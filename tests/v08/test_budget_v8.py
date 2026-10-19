"""The inline Budget and the class-based computation() keep v0.7's counters, messages and call forms.

The expected deltas and messages below were recorded on the unmodified v0.7 code (commit 007a4b3).
"""

from dataclasses import dataclass, field
import unittest
from unittest.mock import patch

from mathforge import limits
from mathforge.errors import InvalidInput
from mathforge.limits import Budget, ComputationLimits, ResourceLimitError, computation, current_budget
from mathforge.model import Domain, Rational, Symbol


SMALL = ComputationLimits(max_work=3, max_nodes=3, max_refinements=3, max_evidence_items=3, max_integer_bits=8,
                          max_matrix_entries=4, max_depth=2)
BAD_CHARGE = "Budget charge must be a nonnegative integer"


@dataclass(frozen=True)
class Box:
    walked: object
    skipped: object = field(default=None, metadata={"budget": "skip"})


def counters(budget):
    return budget.work, budget.node_count, budget.refinements, budget.evidence_count


def nested(depth):
    value = 1
    for _ in range(depth):
        value = (value,)
    return value


class BudgetV8Test(unittest.TestCase):
    def assertCharge(self, budget, thunk, delta, error=None, message=None):
        """Run one charge; compare the (work, nodes, refinements, evidence) delta and the exact error."""
        before = counters(budget)
        if error is None:
            thunk()
        else:
            with self.assertRaises(error) as caught:
                thunk()
            self.assertIs(type(caught.exception), error)
            self.assertEqual(str(caught.exception), message)
        self.assertEqual(tuple(after - start for after, start in zip(counters(budget), before)), delta)

    def test_counter_deltas_and_messages(self):
        limit = ResourceLimitError
        table = (  # (method, arguments, delta, error, message) on a fresh Budget(SMALL)
            ("tick", (), (1, 0, 0, 0), None, None),
            ("tick", (2,), (2, 0, 0, 0), None, None),
            ("tick", (4,), (0, 0, 0, 0), limit, "Computation limit exceeded: work (3)"),
            ("nodes", (), (0, 1, 0, 0), None, None),
            ("nodes", (3,), (0, 3, 0, 0), None, None),
            ("nodes", (4,), (0, 0, 0, 0), limit, "Computation limit exceeded: node_count (3)"),
            ("refine", (), (1, 0, 1, 0), None, None),
            ("refine", (3,), (3, 0, 3, 0), None, None),
            ("refine", (4,), (0, 0, 0, 0), limit, "Computation limit exceeded: refinements (3)"),
            ("evidence", (), (0, 0, 0, 1), None, None),
            ("evidence", (4,), (0, 0, 0, 0), limit, "Computation limit exceeded: evidence_count (3)"),
            ("check_int", (255,), (0, 0, 0, 0), None, None),
            ("check_int", (256,), (0, 0, 0, 0), limit, "Computation limit exceeded: integer bits"),
            ("check_int", (-256,), (0, 0, 0, 0), limit, "Computation limit exceeded: integer bits"),
            ("product", (15, 31), (1, 0, 0, 0), None, None),
            ("product", (255, 2), (1, 0, 0, 0), limit, "Computation limit exceeded: product integer bits"),
            ("product", (0, 2**100), (1, 0, 0, 0), None, None),
            ("product", (), (1, 0, 0, 0), None, None),
            ("product", (2**8,), (1, 0, 0, 0), limit, "Computation limit exceeded: product integer bits"),
            ("product", (2**9,), (1, 0, 0, 0), limit, "Computation limit exceeded: product integer bits"),
            ("product", (1.5, 0), (1, 0, 0, 0), None, None),
            ("product", (1.5, 2), (1, 0, 0, 0), AttributeError, "'float' object has no attribute 'bit_length'"),
            ("matrix", (2, 2), (0, 0, 0, 0), None, None),
            ("matrix", (1, 5), (0, 0, 0, 0), limit, "Computation limit exceeded: matrix entries"),
        )
        for method, arguments, delta, error, message in table:
            with self.subTest(method=method, arguments=arguments):
                budget = Budget(SMALL)
                self.assertCharge(budget, lambda: getattr(budget, method)(*arguments), delta, error, message)
        budget = Budget(ComputationLimits(max_work=1, max_refinements=10))  # Refinements count before work fails.
        self.assertCharge(budget, lambda: budget.refine(2), (0, 0, 2, 0), limit, "Computation limit exceeded: work (1)")
        budget = Budget(ComputationLimits(max_work=10, max_refinements=1))
        self.assertCharge(budget, lambda: budget.refine(2), (0, 0, 0, 0), limit,
                          "Computation limit exceeded: refinements (1)")
        budget = Budget(ComputationLimits(max_work=1))
        budget.tick()
        self.assertCharge(budget, lambda: budget.product(1, 1), (0, 0, 0, 0), limit,
                          "Computation limit exceeded: work (1)")
        budget = Budget(SMALL)
        for _ in range(3):
            budget.tick()
        self.assertEqual(counters(budget), (3, 0, 0, 0))

    def test_inspect_counts_unchanged(self):
        with computation() as budget:
            budget.inspect(Rational(1, 2))
            self.assertEqual(budget.node_count, 3)
        with computation() as budget:
            budget.inspect((1, (2, 3)))
            self.assertEqual(budget.node_count, 5)
        records = (  # (value, evidence, node_count, evidence_count) from v0.7
            (Box(1, (2, 3)), False, 2, 0),  # The marked field is skipped.
            ({"a": [1, 2]}, False, 5, 0),
            ([None, "x", True, 1.5], False, 5, 0),
            (Rational(1, 2), True, 0, 3),
            (Symbol("x", uid="00000000-0000-4000-8000-000000000802"), False, 5, 0),
            (Box, False, 1, 0),  # A dataclass type is a leaf, not a record.
            ((True, 1.5), False, 3, 0),
            ((Domain.REALS, Rational(3), (Rational(1, 2), {Rational(1, 3): Box(Rational(1, 4))})), False, 17, 0),
        )
        for value, evidence, nodes, items in records:
            with self.subTest(value=value):
                budget = Budget()
                budget.inspect(value, evidence=evidence)
                self.assertEqual(counters(budget), (0, nodes, 0, items))
        limit = ResourceLimitError
        failures = (  # (limits, value, evidence, delta, message): the first failing limit and the count it left
            (ComputationLimits(max_depth=4), nested(5), False, (0, 5, 0, 0),
             "Computation limit exceeded: expression depth"),
            (ComputationLimits(max_depth=1), (Rational(1, 2),), False, (0, 2, 0, 0),
             "Computation limit exceeded: expression depth"),
            (ComputationLimits(max_nodes=2), Rational(1, 2), False, (0, 2, 0, 0),
             "Computation limit exceeded: node_count (2)"),
            (ComputationLimits(max_integer_bits=4), Rational(1, 32), False, (0, 2, 0, 0),
             "Computation limit exceeded: integer bits"),
            (ComputationLimits(max_integer_bits=4), Rational(32, 1), False, (0, 3, 0, 0),
             "Computation limit exceeded: integer bits"),
            (ComputationLimits(max_evidence_items=2), Rational(1, 2), True, (0, 0, 0, 2),
             "Computation limit exceeded: evidence_count (2)"),
            (ComputationLimits(max_integer_bits=64), Box(1, 2**100), False, (0, 2, 0, 0), None),
            (ComputationLimits(max_integer_bits=64), Box(2**100), False, (0, 2, 0, 0),
             "Computation limit exceeded: integer bits"),
        )
        for limits_, value, evidence, delta, message in failures:
            with self.subTest(limits=limits_, value=value):
                budget = Budget(limits_)
                self.assertCharge(budget, lambda: budget.inspect(value, evidence=evidence), delta,
                                  None if message is None else limit, message)
        budget = Budget(ComputationLimits(max_depth=1))
        budget.inspect(Rational(1, 2))  # Depth 0 with integer children at depth 1: inside the limit.
        self.assertEqual(budget.node_count, 3)
        budget = Budget(ComputationLimits(max_nodes=5))
        budget.nodes(2)
        self.assertCharge(budget, lambda: budget.inspect((Rational(1, 2),)), (0, 3, 0, 0), limit,
                          "Computation limit exceeded: node_count (5)")

    def test_bad_charges_rejected(self):
        budget = Budget()
        for thunk in (lambda: budget.tick(-1), lambda: budget.tick(1.0), lambda: budget.nodes(True),
                      lambda: budget.refine(-1), lambda: budget.evidence(1.5)):
            with self.subTest(thunk=thunk):
                self.assertCharge(budget, thunk, (0, 0, 0, 0), InvalidInput, BAD_CHARGE)
        for value in (True, 1.0):
            with self.subTest(value=value):
                self.assertCharge(budget, lambda: budget.check_int(value), (0, 0, 0, 0), InvalidInput,
                                  "An exact integer is required")
        for bad in ("x", ComputationLimits):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput) as caught:
                    Budget(bad)
                self.assertEqual(str(caught.exception), "limits must be ComputationLimits or None")
                context = computation(bad)  # As in v0.7 the limits are checked on entry, not by the call.
                with self.assertRaises(InvalidInput) as caught:
                    with context:
                        self.fail("entered with invalid limits")
                self.assertEqual(str(caught.exception), "limits must be ComputationLimits or None")
        self.assertIsNone(current_budget())

    def test_computation_nesting(self):
        self.assertIsNone(current_budget())
        with computation(ComputationLimits(max_work=50)) as outer:
            self.assertIs(current_budget(), outer)
            self.assertEqual(outer.limits, ComputationLimits(max_work=50))
            with computation(ComputationLimits(max_work=1)) as inner:  # Nested: the active budget, limits ignored.
                self.assertIs(inner, outer)
                inner.tick(10)
            with computation(ComputationLimits(max_work=7), fresh=True) as separate:
                self.assertIsNot(separate, outer)
                self.assertIs(current_budget(), separate)
                self.assertEqual((separate.limits.max_work, separate.work), (7, 0))
            self.assertIs(current_budget(), outer)
            with self.assertRaises(KeyError):
                with computation(fresh=True) as failing:
                    self.assertIs(current_budget(), failing)
                    raise KeyError("inside")
            self.assertIs(current_budget(), outer)
            with self.assertRaises(KeyError):
                with computation() as same:
                    self.assertIs(same, outer)
                    raise KeyError("inside")
            self.assertIs(current_budget(), outer)
            self.assertEqual(outer.work, 10)
        self.assertIsNone(current_budget())
        with computation() as default:
            self.assertEqual(default.limits, ComputationLimits())
        self.assertIsNone(current_budget())

    def test_computation_reuse_and_decorator(self):
        context = computation(fresh=True)
        for _ in range(2):  # Entering one object again after leaving it opens a new budget.
            with context as budget:
                self.assertIs(current_budget(), budget)
            self.assertIsNone(current_budget())
        with context as budget:
            with self.assertRaises(RuntimeError):  # ...but one object cannot be active twice at once.
                with context:
                    self.fail("entered an active computation() object again")
            self.assertIs(current_budget(), budget)
        self.assertIsNone(current_budget())

        seen = []

        @computation(ComputationLimits(max_work=5), fresh=True)
        def charged(depth):
            budget = current_budget()
            seen.append(budget)
            budget.tick()
            if depth:
                charged(depth - 1)
            return budget.limits.max_work

        self.assertEqual(charged(1), 5)  # As v0.7's context decorator: one budget per call.
        self.assertEqual(len({id(budget) for budget in seen}), 2)
        self.assertIsNone(current_budget())

    def test_computation_builds_budgets_through_the_module_name(self):
        recorded = []

        class RecordedBudget(Budget):  # The pattern of tools/check_benchmarks.py and benchmarks/exact_algebra.py
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                recorded.append(self)

        a, b = Rational(1, 2), Rational(1, 3)
        with patch.object(limits, "Budget", RecordedBudget):
            with computation(fresh=True) as budget:
                a + b
                budget.inspect(a)
        self.assertEqual(recorded, [budget])
        self.assertEqual((budget.work, budget.node_count), (1, 3))

    def test_limits_stay_readable_and_replaceable(self):
        budget = Budget(ComputationLimits(max_work=2))
        self.assertEqual(budget.limits, ComputationLimits(max_work=2))
        budget.limits = ComputationLimits(max_work=1)  # A replaced limit object takes effect, as in v0.7.
        budget.tick()
        self.assertCharge(budget, budget.tick, (0, 0, 0, 0), ResourceLimitError, "Computation limit exceeded: work (1)")


if __name__ == "__main__":
    unittest.main()
