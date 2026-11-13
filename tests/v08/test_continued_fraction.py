"""The pilot v6 family: finite simple continued fractions (mathforge.ntheory.continued_fraction).

It drives the v6 path end to end: registry, generated wrappers, operations_v6.run, the independent checker,
JSON v6 and rendering. rendering_v6 also presents the generic deferral records, tested on the demo family.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge import operations_v6, registry
from mathforge.areas.ntheory import continued_fraction as family
from mathforge.areas.ntheory.continued_fraction import ContinuedFraction
from mathforge.certify.ntheory.continued_fraction import check_operation
from mathforge.contracts import CheckStatus, ErrorInfo, ExecutionStatus
from mathforge.deferred import Reason

from . import demo_family
from .demo_family import DemoPair

Q = m.Rational
CLAIMS = ["result.source", "result.contract", "result.trace", "result.mathematics"]
SPEC = registry.OperationSpec(
    "continued_fraction", "ntheory", (registry.Argument("value", "Rational | int", None, "rational"),),
    "ContinuedFraction", "Return the finite simple continued fraction of an exact rational.",
    "exact rationals; canonical finite simple continued fraction", version=6)


def forged(terms):
    """A ContinuedFraction that skips the canonical-form validation, which no decoded document can do."""
    record = object.__new__(ContinuedFraction)
    object.__setattr__(record, "terms", terms)
    return record


@dataclass(frozen=True)
class Impostor:
    """Another record class whose terms field holds the right integers."""
    terms: tuple[int, ...]


def evaluate(terms):
    """The value of [a0; a1, ..., an], folded from the right with exact fractions, whatever the form."""
    value = Fraction(terms[-1])
    for term in reversed(terms[:-1]):
        value = term + 1 / value
    return value


def statuses(report):
    """Check status by claim."""
    return {check.claim: check.status for check in report.checks}


class ContinuedFractionTests(unittest.TestCase):
    def test_known_values(self):
        for value, terms in ((Q(415, 93), (4, 2, 6, 7)), (5, (5,)), (Q(1, 2), (0, 2)), (Q(-7, 3), (-3, 1, 2)),
                             (0, (0,)), (Q(5), (5,)), (Q(-1, 2), (-1, 2)), (-5, (-5,))):
            with self.subTest(value=value):
                self.assertEqual(m.ntheory.continued_fraction(value).terms, terms)
                result = m.operations.continued_fraction(value)
                self.assertEqual(result.value, ContinuedFraction(terms))
                self.assertEqual([check.claim for check in result.verification], CLAIMS)
                self.assertTrue(m.verify(result).verified)

    def test_registry_entry(self):
        self.assertEqual(registry.v6_families(), ("mathforge.areas.algebra.roots",
                                                  "mathforge.areas.algebra.polynomial_tools",
                                                  "mathforge.areas.algebra.radicals",
                                                  "mathforge.areas.algebra.sequences",
                                                  "mathforge.areas.discrete.counting",
                                                  "mathforge.areas.equations.piecewise",
                                                  "mathforge.areas.geometry.plane",
                                                  "mathforge.areas.geometry.hull",
                                                  "mathforge.areas.integrals.trig_powers",
                                                  "mathforge.areas.linalg.powers",
                                                  "mathforge.areas.linalg.eigen",
                                                  "mathforge.areas.linalg.forms",
                                                  "mathforge.areas.linalg.lattice",
                                                  "mathforge.areas.ntheory.continued_fraction",
                                                  "mathforge.areas.ntheory.modular",
                                                  "mathforge.areas.ntheory.prime_search",
                                                  "mathforge.areas.ntheory.diophantine",
                                                  "mathforge.areas.ntheory.quadratic_irrational",
                                                  "mathforge.areas.ntheory.arithmetic_functions",
                                                  "mathforge.areas.ntheory.discrete",
                                                  "mathforge.areas.numeric.digits",
                                                  "mathforge.areas.numeric.approximation",
                                                  "mathforge.areas.series.sums",
                                                  "mathforge.areas.series.power_series",
                                                  "mathforge.areas.series.laurent",
                                                  "mathforge.areas.sets.interval_algebra",
                                                  "mathforge.areas.sets.symmetric",
                                                  "mathforge.areas.stats.bivariate",
                                                  "mathforge.areas.stats.order",
                                                  "mathforge.areas.stats.markov",
                                                  "mathforge.areas.stats.exact_tests",
                                                  "mathforge.areas.vector.products"))
        self.assertEqual(registry.spec("continued_fraction"), SPEC)
        self.assertIs(registry.family("continued_fraction"), family)
        self.assertEqual(registry.checker("continued_fraction").__name__,
                         "mathforge.certify.ntheory.continued_fraction")
        self.assertIs(registry.record_types(6)["continued_fraction"], ContinuedFraction)
        self.assertTrue(registry.is_v6_operation("continued_fraction"))
        self.assertNotIn("continued_fraction", registry.operation_arguments(5))  # mathforge.compute stays v0.7.

    def test_checker_rejects_forged_value(self):
        args = {"value": Q(415, 93)}
        self.assertTrue(check_operation("continued_fraction", args, ContinuedFraction((4, 2, 6, 7)), ()))
        self.assertFalse(check_operation("continued_fraction", args, ContinuedFraction((4, 2, 6, 6)), ()))
        # Each forged expansion is mathematically right, so only the canonical-form rule it breaks can reject it:
        # a last term of 1, a zero inner term, a negative inner term, a bool term.
        for value, terms in ((Q(415, 93), (4, 2, 6, 6, 1)), (Q(415, 93), (4, 2, 3, 0, 3, 7)),
                             (Q(415, 93), (5, -2, 7, 7)), (Q(1, 2), (False, 2))):
            with self.subTest(terms=terms):
                self.assertEqual(evaluate(terms), Fraction(value.numerator, value.denominator))
                self.assertFalse(check_operation("continued_fraction", {"value": value}, forged(terms), ()))
        # A float term, a list, no terms, another type, or evidence are rejected as well.
        for value, evidence in ((forged((4, 2, 6, 7.0)), ()), (forged([4, 2, 6, 7]), ()), (forged(()), ()),
                                ((4, 2, 6, 7), ()), (Impostor((4, 2, 6, 7)), ()),
                                (ContinuedFraction((4, 2, 6, 7)), (1,))):
            with self.subTest(value=value, evidence=evidence):
                self.assertFalse(check_operation("continued_fraction", args, value, evidence))
        self.assertTrue(check_operation("continued_fraction", {"value": 5}, ContinuedFraction((5,)), ()))
        self.assertFalse(check_operation("continued_fraction", {"value": True}, ContinuedFraction((1,)), ()))
        self.assertFalse(check_operation("integer_gcd", args, ContinuedFraction((4, 2, 6, 7)), ()))
        # verify() consults the checker: a forged value fails the mathematics check.
        result = m.operations.continued_fraction(Q(415, 93))
        wrong = ContinuedFraction((4, 2, 6, 6))
        tampered = replace(result, value=wrong, verification=(),
                           steps=(replace(result.steps[0], outputs=(wrong, ())),))
        report = statuses(m.verify(tampered))
        self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
        self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_checker_is_independent_of_producer(self):
        result = m.operations.continued_fraction(Q(415, 93))
        decoded = m.loads(m.dumps(result))
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            self.assertTrue(m.verify(result).verified)
            self.assertTrue(m.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = m.operations.continued_fraction(Q(415, 93))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_v6_result_render_roundtrip(self):
        r = m.operations.continued_fraction(m.Rational(415, 93))
        self.assertEqual(m.to_data(r)["version"], 6)
        self.assertEqual(m.loads(m.dumps(r)), r)
        self.assertEqual(m.render_text(r.value), "[4; 2, 6, 7]")
        self.assertEqual(m.to_latex(r.value), "[4; 2, 6, 7]")
        self.assertEqual(m.render_text(r), "completed / value [exact; complete]: [4; 2, 6, 7]")
        five = m.ntheory.continued_fraction(5)
        self.assertEqual((m.render_text(five), m.to_latex(five)), ("[5]", "[5]"))
        self.assertEqual(m.render_text(ContinuedFraction((-3, 1, 2))), "[-3; 1, 2]")
        terms = [{"type": "integer", "value": text} for text in ("4", "2", "6", "7")]
        self.assertEqual(m.to_data(r.value)["data"],
                         {"type": "v6_continued_fraction", "terms": {"type": "tuple", "items": terms}})

    def test_rendering_keeps_the_shared_limits(self):
        # Python refuses str() on integers over 4300 digits; the shared renderer's integer helper does not.
        self.assertEqual(m.render_text(ContinuedFraction((10**5000,))), "[1" + "0" * 5000 + "]")
        with self.assertRaisesRegex(m.ResourceLimitError, "integer characters"):
            m.render_text(ContinuedFraction((10**5000,)), limits=m.RenderLimits(max_chars=1000))
        with self.assertRaisesRegex(m.ResourceLimitError, "characters"):
            m.render_text(ContinuedFraction((4, 2, 6, 7)), limits=m.RenderLimits(max_chars=8))
        with self.assertRaisesRegex(m.ResourceLimitError, "nodes"):
            m.to_latex(ContinuedFraction((1,) * 20 + (2,)), limits=m.RenderLimits(max_nodes=10))

    def test_rejects_inexact_inputs(self):
        for value in (0.5, True, "1/2", m.Symbol("x"), m.sqrt(2), (1, 2)):
            with self.subTest(value=value):
                r = m.operations.continued_fraction(value)
                self.assertIs(r.execution_status, ExecutionStatus.INVALID_INPUT)
                self.assertEqual(r.error.code, "invalid_input")  # Never the producer's internal_error.
        with self.assertRaises(m.OperationError) as raised:
            m.ntheory.continued_fraction(False)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_noncanonical_record_rejected(self):
        with self.assertRaises(m.InvalidInput):
            ContinuedFraction((4, 2, 6, 6, 1))
        for terms in ((), (4, 0, 2), (4, -1, 2), (4, 1), (True,), (4.0,), (4, 2.0), [4, 2]):
            with self.subTest(terms=terms), self.assertRaises(m.InvalidInput):
                ContinuedFraction(terms)
        for terms in ((0,), (-5,), (1, 2), (-3, 1, 2), (0, 1, 2)):
            with self.subTest(terms=terms):
                self.assertEqual(ContinuedFraction(terms).terms, terms)
        document = m.to_data(ContinuedFraction((4, 2, 6, 7)))
        document["data"]["terms"]["items"][-1]["value"] = "1"
        with self.assertRaisesRegex(m.SerializationError, "^Invalid mathematical document$"):
            m.from_data(document)


class V6RenderingTests(unittest.TestCase):
    def test_deferral_records_render(self):
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_defer", {"n": 1})
            self.assertEqual(m.render_text(r.deferred), "Unevaluated(sum, demo_defer, Reason(no_applicable_rule))")
            self.assertEqual(m.to_latex(r.deferred),
                             r"\text{Unevaluated}\left(\text{sum}, \text{demo\_defer}, "
                             r"\text{Reason}\left(\text{no\_applicable\_rule}\right)\right)")
            self.assertEqual(m.render_text(r), "completed / no_conclusion [not_applicable; unknown]: "
                                               "Unevaluated(sum, demo_defer, Reason(no_applicable_rule))")
        reason = Reason("verification_undecided", "50% of the cases")
        self.assertEqual(m.render_text(reason), "Reason(verification_undecided, 50% of the cases)")
        self.assertEqual(m.to_latex(reason),
                         r"\text{Reason}\left(\text{verification\_undecided}, \text{50\% of the cases}\right)")

    def test_family_records_without_render_list_their_fields(self):
        with registry._testing_families(demo_family):
            self.assertEqual(m.render_text(DemoPair(1, -2)), "DemoPair(a=1, b=-2)")
            self.assertEqual(m.to_latex(DemoPair(1, 2)), r"\text{DemoPair}\left(\text{a}=1, \text{b}=2\right)")


if __name__ == "__main__":
    unittest.main()
