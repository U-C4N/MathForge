"""The independent checker of mathforge.areas.stats.order (mathforge.certify.stats.order), tested on its own before
the family's producer exists.

Every true claim is computed here, never by the producer: the type-7 quantile from the data sorted as Fractions and
the card's formula, its evidence from the same sorted list; every most frequent value with collections.Counter. The
evidence follows the card's layout: ((j, x_(j)),) or ((j, x_(j)), (j + 1, x_(j+1))) for quantile, (multiplicity,)
for mode. The checker answers False to every corrupted claim, never with an exception, and refuses an oversized
forgery within FAST seconds under the default budget.

Run by module name: python -m unittest tests.v09.stats.test_order_checker
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable, Sequence
from fractions import Fraction
import inspect
from math import floor
import random
import time
from types import MappingProxyType

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.stats import order as family
from mathforge.certify.stats import order as checker
from mathforge.contracts import CheckStatus
from mathforge.errors import UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational, Sqrt
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.

Number = int | Fraction | Rational


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def frac(number: Number) -> Fraction:
    """An exact number of the tests as a Fraction."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def exact(number: Number) -> int | Rational:
    """The number as a request holds it: an int when it is integral, else a Rational."""
    number = frac(number)
    return number.numerator if number.denominator == 1 else Rational(number.numerator, number.denominator)


def rational(number: Number) -> Rational:
    """The number as a value holds it: always a Rational."""
    number = frac(number)
    return Rational(number.numerator, number.denominator)


def data_args(data: Iterable[Number]) -> dict[str, object]:
    """The arguments of mode as a request holds them."""
    return {"data": tuple(exact(x) for x in data)}


def quantile_args(data: Iterable[Number], p: Number) -> dict[str, object]:
    """The arguments of quantile as a request holds them."""
    return {"data": tuple(exact(x) for x in data), "p": exact(p)}


def quantile_of(data: Sequence[Number], p: Number) -> tuple[Rational, tuple]:
    """(value, evidence) of the type-7 quantile by its definition: sort, h = (n - 1) p, j = floor(h)."""
    ordered = sorted(frac(x) for x in data)
    h = (len(ordered) - 1) * frac(p)
    j = floor(h)
    if h == j:
        return rational(ordered[j]), ((j, rational(ordered[j])),)
    value = ordered[j] + (h - j) * (ordered[j + 1] - ordered[j])
    return rational(value), ((j, rational(ordered[j])), (j + 1, rational(ordered[j + 1])))


def mode_of(data: Sequence[Number]) -> tuple[tuple[Rational, ...], tuple[int]]:
    """(value, evidence) of mode by its definition: the counts, then every value of the largest count, ascending."""
    counts = Counter(frac(x) for x in data)
    top = max(counts.values())
    return tuple(rational(x) for x in sorted(x for x, count in counts.items() if count == top)), (top,)


def quantile_claim(data: Sequence[Number], p: Number) -> tuple[dict[str, object], Rational, tuple]:
    """(args, value, evidence) of a true quantile claim."""
    return (quantile_args(data, p), *quantile_of(data, p))


def mode_claim(data: Sequence[Number]) -> tuple[dict[str, object], tuple[Rational, ...], tuple[int]]:
    """(args, value, evidence) of a true mode claim."""
    return (data_args(data), *mode_of(data))


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


def seeded_data(rng: random.Random, n: int, spread: int) -> list[Fraction]:
    """n random exact rationals with many repeats: small numerators over a few denominators."""
    return [Fraction(rng.randrange(-spread, spread + 1), rng.choice((1, 1, 2, 3))) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values and arguments that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Fractional(Rational):
    """A Rational subclass, equal to the right number but not of the class Rational."""


class Values(tuple):
    """A tuple subclass."""


class OrderCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Iterable[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        data = Argument("data", "", None, "rational_list")
        expected = (
            OperationSpec("quantile", "stats", (data, Argument("p", "Rational | int", None, "rational")), "Rational",
                          "Compute the exact type-7 quantile of rational data.", SCOPE, version=6),
            OperationSpec("mode", "stats", (data,), "tuple[Rational, ...]",
                          "Find every most frequent value of rational data.", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.SCOPE, SCOPE)
        self.assertEqual(family.TYPES, {})
        self.assertFalse(hasattr(family, "RENDER"))  # No record class, so nothing to render.
        self.assertEqual(family.CHECKER, "mathforge.certify.stats.order")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)
        # The producer exists now (it raised UnsupportedOperation while only the checker was written): its answers
        # are the claims of this module's references, which the checker accepts. A wrong operation name is still
        # UnsupportedOperation.
        for operation, args, value, evidence in (("quantile", *quantile_claim((1, 2), 0)),
                                                 ("quantile", *quantile_claim((1, 2, 3, 4), Fraction(1, 4))),
                                                 ("mode", *mode_claim((1, 2, 2, 3)))):
            with self.subTest(operation=operation, produced=args):
                produced_value, produced_evidence = family.produce(operation, dict(args))
                self.assertEqual(produced_value, value)
                self.assertIs(checker.check_operation(operation, args, produced_value, produced_evidence), True)
        with self.assertRaises(UnsupportedOperation):
            family.produce("median", {"data": (1, 2)})

    # -----------------------------------------------------------------------------------------------------------
    # True claims

    def test_quantile_accepts_the_known_values_of_the_card(self):
        cases = [((1, 2, 3, 4), Fraction(1, 2), Rational(5, 2)), ((1, 2, 3, 4), Fraction(1, 4), Rational(7, 4)),
                 ((5,), Fraction(1, 3), Rational(5)), ((3, 1, 2), 0, Rational(1)), ((3, 1, 2), 1, Rational(3))]
        for data, p, expected in cases:
            with self.subTest(data=data, p=p):
                args, value, evidence = quantile_claim(data, p)
                self.assertEqual(value, expected)
                self.assertIs(type(value), Rational)
                self.assertAccepts("quantile", args, value, evidence)
        # The layouts of the card, written out.
        self.assertAccepts("quantile", quantile_args((1, 2, 3, 4), Fraction(1, 2)), Rational(5, 2),
                           ((1, Rational(2)), (2, Rational(3))))
        self.assertAccepts("quantile", quantile_args((3, 1, 2), 1), Rational(3), ((2, Rational(3)),))

    def test_quantile_accepts_special_cases(self):
        cases: list[tuple[str, Sequence[Number], Number]] = [
            ("one item, p = 0", (7,), 0), ("one item, p = 1", (Fraction(-2, 3),), 1),
            ("one item, p = 1/2", (Fraction(9, 4),), Fraction(1, 2)),
            ("p = 0 is the minimum", (4, -1, 9, 2), 0), ("p = 1 is the maximum", (4, -1, 9, 2), 1),
            ("h integral inside", (10, 40, 20, 30, 50), Fraction(1, 2)),
            ("a tie across j and j + 1", (1, 2, 2, 2, 5), Fraction(3, 8)),
            ("all equal", (3, 3, 3, 3), Fraction(2, 7)),
            ("negative rationals", (Fraction(-1, 2), Fraction(-7, 3), Fraction(-5, 6)), Fraction(1, 3)),
            ("mixed ints and fractions", (Fraction(1, 2), 2, Fraction(-3, 4), 0, 1), Fraction(5, 7)),
            ("two items", (0, 1), Fraction(1, 3)),
            ("long integers", (2 ** 200 + 1, -(2 ** 199), 3 ** 100), Fraction(2, 5)),
            ("unsorted with repeats", (5, 1, 5, 1, 3, 3, 3), Fraction(1, 6)),
        ]
        for label, data, p in cases:
            with self.subTest(case=label):
                args, value, evidence = quantile_claim(data, p)
                self.assertAccepts("quantile", args, value, evidence)
        # An order statistic of the evidence may be an exact int as well as a Rational: the same number.
        args, value, evidence = quantile_claim((4, 1, 3, 2), Fraction(1, 2))
        self.assertAccepts("quantile", args, value, tuple((rank, exact(v)) for rank, v in evidence))
        self.assertAccepts("quantile", args, value, ((evidence[0][0], exact(evidence[0][1])), evidence[1]))
        # p given as a Rational with denominator 1, and data items given as Rationals with denominator 1.
        self.assertAccepts("quantile", {"data": (Rational(3), Rational(1), 2), "p": Rational(1)}, Rational(3),
                           ((2, 3),))
        self.assertAccepts("quantile", {"data": (Rational(3), Rational(1), 2), "p": Rational(0)}, Rational(1),
                           ((0, 1),))
        # A forged Rational in canonical form is the same number: the checker reads its fields.
        self.assertAccepts("quantile", {"data": (forged_rational(1, 2), 1), "p": forged_rational(1, 2)},
                           forged_rational(3, 4), ((0, forged_rational(1, 2)), (1, 1)))

    def test_quantile_accepts_seeded_true_claims(self):
        rng = random.Random(20261113)
        for trial in range(150):
            n = rng.randrange(1, 40)
            data = seeded_data(rng, n, rng.choice((3, 50, 10 ** 6)))
            p = rng.choice((Fraction(0), Fraction(1), Fraction(rng.randrange(0, 101), 100),
                            Fraction(rng.randrange(0, n + 1), max(n, 1))))
            with self.subTest(trial=trial):
                self.assertAccepts("quantile", *quantile_claim(data, p))

    def test_mode_accepts_the_known_values_of_the_card(self):
        cases = [((1, 2, 2, 3), (Rational(2),), 2), ((1, 1, 2, 2, 3), (Rational(1), Rational(2)), 2),
                 ((1, 2, 3), (Rational(1), Rational(2), Rational(3)), 1),
                 ((Fraction(1, 2), Fraction(2, 4), 1), (Rational(1, 2),), 2)]
        for data, expected, multiplicity in cases:
            with self.subTest(data=data):
                args, value, evidence = mode_claim(data)
                self.assertEqual(value, expected)
                self.assertEqual(evidence, (multiplicity,))
                self.assertAccepts("mode", args, value, evidence)
        # 2/4 as a request holds it: the Rational constructor has reduced it to 1/2.
        self.assertAccepts("mode", {"data": (Rational(1, 2), Rational(2, 4), 1)}, (Rational(1, 2),), (2,))

    def test_mode_accepts_special_cases(self):
        cases: list[tuple[str, Sequence[Number]]] = [
            ("one item", (Fraction(-5, 7),)), ("all equal", (4, 4, 4)), ("negative values", (-3, -1, -3, -2, -1)),
            ("zero among others", (0, 0, 1, -1)), ("a tie of three", (3, 1, 2, 1, 2, 3, 9)),
            ("fractions and ints", (Fraction(3, 2), 1, Fraction(3, 2), 1, Fraction(1, 2))),
            ("long integers", (2 ** 300, 2 ** 300, -(2 ** 300), 7)),
            ("every item once, unsorted", (5, -2, Fraction(1, 3), 0)),
        ]
        for label, data in cases:
            with self.subTest(case=label):
                self.assertAccepts("mode", *mode_claim(data))
        # Equal numbers in different spellings of the request count as one value.
        self.assertAccepts("mode", {"data": (2, Rational(2), Rational(4, 2), 3, 3)}, (Rational(2),), (3,))
        self.assertAccepts("mode", {"data": (forged_rational(2, 1), 2, 5)}, (Rational(2),), (2,))

    def test_mode_accepts_seeded_true_claims(self):
        rng = random.Random(20261114)
        for trial in range(150):
            data = seeded_data(rng, rng.randrange(1, 60), rng.choice((2, 10, 1000)))
            with self.subTest(trial=trial):
                self.assertAccepts("mode", *mode_claim(data))

    def test_full_size_claims_fit_the_default_budget(self):
        """About the most data a request may hold, checked within FAST seconds under the default budget, and
        accepted by mf.verify."""
        rng = random.Random(20261115)
        data = [Fraction(rng.randrange(-10 ** 9, 10 ** 9), rng.randrange(1, 10 ** 6)) for _ in range(20_000)]
        repeated = [Fraction(rng.randrange(0, 500)) for _ in range(20_000)]
        claims = [("quantile", *quantile_claim(data, Fraction(37, 101))), ("quantile", *quantile_claim(repeated, 1)),
                  ("mode", *mode_claim(data)), ("mode", *mode_claim(repeated))]
        for operation, args, value, evidence in claims:
            with self.subTest(operation=operation, value=str(value)[:40]):
                start = time.perf_counter()
                self.assertAccepts(operation, args, value, evidence)
                self.assertLess(time.perf_counter() - start, FAST)
        # A Result holds 20,000 integers, or 10,000 fractions (more nodes each), within max_evidence_items; the
        # value of mode on distinct data holds every item again.
        for claim in (claims[1], claims[3], ("quantile", *quantile_claim(data[:10_000], Fraction(37, 101))),
                      ("mode", *mode_claim(data[:5_000]))):
            report = mf.verify(claimed(*claim))
            self.assertTrue(report.verified, support.statuses(report))

    # -----------------------------------------------------------------------------------------------------------
    # Wrong and non-canonical values

    def test_quantile_rejects_wrong_values(self):
        args, value, evidence = quantile_claim((1, 2, 3, 4), Fraction(1, 2))  # 5/2 from x_(1) = 2 and x_(2) = 3.
        single = quantile_claim((3, 1, 2), 1)
        tiny = Rational(1, 10 ** 30)
        self.assertRejectsAll("quantile", [
            ("2 for 5/2", args, Rational(2), evidence), ("3 for 5/2", args, Rational(3), evidence),
            ("5/2 plus 1/10^30", args, value + tiny, evidence), ("-5/2", args, Rational(-5, 2), evidence),
            ("the median 2 for the maximum 3", single[0], Rational(2), single[2]),
            ("1 for the maximum 3", single[0], Rational(1), single[2]),
            ("the mean of the data", args, Rational(5, 2) + Rational(1, 4), evidence),
            ("7/4 for p = 1/2", args, Rational(7, 4), evidence),
        ])

    def test_quantile_rejects_non_canonical_values(self):
        args, value, evidence = quantile_claim((1, 2, 3, 4), Fraction(1, 2))
        whole_args, whole, whole_evidence = quantile_claim((5,), Fraction(1, 3))
        self.assertRejectsAll("quantile", [
            ("the int 5 for Rational(5)", whole_args, 5, whole_evidence),
            ("an int subclass", whole_args, Integer(5), whole_evidence),
            ("a Fraction", args, Fraction(5, 2), evidence), ("a float", args, 2.5, evidence),
            ("a str", args, "5/2", evidence), ("a bool", whole_args, True, whole_evidence),
            ("5/2 as 10/4", args, forged_rational(10, 4), evidence),
            ("5/2 as -5/-2", args, forged_rational(-5, -2), evidence),
            ("5 as 5/1 with a float denominator", whole_args, forged_rational(5, 1.0), whole_evidence),
            ("a Rational of a bool", quantile_args((1, 2), 0), forged_rational(True, 1), ((0, 1),)),
            ("a Rational with a zero denominator", args, forged_rational(5, 0), evidence),
            ("a Rational without a denominator", args, support.forged(Rational, numerator=5), evidence),
            ("a Rational without slots", args, support.forged(Rational), evidence),
            ("an impostor Rational", args, support.impostor(Rational(5, 2)), evidence),
            ("a Rational subclass", args, Fractional(5, 2), evidence),
            ("a one-item tuple", args, (Rational(5, 2),), evidence),
            ("a Sqrt", args, Sqrt(Rational(25, 4)), evidence),
            ("None", args, None, evidence),
        ])
        self.assertAccepts("quantile", args, value, evidence)  # The canonical value itself passes.
        self.assertAccepts("quantile", whole_args, whole, whole_evidence)

    def test_mode_rejects_wrong_values(self):
        args, value, evidence = mode_claim((1, 1, 2, 2, 3))  # (1, 2), multiplicity 2.
        self.assertRejectsAll("mode", [
            ("a missing value", args, (Rational(1),), evidence), ("the other missing", args, (Rational(2),), evidence),
            ("an extra value of count 1", args, (Rational(1), Rational(2), Rational(3)), evidence),
            ("a value not in the data", args, (Rational(1), Rational(2), Rational(4)), evidence),
            ("the least frequent value", args, (Rational(3),), evidence),
            ("a substitute", args, (Rational(1), Rational(5, 2)), evidence),
            ("every value with multiplicity 1", args, (Rational(1), Rational(2), Rational(3)), (1,)),
            ("the values of count 1", args, (Rational(3),), (1,)),
        ])
        single = mode_claim((1, 2, 2, 3))
        self.assertRejectsAll("mode", [
            ("all values", single[0], (Rational(1), Rational(2), Rational(3)), (1,)),
            ("1 for 2", single[0], (Rational(1),), (2,)),
        ])

    def test_mode_rejects_non_canonical_values(self):
        args, value, evidence = mode_claim((1, 1, 2, 2, 3))
        self.assertRejectsAll("mode", [
            ("descending", args, (Rational(2), Rational(1)), evidence),
            ("a repeated value", args, (Rational(1), Rational(1), Rational(2)), evidence),
            ("ints for Rationals", args, (1, 2), evidence), ("one int", args, (Rational(1), 2), evidence),
            ("an int subclass", args, (Rational(1), Integer(2)), evidence),
            ("a list for a tuple", args, [Rational(1), Rational(2)], evidence),
            ("a tuple subclass", args, Values((Rational(1), Rational(2))), evidence),
            ("a float", args, (Rational(1), 2.0), evidence), ("a str", args, (Rational(1), "2"), evidence),
            ("a bool", args, (True, Rational(2)), evidence), ("a Fraction", args, (Fraction(1), Rational(2)), evidence),
            ("2 as 4/2", args, (Rational(1), forged_rational(4, 2)), evidence),
            ("1 as -1/-1", args, (forged_rational(-1, -1), Rational(2)), evidence),
            ("a Rational of a bool", args, (forged_rational(True, 1), Rational(2)), evidence),
            ("a Rational without slots", args, (support.forged(Rational), Rational(2)), evidence),
            ("an impostor Rational", args, (Rational(1), support.impostor(Rational(2))), evidence),
            ("a Rational subclass", args, (Fractional(1), Rational(2)), evidence),
            ("an empty tuple", args, (), evidence), ("a bare Rational", mode_claim((1, 2, 2))[0], Rational(2), (2,)),
            ("a nested tuple", args, ((Rational(1), Rational(2)),), evidence), ("None", args, None, evidence),
        ])
        self.assertAccepts("mode", args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # Arguments and evidence

    def test_rejects_claims_on_inputs_outside_the_domain(self):
        evidence = ((0, Rational(1)),)
        self.assertRejectsAll("quantile", [
            ("p = -1/2", quantile_args((1, 2), Fraction(-1, 2)), Rational(1), evidence),
            ("p = 3/2", quantile_args((1, 2), Fraction(3, 2)), Rational(2), ((1, 2),)),
            ("p = 2", quantile_args((1, 2), 2), Rational(3), ((2, 3),)),
            ("p = -1", quantile_args((1, 2), -1), Rational(0), ((-1, 0),)),
            ("empty data", {"data": (), "p": 0}, Rational(0), ((0, 0),)),
        ])
        self.assertRejectsAll("mode", [("empty data", {"data": ()}, (Rational(0),), (0,)),
                                       ("empty data, empty value", {"data": ()}, (), (0,))])

    def test_rejects_corrupted_arguments(self):
        q_args, q_value, q_evidence = quantile_claim((1, 2, 3, 4), Fraction(1, 2))
        m_args, m_value, m_evidence = mode_claim((1, 1, 2, 3))
        stand_ins = [("a bool", True), ("a float", 2.0), ("a str", "2"), ("None", None), ("a Fraction", Fraction(2)),
                     ("an int subclass", Integer(2)), ("a Rational subclass", Fractional(2)),
                     ("an impostor Rational", support.impostor(Rational(2))),
                     ("a Rational in higher terms", forged_rational(4, 2)),
                     ("a Rational with a negative denominator", forged_rational(-2, -1)),
                     ("a Rational of a bool", forged_rational(True, 1)), ("a Rational without slots",
                                                                          support.forged(Rational))]
        for label, item in stand_ins:
            with self.subTest(item=label):
                self.assertRejects("quantile", dict(q_args, data=(1, item, 3, 4)), q_value, q_evidence)
                self.assertRejects("quantile", dict(q_args, p=item), q_value, q_evidence)
                self.assertRejects("mode", {"data": (1, 1, item, 3)}, m_value, m_evidence)
        self.assertRejectsAll("quantile", [
            ("a list for data", dict(q_args, data=list(q_args["data"])), q_value, q_evidence),
            ("a tuple subclass for data", dict(q_args, data=Values(q_args["data"])), q_value, q_evidence),
            ("p as a 1-tuple", dict(q_args, p=(Rational(1, 2),)), q_value, q_evidence),
            ("p as a float", dict(q_args, p=0.5), q_value, q_evidence),
            ("p as 1/2 in higher terms", dict(q_args, p=forged_rational(2, 4)), q_value, q_evidence),
            ("missing p", {"data": q_args["data"]}, q_value, q_evidence),
            ("missing data", {"p": q_args["p"]}, q_value, q_evidence),
            ("an extra argument", dict(q_args, q=1), q_value, q_evidence),
            ("a renamed argument", {"xs": q_args["data"], "p": q_args["p"]}, q_value, q_evidence),
            ("a non-str key", {"data": q_args["data"], 1: q_args["p"]}, q_value, q_evidence),
            ("a mapping proxy", MappingProxyType(q_args), q_value, q_evidence),
            ("arguments as a tuple", tuple(q_args.items()), q_value, q_evidence),
        ])
        self.assertRejectsAll("mode", [
            ("a list for data", {"data": list(m_args["data"])}, m_value, m_evidence),
            ("an extra argument p", dict(m_args, p=0), m_value, m_evidence),
            ("a renamed argument", {"xs": m_args["data"]}, m_value, m_evidence),
            ("no arguments", {}, m_value, m_evidence),
            ("data as a bare number", {"data": 1}, m_value, m_evidence),
        ])
        self.assertAccepts("quantile", q_args, q_value, q_evidence)
        self.assertAccepts("mode", m_args, m_value, m_evidence)

    def test_rejects_wrong_evidence(self):
        args, value, evidence = quantile_claim((1, 2, 3, 4), Fraction(1, 2))  # ((1, 2), (2, 3)).
        whole = quantile_claim((10, 40, 20, 30, 50), Fraction(1, 2))  # h = 2: ((2, 30),).
        two, three = Rational(2), Rational(3)
        self.assertRejectsAll("quantile", [
            ("no evidence", args, value, ()), ("one entry for two", args, value, ((1, two),)),
            ("the second entry alone", args, value, ((2, three),)),
            ("an extra entry", args, value, evidence + ((3, Rational(4)),)),
            ("the entries swapped", args, value, ((2, three), (1, two))),
            ("ranks off by one", args, value, ((2, two), (3, three))),
            ("ranks not j and j + 1", args, value, ((0, Rational(1)), (2, three))),
            ("a list of entries", args, value, list(evidence)),
            ("an entry as a list", args, value, ([1, two], evidence[1])),
            ("an entry of three", args, value, ((1, two, 0), evidence[1])),
            ("an entry of one", args, value, ((1,), evidence[1])),
            ("a bare pair", args, value, (1, two)),
            ("a bool rank 1", args, value, ((True, two), evidence[1])),
            ("a float rank", args, value, ((1.0, two), evidence[1])),
            ("a str rank", args, value, (("1", two), evidence[1])),
            ("an int subclass rank", args, value, ((Integer(1), two), evidence[1])),
            ("a float statistic", args, value, ((1, 2.0), evidence[1])),
            ("a str statistic", args, value, ((1, "2"), evidence[1])),
            ("a bool statistic", quantile_args((0, 1, 1), 0), Rational(0), ((0, False),)),
            ("a Fraction statistic", args, value, ((1, Fraction(2)), evidence[1])),
            ("a statistic in higher terms", args, value, ((1, forged_rational(4, 2)), evidence[1])),
            ("an impostor statistic", args, value, ((1, support.impostor(two)), evidence[1])),
            ("a list for the evidence of one", whole[0], whole[1], [(2, Rational(30))]),
            ("two entries where h is integral", whole[0], whole[1], ((2, Rational(30)), (3, Rational(40)))),
            # Forged evidence that agrees with the claimed value by the formula, but is not the order statistics.
            ("consistent but wrong statistics", args, Rational(7, 2), ((1, three), (2, Rational(4)))),
            ("consistent, a statistic outside the data", args, Rational(9, 4), ((1, Rational(3, 2)), (2, three))),
            ("consistent with a wrong p", args, Rational(7, 4), ((0, Rational(1)), (1, two))),
            ("consistent single, wrong rank", whole[0], Rational(20), ((1, Rational(20)),)),
            ("consistent single, wrong statistic", whole[0], Rational(40), ((2, Rational(40)),)),
        ])
        m_args, m_value, m_evidence = mode_claim((1, 1, 2, 2, 3))  # (1, 2), (2,).
        self.assertRejectsAll("mode", [
            ("no evidence", m_args, m_value, ()), ("an extra item", m_args, m_value, (2, 2)),
            ("a wrong multiplicity", m_args, m_value, (3,)), ("multiplicity 1", m_args, m_value, (1,)),
            ("multiplicity 0", m_args, m_value, (0,)), ("a negative multiplicity", m_args, m_value, (-2,)),
            ("a bool multiplicity", mode_claim((1, 2))[0], (Rational(1), Rational(2)), (True,)),
            ("a float multiplicity", m_args, m_value, (2.0,)), ("a str multiplicity", m_args, m_value, ("2",)),
            ("a Rational multiplicity", m_args, m_value, (Rational(2),)),
            ("an int subclass", m_args, m_value, (Integer(2),)), ("a list", m_args, m_value, [2]),
            ("a bare int", m_args, m_value, 2), ("a nested tuple", m_args, m_value, ((2,),)),
            # Forged evidence that agrees with the value it comes with, but is not the maximal multiplicity.
            ("consistent but not maximal", m_args, (Rational(3),), (1,)),
        ])

    def test_rejects_wrong_operations(self):
        q = quantile_claim((1, 2, 3, 4), Fraction(1, 2))
        m = mode_claim((1, 1, 2, 3))
        for operation in ("Quantile", "quantile ", "median", "mode_", "MODE", "", "v6.quantile", "correlation"):
            with self.subTest(operation=operation):
                self.assertRejects(operation, *q)
                self.assertRejects(operation, *m)
        self.assertRejects("mode", *q)  # The operations swapped.
        self.assertRejects("quantile", *m)
        self.assertRejects("mode", q[0], *m[1:])
        self.assertRejects("quantile", m[0], *q[1:])

    def test_never_raises_on_junk(self):
        """Every slot of a claim filled with stand-ins of every kind: the checker answers False, never raising."""
        junk: list[object] = [None, 0, 1.5, "x", b"x", (), [], {}, Rational(7), Sqrt(Rational(2)), object(), True,
                              support.forged(Rational), Values((1,)), MappingProxyType({}), frozenset(),
                              Fraction(1, 2), (1, 2), ((0, 1),), {"data": (1, 2)}, Integer(1), (None,), ((None, None),)]
        claims = {"quantile": quantile_claim((4, 1, 3), Fraction(1, 4)), "mode": mode_claim((4, 1, 4))}
        for operation, (args, value, evidence) in claims.items():
            for index, item in enumerate(junk):
                with self.subTest(operation=operation, item=index, kind=type(item).__name__):
                    self.assertRejects(item, args, value, evidence)
                    self.assertRejects(operation, item, value, evidence)
                    self.assertRejects(operation, args, item, evidence)
                    if item != evidence:
                        self.assertRejects(operation, args, value, item)
                    for name in args:
                        self.assertRejects(operation, dict(args, **{name: item}), value, evidence)
                    # Among other data, an exact item changes the answer: (4, 1, 3) and (4, 1, 4) become other data.
                    self.assertRejects(operation, dict(args, data=(item, 50, 60)), value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # Oversized forgeries and the budget

    def test_oversized_forgeries_fail_fast(self):
        rng = random.Random(20261116)
        big = [rng.getrandbits(65_000) | 1 << 64_999 | 1 for _ in range(4)]  # Within max_integer_bits, but long.
        long_data = tuple(range(20_000))
        small, value, evidence = quantile_claim((1, 2, 3, 4), Fraction(1, 2))
        long_q = quantile_claim(long_data, Fraction(1, 3))
        m_args, m_value, m_evidence = mode_claim((1, 1, 2, 3))
        long_m = mode_claim(long_data)
        q_cases = [
            (small, forged_rational(GIANT, 1), evidence), (small, forged_rational(1, GIANT + 2), evidence),
            (small, forged_rational(-GIANT, GIANT + 1), evidence), (small, Rational(big[0], big[1]), evidence),
            (long_q[0], Rational(big[0], big[1]), long_q[2]),
            (small, value, ((1, GIANT), evidence[1])), (small, value, ((1, forged_rational(1, GIANT)), evidence[1])),
            (small, value, ((GIANT, Rational(2)), evidence[1])), (small, value, ((1, big[0]), (2, big[1]))),
            (long_q[0], long_q[1], ((long_q[2][0][0], big[0]), (long_q[2][1][0], big[1]))),
            (small, value, evidence * 50_000), (small, value, ((1, Rational(2)),) * 100_000),
            (small, value, tuple(range(100_000))), (small, tuple(range(100_000)), evidence),
        ]
        m_cases = [
            (m_args, m_value, (GIANT,)), (m_args, m_value, (big[0],)), (m_args, m_value, (2,) * 100_000),
            (m_args, tuple(Rational(k) for k in range(100_000)), m_evidence),
            (m_args, (Rational(1), forged_rational(GIANT)), m_evidence),
            (m_args, (forged_rational(1, GIANT + 2), Rational(1)), m_evidence),
            (m_args, (Rational(big[0], big[1]), Rational(big[2], big[3])), m_evidence),
            (long_m[0], tuple(Rational(big[k % 4] + k) for k in range(20_000)), long_m[2]),
            (long_m[0], long_m[1][:-1] + (Rational(big[0]),), long_m[2]),
            (long_m[0], long_m[1], (GIANT,)),
        ]
        for operation, cases in (("quantile", q_cases), ("mode", m_cases)):
            for index, (args, claim, proof) in enumerate(cases):
                with self.subTest(operation=operation, case=index):
                    start = time.perf_counter()
                    self.assertRejects(operation, args, claim, proof)  # Under the default budget, never raising.
                    self.assertLess(time.perf_counter() - start, FAST)
        # A request integer over max_integer_bits raises ResourceLimitError, as the request's inspection does before
        # any checker runs.
        requests = [
            ("quantile", {"data": (1, GIANT, 3), "p": 0}, Rational(1), ((0, 1),)),
            ("quantile", {"data": (1, Rational(1, GIANT), 3), "p": 0}, Rational(0), ((0, 0),)),
            ("quantile", {"data": (1, 2, 3), "p": Rational(1, GIANT)}, Rational(1), ((0, 1),)),
            ("quantile", {"data": (1, 2, 3), "p": GIANT}, Rational(1), ((0, 1),)),
            ("mode", {"data": (1, -GIANT, 3)}, (Rational(1),), (1,)),
            ("mode", {"data": (1, Rational(GIANT, 3), 3)}, (Rational(1),), (1,)),
        ]
        for index, (operation, args, claim, proof) in enumerate(requests):
            with self.subTest(operation=operation, request=index):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, claim, proof)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_colliding_integer_hashes_stay_cheap(self):
        """Integers that share Python's integer hash (multiples of 2**61 - 1) cannot drive the counting dict of mode
        quadratic: its keys are byte strings with a randomized hash."""
        modulus = 2 ** 61 - 1
        data = tuple(k * modulus for k in range(1, 20_001)) + (modulus,)
        self.assertEqual(len({hash(x) for x in data}), 1)
        start = time.perf_counter()
        self.assertAccepts("mode", {"data": data}, (Rational(modulus),), (2,))
        self.assertRejects("mode", {"data": data}, (Rational(2 * modulus),), (2,))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_work_is_charged_before_it_runs(self):
        rng = random.Random(20261117)
        data = [Fraction(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 50)) for _ in range(300)]
        for operation, (args, value, evidence) in (("quantile", quantile_claim(data, Fraction(2, 7))),
                                                   ("mode", mode_claim(data))):
            with self.subTest(operation=operation):
                # The charges depend on the claim alone: a budget one unit short refuses it, its exact size passes.
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, evidence), True)
                work = budget.work
                self.assertGreater(work, 2 * 300)  # At least a unit per item and pass.
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, value, evidence, limits=ComputationLimits(max_work=work))
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence, limits=ComputationLimits(max_work=50))
                self.assertLess(time.perf_counter() - start, FAST)
        # Long integers cost more than short ones: each comparison of the counting pass pays for its products.
        scale = 2 ** 30_000 + 1
        charged = []
        for factor in (1, scale):
            claim = quantile_claim([x * factor for x in data[:40]], Fraction(1, 2))
            with computation(fresh=True) as budget:
                self.assertIs(checker.check_operation("quantile", *claim), True)
            charged.append(budget.work)
        self.assertGreater(charged[1], 4 * charged[0])
        # A value or evidence integer over a smaller max_integer_bits is refused, as no computation under that budget
        # forms it (the value is read before the data); a request integer over it raises.
        narrow = ComputationLimits(max_integer_bits=20)
        self.assertRejects("quantile", quantile_args((1, 2), 0), Rational(1, 2 ** 30), ((0, 1),), limits=narrow)
        self.assertRejects("quantile", quantile_args((1, 2), 0), Rational(1), ((0, 2 ** 30),), limits=narrow)
        self.assertRejects("mode", data_args((1, 2)), (Rational(1), Rational(2 ** 30)), (1,), limits=narrow)
        self.assertRejects("mode", data_args((1, 2 ** 30)), (Rational(1), Rational(2 ** 30)), (1,), limits=narrow)
        self.assertAccepts("mode", data_args((1, 2 ** 30)), (Rational(1), Rational(2 ** 30)), (1,))
        for operation, args, claim, proof in (("mode", data_args((1, 2 ** 30)), (Rational(1),), (1,)),
                                              ("quantile", quantile_args((1, 2 ** 30), 0), Rational(1), ((0, 1),)),
                                              ("quantile", quantile_args((1, 2), Fraction(1, 2 ** 30)), Rational(1),
                                               ((0, 1), (1, 2)))):
            with self.subTest(operation=operation, narrow=claim):
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, claim, proof, limits=narrow)

    def test_no_power_is_formed_by_the_builtin(self):
        """No pow call and no ** in the checker, so that nothing escapes the charges; it imports nothing but the
        allowed layers and the standard library, and never the producer."""
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertEqual(imported, {"__future__", "collections.abc", "math", "mathforge.limits", "mathforge.model",
                                    "mathforge.trusted.integers"})
        for operation in ("quantile(data, p) = q, evidence ((j, v),) or ((j, v), (j + 1, w)).",
                          "mode(data) = (m_1, ..., m_s), evidence (c,)."):
            self.assertIn(operation, checker.__doc__)

    # -----------------------------------------------------------------------------------------------------------
    # The verifier

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip, and a wrong value fails
        the mathematics check."""
        q = quantile_claim((1, 2, 3, 4), Fraction(1, 4))
        m = mode_claim((1, 1, 2, 2, 3))
        claims = [("quantile", q[0], q[1], q[2], Rational(2)),
                  ("quantile", *quantile_claim((Fraction(1, 2), -3, 7), Fraction(2, 3)), Rational(1)),
                  ("mode", m[0], m[1], m[2], (Rational(1),)),
                  ("mode", *mode_claim((Fraction(1, 2), Fraction(2, 4), 1)), (Rational(1),))]
        for operation, args, value, evidence, wrong in claims:
            with self.subTest(operation=operation, value=value):
                self.roundtrip(claimed(operation, args, value, evidence))
                report = support.statuses(mf.verify(claimed(operation, args, wrong, evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
