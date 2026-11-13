"""The sums family of mathforge.series: sum_polynomial, geometric_sum, geometric_series, polynomial_geometric_series.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register series sums

Every expected value comes from the card or from a reference of this module, written here, slow on purpose and
independent of the producer: the closed form of a polynomial sum interpolates the partial sums, added up term by term,
by Lagrange's formula; a finite geometric sum adds its terms one by one; an infinite one is a/(1 - r), confirmed by the
exact tail of its partial sums; the series of p(k) r^k is Newton's series, and its Q is interpolated through
Q(n) = (S_n + Q(0))/r^n, with S_n the partial sums that the identity telescopes. Never from the producer.
tests/v09/series/test_sums_checker.py tests the checker on its own.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.series import sums as family
from mathforge.contracts import CheckStatus, ErrorInfo, ExecutionStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Integers, Rational, Symbol, Truth
from mathforge.polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial

from .. import support

FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
X = Symbol("x")
N = Symbol("n", Integers, (Assumption("positive", Truth.TRUE),))  # A variable with a domain and an assumption.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact number: a float, a bool, a str, None and two containers.
UNCHECKABLE = "for the independent check of the claim"  # The producer's refusal of a claim beyond the checker's budget.

Number = Fraction | int


class Ratio(Rational):
    """A Rational subclass: no exact Rational."""
    __slots__ = ()


class Integer(int):
    """An int subclass, equal to an exact int but none."""


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these, or from the card.

def fraction(number: object) -> Fraction:
    """A Fraction of an int, a Fraction or a mathforge Rational (which Fraction itself does not take)."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)  # type: ignore[arg-type]


def rational(number: object) -> Rational:
    value = fraction(number)
    return Rational(value.numerator, value.denominator)


def exact(number: Number) -> int | Rational:
    """An argument of the kind "rational": an int stays an int, a Fraction becomes a Rational."""
    return number if type(number) is int else rational(number)


def polynomial(coefficients: Sequence[object], variable: Symbol = X) -> Polynomial:
    return Polynomial(tuple(rational(c) for c in coefficients), variable)


def trimmed(coefficients: Sequence[object]) -> list[Fraction]:
    """The coefficients as Fractions without trailing zeros, [0] for the zero polynomial."""
    values = [fraction(c) for c in coefficients]
    while len(values) > 1 and not values[-1]:
        values.pop()
    return values or [Fraction(0)]


def evaluate(coefficients: Sequence[object], point: Number) -> Fraction:
    total = Fraction(0)
    for c in reversed(coefficients):
        total = total * point + fraction(c)
    return total


def through(points: Sequence[tuple[Number, Fraction]]) -> list[Fraction]:
    """The polynomial of degree below len(points) through the points, by Lagrange's formula."""
    result = [Fraction(0)] * len(points)
    for i, (xi, yi) in enumerate(points):
        basis, scale = [Fraction(1)], Fraction(1)
        for j, (xj, _) in enumerate(points):
            if j != i:
                basis = [Fraction(0)] + basis  # basis * (x - xj)
                for k in range(len(basis) - 1):
                    basis[k] -= xj * basis[k + 1]
                scale *= xi - xj
        for k, c in enumerate(basis):
            result[k] += yi * c / scale
    return result


def sum_reference(p: Sequence[object]) -> list[Fraction]:
    """The closed form F of p(1) + ... + p(n): the partial sums S(0), ..., S(d + 1), added up term by term, interpolated
    by Lagrange's formula; [0] for p = 0."""
    p = trimmed(p)
    if p == [0]:
        return [Fraction(0)]
    points, total = [(0, Fraction(0))], Fraction(0)
    for k in range(1, len(p) + 1):
        total += evaluate(p, k)
        points.append((k, total))
    return trimmed(through(points))


def geometric_terms(first: Number, ratio: Number, count: int) -> Fraction:
    """first + first*ratio + ... + first*ratio^(count - 1), term by term."""
    total, term, ratio = Fraction(0), Fraction(first), Fraction(ratio)
    for _ in range(count):
        total += term
        term *= ratio
    return total


def series_reference(first: Number, ratio: Number) -> Fraction:
    """first/(1 - ratio) for |ratio| < 1, confirmed as the limit of the partial sums by their exact tail:
    a/(1 - r) - S_n = a r^n/(1 - r)."""
    a, r = Fraction(first), Fraction(ratio)
    value = a / (1 - r)
    assert value - geometric_terms(a, r, 25) == a * r ** 25 / (1 - r)
    return value


def newton_series(p: Sequence[object], r: Number) -> Fraction:
    """sum_k p(k) r^k for |r| < 1 by Newton's series: p(k) = sum_j Delta^j p(0) binomial(k, j), and
    sum_k binomial(k, j) r^k = r^j/(1 - r)^(j + 1)."""
    r = Fraction(r)
    values = [evaluate(p, k) for k in range(len(p))]
    total = Fraction(0)
    for j in range(len(p)):
        total += values[0] * r ** j / (1 - r) ** (j + 1)
        values = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return total


def q_reference(p: Sequence[object], r: Number) -> list[Fraction]:
    """Q with r Q(x + 1) - Q(x) = p(x) and deg Q = deg p, for |r| < 1: -p for r = 0; otherwise the polynomial through
    Q(n) = (S_n + Q(0))/r^n, n = 0, ..., d, with S_n = sum_(k<n) p(k) r^k and Q(0) = -(the sum of the series): the
    identity telescopes the partial sums to S_n = Q(n) r^n - Q(0)."""
    p, r = trimmed(p), Fraction(r)
    if r == 0:
        return [-c for c in p]
    value = newton_series(p, r)
    points, partial = [], Fraction(0)
    for n in range(len(p)):
        points.append((n, (partial - value) / r ** n))
        partial += evaluate(p, n) * r ** n
    return trimmed(through(points))


def random_fraction(rng: random.Random, size: int = 40, denominators: int = 40) -> Fraction:
    return Fraction(rng.randrange(-size, size + 1), rng.randrange(1, denominators + 1))


def random_polynomial(rng: random.Random, degree: int, size: int = 40, denominators: int = 40) -> list[Fraction]:
    p = [random_fraction(rng, size, denominators) for _ in range(degree + 1)]
    while not p[-1]:
        p[-1] = random_fraction(rng, size, denominators)
    return p


def random_ratio(rng: random.Random) -> Fraction:
    """A ratio with |ratio| < 1: often a short fraction, sometimes 0 or one close to 1 in absolute value."""
    choice = rng.randrange(6)
    if choice == 0:
        return Fraction(0)
    if choice == 1:
        return Fraction(rng.choice((-1, 1)) * rng.randrange(90, 100), 100)
    denominator = rng.randrange(2, 13)
    return Fraction(rng.randrange(1 - denominator, denominator), denominator)


# The known values of the card.
F = Fraction
SUM_KNOWN = (((0, 1), (0, F(1, 2), F(1, 2))), ((0, 0, 1), (0, F(1, 6), F(1, 2), F(1, 3))),
             ((0, 0, 0, 1), (0, 0, F(1, 4), F(1, 2), F(1, 4))), ((1,), (0, 1)), ((0,), (0,)), ((-1, 2), (0, 0, 1)))
GEOMETRIC_SUM_KNOWN = ((1, 2, 10, 1023), (1, F(1, 2), 3, F(7, 4)), (5, 1, 4, 20), (3, 7, 0, 0), (2, -1, 5, 2))
GEOMETRIC_SERIES_KNOWN = ((1, F(1, 2), 2), (3, F(-1, 3), F(9, 4)), (0, 5, 0))
SERIES_KNOWN = (((0, 1), F(1, 2), 2), ((0, 0, 1), F(1, 2), 6), ((1,), F(1, 2), 2), ((0, 1), F(-1, 2), F(-2, 9)))

# One small request per operation, the SMALL_INPUTS of the card's integration notes where it names them.
SMALL = {
    "sum_polynomial": dict(polynomial=polynomial((0, 0, 1))),
    "geometric_sum": dict(first=1, ratio=2, count=10),
    "geometric_series": dict(first=1, ratio=Rational(1, 2)),
    "polynomial_geometric_series": dict(polynomial=polynomial((0, 1)), ratio=Rational(1, 2)),
}

# Inputs outside the domains of the card, each invalid input, with a part of the message.
INVALID = (
    ("geometric_sum", dict(first=1, ratio=2, count=-1), "count >= 0"),
    ("geometric_sum", dict(first=0, ratio=2, count=-1), "count >= 0"),
    ("geometric_sum", dict(first=Rational(1, 2), ratio=1, count=-5), "count >= 0"),
    ("geometric_sum", dict(first=1, ratio=0, count=-(2 ** 100)), "count >= 0"),
    ("geometric_series", dict(first=1, ratio=1), "the series diverges"),
    ("geometric_series", dict(first=1, ratio=-1), "the series diverges"),
    ("geometric_series", dict(first=1, ratio=2), "the series diverges"),
    ("geometric_series", dict(first=-4, ratio=5), "the series diverges"),
    ("geometric_series", dict(first=Rational(1, 3), ratio=Rational(7, 3)), "the series diverges"),
    ("geometric_series", dict(first=1, ratio=Rational(-3, 2)), "the series diverges"),
    ("polynomial_geometric_series", dict(polynomial=polynomial((0, 1)), ratio=1), "the series diverges"),
    ("polynomial_geometric_series", dict(polynomial=polynomial((0, 1)), ratio=-1), "the series diverges"),
    ("polynomial_geometric_series", dict(polynomial=polynomial((1,)), ratio=2), "the series diverges"),
    ("polynomial_geometric_series", dict(polynomial=polynomial((1, 1)), ratio=Rational(3, 2)), "the series diverges"),
    ("polynomial_geometric_series", dict(polynomial=polynomial((5,)), ratio=Rational(-7, 3)), "the series diverges"),
)


class SumsTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with its exact type and four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), type(value))
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertCanonical(self, value: object, evidence: tuple, args: dict[str, object]) -> None:
        """Every number of a value and its evidence is an exact Rational, never an int, and a Polynomial value lies in
        the very Symbol of the argument."""
        if type(value) is Polynomial:
            self.assertIs(value.variable, args["polynomial"].variable)
            self.assertTrue(all(type(c) is Rational for c in value.coefficients), value)
        else:
            self.assertIs(type(value), Rational)
        for item in evidence:
            self.assertIs(type(item), tuple)
            self.assertTrue(item and all(type(c) is Rational for c in item), evidence)

    def assertAgrees(self, operation: str, args: dict[str, object], value: object, evidence: tuple = ()) -> None:
        """produce() answers value with exactly this evidence, both canonical, and the checker accepts them."""
        produced, proof = family.produce(operation, dict(args))
        self.assertEqual((produced, proof), (value, evidence))
        self.assertCanonical(produced, proof, args)
        self.assertIs(support.verdict(family, operation, args, produced, proof), True)

    def estimate(self, operation: str, args: dict[str, object]) -> tuple[object, tuple, int]:
        """(value, evidence, the work that produce() computed for the checker), under the default limits."""
        with patch.object(family, "_refuse_unchecked", wraps=family._refuse_unchecked) as refuse:
            value, evidence = family.produce(operation, dict(args))
        return value, evidence, refuse.call_args.args[1]

    def checker_work(self, operation: str, args: dict[str, object], value: object, evidence: tuple, *,
                     limits: ComputationLimits | None = None) -> int:
        """What the checker charges when it accepts the claim in a fresh budget, as the verifier runs it."""
        with computation(limits, fresh=True) as budget:
            self.assertIs(support.checker(family).check_operation(operation, args, value, evidence), True)
        return budget.work

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for p, f in SUM_KNOWN:
            with self.subTest(operation="sum_polynomial", p=p):
                self.assertEqual(sum_reference(p), trimmed(f))  # The card's value, by the reference.
                result = support.run("sum_polynomial", polynomial=polynomial(p))
                self.assertVerified(result, polynomial(f))
                self.assertEqual(result.steps[0].outputs[1], ())
                self.roundtrip(result)
        for first, ratio, count, value in GEOMETRIC_SUM_KNOWN:
            with self.subTest(operation="geometric_sum", first=first, ratio=ratio, count=count):
                self.assertEqual(geometric_terms(first, ratio, count), value)
                result = support.run("geometric_sum", first=exact(first), ratio=exact(ratio), count=count)
                self.assertVerified(result, rational(value))  # Rational(1023), never the int 1023.
                self.assertEqual(result.steps[0].outputs[1], ())
                self.roundtrip(result)
        for first, ratio, value in GEOMETRIC_SERIES_KNOWN:
            with self.subTest(operation="geometric_series", first=first, ratio=ratio):
                if first:
                    self.assertEqual(series_reference(first, ratio), value)
                result = support.run("geometric_series", first=exact(first), ratio=exact(ratio))
                self.assertVerified(result, rational(value))
                self.assertEqual(result.steps[0].outputs[1], ())
                self.roundtrip(result)
        for p, ratio, value in SERIES_KNOWN:
            with self.subTest(operation="polynomial_geometric_series", p=p, ratio=ratio):
                self.assertEqual(newton_series(p, ratio), value)
                q = q_reference(p, ratio)
                self.assertEqual(-q[0], value)
                result = support.run("polynomial_geometric_series", polynomial=polynomial(p), ratio=exact(ratio))
                self.assertVerified(result, rational(value))
                self.assertEqual(result.steps[0].outputs[1], (tuple(map(rational, q)),))
                self.roundtrip(result)
        self.assertEqual(q_reference((0, 1), F(1, 2)), [-2, -2])  # (x, 1/2): Q = -2x - 2, so Q(0) = -2.
        # Every special case: a variable with a domain and an assumption, a constant, ratio 0 (the sum is p(0)), the
        # zero polynomial with a divergent ratio and ratio 1, first = 0 with a divergent ratio, and an int ratio.
        extra = (
            ("sum_polynomial", dict(polynomial=polynomial((0, 0, 1), N)), polynomial((0, F(1, 6), F(1, 2), F(1, 3)), N),
             ()),
            ("sum_polynomial", dict(polynomial=polynomial((F(-3, 7),))), polynomial((0, F(-3, 7))), ()),
            ("polynomial_geometric_series", dict(polynomial=polynomial((F(7, 2), -1, 4)), ratio=0), Rational(7, 2),
             ((Rational(-7, 2), Rational(1), Rational(-4)),)),
            ("polynomial_geometric_series", dict(polynomial=polynomial((0,)), ratio=5), Rational(0), ((Rational(0),),)),
            ("polynomial_geometric_series", dict(polynomial=polynomial((0,)), ratio=1), Rational(0), ((Rational(0),),)),
            ("geometric_series", dict(first=0, ratio=Rational(-9, 2)), Rational(0), ()),
            ("geometric_sum", dict(first=Rational(-7, 3), ratio=0, count=5), Rational(-7, 3), ()),
            ("geometric_sum", dict(first=4, ratio=-1, count=8), Rational(0), ()),
        )
        for operation, args, value, evidence in extra:
            with self.subTest(operation=operation, args=args):
                result = support.run(operation, **args)
                self.assertVerified(result, value)
                self.assertEqual(result.steps[0].outputs[1], evidence)
                self.assertCanonical(result.value, evidence, args)
                self.roundtrip(result)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the domains of the card, every Polynomial, Rational or Symbol made
        with object.__new__ outside the canonical form that run() admits by its class, and an oversized matrix
        (no argument takes one). produce() tests the types itself too."""
        others = (F(1, 2), Ratio(1, 2), Integer(1), X, polynomial((1, 2)).coefficients, support.oversized_matrix())
        for operation, args in SMALL.items():
            for name in args:
                for item in STAND_INS + others + ((polynomial((1, 2)),) if name != "polynomial" else ()):
                    with self.subTest(operation=operation, argument=name, item=type(item).__name__):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: item})))
            if "count" in args:
                with self.subTest(operation=operation, argument="count", item="Rational"):
                    self.assertInvalidInput(support.run(operation, **dict(args, count=Rational(10))))
        for operation, args, text in INVALID:
            with self.subTest(operation=operation, **{name: str(item) for name, item in args.items()}):
                result = support.run(operation, **args)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(args))
        # Records made with object.__new__ outside the canonical form: run() admits them by their class.
        cubic = polynomial((-1, 0, 0, 1))

        def forged_polynomial(coefficients: object = cubic.coefficients, variable: object = X) -> Polynomial:
            return support.forged(Polynomial, coefficients=coefficients, variable=variable)

        forged_polynomials = (
            ("a trailing zero", forged_polynomial(cubic.coefficients + (Rational(0),))),
            ("int coefficients", forged_polynomial((-1, 0, 0, 1))),
            ("a coefficient outside lowest terms", forged_polynomial(
                (support.forged(Rational, numerator=2, denominator=4), Rational(1)))),
            ("a negative denominator", forged_polynomial(
                (support.forged(Rational, numerator=1, denominator=-2), Rational(1)))),
            ("a Rational subclass coefficient", forged_polynomial((Ratio(1, 2), Rational(1)))),
            ("a str variable", forged_polynomial(variable="x")),
            ("a Symbol with an int name", forged_polynomial(variable=support.forged(
                Symbol, name=7, domain=X.domain, assumptions=(), uid=X.uid))),
            ("a Symbol with a str domain", forged_polynomial(variable=support.forged(
                Symbol, name="x", domain="reals", assumptions=(), uid=X.uid))),
            ("a Symbol with four assumptions", forged_polynomial(variable=support.forged(
                Symbol, name="n", domain=Integers, assumptions=N.assumptions * 4, uid=X.uid))),
            ("no coefficient", forged_polynomial(())),
            ("1,026 coefficients", forged_polynomial((Rational(1),) * (MAX_POLYNOMIAL_DEGREE + 2))),
        )
        for label, item in forged_polynomials:
            for operation, args in SMALL.items():
                if "polynomial" in args:
                    with self.subTest(operation=operation, forged=label):
                        result = support.run(operation, **dict(args, polynomial=item))
                        self.assertInvalidInput(result)
                        self.assertIn("canonical Polynomial", result.error.message)
        forged_rationals = (support.forged(Rational, numerator=10, denominator=2),
                            support.forged(Rational, numerator=-4, denominator=-2),
                            support.forged(Rational, numerator=5, denominator=0))
        for item in forged_rationals:
            for operation, args in SMALL.items():
                for name in ("first", "ratio"):
                    if name in args:
                        with self.subTest(operation=operation, argument=name, forged=(item.numerator,
                                                                                      item.denominator)):
                            result = support.run(operation, **dict(args, **{name: item}))
                            self.assertInvalidInput(result)
                            self.assertIn("lowest terms with a positive denominator", result.error.message)
        # produce() itself refuses every stand-in and each forged record, also those whose missing or float fields
        # run()'s inspection of the arguments meets first; other argument sets; other operations.
        missing = (support.forged(Polynomial, variable=X), support.forged(Polynomial, coefficients=cubic.coefficients),
                   forged_polynomial(list(cubic.coefficients)), forged_polynomial((0.0, -1.0)),
                   forged_polynomial(variable=support.forged(Symbol, name="x")),
                   support.forged(Rational, numerator=1), support.forged(Rational, numerator=1.0, denominator=2),
                   support.forged(Rational, numerator=True, denominator=1))
        for operation, args in SMALL.items():
            for name in args:
                for item in (*STAND_INS, *others, False, *forged_rationals, *missing,
                             *(forged for _, forged in forged_polynomials)):
                    with self.subTest(operation=operation, argument=name, item=type(item).__name__, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: item}))
            for wrong in (dict(args, extra=1), {}, tuple(args.items()), None, {**args, 1: 2}):
                with self.subTest(operation=operation, arguments=repr(wrong)[:40]), self.assertRaises(InvalidInput):
                    family.produce(operation, wrong)
        for operation in ("geometric", "Sum_polynomial", "sum_polynomial ", "continued_fraction", "", None, 1):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(SMALL["sum_polynomial"]))
        # The value function raises the structured path's error.
        for operation, args, _ in INVALID[:1] + INVALID[4:5] + INVALID[10:11]:
            with self.subTest(value_function=operation), self.assertRaises(mf.OperationError) as raised:
                support.value(operation, **args)
            self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in an int argument, also inside a Rational or a coefficient, and max_work=1 exceed
        the budget; no argument takes a matrix (support.oversized_matrix() is invalid input, test_invalid_inputs). The
        card's limit of geometric_sum, count * bits(ratio) > max_integer_bits, is refused at once; the ratios 1, 0 and
        -1, whose powers never grow, are exempt, and their sums are refused only when they do not fit."""
        huge = support.HUGE
        for operation, args in SMALL.items():
            for name in args:
                items = ((polynomial((huge, 1)), polynomial((F(1, huge), 1)), polynomial((1, 0, F(huge + 1, 7))))
                         if name == "polynomial" else (huge, -huge) if name == "count"
                         else (huge, Rational(huge + 1, 3), Rational(1, huge)))
                for item in items:
                    with self.subTest(operation=operation, argument=name):
                        self.assertResourceLimit(support.run(operation, **dict(args, **{name: item})))
                        with self.assertRaises(ResourceLimitError):
                            family.produce(operation, dict(args, **{name: item}))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
                with computation(ComputationLimits(max_work=1), fresh=True), self.assertRaises(ResourceLimitError):
                    family.produce(operation, dict(args))
        start = time.perf_counter()
        for first, ratio, count in ((1, 2, 10 ** 6), (1, F(-3, 2), 2 ** 40), (1, 2 ** 1000 + 1, 10 ** 4),
                                    (1, 2, 32769), (F(1, 3), F(2, 3), 32769), (1, 3, 2 ** 65535)):
            with self.subTest(first=first, ratio=ratio, count=count):
                result = support.run("geometric_sum", first=exact(first), ratio=exact(ratio), count=count)
                self.assertResourceLimit(result)
                self.assertIn("count * bits(ratio)", result.error.message)
        self.assertLess(time.perf_counter() - start, FAST)
        # At the limit, 3/2 to the power 32768 (count * 2 = max_integer_bits) is still formed.
        result = support.run("geometric_sum", first=1, ratio=Rational(3, 2), count=32768)
        self.assertVerified(result, rational(2 * (1 - F(3, 2) ** 32768) / -1))
        # The exempt ratios: count * first, first, first or 0, also for a count of 65,536 bits.
        big = 10 ** 40
        for first, ratio, count, value in ((F(2, 3), 1, big, F(2, 3) * big), (F(2, 3), -1, big + 1, F(2, 3)),
                                           (F(2, 3), -1, big, 0), (F(2, 3), 0, big, F(2, 3)),
                                           (1, 1, 2 ** 65535, 2 ** 65535), (-5, -1, 2 ** 65535 + 1, -5)):
            with self.subTest(first=first, ratio=ratio, exempt=count.bit_length()):
                self.assertVerified(support.run("geometric_sum", first=exact(first), ratio=exact(ratio), count=count),
                                    rational(value))
        # 3 * 2**65535 has 65,537 bits: no Rational of the budget holds the sum.
        self.assertResourceLimit(support.run("geometric_sum", first=3, ratio=1, count=2 ** 65535))
        # Coefficients that fit with a common denominator q1 q2 that does not: the checker refuses that argument, so
        # the producer refuses it too, also for ratio 0, where Q = -p fits and was once returned unverifiable.
        q1, q2 = 2 ** 40_000 + 1, 2 ** 40_000 - 1  # Coprime: both odd, 2 apart.
        wide = polynomial((F(1, q1), F(1, q2)))
        for operation, args, claim in (
                ("sum_polynomial", dict(polynomial=wide), (polynomial((0, 1, 1)), ())),
                ("polynomial_geometric_series", dict(polynomial=wide, ratio=0),
                 (Rational(1, q1), (tuple(-c for c in wide.coefficients),)))):
            with self.subTest(common_denominator=operation):
                result = support.run(operation, **args)
                self.assertResourceLimit(result)
                self.assertIn("common denominator of the polynomial", result.error.message)
                with self.assertRaises(ResourceLimitError):
                    family.produce(operation, dict(args))
                with computation(fresh=True), self.assertRaises(ResourceLimitError):  # Any claim of the right shape.
                    support.checker(family).check_operation(operation, args, *claim)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """Every claim of the family is unique: the closed form of a sum, the value of a series and its Q are fixed,
        and the evidence is () or Q. The checker accepts the true claims that the references of this module build,
        whatever form the arguments take (an int or a Rational with denominator 1, any Symbol), and the true sums
        that the producer refuses under the card's limit of geometric_sum."""
        rng = random.Random(20261201)
        for _ in range(40):
            p = random_polynomial(rng, rng.randrange(8))
            variable = rng.choice((X, N, Symbol("y")))
            args = {"polynomial": polynomial(p, variable)}
            with self.subTest(operation="sum_polynomial", p=p):
                self.assertAccepts("sum_polynomial", args, polynomial(sum_reference(p), variable))
            r = random_ratio(rng)
            q = q_reference(p, r)
            with self.subTest(operation="polynomial_geometric_series", p=p, r=r):
                self.assertAccepts("polynomial_geometric_series", {"polynomial": polynomial(p), "ratio": exact(r)},
                                   rational(-q[0]), (tuple(map(rational, q)),))
        for first, ratio, count in ((3, 2, 9), (Rational(3), Rational(2), 9), (Rational(-4), Rational(1), 6),
                                    (Rational(5, 2), Rational(0), 3), (7, Rational(-1), 7)):
            with self.subTest(operation="geometric_sum", first=first, ratio=ratio, count=count):
                value = rational(geometric_terms(fraction(first), fraction(ratio), count))
                args = {"first": first, "ratio": ratio, "count": count}
                self.assertAccepts("geometric_sum", args, value)
                self.assertEqual(support.run("geometric_sum", **args).value, value)
        for first, ratio in ((3, Rational(-1, 3)), (Rational(3), Rational(-1, 3)), (Rational(0), Rational(7))):
            with self.subTest(operation="geometric_series", first=first, ratio=ratio):
                value = rational(series_reference(fraction(first), fraction(ratio))) if first else Rational(0)
                self.assertAccepts("geometric_series", {"first": first, "ratio": ratio}, value)
        # Sums that fit, beyond the card's a priori limit: the producer refuses them, the checker verifies them.
        for count in (40_000, 65_535):
            args = {"first": 1, "ratio": 2, "count": count}
            with self.subTest(beyond_the_limit=count):
                self.assertResourceLimit(support.run("geometric_sum", **args))
                self.assertAccepts("geometric_sum", args, Rational(2 ** count - 1))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        giant = 2 ** 70_000  # Longer than max_integer_bits allows.
        claims = {
            "sum_polynomial": dict(polynomial=polynomial((F(1, 2), -1, 3))),
            "geometric_sum": dict(first=Rational(3, 2), ratio=Rational(-2, 3), count=7),
            "geometric_series": dict(first=3, ratio=Rational(-1, 3)),
            "polynomial_geometric_series": dict(polynomial=polynomial((F(1, 2), -3, 1)), ratio=Rational(-2, 5)),
        }
        for operation, args in claims.items():
            result = support.run(operation, **args)
            value, evidence = result.value, result.steps[0].outputs[1]
            self.assertVerified(result, value)
            self.assertAccepts(operation, args, value, evidence)
            cases: list[tuple[str, dict[str, object], object, object]] = []
            if operation == "sum_polynomial":
                f = list(value.coefficients)
                p = args["polynomial"]
                cases += [
                    ("a constant term", args, polynomial([1, *f[1:]]), evidence),
                    ("a wrong leading coefficient", args, polynomial([*f[:-1], 2 * fraction(f[-1])]), evidence),
                    ("the sum from k = 0", args, polynomial([fraction(p.coefficients[0]), *f[1:]]), evidence),
                    ("the sum in another Symbol named x", args, polynomial(f, Symbol("x")), evidence),
                    ("the sum in N", args, polynomial(f, N), evidence),
                    ("p itself", args, p, evidence),
                    ("a trailing zero", args, support.forged(Polynomial, coefficients=value.coefficients
                                                             + (Rational(0),), variable=X), evidence),
                    ("int coefficients", args, support.forged(Polynomial, coefficients=(0, 1, 1), variable=X),
                     evidence),
                    ("a giant coefficient", args, polynomial([*f[:-1], giant]), evidence),
                    ("the value as evidence", args, value, (value,)),
                    ("a Rational", args, Rational(1), evidence),
                    ("the coefficient tuple", args, value.coefficients, evidence),
                ]
            elif operation == "polynomial_geometric_series":
                q = list(evidence[0])
                cases += [
                    ("plus one", args, rational(fraction(value) + 1), evidence),
                    ("the opposite sign", args, -value, evidence),
                    ("Q + 1 and its value", args, rational(fraction(value) - 1),
                     (tuple(map(rational, [fraction(q[0]) + 1, *q[1:]])),)),
                    ("Q with another leading coefficient", args, value, (tuple(q[:-1]) + (2 * q[-1],),)),
                    ("Q as a list", args, value, (q,)),
                    ("Q unwrapped", args, value, tuple(q)),
                    ("Q twice", args, value, (tuple(q), tuple(q))),
                    ("Q of ints", args, value, (tuple(int(c.numerator) for c in q),)),
                    ("Q one entry short", args, value, (tuple(q[:-1]),)),
                    ("oversized Q", args, value, ((Rational(0),) * 100_000,)),
                    ("a Rational subclass value", args, Ratio(value.numerator, value.denominator), evidence),
                ]
            else:
                v = fraction(value)
                cases += [
                    ("plus one", args, rational(v + 1), evidence),
                    ("the opposite sign", args, rational(-v), evidence),
                    ("twice the value", args, rational(2 * v), evidence),
                    ("a Fraction", args, v, evidence),
                    ("a float", args, float(v), evidence),
                    ("a str", args, str(v), evidence),
                    ("a Rational subclass", args, Ratio(v.numerator, v.denominator), evidence),
                    ("a Rational outside lowest terms", args,
                     support.forged(Rational, numerator=3 * v.numerator, denominator=3 * v.denominator), evidence),
                    ("an impostor of Rational", args, support.impostor(value), evidence),
                    ("a giant value", args, Rational(giant), evidence),
                    ("first as a bool", dict(args, first=True), value, evidence),
                    ("ratio as a float", dict(args, ratio=float(fraction(args["ratio"]))), value, evidence),
                ]
                if operation == "geometric_sum":
                    cases += [("one term more", args, rational(geometric_terms(F(3, 2), F(-2, 3), 8)), evidence),
                              ("count as a bool", dict(args, count=True), value, evidence),
                              ("count as a float", dict(args, count=7.0), value, evidence),
                              ("count as a str", dict(args, count="7"), value, evidence),
                              ("count as a Rational", dict(args, count=Rational(7)), value, evidence)]
            if type(value) is Polynomial:
                cases += [("an impostor of Polynomial", args, support.impostor(value), evidence),
                          ("a Polynomial forged without its variable", args,
                           support.forged(Polynomial, coefficients=value.coefficients), evidence)]
            cases += [("evidence as a list", args, value, list(evidence)), ("evidence None", args, value, None),
                      ("an extra evidence item", args, value, evidence + ((),)),
                      ("oversized evidence", args, value, evidence + (0,) * 100_000)]
            if evidence:
                cases.append(("the evidence missing", args, value, ()))
            for label, claim_args, item, proof in cases:
                with self.subTest(operation=operation, case=label):
                    self.assertRejects(operation, claim_args, item, proof)
            for other in claims:  # The true claim under the name of another operation.
                if other != operation:
                    with self.subTest(operation=operation, renamed=other):
                        self.assertRejects(other, args, value, evidence)
            report = support.statuses(mf.verify(support.tampered(result, cases[0][2])))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = list(SMALL.items()) + [
            ("sum_polynomial", dict(polynomial=polynomial((F(1, 2), -1, 3), N))),
            ("geometric_sum", dict(first=Rational(5, 7), ratio=Rational(-3, 2), count=300)),
            ("geometric_series", dict(first=0, ratio=7)),
            ("polynomial_geometric_series", dict(polynomial=polynomial((0,)), ratio=-3)),
            ("polynomial_geometric_series", dict(polynomial=polynomial((F(1, 2), -3, 1)), ratio=Rational(-2, 5))),
        ]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, _), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("geometric_sum", first=1, ratio=2, count=10)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a Polynomial (also
        the zero polynomial) and a Rational, integral, fractional, negative and zero."""
        rendered = (
            ("sum_polynomial", dict(polynomial=polynomial((0, 0, 1))), "1/3*x^3 + 1/2*x^2 + 1/6*x",
             r"\frac{1}{3}\, x^{3} + \frac{1}{2}\, x^{2} + \frac{1}{6}\, x"),
            ("sum_polynomial", dict(polynomial=polynomial((-1, 2))), "x^2", "x^{2}"),
            ("sum_polynomial", dict(polynomial=polynomial((0, 0, 0, 1), N)), "1/4*n^4 + 1/2*n^3 + 1/4*n^2",
             r"\frac{1}{4}\, n^{4} + \frac{1}{2}\, n^{3} + \frac{1}{4}\, n^{2}"),
            ("sum_polynomial", dict(polynomial=polynomial((0,))), "0", "0"),
            ("geometric_sum", dict(first=1, ratio=2, count=10), "1023", "1023"),
            ("geometric_sum", dict(first=1, ratio=Rational(1, 2), count=3), "7/4", r"\frac{7}{4}"),
            ("geometric_series", dict(first=3, ratio=Rational(-1, 3)), "9/4", r"\frac{9}{4}"),
            ("geometric_series", dict(first=0, ratio=5), "0", "0"),
            ("polynomial_geometric_series", dict(polynomial=polynomial((0, 1)), ratio=Rational(-1, 2)), "-2/9",
             r"-\frac{2}{9}"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(operation=operation, text=text):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual((mf.render_text(result.value), mf.to_latex(result.value)), (text, latex))
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("geometric_sum", first=1, ratio=2, count=10).value)["data"],
                         {"type": "rational", "numerator": "1023", "denominator": "1"})
        data = mf.to_data(support.run("sum_polynomial", polynomial=polynomial((1,))).value)["data"]
        self.assertEqual(data["type"], "polynomial")
        self.assertEqual(data["coefficients"], [{"type": "rational", "numerator": "0", "denominator": "1"},
                                                {"type": "rational", "numerator": "1", "denominator": "1"}])
        self.assertEqual(data["variable"]["uid"], X.uid)
        evidence = mf.to_data(support.run("polynomial_geometric_series", **SMALL["polynomial_geometric_series"])
                              .steps[0].outputs[1])["data"]
        self.assertEqual(evidence["type"], "tuple")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path. On an
        input without a value the lean path falls back, and the value function raises the structured path's error."""
        rng = random.Random(20261202)
        cases: list[tuple[str, dict[str, object]]] = []
        for _ in range(12):
            cases.append(("sum_polynomial", dict(polynomial=polynomial(random_polynomial(rng, rng.randrange(9)),
                                                                       rng.choice((X, N))))))
            cases.append(("geometric_sum", dict(first=exact(random_fraction(rng)),
                                                ratio=exact(random_fraction(rng, 9, 9)), count=rng.randrange(40))))
            cases.append(("geometric_series", dict(first=exact(random_fraction(rng)), ratio=exact(random_ratio(rng)))))
            cases.append(("polynomial_geometric_series", dict(polynomial=polynomial(random_polynomial(rng, 5)),
                                                              ratio=exact(random_ratio(rng)))))
        for operation, args in cases:
            with self.subTest(operation=operation, args=args):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, **args), expected)
        failing = (("geometric_series", dict(first=1, ratio=1), "invalid_input"),
                   ("sum_polynomial", dict(polynomial=polynomial([0] * MAX_POLYNOMIAL_DEGREE + [1])),
                    "unsupported_operation"))
        for operation, args, code in failing:
            with self.subTest(failing=operation):
                self.assertIs(support.lean_value(operation, **args), lean.FALLBACK)
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(operation, **args)
                self.assertEqual(raised.exception.code, code)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: a polynomial travels as
        its text and variable, an exact rational as a JSON integer or a string "p/q", a count as a JSON integer."""
        requests = (
            ("sum_polynomial", dict(polynomial=polynomial((F(1, 2), -1, 3)))),
            ("geometric_sum", dict(first=Rational(3, 2), ratio=Rational(-2, 3), count=7)),
            ("geometric_series", dict(first=3, ratio=Rational(-1, 3))),
            ("polynomial_geometric_series", dict(polynomial=polynomial((0, 0, 1)), ratio=Rational(1, 2))),
        )
        for operation, args in requests:
            with self.subTest(operation=operation):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        raw = (
            ("geometric_sum", {"first": 1, "ratio": 2, "count": 10}, Rational(1023)),
            ("geometric_sum", {"first": "1", "ratio": "1/2", "count": 3}, Rational(7, 4)),
            ("geometric_series", {"first": 1, "ratio": "1/2"}, Rational(2)),
            ("polynomial_geometric_series", {"polynomial": {"text": "x", "variable": "x"}, "ratio": "-1/2"},
             Rational(-2, 9)),
        )
        for operation, request, value in raw:
            with self.subTest(operation=operation, request=request):
                result = self.answer(support.call(operation, request))
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call("sum_polynomial", {"polynomial": {"text": "k^2", "variable": "k"}}))
        self.assertEqual(result.value.coefficients, tuple(map(rational, (0, F(1, 6), F(1, 2), F(1, 3)))))
        self.assertEqual(result.value.variable.name, "k")
        self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("geometric_series", {"first": 1, "ratio": 2}), "invalid_input")
        self.assertIn("the series diverges", error["message"])
        error = self.assertToolError(support.call("geometric_sum", {"first": 1, "ratio": 2, "count": -1}),
                                     "invalid_input")
        self.assertIn("count >= 0", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 200 seeded inputs per operation, and the
        checker accepts every answer with its evidence."""
        rng = random.Random(20261203)
        counts = dict.fromkeys(SMALL, 0)
        while counts["sum_polynomial"] < 220:
            degree = rng.randrange(13)
            p = random_polynomial(rng, degree, rng.choice((9, 40, 10 ** 12)), rng.choice((1, 12, 10 ** 6)))
            variable = rng.choice((X, N, Symbol("t")))
            f = sum_reference(p)
            with self.subTest(operation="sum_polynomial", p=p):
                self.assertEqual(len(f), degree + 2)
                self.assertEqual([evaluate(f, n) for n in range(degree + 6)],  # The partial sums, beyond the nodes.
                                 [sum((evaluate(p, k) for k in range(1, n + 1)), Fraction(0))
                                  for n in range(degree + 6)])
                self.assertAgrees("sum_polynomial", {"polynomial": polynomial(p, variable)}, polynomial(f, variable))
                counts["sum_polynomial"] += 1
        while counts["geometric_sum"] < 220:
            first = random_fraction(rng, rng.choice((9, 10 ** 9)), rng.choice((1, 9, 10 ** 9)))
            ratio = rng.choice((random_fraction(rng, 9, 9), Fraction(rng.choice((-1, 0, 1))),
                                random_fraction(rng, 10 ** 6, 10 ** 6)))
            count = rng.choice((0, 1, 2, rng.randrange(60), rng.randrange(300)))
            args = {"first": exact(first), "ratio": exact(ratio), "count": count}
            with self.subTest(operation="geometric_sum", **{name: str(item) for name, item in args.items()}):
                self.assertAgrees("geometric_sum", args, rational(geometric_terms(first, ratio, count)))
                counts["geometric_sum"] += 1
        while counts["geometric_series"] < 220:
            first = random_fraction(rng, rng.choice((9, 10 ** 9)), rng.choice((1, 9, 10 ** 9)))
            ratio = random_ratio(rng)
            if not rng.randrange(10):  # first = 0 gives 0 for every ratio, also a divergent one.
                first, ratio = Fraction(0), random_fraction(rng, 30, 3)
            args = {"first": exact(first), "ratio": exact(ratio)}
            value = series_reference(first, ratio) if first else Fraction(0)
            with self.subTest(operation="geometric_series", **{name: str(item) for name, item in args.items()}):
                self.assertAgrees("geometric_series", args, rational(value))
                counts["geometric_series"] += 1
        while counts["polynomial_geometric_series"] < 220:
            degree = rng.randrange(9)
            p = random_polynomial(rng, degree, rng.choice((9, 40, 10 ** 9)), rng.choice((1, 12, 10 ** 6)))
            r = random_ratio(rng)
            q = q_reference(p, r)
            with self.subTest(operation="polynomial_geometric_series", p=p, r=r):
                # The reference's own check: the identity at d + 2 points and the telescoped partial sums.
                self.assertTrue(all(r * evaluate(q, k + 1) - evaluate(q, k) == evaluate(p, k)
                                    for k in range(degree + 2)))
                self.assertEqual(sum((evaluate(p, k) * r ** k for k in range(12)), Fraction(0)),
                                 evaluate(q, 12) * r ** 12 - q[0])
                self.assertAgrees("polynomial_geometric_series", {"polynomial": polynomial(p), "ratio": exact(r)},
                                  rational(newton_series(p, r)), (tuple(map(rational, q)),))
                counts["polynomial_geometric_series"] += 1
        # The zero polynomial with any ratio.
        for r in (0, 1, -1, 5, F(1, 2), F(-7, 3)):
            with self.subTest(zero_polynomial=r):
                self.assertAgrees("polynomial_geometric_series", {"polynomial": polynomial((0,)), "ratio": exact(r)},
                                  Rational(0), ((Rational(0),),))
        self.assertTrue(all(count >= 200 for count in counts.values()), counts)

    def test_full_size_inputs_fit_the_default_budget(self):
        """The sum of x^1023 (degree 1024, the largest Polynomial) and the series of a dense polynomial of degree 1024
        complete under the default limits, verified; the sum of a polynomial of degree 1024, whose closed form has
        degree 1025, is beyond every Polynomial: unsupported_operation."""
        f = sum_reference([0, 0, 0, 0, 1])  # A small power by the reference: the producer agrees on it.
        self.assertAgrees("sum_polynomial", {"polynomial": polynomial([0, 0, 0, 0, 1])}, polynomial(f))
        start = time.perf_counter()
        result = support.run("sum_polynomial", polynomial=polynomial([0] * 1023 + [1]))
        self.assertLess(time.perf_counter() - start, 3.0)  # About 0.3 seconds.
        self.assertTrue(mf.verify(result).verified)
        value = result.value
        self.assertEqual(len(value.coefficients), 1025)
        self.assertEqual(value.coefficients[-1], Rational(1, 1024))  # The leading term of Faulhaber's formula ...
        self.assertEqual(value.coefficients[-2], Rational(1, 2))  # ... and the next one, n^1023/2.
        self.assertEqual([evaluate(value.coefficients, n) for n in range(4)],
                         [0, 1, 1 + 2 ** 1023, 1 + 2 ** 1023 + 3 ** 1023])
        rng = random.Random(20261204)
        p = [rng.randrange(-99, 100) for _ in range(1024)] + [1]
        start = time.perf_counter()
        result = support.run("polynomial_geometric_series", polynomial=polynomial(p), ratio=Rational(1, 2))
        self.assertLess(time.perf_counter() - start, 3.0)
        self.assertTrue(mf.verify(result).verified)
        q = result.steps[0].outputs[1][0]
        self.assertEqual((len(q), q[-1]), (1025, Rational(-2)))  # Q_d = p_d/(r - 1).
        top = polynomial([0] * MAX_POLYNOMIAL_DEGREE + [1])
        result = support.run("sum_polynomial", polynomial=top)
        self.assertIs(result.execution_status, ExecutionStatus.UNSUPPORTED)
        self.assertEqual(result.error.code, "unsupported_operation")
        self.assertIn("degree 1025", result.error.message)
        with self.assertRaises(UnsupportedOperation):
            family.produce("sum_polynomial", {"polynomial": top})
        self.assertEqual(len(support.run("polynomial_geometric_series", polynomial=top, ratio=0)
                             .steps[0].outputs[1][0]), 1025)  # Q = -p has degree 1024 too.

    def test_estimate_of_the_check_is_exact(self):
        """Before it returns a claim, produce() computes the work that the checker will charge for it: on seeded
        claims of every operation, exactly the work that the checker charges in a fresh budget of the default limits,
        for long coefficients and denominators, every special ratio and long counts."""
        rng = random.Random(20261205)
        cases: list[tuple[str, dict[str, object]]] = []
        for _ in range(25):
            p = random_polynomial(rng, rng.choice((0, 1, 2, 5, 13, 40)), rng.choice((9, 2 ** 200, 2 ** 2000)),
                                  rng.choice((1, 7, 2 ** 100, 2 ** 1500)))
            cases.append(("sum_polynomial", {"polynomial": polynomial(p, rng.choice((X, N)))}))
            ratio = rng.choice((random_ratio(rng), Fraction(rng.randrange(-2 ** 200, 2 ** 200), 2 ** 200 + 1)))
            cases.append(("polynomial_geometric_series", {"polynomial": polynomial(p), "ratio": exact(ratio)}))
        for _ in range(60):
            first = random_fraction(rng, rng.choice((9, 2 ** 500)), rng.choice((1, 9, 2 ** 300)))
            ratio = random_fraction(rng, rng.choice((1, 3, 40, 2 ** 64)), rng.choice((1, 5, 2 ** 64)))
            count = rng.choice((0, 1, 2, 3, rng.randrange(100), rng.randrange(1000)))
            if ratio in (0, 1, -1) or count * max(abs(ratio.numerator).bit_length(),
                                                  ratio.denominator.bit_length()) <= 65_536:
                cases.append(("geometric_sum", {"first": exact(first), "ratio": exact(ratio), "count": count}))
            cases.append(("geometric_series", {"first": exact(first), "ratio": exact(random_ratio(rng))}))
        cases += [("geometric_sum", {"first": 1, "ratio": 1, "count": 2 ** 65535}),
                  ("geometric_sum", {"first": Rational(2, 3), "ratio": -1, "count": 10 ** 40}),
                  ("geometric_sum", {"first": Rational(-7, 3), "ratio": 0, "count": 2 ** 5000 + 12345}),
                  ("geometric_sum", {"first": 1, "ratio": 3, "count": 30_000}),
                  ("sum_polynomial", {"polynomial": polynomial([2 ** 20_000] + [0] * 199 + [1])}),
                  ("polynomial_geometric_series", {"polynomial": polynomial([F(1, 3 ** 900), 0, 7]),
                                                   "ratio": Rational(-5, 2 ** 300 + 1)})]
        compared = dict.fromkeys(SMALL, 0)
        for operation, args in cases:
            with self.subTest(operation=operation, args=sorted(args)):
                try:
                    value, evidence, work = self.estimate(operation, args)
                except ResourceLimitError:  # Long coefficients at the highest degrees exceed the default budget.
                    continue
                self.assertEqual(work, self.checker_work(operation, args, value, evidence))
                compared[operation] += 1
        self.assertTrue(all(count >= 20 for count in compared.values()), compared)

    def test_claims_beyond_the_check_are_refused(self):
        """A claim whose check needs more than max_work is the producer's to refuse, so that both sides fail at the
        same point. The checker's shift is charged by its longest integer, the producer's passes by the length of all
        of them, so a long constant term makes a claim cheap to produce and dear to check; geometric_sum's binary
        splitting costs its checker more than the closed form costs the producer. With max_work equal to the check,
        the request verifies; one unit less, the producer refuses the claim it could still make."""
        claims = (
            ("sum_polynomial", dict(polynomial=polynomial([2 ** 20_000] + [0] * 199 + [1]))),
            ("polynomial_geometric_series", dict(polynomial=polynomial([2 ** 30_000] + [0] * 99 + [1]),
                                                 ratio=Rational(-1, 3))),
            ("geometric_sum", dict(first=1, ratio=3, count=30_000)),
        )
        wide = ComputationLimits(max_work=10 ** 9)
        for operation, args in claims:
            with self.subTest(operation=operation):
                with computation(wide, fresh=True) as budget:
                    value, evidence = family.produce(operation, dict(args))
                produced = budget.work
                checked = self.checker_work(operation, args, value, evidence)
                self.assertLess(produced, checked)  # The producer alone would get through max_work = checked - 1.
                self.assertVerified(support.run(operation, limits=ComputationLimits(max_work=checked), **args), value)
                result = support.run(operation, limits=ComputationLimits(max_work=checked - 1), **args)
                self.assertResourceLimit(result)
                self.assertIn(UNCHECKABLE, result.error.message)
        # Under the default limits: x^800 with a constant term of 60,000 bits takes the producer about 0.8 million
        # units and would take its check 2.34 million; the producer refuses it at once.
        start = time.perf_counter()
        for operation, args in (("sum_polynomial", dict(polynomial=polynomial([2 ** 60_000] + [0] * 799 + [1]))),
                                ("polynomial_geometric_series",
                                 dict(polynomial=polynomial([2 ** 60_000] + [0] * 799 + [1]), ratio=Rational(1, 2)))):
            with self.subTest(default_limits=operation):
                result = support.run(operation, **args)
                self.assertResourceLimit(result)
                self.assertIn(UNCHECKABLE, result.error.message)
        self.assertLess(time.perf_counter() - start, 2 * FAST)

    def test_work_is_charged_before_it_runs(self):
        """Every pass is charged before it runs, so a request that needs more than its budget fails at once; produce()
        alone charges at least two units for every input, so max_work=1 refuses every request."""
        rng = random.Random(20261206)
        dense = polynomial([rng.randrange(-99, 100) for _ in range(600)] + [1])
        start = time.perf_counter()
        for operation, args in (("sum_polynomial", dict(polynomial=dense)),
                                ("polynomial_geometric_series", dict(polynomial=dense, ratio=Rational(1, 3)))):
            with self.subTest(operation=operation):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=20_000), **args))
        self.assertLess(time.perf_counter() - start, FAST)
        for operation, args in [*SMALL.items(), ("geometric_sum", dict(first=0, ratio=2, count=0)),
                                ("polynomial_geometric_series", dict(polynomial=polynomial((0,)), ratio=3))]:
            with self.subTest(operation=operation, args=args), computation(ComputationLimits(max_work=10 ** 9),
                                                                           fresh=True) as budget:
                family.produce(operation, dict(args))
                self.assertGreaterEqual(budget.work, 2)

    def test_producer_computes_no_builtin_pow(self):
        """The powers of geometric_sum come from the producer's own chain of charged products: nothing in the family
        module calls the builtin pow or uses the power operator, whose work could not be interrupted."""
        tree = ast.parse(inspect.getsource(family))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        self.assertIn("_chain", called)
        self.assertEqual([node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                          and isinstance(node.op, ast.Pow)], [])
