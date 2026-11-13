"""The independent checker of mathforge.areas.stats.bivariate (mathforge.certify.stats.bivariate), tested on its own
before the family's producer exists.

Every true claim is computed here, never by the producer: the centred sums by their definition and again from raw
moments, with fractions.Fraction (two references that check each other); the correlation coefficient in its canonical
form built by hand and compared with what mathforge.model.mul(sign, mathforge.model.sqrt(q)) builds, the card's
construction; the least-squares line from the centred sums and again by Cramer's rule on the normal equations. The
evidence is always (), the card's layout. The checker answers False to every corrupted claim, never with an exception,
and refuses an oversized forgery within FAST seconds under the default budget.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from fractions import Fraction
import inspect
from math import gcd, isqrt
import random
import time
from types import MappingProxyType
from typing import Any

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.stats import bivariate as family
from mathforge.areas.stats.bivariate import LinearFit
from mathforge.certify.stats import bivariate as checker
from mathforge.contracts import CheckStatus
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Rational, Sqrt, mul, sqrt
from mathforge.registry import Argument, OperationSpec
from mathforge.trusted.integers import product_work

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


def arguments(xs: Iterable[Number], ys: Iterable[Number]) -> dict[str, object]:
    """The arguments of both operations as a request holds them: tuples of ints and Rationals."""
    return {"xs": tuple(exact(x) for x in xs), "ys": tuple(exact(y) for y in ys)}


def centred_sums(xs: Sequence[Number], ys: Sequence[Number]) -> tuple[Fraction, Fraction, Fraction]:
    """(S_xx, S_yy, S_xy) by their definition: the means first, then the products of the deviations."""
    fx, fy = [frac(x) for x in xs], [frac(y) for y in ys]
    mx, my = sum(fx, Fraction(0)) / len(fx), sum(fy, Fraction(0)) / len(fy)
    return (sum(((x - mx) * (x - mx) for x in fx), Fraction(0)), sum(((y - my) * (y - my) for y in fy), Fraction(0)),
            sum(((x - mx) * (y - my) for x, y in zip(fx, fy)), Fraction(0)))


def moment_sums(xs: Sequence[Number], ys: Sequence[Number]) -> tuple[Fraction, Fraction, Fraction]:
    """The same sums from the raw moments: S_xy = sum x*y - (sum x)(sum y)/n, and so on."""
    fx, fy = [frac(x) for x in xs], [frac(y) for y in ys]
    n, sx, sy = len(fx), sum(fx, Fraction(0)), sum(fy, Fraction(0))
    return (sum((x * x for x in fx), Fraction(0)) - sx * sx / n, sum((y * y for y in fy), Fraction(0)) - sy * sy / n,
            sum((x * y for x, y in zip(fx, fy)), Fraction(0)) - sx * sy / n)


def square_root(number: Fraction) -> Fraction | None:
    """The rational square root of a Fraction >= 0, None when it has none."""
    p, q = isqrt(number.numerator), isqrt(number.denominator)
    return Fraction(p, q) if p * p == number.numerator and q * q == number.denominator else None


def signed_radical(sign: int, q: Fraction) -> Any:
    """sign * sqrt(q) in the card's canonical form, built by hand: a Rational when q is a rational square, else
    Sqrt(q) for sign 1 and Mul((Rational(-1), Sqrt(q))) for sign -1."""
    root = square_root(q)
    if root is not None:
        return rational(sign * root)
    radical = Sqrt(rational(q))
    return radical if sign > 0 else Mul((Rational(-1), radical))


def correlation_of(xs: Sequence[Number], ys: Sequence[Number]) -> Any:
    """The canonical Pearson coefficient sign(S_xy) sqrt(S_xy^2 / (S_xx S_yy)); both references agree on the sums,
    and the hand-built form is what mathforge.model.mul(sign, mathforge.model.sqrt(q)) builds."""
    sums = centred_sums(xs, ys)
    assert sums == moment_sums(xs, ys)
    sxx, syy, sxy = sums
    assert sxx > 0 and syy > 0
    q = sxy * sxy / (sxx * syy)
    sign = (sxy > 0) - (sxy < 0)
    value = signed_radical(sign, q)
    assert value == mul(sign, sqrt(rational(q)))  # The card's construction of the canonical value.
    return value


def correlated(xs: Sequence[Number], ys: Sequence[Number]) -> tuple[dict[str, object], Any]:
    """(args, value) of a true correlation claim."""
    return arguments(xs, ys), correlation_of(xs, ys)


def negative(value: Any) -> Any:
    """-r in the canonical form, for r in the canonical form."""
    if type(value) is Rational:
        return -value
    if type(value) is Sqrt:
        return Mul((Rational(-1), value))
    return value.factors[1]


def primes_below(limit: int) -> list[int]:
    """The primes below limit, by Eratosthenes' sieve."""
    flags = bytearray([1]) * limit
    flags[:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return [n for n in range(limit) if flags[n]]


def fit_of(xs: Sequence[Number], ys: Sequence[Number]) -> LinearFit:
    """The least-squares line: slope S_xy / S_xx and intercept mean(ys) - slope * mean(xs), equal to the solution of
    the normal equations [[sum x^2, sum x], [sum x, n]] (slope, intercept) = (sum x*y, sum y) by Cramer's rule."""
    fx, fy = [frac(x) for x in xs], [frac(y) for y in ys]
    n = len(fx)
    sxx, _, sxy = centred_sums(fx, fy)
    slope = sxy / sxx
    intercept = sum(fy, Fraction(0)) / n - slope * sum(fx, Fraction(0)) / n
    sx, sy = sum(fx, Fraction(0)), sum(fy, Fraction(0))
    squares, products = sum((x * x for x in fx), Fraction(0)), sum((x * y for x, y in zip(fx, fy)), Fraction(0))
    determinant = n * squares - sx * sx
    assert determinant > 0
    assert slope == (n * products - sx * sy) / determinant
    assert intercept == (squares * sy - sx * products) / determinant
    return LinearFit(rational(slope), rational(intercept))


def fitted(xs: Sequence[Number], ys: Sequence[Number]) -> tuple[dict[str, object], LinearFit]:
    """(args, value) of a true linear_regression claim."""
    return arguments(xs, ys), fit_of(xs, ys)


def squared_error(xs: Sequence[Number], ys: Sequence[Number], slope: Number, intercept: Number) -> Fraction:
    """sum (y_i - slope x_i - intercept)^2, the quantity that the least-squares line minimizes."""
    a, b = frac(slope), frac(intercept)
    return sum(((frac(y) - a * frac(x) - b) ** 2 for x, y in zip(xs, ys)), Fraction(0))


def residuals(xs: Sequence[Number], ys: Sequence[Number], fit: LinearFit) -> tuple[Rational, ...]:
    """The residuals y_i - (slope x_i + intercept) of a line, as Rationals."""
    a, b = frac(fit.slope), frac(fit.intercept)
    return tuple(rational(frac(y) - a * frac(x) - b) for x, y in zip(xs, ys))


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


def seeded_number(rng: random.Random, bits: int) -> Fraction:
    """A random exact rational: an integer of up to bits bits, or a fraction of two such integers."""
    numerator = rng.randrange(-(2 ** bits), 2 ** bits + 1)
    return Fraction(numerator) if rng.random() < 0.5 else Fraction(numerator, rng.randrange(1, 2 ** bits + 1))


def varying(rng: random.Random, n: int, bits: int) -> list[Fraction]:
    """n random exact rationals that are not all equal."""
    while True:
        values = [seeded_number(rng, bits) for _ in range(n)]
        if len(set(values)) > 1:
            return values


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values and arguments that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Fractional(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Radical(Sqrt):
    """A Sqrt subclass: the right radicand, another class."""


class Product(Mul):
    """A Mul subclass: the right factors, another class."""


class SubFit(LinearFit):
    """A LinearFit subclass: the right fields, another class."""


class Values(tuple):
    """A tuple subclass for a data tuple, the factors of a product or the evidence."""


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def forged_fit(slope: object, intercept: object) -> LinearFit:
    """A LinearFit made with object.__new__, so that its fields may break the canonical form."""
    return support.forged(LinearFit, slope=slope, intercept=intercept)


def inexact(number: Number) -> list[tuple[str, object]]:
    """(label, stand-in) for an exact number that is no exact int and no canonical Rational; several are equal to it
    as numbers."""
    p, q = frac(number).numerator, frac(number).denominator
    return [("a bool", bool(p)), ("a float", p / q), ("a str", str(number)), ("None", None),
            ("a Fraction", Fraction(p, q)), ("an int subclass", Integer(p)),
            ("a Rational subclass", Fractional(p, q)), ("a list", [number]), ("a tuple", (number,)),
            ("a Rational of a bool", forged_rational(bool(p), 1)), ("a Rational of a float", forged_rational(p / q, 1)),
            ("a Rational with a float denominator", forged_rational(p, float(q))),
            ("a Rational in higher terms", forged_rational(2 * p, 2 * q)),
            ("a Rational with a negative denominator", forged_rational(-p, -q)),
            ("a Rational with a zero denominator", forged_rational(p, 0)),
            ("a Rational without a denominator", support.forged(Rational, numerator=p)),
            ("a Rational without a numerator", support.forged(Rational, denominator=q))]


# The datasets of the card and of the benchmark probes, with their values.
LINE = ((1, 2, 3), (2, 4, 6))  # r = 1, the line y = 2*x.
FALLING = ((1, 2, 3), (3, 2, 1))  # r = -1.
CARD = ((1, 2, 3, 4), (1, 3, 2, 4))  # r = 4/5: q = 16/25 is a rational square.
PROBE = ((1, 2, 3), (2, 4, 7))  # r = sqrt(75/76), and the line y = 5/2*x - 2/3.
MIRROR = ((1, 2, 3), (7, 4, 2))  # r = -sqrt(75/76).
FLAT = ((1, 2, 3), (1, 0, 1))  # S_xy = 0: r = 0, and the line y = 2/3.
Q = Rational(75, 76)


class BivariateCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Iterable[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    def assertAcceptsAll(self, operation: str, cases: Iterable[tuple[str, object, object]]) -> None:
        """The checker answers True to each (label, args, value) case, with the evidence ()."""
        for label, args, value in cases:
            with self.subTest(operation=operation, case=label):
                self.assertAccepts(operation, args, value)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        data = (Argument("xs", "", None, "rational_list"), Argument("ys", "", None, "rational_list"))
        expected = (
            OperationSpec("correlation", "stats", data, "Expression",
                          "Compute the exact Pearson correlation coefficient of paired rational data.", SCOPE,
                          version=6),
            OperationSpec("linear_regression", "stats", data, "LinearFit",
                          "Fit the exact least-squares line y = slope*x + intercept to paired rational data.", SCOPE,
                          version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.SCOPE, SCOPE)
        self.assertEqual(family.TYPES, {"linear_fit": LinearFit})
        self.assertEqual(set(family.RENDER), {LinearFit})
        self.assertEqual(family.CHECKER, "mathforge.certify.stats.bivariate")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)
        # The checker names the record without importing the family module.
        self.assertEqual(checker.LINEAR_FIT, (LinearFit.__module__, LinearFit.__qualname__))

    def test_linear_fit_keeps_its_canonical_form(self):
        """The record takes two canonical Rationals only, and renders as the card shows: y = 2*x + 1."""
        LinearFit(Rational(2), Rational(1))
        LinearFit(forged_rational(3, 2), Rational(-5, 6))  # Forging a canonical Rational keeps it canonical.
        for label, item in inexact(2) + [("an int", 2)]:
            for fields in ((item, Rational(1)), (Rational(1), item)):
                with self.subTest(case=label, slope=fields[0] is item):
                    with self.assertRaises(InvalidInput):
                        LinearFit(*fields)
        with self.assertRaises(ResourceLimitError):
            LinearFit(Rational(GIANT), Rational(1))
        cases = [
            ((2, 1), "y = 2*x + 1", r"y = 2\, x + 1"),
            ((Fraction(3, 2), Fraction(5, 6)), "y = 3/2*x + 5/6", r"y = \frac{3}{2}\, x + \frac{5}{6}"),
            ((Fraction(5, 2), Fraction(-2, 3)), "y = 5/2*x - 2/3", r"y = \frac{5}{2}\, x - \frac{2}{3}"),
            ((-2, Fraction(-1, 3)), "y = -2*x - 1/3", r"y = -2\, x - \frac{1}{3}"),
            ((Fraction(-7, 3), -1), "y = -7/3*x - 1", r"y = -\frac{7}{3}\, x - 1"),
            ((1, 0), "y = x", "y = x"),
            ((-1, 4), "y = -x + 4", "y = -x + 4"),
            ((2, 0), "y = 2*x", r"y = 2\, x"),
            ((0, 5), "y = 5", "y = 5"),
            ((0, Fraction(-2, 3)), "y = -2/3", r"y = -\frac{2}{3}"),
            ((0, 0), "y = 0", "y = 0"),
        ]
        for (slope, intercept), text, latex in cases:
            with self.subTest(text=text):
                fit = LinearFit(rational(slope), rational(intercept))
                self.assertEqual(mf.render_text(fit), text)
                self.assertEqual(mf.to_latex(fit), latex)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_correlation_accepts_the_known_values_of_the_card(self):
        cases = [(LINE, Rational(1)), (FALLING, Rational(-1)), (CARD, Rational(4, 5)), (PROBE, Sqrt(Q)),
                 (MIRROR, Mul((Rational(-1), Sqrt(Q)))), (FLAT, Rational(0))]
        for (xs, ys), value in cases:
            with self.subTest(xs=xs, ys=ys):
                self.assertEqual(correlation_of(xs, ys), value)
                self.assertAccepts("correlation", arguments(xs, ys), value)
        # The benchmark probe expects sqrt(75/76) exactly, the radicand reduced and nothing taken out of the radical.
        self.assertEqual(centred_sums(*PROBE), (Fraction(2), Fraction(38, 3), Fraction(5)))

    def test_correlation_accepts_special_cases(self):
        half, third = Fraction(1, 2), Fraction(1, 3)
        cases = [
            ("r = 1/2", (-1, 0, 1), (0, 2, 1)),
            ("r = -1/2", (-1, 0, 1), (1, 2, 0)),
            ("two points rising", (0, 5), (1, 3)),
            ("two points falling", (half, third), (2, 7)),
            ("rational data", (half, -third, Fraction(7, 4), 2, Fraction(-5, 6)), (third, 1, Fraction(-2, 9), 0, 5)),
            ("unsorted data", (5, -3, 0, 2, 9, -1), (1, 4, 4, -2, 7, 0)),
            ("repeated pairs", (1, 1, 2, 2, 3), (2, 2, 3, 5, 5)),
            ("a negative coefficient", (0, 1, 2, 3), (5, 3, 4, -1)),
            ("long integers", (2 ** 200, -(2 ** 150) + 7, 3 ** 90, 1), (5, 2 ** 300 + 1, -(7 ** 80), 11)),
            ("long denominators", (Fraction(1, 2 ** 100), Fraction(3, 3 ** 50), 1), (Fraction(2, 5 ** 40), 0, 7)),
            ("different denominators in xs and ys", (half, third, Fraction(1, 5)),
             (Fraction(1, 7), 1, Fraction(2, 11))),
            ("linear rational data", (half, third, 4, -2), tuple(Fraction(3, 7) * x - 2 for x in (half, third, 4, -2))),
            ("a falling line", (1, 2, 3, 4, 5), (10, 8, 6, 4, 2)),
            ("a hundred and one points on a parabola", tuple(range(-50, 51)), tuple(x * x for x in range(-50, 51))),
            ("a thousand points", tuple(range(1000)), tuple((7 * k) % 101 for k in range(1000))),
        ]
        for label, xs, ys in cases:
            with self.subTest(case=label):
                self.assertAccepts("correlation", *correlated(xs, ys))
        self.assertEqual(correlation_of((-1, 0, 1), (0, 2, 1)), Rational(1, 2))
        self.assertEqual(correlation_of(tuple(range(-50, 51)), tuple(x * x for x in range(-50, 51))), Rational(0))
        # Ints and Rationals, also with denominator 1, are the same exact numbers in a request.
        mixed = {"xs": (Rational(1), 2, Rational(3), 4), "ys": (1, Rational(3), 2, Rational(4))}
        self.assertAccepts("correlation", mixed, Rational(4, 5))
        # The data in the other order, scaled by a positive number or shifted, has the same coefficient; a negative
        # scale changes its sign.
        xs, ys = (half, 3, -2, Fraction(9, 4), 0), (1, Fraction(-4, 5), 6, 2, third)
        value = correlation_of(xs, ys)
        self.assertIs(type(value), Mul)
        for label, other_xs, other_ys, expected in (
                ("reversed", xs[::-1], ys[::-1], value),
                ("scaled", [Fraction(3, 7) * frac(x) for x in xs], ys, value),
                ("shifted", [frac(x) - 10 for x in xs], [frac(y) + Fraction(1, 3) for y in ys], value),
                ("negated", [-frac(x) for x in xs], ys, negative(value)),
                ("swapped", ys, xs, value)):
            with self.subTest(case=label):
                self.assertEqual(correlation_of(other_xs, other_ys), expected)
                self.assertAccepts("correlation", arguments(other_xs, other_ys), expected)

    def test_correlation_accepts_seeded_true_claims(self):
        rng = random.Random(20261112)
        shapes = {Rational: 0, Sqrt: 0, Mul: 0}
        for case in range(300):
            n = rng.randrange(2, 12)
            bits = rng.choice((2, 3, 8, 40))
            xs = varying(rng, n, bits)
            if rng.random() < 0.15:  # Exactly linear data: r = 1 or -1.
                slope = seeded_number(rng, 4) or Fraction(1)
                ys = [slope * x + seeded_number(rng, 4) for x in xs]
            else:
                ys = varying(rng, n, rng.choice((2, 3, 8, 40)))
            args, value = correlated(xs, ys)
            shapes[type(value)] += 1
            with self.subTest(case=case):
                self.assertAccepts("correlation", args, value)
        self.assertTrue(all(shapes.values()), shapes)  # Every canonical shape was met.

    def test_linear_regression_accepts_the_known_values_of_the_card(self):
        cases = [((1, 2, 3), (3, 5, 7), 2, 1), ((0, 1, 2), (1, 2, 4), Fraction(3, 2), Fraction(5, 6)),
                 (*PROBE, Fraction(5, 2), Fraction(-2, 3))]
        for xs, ys, slope, intercept in cases:
            with self.subTest(xs=xs, ys=ys):
                value = LinearFit(rational(slope), rational(intercept))
                self.assertEqual(fit_of(xs, ys), value)
                self.assertAccepts("linear_regression", arguments(xs, ys), value)
                # Brute force around the line: any small step in slope or intercept increases the squared error.
                best = squared_error(xs, ys, slope, intercept)
                step = Fraction(1, 1000)
                for da, db in ((step, 0), (-step, 0), (0, step), (0, -step), (step, -step)):
                    self.assertGreater(squared_error(xs, ys, frac(slope) + da, frac(intercept) + db), best)

    def test_linear_regression_accepts_special_cases(self):
        half, third = Fraction(1, 2), Fraction(1, 3)
        cases = [
            ("slope 0, constant ys", (1, 2, 3), (4, 4, 4), 0, 4),
            ("slope 0, uncorrelated ys", *FLAT, 0, Fraction(2, 3)),
            ("intercept 0", *LINE, 2, 0),
            ("the line through two points", (1, 2), (3, 8), 5, -2),
            ("two rational points", (half, third), (2, 7), -30, 17),
            ("repeated xs that are not all equal", (1, 1, 2), (1, 3, 5), 3, -1),
            ("a falling line", (1, 2, 3, 4, 5), (10, 8, 6, 4, 2), -2, 12),
            ("all ys zero", (3, -1, 4), (0, 0, 0), 0, 0),
        ]
        for label, xs, ys, slope, intercept in cases:
            with self.subTest(case=label):
                value = LinearFit(rational(slope), rational(intercept))
                self.assertEqual(fit_of(xs, ys), value)
                self.assertAccepts("linear_regression", arguments(xs, ys), value)
        others = [
            ("rational data", (half, -third, Fraction(7, 4), 2, Fraction(-5, 6)), (third, 1, Fraction(-2, 9), 0, 5)),
            ("unsorted data", (5, -3, 0, 2, 9, -1), (1, 4, 4, -2, 7, 0)),
            ("long integers", (2 ** 200, -(2 ** 150) + 7, 3 ** 90, 1), (5, 2 ** 300 + 1, -(7 ** 80), 11)),
            ("long denominators", (Fraction(1, 2 ** 100), Fraction(3, 3 ** 50), 1), (Fraction(2, 5 ** 40), 0, 7)),
            ("different denominators in xs and ys", (half, third, Fraction(1, 5)),
             (Fraction(1, 7), 1, Fraction(2, 11))),
            ("a thousand points", tuple(range(1000)), tuple((7 * k) % 101 for k in range(1000))),
        ]
        for label, xs, ys in others:
            with self.subTest(case=label):
                self.assertAccepts("linear_regression", *fitted(xs, ys))
        mixed = {"xs": (Rational(1), 2, Rational(3)), "ys": (3, Rational(5), 7)}
        self.assertAccepts("linear_regression", mixed, LinearFit(Rational(2), Rational(1)))
        # Forging a canonical record keeps it valid: the checker reads the fields, whatever made the record.
        self.assertAccepts("linear_regression", arguments(*PROBE), forged_fit(Rational(5, 2), Rational(-2, 3)))
        self.assertAccepts("linear_regression", arguments(*PROBE), forged_fit(forged_rational(5, 2),
                                                                              forged_rational(-2, 3)))

    def test_linear_regression_accepts_seeded_true_claims(self):
        rng = random.Random(20261113)
        for case in range(300):
            n = rng.randrange(2, 12)
            xs = varying(rng, n, rng.choice((2, 3, 8, 40)))
            if rng.random() < 0.15:  # Exactly linear data: the line itself.
                slope, intercept = seeded_number(rng, 5), seeded_number(rng, 5)
                ys = [slope * x + intercept for x in xs]
                self.assertEqual(fit_of(xs, ys), LinearFit(rational(slope), rational(intercept)))
            else:
                ys = [seeded_number(rng, rng.choice((2, 3, 8, 40))) for _ in range(n)]
            with self.subTest(case=case):
                self.assertAccepts("linear_regression", *fitted(xs, ys))

    def test_full_size_claims_fit_the_default_budget(self):
        """20,000 pairs of 64-bit integers, or 8,000 pairs of rationals, verify under the default limits, through
        mf.verify too (the request then holds most of the nodes that the verifier admits)."""
        rng = random.Random(20261114)
        n = 20_000
        xs = tuple(rng.randrange(-(2 ** 64), 2 ** 64) for _ in range(n))
        ys = tuple(x // 3 + rng.randrange(-(2 ** 62), 2 ** 62) for x in xs)
        args = {"xs": xs, "ys": ys}
        rationals = arguments([Fraction(rng.randrange(-1000, 1000), rng.randrange(1, 256)) for _ in range(8000)],
                              [Fraction(rng.randrange(-1000, 1000), rng.randrange(1, 256)) for _ in range(8000)])
        for label, claim_args in (("integers", args), ("rationals", rationals)):
            for operation, value in (("correlation", correlation_of(claim_args["xs"], claim_args["ys"])),
                                     ("linear_regression", fit_of(claim_args["xs"], claim_args["ys"]))):
                with self.subTest(data=label, operation=operation):
                    with computation(fresh=True) as budget:
                        start = time.perf_counter()
                        self.assertIs(checker.check_operation(operation, claim_args, value, ()), True)
                        elapsed = time.perf_counter() - start
                    self.assertLess(budget.work, ComputationLimits().max_work // 2)
                    self.assertLess(elapsed, FAST)
                    report = mf.verify(claimed(operation, claim_args, value))
                    self.assertTrue(report.verified, support.statuses(report))

    # -----------------------------------------------------------------------------------------------------------
    # Wrong and non-canonical values

    def test_correlation_rejects_wrong_values(self):
        card, probe, mirror, flat = (arguments(*data) for data in (CARD, PROBE, MIRROR, FLAT))
        tiny = Rational(1, 10 ** 30)
        cases: list[tuple[str, object, object, object]] = [
            ("3/5 for 4/5", card, Rational(3, 5), ()),
            ("-4/5 for 4/5", card, Rational(-4, 5), ()),
            ("1 for 4/5", card, Rational(1), ()),
            ("0 for 4/5", card, Rational(0), ()),
            ("q = 16/25 for r = 4/5", card, Rational(16, 25), ()),
            ("sqrt(4/5) for 4/5", card, Sqrt(Rational(4, 5)), ()),
            ("5/4 for 4/5", card, Rational(5, 4), ()),
            ("4/5 plus 1/10^30", card, Rational(4, 5) + tiny, ()),
            ("-sqrt(75/76) for sqrt(75/76)", probe, Mul((Rational(-1), Sqrt(Q))), ()),
            ("sqrt(76/75)", probe, Sqrt(Rational(76, 75)), ()),
            ("sqrt(75/77)", probe, Sqrt(Rational(75, 77)), ()),
            ("sqrt(37/38)", probe, Sqrt(Rational(37, 38)), ()),
            ("99/100, close to sqrt(75/76)", probe, Rational(99, 100), ()),
            ("1 for sqrt(75/76)", probe, Rational(1), ()),
            ("sqrt(75/76) for -sqrt(75/76)", mirror, Sqrt(Q), ()),
            ("-sqrt(76/75)", mirror, Mul((Rational(-1), Sqrt(Rational(76, 75)))), ()),
            # The radicand right, the factor before it wrong: only Rational(-1) makes -sqrt(q).
            ("-2*sqrt(75/76)", mirror, Mul((Rational(-2), Sqrt(Q))), ()),
            ("-1/2*sqrt(75/76)", mirror, Mul((Rational(-1, 2), Sqrt(Q))), ()),
            ("1*sqrt(75/76) for -sqrt(75/76)", mirror, Mul((Rational(1), Sqrt(Q))), ()),
            ("0*sqrt(75/76)", mirror, Mul((Rational(0), Sqrt(Q))), ()),
            ("2*sqrt(75/76)", probe, Mul((Rational(2), Sqrt(Q))), ()),
            ("1/10^30 for 0", flat, tiny, ()),
            ("-1/10^30 for 0", flat, -tiny, ()),
            ("sqrt(1/10^30) for 0", flat, Sqrt(tiny), ()),
            ("the population covariance, 1", card, Rational(1), ()),
            ("the value of other data", arguments(*LINE), Rational(4, 5), ()),
        ]
        self.assertRejectsAll("correlation", cases)

    def test_correlation_rejects_non_canonical_values(self):
        """Mathematically right values in a form other than the canonical one; each refused."""
        card, probe, mirror, line, flat = (arguments(*data) for data in (CARD, PROBE, MIRROR, LINE, FLAT))
        falling_card = arguments(CARD[0], CARD[1][::-1])  # r = -4/5.
        self.assertEqual(correlation_of(*CARD[:1], CARD[1][::-1]), Rational(-4, 5))
        minus = Mul((Rational(-1), Sqrt(Q)))
        # Forging a canonical value keeps it valid.
        self.assertAcceptsAll("correlation", [
            ("a forged 4/5", card, forged_rational(4, 5)),
            ("a forged Sqrt", probe, support.forged(Sqrt, radicand=Q)),
            ("a forged Mul", mirror, support.forged(Mul, factors=(Rational(-1), Sqrt(Q)))),
            ("a forged -1 in a Mul", mirror, Mul((forged_rational(-1), Sqrt(Q)))),
        ])
        cases: list[tuple[str, object, object, object]] = [
            # A rational square q: the Rational is canonical, never a radical.
            ("sqrt(16/25) for 4/5", card, Sqrt(Rational(16, 25)), ()),
            ("-sqrt(16/25) for -4/5", falling_card, Mul((Rational(-1), Sqrt(Rational(16, 25)))), ()),
            ("sqrt(1) for 1", line, Sqrt(Rational(1)), ()),
            ("sqrt(0) for 0", flat, Sqrt(Rational(0)), ()),
            ("-sqrt(0) for 0", flat, Mul((Rational(-1), Sqrt(Rational(0)))), ()),
            ("4/5 times 1", card, Mul((Rational(4, 5), Rational(1))), ()),
            ("4/5 plus 0", card, Add((Rational(4, 5), Rational(0))), ()),
            ("4/5 to the first power", card, Pow(Rational(4, 5), 1), ()),
            # Numbers that are no canonical Rational.
            ("the int 1", line, 1, ()),
            ("the int 0", flat, 0, ()),
            ("the int -1", arguments(*FALLING), -1, ()),
            ("True for 1", line, True, ()),
            ("the float 1.0", line, 1.0, ()),
            ("the float 0.8", card, 0.8, ()),
            ("the str 4/5", card, "4/5", ()),
            ("a Fraction", card, Fraction(4, 5), ()),
            ("an int subclass", line, Integer(1), ()),
            ("a Rational subclass", card, Fractional(4, 5), ()),
            ("4/5 as 8/10", card, forged_rational(8, 10), ()),
            ("4/5 as -4/-5", card, forged_rational(-4, -5), ()),
            ("1 as 2/2", line, forged_rational(2, 2), ()),
            ("0 as 0/5", flat, forged_rational(0, 5), ()),
            ("a Rational of a bool", line, forged_rational(True, 1), ()),
            ("a Rational of a float", card, forged_rational(0.8, 1), ()),
            ("a Rational with a float denominator", card, forged_rational(4, 5.0), ()),
            ("a Rational without slots", card, support.forged(Rational), ()),
            ("an impostor Rational", card, support.impostor(Rational(4, 5)), ()),
            # The radical: no factor taken out, the sign as Rational(-1) first, nothing else.
            ("5*sqrt(3/76)", probe, Mul((Rational(5), Sqrt(Rational(3, 76)))), ()),
            ("sqrt(5700)/76", probe, Mul((Rational(1, 76), Sqrt(Rational(5700)))), ()),
            ("-5*sqrt(3/76)", mirror, Mul((Rational(-5), Sqrt(Rational(3, 76)))), ()),
            ("1*sqrt(q)", probe, Mul((Rational(1), Sqrt(Q))), ()),
            ("sqrt(q) to the first power", probe, Pow(Sqrt(Q), 1), ()),
            ("sqrt(q) plus 0", probe, Add((Sqrt(Q), Rational(0))), ()),
            ("sqrt(q)*(-1)", mirror, Mul((Sqrt(Q), Rational(-1))), ()),
            ("-1*sqrt(q)*1", mirror, Mul((Rational(-1), Sqrt(Q), Rational(1))), ()),
            ("-1*(1*sqrt(q))", mirror, Mul((Rational(-1), Mul((Rational(1), Sqrt(Q))))), ()),
            ("-1 as -2/2", mirror, Mul((forged_rational(-2, 2), Sqrt(Q))), ()),
            ("-1 as an int", mirror, support.forged(Mul, factors=(-1, Sqrt(Q))), ()),
            ("-1 as a float", mirror, support.forged(Mul, factors=(-1.0, Sqrt(Q))), ()),
            ("the factors as a list", mirror, support.forged(Mul, factors=[Rational(-1), Sqrt(Q)]), ()),
            ("the factors as a tuple subclass", mirror, support.forged(Mul, factors=Values((Rational(-1), Sqrt(Q)))),
             ()),
            ("one factor", probe, support.forged(Mul, factors=(Sqrt(Q),)), ()),
            ("a Mul without factors", mirror, support.forged(Mul), ()),
            ("the radicand unreduced", probe, support.forged(Sqrt, radicand=forged_rational(150, 152)), ()),
            ("the radicand as a Fraction", probe, support.forged(Sqrt, radicand=Fraction(75, 76)), ()),
            ("the radicand as a float", probe, support.forged(Sqrt, radicand=75 / 76), ()),
            ("the radicand as a str", probe, support.forged(Sqrt, radicand="75/76"), ()),
            ("the radicand as an int", line, support.forged(Sqrt, radicand=1), ()),
            ("the radicand of a negative denominator", probe, support.forged(Sqrt, radicand=forged_rational(-75, -76)),
             ()),
            ("a Sqrt without a radicand", probe, support.forged(Sqrt), ()),
            ("a Sqrt subclass", probe, Radical(Q), ()),
            ("a Mul subclass", mirror, Product((Rational(-1), Sqrt(Q))), ()),
            ("a Sqrt subclass in a Mul", mirror, support.forged(Mul, factors=(Rational(-1), Radical(Q))), ()),
            ("an impostor Sqrt", probe, support.impostor(Sqrt(Q)), ()),
            ("an impostor Mul", mirror, support.impostor(minus), ()),
            ("an impostor Sqrt in a Mul", mirror,
             support.forged(Mul, factors=(Rational(-1), support.impostor(Sqrt(Q)))), ()),
            ("the radical in a tuple", probe, (Sqrt(Q),), ()),
            ("the sign and the radicand", mirror, (-1, Q), ()),
        ]
        self.assertRejectsAll("correlation", cases)

    def test_linear_regression_rejects_wrong_values(self):
        args = arguments((0, 1, 2), (1, 2, 4))  # The line y = 3/2*x + 5/6.
        value = LinearFit(Rational(3, 2), Rational(5, 6))
        self.assertAccepts("linear_regression", args, value)
        # Each normal equation is live: a line through the means satisfies sum r_i = 0 only, the least-squares line
        # through the origin satisfies sum x_i r_i = 0 only.
        horizontal, origin = LinearFit(Rational(0), Rational(7, 3)), LinearFit(Rational(2), Rational(0))
        self.assertEqual(sum(map(frac, residuals((0, 1, 2), (1, 2, 4), horizontal))), 0)
        self.assertNotEqual(sum(x * frac(r) for x, r in zip((0, 1, 2), residuals((0, 1, 2), (1, 2, 4), horizontal))),
                            0)
        self.assertEqual(sum(x * frac(r) for x, r in zip((0, 1, 2), residuals((0, 1, 2), (1, 2, 4), origin))), 0)
        self.assertNotEqual(sum(map(frac, residuals((0, 1, 2), (1, 2, 4), origin))), 0)
        tiny = Rational(1, 10 ** 30)
        cases: list[tuple[str, object, object, object]] = [
            ("the intercept off", args, LinearFit(Rational(3, 2), Rational(5, 7)), ()),
            ("the slope off by 1/10^30", args, LinearFit(Rational(3, 2) + tiny, Rational(5, 6)), ()),
            ("the intercept off by 1/10^30", args, LinearFit(Rational(3, 2), Rational(5, 6) - tiny), ()),
            ("slope and intercept swapped", args, LinearFit(Rational(5, 6), Rational(3, 2)), ()),
            ("the slope negated", args, LinearFit(Rational(-3, 2), Rational(5, 6)), ()),
            ("a horizontal line through the means", args, horizontal, ()),
            ("the least-squares line through the origin", args, origin, ()),
            ("the line through the first and last points", args, LinearFit(Rational(3, 2), Rational(1)), ()),
            ("x regressed on y, through the means too", args, LinearFit(Rational(14, 9), Rational(7, 9)), ()),
            ("the line of other data", arguments((1, 2, 3), (3, 5, 7)), value, ()),
            ("the line of the data reversed", arguments((1, 2, 4), (0, 1, 2)), value, ()),
        ]
        self.assertRejectsAll("linear_regression", cases)

    def test_linear_regression_rejects_non_canonical_values(self):
        """The right line in a form that no LinearFit record holds, or in another class; each refused."""
        args = arguments((1, 2, 3), (3, 5, 7))  # The line y = 2*x + 1.
        fraction_args = arguments((0, 1, 2), (1, 2, 4))  # The line y = 3/2*x + 5/6.
        value = LinearFit(Rational(2), Rational(1))
        self.assertAccepts("linear_regression", args, value)
        cases: list[tuple[str, object, object, object]] = [
            ("ints", args, forged_fit(2, 1), ()),
            ("an int slope", args, forged_fit(2, Rational(1)), ()),
            ("an int intercept", args, forged_fit(Rational(2), 1), ()),
            ("the slope 2/1 as 4/2", args, forged_fit(forged_rational(4, 2), Rational(1)), ()),
            ("the intercept as -1/-1", args, forged_fit(Rational(2), forged_rational(-1, -1)), ()),
            ("the slope 3/2 as 6/4", fraction_args, forged_fit(forged_rational(6, 4), Rational(5, 6)), ()),
            ("Fractions", fraction_args, forged_fit(Fraction(3, 2), Fraction(5, 6)), ()),
            ("floats", args, forged_fit(2.0, 1.0), ()),
            ("strs", args, forged_fit("2", "1"), ()),
            ("bools", arguments((0, 1), (1, 2)), forged_fit(True, True), ()),
            ("int subclasses", args, forged_fit(Integer(2), Integer(1)), ()),
            ("Rational subclasses", args, forged_fit(Fractional(2), Fractional(1)), ()),
            ("no slope", args, support.forged(LinearFit, intercept=Rational(1)), ()),
            ("no intercept", args, support.forged(LinearFit, slope=Rational(2)), ()),
            ("no fields", args, support.forged(LinearFit), ()),
            ("an impostor record", args, support.impostor(value), ()),
            ("a LinearFit subclass", args, SubFit(Rational(2), Rational(1)), ()),
            ("a tuple", args, (Rational(2), Rational(1)), ()),
            ("a list", args, [Rational(2), Rational(1)], ()),
            ("a dict", args, {"slope": Rational(2), "intercept": Rational(1)}, ()),
            ("the slope alone", args, Rational(2), ()),
            ("the residuals", args, residuals((1, 2, 3), (3, 5, 7), value), ()),
        ]
        for index, field in enumerate(("slope", "intercept")):
            for label, item in inexact(2 - index):
                fields = [Rational(2), Rational(1)]
                fields[index] = item
                cases.append((f"the {field}: {label}", args, forged_fit(*fields), ()))
        self.assertRejectsAll("linear_regression", cases)

    # -----------------------------------------------------------------------------------------------------------
    # Inputs outside the domain, wrong arguments, evidence and operations

    def test_rejects_claims_on_inputs_outside_the_domain(self):
        """Lists of different lengths, fewer than 2 pairs and zero variance have no value: every claim answers
        False, also one that satisfies the identities that the checker tests on valid inputs."""
        zero, one = Rational(0), Rational(1)
        domain = [
            ("different lengths", arguments((1, 2, 3), (1, 2))),
            ("ys longer", arguments((1, 2), (1, 2, 3))),
            ("one pair", arguments((1,), (2,))),
            ("no pair", arguments((), ())),
        ]
        for operation, value in (("correlation", one), ("correlation", zero),
                                 ("linear_regression", LinearFit(one, zero))):
            self.assertRejectsAll(operation, [(label, args, value, ()) for label, args in domain])
        constant_x, constant_y = arguments((2, 2, 2), (1, 2, 3)), arguments((1, 2, 3), (5, 5, 5))
        both = arguments((4, 4), (Rational(1, 3), Rational(1, 3)))
        self.assertRejectsAll("correlation", [
            (f"{label}: {value}", args, value, ()) for label, args in (("constant xs", constant_x),
                                                                      ("constant ys", constant_y),
                                                                      ("both constant", both))
            for value in (zero, one, Rational(-1), Sqrt(Rational(1, 2)))])
        # All xs equal: every line through (2, mean(ys)) satisfies both normal equations, none is unique.
        lines = [LinearFit(zero, Rational(2)), LinearFit(one, zero), LinearFit(Rational(-1), Rational(4))]
        for line in lines:
            self.assertEqual(sum(map(frac, residuals((2, 2, 2), (1, 2, 3), line))), 0)
        self.assertRejectsAll("linear_regression", [(f"constant xs, {index}", constant_x, line, ())
                                                    for index, line in enumerate(lines)])
        self.assertRejectsAll("linear_regression", [("constant xs and ys", both, LinearFit(zero, Rational(1, 3)), ())])
        # Constant ys are fine for a line (slope 0), never for a correlation.
        self.assertAccepts("linear_regression", constant_y, LinearFit(zero, Rational(5)))

    def test_rejects_corrupted_arguments(self):
        cases = {"correlation": correlated(*CARD), "linear_regression": fitted(*CARD)}
        for operation, (args, value) in cases.items():
            self.assertAccepts(operation, args, value)
            corrupted: list[tuple[str, object, object, object]] = []
            for name in ("xs", "ys"):
                for index in (0, 3):
                    for label, item in inexact(args[name][index]):
                        changed = args[name][:index] + (item,) + args[name][index + 1:]
                        corrupted.append((f"{name}[{index}]: {label}", dict(args, **{name: changed}), value, ()))
                corrupted += [
                    (f"{name} as a list", dict(args, **{name: list(args[name])}), value, ()),
                    (f"{name} as a tuple subclass", dict(args, **{name: Values(args[name])}), value, ()),
                    (f"{name} as None", dict(args, **{name: None}), value, ()),
                    (f"{name} as a str", dict(args, **{name: "1, 2, 3, 4"}), value, ()),
                    (f"{name} as a generator", dict(args, **{name: (item for item in args[name])}), value, ()),
                    (f"{name} as an int", dict(args, **{name: 4}), value, ()),
                ]
            corrupted += [
                ("the arguments as pairs", tuple(args.items()), value, ()),
                ("the arguments as a list of pairs", list(args.items()), value, ()),
                ("the arguments as a read-only mapping", MappingProxyType(args), value, ()),
                ("None as the arguments", None, value, ()),
                ("ys missing", {"xs": args["xs"]}, value, ()),
                ("an extra argument", dict(args, ddof=0), value, ()),
                ("the names of covariance", {"x": args["xs"], "y": args["ys"]}, value, ()),
                ("a key that is no str", {"xs": args["xs"], 1: args["ys"]}, value, ()),
            ]
            self.assertRejectsAll(operation, corrupted)

    def test_rejects_wrong_evidence(self):
        """The evidence is (): any other, also the right sums or residuals, is refused."""
        sxx, syy, sxy = (rational(s) for s in centred_sums(*CARD))
        for operation, (args, value) in (("correlation", correlated(*CARD)), ("linear_regression", fitted(*CARD))):
            self.assertAccepts(operation, args, value)
            consistent = residuals(*CARD, fit_of(*CARD)) if operation == "linear_regression" else (Rational(16, 25),)
            self.assertRejectsAll(operation, [
                ("evidence None", args, value, None),
                ("evidence as a list", args, value, []),
                ("evidence as a tuple subclass", args, value, Values()),
                ("evidence as a dict", args, value, {}),
                ("an empty item", args, value, ((),)),
                ("a number", args, value, (0,)),
                ("a str", args, value, ("centred",)),
                ("the centred sums", args, value, (sxx, syy, sxy)),
                ("the centred sums, nested", args, value, ((sxx, syy, sxy),)),
                ("the right q or residuals", args, value, consistent),
                ("the value itself", args, value, (value,)),
            ])

    def test_rejects_wrong_operations(self):
        args, value = correlated(*PROBE)
        fit_args, fit = fitted(*PROBE)
        for operation in ("Correlation", "correlation ", "pearson", "covariance", "stddev", "linear_fit",
                          "continued_fraction", "", None, 7, ("correlation",), b"correlation", ["correlation"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value)
                self.assertRejects(operation, fit_args, fit)
        # A true claim of one operation, under the name of the other.
        self.assertRejects("linear_regression", args, value)
        self.assertRejects("correlation", fit_args, fit)
        self.assertRejects("linear_regression", arguments(*LINE), Rational(1))
        self.assertRejects("correlation", arguments(*LINE), LinearFit(Rational(2), Rational(0)))

    def test_never_raises_on_junk(self):
        """Every slot of a claim filled with stand-ins of every kind: the checker answers False, never raising."""
        junk: list[object] = [None, 0, 1.5, "x", b"x", (), [], {}, Rational(7), Sqrt(Rational(2)),
                              Mul((Rational(-3), Sqrt(Rational(2)))), LinearFit(Rational(1), Rational(1)), object(),
                              True, support.forged(Rational), support.forged(Sqrt), support.forged(Mul),
                              support.forged(LinearFit), Values((1,)), MappingProxyType({}), frozenset(),
                              Fraction(1, 2), (1, 2), {"xs": (1, 2)}]
        claims = {"correlation": correlated(*PROBE), "linear_regression": fitted(*PROBE)}
        for operation, (args, value) in claims.items():
            for index, item in enumerate(junk):
                with self.subTest(operation=operation, item=index, kind=type(item).__name__):
                    self.assertRejects(item, args, value)
                    self.assertRejects(operation, item, value)
                    self.assertRejects(operation, args, item)
                    self.assertRejects(operation, args, value, item if item != () else (item,))
                    for name in ("xs", "ys"):
                        # In the middle of the data, an exact item makes other data with another value.
                        self.assertRejects(operation, dict(args, **{name: item}), value)
                        self.assertRejects(operation, dict(args, **{name: (1, item, 3)}), value)

    # -----------------------------------------------------------------------------------------------------------
    # Oversized forgeries and the budget

    def test_oversized_forgeries_fail_fast(self):
        rng = random.Random(20261115)
        big = [rng.getrandbits(65_000) | 1 << 64_999 for _ in range(4)]  # Within max_integer_bits, but long.
        long_args = arguments(tuple(range(20_000)), tuple((k * k) % 1009 for k in range(20_000)))
        small = arguments(*PROBE)
        cases = [
            ("correlation", small, forged_rational(GIANT, 1), ()),
            ("correlation", small, forged_rational(1, GIANT), ()),
            ("correlation", small, forged_rational(-GIANT, GIANT + 1), ()),
            ("correlation", small, support.forged(Sqrt, radicand=forged_rational(GIANT, GIANT + 2)), ()),
            ("correlation", small, Mul((forged_rational(-GIANT), Sqrt(Q))), ()),
            ("correlation", small, Mul((Rational(-1), support.forged(Sqrt, radicand=forged_rational(3, GIANT)))), ()),
            ("correlation", small, Rational(big[0], big[1]), ()),
            ("correlation", small, Sqrt(Rational(big[0], big[1])), ()),
            ("correlation", small, Mul((Rational(-1), Sqrt(Rational(big[2], big[3])))), ()),
            ("correlation", long_args, Rational(big[0], big[1]), ()),
            ("correlation", long_args, Sqrt(Rational(big[2], big[3])), ()),
            ("correlation", small, tuple(range(100_000)), ()),
            ("correlation", small, Sqrt(Q), (0,) * 100_000),
            ("correlation", small, Sqrt(Q), (GIANT,)),
            ("linear_regression", small, forged_fit(forged_rational(GIANT), Rational(1)), ()),
            ("linear_regression", small, forged_fit(Rational(1), forged_rational(1, GIANT)), ()),
            ("linear_regression", small, LinearFit(Rational(big[0], big[1]), Rational(big[2], big[3])), ()),
            ("linear_regression", long_args, LinearFit(Rational(big[0], big[1]), Rational(big[2], big[3])), ()),
            ("linear_regression", long_args, LinearFit(Rational(big[0]), Rational(-big[1])), ()),
            ("linear_regression", small, tuple(range(100_000)), ()),
            ("linear_regression", small, fit_of(*PROBE), (0,) * 100_000),
            ("linear_regression", small, fit_of(*PROBE), (GIANT,)),
        ]
        for index, (operation, args, value, evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, evidence)  # Under the default budget, never raising.
                self.assertLess(time.perf_counter() - start, FAST)
        # A request integer over max_integer_bits raises ResourceLimitError, as the request's inspection does before
        # any checker runs; so does a common denominator of the data over max_integer_bits.
        coprime = tuple(Rational(1, p) for p in primes_below(60_000))  # Their lcm has far more than 65,536 bits.
        requests = [
            ("a giant x", {"xs": (1, GIANT, 3), "ys": (1, 2, 3)}),
            ("a giant y", {"xs": (1, 2, 3), "ys": (1, 2, -GIANT)}),
            ("a giant denominator", {"xs": (1, Rational(1, GIANT), 3), "ys": (1, 2, 3)}),
            ("a giant numerator", {"xs": (1, 2, 3), "ys": (1, Rational(GIANT, 3), 3)}),
            ("coprime denominators", {"xs": coprime, "ys": tuple(range(len(coprime)))}),
        ]
        for operation, value in (("correlation", Rational(1)), ("linear_regression", LinearFit(Rational(1),
                                                                                               Rational(0)))):
            for label, args in requests:
                with self.subTest(operation=operation, request=label):
                    start = time.perf_counter()
                    with self.assertRaises(ResourceLimitError):
                        support.verdict(family, operation, args, value)
                    self.assertLess(time.perf_counter() - start, FAST)

    def test_work_is_charged_before_it_runs(self):
        rng = random.Random(20261116)
        xs = [Fraction(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 50)) for _ in range(200)]
        ys = [Fraction(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 50)) for _ in range(200)]
        for operation, (args, value) in (("correlation", correlated(xs, ys)), ("linear_regression", fitted(xs, ys))):
            with self.subTest(operation=operation):
                # The charges depend on the claim alone: a budget one unit short refuses it, its exact size passes.
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, ()), True)
                work = budget.work
                self.assertGreater(work, 4 * 200)  # At least a unit per item and pass.
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, value, limits=ComputationLimits(max_work=work))
                # A small budget stops a large claim at once.
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, limits=ComputationLimits(max_work=300))
                self.assertLess(time.perf_counter() - start, FAST)
        # Long integers cost more than short ones: the same data scaled by a 20,000-bit factor. Each pair pays for
        # the products of its deviations (U_i^2, V_i^2, U_i V_i) or of its scaled values (X_i^2, X_i Y_i) by their
        # length, as product_work prices them.
        scale = 2 ** 20_000 + 1
        short_xs, short_ys = xs[:20], ys[:20]
        long_xs, long_ys = [x * scale for x in short_xs], [y * scale + 1 for y in short_ys]
        for operation, reference, products in (("correlation", correlated, 3), ("linear_regression", fitted, 2)):
            charged = []
            for claim in (reference(short_xs, short_ys), reference(long_xs, long_ys)):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, claim[0], claim[1], ()), True)
                charged.append(budget.work)
            with self.subTest(operation=operation, charged=charged):
                self.assertGreater(charged[1], 2 * charged[0])
                self.assertGreaterEqual(charged[1], 20 * products * product_work(20_000, 20_000))
        # Every gcd is paid for before it runs, also when the claim is refused right after it: a long value whose
        # sign is wrong, and long data whose next item is not in lowest terms.
        big = [rng.getrandbits(65_000) | 1 << 64_999 | 1 for _ in range(2)]
        while gcd(*big) != 1:
            big[1] += 2
        tight = ComputationLimits(max_work=5_000)
        self.assertRejects("correlation", arguments(*PROBE), Rational(-big[0], big[1]))
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "correlation", arguments(*PROBE), Rational(-big[0], big[1]), limits=tight)
        unreduced = {"xs": (Rational(big[0], big[1]), forged_rational(2, 4), 3), "ys": (1, 2, 3)}
        self.assertRejects("correlation", unreduced, Rational(1))
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "correlation", unreduced, Rational(1), limits=tight)
        # A smaller max_integer_bits refuses a common denominator (2003 * 2011 has 22 bits), or a request integer,
        # that the default admits; a value integer over it is refused as no computation under that budget forms it.
        args = arguments((Fraction(1, 2003), Fraction(1, 2011)), (0, 1))
        self.assertAccepts("correlation", args, Rational(-1))
        narrow = ComputationLimits(max_integer_bits=20)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "correlation", args, Rational(-1), limits=narrow)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "correlation", arguments((1, 2 ** 30, 3), (1, 2, 3)), Rational(1), limits=narrow)
        fit = fit_of(*args.values())
        self.assertEqual(fit.slope, Rational(-2003 * 2011, 8))
        self.assertAccepts("linear_regression", args, fit)
        self.assertRejects("linear_regression", args, fit, limits=narrow)

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
        # The docstring states the theorem of each operation.
        for operation in ("correlation(xs, ys) = r, evidence ().",
                          "linear_regression(xs, ys) = LinearFit(slope, intercept), evidence ()."):
            self.assertIn(operation, checker.__doc__)

    # -----------------------------------------------------------------------------------------------------------
    # The verifier

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip, and a wrong value fails
        the mathematics check."""
        claims = [
            ("correlation", arguments(*CARD), Rational(4, 5), Sqrt(Rational(16, 25))),
            ("correlation", arguments(*PROBE), Sqrt(Q), Mul((Rational(-1), Sqrt(Q)))),
            ("correlation", arguments(*MIRROR), Mul((Rational(-1), Sqrt(Q))), Sqrt(Q)),
            ("correlation", arguments(*FLAT), Rational(0), Rational(1, 2)),
            ("correlation", arguments((Fraction(1, 2), 2, -3), (0, Fraction(4, 5), 1)),
             correlation_of((Fraction(1, 2), 2, -3), (0, Fraction(4, 5), 1)), Rational(1)),
            ("linear_regression", arguments((1, 2, 3), (3, 5, 7)), LinearFit(Rational(2), Rational(1)),
             LinearFit(Rational(1), Rational(2))),
            ("linear_regression", arguments(*PROBE), LinearFit(Rational(5, 2), Rational(-2, 3)),
             LinearFit(Rational(5, 2), Rational(2, 3))),
        ]
        for operation, args, value, wrong in claims:
            with self.subTest(operation=operation, value=value):
                self.roundtrip(claimed(operation, args, value))
                report = support.statuses(mf.verify(claimed(operation, args, wrong)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
