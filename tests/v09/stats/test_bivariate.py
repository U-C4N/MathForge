"""The bivariate family of mathforge.stats: correlation, linear_regression.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register stats bivariate

Every expected value comes from the card or from the references of this module, written here and slow on purpose, never
from the producer: the sums of squares and products by Lagrange's pairwise form, n S_xy = sum over i < j of
(x_i - x_j)(y_i - y_j), in fractions.Fraction (neither the producer's raw moments nor the checker's centred deviations);
the coefficient's canonical form built by hand from a rational square root of its own; the least-squares line from the
same sums, slope S_xy / S_xx through the mean point, which a search around the line confirms on the card's data. The
family has no deferred case: every valid input has a value, and every other input is invalid input.
tests/v09/stats/test_bivariate_checker.py tests the checker on its own.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
from math import isqrt
import random
import time
from typing import Any
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.stats import bivariate as family
from mathforge.areas.stats.bivariate import LinearFit
from mathforge.contracts import CheckStatus, ErrorInfo, Outcome
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Mul, Pow, Rational, Sqrt

from .. import support

FAST = 1.0  # Seconds: a request that its budget or the domain refuses fails well within this.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
OPERATIONS = ("correlation", "linear_regression")
STAND_INS = (0.5, True, "1", None)  # No exact rational: a float, a bool, a str and None.
Q = Rational(75, 76)  # The radicand of the benchmark probe: r = sqrt(75/76) for (1, 2, 3), (2, 4, 7).

Number = int | Fraction | Rational

# One small request per operation.
SMALL = {
    "correlation": dict(xs=(1, 2, 3, 4), ys=(1, 3, 2, 4)),
    "linear_regression": dict(xs=(0, 1, 2), ys=(1, 2, 4)),
}

# The known values of the card, the benchmark probe and special cases: (operation, xs, ys, value).
KNOWN = (
    ("correlation", (1, 2, 3), (2, 4, 6), Rational(1)),
    ("correlation", (1, 2, 3), (3, 2, 1), Rational(-1)),
    ("correlation", (1, 2, 3, 4), (1, 3, 2, 4), Rational(4, 5)),
    ("correlation", (1, 2, 3), (2, 4, 7), Sqrt(Q)),
    ("correlation", (1, 2, 3), (7, 4, 2), Mul((Rational(-1), Sqrt(Q)))),
    ("correlation", (1, 2, 3), (1, 0, 1), Rational(0)),
    ("correlation", (-1, 0, 1), (0, 2, 1), Rational(1, 2)),
    ("correlation", (Rational(1, 2), Rational(1, 3)), (2, 7), Rational(-1)),  # Two points always lie on a line.
    # n S_xx = 75/2, n S_yy = 1862/25, n S_xy = -51: q = 2601/2793 = 867/931, which is no rational square.
    ("correlation", (Rational(1, 2), 3, -2), (1, Rational(-4, 5), 6), Mul((Rational(-1), Sqrt(Rational(867, 931))))),
    ("linear_regression", (1, 2, 3), (3, 5, 7), LinearFit(Rational(2), Rational(1))),
    ("linear_regression", (0, 1, 2), (1, 2, 4), LinearFit(Rational(3, 2), Rational(5, 6))),
    ("linear_regression", (1, 2, 3), (2, 4, 7), LinearFit(Rational(5, 2), Rational(-2, 3))),
    ("linear_regression", (1, 2, 3), (4, 4, 4), LinearFit(Rational(0), Rational(4))),  # Constant ys: slope 0.
    ("linear_regression", (1, 2, 3), (1, 0, 1), LinearFit(Rational(0), Rational(2, 3))),
    ("linear_regression", (1, 1, 2), (1, 3, 5), LinearFit(Rational(3), Rational(-1))),  # Repeated xs, not all equal.
    ("linear_regression", (Rational(1, 2), Rational(1, 3)), (2, 7), LinearFit(Rational(-30), Rational(17))),
    ("linear_regression", (1, 2, 3), (1, 2, 3), LinearFit(Rational(1), Rational(0))),
    ("linear_regression", (Rational(1, 2), 3, -2), (1, Rational(-4, 5), 6), LinearFit(Rational(-34, 25),
                                                                                   Rational(206, 75))),
)

# Inputs outside the domains of the card, each invalid input: (operation, xs, ys, a part of the message).
INVALID = (
    ("correlation", (1, 2, 3), (1, 2), "the same length"),
    ("correlation", (1, 2), (1, 2, 3), "the same length"),
    ("correlation", (1,), (2,), "at least 2 pairs"),
    ("correlation", (), (), "at least 2 pairs"),
    ("correlation", (2, 2, 2), (1, 2, 3), "xs that are not all equal"),
    ("correlation", (1, 2, 3), (5, 5, 5), "ys that are not all equal"),
    ("correlation", (4, 4), (Rational(1, 3), Rational(1, 3)), "xs that are not all equal"),
    ("correlation", (Rational(1, 3), Rational(1, 3), Rational(1, 3)), (1, 2, 3), "xs that are not all equal"),
    ("correlation", (1, 2, 3), (Rational(-7, 2), Rational(-7, 2), Rational(-7, 2)), "ys that are not all equal"),
    ("linear_regression", (1, 2, 3), (1, 2), "the same length"),
    ("linear_regression", (1,), (2,), "at least 2 pairs"),
    ("linear_regression", (), (), "at least 2 pairs"),
    ("linear_regression", (2, 2, 2), (1, 2, 3), "xs that are not all equal"),
    ("linear_regression", (Rational(1, 2), Rational(1, 2)), (5, 5), "xs that are not all equal"),
)


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

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


def arguments(xs: Sequence[Number], ys: Sequence[Number]) -> dict[str, tuple[int | Rational, ...]]:
    """The arguments of both operations as a request holds them: tuples of ints and Rationals."""
    return {"xs": tuple(exact(x) for x in xs), "ys": tuple(exact(y) for y in ys)}


def pairwise_sums(xs: Sequence[Number], ys: Sequence[Number]) -> tuple[Fraction, Fraction, Fraction]:
    """(n S_xx, n S_yy, n S_xy) by Lagrange's identity, n S_xy = sum over i < j of (x_i - x_j)(y_i - y_j): a loop over
    every pair of points, quadratic and slow on purpose, with no mean and no raw moment."""
    fx, fy = [frac(x) for x in xs], [frac(y) for y in ys]
    sxx = syy = sxy = Fraction(0)
    for i in range(len(fx)):
        for j in range(i + 1, len(fx)):
            dx, dy = fx[i] - fx[j], fy[i] - fy[j]
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
    return sxx, syy, sxy


def rational_root(number: Fraction) -> Fraction | None:
    """The rational square root of a Fraction >= 0, None when it has none."""
    p, q = isqrt(number.numerator), isqrt(number.denominator)
    return Fraction(p, q) if p * p == number.numerator and q * q == number.denominator else None


def reference_correlation(xs: Sequence[Number], ys: Sequence[Number]) -> Any:
    """The Pearson coefficient in the card's canonical form, built by hand; None outside the domain (different
    lengths, fewer than 2 pairs, a constant variable)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    sxx, syy, sxy = pairwise_sums(xs, ys)
    if not sxx or not syy:
        return None
    q = sxy * sxy / (sxx * syy)  # The factors n cancel.
    sign = (sxy > 0) - (sxy < 0)
    root = rational_root(q)
    if root is not None:
        return Rational(sign * root.numerator, root.denominator)
    radical = Sqrt(Rational(q.numerator, q.denominator))
    return radical if sign > 0 else Mul((Rational(-1), radical))


def reference_fit(xs: Sequence[Number], ys: Sequence[Number]) -> LinearFit | None:
    """The least-squares line, slope S_xy / S_xx through the mean point; None outside the domain (different lengths,
    fewer than 2 pairs, xs all equal)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    sxx, _, sxy = pairwise_sums(xs, ys)
    if not sxx:
        return None
    slope = sxy / sxx
    n = len(xs)
    intercept = sum(map(frac, ys), Fraction(0)) / n - slope * sum(map(frac, xs), Fraction(0)) / n
    return LinearFit(rational(slope), rational(intercept))


REFERENCES = {"correlation": reference_correlation, "linear_regression": reference_fit}


def squared_error(xs: Sequence[Number], ys: Sequence[Number], slope: Number, intercept: Number) -> Fraction:
    """sum (y_i - slope x_i - intercept)^2, which the least-squares line minimizes."""
    a, b = frac(slope), frac(intercept)
    return sum(((frac(y) - a * frac(x) - b) ** 2 for x, y in zip(xs, ys)), Fraction(0))


def seeded_number(rng: random.Random, bits: int) -> Fraction:
    """A random exact rational: an integer of up to bits bits, or a fraction of two such integers."""
    numerator = rng.randrange(-(2 ** bits), 2 ** bits + 1)
    return Fraction(numerator) if rng.random() < 0.5 else Fraction(numerator, rng.randrange(1, 2 ** bits + 1))


def seeded_data(rng: random.Random) -> tuple[list[Fraction], list[Fraction]]:
    """Random paired data of 2 to 12 points: mostly arbitrary, sometimes exactly linear (r = 1 or -1, the line
    itself), with constant ys or with constant xs (outside one domain or both), or with few distinct values."""
    n = rng.randrange(2, 13)
    bits = rng.choice((1, 2, 3, 8, 40))
    xs = [seeded_number(rng, bits) for _ in range(n)]
    style = rng.random()
    if style < 0.15:
        slope, intercept = seeded_number(rng, 4), seeded_number(rng, 4)
        ys = [slope * x + intercept for x in xs]
    elif style < 0.2:
        ys = [seeded_number(rng, 5)] * n
    elif style < 0.25:
        xs = [seeded_number(rng, 5)] * n
        ys = [seeded_number(rng, bits) for _ in range(n)]
    else:
        ys = [seeded_number(rng, rng.choice((1, 2, 3, 8, 40))) for _ in range(n)]
    return xs, ys


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def negated(value: Any) -> Any:
    """-r in the canonical form, for a correlation value r in the canonical form."""
    if type(value) is Rational:
        return -value
    if type(value) is Sqrt:
        return Mul((Rational(-1), value))
    return value.factors[1]


def primes_below(limit: int) -> list[int]:
    """The primes below limit, by trial division."""
    return [p for p in range(2, limit) if all(p % d for d in range(2, isqrt(p) + 1))]


def forged_copy(value: Any) -> Any:
    """An equal copy of a canonical value made with object.__new__ at every level: no constructor ran on it."""
    if type(value) is Rational:
        return forged_rational(value.numerator, value.denominator)
    if type(value) is Sqrt:
        return support.forged(Sqrt, radicand=forged_copy(value.radicand))
    if type(value) is Mul:
        return support.forged(Mul, factors=tuple(forged_copy(factor) for factor in value.factors))
    return support.forged(LinearFit, slope=forged_copy(value.slope), intercept=forged_copy(value.intercept))


class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class BivariateTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, of its exact type, with the evidence () and its four checks, which
        mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), type(value))
        self.assertIs(result.outcome, Outcome.VALUE)
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, xs, ys, value in KNOWN:
            with self.subTest(operation=operation, xs=xs, ys=ys):
                self.assertEqual(REFERENCES[operation](xs, ys), value)  # The card's value, by the slow reference.
                result = support.run(operation, xs=xs, ys=ys)
                self.assertVerified(result, value)
                self.roundtrip(result)
        # The card's lines minimize the squared error: any small step in slope or intercept increases it.
        step = Fraction(1, 1000)
        for operation, xs, ys, fit in KNOWN:
            if operation == "linear_regression" and len(set(xs)) > 1:
                best = squared_error(xs, ys, fit.slope, fit.intercept)
                for da, db in ((step, 0), (-step, 0), (0, step), (0, -step), (step, -step), (-step, step)):
                    with self.subTest(xs=xs, ys=ys, da=da, db=db):
                        self.assertGreater(squared_error(xs, ys, frac(fit.slope) + da, frac(fit.intercept) + db), best)
        # Lists are accepted as well as tuples, and ints and Rationals of denominator 1 are the same numbers.
        for operation, args, value in (
                ("correlation", dict(xs=[1, 2, 3], ys=[2, 4, 7]), Sqrt(Q)),
                ("correlation", dict(xs=(Rational(1), 2, Rational(3), 4), ys=(1, Rational(3), 2, Rational(4))),
                 Rational(4, 5)),
                ("linear_regression", dict(xs=[1, 2, 3], ys=[2, 4, 7]), LinearFit(Rational(5, 2), Rational(-2, 3))),
                ("linear_regression", dict(xs=(Rational(1), 2, 3), ys=(3, Rational(5), 7)),
                 LinearFit(Rational(2), Rational(1)))):
            with self.subTest(operation=operation, args=args):
                self.assertVerified(support.run(operation, **args), value)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument, as the argument itself and as an item of it; so is every input outside the domains of the
        card. produce() tests the types itself too."""
        containers = ({1, 2, 3, 4}, {"a": 1}, "1, 2, 3, 4", 3, Rational(3), Fraction(1, 2), support.oversized_matrix())
        items = (*STAND_INS, Fraction(1, 2), (1, 2), [1], 1j)
        for operation, args in SMALL.items():
            for name in args:
                for item in (*STAND_INS, *containers):
                    with self.subTest(operation=operation, argument=name, item=item):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: item})))
                for item in items:
                    changed = args[name][:1] + (item,) + args[name][2:]
                    with self.subTest(operation=operation, argument=name, entry=item):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: changed})))
        for operation, xs, ys, text in INVALID:
            with self.subTest(operation=operation, xs=xs, ys=ys):
                self.assertIsNone(REFERENCES[operation](xs, ys))
                result = support.run(operation, xs=xs, ys=ys)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
        # Constant ys are fine for a line (slope 0), never for a correlation.
        self.assertVerified(support.run("linear_regression", xs=(1, 2, 3), ys=(5, 5, 5)),
                            LinearFit(Rational(0), Rational(5)))
        self.assertInvalidInput(support.run("correlation", xs=(1, 2, 3), ys=(5, 5, 5)))
        # operations_v6.run rejects a stand-in by its kind before produce() runs; produce() refuses one itself, also
        # a Rational made with object.__new__ that breaks the canonical form, and arguments of another shape.
        # Labels, not the items, name the subtests: a Rational without slots has no repr.
        entries = [(repr(item), item) for item in (*STAND_INS, Integer(3), Fraction(1, 2))]
        entries += [("2/4", forged_rational(2, 4)), ("-1/-2", forged_rational(-1, -2)), ("1/0", forged_rational(1, 0)),
                    ("True/1", forged_rational(True, 1)), ("1.5/1", forged_rational(1.5, 1)),
                    ("1/2.0", forged_rational(1, 2.0)), ("no denominator", support.forged(Rational, numerator=1)),
                    ("no slots", support.forged(Rational))]
        for operation, args in SMALL.items():
            for name in args:
                for label, item in entries:
                    changed = args[name][:1] + (item,) + args[name][2:]
                    with self.subTest(operation=operation, argument=name, entry=label, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: changed}))
                for item in (*STAND_INS, {1, 2}, "12", 3, Rational(3), (x for x in (1, 2)), range(3)):
                    with self.subTest(operation=operation, argument=name, item=item, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: item}))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                for shape in (dict(args, extra=1), {"xs": args["xs"]}, {"x": args["xs"], "y": args["ys"]},
                              tuple(args.items()), None):
                    with self.assertRaises(InvalidInput):
                        family.produce(operation, shape)
        for operation in ("covariance", "Correlation", "linear_fit", "", None, 7):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(SMALL["correlation"]))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("correlation", xs=(2, 2, 2), ys=(1, 2, 3))
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in either argument and max_work=1 exceed the budget. No operation of the family
        takes a matrix, so support.oversized_matrix() has nothing to exceed here (as xs or ys it is invalid input,
        test_invalid_inputs). Work is charged before it runs, so a request that needs more than its budget fails at
        once; a common denominator or a value integer over max_integer_bits exceeds the budget too."""
        for operation, args in SMALL.items():
            for name in args:
                for item in (support.HUGE, -support.HUGE, Rational(1, support.HUGE), Rational(support.HUGE, 3)):
                    changed = args[name][:1] + (item,) + args[name][2:]
                    with self.subTest(operation=operation, argument=name, item=type(item).__name__):
                        self.assertResourceLimit(support.run(operation, **dict(args, **{name: changed})))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
        rng = random.Random(20261125)
        primes = primes_below(5000)  # 669 primes, whose product has 7,087 bits; the first 600 have 6,245.
        coprime = tuple(Rational(1, p) for p in primes) * 10
        long_data = [tuple(rng.getrandbits(65_536) for _ in range(400)) for _ in range(2)]
        start = time.perf_counter()
        for operation in OPERATIONS:
            with self.subTest(operation=operation, case="charged before it runs"):
                # 400 pairs of 65,536-bit integers: some 5 million units of products, refused at once.
                self.assertResourceLimit(support.run(operation, xs=long_data[0], ys=long_data[1],
                                                     limits=ComputationLimits(max_work=5_000)))
                self.assertResourceLimit(support.run(operation, xs=coprime, ys=tuple(range(len(coprime))),
                                                     limits=ComputationLimits(max_work=2_000)))
            with self.subTest(operation=operation, case="common denominator"):
                # Each denominator has 11 bits, their product 22: over a max_integer_bits of 20.
                self.assertResourceLimit(support.run(operation, xs=(Rational(1, 2003), Rational(1, 2011)), ys=(0, 1),
                                                     limits=ComputationLimits(max_integer_bits=20)))
                giant = tuple(Rational(1, p) for p in primes[:600])
                self.assertResourceLimit(support.run(operation, xs=giant, ys=tuple(range(len(giant))),
                                                     limits=ComputationLimits(max_integer_bits=4_096)))
        self.assertLess(time.perf_counter() - start, FAST)
        # Values that no Result can hold: ten pairs of 65,536-bit integers give a coefficient whose reduced q has about
        # four times as many bits, and a line whose slope has about twice and intercept three times as many; the
        # request itself fits.
        short = tuple(rng.getrandbits(65_536) for _ in range(10)), tuple(rng.getrandbits(65_536) for _ in range(10))
        start = time.perf_counter()
        for operation in OPERATIONS:
            with self.subTest(operation=operation, case="value integer bits"):
                result = support.run(operation, xs=short[0], ys=short[1])
                self.assertResourceLimit(result)
                self.assertIn("integer bits", result.error.message)
        self.assertLess(time.perf_counter() - start, FAST)
        # Within the limit by one bit: the slope 2**65535 of the line through (0, 0) and (1/2**65535, 1).
        result = support.run("linear_regression", xs=(0, Rational(1, 2 ** 65_535)), ys=(0, 1))
        self.assertVerified(result, LinearFit(Rational(2 ** 65_535), Rational(0)))
        # A request of 25,000 pairs exceeds the nodes that a request and its Result may hold.
        self.assertResourceLimit(support.run("correlation", xs=tuple(range(25_000)), ys=tuple(range(25_000))))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The evidence of both operations is always (), so the claims that the producer does not make differ in the
        form of the request or of the value object: ints written as Rationals of denominator 1, the pairs in another
        order, a canonical value made with object.__new__, and data that the producer never saw whose value follows
        from the producer's by hand (a positive scale or a shift of the data)."""
        for operation, xs, ys, value in KNOWN:
            args = arguments(xs, ys)
            produced, evidence = family.produce(operation, dict(args))
            self.assertEqual((produced, evidence), (value, ()))
            widened = {name: tuple(Rational(item) if type(item) is int else item for item in data)
                       for name, data in args.items()}
            permutation = list(range(len(xs)))
            random.Random(len(xs)).shuffle(permutation)
            shuffled = {name: tuple(data[i] for i in permutation) for name, data in args.items()}
            cases = [("Rationals for ints", widened, value), ("forged copy", args, forged_copy(value)),
                     ("reversed", {name: data[::-1] for name, data in args.items()}, value),
                     ("shuffled", shuffled, value)]
            for label, claim_args, claim in cases:
                with self.subTest(operation=operation, xs=xs, ys=ys, case=label):
                    self.assertAccepts(operation, claim_args, claim)
        # Data the producer never saw. r is invariant under positive scales and shifts of either variable and changes
        # sign with a negative scale; the line of (s x, y + t) is y = (slope / s) x + intercept + t.
        scale, shift = Fraction(3, 7), Fraction(-5, 2)
        for operation, xs, ys, value in KNOWN:
            moved_xs, moved_ys = [scale * frac(x) for x in xs], [frac(y) + shift for y in ys]
            if operation == "correlation":
                claims = ((moved_xs, moved_ys, value), ([-x for x in moved_xs], moved_ys, negated(value)))
            else:
                claims = ((moved_xs, moved_ys, LinearFit(rational(frac(value.slope) / scale),
                                                         rational(frac(value.intercept) + shift))),)
            for claim_xs, claim_ys, claim in claims:
                with self.subTest(operation=operation, xs=xs, ys=ys, data="moved"):
                    self.assertEqual(REFERENCES[operation](claim_xs, claim_ys), claim)
                    self.assertAccepts(operation, arguments(claim_xs, claim_ys), claim)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where a Rational belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        claims = {
            "correlation": [(dict(xs=(1, 2, 3), ys=(2, 4, 7)), [Rational(99, 100), Sqrt(Rational(76, 75)),
                                                               Mul((Rational(-1), Sqrt(Q))), Rational(1)]),
                            (dict(xs=(1, 2, 3, 4), ys=(1, 3, 2, 4)), [Rational(3, 5), Rational(-4, 5),
                                                                     Sqrt(Rational(4, 5)), Rational(16, 25)]),
                            (dict(xs=(1, 2, 3), ys=(7, 4, 2)), [Sqrt(Q), Mul((Rational(-2), Sqrt(Q)))])],
            "linear_regression": [(dict(xs=(1, 2, 3), ys=(2, 4, 7)), [LinearFit(Rational(5, 2), Rational(2, 3)),
                                                                     LinearFit(Rational(-2, 3), Rational(5, 2)),
                                                                     LinearFit(Rational(5, 2), Rational(-1, 2))]),
                                  (dict(xs=(Rational(1, 2), Rational(1, 3)), ys=(2, 7)),
                                   [LinearFit(Rational(30), Rational(17)), LinearFit(Rational(-30), Rational(0))])],
        }
        for operation, requests in claims.items():
            for args, wrong in requests:
                result = support.run(operation, **args)
                value, evidence = result.value, result.steps[0].outputs[1]
                self.assertVerified(result, REFERENCES[operation](args["xs"], args["ys"]))
                self.assertAccepts(operation, args, value, evidence)
                cases: list[tuple[str, object, object, object]] = [(f"wrong {item}", args, item, ()) for item in wrong]
                cases += [("evidence as a list", args, value, []), ("evidence None", args, value, None),
                          ("an extra evidence item", args, value, ((),)), ("a number as evidence", args, value, (0,)),
                          ("oversized evidence", args, value, (0,) * 100_000), ("a giant evidence item", args, value,
                                                                                (GIANT,)),
                          ("xs as a list", dict(args, xs=list(args["xs"])), value, ()),
                          ("ys as a list", dict(args, ys=list(args["ys"])), value, ()),
                          ("the value in a tuple", args, (value,), ()), ("the value in a list", args, [value], ()),
                          ("a giant value", args, forged_rational(GIANT), ())]
                if operation == "correlation":
                    cases += self.correlation_corruptions(args, value)
                else:
                    cases += self.fit_corruptions(args, value)
                for label, claim_args, item, proof in cases:
                    with self.subTest(operation=operation, args=args, case=label):
                        self.assertRejects(operation, claim_args, item, proof)
                other = "linear_regression" if operation == "correlation" else "correlation"
                for name in (other, operation.upper(), "covariance", ""):
                    with self.subTest(operation=operation, renamed=name):
                        self.assertRejects(name, args, value, evidence)
                report = support.statuses(mf.verify(support.tampered(result, wrong[0])))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    @staticmethod
    def correlation_corruptions(args: dict[str, tuple], value: Any) -> list[tuple[str, object, object, object]]:
        """Mathematically right correlation values in another form than the canonical one, and stand-ins."""
        cases: list[tuple[str, object, object, object]] = [
            ("an impostor", args, support.impostor(value), ()),
            ("a float", args, 0.5, ()), ("a str", args, str(value), ()), ("a bool", args, True, ()),
            ("a Fraction", args, Fraction(1, 2), ()), ("an int", args, 1, ()),
        ]
        if type(value) is Rational:
            p, s = value.numerator, value.denominator
            cases += [("the radical of a rational square", args, Sqrt(Rational(p * p, s * s)), ()),
                      ("unreduced", args, forged_rational(2 * p, 2 * s), ()),
                      ("a negative denominator", args, forged_rational(-p, -s), ()),
                      ("times 1", args, Mul((value, Rational(1))), ()), ("to the first power", args, Pow(value, 1), ()),
                      ("a float", args, p / s, ())]
            if s == 1:
                cases.append(("the int", args, p, ()))
            return cases
        radical = value if type(value) is Sqrt else value.factors[1]
        a, b = radical.radicand.numerator, radical.radicand.denominator
        sign = Rational(1) if type(value) is Sqrt else Rational(-1)

        def signed(node: Any) -> Any:
            """The node with the sign of the value: itself, or Rational(-1) times it."""
            return node if sign > 0 else Mul((Rational(-1), node))

        # sign * sqrt(a/b) = 5 sign sqrt(a / (25 b)) = (sign / 2) sqrt(4 a / b): right values, never canonical.
        taken_out = Mul((sign * 5, Sqrt(Rational(a, 25 * b)))) if a % 25 == 0 else Mul((sign / 2,
                                                                                       Sqrt(Rational(4 * a, b))))
        cases += [("a factor taken out", args, taken_out, ()),
                  ("the radicand unreduced", args, signed(support.forged(Sqrt, radicand=forged_rational(2 * a, 2 * b))),
                   ()),
                  ("the factors swapped", args, support.forged(Mul, factors=(radical, sign)), ()),
                  ("the sign as an int", args, support.forged(Mul, factors=(sign.numerator, radical)), ()),
                  ("the radicand as a Fraction", args, support.forged(Sqrt, radicand=Fraction(a, b)), ()),
                  ("a Sqrt impostor", args, support.impostor(radical), ()),
                  ("the radical to the first power", args, Pow(value, 1), ())]
        return cases

    @staticmethod
    def fit_corruptions(args: dict[str, tuple], value: LinearFit) -> list[tuple[str, object, object, object]]:
        """The right line in a form that no LinearFit record holds, or in another class."""
        slope, intercept = value.slope, value.intercept
        cases: list[tuple[str, object, object, object]] = [
            ("an impostor record", args, support.impostor(value), ()),
            ("a tuple", args, (slope, intercept), ()),
            ("a dict", args, {"slope": slope, "intercept": intercept}, ()),
            ("no intercept", args, support.forged(LinearFit, slope=slope), ()),
            ("no fields", args, support.forged(LinearFit), ()),
            ("Fractions", args, support.forged(LinearFit, slope=frac(slope), intercept=frac(intercept)), ()),
            ("floats", args, support.forged(LinearFit, slope=slope.numerator / slope.denominator,
                                            intercept=intercept.numerator / intercept.denominator), ()),
            ("strs", args, support.forged(LinearFit, slope=str(slope), intercept=str(intercept)), ()),
            ("bools", args, support.forged(LinearFit, slope=True, intercept=False), ()),
            ("the slope unreduced", args, support.forged(LinearFit, slope=forged_rational(2 * slope.numerator,
                                                                                         2 * slope.denominator),
                                                         intercept=intercept), ()),
        ]
        if slope.denominator == 1:
            cases.append(("an int slope", args, support.forged(LinearFit, slope=slope.numerator, intercept=intercept),
                          ()))
            cases.append(("an int subclass slope", args, support.forged(LinearFit, slope=Integer(slope.numerator),
                                                                        intercept=intercept), ()))
        return cases

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [(operation, dict(xs=xs, ys=ys)) for operation, xs, ys, _ in KNOWN]
        rng = random.Random(20261126)
        requests += [(operation, dict(xs=tuple(rng.randrange(-10 ** 9, 10 ** 9) for _ in range(500)),
                                      ys=tuple(Rational(rng.randrange(-999, 999), rng.randrange(1, 99))
                                               for _ in range(500)))) for operation in OPERATIONS]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, _), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, value=result.value):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("correlation", xs=(1, 2, 3), ys=(2, 4, 7))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: each shape of the
        coefficient (a Rational, a positive and a negative radical) and lines with every kind of slope and intercept."""
        rendered = (
            ("correlation", (1, 2, 3), (2, 4, 6), "1", "1"),
            ("correlation", (1, 2, 3), (3, 2, 1), "-1", "-1"),
            ("correlation", (1, 2, 3, 4), (1, 3, 2, 4), "4/5", r"\frac{4}{5}"),
            ("correlation", (1, 2, 3), (1, 0, 1), "0", "0"),
            ("correlation", (-1, 0, 1), (1, 2, 0), "-1/2", r"-\frac{1}{2}"),
            ("correlation", (1, 2, 3), (2, 4, 7), "sqrt(75/76)", r"\sqrt{\frac{75}{76}}"),
            ("correlation", (1, 2, 3), (7, 4, 2), "-sqrt(75/76)", r"-\sqrt{\frac{75}{76}}"),
            ("linear_regression", (1, 2, 3), (3, 5, 7), "y = 2*x + 1", r"y = 2\, x + 1"),
            ("linear_regression", (0, 1, 2), (1, 2, 4), "y = 3/2*x + 5/6", r"y = \frac{3}{2}\, x + \frac{5}{6}"),
            ("linear_regression", (1, 2, 3), (2, 4, 7), "y = 5/2*x - 2/3", r"y = \frac{5}{2}\, x - \frac{2}{3}"),
            ("linear_regression", (Rational(1, 2), Rational(1, 3)), (2, 7), "y = -30*x + 17", r"y = -30\, x + 17"),
            ("linear_regression", (1, 2, 3), (1, 2, 3), "y = x", "y = x"),
            ("linear_regression", (1, 2, 3), (3, 2, 1), "y = -x + 4", "y = -x + 4"),
            ("linear_regression", (1, 2, 3), (2, 4, 6), "y = 2*x", r"y = 2\, x"),
            ("linear_regression", (1, 2, 3), (4, 4, 4), "y = 4", "y = 4"),
            ("linear_regression", (1, 2, 3), (1, 0, 1), "y = 2/3", r"y = \frac{2}{3}"),
            ("linear_regression", (1, 2), (0, 0), "y = 0", "y = 0"),
        )
        for operation, xs, ys, text, latex in rendered:
            with self.subTest(operation=operation, xs=xs, ys=ys):
                result = support.run(operation, xs=xs, ys=ys)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("linear_regression", xs=(1, 2, 3), ys=(2, 4, 7)).value)["data"],
                         {"type": "v6_linear_fit", "slope": {"type": "rational", "numerator": "5", "denominator": "2"},
                          "intercept": {"type": "rational", "numerator": "-2", "denominator": "3"}})
        self.assertEqual(mf.to_data(support.run("correlation", xs=(1, 2, 3), ys=(7, 4, 2)).value)["data"],
                         {"type": "mul", "factors": [{"type": "rational", "numerator": "-1", "denominator": "1"},
                                                     {"type": "sqrt", "radicand": {"type": "rational",
                                                                                   "numerator": "75",
                                                                                   "denominator": "76"}}]})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261127)
        cases = []
        while len(cases) < 40:
            xs, ys = seeded_data(rng)
            for operation in OPERATIONS:
                if REFERENCES[operation](xs, ys) is not None:
                    cases.append((operation, arguments(xs, ys)))
        cases += [(operation, dict(xs=[1, 2, 3], ys=[2, 4, 7])) for operation in OPERATIONS]  # Lists, as typed.
        for operation, args in cases:
            with self.subTest(operation=operation, **args):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertIs(type(found), type(expected))
                self.assertEqual(support.value(operation, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, args in SMALL.items():
            with self.subTest(operation=operation):
                result = self.answer(support.call(operation, {name: list(data) for name, data in args.items()}))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        # Rationals travel as "p/q" strings and integers from 2**53 on as decimal strings.
        values = dict(xs=(Rational(1, 2), -3, 2 ** 70, Rational(-7, 4)), ys=(Rational(2, 3), 5, -1, 2 ** 60 + 1))
        for operation in OPERATIONS:
            with self.subTest(operation=operation, encoded=True):
                result = self.answer(support.call_values(operation, **values))
                self.assertEqual(result, support.run(operation, **values))
                self.assertEqual(result.value, REFERENCES[operation](values["xs"], values["ys"]))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("correlation", {"xs": [2, 2], "ys": [1, 2]}), "invalid_input")
        self.assertIn("xs that are not all equal", error["message"])
        error = self.assertToolError(support.call("linear_regression", {"xs": [1, 2, 3], "ys": [1, 2]}),
                                     "invalid_input")
        self.assertIn("the same length", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the slow pairwise references of this module on at least 250 seeded inputs per
        operation, each value of its exact type with the evidence (), and the checker accepts every answer; the
        inputs outside a domain are refused as invalid input, lists give the values of tuples, and every shape of
        the coefficient is met."""
        rng = random.Random(20261128)
        counts = dict.fromkeys(OPERATIONS, 0)
        refused = dict.fromkeys(OPERATIONS, 0)
        shapes = {Rational: 0, Sqrt: 0, Mul: 0}
        for _ in range(320):  # A fixed number of datasets, so that a failing producer cannot keep the loop going.
            xs, ys = seeded_data(rng)
            args = arguments(xs, ys)
            for operation in OPERATIONS:
                expected = REFERENCES[operation](xs, ys)
                with self.subTest(operation=operation, xs=xs, ys=ys):
                    if expected is None:
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args))
                        refused[operation] += 1
                        continue
                    value, evidence = family.produce(operation, dict(args))
                    self.assertEqual(value, expected)
                    self.assertIs(type(value), type(expected))
                    self.assertEqual(evidence, ())
                    self.assertEqual(family.produce(operation, {name: list(data) for name, data in args.items()}),
                                     (value, ()))
                    self.assertIs(support.verdict(family, operation, args, value, evidence), True)
                    counts[operation] += 1
                    if operation == "correlation":
                        shapes[type(value)] += 1
        self.assertGreaterEqual(min(counts.values()), 250, counts)
        self.assertTrue(all(shapes.values()), shapes)
        self.assertTrue(all(refused.values()), refused)  # Both domains were left too.

    def test_full_size_inputs_fit_the_default_budget(self):
        """20,000 pairs of 64-bit integers, or 8,000 pairs of rationals, complete and verify under the default limits,
        with the values of the references (here the centred sums in Fractions: the pairwise loop is quadratic)."""
        rng = random.Random(20261129)
        xs = tuple(rng.randrange(-(2 ** 64), 2 ** 64) for _ in range(20_000))
        ys = tuple(x // 3 + rng.randrange(-(2 ** 62), 2 ** 62) for x in xs)
        ratios = (tuple(Rational(rng.randrange(-1000, 1000), rng.randrange(1, 256)) for _ in range(8000)),
                  tuple(Rational(rng.randrange(-1000, 1000), rng.randrange(1, 256)) for _ in range(8000)))
        for label, data in (("integers", (xs, ys)), ("rationals", ratios)):
            fx, fy = [frac(x) for x in data[0]], [frac(y) for y in data[1]]
            mx, my = sum(fx, Fraction(0)) / len(fx), sum(fy, Fraction(0)) / len(fy)
            sxx = sum(((x - mx) ** 2 for x in fx), Fraction(0))
            syy = sum(((y - my) ** 2 for y in fy), Fraction(0))
            sxy = sum(((x - mx) * (y - my) for x, y in zip(fx, fy)), Fraction(0))
            q = sxy * sxy / (sxx * syy)
            self.assertIsNone(rational_root(q))
            slope = sxy / sxx
            expected = {"correlation": Mul((Rational(-1), Sqrt(rational(q)))) if sxy < 0 else Sqrt(rational(q)),
                        "linear_regression": LinearFit(rational(slope), rational(my - slope * mx))}
            for operation in OPERATIONS:
                with self.subTest(data=label, operation=operation):
                    with computation(fresh=True) as budget:
                        self.assertEqual(family.produce(operation, {"xs": data[0], "ys": data[1]}),
                                         (expected[operation], ()))
                    self.assertLess(budget.work, ComputationLimits().max_work // 4)
                    start = time.perf_counter()
                    result = support.run(operation, xs=data[0], ys=data[1])
                    self.assertLess(time.perf_counter() - start, 2 * FAST)
                    self.assertVerified(result, expected[operation])

    def test_rational_coefficient_whose_square_is_too_long(self):
        """Review regression: a rational coefficient r is computed and verified although q = r^2 has more bits than
        max_integer_bits; only the value's integers must fit. The ys permute the xs, so S_yy = S_xx and
        r = S_xy / S_xx is rational, with about 34,000 bits for 17,000-bit data while q has about 68,000. A radical
        holds q itself, so there the bound still applies (test_resource_limits)."""
        rng = random.Random(20261113)
        xs = tuple(rng.getrandbits(17_000) | 1 << 16_999 for _ in range(4))
        ys = xs[1:] + xs[:1]
        expected = reference_correlation(xs, ys)
        self.assertIs(type(expected), Rational)
        limit = ComputationLimits().max_integer_bits
        self.assertGreater((expected.numerator ** 2).bit_length(), limit)
        self.assertLessEqual(max(expected.numerator.bit_length(), expected.denominator.bit_length()), limit)
        with computation(fresh=True):
            self.assertEqual(family.produce("correlation", {"xs": xs, "ys": ys}), (expected, ()))
        self.assertVerified(support.run("correlation", xs=xs, ys=ys), expected)

    def test_work_is_charged_before_it_runs(self):
        """The producer's charges depend on the request alone: a budget one unit short refuses it, its exact size
        passes. Long integers cost more than short ones, by the products of their length, and the square tests of
        the radical are paid before they run."""
        rng = random.Random(20261130)
        xs = [Fraction(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 50)) for _ in range(200)]
        ys = [Fraction(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 50)) for _ in range(200)]
        args = arguments(xs, ys)
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    value, _ = family.produce(operation, dict(args))
                work = budget.work
                self.assertGreater(work, 3 * 200)  # At least a unit per item and per pair.
                with self.assertRaises(ResourceLimitError), computation(ComputationLimits(max_work=work - 1),
                                                                        fresh=True):
                    family.produce(operation, dict(args))
                with computation(ComputationLimits(max_work=work), fresh=True) as budget:
                    self.assertEqual(family.produce(operation, dict(args)), (value, ()))
                self.assertEqual(budget.work, work)
        # The same data scaled by a 20,000-bit factor: each pair pays for its products (X_i^2, X_i Y_i, and Y_i^2 for
        # correlation) by their length, as trusted.integers.product_work prices them. Twenty pairs more cost at least
        # their products: the steps after the pass over the pairs (the spreads, the reductions) grow by a few bits only.
        from mathforge.trusted.integers import product_work

        scale = 2 ** 20_000 + 1
        claims = [arguments(xs[:20], ys[:20])]
        claims += [arguments([x * scale for x in xs[:count]], [y * scale + 1 for y in ys[:count]])
                   for count in (20, 40)]
        for operation, products in (("correlation", 3), ("linear_regression", 2)):
            charged = []
            for claim in claims:
                with computation(fresh=True) as budget:
                    family.produce(operation, dict(claim))
                charged.append(budget.work)
            with self.subTest(operation=operation, charged=charged):
                self.assertGreater(charged[1], 2 * charged[0])
                self.assertGreaterEqual(charged[2] - charged[1], 20 * products * product_work(20_000, 20_000))
        # The square tests of the radical (math.isqrt inside mathforge.model.sqrt, which mul repeats) are paid before
        # model.sqrt runs: at least two products of their length for each of the six of a reduced q.
        calls: list[str] = []
        charge, root = family._charge_radical, family.sqrt
        with patch.object(family, "_charge_radical", lambda budget, q: (calls.append("charge"), charge(budget, q))), \
                patch.object(family, "sqrt", lambda q: (calls.append("sqrt"), root(q))[1]):
            self.assertEqual(family.produce("correlation", {"xs": (1, 2, 3), "ys": (2, 4, 7)}), (Sqrt(Q), ()))
        self.assertEqual(calls, ["charge", "sqrt"])
        long_q = Rational(rng.getrandbits(64_000) | 1 << 63_999, rng.getrandbits(64_000) | 1 << 63_999)
        bits = min(long_q.numerator.bit_length(), long_q.denominator.bit_length())
        with computation(fresh=True) as budget:
            family._charge_radical(budget, long_q)
        self.assertGreaterEqual(budget.work, 6 * 2 * product_work(bits, bits))

    def test_family_module_imports(self):
        """The family module imports at its top only modules that `import mathforge` loads, and nowhere the checker
        layer, the lean path, the MCP layer or generated code; it builds the radical with mathforge.model, the card's
        construction."""
        package = family.__name__.split(".")[:-1]  # mathforge.areas.stats

        def absolute(node: ast.ImportFrom) -> str:
            """The module that an import names, a relative one resolved against the family's package."""
            base = package[:len(package) - node.level + 1] if node.level else []
            return ".".join([*base, node.module or ""])

        tree = ast.parse(inspect.getsource(family))
        top = {absolute(node) for node in tree.body if isinstance(node, ast.ImportFrom)}
        self.assertEqual(top, {"__future__", "collections.abc", "dataclasses", "math", "typing", "mathforge.errors",
                               "mathforge.limits", "mathforge.model", "mathforge.registry"})
        self.assertFalse([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
        everywhere = {absolute(node) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        for name in everywhere:
            self.assertFalse(any(part in name.split(".") for part in ("certify", "lean", "mcp", "_generated")), name)
        # The cost model, imported where it charges: import mathforge does not load it.
        self.assertEqual(everywhere - top, {"mathforge.trusted.integers"})
