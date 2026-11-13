"""The approximation family of mathforge.numeric: rational_approximation.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register numeric approximation

Every expected value comes from the card or from a reference of this module, never from the producer. For a rational
x the reference is fractions.Fraction.limit_denominator, which returns a closest fraction with denominator <= N, with
the card's tie rule applied by hand: on a tie the other closest fraction is 2x - r, and the one with the smaller
denominator (then the smaller one) wins; brute force over every denominator q <= N checks it for small N. For an
irrational x the reference runs on a dyadic bracket lo < x < hi of width 2**-K computed with integer arithmetic of
its own (math.isqrt, an integer Newton iteration for k-th roots): the answer is a step function of x whose steps are
the fractions with denominator <= N and the midpoints of Farey neighbours, any two of them at least 1/(4 N**4)
apart, so once K > 4 log2(N) + 2 at most one step lies inside the bracket, and when the reference answers the same
at both ends it is the answer for every irrational number between them. The Farey neighbours (l, u) of the evidence
are unique given u.numerator * l.denominator - l.numerator * u.denominator == 1, l.denominator + u.denominator > N
and l < x < u (the checker's theorem), so the tests check those three facts, with l <= lo and hi <= u for an
irrational x. tests/v09/numeric/test_approximation_checker.py tests the checker on its own.

The card has no deferred (Unevaluated) case for rational_approximation: every value in scope gets its approximation
or exceeds the budget.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.numeric import approximation as family
from mathforge.certify.numeric import approximation as checker
from mathforge.contracts import ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Sqrt

from .. import support

Q = mf.Rational
OP = "rational_approximation"
FAST = 3.0  # Seconds: a request that its budget refuses fails, and a full-size request completes, well within this.
BIG = 2 ** 64  # The largest max_denominator of the card.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact number: a float, a bool, a str, None and two containers.
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
SMALL = dict(value=Q(2, 3), max_denominator=5)  # One small request, for the argument tests.


class Integer(int):
    """An int subclass: no exact int for the family, which tests types with `type(x) is int`."""


# ---------------------------------------------------------------------------------------------------------------
# Values

def sqrt(value: object) -> Sqrt:
    """A Sqrt node of a rational, never simplified (mf.sqrt would return a Rational for a square)."""
    value = Fraction(value)
    return Sqrt(Q(value.numerator, value.denominator))


def rational(value: object) -> Q:
    value = Fraction(value)
    return Q(value.numerator, value.denominator)


def root(coefficients: tuple[int, ...], index: int) -> mf.RealAlgebraicRoot:
    return mf.RealAlgebraicRoot(coefficients, index)


def fraction(value: Q) -> Fraction:
    return Fraction(value.numerator, value.denominator)


# ---------------------------------------------------------------------------------------------------------------
# References

def best_rational(x: Fraction, n: int) -> Fraction:
    """The card's answer for a rational x: Fraction.limit_denominator gives a closest fraction r with denominator
    <= n; the only other candidate at the same distance is 2x - r, and on such a tie the smaller denominator wins,
    then the smaller fraction."""
    r = x.limit_denominator(n)
    other = 2 * x - r
    if other != r and other.denominator <= n:
        return min(r, other, key=lambda s: (s.denominator, s))
    return r


def brute_force(x: Fraction, n: int) -> Fraction:
    """The card's answer for a rational x over every denominator q <= n: floor(x*q)/q and the next fraction up,
    ordered by the distance, then the denominator, then the value."""
    candidates = []
    for q in range(1, n + 1):
        p = x.numerator * q // x.denominator
        candidates += [Fraction(p, q), Fraction(p + 1, q)]
    return min(candidates, key=lambda s: (abs(x - s), s.denominator, s))


def farey_pair(x: Fraction, n: int) -> tuple[Fraction, Fraction]:
    """The Farey neighbours l < x < u of order n of a rational x with denominator > n, from the closest fraction
    r = best_rational(x, n), which is one of them, by modular inverses: for r = a/b < x, u = c/d is the neighbour with
    c*b - a*d == 1 and the largest d <= n, so d is the largest d <= n with d == -a**-1 (mod b); for r = c/d > x, l = a/b
    has the largest b <= n with b == c**-1 (mod d)."""
    r = best_rational(x, n)
    if r < x:
        a, b = r.numerator, r.denominator
        d = n - (n + pow(a, -1, b)) % b if b > 1 else n
        return r, Fraction((1 + a * d) // b, d)
    c, d = r.numerator, r.denominator
    b = n - (n - pow(c, -1, d)) % d if d > 1 else n
    return Fraction((c * b - 1) // d, b), r


def iroot(n: int, m: int) -> int:
    """floor(n ** (1/m)) for n >= 0 and m >= 1: math.isqrt for squares, otherwise the integer Newton iteration from
    above (it decreases strictly until it reaches the floor)."""
    if m == 1 or n < 2:
        return n
    if m == 2:
        return isqrt(n)
    x = 1 << -(-n.bit_length() // m)
    while True:
        y = ((m - 1) * x + n // x ** (m - 1)) // m
        if y >= x:
            return x
        x = y


def sqrt_bracket(a: Fraction, k: int) -> tuple[Fraction, Fraction]:
    """lo < sqrt(a) < hi with hi - lo = 2**-k, for a rational a >= 0 that is no square: floor(sqrt(z)) is
    isqrt(floor(z)) for z = a * 4**k."""
    s = isqrt(a.numerator * 4 ** k // a.denominator)
    return Fraction(s, 2 ** k), Fraction(s + 1, 2 ** k)


def linear_bracket(c: Fraction, d: Fraction, a: Fraction, k: int) -> tuple[Fraction, Fraction]:
    """A bracket of c + d*sqrt(a), d != 0, a no square."""
    lo, hi = sqrt_bracket(a, k)
    return (c + d * lo, c + d * hi) if d > 0 else (c + d * hi, c + d * lo)


def binomial(q: int, s: int, m: int, n: int) -> tuple[int, ...]:
    """The coefficients, lowest degree first, of (q*x - s)**m - n."""
    coefficients = [comb(m, j) * q ** j * (-s) ** (m - j) for j in range(m + 1)]
    coefficients[0] -= n
    return tuple(coefficients)


def binomial_bracket(q: int, s: int, m: int, n: int, t: int, k: int) -> tuple[Fraction, Fraction]:
    """A bracket of x = (s + t*y)/q, y = n ** (1/m) for n >= 2 no perfect m-th power, t = -1 or 1: the integer
    m-th root z = floor(y * 2**k) gives z/2**k < y < (z + 1)/2**k."""
    z = iroot(n * 2 ** (m * k), m)
    lo, hi = Fraction(z, 2 ** k), Fraction(z + 1, 2 ** k)
    ends = sorted((Fraction(s) + t * lo, Fraction(s) + t * hi))
    return ends[0] / q, ends[1] / q


def irrational_reference(bracket, n: int) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    """(the answer, a bracket lo < x < hi on which best_rational answers it at both ends) for an irrational x
    given by bracket(k): a precision K > 4 log2(n) + 2 first, doubled until the two ends agree (module docstring)."""
    k = 4 * n.bit_length() + 16
    while True:
        lo, hi = bracket(k)
        found = best_rational(lo, n)
        if found == best_rational(hi, n):
            return found, (lo, hi)
        k *= 2


def is_square(a: Fraction) -> bool:
    return isqrt(a.numerator) ** 2 == a.numerator and isqrt(a.denominator) ** 2 == a.denominator


# The known values of the card, and a few more: (value, N, answer, evidence).
KNOWN = (
    (sqrt(2), 10, Q(7, 5), (Q(7, 5), Q(10, 7))),
    (Q(314159265, 100000000), 1000, Q(355, 113), (Q(2818, 897), Q(355, 113))),
    (Q(1, 3), 2, Q(1, 2), (Q(0), Q(1, 2))),
    (Q(1, 2), 1, Q(0), (Q(0), Q(1))),  # A tie of equal denominators: the smaller.
    (Q(3, 7), 7, Q(3, 7), ()),
    (Mul((Q(-1), sqrt(2))), 10, Q(-7, 5), (Q(-10, 7), Q(-7, 5))),
    (root((-2, 0, 1), 0), 10, Q(-7, 5), (Q(-10, 7), Q(-7, 5))),  # -sqrt(2) as a root.
    (Q(5, 12), 4, Q(1, 2), (Q(1, 3), Q(1, 2))),  # A tie: the smaller denominator, on the right.
    (Q(-3, 4), 2, Q(-1), (Q(-1), Q(-1, 2))),  # A tie: the smaller denominator, on the left.
    (0, 5, Q(0), ()),
    (7, 1, Q(7), ()),
    (root((2, -4, -1, 2), 1), 2, Q(1, 2), ()),  # (2x - 1)(x**2 - 2): the rational root 1/2.
    (root((2, -4, -1, 2), 1), 1, Q(0), (Q(0), Q(1))),  # The same root at a tie.
    (Mul((sqrt(2), sqrt(2), Q(1, 4))), 1, Q(0), (Q(0), Q(1))),  # 1/2 as an expression, at a tie.
    (Mul((sqrt(2), sqrt(2), Q(1, 3))), 3, Q(2, 3), ()),
    (root((-2, 0, 0, 1), 0), 100, Q(63, 50), (Q(97, 77), Q(63, 50))),  # The cube root of 2, 1.2599...
)


class ApproximationTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with its four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), Q)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def produced(self, value: object, n: int) -> tuple[Q, tuple]:
        """produce()'s answer, which the checker accepts; its types are those of the card."""
        args = {"value": value, "max_denominator": n}
        found, evidence = family.produce(OP, dict(args))
        self.assertIs(type(found), Q)
        self.assertIs(type(evidence), tuple)
        self.assertTrue(all(type(item) is Q for item in evidence))
        self.assertIs(support.verdict(family, OP, args, found, evidence), True)
        return found, evidence

    def assertFarey(self, evidence: tuple, n: int, lo: Fraction, hi: Fraction) -> None:
        """The evidence is the Farey pair of order n around every x with lo <= x <= hi (module docstring)."""
        self.assertEqual(len(evidence), 2)
        left, right = (fraction(item) for item in evidence)
        self.assertEqual(right.numerator * left.denominator - left.numerator * right.denominator, 1)
        self.assertTrue(left.denominator <= n and right.denominator <= n < left.denominator + right.denominator)
        self.assertTrue(left <= lo and hi <= right, (left, lo, hi, right))

    def assertRational(self, x: Fraction, n: int, value: object = None) -> None:
        """produce() answers the reference for the rational x, with () exactly when x has denominator <= n."""
        found, evidence = self.produced(rational(x) if value is None else value, n)
        self.assertEqual(fraction(found), best_rational(x, n), (x, n))
        if x.denominator <= n:
            self.assertEqual(evidence, ())
        else:
            self.assertFarey(evidence, n, x, x)
            self.assertNotIn(x, (fraction(item) for item in evidence))
            self.assertEqual(tuple(fraction(item) for item in evidence), farey_pair(x, n))

    def assertIrrational(self, value: object, n: int, bracket) -> None:
        """produce() answers the bracket reference for an irrational value, with its Farey pair."""
        found, evidence = self.produced(value, n)
        expected, (lo, hi) = irrational_reference(bracket, n)
        self.assertEqual(fraction(found), expected, (str(value), n))
        self.assertEqual(len(evidence), 2)
        left, right = (fraction(item) for item in evidence)
        k = 4 * n.bit_length() + 16
        while not (left <= lo and hi <= right) and k < 100_000:  # A neighbour inside the bracket: narrow it.
            k *= 2
            lo, hi = bracket(k)
        self.assertFarey(evidence, n, lo, hi)

    # -----------------------------------------------------------------------------------------------------------

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts, with the card's evidence."""
        self.assertEqual(family.MAX_DENOMINATOR, checker.MAX_DENOMINATOR)
        for value, n, expected, evidence in KNOWN:
            with self.subTest(value=str(value), n=n):
                result = support.run(OP, value=value, max_denominator=n)
                self.assertVerified(result, expected)
                self.assertEqual(result.steps[0].outputs[1], evidence)
                self.roundtrip(result)
        # The references agree with the card.
        self.assertEqual(best_rational(Fraction(314159265, 100000000), 1000), Fraction(355, 113))
        self.assertEqual(best_rational(Fraction(1, 2), 1), 0)
        self.assertEqual(best_rational(Fraction(5, 12), 4), Fraction(1, 2))
        self.assertEqual(irrational_reference(lambda k: sqrt_bracket(Fraction(2), k), 10)[0], Fraction(7, 5))
        self.assertEqual(irrational_reference(lambda k: binomial_bracket(1, 0, 3, 2, 1, k), 100)[0],
                         Fraction(63, 50))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the domain of the card. produce() tests the types itself too."""
        for name in SMALL:
            for item in STAND_INS:
                with self.subTest(argument=name, item=item):
                    self.assertInvalidInput(support.run(OP, **dict(SMALL, **{name: item})))
        x = mf.symbol("x")
        invalid = (
            dict(value=1, max_denominator=0), dict(value=sqrt(2), max_denominator=-1),
            dict(value=Q(1, 3), max_denominator=BIG + 1), dict(value=sqrt(2), max_denominator=-(10 ** 30)),
            dict(value=x, max_denominator=3), dict(value=Add((x, sqrt(2))), max_denominator=3),
            dict(value=Pow(x, 2), max_denominator=3),
            dict(value=Add((root((-2, 0, 0, 1), 0), sqrt(2))), max_denominator=3),
            dict(value=Mul((Q(2), root((-2, 0, 0, 1), 0))), max_denominator=3),
            dict(value=mf.Matrix(((1, 2), (3, 4))), max_denominator=3),
            # rational_approximation takes no matrix: an oversized one is out of scope before it is out of budget.
            dict(value=support.oversized_matrix(), max_denominator=3),
        )
        for args in invalid:
            with self.subTest(value=str(args["value"])[:40], n=args["max_denominator"]):
                self.assertInvalidInput(support.run(OP, **args))
        self.assertIn("from 1 to 2**64", support.run(OP, value=1, max_denominator=0).error.message)
        # Records made with object.__new__ outside their canonical form, which no decoded document holds.
        forged_values = (
            support.forged(Q, numerator=4, denominator=6), support.forged(Q, numerator=1, denominator=0),
            support.forged(Q, numerator=1, denominator=-2), support.forged(Q, numerator=True, denominator=1),
            support.forged(mf.RealAlgebraicRoot, integer_coefficients=(4, 0, -4, 0, 1), real_index=0),  # (x^2-2)^2
            support.forged(mf.RealAlgebraicRoot, integer_coefficients=(-4, 0, 2), real_index=0),  # Not primitive.
            support.forged(mf.RealAlgebraicRoot, integer_coefficients=(2, 0, -1), real_index=0),  # Negative lead.
            support.forged(mf.RealAlgebraicRoot, integer_coefficients=(-2, 0, 1), real_index=2),  # Two roots.
            support.forged(mf.RealAlgebraicRoot, integer_coefficients=(-2, 0, 1), real_index=True),
            support.forged(mf.RealAlgebraicRoot, integer_coefficients=[-2, 0, 1], real_index=0),
            support.forged(mf.RealAlgebraicRoot, integer_coefficients=(3,), real_index=0),
            support.forged(Sqrt, radicand=Q(-2)), support.forged(Sqrt, radicand=sqrt(2)),
            support.forged(Sqrt, radicand=2),
            support.forged(Pow, base=sqrt(2), exponent=-1), support.forged(Pow, base=sqrt(2), exponent=True),
            support.forged(Add, terms=(sqrt(2),)), support.forged(Add, terms=[sqrt(2), Q(1)]),
            support.forged(Add, terms=(1, sqrt(2))), support.forged(Mul, factors=(sqrt(2), 0.5)),
        )
        for number, item in enumerate(forged_values):
            with self.subTest(forged=type(item).__name__, number=number):
                with self.assertRaises(InvalidInput):
                    family.produce(OP, {"value": item, "max_denominator": 3})
                self.assertInvalidInput(support.run(OP, value=item, max_denominator=3))
        # A node inside itself, and a forged root without its fields: produce() refuses them (operations_v6.run
        # reads the arguments before produce() runs, so these reach produce() only directly).
        loop = support.forged(Add)
        object.__setattr__(loop, "terms", (sqrt(2), loop))
        for item in (loop, support.forged(mf.RealAlgebraicRoot, real_index=0)):
            with self.subTest(forged=type(item).__name__), self.assertRaises(InvalidInput):
                family.produce(OP, {"value": item, "max_denominator": 3})
        for item in (True, False, 0.5, "1", None, Integer(3), Fraction(1, 2), 1j, 0, -5, BIG + 1, 2 ** 70_000):
            names = SMALL if type(item) is not int else ("max_denominator",)
            for name in names:
                label = item if type(item) is not int else f"int of {item.bit_length()} bits"
                with self.subTest(argument=name, item=label, path="produce"), self.assertRaises(InvalidInput):
                    family.produce(OP, dict(SMALL, **{name: item}))
        with self.assertRaises(InvalidInput):
            family.produce(OP, dict(SMALL, extra=1))
        with self.assertRaises(InvalidInput):
            family.produce(OP, {"value": Q(2, 3)})
        with self.assertRaises(InvalidInput):
            family.produce(OP, tuple(SMALL.items()))
        for operation in ("Rational_approximation", "digits", "", None):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(SMALL))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OP, value=1, max_denominator=0)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in either argument and max_work=1 exceed the budget. rational_approximation takes
        no matrix argument: a matrix value is outside its scope (test_invalid_inputs). Work is charged before it
        runs, so a request that needs more than its budget fails at once."""
        for args in (dict(value=support.HUGE, max_denominator=3), dict(value=1, max_denominator=support.HUGE),
                     dict(value=Q(support.HUGE, 3), max_denominator=3),
                     dict(value=Q(1, support.HUGE), max_denominator=3),
                     dict(value=sqrt(support.HUGE), max_denominator=3),
                     dict(value=Add((Q(support.HUGE), sqrt(2))), max_denominator=3),
                     dict(value=support.forged(mf.RealAlgebraicRoot, integer_coefficients=(-support.HUGE, 0, 1),
                                               real_index=0), max_denominator=3)):
            with self.subTest(value=type(args["value"]).__name__, n=type(args["max_denominator"]).__name__):
                self.assertResourceLimit(support.run(OP, **args))
        for value in (Q(1, 3), 7, sqrt(2), root((-2, 0, 0, 1), 0), Mul((sqrt(2), sqrt(2)))):
            with self.subTest(value=str(value), max_work=1):
                self.assertResourceLimit(support.run(OP, value=value, max_denominator=BIG,
                                                     limits=ComputationLimits(max_work=1)))
        deep = sqrt(2)
        for _ in range(300):
            deep = Add((deep, Q(1)))
        start = time.perf_counter()
        refused = (
            dict(value=deep, max_denominator=3),  # Deeper than max_depth.
            dict(value=Pow(sqrt(2), 10 ** 6), max_denominator=3),  # 2**500000: an enclosure end past max_integer_bits.
            # Eleven coprime radicands with the integer value 160: no enclosure separates it from 160, and the
            # algebra that would prove it equal stops at MAX_RADICALS, as the checker's does.
            dict(value=Add(tuple(Mul((sqrt(p), sqrt(p))) for p in PRIMES)), max_denominator=3),
            # The neighbours of 2**65530 + sqrt(2) of order 2**64 have numerators of some 65,594 bits, past
            # max_integer_bits, which the checker refuses in a claim.
            dict(value=Add((Q(2 ** 65_530), sqrt(2))), max_denominator=BIG),
        )
        for number, args in enumerate(refused):  # str() of a 65,530-bit integer is too long.
            with self.subTest(case=number):
                self.assertResourceLimit(support.run(OP, **args))
        for value in (root((-1, -1, 0, 0, 0, 1), 0), sqrt(2), Pow(Add((Q(1), sqrt(2))), 3), Q(10 ** 40 + 1, 10 ** 30)):
            with self.subTest(value=str(value), max_work=100):
                self.assertResourceLimit(support.run(OP, value=value, max_denominator=BIG,
                                                     limits=ComputationLimits(max_work=100)))
        self.assertLess(time.perf_counter() - start, FAST)
        # Eleven coprime radicands whose comparisons the first enclosure settles: produced, and verified.
        result = support.run(OP, value=Add(tuple(sqrt(p) for p in PRIMES)), max_denominator=10)
        self.assertTrue(mf.verify(result).verified, result.error)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim built by hand from the references. The card's certificate leaves no
        choice (the answer and the Farey pair of order N are unique, and () is the only evidence of a value with a
        denominator <= N), so these are the producer's claims too: they are built here without it, over N from 1 to
        2**64, and produce() agrees with each."""
        cases = []
        for n in (1, 2, 3, 10, 113, 10 ** 6, BIG):
            cases.append((Fraction(314159265, 100000000), n))
            cases.append((Fraction(-22, 7), n))
            cases.append((Fraction(10 ** 30 + 7, 10 ** 25 + 3), n))
        for x, n in cases:
            expected = best_rational(x, n)
            evidence: tuple = () if x.denominator <= n else tuple(rational(item) for item in farey_pair(x, n))
            with self.subTest(x=x, n=n):
                self.assertAccepts(OP, {"value": rational(x), "max_denominator": n}, rational(expected), evidence)
                self.assertEqual(family.produce(OP, {"value": rational(x), "max_denominator": n}),
                                 (rational(expected), evidence))
        # Hand-made pairs, checked by their three facts alone: sqrt(2) at N = 10 and N = 12, 1/3 at N = 2.
        self.assertAccepts(OP, {"value": sqrt(2), "max_denominator": 10}, Q(7, 5), (Q(7, 5), Q(10, 7)))
        self.assertAccepts(OP, {"value": sqrt(2), "max_denominator": 12}, Q(17, 12), (Q(7, 5), Q(17, 12)))
        self.assertAccepts(OP, {"value": Q(1, 3), "max_denominator": 2}, Q(1, 2), (Q(0), Q(1, 2)))
        self.assertAccepts(OP, {"value": 4, "max_denominator": 1}, Q(4), ())

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        requests = ((sqrt(2), 10), (Q(314159265, 100000000), 1000), (root((-2, 0, 0, 1), 0), 100), (Q(1, 2), 1),
                    (Q(3, 7), 7), (Mul((sqrt(2), sqrt(2), Q(1, 3))), 3), (Add((sqrt(2), sqrt(3))), BIG))
        for value, n in requests:
            args = {"value": value, "max_denominator": n}
            true, evidence = family.produce(OP, dict(args))
            corrupted = [
                ("value + 1", true + 1, evidence),
                ("an int", true.numerator if true.denominator == 1 else true.numerator // true.denominator, evidence),
                ("not in lowest terms", support.forged(Q, numerator=2 * true.numerator,
                                                       denominator=2 * true.denominator), evidence),
                ("negative denominator", support.forged(Q, numerator=-true.numerator, denominator=-true.denominator),
                 evidence),
                ("missing field", support.forged(Q, numerator=true.numerator), evidence),
                ("bool numerator", support.forged(Q, numerator=True, denominator=1), evidence),
                ("float numerator", support.forged(Q, numerator=float(true.numerator), denominator=true.denominator),
                 evidence),
                ("Fraction", fraction(true), evidence),
                ("str", str(true), evidence),
                ("list evidence", true, list(evidence)),
                ("extra evidence", true, evidence + (true,)),
            ]
            if evidence:
                left, right = evidence
                other = right if true == left else left
                corrupted += [
                    ("the other neighbour", other, evidence),
                    ("missing evidence", true, ()),
                    ("swapped evidence", true, (right, left)),
                    ("one neighbour", true, (left,)),
                    ("int neighbours", true, tuple(item.numerator // item.denominator for item in evidence)),
                    ("str neighbours", true, tuple(str(item) for item in evidence)),
                    ("wider pair", true, (left - 1, right)),
                ]
            else:
                corrupted.append(("invented pair", true, (true - 1, true)))
            for label, value_, evidence_ in corrupted:
                with self.subTest(value=str(value)[:30], n=n, corruption=label):
                    self.assertRejects(OP, args, value_, evidence_)
            with self.subTest(value=str(value)[:30], corruption="operation"):
                self.assertRejects("Rational_approximation", args, true, evidence)
                self.assertRejects(OP, dict(args, extra=1), true, evidence)
                self.assertRejects(OP, {"value": value}, true, evidence)
                for item in (0, True, 0.5, str(n), None, BIG + 1):
                    self.assertRejects(OP, dict(args, max_denominator=item), true, evidence)
        # The answer for N is not the answer for a smaller N.
        self.assertRejects(OP, {"value": sqrt(2), "max_denominator": 4}, Q(7, 5), (Q(7, 5), Q(10, 7)))
        # Oversized forged evidence and values are refused at once.
        args = {"value": sqrt(2), "max_denominator": 10}
        start = time.perf_counter()
        oversized = (("huge value", Q(2 ** 70_000 + 1, 7), (Q(7, 5), Q(10, 7))),
                     ("huge neighbour", Q(7, 5), (Q(7, 5), Q(2 ** 70_000 + 1, 7))),
                     ("long evidence", Q(7, 5), (Q(7, 5),) * 1_000_000),
                     ("huge denominator", Q(7, 5), (Q(7, 5), Q(1, 2 ** 70_000))))
        for label, value_, evidence_ in oversized:
            with self.subTest(oversized=label):
                self.assertRejects(OP, args, value_, evidence_, may_exhaust=True)
        self.assertLess(time.perf_counter() - start, FAST)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text: a
        rational, a real algebraic root, an irrational expression and an expression with a rational value."""
        requests = [dict(value=Q(314159265, 100000000), max_denominator=1000), dict(value=0, max_denominator=4),
                    dict(value=root((-2, 0, 0, 1), 0), max_denominator=BIG), dict(value=sqrt(2), max_denominator=BIG),
                    dict(value=Mul((sqrt(2), sqrt(2), Q(1, 3))), max_denominator=3)]
        results = [support.run(OP, **args) for args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for args, result, text in zip(requests, results, texts):
                with self.subTest(value=str(args["value"])):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(OP, value=Q(1, 3), max_denominator=2)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a fraction, a
        negative fraction, an integer and zero."""
        cases = ((sqrt(2), 10, "7/5", r"\frac{7}{5}"), (Mul((Q(-1), sqrt(2))), 10, "-7/5", r"-\frac{7}{5}"),
                 (Q(1, 2), 1, "0", "0"), (Q(-3, 4), 2, "-1", "-1"), (Q(314159265, 100000000), 1000, "355/113",
                                                                      r"\frac{355}{113}"))
        for value, n, text, latex in cases:
            with self.subTest(value=str(value), n=n):
                result = support.run(OP, value=value, max_denominator=n)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertIs(type(decoded.value), Q)
                self.assertTrue(all(type(item) is Q for item in decoded.steps[0].outputs[1]))
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run(OP, value=sqrt(2), max_denominator=10).value)["data"],
                         mf.to_data(Q(7, 5))["data"])

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261113)
        cases = [dict(value=rng.randrange(-10 ** 9, 10 ** 9), max_denominator=rng.randrange(1, 100)) for _ in range(4)]
        cases += [dict(value=Q(rng.randrange(-10 ** 9, 10 ** 9), rng.randrange(1, 10 ** 6)),
                       max_denominator=rng.choice((1, 7, 100, 10 ** 4, BIG))) for _ in range(6)]
        cases += [dict(value=sqrt(Fraction(rng.randrange(2, 500), rng.randrange(1, 30))),
                       max_denominator=rng.choice((1, 10, 10 ** 6, BIG))) for _ in range(5)]
        cases += [dict(value=Add((sqrt(rng.choice((2, 3, 5))), sqrt(rng.choice((7, 11, 13))))),
                       max_denominator=rng.choice((3, 1000, BIG))) for _ in range(3)]
        cases += [dict(value=root(binomial(rng.randrange(1, 5), rng.randrange(-5, 6), 3, rng.randrange(2, 50)), 0),
                       max_denominator=rng.choice((2, 100, BIG))) for _ in range(4)]
        for args in cases:
            with self.subTest(value=str(args["value"])[:40], n=args["max_denominator"]):
                expected = support.run(OP, **args).value
                self.assertIs(type(expected), Q)
                found = support.lean_value(OP, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(OP, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: friendly expression
        strings, a rational string, and a real algebraic root through its kind's encoding."""
        for raw, value, n in (("sqrt(2)", sqrt(2), 10), ("314159265/100000000", Q(314159265, 100000000), 1000),
                              ("sqrt(2) + sqrt(3)", Add((sqrt(2), sqrt(3))), BIG)):
            with self.subTest(raw=raw):
                result = self.answer(support.call(OP, {"value": raw, "max_denominator": n}))
                self.assertEqual(result, support.run(OP, value=value, max_denominator=n))
                self.assertTrue(mf.verify(result).verified)
        for value, n in ((root((-2, 0, 0, 1), 0), BIG), (Mul((sqrt(2), sqrt(2), Q(1, 3))), 3)):
            with self.subTest(value=str(value)):
                result = self.answer(support.call_values(OP, value=value, max_denominator=n))
                self.assertEqual(result, support.run(OP, value=value, max_denominator=n))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call(OP, {"value": "sqrt(2)", "max_denominator": 0}), "invalid_input")
        self.assertIn("from 1 to 2**64", error["message"])

    def test_seeded_agreement_with_references(self):
        """At least 200 seeded inputs of each kind of value, where produce() succeeds and the checker accepts, agree
        with the references of this module: rationals and ints, real roots of binomials, irrational expressions
        with square roots, and rational-valued expressions."""
        rng = random.Random(20261114)
        counts = dict.fromkeys(("rational", "root", "expression", "exact expression"), 0)

        def denominator_limit() -> int:
            return rng.choice((1, 2, 3, rng.randrange(1, 50), rng.randrange(1, 10 ** 4), 10 ** rng.randrange(1, 19),
                               rng.randrange(1, BIG + 1), BIG))

        for _ in range(200):
            numerator = rng.randrange(-10 ** rng.randrange(1, 40), 10 ** rng.randrange(1, 40))
            x = Fraction(numerator, rng.choice((1, 2, 3, 7, 10, rng.randrange(1, 10 ** rng.randrange(1, 40)))))
            value = x.numerator if x.denominator == 1 and rng.randrange(2) else rational(x)
            n = denominator_limit()
            with self.subTest(kind="rational", x=x, n=n):
                self.assertRational(x, n, value)
                counts["rational"] += 1
        while counts["root"] < 200:
            q, s, m = rng.randrange(1, 12), rng.randrange(-20, 21), rng.choice((2, 2, 3, 3, 4, 5))
            radicand = rng.randrange(2, 10 ** rng.randrange(1, 8))
            if iroot(radicand, m) ** m == radicand:
                continue
            signs = (-1, 1) if m % 2 == 0 else (1,)
            index = rng.randrange(len(signs))
            n = denominator_limit()
            value = root(binomial(q, s, m, radicand), index)
            with self.subTest(kind="root", q=q, s=s, m=m, n_=radicand, index=index, n=n):
                self.assertIrrational(value, n, lambda k, q=q, s=s, m=m, r=radicand, t=signs[index]:
                                      binomial_bracket(q, s, m, r, t, k))
                counts["root"] += 1
        while counts["expression"] < 200:
            a = Fraction(rng.randrange(1, 400), rng.randrange(1, 40))
            if is_square(a):
                continue
            c = Fraction(rng.randrange(-100, 101), rng.randrange(1, 20))
            d = Fraction(rng.choice((-1, 1)) * rng.randrange(1, 30), rng.randrange(1, 12))
            n = denominator_limit()
            kind = rng.randrange(3)
            if kind == 0:
                value, bracket = sqrt(a), (lambda k, a=a: sqrt_bracket(a, k))
            elif kind == 1:
                value = Add((rational(c), Mul((rational(d), sqrt(a)))))
                bracket = lambda k, c=c, d=d, a=a: linear_bracket(c, d, a, k)  # noqa: E731
            else:
                b = Fraction(rng.randrange(1, 400), rng.randrange(1, 40))
                if is_square(b) or is_square(a * b):
                    continue
                value = Add((sqrt(a), sqrt(b)))

                def bracket(k: int, a: Fraction = a, b: Fraction = b) -> tuple[Fraction, Fraction]:
                    (lo_a, hi_a), (lo_b, hi_b) = sqrt_bracket(a, k + 1), sqrt_bracket(b, k + 1)
                    return lo_a + lo_b, hi_a + hi_b
            with self.subTest(kind="expression", value=str(value), n=n):
                self.assertIrrational(value, n, bracket)
                counts["expression"] += 1
        for _ in range(200):
            b, c = rng.choice((2, 3, 5, 6, 7, 10, 11)), Fraction(rng.randrange(-500, 500), rng.randrange(1, 60))
            a = rng.randrange(1, 20)
            kind = rng.randrange(3)
            if kind == 0:  # sqrt(b) * sqrt(b) * c = b*c.
                value, x = Mul((sqrt(b), sqrt(b), rational(c))), b * c
            elif kind == 1:  # sqrt(a*a*b) - a*sqrt(b) + c = c.
                value, x = Add((sqrt(a * a * b), Mul((Q(-a), sqrt(b))), rational(c))), c
            else:  # (1 + sqrt(b)) * (1 - sqrt(b)) * c = (1 - b) * c.
                value, x = Mul((Add((Q(1), sqrt(b))), Add((Q(1), Mul((Q(-1), sqrt(b))))), rational(c))), (1 - b) * c
            n = denominator_limit()
            with self.subTest(kind="exact expression", value=str(value), n=n):
                self.assertRational(x, n, value)
                counts["exact expression"] += 1
        self.assertTrue(all(count >= 200 for count in counts.values()), counts)

    def test_references_agree_with_brute_force(self):
        """The rational reference (limit_denominator with the tie rule) equals brute force on seeded rationals, ties
        included, and produce() equals both; the bracket reference equals brute force at both ends of a bracket."""
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randrange(1, 40)
            x = Fraction(rng.randrange(-300, 300), rng.randrange(1, 90))
            if rng.randrange(4) == 0:  # A midpoint of two fractions with denominators <= n: a tie, or close to one.
                s, t = Fraction(rng.randrange(-50, 50), rng.randrange(1, n + 1)), Fraction(rng.randrange(-50, 50),
                                                                                           rng.randrange(1, n + 1))
                x = (s + t) / 2
            with self.subTest(x=x, n=n):
                expected = brute_force(x, n)
                self.assertEqual(best_rational(x, n), expected)
                self.assertEqual(fraction(self.produced(rational(x), n)[0]), expected)
        for radicand in (2, 3, 5, Fraction(7, 3), 1000):
            for n in (1, 2, 5, 17, 60):
                with self.subTest(radicand=radicand, n=n):
                    expected, (lo, hi) = irrational_reference(lambda k, a=Fraction(radicand): sqrt_bracket(a, k), n)
                    self.assertEqual(brute_force(lo, n), expected)
                    self.assertEqual(brute_force(hi, n), expected)
                    self.assertEqual(fraction(self.produced(sqrt(radicand), n)[0]), expected)

    def test_full_size_inputs_fit_the_default_budget(self):
        """N = 2**64 on each kind of value, and values far from 0: each a verified Result within FAST seconds under
        the default limits, equal to the references."""
        cases = (
            (sqrt(2), lambda k: sqrt_bracket(Fraction(2), k)),
            (root((-2, 0, 0, 1), 0), lambda k: binomial_bracket(1, 0, 3, 2, 1, k)),
            (root(binomial(3, -5, 4, 7), 1), lambda k: binomial_bracket(3, -5, 4, 7, 1, k)),
            (Add((Q(2 ** 60_000), sqrt(2))),
             lambda k: linear_bracket(Fraction(2 ** 60_000), Fraction(1), Fraction(2), k)),
            (sqrt(Fraction(2, 10 ** 30)), lambda k: sqrt_bracket(Fraction(2, 10 ** 30), k)),
            (root((-1, -1, 0, 0, 0, 1), 0), None),
            (Pow(Add((Q(1), sqrt(2))), 30), None),
            # Review regression: a root of 32,500 bits, which the walk once compared by a Horner evaluation of S per
            # step (3.4 million units, past the default budget, for a claim the checker verifies in 155,000); the
            # bracket of width 2**-130 from digits' refinement now decides nearly every step by its ends.
            (root((-(2 ** 65_000 + 1), 0, 1), 1), lambda k: sqrt_bracket(Fraction(2 ** 65_000 + 1), k)),
        )
        for number, (value, bracket) in enumerate(cases):  # str() of 2**60000 is too long.
            with self.subTest(case=number):
                start = time.perf_counter()
                result = support.run(OP, value=value, max_denominator=BIG)
                self.assertTrue(mf.verify(result).verified, result.error)
                self.assertLess(time.perf_counter() - start, FAST)
                if bracket is not None:
                    self.assertEqual(fraction(result.value), irrational_reference(bracket, BIG)[0])
        for number, x in enumerate((Fraction(2 ** 60_000 + 1, 3 ** 50), Fraction(-(10 ** 1000) - 1, 10 ** 999 + 7),
                                    Fraction(1, BIG + 1))):
            with self.subTest(rational=number):
                start = time.perf_counter()
                result = support.run(OP, value=rational(x), max_denominator=BIG)
                self.assertVerified(result, rational(best_rational(x, BIG)))
                self.assertLess(time.perf_counter() - start, FAST)

    def test_claims_verify_within_the_producer_budget(self):
        """A claim that produce() makes within a budget verifies within the same budget: the producer's charges,
        the replayed enclosures and the Sturm allowance among them, bound the checker's."""
        dense = Add(tuple(sqrt(p) for p in PRIMES[:10]))
        cases = (
            (root((-3,) + (0,) * 9 + (1,), 0), (1, 2, 10, BIG)),
            (root((-1, 4, -7, 2, 9, -3, 5, 0, -8, 6, 1, 0, 1), 0), (1, 3, 1000, BIG)),
            (root((2, -4, -1, 2), 1), (1, 2, 3)), (Q(1, 2), (1,)), (Q(3, 7), (7,)), (0, (1,)),
            (Q(10 ** 30 + 1, 10 ** 30), (5, BIG)),
            (Add(tuple(Mul((sqrt(p), sqrt(p))) for _ in range(50) for p in PRIMES[:10])), (1,)),
            (Add((Pow(dense, 4), Mul((Q(-1), Pow(Add(tuple(sqrt(p) for p in PRIMES[:10])), 4))), Q(1, 8))), (3, 9)),
            (Pow(Add((Q(1), sqrt(2))), 30), (BIG,)), (dense, (1, BIG)),
        )
        wide = ComputationLimits(max_work=10 ** 9)
        for value, limits in cases:
            for n in limits:
                args = {"value": value, "max_denominator": n}
                with self.subTest(value=str(value)[:40], n=n):
                    with computation(wide, fresh=True) as budget:
                        found, evidence = family.produce(OP, dict(args))
                    produced = budget.work
                    with computation(wide, fresh=True) as budget:
                        self.assertIs(checker.check_operation(OP, args, found, evidence), True)
                    self.assertLessEqual(budget.work, produced)
                    try:  # Under the default limits: produced means verified.
                        with computation(fresh=True):
                            found, evidence = family.produce(OP, dict(args))
                    except ResourceLimitError:
                        continue
                    self.assertIs(support.verdict(family, OP, args, found, evidence), True)
