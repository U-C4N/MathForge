"""The digits family of mathforge.numeric: digits.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register numeric digits

Every expected value comes from the card or from a reference of this module, never from the producer: Fraction
division for rationals (the card's definition), integer k-th roots for the real roots of (q*x - s)**m - n (math.isqrt
and an integer Newton iteration of its own, no bisection on the polynomial), exact Fractions for expressions whose
rational value is known by construction, and Decimal arithmetic at two precisions, with a margin, for the other
expressions. tests/v09/numeric/test_digits_checker.py tests the checker on its own.

The card has no deferred (Unevaluated) case for digits: every value in scope gets its digits or exceeds the budget.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
import functools
from math import comb, isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.numeric import digits as family
from mathforge.areas.numeric.digits import TruncatedDecimal
from mathforge.certify.numeric import digits as checker
from mathforge.contracts import ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Sqrt

from .. import support

Q = mf.Rational
OP = "digits"
FAST = 3.0  # Seconds: a request that its budget refuses fails, and a full-size request completes, well within this.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact number: a float, a bool, a str, None and two containers.
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
SMALL = dict(value=Q(2, 3), places=5)  # One small request, for the argument tests.


class Integer(int):
    """An int subclass: no exact int for the family, which tests types with `type(x) is int`."""


# ---------------------------------------------------------------------------------------------------------------
# Values and references

def sqrt(value: object) -> Sqrt:
    """A Sqrt node of a rational, never simplified (mf.sqrt would return a Rational for a square)."""
    value = Fraction(value)
    return Sqrt(Q(value.numerator, value.denominator))


def rational(value: object) -> Q:
    value = Fraction(value)
    return Q(value.numerator, value.denominator)


def record(sign: int, magnitude: int, places: int, is_exact: bool) -> TruncatedDecimal:
    return TruncatedDecimal(sign, magnitude, places, is_exact)


def truncated(value: Fraction, places: int) -> TruncatedDecimal:
    """The record of a rational by the card's definition: the floor of |x| * 10**places, exact when it is an
    integer, with Fractions."""
    scaled = abs(Fraction(value)) * 10 ** places
    return record((value > 0) - (value < 0), scaled.numerator // scaled.denominator, places, scaled.denominator == 1)


def iroot(n: int, m: int) -> int:
    """floor(n ** (1/m)) for n >= 0 and m >= 1: math.isqrt for squares, otherwise the integer Newton iteration from
    above (it decreases strictly until it reaches the floor)."""
    if m == 1 or n < 2:
        return n
    if m == 2:
        return isqrt(n)
    x = 1 << -(-n.bit_length() // m)  # Above the root.
    while True:
        y = ((m - 1) * x + n // x ** (m - 1)) // m
        if y >= x:
            return x
        x = y


def binomial(q: int, s: int, m: int, n: int) -> tuple[int, ...]:
    """The coefficients, lowest degree first, of (q*x - s)**m - n."""
    coefficients = [comb(m, j) * q ** j * (-s) ** (m - j) for j in range(m + 1)]
    coefficients[0] -= n
    return tuple(coefficients)


def binomial_signs(m: int, n: int) -> tuple[int, ...]:
    """The real roots of (q*x - s)**m - n (q > 0, n != 0) in increasing order as the signs t of (s + t*y)/q, with
    y = |n| ** (1/m): both signs for an even m and n > 0, -1 for an odd m and n < 0, 1 for an odd m and n > 0."""
    if m % 2 == 0:
        return (-1, 1) if n > 0 else ()
    return (1,) if n > 0 else (-1,)


def binomial_digits(q: int, s: int, m: int, n: int, t: int, places: int) -> TruncatedDecimal:
    """The record of x = (s + t*y)/q, y = |n| ** (1/m), q >= 1. With T = 10**places, W = x*q*T = s*T + t*y*T, and
    Z = floor(y*T) is an integer m-th root. Z exact: W is an integer and x the rational W/(q*T). Otherwise W lies in
    the open unit interval (A, A + 1) for an integer A, so W > 0 exactly when A >= 0, and floor(|W|/q) is A // q, or
    (-A - 1) // q for W < 0 (floor((A + f)/q) = A // q for 0 < f < 1)."""
    scale = 10 ** places
    radicand = abs(n) * scale ** m
    z = iroot(radicand, m)
    if z ** m == radicand:
        return truncated(Fraction(s * scale + t * z, q * scale), places)
    low = s * scale + z if t > 0 else s * scale - z - 1
    if low >= 0:
        return record(1, low // q, places, False)
    return record(-1, (-low - 1) // q, places, False)


def root(coefficients: tuple[int, ...], index: int) -> mf.RealAlgebraicRoot:
    return mf.RealAlgebraicRoot(coefficients, index)


def cube_root_two() -> mf.RealAlgebraicRoot:
    return root((-2, 0, 0, 1), 0)


def decimal_value(node: object) -> Decimal:
    """The value of an expression in the current Decimal context (correctly rounded square roots)."""
    if type(node) is int:
        return Decimal(node)
    if type(node) is Q:
        return Decimal(node.numerator) / Decimal(node.denominator)
    if type(node) is Sqrt:
        return decimal_value(node.radicand).sqrt()
    if type(node) is Add:
        return sum((decimal_value(term) for term in node.terms), Decimal(0))
    if type(node) is Mul:
        return functools.reduce(lambda a, b: a * b, (decimal_value(factor) for factor in node.factors))
    if type(node) is Pow:
        return decimal_value(node.base) ** node.exponent if node.exponent else Decimal(1)
    raise TypeError(type(node).__name__)


def decimal_digits(node: object, places: int) -> TruncatedDecimal | None:
    """The record of an irrational expression from Decimal arithmetic at two precisions, places + 60 and places + 90
    digits beyond its integer part; None unless both agree and the fractional part of |x| * 10**places stays at least
    10**-25 away from 0 and 1, which these sizes of expression never reach by rounding."""
    found = []
    for extra in (60, 90):
        with localcontext() as context:
            context.prec = places + extra + 40
            scaled = abs(decimal_value(node)).scaleb(places)
            magnitude = scaled.to_integral_value(rounding=ROUND_FLOOR)
            fraction = scaled - magnitude
            if not Decimal("1e-25") < fraction < 1 - Decimal("1e-25"):
                return None
            found.append((1 if decimal_value(node) > 0 else -1, int(magnitude)))
    if found[0] != found[1]:
        return None
    return record(found[0][0], found[0][1], places, False)


def random_expression(rng: random.Random) -> object:
    """A closed expression of Rationals and square roots: a sum of two to four terms or a product of two."""
    def radicand() -> Fraction:
        return Fraction(rng.randrange(1, 300), rng.randrange(1, 40))

    def coefficient() -> Fraction:
        return Fraction(rng.choice((-1, 1)) * rng.randrange(1, 60), rng.randrange(1, 25))

    def term() -> object:
        kind = rng.randrange(5)
        if kind == 0:
            return sqrt(radicand())
        if kind == 1:
            return Mul((rational(coefficient()), sqrt(radicand())))
        if kind == 2:
            return rational(coefficient())
        if kind == 3:
            return Pow(Add((rational(coefficient()), sqrt(radicand()))), rng.randrange(0, 5))
        return Mul((sqrt(radicand()), sqrt(radicand()), rational(coefficient())))

    if rng.randrange(4) == 0:
        return Mul((term(), term()))
    return Add(tuple(term() for _ in range(rng.randrange(2, 5))))


def rational_expression(rng: random.Random) -> tuple[object, Fraction]:
    """An expression with square roots whose value is a rational known by construction, and that value."""
    c = Fraction(rng.randrange(-400, 400), rng.choice((1, 2, 4, 5, 8, 10, 3, 7, 16, 25, 125)))
    a, b = rng.randrange(1, 30), rng.choice((2, 3, 5, 6, 7, 10, 11, 15))
    kind = rng.randrange(4)
    if kind == 0:  # sqrt(b) * sqrt(b) * c = b*c.
        return Mul((sqrt(b), sqrt(b), rational(c))), b * c
    if kind == 1:  # sqrt(a*a*b) - a*sqrt(b) + c = c.
        return Add((sqrt(a * a * b), Mul((rational(-a), sqrt(b))), rational(c))), c
    if kind == 2:  # sqrt(u/v) ** (2j) = (u/v) ** j.
        base, j = Fraction(rng.randrange(1, 20), rng.choice((1, 2, 5, 3))), rng.randrange(1, 6)
        return Pow(sqrt(base), 2 * j), base ** j
    # (1 + sqrt(b)) * (1 - sqrt(b)) * c = (1 - b) * c.
    return Mul((Add((Q(1), sqrt(b))), Add((Q(1), Mul((Q(-1), sqrt(b))))), rational(c))), (1 - b) * c


# The known values of the card, and a few more: (value, places, record, evidence, text, LaTeX).
KNOWN = (
    (Q(2, 3), 5, record(1, 66666, 5, False), (), "0.66666...", r"0.66666\ldots"),
    (Q(1, 4), 3, record(1, 250, 3, True), (), "0.250", "0.250"),
    (Q(-1, 8), 2, record(-1, 12, 2, False), (), "-0.12...", r"-0.12\ldots"),
    (0, 4, record(0, 0, 4, True), (), "0.0000", "0.0000"),
    (5, 0, record(1, 5, 0, True), (), "5", "5"),
    (Q(-7, 2), 0, record(-1, 3, 0, False), (), "-3...", r"-3\ldots"),
    (sqrt(2), 30, record(1, 1414213562373095048801688724209, 30, False), ("bits", 164),
     "1.414213562373095048801688724209...", r"1.414213562373095048801688724209\ldots"),
    (cube_root_two(), 20, record(1, 125992104989487316476, 20, False), ("root",),
     "1.25992104989487316476...", r"1.25992104989487316476\ldots"),
    (Mul((sqrt(2), sqrt(2), Q(1, 2))), 3, record(1, 1000, 3, True), ("rational", Q(1)), "1.000", "1.000"),
    (root((-2, 0, 1), 0), 12, record(-1, 1414213562373, 12, False), ("root",), "-1.414213562373...",
     r"-1.414213562373\ldots"),
    (root((-1, 0, 500000), 0), 8, record(-1, 141421, 8, False), ("root",), "-0.00141421...", r"-0.00141421\ldots"),
    (root((0, -1, 0, 1), 1), 3, record(0, 0, 3, True), ("root",), "0.000", "0.000"),
    (root((2, -4, -1, 2), 1), 5, record(1, 50000, 5, True), ("root",), "0.50000", "0.50000"),
    (sqrt(5), 0, record(1, 2, 0, False), ("bits", 64), "2...", r"2\ldots"),
    (Add((sqrt(8), Mul((Q(-2), sqrt(2))))), 3, record(0, 0, 3, True), ("rational", Q(0)), "0.000", "0.000"),
)


def reference(value: object, places: int) -> TruncatedDecimal:
    """The record of a KNOWN value from the references of this module."""
    if type(value) is int or type(value) is Q:
        return truncated(Fraction(value.numerator, value.denominator) if type(value) is Q else Fraction(value),
                         places)
    known = {
        "RootOf((-2, 0, 0, 1), 0)": lambda: binomial_digits(1, 0, 3, 2, 1, places),
        "RootOf((-2, 0, 1), 0)": lambda: binomial_digits(1, 0, 2, 2, -1, places),
        "RootOf((-1, 0, 500000), 0)": lambda: binomial_digits(1000, 0, 2, 2, -1, places),  # -sqrt(2)/1000.
        "RootOf((0, -1, 0, 1), 1)": lambda: truncated(Fraction(0), places),  # x**3 - x: -1, 0, 1.
        "RootOf((2, -4, -1, 2), 1)": lambda: truncated(Fraction(1, 2), places),  # (2x - 1)(x**2 - 2).
    }
    if type(value) is mf.RealAlgebraicRoot:
        return known[str(value)]()
    if value == sqrt(2):
        return binomial_digits(1, 0, 2, 2, 1, places)
    if value == sqrt(5):
        return binomial_digits(1, 0, 2, 5, 1, places)
    if value == Mul((sqrt(2), sqrt(2), Q(1, 2))):
        return truncated(Fraction(1), places)
    return truncated(Fraction(0), places)  # sqrt(8) - 2*sqrt(2).


class DigitsTests(support.FamilyTestCase):
    FAMILY = family

    def assertAgrees(self, value: object, places: int, expected: TruncatedDecimal) -> tuple:
        """produce() answers expected, and the checker accepts that answer with its evidence, which is returned."""
        found, evidence = family.produce(OP, {"value": value, "places": places})
        self.assertEqual(found, expected, (str(value), places))
        self.assertIs(type(found), TruncatedDecimal)
        self.assertIs(support.verdict(family, OP, {"value": value, "places": places}, found, evidence), True)
        return evidence

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with its four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts, with the evidence layout of
        its kind of value."""
        for value, places, expected, evidence, _, _ in KNOWN:
            with self.subTest(value=str(value), places=places):
                self.assertEqual(reference(value, places), expected)
                result = support.run(OP, value=value, places=places)
                self.assertVerified(result, expected)
                self.assertEqual(result.steps[0].outputs[1], evidence)
                self.roundtrip(result)
        self.assertEqual((family.MAX_PLACES, family.MAX_RADICALS), (checker.MAX_PLACES, checker.MAX_RADICALS))
        # The integral power 2**65520 is on a decimal point (it is exact at places 0), and the enclosure at the last
        # precision, max_integer_bits, is still wider than a cell: the exact decision of that precision gives it.
        value = Pow(sqrt(2), 131040)
        result = support.run(OP, value=value, places=0)
        self.assertVerified(result, record(1, 2 ** 65520, 0, True))
        self.assertEqual(result.steps[0].outputs[1], ("rational", Q(2 ** 65520)))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the domain of the card. produce() tests the types itself too."""
        for name in SMALL:
            for item in STAND_INS:
                with self.subTest(argument=name, item=item):
                    self.assertInvalidInput(support.run(OP, **dict(SMALL, **{name: item})))
        x = mf.symbol("x")
        invalid = (
            dict(value=1, places=-1), dict(value=1, places=10_001), dict(value=sqrt(2), places=-(10 ** 30)),
            dict(value=x, places=3), dict(value=Add((x, sqrt(2))), places=3), dict(value=Pow(x, 2), places=3),
            dict(value=Add((cube_root_two(), sqrt(2))), places=3), dict(value=Mul((Q(2), cube_root_two())), places=3),
            dict(value=mf.Matrix(((1, 2), (3, 4))), places=3), dict(value=support.oversized_matrix(), places=3),
        )
        for args in invalid:
            with self.subTest(value=str(args["value"])[:40], places=args["places"]):
                self.assertInvalidInput(support.run(OP, **args))
        messages = ((dict(value=1, places=10_001), "from 0 to 10,000"),
                    (dict(value=Add((x, sqrt(2))), places=3), "closed number"),
                    (dict(value=Add((cube_root_two(), sqrt(2))), places=3), "never inside an expression"))
        for args, text in messages:
            with self.subTest(message=text):
                self.assertIn(text, support.run(OP, **args).error.message)
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
            support.forged(Pow, base=sqrt(2), exponent=Q(1, 2)),
            support.forged(Add, terms=(sqrt(2),)), support.forged(Add, terms=[sqrt(2), Q(1)]),
            support.forged(Add, terms=(1, sqrt(2))), support.forged(Mul, factors=(sqrt(2), 0.5)),
        )
        for number, item in enumerate(forged_values):
            with self.subTest(forged=type(item).__name__, number=number):
                with self.assertRaises(InvalidInput):
                    family.produce(OP, {"value": item, "places": 3})
                self.assertInvalidInput(support.run(OP, value=item, places=3))
        # A node inside itself, and a forged root without its fields: produce() refuses them (operations_v6.run
        # reads the arguments before produce() runs, so these reach produce() only directly).
        loop = support.forged(Add)
        object.__setattr__(loop, "terms", (sqrt(2), loop))
        for item in (loop, support.forged(mf.RealAlgebraicRoot, real_index=0)):
            with self.subTest(forged=type(item).__name__), self.assertRaises(InvalidInput):
                family.produce(OP, {"value": item, "places": 3})
        for item in (True, False, 0.5, "1", None, Integer(3), Fraction(1, 2), 1j):
            for name in SMALL:
                with self.subTest(argument=name, item=item, path="produce"), self.assertRaises(InvalidInput):
                    family.produce(OP, dict(SMALL, **{name: item}))
        with self.assertRaises(InvalidInput):
            family.produce(OP, dict(SMALL, extra=1))
        with self.assertRaises(InvalidInput):
            family.produce(OP, {"value": Q(2, 3)})
        with self.assertRaises(InvalidInput):
            family.produce(OP, tuple(SMALL.items()))
        for operation in ("Digits", "pi_digits", "", None):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(SMALL))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OP, value=1, places=10_001)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in either argument and max_work=1 exceed the budget. digits takes no matrix: a
        matrix value is outside its scope (test_invalid_inputs). Work is charged before it runs, so a request that
        needs more than its budget fails at once."""
        for args in (dict(value=support.HUGE, places=3), dict(value=1, places=support.HUGE),
                     dict(value=Q(support.HUGE, 3), places=3), dict(value=Q(1, support.HUGE), places=3),
                     dict(value=sqrt(support.HUGE), places=3), dict(value=Add((Q(support.HUGE), sqrt(2))), places=3),
                     dict(value=support.forged(mf.RealAlgebraicRoot, integer_coefficients=(-support.HUGE, 0, 1),
                                               real_index=0), places=3)):
            with self.subTest(value=type(args["value"]).__name__, places=type(args["places"]).__name__):
                self.assertResourceLimit(support.run(OP, **args))
        for value in (Q(1, 3), 7, sqrt(2), cube_root_two(), Mul((sqrt(2), sqrt(2)))):
            with self.subTest(value=str(value), max_work=1):
                self.assertResourceLimit(support.run(OP, value=value, places=5, limits=ComputationLimits(max_work=1)))
        deep = sqrt(2)
        for _ in range(300):
            deep = Add((deep, Q(1)))
        start = time.perf_counter()
        refused = (
            dict(value=deep, places=3),  # Deeper than max_depth.
            dict(value=Pow(sqrt(2), 10 ** 6), places=0),  # 2**500000: an enclosure end past max_integer_bits.
            # Eleven coprime radicands on a decimal point (the value 160 is an integer): no enclosure separates it,
            # and the algebra of the ("rational", r) layout stops at MAX_RADICALS.
            dict(value=Add(tuple(Mul((sqrt(p), sqrt(p))) for p in PRIMES)), places=0),
        )
        for args in refused:
            with self.subTest(value=str(args["value"])[:40]):
                self.assertResourceLimit(support.run(OP, **args))
        for value in (root((-1, -1, 0, 0, 0, 1), 0), sqrt(2), Pow(Add((Q(1), sqrt(2))), 3), Q(1, 7)):
            with self.subTest(value=str(value), max_work=200):
                self.assertResourceLimit(support.run(OP, value=value, places=10_000,
                                                     limits=ComputationLimits(max_work=200)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: a finer
        enclosure than the first that fits, and ("rational", r) for a rational expression strictly inside a cell."""
        for places in (0, 1, 7, 30, 100):
            expected = binomial_digits(1, 0, 2, 2, 1, places)
            for b in (-(-10 * places // 3) + 64, 4 * places + 128, 1000, 65_536):
                with self.subTest(places=places, bits=b):
                    self.assertAccepts(OP, {"value": sqrt(2), "places": places}, expected, ("bits", b))
        for value, places, exact_value in ((Mul((sqrt(2), sqrt(2), Q(1, 3))), 5, Fraction(2, 3)),
                                           (Add((sqrt(12), Mul((Q(-2), sqrt(3))), Q(-1, 7))), 9, Fraction(-1, 7))):
            with self.subTest(value=str(value)):
                expected = truncated(exact_value, places)
                evidence = self.assertAgrees(value, places, expected)
                self.assertEqual(evidence[0], "bits")  # The producer's choice: the value is inside an open cell.
                self.assertAccepts(OP, {"value": value, "places": places}, expected,
                                   ("rational", rational(exact_value)))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        for value, places in ((Q(2, 3), 5), (cube_root_two(), 20), (sqrt(2), 30),
                              (Mul((sqrt(2), sqrt(2), Q(1, 2))), 3)):
            args = {"value": value, "places": places}
            true, evidence = family.produce(OP, dict(args))
            sign, magnitude = true.sign, true.magnitude
            corrupted = [
                ("magnitude + 1", record(sign, magnitude + 1, places, False), evidence),
                ("magnitude - 1", record(sign, magnitude - 1, places, False), evidence),
                ("negated", record(-sign, magnitude, places, true.exact), evidence),
                ("exactness", support.forged(TruncatedDecimal, sign=sign, magnitude=magnitude, places=places,
                                             exact=not true.exact), evidence),
                ("places", record(sign, magnitude, places + 1, true.exact), evidence),
                ("noncanonical zero", support.forged(TruncatedDecimal, sign=0, magnitude=0, places=places,
                                                     exact=False), evidence),
                ("impostor", support.impostor(true), evidence),
                ("missing field", support.forged(TruncatedDecimal, sign=sign, magnitude=magnitude, places=places),
                 evidence),
                ("bool sign", support.forged(TruncatedDecimal, sign=True, magnitude=magnitude, places=places,
                                             exact=true.exact), evidence),
                ("float magnitude", support.forged(TruncatedDecimal, sign=sign, magnitude=float(magnitude),
                                                   places=places, exact=true.exact), evidence),
                ("str magnitude", support.forged(TruncatedDecimal, sign=sign, magnitude=str(magnitude),
                                                 places=places, exact=true.exact), evidence),
                ("int exact", support.forged(TruncatedDecimal, sign=sign, magnitude=magnitude, places=places,
                                             exact=int(true.exact)), evidence),
                ("tuple value", (sign, magnitude, places, true.exact), evidence),
                ("list evidence", true, list(evidence)),
                ("extra evidence", true, evidence + (1,)),
                ("other layout", true, ("root",) if evidence != ("root",) else ()),
            ]
            if evidence:
                corrupted.append(("missing evidence", true, ()))
                corrupted.append(("bits 0", true, ("bits", 0)))
                corrupted.append(("rational 1 as an int", true, ("rational", 1)))
            for label, value_, evidence_ in corrupted:
                with self.subTest(value=str(value), corruption=label):
                    self.assertRejects(OP, args, value_, evidence_)
            with self.subTest(value=str(value), corruption="operation"):
                self.assertRejects("Digits", args, true, evidence)
                self.assertRejects(OP, dict(args, places=places + 1), true, evidence)
                self.assertRejects(OP, dict(args, extra=1), true, evidence)
        # Oversized forged evidence and values are refused at once.
        args = {"value": sqrt(2), "places": 30}
        true, _ = family.produce(OP, dict(args))
        start = time.perf_counter()
        oversized = (("huge precision", true, ("bits", 2 ** 70_000)), ("huge r", true, ("rational", Q(2 ** 70_000))),
                     ("long evidence", true, ("bits",) + (64,) * 1_000_000),
                     ("huge magnitude", support.forged(TruncatedDecimal, sign=1, magnitude=2 ** 70_000, places=30,
                                                       exact=False), ("bits", 164)))
        for label, value_, evidence_ in oversized:
            with self.subTest(oversized=label):
                self.assertRejects(OP, args, value_, evidence_, may_exhaust=True)
        self.assertLess(time.perf_counter() - start, FAST)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text: a
        rational, a real algebraic root, an enclosed expression and an expression with a rational value."""
        requests = [dict(value=Q(2, 3), places=5), dict(value=0, places=4), dict(value=cube_root_two(), places=20),
                    dict(value=sqrt(2), places=30), dict(value=Mul((sqrt(2), sqrt(2), Q(1, 2))), places=3)]
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
            failed = support.run(OP, value=Q(1, 3), places=2)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: truncated and exact,
        positive, negative and zero, with and without places."""
        for value, places, expected, _, text, latex in KNOWN:
            with self.subTest(value=str(value), places=places):
                result = support.run(OP, value=value, places=places)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertIs(type(decoded.value), TruncatedDecimal)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run(OP, value=Q(1, 4), places=3).value)["data"],
                         {"type": "v6_truncated_decimal", "sign": {"type": "integer", "value": "1"},
                          "magnitude": {"type": "integer", "value": "250"},
                          "places": {"type": "integer", "value": "3"}, "exact": {"type": "boolean", "value": True}})
        # Leading zeros of a long expansion.
        result = support.run(OP, value=Q(1, 10 ** 40), places=45)
        self.assertEqual(mf.render_text(result.value), "0." + "0" * 39 + "100000")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261131)
        cases = [dict(value=rng.randrange(-10 ** 9, 10 ** 9), places=rng.randrange(0, 20)) for _ in range(5)]
        cases += [dict(value=Q(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 10 ** 4)),
                       places=rng.randrange(0, 40)) for _ in range(5)]
        cases += [dict(value=random_expression(rng), places=rng.randrange(0, 40)) for _ in range(5)]
        cases += [dict(value=rational_expression(rng)[0], places=rng.randrange(0, 10)) for _ in range(3)]
        for _ in range(4):
            q, s, m, n = rng.randrange(1, 5), rng.randrange(-5, 6), rng.randrange(2, 6), rng.randrange(2, 50)
            cases.append(dict(value=root(binomial(q, s, m, n), 0), places=rng.randrange(0, 40)))
        for args in cases:
            with self.subTest(value=str(args["value"])[:40], places=args["places"]):
                expected = support.run(OP, **args).value
                self.assertIs(type(expected), TruncatedDecimal)
                found = support.lean_value(OP, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(OP, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: a friendly
        expression string, a rational string, and a real algebraic root through its kind's encoding."""
        for raw, value, places in (("sqrt(2)", sqrt(2), 30), ("2/3", Q(2, 3), 5), ("(1+sqrt(2))^3",
                                                                                   Pow(Add((Q(1), sqrt(2))), 3), 12)):
            with self.subTest(raw=raw):
                result = self.answer(support.call(OP, {"value": raw, "places": places}))
                self.assertEqual(result, support.run(OP, value=value, places=places))
                self.assertTrue(mf.verify(result).verified)
        for value, places in ((cube_root_two(), 20), (Mul((sqrt(2), sqrt(2), Q(1, 2))), 3)):
            with self.subTest(value=str(value)):
                result = self.answer(support.call_values(OP, value=value, places=places))
                self.assertEqual(result, support.run(OP, value=value, places=places))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call(OP, {"value": "sqrt(2)", "places": 10_001}), "invalid_input")
        self.assertIn("from 0 to 10,000", error["message"])

    def test_seeded_agreement_with_references(self):
        """At least 200 seeded inputs of each kind of value, where produce() succeeds and the checker accepts, agree
        with the references of this module: rationals, real roots of binomials, rational-valued expressions with
        square roots and other expressions."""
        rng = random.Random(20261113)
        counts = dict.fromkeys(("rational", "root", "exact expression", "expression"), 0)
        for _ in range(200):
            numerator = rng.randrange(-10 ** rng.randrange(1, 30), 10 ** rng.randrange(1, 30))
            denominator = rng.choice((1, 2, 3, 7, 8, 10, 125, rng.randrange(1, 10 ** rng.randrange(1, 20))))
            value = rational(Fraction(numerator, denominator))
            value = value.numerator if value.denominator == 1 and rng.randrange(2) else value
            places = rng.randrange(0, 60)
            with self.subTest(kind="rational", value=str(value), places=places):
                self.assertEqual(self.assertAgrees(value, places, truncated(Fraction(numerator, denominator),
                                                                             places)), ())
                counts["rational"] += 1
        while counts["root"] < 200:
            q, s, m = rng.randrange(1, 12), rng.randrange(-20, 21), rng.randrange(1, 7)
            n = rng.choice((-1, 1)) * rng.randrange(1, 10 ** rng.randrange(1, 8))
            if m > 1 and rng.randrange(6) == 0:  # A perfect power: rational roots.
                n = rng.choice((-1, 1)) * rng.randrange(1, 30) ** m
            signs = binomial_signs(m, n)
            if not signs:
                continue
            index = rng.randrange(len(signs))
            places = rng.randrange(0, 50)
            value = root(binomial(q, s, m, n), index)
            with self.subTest(kind="root", q=q, s=s, m=m, n=n, index=index, places=places):
                self.assertEqual(self.assertAgrees(value, places, binomial_digits(q, s, m, n, signs[index], places)),
                                 ("root",))
                counts["root"] += 1
        # Roots of (a*x - b)(x**2 - n) with n no square: the rational root b/a among -sqrt(n) and sqrt(n).
        for _ in range(40):
            a, b, n = rng.randrange(1, 9), rng.randrange(-30, 31), rng.choice((2, 3, 5, 6, 7, 10, 11, 13))
            candidates = [("surd", -1), ("surd", 1), ("rational", Fraction(b, a))]

            def compare(left: tuple, right: tuple, n: int = n) -> int:
                """Exact order of +-sqrt(n) and rationals: sign(t*sqrt(n) - r) from r and r*r against n."""
                if left[0] == right[0]:
                    return (left[1] > right[1]) - (left[1] < right[1])
                if left[0] == "rational":
                    return -compare(right, left)
                t, r = left[1], right[1]
                if t > 0:
                    return 1 if r < 0 or r * r < n else -1
                return -1 if r > 0 or r * r < n else 1

            ordered = sorted(candidates, key=functools.cmp_to_key(compare))
            coefficients = (b * n, -a * n, -b, a)  # (a*x - b)(x**2 - n), lowest degree first.
            places = rng.randrange(0, 40)
            for index, (kind, item) in enumerate(ordered):
                expected = (truncated(item, places) if kind == "rational"
                            else binomial_digits(1, 0, 2, n, item, places))
                with self.subTest(kind="root", coefficients=coefficients, index=index, places=places):
                    self.assertAgrees(root(coefficients, index), places, expected)
                    counts["root"] += 1
        for _ in range(200):
            value, exact_value = rational_expression(rng)
            places = rng.randrange(0, 40)
            with self.subTest(kind="exact expression", value=str(value), places=places):
                evidence = self.assertAgrees(value, places, truncated(exact_value, places))
                expected_layout = "rational" if truncated(exact_value, places).exact else "bits"
                self.assertEqual(evidence[0], expected_layout)
                counts["exact expression"] += 1
        attempts = 0
        while counts["expression"] < 200:
            attempts += 1
            value, places = random_expression(rng), rng.randrange(0, 60)
            expected = decimal_digits(value, places)
            if expected is None:  # A rational value, or one too close to a decimal point for the reference.
                continue
            with self.subTest(kind="expression", value=str(value), places=places):
                evidence = self.assertAgrees(value, places, expected)
                self.assertEqual(evidence[0], "bits")
                counts["expression"] += 1
        self.assertLess(attempts, 400)
        self.assertTrue(all(count >= 200 for count in counts.values()), counts)

    def test_full_size_inputs_fit_the_default_budget(self):
        """10,000 places, the most that the card allows, of a rational, a square root, a cube root, a root of a
        quintic, a power of a sum and a rational expression on a decimal point: each a verified Result within
        FAST seconds under the default limits, its digits from the references where they reach."""
        cases = (
            (Q(1, 7), truncated(Fraction(1, 7), 10_000)),
            (sqrt(2), binomial_digits(1, 0, 2, 2, 1, 10_000)),
            (cube_root_two(), binomial_digits(1, 0, 3, 2, 1, 10_000)),
            (root(binomial(3, -5, 4, 7), 1), binomial_digits(3, -5, 4, 7, 1, 10_000)),
            (root((-1, -1, 0, 0, 0, 1), 0), None),
            (Pow(Add((Q(1), sqrt(2))), 3), binomial_digits(1, 7, 2, 50, 1, 10_000)),  # 7 + 5*sqrt(2).
            (Mul((sqrt(2), sqrt(2), Q(1, 2))), record(1, 10 ** 10_000, 10_000, True)),
        )
        for value, expected in cases:
            with self.subTest(value=str(value)):
                start = time.perf_counter()
                result = support.run(OP, value=value, places=10_000)
                self.assertTrue(mf.verify(result).verified, result.error)
                self.assertLess(time.perf_counter() - start, FAST)
                if expected is not None:
                    self.assertEqual(result.value, expected)
                else:  # x**5 - x - 1 changes sign across the claimed cell, its only real root inside.
                    low = Fraction(result.value.magnitude, 10 ** 10_000)
                    high = low + Fraction(1, 10 ** 10_000)
                    self.assertTrue(low ** 5 - low - 1 < 0 < high ** 5 - high - 1)

    def test_roots_far_from_their_root_bound(self):
        """Regression (review of wave m2b): real roots that the root bound overestimates by thousands of bits get
        their digits within the default budget. Before, the isolating interval of -sqrt(2**20000 + 1) was
        (-2**20000, 0), from which Newton's method and bisection gained one bit per step (some 10**8 units of work),
        and kernels.upoly.isolate separated the five roots of the polynomial below, about 2**1000 in size under a
        root bound of about 2**4000, by plain bisection from the bound (some 5 * 10**7 units)."""
        n = 2 ** 20_000 + 1
        scale = 2 ** 1_000
        factors = ((-2, 0, 1), (-3, 1), (5, 1), (-1, 1))
        product = (1,)
        for factor in factors:
            product = tuple(sum(product[i] * factor[k - i] for i in range(len(product)) if 0 <= k - i < len(factor))
                            for k in range(len(product) + len(factor) - 1))
        # (x^2 - 2)(x - 3)(x + 5)(x - 1) at x / 2**1000: the roots -5, -sqrt(2), 1, sqrt(2), 3 times 2**1000.
        clustered = tuple(c * scale ** (len(product) - 1 - k) for k, c in enumerate(product))
        cases = (
            (root((-n, 0, 1), 0), 10, record(-1, isqrt(n * 10 ** 20), 10, False)),
            (root((-n, 0, 1), 1), 3_000, record(1, isqrt(n * 10 ** 6_000), 3_000, False)),
            (root((-(10 ** 1_200), 0, 1), 1), 5, record(1, 10 ** 605, 5, True)),
            (root((-2, 0, 10 ** 1_200), 1), 700, record(1, isqrt(2 * 10 ** 1_400 // 10 ** 1_200), 700, False)),
            (root(clustered, 3), 200, record(1, isqrt(2 * scale ** 2 * 10 ** 400), 200, False)),
            (root(clustered, 0), 100, record(-1, 5 * scale * 10 ** 100, 100, True)),
            (root(clustered, 1), 0, record(-1, isqrt(2 * scale ** 2), 0, False)),
        )
        for number, (value, places, expected) in enumerate(cases):
            with self.subTest(case=number, places=places):  # str() of the 20,000-bit coefficients is too long.
                start = time.perf_counter()
                result = support.run(OP, value=value, places=places)
                self.assertVerified(result, expected)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_claims_verify_within_the_producer_budget(self):
        """Regression (review of wave m2b): a claim that produce() makes within a budget verifies within the same
        budget, as the producer's charges bound the checker's. Before, a rational expression of 500 square roots over
        ten primes, a dense power in the algebra of ten square roots and the roots of polynomials of degree 10 and
        more at 10,000 places came out of produce() within the default budget but ran the checker out of it."""
        dense = Add(tuple(sqrt(p) for p in PRIMES[:10]))
        cases = (
            (Add(tuple(Mul((sqrt(p), sqrt(p))) for _ in range(50) for p in PRIMES[:10])), 0),
            (Add(tuple(term for _ in range(25) for p in PRIMES[:10] for term in (sqrt(p), Mul((Q(-1), sqrt(p)))))), 3),
            (Add((Pow(dense, 4), Mul((Q(-1), Pow(Add(tuple(sqrt(p) for p in PRIMES[:10])), 4))), Q(1, 8))), 3),
            (Add((Pow(dense, 8), Mul((Q(-1), Pow(Add(tuple(sqrt(p) for p in PRIMES[:10])), 8))), Q(1, 8))), 3),
            (root((-3,) + (0,) * 9 + (1,), 0), 10_000),
            (root((-1, 4, -7, 2, 9, -3, 5, 0, -8, 6, 1, 0, 1), 0), 10_000),
            (sqrt(2), 10_000), (Q(1, 7), 10_000), (Pow(Add((Q(1), sqrt(2))), 30), 1_000),
            (Mul((sqrt(Fraction(37, 5)), sqrt(Fraction(5, 37)), Q(-16, 7))), 10_000),
        )
        wide = ComputationLimits(max_work=10 ** 9)
        for value, places in cases:
            args = {"value": value, "places": places}
            with self.subTest(value=str(value)[:40], places=places):
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

    def test_truncated_decimal_canonical_form(self):
        """The record refuses fields outside its canonical form: sign 0 exactly for the number 0."""
        for number, fields in enumerate(((2, 1, 3, False), (True, 1, 3, False), (1, -1, 3, False), (1, 1.0, 3, False),
                       (1, 1, -1, False), (1, 1, 3, 1), (0, 0, 3, False), (0, 1, 3, True), (1, 0, 3, True),
                       (-1, 0, 3, True), (1, support.HUGE, 3, False))):
            with self.subTest(number=number):
                with self.assertRaises(ResourceLimitError if fields[1] is support.HUGE else InvalidInput):
                    record(*fields)
        self.assertEqual(record(1, 0, 3, False).magnitude, 0)  # A positive number below 10**-3.
