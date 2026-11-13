"""The roots family of mathforge.algebra: count_real_roots, complex_roots.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register algebra roots

Every expected value comes from the card or from a reference of this module, never from the producer:
- Built: lead * prod (x - r)**m * prod (x**2 - s) * prod (x**2 + b*x + c), multiplied out of chosen rational roots r,
  radicands s that are no rational squares (the roots -+sqrt(s)) and quadratics without a real root; Built.count
  counts its distinct real roots in an interval by exact comparisons of rationals and squares.
- sturm_count(): Sturm's theorem on the classic remainder sequence of the square-free part over fractions.Fraction,
  for random dense polynomials (the producer runs v0.7's signed primitive pseudo-remainder chain on integers).
- complex_claim(): the complex roots of a product of chosen monic irreducible factors of degree one or two, spelled out
  from the closed forms of the family's docstring and sorted by rational enclosures of their parts that isqrt refines
  until they separate, with the evidence: the leading coefficient and the factors in the order of their key.
tests/v09/algebra/test_roots_checker.py tests the checker on its own.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from math import gcd, isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.algebra import roots as family
from mathforge.complex_numbers import ComplexAlgebraic, ComplexRoot
from mathforge.contracts import CheckStatus, Completeness, ErrorInfo, ExecutionStatus, Outcome
from mathforge.deferred import Reason, Unevaluated
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational, Symbol, sqrt
from mathforge.polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from mathforge.polynomial_exact import factor_with_evidence

from .. import support

Q = Rational
F = Fraction
X = Symbol("x")
COUNT, COMPLEX = "count_real_roots", "complex_roots"
FAST = 2.0  # Seconds: a request beyond its budget fails well within this, one of the larger sizes within twice this.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact number: a float, a bool, a str, None and two containers.
DEFERRAL = Reason("no_applicable_rule", "an irreducible factor of degree three or more")
VERIFIED, FAILED, UNKNOWN = CheckStatus.VERIFIED, CheckStatus.FAILED, CheckStatus.UNKNOWN
CUBIC = (0, -1, 0, 1)  # x^3 - x, with the roots -1, 0 and 1.
# Irreducible polynomials over Q of degree three or more, lowest degree first: x^3 - 2 (Eisenstein), x^3 + x + 1 and
# x^3 - 3x + 1 (cubics without a rational root), x^4 + 1 (the eighth cyclotomic polynomial), x^4 - 10x^2 + 1 (the
# minimal polynomial of sqrt 2 + sqrt 3) and x^5 - x - 1 (an Artin-Schreier polynomial modulo 5).
IRREDUCIBLE = ((-2, 0, 0, 1), (1, 1, 0, 1), (1, -3, 0, 1), (1, 0, 0, 0, 1), (1, 0, -10, 0, 1), (-1, -1, 0, 0, 0, 1))

# The card's counts: (coefficients, lower, upper, count).
COUNT_KNOWN = (
    ((-2, 0, 1), "-inf", "+inf", 2),
    ((-2, 0, 1), 0, 2, 1),
    (CUBIC, -1, 1, 3),
    (CUBIC, "-inf", 0, 2),
    ((1, -2, 1), 0, 2, 1),
    ((1, 0, 1), "-inf", "+inf", 0),
    ((5,), "-inf", "+inf", 0),
)
# Further counts of x^3 - x: ends that are roots, point intervals, Rational ends and one infinite end.
COUNT_SPECIAL = ((-1, -1, 1), (Q(1, 2), Q(1, 2), 0), (0, 0, 1), (-1, Q(-1, 2), 1), (Q(-1, 2), "+inf", 2),
                 ("-inf", -2, 0), (1, "+inf", 1), (2, "+inf", 0), (Q(-3, 2), Q(3, 2), 3), (Q(1, 3), 1, 1),
                 ("-inf", Q(-1, 10 ** 30), 1), (Q(1, 10 ** 30), "+inf", 1))
# The base claim of the corruption tests: -3/2 (x + 1/2)(x^2 - 2)(x^2 + 1)(x^2 + x + 1)^2, of degree 9, whose seven
# roots have every kind of part: -sqrt 2, -1/2 -+ (sqrt 3/2) i (double), -1/2, -i, i, sqrt 2.
BASE_FACTORS = (((F(1, 2), 1), 1), ((-2, 0, 1), 1), ((1, 0, 1), 1), ((1, 1, 1), 2))


# ---------------------------------------------------------------------------------------------------------------
# References: polynomials as lists of Fractions, lowest degree first, without trailing zeros ([] is zero).

def trim(p: Iterable[object]) -> list[Fraction]:
    values = [F(c) for c in p]
    while values and values[-1] == 0:
        values.pop()
    return values


def times(p: Sequence[Fraction], q: Sequence[Fraction]) -> list[Fraction]:
    if not p or not q:
        return []
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def expand(lead: object, factors: Sequence[tuple[Sequence[object], int]]) -> list[Fraction]:
    """lead * prod f**e for factors (coefficients lowest degree first, exponent)."""
    p = [F(lead)]
    for coefficients, exponent in factors:
        for _ in range(exponent):
            p = times(p, [F(c) for c in coefficients])
    return trim(p)


def rational(value: object) -> Q:
    value = F(value)
    return Q(value.numerator, value.denominator)


def polynomial(p: Iterable[object]) -> Polynomial:
    return Polynomial(tuple(rational(c) for c in trim(p)) or (Q(0),), X)


def primitive(p: Sequence[object]) -> tuple[int, ...]:
    """The primitive integer form of a nonzero polynomial, with a positive leading coefficient."""
    values = [F(c) for c in p]
    common = 1
    for c in values:
        common = common * c.denominator // gcd(common, c.denominator)
    integers = [int(c * common) for c in values]
    divisor = gcd(*integers) * (1 if integers[-1] > 0 else -1)
    return tuple(value // divisor for value in integers)


def rational_sqrt(value: Fraction) -> Fraction | None:
    """The rational square root of a Fraction, or None."""
    if value < 0:
        return None
    top, bottom = isqrt(value.numerator), isqrt(value.denominator)
    return F(top, bottom) if top * top == value.numerator and bottom * bottom == value.denominator else None


def evaluate(p: Sequence[Fraction], x: Fraction) -> Fraction:
    value = F(0)
    for c in reversed(p):
        value = value * x + c
    return value


def derivative(p: Sequence[Fraction]) -> list[Fraction]:
    return [i * c for i, c in enumerate(p)][1:]


def divide(p: Sequence[Fraction], q: Sequence[Fraction]) -> tuple[list[Fraction], list[Fraction]]:
    """(quotient, remainder) of p by a nonzero q over the rationals, by long division."""
    p = list(p)
    quotient = [F(0)] * max(len(p) - len(q) + 1, 0)
    for shift in range(len(p) - len(q), -1, -1):
        factor = p[shift + len(q) - 1] / q[-1]
        quotient[shift] = factor
        for i, c in enumerate(q):
            p[shift + i] -= factor * c
    return trim(quotient), trim(p[:len(q) - 1])


def sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def sturm_count(p: Iterable[object], low: Fraction | None, high: Fraction | None) -> int:
    """The distinct real roots of a nonzero polynomial in [low, high], None for an infinite end: Sturm's theorem on the
    sequence s, s', then minus each remainder, of the square-free part s = p / gcd(p, p'), over the fractions. Sign
    variations skip zeros, so V(low) - V(high) counts the roots in (low, high]; low is added when it is a root."""
    p = trim(p)
    if len(p) <= 1:
        return 0
    common, rest = p, derivative(p)
    while rest:
        common, rest = rest, divide(common, rest)[1]
    s = divide(p, common)[0]
    chain = [s, derivative(s)]
    while len(chain[-1]) > 1:
        remainder = divide(chain[-2], chain[-1])[1]
        if not remainder:
            break
        chain.append([-c for c in remainder])

    def variations(x: Fraction | None, positive: bool) -> int:
        if x is None:  # The sign of each member at plus or minus infinity.
            signs = [sign(member[-1]) * (1 if positive or len(member) % 2 else -1) for member in chain]
        else:
            signs = [sign(evaluate(member, x)) for member in chain]
        signs = [item for item in signs if item]
        return sum(a != b for a, b in zip(signs, signs[1:]))

    return variations(low, False) - variations(high, True) + (low is not None and evaluate(s, low) == 0)


def bound(end: object) -> Fraction | None:
    """The Fraction of an end of a request (an int or a Rational), None for an infinity."""
    if type(end) is str:
        return None
    if type(end) is Q:
        return F(end.numerator, end.denominator)
    return F(end)


def request_ends(low: Fraction | None, high: Fraction | None) -> tuple[object, object]:
    """The ends of a request for the interval [low, high], None for an infinite end: an int for an integer, else a
    Rational."""
    def end(value: Fraction | None, infinity: str) -> object:
        if value is None:
            return infinity
        return value.numerator if value.denominator == 1 else Q(value.numerator, value.denominator)
    return end(low, "-inf"), end(high, "+inf")


@dataclass(frozen=True)
class Built:
    """lead * prod (x - r)**m * prod (x**2 - s) * prod (x**2 + b*x + c): distinct rational roots r with multiplicities,
    distinct positive radicands s that are no rational squares (the roots -+sqrt(s)), and quadratics (c, b) with
    b**2 < 4c, which have no real root."""
    lead: Fraction
    rationals: tuple[tuple[Fraction, int], ...] = ()
    radicands: tuple[Fraction, ...] = ()
    definite: tuple[tuple[Fraction, Fraction], ...] = ()

    def coefficients(self) -> list[Fraction]:
        factors = [((-r, 1), m) for r, m in self.rationals]
        factors += [((-s, 0, 1), 1) for s in self.radicands] + [((c, b, 1), 1) for c, b in self.definite]
        return expand(self.lead, factors)

    def count(self, low: Fraction | None, high: Fraction | None) -> int:
        """The distinct real roots in [low, high], each decided by exact comparisons of rationals and squares."""
        total = sum((low is None or low <= r) and (high is None or r <= high) for r, _ in self.rationals)
        for s in self.radicands:
            # sqrt(s) >= low, sqrt(s) <= high, -sqrt(s) >= low and -sqrt(s) <= high, s being no rational square.
            total += (low is None or low <= 0 or low * low < s) and (high is None or (high > 0 and s < high * high))
            total += (low is None or (low < 0 and s < low * low)) and (high is None or high >= 0 or s > high * high)
        return total

    def points(self, rng: random.Random) -> list[Fraction]:
        """Ends at the roots and beside them, near the irrational roots on both sides, and a few others."""
        near = (F(0), F(1, 1000), F(-1, 1000), F(1, 10 ** 12), F(-1, 10 ** 12))
        points = [r + delta for r, _ in self.rationals for delta in near]
        for s in self.radicands:
            approximation = F(isqrt(s.numerator * 10 ** 12 // s.denominator), 10 ** 6)
            points += [approximation, -approximation, approximation + F(1, 10 ** 6), -approximation - F(1, 10 ** 6)]
        return points + [F(rng.randrange(-40, 41), rng.randrange(1, 5)) for _ in range(3)]


def random_built(rng: random.Random) -> Built:
    rationals: dict[Fraction, int] = {}
    for _ in range(rng.randrange(0, 4)):
        rationals[F(rng.randrange(-30, 31), rng.randrange(1, 7))] = rng.randrange(1, 4)
    radicands = set()
    for _ in range(rng.randrange(0, 3)):
        s = F(rng.randrange(1, 60), rng.randrange(1, 5))
        if rational_sqrt(s) is None:
            radicands.add(s)
    definite = tuple((F(rng.randrange(5, 40)), F(rng.randrange(-4, 5))) for _ in range(rng.randrange(0, 2)))
    lead = F(rng.choice((-1, 1)) * rng.randrange(1, 9), rng.randrange(1, 4))
    return Built(lead, tuple(rationals.items()), tuple(sorted(radicands)), definite)


# Records spelled out from the family's docstring: the parts of the roots of each monic factor.

def complex_value(real: object, imag: object) -> ComplexAlgebraic:
    """A ComplexAlgebraic of given parts, both fields set on a new object, as v0.7's producers build theirs."""
    value = object.__new__(ComplexAlgebraic)
    object.__setattr__(value, "real", real)
    object.__setattr__(value, "imag", imag)
    return value


def record(real: object, imag: object, multiplicity: int) -> ComplexRoot:
    return ComplexRoot(complex_value(real, imag), multiplicity)


# A surd (u, v, d) of Fractions is u + v * sqrt(d); v = 0 and d = 0 for a rational.
Surd = tuple[Fraction, Fraction, Fraction]
ZERO: Surd = (F(0), F(0), F(0))


def factor_roots(coefficients: Sequence[object]) -> list[tuple[object, object, Surd, Surd]]:
    """(real part, imaginary part, real value, imaginary value) of each root of a monic linear or irreducible quadratic
    factor: -a for x + a; for x^2 + b x + c with D = b^2 - 4c, the real parts RealAlgebraicRoot(F, 0) and (F, 1), F the
    primitive integer form of the factor, and the imaginary part 0 when D > 0; the real part -b/2 and the imaginary
    parts -+sqrt(-D)/2 when D < 0: Rationals when -D/4 is a rational square, else RealAlgebraicRoot(G, 0) and (G, 1), G
    the primitive integer form of 4y^2 + D."""
    if len(coefficients) == 2:
        a = F(coefficients[0])
        return [(rational(-a), Q(0), (-a, F(0), F(0)), ZERO)]
    c, b = F(coefficients[0]), F(coefficients[1])
    discriminant = b * b - 4 * c
    assert rational_sqrt(discriminant) is None, "a reducible quadratic"
    if discriminant > 0:
        form = primitive(coefficients)
        return [(RealAlgebraicRoot(form, k), Q(0), (-b / 2, F(2 * k - 1, 2), discriminant), ZERO) for k in (0, 1)]
    e = -discriminant / 4
    s = rational_sqrt(e)
    if s is not None:
        return [(rational(-b / 2), rational(t * s), (-b / 2, F(0), F(0)), (t * s, F(0), F(0))) for t in (-1, 1)]
    form = primitive([discriminant, 0, 4])
    return [(rational(-b / 2), RealAlgebraicRoot(form, k), (-b / 2, F(0), F(0)), (F(0), F(2 * k - 1), e))
            for k in (0, 1)]


def enclosure(x: Surd, bits: int) -> tuple[Fraction, Fraction]:
    """A rational interval around u + v sqrt(d): sqrt(d) = sqrt(n m)/m for d = n/m, within 2**-bits / m."""
    u, v, d = x
    if v == 0:
        return u, u
    root = isqrt(d.numerator * d.denominator << 2 * bits)
    low, high = F(root, d.denominator << bits), F(root + 1, d.denominator << bits)
    return (u + v * low, u + v * high) if v > 0 else (u + v * high, u + v * low)


def compare(x: Surd, y: Surd) -> int:
    """-1, 0 or 1 as x is below, equal to or above y: equal surds are equal, others are separated by enclosures refined
    until they part (the roots of distinct factors differ, so no number has two spellings here)."""
    if x == y:
        return 0
    bits = 32
    while True:
        (x_low, x_high), (y_low, y_high) = enclosure(x, bits), enclosure(y, bits)
        if x_high < y_low:
            return -1
        if y_high < x_low:
            return 1
        bits *= 2
        assert bits <= 1 << 16, "the enclosures do not separate"


def factor_key(coefficients: Sequence[object]) -> tuple[int, tuple[tuple[int, int], ...]]:
    """The order of the evidence's factors: (degree, ((numerator, denominator), ...)), lowest degree first."""
    values = [F(c) for c in coefficients]
    return len(values) - 1, tuple((c.numerator, c.denominator) for c in values)


def complex_claim(lead: object, factors: Sequence[tuple[Sequence[object], int]]) -> tuple[Polynomial, tuple, tuple]:
    """(polynomial, value, evidence) of complex_roots(lead * prod f**e) for distinct monic irreducible factors f of
    degree 1 or 2 (coefficients lowest degree first) with their exponents."""
    entries = sorted(factors, key=lambda entry: factor_key(entry[0]))
    evidence = (rational(lead), tuple((tuple(rational(c) for c in coefficients), exponent)
                                      for coefficients, exponent in entries))
    roots = []
    for coefficients, exponent in factors:
        for real, imag, real_value, imag_value in factor_roots(coefficients):
            roots.append((real_value, imag_value, record(real, imag, exponent)))
    roots.sort(key=cmp_to_key(lambda x, y: compare(x[0], y[0]) or compare(x[1], y[1])))
    return polynomial(expand(lead, factors)), tuple(item for _, _, item in roots), evidence


def random_factors(rng: random.Random, count: int) -> list[tuple[tuple[object, ...], int]]:
    """Up to count distinct monic irreducible factors of degree 1 or 2 with exponents from 1 to 3; about a third of the
    quadratics have rational imaginary parts (D = -4s^2)."""
    keys, factors = set(), []
    for _ in range(count):
        choice = rng.random()
        if choice < 0.45:
            coefficients: tuple[object, ...] = (F(rng.randrange(-12, 13), rng.randrange(1, 4)), 1)
        elif choice < 0.65:
            b, s = F(rng.randrange(-8, 9), rng.randrange(1, 3)), F(rng.randrange(1, 6), rng.randrange(1, 3))
            coefficients = (b * b / 4 + s * s, b, 1)  # The roots -b/2 -+ s i.
        else:
            c, b = F(rng.randrange(-15, 16), rng.randrange(1, 4)), F(rng.randrange(-8, 9), rng.randrange(1, 3))
            if rational_sqrt(b * b - 4 * c) is not None:  # Reducible over Q (D = 0 included).
                continue
            coefficients = (c, b, 1)
        if factor_key(coefficients) not in keys:
            keys.add(factor_key(coefficients))
            factors.append((coefficients, rng.choice((1, 1, 1, 2, 3))))
    return factors


def forged_rational(numerator: object, denominator: object = 1, *, missing: str = "") -> Q:
    """A Rational made with object.__new__, whose fields may break lowest terms, a positive denominator or exactness;
    the field named by missing stays unset."""
    value = object.__new__(Q)
    for name, item in (("numerator", numerator), ("denominator", denominator)):
        if name != missing:
            object.__setattr__(value, name, item)
    return value


class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class SubRational(Q):
    """A Rational subclass, equal to the right number but no Rational of the model."""


class RootsTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object, evidence: object) -> None:
        """A completed v6 Result of exactly that value and evidence, its parts of their exact types, with its four
        checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), type(value))
        if type(value) is tuple:
            for root in result.value:
                self.assertIs(type(root), ComplexRoot)
                self.assertIs(type(root.multiplicity), int)
                self.assertIs(type(root.root), ComplexAlgebraic)
                self.assertIn(type(root.root.real), (Q, RealAlgebraicRoot))
                self.assertIn(type(root.root.imag), (Q, RealAlgebraicRoot))
        self.assertEqual(result.steps[0].outputs[1], evidence)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertUnevaluated(self, result: mf.Result) -> None:
        """A completed Result without a value: NO_CONCLUSION with the card's Unevaluated record, whose request is the
        Result's own; mf.verify verifies the source and the deferral and leaves the mathematics UNKNOWN, also after
        the JSON round trip."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertIs(result.outcome, Outcome.NO_CONCLUSION)
        self.assertIsNone(result.value)
        self.assertEqual(result.steps, ())
        self.assertIs(result.completeness, Completeness.UNKNOWN)
        self.assertEqual(result.deferred, Unevaluated("solve", result.request, DEFERRAL))
        expected = {"result.source": VERIFIED, "result.unevaluated": VERIFIED, "result.mathematics": UNKNOWN}
        self.assertEqual(support.statuses(mf.verify(result)), expected)
        decoded = mf.loads(mf.dumps(result))
        self.assertEqual(decoded, result)
        self.assertEqual(support.statuses(mf.verify(decoded)), expected)
        with self.assertRaises(mf.OperationError):
            result.unwrap()

    def assertProduces(self, operation: str, args: dict[str, object], value: object, evidence: object) -> None:
        """produce() answers exactly that value with that evidence, and the checker accepts the claim."""
        produced = family.produce(operation, dict(args))
        self.assertEqual(produced, (value, evidence), args)
        self.assertIs(type(produced[0]), type(value))
        self.assertIs(support.verdict(family, operation, args, *produced), True, args)

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts: the card's counts and further
        ones on x^3 - x, the card's complex roots with their evidence, and its input without a value, x^3 - 2."""
        for coefficients, lower, upper, count in COUNT_KNOWN:
            with self.subTest(operation=COUNT, coefficients=coefficients, lower=lower, upper=upper):
                self.assertEqual(sturm_count(coefficients, bound(lower), bound(upper)), count)
                result = support.run(COUNT, polynomial=polynomial(coefficients), lower=lower, upper=upper)
                self.assertVerified(result, count, ())
                self.roundtrip(result)
        for lower, upper, count in COUNT_SPECIAL:
            with self.subTest(operation=COUNT, cubic=True, lower=lower, upper=upper):
                self.assertEqual(sturm_count(CUBIC, bound(lower), bound(upper)), count)
                self.assertVerified(support.run(COUNT, polynomial=polynomial(CUBIC), lower=lower, upper=upper),
                                    count, ())
        # The defaults are the whole real line, and the request names them.
        result = support.run(COUNT, polynomial=polynomial([-2, 0, 1]))
        self.assertVerified(result, 2, ())
        self.assertEqual(dict(result.request.arguments)["lower"], "-inf")
        self.assertEqual(dict(result.request.arguments)["upper"], "+inf")
        one, zero = Q(1), Q(0)
        sqrt3_half = [RealAlgebraicRoot((-3, 0, 4), k) for k in (0, 1)]
        complex_known = (
            # x^3 - 1: by real part, the pair -1/2 -+ (sqrt 3/2) i comes before 1.
            ((-1, 0, 0, 1), (((-1, 1), 1), ((1, 1, 1), 1)),
             (record(Q(-1, 2), sqrt3_half[0], 1), record(Q(-1, 2), sqrt3_half[1], 1), record(one, zero, 1))),
            # x^4 - 1: -1, -i, i, 1 by real part, then imaginary part.
            ((-1, 0, 0, 0, 1), (((-1, 1), 1), ((1, 1), 1), ((1, 0, 1), 1)),
             (record(-one, zero, 1), record(zero, -one, 1), record(zero, one, 1), record(one, zero, 1))),
            # (x^2 + 1)^2 (x - 2): -i (2), i (2), 2 (1).
            ((-2, 1, -4, 2, -2, 1), (((-2, 1), 1), ((1, 0, 1), 2)),
             (record(zero, -one, 2), record(zero, one, 2), record(Q(2), zero, 1))),
            # x^4 + 4 = (x^2 - 2x + 2)(x^2 + 2x + 2): -1 -+ i, then 1 -+ i.
            ((4, 0, 0, 0, 1), (((2, -2, 1), 1), ((2, 2, 1), 1)),
             (record(-one, -one, 1), record(-one, one, 1), record(one, -one, 1), record(one, one, 1))),
            # A nonzero constant has no root.
            ((5,), (), ()),
        )
        for coefficients, factors, value in complex_known:
            with self.subTest(operation=COMPLEX, coefficients=coefficients):
                p, expected, evidence = complex_claim(coefficients[-1], factors)
                self.assertEqual(p, polynomial(coefficients))
                self.assertEqual(expected, value)
                result = support.run(COMPLEX, polynomial=p)
                self.assertVerified(result, value, evidence)
                self.roundtrip(result)
        self.assertEqual(support.run(COMPLEX, polynomial=polynomial([5])).steps[0].outputs[1], (Q(5), ()))
        self.assertUnevaluated(support.run(COMPLEX, polynomial=polynomial([-2, 0, 0, 1])))

    def test_irreducible_factors_of_degree_three_or_more_are_unevaluated(self):
        """A polynomial with an irreducible factor of degree three or more gives the card's Unevaluated record and no
        value, also beside linear and quadratic factors; produce() answers the record with the evidence ()."""
        rng = random.Random(20261140)
        for index, irreducible in enumerate(IRREDUCIBLE):
            for extra in ([], random_factors(rng, 3)):
                with self.subTest(irreducible=irreducible, extra=extra):
                    factors = [(irreducible, 1 + index % 2)] + extra
                    p = polynomial(expand(F(rng.randrange(1, 9), rng.randrange(1, 4)), factors))
                    result = support.run(COMPLEX, polynomial=p)
                    self.assertUnevaluated(result)
                    self.assertEqual(family.produce(COMPLEX, {"polynomial": p}), (result.deferred, ()))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the card's domain (the zero polynomial, an end of another type or the
        other side's infinity, lower > upper), and every Polynomial or Rational made with object.__new__ that breaks
        the canonical form while run() admits it. produce() tests the types itself too."""
        cubic = polynomial(CUBIC)
        small = {COUNT: dict(polynomial=cubic, lower=-1, upper=Q(1, 2)), COMPLEX: dict(polynomial=cubic)}
        inexact = (*STAND_INS, F(1, 2), Integer(1), SubRational(1, 2), sqrt(2), X, cubic.coefficients,
                   support.oversized_matrix())
        for operation, args in small.items():
            for name in args:
                for item in inexact + ((cubic,) if name != "polynomial" else ()):
                    with self.subTest(operation=operation, argument=name, item=type(item).__name__):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: item})))
        zero = polynomial([])
        self.assertEqual(zero.coefficients, (Q(0),))
        cases = [(COUNT, dict(polynomial=zero, lower=lower, upper=upper), "a nonzero polynomial")
                 for lower, upper in (("-inf", "+inf"), (0, 1), (Q(1, 2), "+inf"))]
        cases.append((COMPLEX, dict(polynomial=zero), "a nonzero polynomial"))
        for lower, upper in (("+inf", "+inf"), ("+inf", 0), ("inf", 0), ("-inf ", 0), ("−inf", 0), ("-INF", 0),
                             ("-infinity", 0), ("", 0)):
            cases.append((COUNT, dict(polynomial=cubic, lower=lower, upper=upper),
                          "lower to be an int, a Rational or '-inf'"))
        for lower, upper in (("-inf", "-inf"), (0, "-inf"), (0, "inf"), (0, "+inf "), (0, "∞")):
            cases.append((COUNT, dict(polynomial=cubic, lower=lower, upper=upper),
                          "upper to be an int, a Rational or '+inf'"))
        for lower, upper in ((1, Q(1, 2)), (Q(3, 2), 1), (2, -2), (Q(1, 3), Q(1, 4)), (1, 0)):
            cases.append((COUNT, dict(polynomial=cubic, lower=lower, upper=upper), "lower <= upper"))
        for operation, args, text in cases:
            with self.subTest(operation=operation, **{name: repr(item) for name, item in args.items()}):
                result = support.run(operation, **args)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(args))
        # Polynomials and Rationals made with object.__new__ outside the canonical form: run() admits them by their
        # class, and produce() refuses them.
        trailing = cubic.coefficients + (Q(0),)
        forged_polynomials = (
            ("a trailing zero", support.forged(Polynomial, coefficients=trailing, variable=X)),
            ("int coefficients", support.forged(Polynomial, coefficients=CUBIC, variable=X)),
            ("a coefficient outside lowest terms",
             support.forged(Polynomial, coefficients=(forged_rational(2, 4), Q(1)), variable=X)),
            ("a negative denominator",
             support.forged(Polynomial, coefficients=(forged_rational(1, -2), Q(1)), variable=X)),
            ("a str variable", support.forged(Polynomial, coefficients=cubic.coefficients, variable="x")),
            ("no coefficient", support.forged(Polynomial, coefficients=(), variable=X)),
            ("1,026 coefficients",
             support.forged(Polynomial, coefficients=(Q(1),) * (MAX_POLYNOMIAL_DEGREE + 2), variable=X)),
        )
        for label, item in forged_polynomials:
            for operation, args in small.items():
                with self.subTest(operation=operation, forged=label):
                    result = support.run(operation, **dict(args, polynomial=item))
                    self.assertInvalidInput(result)
                    self.assertIn("canonical Polynomial", result.error.message)
        for item in (forged_rational(10, 2), forged_rational(-4, -2), forged_rational(6, 4), forged_rational(5, -1),
                     forged_rational(2, 0), forged_rational(True, 1)):
            for name in ("lower", "upper"):
                args = dict(polynomial=cubic, lower="-inf", upper="+inf")
                with self.subTest(argument=name, forged=(item.numerator, item.denominator)):
                    result = support.run(COUNT, **dict(args, **{name: item}))
                    self.assertInvalidInput(result)
                    self.assertIn("lowest terms with a positive denominator", result.error.message)
        # produce() itself refuses every stand-in, each forged record (a float field and a missing field too, which
        # run()'s inspection of the arguments meets first), other operations and other argument sets.
        missing = (support.forged(Polynomial, variable=X), support.forged(Polynomial, coefficients=cubic.coefficients),
                   support.forged(Polynomial, coefficients=[*cubic.coefficients], variable=X),
                   support.forged(Polynomial, coefficients=(0.0, -1.0, 0.0, 1.0), variable=X))
        for operation, args in small.items():
            for name in args:
                for item in (*inexact, False, forged_rational(3.0), forged_rational(1, missing="numerator"),
                             forged_rational(1, missing="denominator"), forged_rational(10, 2),
                             *(forged for _, forged in forged_polynomials), *missing):
                    with self.subTest(operation=operation, argument=name, item=type(item).__name__, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: item}))
            for wrong in (dict(args, extra=1), {}, tuple(args.items()), None, [("polynomial", cubic)]):
                with self.subTest(operation=operation, arguments=type(wrong).__name__), self.assertRaises(InvalidInput):
                    family.produce(operation, wrong)
        with self.assertRaises(InvalidInput):  # The arguments of the other operation.
            family.produce(COMPLEX, dict(small[COUNT]))
        for operation in ("count_real_root", "Complex_roots", "complex_roots ", "sturm_sequence", "denest_sqrt", "",
                          None, 1):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(small[COMPLEX]))
        # The value function raises the structured path's error.
        for operation, args in ((COUNT, dict(polynomial=zero)), (COUNT, dict(polynomial=cubic, lower=1, upper=0)),
                                (COMPLEX, dict(polynomial=zero))):
            with self.subTest(value_function=operation), self.assertRaises(mf.OperationError) as raised:
                support.value(operation, **args)
            self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in an int argument (an end), also inside a Rational end or a coefficient, and
        max_work=1 exceed the budget. No argument takes a matrix: support.oversized_matrix() in any argument is invalid
        input (test_invalid_inputs). A factorization whose Berlekamp matrix exceeds max_matrix_entries, and a Sturm
        chain beyond a small max_work, fail at once."""
        cubic = polynomial(CUBIC)
        huge_ends = ((support.HUGE, "+inf"), ("-inf", -support.HUGE), (Q(support.HUGE + 1, 3), "+inf"),
                     ("-inf", Q(1, support.HUGE)), (Q(-support.HUGE - 1, support.HUGE), 0))
        for lower, upper in huge_ends:
            args = dict(polynomial=cubic, lower=lower, upper=upper)
            with self.subTest(lower=type(lower).__name__, upper=type(upper).__name__):
                self.assertResourceLimit(support.run(COUNT, **args))
                with self.assertRaises(ResourceLimitError):
                    family.produce(COUNT, args)
        huge_polynomials = (Polynomial((Q(support.HUGE), Q(1)), X), Polynomial((Q(1, support.HUGE), Q(1)), X),
                            Polynomial((Q(1), Q(0), Q(support.HUGE + 1, 7)), X))
        for item in huge_polynomials:
            for operation, args in ((COUNT, dict(polynomial=item, lower="-inf", upper="+inf")),
                                    (COMPLEX, dict(polynomial=item))):
                with self.subTest(operation=operation, huge_coefficient=True):
                    self.assertResourceLimit(support.run(operation, **args))
                    with self.assertRaises(ResourceLimitError):
                        family.produce(operation, args)
        for operation, args in ((COUNT, dict(polynomial=cubic, lower=-1, upper=1)),
                                (COUNT, dict(polynomial=polynomial([5]), lower="-inf", upper="+inf")),
                                (COMPLEX, dict(polynomial=cubic)), (COMPLEX, dict(polynomial=polynomial([5]))),
                                (COMPLEX, dict(polynomial=polynomial(IRREDUCIBLE[0])))):
            with self.subTest(operation=operation, max_work=1, args=args):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
                with computation(ComputationLimits(max_work=1), fresh=True), self.assertRaises(ResourceLimitError):
                    family.produce(operation, dict(args))
        # prod (x^2 + k) for k = 1..33 is square-free of degree 66: its Berlekamp matrix has 66 * 66 = 4,356 entries,
        # more than max_matrix_entries (4,096). (The product for k = 1..32, of degree 64, fits the matrix, and its
        # Zassenhaus recombination runs past the default max_work.)
        wide = polynomial(expand(1, [((k, 0, 1), 1) for k in range(1, 34)]))
        start = time.perf_counter()
        self.assertResourceLimit(support.run(COMPLEX, polynomial=wide))
        self.assertLess(time.perf_counter() - start, FAST)
        rng = random.Random(160)
        dense = polynomial([rng.randrange(-9, 10) for _ in range(160)] + [rng.randrange(1, 10)])
        start = time.perf_counter()
        self.assertResourceLimit(support.run(COUNT, polynomial=dense, limits=ComputationLimits(max_work=100_000)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The evidence of both operations is fixed by the claim: () for a count, and for complex roots the leading
        coefficient with the unique factorization into monic irreducible factors. A true claim that the producer did
        not make is therefore one built by hand, here from chosen roots and factors, or one on arguments that write the
        same input another way: an integral end as an int or as a Rational, or the polynomial times a nonzero rational
        (the same roots under another leading coefficient). The checker accepts each, and the producer, asked, answers
        the same."""
        rng = random.Random(20261141)
        accepted = 0
        while accepted < 40:
            built = random_built(rng)
            p = polynomial(built.coefficients())
            points = built.points(rng) + [None]
            low, high = rng.choice(points), rng.choice(points)
            if low is not None and high is not None and low > high:
                low, high = high, low
            lower, upper = request_ends(low, high)
            count = built.count(low, high)
            with self.subTest(built=built, lower=lower, upper=upper):
                self.assertAccepts(COUNT, dict(polynomial=p, lower=lower, upper=upper), count)
                self.assertEqual(family.produce(COUNT, dict(polynomial=p, lower=lower, upper=upper)), (count, ()))
                scaled = polynomial(F(-3, 7) * c for c in built.coefficients())
                self.assertAccepts(COUNT, dict(polynomial=scaled, lower=lower, upper=upper), count)
                self.assertEqual(family.produce(COUNT, dict(polynomial=scaled, lower=lower, upper=upper)), (count, ()))
                accepted += 1
        cubic = polynomial(CUBIC)
        for lower, upper, count in ((-1, 1, 3), (0, "+inf", 2), ("-inf", -1, 1), (1, 1, 1)):
            for low, high in ((lower, upper), (Q(lower) if type(lower) is int else lower,
                                               Q(upper) if type(upper) is int else upper)):
                with self.subTest(lower=repr(low), upper=repr(high)):
                    self.assertAccepts(COUNT, dict(polynomial=cubic, lower=low, upper=high), count)
                    self.assertEqual(family.produce(COUNT, dict(polynomial=cubic, lower=low, upper=high)), (count, ()))
        for _ in range(30):
            factors = random_factors(rng, rng.randrange(1, 5))
            if not factors:
                continue
            lead = F(rng.choice((-1, 1)) * rng.randrange(1, 20), rng.randrange(1, 6))
            p, value, evidence = complex_claim(lead, factors)
            with self.subTest(lead=lead, factors=factors):
                self.assertAccepts(COMPLEX, dict(polynomial=p), value, evidence)
                self.assertEqual(family.produce(COMPLEX, dict(polynomial=p)), (value, evidence))
                scaled, _, scaled_evidence = complex_claim(lead * F(-5, 3), factors)
                self.assertEqual(scaled_evidence, (rational(lead * F(-5, 3)), evidence[1]))
                self.assertAccepts(COMPLEX, dict(polynomial=scaled), value, scaled_evidence)
                self.assertEqual(family.produce(COMPLEX, dict(polynomial=scaled)), (value, scaled_evidence))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence, all on Results of the producer;
        mf.verify fails the mathematics of a Result whose value was replaced, while its trace stays verified."""
        cubic = polynomial(CUBIC)
        args = dict(polynomial=cubic, lower=-1, upper=Q(1, 2))  # The roots -1 and 0.
        result = support.run(COUNT, **args)
        self.assertVerified(result, 2, ())
        cases = [(f"value {item!r}", item, ()) for item in (1, 3, 0, -2, 4, 2.0, "2", None, Q(2), Integer(2), (2,),
                                                            [2], True)]
        cases.append(("a giant value", GIANT, ()))
        cases += [(f"evidence {item!r}", 2, item) for item in ([], None, (2,), ((),), ("",))]
        cases.append(("oversized evidence", 2, (0,) * 100_000))
        for label, item, evidence in cases:
            with self.subTest(operation=COUNT, case=label):
                self.assertRejects(COUNT, args, item, evidence)
        single = dict(polynomial=cubic, lower=0, upper=Q(1, 2))  # The root 0: the right int is 1, which True equals.
        self.assertEqual(support.run(COUNT, **single).value, 1)
        for item in (True, 1.0, Q(1), Integer(1), "1"):
            with self.subTest(operation=COUNT, single=repr(item)):
                self.assertRejects(COUNT, single, item, ())
        for operation in (COMPLEX, "count_real_root", "COUNT_REAL_ROOTS", "", None):
            with self.subTest(operation=operation, claim=COUNT):
                self.assertRejects(operation, args, 2, ())
        report = support.statuses(mf.verify(support.tampered(result, 3)))
        self.assertIs(report["result.trace"], VERIFIED)
        self.assertIs(report["result.mathematics"], FAILED)
        report = support.statuses(mf.verify(support.tampered(result, 2, ((),))))
        self.assertIs(report["result.mathematics"], FAILED)

        p, value, evidence = complex_claim(F(-3, 2), BASE_FACTORS)
        args = dict(polynomial=p)
        result = support.run(COMPLEX, **args)
        self.assertVerified(result, value, evidence)
        lead, factors = evidence
        self.assertEqual([root.multiplicity for root in value], [1, 2, 1, 2, 1, 1, 1])
        sqrt2 = RealAlgebraicRoot((-2, 0, 1), 1)
        self.assertEqual(value[-1].root.real, sqrt2)

        def replaced(index: int, **parts: object) -> tuple:
            root = value[index]
            item = record(parts.get("real", root.root.real), parts.get("imag", root.root.imag),
                          parts.get("multiplicity", root.multiplicity))
            return value[:index] + (item,) + value[index + 1:]

        factorization, qq = factor_with_evidence(p)  # v0.7's factorization record and its irreducibility evidence.
        wrong_values = [
            ("reversed", value[::-1]), ("rotated", value[1:] + value[:1]), ("a root missing", value[:-1]),
            ("an extra root", value + (record(Q(5), Q(0), 1),)), ("a root twice", value[:1] + value),
            ("multiplicity 1 for a double root", replaced(1, multiplicity=1)),
            ("multiplicity 2 for a simple root", replaced(0, multiplicity=2)),
            ("the conjugate's sign", replaced(3, imag=RealAlgebraicRoot((-3, 0, 4), 0))),
            ("sqrt 3 for sqrt 2", replaced(6, real=RealAlgebraicRoot((-3, 0, 1), 1))),
            ("-1/2 as a root of 2x + 1", replaced(2, real=RealAlgebraicRoot((1, 2), 0))),
            ("an int imaginary part", replaced(0, imag=0)),
            ("a float imaginary part", replaced(0, imag=0.0)),
            ("an expression for sqrt 2", replaced(6, real=sqrt(2))),
            ("a forged ComplexRoot without its multiplicity",
             value[:1] + (support.forged(ComplexRoot, root=value[1].root),) + value[2:]),
            ("an impostor ComplexRoot", (support.impostor(value[0]),) + value[1:]),
            ("a list", list(value)), ("one record", value[0]), ("no root", ()), ("None", None),
            ("the evidence as the value", evidence),
        ]
        wrong_evidence = [
            ("a list", [lead, factors]), ("None", None), ("the leading coefficient only", (lead,)),
            ("the factors only", (factors,)), ("an extra item", (lead, factors, ())),
            ("v0.7's irreducibility evidence too", (lead, factors, qq)), ("v0.7's evidence", qq),
            ("a float leading coefficient", (-1.5, factors)), ("the leading coefficient negated", (-lead, factors)),
            ("1 as the leading coefficient", (Q(1), factors)), ("factors as a list", (lead, list(factors))),
            ("a factor missing", (lead, factors[1:])), ("factors reversed", (lead, factors[::-1])),
            ("an exponent too small", (lead, factors[:3] + ((factors[3][0], 1),))),
            ("the factorization record", (lead, factorization)), ("the record alone", (factorization,)),
            ("oversized", (lead, factors * 30_000)),
        ]
        for label, item in wrong_values:
            with self.subTest(operation=COMPLEX, value=label):
                self.assertRejects(COMPLEX, args, item, evidence)
        for label, item in wrong_evidence:
            with self.subTest(operation=COMPLEX, evidence=label):
                self.assertRejects(COMPLEX, args, value, item, may_exhaust=label == "oversized")
        for operation in (COUNT, "Complex_roots", "roots", "", None):
            with self.subTest(operation=operation, claim=COMPLEX):
                self.assertRejects(operation, args, value, evidence)
        report = support.statuses(mf.verify(support.tampered(result, value[::-1])))
        self.assertIs(report["result.trace"], VERIFIED)
        self.assertIs(report["result.mathematics"], FAILED)
        report = support.statuses(mf.verify(support.tampered(result, value, (lead, factors[1:]))))
        self.assertIs(report["result.mathematics"], FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text; an
        Unevaluated Result keeps its report as well."""
        requests = [(COUNT, dict(polynomial=polynomial(coefficients), lower=lower, upper=upper))
                    for coefficients, lower, upper, _ in COUNT_KNOWN]
        requests += [(COMPLEX, dict(polynomial=polynomial(coefficients)))
                     for coefficients in ((-1, 0, 0, 1), (-1, 0, 0, 0, 1), (-2, 1, -4, 2, -2, 1), (4, 0, 0, 0, 1),
                                          (5,))]
        requests.append((COMPLEX, dict(polynomial=complex_claim(F(-3, 2), BASE_FACTORS)[0])))
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        unevaluated = support.run(COMPLEX, polynomial=polynomial(IRREDUCIBLE[0]))
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, args), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, args=args):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            self.assertUnevaluated(unevaluated)
            # The patch is live: a new run reaches the patched producer and fails.
            for operation, args in requests[:1] + requests[-1:]:
                failed = support.run(operation, **args)
                self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: an int, and tuples of
        v0.7's ComplexRoot records with rational and irrational parts, multiplicities and no root. An Unevaluated
        Result renders its record."""
        cubic = polynomial(CUBIC)
        rendered = (
            (COUNT, dict(polynomial=cubic, lower=-1, upper=1), "3", "3"),
            (COUNT, dict(polynomial=polynomial([1, 0, 1]), lower="-inf", upper="+inf"), "0", "0"),
            (COMPLEX, dict(polynomial=polynomial([1, 0, 1])), "((0 + -1*i); multiplicity=1, (0 + 1*i); multiplicity=1)",
             r"\left(\left(0 + -1\,i\right); \text{multiplicity=}1, \left(0 + 1\,i\right); "
             r"\text{multiplicity=}1\right)"),
            (COMPLEX, dict(polynomial=polynomial([-2, 1, -4, 2, -2, 1])),
             "((0 + -1*i); multiplicity=2, (0 + 1*i); multiplicity=2, (2 + 0*i); multiplicity=1)",
             r"\left(\left(0 + -1\,i\right); \text{multiplicity=}2, \left(0 + 1\,i\right); \text{multiplicity=}2, "
             r"\left(2 + 0\,i\right); \text{multiplicity=}1\right)"),
            (COMPLEX, dict(polynomial=polynomial([-2, 0, 1])),
             "((RootOf(t^2 - 2; real_index=0) + 0*i); multiplicity=1, (RootOf(t^2 - 2; real_index=1) + 0*i); "
             "multiplicity=1)",
             r"\left(\left(\operatorname{RootOf}_{\mathbb{R}}\left(t^{2} - 2; 0\right) + 0\,i\right); "
             r"\text{multiplicity=}1, \left(\operatorname{RootOf}_{\mathbb{R}}\left(t^{2} - 2; 1\right) + 0\,i\right); "
             r"\text{multiplicity=}1\right)"),
            (COMPLEX, dict(polynomial=polynomial([1, 1, 1])),
             "((-1/2 + RootOf(4*t^2 - 3; real_index=0)*i); multiplicity=1, (-1/2 + RootOf(4*t^2 - 3; real_index=1)*i); "
             "multiplicity=1)",
             r"\left(\left(-\frac{1}{2} + \operatorname{RootOf}_{\mathbb{R}}\left(4\, t^{2} - 3; 0\right)\,i\right); "
             r"\text{multiplicity=}1, \left(-\frac{1}{2} + \operatorname{RootOf}_{\mathbb{R}}\left(4\, t^{2} - 3; "
             r"1\right)\,i\right); \text{multiplicity=}1\right)"),
            (COMPLEX, dict(polynomial=polynomial([5])), "()", r"\left(\right)"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(operation=operation, text=text):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual((mf.render_text(result.value), mf.to_latex(result.value)), (text, latex))
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run(COUNT, polynomial=cubic, lower=-1, upper=1).value)["data"],
                         {"type": "integer", "value": "3"})
        data = mf.to_data(support.run(COMPLEX, polynomial=polynomial([1, 0, 1])).value)["data"]
        self.assertEqual(data["type"], "tuple")
        self.assertEqual(data["items"][0], {"type": "v5_complex_root", "root": {
            "type": "v5_complex_algebraic", "real": {"type": "rational", "numerator": "0", "denominator": "1"},
            "imag": {"type": "rational", "numerator": "-1", "denominator": "1"}},
            "multiplicity": {"type": "integer", "value": "1"}})
        unevaluated = support.run(COMPLEX, polynomial=polynomial(IRREDUCIBLE[0]))
        self.assertEqual(mf.loads(mf.dumps(unevaluated)), unevaluated)
        self.assertEqual(mf.render_text(unevaluated),
                         "completed / no_conclusion [not_applicable; unknown]: Unevaluated(solve, complex_roots, "
                         f"Reason(no_applicable_rule, {DEFERRAL.detail}))")
        self.assertIn(r"\text{Unevaluated}", mf.to_latex(unevaluated))

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path. On an
        input without a value the lean path falls back, and the value function raises the structured path's error."""
        rng = random.Random(20261142)
        cases: list[tuple[str, dict[str, object]]] = []
        for _ in range(20):
            built = random_built(rng)
            points = built.points(rng) + [None]
            low, high = sorted((rng.choice(points[:-1]), rng.choice(points[:-1])))
            lower, upper = request_ends(rng.choice((low, None)), rng.choice((high, None)))
            cases.append((COUNT, dict(polynomial=polynomial(built.coefficients()), lower=lower, upper=upper)))
        for _ in range(16):
            factors = random_factors(rng, rng.randrange(1, 5))
            if factors:
                cases.append((COMPLEX, dict(polynomial=complex_claim(F(rng.randrange(1, 9), 2), factors)[0])))
        for operation, args in cases:
            with self.subTest(operation=operation, args=args):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, **args), expected)
        self.assertEqual(support.value(COUNT, polynomial=polynomial([-2, 0, 1])), 2)  # The defaults filled in.
        for irreducible in IRREDUCIBLE[:3]:
            with self.subTest(unevaluated=irreducible):
                self.assertIs(support.lean_value(COMPLEX, polynomial=polynomial(irreducible)), lean.FALLBACK)
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(COMPLEX, polynomial=polynomial(irreducible))
                self.assertEqual(raised.exception.code, "operation_failed")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: a polynomial travels as
        its text and variable, an end as a JSON integer, a string "p/q" or an infinity."""
        cubic = polynomial(CUBIC)
        requests = (
            (COUNT, dict(polynomial=cubic, lower=Q(-1, 2), upper="+inf")),
            (COUNT, dict(polynomial=cubic, lower=-1, upper=1)),
            (COUNT, dict(polynomial=polynomial([F(1, 3), 0, -2, 0, 1]), lower="-inf", upper=Q(1, 3))),
            (COUNT, dict(polynomial=cubic, lower=2 ** 60, upper=2 ** 61)),
            (COMPLEX, dict(polynomial=polynomial([4, 0, 0, 0, 1]))),
            (COMPLEX, dict(polynomial=complex_claim(F(-3, 2), BASE_FACTORS)[0])),
        )
        for operation, args in requests:
            with self.subTest(operation=operation, args=args):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        raw = (
            (COUNT, {"polynomial": {"text": "x^3 - x", "variable": "x"}, "lower": "-1/2"}, 2),
            (COUNT, {"polynomial": {"text": "x^2 - 2", "variable": "x"}, "lower": 0, "upper": 2}, 1),
            (COUNT, {"polynomial": {"text": "y^2 - 2*y + 1", "variable": "y"}, "lower": 0, "upper": 2}, 1),
            (COUNT, {"polynomial": {"text": "x^2 + 1", "variable": "x"}, "lower": "-inf", "upper": "+inf"}, 0),
            (COUNT, {"polynomial": {"text": "5", "variable": "x"}}, 0),
            (COMPLEX, {"polynomial": {"text": "x^4 - 1", "variable": "x"}},
             (record(Q(-1), Q(0), 1), record(Q(0), Q(-1), 1), record(Q(0), Q(1), 1), record(Q(1), Q(0), 1))),
        )
        for operation, request, value in raw:
            with self.subTest(operation=operation, request=request):
                result = self.answer(support.call(operation, request))
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)
        self.assertUnevaluated(self.answer(support.call(COMPLEX, {"polynomial": {"text": "x^3 - 2", "variable": "x"}})))
        errors = (
            (COUNT, {"polynomial": {"text": "0", "variable": "x"}}, "a nonzero polynomial"),
            (COMPLEX, {"polynomial": {"text": "0", "variable": "x"}}, "a nonzero polynomial"),
            (COUNT, {"polynomial": {"text": "x^3 - x", "variable": "x"}, "lower": 3, "upper": 1}, "lower <= upper"),
            (COUNT, {"polynomial": {"text": "x^3 - x", "variable": "x"}, "lower": "+inf"}, "or '-inf'"),
            (COUNT, {"polynomial": {"text": "x^3 - x", "variable": "x"}, "upper": "-inf"}, "or '+inf'"),
            (COUNT, {"polynomial": {"text": "x^3 - x", "variable": "x"}, "lower": "sqrt(2)"}, "or '-inf'"),
            (COUNT, {"polynomial": {"text": "x^3 - x", "variable": "x"}, "lower": 0.5}, "A point is"),
        )
        for operation, request, text in errors:
            with self.subTest(operation=operation, error=request):
                error = self.assertToolError(support.call(operation, request), "invalid_input")
                self.assertIn(text, error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 200 seeded inputs per operation, and the
        checker accepts every answer with its evidence: counts on built polynomials with ends at, beside and near their
        roots (Built.count, confirmed by sturm_count) and on random dense polynomials (sturm_count); complex roots of
        random products of linear and quadratic factors (complex_claim), and the Unevaluated record when an irreducible
        factor of degree three or more joins them."""
        rng = random.Random(20261143)
        counts: Counter[str] = Counter()
        while counts["count built"] < 240:
            built = random_built(rng)
            coefficients = built.coefficients()
            p = polynomial(coefficients)
            points = built.points(rng) + [None]
            for _ in range(6):
                low, high = rng.choice(points), rng.choice(points)
                if low is not None and high is not None and low > high:
                    low, high = high, low
                lower, upper = request_ends(low, high)
                count = built.count(low, high)
                with self.subTest(built=built, lower=lower, upper=upper):
                    self.assertEqual(sturm_count(coefficients, low, high), count)  # The two references agree.
                    self.assertProduces(COUNT, dict(polynomial=p, lower=lower, upper=upper), count, ())
                    counts["count built"] += 1
                    counts[f"count {min(count, 3)}"] += 1
        while counts["count dense"] < 200:
            coefficients = trim(F(rng.randrange(-9, 10), rng.choice((1, 1, 2, 3))) for _ in range(rng.randrange(1, 10)))
            ends = [None] + [F(rng.randrange(-12, 13), rng.randrange(1, 4)) for _ in range(3)]
            low, high = rng.choice(ends), rng.choice(ends)
            lower, upper = request_ends(low, high)
            args = dict(polynomial=polynomial(coefficients), lower=lower, upper=upper)
            if not coefficients or (low is not None and high is not None and low > high):
                with self.subTest(refused=args), self.assertRaises(InvalidInput):
                    family.produce(COUNT, args)
                counts["count refused"] += 1
                continue
            with self.subTest(coefficients=coefficients, lower=lower, upper=upper):
                self.assertProduces(COUNT, args, sturm_count(coefficients, low, high), ())
                counts["count dense"] += 1
        while counts["complex"] < 220:
            factors = random_factors(rng, rng.randrange(1, 5))
            if not factors:
                continue
            lead = F(rng.choice((-1, 1)) * rng.randrange(1, 20), rng.randrange(1, 6))
            p, value, evidence = complex_claim(lead, factors)
            with self.subTest(lead=lead, factors=factors):
                self.assertProduces(COMPLEX, dict(polynomial=p), value, evidence)
                counts["complex"] += 1
                counts["complex roots"] += len(value)
                parts = [(type(root.root.real), type(root.root.imag)) for root in value]
                counts["complex irrational parts"] += sum(RealAlgebraicRoot in types for types in parts)
        for _ in range(12):
            factors = [(rng.choice(IRREDUCIBLE), rng.choice((1, 2)))] + random_factors(rng, rng.randrange(0, 3))
            p = polynomial(expand(F(rng.randrange(1, 9), rng.randrange(1, 4)), factors))
            with self.subTest(unevaluated=factors):
                value, evidence = family.produce(COMPLEX, {"polynomial": p})
                self.assertEqual((value, evidence), (Unevaluated("solve", _request(p), DEFERRAL), ()))
                counts["unevaluated"] += 1
        self.assertGreaterEqual(counts["count built"], 240, counts)
        self.assertGreaterEqual(counts["count dense"], 200, counts)
        self.assertGreater(min(counts["count 0"], counts["count 1"], counts["count 2"], counts["count 3"]), 20, counts)
        self.assertGreater(counts["count refused"], 5, counts)
        self.assertGreaterEqual(counts["complex"], 220, counts)
        self.assertGreater(counts["complex irrational parts"], 150, counts)

    # -----------------------------------------------------------------------------------------------------------
    # The producer's choices, sizes and work

    def test_close_roots_shared_parts_and_large_coefficients(self):
        """The exact order holds where approximations would hesitate: real parts shared by several roots, real roots
        close to each other (7/5 < 140/99 < sqrt 2 < 99/70, and a convergent of sqrt 2 some 1.6e-12 above it),
        irrational roots of two fields 7e-7 apart, and coefficients of 2**90."""
        cases = (
            (F(1), (((0, 1), 1), ((1, 0, 1), 1), ((4, 0, 1), 2), ((2, 0, 1), 1))),  # Real part 0 seven times.
            (F(5), (((2, 2, 1), 1), ((3, 2, 1), 2), ((1, 1), 1))),  # -1 - sqrt(2) i, -1 - i, -1, -1 + i, ...
            (F(1), (((-2, 0, 1), 1), ((-3, 0, 1), 1), ((F(-7, 5), 1), 1), ((F(-140, 99), 1), 2), ((F(-99, 70), 1), 1),
                    ((F(3, 2), 1), 1))),
            (F(1), (((F(-665857, 470832), 1), 1), ((-2, 0, 1), 1))),
            (F(1), (((-2, 0, 1), 1), ((F(-1000001, 500000), 0, 1), 1))),
            (F(2 ** 40 + 1, 3), (((3, 2 ** 80 + 1, 1), 2), ((-(2 ** 90), 1), 1), ((2 ** 60, 0, 1), 1))),
        )
        for lead, factors in cases:
            with self.subTest(lead=lead, factors=factors):
                p, value, evidence = complex_claim(lead, factors)
                self.assertVerified(support.run(COMPLEX, polynomial=p), value, evidence)
        # 665857^2 - 2 * 470832^2 = 1 (Pell): the convergent exceeds sqrt 2 by 1/(470832^2 (665857/470832 + sqrt 2)).
        self.assertEqual(F(665857, 470832) ** 2 - 2, F(1, 470832 ** 2))

    def test_refusals_after_validation_are_internal_errors(self):
        """Once the inputs are validated, an InvalidInput from a library function is a bug, never rejected input:
        produce() raises ArithmeticError, and run() reports the producer's internal_error."""
        count_args = dict(polynomial=polynomial(CUBIC), lower=-1, upper=1)
        for operation, args, target in (
                (COUNT, count_args, "mathforge.root_isolation.count_real_roots"),
                (COMPLEX, dict(polynomial=polynomial(CUBIC)), "mathforge.polynomial_exact.factor_with_evidence"),
                (COMPLEX, dict(polynomial=polynomial([1, 1, 1])), "mathforge.complex_numbers._quadratic")):
            with self.subTest(target=target), patch(target, side_effect=InvalidInput("refused")):
                with self.assertRaises(ArithmeticError):
                    family.produce(operation, dict(args))
                self.assertEqual(support.run(operation, **args).error,
                                 ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_records_are_those_of_quadratic_roots(self):
        """The card's producer: the records of an irreducible quadratic factor are what mf.quadratic_roots(1, b, c)
        builds, each with the factor's exponent as its multiplicity, and the evidence is the factorization that
        v0.7's factor finds, without its irreducibility evidence."""
        rng = random.Random(20261144)
        tried = 0
        while tried < 12:
            c, b = F(rng.randrange(-20, 21), rng.randrange(1, 5)), F(rng.randrange(-9, 10), rng.randrange(1, 4))
            if rational_sqrt(b * b - 4 * c) is not None:
                continue
            tried += 1
            exponent = rng.choice((1, 2, 3))
            p = polynomial(expand(1, [((c, b, 1), exponent)]))
            with self.subTest(c=c, b=b, exponent=exponent):
                legacy = mf.quadratic_roots(1, rational(b), rational(c))
                expected = tuple(ComplexRoot(root.root, exponent) for root in legacy)
                result = support.run(COMPLEX, polynomial=p)
                self.assertVerified(result, expected, (Q(1), (((rational(c), rational(b), Q(1)), exponent),)))
        p, value, evidence = complex_claim(F(-3, 2), BASE_FACTORS)
        factorization = mf.operations.factor(p).value  # v0.7's public factor.
        self.assertEqual(evidence, (factorization.coefficient, tuple((entry.polynomial.coefficients, entry.multiplicity)
                                                                     for entry in factorization.factors)))
        self.assertEqual(family.produce(COMPLEX, {"polynomial": p}), (value, evidence))
        self.assertNotIn("qq-factorization-v1", repr(evidence))

    def test_larger_inputs_fit_the_default_budget(self):
        """Under the default limits: count_real_roots of a dense polynomial of degree 160 with one-digit coefficients,
        about 560,000 units for the producer's chain, and complex_roots of a polynomial with 24 quadratic and 13 linear
        factors and of a square-free product of 16 quadratics, each verified within a few seconds."""
        rng = random.Random(160)
        coefficients = [rng.randrange(-9, 10) for _ in range(160)] + [rng.randrange(1, 10)]
        p = polynomial(coefficients)
        for lower, upper in (("-inf", "+inf"), (Q(-1, 3), Q(5, 7))):
            with self.subTest(degree=160, lower=lower, upper=upper):
                start = time.perf_counter()
                result = support.run(COUNT, polynomial=p, lower=lower, upper=upper)
                self.assertLess(time.perf_counter() - start, 2 * FAST)  # About 0.5 s, producer and checker.
                self.assertTrue(mf.verify(result).verified, result.error)
                self.assertIs(type(result.value), int)
                self.assertLessEqual(result.value, 160)
        rng = random.Random(20261145)
        factors, keys = [], set()
        while len(factors) < 24:
            c, b = F(rng.randrange(-200, 201), rng.randrange(1, 9)), F(rng.randrange(-60, 61), rng.randrange(1, 5))
            if rational_sqrt(b * b - 4 * c) is None and factor_key((c, b, 1)) not in keys:
                keys.add(factor_key((c, b, 1)))
                factors.append(((c, b, 1), rng.choice((1, 1, 2))))
        factors += [((F(k, 7), 1), 1) for k in range(-6, 7)]
        p, value, evidence = complex_claim(F(-11, 13), factors)
        self.assertGreater(len(p.coefficients), 61)
        start = time.perf_counter()
        result = support.run(COMPLEX, polynomial=p)
        self.assertLess(time.perf_counter() - start, 2 * FAST)
        self.assertVerified(result, value, evidence)
        p, value, evidence = complex_claim(1, [((k, 0, 1), 1) for k in range(1, 17)])
        self.assertEqual(len(p.coefficients), 33)
        start = time.perf_counter()
        self.assertVerified(support.run(COMPLEX, polynomial=p), value, evidence)
        self.assertLess(time.perf_counter() - start, 2 * FAST)

    def test_a_power_of_x_is_charged_no_lifting(self):
        """Regression: the estimate of the uncharged Hensel liftings was taken from the whole polynomial, so x^1024,
        whose factorization costs about 1.1 million units and needs no lifting, was refused with an estimate of
        billions. The liftings run on the square-free parts of x * q for p = x^k q, which the estimate now follows; it
        still refuses at once a hard factor beside a power of x."""
        zero, one = Q(0), Q(1)
        x_factor = (zero, one)
        # p = lead * x^k * q, written out (multiplying x^k out with expand() would take seconds), with its claim.
        cases = [
            (Q(1), 1024, (one,), (record(zero, zero, 1024),), (one, ((x_factor, 1024),))),
            (Q(-7, 3), 1023, (Q(-7, 3),), (record(zero, zero, 1023),), (Q(-7, 3), ((x_factor, 1023),))),
            # x^600 (x^2 + 1): -i, 0 and i share the real part 0.
            (Q(1), 600, (one, zero, one), (record(zero, -one, 1), record(zero, zero, 600), record(zero, one, 1)),
             (one, ((x_factor, 600), ((one, zero, one), 1)))),
            # x^1000 (x^2 - 2): -sqrt 2, 0, sqrt 2.
            (Q(1), 1000, (Q(-2), zero, one),
             (record(RealAlgebraicRoot((-2, 0, 1), 0), zero, 1), record(zero, zero, 1000),
              record(RealAlgebraicRoot((-2, 0, 1), 1), zero, 1)), (one, ((x_factor, 1000), ((Q(-2), zero, one), 1)))),
        ]
        for lead, power, q, value, evidence in cases:
            p = Polynomial((zero,) * power + q, X)
            with self.subTest(degree=len(p.coefficients) - 1):
                self.assertEqual(p.leading_coefficient, lead)
                start = time.perf_counter()
                self.assertVerified(support.run(COMPLEX, polynomial=p), value, evidence)
                self.assertLess(time.perf_counter() - start, 2 * FAST)
        hard = Polynomial((zero,) * 1000 + (Q(2 ** 60000 + 1), zero, one), X)
        start = time.perf_counter()
        self.assertResourceLimit(support.run(COMPLEX, polynomial=hard))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_a_count_within_the_producers_budget_is_verified(self):
        """Regression: the checker ran its own Euclid for the square-free part before the Sturm chain, about four times
        the chain's work, so this dense polynomial of degree 210, which the producer counts in about 1.8 million units
        of the default 2 million, cost the checker 2.3 million, and run() reported resource_limit_exceeded. The
        checker now reads the square-free part from the chain of P (a constant last member: P is square-free), about
        0.5 million units here."""
        rng = random.Random(211)
        p = polynomial([rng.randrange(-9, 10) for _ in range(210)] + [rng.randrange(1, 10)])
        with computation(fresh=True) as budget:
            produced = family.produce(COUNT, dict(polynomial=p, lower="-inf", upper="+inf"))
        self.assertGreater(budget.work, 1_500_000)  # Close to the default max_work: the gap that the fix closes.
        with computation(fresh=True) as budget:  # The default limits, as the verifier's.
            self.assertIs(support.checker(family).check_operation(COUNT, dict(polynomial=p, lower="-inf",
                                                                              upper="+inf"), *produced), True)
        self.assertLess(budget.work, 1_000_000)
        start = time.perf_counter()
        result = support.run(COUNT, polynomial=p)
        self.assertLess(time.perf_counter() - start, 2 * FAST)
        self.assertVerified(result, produced[0], ())
        # A repeated root takes the checker's other path: S = P / G, G from the last member of the chain of P.
        q = [rng.randrange(-9, 10) for _ in range(12)] + [1]
        r = [rng.randrange(-9, 10) for _ in range(20)] + [1]
        repeated = times(times(q, q), r)
        result = support.run(COUNT, polynomial=polynomial(repeated))
        self.assertVerified(result, sturm_count(repeated, None, None), ())

    def test_work_is_charged_before_it_runs(self):
        """Every charge depends on the input alone: a budget one unit short of a request's work refuses it with
        ResourceLimitError, its exact work completes with the same answer, and a larger input costs more."""
        built = Built(F(3), ((F(1, 2), 2), (F(-3), 1)), (F(2), F(5)), ((F(7), F(1)),))
        cases = [
            (COUNT, dict(polynomial=polynomial(built.coefficients()), lower=Q(-5, 2), upper="+inf"), 5),
            (COMPLEX, dict(polynomial=complex_claim(F(5, 3), (((F(1, 2), F(-1, 3), 1), 2), ((F(1, 3), F(1, 2), 1), 1),
                                                             ((2, 0, 1), 1), ((F(-7, 4), 1), 3)))[0]), None),
        ]
        self.assertEqual(built.count(F(-5, 2), None), 5)  # 1/2, -+sqrt 2 and -+sqrt 5, not -3.
        for operation, args, expected in cases:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    produced = family.produce(operation, dict(args))
                work = budget.work
                if expected is not None:
                    self.assertEqual(produced, (expected, ()))
                with computation(ComputationLimits(max_work=work), fresh=True):
                    self.assertEqual(family.produce(operation, dict(args)), produced)
                with computation(ComputationLimits(max_work=work - 1), fresh=True):
                    with self.assertRaises(ResourceLimitError):
                        family.produce(operation, dict(args))
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=work - 1), **args))
                self.assertGreater(work, 20)
                with computation(fresh=True) as small:
                    family.produce(operation, dict(args, polynomial=polynomial([-2, 0, 1])))
                self.assertLess(small.work, work)
        with computation(fresh=True) as constant:
            self.assertEqual(family.produce(COUNT, dict(polynomial=polynomial([5]), lower="-inf", upper="+inf")),
                             (0, ()))
        self.assertGreaterEqual(constant.work, 2)  # The call and the coefficient: max_work=1 refuses every input.


def _request(p: Polynomial) -> mf.OperationRequest:
    """The request that operations_v6.run builds for complex_roots of p, which an Unevaluated record carries."""
    return mf.OperationRequest(COMPLEX, (("polynomial", p),), 6)
