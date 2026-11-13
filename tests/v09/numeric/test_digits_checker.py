"""Tests of the independent checker of mathforge.areas.numeric.digits (mathforge.certify.numeric.digits) alone,
written before the family's producer exists: every claim is built here, and the checker answers it as the verifier
asks it (support.verdict, in a fresh budget of the default limits unless a test passes others).

The true claims come from references of this module, never from a producer: integer division of exact Fractions for
rationals (the card's definition), integer square roots of closed forms for square-root expressions
(floor(sqrt(y)) = isqrt(floor(y)) for real y >= 0), Decimal arithmetic at places + 80 digits as an independent
confirmation of every expression, and, for real algebraic roots, exact Fraction signs on a grid of step 1/64 that
isolate each simple root, then integer bisection (no Sturm chain). The evidence is the card's layout: () for a
rational, ("root",) for a RealAlgebraicRoot, and for an expression ("bits", b) with the generous b = 4*places + 128 or
("rational", r).

The checker answers False to every corrupted claim, never with an exception, and refuses oversized forged values and
evidence within FAST seconds under the default budget. ResourceLimitError comes only where the checker's docstring
says so: from an argument beyond the budget (an integer longer than max_integer_bits, an expression deeper than
max_depth or whose values pass 2**max_integer_bits) and from an expression with more than MAX_RADICALS independent
square roots under ("rational", r).
"""

from __future__ import annotations

import ast
import dataclasses
from decimal import Decimal, localcontext
import enum
from fractions import Fraction
import inspect
from math import ceil, floor, isqrt
import random
import time
from types import MappingProxyType

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.numeric import digits as family
from mathforge.areas.numeric.digits import TruncatedDecimal
from mathforge.certify.numeric import digits as checker
from mathforge.contracts import CheckStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Sqrt
from mathforge.registry import Argument, OperationSpec

from .. import support

Q = mf.Rational
OP = "digits"
FAST = 3.0  # Seconds: an oversized forgery is refused, and a full-size claim verified, well within this.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
         "and decimal digits")
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
ROOT = ("root",)
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def bits_for(places: int) -> int:
    """A precision that every enclosed expression of these tests needs at most: 4 bits per decimal place (log2(10) is
    about 3.32) and 128 more for the size and the depth of the expression."""
    return 4 * places + 128


def bits(places: int) -> tuple[str, int]:
    return ("bits", bits_for(places))


def sqrt(value: object) -> Sqrt:
    """A Sqrt node of a rational, never simplified (mf.sqrt would return a Rational for a square)."""
    value = Fraction(value)
    return Sqrt(Q(value.numerator, value.denominator))


def rational(value: object) -> Q:
    value = Fraction(value)
    return Q(value.numerator, value.denominator)


def exact(value: object) -> int | Q:
    """An exact rational argument as a request may hold it: an int for an integer, a Rational otherwise."""
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else Q(value.numerator, value.denominator)


def args_of(value: object, places: object) -> dict[str, object]:
    return {"value": value, "places": places}


def record(sign: int, magnitude: int, places: int, is_exact: bool) -> TruncatedDecimal:
    return TruncatedDecimal(sign, magnitude, places, is_exact)


def forged(sign: object, magnitude: object, places: object, is_exact: object) -> TruncatedDecimal:
    """A record made with object.__new__, whatever its fields: what no decoded document can hold."""
    return support.forged(TruncatedDecimal, sign=sign, magnitude=magnitude, places=places, exact=is_exact)


def claim(triple: tuple[int, int, bool], places: int) -> TruncatedDecimal:
    sign, magnitude, is_exact = triple
    return TruncatedDecimal(sign, magnitude, places, is_exact)


def truncated(x: object, places: int) -> tuple[int, int, bool]:
    """(sign, magnitude, exact) of an exact rational by integer division: the card's definition."""
    x = Fraction(x)
    scaled = abs(x) * 10 ** places
    return (x > 0) - (x < 0), scaled.numerator // scaled.denominator, scaled.denominator == 1


def sqrt_floor(n: object, places: int) -> int:
    """floor(sqrt(n) * 10**places) for a rational n >= 0, as isqrt(floor(n * 10**(2*places)))."""
    scaled = Fraction(n) * 10 ** (2 * places)
    return isqrt(scaled.numerator // scaled.denominator)


def icbrt(n: int) -> int:
    """floor(n ** (1/3)) for an int n >= 0, by Newton's method on integers."""
    if n < 2:
        return n
    x = 1 << -(-n.bit_length() // 3)  # At least the cube root.
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            break
        x = y
    while x * x * x > n:
        x -= 1
    while (x + 1) * (x + 1) * (x + 1) <= n:
        x += 1
    return x


def iroot(n: int, k: int) -> int:
    """floor(n ** (1/k)) for an int n >= 0 and k >= 1, by Newton's method on integers from above."""
    if n < 2 or k == 1:
        return n
    x = 1 << -(-n.bit_length() // k)  # At least the k-th root.
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    while x ** k > n:
        x -= 1
    while (x + 1) ** k <= n:
        x += 1
    return x


def decimal_value(node: object, precision: int) -> Decimal:
    """The value of a closed expression of Rational, Sqrt, Add, Mul and Pow nodes in Decimal arithmetic with that many
    significant digits: an approximation, independent of every exact method of the checker."""
    with localcontext() as context:
        context.prec = precision

        def visit(item: object) -> Decimal:
            kind = type(item)
            if kind is Q:
                return Decimal(item.numerator) / Decimal(item.denominator)
            if kind is Sqrt:
                return visit(item.radicand).sqrt()
            if kind is Add:
                total = Decimal(0)
                for term in item.terms:
                    total += visit(term)
                return total
            if kind is Mul:
                total = Decimal(1)
                for factor in item.factors:
                    total *= visit(factor)
                return total
            if kind is Pow:
                return visit(item.base) ** item.exponent
            raise TypeError(kind)

        return +visit(node)


def decimal_truncated(node: object, places: int) -> tuple[int, int, bool]:
    """(sign, magnitude, False) of an irrational expression from decimal_value at places + 80 digits; the scaled
    value must lie farther than 10**-40 from every integer, so that the approximation decides the digits."""
    value = decimal_value(node, places + 80)
    with localcontext() as context:
        context.prec = places + 80
        scaled = abs(value).scaleb(places)
        magnitude = int(scaled)  # int() of a Decimal truncates toward zero.
        rest = scaled - magnitude
        assert Decimal(10) ** -40 < rest < 1 - Decimal(10) ** -40, (node, places, rest)
    return (value > 0) - (value < 0), magnitude, False


def poly_at(coefficients: tuple[int, ...], t: Fraction) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * t + coefficient
    return value


def isolated_roots(coefficients: tuple[int, ...]) -> list[tuple[Fraction, Fraction]]:
    """The real roots of an integer polynomial whose real roots are simple and more than 1/64 apart, in increasing
    order, each as (a, b): a == b for a root on the grid of step 1/64, else grid neighbours a < root < b at which the
    polynomial has opposite signs. Exact signs on the grid alone, within the Cauchy bound: no Sturm chain."""
    bound = 2 + max(abs(c) for c in coefficients[:-1]) // abs(coefficients[-1])  # Above 1 + max|c_i|/|c_n|.
    grid = [Fraction(j, 64) for j in range(-64 * bound, 64 * bound + 1)]
    values = [poly_at(coefficients, t) for t in grid]
    roots = []
    for j, (t, value) in enumerate(zip(grid, values)):
        if value == 0:
            roots.append((t, t))
        elif j + 1 < len(grid) and value * values[j + 1] < 0:
            roots.append((t, grid[j + 1]))
    return roots


def root_truncated(coefficients: tuple[int, ...], index: int, places: int) -> tuple[int, int, bool]:
    """(sign, magnitude, exact) of the index-th real root of the polynomial, by bisection over the integers m with the
    sign of the polynomial at m/10**places inside the root's grid interval."""
    a, b = isolated_roots(coefficients)[index]
    if a == b:
        return truncated(a, places)
    scale = 10 ** places
    left = poly_at(coefficients, a) > 0
    low, high = floor(a * scale), ceil(b * scale)  # low/scale <= a < root < b <= high/scale.
    while high - low > 1:
        middle = (low + high) // 2
        u = Fraction(middle, scale)
        if u <= a:
            low = middle
            continue
        if u >= b:
            high = middle
            continue
        value = poly_at(coefficients, u)
        if value == 0:
            low = high = middle
            break
        if (value > 0) == left:
            low = middle
        else:
            high = middle
    on_point = poly_at(coefficients, Fraction(low, scale)) == 0
    if a >= 0:  # The root lies above a >= 0; otherwise below b <= 0, as 0 is a point of the grid.
        return 1, low, on_point
    return -1, -low if on_point else -low - 1, on_point


def claimed(args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(OP, tuple(args.items()), 6)
    step = contracts.Step("v6." + OP, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=OP, request=request, steps=(step,))


# The real cube root of 2, and sqrt(2)*sqrt(2)/2 as a tree that no normalizer touched (the parser makes it 1).
CUBE_ROOT_2 = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
HALF_TWO = Mul((sqrt(2), sqrt(2), Q(1, 2)))

# The known values of the card: (value, places, (sign, magnitude, exact), evidence, text).
KNOWN = (
    (Q(2, 3), 5, (1, 66666, False), (), "0.66666..."),
    (Q(1, 4), 3, (1, 250, True), (), "0.250"),
    (Q(-1, 8), 2, (-1, 12, False), (), "-0.12..."),
    (0, 4, (0, 0, True), (), "0.0000"),
    (sqrt(2), 30, (1, 1414213562373095048801688724209, False), bits(30), "1.414213562373095048801688724209..."),
    (CUBE_ROOT_2, 20, (1, 125992104989487316476, False), ROOT, "1.25992104989487316476..."),
    (HALF_TWO, 3, (1, 1000, True), ("rational", Q(1)), "1.000"),
)


def minus(node: object) -> Mul:
    return Mul((Q(-1), node))


# Irrational expressions with closed-form references: (name, expression, places -> magnitude, sign).
ENCLOSED = (
    ("sqrt(2)", sqrt(2), lambda k: sqrt_floor(2, k), 1),
    ("-sqrt(2)", minus(sqrt(2)), lambda k: sqrt_floor(2, k), -1),
    ("1 - sqrt(2)", Add((Q(1), minus(sqrt(2)))), lambda k: sqrt_floor(2, k) - 10 ** k, -1),
    ("(1 + sqrt(2))^3 = 7 + 5*sqrt(2)", Pow(Add((Q(1), sqrt(2))), 3), lambda k: 7 * 10 ** k + sqrt_floor(50, k), 1),
    ("sqrt(2) + sqrt(3) = sqrt(5 + 2*sqrt(6))", Add((sqrt(2), sqrt(3))),
     lambda k: isqrt(5 * 10 ** (2 * k) + isqrt(24 * 10 ** (4 * k))), 1),
    ("sqrt(2) - sqrt(3) = -sqrt(5 - 2*sqrt(6))", Add((sqrt(2), minus(sqrt(3)))),
     lambda k: isqrt(5 * 10 ** (2 * k) - isqrt(24 * 10 ** (4 * k)) - 1), -1),
    ("sqrt(1/2)", sqrt(Fraction(1, 2)), lambda k: sqrt_floor(Fraction(1, 2), k), 1),
    ("sqrt(3)^5 = sqrt(243)", Pow(sqrt(3), 5), lambda k: sqrt_floor(243, k), 1),
    ("3/7*sqrt(5)*sqrt(7) = sqrt(315/49)", Mul((Q(3, 7), sqrt(5), sqrt(7))), lambda k: sqrt_floor(Fraction(315, 49), k),
     1),
    ("-sqrt(2/9)*sqrt(3) = -sqrt(2/3)", Mul((Q(-1), sqrt(Fraction(2, 9)), sqrt(3))),
     lambda k: sqrt_floor(Fraction(2, 3), k), -1),
)

# Expressions with a rational value: (name, expression, value).
RATIONAL_VALUED = (
    ("sqrt(2)*sqrt(2)/2", HALF_TWO, Fraction(1)),
    ("sqrt(8) - 2*sqrt(2)", Add((sqrt(8), Mul((Q(-2), sqrt(2))))), Fraction(0)),
    ("Sqrt(4)", sqrt(4), Fraction(2)),
    ("sqrt(6) - sqrt(2)*sqrt(3)", Add((sqrt(6), Mul((Q(-1), sqrt(2), sqrt(3))))), Fraction(0)),
    ("sqrt(12) - 2*sqrt(3), with the square 4 in the base", Add((sqrt(12), Mul((Q(-2), sqrt(3))))), Fraction(0)),
    ("(sqrt(2) + sqrt(3))^2 - 2*sqrt(6)", Add((Pow(Add((sqrt(2), sqrt(3))), 2), Mul((Q(-2), sqrt(6))))), Fraction(5)),
    ("sqrt(1/2)*sqrt(2)", Mul((sqrt(Fraction(1, 2)), sqrt(2))), Fraction(1)),
    ("(1 + sqrt(2))*(1 - sqrt(2))", Mul((Add((Q(1), sqrt(2))), Add((Q(1), minus(sqrt(2)))))), Fraction(-1)),
    ("sqrt(2)^0", Pow(sqrt(2), 0), Fraction(1)),
    ("sqrt(2)^2", Pow(sqrt(2), 2), Fraction(2)),
    ("sqrt(2)^200", Pow(sqrt(2), 200), Fraction(2 ** 100)),
    ("sqrt(2)*sqrt(2)/3", Mul((sqrt(2), sqrt(2), Q(1, 3))), Fraction(2, 3)),
    ("sqrt(72)*sqrt(2)", Mul((sqrt(72), sqrt(2))), Fraction(12)),
    ("sqrt(18) - 3*sqrt(2)", Add((sqrt(18), Mul((Q(-3), sqrt(2))))), Fraction(0)),
    ("sqrt(8/9) - 2/3*sqrt(2)", Add((sqrt(Fraction(8, 9)), Mul((Q(-2, 3), sqrt(2))))), Fraction(0)),
    ("sqrt(0) + 3/7", Add((sqrt(0), Q(3, 7))), Fraction(3, 7)),
    ("ten coprime radicands, the most", Add(tuple(Mul((sqrt(p), sqrt(p))) for p in PRIMES[:10])),
     Fraction(sum(PRIMES[:10]))),
)


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values, arguments and evidence that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Sign(enum.IntEnum):
    """The signs as an int enumeration: equal to them but no exact ints."""
    MINUS = -1
    ZERO = 0
    PLUS = 1


class SubRational(Q):
    """A Rational subclass: equal to the right number but no Rational of the model."""


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass, for an evidence tuple.
FakeRational = dataclasses.make_dataclass("Rational", ["numerator", "denominator"], frozen=True)  # Another module.


def forged_rational(numerator: object, denominator: object = 1, *, missing: str = "") -> Q:
    """A Rational made with object.__new__: its fields may break lowest terms, a positive denominator or exactness,
    and the field named by missing stays unset."""
    value = object.__new__(Q)
    for name, item in (("numerator", numerator), ("denominator", denominator)):
        if name != missing:
            object.__setattr__(value, name, item)
    return value


def forged_node(cls: type, **fields: object) -> object:
    """An expression node made with object.__new__, outside the form of its constructor."""
    return support.forged(cls, **fields)


def label(item: object) -> str:
    """repr(item) for a subtest's label; a record made with object.__new__ without a field has no repr, and an int
    past Python's limit of digits for str() has none either."""
    try:
        text = repr(item)
    except AttributeError:
        return f"<{type(item).__name__} without a field>"
    except ValueError:
        return f"<{type(item).__name__} with an integer of more than 4,300 digits>"
    return text if len(text) <= 80 else text[:77] + "..."


class DigitsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, cases: list[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (name, args, value, evidence) case."""
        for name, args, value, evidence in cases:
            with self.subTest(case=name):
                self.assertRejects(OP, args, value, evidence)

    def assertRejectsFast(self, args: object, value: object, evidence: object = ()) -> None:
        started = time.perf_counter()
        self.assertRejects(OP, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    def assertAcceptsFast(self, args: object, value: object, evidence: object = ()) -> None:
        started = time.perf_counter()
        self.assertAccepts(OP, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    def assertExhausts(self, args: object, value: object, evidence: object = (), *,
                       limits: ComputationLimits | None = None) -> None:
        """The check raises ResourceLimitError at once: the claim is beyond the budget, not wrong."""
        started = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, args, value, evidence, limits=limits)
        self.assertLess(time.perf_counter() - started, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_card(self):
        expected = OperationSpec(
            "digits", "numeric",
            (Argument("value", "Expression | int", None, "scalar"), Argument("places", "int", None, "int")),
            "TruncatedDecimal",
            "Write the decimal expansion of an exact real number truncated toward zero to a number of places.",
            SCOPE, version=6)
        self.assertEqual(family.SPECS, (expected,))
        self.assertEqual(family.TYPES, {"truncated_decimal": TruncatedDecimal})
        self.assertEqual(family.CHECKER, "mathforge.certify.numeric.digits")
        self.assertEqual(set(family.RENDER), {TruncatedDecimal})
        self.assertEqual([field.name for field in dataclasses.fields(TruncatedDecimal)],
                         ["sign", "magnitude", "places", "exact"])
        self.assertEqual(registry.spec(OP), expected)
        self.assertIs(registry.checker(OP), checker)
        self.assertIs(support.checker(family), checker)
        self.assertEqual(checker.RECORD, (TruncatedDecimal.__module__, TruncatedDecimal.__qualname__))
        self.assertEqual(checker.ROOT, (mf.RealAlgebraicRoot.__module__, mf.RealAlgebraicRoot.__qualname__))
        self.assertEqual((checker.MAX_PLACES, family.MAX_PLACES), (10_000, 10_000))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_accepts_the_known_values_of_the_card(self):
        references = {
            0: truncated(Fraction(2, 3), 5), 1: truncated(Fraction(1, 4), 3), 2: truncated(Fraction(-1, 8), 2),
            3: truncated(0, 4), 4: (1, sqrt_floor(2, 30), False),
            5: root_truncated(CUBE_ROOT_2.integer_coefficients, 0, 20), 6: truncated(1, 3),
        }
        self.assertEqual(references[4], decimal_truncated(sqrt(2), 30))
        self.assertEqual(references[5][1], icbrt(2 * 10 ** 60))
        for position, (value, places, triple, evidence, text) in enumerate(KNOWN):
            with self.subTest(text=text):
                self.assertEqual(references[position], triple)
                self.assertAccepts(OP, args_of(value, places), claim(triple, places), evidence)
                self.assertEqual(mf.render_text(claim(triple, places)), text)
        self.assertLess(abs(decimal_value(HALF_TWO, 50) - 1), Decimal(10) ** -45)  # 1, up to Decimal's rounding.

    def test_accepts_every_small_rational(self):
        """Every p/q with |p| <= 24 and 1 <= q <= 10 at 0 to 4 places, against the card's definition by integer
        division; the neighbours of each claim are refused: the magnitude one more or one less, the other exactness,
        another sign and other places."""
        accepted = exacts = 0
        for p in range(-24, 25):
            for q in range(1, 11):
                x = Fraction(p, q)
                for places in range(5):
                    sign, magnitude, is_exact = truncated(x, places)
                    args = args_of(exact(x), places)
                    with self.subTest(x=str(x), places=places):
                        self.assertAccepts(OP, args, record(sign, magnitude, places, is_exact))
                        self.assertRejects(OP, args, forged(sign or 1, magnitude + 1, places, False))
                        if magnitude:
                            self.assertRejects(OP, args, forged(sign, magnitude - 1, places, False))
                        if sign:
                            self.assertRejects(OP, args, forged(sign, magnitude, places, not is_exact))
                            self.assertRejects(OP, args, forged(-sign, magnitude, places, is_exact))
                        self.assertRejects(OP, args, forged(sign, magnitude, places + 1, is_exact))
                    accepted += 1
                    exacts += is_exact
        self.assertEqual(accepted, 49 * 10 * 5)
        self.assertGreater(exacts, 500)

    def test_accepts_rational_special_cases(self):
        """Integers, places 0, numbers below one unit of the last place (magnitude 0 with a sign), terminating
        decimals at, below and beyond their length, Rationals of denominator 1, and long numerators and denominators."""
        cases = (
            (5, 3), (-7, 0), (0, 0), (Fraction(5, 2), 0), (Fraction(-5, 2), 0), (Fraction(-1, 8), 0),
            (Fraction(1, 8), 0), (Fraction(1, 8), 1), (Fraction(1, 8), 2), (Fraction(1, 8), 3), (Fraction(1, 8), 9),
            (Fraction(3, 40), 3), (Fraction(-3, 40), 2), (Fraction(1, 3), 10), (Fraction(-2, 3), 1),
            (Fraction(10 ** 30 + 1, 10 ** 30), 29), (Fraction(10 ** 30 + 1, 10 ** 30), 30),
            (Fraction(-1, 10 ** 50), 49), (Fraction(-1, 10 ** 50), 50), (Fraction(7, 3 * 2 ** 30), 12),
        )
        for x, places in cases:
            with self.subTest(x=str(x), places=places):
                self.assertAccepts(OP, args_of(exact(x), places), claim(truncated(x, places), places))
        self.assertEqual(truncated(Fraction(-1, 8), 0), (-1, 0, False))  # "-0...": a sign with magnitude 0.
        self.assertAccepts(OP, args_of(Q(6), 2), record(1, 600, 2, True))  # A Rational of denominator 1.
        self.assertAccepts(OP, args_of(Q(-6, 1), 0), record(-1, 6, 0, True))
        # Long integers: a magnitude of 33,674 bits at 10,000 places, and one just within max_integer_bits.
        x = Fraction(3 ** 18_000, 7 ** 10_000)
        self.assertAcceptsFast(args_of(exact(x), 10_000), claim(truncated(x, 10_000), 10_000))
        x = Fraction(2 ** 65_000 + 1)
        triple = truncated(x, 100)
        self.assertLessEqual(triple[1].bit_length(), 65_536)
        self.assertAcceptsFast(args_of(exact(x), 100), claim(triple, 100))

    def test_accepts_real_algebraic_roots(self):
        """Every real root of polynomials with negative, zero, rational, decimal and irrational roots, at several
        places, against the grid-and-bisection reference: the cube roots of 2 and 1/2, both roots of x^2 - 2, x^3 - x
        (-1, 0, 1), 4x^2 - 1 (-1/2, 1/2), 3x - 1, 10x - 7, (x^2 - 2)(x^2 - 3), x^5 - x - 1 and 500000x^2 - 1 (the
        canonical form of 10^6 x^2 - 2: +-sqrt(2)/1000)."""
        polynomials = (
            ((-2, 0, 0, 1), range(0, 26)), ((-1, 0, 0, 2), range(0, 12)), ((-2, 0, 1), range(0, 13)),
            ((0, -1, 0, 1), range(0, 4)), ((-1, 0, 4), range(0, 4)), ((-1, 3), range(0, 6)), ((-7, 10), range(0, 4)),
            ((6, 0, -5, 0, 1), range(0, 7)), ((-1, -1, 0, 0, 0, 1), range(0, 16)), ((-1, 0, 500_000), range(0, 9)),
        )
        signs = set()
        on_points = 0
        for coefficients, places_range in polynomials:
            count = len(isolated_roots(coefficients))
            for index in range(count):
                root = mf.RealAlgebraicRoot(coefficients, index)
                self.assertEqual(root.integer_coefficients, coefficients)  # Already canonical.
                for places in places_range:
                    triple = root_truncated(coefficients, index, places)
                    with self.subTest(polynomial=coefficients, index=index, places=places):
                        self.assertAccepts(OP, args_of(root, places), claim(triple, places), ROOT)
                        if triple[0]:
                            self.assertRejects(OP, args_of(root, places),
                                               forged(triple[0], triple[1] + 1, places, False), ROOT)
                    signs.add(triple[0])
                    on_points += triple[2]
        self.assertEqual(signs, {-1, 0, 1})
        self.assertGreater(on_points, 20)
        # The reference agrees with closed forms where there is one.
        self.assertEqual(root_truncated((-2, 0, 0, 1), 0, 25)[1], icbrt(2 * 10 ** 75))
        self.assertEqual(root_truncated((-2, 0, 1), 0, 12), (-1, sqrt_floor(2, 12), False))
        self.assertEqual(root_truncated((-1, 0, 4), 0, 1), (-1, 5, True))
        # The cube root of 2 to 10,000 places, within FAST seconds.
        self.assertAcceptsFast(args_of(CUBE_ROOT_2, 10_000), record(1, icbrt(2 * 10 ** 30_000), 10_000, False), ROOT)

    def test_accepts_expressions_under_enclosures(self):
        """Irrational expressions with ("bits", 4*places + 128): closed-form references, each confirmed by Decimal
        arithmetic; negative values, radicands with denominators, powers, products and nested sums."""
        for name, node, magnitude, sign in ENCLOSED:
            for places in (0, 1, 2, 5, 17, 40):
                triple = (sign, magnitude(places), False)
                with self.subTest(name=name, places=places):
                    self.assertEqual(decimal_truncated(node, places), triple)
                    self.assertAccepts(OP, args_of(node, places), claim(triple, places), bits(places))
                    self.assertRejects(OP, args_of(node, places), forged(sign, triple[1] + 1, places, False),
                                       bits(places))
        # A number far below the first places: sqrt(2) - 1.4142135623730950488 is about 1.6887e-21.
        tiny = Add((sqrt(2), Q(-14142135623730950488, 10 ** 19)))
        for places in (0, 19, 20, 21, 25, 60):
            triple = decimal_truncated(tiny, places)
            with self.subTest(name="tiny", places=places):
                self.assertEqual(triple[0], 1)
                self.assertAccepts(OP, args_of(tiny, places), claim(triple, places), bits(places))
        self.assertEqual(decimal_truncated(tiny, 20), (1, 0, False))
        self.assertEqual(decimal_truncated(tiny, 25), (1, 16887, False))
        self.assertEqual(decimal_truncated(tiny, 25)[1], sqrt_floor(2, 25) - 14142135623730950488 * 10 ** 6)
        # Shared nodes are one node: sqrt(2) doubled forty times over is 2**40 * sqrt(2) = sqrt(2**81).
        doubled = sqrt(2)
        for _ in range(40):
            doubled = Add((doubled, doubled))
        for places in (0, 7, 30):
            self.assertAccepts(OP, args_of(doubled, places), record(1, sqrt_floor(2 ** 81, places), places, False),
                               bits(places))

    def test_accepts_any_sufficient_precision(self):
        """Evidence that a producer need not choose: every sufficient b passes, up to max_integer_bits; a b too small
        to separate the cell is refused, the claim being true. A rational expression off the decimal boundaries
        passes with an enclosure as well as with its value."""
        args, value = args_of(sqrt(2), 30), record(1, sqrt_floor(2, 30), 30, False)
        for b in (110, 128, 200, 1_000, 65_536):
            with self.subTest(b=b):
                self.assertAccepts(OP, args, value, ("bits", b))
        for b in (1, 8, 32, 64, 90):
            with self.subTest(b=b):
                self.assertRejects(OP, args, value, ("bits", b))
        third = Mul((sqrt(2), sqrt(2), Q(1, 3)))  # 2/3
        for places in (0, 3, 12):
            value = claim(truncated(Fraction(2, 3), places), places)
            self.assertAccepts(OP, args_of(third, places), value, ("rational", Q(2, 3)))
            self.assertAccepts(OP, args_of(third, places), value, bits(places))
        limits = ComputationLimits(max_integer_bits=4096)  # The ceiling of b follows the verifying budget.
        args, value = args_of(sqrt(2), 1000), record(1, sqrt_floor(2, 1000), 1000, False)
        self.assertAccepts(OP, args, value, ("bits", 4096), limits=limits)
        self.assertRejects(OP, args, value, ("bits", 4097), limits=limits)

    def test_accepts_expressions_with_rational_evidence(self):
        """Rational-valued expressions with ("rational", r): on decimal boundaries (exact), at 0, negative, through
        radicands that the coprime base refines (6 with 2 and 3, 12 with 3, 72 with 2, 8/9 with 2), and the ten
        independent square roots that MAX_RADICALS allows."""
        for name, node, value in RATIONAL_VALUED:
            for places in (0, 1, 3, 8):
                triple = truncated(value, places)
                with self.subTest(name=name, places=places):
                    self.assertEqual(Fraction(decimal_value(node, 60)).limit_denominator(10 ** 6), value)
                    self.assertAccepts(OP, args_of(node, places), claim(triple, places), ("rational", rational(value)))
                    self.assertRejects(OP, args_of(node, places), forged(triple[0] or 1, triple[1] + 1, places, False),
                                       ("rational", rational(value)))
        self.assertEqual(checker.MAX_RADICALS, 10)

    def test_accepts_radicands_that_share_high_powers(self):
        """Radicands that share a high power of a small number refine into their coprime base in one split each, so
        the ("rational", r) layout verifies them under the default budget: 2**65001 against 2 (sqrt(2**65001) is
        2**32500 * sqrt(2)), 2**9001 against 2 at 0, 2 * 6**1000 against 2, where the base member 3**1000 is a
        square, and ten radicands 2**65530 * p against the ten primes p: the base member 2**65530 is a square, the ten
        primes span 2**10 coordinates, and a sum of elements with one nonzero coordinate each stays cheap. An
        irrational value with such a radicand passes with an enclosure, its reference from isqrt."""
        primes = PRIMES[1:]  # 3 to 31: 2**65530 * 31 still fits max_integer_bits.
        cases = (
            (Mul((sqrt(2 ** 65001), sqrt(2))), 0, Fraction(2 ** 32501)),
            (Add((sqrt(2 ** 9001), Mul((Q(-(2 ** 4500)), sqrt(2))))), 5, Fraction(0)),
            (Mul((sqrt(2 * 6 ** 1000), sqrt(2))), 3, Fraction(2 * 6 ** 500)),
            (Add(tuple(Mul((sqrt(2 ** 65530 * p), sqrt(p))) for p in primes)), 0, Fraction(2 ** 32765 * sum(primes))),
        )
        for node, places, value in cases:
            with self.subTest(value=label(value), places=places):
                self.assertAcceptsFast(args_of(node, places), claim(truncated(value, places), places),
                                       ("rational", rational(value)))
                self.assertRejectsFast(args_of(node, places), claim(truncated(value + 1, places), places),
                                       ("rational", rational(value + 1)))
        magnitude = sqrt_floor(2 ** 65001, 0)
        fraction = isqrt(2 ** 65001 << 400) - (magnitude << 200)  # The fractional part of sqrt(2**65001), times 2**200.
        self.assertTrue(1 << 100 < fraction < (1 << 200) - (1 << 100))  # Far from both ends of the cell.
        self.assertAcceptsFast(args_of(sqrt(2 ** 65001), 0), record(1, magnitude, 0, False), ("bits", 128))

    def test_roots_through_isolating_cells(self):
        """Regression (review of wave m2b): from COARSE_PLACES places on, a root is decided by Sturm counts on a
        coarse decimal cell that isolates it and two signs at the claimed cell's ends. Every true claim passes, on the
        isolating route and on the fallback to the counts at the claimed cell, where two roots lie within 10**-40 of
        each other; every neighbouring claim, and the claim of another root, is refused. Before, the root of
        x^12 - 3 at 10,000 places needed more than the default budget for the counts at the claimed cell alone."""
        self.assertEqual(checker.COARSE_PLACES, 32)
        m = isqrt(2 * 10 ** 80)  # m/10**40 lies below sqrt(2) by less than 10**-40.
        cases = (
            ((2, -4, -1, 2), 32), ((2, -4, -1, 2), 100), ((2, -4, -1, 2), 257),  # -sqrt(2), 1/2, sqrt(2).
            ((2 * m, -2 * 10 ** 40, -m, 10 ** 40), 60), ((2 * m, -2 * 10 ** 40, -m, 10 ** 40), 33),
        )
        for coefficients, places in cases:
            claims = []
            for index in range(3):
                root = mf.RealAlgebraicRoot(coefficients, index)
                x = (Fraction(m, 10 ** 40) if coefficients[0] == 2 * m else Fraction(1, 2)) if index == 1 else None
                if x is not None:
                    triple = truncated(x, places)
                else:
                    triple = (index - 1, sqrt_floor(2, places), False)
                claims.append((root, triple))
            for root, triple in claims:
                with self.subTest(polynomial=coefficients[-1], index=root.real_index, places=places):
                    self.assertAccepts(OP, args_of(root, places), claim(triple, places), ROOT)
                    sign, magnitude, is_exact = triple
                    wrong = [(sign, magnitude + 1, is_exact), (sign, magnitude - 1, is_exact),
                             (-sign, magnitude, is_exact), (sign, magnitude, not is_exact),
                             (sign, magnitude + 1, not is_exact)]
                    wrong += [other for _, other in claims if other != triple]
                    for candidate in wrong:
                        if candidate[1] >= 0 and (candidate[0] == 0) == (candidate[1] == 0 and candidate[2]):
                            self.assertRejects(OP, args_of(root, places), claim(candidate, places), ROOT)
        # Degree 12 at 10,000 places within the default budget.
        root = mf.RealAlgebraicRoot((-3,) + (0,) * 11 + (1,), 0)
        magnitude = iroot(3 * 10 ** 120_000, 12)
        self.assertAcceptsFast(args_of(root, 10_000), record(-1, magnitude, 10_000, False), ROOT)
        self.assertRejectsFast(args_of(root, 10_000), forged(-1, magnitude + 1, 10_000, False), ROOT)
        self.assertRejectsFast(args_of(root, 10_000), forged(-1, magnitude - 1, 10_000, False), ROOT)
        # A rational root on a decimal point, 1/2 of (2x - 1)(x^10 - 3), is an end of every coarse cell: the counts
        # there show that it is alpha_i itself, and the Rational conditions decide, also at 10,000 places.
        half = mf.RealAlgebraicRoot((3, -6) + (0,) * 8 + (-1, 2), 1)
        self.assertAcceptsFast(args_of(half, 10_000), record(1, 5 * 10 ** 9_999, 10_000, True), ROOT)
        for wrong in ((1, 5 * 10 ** 9_999, False), (1, 5 * 10 ** 9_999 - 1, False), (1, 5 * 10 ** 9_999 + 1, True)):
            self.assertRejectsFast(args_of(half, 10_000), forged(wrong[0], wrong[1], 10_000, wrong[2]), ROOT)

    def test_field_evaluation_is_sparse(self):
        """Regression (review of wave m2b): ("rational", r) evaluates the expression on the nonzero coordinates of the
        algebra only. Before, every node cost 2**t units for t independent square roots (1,024 for ten), and an
        expression of 500 square roots over ten primes ran out of the default budget."""
        primes = PRIMES[:10]
        sums = Add(tuple(Mul((sqrt(p), sqrt(p))) for _ in range(50) for p in primes))  # 50 * 129 = 6450.
        self.assertAcceptsFast(args_of(sums, 0), record(1, 6450, 0, True), ("rational", Q(6450)))
        self.assertRejectsFast(args_of(sums, 0), forged(1, 6449, 0, True), ("rational", Q(6449)))
        cancel = Add(tuple(term for _ in range(25) for p in primes for term in (sqrt(p), minus(sqrt(p)))))
        self.assertAcceptsFast(args_of(cancel, 3), record(0, 0, 3, True), ("rational", Q(0)))
        dense = Add((Pow(Add(tuple(sqrt(p) for p in primes)), 4), minus(Pow(Add(tuple(sqrt(p) for p in primes)), 4)),
                     Q(1, 8)))
        self.assertAcceptsFast(args_of(dense, 3), record(1, 125, 3, True), ("rational", Q(1, 8)))
        self.assertRejectsFast(args_of(dense, 3), forged(1, 125, 3, True), ("rational", Q(1, 7)))
        # Not rational: the terms with sqrt(29) survive.
        almost = Add((Pow(Add(tuple(sqrt(p) for p in primes)), 4),
                      minus(Pow(Add(tuple(sqrt(p) for p in primes[:9])), 4)), Q(1, 8)))
        self.assertRejectsFast(args_of(almost, 3), forged(1, 125, 3, True), ("rational", Q(1, 8)))

    def test_full_size_claims_fit_the_default_budget(self):
        """10,000 places, the card's ceiling, for every kind of value and evidence: each claim within FAST seconds."""
        k = 10_000
        self.assertAcceptsFast(args_of(Q(1, 7), k), record(1, 10 ** k // 7, k, False))
        self.assertAcceptsFast(args_of(sqrt(2), k), record(1, sqrt_floor(2, k), k, False), bits(k))
        self.assertAcceptsFast(args_of(sqrt(2), k), record(1, sqrt_floor(2, k), k, False), ("bits", 65_536))
        self.assertAcceptsFast(args_of(Pow(Add((Q(1), sqrt(2))), 3), k),
                               record(1, 7 * 10 ** k + sqrt_floor(50, k), k, False), bits(k))
        self.assertAcceptsFast(args_of(HALF_TWO, k), record(1, 10 ** k, k, True), ("rational", Q(1)))
        self.assertAcceptsFast(args_of(CUBE_ROOT_2, k), record(1, icbrt(2 * 10 ** (3 * k)), k, False), ROOT)
        self.assertRejectsFast(args_of(sqrt(2), k), forged(1, sqrt_floor(2, k) + 1, k, False), bits(k))

    def test_accepts_seeded_random_claims(self):
        """150 seeded claims: random rationals at up to 60 places, and random a + b*sqrt(n)/c with Decimal
        references."""
        rng = random.Random(20261112)
        for _ in range(100):
            x = Fraction(rng.randint(-10 ** 12, 10 ** 12), rng.randint(1, 10 ** 9))
            places = rng.randint(0, 60)
            with self.subTest(x=str(x), places=places):
                self.assertAccepts(OP, args_of(exact(x), places), claim(truncated(x, places), places))
        for _ in range(50):
            n = rng.choice([m for m in range(2, 60) if isqrt(m) ** 2 != m])
            a, b, c = Fraction(rng.randint(-50, 50), rng.randint(1, 9)), rng.randint(-9, 9) or 1, rng.randint(1, 9)
            node = Add((Q(a.numerator, a.denominator), Mul((Q(b, c), sqrt(n)))))
            places = rng.randint(0, 40)
            triple = decimal_truncated(node, places)
            with self.subTest(a=str(a), b=b, c=c, n=n, places=places):
                self.assertAccepts(OP, args_of(node, places), claim(triple, places), bits(places))

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_rejects_wrong_values(self):
        """Each known value with its magnitude one more and one less, the other sign, the other exactness and other
        places, in the record or in the arguments."""
        for value, places, (sign, magnitude, is_exact), evidence, text in KNOWN:
            args = args_of(value, places)
            wrong = [forged(sign or 1, magnitude + 1, places, False), forged(-sign or -1, magnitude, places, is_exact),
                     forged(sign, magnitude, places + 1, is_exact), forged(sign, magnitude * 10, places + 1, is_exact)]
            if magnitude:
                wrong += [forged(sign, magnitude - 1, places, is_exact), forged(sign, magnitude, places, not is_exact)]
            for item in wrong:
                with self.subTest(text=text, value=label(item)):
                    self.assertRejects(OP, args, item, evidence)
            with self.subTest(text=text, places="other"):
                self.assertRejects(OP, args_of(value, places + 1), claim((sign, magnitude, is_exact), places), evidence)
        # A true value of one input is wrong for another.
        self.assertRejects(OP, args_of(Q(1, 3), 5), record(1, 66666, 5, False))
        self.assertRejects(OP, args_of(sqrt(3), 30), record(1, sqrt_floor(2, 30), 30, False), bits(30))
        self.assertRejects(OP, args_of(mf.RealAlgebraicRoot((-3, 0, 0, 1), 0), 20),
                           record(1, 125992104989487316476, 20, False), ROOT)

    def test_rejects_noncanonical_values_that_are_mathematically_right(self):
        """Each forged record denotes the true number, but breaks the canonical form: -0.000 or +0.000 for 0 (sign 0
        only for 0), a bool, an int subclass, an int enumeration, a Rational or a float for an int, an int for the bool
        exact."""
        zero = args_of(0, 3)
        self.assertAccepts(OP, zero, record(0, 0, 3, True))
        quarter, root, expression = args_of(Q(1, 4), 3), args_of(CUBE_ROOT_2, 20), args_of(sqrt(2), 30)
        r20, s30 = 125992104989487316476, sqrt_floor(2, 30)
        cases = [
            ("-0.000", zero, forged(-1, 0, 3, True), ()),
            ("+0.000", zero, forged(1, 0, 3, True), ()),
            ("0 with the sign False", zero, forged(False, 0, 3, True), ()),
            ("0 with the magnitude False", zero, forged(0, False, 3, True), ()),
            ("0 with the sign Sign.ZERO", zero, forged(Sign.ZERO, 0, 3, True), ()),
            ("the sign True", quarter, forged(True, 250, 3, True), ()),
            ("the sign an IntEnum", quarter, forged(Sign.PLUS, 250, 3, True), ()),
            ("the sign an int subclass", quarter, forged(Integer(1), 250, 3, True), ()),
            ("the sign a Rational", quarter, forged(Q(1), 250, 3, True), ()),
            ("the magnitude an int subclass", quarter, forged(1, Integer(250), 3, True), ()),
            ("the magnitude a Rational", quarter, forged(1, Q(250), 3, True), ()),
            ("the magnitude a float", quarter, forged(1, 250.0, 3, True), ()),
            ("the places an int subclass", quarter, forged(1, 250, Integer(3), True), ()),
            ("exact the int 1", quarter, forged(1, 250, 3, 1), ()),
            ("exact the Rational 1", quarter, forged(1, 250, 3, Q(1)), ()),
            ("the places True for 1", args_of(Q(1, 4), 1), forged(1, 2, True, False), ()),
            ("a root, exact the int 0", root, forged(1, r20, 20, 0), ROOT),
            ("a root, the sign True", root, forged(True, r20, 20, False), ROOT),
            ("an expression, exact the int 0", expression, forged(1, s30, 30, 0), bits(30)),
            ("an expression, the magnitude an int subclass", expression, forged(1, Integer(s30), 30, False), bits(30)),
        ]
        self.assertRejectsAll(cases)
        self.assertAccepts(OP, quarter, forged(1, 250, 3, True))  # The forge itself is fine.

    def test_rejects_impostors_and_forged_records(self):
        args, true = args_of(Q(2, 3), 5), record(1, 66666, 5, False)
        self.assertAccepts(OP, args, true)

        class Subclass(TruncatedDecimal):
            """A subclass is another class: its module and name differ."""

        cases = [
            ("an impostor", support.impostor(true)),
            ("a subclass", Subclass(1, 66666, 5, False)),
            ("a tuple of the fields", (1, 66666, 5, False)),
            ("a list of the fields", [1, 66666, 5, False]),
            ("None", None),
            ("the record in a tuple", (true,)),
            ("the number itself", Q(2, 3)),
            ("the magnitude itself", 66666),
            ("the sign missing", support.forged(TruncatedDecimal, magnitude=66666, places=5, exact=False)),
            ("the magnitude missing", support.forged(TruncatedDecimal, sign=1, places=5, exact=False)),
            ("the places missing", support.forged(TruncatedDecimal, sign=1, magnitude=66666, exact=False)),
            ("exact missing", support.forged(TruncatedDecimal, sign=1, magnitude=66666, places=5)),
            ("every field missing", support.forged(TruncatedDecimal)),
            ("the sign 2", forged(2, 66666, 5, False)),
            ("the sign -2", forged(-2, 66666, 5, False)),
            ("a negative magnitude", forged(1, -66666, 5, False)),
            ("the sign 0 with a magnitude", forged(0, 66666, 5, False)),
            ("the sign 0 not exact", forged(0, 0, 5, False)),
            ("negative places", forged(1, 66666, -5, False)),
        ]
        self.assertRejectsAll([(name, args, value, ()) for name, value in cases])
        root_true = record(1, 125992104989487316476, 20, False)
        self.assertRejects(OP, args_of(CUBE_ROOT_2, 20), support.impostor(root_true), ROOT)
        enclosed_true = record(1, sqrt_floor(2, 30), 30, False)
        self.assertRejects(OP, args_of(sqrt(2), 30), support.impostor(enclosed_true), bits(30))

    def test_rejects_inexact_numbers_and_lists(self):
        """A bool, a float, a str, None, a list, a Fraction, a Decimal, an int subclass or a forged Rational where an
        exact number belongs: in the value, in the places, in the record and in the evidence."""
        true = record(1, 66666, 5, False)
        values = (True, False, 0.5, 2 / 3, "2/3", None, [Q(2, 3)], (Q(2, 3),), Fraction(2, 3), Decimal(1),
                  Integer(1), SubRational(2, 3), FakeRational(2, 3), forged_rational(4, 6), forged_rational(2, -3),
                  forged_rational(2, 0), forged_rational(2.0, 3), forged_rational(True, 3),
                  forged_rational(2, 3, missing="denominator"), forged_rational(Integer(2), 3), mf.symbol("x"),
                  mf.Polynomial((0, 1), mf.symbol("t")), {"value": Q(2, 3)})
        cases = [(f"value = {label(item)}", args_of(item, 5), true, ()) for item in values]
        cases += [(f"places = {label(item)}", args_of(Q(2, 3), item), true, ())
                  for item in (True, 5.0, "5", None, [5], (5,), Q(5), Fraction(5), Integer(5), Decimal(5))]
        cases += [(f"field {name} = {label(item)}", args_of(Q(2, 3), 5),
                   forged(**dict(dict(sign=1, magnitude=66666, places=5, is_exact=False), **{name: item})), ())
                  for name in ("sign", "magnitude", "places", "is_exact")
                  for item in (True, 1.0, "1", None, [1], (1,), Fraction(1), Decimal(1))
                  if not (name == "is_exact" and item is True)]
        expression, root = args_of(sqrt(2), 30), args_of(CUBE_ROOT_2, 20)
        enclosed, root_true = record(1, sqrt_floor(2, 30), 30, False), record(1, 125992104989487316476, 20, False)
        cases += [(f"bits = {label(item)}", expression, enclosed, ("bits", item))
                  for item in (True, 200.0, "200", None, [200], (200,), Q(200), Fraction(200), Integer(200),
                               Decimal(200))]
        cases += [(f"r = {label(item)}", args_of(HALF_TWO, 3), record(1, 1000, 3, True), ("rational", item))
                  for item in (True, 1, 1.0, "1", None, [Q(1)], (Q(1),), Fraction(1), Decimal(1), Integer(1),
                               SubRational(1), FakeRational(1, 1), forged_rational(2, 2), forged_rational(1, 0),
                               forged_rational(1.0, 1), forged_rational(1, 1, missing="numerator"))]
        cases += [("evidence as a list", expression, enclosed, list(bits(30))),
                  ("root evidence as a list", root, root_true, ["root"]),
                  ("evidence a tuple subclass", expression, enclosed, Evidence(bits(30))),
                  ("root evidence a tuple subclass", root, root_true, Evidence(ROOT)),
                  ("rational evidence a tuple subclass", args_of(Q(2, 3), 5), true, Evidence()),
                  ("evidence None", args_of(Q(2, 3), 5), true, None),
                  ("the tag as bytes", expression, enclosed, (b"bits", bits_for(30))),
                  ("the root tag as bytes", root, root_true, (b"root",))]
        self.assertRejectsAll(cases)
        self.assertAccepts(OP, args_of(Q(2, 3), 5), true)
        self.assertAccepts(OP, expression, enclosed, bits(30))

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        """Each kind of value has its one layout: () for a rational, ("root",) for a root, ("bits", b) or
        ("rational", r) for an expression; every other layout is refused, also one that fits another kind."""
        rational_case = (args_of(Q(2, 3), 5), record(1, 66666, 5, False))
        root_case = (args_of(CUBE_ROOT_2, 20), record(1, 125992104989487316476, 20, False))
        enclosed_case = (args_of(sqrt(2), 30), record(1, sqrt_floor(2, 30), 30, False))
        exact_case = (args_of(HALF_TWO, 3), record(1, 1000, 3, True))
        layouts = {
            "rational": (rational_case, (ROOT, bits(5), ("rational", Q(2, 3)), (0,), ((),), ((), ()), ("",),
                                         (None,))),
            "root": (root_case, ((), ("root", 1), ("root", "root"), ("ROOT",), ("roots",), (" root",), ("root ",),
                                 bits(20), ("rational", Q(1)), (None,), (ROOT,), ("bits",), (1,))),
            "enclosed": (enclosed_case, ((), ROOT, ("bits",), ("bits", bits_for(30), 1), ("BITS", bits_for(30)),
                                         ("bit", bits_for(30)), (bits_for(30), "bits"), ("bits", 0), ("bits", -200),
                                         ("rational",), ("rational", Q(1), Q(1)), ("digits", 1), (bits(30),),
                                         ("bits", bits_for(30), "bits", bits_for(30)))),
            "exact": (exact_case, ((), ROOT, bits(3), ("rational",), ("rational", Q(1), Q(1)), ("RATIONAL", Q(1)),
                                   (Q(1), "rational"), ("rational", (Q(1),)), (("rational", Q(1)),))),
        }
        for kind, ((args, value), evidences) in layouts.items():
            for evidence in evidences:
                with self.subTest(kind=kind, evidence=label(evidence)):
                    self.assertRejects(OP, args, value, evidence)
        self.assertAccepts(OP, *rational_case)
        self.assertAccepts(OP, *root_case, ROOT)
        self.assertAccepts(OP, *enclosed_case, bits(30))
        self.assertAccepts(OP, *exact_case, ("rational", Q(1)))

    def test_rejects_consistent_but_wrong_evidence(self):
        """Evidence that agrees with the claim but not with the value: a rational r whose digits are the claim's for
        an irrational or another rational value, a precision too small to separate a true claim, an enclosure for a
        number on a boundary, and root claims whose cell holds another root of the polynomial or has the wrong sign or
        exactness: a sign change of the polynomial across such a cell would accept them, the counts refuse them."""
        cases = [
            ("sqrt(2) as 141421/100000", args_of(sqrt(2), 5), record(1, 141421, 5, True),
             ("rational", Q(141421, 100000))),
            ("sqrt(2) as 1414214/1000000", args_of(sqrt(2), 5), record(1, 141421, 5, False),
             ("rational", Q(1414214, 1000000))),
            ("sqrt(2) as 7/5 at 0 places", args_of(sqrt(2), 0), record(1, 1, 0, False), ("rational", Q(7, 5))),
            ("sqrt(2)*sqrt(2)/2 as 1.000001", args_of(HALF_TWO, 3), record(1, 1000, 3, False),
             ("rational", Q(1_000_001, 1_000_000))),
            ("sqrt(8) - 2*sqrt(2) as 1/10**9", args_of(RATIONAL_VALUED[1][1], 3), record(1, 0, 3, False),
             ("rational", Q(1, 10 ** 9))),
            ("sqrt(2) - sqrt(2) as a tiny positive", args_of(Add((sqrt(2), minus(sqrt(2)))), 3),
             record(1, 0, 3, False), ("rational", Q(1, 10 ** 9))),
        ]
        for b in (1, 8, 64):
            cases.append((f"sqrt(2) at 30 places with {b} bits", args_of(sqrt(2), 30),
                          record(1, sqrt_floor(2, 30), 30, False), ("bits", b)))
        for b in (200, 65_536):  # 1.000 exactly: no enclosure lies inside an open cell next to it.
            cases.append((f"1.000 claimed below with {b} bits", args_of(HALF_TWO, 3), record(1, 999, 3, False),
                          ("bits", b)))
            cases.append((f"1.000 claimed inexact with {b} bits", args_of(HALF_TWO, 3), record(1, 1000, 3, False),
                          ("bits", b)))
            cases.append((f"0 claimed with {b} bits", args_of(RATIONAL_VALUED[1][1], 3), record(0, 0, 3, True),
                          ("bits", b)))
        quartic = (6, 0, -5, 0, 1)  # Roots -sqrt(3), -sqrt(2), sqrt(2), sqrt(3).
        cases += [
            ("-sqrt(2) claimed in the cell of -sqrt(3)", args_of(mf.RealAlgebraicRoot(quartic, 1), 1),
             record(-1, 17, 1, False), ROOT),
            ("sqrt(2) claimed in the cell of sqrt(3)", args_of(mf.RealAlgebraicRoot(quartic, 2), 1),
             record(1, 17, 1, False), ROOT),
            ("-sqrt(3) claimed in the cell of -sqrt(2)", args_of(mf.RealAlgebraicRoot(quartic, 0), 1),
             record(-1, 14, 1, False), ROOT),
            ("sqrt(3) claimed as sqrt(2)", args_of(mf.RealAlgebraicRoot(quartic, 3), 3),
             record(1, 1414, 3, False), ROOT),
            ("-sqrt(2) claimed positive", args_of(mf.RealAlgebraicRoot((-2, 0, 1), 0), 3),
             record(1, 1414, 3, False), ROOT),
            ("0 claimed positive", args_of(mf.RealAlgebraicRoot((0, -1, 0, 1), 1), 3), record(1, 0, 3, False), ROOT),
            ("0 claimed negative", args_of(mf.RealAlgebraicRoot((0, -1, 0, 1), 1), 3), record(-1, 0, 3, False), ROOT),
            ("1 claimed as 0", args_of(mf.RealAlgebraicRoot((0, -1, 0, 1), 2), 3), record(0, 0, 3, True), ROOT),
            ("1/2 claimed inexact", args_of(mf.RealAlgebraicRoot((-1, 0, 4), 1), 1), record(1, 5, 1, False), ROOT),
            ("-1/2 claimed inexact", args_of(mf.RealAlgebraicRoot((-1, 0, 4), 0), 1), record(-1, 5, 1, False), ROOT),
            ("-1/2 claimed as -0.4", args_of(mf.RealAlgebraicRoot((-1, 0, 4), 0), 1), record(-1, 4, 1, False), ROOT),
            ("sqrt(2) claimed exact", args_of(mf.RealAlgebraicRoot((-2, 0, 1), 1), 3), record(1, 1414, 3, True), ROOT),
        ]
        self.assertRejectsAll(cases)
        self.assertAccepts(OP, args_of(mf.RealAlgebraicRoot(quartic, 1), 1), record(-1, 14, 1, False), ROOT)

    def test_rejects_enclosures_that_touch_a_boundary(self):
        """A number exactly on a decimal boundary, reached through every rule of the enclosure (Rational leaves, square
        roots of a square and of non-squares, sums, products, squares and powers), has no enclosure inside the open
        cell on either side at any precision: every end is rounded outward and the comparisons are strict. The claims
        of both neighbouring cells are refused for every b from 60 to 199, for the number and its negative; the true
        claim needs ("rational", r)."""
        boundaries = (
            ("1/4 as a sum of Rationals", Add((Q(1, 4), Q(0))), Fraction(1, 4), 2),
            ("1/4 as a square of 1/2", Pow(Q(1, 2), 2), Fraction(1, 4), 2),
            ("1/100 as (1/10)^2", Pow(Q(1, 10), 2), Fraction(1, 100), 2),
            ("1/100 as 1/10 * 1/10", Mul((Q(1, 10), Q(1, 10))), Fraction(1, 100), 2),
            ("27/1000 as (3/10)^3", Pow(Q(3, 10), 3), Fraction(27, 1000), 3),
            ("9/100 as (1/10 + 1/5)^2", Pow(Add((Q(1, 10), Q(1, 5))), 2), Fraction(9, 100), 2),
            ("1/2 as sqrt(1/4)", sqrt(Fraction(1, 4)), Fraction(1, 2), 1),
            ("2 as sqrt(2)^2", Pow(sqrt(2), 2), Fraction(2), 0),
            ("4 as sqrt(2)*sqrt(8)", Mul((sqrt(2), sqrt(8))), Fraction(4), 0),
            ("1/50 as (sqrt(2)/10)^2", Pow(Mul((sqrt(2), Q(1, 10))), 2), Fraction(1, 50), 2),
            ("1 as sqrt(2)*sqrt(2)/2", HALF_TWO, Fraction(1), 3),
        )
        for name, node, value, places in boundaries:
            _, magnitude, is_exact = truncated(value, places)
            self.assertTrue(is_exact and magnitude > 0)
            for sign in (1, -1):
                args = args_of(node if sign > 0 else minus(node), places)
                with self.subTest(name=name, sign=sign):
                    self.assertAccepts(OP, args, record(sign, magnitude, places, True),
                                       ("rational", rational(sign * value)))
                    for b in range(60, 200):
                        self.assertRejects(OP, args, forged(sign, magnitude - 1, places, False), ("bits", b))
                        self.assertRejects(OP, args, forged(sign, magnitude, places, False), ("bits", b))

    def test_rejects_root_arguments_outside_their_canonical_form(self):
        """RealAlgebraicRoot arguments made with object.__new__: the checker reads only the canonical form that the
        constructor makes (primitive, square-free, positive leading coefficient, an index among the real roots), even
        where another reading would give the true number."""
        true = record(1, 1414, 3, False)  # sqrt(2) to 3 places.
        self.assertAccepts(OP, args_of(mf.RealAlgebraicRoot((-2, 0, 1), 1), 3), true, ROOT)

        def root(coefficients: object, index: object = 1) -> object:
            return support.forged(mf.RealAlgebraicRoot, integer_coefficients=coefficients, real_index=index)

        cases = [
            ("the forge itself", root((-2, 0, 1)), None),
            ("(x^2 - 2)^2, not square-free", root((4, 0, -4, 0, 1)), False),
            ("a negative leading coefficient", root((2, 0, -1)), False),
            ("not primitive", root((-4, 0, 2)), False),
            ("a trailing zero", root((-2, 0, 1, 0)), False),
            ("an index past the roots", root((-2, 0, 1), 2), False),
            ("a negative index", root((-2, 0, 1), -1), False),
            ("the index True", root((-2, 0, 1), True), False),
            ("the index 1.0", root((-2, 0, 1), 1.0), False),
            ("a list of coefficients", root([-2, 0, 1]), False),
            ("a bool coefficient", root((-2, False, 1)), False),
            ("a float coefficient", root((-2.0, 0, 1)), False),
            ("an int subclass coefficient", root((Integer(-2), 0, 1)), False),
            ("a constant", root((5,), 0), False),
            ("no coefficient", root((), 0), False),
            ("the coefficients missing", support.forged(mf.RealAlgebraicRoot, real_index=1), False),
            ("the index missing", support.forged(mf.RealAlgebraicRoot, integer_coefficients=(-2, 0, 1)), False),
        ]
        for name, value, expected in cases:
            with self.subTest(case=name):
                if expected is None:
                    self.assertAccepts(OP, args_of(value, 3), true, ROOT)
                else:
                    self.assertRejects(OP, args_of(value, 3), true, ROOT)

    def test_rejects_expressions_outside_the_scope(self):
        """Only closed trees of Rational, Sqrt, Add, Mul and Pow nodes in their constructors' form: a Symbol, a
        RealAlgebraicRoot inside an expression, an int, a float or a forged Rational in the tree, a list of terms, an
        Add of one term, a negative, bool or float exponent, a negative or non-Rational radicand, a missing field and a
        node inside itself are refused under either evidence."""
        x = mf.symbol("x")
        add_list = forged_node(Add, terms=[sqrt(2), Q(1)])
        cycle = forged_node(Add, terms=None)
        object.__setattr__(cycle, "terms", (cycle, Q(1)))
        deep_cycle = forged_node(Mul, factors=None)
        object.__setattr__(deep_cycle, "factors", (Q(2), Add((sqrt(2), Pow(deep_cycle, 2)))))
        nodes = [
            ("a Symbol", x), ("a sum with a Symbol", Add((x, sqrt(2)))),
            ("a power of a Symbol", Pow(Add((x, Q(1))), 2)),
            ("a root inside a sum", Add((CUBE_ROOT_2, sqrt(2)))), ("a root inside a product", Mul((Q(2), CUBE_ROOT_2))),
            ("a Polynomial", mf.Polynomial((0, 1), x)),
            ("an int in a sum", forged_node(Add, terms=(sqrt(2), 1))),
            ("a float in a product", forged_node(Mul, factors=(sqrt(2), 0.5))),
            ("a forged Rational in a sum", forged_node(Add, terms=(sqrt(2), forged_rational(2, 4)))),
            ("a list of terms", add_list), ("an Add of one term", forged_node(Add, terms=(sqrt(2),))),
            ("an empty Mul", forged_node(Mul, factors=())),
            ("a negative exponent", forged_node(Pow, base=sqrt(2), exponent=-1)),
            ("a bool exponent", forged_node(Pow, base=sqrt(2), exponent=True)),
            ("a float exponent", forged_node(Pow, base=sqrt(2), exponent=2.0)),
            ("a negative radicand", forged_node(Sqrt, radicand=Q(-2))),
            ("an int radicand", forged_node(Sqrt, radicand=2)),
            ("a Fraction radicand", forged_node(Sqrt, radicand=Fraction(2))),
            ("a Sqrt of a Sqrt", forged_node(Sqrt, radicand=sqrt(2))),
            ("a forged radicand", forged_node(Sqrt, radicand=forged_rational(4, 2))),
            ("the radicand missing", forged_node(Sqrt)), ("the base missing", forged_node(Pow, exponent=2)),
            ("the terms missing", forged_node(Add)),
            ("a node inside itself", cycle), ("a node inside its own grandchild", deep_cycle),
        ]
        cases = []
        for name, node in nodes:
            cases.append((name + ", bits", args_of(node, 3), record(1, 2414, 3, False), bits(3)))
            cases.append((name + ", rational", args_of(node, 3), record(1, 1000, 3, True), ("rational", Q(1))))
        self.assertRejectsAll(cases)

    def test_rejects_wrong_operations_and_arguments(self):
        args, true = args_of(Q(2, 3), 5), record(1, 66666, 5, False)
        self.assertAccepts(OP, args, true)
        for operation in ("digit", "Digits", "digits ", "DIGITS", "decimal_digits", "pi_digits", "continued_fraction",
                          "", None, 1, ("digits",), b"digits", ["digits"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, true)
        self.assertRejectsAll([
            ("arguments as pairs", tuple(args.items()), true, ()),
            ("arguments as a list of pairs", list(args.items()), true, ()),
            ("arguments as a read-only mapping", MappingProxyType(args), true, ()),
            ("None as arguments", None, true, ()),
            ("no arguments", {}, true, ()),
            ("the value missing", {"places": 5}, true, ()),
            ("the places missing", {"value": Q(2, 3)}, true, ()),
            ("an extra argument", dict(args, base=10), true, ()),
            ("the value renamed", {"expression": Q(2, 3), "places": 5}, true, ()),
            ("negative places", args_of(Q(2, 3), -5), forged(1, 66666, -5, False), ()),
            ("places past the ceiling", args_of(Q(2, 3), 10_001), record(1, 10 ** 10_001 * 2 // 3, 10_001, False), ()),
            ("places far past the ceiling", args_of(Q(2, 3), 10 ** 30), forged(1, 66666, 10 ** 30, False), ()),
        ])
        self.assertAccepts(OP, args_of(Q(2, 3), 10_000), record(1, 10 ** 10_000 * 2 // 3, 10_000, False))

    def test_oversized_forgeries_fail_fast(self):
        """A magnitude longer than max_integer_bits, places, signs and precisions far out of range, an r longer than
        max_integer_bits and evidence of a hundred thousand items: False before any arithmetic on them. A magnitude of
        ten million bits shows that nothing is computed with it: one product alone would take seconds."""
        quarter, expression, root = args_of(Q(1, 4), 3), args_of(sqrt(2), 30), args_of(CUBE_ROOT_2, 20)
        enclosed = record(1, sqrt_floor(2, 30), 30, False)
        for giant in (GIANT, 1 << 10_000_000):
            for args, evidence, places in ((quarter, (), 3), (expression, bits(30), 30), (root, ROOT, 20)):
                for value in (forged(1, giant, places, False), forged(1, giant, places, True),
                              forged(-1, giant + 1, places, False), forged(giant, 250, places, True),
                              forged(1, 250, giant, True), forged(1, giant, giant, False)):
                    with self.subTest(giant=giant.bit_length(), args=label(args["value"]), value=label(value)):
                        self.assertRejectsFast(args, value, evidence)
            for evidence in (("bits", giant), ("bits", -giant), ("rational", Q(giant + 1, 3)),
                             ("rational", Q(3, giant + 1)), ("rational", forged_rational(giant, giant + 1))):
                with self.subTest(giant=giant.bit_length(), evidence=label(evidence)):
                    self.assertRejectsFast(expression, enclosed, evidence)
        for b in (65_537, 10 ** 9):
            self.assertRejectsFast(expression, enclosed, ("bits", b))
        for evidence in ((0,) * 100_000, ("bits",) * 100_000, ROOT * 100_000, (("bits", 200),) * 100_000):
            with self.subTest(evidence=len(evidence)):
                self.assertRejectsFast(expression, enclosed, evidence)
                self.assertRejectsFast(quarter, record(1, 250, 3, True), evidence)
                self.assertRejectsFast(root, record(1, 125992104989487316476, 20, False), evidence)
        # The longest magnitude a record may hold under the default limits, refused by one comparison.
        self.assertRejectsFast(quarter, forged(1, 2 ** 65_535, 3, False))

    def test_arguments_beyond_the_budget_exceed_it(self):
        """An argument beyond the budget raises ResourceLimitError, as run() and the verifier refuse it before any
        checker runs: such a claim is beyond the budget, not wrong. So does an expression whose enclosure passes
        2**max_integer_bits or whose algebra does, and one with more than MAX_RADICALS independent square roots under
        ("rational", r), which the producer refuses as well. Each refusal is immediate."""
        self.assertExhausts(args_of(GIANT, 0), record(1, 1, 0, True))
        self.assertExhausts(args_of(Q(1, GIANT), 0), record(1, 0, 0, False))
        self.assertExhausts(args_of(Q(GIANT + 1, 3), 0), record(1, 1, 0, False))
        huge_root = support.forged(mf.RealAlgebraicRoot, integer_coefficients=(-GIANT, 1), real_index=0)
        self.assertExhausts(args_of(huge_root, 0), record(1, 1, 0, False), ROOT)
        self.assertExhausts(args_of(Mul((sqrt(2), Q(GIANT))), 0), record(1, 1, 0, False), bits(0))
        self.assertExhausts(args_of(Pow(sqrt(2), 10 ** 6), 0), record(1, 1, 0, False), bits(0))
        self.assertExhausts(args_of(Pow(sqrt(2), 10 ** 6), 0), record(1, 1, 0, True), ("rational", Q(1)))
        eleven = Add(tuple(Mul((sqrt(p), sqrt(p))) for p in PRIMES))
        self.assertExhausts(args_of(eleven, 0), record(1, sum(PRIMES), 0, True), ("rational", Q(sum(PRIMES))))
        deep: object = sqrt(2)
        for _ in range(200):
            deep = Add((deep, Q(1)))
        self.assertExhausts(args_of(deep, 0), record(1, 201, 0, False), bits(0))
        # A million terms, all one shared node: two distinct nodes, but every term is paid for before it is walked.
        wide = forged_node(Add, terms=(Q(1),) * 1_000_000)
        self.assertExhausts(args_of(wide, 0), record(1, 10 ** 6, 0, True), ("rational", Q(10 ** 6)))
        # Within their limits: eleven radicands under an enclosure (no algebra), a power whose algebra stays small.
        irrational_eleven = Add(tuple(sqrt(p) for p in PRIMES))
        triple = decimal_truncated(irrational_eleven, 10)
        self.assertAccepts(OP, args_of(irrational_eleven, 10), claim(triple, 10), bits(10))
        self.assertAccepts(OP, args_of(Pow(sqrt(2), 1000), 3), record(1, 2 ** 500 * 1000, 3, True),
                           ("rational", Q(2 ** 500)))
        # A smaller budget scales every ceiling: 2**64 has 65 bits.
        limits = ComputationLimits(max_integer_bits=64)
        self.assertAccepts(OP, args_of(Q(2 ** 63, 3), 0), record(1, 2 ** 63 // 3, 0, False), limits=limits)
        self.assertExhausts(args_of(2 ** 64, 0), record(1, 1, 0, False), limits=limits)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_before_it_runs(self):
        """The charges depend on the claim alone: a budget one unit short of a check refuses it with
        ResourceLimitError and its exact size passes; more places cost more. (A root's Sturm chain is cached between
        checks by certify.upoly, so its work is measured on the other routes.)"""
        claims = (
            (args_of(sqrt(2), 10_000), record(1, sqrt_floor(2, 10_000), 10_000, False), bits(10_000)),
            (args_of(Q(2, 3), 10_000), record(1, 2 * 10 ** 10_000 // 3, 10_000, False), ()),
            (args_of(RATIONAL_VALUED[5][1], 50), record(1, 5 * 10 ** 50, 50, True), ("rational", Q(5))),
        )
        for args, value, evidence in claims:
            with self.subTest(evidence=label(evidence)):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(OP, args, value, evidence), True)
                work = budget.work
                self.assertGreater(work, 10)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, OP, args, value, evidence, limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(OP, args, value, evidence, limits=ComputationLimits(max_work=work))
        works = []
        for places in (100, 1_000, 10_000):
            with computation(fresh=True) as budget:
                self.assertIs(checker.check_operation(OP, args_of(sqrt(2), places),
                                                      record(1, sqrt_floor(2, places), places, False), bits(places)),
                              True)
            works.append(budget.work)
        self.assertEqual(works, sorted(works))
        self.assertGreater(works[2], 4 * works[0])

    def test_no_power_and_no_producer_code(self):
        """The checker forms no power with pow or ** (powers are charged products), imports only the standard
        library's math and the layers that a checker may use, and names no importlib, __import__, sys or builtins."""
        tree = ast.parse(inspect.getsource(checker))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertEqual(names & {"pow", "importlib", "__import__", "sys", "builtins", "eval", "exec"}, set())
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imported, {"__future__", "math", "mathforge.certify.upoly", "mathforge.limits",
                                    "mathforge.model", "mathforge.trusted.surd", "mathforge.trusted.upoly"})

    # -----------------------------------------------------------------------------------------------------------
    # The verifier, the record and its presentation

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip, and a wrong value or
        wrong evidence fails the mathematics check while the trace stays verified."""
        for value, places, triple, evidence, text in KNOWN:
            args = args_of(value, places)
            with self.subTest(text=text):
                decoded = self.roundtrip(claimed(args, claim(triple, places), evidence))
                self.assertIs(type(decoded.value), TruncatedDecimal)
                sign, magnitude, is_exact = triple
                wrong = record(sign or 1, magnitude + 1, places, False)
                report = support.statuses(mf.verify(claimed(args, wrong, evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
                report = support.statuses(mf.verify(claimed(args, claim(triple, places), evidence + (1,))))
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_record_canonical_form(self):
        for sign, magnitude, places, is_exact in ((1, 0, 5, False), (0, 0, 0, True), (-1, 12, 2, False),
                                                  (1, 250, 3, True), (1, 10 ** 40, 0, True), (-1, 0, 0, False),
                                                  (-1, 7, 0, True), (1, 1, 10 ** 30, False)):
            with self.subTest(fields=(sign, magnitude, places, is_exact)):
                value = TruncatedDecimal(sign, magnitude, places, is_exact)
                self.assertEqual((value.sign, value.magnitude, value.places, value.exact),
                                 (sign, magnitude, places, is_exact))
                self.assertEqual(mf.from_data(mf.to_data(value)), value)
        for fields in ((2, 1, 3, False), (-2, 1, 3, False), (True, 1, 3, False), (1.0, 1, 3, False), ("1", 1, 3, False),
                       (None, 1, 3, False), (Q(1), 1, 3, False), (Sign.PLUS, 1, 3, False), (1, -1, 3, False),
                       (1, True, 3, False), (1, 1.0, 3, False), (1, Q(1), 3, False), (1, "1", 3, False),
                       (1, Integer(1), 3, False), (1, 1, -1, False), (1, 1, True, False), (1, 1, 3.0, False),
                       (1, 1, 3, 1), (1, 1, 3, 0), (1, 1, 3, None), (1, 1, 3, "True"), (0, 1, 3, True),
                       (0, 0, 3, False), (1, 0, 3, True), (-1, 0, 3, True), (0, 5, 3, False)):
            with self.subTest(fields=label(fields)), self.assertRaises(mf.InvalidInput):
                TruncatedDecimal(*fields)
        for fields in ((1, GIANT, 3, False), (1, 1, GIANT, False)):
            with self.subTest(fields="giant"), self.assertRaises(ResourceLimitError):
                TruncatedDecimal(*fields)
        document = mf.to_data(TruncatedDecimal(0, 0, 3, True))
        document["data"]["sign"] = {"type": "integer", "value": "-1"}  # -0.000: the same number, not canonical.
        with self.assertRaisesRegex(mf.SerializationError, "^Invalid mathematical document$"):
            mf.from_data(document)
        document = mf.to_data(TruncatedDecimal(1, 250, 3, True))
        document["data"]["exact"] = {"type": "integer", "value": "1"}  # An int for the bool.
        with self.assertRaisesRegex(mf.SerializationError, "^Invalid mathematical document$"):
            mf.from_data(document)

    def test_rendering(self):
        r"""The digits with the point before the last places of them, a leading - for a negative number and ... after
        a truncated one; LaTeX writes \ldots."""
        for value, text, latex in (
                (record(1, 66666, 5, False), "0.66666...", r"0.66666\ldots"),
                (record(1, 250, 3, True), "0.250", "0.250"),
                (record(-1, 12, 2, False), "-0.12...", r"-0.12\ldots"),
                (record(0, 0, 4, True), "0.0000", "0.0000"),
                (record(1, 1414213562373095048801688724209, 30, False), "1.414213562373095048801688724209...",
                 r"1.414213562373095048801688724209\ldots"),
                (record(1, 125992104989487316476, 20, False), "1.25992104989487316476...",
                 r"1.25992104989487316476\ldots"),
                (record(1, 1000, 3, True), "1.000", "1.000"),
                (record(1, 2, 0, False), "2...", r"2\ldots"),
                (record(-1, 2, 0, False), "-2...", r"-2\ldots"),
                (record(-1, 0, 0, False), "-0...", r"-0\ldots"),
                (record(1, 3, 0, True), "3", "3"),
                (record(0, 0, 0, True), "0", "0"),
                (record(1, 16887, 25, False), "0.0000000000000000000016887...", r"0.0000000000000000000016887\ldots"),
                (record(-1, 123456, 3, True), "-123.456", "-123.456")):
            with self.subTest(text=text):
                self.assertEqual((mf.render_text(value), mf.to_latex(value)), (text, latex))
        # Places far beyond the render limits stop at the character limit without building the zeros.
        started = time.perf_counter()
        for value in (record(1, 1, 10 ** 30, False), record(1, 10 ** 19_000, 10 ** 6, False),
                      record(0, 0, 2 ** 65_535, True)):
            for render in (mf.render_text, mf.to_latex):
                with self.subTest(places=value.places.bit_length(), render=render.__name__):
                    with self.assertRaisesRegex(mf.ResourceLimitError, "characters"):
                        render(value)
        self.assertLess(time.perf_counter() - started, FAST)
        with self.assertRaisesRegex(mf.ResourceLimitError, "characters"):
            mf.render_text(record(1, 10 ** 30, 2, False), limits=mf.RenderLimits(max_chars=20))
