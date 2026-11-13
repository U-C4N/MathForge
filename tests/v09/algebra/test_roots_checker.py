"""The independent checker of mathforge.areas.algebra.roots (mathforge.certify.algebra.roots), tested on its own before
the family's producer exists.

Every true claim is computed here, never by the producer: root counts from polynomials multiplied out of known roots
(rationals, pairs -+sqrt(s) and quadratics without real roots, each counted in an interval by exact comparisons), and
from a short reference of this module, Sturm's theorem on the classic remainder sequence over fractions.Fraction;
complex roots from factorizations that the tests choose and multiply out, the records spelled out from the formulas of
the family's docstring (the primitive integer form F of a quadratic factor, G of 4y^2 + D), compared with what v0.7's
quadratic_roots builds (the card's canonical form) and sorted by exact rational enclosures. The evidence follows the
layouts of the family's docstring. The checker answers False to every corrupted claim, never with an exception, and
refuses an oversized forgery within FAST seconds under the default budget.

Run this module by name with

    python -m unittest tests.v09.algebra.test_roots_checker
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
import inspect
from math import gcd, isqrt
import random
import time
from types import MappingProxyType

import mathforge as mf
from mathforge import contracts, registry
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.algebra import roots as family
from mathforge.certify.algebra import roots as checker
from mathforge.complex_numbers import ComplexAlgebraic, ComplexRoot
from mathforge.contracts import CheckStatus
from mathforge.deferred import Reason, Unevaluated
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational, Symbol, sqrt
from mathforge.polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
         "and decimal digits")
POINT = "Rational | int | Literal['-inf', '+inf']"
X = Symbol("x")
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
F = Fraction


# ---------------------------------------------------------------------------------------------------------------
# References: polynomials as lists of Fractions, lowest degree first, without trailing zeros ([] is zero).

def trim(p: Iterable[Fraction]) -> list[Fraction]:
    p = [F(c) for c in p]
    while p and p[-1] == 0:
        p.pop()
    return p


def times(p: Sequence[Fraction], q: Sequence[Fraction]) -> list[Fraction]:
    if not p or not q:
        return []
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def expand(lead: object, factors: Sequence[tuple[Sequence[object], int]]) -> list[Fraction]:
    """lead * prod f**e, for factors (coefficients lowest degree first, exponent)."""
    p = [F(lead)]
    for coefficients, exponent in factors:
        for _ in range(exponent):
            p = times(p, [F(c) for c in coefficients])
    return trim(p)


def rational(value: object) -> Rational:
    value = F(value)
    return Rational(value.numerator, value.denominator)


def end(value: object) -> Fraction | None:
    """The Fraction of an end of the interval (an int or a Rational), None for an infinity."""
    if type(value) is str:
        return None
    if type(value) is Rational:
        return F(value.numerator, value.denominator)
    return F(value)


def polynomial(p: Iterable[object]) -> Polynomial:
    return Polynomial(tuple(rational(c) for c in trim(p)) or (Rational(0),), X)


def primitive(p: Sequence[object]) -> tuple[int, ...]:
    """The primitive integer form of a nonzero polynomial, with a positive leading coefficient."""
    p = [F(c) for c in p]
    common = 1
    for c in p:
        common = common * c.denominator // gcd(common, c.denominator)
    integers = [int(c * common) for c in p]
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


def sturm_count(p: Sequence[object], low: Fraction | None, high: Fraction | None) -> int:
    """The distinct real roots of a nonzero polynomial in [low, high], None for an infinite end: Sturm's theorem on the
    classic sequence s, s', then minus each remainder, of the square-free part s = p / gcd(p, p'), over the fractions.
    Sign variations skip zeros, so V(low) - V(high) counts the roots in (low, high]; low is added when it is a root."""
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

    count = variations(low, False) - variations(high, True)
    return count + (low is not None and evaluate(s, low) == 0)


@dataclass(frozen=True)
class Rooted:
    """lead * prod (x - r)**m * prod (x**2 - s) * prod (x**2 + b x + c): distinct rational roots r with multiplicities,
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


# Records spelled out from the family's docstring: the parts of the roots of each monic factor.

def complex_value(real: object, imag: object) -> ComplexAlgebraic:
    """A ComplexAlgebraic of given parts, built as v0.7's producers build theirs (fields set on a new object), without
    the normalizing constructor."""
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
    factor: -a for x + a; for x^2 + b x + c with D = b^2 - 4c, the parts RealAlgebraicRoot(F, 0) and (F, 1), F the
    primitive integer form of the factor, with imaginary part 0 when D > 0, and the real part -b/2 with the imaginary
    parts -+sqrt(-D)/2 when D < 0: Rationals when -D is a rational square, else RealAlgebraicRoot(G, 0) and (G, 1), G
    the primitive integer form of 4y^2 + D."""
    if len(coefficients) == 2:
        a = F(coefficients[0])
        return [(rational(-a), Rational(0), (-a, F(0), F(0)), ZERO)]
    c, b = F(coefficients[0]), F(coefficients[1])
    discriminant = b * b - 4 * c
    if discriminant > 0:
        form = primitive(coefficients)
        return [(RealAlgebraicRoot(form, k), Rational(0), (-b / 2, F(2 * k - 1, 2), discriminant), ZERO)
                for k in (0, 1)]
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
    """-1, 0 or 1 as x is below, equal to or above y: rationals exactly, others by enclosures refined until they
    separate (the claims of these tests never hold one irrational number under two spellings)."""
    if x == y:
        return 0
    if x[1] == 0 and y[1] == 0:
        return sign(x[0] - y[0])
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
    """(polynomial, value, evidence) of the true claim complex_roots(lead * prod f**e), for distinct monic irreducible
    factors f of degree 1 or 2 (coefficients lowest degree first) with their exponents."""
    entries = sorted(factors, key=lambda entry: factor_key(entry[0]))
    evidence = (rational(lead), tuple((tuple(rational(c) for c in coefficients), exponent)
                                      for coefficients, exponent in entries))
    roots = []
    for coefficients, exponent in factors:
        for real, imag, real_value, imag_value in factor_roots(coefficients):
            roots.append((real_value, imag_value, record(real, imag, exponent)))
    roots.sort(key=cmp_to_key(lambda x, y: compare(x[0], y[0]) or compare(x[1], y[1])))
    return polynomial(expand(lead, factors)), tuple(item for _, _, item in roots), evidence


def with_root(value: tuple, index: int, replacement: object) -> tuple:
    return value[:index] + (replacement,) + value[index + 1:]


def with_part(value: tuple, index: int, **parts: object) -> tuple:
    """The value with the root at index rebuilt with other parts (real=..., imag=...), its multiplicity kept."""
    root = value[index]
    return with_root(value, index, record(parts.get("real", root.root.real), parts.get("imag", root.root.imag),
                                          root.multiplicity))


def forged_root(value: tuple, index: int, **fields: object) -> tuple:
    """The value with the root at index forged with object.__new__: its fields as given, the others kept."""
    root = value[index]
    return with_root(value, index, support.forged(ComplexRoot, **{"root": root.root,
                                                                  "multiplicity": root.multiplicity, **fields}))


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass.
Arguments = type("Arguments", (dict,), {})  # A dict subclass, which could answer differently on each lookup.

# The base claim of the corruption tables of complex_roots: -3/2 (x + 1/2)(x^2 - 2)(x^2 + 1)(x^2 + x + 1)^2, of
# degree 9, whose seven roots exercise every kind of part: -sqrt 2, -1/2 - (sqrt 3/2) i (double), -1/2, -1/2 +
# (sqrt 3/2) i (double), -i, i, sqrt 2; three of them share the real part -1/2.
BASE_FACTORS = (((F(1, 2), 1), 1), ((-2, 0, 1), 1), ((1, 0, 1), 1), ((1, 1, 1), 2))
BASE = complex_claim(F(-3, 2), BASE_FACTORS)


class RootsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        polynomial_argument = Argument("polynomial", "Polynomial", None, "polynomial")
        expected = (
            OperationSpec("count_real_roots", "algebra",
                          (polynomial_argument, Argument("lower", POINT, "'-inf'", "point"),
                           Argument("upper", POINT, "'+inf'", "point")), "int",
                          "Count the distinct real roots of a rational polynomial in a closed interval with rational "
                          "or infinite ends.", SCOPE, version=6),
            OperationSpec("complex_roots", "algebra", (polynomial_argument,), "tuple[ComplexRoot, ...]",
                          "List every complex root with multiplicity of a rational polynomial whose irreducible "
                          "factors have degree at most two.", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {})  # An int, and v0.7's ComplexRoot records: no record class of its own.
        self.assertEqual(family.CHECKER, "mathforge.certify.algebra.roots")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)
        self.assertEqual(checker.MAX_DEGREE, MAX_POLYNOMIAL_DEGREE)  # The checker cannot import the constant.

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_count_real_roots_accepts_the_known_values(self):
        cubic = [0, -1, 0, 1]  # x^3 - x
        for coefficients, lower, upper, count in (([-2, 0, 1], "-inf", "+inf", 2), ([-2, 0, 1], 0, 2, 1),
                                                  (cubic, -1, 1, 3), (cubic, "-inf", 0, 2), ([1, -2, 1], 0, 2, 1),
                                                  ([1, 0, 1], "-inf", "+inf", 0), ([5], "-inf", "+inf", 0)):
            with self.subTest(coefficients=coefficients, lower=lower, upper=upper):
                low = end(lower)
                high = end(upper)
                self.assertEqual(sturm_count(coefficients, low, high), count)
                args = dict(polynomial=polynomial(coefficients), lower=lower, upper=upper)
                self.assertAccepts("count_real_roots", args, count)
                self.assertRejects("count_real_roots", args, count + 1)
        # Ends that are roots, a point interval, Rational ends and the defaults' infinities on one side only.
        for lower, upper, count in ((-1, -1, 1), (Rational(1, 2), Rational(1, 2), 0), (0, 0, 1),
                                    (-1, Rational(-1, 2), 1), (Rational(-1, 2), "+inf", 2), ("-inf", -2, 0),
                                    (1, "+inf", 1), (2, "+inf", 0), (Rational(-3, 2), Rational(3, 2), 3),
                                    (Rational(1, 3), 1, 1)):
            with self.subTest(cubic=True, lower=lower, upper=upper):
                self.assertAccepts("count_real_roots", dict(polynomial=polynomial(cubic), lower=lower, upper=upper),
                                   count)

    def test_count_real_roots_accepts_constructed_claims(self):
        rng = random.Random(20261112)
        near = (F(0), F(1, 1000), F(-1, 1000), F(1, 10 ** 12), F(-1, 10 ** 12))
        for trial in range(120):
            rationals = {}
            for _ in range(rng.randrange(0, 4)):
                rationals[F(rng.randrange(-30, 31), rng.randrange(1, 7))] = rng.randrange(1, 4)
            radicands = set()
            for _ in range(rng.randrange(0, 3)):
                s = F(rng.randrange(1, 60), rng.randrange(1, 5))
                if rational_sqrt(s) is None:
                    radicands.add(s)
            definite = [(F(rng.randrange(5, 40)), F(rng.randrange(-4, 5))) for _ in range(rng.randrange(0, 2))]
            lead = F(rng.choice((-1, 1)) * rng.randrange(1, 9), rng.randrange(1, 4))
            built = Rooted(lead, tuple(rationals.items()), tuple(radicands), tuple(definite))
            p = built.coefficients()
            # Ends at the roots, beside them, at approximations of the irrational roots, and infinite.
            points = [r + delta for r in rationals for delta in near]
            for s in radicands:
                approximation = F(isqrt(s.numerator * 10 ** 12 // s.denominator), 10 ** 6)
                points += [approximation, -approximation, approximation + F(1, 10 ** 6), -approximation - F(1, 10 ** 6)]
            points += [F(rng.randrange(-40, 41), rng.randrange(1, 5)) for _ in range(3)]
            for _ in range(6):
                low, high = rng.choice(points + [None]), rng.choice(points + [None])
                if low is not None and high is not None and low > high:
                    low, high = high, low
                count = built.count(low, high)
                with self.subTest(trial=trial, low=low, high=high):
                    self.assertEqual(sturm_count(p, low, high), count)  # The two references agree.
                    lower = "-inf" if low is None else rational(low) if low.denominator > 1 else int(low)
                    upper = "+inf" if high is None else rational(high) if high.denominator > 1 else int(high)
                    args = dict(polynomial=polynomial(p), lower=lower, upper=upper)
                    self.assertAccepts("count_real_roots", args, count)
                    self.assertRejects("count_real_roots", args, count + 1)
                    if count:
                        self.assertRejects("count_real_roots", args, count - 1)

    def test_count_real_roots_accepts_reference_counts(self):
        rng = random.Random(20261113)
        for trial in range(150):
            p = trim(F(rng.randrange(-9, 10)) for _ in range(rng.randrange(1, 9)))
            if not p:
                continue
            ends = [None] + [F(rng.randrange(-12, 13), rng.randrange(1, 4)) for _ in range(3)]
            low, high = rng.choice(ends), rng.choice(ends)
            if low is not None and high is not None and low > high:
                low, high = high, low
            count = sturm_count(p, low, high)
            args = dict(polynomial=polynomial(p), lower="-inf" if low is None else rational(low),
                        upper="+inf" if high is None else rational(high))
            with self.subTest(trial=trial, p=p, low=low, high=high):
                self.assertAccepts("count_real_roots", args, count)
                self.assertRejects("count_real_roots", args, count + 1)
        # Chebyshev's T_24, all of whose 24 roots are real and in (-1, 1), with large coefficients.
        t = [[F(1)], [F(0), F(1)]]
        for _ in range(23):
            t.append(trim(a - b for a, b in zip(times([F(0), F(2)], t[-1]), t[-2] + [F(0), F(0)])))
        chebyshev = polynomial(t[24])
        for lower, upper, count in (("-inf", "+inf", 24), (-1, 1, 24), (0, "+inf", 12), (Rational(1, 2), 1, 8)):
            with self.subTest(chebyshev=True, lower=lower, upper=upper):
                low = end(lower)
                high = end(upper)
                self.assertEqual(sturm_count(t[24], low, high), count)
                self.assertAccepts("count_real_roots", dict(polynomial=chebyshev, lower=lower, upper=upper), count)

    def test_complex_roots_accepts_the_known_values(self):
        one = Rational(1)
        i, minus_i = record(Rational(0), one, 1), record(Rational(0), -one, 1)
        sqrt3_half = [RealAlgebraicRoot((-3, 0, 4), k) for k in (0, 1)]
        cases = [
            # x^3 - 1: sorted by real part, the conjugate pair -1/2 -+ (sqrt 3/2) i comes before 1.
            ([-1, 0, 0, 1], (((-1, 1), 1), ((1, 1, 1), 1)),
             (record(Rational(-1, 2), sqrt3_half[0], 1), record(Rational(-1, 2), sqrt3_half[1], 1),
              record(one, Rational(0), 1))),
            # x^4 - 1: -1, -i, i, 1 by real part, then imaginary part.
            ([-1, 0, 0, 0, 1], (((-1, 1), 1), ((1, 1), 1), ((1, 0, 1), 1)),
             (record(-one, Rational(0), 1), minus_i, i, record(one, Rational(0), 1))),
            # (x^2 + 1)^2 (x - 2): -i (2), i (2), 2 (1).
            ([-2, 1, -4, 2, -2, 1], (((-2, 1), 1), ((1, 0, 1), 2)),
             (record(Rational(0), -one, 2), record(Rational(0), one, 2), record(Rational(2), Rational(0), 1))),
            # x^4 + 4 = (x^2 - 2x + 2)(x^2 + 2x + 2): -1 -+ i, then 1 -+ i.
            ([4, 0, 0, 0, 1], (((2, -2, 1), 1), ((2, 2, 1), 1)),
             (record(-one, -one, 1), record(-one, one, 1), record(one, -one, 1), record(one, one, 1))),
            # A nonzero constant has no root.
            ([5], (), ()),
        ]
        for coefficients, factors, value in cases:
            with self.subTest(coefficients=coefficients):
                p, built, evidence = complex_claim(F(coefficients[-1]), [(tuple(f), e) for f, e in factors])
                self.assertEqual(p, polynomial(coefficients))
                self.assertEqual(built, value)
                self.assertAccepts("complex_roots", dict(polynomial=p), value, evidence)
        # x^3 - 2 has an irreducible cubic factor: the result is Unevaluated, and no tuple passes.
        cubic = dict(polynomial=polynomial([-2, 0, 0, 1]))
        self.assertRejectsAll("complex_roots", [
            ("no root", cubic, (), (one, ())),
            ("a triple root 1", cubic, (record(one, Rational(0), 3),), (one, (((-one, one), 3),))),
            ("the cubic as a factor", cubic, (), (one, (((Rational(-2), Rational(0), Rational(0), one), 1),))),
        ])

    def test_complex_roots_accepts_true_claims(self):
        cases = [
            (F(3), (((F(-1, 3), 1), 1),)),  # 3x - 1.
            (F(-2), (((F(-5, 2), 1), 1),)),  # -2x + 5, a negative leading coefficient.
            (F(1), (((F(1, 2), F(-1, 3), 1), 1),)),  # x^2 + x/2 - 1/3: D = 19/12 > 0, no rational square.
            (F(1), (((5, 2, 1), 1),)),  # x^2 + 2x + 5 = (x + 1)^2 + 4: -1 -+ 2i, Rational parts.
            (F(1), (((F(1, 3), F(1, 2), 1), 1),)),  # D = -13/12: the imaginary parts name 48y^2 - 13.
            (F(-7, 3), (((F(1, 3), F(1, 2), 1), 1), ((F(-1, 2), 1), 3))),  # Rational coefficients, a triple root.
            # Real parts 0 shared by six roots and a root 0: ordered by imaginary part.
            (F(1), (((0, 1), 1), ((1, 0, 1), 1), ((4, 0, 1), 2), ((2, 0, 1), 1))),
            # Real part -1 shared: -1 - sqrt(2) i, -1 - i, -1, -1 + i, -1 + sqrt(2) i.
            (F(5), (((2, 2, 1), 1), ((3, 2, 1), 2), ((1, 1), 1))),
            # Real roots close to each other: 7/5 < 140/99 < sqrt 2 < 99/70, and -sqrt 3 < -3/2 < -sqrt 2.
            (F(1), (((-2, 0, 1), 1), ((-3, 0, 1), 1), ((F(-7, 5), 1), 1), ((F(-140, 99), 1), 2),
                    ((F(-99, 70), 1), 1), ((F(3, 2), 1), 1))),
            # Irrational roots of two fields 7e-7 apart: sqrt 2 and sqrt(1000001/500000).
            (F(1), (((-2, 0, 1), 1), ((F(-1000001, 500000), 0, 1), 1))),
            # Large coefficients.
            (F(2 ** 40 + 1, 3), (((3, 2 ** 80 + 1, 1), 2), ((-(2 ** 90), 1), 1), ((2 ** 60, 0, 1), 1))),
        ]
        for lead, factors in cases:
            with self.subTest(lead=lead, factors=factors):
                p, value, evidence = complex_claim(lead, factors)
                self.assertEqual(sum(root.multiplicity for root in value), len(p.coefficients) - 1)
                self.assertAccepts("complex_roots", dict(polynomial=p), value, evidence)
        # The records of each quadratic factor are those of v0.7's quadratic_roots, which the producer will use.
        for c, b in ((1, 1), (-2, 0), (5, 2), (F(1, 3), F(1, 2)), (F(-1, 3), F(1, 2)), (4, 0), (2, F(2, 3)),
                     (F(7, 25), F(-6, 5))):
            with self.subTest(quadratic=(c, b)):
                ours = sorted(factor_roots((c, b, 1)), key=cmp_to_key(lambda x, y: compare(x[2], y[2])
                                                                     or compare(x[3], y[3])))
                legacy = mf.quadratic_roots(1, rational(b), rational(c))
                self.assertEqual(tuple(complex_value(real, imag) for real, imag, _, _ in ours),
                                 tuple(root.root for root in legacy))

    def test_complex_roots_accepts_random_factorizations(self):
        rng = random.Random(20261114)
        for trial in range(60):
            keys, factors = set(), []
            for _ in range(rng.randrange(1, 6)):
                if rng.random() < 0.45:
                    coefficients: tuple[object, ...] = (F(rng.randrange(-12, 13), rng.randrange(1, 4)), 1)
                else:
                    c, b = F(rng.randrange(-15, 16), rng.randrange(1, 4)), F(rng.randrange(-8, 9), rng.randrange(1, 3))
                    if rational_sqrt(b * b - 4 * c) is not None:  # A reducible quadratic.
                        continue
                    coefficients = (c, b, 1)
                if factor_key(coefficients) not in keys:
                    keys.add(factor_key(coefficients))
                    factors.append((coefficients, rng.choice((1, 1, 1, 2, 3))))
            if not factors:
                continue
            lead = F(rng.choice((-1, 1)) * rng.randrange(1, 20), rng.randrange(1, 6))
            p, value, evidence = complex_claim(lead, factors)
            args = dict(polynomial=p)
            with self.subTest(trial=trial, lead=lead, factors=factors):
                self.assertAccepts("complex_roots", args, value, evidence)
                if len(value) > 1:
                    self.assertRejects("complex_roots", args, value[::-1], evidence)
                    self.assertRejects("complex_roots", args, value[:-1], evidence)

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_count_real_roots_rejects_claims_on_inputs_outside_the_domain(self):
        cubic = polynomial([0, -1, 0, 1])
        zero = polynomial([])
        self.assertEqual(zero.coefficients, (Rational(0),))
        cases = [(f"the zero polynomial, value {k}", dict(polynomial=zero, lower="-inf", upper="+inf"), k, ())
                 for k in range(3)]
        for lower, upper in ((1, Rational(1, 2)), (Rational(3, 2), 1), ("+inf", "+inf"), ("-inf", "-inf"),
                             ("+inf", 0), (0, "-inf"), (2, -2)):
            cases += [(f"lower {lower!r}, upper {upper!r}, value {k}",
                       dict(polynomial=cubic, lower=lower, upper=upper), k, ()) for k in range(4)]
        self.assertRejectsAll("count_real_roots", cases)

    def test_count_real_roots_rejects_corrupted_claims(self):
        cubic = polynomial([0, -1, 0, 1])
        args = dict(polynomial=cubic, lower=-1, upper=Rational(1, 2))  # The roots -1 and 0.
        single = dict(polynomial=cubic, lower=0, upper=Rational(1, 2))  # The root 0.
        self.assertAccepts("count_real_roots", args, 2)
        self.assertAccepts("count_real_roots", single, 1)
        forged_polynomial = support.forged(Polynomial, coefficients=cubic.coefficients + (Rational(0),), variable=X)
        self.assertRejectsAll("count_real_roots", [
            ("one root too many", args, 3, ()),
            ("one root too few", args, 1, ()),
            ("no root", args, 0, ()),
            ("the count of the open interval", args, 1, ()),
            ("above the degree", args, 4, ()),
            ("a negative count", args, -2, ()),
            ("a bool count", single, True, ()),
            ("a float count", args, 2.0, ()),
            ("a str count", args, "2", ()),
            ("a None count", args, None, ()),
            ("a Rational count", args, Rational(2), ()),
            ("an int-subclass count", args, Integer(2), ()),
            ("a tuple count", args, (2,), ()),
            ("evidence where none belongs", args, 2, ((),)),
            ("the count as evidence", args, 2, (2,)),
            ("evidence as a list", args, 2, []),
            ("evidence as a tuple subclass", args, 2, Evidence(())),
            ("evidence None", args, 2, None),
            ("evidence a str", args, 2, ""),
            ("a bool end", dict(args, lower=True), 1, ()),
            ("a float end", dict(args, lower=-1.0), 2, ()),
            ("a Fraction end", dict(args, upper=F(1, 2)), 2, ()),
            ("a str end", dict(args, upper="1/2"), 2, ()),
            ("'inf' as an end", dict(args, upper="inf"), 3, ()),
            ("a unicode minus", dict(args, lower="−inf"), 2, ()),
            ("'-inf ' with a space", dict(args, lower="-inf "), 2, ()),
            ("a None end", dict(args, lower=None), 2, ()),
            ("a list end", dict(args, lower=[-1]), 2, ()),
            ("an irrational end", dict(args, upper=sqrt(2)), 3, ()),
            ("a non-canonical Rational end", dict(args, upper=support.forged(Rational, numerator=2, denominator=4)),
             2, ()),
            ("a Rational end with a negative denominator",
             dict(args, upper=support.forged(Rational, numerator=-1, denominator=-2)), 2, ()),
            ("a Rational end without fields", dict(args, upper=support.forged(Rational)), 2, ()),
            ("a Rational end with a bool numerator",
             dict(args, upper=support.forged(Rational, numerator=True, denominator=2)), 2, ()),
            ("a Rational end with a float denominator",
             dict(args, upper=support.forged(Rational, numerator=1, denominator=2.0)), 2, ()),
            ("an impostor Rational end", dict(args, upper=support.impostor(Rational(1, 2))), 2, ()),
            ("an int-subclass end", dict(args, lower=Integer(-1)), 2, ()),
            ("a forged polynomial with str coefficients",
             dict(args, polynomial=support.forged(Polynomial, coefficients=("0", "-1", "0", "1"), variable=X)), 2, ()),
            ("a forged polynomial with a trailing zero", dict(args, polynomial=forged_polynomial), 2, ()),
            ("a forged polynomial with int coefficients",
             dict(args, polynomial=support.forged(Polynomial, coefficients=(0, -1, 0, 1), variable=X)), 2, ()),
            ("a forged polynomial with float coefficients",
             dict(args, polynomial=support.forged(Polynomial, coefficients=(0.0, -1.0, 0.0, 1.0), variable=X)), 2, ()),
            ("a forged polynomial with a list of coefficients",
             dict(args, polynomial=support.forged(Polynomial, coefficients=list(cubic.coefficients), variable=X)),
             2, ()),
            ("a forged polynomial with a str variable",
             dict(args, polynomial=support.forged(Polynomial, coefficients=cubic.coefficients, variable="x")), 2, ()),
            ("a forged polynomial without fields", dict(args, polynomial=support.forged(Polynomial)), 2, ()),
            ("an impostor polynomial", dict(args, polynomial=support.impostor(cubic)), 2, ()),
            ("the coefficients for the polynomial", dict(args, polynomial=cubic.coefficients), 2, ()),
            ("an expression for the polynomial", dict(args, polynomial=X * X * X - X), 2, ()),
        ])

    def test_complex_roots_rejects_corrupted_values(self):
        p, value, evidence = BASE
        args = dict(polynomial=p)
        self.assertAccepts("complex_roots", args, value, evidence)
        self.assertEqual([root.multiplicity for root in value], [1, 2, 1, 2, 1, 1, 1])
        sqrt2 = RealAlgebraicRoot((-2, 0, 1), 1)
        self.assertEqual(value[-1].root.real, sqrt2)
        self.assertEqual(value[2].root.real, Rational(-1, 2))
        self.assertEqual(value[3].root.imag, RealAlgebraicRoot((-3, 0, 4), 1))
        unevaluated = Unevaluated("solve", contracts.OperationRequest("complex_roots", tuple(args.items()), 6),
                                  Reason("no_applicable_rule", "an irreducible factor of degree three or more"))
        half_imag = value[1].root
        self.assertRejectsAll("complex_roots", [(label, args, corrupted, evidence) for label, corrupted in [
            ("a root missing", value[:-1]),
            ("a root too many", value + (record(Rational(5), Rational(0), 1),)),
            ("a root twice instead of another", with_root(value, 1, value[0])),
            ("a root twice, one too many", value[:1] + value),
            ("two roots of one real part swapped", value[:1] + (value[2], value[1]) + value[3:]),
            ("reversed", value[::-1]),
            ("rotated", value[1:] + value[:1]),
            ("multiplicity 1 for a double root", with_root(value, 1, record(half_imag.real, half_imag.imag, 1))),
            ("multiplicity 2 for a simple root", with_root(value, 0, record(value[0].root.real, Rational(0), 2))),
            ("sqrt 3 for sqrt 2", with_part(value, 6, real=RealAlgebraicRoot((-3, 0, 1), 1))),
            ("-sqrt 2 for sqrt 2", with_part(value, 6, real=RealAlgebraicRoot((-2, 0, 1), 0))),
            ("2 for -i", with_part(value, 4, real=Rational(2), imag=Rational(0))),
            ("the conjugate's sign flipped", with_part(value, 3, imag=RealAlgebraicRoot((-3, 0, 4), 0))),
            # Correct names of the right numbers that are not canonical.
            ("-1/2 as a root of 2x + 1", with_part(value, 2, real=RealAlgebraicRoot((1, 2), 0))),
            ("the imaginary part 1 as a root of x^2 - 1", with_part(value, 5, imag=RealAlgebraicRoot((-1, 0, 1), 1))),
            ("sqrt 2 as a root of (x^2 - 2)(x - 3)", with_part(value, 6, real=RealAlgebraicRoot((6, -2, -3, 1), 1))),
            ("sqrt 2 as a root of (x^2 - 2)(x^2 + 1)",
             with_part(value, 6, real=RealAlgebraicRoot((-2, 0, -1, 0, 1), 1))),
            ("sqrt 3/2 named by 8y^2 - 6", with_part(value, 3, imag=support.forged(
                RealAlgebraicRoot, integer_coefficients=(-6, 0, 8), real_index=1))),
            ("sqrt 3/2 named by -4y^2 + 3", with_part(value, 3, imag=support.forged(
                RealAlgebraicRoot, integer_coefficients=(3, 0, -4), real_index=0))),
            ("-1/2 as -2/4", with_part(value, 2, real=support.forged(Rational, numerator=-2, denominator=4))),
            ("-1/2 as 1/-2", with_part(value, 2, real=support.forged(Rational, numerator=1, denominator=-2))),
            ("a Rational without fields",  # Forged throughout: ComplexRoot's own __post_init__ reads the fields.
             forged_root(value, 2, root=complex_value(support.forged(Rational), Rational(0)))),
            # Exact types.
            ("an int imaginary part", with_part(value, 0, imag=0)),
            ("a float imaginary part", with_part(value, 0, imag=0.0)),
            ("a bool imaginary part", with_part(value, 0, imag=False)),
            ("a str imaginary part", with_part(value, 0, imag="0")),
            ("a Fraction real part", with_part(value, 2, real=F(-1, 2))),
            ("an impostor Rational real part", with_part(value, 2, real=support.impostor(Rational(-1, 2)))),
            ("an expression for sqrt 2", with_part(value, 6, real=sqrt(2))),
            ("multiplicity True", forged_root(value, 0, multiplicity=True)),
            ("multiplicity 0", forged_root(value, 0, multiplicity=0)),
            ("multiplicity -1", forged_root(value, 0, multiplicity=-1)),
            ("multiplicity 1.0", forged_root(value, 0, multiplicity=1.0)),
            ("multiplicity '1'", forged_root(value, 0, multiplicity="1")),
            ("a Rational multiplicity", forged_root(value, 0, multiplicity=Rational(1))),
            ("an int-subclass multiplicity", forged_root(value, 0, multiplicity=Integer(1))),
            ("a bool index", with_part(value, 6, real=support.forged(
                RealAlgebraicRoot, integer_coefficients=(-2, 0, 1), real_index=True))),
            ("a str index", with_part(value, 6, real=support.forged(
                RealAlgebraicRoot, integer_coefficients=(-2, 0, 1), real_index="1"))),
            ("bool coefficients", with_part(value, 6, real=support.forged(
                RealAlgebraicRoot, integer_coefficients=(-2, False, True), real_index=1))),
            ("str coefficients", with_part(value, 6, real=support.forged(
                RealAlgebraicRoot, integer_coefficients=("-2", "0", "1"), real_index=1))),
            ("a list of coefficients", with_part(value, 6, real=support.forged(
                RealAlgebraicRoot, integer_coefficients=[-2, 0, 1], real_index=1))),
            ("a RealAlgebraicRoot without fields",
             forged_root(value, 6, root=complex_value(support.forged(RealAlgebraicRoot), Rational(0)))),
            # Records of other classes and forged records.
            ("an impostor ComplexRoot", with_root(value, 0, support.impostor(value[0]))),
            ("an impostor ComplexAlgebraic", forged_root(value, 0, root=support.impostor(value[0].root))),
            ("an impostor RealAlgebraicRoot", with_part(value, 6, real=support.impostor(sqrt2))),
            ("a forged ComplexRoot without fields", with_root(value, 0, support.forged(ComplexRoot))),
            ("a ComplexRoot without its multiplicity", with_root(value, 0, support.forged(ComplexRoot,
                                                                                          root=value[0].root))),
            ("a ComplexAlgebraic without its imaginary part",
             forged_root(value, 0, root=support.forged(ComplexAlgebraic, real=value[0].root.real))),
            ("the bare ComplexAlgebraic for the record", with_root(value, 0, value[0].root)),
            ("a pair for the record", with_root(value, 0, (value[0].root, 1))),
            # Containers.
            ("a list", list(value)),
            ("a tuple subclass", Evidence(value)),
            ("None", None),
            ("one record", value[0]),
            ("an Unevaluated record", unevaluated),
            ("no root", ()),
        ]])

    def test_complex_roots_rejects_corrupted_evidence(self):
        p, value, evidence = BASE
        args = dict(polynomial=p)
        lead, factors = evidence
        self.assertEqual(lead, Rational(-3, 2))
        linear, minus_two, plus_one, double = factors
        self.assertEqual([f[0] for f in factors], [(Rational(1, 2), Rational(1)),
                                                   (Rational(-2), Rational(0), Rational(1)),
                                                   (Rational(1), Rational(0), Rational(1)),
                                                   (Rational(1), Rational(1), Rational(1))])
        one, half = Rational(1), Rational(1, 2)
        self.assertRejectsAll("complex_roots", [(label, args, value, corrupted) for label, corrupted in [
            ("no evidence", ()),
            ("the leading coefficient only", (lead,)),
            ("the factors only", (factors,)),
            ("an extra item", (lead, factors, ())),
            ("a list", [lead, factors]),
            ("a tuple subclass", Evidence(evidence)),
            ("None", None),
            ("the items swapped", (factors, lead)),
            ("the leading coefficient negated", (-lead, factors)),
            ("the leading coefficient doubled", (2 * lead, factors)),
            ("1 as the leading coefficient", (one, factors)),
            ("a Fraction leading coefficient", (F(-3, 2), factors)),
            ("a float leading coefficient", (-1.5, factors)),
            ("a str leading coefficient", ("-3/2", factors)),
            ("an impostor leading coefficient", (support.impostor(lead), factors)),
            ("a non-canonical leading coefficient", (support.forged(Rational, numerator=-6, denominator=4), factors)),
            ("factors as a list", (lead, list(factors))),
            ("an entry as a list", (lead, (list(linear),) + factors[1:])),
            ("coefficients as a list", (lead, ((list(linear[0]), 1),) + factors[1:])),
            ("an entry of three items", (lead, ((linear[0], 1, ()),) + factors[1:])),
            ("an entry of one item", (lead, ((linear[0],),) + factors[1:])),
            ("a bool exponent", (lead, ((linear[0], True),) + factors[1:])),
            ("exponent 0", (lead, ((linear[0], 0),) + factors[1:])),
            ("a negative exponent", (lead, ((linear[0], -1),) + factors[1:])),
            ("a float exponent", (lead, ((linear[0], 1.0),) + factors[1:])),
            ("a str exponent", (lead, ((linear[0], "1"),) + factors[1:])),
            ("a Rational exponent", (lead, ((linear[0], Rational(1)),) + factors[1:])),
            ("exponent 1 for the double factor", (lead, factors[:3] + ((double[0], 1),))),
            ("exponent 3 for the double factor", (lead, factors[:3] + ((double[0], 3),))),
            ("a factor missing", (lead, factors[1:])),
            ("an extra factor", (lead, ((((Rational(-5), one), 1),) + factors))),
            ("factors unsorted", (lead, (minus_two, linear, plus_one, double))),
            ("the quadratics unsorted", (lead, (linear, plus_one, minus_two, double))),
            ("the double factor twice", (lead, factors[:3] + ((double[0], 1), (double[0], 1)))),
            ("a non-monic factor", (lead, (((one, Rational(2)), 1),) + factors[1:])),
            ("a non-monic factor with the leading coefficient that makes the product right",
             (Rational(-3, 4), (((one, Rational(2)), 1),) + factors[1:])),
            ("an int coefficient", (lead, (((half, 1), 1),) + factors[1:])),
            ("a float coefficient", (lead, (((0.5, one), 1),) + factors[1:])),
            ("a Fraction coefficient", (lead, (((F(1, 2), one), 1),) + factors[1:])),
            ("a bool coefficient", (lead, (((half, True), 1),) + factors[1:])),
            ("a str coefficient", (lead, ((("1/2", one), 1),) + factors[1:])),
            ("an impostor coefficient", (lead, (((support.impostor(half), one), 1),) + factors[1:])),
            ("a non-canonical coefficient",
             (lead, (((support.forged(Rational, numerator=2, denominator=4), one), 1),) + factors[1:])),
            ("a trailing zero", (lead, (((half, one, Rational(0)), 1),) + factors[1:])),
            ("a constant factor", (lead, ((((one,), 1),) + factors))),
            ("a cubic factor with the right product",
             (lead, (((Rational(-1), Rational(-2), half, one), 1), plus_one, double))),
            ("a quartic factor with the right product",
             (lead, (linear, minus_two, ((one, one, Rational(2), one, one), 1), ((one, one, one), 1)))),
            ("a factor shifted by 2**61 - 1", (lead, (((half + 2 ** 61 - 1, one), 1),) + factors[1:])),
            ("the factors of another polynomial", (lead, (((Rational(-1, 2), one), 1),) + factors[1:])),
        ]])
        # Factorizations that multiply out to the polynomial with factors that are not irreducible.
        quartic = dict(polynomial=polynomial([-1, 0, 0, 0, 1]))
        _, roots, true_evidence = complex_claim(1, (((-1, 1), 1), ((1, 1), 1), ((1, 0, 1), 1)))
        square = dict(polynomial=polynomial([1, 2, 1]))
        one_root = (record(-one, Rational(0), 2),)
        self.assertAccepts("complex_roots", quartic, roots, true_evidence)
        self.assertAccepts("complex_roots", square, one_root, (one, (((one, one), 2),)))
        reducible = (one, (((Rational(-1), Rational(0), one), 1), true_evidence[1][2]))
        self.assertRejectsAll("complex_roots", [
            ("x^2 - 1 as one factor", quartic, roots, reducible),
            ("(x + 1)^2 as x^2 + 2x + 1", square, one_root, (one, (((one, Rational(2), one), 1),))),
            ("(x + 1)^2 as x^2 + 2x + 1, roots twice", square, (record(-one, Rational(0), 1),) * 2,
             (one, (((one, Rational(2), one), 1),))),
            ("(x^2 + 1)^2 for x^4 - 1", quartic, roots, (one, (((one, Rational(0), one), 2),))),
        ])

    def test_rejects_wrong_operations_and_arguments(self):
        p, value, evidence = BASE
        args = dict(polynomial=p)
        self.assertAccepts("complex_roots", args, value, evidence)
        self.assertRejectsAll("complex_roots", [
            ("arguments as pairs", tuple(args.items()), value, evidence),
            ("arguments as a list of pairs", list(args.items()), value, evidence),
            ("arguments as a read-only mapping", MappingProxyType(args), value, evidence),
            ("arguments as a dict subclass", Arguments(args), value, evidence),
            ("None as arguments", None, value, evidence),
            ("no argument", {}, value, evidence),
            ("an extra argument", dict(args, lower="-inf"), value, evidence),
            ("the argument renamed", dict(f=p), value, evidence),
            ("the zero polynomial", dict(polynomial=polynomial([])), (), (Rational(0), ())),
            ("a forged polynomial with a trailing zero",
             dict(polynomial=support.forged(Polynomial, coefficients=p.coefficients + (Rational(0),), variable=X)),
             value, evidence),
            ("an impostor polynomial", dict(polynomial=support.impostor(p)), value, evidence),
            ("a forged polynomial of degree 1025", dict(polynomial=support.forged(
                Polynomial, coefficients=(Rational(1),) * (MAX_POLYNOMIAL_DEGREE + 2), variable=X)), (),
             (Rational(1), ())),
        ])
        count_args = dict(polynomial=polynomial([-2, 0, 1]), lower="-inf", upper="+inf")
        for operation in ("count_real_roots", "Complex_roots", "complex_roots ", "complex", "roots", "", None, 7,
                          ("complex_roots",), b"complex_roots", ["complex_roots"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)
        for operation in ("complex_roots", "count_real_root", "sturm_sequence", None):
            with self.subTest(operation=operation, claim="count_real_roots"):
                self.assertRejects(operation, count_args, 2, ())
        self.assertAccepts("count_real_roots", count_args, 2)
        self.assertRejects("count_real_roots", dict(polynomial=p), value, evidence)  # Arguments of the other one.

    def test_oversized_forgeries_fail_fast(self):
        p, value, evidence = BASE
        args = dict(polynomial=p)
        lead, factors = evidence
        cubic = dict(polynomial=polynomial([0, -1, 0, 1]), lower="-inf", upper="+inf")
        power = polynomial(expand(1, (((-1, 1), 200),)))  # (x - 1)^200, whose coefficients reach 2**196.
        large = Rational(2 ** 60_000 + 1)  # Within max_integer_bits, far beyond Mignotte's bound for BASE.
        cases = [
            ("count_real_roots", cubic, GIANT, ()),
            ("count_real_roots", cubic, 10 ** 6, ()),
            ("count_real_roots", cubic, 3, (0,) * 100_000),
            ("count_real_roots", cubic, 3, (GIANT,)),
            ("count_real_roots", dict(cubic, lower=GIANT), 0, ()),
            ("count_real_roots", dict(cubic, lower=-GIANT), 3, ()),
            ("count_real_roots", dict(cubic, upper=Rational(1, GIANT)), 2, ()),
            ("count_real_roots", dict(cubic, polynomial=support.forged(
                Polynomial, coefficients=(Rational(1),) * 1_000_000, variable=X)), 0, ()),
            ("count_real_roots", dict(cubic, polynomial=support.forged(
                Polynomial, coefficients=(Rational(GIANT), Rational(1)), variable=X)), 1, ()),
            ("complex_roots", args, value * 20_000, evidence),
            ("complex_roots", args, value, (lead, factors * 30_000)),
            ("complex_roots", args, value, (lead, ((factors[0][0], GIANT),) + factors[1:])),
            ("complex_roots", args, value, (lead, ((factors[0][0], 10 ** 6),) + factors[1:])),
            ("complex_roots", args, value, (Rational(GIANT), factors)),
            ("complex_roots", args, value, (lead, (((Rational(GIANT), Rational(1)), 1),) + factors[1:])),
            ("complex_roots", args, value, (lead, (((large, Rational(1)), 1),) + factors[1:])),
            ("complex_roots", args, value, (lead, (((Rational(1, 2 ** 60_000), Rational(1)), 1),) + factors[1:])),
            ("complex_roots", args, value, (lead, factors[:3] + (((large, large, Rational(1)), 2),))),
            # Forged records throughout: ComplexRoot's own __post_init__ refuses parts this long.
            ("complex_roots", args, forged_root(value, 6, root=complex_value(support.forged(
                RealAlgebraicRoot, integer_coefficients=(GIANT, 0, 1), real_index=0), Rational(0))), evidence),
            ("complex_roots", args, forged_root(value, 6, root=complex_value(support.forged(
                RealAlgebraicRoot, integer_coefficients=(1,) * 1_000_000, real_index=0), Rational(0))), evidence),
            ("complex_roots", args, forged_root(value, 2, root=complex_value(Rational(-1, 2), Rational(1, GIANT))),
             evidence),
            ("complex_roots", args, forged_root(value, 0, multiplicity=GIANT), evidence),
            # A factor within the bound whose powers outgrow it: refused as soon as one power leaves the bound.
            ("complex_roots", dict(polynomial=power), (record(Rational(1 + 2 ** 61), Rational(0), 200),),
             (Rational(1), (((Rational(-1 - 2 ** 61), Rational(1)), 200),))),
        ]
        for index, (operation, case_args, case_value, case_evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, case_args, case_value, case_evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # The true claim for (x - 1)^200 passes, within the same bound.
        start = time.perf_counter()
        self.assertAccepts("complex_roots", dict(polynomial=power), (record(Rational(1), Rational(0), 200),),
                           (Rational(1), (((Rational(-1), Rational(1)), 200),)))
        self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_deterministically(self):
        """The charges depend on the claim alone: a budget one unit short of a check refuses it, its exact size
        passes, and a tiny budget runs out at once."""
        built = Rooted(F(3), ((F(1, 2), 2), (F(-3), 1)), (F(2), F(5)), ((F(7), F(1)),))
        count = built.count(F(-5, 2), None)
        self.assertEqual(count, 5)  # 1/2, -+sqrt 2 and -+sqrt 5, not -3.
        count_args = dict(polynomial=polynomial(built.coefficients()), lower=Rational(-5, 2), upper="+inf")
        p, value, evidence = complex_claim(F(5, 3), (((F(1, 2), F(-1, 3), 1), 2), ((F(1, 3), F(1, 2), 1), 1),
                                                     ((2, 0, 1), 1), ((F(-7, 4), 1), 3)))
        for operation, args, claim_value, claim_evidence in (("count_real_roots", count_args, count, ()),
                                                             ("complex_roots", dict(polynomial=p), value, evidence)):
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, claim_value, claim_evidence), True)
                work = budget.work
                self.assertGreater(work, 20)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, claim_value, claim_evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, claim_value, claim_evidence,
                                   limits=ComputationLimits(max_work=work))
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, claim_value, claim_evidence,
                                    limits=ComputationLimits(max_work=5))

    def test_larger_claims_fit_the_default_budget(self):
        rng = random.Random(20261115)
        factors, keys = [], set()
        while len(factors) < 24:
            c, b = F(rng.randrange(-200, 201), rng.randrange(1, 9)), F(rng.randrange(-60, 61), rng.randrange(1, 5))
            coefficients = (c, b, 1)
            if rational_sqrt(b * b - 4 * c) is None and factor_key(coefficients) not in keys:
                keys.add(factor_key(coefficients))
                factors.append((coefficients, rng.choice((1, 1, 2))))
        factors += [((F(k, 7), 1), 1) for k in range(-6, 7)]
        p, value, evidence = complex_claim(F(-11, 13), factors)
        self.assertGreater(len(p.coefficients), 60)
        start = time.perf_counter()
        self.assertAccepts("complex_roots", dict(polynomial=p), value, evidence)
        self.assertRejects("complex_roots", dict(polynomial=p), value[1:] + value[:1], evidence)
        self.assertLess(time.perf_counter() - start, 5 * FAST)

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_every_power_is_charged(self):
        """No builtin pow and no ** in the checker: every product it forms is charged before it is formed."""
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])

    def test_checker_imports_no_producer_code(self):
        tree = ast.parse(inspect.getsource(checker))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
        allowed = ("mathforge.certify", "mathforge.trusted", "mathforge.errors", "mathforge.limits", "mathforge.model")
        for module in imported:
            with self.subTest(module=module):
                self.assertTrue(module in ("__future__", "math") or module.startswith(allowed), module)

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layouts
        survive the format, and a wrong value fails the mathematics check."""
        p, value, evidence = BASE
        cubic = polynomial([0, -1, 0, 1])
        claims = [
            ("count_real_roots", dict(polynomial=cubic, lower="-inf", upper=Rational(1, 2)), 2, (), 3),
            ("count_real_roots", dict(polynomial=cubic, lower=-1, upper="+inf"), 3, (), 2),
            ("complex_roots", dict(polynomial=p), value, evidence, value[::-1]),
            ("complex_roots", dict(polynomial=polynomial([5])), (), (Rational(5), ()),
             (record(Rational(0), Rational(0), 1),)),
        ]
        for operation, args, claim_value, claim_evidence, wrong in claims:
            with self.subTest(operation=operation, args=args):
                self.roundtrip(claimed(operation, args, claim_value, claim_evidence))
                report = support.statuses(mf.verify(claimed(operation, args, wrong, claim_evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
