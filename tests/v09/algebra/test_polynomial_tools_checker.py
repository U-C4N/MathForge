"""The independent checker of mathforge.areas.algebra.polynomial_tools (mathforge.certify.algebra.polynomial_tools),
tested on its own before the family's producer exists.

Every true claim is computed here, never by the producer: by Lagrange's formula and by Gauss-Jordan elimination on the
Vandermonde system, with fractions.Fraction (two references that check each other), or by construction, a polynomial
chosen first and evaluated at the nodes by Horner's rule written here. The evidence is always (), the card's layout.
The checker answers False to every corrupted claim, never with an exception, and refuses an oversized forgery within
FAST seconds under the default budget.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
import dataclasses
from fractions import Fraction
import inspect
import random
import time
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.algebra import polynomial_tools as family
from mathforge.certify.algebra import polynomial_tools as checker
from mathforge.contracts import CheckStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Integers, Rational, Reals, Symbol, Truth
from mathforge.polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
         "and decimal digits")
SUMMARY = "Find the polynomial of least degree through given points with distinct rational abscissas."
X = Symbol("x")
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


def trimmed(coefficients: Iterable[Fraction]) -> list[Fraction]:
    """Coefficients without trailing zeros; [0] for the zero polynomial."""
    values = list(coefficients)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values or [Fraction(0)]


def times_linear(polynomial: Sequence[Fraction], root: Fraction) -> list[Fraction]:
    """polynomial * (x - root), coefficients constant first."""
    result = [Fraction(0)] * (len(polynomial) + 1)
    for index, coefficient in enumerate(polynomial):
        result[index + 1] += coefficient
        result[index] -= root * coefficient
    return result


def lagrange(xs: Sequence[Number], ys: Sequence[Number]) -> list[Fraction]:
    """The interpolating polynomial by Lagrange's formula, sum_i y_i prod_{j != i} (x - x_j) / (x_i - x_j)."""
    points, values = [frac(x) for x in xs], [frac(y) for y in ys]
    total = [Fraction(0)] * len(points)
    for i, (point, value) in enumerate(zip(points, values)):
        basis, scale = [Fraction(1)], Fraction(1)
        for j, other in enumerate(points):
            if j != i:
                basis = times_linear(basis, other)
                scale *= point - other
        for index, coefficient in enumerate(basis):
            total[index] += value * coefficient / scale
    return trimmed(total)


def vandermonde(xs: Sequence[Number], ys: Sequence[Number]) -> list[Fraction]:
    """The coefficients c_0, ..., c_(n-1) with sum_k c_k x_i**k = y_i, by Gauss-Jordan elimination on the Vandermonde
    matrix, which distinct nodes make invertible."""
    n = len(xs)
    rows = [[frac(x) ** k for k in range(n)] + [frac(y)] for x, y in zip(xs, ys)]
    for column in range(n):
        pivot = next(row for row in range(column, n) if rows[row][column] != 0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(n):
            if row != column and rows[row][column] != 0:
                factor = rows[row][column] / rows[column][column]
                rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
    return trimmed(rows[k][n] / rows[k][k] for k in range(n))


def lagrange_by_division(xs: Sequence[Number], ys: Sequence[Number]) -> list[Fraction]:
    """Lagrange's formula in a quadratic number of steps, for long inputs: the master polynomial prod_j (x - x_j) once,
    and each basis numerator prod_{j != i} (x - x_j) from it by synthetic division by x - x_i."""
    points, values = [frac(x) for x in xs], [frac(y) for y in ys]
    master = [Fraction(1)]
    for point in points:
        master = times_linear(master, point)
    total = [Fraction(0)] * len(points)
    for i, (point, value) in enumerate(zip(points, values)):
        scale = Fraction(1)
        for j, other in enumerate(points):
            if j != i:
                scale *= point - other
        factor, carry = value / scale, Fraction(0)
        for k in range(len(points), 0, -1):  # The quotient's coefficients, highest first; the remainder is 0.
            carry = master[k] + carry * point
            total[k - 1] += factor * carry
    return trimmed(total)


def evaluate(coefficients: Sequence[Number], point: Number) -> Fraction:
    """The polynomial at the point, by Horner's rule."""
    total, point = Fraction(0), frac(point)
    for coefficient in reversed(coefficients):
        total = total * point + frac(coefficient)
    return total


def polynomial(coefficients: Iterable[Number], variable: Symbol = X) -> Polynomial:
    """The Polynomial record of the coefficients (constant first) in the variable."""
    return Polynomial(tuple(Rational(frac(c).numerator, frac(c).denominator) for c in coefficients), variable)


def arguments(xs: Iterable[Number], ys: Iterable[Number], variable: object = X) -> dict[str, object]:
    """The arguments of interpolate as a request holds them: tuples of ints and Rationals, and the variable."""
    return {"xs": tuple(exact(x) for x in xs), "ys": tuple(exact(y) for y in ys), "variable": variable}


def interpolated(xs: Sequence[Number], ys: Sequence[Number], variable: Symbol = X) -> tuple[dict[str, object],
                                                                                             Polynomial]:
    """(args, value) of a true claim; the two references agree on the value."""
    reference = lagrange(xs, ys)
    if len(xs) <= 12:
        assert vandermonde(xs, ys) == reference
    return arguments(xs, ys, variable), polynomial(reference, variable)


def constructed(coefficients: Sequence[Number], xs: Sequence[Number]) -> tuple[dict[str, object], Polynomial]:
    """(args, value) of a true claim built from its value: the polynomial evaluated at distinct nodes, more of them
    than its coefficients."""
    assert len(set(map(frac, xs))) == len(xs) >= len(trimmed(map(frac, coefficients)))
    return arguments(xs, [evaluate(coefficients, x) for x in xs]), polynomial(coefficients)


def integer_values(coefficients: Sequence[int], xs: Sequence[int]) -> tuple[int, ...]:
    """The integer polynomial at integer nodes, by Horner's rule on ints (fast enough for the full size)."""
    values = []
    for x in xs:
        total = 0
        for coefficient in reversed(coefficients):
            total = total * x + coefficient
        values.append(total)
    return tuple(values)


def claimed(args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest("interpolate", tuple(args.items()), 6)
    step = contracts.Step("v6.interpolate", (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method="interpolate", request=request,
                            for_=args["variable"], domain=Reals, steps=(step,))


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values and arguments that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Fractional(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Coefficients(tuple):
    """A tuple subclass for a coefficient tuple, an argument list or the evidence."""


class Subpolynomial(Polynomial):
    """A Polynomial subclass: the right fields, another class."""


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def inexact(number: int | Rational) -> list[tuple[str, object]]:
    """(label, stand-in) for an exact number that is no exact int and no canonical Rational."""
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


class PolynomialToolsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, cases: Iterable[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case of interpolate."""
        for label, args, value, evidence in cases:
            with self.subTest(case=label):
                self.assertRejects("interpolate", args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_spec_of_the_card(self):
        expected = OperationSpec("interpolate", "algebra",
                                 (Argument("xs", "", None, "rational_list"), Argument("ys", "", None, "rational_list"),
                                  Argument("variable", "Symbol", None, "symbol")),
                                 "Polynomial", SUMMARY, SCOPE, version=6)
        self.assertEqual(family.SPECS, (expected,))
        self.assertEqual(family.SCOPE, SCOPE)
        self.assertEqual(family.TYPES, {})  # The value is the v0.7 Polynomial record.
        self.assertEqual(getattr(family, "RENDER", {}), {})  # The shared renderer presents a Polynomial.
        self.assertEqual(family.CHECKER, "mathforge.certify.algebra.polynomial_tools")
        self.assertEqual(registry.spec("interpolate"), expected)
        self.assertIs(registry.checker("interpolate"), checker)
        self.assertIs(support.checker(family), checker)
        # The checker names the record and its bound without importing mathforge.polynomial.
        self.assertEqual(checker.POLYNOMIAL, (Polynomial.__module__, Polynomial.__qualname__))
        self.assertEqual(checker.MAX_LENGTH, MAX_POLYNOMIAL_DEGREE + 1)
        self.assertEqual(checker.PRIME, 2 ** 61 - 1)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_accepts_the_known_values_of_the_card(self):
        half = Rational(1, 2)
        cases = [
            ((0, 1, 2), (1, 3, 7), (1, 1, 1)),  # x^2 + x + 1
            ((1, 2), (5, 5), (5,)),  # 5: two points on a constant
            ((0,), (3,), (3,)),  # 3: one point
            ((half, 1), (1, 0), (2, -2)),  # -2x + 2, through (1/2, 1) and (1, 0)
            ((half, 2), (0, 1), (Fraction(-1, 3), Fraction(2, 3))),  # The card's literal input: 2/3 x - 1/3.
        ]
        for xs, ys, coefficients in cases:
            with self.subTest(xs=xs, ys=ys):
                self.assertEqual(lagrange(xs, ys), [frac(c) for c in coefficients])
                self.assertEqual(vandermonde(xs, ys), [frac(c) for c in coefficients])
                self.assertAccepts("interpolate", arguments(xs, ys), polynomial(coefficients))
        # The card prints -2x + 2 for xs (1/2, 2) and ys (0, 1), but -2x + 2 takes 1 at 1/2 and -2 at 2.
        self.assertEqual((evaluate((2, -2), half), evaluate((2, -2), 2)), (1, -2))
        self.assertRejects("interpolate", arguments((half, 2), (0, 1)), polynomial((2, -2)))

    def test_accepts_special_cases(self):
        third = Rational(1, 3)
        cases = [
            ("one point on zero", (4,), (0,)),
            ("every ordinate zero", (-1, 0, 5, third), (0, 0, 0, 0)),
            ("one rational point", (third,), (Rational(-5, 3),)),
            ("ten points on a line", tuple(range(-3, 7)), tuple(3 * x - 2 for x in range(-3, 7))),
            ("thirty points on a parabola", tuple(range(30)), tuple(x * x - 7 for x in range(30))),
            ("unsorted negative and rational abscissas", (Rational(5, 2), -3, third, 0, Rational(-7, 4)),
             (1, Rational(-2, 9), 4, 0, 11)),
            ("long abscissas", (2 ** 200, -(2 ** 150) + 1, Rational(1, 2 ** 100)), (1, 2, 3)),
            ("long ordinates", (1, 2, 3), (2 ** 300, Rational(-1, 3 ** 90), 5)),
            ("a negative constant", (1, 2, 3, 4), (-6, -6, -6, -6)),
        ]
        for label, xs, ys in cases:
            with self.subTest(case=label):
                self.assertAccepts("interpolate", *interpolated(xs, ys))
        # Ints and Rationals, also with denominator 1, are the same exact numbers in a request.
        mixed = {"xs": (Rational(0), 1, Rational(2)), "ys": (1, Rational(3), 7), "variable": X}
        self.assertAccepts("interpolate", mixed, polynomial((1, 1, 1)))
        # The zero polynomial is the single zero coefficient.
        self.assertEqual(polynomial((0, 0, 0)).coefficients, (Rational(0),))
        self.assertAccepts("interpolate", arguments((1, 2), (0, 0)), polynomial((0,)))
        # Any Symbol is a variable: its name, domain and assumptions travel with it into the value.
        for variable in (Symbol("t"), Symbol("x", Integers), Symbol("y", assumptions=(Assumption("positive",
                                                                                              Truth.TRUE),))):
            with self.subTest(variable=variable):
                self.assertAccepts("interpolate", *interpolated((0, 1, 2), (1, 3, 7), variable))
        # More points than the record's coefficient bound, on a polynomial of low degree.
        many = tuple(range(-1000, 1000))
        self.assertAccepts("interpolate", *constructed((5, 0, -1, 2), many))

    def test_accepts_seeded_true_claims(self):
        rng = random.Random(20261112)

        def number(bits: int) -> Fraction:
            return Fraction(rng.randrange(-(2 ** bits), 2 ** bits + 1), rng.randrange(1, 2 ** bits + 1))

        for case in range(250):
            n = rng.randrange(1, 10)
            bits = rng.choice((2, 4, 16, 64))
            nodes: dict[Fraction, None] = {}
            while len(nodes) < n:
                nodes[number(bits)] = None
            xs = list(nodes)
            ys = [number(rng.choice((1, 3, 20))) if rng.random() > 0.1 else Fraction(0) for _ in range(n)]
            with self.subTest(case=case, xs=xs, ys=ys):
                self.assertAccepts("interpolate", *interpolated(xs, ys))
        for case in range(100):  # By construction: a polynomial first, then as many nodes as it needs or more.
            degree = rng.randrange(0, 12)
            coefficients = [number(rng.choice((2, 8, 30))) for _ in range(degree)] + [number(5) or Fraction(1)]
            nodes = {}
            while len(nodes) < degree + 1 + rng.randrange(0, 4):
                nodes[number(rng.choice((3, 10)))] = None
            args, value = constructed(coefficients, list(nodes))
            with self.subTest(case=f"constructed {case}"):
                self.assertEqual(value, polynomial(lagrange(args["xs"], args["ys"])))
                self.assertAccepts("interpolate", args, value)

    def test_accepts_long_rational_coefficients(self):
        """The interpolants of small ordinates at many nodes, whose coefficients are long rationals (what a producer
        returns for such data), verify under the default limits, and a change of one coefficient is refused."""
        rng = random.Random(20261117)
        for xs, ys in [((1, -2, Rational(1, 3), 5), (Rational(2, 7), 0, -1, 4)),
                       (tuple(range(12)), tuple(rng.randrange(-9, 10) for _ in range(12)))]:
            self.assertEqual(lagrange_by_division(xs, ys), lagrange(xs, ys))  # The two references agree.
        nodes: dict[Fraction, None] = {}
        while len(nodes) < 60:
            nodes[Fraction(rng.randrange(-60, 61), rng.randrange(1, 25))] = None
        cases = [("150 consecutive integers", tuple(range(-75, 75))), ("60 short rationals", tuple(nodes))]
        for label, xs in cases:
            ys = tuple(rng.randrange(-9, 10) for _ in xs)
            coefficients = lagrange_by_division(xs, ys)
            self.assertEqual(len(coefficients), len(xs))  # Degree n - 1 for these seeds.
            self.assertGreater(max(c.denominator.bit_length() for c in coefficients), 500)
            args, value = arguments(xs, ys), polynomial(coefficients)
            with self.subTest(case=label):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation("interpolate", args, value, ()), True)
                self.assertLess(budget.work, ComputationLimits().max_work // 4)
                wrong = polynomial(coefficients[:-1] + [coefficients[-1] + Fraction(1, 3)])
                start = time.perf_counter()
                self.assertRejects("interpolate", args, wrong)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_full_size_claim_fits_the_default_budget(self):
        """1,025 nodes and a polynomial of degree 1,024, the record's bound, verify under the default limits."""
        rng = random.Random(20261113)
        size = MAX_POLYNOMIAL_DEGREE + 1
        coefficients = [rng.randrange(-10 ** 6, 10 ** 6) for _ in range(size - 1)] + [rng.randrange(1, 10 ** 6)]
        xs = tuple(range(-(size // 2), size - size // 2))
        args = {"xs": xs, "ys": integer_values(coefficients, xs), "variable": X}
        value = polynomial(coefficients)
        with computation(fresh=True) as budget:
            start = time.perf_counter()
            self.assertIs(checker.check_operation("interpolate", args, value, ()), True)
            elapsed = time.perf_counter() - start
        self.assertLess(budget.work, ComputationLimits().max_work)
        self.assertLess(elapsed, 4 * FAST)
        wrong = polynomial(coefficients[:-1] + [coefficients[-1] + 1])
        start = time.perf_counter()
        self.assertRejects("interpolate", args, wrong)
        self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Wrong and non-canonical values

    def test_rejects_wrong_values(self):
        args, value = interpolated((0, 1, 2), (1, 3, 7))  # x^2 + x + 1
        self.assertAccepts("interpolate", args, value)
        other = Symbol("x")  # The same name, another identity.
        cases: list[tuple[str, object, object, object]] = [
            (f"coefficient {k} off by one", args,
             polynomial(c + (1 if i == k else 0) for i, c in enumerate((1, 1, 1))), ()) for k in range(3)]
        line_args, line = interpolated((Rational(1, 2), 2), (0, 1))  # 2/3 x - 1/3
        cases += [
            ("a line", args, polynomial((1, 3)), ()),
            ("the zero polynomial", args, polynomial((0,)), ()),
            ("the highest coefficient first", line_args, polynomial(reversed(line.coefficients)), ()),
            ("a constant", args, polynomial((1,)), ()),
            ("another Symbol named x", args, polynomial((1, 1, 1), other), ()),
            ("the variable with another domain", args, polynomial((1, 1, 1), dataclasses.replace(X, domain=Integers)),
             ()),
            ("the variable with an assumption", args,
             polynomial((1, 1, 1), dataclasses.replace(X, assumptions=(Assumption("nonzero", Truth.TRUE),))), ()),
            ("the right polynomial in another request variable", arguments((0, 1, 2), (1, 3, 7), other), value, ()),
            ("an expression", args, value.to_expression(), ()),
            ("the coefficient tuple", args, value.coefficients, ()),
            ("None", args, None, ()),
            ("a Rational", arguments((1, 2), (5, 5)), Rational(5), ()),
            ("an int", arguments((1, 2), (5, 5)), 5, ()),
        ]
        # Right at every node but the last, with degree n - 1.
        xs, ys = (Rational(-1, 2), 0, 3, 4, Rational(7, 3)), (2, -1, 0, 5, Rational(1, 9))
        args5, value5 = interpolated(xs, ys)
        bump = [Fraction(1)]
        for x in xs[:-1]:
            bump = times_linear(bump, frac(x))
        near = [frac(c) + b for c, b in zip(list(value5.coefficients) + [Rational(0)] * 5, bump)]
        self.assertEqual([evaluate(near, x) == frac(y) for x, y in zip(xs, ys)], [True] * 4 + [False])
        cases.append(("right at all nodes but the last", args5, polynomial(near), ()))
        # Right at every node, with degree n: not the polynomial of least degree.
        full = times_linear(bump, frac(xs[-1]))  # prod_i (x - x_i), of degree n.
        through = [frac(c) + b for c, b in zip(list(value5.coefficients) + [Rational(0)] * 6, full)]
        self.assertEqual(len(trimmed(through)), len(xs) + 1)
        self.assertTrue(all(evaluate(through, x) == frac(y) for x, y in zip(xs, ys)))
        cases.append(("right at every node with degree n", args5, polynomial(through), ()))
        self.assertRejectsAll(cases)

    def test_rejects_non_canonical_values(self):
        """Mathematically right values in a form that no Polynomial record holds: each forged with object.__new__."""
        args, value = interpolated((0, 1, 2), (1, 3, 7))
        one, zero = Rational(1), Rational(0)

        def forged(coefficients: object = (one, one, one), variable: object = X, **fields: object) -> Any:
            return support.forged(Polynomial, coefficients=coefficients, variable=variable, **fields)

        self.assertAccepts("interpolate", args, forged())  # The forging itself keeps a canonical record valid.
        zero_args = arguments((1, 2), (0, 0))
        five = arguments((0, 1, 2, 3, 4), (1, 3, 7, 13, 21))  # x^2 + x + 1 again: room for four coefficients.
        # Each of these holds the right polynomial, x^2 + x + 1 or zero, in a form that no record holds.
        right = [
            ("a trailing zero", five, (one, one, one, zero)),
            ("the zero polynomial as ()", zero_args, ()),
            ("the zero polynomial with two zeros", zero_args, (zero, zero)),
            ("zero as 0/5", zero_args, (forged_rational(0, 5),)),
            ("int coefficients", args, (1, 1, 1)),
            ("a list of coefficients", args, [one, one, one]),
            ("a tuple subclass of coefficients", args, Coefficients((one, one, one))),
            ("one as 2/2", args, (one, forged_rational(2, 2), one)),
            ("one as -1/-1", args, (one, one, forged_rational(-1, -1))),
        ]
        for label, claim_args, coefficients in right:
            with self.subTest(case=label, right=True):
                numbers = [Fraction(c.numerator, c.denominator) if type(c) is Rational else Fraction(c)
                           for c in coefficients]
                self.assertEqual(trimmed(numbers), lagrange(claim_args["xs"], claim_args["ys"]))
        cases: list[tuple[str, object, object, object]] = [
            (label, claim_args, forged(coefficients), ()) for label, claim_args, coefficients in right]
        cases += [
            ("no coefficients field", args, support.forged(Polynomial, variable=X), ()),
            ("no variable field", args, support.forged(Polynomial, coefficients=(one, one, one)), ()),
            ("no fields", args, support.forged(Polynomial), ()),
            ("the variable as a str", args, forged(variable="x"), ()),
            ("the variable as None", args, forged(variable=None), ()),
            ("a forged variable", args, forged(variable=support.forged(Symbol, name="x", domain=Reals,
                                                                       assumptions=(), uid=1)), ()),
            ("an impostor variable", args, forged(variable=support.impostor(X)), ()),
            ("an impostor Polynomial", args, support.impostor(value), ()),
            ("a Polynomial subclass", args, Subpolynomial((1, 1, 1), X), ()),
        ]
        for index in range(3):
            for label, item in inexact(1):
                coefficients = tuple(item if i == index else one for i in range(3))
                cases.append((f"coefficient {index}: {label}", args, forged(coefficients), ()))
        self.assertRejectsAll(cases)

    # -----------------------------------------------------------------------------------------------------------
    # Wrong arguments, evidence and operations

    def test_rejects_corrupted_arguments(self):
        args, value = interpolated((0, 1, 2), (1, 3, 7))
        cases: list[tuple[str, object, object, object]] = []
        for name in ("xs", "ys"):
            for index in range(3):
                for label, item in inexact(args[name][index]):
                    changed = args[name][:index] + (item,) + args[name][index + 1:]
                    cases.append((f"{name}[{index}]: {label}", dict(args, **{name: changed}), value, ()))
            cases += [(f"{name} as a list", dict(args, **{name: list(args[name])}), value, ()),
                      (f"{name} as a tuple subclass", dict(args, **{name: Coefficients(args[name])}), value, ()),
                      (f"{name} one point shorter", dict(args, **{name: args[name][:2]}), value, ()),
                      (f"{name} one point longer", dict(args, **{name: args[name] + (9,)}), value, ()),
                      (f"{name} as None", dict(args, **{name: None}), value, ())]
        constant = polynomial((2,))
        cases += [
            ("no point", arguments((), ()), constant, ()),
            ("a repeated abscissa", arguments((1, 1), (2, 2)), constant, ()),
            ("a repeated abscissa, int and Rational", {"xs": (2, Rational(2)), "ys": (2, 2), "variable": X}, constant,
             ()),
            ("a repeated rational abscissa", arguments((Rational(1, 2), 3, Rational(1, 2)), (2, 2, 2)), constant, ()),
            ("a repeated abscissa in higher terms", {"xs": (Rational(1, 2), forged_rational(2, 4)), "ys": (2, 2),
                                                     "variable": X}, constant, ()),
            ("the variable as a str", dict(args, variable="x"), value, ()),
            ("the variable as None", dict(args, variable=None), value, ()),
            ("an impostor variable", dict(args, variable=support.impostor(X)), value, ()),
            ("a forged variable", dict(args, variable=support.forged(Symbol, name="x")), value, ()),
            ("the arguments as pairs", tuple(args.items()), value, ()),
            ("the arguments as a list of pairs", list(args.items()), value, ()),
            ("the arguments as a read-only mapping", MappingProxyType(args), value, ()),
            ("None as the arguments", None, value, ()),
            ("an argument missing", {"xs": args["xs"], "ys": args["ys"]}, value, ()),
            ("an extra argument", dict(args, points=args["xs"]), value, ()),
            ("an argument renamed", {"x": args["xs"], "ys": args["ys"], "variable": X}, value, ()),
            ("a key that is no str", {"xs": args["xs"], "ys": args["ys"], 1: X}, value, ()),
        ]
        self.assertRejectsAll(cases)

    def test_rejects_wrong_evidence(self):
        args, value = interpolated((0, 1, 2), (1, 3, 7))
        differences = (1, 2, 1)  # Newton's divided differences of the points: consistent, but not the card's layout.
        self.assertRejectsAll([
            ("evidence None", args, value, None),
            ("evidence as a list", args, value, []),
            ("evidence as a tuple subclass", args, value, Coefficients()),
            ("an extra item", args, value, ((),)),
            ("a number", args, value, (0,)),
            ("a str", args, value, ("lagrange",)),
            ("the divided differences", args, value, (differences,)),
            ("the divided differences, flat", args, value, differences),
            ("the value itself", args, value, (value,)),
            ("a dict", args, value, {}),
        ])

    def test_rejects_wrong_operations(self):
        args, value = interpolated((0, 1, 2), (1, 3, 7))
        for operation in ("Interpolate", "interpolate ", "interpolation", "lagrange", "continued_fraction", "powmod",
                          "", None, 7, ("interpolate",), b"interpolate", ["interpolate"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value)

    def test_never_raises_on_junk(self):
        """Every slot of a claim filled with stand-ins of every kind: the checker answers False, never raising."""
        args, value = interpolated((0, 1, 2), (1, 3, 7))
        # Neither the true value nor the request's variable is junk: either would complete a true claim.
        junk: list[object] = [None, 0, 1.5, "x", b"x", (), [], {}, Rational(1), Symbol("x"), polynomial((1, 1)),
                              object(), True, support.forged(Polynomial), support.forged(Rational),
                              support.forged(Symbol), Coefficients((1,)), MappingProxyType({}), frozenset(),
                              Fraction(1, 2)]
        for index, item in enumerate(junk):
            with self.subTest(item=index, kind=type(item).__name__):  # A forged record without fields has no repr.
                self.assertRejects(item, args, value)
                self.assertRejects("interpolate", item, value)
                self.assertRejects("interpolate", args, item)
                self.assertRejects("interpolate", args, value, item if item != () else (item,))
                for name in ("xs", "ys", "variable"):
                    self.assertRejects("interpolate", dict(args, **{name: item}), value)

    # -----------------------------------------------------------------------------------------------------------
    # Oversized forgeries and the budget

    def test_oversized_forgeries_fail_fast(self):
        rng = random.Random(20261114)
        size = MAX_POLYNOMIAL_DEGREE + 1
        sevens = {"xs": tuple(range(size)), "ys": (7,) * size, "variable": X}  # The true value is the constant 7.
        # Pairwise different numerators and denominators of 60,000 bits: the modular filter refuses before any gcd.
        heavy = tuple(forged_rational(rng.getrandbits(60_000) | 1, rng.getrandbits(60_000) | 1 << 59_999)
                      for _ in range(size))
        right_at_zero = (Rational(7),) + heavy[1:]
        small = arguments((0, 1), (1, 1))
        cases = [
            ("100,000 coefficients", sevens, support.forged(Polynomial, coefficients=(Rational(7),) * 100_000,
                                                            variable=X), ()),
            ("1,026 coefficients for 2,000 nodes", {"xs": tuple(range(2000)), "ys": (7,) * 2000, "variable": X},
             support.forged(Polynomial, coefficients=(Rational(7),) + (Rational(0),) * 1024 + (Rational(1),),
                            variable=X), ()),
            ("a giant numerator", small, support.forged(Polynomial, coefficients=(forged_rational(GIANT),),
                                                        variable=X), ()),
            ("a giant denominator", small, support.forged(Polynomial, coefficients=(forged_rational(1, GIANT),),
                                                          variable=X), ()),
            ("1,025 heavy coefficients", sevens, support.forged(Polynomial, coefficients=heavy, variable=X), ()),
            ("1,025 heavy coefficients, right at 0", sevens,
             support.forged(Polynomial, coefficients=right_at_zero, variable=X), ()),
            ("giant evidence", small, polynomial((1,)), (0,) * 100_000),
            ("a giant evidence integer", small, polynomial((1,)), (GIANT,)),
            ("a long tuple as the value", small, tuple(range(100_000)), ()),
        ]
        for label, args, value, evidence in cases:
            with self.subTest(case=label):
                start = time.perf_counter()
                self.assertRejects("interpolate", args, value, evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # A request integer over max_integer_bits raises ResourceLimitError, as the request's inspection does.
        for args in (arguments((0, GIANT), (1, 1)), arguments((0, 1), (1, Rational(1, GIANT)))):
            start = time.perf_counter()
            with self.assertRaises(ResourceLimitError):
                support.verdict(family, "interpolate", args, polynomial((1,)))
            self.assertLess(time.perf_counter() - start, FAST)

    def test_a_claim_crafted_against_the_filter_is_never_verified(self):
        """Coefficients that agree with the truth modulo PRIME pass the filter, which never accepts: the exact
        evaluation refuses a light one, and a heavy one runs out of the budget at once, unverified."""
        args = arguments((0, 1, 2), (1, 3, 7))
        crafted = polynomial((1, 1 + checker.PRIME, 1))
        with computation(fresh=True) as budget:
            pairs = [(c.numerator, c.denominator) for c in crafted.coefficients]
            self.assertIs(checker._filter(budget, pairs, [(0, 1), (1, 1), (2, 1)], [(1, 1), (3, 1), (7, 1)]), True)
        self.assertRejects("interpolate", args, crafted)
        rng = random.Random(20261115)
        size = MAX_POLYNOMIAL_DEGREE + 1
        heavy = [Rational(7)] + [forged_rational(checker.PRIME * (rng.getrandbits(60_000) | 1),
                                                 rng.getrandbits(60_000) | 1 << 59_999 | 1) for _ in range(size - 1)]
        sevens = {"xs": tuple(range(size)), "ys": (7,) * size, "variable": X}
        start = time.perf_counter()
        self.assertRejects("interpolate", sevens, support.forged(Polynomial, coefficients=tuple(heavy), variable=X),
                           may_exhaust=True)
        self.assertLess(time.perf_counter() - start, FAST)

    def test_both_stages_are_live(self):
        """Each of the filter and the exact evaluation refuses wrong claims on its own."""
        args, value = interpolated((Rational(1, 2), 2, -3), (0, 1, Rational(4, 5)))
        wrong = polynomial(list(value.coefficients[:-1]) + [value.coefficients[-1] + 1])
        with patch.object(checker, "_filter", lambda budget, pairs, nodes, values: True):
            self.assertAccepts("interpolate", args, value)
            self.assertRejects("interpolate", args, wrong)
            self.assertRejects("interpolate", args, polynomial((1, 1 + checker.PRIME, 1)))
        with patch.object(checker, "_exact", lambda budget, pairs, nodes, values: True):
            self.assertAccepts("interpolate", args, value)
            self.assertRejects("interpolate", args, wrong)
        # The exact stage also refuses when a denominator is a multiple of PRIME, where the filter learns little.
        args_p, value_p = interpolated((0, 1), (Rational(1, checker.PRIME), Rational(2, checker.PRIME)))
        self.assertAccepts("interpolate", args_p, value_p)
        self.assertRejects("interpolate", args_p, polynomial((Rational(1, checker.PRIME), Rational(2, checker.PRIME))))

    def test_work_is_charged_before_it_runs(self):
        rng = random.Random(20261116)
        coefficients = [rng.randrange(-1000, 1000) for _ in range(199)] + [3]
        xs = tuple(range(-100, 100))
        args = {"xs": xs, "ys": integer_values(coefficients, xs), "variable": X}
        value = polynomial(coefficients)
        # The charges depend on the claim alone: a budget one unit short of the check refuses it, its exact size passes.
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation("interpolate", args, value, ()), True)
        work = budget.work
        self.assertGreater(work, 200 * 199)  # At least one unit per Horner step at every node.
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "interpolate", args, value, limits=ComputationLimits(max_work=work - 1))
        self.assertAccepts("interpolate", args, value, limits=ComputationLimits(max_work=work))
        # A small budget stops a large claim at once.
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "interpolate", args, value, limits=ComputationLimits(max_work=5_000))
        self.assertLess(time.perf_counter() - start, FAST)
        # Long integers cost more than short ones: the same shape with 20,000-bit coefficients.
        long = [c * (2 ** 20_000 + 1) for c in coefficients[:20]]
        args_long = {"xs": xs[:20], "ys": integer_values(long, xs[:20]), "variable": X}
        short = coefficients[:20]
        args_short = {"xs": xs[:20], "ys": integer_values(short, xs[:20]), "variable": X}
        charged = []
        for claim_args, claim in ((args_short, short), (args_long, long)):
            with computation(fresh=True) as budget:
                self.assertIs(checker.check_operation("interpolate", claim_args, polynomial(claim), ()), True)
            charged.append(budget.work)
        self.assertGreater(charged[1], 2 * charged[0])

    def test_no_power_is_formed_by_the_builtin(self):
        """The checker forms powers by repeated products inside Horner's rule: no pow call and no ** at all, so
        nothing escapes the charges; it imports nothing but the allowed layers and the standard library."""
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
        self.assertEqual(imported, {"__future__", "math", "mathforge.limits", "mathforge.model",
                                    "mathforge.trusted.integers"})

    # -----------------------------------------------------------------------------------------------------------
    # The verifier

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip, and a wrong value fails
        the mathematics check."""
        t = Symbol("t", Integers)
        claims = [
            (arguments((0, 1, 2), (1, 3, 7)), polynomial((1, 1, 1)), polynomial((1, 1, 2))),
            (arguments((Rational(1, 2), 2), (0, 1)), polynomial((Fraction(-1, 3), Fraction(2, 3))),
             polynomial((2, -2))),
            (arguments((5,), (0,), t), polynomial((0,), t), polynomial((1,), t)),
        ]
        for args, value, wrong in claims:
            with self.subTest(value=value):
                self.roundtrip(claimed(args, value))
                report = support.statuses(mf.verify(claimed(args, wrong)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
