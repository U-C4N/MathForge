"""The polynomial_tools family of mathforge.algebra: interpolate.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register algebra polynomial_tools

Every expected value comes from the card or from a reference of this module, never from the producer: Lagrange's
formula and Gauss-Jordan elimination on the Vandermonde system over fractions.Fraction (two references that check each
other), Lagrange's formula in a quadratic number of steps for longer inputs, or a polynomial chosen first and evaluated
at more nodes than it has coefficients. The card's fourth known value is misprinted: the line through (1/2, 0) and
(2, 1), the card's xs (1/2, 2) and ys (0, 1), is 2/3 x - 1/3, and -2x + 2 interpolates xs (1/2, 1) and ys (1, 0); both
are tested, as the checker's tests do. interpolate has no deferred input (no Unevaluated result): every valid input has
its polynomial, which only a degree above the record's bound (unsupported_operation) or the budget keeps from it.
tests/v09/algebra/test_polynomial_tools_checker.py tests the checker on its own.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable, Sequence
import dataclasses
from fractions import Fraction
import inspect
import math
import random
import time
from typing import Any
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.algebra import polynomial_tools as family
from mathforge.contracts import CheckStatus, ErrorInfo, ExecutionStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Integers, Rational, Reals, Symbol, Truth, sqrt
from mathforge.polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial

from .. import support

F = Fraction
Q = Rational
X = Symbol("x")
OPERATION = "interpolate"
FAST = 2.0  # Seconds: a request beyond its budget fails well within this.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
STAND_INS = (0.5, True, "1", None)  # No exact number: a float, a bool, a str and None.
VERIFIED, FAILED = CheckStatus.VERIFIED, CheckStatus.FAILED
SMALL = dict(xs=(0, 1, 2), ys=(1, 3, 7), variable=X)  # x^2 + x + 1, the card's first known value.

Number = int | Fraction | Rational

# The card's known values: (xs, ys, coefficients constant first).
KNOWN = (
    ((0, 1, 2), (1, 3, 7), (1, 1, 1)),  # x^2 + x + 1.
    ((1, 2), (5, 5), (5,)),  # 5: two points on a constant.
    ((0,), (3,), (3,)),  # 3: one point.
    ((Q(1, 2), 2), (0, 1), (F(-1, 3), F(2, 3))),  # The card's fourth input: 2/3 x - 1/3 (the card prints -2x + 2).
    ((Q(1, 2), 1), (1, 0), (2, -2)),  # -2x + 2, the card's fourth value: through (1/2, 1) and (1, 0).
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
    value = frac(number)
    return value.numerator if value.denominator == 1 else Rational(value.numerator, value.denominator)


def trimmed(coefficients: Iterable[Fraction]) -> list[Fraction]:
    """Coefficients without trailing zeros; [0] for the zero polynomial."""
    values = list(coefficients)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values or [Fraction(0)]


def times_linear(coefficients: Sequence[Fraction], root: Fraction) -> list[Fraction]:
    """The polynomial times (x - root), coefficients constant first."""
    result = [Fraction(0)] * (len(coefficients) + 1)
    for index, coefficient in enumerate(coefficients):
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
    """Lagrange's formula in a quadratic number of steps, for longer inputs: the master polynomial prod_j (x - x_j)
    once, and each basis numerator prod_{j != i} (x - x_j) from it by synthetic division by x - x_i."""
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
    total, at = Fraction(0), frac(point)
    for coefficient in reversed(coefficients):
        total = total * at + frac(coefficient)
    return total


def integer_values(coefficients: Sequence[int], xs: Sequence[int]) -> tuple[int, ...]:
    """The integer polynomial at integer nodes, by Horner's rule on ints (fast enough for the full size)."""
    values = []
    for x in xs:
        total = 0
        for coefficient in reversed(coefficients):
            total = total * x + coefficient
        values.append(total)
    return tuple(values)


def polynomial(coefficients: Iterable[Number], variable: Symbol = X) -> Polynomial:
    """The Polynomial record of the coefficients (constant first) in the variable."""
    return Polynomial(tuple(Rational(frac(c).numerator, frac(c).denominator) for c in coefficients), variable)


def arguments(xs: Iterable[Number], ys: Iterable[Number], variable: object = X) -> dict[str, object]:
    """The arguments of interpolate as a request holds them: tuples of ints and Rationals, and the variable."""
    return {"xs": tuple(exact(x) for x in xs), "ys": tuple(exact(y) for y in ys), "variable": variable}


def distinct(rng: random.Random, count: int, draw: Any) -> list[Fraction]:
    """count pairwise distinct numbers from draw(rng), in the order drawn."""
    found: dict[Fraction, None] = {}
    while len(found) < count:
        found[frac(draw(rng))] = None
    return list(found)


def random_rational(rng: random.Random, bits: int) -> Fraction:
    """A random rational with numerator and denominator of at most bits bits."""
    return Fraction(rng.randrange(-(2 ** bits), 2 ** bits + 1), rng.randrange(1, 2 ** bits + 1))


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for inputs that are no exact rationals or no Symbol.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class SubRational(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Subpolynomial(Polynomial):
    """A Polynomial subclass: the right fields, another class."""


def forged_rational(numerator: object, denominator: object = 1, **missing: str) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything; missing="numerator" (or
    "denominator") leaves that slot unset."""
    fields = {"numerator": numerator, "denominator": denominator}
    for name in missing.values():
        fields.pop(name)
    return support.forged(Rational, **fields)


class PolynomialToolsTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with the evidence () and its four checks, which mf.verify accepts."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertEqual(result.value, value)
        self.assertIs(type(result.value), Polynomial)
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertProduces(self, args: dict[str, object], coefficients: Sequence[Number]) -> Polynomial:
        """produce() answers the polynomial of these coefficients in the request's variable, with the evidence (),
        every coefficient an exact canonical Rational, and the checker accepts the claim."""
        value, evidence = family.produce(OPERATION, dict(args))
        expected = polynomial(coefficients, args["variable"])
        self.assertEqual(value, expected, args)
        self.assertEqual(evidence, ())
        self.assertIs(value.variable, args["variable"])
        self.assertTrue(all(type(c) is Rational and type(c.numerator) is int and type(c.denominator) is int
                            for c in value.coefficients))
        self.assertIs(support.verdict(family, OPERATION, args, value, evidence), True, args)
        return value

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts: the card's, with its fourth
        input and value each in its own request, the zero polynomial and a variable with a domain and an assumption."""
        for xs, ys, coefficients in KNOWN:
            with self.subTest(xs=xs, ys=ys):
                expected = [frac(c) for c in coefficients]
                self.assertEqual(lagrange(xs, ys), expected)
                self.assertEqual(vandermonde(xs, ys), expected)
                result = support.run(OPERATION, xs=xs, ys=ys, variable=X)
                self.assertVerified(result, polynomial(coefficients))
                self.assertEqual(result.for_, X)
                self.roundtrip(result)
        # The card prints -2x + 2 for xs (1/2, 2) and ys (0, 1), but -2x + 2 takes 1 at 1/2 and -2 at 2.
        self.assertEqual((evaluate((2, -2), Q(1, 2)), evaluate((2, -2), 2)), (1, -2))
        self.assertNotEqual(support.run(OPERATION, xs=(Q(1, 2), 2), ys=(0, 1), variable=X).value, polynomial((2, -2)))
        t = Symbol("t", Integers, (Assumption("positive", Truth.TRUE),))
        special = (
            ((1, 2), (0, 0), (0,), X),  # Every ordinate 0: the zero polynomial, Rational(0) alone.
            ((-1, 0, 5, Q(1, 3)), (0, 0, 0, 0), (0,), X),
            ((Q(1, 3),), (Q(-5, 3),), (F(-5, 3),), X),  # One rational point.
            ((0, 1, 2, 3), (0, 1, 8, 27), (0, 0, 0, 1), X),  # x^3 through four of its points.
            ((0, 1, 2), (1, 3, 7), (1, 1, 1), t),  # The variable, its domain and assumption travel into the value.
        )
        for xs, ys, coefficients, variable in special:
            with self.subTest(xs=xs, ys=ys, variable=variable.name):
                self.assertEqual(lagrange(xs, ys), [frac(c) for c in coefficients])
                result = support.run(OPERATION, xs=xs, ys=ys, variable=variable)
                self.assertVerified(result, polynomial(coefficients, variable))
                self.assertEqual(result.value.variable, variable)
                self.roundtrip(result)
        self.assertEqual(support.run(OPERATION, xs=(1, 2), ys=(0, 0), variable=X).value.coefficients, (Q(0),))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument and in every place of xs and ys; so are the card's cases (different lengths, no point, a repeated
        abscissa, also 2 beside Rational(2)), and Rationals or Symbols made with object.__new__ outside their canonical
        form, which run() admits by their class. produce() tests the types itself too."""
        matrix = support.oversized_matrix()  # No argument takes a matrix: in any of them it is invalid input.
        containers = (*STAND_INS, 7, Q(1, 2), X, matrix, polynomial((1, 1)), "12", {1: 2})
        for name in ("xs", "ys"):
            for item in containers:
                with self.subTest(argument=name, item=type(item).__name__):
                    self.assertInvalidInput(support.run(OPERATION, **dict(SMALL, **{name: item})))
            for index in range(3):
                for item in (*STAND_INS, (1,), [1], F(1, 2), Integer(1), SubRational(1, 2), sqrt(2), X, matrix,
                             forged_rational(2, 4), forged_rational(-1, -2), forged_rational(1, 0),
                             forged_rational(True, 1), forged_rational(3, -1)):
                    changed = SMALL[name][:index] + (item,) + SMALL[name][index + 1:]
                    with self.subTest(argument=name, index=index, item=type(item).__name__):
                        self.assertInvalidInput(support.run(OPERATION, **dict(SMALL, **{name: changed})))
        for item in (*STAND_INS, (1, 2), [1], "x", 1, Q(1), polynomial((1, 1)), matrix, support.impostor(X),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=(), uid=1)):
            with self.subTest(argument="variable", item=type(item).__name__):
                self.assertInvalidInput(support.run(OPERATION, **dict(SMALL, variable=item)))
        card = (
            (dict(xs=(), ys=(), variable=X), "at least one point"),
            (dict(xs=(1, 2), ys=(1,), variable=X), "the same length"),
            (dict(xs=(1,), ys=(1, 2), variable=X), "the same length"),
            (dict(xs=(), ys=(4,), variable=X), "the same length"),
            (dict(xs=(1, 1), ys=(1, 2), variable=X), "pairwise distinct abscissas"),
            (dict(xs=(2, Q(2)), ys=(1, 1), variable=X), "pairwise distinct abscissas"),
            (dict(xs=(Q(1, 2), 3, Q(1, 2)), ys=(1, 2, 3), variable=X), "pairwise distinct abscissas"),
            (dict(xs=(Q(-7, 3), 0, Q(-14, 6)), ys=(1, 2, 3), variable=X), "pairwise distinct abscissas"),
            (dict(xs=tuple(range(50)) + (49,), ys=(0,) * 51, variable=X), "pairwise distinct abscissas"),
        )
        for args, text in card:
            with self.subTest(**{name: repr(item) for name, item in args.items() if name != "variable"}):
                result = support.run(OPERATION, **args)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
                with self.assertRaises(InvalidInput):
                    family.produce(OPERATION, dict(args))
        for item, text in ((forged_rational(2, 4), "lowest terms"), (forged_rational(-1, -2), "canonical Rationals")):
            with self.subTest(forged=(item.numerator, item.denominator)):
                self.assertIn(text, support.run(OPERATION, **dict(SMALL, ys=(1, item, 7))).error.message)
        # produce() itself refuses every stand-in, also those that run() does not bring to it: a float slot, a missing
        # slot and a Symbol without fields, which run()'s inspection of the arguments meets first, and a Symbol whose
        # domain is a str, with which run() cannot even build its failure Result (operations._failure raises
        # TypeError on it, outside this family); other argument sets and other operations.
        for name in ("xs", "ys"):
            for item in (*containers, frozenset({1, 2}), b"12"):
                with self.subTest(argument=name, item=type(item).__name__, path="produce"):
                    with self.assertRaises(InvalidInput):
                        family.produce(OPERATION, dict(SMALL, **{name: item}))
            for item in (*STAND_INS, (1,), F(1, 2), Integer(1), SubRational(1, 2), sqrt(2), X, forged_rational(2, 4),
                         forged_rational(1, 0), forged_rational(True, 1), forged_rational(1.0), forged_rational(1, 2.0),
                         forged_rational(1, missing="numerator"), forged_rational(1, missing="denominator")):
                with self.subTest(argument=name, item=repr(item) if type(item) is not Rational else "forged",
                                  path="produce"):
                    with self.assertRaises(InvalidInput):
                        family.produce(OPERATION, dict(SMALL, **{name: (0, item, 2)}))
        for item in (*STAND_INS, "x", support.impostor(X), support.forged(Symbol, name="x"),
                     support.forged(Symbol, name="x", domain="reals", assumptions=(), uid=X.uid),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=[], uid=X.uid),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=("positive",), uid=X.uid),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=(Assumption("positive"),) * 4,
                                    uid=X.uid)):
            with self.subTest(argument="variable", item=type(item).__name__, path="produce"):
                with self.assertRaises(InvalidInput):
                    family.produce(OPERATION, dict(SMALL, variable=item))
        for wrong in (dict(SMALL, extra=1), {"xs": (0,), "ys": (1,)}, {}, tuple(SMALL.items()), None,
                      list(SMALL.items()), {"x": (0,), "ys": (1,), "variable": X}):
            with self.subTest(arguments=repr(wrong)[:40]), self.assertRaises(InvalidInput):
                family.produce(OPERATION, wrong)
        for operation in ("Interpolate", "interpolate ", "interpolation", "lagrange", "cyclotomic_polynomial", "",
                          None, 1):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(SMALL))
        # Lists are what a caller of produce() may pass, as the kind admits them: run() freezes them into tuples.
        self.assertEqual(family.produce(OPERATION, dict(SMALL, xs=[0, 1, 2], ys=[1, 3, 7])),
                         (polynomial((1, 1, 1)), ()))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OPERATION, xs=(1, 1), ys=(1, 2), variable=X)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in xs or ys (also inside a Rational) and max_work=1 exceed the budget. No argument
        takes a matrix: support.oversized_matrix() in any argument is invalid input (test_invalid_inputs). A value with
        a coefficient longer than max_integer_bits, and a large request beyond a small budget, fail at once."""
        for name in ("xs", "ys"):
            for item in (support.HUGE, -support.HUGE, Q(support.HUGE + 1, 3), Q(1, support.HUGE)):
                changed = SMALL[name][:1] + (item,) + SMALL[name][2:]
                with self.subTest(argument=name, item=type(item).__name__):
                    self.assertResourceLimit(support.run(OPERATION, **dict(SMALL, **{name: changed})))
                    with self.assertRaises(ResourceLimitError):
                        family.produce(OPERATION, dict(SMALL, **{name: changed}))
        for args in (SMALL, dict(xs=(0,), ys=(0,), variable=X), dict(xs=(Q(1, 2), 2), ys=(0, 1), variable=X)):
            with self.subTest(max_work=1, args=args):
                self.assertResourceLimit(support.run(OPERATION, limits=ComputationLimits(max_work=1), **args))
                with computation(ComputationLimits(max_work=1), fresh=True), self.assertRaises(ResourceLimitError):
                    family.produce(OPERATION, dict(args))
        # Three nodes 0, H, 2H with H = 2**40000: x(x - H)/(2H**2) has a denominator of 80,001 bits, no Rational of
        # the default limits; the producer refuses it as soon as it forms it.
        h = 2 ** 40_000
        self.assertEqual(lagrange((0, 1, 2), (0, 0, 1)), [0, F(-1, 2), F(1, 2)])
        start = time.perf_counter()
        self.assertResourceLimit(support.run(OPERATION, xs=(0, h, 2 * h), ys=(0, 0, 1), variable=X))
        rng = random.Random(20261170)
        ys = tuple(rng.randrange(-9, 10) for _ in range(600))
        self.assertResourceLimit(support.run(OPERATION, xs=tuple(range(600)), ys=ys, variable=X,
                                             limits=ComputationLimits(max_work=50_000)))
        nodes = tuple(rng.sample(range(-3000, 3001), 400))
        self.assertResourceLimit(support.run(OPERATION, xs=nodes, ys=ys[:400], variable=X,
                                             limits=ComputationLimits(max_work=100_000)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The evidence is always () and the value is unique, so a true claim that the producer did not make is one
        built by hand: from a chosen polynomial evaluated at more nodes than it has coefficients, or on arguments that
        write the same points another way (an integral number as an int or as a Rational, another order of the
        points). The checker accepts each, and the producer, asked, answers the same."""
        rng = random.Random(20261171)
        for case in range(40):
            degree = rng.randrange(0, 8)
            coefficients = [random_rational(rng, rng.choice((2, 8, 30))) for _ in range(degree)]
            coefficients.append(random_rational(rng, 5) or F(1))
            xs = distinct(rng, degree + 1 + rng.randrange(0, 5), lambda r: random_rational(r, r.choice((3, 10))))
            args = arguments(xs, [evaluate(coefficients, x) for x in xs])
            with self.subTest(case=case, coefficients=coefficients):
                self.assertAccepts(OPERATION, args, polynomial(coefficients))
                self.assertProduces(args, coefficients)
                order = list(range(len(xs)))
                rng.shuffle(order)
                shuffled = dict(args, xs=tuple(args["xs"][i] for i in order), ys=tuple(args["ys"][i] for i in order))
                self.assertAccepts(OPERATION, shuffled, polynomial(coefficients))
                self.assertProduces(shuffled, coefficients)
        as_rationals = {"xs": (Q(0), 1, Q(2)), "ys": (1, Q(3), Q(7)), "variable": X}
        self.assertAccepts(OPERATION, as_rationals, polynomial((1, 1, 1)))
        self.assertProduces(as_rationals, (1, 1, 1))
        # More points than coefficients by far: 2,000 on a cubic, beyond the record's bound of 1,025 coefficients.
        many = tuple(range(-1000, 1000))
        args = {"xs": many, "ys": integer_values((5, 0, -1, 2), many), "variable": X}
        self.assertAccepts(OPERATION, args, polynomial((5, 0, -1, 2)))
        self.assertProduces(args, (5, 0, -1, 2))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, an int where a Rational belongs, a list for a tuple, missing or extra evidence, a
        wrong operation name and oversized forged evidence, all on a Result of the producer; mf.verify fails the
        mathematics of a Result whose value or evidence was replaced, while its trace stays verified."""
        xs, ys = (Q(-1, 2), 0, 3, 4, Q(7, 3)), (2, -1, 0, 5, Q(1, 9))
        args = arguments(xs, ys)
        result = support.run(OPERATION, **args)
        coefficients = lagrange(xs, ys)
        value = polynomial(coefficients)
        self.assertVerified(result, value)
        self.assertEqual(len(value.coefficients), 5)
        bump = [F(1)]
        for x in xs[:-1]:
            bump = times_linear(bump, frac(x))  # Zero at every node but the last.
        full = times_linear(bump, frac(xs[-1]))  # prod (x - x_i), zero at every node, of degree n.
        padded = coefficients + [F(0)] * 2
        near = [c + b for c, b in zip(padded, bump)]
        through = [c + b for c, b in zip(padded, full)]
        self.assertTrue(all(evaluate(through, x) == frac(y) for x, y in zip(xs, ys)))
        self.assertEqual([evaluate(near, x) == frac(y) for x, y in zip(xs, ys)], [True] * 4 + [False])
        zero = Q(0)
        wrong_values = [(f"coefficient {k} plus 1/7", polynomial(c + (F(1, 7) if i == k else 0)
                                                                 for i, c in enumerate(coefficients)))
                        for k in range(5)]
        wrong_values += [
            ("right at every node but the last", polynomial(near)),
            ("right at every node with degree n", polynomial(through)),
            ("the zero polynomial", polynomial((0,))), ("a constant", polynomial((2,))),
            ("the coefficients reversed", polynomial(reversed(coefficients))),
            ("another Symbol named x", polynomial(coefficients, Symbol("x"))),
            ("the variable with another domain", polynomial(coefficients, dataclasses.replace(X, domain=Integers))),
            ("an expression", value.to_expression()), ("the coefficient tuple", value.coefficients), ("None", None),
            ("a Rational", Q(2)), ("an int", 2),
            ("a trailing zero", support.forged(Polynomial, coefficients=value.coefficients + (zero,), variable=X)),
            ("int coefficients", support.forged(Polynomial, coefficients=(1, 1, 1), variable=X)),
            ("a list of coefficients", support.forged(Polynomial, coefficients=list(value.coefficients), variable=X)),
            ("a coefficient in higher terms",
             support.forged(Polynomial, coefficients=(forged_rational(2 * value.coefficients[0].numerator,
                                                                      2 * value.coefficients[0].denominator),)
                            + value.coefficients[1:], variable=X)),
            ("no fields", support.forged(Polynomial)), ("an impostor Polynomial", support.impostor(value)),
            ("a Polynomial subclass", Subpolynomial(value.coefficients, X)),
            ("a variable as a str", support.forged(Polynomial, coefficients=value.coefficients, variable="x")),
        ]
        for label, item in wrong_values:
            with self.subTest(value=label):
                self.assertRejects(OPERATION, args, item, ())
        heads = (2, -6, 1)  # Numbers that look like Newton's coefficients: evidence of no layout of the card.
        for label, evidence in (("None", None), ("a list", []), ("an extra item", ((),)), ("a number", (0,)),
                                ("a str", ("newton",)), ("the value", (value,)), ("heads", (heads,)),
                                ("heads, flat", heads), ("a dict", {}), ("oversized", (0,) * 100_000),
                                ("a giant integer", (GIANT,))):
            with self.subTest(evidence=label):
                self.assertRejects(OPERATION, args, value, evidence)
        for operation in ("Interpolate", "interpolation", "lagrange", "cyclotomic_polynomial", "", None):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, ())
        report = support.statuses(mf.verify(support.tampered(result, polynomial(near))))
        self.assertIs(report["result.trace"], VERIFIED)
        self.assertIs(report["result.mathematics"], FAILED)
        report = support.statuses(mf.verify(support.tampered(result, value, (heads,))))
        self.assertIs(report["result.mathematics"], FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [dict(xs=xs, ys=ys, variable=X) for xs, ys, _ in KNOWN]
        requests += [dict(xs=(1, 2), ys=(0, 0), variable=X),
                     dict(xs=(Q(-1, 2), 0, 3, 4, Q(7, 3)), ys=(2, -1, 0, 5, Q(1, 9)), variable=Symbol("t")),
                     dict(xs=tuple(range(-1000, 1000)), ys=integer_values((5, 0, -1, 2), range(-1000, 1000)),
                          variable=X)]
        results = [support.run(OPERATION, **args) for args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for args, result, text in zip(requests, results, texts):
                with self.subTest(xs=args["xs"][:5]):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(OPERATION, **SMALL)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a monic polynomial,
        rational coefficients with signs, a constant, zero, a single power and another variable."""
        t = Symbol("t")
        rendered = (
            (dict(xs=(0, 1, 2), ys=(1, 3, 7), variable=X), "x^2 + x + 1", "x^{2} + x + 1"),
            (dict(xs=(Q(1, 2), 2), ys=(0, 1), variable=X), "2/3*x - 1/3", r"\frac{2}{3}\, x - \frac{1}{3}"),
            (dict(xs=(Q(1, 2), 1), ys=(1, 0), variable=X), "-2*x + 2", r"-2\, x + 2"),
            (dict(xs=(0, 1, 2, 3), ys=(1, 0, 1, Q(-1, 2)), variable=X), "-3/4*x^3 + 13/4*x^2 - 7/2*x + 1",
             r"-\frac{3}{4}\, x^{3} + \frac{13}{4}\, x^{2} - \frac{7}{2}\, x + 1"),
            (dict(xs=(0, 1, 2, 3), ys=(0, 1, 8, 27), variable=X), "x^3", "x^{3}"),
            (dict(xs=(1, 2), ys=(5, 5), variable=X), "5", "5"),
            (dict(xs=(Q(1, 3),), ys=(Q(-5, 3),), variable=t), "-5/3", r"-\frac{5}{3}"),
            (dict(xs=(1, 2), ys=(0, 0), variable=X), "0", "0"),
            (dict(xs=(1, 2, 3), ys=(1, 4, 9), variable=t), "t^2", "t^{2}"),
        )
        for args, text, latex in rendered:
            with self.subTest(text=text):
                result = support.run(OPERATION, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual((mf.render_text(result.value), mf.to_latex(result.value)), (text, latex))
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        data = mf.to_data(support.run(OPERATION, xs=(Q(1, 2), 2), ys=(0, 1), variable=X).value)["data"]
        self.assertEqual(data["type"], "polynomial")
        self.assertEqual(data["coefficients"], [{"type": "rational", "numerator": "-1", "denominator": "3"},
                                                {"type": "rational", "numerator": "2", "denominator": "3"}])
        self.assertEqual((data["variable"]["name"], data["variable"]["uid"]), ("x", X.uid))

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path; on an
        input without a value the lean path falls back and the value function raises the structured path's error."""
        rng = random.Random(20261172)
        cases = []
        for _ in range(16):
            n = rng.randrange(1, 9)
            xs = distinct(rng, n, lambda r: random_rational(r, r.choice((2, 6, 20))))
            cases.append(arguments(xs, [random_rational(rng, rng.choice((1, 4, 30))) for _ in range(n)]))
        cases.append(arguments(range(-40, 40), [rng.randrange(-9, 10) for _ in range(80)], Symbol("t")))
        cases.append(dict(SMALL, xs=[0, 1, 2], ys=[1, 3, 7]))  # Lists, frozen into tuples on both paths.
        for args in cases:
            with self.subTest(xs=args["xs"][:4]):
                expected = support.run(OPERATION, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(OPERATION, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(OPERATION, **args), expected)
        over = dict(xs=tuple(range(1100)), ys=tuple(k % 7 for k in range(1100)), variable=X)
        for args, code in ((dict(xs=(1, 1), ys=(1, 2), variable=X), "invalid_input"),
                           (dict(xs=(), ys=(), variable=X), "invalid_input"), (over, "unsupported_operation")):
            with self.subTest(code=code, points=len(args["xs"])):
                self.assertIs(support.lean_value(OPERATION, **args), lean.FALLBACK)
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(OPERATION, **args)
                self.assertEqual(raised.exception.code, code)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: the numbers travel as
        JSON integers or strings "p/q", the variable as its name."""
        requests = (
            SMALL,
            dict(xs=(Q(1, 2), 2), ys=(0, 1), variable=X),
            dict(xs=(Q(-1, 2), 0, 3, 4, Q(7, 3)), ys=(2, -1, 0, 5, Q(1, 9)), variable=Symbol("t")),
            dict(xs=(2 ** 80, -3, 5), ys=(Q(-7, 9), -(3 ** 50), 0), variable=X),
            dict(xs=(1, 2), ys=(0, 0), variable=X),
        )
        for args in requests:
            with self.subTest(xs=args["xs"]):
                result = self.answer(support.call_values(OPERATION, **args))
                self.assertEqual(result, support.run(OPERATION, **args))
                self.assertTrue(mf.verify(result).verified)
        raw = (
            ({"xs": [0, 1, 2], "ys": [1, 3, 7], "variable": "x"}, (1, 1, 1), "x"),
            ({"xs": ["1/2", 2], "ys": [0, "1"], "variable": "t"}, (F(-1, 3), F(2, 3)), "t"),
            ({"xs": ["-3/4"], "ys": ["5/2"], "variable": "y"}, (F(5, 2),), "y"),
            ({"xs": [1, 2, 3], "ys": [str(10 ** 30), str(10 ** 30), str(10 ** 30)], "variable": "x"}, (10 ** 30,), "x"),
        )
        for request, coefficients, name in raw:
            with self.subTest(request=request):
                result = self.answer(support.call(OPERATION, request))
                self.assertEqual(result.value.coefficients, polynomial(coefficients).coefficients)
                self.assertEqual(result.value.variable.name, name)
                self.assertTrue(mf.verify(result).verified)
        errors = (
            ({"xs": [1, 1], "ys": [0, 1], "variable": "x"}, "invalid_input", "pairwise distinct abscissas"),
            ({"xs": [1, "2/2"], "ys": [0, 1], "variable": "x"}, "invalid_input", "pairwise distinct abscissas"),
            ({"xs": [1, 2], "ys": [3], "variable": "x"}, "invalid_input", "the same length"),
            ({"xs": [], "ys": [], "variable": "x"}, "invalid_input", "at least one point"),
            ({"xs": [0.5], "ys": [0], "variable": "x"}, "invalid_input", "An exact rational is required"),
            ({"xs": list(range(1100)), "ys": [k % 7 for k in range(1100)], "variable": "x"}, "unsupported_operation",
             "a degree above 1024"),
        )
        for request, code, text in errors:
            with self.subTest(error=str(request)[:60]):
                error = self.assertToolError(support.call(OPERATION, request), code)
                self.assertIn(text, error["message"])
        self.assertToolError(support.call(OPERATION, {"xs": [0, 1], "ys": [1, 2], "variable": "x"},
                                          limits={"max_work": 1}), "resource_limit_exceeded")

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 200 seeded inputs, and the checker accepts
        every answer: random rational points (Lagrange's formula, and Gauss-Jordan elimination where short), equally
        spaced nodes of every step, which take the forward differences, unequally spaced integer nodes on polynomials
        with integer coefficients, where every division of the table is exact, chosen polynomials at more nodes than
        they need, constant and zero data, and longer inputs against Lagrange's formula in quadratic steps."""
        rng = random.Random(20261173)
        counts: Counter[str] = Counter()

        def check(label: str, xs: Sequence[Number], ys: Sequence[Number], coefficients: Sequence[Number]) -> None:
            with self.subTest(case=label, number=counts[label], xs=[str(x) for x in xs[:4]]):
                self.assertProduces(arguments(xs, ys), coefficients)
                counts[label] += 1

        while counts["random"] < 110:
            n = rng.randrange(1, 10)
            xs = distinct(rng, n, lambda r: random_rational(r, r.choice((2, 4, 16, 64))))
            ys = [random_rational(rng, rng.choice((1, 3, 20))) if rng.random() > 0.1 else F(0) for _ in range(n)]
            reference = lagrange(xs, ys)
            if n <= 7:
                self.assertEqual(vandermonde(xs, ys), reference)
            check("random", xs, ys, reference)
        while counts["equally spaced"] < 60:
            n = rng.randrange(2, 15)
            start, step = random_rational(rng, rng.choice((1, 6))), random_rational(rng, rng.choice((1, 4))) or F(1)
            xs = [start + k * step for k in range(n)]
            rng.shuffle(xs)  # Sorting finds the step, also for a negative one.
            ys = [F(rng.randrange(-20, 21), rng.choice((1, 1, 2, 3, 7))) for _ in range(n)]
            check("equally spaced", xs, ys, lagrange(xs, ys))
        while counts["integer polynomial"] < 30:
            degree = rng.randrange(0, 12)
            coefficients = [rng.randrange(-50, 51) for _ in range(degree)] + [rng.randrange(1, 50)]
            xs = rng.sample(range(-300, 300), degree + 1 + rng.randrange(0, 6))
            check("integer polynomial", xs, integer_values(coefficients, xs), coefficients)
        while counts["constructed"] < 50:
            degree = rng.randrange(0, 10)
            coefficients = [random_rational(rng, rng.choice((2, 8, 30))) for _ in range(degree)]
            coefficients.append(random_rational(rng, 5) or F(1))
            xs = distinct(rng, degree + 1 + rng.randrange(0, 5), lambda r: random_rational(r, r.choice((3, 10))))
            check("constructed", xs, [evaluate(coefficients, x) for x in xs], coefficients)
        while counts["constant"] < 12:
            n = rng.randrange(1, 30)
            xs = distinct(rng, n, lambda r: random_rational(r, 8))
            constant = random_rational(rng, 20) if counts["constant"] % 3 else F(0)
            check("constant", xs, [constant] * n, [constant])
        for n in (30, 45, 60):
            nodes = distinct(rng, n, lambda r: F(r.randrange(-60, 61), r.randrange(1, 13)))
            ys = [rng.randrange(-9, 10) for _ in range(n)]
            check("longer", nodes, ys, lagrange_by_division(nodes, ys))
            consecutive = range(-(n // 2), n - n // 2)
            check("longer", consecutive, ys, lagrange_by_division(consecutive, ys))
        self.assertGreaterEqual(sum(counts.values()), 200, counts)
        self.assertGreaterEqual(min(counts.values()), 6, counts)

    # -----------------------------------------------------------------------------------------------------------
    # The producer's sizes, paths and work

    def test_full_size_inputs_fit_the_default_budget(self):
        """Requests of the record's full size complete under the default limits, verified: 1,025 consecutive integer
        abscissas on a polynomial of degree 1,024 with 20-bit integer coefficients, and 600 consecutive integers with
        one-digit ordinates, whose coefficients have some 4,600 bits. At 1,025 such ordinates (coefficients of some
        8,800 bits) the producer still completes within its budget, and the checker's ends the request with
        resource_limit_exceeded, never another failure."""
        rng = random.Random(20261113)
        size = MAX_POLYNOMIAL_DEGREE + 1
        coefficients = [rng.randrange(-10 ** 6, 10 ** 6) for _ in range(size - 1)] + [rng.randrange(1, 10 ** 6)]
        xs = tuple(range(-(size // 2), size - size // 2))
        start = time.perf_counter()
        with computation(fresh=True) as budget:  # The default limits, as for a request.
            value, evidence = family.produce(OPERATION, {"xs": xs, "ys": integer_values(coefficients, xs),
                                                         "variable": X})
        self.assertEqual((value, evidence), (polynomial(coefficients), ()))
        self.assertLess(budget.work, 1_700_000)  # Measured 1,413,116 units.
        result = support.run(OPERATION, xs=xs, ys=integer_values(coefficients, xs), variable=X)
        self.assertVerified(result, polynomial(coefficients))
        self.assertLess(time.perf_counter() - start, 4 * FAST)
        ys = tuple(rng.randrange(-9, 10) for _ in range(600))
        result = support.run(OPERATION, xs=tuple(range(-300, 300)), ys=ys, variable=X)
        self.assertEqual(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertEqual(len(result.value.coefficients), 600)
        self.assertGreater(max(c.denominator.bit_length() for c in result.value.coefficients), 4_000)
        self.assertTrue(mf.verify(result).verified)
        ys = tuple(rng.randrange(-9, 10) for _ in range(size))
        with computation(fresh=True) as budget:
            value, _ = family.produce(OPERATION, {"xs": xs, "ys": ys, "variable": X})
        self.assertLess(budget.work, ComputationLimits().max_work)  # Measured about 1.52 million units.
        self.assertGreater(max(c.denominator.bit_length() for c in value.coefficients), 8_000)
        # The checker needs about 2.1 million units for this claim (measured with its present cost model), more than
        # its default 2 million: run() reports resource_limit_exceeded, and never another failure.
        start = time.perf_counter()
        result = support.run(OPERATION, xs=xs, ys=ys, variable=X)
        if result.execution_status is ExecutionStatus.COMPLETED:
            self.assertVerified(result, value)
        else:
            self.assertResourceLimit(result)
        self.assertLess(time.perf_counter() - start, 2 * FAST)

    def test_degree_above_the_record_bound_is_unsupported(self):
        """More than 1,025 points whose interpolant has a degree above 1,024 have no Polynomial record:
        unsupported_operation, found when the table reaches order 1,025, also for points in another order. Low-degree
        data on as many points stays a value."""
        rng = random.Random(20261174)
        for n in (1026, 1100):
            xs = list(range(n))
            ys = [rng.randrange(-9, 10) for _ in range(n)]
            rng.shuffle(xs)
            with self.subTest(points=n):
                start = time.perf_counter()
                result = support.run(OPERATION, xs=tuple(xs), ys=tuple(ys), variable=X)
                self.assertIs(result.execution_status, ExecutionStatus.UNSUPPORTED, result.error)
                self.assertEqual(result.error.code, "unsupported_operation")
                self.assertIn(f"a degree above {MAX_POLYNOMIAL_DEGREE}", result.error.message)
                self.assertLess(time.perf_counter() - start, FAST)
                with self.assertRaises(UnsupportedOperation):
                    family.produce(OPERATION, {"xs": tuple(xs), "ys": tuple(ys), "variable": X})
        xs = tuple(range(1100))
        self.assertVerified(support.run(OPERATION, xs=xs, ys=integer_values((3, 0, 1), xs), variable=X),
                            polynomial((3, 0, 1)))

    def test_both_paths_of_the_table_agree(self):
        """Equally spaced nodes take the forward differences without a division, other nodes the divisions with the
        least scale that keeps them exact. Forced onto the division path, equally spaced nodes give the same values,
        exact and inexact divisions alike, as the references do."""
        rng = random.Random(20261175)
        original = family._next_level

        def dividing(budget, level, total, points, order, node_bits, spacing, rational=None):
            return original(budget, level, total, points, order, node_bits, None, rational)

        cases = [(tuple(range(-6, 7)), tuple(rng.randrange(-9, 10) for _ in range(13))),
                 (tuple(Q(k, 3) for k in range(10)), tuple(Q(rng.randrange(-9, 10), rng.randrange(1, 5))
                                                          for _ in range(10))),
                 (tuple(range(0, 100, 7)), integer_values((1, -2, 0, 3), range(0, 100, 7)))]
        for xs, ys in cases:
            with self.subTest(xs=xs[:3]):
                expected = lagrange(xs, ys)
                self.assertProduces(arguments(xs, ys), expected)
                with patch.object(family, "_next_level", dividing):
                    self.assertProduces(arguments(xs, ys), expected)

    def test_work_is_charged_before_it_runs(self):
        """The producer charges the budget before each step: its work on an input is the same on every run and grows
        with the lengths of the numbers, a budget one unit short of it refuses the input, and a small budget stops a
        large input at once. Nothing in the family module calls the builtin pow or uses **, which could not be
        interrupted."""
        rng = random.Random(20261176)
        xs = tuple(range(-50, 50))
        ys = tuple(rng.randrange(-9, 10) for _ in xs)
        works = []
        for _ in range(2):
            with computation(fresh=True) as budget:
                family.produce(OPERATION, {"xs": xs, "ys": ys, "variable": X})
            works.append(budget.work)
        self.assertEqual(works[0], works[1])
        with computation(ComputationLimits(max_work=works[0]), fresh=True):
            family.produce(OPERATION, {"xs": xs, "ys": ys, "variable": X})
        with computation(ComputationLimits(max_work=works[0] - 1), fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OPERATION, {"xs": xs, "ys": ys, "variable": X})
        long = tuple(y * (2 ** 20_000 + 1) for y in ys)  # The same table, with every number 20,000 bits longer.
        with computation(fresh=True) as budget:
            family.produce(OPERATION, {"xs": xs, "ys": long, "variable": X})
        self.assertGreater(budget.work, 2 * works[0])
        nodes = tuple(rng.sample(range(-3000, 3001), 100))  # Unequally spaced: a division for every entry.
        with computation(fresh=True) as budget:
            family.produce(OPERATION, {"xs": nodes, "ys": ys, "variable": X})
        self.assertGreater(budget.work, works[0])
        start = time.perf_counter()
        with computation(ComputationLimits(max_work=20_000), fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OPERATION, {"xs": tuple(range(1025)), "ys": tuple(rng.randrange(-9, 10)
                                                                            for _ in range(1025)), "variable": X})
        self.assertLess(time.perf_counter() - start, FAST)
        tree = ast.parse(inspect.getsource(family))
        called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func,
                                                                                                         ast.Name)}
        self.assertNotIn("pow", called)
        self.assertEqual([node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                          and isinstance(node.op, ast.Pow)], [])

    def test_coefficients_reduce_against_the_part_coprime_to_the_scale(self):
        """Regression (review of wave m2a): the coefficient of x**i, A_i * Q**i / L_0, is reduced by a gcd of A_i with
        L_0 / gcd(Q**i, L_0) alone, and gcd(Q**i, L_0) builds up from gcd(Q, rest) one coefficient at a time. Before,
        A_i * Q**i was formed and reduced against the whole of L_0, a gcd up to i * bits(Q) bits longer: the 50 random
        abscissas with 16-bit denominators below took about 1.34 million units, now about 990,000, and 40 with 30-bit
        denominators 2.66 million, beyond the default budget, now about 1.92 million. Values are canonical and equal
        Fraction arithmetic wherever gcd(Q**i, L_0) is partial, full or 1."""
        rng = random.Random(20261180)
        cases = [(12, 2 ** 7 * 3 ** 2 * 5 * 7), (12, 1), (1, 2 ** 5 * 11), (35, 2 ** 3 * 3), (6, 2 ** 40 * 3 ** 3),
                 (2 ** 3 * 3 ** 4, 2 ** 2 * 3 ** 30 * 13)]
        for scale, denominator in cases:
            numerators = [rng.choice((0, 1, -1, scale, denominator, rng.randrange(-10 ** 9, 10 ** 9),
                                      rng.randrange(1, 10 ** 30) * 2 ** rng.randrange(0, 20))) for _ in range(40)]
            with self.subTest(scale=scale, denominator=denominator), computation(fresh=True) as budget:
                found = family._coefficients(budget, numerators, denominator, scale)
                expected = [F(numerator * scale ** index, denominator) for index, numerator in enumerate(numerators)]
                self.assertEqual([F(c.numerator, c.denominator) for c in found], expected)
                self.assertTrue(all(type(c) is Rational and c.denominator >= 1
                                    and math.gcd(c.numerator, c.denominator) == 1 for c in found))
        nodes = distinct(rng, 50, lambda r: F(r.randrange(-2 ** 16, 2 ** 16), r.randrange(1, 2 ** 16)))
        ys = [rng.randrange(-9, 10) for _ in nodes]
        args = arguments(nodes, ys)
        with computation(fresh=True) as budget:
            value, _ = family.produce(OPERATION, dict(args))
        # Measured 391,927 units since such abscissas run the table over themselves (989,898 over the integer nodes
        # Q x_i after this change, 1,338,561 before it).
        self.assertLess(budget.work, 1_150_000)
        self.assertEqual(value, polynomial(lagrange_by_division(nodes, ys)))
        self.assertIs(support.verdict(family, OPERATION, args, value), True)

    def test_rational_abscissas_run_over_the_abscissas(self):
        """Regression (review of wave m2a, the completeness gap): with many different denominators, the table over the
        integer nodes Q x_i carried Q**j at level j, so that 200 abscissas k/p_k with distinct 20-bit primes p_k took
        about 299 million units for a value whose coefficients have under 8,000 bits, which the checker verifies with
        about 577,000; run() reported resource_limit_exceeded. Unequally spaced abscissas whose common denominator Q
        is longer than twice their denominators on average now run the table and Horner's rule over the abscissas
        themselves: the request fits the default budget (about 311,000 units) and verifies. 40 abscissas with 30-bit
        denominators went from about 1.8 million units to about 630,000. Equally spaced and same-denominator
        abscissas keep the integer nodes, and both tables give the values of Fraction arithmetic."""
        primes, candidate = [], 2 ** 19 + 1
        while len(primes) < 200:
            if all(candidate % d for d in range(3, math.isqrt(candidate) + 1, 2)):
                primes.append(candidate)
            candidate += 2
        xs = [F(k + 1, p) for k, p in enumerate(primes)]
        last = xs[-1]  # The data is 1 except at the last abscissa: P = 1 - prod_(k < 199) (x - x_k) / (last - x_k).
        basis, scale = [F(1)], F(1)
        for node in xs[:-1]:
            basis, scale = times_linear(basis, node), scale * (last - node)
        expected = trimmed([(1 if index == 0 else 0) - c / scale for index, c in enumerate(basis)])
        start = time.perf_counter()
        result = support.run(OPERATION, **arguments(xs, [1] * 199 + [0]))
        self.assertVerified(result, polynomial(expected))
        self.assertLess(time.perf_counter() - start, 3 * FAST)
        with computation(fresh=True) as budget:
            family.produce(OPERATION, arguments(xs, [1] * 199 + [0]))
        self.assertLess(budget.work, 600_000)  # Measured 311,179 units.
        rng = random.Random(20261181)
        nodes = distinct(rng, 40, lambda r: F(r.randrange(-2 ** 30, 2 ** 30), r.randrange(1, 2 ** 30)))
        ys = [rng.randrange(-9, 10) for _ in nodes]
        with computation(fresh=True) as budget:
            value, _ = family.produce(OPERATION, arguments(nodes, ys))
        self.assertLess(budget.work, 900_000)  # Measured 620,429 units.
        self.assertEqual(value, polynomial(lagrange_by_division(nodes, ys)))
        # The choice of the table: integer nodes for one common denominator and for equal steps, the abscissas for
        # integers next to a few rationals of different denominators; both agree with Fraction arithmetic.
        original, chosen = family._next_level, []

        def spy(budget, level, total, points, order, node_bits, spacing, rational=None):
            chosen.append(rational is not None)
            return original(budget, level, total, points, order, node_bits, spacing, rational)

        cases = [(rng.sample([F(k, 7) for k in range(-200, 200)], 30), False),
                 ([F(k, 3) for k in range(25)], False),
                 (rng.sample(range(-300, 300), 30) + [F(1, 3), F(2, 5), F(-3, 7), F(5, 11)], True),
                 (distinct(rng, 30, lambda r: F(r.randrange(-50, 50), r.randrange(1, 30))), True)]
        for xs, over_abscissas in cases:
            ys = [rng.randrange(-9, 10) for _ in xs]
            with self.subTest(xs=xs[:3]), patch.object(family, "_next_level", spy):
                chosen.clear()
                self.assertProduces(arguments(xs, ys), lagrange_by_division(xs, ys))
                self.assertEqual(set(chosen), {over_abscissas})

    def test_sorting_is_charged_at_its_measured_cost(self):
        """Regression (review of wave m2a): sorting the nodes was charged a unit per comparison, some 20 to 200 times
        its measured cost, so that produce() refused 100,000 points on a line (about 1.7 million units for the sort
        alone) whose value the checker verifies with 1.3 million. operations_v6.run limits a request to 100,000 nodes
        (some 25,000 points) before the producer runs; produce() called directly now takes about 700,000."""
        xs = tuple(range(-150_000, 150_000, 3))
        with computation(fresh=True) as budget:
            value, evidence = family.produce(OPERATION, {"xs": xs, "ys": tuple(2 * x - 7 for x in xs), "variable": X})
        self.assertEqual((value, evidence), (polynomial((-7, 2)), ()))
        self.assertLess(budget.work, 1_000_000)
        self.assertEqual(family._sort_work(100_000, 20), 1 + (100_000 * 17 >> 4))
        self.assertGreater(family._sort_work(2_000, 60_000), 2_000 * 11 * 7)  # Long ints with a shared prefix.
