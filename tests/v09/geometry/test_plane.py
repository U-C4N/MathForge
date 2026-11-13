"""The plane family of mathforge.geometry: distance, polygon_area, line_intersection, orientation.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register geometry plane

Every expected value comes from the card or from a reference of this module, written here with fractions.Fraction and
independent of the producer: the squared distance with an integer square root by Newton's method, the signed area as a
fan of triangles checked against the trapezoid form, the intersection by elimination on the implicit equations of the
lines, the orientation as the sign of a 3 x 3 determinant; never from the producer. The card defers no input of this
family, so no request ends without a value (NO_CONCLUSION). tests/v09/geometry/test_plane_checker.py
tests the checker on its own.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import inspect
import itertools
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.geometry import plane as family
from mathforge.areas.geometry.plane import LineIntersection
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.matrices import Matrix
from mathforge.model import Mul, Rational, Sqrt

from .. import support

FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
HALF = Rational(1, 2)
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
STAND_INS = (0.5, True, "1", None)  # No exact number: a float, a bool, a str and None.
OPERATIONS = ("distance", "polygon_area", "line_intersection", "orientation")

Number = int | Fraction | Rational

# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these, from the card, or from brute force.


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
    """The number as a value holds it: always a Rational, also when it is integral."""
    number = frac(number)
    return Rational(number.numerator, number.denominator)


def point(*coordinates: Number) -> tuple[int | Rational, ...]:
    """A point as a request holds it."""
    return tuple(exact(coordinate) for coordinate in coordinates)


def polygon(vertices: Sequence[Sequence[Number]]) -> Matrix:
    """The polygon as polygon_area takes it: a Matrix of two columns, one row per vertex."""
    return Matrix(tuple((exact(x), exact(y)) for x, y in vertices), ncols=2)


def integer_root(n: int) -> int:
    """floor(sqrt(n)) for n >= 0, by Newton's method on integers written here (the producer uses math.isqrt)."""
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + 1) // 2)  # At least sqrt(n).
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def distance_reference(p: Sequence[Number], q: Sequence[Number]) -> Rational | Sqrt:
    """The card's value: the Rational square root of r = sum (p_i - q_i)**2 when r has one, else the node Sqrt(r)."""
    r = sum(((frac(a) - frac(b)) ** 2 for a, b in zip(p, q)), Fraction(0))
    top, bottom = integer_root(r.numerator), integer_root(r.denominator)
    if top * top == r.numerator and bottom * bottom == r.denominator:
        return Rational(top, bottom)
    return Sqrt(Rational(r.numerator, r.denominator))


def minus(a: Sequence[Number], b: Sequence[Number]) -> tuple[Fraction, Fraction]:
    """a - b in the plane."""
    return frac(a[0]) - frac(b[0]), frac(a[1]) - frac(b[1])


def cross(u: Sequence[Number], v: Sequence[Number]) -> Fraction:
    """cross2(u, v) = u_x v_y - u_y v_x."""
    return frac(u[0]) * frac(v[1]) - frac(u[1]) * frac(v[0])


def sign(number: Fraction) -> int:
    return (number > 0) - (number < 0)


def fan_area(vertices: Sequence[Sequence[Number]]) -> Fraction:
    """The signed area as a fan of triangles from the first vertex, (1/2) sum_i cross2(v_i - v_0, v_{i+1} - v_0)."""
    if len(vertices) < 3:
        return Fraction(0)
    origin = vertices[0]
    return sum((cross(minus(a, origin), minus(b, origin)) for a, b in zip(vertices[1:], vertices[2:])),
               Fraction(0)) / 2


def trapezoid_area(vertices: Sequence[Sequence[Number]]) -> Fraction:
    """The signed area by the trapezoid form, (1/2) sum_i (x_i + x_{i+1}) (y_{i+1} - y_i), cyclically."""
    count, total = len(vertices), Fraction(0)
    for index in range(count):
        (x0, y0), (x1, y1) = vertices[index], vertices[(index + 1) % count]
        total += (frac(x0) + frac(x1)) * (frac(y1) - frac(y0))
    return total / 2


def area_reference(vertices: Sequence[Sequence[Number]]) -> Rational:
    """The signed area as a value: the fan of triangles, which the trapezoid form confirms."""
    area = fan_area(vertices)
    assert trapezoid_area(vertices) == area
    return rational(area)


def implicit(a: Sequence[Number], b: Sequence[Number]) -> tuple[Fraction, Fraction, Fraction]:
    """(A, B, C) with A x + B y = C exactly on the line through the points a != b."""
    first, second = frac(b[1]) - frac(a[1]), frac(a[0]) - frac(b[0])
    return first, second, first * frac(a[0]) + second * frac(a[1])


def eliminated(p1: Sequence[Number], p2: Sequence[Number], q1: Sequence[Number],
               q2: Sequence[Number]) -> tuple[Fraction, Fraction] | str:
    """The common point of the two lines by elimination on their implicit equations (the producer uses Cramer's rule
    on the parametric form), substituted into both equations; else "parallel" or "coincident", told apart by whether
    both q1 and q2 solve the first equation."""
    a1, b1, c1 = implicit(p1, p2)
    a2, b2, c2 = implicit(q1, q2)
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:  # Parallel normals.
        on_first = [a1 * frac(x) + b1 * frac(y) == c1 for x, y in (q1, q2)]
        assert on_first[0] == on_first[1]  # Parallel lines share both points or neither.
        return "coincident" if on_first[0] else "parallel"
    x, y = (c1 * b2 - c2 * b1) / determinant, (a1 * c2 - a2 * c1) / determinant
    assert a1 * x + b1 * y == c1 and a2 * x + b2 * y == c2
    return x, y


def intersection_reference(p1: Sequence[Number], p2: Sequence[Number], q1: Sequence[Number],
                           q2: Sequence[Number]) -> LineIntersection:
    """The card's value, from the elimination."""
    found = eliminated(p1, p2, q1, q2)
    if type(found) is str:
        return LineIntersection(found, None)
    return LineIntersection("point", (rational(found[0]), rational(found[1])))


def orientation_reference(p: Sequence[Number], q: Sequence[Number], r: Sequence[Number]) -> int:
    """The sign of det [[1, px, py], [1, qx, qy], [1, rx, ry]] by cofactor expansion along the first column,
    confirmed by the sign of the triangle's trapezoid area."""
    (px, py), (qx, qy), (rx, ry) = ((frac(x), frac(y)) for x, y in (p, q, r))
    determinant = (qx * ry - qy * rx) - (px * ry - py * rx) + (px * qy - py * qx)
    assert sign(determinant) == sign(trapezoid_area((p, q, r)))
    return sign(determinant)


def number(rng: random.Random, bits: int, denominator_bits: int = 0) -> Fraction:
    """A random exact number: an integer of up to bits bits, over a denominator of up to denominator_bits bits."""
    numerator = rng.randrange(-(1 << bits), (1 << bits) + 1)
    return Fraction(numerator, rng.randrange(1, (1 << denominator_bits) + 1) if denominator_bits else 1)


def reference(operation: str, args: dict[str, object]) -> object:
    """The value of a request by the references."""
    if operation == "distance":
        return distance_reference(args["p"], args["q"])
    if operation == "polygon_area":
        return area_reference(args["points"].rows)
    if operation == "line_intersection":
        return intersection_reference(args["p1"], args["p2"], args["q1"], args["q2"])
    return orientation_reference(args["p"], args["q"], args["r"])


# ---------------------------------------------------------------------------------------------------------------
# Requests

SQUARE = ((0, 0), (1, 0), (1, 1), (0, 1))  # Area 1 counterclockwise.
TRIANGLE = ((0, 0), (4, 0), (0, 3))  # Area 6.
BOW_TIE = ((0, 0), (1, 1), (1, 0), (0, 1))  # Two loops of opposite orientation that cancel.
UNEQUAL_BOW_TIE = ((0, 0), (2, 2), (2, 0), (0, 1))  # Loops of 1/3 counterclockwise and 4/3 clockwise: -1.
FIVE = dict(p=(0, 0), q=(3, 4))  # distance 5
EIGHT = dict(p=(0, 0), q=(2, 2))  # distance sqrt(8)
HALVES = dict(p=(0, 0), q=(HALF, HALF))  # distance sqrt(1/2)
CROSSING = dict(p1=(0, 0), p2=(1, 1), q1=(0, 1), q2=(1, 0))  # the point (1/2, 1/2)
PARALLEL = dict(p1=(0, 0), p2=(1, 0), q1=(0, 1), q2=(1, 1))
SAME = dict(p1=(0, 0), p2=(1, 1), q1=(2, 2), q2=(3, 3))  # coincident
CCW = dict(p=(0, 0), q=(1, 0), r=(0, 1))  # orientation 1
CW = dict(p=(0, 0), q=(0, 1), r=(1, 0))  # orientation -1
COLLINEAR = dict(p=(0, 0), q=(1, 1), r=(2, 2))  # orientation 0

# One small request per operation.
SMALL = {
    "distance": FIVE,
    "polygon_area": dict(points=polygon(TRIANGLE)),
    "line_intersection": CROSSING,
    "orientation": CCW,
}

# The known values of the card, and the cases that its text names: (operation, arguments, value).
KNOWN = (
    ("distance", FIVE, Rational(5)),
    ("distance", dict(p=(0, 0), q=(1, 1)), Sqrt(Rational(2))),
    ("distance", dict(p=(1, 2, 3), q=(1, 2, 3)), Rational(0)),
    ("distance", EIGHT, Sqrt(Rational(8))),  # No factor taken out.
    ("distance", HALVES, Sqrt(HALF)),
    ("distance", dict(p=(7,), q=(-2,)), Rational(9)),
    ("distance", dict(p=(0, 0), q=(Rational(3, 10), Rational(2, 5))), HALF),
    ("distance", dict(p=(-1, -2, -3), q=(2, 2, 9)), Rational(13)),
    ("polygon_area", dict(points=polygon(SQUARE)), Rational(1)),
    ("polygon_area", dict(points=polygon(SQUARE[::-1])), Rational(-1)),
    ("polygon_area", dict(points=polygon(TRIANGLE)), Rational(6)),
    ("polygon_area", dict(points=polygon(BOW_TIE)), Rational(0)),
    ("polygon_area", dict(points=polygon(UNEQUAL_BOW_TIE)), Rational(-1)),
    ("polygon_area", dict(points=polygon(SQUARE + ((0, 0),))), Rational(1)),  # A repeated closing vertex.
    ("polygon_area", dict(points=polygon(((1, 2), (5, -7)))), Rational(0)),
    ("polygon_area", dict(points=polygon(((3, 5),))), Rational(0)),
    ("polygon_area", dict(points=polygon(((0, 0), (1, 1), (2, 2)))), Rational(0)),
    ("polygon_area", dict(points=polygon(((0, 0), (1, 0), (0, 1)))), HALF),
    ("polygon_area", dict(points=Matrix((), ncols=2)), Rational(0)),
    ("polygon_area", dict(points=Matrix([])), Rational(0)),
    ("line_intersection", CROSSING, LineIntersection("point", (HALF, HALF))),
    ("line_intersection", PARALLEL, LineIntersection("parallel", None)),
    ("line_intersection", SAME, LineIntersection("coincident", None)),
    ("line_intersection", dict(p1=(0, 0), p2=(2, 0), q1=(1, -1), q2=(1, 1)),
     LineIntersection("point", (Rational(1), Rational(0)))),
    ("line_intersection", dict(p1=(0, 0), p2=(1, 2), q1=(0, 1), q2=(1, 0)),
     LineIntersection("point", (Rational(1, 3), Rational(2, 3)))),
    ("line_intersection", dict(p1=(0, 0), p2=(0, 1), q1=(1, 5), q2=(1, -5)), LineIntersection("parallel", None)),
    ("line_intersection", dict(p1=(0, 0), p2=(2, 1), q1=(6, 3), q2=(-4, -2)), LineIntersection("coincident", None)),
    ("orientation", CCW, 1),
    ("orientation", CW, -1),
    ("orientation", COLLINEAR, 0),
    ("orientation", dict(p=(1, 2), q=(1, 2), r=(5, 7)), 0),
    ("orientation", dict(p=(HALF, 0), q=(1, Rational(1, 7)), r=(Rational(-5, 2), 2)), 1),
)

# The inputs that the card rejects, each invalid input.
INVALID = (
    ("distance", dict(p=(0, 0), q=(3, 4, 0))),
    ("distance", dict(p=(3,), q=(0, 0))),
    ("distance", dict(p=(), q=())),
    ("distance", dict(p=(), q=(1,))),
    ("distance", dict(p=(1,), q=())),
    ("polygon_area", dict(points=Matrix([[0, 0, 0], [4, 0, 0], [0, 3, 0]]))),
    ("polygon_area", dict(points=Matrix([[0], [4], [0]]))),
    ("polygon_area", dict(points=Matrix([[], []]))),  # Two rows without columns.
    ("polygon_area", dict(points=Matrix((), ncols=3))),
    ("polygon_area", dict(points=Matrix((), ncols=1))),
    ("polygon_area", dict(points=Matrix([[0, 4, 0], [0, 0, 3]]))),  # The triangle transposed.
    ("line_intersection", dict(p1=(0, 0), p2=(0, 0), q1=(0, 1), q2=(1, 0))),
    ("line_intersection", dict(p1=(0, 0), p2=(1, 1), q1=(2, 2), q2=(2, 2))),
    ("line_intersection", dict(p1=(5, 5), p2=(5, 5), q1=(5, 5), q2=(5, 5))),
    ("line_intersection", dict(p1=(2, HALF), p2=(Rational(2), HALF), q1=(0, 1), q2=(1, 0))),  # 2 == Rational(2).
    ("line_intersection", dict(p1=(0, 0, 0), p2=(1, 1, 0), q1=(0, 1, 0), q2=(1, 0, 0))),
    ("line_intersection", dict(p1=(0, 0), p2=(1, 1), q1=(0, 1), q2=(1, 0, 0))),
    ("line_intersection", dict(p1=(0,), p2=(1,), q1=(2,), q2=(3,))),
    ("line_intersection", dict(p1=(), p2=(1, 1), q1=(0, 1), q2=(1, 0))),
    ("orientation", dict(p=(0, 0, 0), q=(1, 0, 0), r=(0, 1, 0))),
    ("orientation", dict(p=(0, 0), q=(1, 0), r=(0, 1, 5))),
    ("orientation", dict(p=(0,), q=(1,), r=(2,))),
    ("orientation", dict(p=(), q=(1, 0), r=(0, 1))),
)


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins that the producer and the checker refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Fractional(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Coordinates(tuple):
    """A tuple subclass for a point."""


class Crossing(LineIntersection):
    """A LineIntersection subclass: the right fields, another class."""


@dataclass(frozen=True)
class Answer:
    """A record that holds a right answer: the checker refuses it, an impostor of it and one forged with
    object.__new__."""
    value: object


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, whose slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def forged_matrix(**fields: object) -> Matrix:
    """A Matrix made with object.__new__, whose fields may break its form."""
    return support.forged(Matrix, **fields)


def mixed(rng: random.Random, coordinates: Sequence[Number]) -> tuple[int | Rational, ...]:
    """A point with each integral coordinate written as an int or as a Rational, at random: two requests of the same
    point."""
    return tuple(rational(item) if frac(item).denominator == 1 and rng.random() < 0.5 else exact(item)
                 for item in coordinates)


class PlaneTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, of its exact type, with evidence () and the four checks, which
        mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), type(value))
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertAgrees(self, operation: str, args: dict[str, object], expected: object) -> None:
        """produce() answers expected, of its exact type, with the evidence (), and the checker accepts it."""
        value, evidence = family.produce(operation, dict(args))
        self.assertEqual(value, expected)
        self.assertIs(type(value), type(expected))
        if type(value) is LineIntersection and value.point is not None:
            self.assertTrue(all(type(item) is Rational for item in value.point), value)
        self.assertEqual(evidence, ())
        self.assertIs(support.verdict(family, operation, args, value, evidence), True)

    # -----------------------------------------------------------------------------------------------------------

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, args, value in KNOWN:
            with self.subTest(operation=operation, args=args):
                self.assertEqual(reference(operation, args), value)  # The card's value, by the references.
                result = support.run(operation, **args)
                self.assertVerified(result, value)
                self.roundtrip(result)
        # Lists are frozen into tuples by run(): the request holds the points as tuples.
        result = support.run("distance", p=[0, 0], q=[3, 4])
        self.assertVerified(result, Rational(5))
        self.assertEqual(dict(result.request.arguments), FIVE)
        # 3 and Rational(3) are the same coordinate: the same value, from different requests.
        self.assertVerified(support.run("distance", p=(Rational(3), Rational(0)), q=(0, 4)), Rational(5))
        self.assertVerified(support.run("orientation", p=(Rational(0), 0), q=(1, 0), r=(0, Rational(1))), 1)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument and in every coordinate; so is every input that the card rejects. produce() tests the types
        itself too."""
        for operation, args in SMALL.items():
            for name, item in args.items():
                for stand_in in STAND_INS:
                    with self.subTest(operation=operation, argument=name, item=stand_in):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: stand_in})))
                if type(item) is tuple:  # A point: each coordinate in turn, and other containers.
                    for index in range(len(item)):
                        for stand_in in STAND_INS + (Fraction(1, 2), Integer(1), (1,), [1]):
                            changed = item[:index] + (stand_in,) + item[index + 1:]
                            with self.subTest(operation=operation, argument=name, index=index, item=stand_in):
                                self.assertInvalidInput(support.run(operation, **dict(args, **{name: changed})))
                    for container in (Matrix([list(item)]), str(item), (item,)):
                        with self.subTest(operation=operation, argument=name, container=type(container).__name__):
                            self.assertInvalidInput(support.run(operation, **dict(args, **{name: container})))
                else:  # The polygon: rows that are no Matrix, and a point.
                    for container in (item.rows, [list(row) for row in item.rows], (0, 0)):
                        with self.subTest(operation=operation, argument=name, container=container):
                            self.assertInvalidInput(support.run(operation, **dict(args, **{name: container})))
        for operation, args in INVALID:
            with self.subTest(operation=operation, args=args):
                self.assertInvalidInput(support.run(operation, **args))
        messages = (("distance", dict(p=(0, 0), q=(3, 4, 0)), "the same dimension"),
                    ("distance", dict(p=(), q=()), "at least 1"),
                    ("polygon_area", dict(points=Matrix([[0], [4], [0]])), "two columns"),
                    ("line_intersection", dict(p1=(0, 0), p2=(0, 0), q1=(0, 1), q2=(1, 0)), "p1 != p2 and q1 != q2"),
                    ("line_intersection", dict(p1=(0,), p2=(1,), q1=(2,), q2=(3,)), "dimension 2"),
                    ("orientation", dict(p=(0, 0, 0), q=(1, 0), r=(0, 1)), "dimension 2"))
        for operation, args, text in messages:
            with self.subTest(operation=operation, message=text):
                self.assertIn(text, support.run(operation, **args).error.message)
        # Rationals and matrices that only object.__new__ makes reach produce() through run(), which checks the class
        # alone: produce() refuses them as its checker does.
        rows = SMALL["polygon_area"]["points"].rows
        reaching = [
            ("distance", dict(p=(forged_rational(2, 4), 0), q=(0, 0))),
            ("distance", dict(p=(forged_rational(-1, -2), 0), q=(0, 0))),
            ("orientation", dict(CCW, r=(0, forged_rational(0, 5)))),
            ("line_intersection", dict(CROSSING, q2=(forged_rational(3, 3), 0))),
            # The public Matrix constructor keeps a Rational as it is given.
            ("polygon_area", dict(points=Matrix([[0, 0], [4, 0], [forged_rational(0, 2), 3]]))),
            ("polygon_area", dict(points=forged_matrix(rows=TRIANGLE, ncols=2))),  # int entries
            ("polygon_area", dict(points=forged_matrix(rows=list(rows), ncols=2))),
            ("polygon_area", dict(points=forged_matrix(rows=rows, ncols=3))),
            ("polygon_area", dict(points=forged_matrix(rows=rows[:2] + (rows[2] + (Rational(0),),), ncols=2))),
        ]
        for index, (operation, args) in enumerate(reaching):
            with self.subTest(operation=operation, forged=index):
                self.assertInvalidInput(support.run(operation, **args))
        # produce() called directly: every stand-in, in a coordinate and as a point, and arguments of other names.
        stand_ins = (True, False, 0.5, "1", None, Fraction(1, 2), Integer(1), Fractional(1, 2), forged_rational(2, 4),
                     forged_rational(1, -2), forged_rational(1, 0), forged_rational(1.0, 2), forged_rational(True, 1),
                     support.forged(Rational, numerator=1), support.forged(Rational, denominator=1))
        for operation, args in SMALL.items():
            for name, item in args.items():
                if type(item) is tuple:
                    for index, stand_in in enumerate(stand_ins):
                        with self.subTest(operation=operation, argument=name, stand_in=index, path="produce"):
                            with self.assertRaises(InvalidInput):
                                family.produce(operation, dict(args, **{name: (stand_in,) + item[1:]}))
                    containers = (list(item), Coordinates(item), None, 0, Rational(1), polygon(TRIANGLE))
                else:
                    containers = (rows, [list(row) for row in rows], None, 0, (0, 0))
                for index, container in enumerate(containers):
                    with self.subTest(operation=operation, argument=name, container=index, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: container}))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(args, extra=1))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, tuple(args.items()))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, {name: value for name, value in list(args.items())[:-1]})
        forgeries = [forged_matrix(rows=list(rows), ncols=2), forged_matrix(rows=rows, ncols=True),
                     forged_matrix(rows=rows, ncols=2.0), forged_matrix(rows=rows, ncols=-2),
                     forged_matrix(rows=TRIANGLE, ncols=2), forged_matrix(rows=rows[:2] + ([Rational(0), Rational(3)],),
                                                                          ncols=2),
                     forged_matrix(rows=rows[:2] + (rows[2][:1],), ncols=2), forged_matrix(rows=rows, ncols=0),
                     forged_matrix(rows=rows[:2] + ((Rational(0), forged_rational(6, 2)),), ncols=2),
                     forged_matrix(ncols=2), forged_matrix(rows=rows), forged_matrix(rows=None, ncols=2)]
        for index, forgery in enumerate(forgeries):
            with self.subTest(forgery=index, path="produce"), self.assertRaises(InvalidInput):
                family.produce("polygon_area", {"points": forgery})
        for operation in ("Distance", "area", "convex_hull", "", None, ("distance",)):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(FIVE))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("line_intersection", p1=(0, 0), p2=(0, 0), q1=(0, 1), q2=(1, 0))
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in any coordinate, a 65 x 65 matrix where one is taken, and max_work=1 exceed the
        budget. The polygon Matrix is charged before its shape is validated, and work is charged before it runs, so
        that a request that needs more than its budget fails at once."""
        for operation, args in SMALL.items():
            for name, item in args.items():
                if type(item) is not tuple:
                    continue
                for index in range(len(item)):
                    for huge in (support.HUGE, -support.HUGE, Rational(1, support.HUGE), Rational(support.HUGE, 3)):
                        changed = item[:index] + (huge,) + item[index + 1:]
                        with self.subTest(operation=operation, argument=name, index=index):
                            self.assertResourceLimit(support.run(operation, **dict(args, **{name: changed})))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
        self.assertResourceLimit(support.run("polygon_area", points=support.oversized_matrix()))
        # More entries than max_matrix_entries in a shape that polygon_area rejects too: the budget speaks first.
        with computation(ComputationLimits(max_matrix_entries=5000), fresh=True):
            vertices = Matrix(tuple((k, k * k % 101) for k in range(2049)))  # 4,098 entries.
            column = Matrix(tuple((k,) for k in range(4097)))
        for matrix in (vertices, column):
            with self.subTest(shape=matrix.shape):
                self.assertResourceLimit(support.run("polygon_area", points=matrix))
        with computation(ComputationLimits(max_integer_bits=80_000), fresh=True):
            giant = Matrix([[0, 0], [support.HUGE, 0], [0, 3]])
        self.assertResourceLimit(support.run("polygon_area", points=giant))
        rng = random.Random(20261125)
        limits = ComputationLimits(max_work=1000)
        start = time.perf_counter()
        long_p = tuple(rng.getrandbits(60_000) for _ in range(40))
        self.assertResourceLimit(support.run("distance", p=long_p, q=tuple(-x for x in long_p), limits=limits))
        many = polygon(tuple((k, k * k % 101) for k in range(2000)))
        self.assertResourceLimit(support.run("polygon_area", points=many, limits=limits))
        corners = [(rng.getrandbits(60_000), rng.getrandbits(60_000)) for _ in range(4)]
        self.assertResourceLimit(support.run("line_intersection", **dict(zip(("p1", "p2", "q1", "q2"), corners)),
                                             limits=limits))
        self.assertResourceLimit(support.run("orientation", p=corners[0], q=corners[1], r=corners[2], limits=limits))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_values_beyond_max_integer_bits(self):
        """A value whose integers are longer than max_integer_bits exceeds the budget where it is built, while the
        numbers that lead to it may be longer: a rational distance needs only its root within the limit."""
        big = 2 ** 40_000
        # r = 2**80000 is longer than max_integer_bits, its root is not.
        self.assertVerified(support.run("distance", p=(0, 0), q=(big, 0)), Rational(big))
        # A primitive Pythagorean triple of 40,001-bit numbers (m odd, n even, coprime): terms of 80,000 bits cancel to
        # r = 1.
        m, n = 2 ** 20_000 + 1, 2 ** 19_999
        a, b, c = m * m - n * n, 2 * m * n, m * m + n * n
        self.assertEqual(a * a + b * b, c * c)
        self.assertVerified(support.run("distance", p=(0, 0), q=(Rational(a, c), Rational(b, c))), Rational(1))
        # An irrational distance holds r in its value: Sqrt(2**80000 + 1) cannot be built.
        self.assertResourceLimit(support.run("distance", p=(0, 0), q=(big, 1)))
        self.assertResourceLimit(support.run("polygon_area", points=polygon(((0, 0), (big, 0), (0, big)))))
        # y = 0 meets the line through (0, 1) and (2**40000, 1 - 2**-30000) at x = 2**70000.
        far = dict(p1=(0, 0), p2=(1, 0), q1=(0, 1), q2=(big, Rational(2 ** 30_000 - 1, 2 ** 30_000)))
        self.assertEqual(eliminated(**far), (Fraction(2 ** 70_000), Fraction(0)))  # No record holds this point.
        self.assertResourceLimit(support.run("line_intersection", **far))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The evidence of every operation is (), so a producer chooses nothing but its canonical value: the freedom of
        a true claim is in its request. The checker accepts the references' claims on requests that write integral
        coordinates as ints or as Rationals, and on the requests that the symmetries of each operation give (the
        points of a distance swapped, the vertices of a polygon rotated or closed by a repeated vertex, the points of
        a line or the two lines swapped, the points of an orientation permuted cyclically, or two of them swapped,
        which reverses the sign); the producer, which made none of these claims, answers the same value to each."""
        rng = random.Random(20261126)
        checked = dict.fromkeys(OPERATIONS, 0)
        kinds = set()
        for case in range(40):
            bits, denominator_bits = rng.choice(((3, 0), (12, 0), (6, 3), (20, 8)))
            dimension = rng.randrange(1, 5)
            p = [number(rng, bits, denominator_bits) for _ in range(dimension)]
            q = [number(rng, bits, denominator_bits) for _ in range(dimension)]
            value = distance_reference(p, q)
            for left, right in ((p, q), (q, p)):
                args = dict(p=mixed(rng, left), q=mixed(rng, right))
                with self.subTest(operation="distance", case=case, args=args):
                    self.assertAccepts("distance", args, value)
                    self.assertEqual(family.produce("distance", args), (value, ()))
                    checked["distance"] += 1
            vertices = [(number(rng, bits, denominator_bits), number(rng, bits, denominator_bits))
                        for _ in range(rng.randrange(3, 9))]
            area = area_reference(vertices)
            shift = rng.randrange(len(vertices))
            for variant in (vertices, vertices[shift:] + vertices[:shift], vertices + vertices[:1]):
                args = dict(points=polygon(variant))
                with self.subTest(operation="polygon_area", case=case, vertices=variant):
                    self.assertAccepts("polygon_area", args, area)
                    self.assertEqual(family.produce("polygon_area", args), (area, ()))
                    checked["polygon_area"] += 1
            corners = [(number(rng, 3, rng.choice((0, 2))), number(rng, 3, rng.choice((0, 2)))) for _ in range(4)]
            direction = minus(corners[1], corners[0])
            if case % 4 == 1:  # The second line parallel to the first, through q1.
                corners[3] = tuple(c + 2 * d for c, d in zip(corners[2], direction))
            elif case % 4 == 2:  # The same line: q1 and q2 on the line through p1 and p2.
                corners[2], corners[3] = (tuple(a + t * d for a, d in zip(corners[0], direction))
                                          for t in (Fraction(-1, 2), Fraction(3)))
            if corners[0] == corners[1] or corners[2] == corners[3]:
                continue
            crossing = intersection_reference(*corners)
            kinds.add(crossing.kind)
            first, second, third, fourth = corners
            for variant in ((first, second, third, fourth), (second, first, third, fourth),
                            (first, second, fourth, third), (third, fourth, first, second)):
                args = dict(zip(("p1", "p2", "q1", "q2"), (mixed(rng, corner) for corner in variant)))
                with self.subTest(operation="line_intersection", case=case, args=args):
                    self.assertAccepts("line_intersection", args, crossing)
                    self.assertEqual(family.produce("line_intersection", args), (crossing, ()))
                    checked["line_intersection"] += 1
            turn = orientation_reference(first, second, third)
            for variant, expected in (((first, second, third), turn), ((second, third, first), turn),
                                      ((third, first, second), turn), ((second, first, third), -turn),
                                      ((first, third, second), -turn)):
                args = dict(zip(("p", "q", "r"), (mixed(rng, corner) for corner in variant)))
                with self.subTest(operation="orientation", case=case, args=args):
                    self.assertAccepts("orientation", args, expected)
                    self.assertEqual(family.produce("orientation", args), (expected, ()))
                    checked["orientation"] += 1
        # The polygon without vertices has two requests: the Matrix of two columns and no rows, and Matrix([]).
        for matrix in (Matrix((), ncols=2), Matrix([])):
            self.assertAccepts("polygon_area", dict(points=matrix), Rational(0))
            self.assertEqual(family.produce("polygon_area", dict(points=matrix)), (Rational(0), ()))
        self.assertEqual(kinds, {"point", "parallel", "coincident"})
        self.assertTrue(all(count >= 60 for count in checked.values()), checked)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where a number belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        claims = [("distance", FIVE), ("distance", EIGHT), ("distance", HALVES),
                  ("polygon_area", dict(points=polygon(TRIANGLE))),
                  ("polygon_area", dict(points=polygon(UNEQUAL_BOW_TIE))),
                  ("line_intersection", CROSSING), ("line_intersection", PARALLEL), ("line_intersection", SAME),
                  ("orientation", CCW), ("orientation", CW), ("orientation", COLLINEAR)]
        point_value = LineIntersection("point", (HALF, HALF))
        parallel, coincident = LineIntersection("parallel", None), LineIntersection("coincident", None)
        wrong: dict[str, list[tuple[str, object]]] = {
            "distance": [],
            "polygon_area": [],
            "line_intersection": [],
            "orientation": [],
        }
        for operation, args in claims:
            result = support.run(operation, **args)
            value, evidence = result.value, result.steps[0].outputs[1]
            self.assertVerified(result, value)
            self.assertAccepts(operation, args, value, evidence)
            if operation == "distance" and type(value) is Rational:
                wrong_values = [("the value plus 1", Rational(value.numerator + value.denominator, value.denominator)),
                                ("the negated value", -value), ("the square", value * value),
                                ("the int", value.numerator), ("Sqrt of the square", Sqrt(value * value)),
                                ("Sqrt of the square plus 1", Sqrt(value * value + 1)),
                                ("a float", value.numerator / value.denominator), ("a str", str(value)),
                                ("a Fraction", frac(value)), ("an impostor Rational", support.impostor(value)),
                                ("a Rational subclass", Fractional(value.numerator, value.denominator)),
                                ("a Rational in higher terms", forged_rational(2 * value.numerator,
                                                                               2 * value.denominator))]
            elif operation == "distance":
                radicand = value.radicand
                wrong_values = [("Sqrt of the radicand plus 1", Sqrt(radicand + 1)),
                                ("Sqrt of half the radicand", Sqrt(radicand / 2)),
                                ("the radicand alone", radicand),
                                ("twice the root of a quarter", Mul((Rational(2), Sqrt(radicand / 4)))),
                                ("the negated root", Mul((Rational(-1), value))),
                                ("an impostor Sqrt", support.impostor(value)),
                                ("a Sqrt without a radicand", support.forged(Sqrt)),
                                ("a radicand in higher terms", support.forged(Sqrt, radicand=forged_rational(
                                    3 * radicand.numerator, 3 * radicand.denominator)))]
            elif operation == "polygon_area":
                wrong_values = [("the negated area", -value), ("twice the area", value * 2),
                                ("the area plus 1/2", value + HALF), ("the unsigned area", abs(value)),
                                ("Sqrt of the square", Sqrt(value * value)),
                                ("an impostor Rational", support.impostor(value)),
                                ("a float", value.numerator / value.denominator)]
                if value.denominator == 1:
                    wrong_values.append(("the int", value.numerator))
                wrong_values = [(label, item) for label, item in wrong_values if item != value]
            elif operation == "line_intersection" and value.kind == "point":
                x, y = value.point
                wrong_values = [("parallel", parallel), ("coincident", coincident),
                                ("the point moved by 2**-80", LineIntersection("point", (x + Rational(1, 2 ** 80), y))),
                                ("the point (0, 0)", LineIntersection("point", (Rational(0), Rational(0)))),
                                ("int coordinates", support.forged(LineIntersection, kind="point", point=(0, 0))),
                                ("a list point", support.forged(LineIntersection, kind="point", point=[x, y])),
                                ("a tuple-subclass point", support.forged(LineIntersection, kind="point",
                                                                          point=Coordinates((x, y)))),
                                ("no point", support.forged(LineIntersection, kind="point", point=None)),
                                ("the kind Point", support.forged(LineIntersection, kind="Point", point=(x, y))),
                                ("an impostor record", support.impostor(value)),
                                ("a subclass record", Crossing("point", (x, y))),
                                ("the point alone", (x, y))]
            elif operation == "line_intersection":
                other = coincident if value.kind == "parallel" else parallel
                wrong_values = [("the other kind", other), ("a point", point_value),
                                ("a point for the kind", support.forged(LineIntersection, kind=value.kind,
                                                                        point=(HALF, HALF))),
                                ("an empty point", support.forged(LineIntersection, kind=value.kind, point=())),
                                ("an impostor record", support.impostor(value)),
                                ("the kind alone", value.kind)]
            else:
                wrong_values = [(f"the sign {item}", item) for item in (-1, 0, 1, 2, -2) if item != value]
                wrong_values += [("a bool", bool(value)), ("a float", float(value)), ("a Rational", Rational(value)),
                                 ("an int subclass", Integer(value)), ("a str", str(value))]
            wrong_values += [("a record", Answer(value)), ("an impostor record", support.impostor(Answer(value))),
                             ("a forged record", support.forged(Answer, value=value)), ("None", None),
                             ("the value in a tuple", (value,)), ("the value in a list", [value])]
            wrong[operation].extend(wrong_values)
            cases = [(label, item, ()) for label, item in wrong_values]
            cases += [("evidence as a list", value, []), ("evidence None", value, None),
                      ("an extra evidence item", value, ((),)), ("the value as evidence", value, (value,)),
                      ("oversized evidence", value, (0,) * 100_000), ("a giant value", Rational(GIANT), ()),
                      ("a giant int", GIANT, ())]
            for label, item, proof in cases:
                with self.subTest(operation=operation, args=args, case=label):
                    self.assertRejects(operation, args, item, proof)
            for other in OPERATIONS:  # The true claim under the name of another operation.
                if other != operation:
                    with self.subTest(operation=operation, renamed=other):
                        self.assertRejects(other, args, value, evidence)
            report = support.statuses(mf.verify(support.tampered(result, wrong_values[0][1])))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
        self.assertTrue(all(len(items) >= 10 for items in wrong.values()))

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = list(SMALL.items()) + [("distance", EIGHT), ("distance", HALVES),
                                          ("polygon_area", dict(points=Matrix([]))),
                                          ("polygon_area", dict(points=polygon(UNEQUAL_BOW_TIE))),
                                          ("line_intersection", PARALLEL), ("line_intersection", SAME),
                                          ("orientation", CW), ("orientation", COLLINEAR)]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, args), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, args=args):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("distance", **FIVE)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a Rational and a
        Sqrt node of distance, a Rational area, the three kinds of a LineIntersection and the three signs of an
        orientation."""
        rendered = (
            ("distance", FIVE, "5", "5"),
            ("distance", dict(p=(0, 0), q=(Rational(3, 10), Rational(2, 5))), "1/2", r"\frac{1}{2}"),
            ("distance", EIGHT, "sqrt(8)", r"\sqrt{8}"),
            ("distance", HALVES, "sqrt(1/2)", r"\sqrt{\frac{1}{2}}"),
            ("polygon_area", dict(points=polygon(TRIANGLE)), "6", "6"),
            ("polygon_area", dict(points=polygon(SQUARE[::-1])), "-1", "-1"),
            ("polygon_area", dict(points=polygon(((0, 0), (1, 0), (0, 1)))), "1/2", r"\frac{1}{2}"),
            ("polygon_area", dict(points=Matrix([])), "0", "0"),
            ("line_intersection", CROSSING, "(1/2, 1/2)", r"\left(\frac{1}{2}, \frac{1}{2}\right)"),
            ("line_intersection", dict(p1=(0, 0), p2=(1, 2), q1=(0, 1), q2=(1, 0)), "(1/3, 2/3)",
             r"\left(\frac{1}{3}, \frac{2}{3}\right)"),
            ("line_intersection", dict(p1=(0, -6), p2=(1, -6), q1=(-2, 0), q2=(-2, 1)), "(-2, -6)",
             r"\left(-2, -6\right)"),
            ("line_intersection", PARALLEL, "parallel lines", r"\text{parallel lines}"),
            ("line_intersection", SAME, "coincident lines", r"\text{coincident lines}"),
            ("orientation", CCW, "1", "1"),
            ("orientation", CW, "-1", "-1"),
            ("orientation", COLLINEAR, "0", "0"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(operation=operation, args=args):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        rational_data = {"type": "rational", "numerator": "1", "denominator": "2"}
        self.assertEqual(mf.to_data(support.run("distance", **EIGHT).value)["data"],
                         {"type": "sqrt", "radicand": {"type": "rational", "numerator": "8", "denominator": "1"}})
        self.assertEqual(mf.to_data(support.run("polygon_area", points=polygon(TRIANGLE)).value)["data"],
                         {"type": "rational", "numerator": "6", "denominator": "1"})
        self.assertEqual(mf.to_data(support.run("line_intersection", **CROSSING).value)["data"],
                         {"type": "v6_line_intersection", "kind": {"type": "text", "value": "point"},
                          "point": {"type": "tuple", "items": [rational_data, rational_data]}})
        self.assertEqual(mf.to_data(support.run("line_intersection", **PARALLEL).value)["data"],
                         {"type": "v6_line_intersection", "kind": {"type": "text", "value": "parallel"},
                          "point": {"type": "none"}})
        self.assertEqual(mf.to_data(support.run("orientation", **CW).value)["data"], {"type": "integer", "value": "-1"})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261127)
        cases: list[tuple[str, dict[str, object]]] = []
        while len(cases) < 40:
            bits, denominator_bits = rng.choice(((4, 0), (30, 0), (8, 4)))
            dimension = rng.randrange(1, 5)
            corners = [(number(rng, bits, denominator_bits), number(rng, bits, denominator_bits)) for _ in range(4)]
            cases.append(("distance", dict(p=point(*(number(rng, bits, denominator_bits) for _ in range(dimension))),
                                           q=point(*(number(rng, bits, denominator_bits) for _ in range(dimension))))))
            cases.append(("polygon_area", dict(points=polygon(corners[:rng.randrange(0, 5)]))))
            if corners[0] != corners[1] and corners[2] != corners[3]:
                cases.append(("line_intersection", dict(zip(("p1", "p2", "q1", "q2"),
                                                            (point(*corner) for corner in corners)))))
            cases.append(("orientation", dict(zip(("p", "q", "r"), (point(*corner) for corner in corners[:3])))))
        for operation, args in cases:
            with self.subTest(operation=operation, args=args):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        requests = {
            "distance": ({"p": [0, 0], "q": [3, 4]}, FIVE),
            "polygon_area": ({"points": [[0, 0], [4, 0], [0, 3]]}, dict(points=polygon(TRIANGLE))),
            "line_intersection": ({"p1": [0, 0], "p2": [1, 1], "q1": [0, 1], "q2": [1, 0]}, CROSSING),
            "orientation": ({"p": [0, 0], "q": [1, 0], "r": [0, 1]}, CCW),
        }
        for operation, (request, args) in requests.items():
            with self.subTest(operation=operation):
                result = self.answer(support.call(operation, request))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        # Rationals travel as "p/q" strings, integers from 2**53 on as decimal strings.
        large = 2 ** 70 + 1
        for operation, args in (("distance", dict(p=(HALF, large), q=(Rational(-1, 3), 0))),
                                ("polygon_area", dict(points=polygon(((Rational(1, 3), 0), (large, Rational(1, 7)),
                                                                      (0, 1))))),
                                ("line_intersection", dict(p1=(Rational(1, 3), 0), p2=(1, Rational(1, 7)),
                                                           q1=(Rational(-5, 2), large), q2=(0, Rational(2, 3)))),
                                ("orientation", dict(p=(large, 1), q=(Rational(-1, 9), 2), r=(3, -large)))):
            with self.subTest(operation=operation, values=True):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        # An empty JSON array is the polygon without vertices, Matrix([]).
        result = self.answer(support.call("polygon_area", {"points": []}))
        self.assertVerified(result, Rational(0))
        self.assertEqual(result, support.run("polygon_area", points=Matrix([])))
        error = self.assertToolError(support.call("line_intersection", {"p1": [0, 0], "p2": [0, 0], "q1": [0, 1],
                                                                        "q2": [1, 0]}), "invalid_input")
        self.assertIn("p1 != p2 and q1 != q2", error["message"])
        error = self.assertToolError(support.call("polygon_area", {"points": [[0, 0, 0], [4, 0, 0], [0, 3, 0]]}),
                                     "invalid_input")
        self.assertIn("two columns", error["message"])
        self.assertToolError(support.call("distance", {"p": [0.5, 0], "q": [0, 0]}), "invalid_input")
        self.assertToolError(support.call("orientation", {"p": [0, 0], "q": [1, 0], "r": [True, 1]}), "invalid_input")

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 200 seeded inputs per operation (every
        ordered triple of a 3 x 3 grid for orientation besides), and the checker accepts every answer. The loops have
        fixed lengths and count the inputs that they compare, so that a wrong producer fails them rather than keeping
        them running."""
        rng = random.Random(20261128)
        counts = dict.fromkeys(OPERATIONS, 0)
        directions = ((3, 4), (5, 12), (8, 15), (1, 2, 2), (2, 3, 6), (2, 6, 9), (1, 4, 8), (1,))  # Integer norms.
        for case in range(300):
            if case % 3:  # Random points, mostly at an irrational distance.
                dimension = rng.randrange(1, 7)
                bits, denominator_bits = rng.choice(((3, 0), (20, 0), (8, 4), (40, 20), (200, 0)))
                p = [number(rng, bits, denominator_bits) for _ in range(dimension)]
                q = p[:] if case % 29 == 1 else [number(rng, bits, denominator_bits) for _ in range(dimension)]
            else:  # q = p + k * direction: the rational distance |k| * |direction|.
                direction = rng.choice(directions)
                scale = number(rng, 10, rng.choice((0, 6))) or Fraction(1)
                p = [number(rng, 12, 3) for _ in direction]
                q = [a + scale * b for a, b in zip(p, direction)]
            args = dict(p=point(*p), q=point(*q))
            expected = distance_reference(p, q)
            counts["distance"] += 1
            with self.subTest(operation="distance", args=args):
                if type(expected) is Rational and expected.denominator == 1 and expected.numerator < 10 ** 4:
                    # Brute force on small integral distances: the k with k * k = r.
                    r = sum(((a - b) ** 2 for a, b in zip(p, q)), Fraction(0))
                    self.assertEqual(next(k for k in range(10 ** 4) if k * k >= r), expected.numerator)
                self.assertAgrees("distance", args, expected)
        for case in range(250):
            count = rng.randrange(0, 25)
            bits, denominator_bits = rng.choice(((3, 0), (30, 0), (8, 3), (20, 12)))
            if case % 5 == 0:  # Shared denominators: sevenths and their cubes.
                vertices = [(Fraction(rng.randrange(-10 ** 9, 10 ** 9), 7), Fraction(rng.randrange(10 ** 9), 343))
                            for _ in range(count)]
            else:
                vertices = [(number(rng, bits, denominator_bits), number(rng, bits, denominator_bits))
                            for _ in range(count)]
            args = dict(points=polygon(vertices))
            counts["polygon_area"] += 1
            with self.subTest(operation="polygon_area", vertices=vertices):
                self.assertAgrees("polygon_area", args, area_reference(vertices))
        # Lattice polygons by brute force: a simple orthogonal staircase has the area of its unit cells.
        for steps in range(1, 12):
            vertices = [(0, 0)] + [(k + dx, k + dy) for k in range(steps) for dx, dy in ((1, 0), (1, 1))] + [(0, steps)]
            cells = sum(1 for x in range(steps + 1) for y in range(steps + 1) if x < y + 1 and y < steps)
            self.assertEqual(area_reference(vertices), Rational(cells))
            self.assertAgrees("polygon_area", dict(points=polygon(vertices)), Rational(cells))
        kinds = set()
        for case in range(320):  # About one draw in forty repeats a point and is refused instead.
            bits, denominator_bits = rng.choice(((2, 0), (2, 0), (6, 0), (6, 2), (30, 10)))
            corners = [(number(rng, bits, denominator_bits), number(rng, bits, denominator_bits)) for _ in range(4)]
            if case % 5 == 0:  # The second line parallel to the first: through q1 with the direction of p2 - p1.
                step = number(rng, 3, 1) or Fraction(1)
                corners[3] = tuple(c + step * d for c, d in zip(corners[2], minus(corners[1], corners[0])))
            elif case % 7 == 0:  # The same line: q1 and q2 on the line through p1 and p2.
                corners[2], corners[3] = (tuple(a + t * d for a, d in zip(corners[0], minus(corners[1], corners[0])))
                                          for t in (number(rng, 3, 2), number(rng, 3, 2)))
            args = dict(zip(("p1", "p2", "q1", "q2"), (point(*corner) for corner in corners)))
            if corners[0] == corners[1] or corners[2] == corners[3]:
                with self.subTest(operation="line_intersection", degenerate=args), self.assertRaises(InvalidInput):
                    family.produce("line_intersection", args)
                continue
            expected = intersection_reference(*corners)
            kinds.add(expected.kind)
            counts["line_intersection"] += 1
            with self.subTest(operation="line_intersection", args=args):
                self.assertAgrees("line_intersection", args, expected)
        self.assertEqual(kinds, {"point", "parallel", "coincident"})
        for case in range(300):
            bits, denominator_bits = rng.choice(((4, 0), (30, 0), (10, 5), (60, 30)))
            p, q = [(number(rng, bits, denominator_bits), number(rng, bits, denominator_bits)) for _ in range(2)]
            if case % 4 == 0:  # r on the line through p and q.
                step = number(rng, 5, 3)
                r = tuple(a + step * d for a, d in zip(p, minus(q, p)))
            else:
                r = (number(rng, bits, denominator_bits), number(rng, bits, denominator_bits))
            args = dict(p=point(*p), q=point(*q), r=point(*r))
            counts["orientation"] += 1
            with self.subTest(operation="orientation", args=args):
                self.assertAgrees("orientation", args, orientation_reference(p, q, r))
        grid = list(itertools.product(range(3), repeat=2))
        turns = {-1: 0, 0: 0, 1: 0}
        for p, q, r in itertools.product(grid, repeat=3):
            expected = orientation_reference(p, q, r)
            turns[expected] += 1
            self.assertAgrees("orientation", dict(p=p, q=q, r=r), expected)
        self.assertEqual(turns[1], turns[-1])  # Swapping q and r reverses every turn.
        self.assertEqual(sum(turns.values()), 9 ** 3)
        self.assertTrue(all(count >= 200 for count in counts.values()), counts)

    def test_full_size_inputs_fit_the_default_budget(self):
        """The largest requests of the checker's tests complete under the default limits, within a second each and
        far within max_work: a distance of 32,000-bit coordinates, one of two 10,000-dimensional points, polygons of
        2,048 vertices (4,096 entries, max_matrix_entries) with integer and with shared rational coordinates, a line
        intersection of 16,000-bit corners and an orientation of 65,000-bit coordinates."""
        rng = random.Random(20261116)
        a, b = rng.getrandbits(32_000) | 1 << 31_999, rng.getrandbits(32_000)
        long_p = tuple(rng.randrange(-1000, 1000) for _ in range(10_000))
        long_q = tuple(rng.randrange(-1000, 1000) for _ in range(10_000))
        wide = tuple((rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(-10 ** 6, 10 ** 6)) for _ in range(2048))
        sevenths = tuple((Fraction(rng.randrange(-10 ** 9, 10 ** 9), 7), Fraction(rng.randrange(10 ** 9), 7 ** 3))
                         for _ in range(2048))
        corners = [(rng.getrandbits(16_000), rng.getrandbits(16_000)) for _ in range(4)]
        requests = [
            ("distance", dict(p=(0, 0), q=(a, b)), distance_reference((0, 0), (a, b))),
            ("distance", dict(p=long_p, q=long_q), distance_reference(long_p, long_q)),
            ("polygon_area", dict(points=polygon(wide)), area_reference(wide)),
            ("polygon_area", dict(points=polygon(sevenths)), area_reference(sevenths)),
            ("line_intersection", dict(zip(("p1", "p2", "q1", "q2"), corners)), intersection_reference(*corners)),
            ("orientation", dict(p=(rng.getrandbits(65_000), 3), q=(1, rng.getrandbits(65_000)),
                                 r=(rng.getrandbits(65_000), rng.getrandbits(65_000))), None),
        ]
        self.assertGreater(requests[4][2].point[0].numerator.bit_length(), 40_000)
        for operation, args, expected in requests:
            with self.subTest(operation=operation):
                if expected is None:
                    expected = reference(operation, args)
                start = time.perf_counter()
                with computation(fresh=True) as budget:  # The default limits, as for a request.
                    self.assertEqual(family.produce(operation, args), (expected, ()))
                self.assertLess(budget.work, 200_000)  # Measured at most about 70,000 units.
                result = support.run(operation, **args)
                self.assertVerified(result, expected)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_sums_that_cancel_stay_within_the_budget(self):
        """Regression (review of wave m4b): the shoelace terms of vertices on a line that misses the origin telescope,
        c * (x_i - x_{i+1}) on y = x + c, and their unreduced running sum kept the least common multiple of every
        term's denominator: 200 vertices over nearly coprime 2,000-bit denominators exhausted the default max_work in
        produce(), where the checker verifies the area 0 in about 360,000 units. Requests through run(), on that line
        and on y = 2x (whose shoelace terms are 0 and skipped), and a distance whose terms cancel nothing but repeat
        the same denominators, complete and verify."""
        denominators = [2 ** 1999 + 2 * index + 1 for index in range(200)]  # Pairwise gcds divide differences < 400.
        offset = [(Fraction(3, denominator), Fraction(3, denominator) + 1) for denominator in denominators]
        through = [(Fraction(3, denominator), Fraction(6, denominator)) for denominator in denominators]
        lifted = offset[:60] + [(Fraction(1), Fraction(7))]
        requests = [
            ("polygon_area", dict(points=polygon(offset)), Rational(0)),
            ("polygon_area", dict(points=polygon(offset[::-1])), Rational(0)),
            ("polygon_area", dict(points=polygon(through)), Rational(0)),
            ("polygon_area", dict(points=polygon(lifted)), area_reference(lifted)),
            ("distance", dict(p=point(*(x for x, _ in offset)), q=point(*(y for _, y in offset))),
             Sqrt(Rational(200))),  # Every difference is -1 over its own denominator, unreduced: r = 200.
        ]
        self.assertNotEqual(requests[3][2], Rational(0))
        self.assertEqual(requests[4][2], distance_reference([x for x, _ in offset], [y for _, y in offset]))
        for operation, args, expected in requests:
            with self.subTest(operation=operation, expected=expected):
                with computation(fresh=True) as budget:
                    self.assertEqual(family.produce(operation, args), (expected, ()))
                self.assertLess(budget.work, 500_000)  # Measured at most about 90,000 units.
                self.assertVerified(support.run(operation, **args), expected)

    def test_work_is_charged_before_it_runs(self):
        """The charges depend on the request alone: a budget one unit short of the producer's work refuses the request,
        its exact size completes it. Long products are charged by their size."""
        rng = random.Random(20261117)
        requests = [
            ("distance", dict(p=(Rational(1, 3), 2 ** 3000, 5), q=(7, Rational(-2, 9), 2 ** 2000))),
            ("polygon_area", dict(points=polygon(((0, 0), (Fraction(2 ** 900, 3), 1), (5, Fraction(7, 2 ** 800)),
                                                  (-1, 9))))),
            ("line_intersection", dict(p1=(rng.getrandbits(4000), 1), p2=(2, rng.getrandbits(4000)),
                                       q1=(Rational(1, 3), 5), q2=(rng.getrandbits(3000), 7))),
            ("orientation", dict(p=(rng.getrandbits(5000), 0), q=(1, 1), r=(Rational(1, 7), 2 ** 4000))),
        ]
        for operation, args in requests:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    expected = family.produce(operation, args)
                work = budget.work
                self.assertGreater(work, 20)
                self.assertEqual(expected, (reference(operation, args), ()))
                short = ComputationLimits(max_work=work - 1)
                with self.assertRaises(ResourceLimitError), computation(short, fresh=True):
                    family.produce(operation, args)
                with computation(ComputationLimits(max_work=work), fresh=True):
                    self.assertEqual(family.produce(operation, args), expected)
        works = []
        for args in (dict(p=(0, 0), q=(3, 4)), dict(p=(0, 0), q=(3 * 2 ** 30_000 + 1, 4 * 2 ** 30_000))):
            with computation(fresh=True) as budget:
                family.produce("distance", args)
            works.append(budget.work)
        self.assertGreater(works[1], 50 * works[0])

    def test_module_imports_its_cost_model_inside_produce(self):
        """The family module imports at its top only modules that `import mathforge` loads: the cost model of
        trusted.integers is imported when produce() runs, and nothing imports the checker, lean, mcp or generated
        code."""
        tree = ast.parse(inspect.getsource(family))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                top.add("." * node.level + (node.module or ""))
            elif isinstance(node, ast.Import):
                top.update(alias.name for alias in node.names)
        self.assertEqual(top, {"__future__", "collections.abc", "dataclasses", "math", "...errors", "...limits",
                               "...matrices", "...model", "...registry"})
        everywhere = {"." * node.level + (node.module or "") for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom)}
        self.assertIn("...trusted.integers", everywhere)
        self.assertFalse(any(name in module for module in everywhere
                             for name in ("certify", "lean", "mcp", "_generated", "kernels")))
