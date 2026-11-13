"""The independent checker of mathforge.areas.geometry.plane (mathforge.certify.geometry.plane), tested on its own
before the family's producer exists.

Every true claim is computed here, never by the producer: with fractions.Fraction, by the card's formulas, which the
checker does not use (the sum of squares with an integer square root by Newton's method written here, the shoelace
sum, Cramer's rule, the cross product of q - p and r - p), each checked against a second reference (model.sqrt, a fan
of triangles, elimination on the implicit equations of the lines, the shoelace sign of the triangle) or against brute
force on a grid. The evidence is always (), the card's layout. The checker answers False to every corrupted claim,
never with an exception, and refuses an oversized forgery within FAST seconds under the default budget.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
import enum
from fractions import Fraction
import inspect
import itertools
import random
import time
from types import MappingProxyType

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.geometry import plane as family
from mathforge.areas.geometry.plane import LineIntersection
from mathforge.certify.geometry import plane as checker
from mathforge.contracts import CheckStatus
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.matrices import Matrix
from mathforge.model import Add, Mul, Pow, Rational, Sqrt, sqrt
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
BIG = 2 ** 65_535 + 1  # As long as max_integer_bits allows: arithmetic runs on it.
HALF = Rational(1, 2)

Number = int | Fraction | Rational


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these, or from brute force.

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
    """A point as a request holds it: a tuple of ints and Rationals."""
    return tuple(exact(coordinate) for coordinate in coordinates)


def integer_root(n: int) -> int:
    """floor(sqrt(n)) for n >= 0, by Newton's method on integers written here (the checker uses math.isqrt)."""
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + 1) // 2)  # At least sqrt(n).
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def squared_distance(p: Sequence[Number], q: Sequence[Number]) -> Fraction:
    """r = sum_i (p_i - q_i)**2."""
    return sum(((frac(a) - frac(b)) ** 2 for a, b in zip(p, q)), Fraction(0))


def distance_value(p: Sequence[Number], q: Sequence[Number]) -> Rational | Sqrt:
    """The card's value: the Rational square root of r = |p - q|**2 when r has one, else the node Sqrt(r)."""
    r = squared_distance(p, q)
    top, bottom = integer_root(r.numerator), integer_root(r.denominator)
    if top * top == r.numerator and bottom * bottom == r.denominator:
        return Rational(top, bottom)
    return Sqrt(Rational(r.numerator, r.denominator))


def distance_claim(p: Sequence[Number], q: Sequence[Number]) -> tuple[dict[str, object], Rational | Sqrt]:
    """(args, value) of a true distance claim; the value is model.sqrt(r), the card's definition."""
    r = squared_distance(p, q)
    value = distance_value(p, q)
    assert value == sqrt(Rational(r.numerator, r.denominator))
    return {"p": point(*p), "q": point(*q)}, value


def minus(a: Sequence[Number], b: Sequence[Number]) -> tuple[Fraction, Fraction]:
    """a - b in the plane."""
    return frac(a[0]) - frac(b[0]), frac(a[1]) - frac(b[1])


def cross(u: Sequence[Number], v: Sequence[Number]) -> Fraction:
    """cross2(u, v) = u_x v_y - u_y v_x."""
    return frac(u[0]) * frac(v[1]) - frac(u[1]) * frac(v[0])


def sign(number: Fraction) -> int:
    return (number > 0) - (number < 0)


def shoelace(vertices: Sequence[Sequence[Number]]) -> Fraction:
    """The signed area by the shoelace formula, (1/2) sum_i (x_i y_{i+1} - x_{i+1} y_i), cyclically."""
    count, total = len(vertices), Fraction(0)
    for index in range(count):
        (x0, y0), (x1, y1) = vertices[index], vertices[(index + 1) % count]
        total += frac(x0) * frac(y1) - frac(x1) * frac(y0)
    return total / 2


def fan(vertices: Sequence[Sequence[Number]]) -> Fraction:
    """The signed area as a fan of triangles from the first vertex, (1/2) sum_i cross2(v_i - v_0, v_{i+1} - v_0)."""
    if len(vertices) < 3:
        return Fraction(0)
    origin = vertices[0]
    return sum((cross(minus(a, origin), minus(b, origin)) for a, b in zip(vertices[1:], vertices[2:])),
               Fraction(0)) / 2


def polygon(vertices: Sequence[Sequence[Number]]) -> Matrix:
    """The polygon as polygon_area takes it: a Matrix of two columns, one row per vertex."""
    return Matrix(tuple((exact(x), exact(y)) for x, y in vertices), ncols=2)


def area_claim(vertices: Sequence[Sequence[Number]]) -> tuple[dict[str, object], Rational]:
    """(args, value) of a true polygon_area claim; the two references agree."""
    area = shoelace(vertices)
    assert fan(vertices) == area
    return {"points": polygon(vertices)}, rational(area)


def intersection(p1: Sequence[Number], p2: Sequence[Number], q1: Sequence[Number],
                 q2: Sequence[Number]) -> LineIntersection:
    """The card's value by Cramer's rule on p1 + t (p2 - p1) = q1 + s (q2 - q1)."""
    u, v, w = minus(p2, p1), minus(q2, q1), minus(q1, p1)
    d = cross(u, v)
    if d == 0:
        return LineIntersection("coincident" if cross(u, w) == 0 else "parallel", None)
    t = cross(w, v) / d
    return LineIntersection("point", (rational(frac(p1[0]) + t * u[0]), rational(frac(p1[1]) + t * u[1])))


def implicit(a: Sequence[Number], b: Sequence[Number]) -> tuple[Fraction, Fraction, Fraction]:
    """(A, B, C) with A x + B y = C exactly on the line through the points a != b."""
    first, second = frac(b[1]) - frac(a[1]), frac(a[0]) - frac(b[0])
    return first, second, first * frac(a[0]) + second * frac(a[1])


def eliminated(p1: Sequence[Number], p2: Sequence[Number], q1: Sequence[Number],
               q2: Sequence[Number]) -> LineIntersection:
    """The same value by elimination on the implicit equations of the two lines: the second reference."""
    a1, b1, c1 = implicit(p1, p2)
    a2, b2, c2 = implicit(q1, q2)
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:  # Parallel normals: the lines coincide when q1 solves the first equation.
        return LineIntersection("coincident" if a1 * frac(q1[0]) + b1 * frac(q1[1]) == c1 else "parallel", None)
    return LineIntersection("point", (rational((c1 * b2 - c2 * b1) / determinant),
                                      rational((a1 * c2 - a2 * c1) / determinant)))


def line_claim(p1: Sequence[Number], p2: Sequence[Number], q1: Sequence[Number],
               q2: Sequence[Number]) -> tuple[dict[str, object], LineIntersection]:
    """(args, value) of a true line_intersection claim; the two references agree."""
    value = intersection(p1, p2, q1, q2)
    assert eliminated(p1, p2, q1, q2) == value
    return {"p1": point(*p1), "p2": point(*p2), "q1": point(*q1), "q2": point(*q2)}, value


def orientation_value(p: Sequence[Number], q: Sequence[Number], r: Sequence[Number]) -> int:
    """The card's value, the sign of cross2(q - p, r - p), checked against the shoelace sign of the triangle."""
    value = sign(cross(minus(q, p), minus(r, p)))
    assert value == sign(shoelace((p, q, r)))
    return value


def orientation_claim(p: Sequence[Number], q: Sequence[Number],
                      r: Sequence[Number]) -> tuple[dict[str, object], int]:
    """(args, value) of a true orientation claim."""
    return {"p": point(*p), "q": point(*q), "r": point(*r)}, orientation_value(p, q, r)


def number(rng: random.Random, bits: int, denominator_bits: int = 0) -> Fraction:
    """A random exact number: an integer of up to bits bits, over a denominator of up to denominator_bits bits."""
    numerator = rng.randrange(-(1 << bits), (1 << bits) + 1)
    return Fraction(numerator, rng.randrange(1, (1 << denominator_bits) + 1) if denominator_bits else 1)


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values, arguments and evidence that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Fractional(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Coordinates(tuple):
    """A tuple subclass for a point, the rows of a Matrix or the evidence."""


class Text(str):
    """A str subclass for the kind of a LineIntersection."""


class Sign(enum.IntEnum):
    """The values of orientation as an int enumeration: equal to them but no exact ints."""
    MINUS = -1
    ZERO = 0
    PLUS = 1


class Crossing(LineIntersection):
    """A LineIntersection subclass: the right fields, another class."""


class Rows(Matrix):
    """A Matrix subclass: the right rows, another class."""


@dataclass(frozen=True)
class Answer:
    """A record that holds a right answer: no value of distance, polygon_area or orientation is a record, so the
    checker refuses every record, this one, an impostor of it and one forged with object.__new__."""
    value: object


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def forged_intersection(**fields: object) -> LineIntersection:
    """A LineIntersection made with object.__new__, whose fields may break the canonical form."""
    return support.forged(LineIntersection, **fields)


def inexact(item: Number) -> list[tuple[str, object]]:
    """(label, stand-in) for an exact number that is no exact int and no canonical Rational."""
    p, q = frac(item).numerator, frac(item).denominator
    return [("a bool", bool(p)), ("a float", p / q), ("a str", str(item)), ("None", None),
            ("a Fraction", Fraction(p, q)), ("an int subclass", Integer(p)), ("a Rational subclass", Fractional(p, q)),
            ("a list", [item]), ("a tuple", (item,)),
            ("a Rational of a bool", forged_rational(bool(p), 1)), ("a Rational of a float", forged_rational(p / q, 1)),
            ("a Rational of a str", forged_rational(str(p), 1)),
            ("a Rational with a float denominator", forged_rational(p, float(q))),
            ("a Rational in higher terms", forged_rational(2 * p, 2 * q)),
            ("a Rational with a negative denominator", forged_rational(-p, -q)),
            ("a Rational with a zero denominator", forged_rational(p, 0)),
            ("a Rational without a denominator", support.forged(Rational, numerator=p)),
            ("a Rational without a numerator", support.forged(Rational, denominator=q))]


def value_stand_ins(item: Number) -> list[tuple[str, object]]:
    """(label, stand-in) for a Rational of a value: the inexact stand-ins, an impostor and, when the number is
    integral, the int itself, which no value holds."""
    stand_ins = inexact(item) + [("an impostor Rational", support.impostor(rational(item))),
                                 ("a record", Answer(rational(item)))]
    if frac(item).denominator == 1:
        stand_ins.append(("the int", frac(item).numerator))
    return stand_ins


def corrupted_points(args: dict[str, tuple]) -> list[tuple[str, dict[str, object]]]:
    """(label, args) with each coordinate of each point in turn replaced by each inexact stand-in, and each point by a
    list, a tuple subclass, None, a Matrix and a str."""
    cases: list[tuple[str, dict[str, object]]] = []
    for name, value in args.items():
        for index, coordinate in enumerate(value):
            for label, item in inexact(coordinate):
                changed = value[:index] + (item,) + value[index + 1:]
                cases.append((f"{name}[{index}]: {label}", dict(args, **{name: changed})))
        cases += [(f"{name} as a list", dict(args, **{name: list(value)})),
                  (f"{name} as a tuple subclass", dict(args, **{name: Coordinates(value)})),
                  (f"{name} as None", dict(args, **{name: None})),
                  (f"{name} as a Matrix", dict(args, **{name: Matrix([list(value)])})),
                  (f"{name} as a str", dict(args, **{name: str(value)}))]
    return cases


def wrong_shapes(args: dict[str, object]) -> list[tuple[str, object]]:
    """(label, args) for the arguments in another container, or with a name missing, added, renamed or no str."""
    names = list(args)
    return [("the arguments as pairs", tuple(args.items())), ("the arguments as a list of pairs", list(args.items())),
            ("the arguments as a read-only mapping", MappingProxyType(args)), ("None as the arguments", None),
            ("an argument missing", {name: args[name] for name in names[:-1]}),
            ("an extra argument", dict(args, extra=args[names[0]])),
            ("an argument renamed", {**{name: args[name] for name in names[:-1]}, "renamed": args[names[-1]]}),
            ("a key that is no str", {**{name: args[name] for name in names[:-1]}, 1: args[names[-1]]}),
            ("a key that is a str subclass", {**{name: args[name] for name in names[:-1]},
                                              Text(names[-1]): args[names[-1]]})]


# The true claims that the tables of corruptions start from.
FIVE = {"p": (0, 0), "q": (3, 4)}  # distance 5
EIGHT = {"p": (0, 0), "q": (2, 2)}  # distance sqrt(8)
HALVES = {"p": (0, 0), "q": (HALF, HALF)}  # distance sqrt(1/2)
CROSSING = {"p1": (0, 0), "p2": (1, 1), "q1": (0, 1), "q2": (1, 0)}  # the point (1/2, 1/2)
PARALLEL = {"p1": (0, 0), "p2": (1, 0), "q1": (0, 1), "q2": (1, 1)}
SAME = {"p1": (0, 0), "p2": (1, 1), "q1": (2, 2), "q2": (3, 3)}  # coincident
CCW = {"p": (0, 0), "q": (1, 0), "r": (0, 1)}  # orientation 1
CW = {"p": (0, 0), "q": (0, 1), "r": (1, 0)}  # orientation -1
COLLINEAR = {"p": (0, 0), "q": (1, 1), "r": (2, 2)}  # orientation 0
TRIANGLE = ((0, 0), (4, 0), (0, 3))  # area 6
SQUARE = ((0, 0), (1, 0), (1, 1), (0, 1))  # area 1 counterclockwise


class PlaneCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: object, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        def points(*names: str) -> tuple[Argument, ...]:
            return tuple(Argument(name, "", None, "rational_list") for name in names)

        expected = (
            OperationSpec("distance", "geometry", points("p", "q"), "Expression",
                          "Compute the exact Euclidean distance between two rational points.", SCOPE, version=6),
            OperationSpec("polygon_area", "geometry", (Argument("points", "Matrix", None, "matrix"),), "Rational",
                          "Compute the signed shoelace area of a polygon given by its vertices in order.", SCOPE,
                          version=6),
            OperationSpec("line_intersection", "geometry", points("p1", "p2", "q1", "q2"), "LineIntersection",
                          "Intersect the line through p1, p2 with the line through q1, q2 in the plane.", SCOPE,
                          version=6),
            OperationSpec("orientation", "geometry", points("p", "q", "r"), "int",
                          "Decide the orientation of three rational points in the plane: 1 counterclockwise, "
                          "-1 clockwise, 0 collinear.", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.SCOPE, SCOPE)
        self.assertEqual(family.TYPES, {"line_intersection": LineIntersection})
        self.assertEqual(set(family.RENDER), {LineIntersection})
        self.assertEqual(family.CHECKER, "mathforge.certify.geometry.plane")
        self.assertIs(support.checker(family), checker)
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
                self.assertEqual(checker.ARGUMENTS[spec.name], tuple(argument.name for argument in spec.arguments))
        self.assertEqual(set(checker.ARGUMENTS), {spec.name for spec in expected})
        # The checker names the classes that it reads without importing them.
        self.assertEqual(checker.RECORD, (LineIntersection.__module__, LineIntersection.__qualname__))
        self.assertEqual(checker.MATRIX, (Matrix.__module__, Matrix.__qualname__))
        self.assertEqual(checker.KINDS, family.KINDS)

    def test_the_record_holds_its_canonical_form(self):
        for kind, crossing in (("point", (HALF, Rational(-3))), ("point", (Rational(0), Rational(0))),
                               ("parallel", None), ("coincident", None)):
            with self.subTest(kind=kind):
                record = LineIntersection(kind, crossing)
                self.assertEqual((record.kind, record.point), (kind, crossing))
        invalid = [
            ("point", None), ("point", (1, 2)), ("point", (HALF, 2)), ("point", [HALF, HALF]), ("point", (HALF,)),
            ("point", (HALF, HALF, HALF)), ("point", ()), ("point", (HALF, 0.5)), ("point", (HALF, True)),
            ("point", (HALF, Fraction(1, 2))), ("point", (HALF, Fractional(1, 2))),
            ("point", Coordinates((HALF, HALF))), ("point", (HALF, forged_rational(2, 4))),
            ("point", (HALF, forged_rational(-1, -2))), ("point", (HALF, forged_rational(1, 0))),
            ("point", (HALF, forged_rational(1.0, 2))), ("point", (HALF, support.forged(Rational, numerator=1))),
            ("parallel", (HALF, HALF)), ("parallel", ()), ("coincident", (HALF, HALF)), ("coincident", 0),
            ("crossing", None), ("Point", (HALF, HALF)), ("", None), (Text("point"), (HALF, HALF)),
            (Text("parallel"), None), (b"parallel", None), (None, None), (1, None), (("point",), (HALF, HALF)),
        ]
        for index, (kind, crossing) in enumerate(invalid):
            with self.subTest(case=index):  # A forged Rational without a field has no repr.
                with self.assertRaises(InvalidInput):
                    LineIntersection(kind, crossing)
        with self.assertRaises(ResourceLimitError):  # A coordinate longer than max_integer_bits.
            LineIntersection("point", (HALF, Rational(GIANT)))

    def test_rendering(self):
        cases = [
            (LineIntersection("point", (HALF, Rational(-3))), "(1/2, -3)", r"\left(\frac{1}{2}, -3\right)"),
            (LineIntersection("parallel", None), "parallel lines", r"\text{parallel lines}"),
            (LineIntersection("coincident", None), "coincident lines", r"\text{coincident lines}"),
        ]
        for value, text, latex in cases:
            with self.subTest(kind=value.kind):
                self.assertEqual(mf.render_text(value), text)
                self.assertEqual(mf.to_latex(value), latex)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_distance_accepts_true_claims(self):
        for p, q, value in (((0, 0), (3, 4), Rational(5)), ((0, 0), (1, 1), Sqrt(Rational(2))),
                            ((1, 2, 3), (1, 2, 3), Rational(0))):  # The card's values.
            with self.subTest(p=p, q=q):
                args, found = distance_claim(p, q)
                self.assertEqual(found, value)
                self.assertAccepts("distance", args, value)
        third = Fraction(1, 3)
        cases = [
            ("one dimension", (7,), (-2,), Rational(9)),
            ("a rational distance between rational points", (0, 0), (Fraction(3, 5), Fraction(4, 5)), Rational(1)),
            ("a rational distance that is no integer", (0, 0), (Fraction(3, 10), Fraction(2, 5)), HALF),
            ("the radicand 1/2", (0, 0), (Fraction(1, 2), Fraction(1, 2)), Sqrt(HALF)),
            ("the radicand 8, no factor taken out", (0, 0), (2, 2), Sqrt(Rational(8))),
            ("the radicand 2/9, a square denominator", (0, 0), (third, third), Sqrt(Rational(2, 9))),
            ("the radicand 9/2, a square numerator", (0, 0), (Fraction(3, 2), Fraction(3, 2)), Sqrt(Rational(9, 2))),
            ("negative coordinates", (-1, -2, -3), (2, 2, 9), Rational(13)),
            ("four dimensions", (1, 1, 1, 1), (2, 2, 2, 2), Rational(2)),
            ("fifty dimensions", tuple(range(50)), tuple(range(1, 51)), Sqrt(Rational(50))),
            ("long coordinates", (2 ** 200, -(3 ** 100)), (5, Fraction(1, 2 ** 90)), None),
            ("a long rational distance", (0, 0), (3 * 2 ** 300, 4 * 2 ** 300), Rational(5 * 2 ** 300)),
        ]
        for label, p, q, value in cases:
            with self.subTest(case=label):
                args, found = distance_claim(p, q)
                if value is not None:
                    self.assertEqual(found, value)
                self.assertAccepts("distance", args, found)
        # 3 and Rational(3) are the same coordinate in a request.
        self.assertAccepts("distance", {"p": (Rational(3), Rational(0)), "q": (0, 4)}, Rational(5))
        rng = random.Random(20261112)
        directions = ((3, 4), (5, 12), (8, 15), (1, 2, 2), (2, 3, 6), (2, 6, 9), (1, 4, 8), (1,))  # Integer norms.
        for case in range(150):
            if case % 3:  # Random points, mostly at an irrational distance.
                dimension = rng.randrange(1, 7)
                bits, denominator_bits = rng.choice(((3, 0), (20, 0), (8, 4), (40, 20)))
                p = tuple(number(rng, bits, denominator_bits) for _ in range(dimension))
                q = tuple(number(rng, bits, denominator_bits) for _ in range(dimension))
            else:  # q = p + k * direction: the rational distance |k| * |direction|.
                direction = rng.choice(directions)
                scale = number(rng, 10, rng.choice((0, 6))) or Fraction(1)
                p = tuple(number(rng, 12, 3) for _ in direction)
                q = tuple(a + scale * b for a, b in zip(p, direction))
            with self.subTest(case=case, p=p, q=q):
                self.assertAccepts("distance", *distance_claim(p, q))

    def test_polygon_area_accepts_true_claims(self):
        for label, vertices, area in (("the unit square counterclockwise", SQUARE, 1),
                                      ("the unit square clockwise", SQUARE[::-1], -1),
                                      ("the triangle (0, 0), (4, 0), (0, 3)", TRIANGLE, 6)):  # The card's values.
            with self.subTest(case=label):
                args, value = area_claim(vertices)
                self.assertEqual(value, Rational(area))
                self.assertAccepts("polygon_area", args, value)
        zero = Rational(0)
        third = Fraction(1, 3)
        cases = [
            ("one vertex", ((3, 5),), zero),
            ("two vertices", ((1, 2), (5, -7)), zero),
            ("a repeated closing vertex", SQUARE + ((0, 0),), Rational(1)),
            ("a bow-tie whose loops cancel", ((0, 0), (1, 1), (1, 0), (0, 1)), zero),
            ("a bow-tie with unequal loops", ((0, 0), (2, 2), (2, 0), (0, 1)), None),
            ("three collinear vertices", ((0, 0), (1, 1), (2, 2)), zero),
            ("a non-convex polygon", ((0, 0), (4, 0), (4, 4), (2, 1), (0, 4)), Rational(10)),
            ("a vertex visited twice: two counterclockwise loops add", ((0, 0), (2, 0), (2, 2), (0, 0), (-2, 0),
                                                                         (-2, -2)), Rational(4)),
            ("a vertex visited twice: opposite loops cancel", ((0, 0), (2, 0), (2, 2), (0, 0), (-2, -2), (-2, 0)),
             zero),
            ("rational vertices", ((third, 0), (1, Fraction(1, 7)), (Fraction(-5, 2), Fraction(2, 3))), None),
            ("vertices far from the origin", ((10 ** 30, 10 ** 30), (10 ** 30 + 4, 10 ** 30), (10 ** 30, 10 ** 30 + 3)),
             Rational(6)),
            ("long coordinates", ((2 ** 300, -(3 ** 150)), (Fraction(1, 2 ** 200), 7), (-5, Fraction(2 ** 100, 3))),
             None),
        ]
        for label, vertices, area in cases:
            with self.subTest(case=label):
                args, value = area_claim(vertices)
                if area is not None:
                    self.assertEqual(value, area)
                self.assertAccepts("polygon_area", args, value)
        # No vertex: the empty Matrix of two columns, and Matrix([]), whose column count is 0.
        self.assertEqual(Matrix([]).shape, (0, 0))
        self.assertAccepts("polygon_area", {"points": Matrix((), ncols=2)}, zero)
        self.assertAccepts("polygon_area", {"points": Matrix([])}, zero)
        rng = random.Random(20261113)
        for case in range(120):
            count = rng.randrange(0, 25)
            bits, denominator_bits = rng.choice(((3, 0), (30, 0), (8, 3), (20, 12)))
            vertices = tuple((number(rng, bits, denominator_bits), number(rng, bits, denominator_bits))
                             for _ in range(count))
            with self.subTest(case=case, vertices=vertices):
                self.assertAccepts("polygon_area", *area_claim(vertices))

    def test_line_intersection_accepts_true_claims(self):
        for points, value in ((((0, 0), (1, 1), (0, 1), (1, 0)), LineIntersection("point", (HALF, HALF))),
                              (((0, 0), (1, 0), (0, 1), (1, 1)), LineIntersection("parallel", None)),
                              (((0, 0), (1, 1), (2, 2), (3, 3)), LineIntersection("coincident", None))):  # The card's.
            with self.subTest(points=points):
                args, found = line_claim(*points)
                self.assertEqual(found, value)
                self.assertAccepts("line_intersection", args, value)
        cases = [
            ("a vertical and a horizontal line", (2, -5), (2, 7), (-1, 3), (4, 3), ("point", (2, 3))),
            ("crossing at p1", (1, 1), (2, 3), (1, 1), (5, 0), ("point", (1, 1))),
            ("crossing outside both segments", (0, 0), (1, 0), (5, 5), (5, 4), ("point", (5, 0))),
            ("a rational crossing", (0, 0), (1, 2), (0, 1), (1, 0), ("point", (Fraction(1, 3), Fraction(2, 3)))),
            ("parallel vertical lines", (0, 0), (0, 1), (1, 5), (1, -5), ("parallel", None)),
            ("parallel lines of a rational slope", (0, 0), (3, 1), (0, 1), (6, 3), ("parallel", None)),
            ("the same line in opposite directions", (0, 0), (2, 1), (6, 3), (-4, -2), ("coincident", None)),
            ("the same line through a shared point", (1, 1), (2, 2), (1, 1), (3, 3), ("coincident", None)),
            ("the same line from the same points", (0, 0), (1, 2), (0, 0), (1, 2), ("coincident", None)),
            ("rational points", (Fraction(1, 3), 0), (1, Fraction(1, 7)), (Fraction(-5, 2), 2), (0, Fraction(2, 3)),
             None),
            ("long coordinates", (2 ** 100, 3 ** 60), (-(5 ** 40), 7), (Fraction(1, 2 ** 70), 11), (13, -(2 ** 90)),
             None),
        ]
        for label, p1, p2, q1, q2, expected in cases:
            with self.subTest(case=label):
                args, value = line_claim(p1, p2, q1, q2)
                if expected is not None:
                    kind, crossing = expected
                    self.assertEqual(value, LineIntersection(
                        kind, None if crossing is None else tuple(rational(item) for item in crossing)))
                self.assertAccepts("line_intersection", args, value)
        # 1 and Rational(1) are the same coordinate in a request.
        self.assertAccepts("line_intersection", {"p1": (Rational(0), 0), "p2": (1, Rational(1)), "q1": (0, 1),
                                                 "q2": (Rational(1), Rational(0))},
                           LineIntersection("point", (HALF, HALF)))
        rng = random.Random(20261114)
        kinds = set()
        for case in range(400):
            bits, denominator_bits = rng.choice(((2, 0), (2, 0), (6, 0), (6, 2), (30, 10)))
            points = [(number(rng, bits, denominator_bits), number(rng, bits, denominator_bits)) for _ in range(4)]
            if points[0] == points[1] or points[2] == points[3]:
                continue
            if case % 5 == 0:  # The second line parallel to the first: through q1 with the direction of p2 - p1.
                step = number(rng, 3, 1) or Fraction(1)
                points[3] = tuple(c + step * d for c, d in zip(points[2], minus(points[1], points[0])))
            args, value = line_claim(*points)
            kinds.add(value.kind)
            with self.subTest(case=case, points=points):
                self.assertAccepts("line_intersection", args, value)
        self.assertEqual(kinds, {"point", "parallel", "coincident"})

    def test_orientation_accepts_true_claims(self):
        for args, value in ((CCW, 1), (CW, -1), (COLLINEAR, 0)):
            with self.subTest(args=args):
                self.assertEqual(orientation_claim(args["p"], args["q"], args["r"])[1], value)
                self.assertAccepts("orientation", args, value)
        for label, p, q, r in (("p == q", (1, 2), (1, 2), (5, 7)), ("all three equal", (3, 3), (3, 3), (3, 3)),
                               ("q == r", (0, 0), (1, 5), (1, 5)), ("rational points", (Fraction(1, 3), 0),
                                                                   (1, Fraction(1, 7)), (Fraction(-5, 2), 2)),
                               ("long coordinates", (2 ** 300, 1), (-(3 ** 200), 2 ** 250), (5, Fraction(1, 2 ** 99)))):
            with self.subTest(case=label):
                self.assertAccepts("orientation", *orientation_claim(p, q, r))
        # Brute force on a grid: every ordered triple of the 3 x 3 grid {0, 1, 2}^2.
        grid = list(itertools.product(range(3), repeat=2))
        counts = {-1: 0, 0: 0, 1: 0}
        for p, q, r in itertools.product(grid, repeat=3):
            args, value = orientation_claim(p, q, r)
            counts[value] += 1
            self.assertAccepts("orientation", args, value)
        self.assertEqual(counts[1], counts[-1])  # Swapping q and r reverses every turn.
        self.assertEqual(sum(counts.values()), 9 ** 3)
        rng = random.Random(20261115)
        for case in range(150):
            bits, denominator_bits = rng.choice(((4, 0), (30, 0), (10, 5), (60, 30)))
            p, q = [(number(rng, bits, denominator_bits), number(rng, bits, denominator_bits)) for _ in range(2)]
            if case % 4 == 0:  # r on the line through p and q.
                step = number(rng, 5, 3)
                r = tuple(a + step * d for a, d in zip(p, minus(q, p)))
            else:
                r = (number(rng, bits, denominator_bits), number(rng, bits, denominator_bits))
            with self.subTest(case=case):
                self.assertAccepts("orientation", *orientation_claim(p, q, r))

    def test_full_size_claims_fit_the_default_budget(self):
        """Claims on the largest inputs that a producer can answer under the default limits verify within them."""
        rng = random.Random(20261116)
        limits = ComputationLimits()
        a, b = rng.getrandbits(32_000) | 1 << 31_999, rng.getrandbits(32_000)
        r = a * a + b * b  # Nearly max_integer_bits.
        self.assertLessEqual(r.bit_length(), limits.max_integer_bits)
        dimension = 10_000
        long_p = tuple(rng.randrange(-1000, 1000) for _ in range(dimension))
        long_q = tuple(rng.randrange(-1000, 1000) for _ in range(dimension))
        wide = tuple((rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(-10 ** 6, 10 ** 6)) for _ in range(2048))
        sevenths = tuple((Fraction(rng.randrange(-10 ** 9, 10 ** 9), 7), Fraction(rng.randrange(10 ** 9), 7 ** 3))
                         for _ in range(2048))
        corners = [(rng.getrandbits(16_000), rng.getrandbits(16_000)) for _ in range(4)]
        claims = [
            ("distance", *distance_claim((0, 0), (a, b))),
            ("distance", *distance_claim(long_p, long_q)),
            ("polygon_area", *area_claim(wide)),  # 4,096 entries: max_matrix_entries.
            ("polygon_area", *area_claim(sevenths)),
            ("line_intersection", *line_claim(*corners)),
            ("orientation", *orientation_claim((rng.getrandbits(65_000), 3), (1, rng.getrandbits(65_000)),
                                               (rng.getrandbits(65_000), rng.getrandbits(65_000)))),
        ]
        point_value = claims[4][2]
        self.assertGreater(point_value.point[0].numerator.bit_length(), 40_000)
        for operation, args, value in claims:
            with self.subTest(operation=operation):
                start = time.perf_counter()
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, ()), True)
                self.assertLess(time.perf_counter() - start, FAST)
                self.assertLess(budget.work, limits.max_work)

    def test_sums_that_cancel_stay_within_the_budget(self):
        """Regression (review of wave m4b): sums whose terms cancel verify far within max_work, where the producer
        answers them in a few thousand units. Unreduced, the checker's running sum kept the least common multiple of
        every term's denominator: the distance between two equal points of 300 coordinates over nearly coprime
        2,000-bit denominators, and the trapezoid sum of 200 such vertices on the line y = 2x (whose terms telescope
        to 0), each exhausted the default max_work although produce() answers 0 in under 3,000 units."""
        denominators = [2 ** 1999 + 2 * index + 1 for index in range(300)]  # Pairwise gcds divide differences < 600.
        equal = tuple(Fraction(1, denominator) for denominator in denominators)
        on_line = [(Fraction(3, denominator), Fraction(6, denominator)) for denominator in denominators[:200]]
        claims = [
            ("distance", *distance_claim(equal, equal)),
            ("polygon_area", *area_claim(on_line)),
            ("polygon_area", *area_claim(on_line[::-1])),
        ]
        for operation, args, value in claims:
            with self.subTest(operation=operation):
                self.assertEqual(value, Rational(0))
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, ()), True)
                self.assertLess(budget.work, 500_000)  # Measured about 1,600 and 357,000 units.
                for wrong in (Rational(1), Rational(-1, 2 ** 1999 + 1), Sqrt(HALF)):
                    self.assertRejects(operation, args, wrong)
        # Reducing changes no verdict: the sum over the same denominators that does not cancel is the shoelace area.
        lifted = on_line[:40] + [(Fraction(1), Fraction(5))]
        self.assertAccepts("polygon_area", *area_claim(lifted))

    # -----------------------------------------------------------------------------------------------------------
    # Claims on inputs that have no value

    def test_rejects_claims_on_inputs_outside_the_domain(self):
        # distance: points of different dimensions, and empty points.
        cases = [(f"{p} and {q}, value {value}", {"p": p, "q": q}, value, ())
                 for p, q in (((0, 0), (3, 4, 0)), ((3,), (0, 0)), ((), ()), ((), (1,)), ((1,), ()))
                 for value in (Rational(0), Rational(3), Rational(5), Sqrt(Rational(2)))]
        self.assertRejectsAll("distance", cases)
        # polygon_area: a Matrix whose rows are no plane points, also without rows when it declares the columns.
        cases = []
        for label, matrix in (("three columns", Matrix([[0, 0, 0], [4, 0, 0], [0, 3, 0]])),
                              ("one column", Matrix([[0], [4], [0]])), ("two rows of nothing", Matrix([[], []])),
                              ("no rows of three columns", Matrix((), ncols=3)),
                              ("no rows of one column", Matrix((), ncols=1)),
                              ("the transposed triangle", Matrix([[0, 4, 0], [0, 0, 3]])),
                              ("65 x 65", support.oversized_matrix())):
            cases += [(f"{label}, value {value}", {"points": matrix}, Rational(value), ()) for value in (0, 6, -6)]
        self.assertRejectsAll("polygon_area", cases)
        # line_intersection: p1 == p2 or q1 == q2 determines no line (every determinant vanishes, so "coincident"
        # would pass), also with an int and a Rational of the same number; and points of another dimension.
        values = [LineIntersection("point", (HALF, HALF)), LineIntersection("point", (Rational(0), Rational(0))),
                  LineIntersection("parallel", None), LineIntersection("coincident", None)]
        cases = []
        for label, args in (("p1 == p2", {"p1": (0, 0), "p2": (0, 0), "q1": (0, 1), "q2": (1, 0)}),
                            ("q1 == q2", {"p1": (0, 0), "p2": (1, 1), "q1": (2, 2), "q2": (2, 2)}),
                            ("both lines degenerate", {"p1": (5, 5), "p2": (5, 5), "q1": (5, 5), "q2": (5, 5)}),
                            ("p1 == p2 as 2 and Rational(2)", {"p1": (2, HALF), "p2": (Rational(2), HALF),
                                                               "q1": (0, 1), "q2": (1, 0)}),
                            ("three dimensions", {"p1": (0, 0, 0), "p2": (1, 1, 0), "q1": (0, 1, 0),
                                                  "q2": (1, 0, 0)}),
                            ("one point in three dimensions", dict(CROSSING, q2=(1, 0, 0))),
                            ("one dimension", {"p1": (0,), "p2": (1,), "q1": (2,), "q2": (3,)}),
                            ("an empty point", dict(CROSSING, p1=()))):
            cases += [(f"{label}, {value.kind}", args, value, ()) for value in values]
        self.assertRejectsAll("line_intersection", cases)
        # orientation: points of another dimension.
        cases = [(f"{args}, value {value}", args, value, ())
                 for args in ({"p": (0, 0, 0), "q": (1, 0, 0), "r": (0, 1, 0)}, dict(CCW, r=(0, 1, 5)),
                              {"p": (0,), "q": (1,), "r": (2,)}, dict(CCW, p=()))
                 for value in (-1, 0, 1)]
        self.assertRejectsAll("orientation", cases)

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_distance_rejects_corrupted_claims(self):
        root8, root_half = Sqrt(Rational(8)), Sqrt(HALF)
        self.assertAccepts("distance", FIVE, Rational(5))
        self.assertAccepts("distance", EIGHT, root8)
        self.assertAccepts("distance", HALVES, root_half)
        equal = {"p": (1, 2), "q": (1, 2)}
        wrong = [
            ("4", FIVE, Rational(4)), ("6", FIVE, Rational(6)), ("-5", FIVE, Rational(-5)),
            ("25, the square", FIVE, Rational(25)), ("5/2", FIVE, Rational(5, 2)), ("0", FIVE, Rational(0)),
            ("sqrt(24)", FIVE, Sqrt(Rational(24))), ("sqrt(26)", FIVE, Sqrt(Rational(26))),
            ("sqrt(8) for 5", FIVE, root8), ("sqrt(2) for sqrt(8)", EIGHT, Sqrt(Rational(2))),
            ("sqrt(7) for sqrt(8)", EIGHT, Sqrt(Rational(7))), ("3 for sqrt(8)", EIGHT, Rational(3)),
            ("sqrt(2) for sqrt(1/2)", HALVES, Sqrt(Rational(2))), ("1/2 for sqrt(1/2)", HALVES, HALF),
            ("sqrt(1/4) for sqrt(1/2)", HALVES, Sqrt(Rational(1, 4))), ("1 for equal points", equal, Rational(1)),
            ("the distance of other points", {"p": (0, 0), "q": (3, 5)}, Rational(5)),
            ("a dimension left out", {"p": (0, 0, 2), "q": (3, 4, 0)}, Rational(5)),
        ]
        # Right in value, wrong in form: model.sqrt(r) is the only canonical form.
        non_canonical = [
            ("sqrt(25) for 5", FIVE, Sqrt(Rational(25))), ("sqrt(0) for equal points", equal, Sqrt(Rational(0))),
            ("sqrt(1) for 1", {"p": (0,), "q": (1,)}, Sqrt(Rational(1))),
            ("2*sqrt(2) for sqrt(8)", EIGHT, Mul((Rational(2), Sqrt(Rational(2))))),
            ("sqrt(2) + sqrt(2) for sqrt(8)", EIGHT, Add((Sqrt(Rational(2)), Sqrt(Rational(2))))),
            ("sqrt(8)**1", EIGHT, Pow(root8, 1)),
            ("sqrt(2)/2 for sqrt(1/2)", HALVES, Mul((HALF, Sqrt(Rational(2))))),
            ("-sqrt(8)", EIGHT, Mul((Rational(-1), root8))),
            ("sqrt(8) with the radicand 16/2", EIGHT, support.forged(Sqrt, radicand=forged_rational(16, 2))),
            ("sqrt(8) with the radicand -8/-1", EIGHT, support.forged(Sqrt, radicand=forged_rational(-8, -1))),
            ("sqrt(1/2) with the radicand 2/4", HALVES, support.forged(Sqrt, radicand=forged_rational(2, 4))),
            ("sqrt(8) without a radicand", EIGHT, support.forged(Sqrt)),
            ("a negative radicand", EIGHT, support.forged(Sqrt, radicand=Rational(-8))),
            ("an impostor Sqrt", EIGHT, support.impostor(root8)),
            ("the radicand alone", EIGHT, Rational(8)), ("the tuple (5,)", FIVE, (Rational(5),)),
            ("None", FIVE, None), ("a str", FIVE, "5"), ("the record of another family", FIVE,
                                                       LineIntersection("point", (Rational(5), Rational(0)))),
        ]
        non_canonical += [(f"5 as {label}", FIVE, item) for label, item in value_stand_ins(5)]
        non_canonical += [(f"sqrt(8) with the radicand as {label}", EIGHT, support.forged(Sqrt, radicand=item))
                          for label, item in value_stand_ins(8)]
        cases = [(label, args, value, ()) for label, args, value in wrong + non_canonical]
        cases += [(label, args, Rational(5), ()) for label, args in corrupted_points(FIVE) + wrong_shapes(FIVE)]
        cases += [(label, args, root8, ()) for label, args in corrupted_points(EIGHT)]
        self.assertRejectsAll("distance", cases)

    def test_polygon_area_rejects_corrupted_claims(self):
        args, six = area_claim(TRIANGLE)
        clockwise, minus_one = area_claim(SQUARE[::-1])
        offset, offset_area = area_claim(((1, 1), (5, 1), (1, 4)))  # Area 6 off the origin.
        # The edges of this bow-tie cross at (2/3, 2/3): a counterclockwise loop of area 1/3, a clockwise one of 4/3.
        bow_tie, bow_tie_area = area_claim(((0, 0), (2, 2), (2, 0), (0, 1)))
        self.assertEqual((six, minus_one, offset_area, bow_tie_area),
                         (Rational(6), Rational(-1), Rational(6), Rational(-1)))
        self.assertEqual((shoelace(((0, 0), (Fraction(2, 3), Fraction(2, 3)), (0, 1))),
                          shoelace(((Fraction(2, 3), Fraction(2, 3)), (2, 2), (2, 0)))),
                         (Fraction(1, 3), Fraction(-4, 3)))
        for claim in ((args, six), (clockwise, minus_one), (offset, offset_area), (bow_tie, bow_tie_area)):
            self.assertAccepts("polygon_area", *claim)
        wrong = [
            ("-6", args, Rational(-6)), ("twice the area", args, Rational(12)), ("half the area", args, Rational(3)),
            ("7", args, Rational(7)), ("0", args, Rational(0)), ("the unsigned area of a clockwise square", clockwise,
                                                                 Rational(1)),
            ("the open path without its closing edge", offset, Rational(15, 2)),
            ("the sum of the loops' unsigned areas", bow_tie, Rational(5, 3)),
            ("the counterclockwise loop alone", bow_tie, Rational(1, 3)),
            ("the clockwise loop alone", bow_tie, Rational(-4, 3)), ("the unsigned area", bow_tie, Rational(1)),
            ("1 for no vertex", {"points": Matrix([])}, Rational(1)),
        ]
        non_canonical = [(f"6 as {label}", args, item) for label, item in value_stand_ins(6)]
        non_canonical += [("0 as the int 0", {"points": Matrix([])}, 0), ("the tuple (6,)", args, (Rational(6),)),
                          ("Sqrt(36)", args, Sqrt(Rational(36))), ("None", args, None)]
        matrix = args["points"]
        rows = matrix.rows

        def forged_matrix(**fields: object) -> Matrix:
            return support.forged(Matrix, **fields)

        arguments = [
            ("a tuple of rows", rows), ("a list of rows", [list(row) for row in rows]),
            ("an impostor Matrix", support.impostor(matrix)), ("a Matrix subclass", Rows(rows)),
            ("a forged Matrix with a list of rows", forged_matrix(rows=list(rows), ncols=2)),
            ("a forged Matrix with a tuple subclass of rows", forged_matrix(rows=Coordinates(rows), ncols=2)),
            ("a forged Matrix with a list row", forged_matrix(rows=(list(rows[0]),) + rows[1:], ncols=2)),
            ("a forged Matrix with a tuple-subclass row", forged_matrix(rows=(Coordinates(rows[0]),) + rows[1:],
                                                                        ncols=2)),
            ("a forged Matrix of ints", forged_matrix(rows=TRIANGLE, ncols=2)),
            ("a forged Matrix with ncols 3", forged_matrix(rows=rows, ncols=3)),
            ("a forged Matrix with ncols True", forged_matrix(rows=rows, ncols=True)),
            ("a forged Matrix with ncols 2.0", forged_matrix(rows=rows, ncols=2.0)),
            ("a forged Matrix with a row of three", forged_matrix(rows=rows[:2] + (rows[2] + (Rational(0),),),
                                                                  ncols=2)),
            ("a forged Matrix with a row of one", forged_matrix(rows=rows[:2] + (rows[2][:1],), ncols=2)),
            ("a forged Matrix of ncols 0 with rows", forged_matrix(rows=rows, ncols=0)),
            ("a forged Matrix without rows", forged_matrix(ncols=2)),
            ("a forged Matrix without ncols", forged_matrix(rows=rows)),
            ("a forged Matrix with rows None", forged_matrix(rows=None, ncols=2)),
            ("a list of points", [list(vertex) for vertex in TRIANGLE]), ("None", None),
        ]
        for label, item in inexact(3):
            arguments.append((f"an entry as {label}", forged_matrix(rows=rows[:2] + ((Rational(0), item),), ncols=2)))
        cases = [(label, claim_args, value, ()) for label, claim_args, value in wrong + non_canonical]
        cases += [(label, {"points": item}, six, ()) for label, item in arguments]
        cases += [(label, claim_args, six, ()) for label, claim_args in wrong_shapes(args)]
        self.assertRejectsAll("polygon_area", cases)

    def test_line_intersection_rejects_corrupted_claims(self):
        crossing = LineIntersection("point", (HALF, HALF))
        parallel, coincident = LineIntersection("parallel", None), LineIntersection("coincident", None)
        slanted, slanted_value = line_claim((0, 0), (1, 2), (0, 1), (1, 0))  # The point (1/3, 2/3).
        integral, integral_value = line_claim((0, 0), (2, 0), (1, -1), (1, 1))  # The point (1, 0).
        self.assertEqual(integral_value, LineIntersection("point", (Rational(1), Rational(0))))
        for claim in ((CROSSING, crossing), (PARALLEL, parallel), (SAME, coincident), (slanted, slanted_value),
                      (integral, integral_value)):
            self.assertAccepts("line_intersection", *claim)
        one, zero = Rational(1), Rational(0)

        def at(x: Number, y: Number) -> LineIntersection:
            return LineIntersection("point", (rational(x), rational(y)))

        wrong = [
            ("parallel for crossing lines", CROSSING, parallel),
            ("coincident for crossing lines", CROSSING, coincident),
            ("a point for parallel lines", PARALLEL, crossing), ("coincident for parallel lines", PARALLEL, coincident),
            ("parallel for the same line", SAME, parallel), ("a point of the same line", SAME, at(1, 1)),
            ("a point off both lines", CROSSING, at(1, HALF)), ("a point on the first line only", CROSSING, at(1, 1)),
            ("a point on the second line only", CROSSING, at(0, 1)),
            ("the coordinates swapped", slanted, at(Fraction(2, 3), Fraction(1, 3))),
            ("a point off by 2**-80", slanted, at(Fraction(1, 3) + Fraction(1, 2 ** 80), Fraction(2, 3))),
            ("the crossing of other lines", integral, crossing),
        ]
        non_canonical = [
            ("int coordinates", integral, forged_intersection(kind="point", point=(1, 0))),
            ("a list point", CROSSING, forged_intersection(kind="point", point=[HALF, HALF])),
            ("a tuple-subclass point", CROSSING, forged_intersection(kind="point", point=Coordinates((HALF, HALF)))),
            ("one coordinate", CROSSING, forged_intersection(kind="point", point=(HALF,))),
            ("three coordinates", CROSSING, forged_intersection(kind="point", point=(HALF, HALF, zero))),
            ("no point for the kind point", CROSSING, forged_intersection(kind="point", point=None)),
            ("an empty point for parallel lines", PARALLEL, forged_intersection(kind="parallel", point=())),
            ("a point for parallel lines", PARALLEL, forged_intersection(kind="parallel", point=(HALF, HALF))),
            ("a point for the same line", SAME, forged_intersection(kind="coincident", point=(one, one))),
            ("the kind Point", CROSSING, forged_intersection(kind="Point", point=(HALF, HALF))),
            ("the kind 'point '", CROSSING, forged_intersection(kind="point ", point=(HALF, HALF))),
            ("the kind intersection", CROSSING, forged_intersection(kind="intersection", point=(HALF, HALF))),
            ("an empty kind", PARALLEL, forged_intersection(kind="", point=None)),
            ("the kind as bytes", PARALLEL, forged_intersection(kind=b"parallel", point=None)),
            ("the kind as a str subclass", PARALLEL, forged_intersection(kind=Text("parallel"), point=None)),
            ("the kind as a str subclass, point", CROSSING, forged_intersection(kind=Text("point"),
                                                                                point=(HALF, HALF))),
            ("the kind None", PARALLEL, forged_intersection(kind=None, point=None)),
            ("the kind 1", PARALLEL, forged_intersection(kind=1, point=None)),
            ("the kind as a tuple", CROSSING, forged_intersection(kind=("point",), point=(HALF, HALF))),
            ("no kind field", CROSSING, forged_intersection(point=(HALF, HALF))),
            ("no point field", PARALLEL, forged_intersection(kind="parallel")),
            ("no fields", PARALLEL, forged_intersection()),
            ("an impostor record", CROSSING, support.impostor(crossing)),
            ("an impostor record, parallel", PARALLEL, support.impostor(parallel)),
            ("a subclass record", CROSSING, Crossing("point", (HALF, HALF))),
            ("the point alone", CROSSING, (HALF, HALF)), ("the kind alone", PARALLEL, "parallel"),
            ("None", PARALLEL, None), ("a record holding the point", CROSSING, Answer((HALF, HALF))),
        ]
        for label, item in inexact(HALF):
            non_canonical.append((f"x as {label}", CROSSING, forged_intersection(kind="point", point=(item, HALF))))
            non_canonical.append((f"y as {label}", CROSSING, forged_intersection(kind="point", point=(HALF, item))))
        for label, item in value_stand_ins(1):
            non_canonical.append((f"1 as {label}", integral, forged_intersection(kind="point", point=(item, zero))))
        cases = [(label, args, value, ()) for label, args, value in wrong + non_canonical]
        cases += [(label, args, crossing, ()) for label, args in corrupted_points(CROSSING) + wrong_shapes(CROSSING)]
        cases += [(label, args, parallel, ()) for label, args in corrupted_points(PARALLEL)]
        self.assertRejectsAll("line_intersection", cases)

    def test_orientation_rejects_corrupted_claims(self):
        big = {"p": (0, 0), "q": (4, 0), "r": (0, 3)}  # cross2(q - p, r - p) = 12.
        for args, value in ((CCW, 1), (CW, -1), (COLLINEAR, 0), (big, 1)):
            self.assertAccepts("orientation", args, value)
        wrong = [
            ("-1 for counterclockwise", CCW, -1), ("0 for counterclockwise", CCW, 0), ("1 for clockwise", CW, 1),
            ("0 for clockwise", CW, 0), ("1 for collinear", COLLINEAR, 1), ("-1 for collinear", COLLINEAR, -1),
            ("2", CCW, 2), ("-2", CW, -2), ("the cross product 12", big, 12), ("the area 6", big, 6),
        ]
        non_canonical = [
            ("True", CCW, True), ("False for collinear", COLLINEAR, False), ("1.0", CCW, 1.0), ("'1'", CCW, "1"),
            ("Rational(1)", CCW, Rational(1)), ("Rational(-1)", CW, Rational(-1)), ("an IntEnum", CCW, Sign.PLUS),
            ("an IntEnum 0", COLLINEAR, Sign.ZERO), ("an IntEnum -1", CW, Sign.MINUS), ("an int subclass", CCW,
                                                                                        Integer(1)),
            ("None", CCW, None), ("(1,)", CCW, (1,)), ("[1]", CCW, [1]), ("a record", CCW, Answer(1)),
            ("an impostor record", CCW, support.impostor(Answer(1))), ("a forged record", CCW,
                                                                      support.forged(Answer, value=1)),
            ("the str counterclockwise", CCW, "counterclockwise"), ("-0.0 for collinear", COLLINEAR, -0.0),
            ("a Fraction", CCW, Fraction(1)),
        ]
        cases = [(label, args, value, ()) for label, args, value in wrong + non_canonical]
        cases += [(label, args, 1, ()) for label, args in corrupted_points(CCW) + wrong_shapes(CCW)]
        self.assertRejectsAll("orientation", cases)

    def test_rejects_wrong_operations_and_evidence(self):
        claims = [("distance", FIVE, Rational(5)), ("polygon_area", area_claim(TRIANGLE)[0], Rational(6)),
                  ("line_intersection", CROSSING, LineIntersection("point", (HALF, HALF))), ("orientation", CCW, 1)]
        for operation, args, value in claims:
            self.assertAccepts(operation, args, value)
            for name in ("Distance", "distance ", "area", "polygon area", "polygonArea", "intersection", "orient",
                         "orientation2", "continued_fraction", "powmod", "dot", "cross", "", None, 7,
                         (operation,), b"distance", [operation], Text(operation)):
                with self.subTest(operation=operation, name=name):
                    self.assertRejects(name, args, value)
            # A true claim of one operation under the name of another.
            for other in ("distance", "polygon_area", "line_intersection", "orientation"):
                if other != operation:
                    with self.subTest(operation=operation, other=other):
                        self.assertRejects(other, args, value)
        # The evidence of every operation is the empty tuple: none is missing, everything else is extra or of the
        # wrong shape, also evidence that is consistent with the claim (the radicand, the trapezoid terms, Cramer's
        # determinant and parameter, the cross product).
        consistent = {
            "distance": [(Rational(25),), ((3, 4),), (Rational(3), Rational(4))],
            "polygon_area": [(Rational(12),), (Rational(0), Rational(12), Rational(0)), ((Rational(12), 2),)],
            "line_intersection": [(Rational(-2), HALF), (Rational(-2),), ((HALF, HALF),)],
            "orientation": [(1,), (Rational(1),), ((1, 0), (0, 1))],
        }
        for operation, args, value in claims:
            evidences = [None, [], Coordinates(), ((),), (0,), ("shoelace",), (value,), {}, "", (None,), ((), ())]
            for index, evidence in enumerate(evidences + consistent[operation]):
                with self.subTest(operation=operation, evidence=index):
                    self.assertRejects(operation, args, value, evidence)

    def test_never_raises_on_junk(self):
        """Every slot of a claim filled with stand-ins of every kind: the checker answers False, never raising."""
        claims = [("distance", FIVE, Rational(5)), ("polygon_area", area_claim(TRIANGLE)[0], Rational(6)),
                  ("line_intersection", CROSSING, LineIntersection("point", (HALF, HALF))), ("orientation", CCW, 1)]
        # None of these completes a true claim of the tables above.
        junk: list[object] = [None, 7, 1.5, "x", b"x", (), [], {}, object(), True, frozenset(), MappingProxyType({}),
                              Fraction(1, 2), Rational(2), Sqrt(Rational(2)), Matrix([]), Coordinates((1,)),
                              LineIntersection("parallel", None), support.forged(Rational), support.forged(Sqrt),
                              support.forged(LineIntersection), support.forged(Matrix), Answer(1), ((0, 0),)]
        for operation, args, value in claims:
            for index, item in enumerate(junk):
                with self.subTest(operation=operation, item=index, kind=type(item).__name__):
                    self.assertRejects(item, args, value)
                    self.assertRejects(operation, item, value)
                    self.assertRejects(operation, args, item)
                    self.assertRejects(operation, args, value, item if item != () else (item,))
                    for name in args:
                        self.assertRejects(operation, dict(args, **{name: item}), value)

    # -----------------------------------------------------------------------------------------------------------
    # Oversized forgeries and the budget

    def test_oversized_forgeries_fail_fast(self):
        triangle = area_claim(TRIANGLE)[0]
        crossing = LineIntersection("point", (HALF, HALF))
        cases = [
            ("distance", FIVE, Rational(GIANT), ()), ("distance", FIVE, Rational(1, GIANT), ()),
            ("distance", FIVE, Sqrt(Rational(GIANT)), ()), ("distance", FIVE, Sqrt(Rational(1, GIANT)), ()),
            ("distance", FIVE, Rational(BIG, 3), ()), ("distance", FIVE, Sqrt(Rational(BIG, 7)), ()),
            ("distance", EIGHT, support.forged(Sqrt, radicand=forged_rational(BIG * 8, BIG)), ()),
            ("distance", FIVE, Rational(5), (0,) * 100_000), ("distance", FIVE, Rational(5), (GIANT,)),
            ("polygon_area", triangle, Rational(GIANT), ()), ("polygon_area", triangle, Rational(BIG, 3), ()),
            ("polygon_area", triangle, Rational(6), (Rational(GIANT),)),
            ("polygon_area", triangle, Rational(6), tuple(Rational(k) for k in range(100_000))),
            ("line_intersection", CROSSING, forged_intersection(kind="point", point=(HALF, Rational(GIANT))), ()),
            ("line_intersection", CROSSING, forged_intersection(kind="point", point=(Rational(1, GIANT), HALF)), ()),
            ("line_intersection", CROSSING, LineIntersection("point", (Rational(BIG, 3), Rational(BIG - 2, 5))), ()),
            ("line_intersection", CROSSING, forged_intersection(kind="point", point=(HALF,) * 100_000), ()),
            ("line_intersection", CROSSING, forged_intersection(kind="point" * 100_000, point=(HALF, HALF)), ()),
            ("line_intersection", CROSSING, crossing, ((crossing,) * 100_000)),
            ("orientation", CCW, GIANT, ()), ("orientation", CCW, -GIANT, ()), ("orientation", CCW, BIG, ()),
            ("orientation", CCW, 1, tuple(range(100_000))),
        ]
        for index, (operation, args, value, evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, evidence)  # Under the default budget, never raising.
                self.assertLess(time.perf_counter() - start, FAST)
        # A forged value longer than max_integer_bits is refused before any arithmetic on it.
        giant = Rational(GIANT, 3)  # Made outside a budget, which would refuse it.
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation("polygon_area", triangle, giant, ()), False)
        self.assertLess(budget.work, 1000)

    def test_oversized_requests_exceed_the_budget(self):
        """A request integer longer than max_integer_bits, or a polygon with more entries than max_matrix_entries,
        raises ResourceLimitError, as the request's inspection (operations_v6.run, the verifier) and produce() do:
        the claim stays unverified, never verified."""
        with computation(ComputationLimits(max_integer_bits=80_000), fresh=True):
            giant_polygon = Matrix([[0, 0], [GIANT, 0], [0, 3]])
        cases = [
            ("distance", {"p": (0, GIANT), "q": (0, 0)}, Rational(GIANT)),
            ("distance", {"p": (0, Rational(1, GIANT)), "q": (0, 0)}, Rational(1, GIANT)),
            ("polygon_area", {"points": giant_polygon}, Rational(3 * GIANT, 2)),
            ("line_intersection", dict(CROSSING, q2=(GIANT, 0)), LineIntersection("point", (HALF, HALF))),
            ("orientation", dict(CCW, r=(0, GIANT)), 1),
        ]
        for operation, args, value in cases:
            with self.subTest(operation=operation):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value)
                self.assertLess(time.perf_counter() - start, FAST)
        # 2,049 vertices are 4,098 entries, two more than the default max_matrix_entries.
        with computation(ComputationLimits(max_matrix_entries=4098), fresh=True):
            vertices = tuple((k, k * k % 101) for k in range(2049))
            args, value = area_claim(vertices)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "polygon_area", args, value)
        self.assertAccepts("polygon_area", args, value, limits=ComputationLimits(max_matrix_entries=4098))

    def test_work_is_charged_before_it_runs(self):
        vertices = tuple((k, k * k % 101) for k in range(2000))
        args, value = area_claim(vertices)
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):  # Every vertex is charged: a small budget runs out at once.
            support.verdict(family, "polygon_area", args, value, limits=ComputationLimits(max_work=1000))
        self.assertLess(time.perf_counter() - start, FAST)
        # The charges depend on the claim alone: a budget one unit short of a check refuses it, its exact size passes.
        rng = random.Random(20261117)
        claims = [
            ("distance", *distance_claim((Fraction(1, 3), 2 ** 3000, 5), (7, Fraction(-2, 9), 2 ** 2000))),
            ("polygon_area", *area_claim(((0, 0), (Fraction(2 ** 900, 3), 1), (5, Fraction(7, 2 ** 800)), (-1, 9)))),
            ("line_intersection", *line_claim((rng.getrandbits(4000), 1), (2, rng.getrandbits(4000)),
                                              (Fraction(1, 3), 5), (rng.getrandbits(3000), 7))),
            ("orientation", *orientation_claim((rng.getrandbits(5000), 0), (1, 1), (Fraction(1, 7), 2 ** 4000))),
        ]
        for operation, claim_args, claim_value in claims:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, claim_args, claim_value, ()), True)
                work = budget.work
                self.assertGreater(work, 20)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, claim_args, claim_value,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, claim_args, claim_value, limits=ComputationLimits(max_work=work))
        # Long products are charged by their size: a check on 30,000-bit coordinates costs far more than one on
        # small ones.
        small_args, small_value = distance_claim((0, 0), (3, 4))
        long_args, long_value = distance_claim((0, 0), (3 * 2 ** 30_000 + 1, 4 * 2 ** 30_000))
        works = []
        for claim_args, claim_value in ((small_args, small_value), (long_args, long_value)):
            with computation(fresh=True) as budget:
                self.assertIs(checker.check_operation("distance", claim_args, claim_value, ()), True)
            works.append(budget.work)
        self.assertGreater(works[1], 50 * works[0])

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_the_checker_forms_no_power(self):
        """No builtin pow and no **: nothing in the checker needs a power, so nothing has to be charged for one."""
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the values and the
        arguments survive the format, and a wrong value fails the mathematics check."""
        crossing = LineIntersection("point", (HALF, HALF))
        parallel, coincident = LineIntersection("parallel", None), LineIntersection("coincident", None)
        triangle = area_claim(TRIANGLE)[0]
        claims = [
            ("distance", EIGHT, Sqrt(Rational(8)), Mul((Rational(2), Sqrt(Rational(2))))),
            ("distance", FIVE, Rational(5), Rational(-5)),
            ("distance", HALVES, Sqrt(HALF), Sqrt(Rational(2))),
            ("polygon_area", triangle, Rational(6), Rational(-6)),
            ("polygon_area", {"points": Matrix([])}, Rational(0), Rational(1)),
            ("polygon_area", {"points": Matrix((), ncols=2)}, Rational(0), Rational(-1)),
            ("line_intersection", CROSSING, crossing, parallel),
            ("line_intersection", PARALLEL, parallel, coincident),
            ("line_intersection", SAME, coincident, parallel),
            ("orientation", CCW, 1, -1),
            ("orientation", COLLINEAR, 0, 1),
        ]
        for index, (operation, args, value, wrong) in enumerate(claims):
            with self.subTest(case=index, operation=operation):
                self.roundtrip(claimed(operation, args, value))
                report = support.statuses(mf.verify(claimed(operation, args, wrong)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
