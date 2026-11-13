"""The hull family of mathforge.geometry: convex_hull.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register geometry hull

Every expected value comes from the card or from the reference of this module, never from the producer: the points
are scaled to integers by the least common multiple of their denominators (a positive scaling keeps the hull and the
order of its vertices), a point is a vertex exactly when the directions from it to the other points lie in an open
half-plane (one direction d has every other strictly left of it, or on its own ray), and the vertices are sorted by
angle around the lowest-then-leftmost one. That is neither the producer's monotone chain nor the checker's fan test.
The card defers no input of this family, so no request ends without a value (NO_CONCLUSION).
tests/v09/geometry/test_hull_checker.py tests the checker on its own.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
from functools import cmp_to_key
import inspect
import math
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.geometry import hull as family
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.matrices import Matrix
from mathforge.model import Rational

from .. import support

FAST = 2.0  # Seconds: the largest requests, and a request that its budget refuses, finish well within this.
OP = "convex_hull"
STAND_INS = (0.5, True, "1", None)  # No exact number and no Matrix: a float, a bool, a str and None.
H = Fraction(1, 2)

Number = int | Fraction | Rational
IntPoint = tuple[int, int]

# ---------------------------------------------------------------------------------------------------------------
# The reference: every expected value of the tests comes from it or from the card.


def frac(number: Number) -> Fraction:
    """An exact number of the tests as a Fraction."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def cross(u: IntPoint, v: IntPoint) -> int:
    return u[0] * v[1] - u[1] * v[0]


def extreme(p: IntPoint, points: Sequence[IntPoint]) -> bool:
    """Whether p is a vertex of the hull of the distinct points: the directions from p to the others lie in an open
    half-plane, that is, one of them, d, has every direction strictly left of it or on its own ray."""
    directions = [(q[0] - p[0], q[1] - p[1]) for q in points if q != p]
    if not directions:  # The only point.
        return True
    return any(all(cross(d, v) > 0 or (cross(d, v) == 0 and d[0] * v[0] + d[1] * v[1] > 0) for v in directions)
               for d in directions)


def reference_hull(points: Sequence[Sequence[Number]]) -> list[tuple[Fraction, Fraction]]:
    """The card's value: the extreme points, from the lowest-then-leftmost one, counterclockwise by angle."""
    exact = [(frac(x), frac(y)) for x, y in points]
    scale = math.lcm(*(c.denominator for point in exact for c in point))
    distinct = sorted({(int(x * scale), int(y * scale)) for x, y in exact})
    vertices = [p for p in distinct if extreme(p, distinct)]
    start = min(vertices, key=lambda v: (v[1], v[0]))
    rest = [v for v in vertices if v != start]
    # Every other vertex lies at an angle in [0, pi) from start, no two on one ray: a strictly convex polygon.
    rest.sort(key=cmp_to_key(lambda a, b: -cross((a[0] - start[0], a[1] - start[1]),
                                                (b[0] - start[0], b[1] - start[1]))))
    return [(Fraction(x, scale), Fraction(y, scale)) for x, y in [start] + rest]


def rational(number: Number) -> Rational:
    number = frac(number)
    return Rational(number.numerator, number.denominator)


def matrix(points: Sequence[Sequence[Number]]) -> Matrix:
    """Points as a request or a value holds them: a Matrix of two columns of Rationals."""
    return Matrix(tuple((rational(x), rational(y)) for x, y in points), ncols=2)


def expected(points: Sequence[Sequence[Number]]) -> Matrix:
    return matrix(reference_hull(points))


def forged_rational(numerator: object, denominator: object) -> Rational:
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def forged_matrix(**fields: object) -> Matrix:
    return support.forged(Matrix, **fields)


# ---------------------------------------------------------------------------------------------------------------
# Requests

SQUARE = [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)]
KNOWN = (  # The card's known values.
    (SQUARE, [(0, 0), (2, 0), (2, 2), (0, 2)]),
    ([(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 2)]),
    ([(1, 1), (1, 1)], [(1, 1)]),
    ([(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (2, 0), (1, 1)]),
)
NAMED = (  # The cases that the card's text names, with their values.
    ([(5, -3)], [(5, -3)]),  # One point.
    ([(H, H), (H, H), (H, H)], [(H, H)]),  # One distinct point, repeated.
    ([(3, 0), (1, 0), (2, 0), (1, 0)], [(1, 0), (3, 0)]),  # Horizontal: the leftmost first.
    ([(0, 4), (0, 1), (0, 3)], [(0, 1), (0, 4)]),  # Vertical: the lowest first.
    ([(2, 2), (0, 4), (1, 3), (4, 0)], [(4, 0), (0, 4)]),  # Descending diagonal: the lowest first.
    ([(3, 0), (0, 0), (0, 3), (3, 3), (1, 2), (2, 1), (3, 1)], [(0, 0), (3, 0), (3, 3), (0, 3)]),  # Lowest tie.
    ([(10, 0), (0, 10), (-10, 0), (0, -10), (1, 1), (5, 5)], [(0, -10), (10, 0), (0, 10), (-10, 0)]),  # Edge point.
    ([(Fraction(1, 3), 0), (Fraction(2, 5), Fraction(7, 11)), (0, H)],
     [(Fraction(1, 3), 0), (Fraction(2, 5), Fraction(7, 11)), (0, H)]),
)
SMALL = matrix(SQUARE)


class Fractional(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Sub(Matrix):
    """A Matrix subclass."""


class HullTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: Matrix) -> None:
        """A completed v6 Result of that value, a Matrix of Rationals, with evidence () and the four checks, which
        mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), Matrix)
        self.assertTrue(all(type(entry) is Rational for row in result.value.rows for entry in row))
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertAgrees(self, points: Sequence[Sequence[Number]], value: Matrix | None = None) -> None:
        """produce() answers the reference's value with the evidence (), and the checker accepts it."""
        args = {"points": matrix(points)}
        value = expected(points) if value is None else value
        found, evidence = family.produce(OP, dict(args))
        self.assertEqual(found, value)
        self.assertIs(type(found), Matrix)
        self.assertTrue(all(type(entry) is Rational for row in found.rows for entry in row))
        self.assertEqual(evidence, ())
        self.assertIs(support.verdict(family, OP, args, found, evidence), True)

    # -----------------------------------------------------------------------------------------------------------

    def test_reference(self):
        """The reference gives the card's known values and the named cases."""
        for points, value in KNOWN + NAMED:
            with self.subTest(points=points):
                self.assertEqual(reference_hull(points), [(Fraction(x), Fraction(y)) for x, y in value])

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for points, value in KNOWN + NAMED:
            with self.subTest(points=points):
                result = support.run(OP, points=matrix(points))
                self.assertVerified(result, matrix(value))
                self.roundtrip(result)
        # A Matrix of ints is the same request: the constructor makes every entry a Rational.
        self.assertVerified(support.run(OP, points=Matrix([[0, 0], [1, 0], [2, 0], [1, 1]])),
                            matrix([(0, 0), (2, 0), (1, 1)]))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error; so is
        every input that the card rejects: no rows, and a column count other than 2. produce() tests the types
        itself too."""
        rows = SMALL.rows
        for stand_in in STAND_INS + (rows, [list(row) for row in rows], (0, 0), Rational(1), 0):
            with self.subTest(points=stand_in):
                self.assertInvalidInput(support.run(OP, points=stand_in))
        invalid = (Matrix([]), Matrix((), ncols=2), Matrix((), ncols=3), Matrix([[], []]),
                   Matrix([[0, 0, 0], [4, 0, 0], [0, 3, 0]]), Matrix([[0], [4], [0]]), Matrix([[0, 4, 0], [0, 0, 3]]))
        for points in invalid:
            with self.subTest(points=points):
                result = support.run(OP, points=points)
                self.assertInvalidInput(result)
                self.assertIn("two columns and at least one row", result.error.message)
        # Entries that only object.__new__ makes reach produce() through run(), which checks the class alone, in
        # every row and column. A Rational whose fields are no ints, or that lacks a field, already fails the
        # request's inspection in run() (the plane family's tests leave those to produce() as well): they are
        # refused by produce() below.
        reaching = STAND_INS + (0, 2, forged_rational(2, 4), forged_rational(-1, -2), forged_rational(0, 5),
                                forged_rational(1, 0), forged_rational(True, 1), Fractional(1, 2))
        direct = reaching + (forged_rational(1.0, 2), forged_rational("1", 1), forged_rational(1, 2.0),
                             support.forged(Rational, numerator=1), support.forged(Rational, denominator=1))
        for index, entry in enumerate(direct):
            for row in range(len(rows)):
                for column in range(2):
                    changed = [list(item) for item in rows]
                    changed[row][column] = entry
                    points = forged_matrix(rows=tuple(tuple(item) for item in changed), ncols=2)
                    with self.subTest(entry=index, row=row, column=column):
                        if index < len(reaching):
                            self.assertInvalidInput(support.run(OP, points=points))
                        with self.assertRaises(InvalidInput):
                            family.produce(OP, {"points": points})
        # The public Matrix constructor keeps a Rational as it is given.
        for entry in (forged_rational(2, 4), forged_rational(0, 5), forged_rational(-1, -2)):
            with self.subTest(entry=entry, constructor=True):
                self.assertInvalidInput(support.run(OP, points=Matrix([[0, 0], [entry, 1]])))
        reaching_forgeries = [forged_matrix(rows=list(rows), ncols=2), forged_matrix(rows=rows, ncols=True),
                              forged_matrix(rows=rows, ncols=2.0), forged_matrix(rows=rows, ncols=-2),
                              forged_matrix(rows=rows, ncols=3), forged_matrix(rows=rows, ncols=0),
                              forged_matrix(rows=rows[:2] + (list(rows[2]),), ncols=2),
                              forged_matrix(rows=rows[:2] + (rows[2][:1],), ncols=2),
                              forged_matrix(rows=rows[:2] + (rows[2] + (Rational(0),),), ncols=2),
                              forged_matrix(rows=(), ncols=2), forged_matrix(rows=None, ncols=2)]
        # A Matrix without a field fails the request's inspection in run(); produce() refuses it as well.
        forgeries = reaching_forgeries + [forged_matrix(ncols=2), forged_matrix(rows=rows)]
        for index, forgery in enumerate(forgeries):
            with self.subTest(forgery=index):
                if index < len(reaching_forgeries):
                    self.assertInvalidInput(support.run(OP, points=forgery))
                with self.assertRaises(InvalidInput):
                    family.produce(OP, {"points": forgery})
        # produce() called directly: other containers, other argument sets, other operation names.
        for container in STAND_INS + (rows, [list(row) for row in rows], (0, 0), Rational(1), 0,
                                      support.impostor(SMALL), Matrix([]), Matrix([[1, 2, 3]])):
            with self.subTest(container=container, path="produce"), self.assertRaises(InvalidInput):
                family.produce(OP, {"points": container})
        sub = object.__new__(Sub)
        object.__setattr__(sub, "rows", rows)
        object.__setattr__(sub, "ncols", 2)
        for args in ({}, {"points": SMALL, "extra": 1}, {"point": SMALL}, (("points", SMALL),), None,
                     {"points": sub}):
            with self.subTest(args=args, path="produce"), self.assertRaises(InvalidInput):
                family.produce(OP, args)
        for operation in ("Convex_hull", "convex_hulls", "polygon_area", "", None, (OP,), b"convex_hull"):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, {"points": SMALL})

        class Name(str):
            pass

        with self.assertRaises(UnsupportedOperation):
            family.produce(Name(OP), {"points": SMALL})
        # Review regression: a str subclass equal to "points" is no argument name, which the checker refuses too.
        with self.assertRaises(InvalidInput):
            family.produce(OP, {Name("points"): SMALL})
        self.assertFalse(support.verdict(family, OP, {Name("points"): SMALL}, expected(SQUARE)))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OP, points=Matrix([]))
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in an entry, a 65 x 65 matrix, 2,049 x 2 and 4,097 x 1 matrices (charged before the
        shape is validated) and max_work=1 exceed the budget; a request that needs more work than its budget fails
        at once."""
        with computation(ComputationLimits(max_integer_bits=80_000), fresh=True):
            giants = [Matrix([[0, 0], [huge, 0], [0, 3]]) for huge in (support.HUGE, -support.HUGE,
                                                                      Rational(1, support.HUGE),
                                                                      Rational(support.HUGE, 3))]
        for index, points in enumerate(giants):
            with self.subTest(giant=index):
                self.assertResourceLimit(support.run(OP, points=points))
                with self.assertRaises(ResourceLimitError), computation(fresh=True):
                    family.produce(OP, {"points": points})
        with computation(ComputationLimits(max_matrix_entries=5000), fresh=True):
            tall = Matrix(tuple((k, k * k % 101) for k in range(2049)))  # 4,098 entries.
            column = Matrix(tuple((k,) for k in range(4097)))
        for points in (support.oversized_matrix(), tall, column):
            with self.subTest(shape=points.shape):
                self.assertResourceLimit(support.run(OP, points=points))
                with self.assertRaises(ResourceLimitError), computation(fresh=True):
                    family.produce(OP, {"points": points})
        self.assertResourceLimit(support.run(OP, points=SMALL, limits=ComputationLimits(max_work=1)))
        self.assertResourceLimit(support.run(OP, points=matrix([(1, 1)]), limits=ComputationLimits(max_work=1)))
        start = time.perf_counter()
        many = matrix([(k, k * k) for k in range(2048)])
        self.assertResourceLimit(support.run(OP, points=many, limits=ComputationLimits(max_work=1000)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The evidence is (), so a producer chooses nothing but its canonical value: the freedom of a true claim is
        in its request. The checker accepts the reference's claims on requests with the rows permuted and points
        repeated, and the producer, which made none of these claims, answers the same value to each."""
        rng = random.Random(20261201)
        checked = 0
        for case in range(40):
            points = [(Fraction(rng.randint(-6, 6), rng.choice((1, 1, 2, 3))), rng.randint(-6, 6))
                      for _ in range(rng.randint(1, 10))]
            value = expected(points)
            for variant in (points, points[::-1], rng.sample(points, len(points)), points + points[:3]):
                args = {"points": matrix(variant)}
                with self.subTest(case=case, points=variant):
                    self.assertAccepts(OP, args, value)
                    self.assertEqual(family.produce(OP, dict(args)), (value, ()))
                    checked += 1
        self.assertGreaterEqual(checked, 160)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where a Rational belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        for points in (SQUARE, [(0, 0), (1, 1), (2, 2)], [(1, 1), (1, 1)], [(0, 0), (1, 0), (2, 0), (1, 1)]):
            args = {"points": matrix(points)}
            result = support.run(OP, **args)
            value = result.value
            self.assertVerified(result, expected(points))
            self.assertAccepts(OP, args, value)
            hull = list(value.rows)
            wrong = [("the hull reversed", hull[::-1]), ("the hull rotated", hull[1:] + hull[:1]),
                     ("a repeated vertex", hull + hull[:1]), ("a far point", hull + [(Rational(9), Rational(9))]),
                     ("the first vertex alone", hull[:1]), ("all input points", list(args["points"].rows))]
            if len(hull) > 1:
                wrong.append(("a vertex left out", hull[:-1]))
            cases = [(label, Matrix(tuple(rows), ncols=2), ()) for label, rows in wrong
                     if rows and tuple(rows) != value.rows]
            first = list(hull[0])
            cases += [
                ("an int entry", forged_matrix(rows=((int(frac(first[0])), first[1]),) + value.rows[1:], ncols=2), ()),
                ("a bool entry", forged_matrix(rows=((True, first[1]),) + value.rows[1:], ncols=2), ()),
                ("a float entry", forged_matrix(rows=((0.0, first[1]),) + value.rows[1:], ncols=2), ()),
                ("a str entry", forged_matrix(rows=(("0", first[1]),) + value.rows[1:], ncols=2), ()),
                ("an entry in higher terms", forged_matrix(rows=((forged_rational(2 * first[0].numerator, 2),
                                                                  first[1]),) + value.rows[1:], ncols=2), ()),
                ("a list row", forged_matrix(rows=(first,) + value.rows[1:], ncols=2), ()),
                ("rows as a list", forged_matrix(rows=list(value.rows), ncols=2), ()),
                ("ncols 3", forged_matrix(rows=value.rows, ncols=3), ()),
                ("an impostor Matrix", support.impostor(value), ()),
                ("the rows alone", value.rows, ()), ("None", None, ()), ("Matrix([])", Matrix([]), ()),
                ("evidence as a list", value, []), ("evidence None", value, None),
                ("the value as evidence", value, (value,)), ("oversized evidence", value, (0,) * 100_000),
            ]
            sub = object.__new__(Sub)
            object.__setattr__(sub, "rows", value.rows)
            object.__setattr__(sub, "ncols", 2)
            cases.append(("a Matrix subclass", sub, ()))
            for label, item, evidence in cases:
                with self.subTest(points=points, case=label):
                    self.assertRejects(OP, args, item, evidence, may_exhaust=label == "oversized evidence")
            for operation in ("polygon_area", "orientation", "Convex_hull"):
                with self.subTest(points=points, renamed=operation):
                    self.assertRejects(operation, args, value)
            report = support.statuses(mf.verify(support.tampered(result, cases[0][1])))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [points for points, _ in KNOWN + NAMED]
        results = [support.run(OP, points=matrix(points)) for points in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for points, result, text in zip(requests, results, texts):
                with self.subTest(points=points):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(OP, points=SMALL)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value, a Matrix of one, two or more rows, renders as text
        and as LaTeX."""
        rendered = (
            ([(0, 0), (2, 0), (1, 1)], "[[0, 0], [2, 0], [1, 1]]",
             r"\begin{bmatrix}0 & 0 \\ 2 & 0 \\ 1 & 1\end{bmatrix}"),
            ([(1, 1), (1, 1)], "[[1, 1]]", r"\begin{bmatrix}1 & 1\end{bmatrix}"),
            ([(0, 0), (1, 1), (2, 2)], "[[0, 0], [2, 2]]", r"\begin{bmatrix}0 & 0 \\ 2 & 2\end{bmatrix}"),
            ([(H, 0), (0, Fraction(1, 3)), (-1, -1)], "[[-1, -1], [1/2, 0], [0, 1/3]]",
             r"\begin{bmatrix}-1 & -1 \\ \frac{1}{2} & 0 \\ 0 & \frac{1}{3}\end{bmatrix}"),
        )
        for points, text, latex in rendered:
            with self.subTest(points=points):
                result = support.run(OP, points=matrix(points))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        half = {"type": "rational", "numerator": "1", "denominator": "2"}
        zero = {"type": "rational", "numerator": "0", "denominator": "1"}
        self.assertEqual(mf.to_data(support.run(OP, points=matrix([(H, 0), (H, 0)])).value)["data"],
                         {"type": "matrix", "rows": [[half, zero]], "ncols": "2"})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261202)
        for case in range(40):
            denominators = (1,) if case % 2 else (1, 2, 3, 7)
            points = [(Fraction(rng.randint(-20, 20), rng.choice(denominators)),
                       Fraction(rng.randint(-20, 20), rng.choice(denominators))) for _ in range(rng.randint(1, 12))]
            args = {"points": matrix(points)}
            with self.subTest(points=points):
                value = support.run(OP, **args).value
                self.assertEqual(value, expected(points))
                found = support.lean_value(OP, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, value)
                self.assertEqual(support.value(OP, **args), value)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        result = self.answer(support.call(OP, {"points": [[0, 0], [2, 0], [1, 1], [2, 2], [0, 2]]}))
        self.assertEqual(result, support.run(OP, points=SMALL))
        self.assertVerified(result, matrix([(0, 0), (2, 0), (2, 2), (0, 2)]))
        # Rationals travel as "p/q" strings, integers from 2**53 on as decimal strings.
        large = 2 ** 70 + 1
        points = matrix([(H, large), (Fraction(-1, 3), 0), (large, Fraction(1, 7)), (0, 1), (Fraction(1, 9), 2)])
        result = self.answer(support.call_values(OP, points=points))
        self.assertEqual(result, support.run(OP, points=points))
        self.assertVerified(result, expected(points.rows))
        error = self.assertToolError(support.call(OP, {"points": []}), "invalid_input")
        self.assertIn("two columns", error["message"])
        self.assertToolError(support.call(OP, {"points": [[0, 0, 0], [1, 0, 0]]}), "invalid_input")
        self.assertToolError(support.call(OP, {"points": [[0.5, 0]]}), "invalid_input")
        self.assertToolError(support.call(OP, {"points": [[True, 0]]}), "invalid_input")
        self.assertToolError(support.call(OP, {"points": None}), "invalid_input")

    def test_seeded_agreement_with_reference(self):
        """produce() agrees with the reference on at least 200 seeded inputs, and the checker accepts every answer:
        small grids (many collinear and repeated points), rational points, collinear sets in every direction, points
        in convex position, long integers and long denominators. Every tenth request also runs and verifies."""
        rng = random.Random(20261203)
        sizes = {1: 0, 2: 0, 3: 0}  # Hulls of one point, two points, and three or more.
        compared = 0
        for case in range(300):
            kind = case % 6
            count = rng.randint(1, 14)
            if kind == 0:
                points = [(rng.randint(-4, 4), rng.randint(-4, 4)) for _ in range(count)]
            elif kind == 1:
                points = [(Fraction(rng.randint(-9, 9), rng.randint(1, 6)),
                           Fraction(rng.randint(-9, 9), rng.randint(1, 6))) for _ in range(count)]
            elif kind == 2:  # Collinear: a line through a rational point, or a vertical one.
                base = (Fraction(rng.randint(-5, 5), rng.randint(1, 3)),
                        Fraction(rng.randint(-5, 5), rng.randint(1, 3)))
                direction = rng.choice(((1, 0), (0, 1), (1, 1), (2, -3), (-1, 4), (Fraction(1, 3), Fraction(2, 5))))
                points = [(base[0] + t * direction[0], base[1] + t * direction[1])
                          for t in (Fraction(rng.randint(-6, 6), rng.randint(1, 2)) for _ in range(count))]
            elif kind == 3:  # Convex position on a parabola, shuffled, with interior points.
                xs = rng.sample(range(-30, 30), count)
                points = [(x, x * x) for x in xs] + [(rng.randint(-5, 5), 900) for _ in range(rng.randint(0, 3))]
                rng.shuffle(points)
            elif kind == 4:  # Long integers.
                points = [(rng.getrandbits(80) - 2 ** 79, rng.getrandbits(80) - 2 ** 79) for _ in range(count)]
            else:  # Long, mostly coprime denominators, with repetitions.
                points = [(Fraction(rng.getrandbits(40), rng.getrandbits(30) | 1),
                           Fraction(rng.getrandbits(40) - 2 ** 39, rng.getrandbits(30) | 1)) for _ in range(count)]
                points += rng.sample(points, min(2, len(points)))
            value = expected(points)
            sizes[min(len(value.rows), 3)] += 1
            with self.subTest(case=case, points=points):
                self.assertAgrees(points, value)
                compared += 1
                if case % 10 == 0:
                    self.assertVerified(support.run(OP, points=matrix(points)), value)
        self.assertGreaterEqual(compared, 200)
        self.assertTrue(all(count >= 20 for count in sizes.values()), sizes)

    def test_full_size_inputs_fit_the_default_budget(self):
        """The largest requests, 2,048 points (max_matrix_entries / 2 rows), complete and verify under the default
        limits within FAST seconds: shuffled points of a parabola, all vertices; a rational grid with a hull of four;
        and shuffled rational points of the unit circle over pairwise different denominators, all vertices."""
        rng = random.Random(20261204)
        parabola = [(i, i * i) for i in range(2048)]
        shuffled = rng.sample(parabola, len(parabola))
        grid = [(Fraction(i % 64, 3), Fraction(i // 64, 5)) for i in range(2048)]
        # The point of parameter t, |t| < 1, lies on the unit circle at the angle 2 atan(t), which grows with t from
        # the lowest point (t = -1024/1031): every point is a vertex, in the order of t. (The O(n^3) reference is too
        # slow for 2,048 points; it confirms the same order on a sample below.)
        ts = [Fraction(i, 1031) for i in range(-1024, 1024)]
        circle = [((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)) for t in ts]
        sample = circle[::64]
        self.assertEqual(reference_hull(rng.sample(sample, len(sample))), sample)
        requests = [
            (shuffled, matrix(parabola)),
            (grid, matrix([(0, 0), (21, 0), (21, Fraction(31, 5)), (0, Fraction(31, 5))])),
            (rng.sample(circle, len(circle)), matrix(circle)),
        ]
        for points, value in requests:
            with self.subTest(size=len(points), first=points[0]):
                start = time.perf_counter()
                with computation(fresh=True) as budget:  # The default limits, as for a request.
                    self.assertEqual(family.produce(OP, {"points": matrix(points)}), (value, ()))
                self.assertLess(budget.work, 500_000)  # Measured about 73,000, 159,000 and 175,000 units.
                result = support.run(OP, points=matrix(points))
                self.assertVerified(result, value)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_work_stays_near_the_producers(self):
        """Review regression: on long coordinates the checker's work stays below twice the producer's, so that a
        value the producer finds under the default limits is also verified under them. Before the review, the
        checker spent about nine times the producer's work on points in convex position with 10,000-bit coordinates
        (a 3 x 3 determinant per orientation and a fan search for every vertex), and those requests ended
        unverified."""
        shift, side = 2 ** 10_000, 2 ** 1000
        rng = random.Random(77)
        convex = [(i * shift + 1, (i * shift + 1) ** 2 >> 10_000) for i in range(1024)]  # y = i^2 s + 2 i exactly.
        square = [(0, 0), (side, 0), (side, side), (0, side)]
        inside = [(rng.randrange(1, side), rng.randrange(1, side)) for _ in range(1020)] + square
        rng.shuffle(inside)
        # |t| < 1: the angle 2 atan(t) grows with t from the lowest point, so the vertices are in the order of t.
        ts = sorted({Fraction(rng.getrandbits(200) - 2 ** 199, 2 ** 200 + rng.getrandbits(100)) for _ in range(256)})
        circle = [((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)) for t in ts]
        for points, value in ((rng.sample(convex, len(convex)), matrix(convex)), (inside, matrix(square)),
                              (rng.sample(circle, len(circle)), matrix(circle))):
            args = {"points": matrix(points)}
            with self.subTest(size=len(points), first=points[0]):
                with computation(fresh=True) as budget:
                    self.assertEqual(family.produce(OP, args), (value, ()))
                produced = budget.work
                with computation(fresh=True) as budget:
                    self.assertIs(support.checker(family).check_operation(OP, args, value, ()), True)
                self.assertLess(budget.work, 2 * produced)
                self.assertVerified(support.run(OP, points=args["points"]), value)

    def test_work_is_charged_before_it_runs(self):
        """The charges depend on the request alone: a budget one unit short of the producer's work refuses the
        request, its exact size completes it. Long integers are charged by their size."""
        requests = [SQUARE, [(Rational(1, 3), 2 ** 3000), (7, Rational(-2, 9)), (2 ** 2000, 5), (0, 0)],
                    [(k, k * k % 13) for k in range(40)]]
        for points in requests:
            args = {"points": matrix(points)}
            with self.subTest(points=points[:3]):
                with computation(fresh=True) as budget:
                    answer = family.produce(OP, args)
                work = budget.work
                self.assertGreater(work, 10)
                self.assertEqual(answer, (expected(points), ()))
                with self.assertRaises(ResourceLimitError), computation(ComputationLimits(max_work=work - 1),
                                                                        fresh=True):
                    family.produce(OP, args)
                with computation(ComputationLimits(max_work=work), fresh=True):
                    self.assertEqual(family.produce(OP, args), answer)
        works = []
        for scale in (1, 2 ** 30_000):
            with computation(fresh=True) as budget:
                family.produce(OP, {"points": matrix([(x * scale + 1, y * scale) for x, y in SQUARE])})
            works.append(budget.work)
        self.assertGreater(works[1], 10 * works[0])

    def test_module_imports_only_what_mathforge_loads(self):
        """The family module imports at its top only modules that `import mathforge` loads: the cost model of
        trusted.integers is imported when produce() runs, and nothing imports the checker, lean, mcp, kernels or
        generated code."""
        tree = ast.parse(inspect.getsource(family))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                top.add("." * node.level + (node.module or ""))
            elif isinstance(node, ast.Import):
                top.update(alias.name for alias in node.names)
        self.assertEqual(top, {"__future__", "functools", "math", "...errors", "...limits", "...matrices", "...model",
                               "...registry"})
        everywhere = {"." * node.level + (node.module or "") for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom)}
        self.assertIn("...trusted.integers", everywhere)
        self.assertFalse(any(name in module for module in everywhere
                             for name in ("certify", "lean", "mcp", "_generated", "kernels")))
