"""The independent checker of mathforge.areas.geometry.hull (mathforge.certify.geometry.hull), tested on its own
before the family's producer exists.

Every true claim is computed here, never by the producer: with fractions.Fraction, by brute force over the pairs of
distinct points (a directed pair is a hull edge when every point lies on or left of it and every point on its line
lies on the segment), checked against a second reference, Jarvis's gift wrapping, on seeded random sets. The evidence
is always (), the card's layout. The checker answers False to every corrupted claim, never with an exception, and
refuses an oversized forgery within FAST seconds under the default budget. A request integer longer than
max_integer_bits and a points Matrix over max_matrix_entries raise ResourceLimitError, as for the plane family: the
request's inspection refuses them before a checker runs, and the claim then stays unverified.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import itertools
import random
import time

from mathforge.areas.geometry import hull as family
from mathforge.certify.geometry import hull as checker
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.matrices import Matrix
from mathforge.model import Rational

from .. import support

FAST = 2.0  # Seconds: an oversized forgery, and the largest true claim, are judged well within this.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
OP = "convex_hull"
# The only imports a checker may have: the standard library modules it uses and the allowed mathforge layers.
ALLOWED = r"^from (__future__|math|mathforge[.](certify|trusted|errors|limits|model))[. ]"

Number = int | Fraction | Rational
Pt = tuple[Fraction, Fraction]


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def frac(number: Number) -> Fraction:
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def cross(o: Pt, a: Pt, b: Pt) -> Fraction:
    """cross2(a - o, b - o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def on_segment(p: Pt, a: Pt, b: Pt) -> bool:
    """Whether p, on the line through a and b, lies between them."""
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def lowest(points: Sequence[Pt]) -> Pt:
    return min(points, key=lambda p: (p[1], p[0]))


def brute_hull(points: Sequence[Sequence[Number]]) -> list[Pt]:
    """The card's value by brute force: the directed hull edges a -> b (every point on or left, collinear points on
    the segment), followed from the lowest-then-leftmost point."""
    distinct = sorted({(frac(x), frac(y)) for x, y in points})
    if len(distinct) == 1:
        return distinct
    start = lowest(distinct)
    if all(cross(distinct[0], distinct[1], p) == 0 for p in distinct):
        far = [q for q in distinct if all(on_segment(p, start, q) for p in distinct)]
        assert len(far) == 1
        return [start, far[0]]
    successor = {}
    for a, b in itertools.permutations(distinct, 2):
        if all(cross(a, b, p) > 0 or (cross(a, b, p) == 0 and on_segment(p, a, b)) for p in distinct):
            assert a not in successor
            successor[a] = b
    result = [start]
    while successor[result[-1]] != start:
        result.append(successor[result[-1]])
    assert len(result) == len(successor)
    return result


def gift_wrap(points: Sequence[Sequence[Number]]) -> list[Pt]:
    """The card's value by Jarvis's march: from the lowest-then-leftmost point, the next vertex is the point that
    leaves every other on its left, the farthest one among collinear candidates."""
    distinct = list({(frac(x), frac(y)) for x, y in points})
    start = lowest(distinct)
    result = [start]
    while True:
        here = result[-1]
        best = None
        for p in distinct:
            if p == here:
                continue
            if best is None:
                best = p
                continue
            turn = cross(here, best, p)
            far = (p[0] - here[0]) ** 2 + (p[1] - here[1]) ** 2 > (best[0] - here[0]) ** 2 + (best[1] - here[1]) ** 2
            if turn < 0 or (turn == 0 and far):
                best = p
        if best is None or best == start:
            return result
        if len(result) >= 2 and cross(result[-2], here, best) == 0:  # Collinear set: only the two extremes.
            return result
        result.append(best)


def rational(number: Number) -> Rational:
    number = frac(number)
    return Rational(number.numerator, number.denominator)


def matrix(points: Sequence[Sequence[Number]]) -> Matrix:
    return Matrix(tuple((rational(x), rational(y)) for x, y in points))


def claim(points: Sequence[Sequence[Number]]) -> tuple[dict[str, object], Matrix]:
    """(args, value) of a true claim, the value by brute force."""
    return {"points": matrix(points)}, matrix(brute_hull(points))


def forged_matrix(rows: object, ncols: object = 2) -> Matrix:
    return support.forged(Matrix, rows=rows, ncols=ncols)


def forged_rational(numerator: object, denominator: object) -> Rational:
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def edges(budget, vertices: Sequence[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[int, int]]:
    """The checker's edge directions of a closed polygon given as pairs, as its convex_hull step computes them."""
    corners = [checker._homogeneous(budget, vertex) for vertex in vertices]
    return [checker._direction(budget, corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]


SQUARE = [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)]
KNOWN = (
    (SQUARE, [(0, 0), (2, 0), (2, 2), (0, 2)]),
    ([(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 2)]),
    ([(1, 1), (1, 1)], [(1, 1)]),
    ([(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (2, 0), (1, 1)]),
)
H = Fraction(1, 2)
SPECIAL = (
    [(5, -3)],  # One point.
    [(H, H), (H, H), (H, H)],  # One distinct point, repeated, rational.
    [(3, 0), (1, 0), (2, 0), (1, 0)],  # Horizontal, unordered, repeated: the leftmost first.
    [(0, 4), (0, 1), (0, 3)],  # Vertical: the lowest first.
    [(2, 2), (0, 4), (1, 3), (4, 0)],  # Descending diagonal: the lowest point (4, 0) first.
    [(Fraction(1, 3), Fraction(1, 7)), (Fraction(2, 3), Fraction(2, 7)), (1, Fraction(3, 7))],  # Rational line.
    [(0, 0), (4, 0), (0, 4), (2, 0), (2, 2), (0, 2), (1, 1)],  # Triangle with edge and interior points.
    [(3, 0), (0, 0), (0, 3), (3, 3), (1, 2), (2, 1), (3, 1)],  # Lowest tie: (0, 0) before (3, 0).
    [(-5, -1), (-2, -7), (4, -6), (6, 2), (0, 5), (-3, 3), (0, 0), (1, -2)],  # Negative coordinates.
    [(Fraction(1, 3), 0), (Fraction(2, 5), Fraction(7, 11)), (0, Fraction(1, 2)), (Fraction(1, 7), Fraction(1, 9))],
    [(0, 0), (1, 1), (0, 0), (1, 1), (1, 0)],  # A triangle given with repetitions.
    [(10, 0), (0, 10), (-10, 0), (0, -10), (1, 1), (5, 5)],  # A diamond with a point on an edge.
)


class HullCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertFast(self, operation, args, value, evidence=(), *, may_exhaust: bool = False):
        started = time.perf_counter()
        self.assertRejects(operation, args, value, evidence, may_exhaust=may_exhaust)
        self.assertLess(time.perf_counter() - started, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # The references

    def test_references_agree(self):
        """Brute force and gift wrapping give the same hull on seeded random sets, and the known values."""
        for points, expected in KNOWN:
            self.assertEqual(brute_hull(points), [(Fraction(x), Fraction(y)) for x, y in expected])
        generator = random.Random(9090)
        for _ in range(300):
            count = generator.randint(1, 12)
            points = [(generator.randint(-4, 4), generator.randint(-4, 4)) for _ in range(count)]
            self.assertEqual(brute_hull(points), gift_wrap(points), points)

    # -----------------------------------------------------------------------------------------------------------
    # True claims

    def test_known_values(self):
        for points, expected in KNOWN:
            self.assertAccepts(OP, {"points": matrix(points)}, matrix(expected))

    def test_special_cases(self):
        """One point, repeated points, collinear sets in every direction, edge and interior points, a tie for the
        lowest point, negative and rational coordinates."""
        for points in SPECIAL:
            with self.subTest(points=points):
                self.assertAccepts(OP, *claim(points))

    def test_seeded_random_claims(self):
        generator = random.Random(2026)
        for trial in range(250):
            count = generator.randint(1, 14)
            if trial % 3 == 0:
                points = [(Fraction(generator.randint(-6, 6), generator.randint(1, 4)),
                           Fraction(generator.randint(-6, 6), generator.randint(1, 4))) for _ in range(count)]
            elif trial % 3 == 1:
                slope, offset = Fraction(generator.randint(-3, 3), generator.randint(1, 3)), generator.randint(-2, 2)
                xs = [Fraction(generator.randint(-9, 9), 2) for _ in range(count)]
                points = [(x, slope * x + offset) for x in xs]  # Collinear.
            else:
                points = [(generator.randint(-5, 5), generator.randint(-5, 5)) for _ in range(count)]
            with self.subTest(points=points):
                self.assertAccepts(OP, *claim(points))

    def test_largest_true_claims(self):
        """2,048 points (max_matrix_entries / 2 rows) in convex position, shuffled, and 2,048 points of a grid with
        a small hull, verify under the default budget within FAST seconds."""
        generator = random.Random(7)
        parabola = [(i, i * i) for i in range(2048)]
        hull = list(parabola)  # (0, 0) is the lowest point; the parabola turns left throughout.
        generator.shuffle(parabola)
        started = time.perf_counter()
        self.assertAccepts(OP, {"points": matrix(parabola)}, matrix(hull))
        self.assertLess(time.perf_counter() - started, FAST)
        grid = [(Fraction(i % 64, 3), Fraction(i // 64, 5)) for i in range(2048)]
        expected = [(0, 0), (Fraction(63, 3), 0), (Fraction(63, 3), Fraction(31, 5)), (0, Fraction(31, 5))]
        started = time.perf_counter()
        self.assertAccepts(OP, {"points": matrix(grid)}, matrix(expected))
        self.assertLess(time.perf_counter() - started, FAST)

    def test_rational_points_on_a_circle(self):
        """Rational points of the unit circle, ((1 - t^2) / (1 + t^2), 2t / (1 + t^2)), with pairwise different
        denominators: every one is a vertex."""
        ts = [Fraction(i, 37) for i in range(-150, 151, 3)]
        points = [((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)) for t in ts]
        args, value = claim(points)
        self.assertEqual(len(value.rows), len(points))
        self.assertAccepts(OP, args, value)

    def test_long_coordinates_fit_the_default_budget(self):
        """Review regression: 1,024 points in convex position with 10,000-bit coordinates, and 2,044 random points
        of 1,000 bits inside a square, verify under the default limits. On the first, the containment step alone
        cost about 3,200,000 units (max_work is 2,000,000) when every orientation was a 3 x 3 determinant and every
        vertex went through the fan search; with directions computed once and vertices skipped, the whole check
        costs about 250,000."""
        shift = 2 ** 10_000
        convex = [(i * shift + 1, (i * shift + 1) ** 2 >> 10_000) for i in range(1024)]
        # With s = 2^10000 and x_i = i s + 1, y_i = floor(x_i^2 / s) = i^2 s + 2 i exactly: equally spaced x and a
        # second difference 2 s > 0 in y, so every point is a vertex, counterclockwise from i = 0, the lowest.
        side = 2 ** 1000
        generator = random.Random(4242)
        square = [(0, 0), (side, 0), (side, side), (0, side)]
        inside = [(generator.randrange(1, side), generator.randrange(1, side)) for _ in range(2044)] + square
        generator.shuffle(inside)
        for points, value in ((convex, matrix(convex)), (inside, matrix(square))):
            with self.subTest(size=len(points)):
                with computation(None, fresh=True) as budget:
                    self.assertIs(checker.check_operation(OP, {"points": matrix(points)}, value, ()), True)
                self.assertLess(budget.work, 500_000)

    # -----------------------------------------------------------------------------------------------------------
    # Wrong and non-canonical values

    def test_wrong_values(self):
        """A missing vertex, a point outside the input, a point outside the true hull, the k = 1 and k = 2 answers
        for sets that are not one point or not collinear, and a hull of the wrong set."""
        args = {"points": matrix(SQUARE)}
        for wrong in ([(0, 0), (2, 0), (2, 2)],  # (0, 2) is missing: an input point outside the polygon.
                      [(0, 0), (2, 0), (3, 3), (0, 2)],  # (3, 3) is no input point.
                      [(0, 0), (2, 0), (1, 1)],  # Inputs outside the claimed triangle.
                      [(0, 0)], [(0, 0), (2, 2)], [(0, 0), (2, 0)],
                      [(1, 1), (2, 0), (2, 2), (0, 2)]):
            with self.subTest(wrong=wrong):
                self.assertRejects(OP, args, matrix(wrong))
        line = {"points": matrix([(0, 0), (1, 1), (2, 2)])}
        for wrong in ([(0, 0)], [(0, 0), (1, 1)], [(1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2)], [(2, 2), (0, 0)]):
            with self.subTest(wrong=wrong):
                self.assertRejects(OP, line, matrix(wrong))
        self.assertRejects(OP, {"points": matrix([(1, 1), (1, 1)])}, matrix([(1, 1), (1, 1)]))
        self.assertRejects(OP, {"points": matrix([(1, 1), (1, 2)])}, matrix([(1, 1)]))
        self.assertRejects(OP, {"points": matrix([(0, 0), (1, 0), (2, 1)])}, matrix([(0, 0), (2, 1)]))

    def test_non_canonical_values(self):
        """Correct sets of vertices in another form: another start, clockwise order, an edge point as a vertex, a
        repeated vertex, a collinear middle point, and entries that are no canonical Rationals."""
        args = {"points": matrix(SQUARE)}
        for wrong in ([(2, 0), (2, 2), (0, 2), (0, 0)], [(0, 2), (0, 0), (2, 0), (2, 2)],
                      [(0, 0), (0, 2), (2, 2), (2, 0)], [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
                      [(0, 0), (2, 0), (2, 0), (2, 2), (0, 2)]):
            with self.subTest(wrong=wrong):
                self.assertRejects(OP, args, matrix(wrong))
        edge = {"points": matrix([(0, 0), (1, 0), (2, 0), (1, 1)])}
        self.assertRejects(OP, edge, matrix([(0, 0), (1, 0), (2, 0), (1, 1)]))
        self.assertRejects(OP, edge, matrix([(0, 0), (2, 0), (1, 1), (0, 0)]))
        self.assertRejects(OP, {"points": matrix([(0, 0), (1, 1), (2, 2)])}, matrix([(0, 0), (1, 1), (2, 2)]))
        self.assertRejects(OP, {"points": matrix([(0, 0), (1, 1), (2, 2)])}, matrix([(2, 2), (0, 0)]))
        tie = {"points": matrix([(3, 0), (0, 0), (0, 3)])}
        self.assertAccepts(OP, tie, matrix([(0, 0), (3, 0), (0, 3)]))
        self.assertRejects(OP, tie, matrix([(3, 0), (0, 3), (0, 0)]))  # As low, but not the leftmost.
        true = [(Rational(0), Rational(0)), (Rational(2), Rational(0)), (Rational(2), Rational(2)),
                (Rational(0), Rational(2))]
        self.assertAccepts(OP, args, forged_matrix(tuple(true)))
        for index in range(4):
            for column in range(2):
                for bad in (int(frac(true[index][column])), True, False, 0.0, 2.0, "2", None, Fraction(2),
                            forged_rational(0, 2), forged_rational(4, 2), forged_rational(0, -1),
                            forged_rational(True, 1), forged_rational(2.0, 1), forged_rational(2, 1.0),
                            forged_rational("2", 1), support.forged(Rational, numerator=2)):
                    rows = [list(row) for row in true]
                    rows[index][column] = bad
                    with self.subTest(index=index, column=column, bad=bad):
                        self.assertRejects(OP, args, forged_matrix(tuple(tuple(row) for row in rows)))
        self.assertRejects(OP, args, forged_matrix(tuple(true), ncols=3))
        self.assertRejects(OP, args, forged_matrix(tuple(true), ncols=True))
        self.assertRejects(OP, args, forged_matrix(tuple(true), ncols=2.0))
        self.assertRejects(OP, args, forged_matrix(list(true)))
        self.assertRejects(OP, args, forged_matrix(tuple(list(row) for row in true)))
        self.assertRejects(OP, args, forged_matrix(tuple(row + (Rational(0),) for row in true)))
        self.assertRejects(OP, args, forged_matrix(tuple(row[:1] for row in true)))
        self.assertRejects(OP, args, forged_matrix(()))
        self.assertRejects(OP, args, Matrix([]))
        self.assertRejects(OP, args, support.forged(Matrix, rows=tuple(true)))
        self.assertRejects(OP, args, support.forged(Matrix, ncols=2))

    def test_other_classes(self):
        """An impostor Matrix of another module, a subclass, the rows alone, and other values."""
        args, value = claim(SQUARE)
        self.assertAccepts(OP, args, value)
        self.assertRejects(OP, args, support.impostor(value))

        class Sub(Matrix):
            pass

        sub = object.__new__(Sub)
        object.__setattr__(sub, "rows", value.rows)
        object.__setattr__(sub, "ncols", 2)
        self.assertRejects(OP, args, sub)
        for other in (value.rows, list(value.rows), None, 4, Rational(4), "matrix", value.rows[0]):
            self.assertRejects(OP, args, other)

    def test_winding_number(self):
        """A pentagram over five points in convex position turns left at every vertex but around twice: step (b)
        alone refuses it, and so does the whole check."""
        pentagon = [(0, -10), (10, -3), (6, 8), (-6, 8), (-10, -3)]
        star = [pentagon[i] for i in (0, 2, 4, 1, 3)]
        args = {"points": matrix(pentagon)}
        self.assertAccepts(OP, args, matrix(pentagon))
        self.assertRejects(OP, args, matrix(star))
        with computation(None, fresh=True) as budget:
            vertices = [((x, 1), (y, 1)) for x, y in star]
            self.assertTrue(all(cross(*(tuple(map(Fraction, star[(i + j) % 5])) for j in (-1, 0, 1))) > 0
                                for i in range(5)))
            self.assertFalse(checker._convex_polygon(budget, edges(budget, vertices)))
            square = [((x, 1), (y, 1)) for x, y in [(0, 0), (2, 0), (2, 2), (0, 2)]]
            self.assertTrue(checker._convex_polygon(budget, edges(budget, square)))

    def test_seeded_corruptions(self):
        """Every one-entry change, removal, insertion and rotation of true hulls answers False without raising."""
        generator = random.Random(31)
        for _ in range(60):
            points = [(generator.randint(-4, 4), generator.randint(-4, 4)) for _ in range(generator.randint(1, 9))]
            args, value = claim(points)
            hull = [tuple(row) for row in value.rows]
            candidates = []
            for index in range(len(hull)):
                for column in range(2):
                    rows = [list(row) for row in hull]
                    rows[index][column] = rows[index][column] + Rational(generator.choice((-1, 1)), 2)
                    candidates.append([tuple(row) for row in rows])
                candidates.append(hull[:index] + hull[index + 1:])
                candidates.append(hull[:index] + [hull[index]] + hull[index:])
            candidates.append(hull[1:] + hull[:1])
            candidates.append(hull[::-1])
            candidates.append(hull + [(Rational(100), Rational(100))])
            for rows in candidates:
                if rows == hull or not rows:
                    continue
                with self.subTest(points=points, rows=rows):
                    self.assertRejects(OP, args, Matrix(tuple(rows)))

    # -----------------------------------------------------------------------------------------------------------
    # Requests, evidence and operation names

    def test_request_shapes(self):
        """A request that holds no n x 2 Matrix of canonical Rationals, n >= 1, has no value."""
        args, value = claim(SQUARE)
        rows = [tuple(row) for row in args["points"].rows]
        for points in (rows, tuple(rows), Matrix([]), Matrix([(1, 2, 3)]), Matrix([(1,), (2,)]),
                       forged_matrix(tuple(rows), ncols=3), forged_matrix(list(rows)),
                       forged_matrix(tuple(list(row) for row in rows)), forged_matrix(()),
                       support.impostor(args["points"]), None, 5):
            with self.subTest(points=points):
                self.assertRejects(OP, {"points": points}, value)
        for bad in (0, True, 0.0, "0", None, Fraction(0), forged_rational(0, 2), forged_rational(2, 2),
                    forged_rational(True, 1), forged_rational(0, 0)):
            forged = [list(row) for row in rows]
            forged[0][0] = bad
            with self.subTest(bad=bad):
                self.assertRejects(OP, {"points": forged_matrix(tuple(tuple(row) for row in forged))}, value)
        for bad_args in ({}, {"points": args["points"], "extra": 1}, {"point": args["points"]},
                         {1: args["points"]}, [("points", args["points"])], None):
            with self.subTest(args=bad_args):
                self.assertRejects(OP, bad_args, value)

    def test_evidence_shapes(self):
        args, value = claim(SQUARE)
        for evidence in (None, [], (0,), ((),), (value,), (value.rows,), ("()",), [value], 0, ""):
            with self.subTest(evidence=evidence):
                self.assertRejects(OP, args, value, evidence)

    def test_operation_names(self):
        args, value = claim(SQUARE)
        for operation in ("convex_hulls", "Convex_hull", "orientation", "polygon_area", "", None, 1, b"convex_hull"):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value)

        class Name(str):
            pass

        self.assertRejects(Name(OP), args, value)

    # -----------------------------------------------------------------------------------------------------------
    # Size and budget

    def test_oversized_values_fail_fast(self):
        """A value with more rows than the request has points, or an integer longer than max_integer_bits, is
        refused within FAST seconds under the default budget."""
        args, value = claim(SQUARE)
        row = value.rows[0]
        self.assertFast(OP, args, forged_matrix((row,) * 1_000_000))
        self.assertFast(OP, args, forged_matrix(value.rows + (row,)))
        for giant in (forged_rational(GIANT, 1), forged_rational(1, GIANT), forged_rational(GIANT + 1, GIANT)):
            rows = [list(item) for item in value.rows]
            rows[3][1] = giant
            self.assertFast(OP, args, forged_matrix(tuple(tuple(item) for item in rows)))
        big = {"points": matrix([(i, i * i) for i in range(2048)])}
        self.assertFast(OP, big, forged_matrix(((Rational(0), Rational(0)),) * 2048))
        self.assertFast(OP, big, forged_matrix(tuple((Rational(i), forged_rational(GIANT, 1)) for i in range(2048))))

    def test_resource_limits(self):
        """Request integers longer than max_integer_bits, a request over max_matrix_entries and a tiny max_work
        raise ResourceLimitError (the claim is unverified), never True."""
        args, value = claim(SQUARE)
        rows = [list(row) for row in args["points"].rows]
        rows[1][0] = forged_rational(support.HUGE, 1)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, {"points": forged_matrix(tuple(tuple(row) for row in rows))}, value)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, {"points": support.oversized_matrix()}, value)
        tall = forged_matrix(tuple((Rational(i), Rational(0)) for i in range(2049)))
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, {"points": tall}, value)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, args, value, limits=ComputationLimits(max_work=1))
        self.assertAccepts(OP, args, value, limits=ComputationLimits(max_work=2_000))

    def test_checker_never_reads_the_producer(self):
        """The checker module names no producer module (its imports are only the allowed layers)."""
        with open(checker.__file__, encoding="utf-8") as handle:
            imports = [line for line in handle if line.startswith(("import ", "from "))]
        self.assertTrue(imports)
        for line in imports:
            self.assertRegex(line, ALLOWED, line)
