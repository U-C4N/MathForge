"""Convex hulls of rational points in the plane (mathforge.geometry): convex_hull.

convex_hull(points) -> Matrix
    Domain: points is a Matrix (MCP kind "matrix": operations_v6.run already rejects anything but a Matrix as invalid
    input) of n >= 1 rows (x, y) of exact rationals and two columns; repeated points are allowed. A Matrix without
    rows, Matrix([]) with 0 columns included, and any column count other than 2 are invalid input.
    Value: the k x 2 Matrix of the vertices of the convex hull of the points, counterclockwise, starting at the lowest
    point (smallest y, then smallest x among those), with only strict turns: a point inside an edge is no vertex.
    Every entry is a canonical Rational (the Matrix constructor makes every entry one). One distinct point gives that
    point (k = 1); points that all lie on one line give the two extreme points, the lowest-then-leftmost first
    (k = 2); otherwise k >= 3. Known values: (0,0),(2,0),(1,1),(2,2),(0,2) -> (0,0),(2,0),(2,2),(0,2);
    (0,0),(1,1),(2,2) -> (0,0),(2,2); (1,1),(1,1) -> (1,1); (0,0),(1,0),(2,0),(1,1) -> (0,0),(2,0),(1,1).
    Evidence: (). The checker, mathforge.certify.geometry.hull, verifies that every vertex is an input point, that the
    vertices are distinct, that the first one is the lowest-then-leftmost input point, and for k >= 3 that every
    consecutive triple is a strict left turn, that the polygon turns around exactly once and that every input point
    lies on or left of every edge (for k = 2 on the segment, for k = 1 at the point): the card's checker steps.
    Resource limits: current_budget().matrix(rows, cols) is charged before the shape is validated, so that a 65 x 65
    matrix, or 2 columns of more than max_matrix_entries / 2 rows, is resource_limit_exceeded.

How produce() computes, by the card's producer, Andrew's monotone chain (docs/v09/cards/m7-extended.md). Nothing here
is a proof: the checker verifies the value without sorting and without the chain.

- Every entry is read as a pair (numerator, denominator) of a canonical Rational: exact int fields, a positive
  denominator, lowest terms. Anything else (a Rational made with object.__new__ outside that form, an int or another
  object in a Matrix made without its constructor) is invalid input, as its checker refuses it. Pairs in lowest terms
  make equal points equal keys, so a dictionary keeps each distinct point once, with the Rationals of its first row.
- Each distinct point is written in homogeneous integer coordinates (X, Y, W), W the least common multiple of the two
  denominators. Points are compared by cross-multiplication (directly when their W agree), and the turn of o -> a -> b
  is the sign of cross2(A, B) for A, B positive multiples of a - o and b - o: (X_a - X_o, Y_a - Y_o) when W_a = W_o,
  else (X_a W_o - X_o W_a, Y_a W_o - Y_o W_a).
- The distinct points are sorted by (x, y), and the chain builds the lower hull from left to right and the upper hull
  from right to left, popping the last vertex while it does not make a strict left turn; the two chains without their
  last points give the vertices counterclockwise from the leftmost-then-lowest point, without points inside edges.
  One distinct point gives itself, collinear points their two extremes. The list is then rotated to start at the
  lowest-then-leftmost vertex (an extreme point, so a vertex of the chain), and the value is built with the Matrix
  constructor from the request's own canonical Rationals.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call, one for each row and entry it reads, one per 4,096
bits for hashing a point, one per step of the chains, and every operation on integers before it runs, on the scale of
the cost model of trusted.integers that its checker uses as well (product_work for a product, one unit per 16,384 bits
of an addition or a comparison, a gcd as a long division and Lehmer's algorithm on the shorter operand). It refuses
an integer longer than max_integer_bits, and a points Matrix with more entries than max_matrix_entries before it
validates the Matrix's shape, with ResourceLimitError. The work is O(n log n) comparisons and O(n) turns: 2,048
points in convex position cost well under the default max_work (test_hull measures it).

The family defines no record class: its only value is a Matrix (mathforge.matrices), which the shared renderer
presents, so TYPES is empty and there is no RENDER. The text of the hull of (0,0),(2,0),(1,1) is
"[[0, 0], [2, 0], [1, 1]]" and its LaTeX "\\begin{bmatrix}0 & 0 \\\\ 2 & 0 \\\\ 1 & 1\\end{bmatrix}".
mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/geometry/test_hull.py is the definition of
done.
"""

from __future__ import annotations

from functools import cmp_to_key
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, computation
from ...matrices import Matrix
from ...model import Rational
from ...registry import Argument, OperationSpec

# The producer rules:
# - Imports at the top come only from modules that `import mathforge` already loads; kernels are imported inside
#   produce(). Never import mathforge.certify, mathforge.lean, mathforge.mcp, mathforge._generated or a generated
#   area module (mathforge.ntheory, ...).
# - produce() validates its inputs first, with tests like `type(x) is int` (a bool, a float or a str is never an
#   exact number), and raises InvalidInput. It builds records only from validated data. For a Matrix argument it
#   calls current_budget().matrix(rows, cols) before it validates the shape.
# - The value is never None. The evidence is a tuple of ints, Rationals, tuples, str and records: no list, dict or
#   float. Under the verifier, 23 + 2 * request + 2 * value + evidence nodes stay within max_evidence_items.
# - Every operation takes at least one argument. Only an Expression or a Matrix that Result.expression should hold
#   is named `expression` or `A`.
# - Operation names are new, and nothing is superseded (no SUPERSEDES): tools/new_family.py checked these names.
# - No speed-suite fixture and no hand-written lean module: the generic lean runner serves v6 operations.

# The scope text of the geometry area's v0.9 specs (areas/geometry/plane.py), which already names geometry.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

# The matrix is points, never A: an argument named A would be copied into Result.expression.
SPECS = (
    OperationSpec("convex_hull", "geometry", (Argument("points", "Matrix", None, "matrix"),), "Matrix",
                  "Find the vertices of the convex hull of rational points in the plane.", SCOPE, version=6),
)

TYPES: dict[str, type] = {}  # The value is the v0.7 record mathforge.matrices.Matrix.
CHECKER = "mathforge.certify.geometry.hull"


Pair = tuple[int, int]  # (numerator, denominator) of a canonical rational: lowest terms, denominator >= 1.
Point = tuple[int, int, int]  # (X, Y, W) of a point: x = X / W, y = Y / W, W >= 1 the lcm of the denominators.

_ARGUMENTS = ("points",)
_SHAPE = "convex_hull requires a Matrix of two columns and at least one row (x, y) per point"
_ENTRIES = "convex_hull requires a Matrix of Rationals in lowest terms"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(the hull's vertices as a Matrix, ()) for args = {"points": points}, by Andrew's monotone chain (module
    docstring). UnsupportedOperation for another operation, InvalidInput for arguments outside the domain,
    ResourceLimitError past the budget."""
    if type(operation) is not str or operation != "convex_hull":
        raise UnsupportedOperation("Not an operation of the hull family: convex_hull")
    # Exact str names: a str subclass that compares equal to "points" is no argument name, as the checker rules.
    if type(args) is not dict or any(type(name) is not str for name in args) or set(args) != set(_ARGUMENTS):
        raise InvalidInput("convex_hull takes exactly the argument points")
    with computation() as budget:
        budget.tick()  # The call itself; every valid input costs at least one unit more.
        return _convex_hull(_Exact(budget), args["points"]), ()


def _convex_hull(exact: _Exact, points: object) -> Matrix:
    """The vertices of the hull of the rows of points, counterclockwise from the lowest-then-leftmost one."""
    budget = exact.budget
    if type(points) is not Matrix:
        raise InvalidInput(_SHAPE)
    rows, ncols = getattr(points, "rows", None), getattr(points, "ncols", None)
    if type(rows) is not tuple or type(ncols) is not int or ncols < 0:  # Only a Matrix made without its constructor.
        raise InvalidInput(_SHAPE)
    budget.matrix(len(rows), ncols)  # Before the shape is validated: a 65 x 65 matrix exceeds the budget.
    if ncols != 2 or not rows:
        raise InvalidInput(_SHAPE)
    entries: dict[tuple[Pair, Pair], tuple[Rational, Rational]] = {}  # Distinct points, the first entries of each.
    for row in rows:
        budget.tick()
        if type(row) is not tuple or len(row) != 2:  # Only a Matrix made without its constructor.
            raise InvalidInput(_SHAPE)
        key = (_number(exact, row[0]), _number(exact, row[1]))
        budget.tick(1 + (_bits(key) >> 12))  # Hashing a point reads its integers.
        if key not in entries:
            entries[key] = row
    # Pairs in lowest terms: equal points are equal keys, so the dictionary holds each distinct point once.
    keys = list(entries)
    corners = [exact.homogeneous(x, y) for x, y in keys]
    order = sorted(range(len(keys)), key=cmp_to_key(lambda i, j: exact.lexicographic(corners[i], corners[j])))
    hull = _monotone_chain(exact, [corners[index] for index in order], order)
    # The chain starts at the leftmost-then-lowest vertex; the value starts at the lowest-then-leftmost one, which is
    # an extreme point and therefore a vertex of the chain.
    start = 0
    for position in range(1, len(hull)):
        budget.tick()
        if exact.lowest_first(corners[hull[position]], corners[hull[start]]) < 0:
            start = position
    vertices = hull[start:] + hull[:start]
    return Matrix(tuple(entries[keys[index]] for index in vertices), ncols=2)


def _monotone_chain(exact: _Exact, ordered: list[Point], labels: list[int]) -> list[int]:
    """Andrew's monotone chain on distinct points sorted by (x, y): the labels of the hull's vertices,
    counterclockwise from the first point, only strict turns. One point gives itself and collinear points their two
    extremes, the first and the last point."""
    count = len(ordered)
    if count == 1:
        return [labels[0]]
    lower: list[int] = []  # Positions in ordered.
    for position in range(count):
        exact.budget.tick()
        while len(lower) >= 2 and exact.turn(ordered[lower[-2]], ordered[lower[-1]], ordered[position]) <= 0:
            exact.budget.tick()
            lower.pop()
        lower.append(position)
    upper: list[int] = []
    for position in range(count - 1, -1, -1):
        exact.budget.tick()
        while len(upper) >= 2 and exact.turn(ordered[upper[-2]], ordered[upper[-1]], ordered[position]) <= 0:
            exact.budget.tick()
            upper.pop()
        upper.append(position)
    # Each chain ends where the other begins. For collinear points both chains are [first, last] and [last, first].
    return [labels[position] for position in lower[:-1] + upper[:-1]]


def _number(exact: _Exact, number: object) -> Pair:
    """(numerator, denominator) of a canonical Rational: exact int fields, a positive denominator, lowest terms.
    InvalidInput for anything else (an int never reaches here through the Matrix constructor; a Rational made with
    object.__new__ may lack a field or hold anything), ResourceLimitError for an integer longer than
    max_integer_bits."""
    budget = exact.budget
    budget.tick()
    if type(number) is Rational:
        numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
        if type(numerator) is int and type(denominator) is int and denominator >= 1:  # A bool is no exact int.
            budget.check_int(numerator)
            budget.check_int(denominator)
            if denominator == 1 or exact.gcd(numerator, denominator) == 1:
                return numerator, denominator
    raise InvalidInput(_ENTRIES)


def _bits(key: tuple[Pair, Pair]) -> int:
    """The total length of the four integers of a point."""
    (a, b), (c, d) = key
    return a.bit_length() + b.bit_length() + c.bit_length() + d.bit_length()


class _Exact:
    """Exact integer arithmetic for one call of produce(), each operation charged to the budget before it runs on the
    scale of the cost model of trusted.integers (product_work for a product, one unit per 16,384 bits of an addition
    or a comparison, a gcd as a long division and Lehmer's algorithm on the shorter operand)."""

    __slots__ = ("budget", "_product_work", "_division_work")

    def __init__(self, budget: Budget) -> None:
        from ...trusted.integers import division_work, product_work  # The cost model that the checkers use.

        self.budget = budget
        self._product_work = product_work
        self._division_work = division_work

    def times(self, a: int, b: int) -> int:
        """a * b."""
        left, right = a.bit_length(), b.bit_length()
        self.budget.tick(1 + self._product_work(left, right) + ((left + right) >> 12))
        return a * b

    def minus(self, a: int, b: int) -> int:
        """a - b."""
        self.budget.tick(1 + (max(a.bit_length(), b.bit_length()) >> 14))
        return a - b

    def sign(self, a: int, b: int) -> int:
        """The sign of a - b, by comparison."""
        self.budget.tick(1 + (max(a.bit_length(), b.bit_length()) >> 14))
        return (a > b) - (a < b)

    def quotient(self, a: int, b: int) -> int:
        """a // b for b >= 1, charged by the length of the quotient."""
        left, right = a.bit_length(), b.bit_length()
        self.budget.tick(1 + self._division_work(max(left - right, 0) + 64, right) + (left >> 12))
        return a // b

    def gcd(self, a: int, b: int) -> int:
        """math.gcd(a, b): a long division of the longer operand by the shorter, then Lehmer's algorithm on the
        shorter, quadratic in its length."""
        left, right = a.bit_length(), b.bit_length()
        shorter, longer = min(left, right), max(left, right)
        words = shorter >> 6
        self.budget.tick(1 + self._division_work(max(longer - shorter, 0) + 64, shorter) + (longer >> 12)
                         + (words * words >> 6))
        return gcd(a, b)

    def homogeneous(self, x: Pair, y: Pair) -> Point:
        """(X, Y, W) of the point (x, y): W the least common multiple of the two denominators, X = x * W, Y = y * W."""
        (a, b), (c, d) = x, y
        self.budget.tick()
        if b == d:
            return a, c, b
        if b == 1:
            return self.times(a, d), c, d
        if d == 1:
            return a, self.times(c, b), b
        divisor = self.gcd(b, d)
        b_part = b if divisor == 1 else self.quotient(b, divisor)
        d_part = d if divisor == 1 else self.quotient(d, divisor)
        return self.times(a, d_part), self.times(c, b_part), self.times(b, d_part)

    def _coordinate(self, p: Point, q: Point, axis: int) -> int:
        """The sign of p[axis] / p[2] - q[axis] / q[2], cross-multiplied unless the denominators agree."""
        if p[2] == q[2]:
            return self.sign(p[axis], q[axis])
        return self.sign(self.times(p[axis], q[2]), self.times(q[axis], p[2]))

    def lexicographic(self, p: Point, q: Point) -> int:
        """The order of two points by x, then by y: -1, 0 or 1."""
        return self._coordinate(p, q, 0) or self._coordinate(p, q, 1)

    def lowest_first(self, p: Point, q: Point) -> int:
        """The order of two points by y, then by x: -1, 0 or 1."""
        return self._coordinate(p, q, 1) or self._coordinate(p, q, 0)

    def _direction(self, origin: Point, p: Point) -> tuple[int, int]:
        """A positive multiple of p - origin as two integers: (X_p - X_o, Y_p - Y_o), which is W (p - origin), when the
        two denominators agree, else W_o W_p (p - origin)."""
        (xo, yo, wo), (xp, yp, wp) = origin, p
        if wo == wp:
            return self.minus(xp, xo), self.minus(yp, yo)
        return (self.minus(self.times(xp, wo), self.times(xo, wp)),
                self.minus(self.times(yp, wo), self.times(yo, wp)))

    def turn(self, o: Point, a: Point, b: Point) -> int:
        """The sign of cross2(a - o, b - o): 1 for a left turn o -> a -> b, -1 for a right turn, 0 on a line. Each
        direction is a positive multiple of the true one, which keeps the sign."""
        ax, ay = self._direction(o, a)
        bx, by = self._direction(o, b)
        return self.sign(self.times(ax, by), self.times(ay, bx))
