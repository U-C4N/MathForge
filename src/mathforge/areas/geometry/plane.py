"""Exact plane geometry of rational points (mathforge.geometry): distances, signed polygon areas, line intersections and
orientations.

Points are tuples of exact rationals (MCP kind "rational_list": operations_v6.run freezes a list into a tuple and
already rejects a bool, a float, a str or None in it as invalid input; 2 and Rational(2) are the same coordinate). A
polygon is a Matrix (kind "matrix") with one row (x, y) per vertex. For each operation, its domain, its value and the
evidence that its checker, mathforge.certify.geometry.plane, verifies; the evidence is () for all four:

- distance(p, q) -> Expression: the Euclidean distance between two points of the same dimension n >= 1 (different
  lengths and empty points are invalid input). The value is mathforge.model.sqrt(r) for r = sum_i (p_i - q_i)**2: a
  Rational when r is the square of a rational (Rational(5) for (0, 0) and (3, 4), Rational(0) for equal points), else
  the node Sqrt(r) with the canonical Rational r as its radicand and no factor taken out (Sqrt(Rational(8)) for (0, 0)
  and (2, 2), never 2*sqrt(2); Sqrt(Rational(1, 2)) for (0, 0) and (1/2, 1/2)).
- polygon_area(points) -> Rational: the signed shoelace area (1/2) * sum_i (x_i * y_{i+1} - x_{i+1} * y_i) of the
  vertices in order, cyclically: positive for counterclockwise vertices, negative for clockwise ones, and 0 for a
  self-intersecting bow-tie whose two loops cancel. The value is a Rational, also when it is integral (Rational(6),
  never 6). Fewer than 3 vertices give Rational(0), and a repeated closing vertex changes nothing. points has two
  columns; the matrix without rows holds no vertex, also Matrix([]), whose column count is 0 (an empty JSON array over
  MCP). Any other column count is invalid input.
- line_intersection(p1, p2, q1, q2) -> LineIntersection: the intersection of the line through p1 and p2 with the line
  through q1 and q2, four points of dimension 2 (any other dimension is invalid input, and so are p1 == p2 and
  q1 == q2, which determine no line). The value is LineIntersection("point", (x, y)) for lines that cross, with two
  Rationals (never ints), LineIntersection("parallel", None) for distinct parallel lines and
  LineIntersection("coincident", None) for the same line given twice.
- orientation(p, q, r) -> int: the sign of cross2(q - p, r - p) for three points of dimension 2 (any other dimension
  is invalid input): 1 when p, q, r turn counterclockwise, -1 when they turn clockwise, 0 when they lie on a line,
  also when two of them are equal.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, docs/v09/cards/m4-linalg-geometry-stats-discrete.md is the specification, and tests/v09/geometry/
test_plane.py is the definition of done.

How produce() computes, by the card's producer formulas. Nothing here is a proof: the checker verifies every value with
the trapezoid form of the signed area and with cross-multiplication, never with these formulas.

- Every coordinate is read as a pair (numerator, denominator) in lowest terms with a positive denominator: an int over
  1, or a Rational whose two fields are exact ints in that form. A Rational made with object.__new__ outside that
  form, or an int in a Matrix (whose constructor makes every entry a Rational), is invalid input, as its checker refuses
  it. A sum of fractions keeps its denominator at the least common multiple of its terms' denominators, unreduced, so
  that coordinates without a denominator, or with a shared one, stay cheap; one gcd reduces each value at the end,
  and the running sums of distance and polygon_area are also reduced whenever their denominator has doubled since
  the last reduction (_Exact.tidy), so that terms that cancel, as the telescoping shoelace terms of vertices on a
  line, keep the sum short.
- distance sums the squares of the coordinate differences, r = sum_i (p_i - q_i)**2, and returns model.sqrt(r) of the
  reduced r: a Rational when the numerator and the denominator of r are perfect squares (math.isqrt), else Sqrt(r). A
  rational distance needs only its root within max_integer_bits, r may be longer; the radicand of a Sqrt is part of
  the value and must fit.
- polygon_area writes each vertex in homogeneous integer coordinates (X, Y, W), W the least common multiple of the
  denominators of x and y, X = x * W and Y = y * W, and sums the shoelace terms
  x_i * y_{i+1} - x_{i+1} * y_i = (X_i * Y_{i+1} - X_{i+1} * Y_i) / (W_i * W_{i+1}) around the polygon; the area is half
  the sum. Fewer than three vertices give Rational(0) without a sum (their shoelace terms cancel), once every entry is
  validated.
- line_intersection and orientation multiply their points by the least common multiple L of all their coordinates'
  denominators, which changes neither a sign nor a ratio, and work on the integers P = L * p. line_intersection applies
  Cramer's rule to p1 + t * (p2 - p1) = q1 + s * (q2 - q1): with U = P2 - P1, V = Q2 - Q1, W = Q1 - P1 and
  D = cross2(U, V) = L**2 * cross2(p2 - p1, q2 - q1), the lines are parallel for D = 0, and coincident when moreover
  cross2(U, W) = 0 (q1 lies on the first line); otherwise t = T / D with T = cross2(W, V), and the point
  p1 + t * (p2 - p1) is (P1 * D + T * U) / (L * D), each coordinate reduced to lowest terms. orientation is the sign
  of cross2(Q - P, R - P).

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call, one for each coordinate that it reads and one for
each term of a sum, so that every valid input costs at least two units; it refuses an integer longer than
max_integer_bits, and a polygon Matrix with more entries than max_matrix_entries before it validates the Matrix's
shape, with ResourceLimitError; and it charges every operation on integers before it runs, on the scale of the cost
model of trusted.integers that its checker uses as well: a product one unit, product_work of the lengths of its
factors and a unit per 4,096 bits of them; an addition one unit and one per 16,384 bits; a long division one unit,
division_work by the length of its quotient and a unit per 4,096 bits of the dividend; a gcd a long division of the
longer operand by the shorter and Lehmer's algorithm on the shorter, one unit per 64 squared 64-bit words (the scale of
trusted.upoly.charge_gcd); an integer square root as much as a product of its operand with itself. A value with an
integer longer than max_integer_bits raises ResourceLimitError where it is built. Measured with CPython on one
machine, a unit of this producer takes at most about 0.3 microseconds, and the largest requests of the tests fit the
default max_work of 2,000,000 with room to spare, at about the work that the checker spends on them: a distance of
32,000-bit coordinates about 11,000 units, one of two 10,000-dimensional points of three-digit coordinates about
70,000, a polygon of 2,048 vertices (4,096 entries, max_matrix_entries) about 18,000 with integer coordinates and
38,000 over the denominators 7 and 343, a line intersection of 16,000-bit corners about 36,000 and an orientation of
65,000-bit coordinates about 8,000.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, computation
from ...matrices import Matrix
from ...model import Expression, Rational, Sqrt, sqrt, trusted_rational
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

# The scope text of every spec of wave M4 (docs/v09/cards/m4-linalg-geometry-stats-discrete.md).
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"


def _point(name: str) -> Argument:
    """A point argument: unannotated, as v0.7's rational_list arguments are (mean(data), covariance(x, y)), since a
    list is accepted as well as a tuple."""
    return Argument(name, "", None, "rational_list")


SPECS = (
    OperationSpec("distance", "geometry", (_point("p"), _point("q")), "Expression",
                  "Compute the exact Euclidean distance between two rational points.", SCOPE, version=6),
    OperationSpec("polygon_area", "geometry", (Argument("points", "Matrix", None, "matrix"),), "Rational",
                  "Compute the signed shoelace area of a polygon given by its vertices in order.", SCOPE, version=6),
    OperationSpec("line_intersection", "geometry", (_point("p1"), _point("p2"), _point("q1"), _point("q2")),
                  "LineIntersection",
                  "Intersect the line through p1, p2 with the line through q1, q2 in the plane.", SCOPE, version=6),
    OperationSpec("orientation", "geometry", (_point("p"), _point("q"), _point("r")), "int",
                  "Decide the orientation of three rational points in the plane: 1 counterclockwise, -1 clockwise, "
                  "0 collinear.", SCOPE, version=6),
)

KINDS = ("point", "parallel", "coincident")  # The kinds of a LineIntersection.


@dataclass(frozen=True)
class LineIntersection:
    """Where two lines of the plane meet: kind "point" with point = (x, y), the one common point, or kind "parallel"
    (no common point) or "coincident" (the same line) with point None.

    The canonical form, which each answer has exactly once: kind is one of the three exact strs, and point is a tuple
    of two Rationals in lowest terms (never ints) for "point" and None otherwise. The record holds data only; its
    checker tests the same form again."""
    kind: str
    point: tuple[Rational, Rational] | None

    def __post_init__(self):
        with computation() as budget:
            budget.tick()
            if type(self.kind) is not str or self.kind not in KINDS:  # A str subclass is no exact str.
                raise InvalidInput('A line intersection has the kind "point", "parallel" or "coincident"')
            if self.kind != "point":
                if self.point is not None:
                    raise InvalidInput("Only a line intersection of the kind point has a point")
                return
            if type(self.point) is not tuple or len(self.point) != 2:
                raise InvalidInput("The point of a line intersection is a tuple (x, y) of two Rationals")
            for coordinate in self.point:
                budget.tick()
                if type(coordinate) is not Rational:  # An int, a bool or a float is no canonical coordinate.
                    raise InvalidInput("The coordinates of a line intersection are Rationals, never ints")
                # A Rational made with object.__new__ may lack a field or hold anything: only its constructor's form.
                numerator = getattr(coordinate, "numerator", None)
                denominator = getattr(coordinate, "denominator", None)
                if type(numerator) is not int or type(denominator) is not int or denominator < 1:
                    raise InvalidInput("The coordinates of a line intersection are Rationals of exact integers")
                budget.check_int(numerator)
                budget.check_int(denominator)
                words = max(numerator.bit_length(), denominator.bit_length()) >> 6
                budget.tick(words * words >> 6)  # Before the gcd, quadratic in the length (Lehmer).
                if gcd(numerator, denominator) != 1:
                    raise InvalidInput("The coordinates of a line intersection are Rationals in lowest terms")


TYPES = {"line_intersection": LineIntersection}
CHECKER = "mathforge.certify.geometry.plane"

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1, not necessarily reduced.
_TIDY = 4_096  # Bits of a running sum's denominator below which _Exact.tidy never reduces it.

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
_NUMBERS = "{} requires points of exact rationals: tuples of ints and Rationals in lowest terms"
_PLANE = "{} requires points of dimension 2"
_DIMENSIONS = "distance requires two points of the same dimension, at least 1"
_LINES = "line_intersection requires p1 != p2 and q1 != q2: two equal points determine no line"
_POLYGON = ("polygon_area requires a Matrix of two columns, one row (x, y) per vertex, or a Matrix without rows "
            "and columns")
_ENTRIES = "polygon_area requires a Matrix of Rationals in lowest terms"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name and
    kind; the module states the algorithms, and the evidence is () for every operation. InvalidInput for an argument
    outside the operation's domain, ResourceLimitError past the budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation("Not an operation of the plane family: distance, polygon_area, line_intersection "
                                   "or orientation")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    values = [args[name] for name in names]
    with computation() as budget:
        budget.tick()  # The call itself; every valid input costs at least one unit more.
        return _PRODUCERS[operation](_Exact(budget), *values), ()


def _distance(exact: _Exact, p: object, q: object) -> Expression:
    """model.sqrt(r) for r = sum_i (p_i - q_i)**2: a Rational, or Sqrt(r) when r is no rational square."""
    if type(p) is not tuple or type(q) is not tuple:
        raise InvalidInput(_NUMBERS.format("distance"))
    if not p or len(p) != len(q):  # Tested before any coordinate is read.
        raise InvalidInput(_DIMENSIONS)
    message = _NUMBERS.format("distance")
    left = [_number(exact, coordinate, message) for coordinate in p]
    right = [_number(exact, coordinate, message) for coordinate in q]
    total: Pair = (0, 1)
    mark = _TIDY
    for x, y in zip(left, right):
        exact.budget.tick()
        numerator, denominator = exact.difference(x, y)
        if numerator:
            squared_denominator = 1 if denominator == 1 else exact.times(denominator, denominator)
            total = exact.add(total, (exact.times(numerator, numerator), squared_denominator))
            total, mark = exact.tidy(total, mark)
    numerator, denominator = exact.lowest(*total)
    exact.charge_square_roots(numerator, denominator)
    # r in lowest terms, possibly longer than max_integer_bits: a rational distance only needs its root to fit, and
    # model.sqrt builds that root with the Rational constructor, which refuses an integer longer than the budget's.
    value = sqrt(trusted_rational(numerator, denominator))
    if type(value) is Sqrt:
        exact.budget.charge_rational(numerator, denominator)  # The radicand is part of the value: it must fit.
    return value


def _polygon_area(exact: _Exact, points: object) -> Rational:
    """The signed shoelace area of the rows of points, in homogeneous integer coordinates (module docstring)."""
    budget = exact.budget
    if type(points) is not Matrix:
        raise InvalidInput(_POLYGON)
    rows, ncols = getattr(points, "rows", None), getattr(points, "ncols", None)
    if type(rows) is not tuple or type(ncols) is not int or ncols < 0:  # Only a Matrix made without its constructor.
        raise InvalidInput(_POLYGON)
    budget.matrix(len(rows), ncols)  # Before the shape is validated: a 65 x 65 matrix exceeds the budget.
    if not (ncols == 2 or (ncols == 0 and not rows)):
        raise InvalidInput(_POLYGON)
    vertices = []
    for row in rows:
        budget.tick()
        if type(row) is not tuple or len(row) != 2:  # Only a Matrix made without its constructor.
            raise InvalidInput(_POLYGON)
        vertices.append((_number(exact, row[0], _ENTRIES, integers=False),
                         _number(exact, row[1], _ENTRIES, integers=False)))
    count = len(vertices)
    if count < 3:  # One or two vertices: each shoelace term cancels with its reverse; none: no term.
        return exact.rational(0, 1)
    corners = [exact.homogeneous(x, y) for x, y in vertices]
    total: Pair = (0, 1)
    mark = _TIDY
    for index in range(count):
        budget.tick()
        (x0, y0, w0), (x1, y1, w1) = corners[index], corners[(index + 1) % count]
        term = exact.minus(exact.times(x0, y1), exact.times(x1, y0))
        if term:
            scale = w0 if w1 == 1 else w1 if w0 == 1 else exact.times(w0, w1)
            total = exact.add(total, (term, scale))
            total, mark = exact.tidy(total, mark)
    numerator, denominator = total
    return exact.rational(numerator, exact.times(2, denominator))


def _line_intersection(exact: _Exact, p1: object, p2: object, q1: object, q2: object) -> LineIntersection:
    """The intersection by Cramer's rule on the integer points L * p (module docstring)."""
    for point in (p1, p2, q1, q2):
        if type(point) is not tuple or len(point) != 2:
            raise InvalidInput(_PLANE.format("line_intersection"))
    message = _NUMBERS.format("line_intersection")
    first, second, third, fourth = ([_number(exact, coordinate, message) for coordinate in point]
                                    for point in (p1, p2, q1, q2))
    exact.budget.tick()
    if first == second or third == fourth:  # Pairs in lowest terms: equal numbers are equal pairs.
        raise InvalidInput(_LINES)
    scale, (x1, y1, x2, y2, x3, y3, x4, y4) = exact.common(first + second + third + fourth)
    ux, uy = exact.minus(x2, x1), exact.minus(y2, y1)  # U = P2 - P1
    vx, vy = exact.minus(x4, x3), exact.minus(y4, y3)  # V = Q2 - Q1
    wx, wy = exact.minus(x3, x1), exact.minus(y3, y1)  # W = Q1 - P1
    determinant = exact.cross(ux, uy, vx, vy)
    if not determinant:  # Parallel directions: the same line exactly when q1 lies on the first one.
        return LineIntersection("parallel" if exact.cross(ux, uy, wx, wy) else "coincident", None)
    parameter = exact.cross(wx, wy, vx, vy)  # t = parameter / determinant.
    denominator = determinant if scale == 1 else exact.times(scale, determinant)
    x = exact.rational(exact.plus(exact.times(x1, determinant), exact.times(parameter, ux)), denominator)
    y = exact.rational(exact.plus(exact.times(y1, determinant), exact.times(parameter, uy)), denominator)
    return LineIntersection("point", (x, y))


def _orientation(exact: _Exact, p: object, q: object, r: object) -> int:
    """The sign of cross2(q - p, r - p), on the integer points L * p."""
    for point in (p, q, r):
        if type(point) is not tuple or len(point) != 2:
            raise InvalidInput(_PLANE.format("orientation"))
    message = _NUMBERS.format("orientation")
    first, second, third = ([_number(exact, coordinate, message) for coordinate in point] for point in (p, q, r))
    _, (px, py, qx, qy, rx, ry) = exact.common(first + second + third)
    cross = exact.cross(exact.minus(qx, px), exact.minus(qy, py), exact.minus(rx, px), exact.minus(ry, py))
    return (cross > 0) - (cross < 0)


_PRODUCERS: dict[str, Callable[..., object]] = {
    "distance": _distance,
    "polygon_area": _polygon_area,
    "line_intersection": _line_intersection,
    "orientation": _orientation,
}


def _number(exact: _Exact, number: object, message: str, *, integers: bool = True) -> Pair:
    """(numerator, denominator) of an exact rational: an int over 1 when integers allows it, or a Rational whose fields
    are exact ints in lowest terms with a positive denominator. InvalidInput(message) for anything else (a Rational made
    with object.__new__ may lack a field or hold anything), ResourceLimitError for an integer longer than
    max_integer_bits."""
    budget = exact.budget
    budget.tick()
    if integers and type(number) is int:  # A bool is no exact int.
        budget.check_int(number)
        return number, 1
    if type(number) is Rational:
        numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
        if type(numerator) is int and type(denominator) is int and denominator >= 1:
            budget.check_int(numerator)
            budget.check_int(denominator)
            if denominator == 1 or exact.gcd(numerator, denominator) == 1:
                return numerator, denominator
    raise InvalidInput(message)


class _Exact:
    """Exact arithmetic for one call of produce(), on integers and on fractions (numerator, denominator) with a positive
    denominator that need not be in lowest terms. Every operation is charged to the budget before it runs, on the scale
    of trusted.integers (the module's Work paragraph states the charges)."""

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

    def plus(self, a: int, b: int) -> int:
        """a + b."""
        self.budget.tick(1 + (max(a.bit_length(), b.bit_length()) >> 14))
        return a + b

    def minus(self, a: int, b: int) -> int:
        """a - b."""
        self.budget.tick(1 + (max(a.bit_length(), b.bit_length()) >> 14))
        return a - b

    def quotient(self, a: int, b: int) -> int:
        """a // b for b >= 1, charged by the length of the quotient (division_work by the dividend's whole length would
        charge an exact quotient as long as the dividend)."""
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

    def cross(self, ax: int, ay: int, bx: int, by: int) -> int:
        """cross2(a, b) = ax * by - ay * bx."""
        return self.minus(self.times(ax, by), self.times(ay, bx))

    def add(self, x: Pair, y: Pair) -> Pair:
        """x + y, over the least common multiple of the two denominators."""
        (a, b), (c, d) = x, y
        if b == d:
            return self.plus(a, c), b
        if b == 1:
            return self.plus(self.times(a, d), c), d
        if d == 1:
            return self.plus(a, self.times(c, b)), b
        divisor = self.gcd(b, d)
        b_part = b if divisor == 1 else self.quotient(b, divisor)
        d_part = d if divisor == 1 else self.quotient(d, divisor)
        return self.plus(self.times(a, d_part), self.times(c, b_part)), self.times(b, d_part)

    def difference(self, x: Pair, y: Pair) -> Pair:
        """x - y, over the least common multiple of the two denominators."""
        self.budget.tick(1 + (y[0].bit_length() >> 14))  # The negation copies the numerator.
        return self.add(x, (-y[0], y[1]))

    def common(self, pairs: list[Pair]) -> tuple[int, list[int]]:
        """(L, [n * (L / d) for each pair (n, d)]): L, the least common multiple of the denominators, and the numbers
        times L, all integers."""
        scale = 1
        for _, denominator in pairs:
            self.budget.tick()
            if denominator == 1 or denominator == scale:
                continue
            divisor = self.gcd(scale, denominator)
            if divisor != denominator:  # L becomes L * (d / gcd(L, d)) unless d divides it.
                scale = self.times(scale, self.quotient(denominator, divisor))
        if scale == 1:
            return 1, [numerator for numerator, _ in pairs]
        return scale, [numerator if denominator == scale
                       else self.times(numerator, self.quotient(scale, denominator))
                       for numerator, denominator in pairs]

    def homogeneous(self, x: Pair, y: Pair) -> tuple[int, int, int]:
        """(X, Y, W) of the point (x, y): W the least common multiple of the two denominators, X = x * W and Y = y * W,
        all integers."""
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

    def lowest(self, numerator: int, denominator: int) -> Pair:
        """numerator / denominator in lowest terms with a positive denominator, for a nonzero denominator; 0 is 0/1."""
        if denominator < 0:
            self.budget.tick(1 + (max(numerator.bit_length(), denominator.bit_length()) >> 14))
            numerator, denominator = -numerator, -denominator
        divisor = self.gcd(numerator, denominator)
        if divisor == 1:
            return numerator, denominator
        return self.quotient(numerator, divisor), self.quotient(denominator, divisor)

    def tidy(self, total: Pair, mark: int) -> tuple[Pair, int]:
        """(total, mark) for the next term of a running sum: the sum in lowest terms once its denominator is longer than
        mark bits, and then mark twice the reduced denominator's length, at least _TIDY. A sum over the least common
        multiple of its terms' denominators grows with every term even where the terms cancel: the shoelace terms of
        vertices on a line y = m * x + c with c != 0 telescope, c * (x_i - x_{i+1}), and without this reduction 200 such
        vertices over pairwise coprime 2,000-bit denominators exhausted max_work, where the checker needs about
        400,000 units. Where nothing cancels, the reductions at each doubling cost about 4/3 of one gcd at the final
        length."""
        numerator, denominator = total
        if denominator.bit_length() <= mark:
            return total, mark
        total = self.lowest(numerator, denominator)
        return total, max(2 * total[1].bit_length(), _TIDY)

    def rational(self, numerator: int, denominator: int) -> Rational:
        """numerator / denominator as a canonical Rational, for a nonzero denominator: one unit for the Rational, and
        ResourceLimitError when its numerator or its denominator is longer than max_integer_bits."""
        numerator, denominator = self.lowest(numerator, denominator)
        self.budget.charge_rational(numerator, denominator)
        return trusted_rational(numerator, denominator)

    def charge_square_roots(self, numerator: int, denominator: int) -> None:
        """Charge model.sqrt of numerator / denominator before it runs: for each of the two integers math.isqrt, as much
        as a product of the integer with itself, and the square of its root compared with it; then the gcd of the two
        roots that the Rational of a rational square reduces."""
        total = 1
        for number in (numerator, denominator):
            bits = number.bit_length()
            half = (bits + 1) >> 1
            total += 2 + self._product_work(bits, bits) + self._product_work(half, half) + (bits >> 11)
        words = ((max(numerator.bit_length(), denominator.bit_length()) + 1) >> 1) >> 6
        self.budget.tick(total + (words * words >> 6))


def _render(intersection, renderer, child):
    r"""(1/2, 1/2) for the kind point, the shared rendering of the tuple (LaTeX \left(\frac{1}{2}, \frac{1}{2}\right));
    otherwise "parallel lines" or "coincident lines" (LaTeX \text{parallel lines})."""
    if intersection.kind == "point":
        return child(intersection.point)
    return renderer.label(intersection.kind + " lines")


RENDER = {LineIntersection: _render}
