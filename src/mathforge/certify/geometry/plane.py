"""Independent checker of the family mathforge.areas.geometry.plane: distance, polygon_area, line_intersection and
orientation.

The checker never reruns the producer's formulas (model.sqrt of a sum of squares, the shoelace sum, Cramer's rule, the
cross product of q - p and r - p). Its one geometric primitive is the trapezoid form of twice the signed area of a
closed polygon v_1, ..., v_n, indices taken modulo n:

    T(v_1, ..., v_n) = sum_i (x_i + x_{i+1}) * (y_{i+1} - y_i).

T equals the shoelace sum sum_i (x_i * y_{i+1} - x_{i+1} * y_i) by telescoping: each term expands to
x_i * y_{i+1} - x_{i+1} * y_i + (x_{i+1} * y_{i+1} - x_i * y_i), and the brackets cancel around the polygon. For a
triangle, T(a, b, c) = cross2(b - a, c - a) = det[[1, a_x, a_y], [1, b_x, b_y], [1, c_x, c_y]], with
cross2(u, v) = u_x * v_y - u_y * v_x: positive when a, b, c turn counterclockwise, negative when they turn clockwise,
and zero exactly when they lie on a line (when two of them are equal, too).

The theorem that makes True imply the claim, for each operation. The evidence of every operation is the empty tuple.

distance(p, q) = v.
    p and q are points of the same dimension n >= 1 (tuples of exact rationals). The checker sums
    r = sum_i (p_i - q_i)**2 as an exact fraction N/D (D >= 1, not necessarily in lowest terms) and accepts
    - a Rational v = s/t in lowest terms with s >= 0 and s*s*D == t*t*N, that is v >= 0 and v**2 = r: the nonnegative
      square root of r is unique, and so is its canonical Rational;
    - a Sqrt node whose radicand is a Rational u/w in lowest terms with u*D == w*N, so u/w is the canonical Rational of
      r, when u or w is not a perfect square (math.isqrt). A fraction u/w in lowest terms is the square of a rational
      exactly when u and w are perfect squares (a prime that divided both square roots would divide u and w), so the
      node passes exactly when r has no rational square root, with r in its unique canonical form.
    Those are the two cases of model.sqrt(r), the card's canonical value, and no other value passes: an int, a
    product such as 2*sqrt(2) for sqrt(8), a Sqrt of a rational square (sqrt(25) for 5) or of a radicand that is not
    in lowest terms, any other class. Different dimensions and empty points have no value.

polygon_area(points) = v.
    points is a Matrix (identified by MATRIX, read as data) whose rows are the vertices (x, y) in order: two columns
    and any number of rows, or no row and no column, Matrix([]); every other column count has no value. v is a
    Rational s/t in lowest terms with 2*s*D == t*N for T(points) = N/D: v = T/2, the trapezoid form halved, which is
    the signed shoelace area (1/2) * sum_i (x_i * y_{i+1} - x_{i+1} * y_i), and its canonical Rational is unique.
    Fewer than three vertices give 0 (every term cancels with its reverse, or vanishes), and so does no vertex at
    all; a repeated closing vertex adds the term (x_1 + x_1) * (y_1 - y_1) = 0. An int, also an integral area as 6,
    is not the canonical value: the family's values are Rationals.

line_intersection(p1, p2, q1, q2) = LineIntersection(kind, point).
    Four points of dimension 2 with p1 != p2 and q1 != q2, compared as numbers (2 and Rational(2) are one coordinate):
    L1 is the line through p1 and p2, L2 the line through q1 and q2. With A1 = T(p1, p2, q1) and A2 = T(p1, p2, q2),
        A2 - A1 = cross2(p2 - p1, q2 - p1) - cross2(p2 - p1, q1 - p1) = cross2(p2 - p1, q2 - q1) = d
    by the bilinearity of cross2: the card's direction determinant. d != 0 exactly when the directions of L1 and L2 are
    not parallel, and then the lines meet in exactly one point. T(p1, p2, X) = 0 exactly when X lies on L1 (p1 != p2),
    and T(q1, q2, X) = 0 exactly when X lies on L2 (q1 != q2). The record (identified by RECORD) has a kind, one of the
    exact strs of KINDS, and a point, and passes in exactly one of three cases, which exclude each other:
    - "point" when d != 0 and the point is a tuple X of two Rationals in lowest terms with T(p1, p2, X) = 0 and
      T(q1, q2, X) = 0: X lies on both lines, so it is their one common point, whose canonical form is unique;
    - "parallel" with point None when d = 0 and A1 != 0: parallel directions, and q1 is not on L1, so L2 misses L1;
    - "coincident" with point None when d = 0 and A1 = 0: q1 lies on L1 and L2 runs through it parallel to L1, so
      L2 = L1.
    With p1 = p2 every determinant vanishes, and "coincident" would pass whatever the lines are: such a request, and one
    with q1 = q2, has no value, and every claim on it answers False, as does a point of another dimension.

orientation(p, q, r) = s.
    Three points of dimension 2; s is the exact int sign of T(p, q, r) = cross2(q - p, r - p), the card's value: 1 when
    p, q, r turn counterclockwise, -1 when they turn clockwise, 0 when they lie on a line or two of them coincide.

Shapes. The operation must be one of the four strs, args a dict of exactly the operation's argument names, the
evidence (), and every number an exact int or a Rational of the class mathforge.model.Rational with exact int fields,
a positive denominator and lowest terms: a bool, a float, a str, a Fraction, an int or Rational subclass and a Rational
made with object.__new__ outside that form are no exact numbers, a list is no tuple. A request holds a point as a tuple
of ints and Rationals, a Matrix holds Rationals only, and a value holds Rationals only (never an int: the family
normalizes its values). Anything else answers False, never an exception.

Budget. Every loop is charged before it runs, and every operation on long integers in the units of the cost model of
trusted.integers: product_work for a product, division_work by the length of a quotient, a gcd quadratic in its shorter
operand as trusted.upoly.charge_gcd charges it, one unit per 16,384 bits of an addition, and an integer square root as
much as a product of its operand with itself (measured: math.isqrt of a number costs about one such product). No
builtin pow is used. A value integer longer than max_integer_bits answers False, as no value under the budget holds
one, so an oversized forged value costs a few comparisons; values have a fixed number of integers, so a forged value
can cost at most a few products of max_integer_bits. A request integer longer than max_integer_bits raises
ResourceLimitError, as the request's inspection by operations_v6.run and by the verifier does before a checker runs,
and so does a polygon Matrix with more entries than max_matrix_entries, as produce() is charged for it, and work past
max_work: the claim is then unverified, never verified. Sums keep their denominator at the least common multiple of
their terms', so coordinates that share denominators stay cheap, and pairwise coprime ones make the work grow about
with the square of the number of vertices (measured: 2,048 integer vertices take about 20,000 units, 2,048 vertices
over pairwise coprime 64-bit denominators about 1.4 million, when their area already passes max_integer_bits). A
term that is exactly 0 (equal coordinates of a distance, a horizontal edge) is skipped, and a running sum is reduced
to lowest terms whenever its denominator has doubled since the last reduction (_tidy), so that sums whose terms
cancel stay short: the distance between two equal points of 300 coordinates over coprime 2,000-bit denominators
costs about 2,000 units instead of exhausting max_work.

A known limit of the trapezoid form, which no summation order removes: vertices on a line through the origin make
every shoelace term 0, which produce() skips, so it answers 0 for 2,048 vertices of 65,536-bit coprime denominators
in about 20,000 units, while the trapezoid terms (x_i + x_{i+1}) * (y_{i+1} - y_i) stay nonzero (they telescope) and
their products alone, 2,048 products of 131,072-bit integers, cost about 33 million units. Such a request ends in
resource_limit_exceeded (run() cannot verify the value), never in a wrong verdict; up to about 2,048 vertices over
1,000-bit or 500 vertices over 2,000-bit denominators verify within the default max_work.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import gcd, isqrt

from mathforge.limits import Budget, computation
from mathforge.model import Rational, Sqrt
from mathforge.trusted.integers import division_work, product_work

RECORD = ("mathforge.areas.geometry.plane", "LineIntersection")  # Module and name of line_intersection's value class.
MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of polygon_area's argument class, read as data.
KINDS = ("point", "parallel", "coincident")  # The kinds of a LineIntersection.
# The argument names of each operation, in the order of its spec.
ARGUMENTS: dict[str, tuple[str, ...]] = {
    "distance": ("p", "q"),
    "polygon_area": ("points",),
    "line_intersection": ("p1", "p2", "q1", "q2"),
    "orientation": ("p", "q", "r"),
}
_MISSING = object()  # A field that a record made with object.__new__ lacks.
_TIDY = 4_096  # Bits of a running sum's denominator below which _tidy never reduces it: small sums stay unreduced.

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1.
Vertex = tuple[Pair, Pair]  # (x, y) of a point of the plane.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for arguments other than exactly the operation's, for any malformed or
    non-canonical value and for any evidence but (); ResourceLimitError for a request integer over max_integer_bits,
    a polygon over max_matrix_entries and work past the budget."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = ARGUMENTS.get(operation)
    if (names is None or len(args) != len(names) or any(type(name) is not str for name in args)
            or any(name not in args for name in names)):
        return False
    if type(evidence) is not tuple or evidence:
        return False
    with computation() as budget:  # One budget for every helper: the active one, or one of the default limits.
        return _CHECKS[operation](budget, *(args[name] for name in names), value)


# ---------------------------------------------------------------------------------------------------------------
# The operations

def _distance(budget: Budget, p: object, q: object, value: object) -> bool:
    """distance: v >= 0 with v**2 = r, or the node Sqrt(r) when r has no rational square root (module docstring)."""
    kind = type(value)
    if kind is not Rational and kind is not Sqrt:
        return False
    if type(p) is not tuple or type(q) is not tuple or not p or len(p) != len(q):
        return False  # Different dimensions and the empty point are invalid input: no value exists.
    left, right = _point(budget, p), _point(budget, q)
    if left is None or right is None:
        return False
    total: Pair = (0, 1)
    mark = _TIDY
    for a, b in zip(left, right):
        budget.tick()
        if a == b:  # Pairs in lowest terms: equal coordinates add 0, and their denominators must not grow the sum.
            continue
        difference = _subtract(budget, a, b)
        total = _add(budget, total, _multiply(budget, difference, difference))
        total, mark = _tidy(budget, total, mark)
    numerator, denominator = total  # r = numerator / denominator >= 0.
    if kind is Rational:
        root = _canonical(budget, value)
        if root is None or root[0] < 0:
            return False
        s, t = root
        return _times(budget, _times(budget, s, s), denominator) == _times(budget, _times(budget, t, t), numerator)
    radicand = _canonical(budget, getattr(value, "radicand", _MISSING))
    if radicand is None:
        return False
    u, w = radicand
    if _times(budget, u, denominator) != _times(budget, w, numerator):
        return False
    return not (_is_square(budget, u) and _is_square(budget, w))  # u >= 0 here, as r >= 0 and w >= 1.


def _polygon_area(budget: Budget, points: object, value: object) -> bool:
    """polygon_area: the Rational v with 2v = T(points), the trapezoid form (module docstring)."""
    if type(value) is not Rational:
        return False
    vertices = _vertices(budget, points)
    if vertices is None:
        return False
    area = _canonical(budget, value)
    if area is None:
        return False
    numerator, denominator = _twice_area(budget, vertices)
    s, t = area
    return _times(budget, _times(budget, 2, s), denominator) == _times(budget, t, numerator)


def _line_intersection(budget: Budget, p1: object, p2: object, q1: object, q2: object, value: object) -> bool:
    """line_intersection: the classification by A1 = T(p1, p2, q1) and A2 = T(p1, p2, q2), whose difference is the
    direction determinant d, and the point on both lines (module docstring)."""
    fields = _record(value)
    if fields is None:
        return False
    kind, point = fields
    if type(kind) is not str or kind not in KINDS:  # A str subclass is no exact str.
        return False
    if kind == "point":
        if type(point) is not tuple or len(point) != 2:
            return False
    elif point is not None:
        return False
    parsed = [_point(budget, item, 2) for item in (p1, p2, q1, q2)]
    if any(item is None for item in parsed):
        return False
    start1, end1, start2, end2 = parsed  # p1, p2, q1 and q2 as pairs.
    if _same(budget, start1, end1) or _same(budget, start2, end2):
        return False  # p1 == p2 or q1 == q2 determines no line: invalid input, no value exists.
    area1, area2 = _twice_area(budget, (start1, end1, start2)), _twice_area(budget, (start1, end1, end2))
    parallel = _equal(budget, area1, area2)  # d = A2 - A1 = cross2(p2 - p1, q2 - q1) == 0.
    if kind == "parallel":
        return parallel and area1[0] != 0
    if kind == "coincident":
        return parallel and area1[0] == 0
    if parallel:
        return False
    x, y = _canonical(budget, point[0]), _canonical(budget, point[1])
    if x is None or y is None:
        return False
    crossing = (x, y)
    return (_twice_area(budget, (start1, end1, crossing))[0] == 0
            and _twice_area(budget, (start2, end2, crossing))[0] == 0)


def _orientation(budget: Budget, p: object, q: object, r: object, value: object) -> bool:
    """orientation: the sign of T(p, q, r) = cross2(q - p, r - p) (module docstring)."""
    if type(value) is not int or value not in (-1, 0, 1):  # A bool or an IntEnum is no exact int.
        return False
    points = [_point(budget, item, 2) for item in (p, q, r)]
    if any(point is None for point in points):
        return False
    numerator = _twice_area(budget, points)[0]  # The denominator is positive.
    return value == (numerator > 0) - (numerator < 0)


_CHECKS: dict[str, Callable[..., bool]] = {
    "distance": _distance,
    "polygon_area": _polygon_area,
    "line_intersection": _line_intersection,
    "orientation": _orientation,
}


# ---------------------------------------------------------------------------------------------------------------
# Shapes

def _named(value: object, name: tuple[str, str]) -> bool:
    """Whether the value's class has this module and name, both exact strs (a class may set __module__ to any object,
    whose comparison could answer anything)."""
    cls = type(value)
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


def _record(value: object) -> tuple[object, object] | None:
    """(kind, point) of a value of the record class, each field read once; None for any other class (one of another
    module with the same name included) and for a record made with object.__new__ that lacks a field."""
    if not _named(value, RECORD):
        return None
    kind, point = getattr(value, "kind", _MISSING), getattr(value, "point", _MISSING)
    if kind is _MISSING or point is _MISSING:  # Identity tests: a forged field may compare oddly.
        return None
    return kind, point


def _fields(number: object) -> Pair | None:
    """(numerator, denominator) of a Rational, each field read once, when both are exact ints and the denominator is
    positive; None for any other class (an int, a Rational subclass and a class of another module named Rational
    included) and for a Rational made with object.__new__ that lacks a field or holds anything else."""
    if type(number) is not Rational:
        return None
    numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:  # A bool is no exact int.
        return None
    return numerator, denominator


def _lowest(budget: Budget, numerator: int, denominator: int) -> bool:
    """Whether numerator/denominator, with denominator >= 1, is in lowest terms: 0 only as 0/1."""
    if denominator == 1:
        budget.tick()
        return True
    return _gcd(budget, numerator, denominator) == 1


def _argument(budget: Budget, number: object) -> Pair | None:
    """An exact rational of a request as a pair in lowest terms: an int (over 1), or a canonical Rational; None for
    anything else. ResourceLimitError for an integer longer than max_integer_bits."""
    if type(number) is int:
        budget.check_int(number)
        return number, 1
    pair = _fields(number)
    if pair is None:
        return None
    budget.check_int(pair[0])
    budget.check_int(pair[1])
    return pair if _lowest(budget, *pair) else None


def _entry(budget: Budget, number: object) -> Pair | None:
    """An entry of a Matrix as a pair in lowest terms: a canonical Rational, never an int, which no Matrix holds (its
    constructor makes every entry a Rational); None for anything else. ResourceLimitError for an integer longer than
    max_integer_bits."""
    pair = _fields(number)
    if pair is None:
        return None
    budget.check_int(pair[0])
    budget.check_int(pair[1])
    return pair if _lowest(budget, *pair) else None


def _canonical(budget: Budget, number: object) -> Pair | None:
    """A Rational of a value as a pair: exact ints within max_integer_bits, a positive denominator and lowest terms;
    None for anything else, an int included (a value holds Rationals only), and for an integer longer than
    max_integer_bits, which no value under the budget holds, before any arithmetic on it."""
    pair = _fields(number)
    if pair is None:
        return None
    limit = budget.limits.max_integer_bits
    if pair[0].bit_length() > limit or pair[1].bit_length() > limit:
        return None
    return pair if _lowest(budget, *pair) else None


def _point(budget: Budget, point: object, dimension: int | None = None) -> tuple[Pair, ...] | None:
    """The coordinates of a point argument as pairs in lowest terms: a tuple (a request holds lists frozen into
    tuples) of exact rationals, of the given dimension when one is given; None for anything else, the length tested
    before any coordinate is read. ResourceLimitError for an integer longer than max_integer_bits."""
    if type(point) is not tuple or (dimension is not None and len(point) != dimension):
        return None
    coordinates = []
    for coordinate in point:
        budget.tick()
        pair = _argument(budget, coordinate)
        if pair is None:
            return None
        coordinates.append(pair)
    return tuple(coordinates)


def _vertices(budget: Budget, points: object) -> list[Vertex] | None:
    """The rows of a polygon Matrix as vertices: an object of the class named MATRIX, read as data, whose rows are a
    tuple of tuples of two canonical Rationals and whose ncols is 2, or 0 without rows (Matrix([])); None for anything
    else. ResourceLimitError for more entries than max_matrix_entries, which produce() refuses too, before any row is
    read, and for an integer longer than max_integer_bits."""
    if not _named(points, MATRIX):
        return None
    rows, ncols = getattr(points, "rows", _MISSING), getattr(points, "ncols", _MISSING)
    if type(rows) is not tuple or type(ncols) is not int or not (ncols == 2 or (ncols == 0 and not rows)):
        return None
    budget.matrix(len(rows), ncols)
    vertices = []
    for row in rows:
        budget.tick()
        if type(row) is not tuple or len(row) != 2:
            return None
        x, y = _entry(budget, row[0]), _entry(budget, row[1])
        if x is None or y is None:
            return None
        vertices.append((x, y))
    return vertices


# ---------------------------------------------------------------------------------------------------------------
# Exact arithmetic on integers and pairs, every operation charged before it runs

def _times(budget: Budget, a: int, b: int) -> int:
    """a * b: one unit, what a product of these lengths costs beyond it (product_work) and one unit per 4,096 bits of
    the operands for the passes over them (a long integer times a short one is linear)."""
    left, right = a.bit_length(), b.bit_length()
    budget.tick(1 + product_work(left, right) + ((left + right) >> 12))
    return a * b


def _plus(budget: Budget, a: int, b: int) -> int:
    """a + b: one unit, and one per 16,384 bits of the longer operand."""
    budget.tick(1 + (max(a.bit_length(), b.bit_length()) >> 14))
    return a + b


def _quotient(budget: Budget, a: int, b: int) -> int:
    """a // b for b >= 1, charged by the length of the quotient (division_work by the dividend's whole length would
    charge a quotient as long as the dividend) and one unit per 4,096 bits of the dividend."""
    left, right = a.bit_length(), b.bit_length()
    budget.tick(1 + division_work(max(left - right, 0) + 64, right) + (left >> 12))
    return a // b


def _gcd(budget: Budget, a: int, b: int) -> int:
    """math.gcd(a, b): one unit, a division of the longer by the shorter with a pass over the longer, then Lehmer's
    algorithm on the shorter, quadratic, one unit per 64 squared 64-bit words as trusted.upoly.charge_gcd charges it
    (16,384 units at 65,536 bits)."""
    left, right = a.bit_length(), b.bit_length()
    small, large = min(left, right), max(left, right)
    words = small >> 6
    budget.tick(1 + division_work(max(large - small, 0) + 64, small) + (large >> 12) + (words * words >> 6))
    return gcd(a, b)


def _is_square(budget: Budget, n: int) -> bool:
    """Whether n >= 0 is a perfect square: math.isqrt, charged like a product of n with itself, then the square of
    the root compared with n."""
    bits = n.bit_length()
    budget.tick(1 + product_work(bits, bits) + (bits >> 12))
    root = isqrt(n)
    return _times(budget, root, root) == n


def _add(budget: Budget, x: Pair, y: Pair) -> Pair:
    """x + y, with the least common multiple of the two denominators as the denominator of the sum."""
    (a, b), (c, d) = x, y
    if b == d:
        return _plus(budget, a, c), b
    divisor = _gcd(budget, b, d)
    b_part, d_part = (b, d) if divisor == 1 else (_quotient(budget, b, divisor), _quotient(budget, d, divisor))
    return _plus(budget, _times(budget, a, d_part), _times(budget, c, b_part)), _times(budget, b, d_part)


def _subtract(budget: Budget, x: Pair, y: Pair) -> Pair:
    """x - y, as x + (-y)."""
    budget.tick(1 + (y[0].bit_length() >> 14))  # The negation copies the numerator.
    return _add(budget, x, (-y[0], y[1]))


def _multiply(budget: Budget, x: Pair, y: Pair) -> Pair:
    """x * y, not reduced."""
    return _times(budget, x[0], y[0]), _times(budget, x[1], y[1])


def _equal(budget: Budget, x: Pair, y: Pair) -> bool:
    """Whether x == y as numbers, cross-multiplied."""
    return _times(budget, x[0], y[1]) == _times(budget, y[0], x[1])


def _same(budget: Budget, a: tuple[Pair, ...], b: tuple[Pair, ...]) -> bool:
    """Whether two points of pairs in lowest terms are equal as points: lowest terms make equal numbers equal pairs."""
    budget.tick(1 + (sum(number.bit_length() for pair in a for number in pair) >> 14))
    return a == b


def _twice_area(budget: Budget, vertices: Sequence[Vertex]) -> Pair:
    """T(v_1, ..., v_n) = sum_i (x_i + x_{i+1}) * (y_{i+1} - y_i), indices modulo n: twice the signed area of the closed
    polygon (module docstring), as a fraction with a positive denominator; (0, 1) for no vertex."""
    total: Pair = (0, 1)
    mark = _TIDY
    count = len(vertices)
    for index in range(count):
        budget.tick()
        (x0, y0), (x1, y1) = vertices[index], vertices[(index + 1) % count]
        if y0 == y1:  # Pairs in lowest terms: a horizontal edge adds 0, and its denominators must not grow the sum.
            continue
        total = _add(budget, total, _multiply(budget, _add(budget, x0, x1), _subtract(budget, y1, y0)))
        total, mark = _tidy(budget, total, mark)
    return total


def _tidy(budget: Budget, total: Pair, mark: int) -> tuple[Pair, int]:
    """(total, mark) for the next term of a running sum: the sum reduced to lowest terms once its denominator is longer
    than mark bits, and then mark twice the reduced denominator's length, at least _TIDY.

    The reduction is exact, so no verdict depends on it; it bounds the work. A sum keeps the least common multiple of
    its terms' denominators, which grows with every term even where the terms cancel: the trapezoid terms of vertices
    on a line through the origin telescope, (x_i + x_{i+1}) * m * (x_{i+1} - x_i) = m * (x_{i+1}**2 - x_i**2), whose
    shoelace terms are all 0, and produce() skips those and answers 0 within a few thousand units, while the unreduced
    sum of 200 vertices over pairwise coprime 2,000-bit denominators exhausted max_work. Reducing at each doubling of
    the denominator costs, where nothing cancels, at most about 4/3 of one gcd at the final length."""
    numerator, denominator = total
    if denominator.bit_length() <= mark:
        return total, mark
    divisor = _gcd(budget, numerator, denominator)
    if divisor != 1:
        numerator, denominator = _quotient(budget, numerator, divisor), _quotient(budget, denominator, divisor)
    return (numerator, denominator), max(2 * denominator.bit_length(), _TIDY)
