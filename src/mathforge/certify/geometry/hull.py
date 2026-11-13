"""Independent checker of the family mathforge.areas.geometry.hull: convex_hull.

The checker never reruns the producer's algorithm (Andrew's monotone chain: sorting and two stacks). Its geometric
primitive is the orientation of three points a, b, c of the plane, the sign of

    cross2(b - a, c - a) = det[[x_a, y_a, 1], [x_b, y_b, 1], [x_c, y_c, 1]],

positive when a, b, c turn counterclockwise (c strictly left of the directed line a -> b), negative when they turn
clockwise, zero exactly when they lie on one line. Each point x = a/b, y = c/d (lowest terms, b, d >= 1) is written in
homogeneous integer coordinates (X, Y, W) = (a d, c b, b d) with W > 0. The direction from P = (X_p, Y_p, W_p) to Q
is the integer pair (X_q - X_p, Y_q - Y_p) = W (q - p) when W_p = W_q = W, else (X_q W_p - X_p W_q, Y_q W_p - Y_p W_q)
= W_p W_q (q - p): a positive multiple of q - p either way. cross2 is bilinear, so the orientation of a, b, c is the
sign of cross2 of the directions from a to b and from a to c, and since cross2(u, u + w) = cross2(u, w) the turn at b
of a -> b -> c is also the sign of cross2 of the directions a -> b and b -> c. The checker computes each direction
it needs once (the edges of the polygon, the rays from its first vertex) and each orientation then costs two
products.

The theorem that makes True imply the claim. The evidence is the empty tuple.

convex_hull(points) = V, V a k x 2 Matrix of vertices v_0, ..., v_{k-1}.
    points is a Matrix (identified by MATRIX, read as data) of n >= 1 rows (x, y) and 2 columns, S its set of points.
    V is a Matrix (identified by MATRIX) of 1 <= k <= n rows of two canonical Rationals each. The checker accepts when
    (1) every v_i is in S, (2) the v_i are distinct, (3) v_0 is the lowest-then-leftmost point of S (no point of S
    has a smaller y, or the same y and a smaller x), and
    - k = 1: every point of S equals v_0. Then S = {v_0}, whose hull is the point itself.
    - k = 2: every point of S lies on the segment [v_0, v_1]: orientation 0 against v_0, v_1 and both coordinates
      between those of v_0 and v_1. Then S is collinear with at least two distinct points and v_0, v_1 in S, so
      conv(S) = [v_0, v_1] and its two extreme points are v_0 and v_1, in the card's order by (3).
    - k >= 3:
      (a) every cyclically consecutive triple v_{i-1}, v_i, v_{i+1} turns strictly left;
      (b) the polygon turns around exactly once: label each edge e_i = v_{i+1} - v_i "up" when its direction angle
          lies in [0, pi) (dy > 0, or dy = 0 and dx > 0) and "down" otherwise; the label changes exactly twice
          cyclically. By (a) the direction angle grows by a turn in (0, pi) at every vertex, and the half-open range
          of each turn holds at most one multiple of pi, at which the label changes; around the polygon the angle
          grows by 2 pi w for its turning number w >= 1, past exactly 2 w multiples of pi. Two changes give w = 1,
          and a closed polygon whose turns all have the same strict sign and whose turning number is 1 is a simple,
          strictly convex polygon P, counterclockwise. (A pentagram, w = 2, changes the label four times.)
      (c) every point p of S lies in the closed polygon P, decided by the fan of P at v_0: p is on or left of the
          edge v_0 -> v_1 and on or right of the ray v_0 -> v_{k-1}, so p lies in the cone of P at v_0, whose angle
          is below pi; the rays v_0 -> v_i (1 <= i <= k-1) are sorted by strictly increasing angle within that cone,
          so "p on or left of the ray v_0 -> v_i" holds for i = 1 up to some index and fails after it, and a binary
          search finds the largest such i, clamped to k - 2; then p lies in the triangle v_0, v_i, v_{i+1}, the part
          of the cone between those two rays inside P, exactly when p is on or left of the edge v_i -> v_{i+1}. The
          fan triangles cover P, so this holds exactly when p is in P, and the closed convex P is the intersection
          of the closed left half-planes of its edges: (c) is the card's "every input point is on or left of every
          edge", decided in O(log k) orientations per point instead of k.
      A point of S that is a vertex lies in P = conv(v_0, ..., v_{k-1}) without a test, so (c) tests only the other
      distinct points of S; repeated points are tested once.
      Then P contains S, so conv(S) is a subset of P, and P = conv(v_0, ..., v_{k-1}) is a subset of conv(S) by (1):
      P = conv(S). The vertices of the strictly convex P are exactly its extreme points, that is the hull's
      vertices, each once by (2), counterclockwise and without points inside edges by (a) and (b), starting at the
      lowest-then-leftmost point by (3), which an extreme point is. That vertex sequence is unique, so no other value
      passes: another start, clockwise order, a point inside an edge, a repeated vertex, a point outside S, a missing
      extreme point (an input point outside the polygon) all fail, and so do k = 1 or 2 claims for points that are
      not all equal or not collinear (k >= 3 claims for collinear points fail (a)).

Shapes. The operation must be the str "convex_hull", args a dict of exactly the key "points", the evidence (), and
every entry a Rational of the class mathforge.model.Rational with exact int fields, a positive denominator and lowest
terms: a bool, a float, a str, an int (no Matrix holds one: its constructor makes every entry a Rational), a Rational
subclass and a Rational made with object.__new__ outside that form are no canonical entries, a list is no tuple, and a
Matrix row must be a tuple of 2 entries with ncols == 2. Anything else answers False, never an exception.

Budget. Every loop is charged before it runs, and every operation on integers in the units of the cost model of
trusted.integers: product_work for a product, one unit per 16,384 bits of an addition, a gcd quadratic in its shorter
operand as trusted.upoly.charge_gcd charges it, and one unit per 4,096 bits for hashing a point. No builtin pow is
used. A value with more rows than the request's points is refused before any of its entries is read (its vertices
must be distinct points of the request), and a value integer longer than max_integer_bits answers False before any
arithmetic, as no value under the budget holds one: an oversized forged value costs a few comparisons per row. A
request integer longer than max_integer_bits raises ResourceLimitError, as the request's inspection by
operations_v6.run and by the verifier does before a checker runs, and so does a points Matrix with more entries than
max_matrix_entries (charged before its shape is validated, as produce() is charged for it), and work past max_work:
the claim is then unverified, never verified. The work is O(log k) orientations (each two products) for every
distinct input point that is no vertex, O(n + k) for the rest: 2,048 points in convex position fit the default
max_work, and the checker's work stays within a small factor of the producer's (test_hull_checker measures
both on large coordinates).

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.limits import Budget, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of the class of points and of the value, read as data.
OPERATION = "convex_hull"
ARGUMENT = "points"
_MISSING = object()  # A field that a record made with object.__new__ lacks.

Pair = tuple[int, int]  # (numerator, denominator) of a canonical rational: lowest terms, denominator >= 1.
Point = tuple[Pair, Pair]  # (x, y) of a point of the plane; equal points are equal tuples (lowest terms).
Homogeneous = tuple[int, int, int]  # (X, Y, W) with x = X / W, y = Y / W and W >= 1.
Direction = tuple[int, int]  # A positive multiple of the difference of two points, as two integers.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorem of the module.
    False for an unknown operation, for arguments other than exactly {"points": ...}, for any malformed or
    non-canonical value and for any evidence but (); ResourceLimitError for a request integer over max_integer_bits,
    a points Matrix over max_matrix_entries and work past the budget."""
    if type(operation) is not str or operation != OPERATION or type(args) is not dict:
        return False
    if len(args) != 1 or any(type(name) is not str for name in args) or ARGUMENT not in args:
        return False
    if type(evidence) is not tuple or evidence:
        return False
    with computation() as budget:  # One budget for every helper: the active one, or one of the default limits.
        return _convex_hull(budget, args[ARGUMENT], value)


def _convex_hull(budget: Budget, points: object, value: object) -> bool:
    """convex_hull: steps (1)-(3) and the case of k (module docstring)."""
    if not _named(value, MATRIX):
        return False  # Decided before the request is read: no class but Matrix can be the value.
    inputs = _input_points(budget, points)
    if inputs is None:
        return False
    vertices = _value_points(budget, value, len(inputs))
    if vertices is None:
        return False
    known: set[Point] = set()
    distinct: list[Point] = []  # The distinct points of the request in their order: each is judged once below.
    for point in inputs:
        budget.tick(1 + (_bits(point) >> 12))  # Hashing a point reads its integers.
        if point not in known:
            known.add(point)
            distinct.append(point)
    seen: set[Point] = set()
    for vertex in vertices:
        budget.tick(1 + (_bits(vertex) >> 12))
        if vertex not in known or vertex in seen:  # (1) a point of the request, (2) distinct.
            return False
        seen.add(vertex)
    first = vertices[0]
    for point in distinct:  # (3) no point of the request is lower, or as low and further left.
        budget.tick()
        height = _compare(budget, point[1], first[1])
        if height < 0 or (height == 0 and _compare(budget, point[0], first[0]) < 0):
            return False
    if len(vertices) == 1:
        return len(known) == 1  # first is in known, so every point equals it.
    homogeneous = [_homogeneous(budget, vertex) for vertex in vertices]
    if len(vertices) == 2:
        return _on_segment(budget, distinct, vertices, homogeneous)
    count = len(homogeneous)
    edges = []  # edges[i]: a positive multiple of v_{i+1} - v_i, indices modulo k.
    for index in range(count):
        budget.tick()
        edges.append(_direction(budget, homogeneous[index], homogeneous[(index + 1) % count]))
    return _convex_polygon(budget, edges) and _contains(budget, distinct, seen, homogeneous, edges)


def _on_segment(budget: Budget, inputs: list[Point], vertices: list[Point], ends: list[Homogeneous]) -> bool:
    """k = 2: every input point lies on the segment between the two vertices (orientation 0, coordinates between)."""
    (x0, y0), (x1, y1) = vertices
    line = _direction(budget, ends[0], ends[1])
    for point in inputs:
        budget.tick()
        if _cross(budget, line, _direction(budget, ends[0], _homogeneous(budget, point))) != 0:
            return False
        if not (_between(budget, point[0], x0, x1) and _between(budget, point[1], y0, y1)):
            return False
    return True


def _convex_polygon(budget: Budget, edges: list[Direction]) -> bool:
    """k >= 3, from the edge directions e_i (positive multiples of v_{i+1} - v_i): (a) every cyclic triple turns
    strictly left, the turn at v_i being the sign of cross2(e_{i-1}, e_i), and (b) the up/down label of the edges,
    read from the signs of the components of each e_i, changes exactly twice around the polygon, so that it is a
    simple, strictly convex polygon, counterclockwise."""
    count = len(edges)
    for index in range(count):
        budget.tick()
        if _cross(budget, edges[index - 1], edges[index]) <= 0:
            return False
    labels = []
    for index in range(count):
        budget.tick()
        dx, dy = edges[index]
        labels.append(dy > 0 or (dy == 0 and dx > 0))
    changes = 0
    for index in range(count):
        budget.tick()
        changes += labels[index] != labels[index - 1]
    return changes == 2


def _contains(budget: Budget, inputs: list[Point], vertices: set[Point], polygon: list[Homogeneous],
              edges: list[Direction]) -> bool:
    """(c) every distinct input point that is no vertex lies in the closed convex polygon, found in the fan of
    triangles at its first vertex by a binary search over the rays from it (module docstring). A vertex lies in the
    polygon, the convex hull of its vertices, without a test."""
    apex, last = polygon[0], len(polygon) - 1
    rays: list[Direction] = [(0, 0)]  # rays[i]: a positive multiple of v_i - v_0 for 1 <= i <= last.
    for index in range(1, last + 1):
        budget.tick()
        rays.append(_direction(budget, apex, polygon[index]))
    for point in inputs:
        budget.tick(1 + (_bits(point) >> 12))  # Hashing a point reads its integers.
        if point in vertices:
            continue
        p = _homogeneous(budget, point)
        towards = _direction(budget, apex, p)
        if _cross(budget, rays[1], towards) < 0 or _cross(budget, rays[last], towards) > 0:
            return False  # Outside the cone of the polygon at its first vertex.
        low, high = 1, last - 1  # The largest i in [1, last - 1] with p on or left of the ray apex -> polygon[i].
        while low < high:
            budget.tick()
            middle = (low + high + 1) >> 1
            if _cross(budget, rays[middle], towards) >= 0:
                low = middle
            else:
                high = middle - 1
        if _cross(budget, edges[low], _direction(budget, polygon[low], p)) < 0:
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# Shapes

def _named(value: object, name: tuple[str, str]) -> bool:
    """Whether the value's class has this module and name, both exact strs (a class may set __module__ to any object,
    whose comparison could answer anything)."""
    cls = type(value)
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


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


def _rows(matrix: object) -> tuple[tuple, int] | None:
    """(rows, ncols) of an object of the class named MATRIX, each field read once; None when a field is missing or
    is no exact tuple or int (only a Matrix made without its constructor)."""
    if not _named(matrix, MATRIX):
        return None
    rows, ncols = getattr(matrix, "rows", _MISSING), getattr(matrix, "ncols", _MISSING)
    if type(rows) is not tuple or type(ncols) is not int or ncols < 0:
        return None
    return rows, ncols


def _input_points(budget: Budget, points: object) -> list[Point] | None:
    """The rows of the points Matrix as points: n >= 1 rows of two canonical Rationals and ncols == 2; None for
    anything else. ResourceLimitError for more entries than max_matrix_entries, charged before the shape is judged as
    produce() charges it, and for an integer longer than max_integer_bits."""
    found = _rows(points)
    if found is None:
        return None
    rows, ncols = found
    budget.matrix(len(rows), ncols)
    if ncols != 2 or not rows:
        return None
    result = []
    for row in rows:
        budget.tick()
        if type(row) is not tuple or len(row) != 2:
            return None
        x, y = _entry(budget, row[0]), _entry(budget, row[1])
        if x is None or y is None:
            return None
        result.append((x, y))
    return result


def _value_points(budget: Budget, value: object, limit: int) -> list[Point] | None:
    """The rows of the claimed Matrix as points: 1 to limit rows of two canonical Rationals within max_integer_bits
    and ncols == 2; None for anything else, the row count judged before any row is read."""
    found = _rows(value)
    if found is None:
        return None
    rows, ncols = found
    if ncols != 2 or not 1 <= len(rows) <= limit:
        return None
    result = []
    for row in rows:
        budget.tick()
        if type(row) is not tuple or len(row) != 2:
            return None
        x, y = _canonical(budget, row[0]), _canonical(budget, row[1])
        if x is None or y is None:
            return None
        result.append((x, y))
    return result


def _entry(budget: Budget, number: object) -> Pair | None:
    """An entry of the request's Matrix as a pair in lowest terms; None for anything but a canonical Rational.
    ResourceLimitError for an integer longer than max_integer_bits."""
    pair = _fields(number)
    if pair is None:
        return None
    budget.check_int(pair[0])
    budget.check_int(pair[1])
    return pair if _lowest(budget, *pair) else None


def _canonical(budget: Budget, number: object) -> Pair | None:
    """An entry of the value as a pair in lowest terms; None for anything but a canonical Rational, and for an integer
    longer than max_integer_bits, which no value under the budget holds, before any arithmetic on it."""
    pair = _fields(number)
    if pair is None:
        return None
    limit = budget.limits.max_integer_bits
    if pair[0].bit_length() > limit or pair[1].bit_length() > limit:
        return None
    return pair if _lowest(budget, *pair) else None


def _lowest(budget: Budget, numerator: int, denominator: int) -> bool:
    """Whether numerator/denominator, with denominator >= 1, is in lowest terms: 0 only as 0/1. A gcd charged as
    trusted.upoly.charge_gcd charges it: a first division of the longer operand by the shorter, then Lehmer's
    algorithm, one unit per 64 squared 64-bit words of the shorter."""
    if denominator == 1:
        budget.tick()
        return True
    left, right = numerator.bit_length(), denominator.bit_length()
    small, large = min(left, right), max(left, right)
    words = small >> 6
    budget.tick(1 + division_work(max(large - small, 0) + 64, small) + (large >> 12) + (words * words >> 6))
    return gcd(numerator, denominator) == 1


def _bits(point: Point) -> int:
    """The total length of the four integers of a point."""
    (a, b), (c, d) = point
    return a.bit_length() + b.bit_length() + c.bit_length() + d.bit_length()


# ---------------------------------------------------------------------------------------------------------------
# Exact arithmetic, every operation charged before it runs

def _times(budget: Budget, a: int, b: int) -> int:
    """a * b: one unit, what a product of these lengths costs beyond it (product_work) and one unit per 4,096 bits of
    the operands for the passes over them."""
    left, right = a.bit_length(), b.bit_length()
    budget.tick(1 + product_work(left, right) + ((left + right) >> 12))
    return a * b


def _minus(budget: Budget, a: int, b: int) -> int:
    """a - b: one unit, and one per 16,384 bits of the longer operand."""
    budget.tick(1 + (max(a.bit_length(), b.bit_length()) >> 14))
    return a - b


def _compare(budget: Budget, x: Pair, y: Pair) -> int:
    """The sign of x - y for two rationals with positive denominators, cross-multiplied."""
    if x[1] == y[1]:
        budget.tick(1 + (max(x[0].bit_length(), y[0].bit_length()) >> 14))
        left, right = x[0], y[0]
    else:
        left, right = _times(budget, x[0], y[1]), _times(budget, y[0], x[1])
    return (left > right) - (left < right)


def _between(budget: Budget, x: Pair, a: Pair, b: Pair) -> bool:
    """Whether x lies in the closed interval between a and b, in either order."""
    low, high = _compare(budget, x, a), _compare(budget, x, b)
    return low == 0 or high == 0 or low != high


def _homogeneous(budget: Budget, point: Point) -> Homogeneous:
    """(X, Y, W) = (a d, c b, b d) for the point (a/b, c/d): x = X / W, y = Y / W with W >= 1."""
    (a, b), (c, d) = point
    if b == d:
        budget.tick()
        return a, c, b
    return _times(budget, a, d), _times(budget, c, b), _times(budget, b, d)


def _direction(budget: Budget, p: Homogeneous, q: Homogeneous) -> Direction:
    """A positive multiple of q - p as two integers: (X_q - X_p, Y_q - Y_p), which is W (q - p), when the two W agree,
    else (X_q W_p - X_p W_q, Y_q W_p - Y_p W_q), which is W_p W_q (q - p) (module docstring)."""
    (xp, yp, wp), (xq, yq, wq) = p, q
    budget.tick(1 + (max(wp.bit_length(), wq.bit_length()) >> 14))  # Comparing the two W.
    if wp == wq:
        return _minus(budget, xq, xp), _minus(budget, yq, yp)
    return (_minus(budget, _times(budget, xq, wp), _times(budget, xp, wq)),
            _minus(budget, _times(budget, yq, wp), _times(budget, yp, wq)))


def _cross(budget: Budget, u: Direction, v: Direction) -> int:
    """The sign of cross2(u, v) = u_x v_y - u_y v_x: 1 when v points strictly left of u, -1 strictly right, 0 when
    the two are parallel."""
    left, right = _times(budget, u[0], v[1]), _times(budget, u[1], v[0])
    budget.tick(1 + (max(left.bit_length(), right.bit_length()) >> 14))  # The comparison.
    return (left > right) - (left < right)

