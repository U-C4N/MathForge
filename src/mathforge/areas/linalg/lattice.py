"""Hermite and Smith normal forms of integer matrices (mathforge.linalg): hermite_normal_form, smith_normal_form.

Both operations take one argument M, a Matrix (MCP kind and annotation "matrix" / "Matrix"; operations_v6.run already
rejects anything but a Matrix as invalid input) of m rows and n columns, m, n >= 0, the empty shapes included, whose
entries are integers: Rationals with denominator 1. A non-integer entry is invalid input. Resource limits:
current_budget().matrix(m, n) is charged before the shape is validated (a 65 x 65 M is resource_limit_exceeded), every
integer of the request lies within max_integer_bits, and the work runs in the request's budget. The evidence holds
square integer matrices (below) whose sizes the checker charges with current_budget().matrix as well: produce() must
refuse with ResourceLimitError an M whose evidence matrices have more than max_matrix_entries entries (m * m for
hermite_normal_form; m * m or n * n for smith_normal_form: under the default limits more than 64 rows, or columns for
smith_normal_form), since the checker raises ResourceLimitError on exactly those claims.

An integer matrix in the evidence is a tuple of rows, each row a tuple of exact ints (never a Matrix record, a list, a
bool, a float or a str), every int within max_integer_bits; a k x k matrix is k rows of k ints, and the 0 x 0 matrix is
the empty tuple ().

hermite_normal_form(M) -> Matrix
    Value: the row-style Hermite normal form H of M, an m x n Matrix (ncols = n) of integer Rationals (denominator 1),
    with H = U M for a unimodular U: H is in row echelon form, the leading entry (pivot) of each nonzero row is
    positive and strictly to the right of the pivot of the row above, the entries above each pivot lie in [0, pivot),
    and the zero rows come last. It is unique for every integer matrix. Known values (docs/v09/cards/m7-extended.md):
    [[2, 4], [3, 6]] -> [[1, 2], [0, 0]]; [[4, 6], [2, 3]] -> [[2, 3], [0, 0]]; [[2, 3], [5, 7]] -> [[1, 0], [0, 1]];
    [[6, 4], [0, 2]] -> [[6, 0], [0, 2]].
    Evidence: (U, V), two m x m integer matrices with U V == I and H == U M.

smith_normal_form(M) -> tuple[int, ...]
    Value: the invariant factors (d_1, ..., d_k), k = min(m, n), exact ints with d_i >= 0, d_i | d_(i+1) and the zeros
    last: the diagonal of D = U M V with U, V unimodular, D the m x n matrix with the value on its diagonal and zeros
    elsewhere. () when k = 0. Known values: [[2, 4], [6, 8]] -> (2, 4); [[1, 2, 3], [4, 5, 6]] -> (1, 3);
    [[0, 0], [0, 0]] -> (0, 0); [[2, 0], [0, 3]] -> (1, 6).
    Evidence: (U, U_inv, V, V_inv): U and U_inv m x m, V and V_inv n x n integer matrices with U U_inv == I,
    V V_inv == I and U M V == D.

The checker, mathforge.certify.linalg.lattice, verifies these identities and the canonical shapes exactly; its
docstring states the theorems (uniqueness of the Hermite normal form under left multiplication by unimodular matrices,
and of the Smith normal form by the determinantal divisors).

How produce() computes. Nothing here is a proof: the checker verifies U V = I, H = U M (or the four Smith identities)
and the canonical shapes itself.

- It charges one unit for the call and max_matrix_entries for M (current_budget().matrix(m, n)) before it validates
  anything but the class of M and the types of its two fields, then the shape (m tuples of n entries) and the
  entries: exact Rationals with int parts and the denominator 1, each integer within max_integer_bits. A Matrix built
  by its constructor holds canonical Rationals, so a non-integer entry is one with a denominator other than 1; a
  Matrix forged without its constructor may hold 4/2, which the checker refuses (it reads the denominator), so it is
  invalid input too. Then the evidence sizes: budget.matrix(m, m), and budget.matrix(n, n) for smith_normal_form.
- hermite_normal_form: an incremental Hermite normal form in the manner of Kannan and Bachem. The rows of M are
  inserted one at a time into the Hermite basis of the rows before them. A row whose leading column holds a pivot is
  first reduced by the floor quotient of its leading entry by the pivot; a nonzero remainder b < a = pivot is then
  combined with the pivot row by the unimodular 2 x 2 step (s, t; -b/g, a/g), s a + t b = g = gcd(a, b) from the
  extended Euclidean algorithm, which leaves g at the pivot and clears the row's entry; the row goes on to its next
  nonzero column. A row whose leading column holds no pivot becomes a pivot row there (negated when its leading entry
  is negative), and a row that runs out of nonzero entries is a zero row (its row of U lies in the left kernel). After
  every insertion that changed the basis, the entries above every pivot from the first changed one down are reduced
  into [0, pivot) by floor quotients, top to bottom, so that the basis is the Hermite normal form of the rows inserted
  so far and its entries stay as small as that form (bounded through the determinantal divisors) rather than growing
  like those of an elimination without reduction. Finally the pivot rows are put in the order of their columns and the
  zero rows after them. Every row operation on A = U M is applied to U as well, and its inverse to U^-1, kept as its
  transpose (a column operation on U^-1 is a row operation on the transpose), so the evidence (U, U^-1) needs no
  inversion.
- smith_normal_form: alternating Hermite normal forms, of A (row operations, tracked in U and U^-1) and of the
  transpose of A (column operations, tracked in the transpose of V and in V^-1), each by the routine above, until A is
  diagonal; the first diagonal entry divides the one before it at every phase and is fixed once it divides its row and
  column, so the alternation ends, and a matrix that is both a row and a column Hermite normal form is diagonal, with
  its positive entries first. The positive entries are then put in increasing order by one permutation of the rows
  and the columns (a diagonal that is a chain up to order needs nothing more). Then the divisibility chain: at a pair
  i < j with d_i not dividing d_j, the unimodular pair L = (s, t; -b/g, a/g) on rows i, j and
  R = (1, -t b/g; 1, s a/g) on columns i, j turns diag(a, b) into diag(gcd, lcm) = diag(g, a b / g). The pairs are
  the comparators of Batcher's odd-even merge sort on the positive entries (gcd and lcm are the meet and join of the
  distributive lattice of divisibility, which a sorting network sorts, so the network makes the chain; its small
  depth keeps the evidence entries short), followed by every pair i < j as a final pass that only finds the chain
  made. The value is the diagonal (d_1, ..., d_k), the evidence (U, U^-1, V, V^-1).

Limits. ResourceLimitError refuses what the checker refuses: an M past max_matrix_entries (before its shape is read),
an evidence matrix past it (m x m, and n x n for smith_normal_form: under the default limits more than 64 rows, or
columns for smith_normal_form), an integer of the value or of the evidence longer than max_integer_bits (the checker
answers False on it and operations_v6.run refuses it as well), and a stored Result that the verifier would not admit
under max_evidence_items (it counts the Result's own items, the request and the value twice and the evidence once).
One refusal of the algorithm remains for claims that the checker would accept: an intermediate row whose entries pass
max_integer_bits is refused at its next use, although the final entries might have fit; under the default limits
that takes entries of thousands of bits, whose products max_work refuses first. Work past max_work is refused before
it runs: every row operation is charged, before it runs, on the scale of trusted.integers (the checker's), one unit
per entry plus product_work of the coefficient and the longest entry of the source row and (a + b) / 1024 for the
additions; every step of the extended Euclidean algorithm by its division_work and its products; a unit per entry
scanned for a leading column, compared above a pivot or tested off the diagonal. A 64 x 64 matrix of small entries
fits the default budget of 2,000,000 units for both operations (tests/v09/linalg/test_lattice.py measures it); the
checker's own work bounds the evidence it can verify, and a Result that its budget refuses comes back from
operations_v6.run as resource_limit_exceeded.

The family defines no record class: the value of hermite_normal_form is a Matrix (mathforge.matrices), that of
smith_normal_form a tuple of ints, both presented by the shared renderer, so TYPES is empty and there is no RENDER.
mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/linalg/test_lattice.py is the definition of done.
"""

from __future__ import annotations

from bisect import bisect_left

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...matrices import Matrix, trusted_matrix
from ...model import Rational, trusted_rational
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

# The scope text of the area's v0.9 families (areas/linalg/eigen.py): it already names the matrices of this family.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

# The matrix is M, never A: an argument named A would be copied into Result.expression (operations_v6.source_context).
SPECS = (
    OperationSpec("hermite_normal_form", "linalg",
                  (Argument("M", "Matrix", None, "matrix"),),
                  "Matrix", "Compute the row-style Hermite normal form of an integer matrix.", SCOPE, version=6),
    OperationSpec("smith_normal_form", "linalg",
                  (Argument("M", "Matrix", None, "matrix"),),
                  "tuple[int, ...]", "Compute the invariant factors (Smith normal form) of an integer matrix.", SCOPE,
                  version=6),
)

TYPES: dict[str, type] = {}  # The values are a v0.7 Matrix and a tuple of ints: no record class of this family.
CHECKER = "mathforge.certify.linalg.lattice"

IntRows = list[list[int]]

_OPERATIONS = ("hermite_normal_form", "smith_normal_form")
_NOT_AN_INTEGER_MATRIX = "{} requires a Matrix M of integers"
_INTERMEDIATE = "Computation limit exceeded: integer bits (an intermediate entry of the reduction)"
_BITS = "Computation limit exceeded: integer bits"
# What the verifier counts among its evidence items besides the value (twice), the evidence (once) and the Matrix M
# (twice): the Result's own fields with its four stored checks, and in each request the request itself, its operation
# name, its tuple of arguments, the pair ("M", M) and the name "M" (the counts of mathforge.areas.linalg.eigen, whose
# test measures the verifier's walk).
_RESULT_ITEMS = 54
_REQUEST_ITEMS = 5


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(H, (U, U^-1)) for hermite_normal_form and ((d_1, ..., d_k), (U, U^-1, V, V^-1)) for smith_normal_form, for
    args = {"M": M}, computed as the module states. InvalidInput for an M that is no Matrix of integers;
    ResourceLimitError past the budget, whose max_matrix_entries is charged before the shape of M is validated."""
    if type(operation) is not str or operation not in _OPERATIONS:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the lattice family")
    if type(args) is not dict or set(args) != {"M"}:
        raise InvalidInput(f"{operation} takes exactly the argument M")
    matrix = args["M"]
    with computation() as budget:
        budget.tick()  # The call itself.
        int_rows, ncols = _integer_rows(budget, operation, matrix)
        size = len(int_rows)
        budget.matrix(size, size)  # U and U^-1: the checker refuses larger ones with ResourceLimitError.
        if operation == "hermite_normal_form":
            value, evidence = _hermite_normal_form(budget, int_rows, ncols)
        else:
            budget.matrix(ncols, ncols)  # V and V^-1.
            value, evidence = _smith_normal_form(budget, int_rows, ncols)
        _check_items(budget, size, ncols, value, evidence)
        return value, evidence


def _integer_rows(budget: Budget, operation: str, matrix: object) -> tuple[IntRows, int]:
    """The integer entries of M as rows of ints and its column count. budget.matrix(m, n) comes first, after the
    class of M and the types of its fields. InvalidInput for anything but a Matrix of m tuples of n exact Rationals
    with int parts and the denominator 1; ResourceLimitError for an integer longer than max_integer_bits."""
    message = _NOT_AN_INTEGER_MATRIX.format(operation)
    if type(matrix) is not Matrix:  # A subclass, the rows alone or any other object is no Matrix.
        raise InvalidInput(message)
    rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
    if type(rows) is not tuple or type(ncols) is not int:
        raise InvalidInput(message)
    budget.matrix(len(rows), ncols)  # First: a 65 x 65 M exceeds the budget before its shape is judged.
    if ncols < 0:
        raise InvalidInput(message)
    result = []
    for row in rows:
        budget.tick(1 + ncols)
        if type(row) is not tuple or len(row) != ncols:
            raise InvalidInput(message)
        items = []
        for entry in row:
            if type(entry) is not Rational:
                raise InvalidInput(message)
            numerator = getattr(entry, "numerator", None)  # A forged Rational may lack a slot.
            denominator = getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int:
                raise InvalidInput(message)
            if denominator != 1:
                raise InvalidInput(f"{operation} requires integer entries, not a fraction")
            budget.check_int(numerator)
            items.append(numerator)
        result.append(items)
    return result, ncols


def _identity(size: int) -> IntRows:
    return [[int(i == j) for j in range(size)] for i in range(size)]


class _Rows:
    """Unimodular row operations on a matrix a (or none), applied to its transform t as well, and their inverses to w,
    the transpose of t's inverse: a = t A_0 and t w^T = I hold after every operation. Each operation is charged before
    it runs, on the scale of trusted.integers."""

    __slots__ = ("budget", "a", "t", "w", "limit", "product_work", "division_work")

    def __init__(self, budget: Budget, a: IntRows | None, t: IntRows, w: IntRows) -> None:
        from ...trusted.integers import division_work, product_work

        self.budget, self.a, self.t, self.w = budget, a, t, w
        self.limit = budget.limits.max_integer_bits
        self.product_work, self.division_work = product_work, division_work

    def _charge(self, coefficient_bits: int, sources: list[list[int]]) -> None:
        """Charge row operations with a coefficient of that length on each source row (one product and one addition
        per entry). ResourceLimitError for a source entry past max_integer_bits."""
        total = 1
        for row in sources:
            if not row:
                continue
            bits = max(map(int.bit_length, row))
            if bits > self.limit:
                raise ResourceLimitError(_INTERMEDIATE)
            total += len(row) * (1 + self.product_work(coefficient_bits, bits) + ((coefficient_bits + bits) >> 10))
        self.budget.tick(total)

    def _matrices(self) -> tuple[IntRows, ...]:
        return (self.t,) if self.a is None else (self.a, self.t)

    def swap(self, i: int, j: int) -> None:
        """Rows i and j exchanged; the inverse's columns i and j likewise."""
        self.budget.tick()
        for rows in self._matrices() + (self.w,):
            rows[i], rows[j] = rows[j], rows[i]

    def negate(self, i: int) -> None:
        """Row i negated; the inverse's column i likewise."""
        matrices = self._matrices() + (self.w,)
        self._charge(1, [rows[i] for rows in matrices])
        for rows in matrices:
            rows[i] = [-x for x in rows[i]]

    def add(self, i: int, j: int, c: int) -> None:
        """Row i += c row j, for i != j; the inverse's column j -= c times its column i."""
        matrices = self._matrices()
        self._charge(c.bit_length(), [rows[j] for rows in matrices] + [self.w[i]])
        for rows in matrices:
            rows[i] = [x + c * y for x, y in zip(rows[i], rows[j])]
        w = self.w
        w[j] = [x - c * y for x, y in zip(w[j], w[i])]

    def combine(self, p: int, i: int, s: int, t: int, x: int, y: int) -> None:
        """Rows (p, i) <- (s row p + t row i, x row p + y row i) for s y - t x = 1; the inverse's columns
        (p, i) <- (y col p - x col i, -t col p + s col i)."""
        matrices = self._matrices()
        bits = max(s.bit_length(), t.bit_length(), x.bit_length(), y.bit_length())
        sources = [rows[k] for rows in matrices + (self.w,) for k in (p, i)]
        self._charge(bits, sources)
        self._charge(bits, sources)  # Two products and an addition per new entry.
        for rows in matrices:
            first, second = rows[p], rows[i]
            rows[p] = [s * u + t * v for u, v in zip(first, second)]
            rows[i] = [x * u + y * v for u, v in zip(first, second)]
        w = self.w
        first, second = w[p], w[i]
        w[p] = [y * u - x * v for u, v in zip(first, second)]
        w[i] = [s * v - t * u for u, v in zip(first, second)]

    def permute(self, order: list[int]) -> None:
        """Row k of each matrix becomes its row order[k]; the inverse's columns likewise."""
        self.budget.tick(1 + len(order))
        if self.a is not None:
            self.a = [self.a[k] for k in order]
        self.t = [self.t[k] for k in order]
        self.w = [self.w[k] for k in order]

    def xgcd(self, a: int, b: int) -> tuple[int, int, int]:
        """(g, s, t) with s a + t b = g = gcd(a, b) > 0, for ints a > 0 and b > 0, by the extended Euclidean
        algorithm (|s| <= b / g and |t| <= a / g); every step charged before it runs."""
        budget, product_work, division_work = self.budget, self.product_work, self.division_work
        s0, s1, t0, t1 = 1, 0, 0, 1
        while b:
            a_bits = a.bit_length()
            budget.tick(3 + division_work(a_bits, b.bit_length()) + 2 * product_work(a_bits, a_bits))
            quotient, remainder = divmod(a, b)
            a, b = b, remainder
            s0, s1 = s1, s0 - quotient * s1
            t0, t1 = t1, t0 - quotient * t1
        return a, s0, t0


def _lead(budget: Budget, row: list[int], start: int) -> int:
    """The first column at or after start where row is nonzero, -1 if none; a unit per entry scanned."""
    budget.tick(1 + len(row) - start)
    for column in range(start, len(row)):
        if row[column]:
            return column
    return -1


def _hermite(rows: _Rows, ncols: int) -> None:
    """Put rows.a (with its transform) into row Hermite normal form by inserting its rows one at a time into the
    Hermite basis of the rows before them, as the module states."""
    budget = rows.budget
    pivots: list[int] = []  # Row indices of the basis, by increasing pivot column.
    columns: list[int] = []  # Their pivot columns.
    owner: dict[int, int] = {}  # Pivot column -> row index.
    for i in range(len(rows.a)):
        budget.tick()
        start = _insert(rows, i, pivots, columns, owner)
        if start is not None:
            _reduce(rows, pivots, columns, start)
    basis = set(pivots)
    rows.permute(pivots + [i for i in range(len(rows.a)) if i not in basis])


def _insert(rows: _Rows, i: int, pivots: list[int], columns: list[int], owner: dict[int, int]) -> int | None:
    """Insert row i into the Hermite basis: reduce and combine it with the pivot rows of its leading columns until it
    becomes a pivot row or zero. The position in pivots of the first basis row that changed, None if none did."""
    budget, a = rows.budget, rows.a
    changed = None
    lead = -1
    while True:
        lead = _lead(budget, a[i], lead + 1)  # Row i is zero before lead + 1.
        if lead < 0:
            return changed  # A zero row.
        p = owner.get(lead)
        if p is None:  # A new pivot row.
            if a[i][lead] < 0:
                rows.negate(i)
            position = bisect_left(columns, lead)
            pivots.insert(position, i)
            columns.insert(position, lead)
            owner[lead] = i
            return position if changed is None else min(changed, position)
        pivot = a[p][lead]
        quotient = a[i][lead] // pivot  # Floor: the remainder lies in [0, pivot).
        if quotient:
            rows.add(i, p, -quotient)
        remainder = a[i][lead]
        if remainder:
            g, s, t = rows.xgcd(pivot, remainder)
            rows.combine(p, i, s, t, -(remainder // g), pivot // g)  # Determinant (s pivot + t remainder) / g = 1.
            position = bisect_left(columns, lead)
            changed = position if changed is None else min(changed, position)


def _reduce(rows: _Rows, pivots: list[int], columns: list[int], start: int) -> None:
    """Reduce the entries above every pivot from position start down into [0, pivot) by floor quotients, top to
    bottom: a reduction by the pivot row at position k changes the rows above it only in later columns."""
    budget, a = rows.budget, rows.a
    for index in range(start, len(pivots)):
        q, column = pivots[index], columns[index]
        pivot = a[q][column]
        budget.tick(1 + index)
        for p in pivots[:index]:
            entry = a[p][column]
            if entry < 0 or entry >= pivot:
                rows.add(p, q, -(entry // pivot))


def _transpose(rows: IntRows, width: int) -> IntRows:
    """The transpose of integer rows of that width."""
    return [list(column) for column in zip(*rows)] if rows else [[] for _ in range(width)]


def _is_diagonal(budget: Budget, a: IntRows, ncols: int) -> bool:
    """Whether every entry off the diagonal is zero; a unit per entry."""
    budget.tick(1 + len(a) * ncols)
    return all(not entry or index == column for index, row in enumerate(a) for column, entry in enumerate(row))


def _batcher_pairs(count: int) -> list[tuple[int, int]]:
    """The comparators (i, j), i < j < count, of Batcher's odd-even merge sort on count items, in their order: the
    network of the next power of two without the comparators that reach past count (items there would be greatest,
    which a comparator leaves in place)."""
    pairs = []
    p = 1
    while p < count:
        k = p
        while k:
            for j in range(k % p, count - k, 2 * k):
                for i in range(min(k, count - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


def _check_bits(budget: Budget, matrices: tuple[IntRows, ...]) -> None:
    """ResourceLimitError for an entry longer than max_integer_bits: the checker answers False on it."""
    limit = budget.limits.max_integer_bits
    for rows in matrices:
        budget.tick(1 + len(rows))
        for row in rows:
            if row and max(map(int.bit_length, row)) > limit:
                raise ResourceLimitError(_BITS)


def _frozen(rows: IntRows) -> tuple[tuple[int, ...], ...]:
    """Integer rows in the evidence layout: a tuple of tuples of ints."""
    return tuple(map(tuple, rows))


def _hermite_normal_form(budget: Budget, int_rows: IntRows, ncols: int) -> tuple[Matrix, tuple]:
    """(H, (U, U^-1))."""
    size = len(int_rows)
    rows = _Rows(budget, int_rows, _identity(size), _identity(size))
    _hermite(rows, ncols)
    inverse = _transpose(rows.w, size)
    _check_bits(budget, (rows.a, rows.t, inverse))
    budget.tick(1 + size * (ncols + 2 * size))
    value = trusted_matrix(tuple(tuple(trusted_rational(entry, 1) for entry in row) for row in rows.a), ncols)
    return value, (_frozen(rows.t), _frozen(inverse))


def _smith_normal_form(budget: Budget, int_rows: IntRows, ncols: int) -> tuple[tuple[int, ...], tuple]:
    """((d_1, ..., d_k), (U, U^-1, V, V^-1))."""
    size = len(int_rows)
    a = int_rows
    left, left_inverse_t = _identity(size), _identity(size)  # U and the transpose of U^-1.
    right_t, right_inverse = _identity(ncols), _identity(ncols)  # The transpose of V, and V^-1.
    by_rows, phases = True, 0
    # At least one phase: a diagonal M such as diag(-2, 3) or diag(0, 3) is no Smith form yet, but a diagonal Hermite
    # normal form has its positive entries first and its zeros last.
    while not phases or not _is_diagonal(budget, a, ncols):
        if by_rows:  # A <- E A: U <- E U, U^-1 <- U^-1 E^-1.
            rows = _Rows(budget, a, left, left_inverse_t)
            _hermite(rows, ncols)
            a, left, left_inverse_t = rows.a, rows.t, rows.w
        else:  # A <- A F, that is A^T <- F^T A^T: V^T <- F^T V^T, (V^-1)^T <- (V^-1)^T F^-T.
            budget.tick(1 + size * ncols)
            rows = _Rows(budget, _transpose(a, size), right_t, right_inverse)
            _hermite(rows, size)
            budget.tick(1 + size * ncols)
            a, right_t, right_inverse = _transpose(rows.a, ncols), rows.t, rows.w
        by_rows, phases = not by_rows, phases + 1
    count = min(size, ncols)
    factors = [a[index][index] for index in range(count)]  # Positive first, then zeros: a Hermite normal form.
    on_rows = _Rows(budget, None, left, left_inverse_t)
    on_columns = _Rows(budget, None, right_t, right_inverse)
    # The positive entries in increasing order first, by the same permutation P of rows and columns (P D P^T): a
    # diagonal that is a chain up to order, such as diag(2^64, 2^63, ..., 2) which a Hermite normal form leaves as it
    # is, then needs no gcd step, where the steps below would act as swaps made of gcd steps (and the pass alone, which
    # the network below replaced, made one at each of its count^2 / 2 pairs: over twice the default max_work on 64 x 64,
    # for a claim whose check costs about half of it).
    rank = next((index for index, factor in enumerate(factors) if not factor), count)
    budget.tick(1 + rank * rank.bit_length() * (1 + (max(factors, default=0).bit_length() >> 10)))
    order = sorted(range(rank), key=factors.__getitem__)
    if order != list(range(rank)):
        on_rows.permute(order + list(range(rank, size)))
        on_columns.permute(order + list(range(rank, ncols)))
        factors = [factors[index] for index in order] + factors[rank:]
    # The chain: the step (d_i, d_j) -> (gcd, lcm) at the pairs of a comparator network, Batcher's odd-even merge sort
    # on the positive entries, then at every pair i < j as a final pass that finds the chain made (it only divides).
    # Divisibility is a distributive lattice with gcd and lcm as meet and join, where a network that sorts every 0-1
    # input sorts every input, so the network alone makes the chain; its depth (log2(rank)^2 / 2) rather than the pass
    # alone, which makes d_0 the gcd of all, then d_1, and so on, keeps the transforms short: each step multiplies two
    # rows of U and of V^T by cofactors as long as the entries, and the pass alone multiplied the rows of a diagonal
    # of 64 primes by every partial product (evidence entries of 11,568 bits, a check of 10.3M units, against 898 bits
    # and 1.4M units: tests/v09/linalg/test_lattice.py).
    for i, j in _batcher_pairs(rank) + [(i, j) for i in range(rank) for j in range(i + 1, rank)]:
        first, second = factors[i], factors[j]
        budget.tick(1 + on_rows.division_work(second.bit_length(), first.bit_length()))
        if not second % first:
            continue
        g, s, t = on_rows.xgcd(first, second)
        # L diag(a, b) R = diag(g, a b / g): L = (s, t; -b/g, a/g) on the rows, R = (1, -t b/g; 1, s a/g) on the
        # columns, that is R^T = (1, 1; -t b/g, s a/g) on the rows of V^T. Both have determinant 1.
        on_rows.combine(i, j, s, t, -(second // g), first // g)
        on_columns.combine(i, j, 1, 1, -(t * second // g), s * first // g)
        budget.tick(1 + on_rows.product_work(first.bit_length(), second.bit_length()))
        factors[i], factors[j] = g, first // g * second
    # permute() rebinds the lists, combine() replaces their rows: read the transforms back from the two trackers.
    left, left_inverse_t, right_t, right_inverse = on_rows.t, on_rows.w, on_columns.t, on_columns.w
    left_inverse = _transpose(left_inverse_t, size)
    right = _transpose(right_t, ncols)
    _check_bits(budget, (left, left_inverse, right, right_inverse, [factors]))
    return tuple(factors), (_frozen(left), _frozen(left_inverse), _frozen(right), _frozen(right_inverse))


def _check_items(budget: Budget, size: int, ncols: int, value: object, evidence: tuple) -> None:
    """ResourceLimitError when the verifier would not admit the stored Result under max_evidence_items: it counts the
    Result's own items, the request twice, the value twice and the evidence once, as Budget.inspect walks them; a
    separate budget of the same limits counts them here."""
    budget.tick(1 + size)
    counter = Budget(budget.limits)
    request = _REQUEST_ITEMS + 3 + size + 3 * size * ncols  # The Matrix: itself, its rows tuple, the rows, the
    counter.evidence(_RESULT_ITEMS + 2 * request)  # Rationals (three items each) and ncols.
    counter.inspect(value, evidence=True)
    counter.inspect(value, evidence=True)
    counter.inspect(evidence, evidence=True)
    budget.tick(counter.evidence_count >> 4)  # The walk itself.
