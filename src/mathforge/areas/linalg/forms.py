"""Inertia and definiteness of symmetric rational matrices, and Gram-Schmidt orthogonalization of rational rows
(mathforge.linalg).

definiteness(M) takes a square Matrix M of exact Rationals, n x n with n >= 0 (MCP kind and annotation "Matrix"), that
is symmetric (M[i][j] == M[j][i]); a matrix that is not square or not symmetric is invalid input. It returns an
Inertia record: positive, negative and zero, the numbers of positive, negative and zero eigenvalues of M counted with
multiplicity (they add up to n), and kind, derived from them in this order: positive == n gives "positive_definite",
negative == n "negative_definite", negative == 0 "positive_semidefinite" (the zero matrix included), positive == 0
"negative_semidefinite", anything else "indefinite". The 0 x 0 matrix gives Inertia(0, 0, 0, "positive_definite").

Known values (docs/v09/cards/m7-extended.md): [[2, 1], [1, 2]] gives (2, 0, 0) positive_definite; [[1, 2], [2, 1]]
(1, 1, 0) indefinite; [[0, 0], [0, 0]] (0, 0, 2) positive_semidefinite; [[0, 1], [1, 0]] (1, 1, 0) indefinite; [[-1]]
(0, 1, 0) negative_definite; [[1, 1], [1, 1]] (1, 0, 1) positive_semidefinite.

Evidence of definiteness: (S, D), the layout that mathforge.certify.linalg.forms checks.

- S: an invertible rational n x n matrix as a tuple of n row tuples of n Rationals (exact class
  mathforge.model.Rational, never an int; lowest terms are not required); its columns are the new basis.
- D: a tuple of n Rationals with S^T M S = diag(D) exactly.
  For n = 0 the evidence is ((), ()). Every integer in S and D stays within max_integer_bits.

gram_schmidt(M) takes a Matrix M of exact Rationals, k x n with k, n >= 0, whose rows m_0, ..., m_(k-1) are linearly
independent; dependent rows (some q_i below is the zero vector; k > n always) are invalid input. It returns the Matrix
Q, k x n with the column count of M, of canonical Rationals (lowest terms, 0 as 0/1), whose rows are q_i = m_i -
sum_{j<i} mu_ij q_j with mu_ij = (m_i . q_j)/(q_j . q_j): the rows are orthogonal and not normalized. The matrix with no
row gives the matrix with no row and n columns.

Known values: [[3, 1], [2, 2]] gives [[3, 1], [-2/5, 6/5]]; [[1, 0], [0, 1]] gives itself; [[1, 1, 0], [1, 0, 1]] gives
[[1, 1, 0], [1/2, -1/2, 1]].

Evidence of gram_schmidt: mu itself, a tuple of k tuples, the tuple at index i holding the i Rationals mu_i0, ...,
mu_i(i-1) (so the first is ()): ((), (mu_10,), (mu_20, mu_21), ...); () for k = 0. Every integer within
max_integer_bits; lowest terms are not required.

How produce() computes (docs/v09/cards/m7-extended.md). Nothing here is a proof: the checker verifies every claim
with the certificates above.

- It charges one unit for the call and max_matrix_entries for M (current_budget().matrix) before it validates anything
  but the class of M, so that a 65 x 65 M exceeds the budget whatever its shape; then the shape (definiteness: square)
  and the entries: exact Rationals with int parts, a positive denominator and in lowest terms, each integer within
  max_integer_bits. A Matrix built by its constructor holds no other entry; one forged without it may, and produce()
  rejects it as invalid input. definiteness then requires M[i][j] == M[j][i].
- definiteness: m, the least common multiple of the denominators of M, and the symmetric integer matrix M_int = m M.
  A symmetric fraction-free (Bareiss) elimination of M_int: at step k the pivot is the diagonal entry A[k][k]; when it
  is 0, a later diagonal entry that is not 0 is swapped in symmetrically (the same rows and columns), else, for an
  A[k][j] != 0, the congruence step "index k += index j" (rows, then columns) makes the pivot 2 A[k][j]; a row of the
  remaining block that is 0 everywhere is swapped to the end and leaves the elimination (its D entry is 0). Every
  update A[i][j] = (p A[i][j] - A[i][k] A[k][j]) / d_k divides exactly (Sylvester's identity: the entries are minors of
  the congruent integer matrix M' = T^T M_int T, T the unimodular product of the swaps and additions). The same row
  operations on the identity give a lower triangular integer U with U[k][k] = d_k, d_0 = 1 and d_(k+1) the k-th
  pivot, and U M' U^T = diag(d_k d_(k+1)) (0 for the rows that left), so S = T U^T is an invertible integer matrix with
  S^T M S = diag(d_k d_(k+1) / m). S is emitted as Rationals of denominator 1, D in lowest terms, and the counts of
  the signs of D give the Inertia.
- gram_schmidt: each row scaled by the least common multiple r_i of its denominators, M_int[i] = r_i m_i, then the
  integral Gram-Schmidt recurrence on M_int: with d_0 = 1 and the integer vectors Q'_j = d_j q'_j (q'_j the
  Gram-Schmidt rows of M_int), lambda_ij = M_int[i] . Q'_j, v = M_int[i] and v = (d_(j+1) v - lambda_ij Q'_j) / d_j
  for j < i (exact: d_j times a projection onto the span of j integer rows is integral, by Cramer's rule), Q'_i = v and
  d_(i+1) = Q'_i . Q'_i / d_i, the Gram determinant of the first i + 1 rows. Q'_i = 0 means dependent rows: invalid
  input. Then q_i = Q'_i / (d_i r_i) in lowest terms and mu_ij = lambda_ij r_j / (d_(j+1) r_i), also in lowest terms
  (mu_ij of M is that of M_int times r_j / r_i). k > n is refused at once, as the rows are then dependent.

Limits. ResourceLimitError where the checker would answer False to a true claim: an integer of S, D, Q or mu longer
than max_integer_bits. produce() refuses a Result that the verifier would not admit under max_evidence_items (it counts
the Result's own items, the request twice, the value twice and the evidence once), and run() refuses one whose
independent check passes max_work. Every step charges the request's budget before it runs, on the cost model of
trusted.integers (product_work for a product, division_work for an exact division, a unit per entry), from the bit
lengths of the integers it is about to combine: a 64 x 64 matrix of small entries costs some 10^5 units, and long
entries meet max_work before the long-integer work runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...matrices import Matrix
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

# The scope text of the linear-algebra families of v0.9 (mathforge.areas.linalg.eigen): it already names the
# matrices that these operations take.
_SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

SPECS = (
    OperationSpec("definiteness", "linalg",
                  (Argument("M", "Matrix", None, "matrix"),),
                  "Inertia", "Find the inertia and definiteness of a symmetric rational matrix.", _SCOPE, version=6),
    OperationSpec("gram_schmidt", "linalg",
                  (Argument("M", "Matrix", None, "matrix"),),
                  "Matrix", "Orthogonalize the linearly independent rows of a rational matrix without normalizing.",
                  _SCOPE, version=6),
)

def _kind(positive: int, negative: int, zero: int) -> str:
    """The kind of the counts, tested in the card's order."""
    n = positive + negative + zero
    if positive == n:
        return "positive_definite"
    if negative == n:
        return "negative_definite"
    if negative == 0:
        return "positive_semidefinite"
    if positive == 0:
        return "negative_semidefinite"
    return "indefinite"


@dataclass(frozen=True)
class Inertia:
    """The inertia of a symmetric matrix: the numbers of its positive, negative and zero eigenvalues, and its kind.

    The canonical form: positive, negative and zero are exact ints >= 0 (their sum is the size of the matrix), and kind
    is the str that the counts give in this order: positive == n "positive_definite", negative == n
    "negative_definite", negative == 0 "positive_semidefinite", positive == 0 "negative_semidefinite", otherwise
    "indefinite"."""
    positive: int
    negative: int
    zero: int
    kind: str

    def __post_init__(self):
        with computation() as budget:
            budget.tick()
            counts = (self.positive, self.negative, self.zero)
            for count in counts:
                if type(count) is not int or count < 0:  # Neither bool nor float is an exact integer.
                    raise InvalidInput("The counts of an inertia are exact integers >= 0")
                budget.check_int(count)
            if type(self.kind) is not str or self.kind != _kind(*counts):
                raise InvalidInput(f"The kind of the inertia {counts} is {_kind(*counts)!r}")


TYPES = {"inertia": Inertia}
CHECKER = "mathforge.certify.linalg.forms"


_BITS = "Computation limit exceeded: integer bits"
_NOT_A_MATRIX = "{} requires a Matrix M of exact rationals"
# What the verifier counts among its evidence items besides the value (twice), the evidence (once) and the Matrix M
# (twice, in the request of the Result and in that of its step): the Result's own fields with its four stored checks
# (_RESULT_ITEMS, as in mathforge.areas.linalg.eigen), and in each request the request itself, its operation name, its
# tuple of arguments, the pair ("M", M) and the name "M" (_REQUEST_ITEMS). tests/v09/linalg/test_forms.py measures the
# verifier's walk and fails when the layout of contracts.Result changes.
_RESULT_ITEMS = 54
_REQUEST_ITEMS = 5


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(Inertia, (S, D)) for definiteness and (Matrix, mu) for gram_schmidt, args = {"M": M}, computed as the module
    states. InvalidInput for an M that is no Matrix of canonical Rationals, for a definiteness M that is not square or
    not symmetric and for gram_schmidt rows that are linearly dependent; ResourceLimitError past the budget, whose
    max_matrix_entries is charged before anything but the class of M is validated."""
    if type(operation) is not str or operation not in ("definiteness", "gram_schmidt"):
        raise UnsupportedOperation(f"{operation!r} is not an operation of the forms family")
    if type(args) is not dict or set(args) != {"M"}:
        raise InvalidInput(f"{operation} takes exactly the argument M")
    matrix = args["M"]
    with computation() as budget:
        budget.tick()  # The call itself.
        if type(matrix) is not Matrix:  # A subclass, the rows alone or any other object is no Matrix.
            raise InvalidInput(_NOT_A_MATRIX.format(operation))
        rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
        if type(rows) is not tuple or type(ncols) is not int or ncols < 0:
            raise InvalidInput(_NOT_A_MATRIX.format(operation))
        budget.matrix(len(rows), ncols)  # First: a 65 x 65 M exceeds the budget before its shape is judged.
        k = len(rows)
        if operation == "definiteness" and ncols != k:
            raise InvalidInput(f"definiteness requires a square matrix M, not {k} x {ncols}")
        _validate_entries(budget, rows, ncols, operation)
        if operation == "definiteness":
            _require_symmetric(budget, rows)
            value, evidence = _definiteness(budget, rows)
        else:
            if k > ncols:
                raise InvalidInput(f"gram_schmidt requires linearly independent rows: {k} rows in dimension {ncols}")
            value, evidence = _gram_schmidt(budget, rows, ncols)
        _check_items(budget, k, ncols, value, evidence)
        return value, evidence


# ---------------------------------------------------------------------------------------------------------------
# Validation

def _validate_entries(budget: Budget, rows: tuple, ncols: int, operation: str) -> None:
    """InvalidInput for a row that is no tuple of ncols entries and an entry that is no Rational of two exact ints with
    a positive denominator in lowest terms (only a Matrix forged without its constructor holds one);
    ResourceLimitError for an integer longer than max_integer_bits."""
    for row in rows:
        budget.tick(1 + ncols)
        if type(row) is not tuple or len(row) != ncols:
            raise InvalidInput(_NOT_A_MATRIX.format(operation))
        for entry in row:
            if type(entry) is not Rational:  # An int is no Rational here, nor is a bool, a float or a Fraction.
                raise InvalidInput(f"The entries of M are exact Rationals, not {type(entry).__name__}")
            numerator, denominator = getattr(entry, "numerator", None), getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int or denominator < 1:
                raise InvalidInput("The entries of M are Rationals of two exact integers")
            budget.check_int(numerator)
            budget.check_int(denominator)
            if denominator != 1 and _gcd(budget, numerator, denominator) != 1:  # Also 0/2: gcd(0, 2) = 2.
                raise InvalidInput("The entries of M are Rationals in lowest terms")


def _require_symmetric(budget: Budget, rows: tuple) -> None:
    """InvalidInput unless M[i][j] == M[j][i] for every i < j (canonical entries: equal fields)."""
    n = len(rows)
    for i in range(n):
        budget.tick(1 + n)
        for j in range(i + 1, n):
            a, b = rows[i][j], rows[j][i]
            if a.numerator != b.numerator or a.denominator != b.denominator:
                raise InvalidInput(f"definiteness requires a symmetric matrix M: M[{i}][{j}] != M[{j}][{i}]")


# ---------------------------------------------------------------------------------------------------------------
# Integer helpers, each charged before it runs on the cost model of trusted.integers (the checker's measure)

def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: a first remainder, then quadratic in the shorter one."""
    from ...trusted.integers import division_work

    longer, shorter = max(left_bits, right_bits), min(left_bits, right_bits)
    words = shorter >> 6
    return 1 + division_work(longer, shorter) + (words * words >> 6)


def _gcd(budget: Budget, a: int, b: int) -> int:
    """gcd(a, b), charged before it runs."""
    budget.tick(_gcd_work(a.bit_length(), b.bit_length()))
    return gcd(a, b)


def _lcm(budget: Budget, a: int, b: int) -> int:
    """The least common multiple of two positive ints, charged before it runs."""
    from ...trusted.integers import division_work, product_work

    if b == 1 or a == b:
        return a
    if a == 1:
        return b
    a_bits, b_bits = a.bit_length(), b.bit_length()
    budget.tick(2 + division_work(a_bits, b_bits) + product_work(a_bits, b_bits))
    return a // _gcd(budget, a, b) * b


def _lowest(budget: Budget, numerator: int, denominator: int, limit: int) -> Rational:
    """numerator / denominator (denominator != 0) as a Rational in lowest terms, charged before it runs;
    ResourceLimitError when an integer of it is longer than limit bits (the checker answers False to it)."""
    from ...trusted.integers import division_work

    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    if not numerator:
        return trusted_rational(0, 1)
    if denominator != 1:
        divisor = _gcd(budget, numerator, denominator)
        if divisor != 1:
            bits = divisor.bit_length()
            budget.tick(2 + division_work(numerator.bit_length(), bits) + division_work(denominator.bit_length(), bits))
            numerator, denominator = numerator // divisor, denominator // divisor
    if numerator.bit_length() > limit or denominator.bit_length() > limit:
        raise ResourceLimitError(_BITS)
    return trusted_rational(numerator, denominator)


def _widest(values: list[int]) -> int:
    """The longest bit length among some ints (0 for none)."""
    return max(map(int.bit_length, values), default=0)


def _dot(budget: Budget, left: list[int], right: list[int]) -> int:
    """The dot product of two integer vectors of one length, charged before it runs."""
    from ...trusted.integers import product_work

    budget.tick(1 + len(left) + len(left) * product_work(_widest(left), _widest(right)))
    return sum(a * b for a, b in zip(left, right))


def _scaled_rows(budget: Budget, rows: tuple, common: int | None) -> tuple[list[list[int]], list[int]]:
    """The integer rows scale * M[i] and the scale of each: common for every row when it is given, else the least common
    multiple of the denominators of the row."""
    from ...trusted.integers import division_work, product_work

    result, scales = [], []
    for row in rows:
        budget.tick(1 + len(row))
        scale = common
        if scale is None:
            scale = 1
            for entry in row:
                scale = _lcm(budget, scale, entry.denominator)
        scale_bits = scale.bit_length()
        integers = []
        for entry in row:
            if entry.denominator == scale:
                integers.append(entry.numerator)
                continue
            budget.tick(2 + division_work(scale_bits, entry.denominator.bit_length())
                        + product_work(scale_bits, entry.numerator.bit_length()))
            integers.append(entry.numerator * (scale // entry.denominator))
        result.append(integers)
        scales.append(scale)
    return result, scales


# ---------------------------------------------------------------------------------------------------------------
# definiteness

def _definiteness(budget: Budget, rows: tuple) -> tuple[Inertia, tuple]:
    """The value and the evidence (S, D) for a validated symmetric n x n M (module docstring, "How produce()
    computes")."""
    from ...trusted.integers import division_work, product_work

    limit = budget.limits.max_integer_bits
    n = len(rows)
    common = 1
    for row in rows:
        budget.tick(1 + n)
        for entry in row:
            common = _lcm(budget, common, entry.denominator)
    a, _ = _scaled_rows(budget, rows, common)  # The working matrix A, M_int at the start.
    u = [[int(i == j) for j in range(n)] for i in range(n)]  # The lower triangular integer U, U[k][k] = d_k.
    t = [[int(i == j) for j in range(n)] for i in range(n)]  # T: the swaps and additions as column operations.
    budget.tick(1 + 2 * n * n)
    pivots = [0] * n  # d_k d_(k+1) at the position of each step; 0 for the rows that leave the elimination.
    previous = 1  # d_k.
    active = n  # The indices active..n-1 left the elimination: their rows of the remaining block were 0.
    k = 0
    while k < active:
        budget.tick(1 + active - k)
        if not a[k][k]:
            j = next((j for j in range(k + 1, active) if a[j][j]), None)
            if j is not None:
                _swap(budget, a, u, t, k, j, n)
            else:
                j = next((j for j in range(k + 1, active) if a[k][j]), None)
                if j is None:  # Row k of the remaining block is 0: it leaves, its D entry is 0.
                    active -= 1
                    _swap(budget, a, u, t, k, active, n)
                    continue
                _add(budget, a, u, t, k, j, n)  # The pivot becomes 2 A[k][j] != 0 (A[k][k] = A[j][j] = 0).
        pivot = a[k][k]
        pivot_bits, previous_bits = pivot.bit_length(), previous.bit_length()
        pivot_row, pivot_u = a[k], u[k]
        row_bits, u_bits = _widest(pivot_row[k + 1:active]), _widest(pivot_u[:k])
        for i in range(k + 1, active):
            row, u_row = a[i], u[i]
            factor = row[k]
            factor_bits = factor.bit_length()
            entry_bits, entry_u_bits = _widest(row[i:active]), _widest(u_row[:k])
            wide = max(entry_bits + pivot_bits, factor_bits + row_bits) + 1
            wide_u = max(entry_u_bits + pivot_bits, factor_bits + u_bits) + 1
            budget.tick(1 + (active - i) * (3 + product_work(entry_bits, pivot_bits)
                                            + product_work(factor_bits, row_bits) + division_work(wide, previous_bits))
                        + k * (3 + product_work(entry_u_bits, pivot_bits) + product_work(factor_bits, u_bits)
                               + division_work(wide_u, previous_bits)))
            for j in range(i, active):  # Exact (Sylvester's identity); the block stays symmetric.
                entry = (pivot * row[j] - factor * pivot_row[j]) // previous
                row[j] = entry
                a[j][i] = entry
            for c in range(k):
                u_row[c] = (pivot * u_row[c] - factor * pivot_u[c]) // previous
            u_row[k] = -factor  # (p * 0 - factor * U[k][k]) / d_k with U[k][k] = d_k.
            u_row[i] = pivot
        budget.tick(1 + product_work(previous_bits, pivot_bits))
        pivots[k] = previous * pivot
        previous = pivot
        k += 1
    # S = T U^T; T has few nonzero entries in a row (a permutation and a few additions).
    s_rows = []
    budget.tick(1 + n * n)
    u_bits = max((_widest(row) for row in u), default=0)
    for r in range(n):
        terms = [(c, value) for c, value in enumerate(t[r]) if value]
        budget.tick(1 + n + n * len(terms) * (2 + product_work(_widest([v for _, v in terms]), u_bits)))
        entries = []
        for c in range(n):
            u_row = u[c]
            value = sum(factor * u_row[column] for column, factor in terms)
            if value.bit_length() > limit:
                raise ResourceLimitError(_BITS)  # An integer of the evidence that the checker would answer False.
            entries.append(trusted_rational(value, 1))
        s_rows.append(tuple(entries))
    diagonal = tuple(_lowest(budget, pivot, common, limit) for pivot in pivots)
    counts = (sum(1 for x in diagonal if x.numerator > 0), sum(1 for x in diagonal if x.numerator < 0),
              sum(1 for x in diagonal if not x.numerator))
    return Inertia(*counts, _kind(*counts)), (tuple(s_rows), diagonal)


def _swap(budget: Budget, a: list[list[int]], u: list[list[int]], t: list[list[int]], k: int, j: int, n: int) -> None:
    """The symmetric exchange of the indices k < j (both still in the elimination): rows and columns of A, the part of
    the rows of U left of column k (U keeps d_k I on the remaining block), and the columns of T."""
    budget.tick(1 + 3 * n)
    if k == j:
        return
    a[k], a[j] = a[j], a[k]
    for row in a:
        row[k], row[j] = row[j], row[k]
    u[k][:k], u[j][:k] = u[j][:k], u[k][:k]
    for row in t:
        row[k], row[j] = row[j], row[k]


def _add(budget: Budget, a: list[list[int]], u: list[list[int]], t: list[list[int]], k: int, j: int, n: int) -> None:
    """The congruence step "index k += index j" (k < j): row k of A += row j, then column k += column j; the part of
    row k of U left of column k += that of row j (E^T U E^-T keeps d_k I on the remaining block); column k of T +=
    column j."""
    budget.tick(1 + 4 * n + (_widest(a[k]) + _widest(a[j]) + _widest(u[j][:k])) * n // 64)
    row_k, row_j = a[k], a[j]
    for c in range(n):
        row_k[c] += row_j[c]
    for row in a:
        row[k] += row[j]
    u_k, u_j = u[k], u[j]
    for c in range(k):
        u_k[c] += u_j[c]
    for row in t:
        row[k] += row[j]


# ---------------------------------------------------------------------------------------------------------------
# gram_schmidt

def _gram_schmidt(budget: Budget, rows: tuple, ncols: int) -> tuple[Matrix, tuple]:
    """The value Q and the evidence mu for validated k x n rows, k <= n (module docstring, "How produce()
    computes")."""
    from ...trusted.integers import division_work, product_work

    limit = budget.limits.max_integer_bits
    m_int, scales = _scaled_rows(budget, rows, None)
    vectors: list[list[int]] = []  # Q'_j = d_j q'_j.
    determinants = [1]  # d_0, d_1, ...: the Gram determinants of the leading rows of M_int.
    coefficients: list[list[int]] = []  # lambda_ij = M_int[i] . Q'_j.
    for i, row in enumerate(m_int):
        budget.tick(1 + i)
        v = list(row)
        found = []
        for j in range(i):
            other = vectors[j]
            weight = _dot(budget, row, other)
            found.append(weight)
            after, before = determinants[j + 1], determinants[j]
            v_bits, other_bits = _widest(v), _widest(other)
            wide = max(v_bits + after.bit_length(), weight.bit_length() + other_bits) + 1
            budget.tick(1 + ncols * (3 + product_work(v_bits, after.bit_length())
                                     + product_work(weight.bit_length(), other_bits)
                                     + division_work(wide, before.bit_length())))
            v = [(after * x - weight * y) // before for x, y in zip(v, other)]  # Exact (Cramer's rule).
        if not any(v):
            raise InvalidInput(f"gram_schmidt requires linearly independent rows: row {i} lies in the span of the "
                               "rows before it")
        square = _dot(budget, v, v)
        budget.tick(1 + division_work(square.bit_length(), determinants[i].bit_length()))
        determinants.append(square // determinants[i])
        vectors.append(v)
        coefficients.append(found)
    q_rows = []
    for i, v in enumerate(vectors):
        budget.tick(1 + ncols + product_work(determinants[i].bit_length(), scales[i].bit_length()))
        denominator = determinants[i] * scales[i]
        q_rows.append(tuple(_lowest(budget, x, denominator, limit) for x in v))
    mu = []
    for i, found in enumerate(coefficients):
        budget.tick(1 + i)
        entries = []
        for j, weight in enumerate(found):
            budget.tick(2 + product_work(weight.bit_length(), scales[j].bit_length())
                        + product_work(determinants[j + 1].bit_length(), scales[i].bit_length()))
            entries.append(_lowest(budget, weight * scales[j], determinants[j + 1] * scales[i], limit))
        mu.append(tuple(entries))
    return Matrix(tuple(q_rows), ncols=ncols), tuple(mu)


# ---------------------------------------------------------------------------------------------------------------
# The verifier's evidence items

def _check_items(budget: Budget, k: int, ncols: int, value: object, evidence: tuple) -> None:
    """ResourceLimitError when the verifier would not admit the stored Result under max_evidence_items: it counts the
    Result's own items (_RESULT_ITEMS), the request twice, the value twice and the evidence once, as Budget.inspect
    walks them; a separate budget of the same limits counts them here."""
    budget.tick(1 + k)
    counter = Budget(budget.limits)
    request = _REQUEST_ITEMS + 3 + k + 3 * k * ncols  # The Matrix: itself, its rows, k rows, k n Rationals, ncols.
    counter.evidence(_RESULT_ITEMS + 2 * request)
    counter.inspect(value, evidence=True)
    counter.inspect(value, evidence=True)
    counter.inspect(evidence, evidence=True)
    budget.tick(counter.evidence_count >> 4)  # The walk itself.


def _render(inertia: Inertia, renderer, child) -> str:
    r"""positive definite (2, 0, 0): the kind in words, then (positive, negative, zero); in LaTeX
    \text{positive definite}\ \left(2, 0, 0\right)."""
    counts = renderer.parens(renderer.join((child(count) for count in (inertia.positive, inertia.negative,
                                                                         inertia.zero)), ", "))
    return renderer.join((renderer.label(inertia.kind.replace("_", " ")), r"\ " if renderer.latex else " ", counts))


RENDER = {Inertia: _render}
