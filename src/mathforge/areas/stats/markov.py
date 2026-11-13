"""Stationary distributions of finite Markov chains with exact rational transition matrices (mathforge.stats).

stationary_distribution(M) -> tuple[Rational, ...]
    Domain: M is a Matrix (MCP kind "matrix": operations_v6.run already rejects anything but a Matrix as invalid
    input) that is square, n x n with n >= 1, and row stochastic: every entry is a Rational >= 0 and every row sums to
    exactly 1. A non-square M, the 0 x 0 matrix (a chain without states has no distribution), a negative entry and a
    row whose sum is not 1 are invalid input, and so is a chain without a unique stationary distribution: rank(I - M)
    below n - 1, that is more than one closed communicating class (the identity of size 2, for one).
    Value: the unique pi with pi M = pi, sum pi = 1 and pi >= 0, as a tuple of n Rationals in the order of M's states,
    each canonical: exact ints in lowest terms with a positive denominator, an integral one included (Rational(1),
    never an int). [[1]] gives (Rational(1),).
    Evidence: (rows, cols, prime). Let s be the least common multiple of the denominators of M's entries (s = 1 for an
    integer matrix) and K = s (I - M), an integer matrix. rows and cols are tuples of n - 1 exact ints, each strictly
    increasing within 0 .. n - 1: the row and column indices of an (n - 1) x (n - 1) submatrix of K whose determinant
    is nonzero modulo prime, an exact int below 3,317,044,064,679,887,385,961,981 that
    mathforge.certify.ntheory.factorization.prime_proof(prime, ()) proves prime. For n = 1 both index tuples are
    empty (the 0 x 0 determinant is 1) and prime is any such prime, 2 for instance.

The checker, mathforge.certify.stats.markov, verifies that M is stochastic, that pi M = pi, sum pi = 1 and pi >= 0
exactly, proves the prime, and eliminates the submatrix of K modulo the prime: a nonzero determinant makes
rank(I - M) >= n - 1, so the left kernel of I - M is at most one-dimensional and pi is the only stationary
distribution.

How produce() computes. Nothing here is a proof: the checker verifies the value and the rank certificate itself.
- It charges one unit for the call and max_matrix_entries for M (budget.matrix) before it validates anything but the
  class of M and the types of its rows and ncols, so that a 65 x 65 M exceeds the budget whatever its shape; then the
  shape (square, n >= 1) and the entries: exact Rationals of two exact ints, a positive denominator, lowest terms (a
  Matrix forged without its constructor may hold 2/4, whose s, and so the evidence, would depend on how it is
  written; the checker refuses it, so produce() rejects it) and every integer within max_integer_bits.
- t_i, the least common multiple of the denominators of row i, and s, the least common multiple of the t_i, each
  grown one denominator at a time (ResourceLimitError past max_integer_bits, as the checker raises for s); A =
  diag(t) M, an integer matrix; M is stochastic when every entry of A is >= 0 and every row i of A sums to t_i.
- K' = diag(t) (I - M) = diag(t) - A, scaled row by row rather than by s, so that its entries stay as short as the
  rows of M allow (a chain whose rows have different denominators would give K = s (I - M) entries of the length of
  s, and the elimination minors of n times that), and the fraction-free (Bareiss) forward elimination of its
  transpose, rows swapped to find each pivot (not the reduced echelon form of kernels.linalg.echelon, whose
  Gauss-Jordan steps update every row in full and took about twice the time on 64-state chains). Every
  row of I - M sums to 0, so its rank is at most n - 1; a rank below n - 1 means more than one stationary
  distribution, invalid input. With rank n - 1 there is one free column f, and the kernel of K'^T,
  {y : y K' = 0}, is spanned by the integer y with y_f = d, the last pivot, found by exact back substitution
  (Cramer's rule makes every y_p an integer); x = y diag(t) spans the left kernel of I - M. x divided by the gcd of
  its entries and signed to a positive sum is the primitive vector v >= 0 (Perron-Frobenius; a negative entry would
  be a bug, ArithmeticError), and pi_j = v_j / sum v in lowest terms (ResourceLimitError when sum v is longer than
  max_integer_bits: the checker refuses such a value integer).
- The evidence: the minor K'^T[basis_rows][:, pivots] of the elimination (basis_rows the rows that held its
  pivots, in their order) has the determinant d, so K'[pivots, basis_rows] does too, and rows = sorted pivots,
  cols = sorted basis_rows give +-d. K = s (I - M) = diag(s / t_i) K', so
  det K[rows, cols] = +-d times the product of s / t_i over the rows, and prime is the least prime that divides
  neither d nor any of those s / t_i, found by trial division (a prime that does not divide s divides none of them).
  For n = 1, M = [[1]]: pi = (1,) and the evidence ((), (), 2).

Limits. ResourceLimitError where the checker raises it (M past max_matrix_entries, an integer of M or s past
max_integer_bits, work past max_work) and for a value integer past max_integer_bits, which the checker refuses. The
elimination's intermediate minors are bounded by the work they are charged, not by max_integer_bits: the last pivot d
may be longer than the value's integers (it is divided out with the gcd of x), and refusing it would refuse values
that fit.

Work. Every step charges the request's budget before it runs, on the scale of trusted.upoly: a unit per entry,
charge_product and charge_gcd for the long products, quotients and gcds of t, s, A and pi, and one pass over an
integer for each remainder of the prime search. Each elimination step is charged from the actual lengths of its
operands, measured first: a unit per entry update and two products and an exact division of the longest pivot-row
and remaining entries per update; each back-substitution row likewise. Entries of a few dozen bits cost nothing
more: a dense 64-state chain with one-digit weights costs about 130,000 units of the default 2,000,000 and runs in
less than a tenth of a second, and a dense 64-state chain with 36-bit weights, whose value has integers of a few
thousand bits, costs about 1,070,000 (the reduced echelon form that produce() first used was charged 2,230,000).

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/stats/test_markov.py is the definition of done.
"""

from __future__ import annotations

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

# The scope text of the area's v0.9 families (mathforge.areas.stats.bivariate): a transition matrix is one of its
# exact rational matrices.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

# The matrix is M, never A: an argument named A would be copied into Result.expression (operations_v6.source_context).
SPECS = (
    OperationSpec("stationary_distribution", "stats", (Argument("M", "Matrix", None, "matrix"),),
                  "tuple[Rational, ...]",
                  "Find the unique stationary distribution of a finite Markov chain with a rational transition matrix.",
                  SCOPE, version=6),
)

# The value is a tuple of mathforge.model.Rational: the family defines no record class, and the shared renderer
# writes the tuple (so no RENDER either).
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.stats.markov"

_NOT_A_MATRIX = "stationary_distribution requires a square Matrix M of exact rationals"
_BITS = "Computation limit exceeded: integer bits"
_ONE = trusted_rational(1, 1)


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(pi, (rows, cols, prime)) for args = {"M": M}, computed as the module states. InvalidInput for an M that is no
    square row-stochastic Matrix of canonical Rationals with n >= 1, or whose chain has more than one stationary
    distribution; ResourceLimitError past the budget, whose max_matrix_entries is charged before the shape is read."""
    if type(operation) is not str or operation != "stationary_distribution":
        raise UnsupportedOperation(f"{operation!r} is not an operation of the markov family")
    if type(args) is not dict or set(args) != {"M"}:
        raise InvalidInput("stationary_distribution takes exactly the argument M")
    matrix = args["M"]
    with computation() as budget:
        budget.tick()  # The call itself.
        if type(matrix) is not Matrix:  # A subclass, the rows alone or any other object is no Matrix.
            raise InvalidInput(_NOT_A_MATRIX)
        rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
        if type(rows) is not tuple or type(ncols) is not int:
            raise InvalidInput(_NOT_A_MATRIX)
        budget.matrix(len(rows), ncols)  # First: a 65 x 65 M exceeds the budget before its shape is judged.
        n = len(rows)
        if n == 0:
            raise InvalidInput("stationary_distribution requires a chain with at least one state, not a 0 x 0 M")
        if ncols != n:
            raise InvalidInput(f"stationary_distribution requires a square matrix M, not {n} x {ncols}")
        _validate_entries(budget, rows, n)
        limit = budget.limits.max_integer_bits
        scales = [_lcm(budget, [entry.denominator for entry in row], limit) for row in rows]  # t_i, each dividing s.
        s = _lcm(budget, scales, limit)
        scaled = _scaled(budget, rows, scales)  # A = diag(t) M.
        _require_stochastic(budget, scaled, scales)
        if n == 1:  # M = [[1]]: the 0 x 0 minor has the determinant 1, which every prime leaves nonzero.
            return (_ONE,), ((), (), 2)
        return _stationary(budget, scaled, scales, s, n, limit)


def _validate_entries(budget: Budget, rows: tuple, n: int) -> None:
    """InvalidInput for a row that is no tuple of n entries and an entry that is no Rational of two exact ints with a
    positive denominator in lowest terms; ResourceLimitError for an integer longer than max_integer_bits."""
    for row in rows:
        budget.tick(1 + n)
        if type(row) is not tuple or len(row) != n:
            raise InvalidInput(_NOT_A_MATRIX)
        for entry in row:
            if type(entry) is not Rational:  # An int is no entry of a Matrix, nor is a bool or a float.
                raise InvalidInput("The entries of M are exact Rationals")
            numerator, denominator = getattr(entry, "numerator", None), getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int or denominator < 1:
                raise InvalidInput("The entries of M are Rationals of two exact integers")
            budget.check_int(numerator)
            budget.check_int(denominator)
            if denominator != 1 and _gcd(numerator, denominator) != 1:
                raise InvalidInput("The entries of M are Rationals in lowest terms")


def _gcd(a: int, b: int) -> int:
    """gcd(a, b), charged before it runs on the scale of trusted.upoly: the long division of the longer integer by the
    shorter one, then Lehmer's algorithm on the shorter length."""
    from ...trusted import upoly as tp

    longer, shorter = sorted((a.bit_length(), b.bit_length()), reverse=True)
    tp.charge_product(longer, shorter)
    tp.charge_gcd(shorter)
    return gcd(a, b)


def _lcm(budget: Budget, values: list[int], limit: int) -> int:
    """The least common multiple of positive ints, grown one at a time; ResourceLimitError once it is longer than
    limit bits (the checker's refusal for s, which every t_i divides), before a much longer lcm is formed."""
    from ...trusted import upoly as tp

    result = 1
    budget.tick(1 + len(values))
    for value in values:
        if value != 1:
            tp.charge_product(result.bit_length(), value.bit_length(), 3)  # result % v, v // g and the product.
            if result % value:
                result = result // _gcd(result, value) * value
                if result.bit_length() > limit:
                    raise ResourceLimitError(_BITS)
    return result


def _scaled(budget: Budget, rows: tuple, scales: list[int]) -> list[list[int]]:
    """The integer rows of A = diag(t) M: row i times t_i, a multiple of each of its denominators."""
    from ...trusted import upoly as tp

    scaled = []
    for row, scale in zip(rows, scales):
        budget.tick(1 + len(row))
        bits = scale.bit_length()
        values = []
        for entry in row:
            numerator, denominator = entry.numerator, entry.denominator
            tp.charge_product(max(numerator.bit_length(), denominator.bit_length()), bits, 2)
            values.append(numerator * scale if denominator == 1 else numerator * (scale // denominator))
        scaled.append(values)
    return scaled


def _require_stochastic(budget: Budget, scaled: list[list[int]], scales: list[int]) -> None:
    """InvalidInput unless every entry of A is >= 0 and every row i of A sums to t_i: M is row stochastic."""
    from ...trusted import upoly as tp

    for i, (row, scale) in enumerate(zip(scaled, scales)):
        budget.tick(1 + len(row))
        tp.charge(len(row) * (1 + (scale.bit_length() >> 14)))  # The additions: one linear pass per entry.
        if any(entry < 0 for entry in row):
            raise InvalidInput(f"A transition matrix has no negative entry; row {i} of M has one")
        if sum(row) != scale:
            raise InvalidInput(f"Each row of a transition matrix sums to 1; row {i} of M does not")


def _stationary(budget: Budget, scaled: list[list[int]], scales: list[int], s: int, n: int,
                limit: int) -> tuple[tuple, tuple]:
    """The value and the evidence for a validated stochastic n x n chain, n >= 2 (module docstring)."""
    from ...trusted import upoly as tp

    # The transpose of K' = diag(t) (I - M) = diag(t) - A: row j is column j of K'. Its kernel is {y : y K' = 0},
    # and x = y diag(t) runs over the left kernel of I - M.
    transposed = []
    for j in range(n):
        budget.tick(1 + n)
        transposed.append([(scales[i] if i == j else 0) - scaled[i][j] for i in range(n)])
    upper, pivots, order, d = _forward_elimination(budget, transposed, n)
    rank = len(pivots)
    if rank < n - 1:
        raise InvalidInput("The chain has no unique stationary distribution: rank(I - M) is below n - 1 (more than "
                           "one closed communicating class)")
    if rank != n - 1:  # Every row of I - M sums to 0, so its rank is at most n - 1: a full rank is a bug.
        raise ArithmeticError("rank(I - M) = n for a stochastic M")
    free = next(j for j in range(n) if j not in pivots)
    vector = _back_substitution(budget, upper, pivots, free, d, n)  # y, with y_free = d.
    basis_rows = order[:rank]
    budget.tick(n)
    for i in range(n):  # x_i = y_i t_i.
        tp.charge_product(vector[i].bit_length(), scales[i].bit_length())
        vector[i] *= scales[i]
    vector = _primitive_nonnegative(budget, vector)
    total = sum(vector)
    if total.bit_length() > limit:
        raise ResourceLimitError(_BITS)  # The denominator of pi in lowest terms is sum v (the checker's docstring).
    value = []
    for entry in vector:
        budget.tick()
        if entry == 0:
            value.append(trusted_rational(0, 1))
            continue
        divisor = _gcd(entry, total)
        tp.charge_product(total.bit_length(), divisor.bit_length(), 2)
        value.append(trusted_rational(entry // divisor, total // divisor))
    rows = tuple(sorted(pivots))  # The rows of the minor of K' (and of K) ...
    cols = tuple(sorted(basis_rows))  # ... and its columns; sorting changes only the sign of d.
    # K = s (I - M) = diag(s / t_i) K', so det K[rows, cols] = +-d times the product of s / t_i over the rows.
    factors = []
    for i in rows:
        tp.charge_product(s.bit_length(), scales[i].bit_length())
        factors.append(s // scales[i])
    return tuple(value), (rows, cols, _least_prime_not_dividing(budget, d, s, factors))


def _primitive_nonnegative(budget: Budget, vector: list[int]) -> list[int]:
    """The vector divided by the gcd of its entries and signed to a positive sum: the primitive v >= 0 of the kernel
    line (Perron-Frobenius; ArithmeticError for a negative entry, which would be a bug)."""
    from ...trusted import upoly as tp

    widest = max(entry.bit_length() for entry in vector)
    budget.tick(len(vector))
    tp.charge_gcd(widest, len(vector))
    common = 0
    for entry in vector:
        common = gcd(common, entry)
    tp.charge_product(widest, common.bit_length(), len(vector))
    vector = [entry // common for entry in vector]
    if sum(vector) < 0:
        vector = [-entry for entry in vector]
    if any(entry < 0 for entry in vector) or not any(vector):
        raise ArithmeticError("the left kernel of I - M holds no nonnegative vector")
    return vector


def _bits(row: list[int]) -> int:
    """The bit length of the longest entry of a nonempty row of ints."""
    return max(max(row), -min(row)).bit_length()


def _forward_elimination(budget: Budget, matrix: list[list[int]],
                         n: int) -> tuple[list[list[int]], list[int], list[int], int]:
    """(upper, pivots, order, d): fraction-free (Bareiss) forward elimination of the n x n integer matrix, rows
    swapped to find each pivot and a column without one skipped. Row k of upper has its pivot at column pivots[k],
    order lists the rows of the matrix in their final places, and d, the last pivot (1 for rank 0), is the
    determinant of the minor matrix[order[:rank]][:, pivots]: after k pivots, every entry of the rows below is a
    minor of order k + 1 (Sylvester's identity), which makes each division by the previous pivot exact.

    Forward elimination updates only the rows below a pivot and the columns to its right, about n**3 / 3 entry
    updates in all where Gauss-Jordan elimination updates every row in full, n**3, and its longest entries arise in
    the last steps, where few rows remain: on dense 64-state chains it takes about half the time of the reduced
    echelon form.
    Each step is charged before it runs from the actual lengths of its operands: a unit per entry update, and two
    products and an exact long division of the longest pivot-row and remaining entries on the scale of
    trusted.integers (product_work, and division_work, which counts CPython's quadratic long division and is several
    times a product's: charging it as a product would undercharge long entries about fourfold in time)."""
    from ...trusted.integers import division_work, product_work

    a = [list(row) for row in matrix]
    order = list(range(n))
    pivots: list[int] = []
    previous, r = 1, 0
    for j in range(n):
        if r == n:
            break
        budget.tick(1 + n - r)
        pivot_index = r
        while pivot_index < n and not a[pivot_index][j]:
            pivot_index += 1
        if pivot_index == n:  # No pivot in this column: the rows below keep a zero there in every later step.
            continue
        if pivot_index != r:
            a[r], a[pivot_index] = a[pivot_index], a[r]
            order[r], order[pivot_index] = order[pivot_index], order[r]
        pivot_row = a[r]
        pivot = pivot_row[j]
        below, width = n - r - 1, n - j - 1
        if below and width:
            budget.tick(1 + below * (width + 2))  # The entry updates, and the passes that measure them.
            widest = max(_bits(pivot_row[j:]), max(_bits(a[i][j:]) for i in range(r + 1, n)))
            update = 2 * product_work(widest, widest) + division_work(2 * widest + 1, previous.bit_length())
            if update:
                budget.tick(below * width * update)
            tail = pivot_row[j + 1:]
            for i in range(r + 1, n):
                row = a[i]
                factor = row[j]
                if factor:
                    row[j + 1:] = [(pivot * x - factor * y) // previous for x, y in zip(row[j + 1:], tail)]
                    row[j] = 0
                else:  # Scaled all the same, so that every entry stays the minor that the next division needs.
                    row[j + 1:] = [pivot * x // previous for x in row[j + 1:]]
        elif below:  # The last column: the entries below the pivot become 0 and nothing else changes.
            budget.tick(below)
            for i in range(r + 1, n):
                a[i][j] = 0
        pivots.append(j)
        previous = pivot
        r += 1
    return a[:r], pivots, order, previous


def _back_substitution(budget: Budget, upper: list[list[int]], pivots: list[int], free: int, d: int,
                       n: int) -> list[int]:
    """The integer y with y_free = d and upper y = 0, for the rank n - 1 forward elimination of a matrix B: every
    row of upper is a rational combination of rows of the minor B[order[:n - 1]], and y_free = d = det of that minor
    with its pivot columns makes y_pivots = -adj(minor) B[order[:n - 1], free] an integer vector (Cramer's rule), so
    each division below is exact (ArithmeticError otherwise, which would be a bug). Each row is charged before it
    runs: a unit per product, and the products of its longest entries and the long division (product_work and
    division_work of trusted.integers)."""
    from ...trusted.integers import division_work, product_work

    y = [0] * n
    y[free] = d
    widest_y = d.bit_length()
    for k in range(len(upper) - 1, -1, -1):
        row = upper[k]
        p = pivots[k]
        count = n - p - 1
        budget.tick(2 + count)
        row_bits = _bits(row[p:])
        sum_bits = row_bits + widest_y + n.bit_length()
        budget.tick(count * product_work(row_bits, widest_y) + division_work(sum_bits, row[p].bit_length()))
        total = sum(row[c] * y[c] for c in range(p + 1, n) if row[c])
        quotient, remainder = divmod(-total, row[p])
        if remainder:
            raise ArithmeticError("an inexact back substitution in the kernel of I - M")
        y[p] = quotient
        widest_y = max(widest_y, quotient.bit_length())
    return y


def _least_prime_not_dividing(budget: Budget, d: int, s: int, factors: list[int]) -> int:
    """The least prime that divides neither the nonzero d nor any of the factors s / t_i, by trial division (a prime
    that does not divide s divides none of them). Every remainder is charged a pass over its integer. The primes that
    divide d or s have a product of at most |d| s, so a few thousand candidates at most when both are within
    max_integer_bits, and usually the first few."""
    pass_d, pass_s = 1 + (d.bit_length() >> 14), 1 + (s.bit_length() >> 14)
    primes: list[int] = []
    candidate = 2
    while True:
        budget.tick(1 + len(primes) // 64)
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
            budget.tick(pass_d + pass_s)
            if d % candidate and (s % candidate or not _divides_any(budget, candidate, factors)):
                return candidate
        candidate += 1


def _divides_any(budget: Budget, prime: int, factors: list[int]) -> bool:
    """Whether the prime divides one of the factors, each remainder charged a pass over its factor."""
    for factor in factors:
        budget.tick(1 + (factor.bit_length() >> 14))
        if factor % prime == 0:
            return True
    return False
