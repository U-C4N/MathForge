"""Integer linear algebra producers: fraction-free elimination on A_int = diag(s) A.

Every routine works on integer rows (mathforge.trusted.intmatrix.clear_row_denominators) and divides only exactly:

* ``determinant`` is Bareiss's fraction-free Gaussian elimination;
* ``echelon`` is fraction-free Gauss-Jordan elimination. After r pivots every row is det(M_r) times the matching row of
  the exact Gauss-Jordan state, M_r the r x r minor of the pivot rows and columns, so the nonzero rows of the reduced
  row echelon form are its first r rows over the common denominator det(M_r) (the last pivot);
* ``inverse`` is that elimination on [A_int | I], which ends as det(A_int) [I | A_int^-1].

They charge an active budget one batched tick per row rewrite (its length) and check each pivot against the integer
bound, so runaway work stops; with no active budget they charge nothing. The results are certified by
mathforge.certify.linalg, never by rerunning these routines.
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import mul
from typing import NamedTuple

from ..limits import current_budget
from ..model import Rational
from ..trusted.intmatrix import ONE, ZERO, clear_row_denominators, ratio
from ..trusted.primes import moduli


class Echelon(NamedTuple):
    """The reduced row echelon form of an integer matrix A_int (m x ncols) of rank r, with its certificate.

    The nonzero rows of the RREF are ``numerators[i] / denominator``; ``pivots`` are their pivot columns and
    ``basis_rows`` the rows of A_int, in selection order, whose minor M = A_int[basis_rows][:, pivots] has the
    determinant ``denominator`` (not zero modulo ``modulus``, the first prime of trusted.primes that does not divide
    it). Rank 0 gives empty tuples, denominator 1 and modulus 2."""

    pivots: tuple[int, ...]
    basis_rows: tuple[int, ...]
    numerators: tuple[tuple[int, ...], ...]
    denominator: int
    modulus: int


def determinant(rows: Sequence[Sequence[int]]) -> int:
    """det of a square integer matrix by Bareiss's fraction-free elimination."""
    budget = current_budget()
    n = len(rows)
    if n == 0:
        return 1
    a = [list(row) for row in rows]
    sign, previous = 1, 1
    for k in range(n - 1):
        pivot_index = k
        while not a[pivot_index][k]:
            pivot_index += 1
            if pivot_index == n:
                return 0
        if pivot_index != k:
            a[k], a[pivot_index] = a[pivot_index], a[k]
            sign = -sign
        pivot_row = a[k]
        pivot = pivot_row[k]
        start = k + 1
        tail = pivot_row[start:]
        if budget is not None:
            budget.check_int(pivot)
            budget.tick(1 + (n - start) * (n - start))
        for i in range(start, n):
            row = a[i]
            factor = row[k]
            if factor:
                row[start:] = [(pivot * x - factor * y) // previous for x, y in zip(row[start:], tail)]
            elif pivot != previous:
                row[start:] = [pivot * x // previous for x in row[start:]]
        previous = pivot
    return sign * a[n - 1][n - 1]


def _modulus(denominator: int) -> int:
    """The first prime of trusted.primes that does not divide a nonzero integer."""
    for modulus in moduli():
        if denominator % modulus:
            return modulus
    raise ArithmeticError("no modulus")  # Unreachable: a nonzero integer has finitely many prime divisors.


def echelon(rows: Sequence[Sequence[int]], ncols: int) -> Echelon:
    """The reduced row echelon form of an integer matrix by fraction-free Gauss-Jordan elimination."""
    budget = current_budget()
    m = len(rows)
    a = [list(row) for row in rows]
    order = list(range(m))
    pivots: list[int] = []
    previous, r = 1, 0
    for j in range(ncols):
        if r == m:
            break
        pivot_index = r
        while not a[pivot_index][j]:
            pivot_index += 1
            if pivot_index == m:
                break
        if pivot_index == m:
            continue
        if pivot_index != r:
            a[r], a[pivot_index] = a[pivot_index], a[r]
            order[r], order[pivot_index] = order[pivot_index], order[r]
        pivot_row = a[r]
        pivot = pivot_row[j]
        if budget is not None:
            budget.check_int(pivot)
            budget.tick(1 + (m - 1) * ncols)
        scale = pivot != previous
        for i in range(m):
            if i == r:
                continue
            row = a[i]
            factor = row[j]
            if factor:
                a[i] = [(pivot * x - factor * y) // previous for x, y in zip(row, pivot_row)]
            elif scale and (i < r or any(row)):
                a[i] = [pivot * x // previous for x in row]
        pivots.append(j)
        previous = pivot
        r += 1
    numerators = tuple(tuple(a[i]) for i in range(r))
    if r == 0:
        return Echelon((), (), (), 1, 2)
    return Echelon(tuple(pivots), tuple(order[:r]), numerators, previous, _modulus(previous))


def charpoly(rows: Sequence[Sequence[int]]) -> list[int]:
    """The coefficients of det(t I - A) for a square integer matrix, highest power first (1, c_1, ..., c_n), by
    Berkowitz's division-free algorithm: the Toeplitz products of the trailing principal blocks, as v0.7 computed
    them over the rationals."""
    budget = current_budget()
    n = len(rows)
    coefficients = [1]
    for start in range(n - 1, -1, -1):
        size = n - start
        right = rows[start][start + 1:]
        vector = [rows[i][start] for i in range(start + 1, n)]
        block = [row[start + 1:] for row in rows[start + 1:]]
        sequence = [1, -rows[start][start]]
        for power in range(size - 1):
            sequence.append(-sum(map(mul, right, vector)))
            if power < size - 2:
                vector = [sum(map(mul, row, vector)) for row in block]
        if budget is not None:
            budget.tick(1 + size * size * size)
        count = len(coefficients)
        coefficients = [sum(map(mul, sequence[index::-1][:count], coefficients[:index + 1]))
                        for index in range(size + 1)]
    return coefficients


def lu(rows: Sequence[Sequence[Rational]]) -> tuple[tuple[tuple[Rational, ...], ...], ...]:
    """(P, L, U) with P A = L U for a square rational matrix, by v0.7's elimination: column k takes the first row at or
    below k with a nonzero entry as its pivot (no pivot: the column is skipped), so the factors are v0.7's."""
    budget = current_budget()
    n = len(rows)
    U = [list(row) for row in rows]
    L = [[ONE if i == j else ZERO for j in range(n)] for i in range(n)]
    P = [[ONE if i == j else ZERO for j in range(n)] for i in range(n)]
    for k in range(n):
        candidate = k
        while not U[candidate][k].numerator:
            candidate += 1
            if candidate == n:
                break
        if candidate == n:
            continue
        if candidate != k:
            U[k], U[candidate] = U[candidate], U[k]
            P[k], P[candidate] = P[candidate], P[k]
            L[k][:k], L[candidate][:k] = L[candidate][:k], L[k][:k]
        pivot_row = U[k]
        pivot = pivot_row[k]
        start = k + 1
        tail = pivot_row[start:]
        if budget is not None:
            budget.tick(1 + (n - start) * (n - start))
        for i in range(start, n):
            row = U[i]
            multiplier = row[k] / pivot
            L[i][k] = multiplier
            row[k] = ZERO
            if multiplier.numerator:
                row[start:] = [u - multiplier * v for u, v in zip(row[start:], tail)]
    return (tuple(tuple(row) for row in P), tuple(tuple(row) for row in L), tuple(tuple(row) for row in U))


def inverse(rows: Sequence[Sequence[Rational]]) -> tuple[tuple[Rational, ...], ...] | None:
    """The inverse of a square rational matrix, or None when it is singular."""
    budget = current_budget()
    int_rows, scales = clear_row_denominators(rows)
    n = len(int_rows)
    a = [list(row) + [0] * n for row in int_rows]
    for i in range(n):
        a[i][n + i] = 1
    previous = 1
    width = 2 * n
    for k in range(n):
        pivot_index = k
        while not a[pivot_index][k]:
            pivot_index += 1
            if pivot_index == n:
                return None
        if pivot_index != k:
            a[k], a[pivot_index] = a[pivot_index], a[k]
        pivot_row = a[k]
        pivot = pivot_row[k]
        start = k + 1
        tail = pivot_row[start:]
        if budget is not None:
            budget.check_int(pivot)
            budget.tick(1 + (n - 1) * (width - start))
        scale = pivot != previous
        for i in range(n):
            if i == k:
                continue
            row = a[i]
            factor = row[k]
            if factor:
                row[start:] = [(pivot * x - factor * y) // previous for x, y in zip(row[start:], tail)]
            elif scale:
                row[start:] = [pivot * x // previous for x in row[start:]]
        previous = pivot
    # a = d [I | A_int^-1] with d = previous, and A^-1 = A_int^-1 diag(s).
    return tuple(tuple([ratio(row[n + j] * scales[j], previous) for j in range(n)]) for row in a)
