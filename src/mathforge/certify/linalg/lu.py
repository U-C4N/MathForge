"""Independent checker of a PLU factorization P A = L U, by v0.7's criterion in exact integer arithmetic.

P must be a permutation matrix (entries 0 and 1, one 1 in every row and column; P A has row pi(i) of A as its row i),
L unit lower triangular and U upper triangular. The product is compared in integers: with A_int = diag(s) A,
L_int = diag(l) L (l_i the lcm of the denominators of row i of L) and U_int = U diag(u) (u_j that of column j of U),
(L U)[i][j] = A[pi(i)][j] exactly when s_pi(i) (L_int U_int)[i][j] = l_i u_j A_int[pi(i)][j].
"""

from __future__ import annotations

from math import lcm
from operator import attrgetter, mul

from mathforge.limits import current_budget
from mathforge.trusted.intmatrix import clear_row_denominators

from . import square_rows

_denominator = attrgetter("denominator")
_numerator = attrgetter("numerator")


def _permutation(rows):
    """pi with rows[i][pi[i]] = 1 and every other entry 0, when rows is a permutation matrix; else None."""
    n = len(rows)
    positions, seen = [], set()
    for row in rows:
        position = None
        for index, value in enumerate(row):
            if value.numerator:
                if position is not None or value.numerator != 1 or value.denominator != 1:
                    return None
                position = index
        if position is None or position in seen:
            return None
        seen.add(position)
        positions.append(position)
    return positions if len(positions) == n else None


def check_lu(A, P, L, U) -> bool:
    """Whether P, L and U (each a Matrix or a tuple of row tuples) factor the square rational matrix A as P A = L U."""
    rows, p_rows, l_rows, u_rows = square_rows(A), square_rows(P), square_rows(L), square_rows(U)
    if rows is None or p_rows is None or l_rows is None or u_rows is None:
        return False
    n = len(rows)
    if not len(p_rows) == len(l_rows) == len(u_rows) == n:
        return False
    permutation = _permutation(p_rows)
    if permutation is None:
        return False
    for i in range(n):
        l_row, u_row = l_rows[i], u_rows[i]
        diagonal = l_row[i]
        if diagonal.numerator != 1 or diagonal.denominator != 1:
            return False
        if any(value.numerator for value in l_row[i + 1:]) or any(value.numerator for value in u_row[:i]):
            return False
    budget = current_budget()
    int_rows, scales = clear_row_denominators(rows)
    l_int, l_scales = clear_row_denominators(l_rows)
    u_columns = []
    u_scales = []
    for column in zip(*u_rows):
        scale = lcm(*map(_denominator, column))
        u_scales.append(scale)
        if scale == 1:
            u_columns.append(tuple(map(_numerator, column)))
        else:
            u_columns.append(tuple([value.numerator * (scale // value.denominator) for value in column]))
    for i in range(n):
        source = permutation[i]
        left, row, s_source, l_scale = l_int[i], int_rows[source], scales[source], l_scales[i]
        for j in range(n):
            if s_source * sum(map(mul, left, u_columns[j])) != l_scale * u_scales[j] * row[j]:
                return False
        if budget is not None:
            budget.tick(1 + n * n)
    return True
