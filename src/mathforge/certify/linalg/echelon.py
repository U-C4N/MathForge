"""Independent checker of a reduced row echelon form: the echelon certificate, never a rerun elimination.

The claim about an integer matrix A_int (m x n; A_int = diag(s) A has the RREF of A) is a list of pivot columns
p_0 < ... < p_{r-1}, rows R_i = numerators[i] / denominator and basis rows b_0, ..., b_{r-1} with a modulus. The
checker establishes, in exact integer arithmetic:

* C1. R is in reduced row echelon form with pivot columns p: R_i is zero before p_i, R_i[p_i] = 1, and every other
  row of R is zero in column p_i.
* C2. For every row k of A_int and every non-pivot column j, A_int[k][j] = sum_i A_int[k][p_i] R_i[j]. With C1 this is
  A_int = A_int[:, p] R, so the row space of A_int lies in that of R and rank A_int <= r.
* C3. The r x r minor M = A_int[b][:, p] has a determinant that is a unit modulo the modulus (lazy elimination modulo
  it, which gives no conclusion at a pivot that is not a unit, so this is sound for any modulus >= 2). Hence
  rank A_int >= r.

So the row space of A_int is that of R, and R is its reduced row echelon form (which is unique). Then the canonical
nullspace basis read from R (1 at a free column, 0 at the others, -R_i[f] at p_i) satisfies A v = 0 by C2, the pivot
columns of A span its column space, and the rank is r. ``numerators=None`` stands for R = I with every column a pivot
(C1 holds and C2 is empty), which C3 alone then certifies.
"""

from __future__ import annotations

from operator import mul

from mathforge.limits import current_budget

from ._modular import nonsingular_mod

_INT = {int}


def check_echelon(int_rows, ncols, numerators, denominator, pivots, basis_rows, modulus) -> bool:
    """Whether numerators / denominator are the nonzero rows of the RREF of int_rows (a tuple of ncols-long int
    tuples), with pivot columns ``pivots``, certified by the minor on ``basis_rows`` modulo ``modulus``."""
    m = len(int_rows)
    if (type(pivots) is not tuple or type(basis_rows) is not tuple or type(denominator) is not int
            or not denominator or type(modulus) is not int or modulus < 2 or type(ncols) is not int):
        return False
    r = len(pivots)
    if len(basis_rows) != r or r > m:
        return False
    previous = -1
    for column in pivots:
        if type(column) is not int or column <= previous or column >= ncols:
            return False
        previous = column
    if len(set(basis_rows)) != r or any(type(row) is not int or not 0 <= row < m for row in basis_rows):
        return False
    budget = current_budget()
    if numerators is None:
        if pivots != tuple(range(ncols)) or denominator != 1:
            return False
    else:
        # C1
        if type(numerators) is not tuple or len(numerators) != r:
            return False
        for row, column in zip(numerators, pivots):
            if (type(row) is not tuple or len(row) != ncols or set(map(type, row)) - _INT
                    or row[column] != denominator or any(row[:column])):
                return False
        for index, column in enumerate(pivots):
            for other, row in enumerate(numerators):
                if other != index and row[column]:
                    return False
        # C2
        pivot_set = set(pivots)
        free = [column for column in range(ncols) if column not in pivot_set]
        if free:
            columns = [tuple([row[column] for row in numerators]) for column in free]
            for row in int_rows:
                coefficients = [row[column] for column in pivots]
                for column, combination in zip(free, columns):
                    if sum(map(mul, coefficients, combination)) != denominator * row[column]:
                        return False
            if budget is not None:
                budget.tick(1 + m * len(free) * r)
    # C3
    minor = [tuple([int_rows[row][column] for column in pivots]) for row in basis_rows]
    return nonsingular_mod(minor, modulus)
