"""Independent checker of a matrix inverse: one exact integer product, never an elimination.

With A_int = diag(s) A and B_int = B diag(t), t_j the lcm of the denominators of column j of B, A B = I holds exactly
when A_int B_int = diag(s_i t_i). For square matrices A B = I also gives B A = I, the v0.7 criterion. A matrix A with
a row whose denominators' lcm passes max_integer_bits has no integer form within the bound: A B = I is then checked
over Q, entry by entry, as v0.7's product did.
"""

from __future__ import annotations

from operator import mul

from mathforge.limits import ResourceLimitError, current_budget
from mathforge.trusted.intmatrix import ZERO, clear_row_denominators

from . import square_rows


def _rational_identity(rows, inverse_rows) -> bool:
    """Whether A B = I over Q, each entry summed as v0.7's matrix product summed it."""
    n = len(rows)
    for i, row in enumerate(rows):
        for j in range(n):
            total = ZERO
            for k in range(n):
                total = total + row[k] * inverse_rows[k][j]
            if total.denominator != 1 or total.numerator != (1 if i == j else 0):
                return False
    return True


def check_inverse(A, B) -> bool:
    """Whether B is the inverse of the square rational matrix A (each a Matrix or a tuple of row tuples)."""
    rows, inverse_rows = square_rows(A), square_rows(B)
    if rows is None or inverse_rows is None or len(rows) != len(inverse_rows):
        return False
    budget = current_budget()
    try:
        int_rows, scales = clear_row_denominators(rows)
        columns, column_scales = clear_row_denominators(tuple(zip(*inverse_rows)))  # B_int's columns and the t_j.
    except ResourceLimitError:  # A scale past the integer bound: the product over Q decides.
        return _rational_identity(rows, inverse_rows)
    n = len(rows)
    for j, (integers, scale) in enumerate(zip(columns, column_scales)):
        for i, row in enumerate(int_rows):
            if sum(map(mul, row, integers)) != (scales[i] * scale if i == j else 0):
                return False
        if budget is not None:
            budget.tick(n * n)
    return True
