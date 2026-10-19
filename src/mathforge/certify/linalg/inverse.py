"""Independent checker of a matrix inverse: one exact integer product, never an elimination.

With A_int = diag(s) A and B_int = B diag(t), t_j the lcm of the denominators of column j of B, A B = I holds exactly
when A_int B_int = diag(s_i t_i). For square matrices A B = I also gives B A = I, the v0.7 criterion.
"""

from __future__ import annotations

from math import lcm
from operator import attrgetter, mul

from mathforge.limits import current_budget
from mathforge.trusted.intmatrix import clear_row_denominators

from . import square_rows

_denominator = attrgetter("denominator")
_numerator = attrgetter("numerator")


def check_inverse(A, B) -> bool:
    """Whether B is the inverse of the square rational matrix A (each a Matrix or a tuple of row tuples)."""
    rows, inverse_rows = square_rows(A), square_rows(B)
    if rows is None or inverse_rows is None or len(rows) != len(inverse_rows):
        return False
    budget = current_budget()
    int_rows, scales = clear_row_denominators(rows)
    n = len(rows)
    for j, column in enumerate(zip(*inverse_rows)):
        scale = lcm(*map(_denominator, column))
        if scale == 1:
            integers = tuple(map(_numerator, column))
        else:
            integers = tuple([value.numerator * (scale // value.denominator) for value in column])
        for i, row in enumerate(int_rows):
            if sum(map(mul, row, integers)) != (scales[i] * scale if i == j else 0):
                return False
        if budget is not None:
            budget.tick(n * n)
    return True
