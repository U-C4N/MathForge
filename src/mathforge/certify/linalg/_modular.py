"""Gaussian elimination modulo a modulus: the checkers' own determinant and rank tests, never the producers' code.

Elimination is lazy: the rows below the pivot keep unreduced integers, and only the pivot row and each multiplier
are reduced when used, so a row grows by at most modulus**2 per step and no step divides. A row whose entry in the
pivot column is zero modulo the modulus is skipped, so sparse matrices cost little. The modulus may be composite (a
product of primes): every pivot must then be a unit, and a pivot that is not ends the elimination without a
conclusion (None), never with a wrong one. An active budget is charged one batched tick per pivot step.
"""

from __future__ import annotations

from mathforge.limits import current_budget


def det_mod(int_rows, modulus):
    """det of a square integer matrix modulo the modulus, in [0, modulus); None when a pivot is not a unit."""
    budget = current_budget()
    n = len(int_rows)
    rows = [list(row) for row in int_rows]
    det = 1
    for k in range(n):
        pivot_index = k
        while not rows[pivot_index][k] % modulus:
            pivot_index += 1
            if pivot_index == n:
                return 0
        if pivot_index != k:
            rows[k], rows[pivot_index] = rows[pivot_index], rows[k]
            det = -det
        pivot_row = rows[k]
        pivot = pivot_row[k] % modulus
        det = det * pivot % modulus
        start = k + 1
        if start == n:
            break
        try:
            inverse = pow(pivot, -1, modulus)
        except ValueError:
            return None
        tail = [value % modulus for value in pivot_row[start:]]
        updated = 0
        for i in range(start, n):
            row = rows[i]
            factor = row[k] % modulus
            if factor:
                factor = factor * inverse % modulus
                row[start:] = [a - factor * b for a, b in zip(row[start:], tail)]
                updated += 1
        if budget is not None:
            budget.tick(1 + updated * (n - start))
    return det % modulus


def nonsingular_mod(int_rows, modulus):
    """Whether a square integer matrix has a determinant that is a unit modulo the modulus (>= 2), so not zero: the
    elimination answers False at a column without a pivot and at a pivot that is not a unit, so True is sound for any
    modulus."""
    if type(modulus) is not int or modulus < 2:
        return False
    determinant = det_mod(int_rows, modulus)
    if not determinant:  # Zero or no conclusion.
        return False
    try:
        pow(determinant, -1, modulus)
    except ValueError:
        return False
    return True
