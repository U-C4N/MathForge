"""Lean value functions of the linear algebra operations: the integer kernel and its certificates.

Each function takes the public value function's arguments, validates them as the structured path would accept them
(an exact Matrix of canonical Rationals, at most max_matrix_entries entries, integers within max_integer_bits) and
otherwise returns FALLBACK, as it does inside an active budget, for a singular inverse, a non-square determinant and
any failure. A value comes back only after a checker of mathforge.certify.linalg accepted the producer's result (or
v0.7's criterion, for trace and lu), and it equals what ``mathforge.operations.<name>(...).unwrap()`` returns: the
producers compute the same mathematical object, which the structured path also returns.

The producers (mathforge.kernels.linalg) and the checkers run in one fresh budget with the default limits, so a
runaway computation stops there and falls back. A matrix whose integer rows would make eliminations too large (the
sum of the row sizes past the integer bound) falls back before any elimination.
"""

from __future__ import annotations

from math import gcd, prod
from operator import attrgetter

from ..certify.linalg.determinant import check_integer_determinant
from ..certify.linalg.inverse import check_inverse
from ..certify.linalg.trace import check_trace
from ..kernels.linalg import determinant, inverse as integer_inverse
from ..limits import ComputationLimits, computation, current_budget
from ..matrices import Matrix, trusted_matrix
from ..model import Rational
from ..trusted.intmatrix import ONE, ZERO, clear_row_denominators, integer, ratio
from . import FALLBACK

_LIMITS = ComputationLimits()
_MAX_ENTRIES = _LIMITS.max_matrix_entries
_MAX_BITS = _LIMITS.max_integer_bits
_numerator = attrgetter("numerator")
_denominator = attrgetter("denominator")


def _entries(A):
    """The rows of A when it is a Matrix of exact Rationals within the default matrix entry limit; else None."""
    if type(A) is not Matrix:
        return None
    rows, ncols = A.rows, A.ncols
    if type(rows) is not tuple or type(ncols) is not int or ncols < 0 or len(rows) * ncols > _MAX_ENTRIES:
        return None
    for row in rows:
        if type(row) is not tuple or len(row) != ncols:
            return None
        for value in row:
            if type(value) is not Rational:
                return None
    return rows


def _integer_rows(rows):
    """(integer rows, row scales) of validated rows when every integer stays within the default integer bound and
    the eliminations stay bounded (the sum of the row sizes within it as well); else None."""
    int_rows, scales = clear_row_denominators(rows)
    total = 0
    for row, scale in zip(int_rows, scales):
        if row:
            bits = max(max(row), -min(row)).bit_length()
            if bits > _MAX_BITS:
                return None
            total += bits
        if scale.bit_length() > _MAX_BITS:
            return None
    if total + len(rows) * len(rows).bit_length() > _MAX_BITS:
        return None
    return int_rows, scales


def trace(A):
    """Sum of the diagonal over a common denominator; checked by v0.7's criterion (certify.linalg.trace). Every entry
    is type-checked; only the diagonal's integers are held to the integer bound (addendum rule 5: an off-diagonal
    entry past it, which the structured path refuses, is a permissive difference only)."""
    try:
        if current_budget() is not None:
            return FALLBACK
        rows = _entries(A)
        if rows is None or len(rows) != A.ncols:
            return FALLBACK
        numerator, denominator = 0, 1
        for index, row in enumerate(rows):
            entry = row[index]
            top, bottom = entry.numerator, entry.denominator
            if top.bit_length() > _MAX_BITS or bottom.bit_length() > _MAX_BITS:
                return FALLBACK
            if bottom == 1:
                numerator += top * denominator
            else:
                common = gcd(denominator, bottom)
                numerator = numerator * (bottom // common) + top * (denominator // common)
                denominator = denominator // common * bottom
        value = ratio(numerator, denominator)
        with computation(fresh=True):
            if check_trace(rows, value) is not True:
                return FALLBACK
        return value
    except Exception:
        return FALLBACK


def det(A):
    """Bareiss on the integer rows (kernels.linalg.determinant); checked by certify.linalg.determinant."""
    try:
        if current_budget() is not None:
            return FALLBACK
        rows = _entries(A)
        if rows is None or len(rows) != A.ncols:
            return FALLBACK
        integers = _integer_rows(rows)
        if integers is None:
            return FALLBACK
        int_rows, scales = integers
        with computation(fresh=True):
            target = determinant(int_rows)
            if check_integer_determinant(int_rows, target) is not True:
                return FALLBACK
        return ratio(target, prod(scales))
    except Exception:
        return FALLBACK


def inverse(A):
    """Fraction-free Gauss-Jordan on [A_int | I] (kernels.linalg.inverse); checked by certify.linalg.inverse. A
    singular matrix falls back, so that the structured path raises its error."""
    try:
        if current_budget() is not None:
            return FALLBACK
        rows = _entries(A)
        n = A.ncols if rows is not None else -1
        if rows is None or len(rows) != n:
            return FALLBACK
        if _integer_rows(rows) is None:
            return FALLBACK
        with computation(fresh=True):
            value = integer_inverse(rows)
            if value is None or check_inverse(rows, value) is not True:
                return FALLBACK
        return trusted_matrix(value, n)
    except Exception:
        return FALLBACK


LEAN = {"trace": trace, "det": det, "inverse": inverse}
