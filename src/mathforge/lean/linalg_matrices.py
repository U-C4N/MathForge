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

from math import gcd, lcm, prod

from ..certify.linalg.charpoly import check_charpoly
from ..certify.linalg.determinant import check_integer_determinant
from ..certify.linalg.echelon import check_echelon
from ..certify.linalg.inverse import check_inverse
from ..certify.linalg.lu import check_lu
from ..certify.linalg.trace import check_trace
from ..kernels.linalg import (Echelon, charpoly as integer_charpoly, determinant, echelon as integer_echelon,
                              inverse as integer_inverse, lu as plu)
from ..limits import ComputationLimits, ResourceLimitError, computation, current_budget
from ..matrices import Matrix, trusted_matrix, trusted_rref
from ..model import Rational, Reals, Symbol
from ..polynomial import Polynomial
from ..trusted.intmatrix import ONE, ZERO, clear_row_denominators, ratio
from ..trusted.primes import MODULI
from . import FALLBACK

_LIMITS = ComputationLimits()
_MAX_ENTRIES = _LIMITS.max_matrix_entries
_MAX_BITS = _LIMITS.max_integer_bits


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
    the eliminations stay bounded (the sum of the row sizes within it as well); else None. A row scale past the bound
    is refused while it is formed (trusted.intmatrix.clear_row_denominators)."""
    try:
        int_rows, scales = clear_row_denominators(rows)
    except ResourceLimitError:
        return None
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
    entry past it, which the structured path refuses, is a permissive difference only). A running denominator past
    the bound is reduced with its numerator, and the sum falls back when it stays past it: v0.7's partial sums, the
    same Rationals in lowest terms, would have passed the bound there as well."""
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
                if denominator.bit_length() > _MAX_BITS:
                    common = gcd(numerator, denominator)
                    numerator, denominator = numerator // common, denominator // common
                    if denominator.bit_length() > _MAX_BITS:
                        return FALLBACK
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


def _certified_echelon(A):
    """(rows, ncols, certified Echelon) of a validated matrix, else None. A square matrix is first claimed to be
    nonsingular (R = I, numerators None), which the certificate's minor test alone settles; otherwise, and when that
    claim fails, kernels.linalg.echelon produces the certificate. certify.linalg.echelon checks it either way."""
    rows = _entries(A)
    if rows is None:
        return None
    integers = _integer_rows(rows)
    if integers is None:
        return None
    int_rows = integers[0]
    ncols = A.ncols
    with computation(fresh=True):
        if ncols and len(rows) == ncols:
            full = tuple(range(ncols))
            if check_echelon(int_rows, ncols, None, 1, full, full, MODULI[0]) is True:
                return rows, ncols, Echelon(full, full, None, 1, MODULI[0])
        result = integer_echelon(int_rows, ncols)
        if check_echelon(int_rows, ncols, result.numerators, result.denominator, result.pivots,
                         result.basis_rows, result.modulus) is not True:
            return None
    return rows, ncols, result


def _reduced_rows(certificate, ncols):
    """The nonzero rows of the RREF as canonical Rationals."""
    if certificate.numerators is None:
        return tuple(tuple(ONE if i == j else ZERO for j in range(ncols)) for i in range(ncols))
    denominator = certificate.denominator
    return tuple(tuple([ratio(value, denominator) for value in row]) for row in certificate.numerators)


def _lean(function):
    """A lean function: FALLBACK inside an active budget, for a None answer and for any exception."""
    def lean(A):
        try:
            if current_budget() is not None:
                return FALLBACK
            value = function(A)
            return FALLBACK if value is None else value
        except Exception:
            return FALLBACK
    lean.__name__ = lean.__qualname__ = function.__name__.lstrip("_")
    lean.__doc__ = function.__doc__
    return lean


@_lean
def _rref(A):
    """The reduced row echelon form with its zero rows, and the pivot columns: certified by the echelon checker."""
    certified = _certified_echelon(A)
    if certified is None:
        return None
    rows, ncols, certificate = certified
    reduced = _reduced_rows(certificate, ncols)
    zero = (ZERO,) * ncols
    matrix = trusted_matrix(reduced + (zero,) * (len(rows) - len(reduced)), ncols)
    return trusted_rref(matrix, certificate.pivots)


@_lean
def _rank(A):
    """The rank: the number of pivots of the certified echelon form."""
    certified = _certified_echelon(A)
    return None if certified is None else len(certified[2].pivots)


@_lean
def _nullspace(A):
    """The canonical basis read from the certified RREF: for each free column f, 1 at f, 0 at the other free columns
    and -R_i[f] at pivot i (A v = 0 by the certificate's C2)."""
    certified = _certified_echelon(A)
    if certified is None:
        return None
    _, ncols, certificate = certified
    pivots = certificate.pivots
    if certificate.numerators is None:
        return ()
    pivot_set = set(pivots)
    numerators, denominator = certificate.numerators, certificate.denominator
    basis = []
    for free in range(ncols):
        if free in pivot_set:
            continue
        vector = [ZERO] * ncols
        vector[free] = ONE
        for row, pivot in zip(numerators, pivots):
            vector[pivot] = ratio(-row[free], denominator)
        basis.append(tuple(vector))
    return tuple(basis)


@_lean
def _rowspace(A):
    """The nonzero rows of the certified RREF."""
    certified = _certified_echelon(A)
    return None if certified is None else _reduced_rows(certified[2], certified[1])


@_lean
def _columnspace(A):
    """The columns of A at the certified pivot columns."""
    certified = _certified_echelon(A)
    if certified is None:
        return None
    rows, _, certificate = certified
    return tuple(tuple([row[column] for row in rows]) for column in certificate.pivots)


@_lean
def _lu(A):
    """v0.7's PLU elimination (kernels.linalg.lu), checked by v0.7's criterion in integers (certify.linalg.lu)."""
    rows = _entries(A)
    if rows is None or len(rows) != A.ncols or _integer_rows(rows) is None:
        return None
    from ..linear_extensions import trusted_lu
    n = A.ncols
    with computation(fresh=True):
        P, L, U = plu(rows)
        if check_lu(rows, P, L, U) is not True:
            return None
    return trusted_lu(trusted_matrix(P, n), trusted_matrix(L, n), trusted_matrix(U, n))


def _characteristic(A, variable):
    """det(t I - A) as v0.7's Polynomial in variable, or None: Berkowitz on the integer matrix d A (d the lcm of all
    denominators; kernels.linalg.charpoly), checked by certify.linalg.charpoly, then c_k(A) = c_k(d A) / d**k."""
    if type(variable) is not Symbol or variable.domain is not Reals or variable.assumptions:
        return None
    rows = _entries(A)
    if rows is None or len(rows) != A.ncols:
        return None
    integers = _integer_rows(rows)
    if integers is None:
        return None
    int_rows, scales = integers
    common = 1
    for scale in scales:  # The lcm of the row scales, refused as soon as it passes the integer bound.
        common = lcm(common, scale)
        if common.bit_length() > _MAX_BITS:
            return None
    if common != 1:
        int_rows = tuple(tuple([value * (common // scale) for value in row]) for row, scale in zip(int_rows, scales))
        if _integer_rows_fit(int_rows) is False:
            return None
    n = len(rows)
    with computation(fresh=True):
        coefficients = integer_charpoly(int_rows)
        if check_charpoly(int_rows, coefficients) is not True:
            return None
    return Polynomial(tuple(ratio(coefficients[n - power], common ** (n - power)) for power in range(n + 1)),
                      variable)


def _integer_rows_fit(int_rows):
    """Whether rescaled integer rows still keep the eliminations within the integer bound."""
    total = 0
    for row in int_rows:
        if row:
            bits = max(max(row), -min(row)).bit_length()
            if bits > _MAX_BITS:
                return False
            total += bits
    return total + len(int_rows) * len(int_rows).bit_length() <= _MAX_BITS


def charpoly(A, variable):
    """det(t I - A): Berkowitz over the integers, checked modulo primes through a Hessenberg form."""
    try:
        if current_budget() is not None:
            return FALLBACK
        value = _characteristic(A, variable)
        return FALLBACK if value is None else value
    except Exception:
        return FALLBACK


def real_eigenvalues(A):
    """The real roots of the certified characteristic polynomial in v0.7's root variable, through the real_roots
    lean function (whose own checker certifies the roots), as v0.7's real_eigenvalues isolates them."""
    try:
        if current_budget() is not None:
            return FALLBACK
        from ..root_isolation import ROOT_VARIABLE
        polynomial = _characteristic(A, ROOT_VARIABLE)
        if polynomial is None:
            return FALLBACK
        from . import functions
        return functions.real_roots(polynomial, ROOT_VARIABLE)
    except Exception:
        return FALLBACK


LEAN = {"trace": trace, "det": det, "inverse": inverse, "rref": _rref, "rank": _rank, "nullspace": _nullspace,
        "rowspace": _rowspace, "columnspace": _columnspace, "lu": _lu, "charpoly": charpoly,
        "real_eigenvalues": real_eigenvalues}
