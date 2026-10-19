"""Lean functions of the legacy univariate polynomial operations of the algebra area.

Each takes canonical Polynomial operands (lean._poly.coefficients) the way the structured path takes them, runs a
producer of mathforge.kernels.upoly in a fresh default budget, and returns the value only when a checker of
mathforge.certify.upoly accepts it; anything else gives FALLBACK, and the structured path answers.

* poly_divmod: long division over Q (integer pseudo-division, one reduction at the end); checked by the identity
  f = q g + r with deg r < deg g, decided by one Kronecker evaluation.
"""

from __future__ import annotations

from ..certify import upoly as certify
from ..kernels import upoly as kernels
from ..limits import computation, current_budget
from ..polynomial import trusted_polynomial
from . import FALLBACK
from ._poly import coefficients, same_variable


def _operands(f, g):
    """The coefficient tuples of two canonical Polynomials in the same variable, or None."""
    if current_budget() is not None:
        return None
    first, second = coefficients(f), coefficients(g)
    if first is None or second is None or not same_variable(f.variable, g.variable):
        return None
    return first, second


def poly_divmod(f: object, g: object) -> object:
    operands = _operands(f, g)
    if operands is None:
        return FALLBACK
    first, second = operands
    if len(second) == 1 and not second[0].numerator:
        return FALLBACK  # The structured path reports the zero divisor.
    try:
        with computation(fresh=True):
            quotient, remainder = kernels.divmod_rational(first, second)
            if certify.check_division(first, second, quotient, remainder) is not True:
                return FALLBACK
    except Exception:
        return FALLBACK
    variable = f.variable
    if remainder is first:
        return trusted_polynomial(quotient, variable), f
    return trusted_polynomial(quotient, variable), trusted_polynomial(remainder, variable)


LEAN = {"poly_divmod": poly_divmod}
