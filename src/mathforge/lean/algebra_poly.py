"""Lean functions of the legacy univariate polynomial operations of the algebra area.

Each takes canonical Polynomial operands (lean._poly.coefficients) the way the structured path takes them, runs a
producer of mathforge.kernels.upoly in a fresh default budget, and returns the value only when a checker of
mathforge.certify.upoly accepts it; anything else gives FALLBACK, and the structured path answers.

* poly_divmod: long division over Q (integer pseudo-division, one reduction at the end); checked by the identity
  f = q g + r with deg r < deg g, decided by one Kronecker evaluation.
* poly_gcd: the heuristic gcd at a power of two (Euclid when it does not settle) of the primitive integer forms;
  checked through the cofactors, f = h u and g = h v, and u, v coprime modulo a prime.
* poly_xgcd: the extended Euclidean algorithm over Q; checked by s f + t g = h, h dividing f and g, and the cofactor
  degrees that make (s, t) unique, so that the value is v0.7's.
* square_free: v0.7's square-free algorithm over Z with the gcd above; checked as v0.7 checks it: monic square-free
  pairwise coprime factors (the product of the factors square-free modulo a prime) and the product identity.
* rational_function: the reduced fraction from the gcd's cofactors and the exclusion from the square-free part;
  checked by the cross products and coprimality, and the square-free part through its gcd's cofactors.
* resultant, discriminant (v4): v0.7's subresultant PRS on the primitive integer forms; checked by the Euclidean
  algorithm modulo a prime larger than twice the Hadamard bound.
"""

from __future__ import annotations

from ..certify import upoly as certify
from ..kernels import upoly as kernels
from ..limits import computation, current_budget
from ..model import Rational, trusted_rational
from ..polynomial import Polynomial, trusted_polynomial, trusted_square_free
from ..rational_function import RationalFunction
from ..trusted import upoly as tp
from . import FALLBACK, run_legacy
from ._poly import coefficients, real_variable, same_variable, small_int

_ZERO = (trusted_rational(0, 1),)


def _operands(f, g):
    """The coefficient tuples of two canonical Polynomials in the same variable, or None."""
    if current_budget() is not None:
        return None
    first, second = coefficients(f), coefficients(g)
    if first is None or second is None or not same_variable(f.variable, g.variable):
        return None
    return first, second


def _is_zero(values) -> bool:
    return len(values) == 1 and not values[0].numerator


def _poly_divmod(f: object, g: object) -> object:
    operands = _operands(f, g)
    if operands is None:
        return FALLBACK
    first, second = operands
    if _is_zero(second):
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


def _poly_gcd(f: object, g: object) -> object:
    operands = _operands(f, g)
    if operands is None or _is_zero(operands[0]) or _is_zero(operands[1]):
        return FALLBACK  # A zero operand: v0.7's xgcd conventions, left to the structured path.
    try:
        with computation(fresh=True):
            first, second = (kernels.rational_primitive(values) for values in operands)
            common, left, right = kernels.gcd_cofactors(first, second)
            if certify.check_gcd(first, second, common, left, right) is not True:
                return FALLBACK
            return trusted_polynomial(tp.monic_rationals(common), f.variable)
    except Exception:
        return FALLBACK


def _tuple(values):
    return tuple(values) if values else _ZERO


def _poly_xgcd(f: object, g: object) -> object:
    operands = _operands(f, g)
    if operands is None or _is_zero(operands[0]) or _is_zero(operands[1]):
        return FALLBACK
    first, second = operands
    try:
        with computation(fresh=True):
            common, s, t = (_tuple(values) for values in kernels.xgcd_rational(first, second))
            if certify.check_xgcd(first, second, common, s, t) is not True:
                return FALLBACK
    except Exception:
        return FALLBACK
    variable = f.variable
    return (trusted_polynomial(common, variable), trusted_polynomial(s, variable),
            trusted_polynomial(t, variable))


def _square_free(f: object) -> object:
    if current_budget() is not None:
        return FALLBACK
    values = coefficients(f)
    if values is None or len(values) < 2:
        return FALLBACK  # A constant: v0.7's decomposition without factors, left to the structured path.
    try:
        with computation(fresh=True):
            integer = kernels.rational_primitive(values)
            factors = kernels.square_free_integer(integer)
            monic = [(tp.monic_rationals(factor), multiplicity) for factor, multiplicity in factors]
            if certify.check_square_free(values, monic) is not True:
                return FALLBACK
    except Exception:
        return FALLBACK
    variable = f.variable
    return trusted_square_free(values[-1], tuple((trusted_polynomial(factor, variable), multiplicity)
                                                 for factor, multiplicity in monic))


def _polynomial_argument(value, variable):
    """The coefficients of a Polynomial in the variable, or of an int as a constant polynomial; None otherwise."""
    if type(value) is int:
        return (Rational(value),) if small_int(value) else None
    values = coefficients(value)
    if values is None or not same_variable(value.variable, variable):
        return None
    return values


def _rational_function(numerator: object, denominator: object, variable: object) -> object:
    if current_budget() is not None or not real_variable(variable):
        return FALLBACK
    top, bottom = _polynomial_argument(numerator, variable), _polynomial_argument(denominator, variable)
    if top is None or bottom is None or _is_zero(bottom):
        return FALLBACK
    try:
        with computation(fresh=True):
            integer_bottom = kernels.rational_primitive(bottom)
            if _is_zero(top):
                reduced_top, reduced_bottom = _ZERO, (trusted_rational(1, 1),)
            else:
                integer_top = kernels.rational_primitive(top)
                _, left, right = kernels.gcd_cofactors(integer_top, integer_bottom)
                # top = c F and bottom = d G with F = h left and G = h right, so top / bottom is
                # (c / d) left / right; with right made monic, the numerator's leading coefficient is lc(top) / lc(bottom).
                reduced_bottom = tp.monic_rationals(right)
                factor = top[-1] / (left[-1] * bottom[-1])
                reduced_top = tuple([factor * value for value in left])
            if certify.check_reduced_fraction(top, bottom, reduced_top, reduced_bottom) is not True:
                return FALLBACK
            excluded = ()
            if len(integer_bottom) > 1:
                common, part, rest = kernels.gcd_cofactors(integer_bottom,
                                                           tp.primitive(tp.derivative(integer_bottom)))
                if certify.check_square_free_part(integer_bottom, part, common, rest) is not True:
                    return FALLBACK
                excluded = (tp.monic_rationals(part),)
    except Exception:
        return FALLBACK
    # The polynomials' variables equal the variable argument (Symbols compare by all their fields).
    value = object.__new__(RationalFunction)
    fields = value.__dict__
    fields["numerator"] = trusted_polynomial(reduced_top, variable)
    fields["denominator"] = trusted_polynomial(reduced_bottom, variable)
    fields["excluded"] = tuple(trusted_polynomial(part, variable) for part in excluded)
    return value


def _resultant(f: object, g: object) -> object:
    operands = _operands(f, g)
    if operands is None:
        return FALLBACK
    first, second = operands
    if _is_zero(first) or _is_zero(second):
        return FALLBACK  # v0.7 answers 0 after its own validation; left to the generic path.
    try:
        with computation(fresh=True):
            value = kernels.resultant_rational(first, second)[0]
            if certify.check_rational_resultant(first, second, value) is not True:
                return FALLBACK
            return value
    except Exception:
        return FALLBACK


def _discriminant(f: object) -> object:
    if current_budget() is not None:
        return FALLBACK
    values = coefficients(f)
    if values is None or len(values) < 2:
        return FALLBACK  # A constant: v0.7 answers 0; left to the generic path.
    try:
        with computation(fresh=True):
            value = kernels.discriminant_rational(values)
            if certify.check_discriminant(values, value) is not True:
                return FALLBACK
            return value
    except Exception:
        return FALLBACK


def _routed(name, fast, exact=False):
    """The lean function of an operation: the fast path above, else the generic lean path of its legacy entry, for
    the inputs (expressions, zero or constant operands) that the fast path leaves to it."""
    import inspect
    names = tuple(inspect.signature(fast).parameters)

    def lean(*arguments):
        value = fast(*arguments)
        if value is FALLBACK and len(arguments) == len(names):
            return run_legacy(name, dict(zip(names, arguments)), exact=exact)
        return value

    lean.__name__ = lean.__qualname__ = name
    lean.__doc__ = fast.__doc__
    return lean


poly_divmod = _routed("poly_divmod", _poly_divmod)
poly_gcd = _routed("poly_gcd", _poly_gcd)
poly_xgcd = _routed("poly_xgcd", _poly_xgcd)
square_free = _routed("square_free", _square_free)
rational_function = _routed("rational_function", _rational_function)
resultant = _routed("resultant", _resultant, exact=True)
discriminant = _routed("discriminant", _discriminant, exact=True)

LEAN = {"poly_divmod": poly_divmod, "poly_gcd": poly_gcd, "poly_xgcd": poly_xgcd, "square_free": square_free,
        "rational_function": rational_function, "resultant": resultant, "discriminant": discriminant}
