"""Lean dispatchers of the legacy integration operations, whose public functions route on the input type.

``mathforge.operations`` sends a RealExpr or RealFunction integrand to the registry's function_antiderivative or
function_integrate, a RationalFunction to rational_antiderivative or rational_integrate, definite integrals with a
bound that is an Expression but not a Rational to the v4 exact path (``_run_exact``), and every other input through
``_run`` to the ``operations.LEGACY`` entry. These lean functions take the same routes.

A Polynomial object takes the polynomial kernel first: the antiderivative's coefficients (checked by
differentiating them back) returned as the expression tree of ``Polynomial.to_expression``, and a definite integral
between rational bounds as the antiderivative's values at the bounds (checked against the sum of v0.7's
contributions c_k (b**(k+1) - a**(k+1)) / (k+1)).
"""

from __future__ import annotations

from ..certify import upoly as certify
from ..kernels import upoly as kernels
from ..limits import computation, current_budget
from ..model import Expression, Rational
from ..operations import _function_input
from ..polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from ..rational_function import RationalFunction
from ..trusted import upoly as trusted
from . import FALLBACK, functions, run_legacy
from ._poly import MAX_BITS, coefficients, real_variable, same_variable


def antiderivative(expression: object, variable: object) -> object:
    if type(expression) is Polynomial:
        value = _antiderivative_polynomial(expression, variable)
        if value is not FALLBACK:
            return value
    elif _function_input(expression):
        return functions.function_antiderivative(expression, variable)
    elif type(expression) is RationalFunction:
        return functions.rational_antiderivative(expression, variable)
    return run_legacy("antiderivative", {"expression": expression, "variable": variable})


def integrate(expression: object, variable: object, lower: object, upper: object) -> object:
    if type(expression) is Polynomial:
        value = _integrate_polynomial(expression, variable, lower, upper)
        if value is not FALLBACK:
            return value
    elif _function_input(expression):
        return functions.function_integrate(expression, variable, lower, upper)
    elif type(expression) is RationalFunction:
        return functions.rational_integrate(expression, variable, lower, upper)
    arguments = {"expression": expression, "variable": variable, "lower": lower, "upper": upper}
    exact = any(isinstance(bound, Expression) and type(bound) is not Rational for bound in (lower, upper))
    return run_legacy("integrate", arguments, exact=exact)


def _polynomial_input(polynomial, variable):
    if current_budget() is not None or not real_variable(variable):
        return None
    values = coefficients(polynomial)
    if values is None or not same_variable(polynomial.variable, variable):
        return None
    return values


def _antiderivative_polynomial(polynomial: Polynomial, variable: object) -> object:
    """calculus.antiderivative_value of a Polynomial, or FALLBACK."""
    values = _polynomial_input(polynomial, variable)
    if values is None or len(values) - 1 >= MAX_POLYNOMIAL_DEGREE:
        return FALLBACK  # degree >= 1024: the structured path reports the degree limit
    try:  # Linear work in the validated input: no budget is needed to stop it.
        result = kernels.antiderivative(values)
        if certify.check_antiderivative(values, result) is not True:
            return FALLBACK
        return trusted.expression(result, polynomial.variable)
    except Exception:
        return FALLBACK


def _bound(value):
    if type(value) is int:
        return Rational(value) if value.bit_length() <= MAX_BITS else None
    if type(value) is Rational and value.numerator.bit_length() <= MAX_BITS and value.denominator.bit_length() <= MAX_BITS:
        return value
    return None


def _integrate_polynomial(polynomial: Polynomial, variable: object, lower: object, upper: object) -> object:
    """calculus.definite_integral's value for a Polynomial between rational bounds, or FALLBACK."""
    values = _polynomial_input(polynomial, variable)
    lower, upper = _bound(lower), _bound(upper)
    if values is None or lower is None or upper is None:
        return FALLBACK
    try:
        with computation(fresh=True):
            value = kernels.definite_integral(values, lower, upper)
            if certify.check_definite_integral(values, lower, upper, value) is not True:
                return FALLBACK
            return value
    except Exception:
        return FALLBACK


LEAN = {"antiderivative": antiderivative, "integrate": integrate}
