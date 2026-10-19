"""Lean dispatchers of the legacy analysis operations whose public functions route on the input type.

``mathforge.operations`` sends a RealExpr or RealFunction input of differentiate, partial_diff, gradient, jacobian
and hessian to the registry's function_* operation, and every other input through ``_run`` to the operation's
``operations.LEGACY`` entry. These lean functions take the same routes: the registry operation's lean function, or
the lean path of the LEGACY entry. Where the public function makes a tuple of an iterable, a one-shot iterator falls
back first, so that the structured path still receives it whole.

A Polynomial object given to differentiate takes the polynomial kernel first: the derivative's coefficients from
mathforge.kernels.upoly, checked against math.perm's falling factorials by mathforge.certify.upoly, and returned as
the expression tree that ``Polynomial.to_expression`` builds, as the structured path does.
"""

from __future__ import annotations

from typing import Any

from ..certify import upoly as certify
from ..kernels import upoly as kernels
from ..limits import current_budget
from ..operations import _function_input
from ..polynomial import Polynomial
from ..trusted import upoly as trusted
from . import FALLBACK, functions, run_legacy
from ._poly import coefficients, real_variable, same_variable, small_int


def differentiate(expression: Any, variable: Any, n: Any = 1) -> Any:
    if type(expression) is Polynomial:
        value = _differentiate_polynomial(expression, variable, n)
        if value is not FALLBACK:
            return value
    elif _function_input(expression):
        return functions.function_differentiate(expression, variable, n)
    return run_legacy("differentiate", {"expression": expression, "variable": variable, "n": n})


def _differentiate_polynomial(polynomial: Polynomial, variable: Any, n: Any) -> Any:
    """The n-th derivative of a Polynomial as calculus.derivative returns it, or FALLBACK."""
    if current_budget() is not None or not small_int(n) or n < 0 or not real_variable(variable):
        return FALLBACK
    values = coefficients(polynomial)
    if values is None or not same_variable(polynomial.variable, variable):
        return FALLBACK
    try:  # Linear work in the validated input: no budget is needed to stop it.
        result = kernels.derivative(values, n)
        if certify.check_derivative(values, n, result) is not True:
            return FALLBACK
        return trusted.expression(result, polynomial.variable)
    except Exception:
        return FALLBACK


def partial_diff(expression: Any, variable: Any, n: Any = 1) -> Any:
    if _function_input(expression):
        return functions.function_partial_diff(expression, variable, n)
    return run_legacy("partial_diff", {"expression": expression, "variable": variable, "n": n})


def gradient(expression: Any, variables: Any) -> Any:
    if _function_input(expression):
        if type(variables) not in (tuple, list):
            return FALLBACK
        return functions.function_gradient(expression, tuple(variables))
    return run_legacy("gradient", {"expression": expression, "variables": variables})


def jacobian(expressions: Any, variables: Any) -> Any:
    if type(expressions) in (tuple, list) and any(_function_input(item) for item in expressions):
        if type(variables) not in (tuple, list):
            return FALLBACK
        return functions.function_jacobian(tuple(expressions), tuple(variables))
    return run_legacy("jacobian", {"expressions": expressions, "variables": variables})


def hessian(expression: Any, variables: Any) -> Any:
    if _function_input(expression):
        if type(variables) not in (tuple, list):
            return FALLBACK
        return functions.function_hessian(expression, tuple(variables))
    return run_legacy("hessian", {"expression": expression, "variables": variables})


LEAN = {"differentiate": differentiate, "partial_diff": partial_diff, "gradient": gradient, "jacobian": jacobian,
        "hessian": hessian}
