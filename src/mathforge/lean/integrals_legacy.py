"""Lean dispatchers of the legacy integration operations, whose public functions route on the input type.

``mathforge.operations`` sends a RealExpr or RealFunction integrand to the registry's function_antiderivative or
function_integrate, a RationalFunction to rational_antiderivative or rational_integrate, definite integrals with a
bound that is an Expression but not a Rational to the v4 exact path (``_run_exact``), and every other input through
``_run`` to the ``operations.LEGACY`` entry. These lean functions take the same routes.
"""

from __future__ import annotations

from ..model import Expression, Rational
from ..operations import _function_input
from ..rational_function import RationalFunction
from . import functions, run_legacy


def antiderivative(expression: object, variable: object) -> object:
    if _function_input(expression):
        return functions.function_antiderivative(expression, variable)
    if type(expression) is RationalFunction:
        return functions.rational_antiderivative(expression, variable)
    return run_legacy("antiderivative", {"expression": expression, "variable": variable})


def integrate(expression: object, variable: object, lower: object, upper: object) -> object:
    if _function_input(expression):
        return functions.function_integrate(expression, variable, lower, upper)
    if type(expression) is RationalFunction:
        return functions.rational_integrate(expression, variable, lower, upper)
    arguments = {"expression": expression, "variable": variable, "lower": lower, "upper": upper}
    exact = any(isinstance(bound, Expression) and type(bound) is not Rational for bound in (lower, upper))
    return run_legacy("integrate", arguments, exact=exact)


LEAN = {"antiderivative": antiderivative, "integrate": integrate}
