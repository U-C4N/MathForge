"""Lean dispatchers of the legacy analysis operations whose public functions route on the input type.

``mathforge.operations`` sends a RealExpr or RealFunction input of differentiate, partial_diff, gradient, jacobian
and hessian to the registry's function_* operation, and every other input through ``_run`` to the operation's
``operations.LEGACY`` entry. These lean functions take the same routes: the registry operation's lean function, or
the lean path of the LEGACY entry. Where the public function makes a tuple of an iterable, a one-shot iterator falls
back first, so that the structured path still receives it whole.
"""

from __future__ import annotations

from typing import Any

from ..operations import _function_input
from . import FALLBACK, functions, run_legacy


def differentiate(expression: Any, variable: Any, n: Any = 1) -> Any:
    if _function_input(expression):
        return functions.function_differentiate(expression, variable, n)
    return run_legacy("differentiate", {"expression": expression, "variable": variable, "n": n})


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
