"""Lean dispatcher of the legacy limit operation, whose public function routes on the input type.

``mathforge.operations.limit`` sends a RealExpr or RealFunction to the registry's function_limit, a point that is an
Expression but not a Rational to the v4 exact path (``_run_exact``), and every other input through ``_run`` to the
v3 ``operations.LEGACY`` entry. This lean function takes the same routes.
"""

from __future__ import annotations

from ..model import Expression, Rational
from ..operations import _function_input
from . import functions, run_legacy


def limit(expression: object, variable: object, point: object, side: object = "both") -> object:
    if _function_input(expression):
        return functions.function_limit(expression, variable, point, side)
    exact = isinstance(point, Expression) and type(point) is not Rational
    return run_legacy("limit", {"expression": expression, "variable": variable, "point": point, "side": side},
                      exact=exact)


LEAN = {"limit": limit}
