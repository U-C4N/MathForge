"""Lean dispatcher of the legacy evaluate operation, whose public function routes on the input type.

``mathforge.operations.evaluate`` sends a RealExpr or RealFunction to the registry's function_evaluate, with no
bindings meaning empty ones, and every other input through ``_run`` to the ``operations.LEGACY`` entry. This lean
function takes the same routes.
"""

from __future__ import annotations

from ..operations import _function_input
from . import functions, run_legacy


def evaluate(expression: object, bindings: object = None) -> object:
    if _function_input(expression):
        return functions.function_evaluate(expression, {} if bindings is None else bindings)
    return run_legacy("evaluate", {"expression": expression, "bindings": bindings})


LEAN = {"evaluate": evaluate}
