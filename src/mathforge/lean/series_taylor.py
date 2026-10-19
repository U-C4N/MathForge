"""Lean value function of function_taylor (public name ``taylor``): v0.7's producer, checked by power series.

The generic runner would verify the producer's TaylorPolynomial with v0.7's checker, which re-derives every symbolic
derivative of the trace through formal jets and proves each equal with canonical sums: most of a Taylor call. Here
the runner is the same (mathforge.lean's registry plan, with its arguments, budgets and record checks), the value is
v0.7's, and the check is mathforge.certify.series.taylor: it recomputes the coefficients at the center as truncated
power series in exact rational arithmetic, an algorithm independent of the producer's differentiation.

v0.7's checker decides instead wherever the series checker declines (a center at which some node's value is not
rational, an arccosine, pi or E, ...), and also for every source that holds tan or tanh: v0.7's checker refuses some
correct traces of those (see _V07_REFUSES), and the lean path must refuse what the structured path refuses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import FALLBACK

OPERATION = "function_taylor"

# v0.7's checker proves each step of the derivative trace with canonical sums over opaque atoms. It rewrites tan(u)
# and tanh(u) as quotients whose new sin, cos, sinh and cosh atoms it does not identify with the equivalent atoms of
# the other side, so it refuses some correct traces of sources holding tan or tanh, and the structured path then
# fails. tests/v08/series/test_taylor.py compares both paths on seeded sources with and without them.
_V07_REFUSES = frozenset({"tan", "tanh"})

# The registry spec the plan was made for, and the plan: a registry rebuilt with other families gives a new spec.
_PLAN: list[tuple[Any, Callable[[Mapping[str, object]], object]]] = []


def _v07_may_refuse(request: Mapping[str, object]) -> bool:
    """Whether the source holds a node that v0.7's checker may refuse to prove (or is not a RealExpr tree at all)."""
    from ..functions import RealExpr, RealFunction
    source = request.get("expression")
    if type(source) is RealFunction:
        source = source.expression
    if type(source) is not RealExpr:
        return True
    pending, seen = [source], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node.op in _V07_REFUSES:
            return True
        pending.extend(node.args)
    return False


def _plan() -> Callable[[Mapping[str, object]], object] | None:
    from .. import registry
    from . import _prepare, _registry_plan, _spec
    spec = _spec(OPERATION)
    if _PLAN and _PLAN[0][0] is spec:
        return _PLAN[0][1]
    if spec is None or spec.version != 5:
        return None  # Not v0.7's family: the generic runner decides.
    _prepare()
    v07 = registry.checker(OPERATION, spec.version)  # The family module; its check is read at each call.

    def check(operation: str, request: Mapping[str, object], value: object, evidence: tuple) -> bool:
        from ..certify.series.taylor import check_operation
        if not _v07_may_refuse(request) and check_operation(operation, request, value, evidence) is True:
            return True
        return v07.check_operation(operation, request, value, evidence) is True

    plan = _registry_plan(OPERATION, spec, check)
    _PLAN[:] = [(spec, plan)]
    return plan


def function_taylor(expression: object, variable: object, point: object = 0, order: object = 5) -> object:
    from . import run_value
    arguments = {"expression": expression, "variable": variable, "point": point, "order": order}
    try:
        plan = _plan()
        return run_value(OPERATION, arguments) if plan is None else plan(arguments)
    except Exception:
        return FALLBACK


LEAN = {OPERATION: function_taylor}
