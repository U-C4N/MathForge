"""Number-theory row adapters."""

import inspect
from typing import Any

import mathforge
from mathforge import ExecutionStatus, Result

from ..criteria import ADAPTERS, Outcome, same_value
from ..examples import Example
from . import example_call


def continued_fraction(example: Example) -> tuple[Outcome, Result[Any] | None]:
    """Row 27.25, Continued fractions, on mathforge's continued_fraction: the finite expansion of an exact rational.

    Two of the row's SymPy APIs ask what the operation answers when their number is rational: continued_fraction(x),
    and continued_fraction_periodic(p, q, d=0, s=1), the expansion of (p + s*sqrt(d))/q. Their terms are compared
    with SymPy's. A quadratic irrational, whose expansion is periodic, an expression that is not an exact rational as
    written, continued_fraction_convergents and continued_fraction_reduce, which the operation does not compute, and
    any other code are UNEVALUATED: never a match."""
    import sympy

    call = example_call(example)
    if call is None:
        return Outcome.UNEVALUATED, None
    number = _expanded_number(sympy, *call)
    if not isinstance(number, sympy.Rational):  # Integer is a Rational.
        return Outcome.UNEVALUATED, None
    result = mathforge.operations.continued_fraction(mathforge.Rational(int(number.p), int(number.q)))
    if result.execution_status is not ExecutionStatus.COMPLETED or result.value is None:
        return Outcome.UNEVALUATED, None
    if not mathforge.verify(result).verified:
        return Outcome.UNVERIFIED, result
    terms = [int(term) for term in result.value.terms]
    return (Outcome.MATCH if same_value(example, terms) else Outcome.MISMATCH), result


def _expanded_number(sympy: Any, function: Any, arguments: tuple[Any, ...], keywords: dict[str, Any]) -> Any:
    """The number whose continued fraction the SymPy call computes, as SymPy evaluates it; None for another function or
    arguments that SymPy would refuse."""
    if function is not sympy.continued_fraction and function is not sympy.continued_fraction_periodic:
        return None
    try:
        bound = inspect.signature(function).bind(*arguments, **keywords)
        bound.apply_defaults()
        if function is sympy.continued_fraction:
            return sympy.sympify(bound.arguments["a"], strict=True)
        p, q, d, s = (sympy.Integer(bound.arguments[name]) for name in ("p", "q", "d", "s"))
        return (p + s * sympy.sqrt(d)) / q
    except (TypeError, ValueError, sympy.SympifyError):
        return None


ADAPTERS["27.25"] = continued_fraction
