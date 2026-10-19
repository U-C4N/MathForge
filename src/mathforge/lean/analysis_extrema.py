"""Lean value function of extrema_values for rational functions whose real stationary points are all rational.

v0.7's producer analyzes the whole function (zeros, sign sets, monotonic regions) and evaluates the extrema in real
algebraic arithmetic; its checker verifies that whole analysis. When the raw numerator of the derivative, P = N'D -
ND', splits over Q into linear factors and quadratics without real roots, the strict extrema need none of that: they
are the rational roots of odd multiplicity inside the domain, and their values are rational. This producer finds them
from the square-free decomposition of P and reads each kind from the signs of P in the cells beside the root; the
certify checker (mathforge.certify.analysis.extrema) multiplies the factorization back and derives the kinds from its
parity instead. The points are the Rationals that v0.7 reports and the values N(r)/D(r), which v0.7's real algebraic
evaluation gives as the same Rationals. An input with an irrational or undecided stationary point, or a constant
function, takes the generic lean path of the v0.7 entry; any other call that this path does not handle falls back to
the structured path.

It runs through the generic legacy runner of mathforge.lean, with this producer and checker in place of the v0.7
entry, so the arguments, budgets and record checks are those of every legacy lean path.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from . import FALLBACK


class _Decline(Exception):
    """This path does not handle the input."""


def _produce(args: dict) -> tuple[object, tuple]:
    from ..algebraic_arithmetic import validate_exact_scalars
    from ..analysis_types import ExtremumValue
    from ..certify.analysis.extrema import FORMAT
    from ..limits import current_budget
    from ..model import Rational
    from ..polynomial import Polynomial
    from ..rational_function import as_rational_function, validate_variable

    validate_exact_scalars(args)
    variable = args["variable"]
    validate_variable(variable)
    function = as_rational_function(args["expression"], variable)
    numerator, denominator = function.numerator, function.denominator
    stationary = numerator.derivative() * denominator - numerator * denominator.derivative()
    if stationary.is_zero:
        raise _Decline  # A constant function: v0.7 reports its stationary domain.
    lead = stationary.leading_coefficient
    roots: list[tuple[Rational, int]] = []
    quadratics: list[tuple[Rational, Rational, Rational]] = []
    budget = current_budget()
    for factor in (stationary / lead).square_free().factors:
        if budget is not None:
            budget.tick()
        multiplicity = factor.multiplicity
        parts: Any = (factor.polynomial,)
        if factor.polynomial.degree > 2:  # Split it over Q: its linear and quadratic factors are handled below.
            from ..polynomial_exact import factor_with_evidence
            parts = tuple(entry.polynomial for entry in factor_with_evidence(factor.polynomial)[0].factors)
        for part in parts:
            coefficients = part.coefficients
            if len(coefficients) == 2:
                roots.append((-coefficients[0], multiplicity))
            elif len(coefficients) == 3:
                c, b = coefficients[0], coefficients[1]
                discriminant = b * b - 4 * c
                if discriminant < Rational(0):
                    quadratics.extend([coefficients] * multiplicity)
                    continue
                root = _rational_sqrt(discriminant)
                if root is None:
                    raise _Decline  # Irrational stationary points: v0.7's real algebraic arithmetic.
                roots.extend((((-b - root) / 2, multiplicity), ((-b + root) / 2, multiplicity)))
            else:
                raise _Decline  # An irreducible factor of degree three or more: v0.7's root isolation.
    roots.sort(key=lambda pair: pair[0])
    points = [point for point, _ in roots]
    values = []
    for index, (point, multiplicity) in enumerate(roots):
        if not function.is_defined_at(point):
            continue
        left = points[index - 1] if index else point - 1
        right = points[index + 1] if index + 1 < len(points) else point + 1
        before = _sign(stationary.evaluate((left + point) / 2))
        after = _sign(stationary.evaluate((point + right) / 2))
        if before * after == -1:
            values.append(ExtremumValue(point, function.evaluate(point), "minimum" if after > 0 else "maximum"))
    evidence = (FORMAT, lead, tuple(roots), tuple(quadratics))
    return tuple(values), evidence


def _sign(value: Any) -> int:
    return (value.numerator > 0) - (value.numerator < 0)


def _rational_sqrt(value: Any) -> Any:
    from math import isqrt
    from ..model import Rational
    numerator, denominator = value.numerator, value.denominator
    a, b = isqrt(numerator), isqrt(denominator)
    return Rational(a, b) if a * a == numerator and b * b == denominator else None


def _check(request: dict, value: object, evidence: tuple) -> bool:
    from ..certify.analysis.extrema import check_operation
    return check_operation(request, value, evidence)


class _Entry(NamedTuple):
    produce: Any
    check: Any


_ENTRY = _Entry(_produce, _check)


def extrema_values(expression: object, variable: object) -> object:
    """The rational-stationary path; an input it declines takes the generic lean path of v0.7's entry."""
    from . import _legacy, _prepare, run_legacy
    arguments = {"expression": expression, "variable": variable}
    try:
        _prepare()
        return _legacy("extrema_values", arguments, _ENTRY, FALLBACK)
    except _Decline:
        return run_legacy("extrema_values", arguments)
    except Exception:
        return FALLBACK


LEAN = {"extrema_values": extrema_values}
