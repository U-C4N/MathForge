"""Immutable exact records for rational limits and real function analysis.

These records carry mathematical claims. Construction validates their shape;
``verify`` checks their source linkage and mathematical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from .algebraic import RealAlgebraicRoot
from .errors import InvalidInput
from .intervals import Interval, IntervalSet
from .limits import current_budget
from .model import Rational
from .rational_function import RationalFunction


RealPoint: TypeAlias = Rational | RealAlgebraicRoot
LimitKind: TypeAlias = Literal["finite", "positive_infinity", "negative_infinity", "does_not_exist"]


def _choice(value, choices, name):
    if type(value) is not str or value not in choices:
        raise InvalidInput(f"Invalid {name}")


def _point(value):
    if type(value) not in (Rational, RealAlgebraicRoot):
        raise InvalidInput("An exact rational or real algebraic point is required")


def _items(values, cls):
    if type(values) not in (tuple, list):
        raise InvalidInput(f"A sequence of {cls.__name__} records is required")
    result = tuple(values)
    if any(type(value) is not cls for value in result):
        raise InvalidInput(f"Expected {cls.__name__} records")
    return result


def _inspect(value):
    budget = current_budget()
    if budget is not None:
        budget.inspect(value)


@dataclass(frozen=True)
class LimitValue:
    kind: LimitKind
    value: RealPoint | None = None

    def __post_init__(self):
        _choice(self.kind, ("finite", "positive_infinity", "negative_infinity", "does_not_exist"), "limit kind")
        if self.kind == "finite":
            if type(self.value) not in (Rational, RealAlgebraicRoot):
                raise InvalidInput("A finite limit requires an exact real algebraic value")
        elif self.value is not None:
            raise InvalidInput("A non-finite limit cannot carry a scalar value")
        _inspect(self)


@dataclass(frozen=True)
class ExcludedPoint:
    point: RealPoint
    kind: Literal["hole", "pole"]

    def __post_init__(self):
        _point(self.point)
        _choice(self.kind, ("hole", "pole"), "excluded-point kind")
        _inspect(self)


@dataclass(frozen=True)
class MonotonicRegion:
    interval: Interval
    behavior: Literal["increasing", "decreasing", "constant"]

    def __post_init__(self):
        if type(self.interval) is not Interval or self.interval.is_empty:
            raise InvalidInput("A monotonic region requires a nonempty Interval")
        _choice(self.behavior, ("increasing", "decreasing", "constant"), "monotonic behavior")
        _inspect(self)


@dataclass(frozen=True)
class LocalExtremum:
    point: RealPoint
    kind: Literal["minimum", "maximum"]

    def __post_init__(self):
        _point(self.point)
        _choice(self.kind, ("minimum", "maximum"), "extremum kind")
        _inspect(self)


@dataclass(frozen=True)
class ExtremumValue:
    point: RealPoint
    value: RealPoint
    kind: Literal["minimum", "maximum"]

    def __post_init__(self):
        _point(self.point)
        _point(self.value)
        _choice(self.kind, ("minimum", "maximum"), "extremum kind")
        _inspect(self)


@dataclass(frozen=True)
class FunctionAnalysis:
    function: RationalFunction
    domain: IntervalSet
    excluded_points: tuple[ExcludedPoint, ...]
    zero_set: IntervalSet
    positive_set: IntervalSet
    negative_set: IntervalSet
    derivative: RationalFunction
    stationary_set: IntervalSet
    monotonic_regions: tuple[MonotonicRegion, ...]
    strict_extrema: tuple[LocalExtremum, ...]

    def __post_init__(self):
        if type(self.function) is not RationalFunction or type(self.derivative) is not RationalFunction:
            raise InvalidInput("Function analysis requires rational function records")
        if self.function.variable != self.derivative.variable:
            raise InvalidInput("The function and derivative variables must agree")
        for name in ("domain", "zero_set", "positive_set", "negative_set", "stationary_set"):
            if type(getattr(self, name)) is not IntervalSet:
                raise InvalidInput(f"{name} must be an IntervalSet")
        for name, cls in (("excluded_points", ExcludedPoint), ("monotonic_regions", MonotonicRegion),
                          ("strict_extrema", LocalExtremum)):
            object.__setattr__(self, name, _items(getattr(self, name), cls))
        _inspect(self)
