"""Canonical finite unions of exact real intervals."""

from dataclasses import dataclass
from functools import cmp_to_key

from .algebraic import RealAlgebraicRoot, compare_real
from .errors import InvalidInput, UnsupportedOperation
from .limits import computation
from .model import Rational, Reals


def _endpoint(value):
    if type(value) is int:
        return Rational(value)
    if value is None or type(value) in (Rational, RealAlgebraicRoot):
        return value
    raise InvalidInput("Interval endpoints must be rational or real algebraic roots")


@dataclass(frozen=True)
class Interval:
    lower: Rational | RealAlgebraicRoot | None
    upper: Rational | RealAlgebraicRoot | None
    left_closed: bool = False
    right_closed: bool = False

    def __post_init__(self):
        with computation():
            lower, upper = _endpoint(self.lower), _endpoint(self.upper)
            if type(self.left_closed) is not bool or type(self.right_closed) is not bool:
                raise InvalidInput("Interval closure flags must be booleans")
            if lower is None and self.left_closed or upper is None and self.right_closed:
                raise InvalidInput("An infinite endpoint must be open")
            if lower is not None and upper is not None and compare_real(lower, upper) > 0:
                raise InvalidInput("Interval endpoints are reversed")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)

    @property
    def is_empty(self):
        return (self.lower is not None and self.upper is not None
                and compare_real(self.lower, self.upper) == 0
                and not (self.left_closed and self.right_closed))

    def contains(self, value, *, limits=None):
        with computation(limits):
            left = 1 if self.lower is None else compare_real(value, self.lower)
            right = -1 if self.upper is None else compare_real(value, self.upper)
            return ((left > 0 or left == 0 and self.left_closed)
                    and (right < 0 or right == 0 and self.right_closed))


def _lower_order(a, b):
    if a.lower is None:
        return -1 if b.lower is not None else 0
    if b.lower is None:
        return 1
    order = compare_real(a.lower, b.lower)
    return order if order else (b.left_closed - a.left_closed)


def _same_end(a, b):
    return a is None and b is None or (a is not None and b is not None and compare_real(a, b) == 0)


@dataclass(frozen=True)
class IntervalSet:
    intervals: tuple[Interval, ...]

    def __post_init__(self):
        with computation() as budget:
            items = tuple(self.intervals)
            budget.inspect(items)
            if any(type(item) is not Interval for item in items):
                raise InvalidInput("IntervalSet requires Interval entries")
            items = sorted((item for item in items if not item.is_empty), key=cmp_to_key(_lower_order))
            merged = []
            for item in items:
                budget.tick()
                if not merged:
                    merged.append(item)
                    continue
                previous = merged[-1]
                touch = -1 if previous.upper is None or item.lower is None else compare_real(item.lower, previous.upper)
                if touch > 0 or touch == 0 and not (previous.right_closed or item.left_closed):
                    merged.append(item)
                    continue
                upper, right_closed = previous.upper, previous.right_closed
                if upper is not None:
                    order = 1 if item.upper is None else compare_real(item.upper, upper)
                    if order > 0:
                        upper, right_closed = item.upper, item.right_closed
                    elif order == 0:
                        right_closed = right_closed or item.right_closed
                merged[-1] = Interval(previous.lower, upper, previous.left_closed, right_closed)
            object.__setattr__(self, "intervals", tuple(merged))

    @property
    def domain(self):
        return Reals

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        return iter(self.intervals)

    def contains(self, value, *, limits=None):
        with computation(limits):
            return any(interval.contains(value) for interval in self.intervals)

    @classmethod
    def from_solution_set(cls, value):
        from .contracts import EmptySet, UniversalSet
        if type(value) is cls:
            return value
        if type(value) not in (EmptySet, UniversalSet) or value.domain != Reals:
            raise UnsupportedOperation("Only real interval, empty, or universal sets are convertible")
        return cls(()) if type(value) is EmptySet else cls((Interval(None, None),))

    def to_solution_set(self):
        from .contracts import EmptySet, UniversalSet
        if not self.intervals:
            return EmptySet(Reals)
        if len(self.intervals) == 1 and self.intervals[0] == Interval(None, None):
            return UniversalSet(Reals)
        return self

    def union(self, other, *, limits=None):
        with computation(limits):
            other = self.from_solution_set(other)
            return IntervalSet(self.intervals + other.intervals)

    def intersection(self, other, *, limits=None):
        with computation(limits) as budget:
            other = self.from_solution_set(other)
            intersections = []
            i = j = 0
            while i < len(self.intervals) and j < len(other.intervals):
                budget.tick()
                a, b = self.intervals[i], other.intervals[j]
                if _lower_order(a, b) < 0:
                    lower, lc = b.lower, b.left_closed
                else:
                    lower, lc = a.lower, a.left_closed
                if _same_end(a.lower, b.lower):
                    lc = a.left_closed and b.left_closed
                end = (-1 if b.upper is None and a.upper is not None else
                       1 if a.upper is None and b.upper is not None else
                       0 if a.upper is None else compare_real(a.upper, b.upper))
                upper, rc = (a.upper, a.right_closed) if end <= 0 else (b.upper, b.right_closed)
                if end == 0:
                    rc = a.right_closed and b.right_closed
                if lower is None or upper is None or compare_real(lower, upper) <= 0:
                    intersections.append(Interval(lower, upper, lc, rc))
                if end <= 0:
                    i += 1
                if end >= 0:
                    j += 1
            return IntervalSet(tuple(intersections))


def equal_interval_sets(left, right):
    left, right = IntervalSet.from_solution_set(left), IntervalSet.from_solution_set(right)
    return len(left) == len(right) and all(
        _same_end(a.lower, b.lower) and _same_end(a.upper, b.upper)
        and a.left_closed == b.left_closed and a.right_closed == b.right_closed
        for a, b in zip(left, right))
