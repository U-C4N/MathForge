"""Exact real rational-function analysis from certified roots and sign cells.

Evidence is ``(groups, samples)``. Each group is ``(polynomial, roots,
root_certificate)``; each sample is ``(rational_point, function_sign,
derivative_sign)`` for one open cell in the complete root partition.
The checker validates those certificates and claims without invoking a producer.
"""

from functools import cmp_to_key

from .algebraic import as_endpoint, compare_real, rational_between, sign_at
from .analysis_types import ExcludedPoint, FunctionAnalysis, LocalExtremum, MonotonicRegion
from .errors import UnsupportedOperation
from .intervals import Interval, IntervalSet, equal_interval_sets
from .limits import current_budget
from .model import Rational
from .rational_function import as_rational_function, same_domain, validate_variable
from .root_certificates import check_root_certificate, root_certificate
from .root_isolation import real_roots, root_work


def _sources(function, derivative):
    """A stable, deduplicated list; never enumerate the zero polynomial."""
    sources = []
    seen = set()
    for polynomial in (function.numerator, derivative.numerator, *function.excluded):
        current_budget().tick()
        if not polynomial.is_zero and polynomial not in seen:
            sources.append(polynomial)
            seen.add(polynomial)
    return tuple(sources)


def _points(groups):
    points = sorted((as_endpoint(record.root) for _, records, _ in groups for record in records),
                    key=cmp_to_key(compare_real))
    unique = []
    for point in points:
        current_budget().tick()
        if not unique or compare_real(unique[-1], point) != 0:
            unique.append(point)
    return tuple(unique)


def _sign(function, point):
    return sign_at(function.numerator, point) * sign_at(function.denominator, point)


def _point_values(function, derivative, points):
    """Point membership and exact signs, independent of open-cell samples."""
    values = []
    for point in points:
        current_budget().tick()
        if function.is_defined_at(point):
            values.append((_sign(function, point), _sign(derivative, point)))
        else:
            values.append(None)
    return tuple(values)


def _cell_set(points, open_membership, point_membership):
    """Canonical union of selected open cells and singleton boundary points."""
    bounds = (None, *points, None)
    intervals = []
    for index, included in enumerate(open_membership):
        current_budget().tick()
        if included:
            intervals.append(Interval(bounds[index], bounds[index + 1]))
    for point, included in zip(points, point_membership):
        current_budget().tick()
        if included:
            intervals.append(Interval(point, point, True, True))
    return IntervalSet(tuple(intervals))


def _producer_regions(points, samples, point_values):
    bounds = (None, *points, None)
    regions = []
    for index, (_, _, sign) in enumerate(samples):
        current_budget().tick()
        behavior = {1: "increasing", -1: "decreasing", 0: "constant"}[sign]
        lc = index > 0 and point_values[index - 1] is not None
        rc = index < len(points) and point_values[index] is not None
        interval = Interval(bounds[index], bounds[index + 1], lc, rc)
        if regions and regions[-1].behavior == behavior and lc:
            previous = regions.pop().interval
            interval = Interval(previous.lower, interval.upper, previous.left_closed, rc)
        regions.append(MonotonicRegion(interval, behavior))
    return tuple(regions)


def _assemble_analysis(function, derivative, points, samples):
    """Producer-only result assembly; the checker does not use this function."""
    values = _point_values(function, derivative, points)
    domain = _cell_set(points, (True for _ in samples), (v is not None for v in values))
    sets = {}
    for name, component, sign in (("zero", 0, 0), ("positive", 0, 1),
                                  ("negative", 0, -1), ("stationary", 1, 0)):
        sets[name] = _cell_set(points, (s[component + 1] == sign for s in samples),
                              (v is not None and v[component] == sign for v in values))
    excluded = tuple(ExcludedPoint(point, "pole" if sign_at(function.denominator, point) == 0 else "hole")
                     for point, value in zip(points, values) if value is None)
    extrema = []
    for index, (point, value) in enumerate(zip(points, values)):
        current_budget().tick()
        if value is not None and value[1] == 0:
            left, right = samples[index][2], samples[index + 1][2]
            if left == -1 and right == 1:
                extrema.append(LocalExtremum(point, "minimum"))
            elif left == 1 and right == -1:
                extrema.append(LocalExtremum(point, "maximum"))
    return FunctionAnalysis(function, domain, excluded, sets["zero"], sets["positive"],
                            sets["negative"], derivative, sets["stationary"],
                            _producer_regions(points, samples, values), tuple(extrema))


def analyze_function(expression, variable, *, limits=None):
    """Produce complete exact analysis of a real univariate rational function.

    Monotonic regions are maximal connected intervals, include defined critical
    endpoints, and never bridge a hole. Extrema are strict and record x only.
    A constant function is stationary on its entire (possibly punctured) domain.
    """
    with root_work(limits) as budget:
        budget.inspect((expression, variable))
        validate_variable(variable)
        function = as_rational_function(expression, variable)
        derivative = function.derivative()
        groups = []
        for polynomial in _sources(function, derivative):
            records = real_roots(polynomial)
            groups.append((polynomial, records, root_certificate(polynomial, records)))
        groups = tuple(groups)
        points = _points(groups)
        bounds = (None, *points, None)
        samples = []
        for lower, upper in zip(bounds, bounds[1:]):
            sample = rational_between(lower, upper)
            samples.append((sample, _sign(function, sample), _sign(derivative, sample)))
        samples = tuple(samples)
        evidence = (groups, samples)
        budget.inspect(evidence, evidence=True)
        value = _assemble_analysis(function, derivative, points, samples)
        budget.inspect(value)
        return value, evidence


def _boundary_index(point, points, *, upper):
    if point is None:
        return len(points) + 1 if upper else 0
    for index, candidate in enumerate(points, 1):
        current_budget().tick()
        if compare_real(point, candidate) == 0:
            return index
    return None


def _check_regions(regions, points, samples, values):
    """Validate coverage, direction, endpoint closure and maximality directly.

    Each claimed region must cover consecutive complete open cells exactly once.
    Its endpoints must be partition boundaries; an adjacent cell with the same
    direction and a defined joining point proves a region is not maximal.
    """
    cursor = 0
    signs = {"increasing": 1, "decreasing": -1, "constant": 0}
    for region in regions:
        current_budget().tick()
        interval = region.interval
        lower = _boundary_index(interval.lower, points, upper=False)
        upper = _boundary_index(interval.upper, points, upper=True)
        if lower != cursor or upper is None or upper <= lower:
            return False
        expected_sign = signs[region.behavior]
        if (interval.left_closed != (lower > 0 and values[lower - 1] is not None)
                or interval.right_closed != (upper <= len(points) and values[upper - 1] is not None)):
            return False
        for cell in range(lower, upper):
            current_budget().tick()
            if samples[cell][2] != expected_sign:
                return False
            if cell > lower and values[cell - 1] is None:
                return False
        if lower > 0 and values[lower - 1] is not None and samples[lower - 1][2] == expected_sign:
            return False
        if upper < len(samples) and values[upper - 1] is not None and samples[upper][2] == expected_sign:
            return False
        cursor = upper
    return cursor == len(samples)


def _check_analysis(expression, variable, value, evidence):
    if type(value) is not FunctionAnalysis or type(evidence) is not tuple or len(evidence) != 2:
        return False
    validate_variable(variable)
    function = as_rational_function(expression, variable)
    recorded = value.function
    if (recorded.numerator != function.numerator or recorded.denominator != function.denominator
            or not same_domain(recorded, function)):
        return False
    derivative = value.derivative
    if derivative.variable != variable or not same_domain(function, derivative):
        return False
    n, d = function.numerator, function.denominator
    raw_numerator = n.derivative() * d - n * d.derivative()
    raw_denominator = d * d
    # Exact division avoids multiplying already-reduced claims into degrees
    # larger than either the source derivative or the polynomial degree cap.
    denominator_factor, denominator_remainder = raw_denominator.divmod(derivative.denominator)
    if not denominator_remainder.is_zero:
        return False
    if derivative.numerator.is_zero:
        if not raw_numerator.is_zero:
            return False
    else:
        numerator_factor, numerator_remainder = raw_numerator.divmod(derivative.numerator)
        if not numerator_remainder.is_zero or numerator_factor != denominator_factor:
            return False
    groups, samples = evidence
    sources = _sources(function, derivative)
    if type(groups) is not tuple or len(groups) != len(sources) or type(samples) is not tuple:
        return False
    for source, group in zip(sources, groups):
        current_budget().tick()
        if type(group) is not tuple or len(group) != 3:
            return False
        polynomial, records, certificate = group
        if (polynomial != source or type(records) is not tuple
                or not check_root_certificate(source, records, certificate)):
            return False
    points = _points(groups)
    bounds = (None, *points, None)
    if len(samples) != len(points) + 1:
        return False
    for lower, upper, sample in zip(bounds, bounds[1:], samples):
        current_budget().tick()
        if type(sample) is not tuple or len(sample) != 3:
            return False
        point, f_sign, d_sign = sample
        if (type(point) is not Rational or type(f_sign) is not int or type(d_sign) is not int
                or f_sign not in (-1, 0, 1) or d_sign not in (-1, 0, 1)
                or lower is not None and compare_real(point, lower) <= 0
                or upper is not None and compare_real(point, upper) >= 0
                or not function.is_defined_at(point)
                or f_sign != _sign(function, point) or d_sign != _sign(derivative, point)):
            return False
        if (f_sign == 0) != function.numerator.is_zero or (d_sign == 0) != derivative.numerator.is_zero:
            return False
    values = _point_values(function, derivative, points)
    domain = _cell_set(points, (True for _ in samples), (v is not None for v in values))
    if not equal_interval_sets(value.domain, domain):
        return False
    for reported, component, sign in ((value.zero_set, 0, 0), (value.positive_set, 0, 1),
                                      (value.negative_set, 0, -1), (value.stationary_set, 1, 0)):
        expected = _cell_set(points, (s[component + 1] == sign for s in samples),
                             (v is not None and v[component] == sign for v in values))
        if not equal_interval_sets(reported, expected):
            return False
    excluded_index = extrema_index = 0
    for index, (point, signs) in enumerate(zip(points, values)):
        current_budget().tick()
        if signs is None:
            if excluded_index >= len(value.excluded_points):
                return False
            record = value.excluded_points[excluded_index]
            kind = "pole" if sign_at(function.denominator, point) == 0 else "hole"
            if record.kind != kind or compare_real(record.point, point) != 0:
                return False
            excluded_index += 1
        elif samples[index][2] * samples[index + 1][2] == -1:
            if signs[1] != 0 or extrema_index >= len(value.strict_extrema):
                return False
            record = value.strict_extrema[extrema_index]
            kind = "minimum" if samples[index][2] == -1 else "maximum"
            if record.kind != kind or compare_real(record.point, point) != 0:
                return False
            extrema_index += 1
    return (excluded_index == len(value.excluded_points) and extrema_index == len(value.strict_extrema)
            and _check_regions(value.monotonic_regions, points, samples, values))


def check_analysis(expression, variable, value, evidence, *, limits=None):
    """Independently check every analysis claim; resource exhaustion propagates."""
    with root_work(limits) as budget:
        budget.inspect((expression, variable, value, evidence), evidence=True)
        try:
            return _check_analysis(expression, variable, value, evidence)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, UnsupportedOperation):
            return False
