"""Exact sign charts with independently checkable root and cell evidence."""

from functools import cmp_to_key

from .algebraic import as_endpoint, compare_real, rational_between, sign_at
from .contracts import Eq, Inequality, EmptySet, FiniteSet
from .intervals import Interval, IntervalSet, equal_interval_sets
from .limits import computation
from .model import Rational, Reals
from .rational_function import as_rational_function, validate_variable
from .root_isolation import real_roots, verify_root_records
from .root_certificates import root_certificate, check_root_certificate


def source_function(problem, variable):
    validate_variable(variable)
    return as_rational_function(problem.lhs, variable) - as_rational_function(problem.rhs, variable)


def _sources(function):
    return (() if function.numerator.is_zero else (function.numerator,)) + function.excluded


def _points(groups):
    points = sorted((as_endpoint(record.root) for _, records, _ in groups for record in records),
                    key=cmp_to_key(compare_real))
    unique = []
    for point in points:
        if not unique or compare_real(point, unique[-1]) != 0:
            unique.append(point)
    return tuple(unique)


def _sign(function, point):
    return sign_at(function.numerator, point) * sign_at(function.denominator, point)


def _accept(sign, relation):
    return {"eq": sign == 0, "lt": sign < 0, "le": sign <= 0,
            "gt": sign > 0, "ge": sign >= 0}[relation]


def _assemble(function, points, samples, relation):
    intervals = []
    bounds = (None,) + points + (None,)
    for index, (_, sign) in enumerate(samples):
        if _accept(sign, relation):
            intervals.append(Interval(bounds[index], bounds[index + 1]))
    for point in points:
        if function.is_defined_at(point) and _accept(_sign(function, point), relation):
            intervals.append(Interval(point, point, True, True))
    return IntervalSet(tuple(intervals)).to_solution_set()


def solve_real_set(problem, variable, *, limits=None):
    with computation(limits) as budget:
        function = source_function(problem, variable)
        groups = tuple((poly, records, root_certificate(poly, records))
                       for poly in _sources(function) for records in (real_roots(poly),))
        if type(problem) is Eq and not function.numerator.is_zero:
            values = tuple(record.root for record in groups[0][1] if function.is_defined_at(record.root))
            value = FiniteSet(values, Reals) if values else EmptySet(Reals)
            evidence = (groups, ())
        else:
            points = _points(groups)
            bounds = (None,) + points + (None,)
            samples = tuple((sample, _sign(function, sample)) for sample in
                            (rational_between(a, b) for a, b in zip(bounds, bounds[1:])))
            value = _assemble(function, points, samples,
                              "eq" if type(problem) is Eq else problem.relation)
            evidence = (groups, samples)
        budget.inspect(evidence, evidence=True)
        return value, evidence


def check_solution(problem, variable, value, evidence, *, limits=None):
    """Check all root counts, signs and cells without invoking the solver."""
    with computation(limits):
        if type(problem) not in (Eq, Inequality):
            return False
        function = source_function(problem, variable)
        groups, samples = evidence
        sources = _sources(function)
        if len(groups) != len(sources):
            return False
        for source, (poly, records, certificate) in zip(sources, groups):
            if source != poly or not check_root_certificate(source, records, certificate):
                return False
        if type(problem) is Eq and not function.numerator.is_zero:
            if samples:
                return False
            expected = tuple(record.root for record in groups[0][1] if function.is_defined_at(record.root))
            if not expected:
                return type(value) is EmptySet and value.domain == Reals
            return (type(value) is FiniteSet and value.domain == Reals and len(value) == len(expected)
                    and all(compare_real(a, b) == 0 for a, b in zip(expected, value.values)))
        points = _points(groups)
        bounds = (None,) + points + (None,)
        if len(samples) != len(bounds) - 1:
            return False
        for lower, upper, (sample, sign) in zip(bounds, bounds[1:], samples):
            if (type(sample) is not Rational or type(sign) is not int or sign not in (-1, 0, 1)
                    or lower is not None and compare_real(sample, lower) <= 0
                    or upper is not None and compare_real(sample, upper) >= 0
                    or not function.is_defined_at(sample) or _sign(function, sample) != sign):
                return False
        expected = _assemble(function, points, samples,
                             "eq" if type(problem) is Eq else problem.relation)
        return equal_interval_sets(value, expected)
