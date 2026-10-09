"""Exact real root isolation over Q, with signed Sturm chains.

The producer and the membership/completeness checkers share exact primitives,
but the checkers never call ``real_roots``.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import cmp_to_key
from math import gcd

from .errors import InvalidInput, UnsupportedOperation
from .limits import computation
from .model import Rational, Reals, Symbol, sqrt
from .polynomial import Polynomial


ROOT_VARIABLE = Symbol("_root", uid="00000000-0000-4000-8000-000000000002")
_CACHE = ContextVar("mathforge_root_work_cache", default=None)


@contextmanager
def root_work(limits=None):
    with computation(limits) as budget:
        token = _CACHE.set({}) if _CACHE.get() is None else None
        try:
            yield budget
        finally:
            if token is not None:
                _CACHE.reset(token)


def _rational(value):
    if type(value) is int:
        return Rational(value)
    if type(value) is not Rational:
        raise InvalidInput("Root bounds must be exact rationals")
    return value


def evaluate_rational(poly, value):
    """Horner evaluation without interpreting the symbol's metadata."""
    result = Rational(0)
    for coefficient in reversed(poly.coefficients):
        result = result * value + coefficient
    return result


def primitive_integer_coefficients(poly, *, positive_leading=True):
    """Clear rational denominators and content; optionally preserve sign."""
    with root_work() as budget:
        denominator = 1
        for value in poly.coefficients:
            budget.tick()
            budget.check_int(value.numerator)
            budget.check_int(value.denominator)
            factor = denominator // gcd(denominator, value.denominator)
            budget.product(factor, value.denominator)
            denominator = factor * value.denominator
            budget.check_int(denominator)
        values = []
        for value in poly.coefficients:
            budget.product(value.numerator, denominator // value.denominator)
            integer = value.numerator * (denominator // value.denominator)
            budget.check_int(integer)
            values.append(integer)
        content = 0
        for integer in values:
            content = gcd(content, abs(integer))
        if not content:
            return (0,)
        sign = -1 if positive_leading and values[-1] < 0 else 1
        return tuple(sign * value // content for value in values)


def _square_free_part(poly):
    if poly.is_zero:
        raise InvalidInput("The zero polynomial has no finite root set", code="non_finite_root_set")
    if poly.degree <= 0:
        return Polynomial((1,), poly.variable)
    return poly.exact_div(poly.gcd(poly.derivative())).monic()


def sturm_sequence(poly, *, limits=None):
    """Return a signed primitive Sturm chain for the square-free part."""
    if type(poly) is not Polynomial:
        raise InvalidInput("Sturm input must be a Polynomial")
    with root_work(limits):
        key = ("sturm", poly)
        cache = _CACHE.get()
        if key in cache:
            return cache[key]
        part = _square_free_part(poly)
        first = Polynomial(primitive_integer_coefficients(part), poly.variable)
        if first.degree <= 0:
            result = (first,)
        else:
            chain = [first, first.derivative()]
            while not chain[-1].is_zero:
                remainder = -chain[-2].divmod(chain[-1])[1]
                if remainder.is_zero:
                    break
                # Positive rescaling only: making a negative-leading remainder
                # monic would invalidate Sturm sign variations.
                chain.append(Polynomial(primitive_integer_coefficients(
                    remainder, positive_leading=False), poly.variable))
            result = tuple(chain)
        cache[key] = result
        return result


def _variations(chain, at=None, *, positive_infinity=False):
    signs = []
    for polynomial in chain:
        if at is None:
            sign = 1 if polynomial.leading_coefficient.numerator > 0 else -1
            if not positive_infinity and polynomial.degree % 2:
                sign = -sign
        else:
            value = evaluate_rational(polynomial, at).numerator
            sign = (value > 0) - (value < 0)
        if sign:
            signs.append(sign)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _count_open(chain, lower, upper):
    count = _variations(chain, lower) - _variations(
        chain, upper, positive_infinity=upper is None)
    if upper is not None and evaluate_rational(chain[0], upper).numerator == 0:
        count -= 1
    return count


def count_real_roots(poly, lower=None, upper=None, *, include_lower=False,
                     include_upper=False, limits=None):
    """Count distinct roots, with explicitly open/closed finite endpoints."""
    if type(poly) is not Polynomial:
        raise InvalidInput("Root counting requires a Polynomial")
    if type(include_lower) is not bool or type(include_upper) is not bool:
        raise InvalidInput("Root endpoint flags must be booleans")
    lower = None if lower is None else _rational(lower)
    upper = None if upper is None else _rational(upper)
    if (lower is None and include_lower) or (upper is None and include_upper):
        raise InvalidInput("Infinite root-count bounds cannot be closed")
    if lower is not None and upper is not None and lower > upper:
        raise InvalidInput("Root-count bounds are reversed")
    with root_work(limits) as budget:
        budget.inspect(poly)
        chain = sturm_sequence(poly)
        if lower is not None and lower == upper:
            return int(include_lower and include_upper and
                       evaluate_rational(chain[0], lower).numerator == 0)
        count = _count_open(chain, lower, upper)
        if include_lower and evaluate_rational(chain[0], lower).numerator == 0:
            count += 1
        if include_upper and evaluate_rational(chain[0], upper).numerator == 0:
            count += 1
        return count


def _root_bound(poly):
    largest = 0
    for coefficient in poly.coefficients[:-1]:
        ratio = abs(coefficient / poly.leading_coefficient)
        largest = max(largest, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
    return Rational(largest + 2)


def isolate_real_roots(poly, *, limits=None):
    """Disjoint sorted rational points/open intervals for all distinct roots."""
    if type(poly) is not Polynomial:
        raise InvalidInput("Root isolation requires a Polynomial")
    with root_work(limits) as budget:
        cache = _CACHE.get()
        key = ("isolate", poly)
        if key in cache:
            return cache[key]
        chain = sturm_sequence(poly)
        polynomial = chain[0]
        if polynomial.degree <= 0:
            return ()
        bound = _root_bound(polynomial)
        pending = [(-bound, bound, _count_open(chain, -bound, bound))]
        intervals = []
        while pending:
            budget.refine()
            lower, upper, count = pending.pop()
            if not count:
                continue
            lower_zero = evaluate_rational(polynomial, lower).numerator == 0
            upper_zero = evaluate_rational(polynomial, upper).numerator == 0
            if count == 1 and not lower_zero and not upper_zero:
                intervals.append((lower, upper))
                continue
            middle = (lower + upper) / 2
            middle_zero = evaluate_rational(polynomial, middle).numerator == 0
            if middle_zero:
                intervals.append((middle, middle))
            left_count = _count_open(chain, lower, middle)
            right_count = _count_open(chain, middle, upper)
            if left_count + right_count + int(middle_zero) != count:
                raise ArithmeticError("Sturm subdivision did not preserve the root count")
            if right_count:
                pending.append((middle, upper, right_count))
            if left_count:
                pending.append((lower, middle, left_count))
        intervals.sort(key=lambda interval: interval[0])
        result = tuple(intervals)
        cache[key] = result
        return result


def refine_interval(poly, interval, *, limits=None):
    """Bisect a certified isolator once; exact rational roots become points."""
    with root_work(limits) as budget:
        budget.refine()
        lower, upper = interval
        if lower == upper:
            return interval
        middle = (lower + upper) / 2
        chain = sturm_sequence(poly)
        if evaluate_rational(chain[0], middle).numerator == 0:
            return middle, middle
        return (lower, middle) if _count_open(chain, lower, middle) else (middle, upper)


def _simplest_open_fraction(lower, upper):
    """Minimum-denominator fraction strictly between rational endpoints.

    Continued-fraction reciprocal steps avoid enumerating denominators.
    """
    if not lower < upper:
        raise InvalidInput("Rational reconstruction requires a nonempty interval")
    if lower < Rational(0) < upper:
        return Rational(0)
    if upper <= Rational(0):
        return -_simplest_open_fraction(-upper, -lower)
    with root_work() as budget:
        prefix = []
        while True:
            budget.refine()
            integer = lower.numerator // lower.denominator
            candidate = Rational(integer + 1)
            if candidate < upper:
                break
            low, high = lower - integer, upper - integer
            if low.numerator == 0:
                reciprocal = Rational(1) / high
                denominator = reciprocal.numerator // reciprocal.denominator + 1
                candidate = Rational(integer) + Rational(1, denominator)
                break
            prefix.append(integer)
            lower, upper = Rational(1) / high, Rational(1) / low
        for integer in reversed(prefix):
            candidate = Rational(integer) + Rational(1) / candidate
        return candidate


def reconstruct_rational_root(poly, interval, *, limits=None):
    """Return (rational root or None, refined isolator), without heuristics."""
    with root_work(limits) as budget:
        lower, upper = interval
        if lower == upper:
            if evaluate_rational(poly, lower).numerator != 0:
                raise InvalidInput("A point isolator must be an exact root")
            return lower, (lower, upper)
        coefficients = primitive_integer_coefficients(poly)
        bound = abs(coefficients[-1])
        budget.product(2, bound, bound)
        width = Rational(1, 2 * bound * bound)
        while lower != upper and upper - lower >= width:
            lower, upper = refine_interval(poly, (lower, upper))
        if lower == upper:
            return lower, (lower, upper)
        candidate = _simplest_open_fraction(lower, upper)
        if candidate.denominator <= bound and evaluate_rational(poly, candidate).numerator == 0:
            return candidate, (candidate, candidate)
        return None, (lower, upper)


def _small_roots(poly):
    if poly.degree <= 0:
        return ()
    if poly.degree == 1:
        return (-poly.coefficients[0] / poly.coefficients[1],)
    c, b, a = poly.coefficients
    discriminant = b * b - 4 * a * c
    if discriminant < Rational(0):
        return ()
    center = -b / (2 * a)
    if discriminant == Rational(0):
        return (center,)
    radius = sqrt(discriminant / (4 * a * a))
    return center - radius, center + radius


def real_roots(poly, *, limits=None):
    """All real roots with multiplicities; rational roots are always recognized."""
    from .algebraic import RealAlgebraicRoot, RootRecord, compare_real
    if type(poly) is not Polynomial:
        raise InvalidInput("real_roots requires a Polynomial")
    if poly.variable.domain is not Reals or poly.variable.assumptions:
        raise UnsupportedOperation("Root solving requires an unconstrained real variable")
    with root_work(limits) as budget:
        budget.inspect(poly)
        if poly.is_zero:
            raise InvalidInput("The zero polynomial has no finite root set", code="non_finite_root_set")
        records = []
        for factor in poly.square_free().factors:
            polynomial, multiplicity = factor.polynomial, factor.multiplicity
            if polynomial.degree <= 2:
                records.extend(RootRecord(root, multiplicity) for root in _small_roots(polynomial))
                continue
            intervals = isolate_real_roots(polynomial)
            rationals, irrational_intervals = [], []
            for interval in intervals:
                root, refined = reconstruct_rational_root(polynomial, interval)
                if root is None:
                    irrational_intervals.append(refined)
                else:
                    rationals.append(root)
            residual = polynomial
            for root in rationals:
                residual = residual.exact_div(Polynomial((-root, 1), polynomial.variable))
                records.append(RootRecord(root, multiplicity))
            if residual.degree <= 2:
                records.extend(RootRecord(root, multiplicity) for root in _small_roots(residual))
            else:
                coefficients = primitive_integer_coefficients(residual)
                for index in range(len(irrational_intervals)):
                    records.append(RootRecord(RealAlgebraicRoot(coefficients, index), multiplicity))
        records.sort(key=cmp_to_key(lambda a, b: compare_real(a.root, b.root)))
        budget.evidence(len(records))
        return tuple(records)


def verify_root_records(poly, records, *, limits=None):
    """Independent membership, multiplicity, ordering and completeness check."""
    from .algebraic import RootRecord, compare_real, sign_at
    with root_work(limits):
        if type(poly) is not Polynomial or poly.is_zero:
            return False
        records = tuple(records)
        if any(type(record) is not RootRecord for record in records):
            return False
        if len(records) != count_real_roots(poly):
            return False
        for left, right in zip(records, records[1:]):
            if compare_real(left.root, right.root) >= 0:
                return False
        for record in records:
            derivative = poly
            multiplicity = 0
            while derivative.degree >= 0 and sign_at(derivative, record.root) == 0:
                multiplicity += 1
                derivative = derivative.derivative()
            if multiplicity != record.multiplicity:
                return False
        return True


def verify_isolator(value, interval, *, limits=None):
    """Bind a point/open isolator to the actual descriptor's root index."""
    from .algebraic import RealAlgebraicRoot, as_endpoint
    with root_work(limits):
        value = as_endpoint(value)
        if type(interval) is not tuple or len(interval) != 2:
            return False
        lower, upper = interval
        if type(lower) is not Rational or type(upper) is not Rational or lower > upper:
            return False
        if type(value) is Rational:
            return lower == upper == value
        if type(value) is not RealAlgebraicRoot:
            return False
        polynomial = Polynomial(value.integer_coefficients, ROOT_VARIABLE)
        if lower == upper:
            return (evaluate_rational(polynomial, lower).numerator == 0 and
                    count_real_roots(polynomial, upper=lower) == value.real_index)
        return (evaluate_rational(polynomial, lower).numerator != 0 and
                evaluate_rational(polynomial, upper).numerator != 0 and
                count_real_roots(polynomial, upper=lower) == value.real_index and
                count_real_roots(polynomial, lower, upper) == 1)
