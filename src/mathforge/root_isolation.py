"""Exact real root isolation over Q, with signed Sturm chains.

The producers run on dense integer polynomials (mathforge.kernels.upoly): v0.7's signed primitive Sturm chain of the
square-free part, its bisection from the root bound and its refinements, so that chains, isolating intervals and
root records are those of v0.7. The membership and completeness checkers (verify_root_records, verify_isolator)
count roots with the independent Sturm chains of mathforge.certify.upoly and never call ``real_roots``.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from math import gcd

from .errors import InvalidInput, UnsupportedOperation
from .limits import computation
from .model import Rational, Reals, Symbol, sqrt, trusted_rational
from .polynomial import Polynomial, trusted_polynomial


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


# ---------------------------------------------------------------------------------------------------------------
# The integer forms the kernels work on

def _integer(poly):
    """The primitive integer coefficients (positive leading coefficient) of a nonzero Polynomial."""
    from .trusted import upoly as tp
    return tp.primitive(tp.from_rationals(poly.coefficients)[0])


def _chain(poly):
    """The integer Sturm chain (v0.7's normalization) of a Polynomial's square-free part, kept for the work."""
    from .kernels import upoly as kernels
    cache = _CACHE.get()
    key = ("chain", poly)
    if cache is not None and key in cache:
        return cache[key]
    if poly.is_zero:
        raise InvalidInput("The zero polynomial has no finite root set", code="non_finite_root_set")
    chain = kernels.sturm_chain(kernels.squarefree_part(_integer(poly)))
    if cache is not None:
        cache[key] = chain
    return chain


def _as_polynomial(values, variable):
    return trusted_polynomial(tuple(trusted_rational(value, 1) for value in values), variable)


def _as_monic(values, variable):
    from .trusted import upoly as tp
    return trusted_polynomial(tp.monic_rationals(values), variable)


def _point(value):
    return value.numerator, value.denominator


def _rational_pair(point):
    return trusted_rational(point[0], point[1])


def _is_root(chain, value):
    from .kernels import upoly as kernels
    return kernels._value(chain[0], value.numerator, value.denominator) == 0


# ---------------------------------------------------------------------------------------------------------------
# Sturm chains, counts and isolation

def sturm_sequence(poly, *, limits=None):
    """Return a signed primitive Sturm chain for the square-free part."""
    if type(poly) is not Polynomial:
        raise InvalidInput("Sturm input must be a Polynomial")
    with root_work(limits):
        key = ("sturm", poly)
        cache = _CACHE.get()
        if key in cache:
            return cache[key]
        result = tuple(_as_polynomial(member, poly.variable) for member in _chain(poly))
        cache[key] = result
        return result


def _variations(chain, at=None, *, positive_infinity=False):
    """Sign variations of an integer chain at a Rational, or at -infinity / +infinity for at=None."""
    from .kernels import upoly as kernels
    if at is None:
        return kernels.variations_at_infinity(chain, positive_infinity)
    return kernels.endpoint(chain, at.numerator, at.denominator)[0]


def _count_open(chain, lower, upper):
    """Roots of an integer chain's first member in the open interval (lower, upper); None is infinite."""
    from .kernels import upoly as kernels
    return kernels.count_open(chain, None if lower is None else _point(lower),
                              None if upper is None else _point(upper))


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
        chain = _chain(poly)
        if lower is not None and lower == upper:
            return int(include_lower and include_upper and _is_root(chain, lower))
        count = _count_open(chain, lower, upper)
        if include_lower and _is_root(chain, lower):
            count += 1
        if include_upper and _is_root(chain, upper):
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
    from .kernels import upoly as kernels
    if type(poly) is not Polynomial:
        raise InvalidInput("Root isolation requires a Polynomial")
    with root_work(limits):
        cache = _CACHE.get()
        key = ("isolate", poly)
        if key in cache:
            return cache[key]
        chain = _chain(poly)
        if len(chain[0]) <= 1:
            return ()
        result = tuple((_rational_pair(lower), _rational_pair(upper))
                       for lower, upper in kernels.isolate(chain[0]))
        cache[key] = result
        return result


def refine_interval(poly, interval, *, limits=None):
    """Bisect a certified isolator once; exact rational roots become points."""
    from .kernels import upoly as kernels
    with root_work(limits) as budget:
        lower, upper = interval
        if lower == upper:
            budget.refine()
            return interval
        refined = kernels.refine(_chain(poly), _point(lower), _point(upper))
        return _rational_pair(refined[0]), _rational_pair(refined[1])


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


def _rational_root(integer, chain, interval):
    """reconstruct_rational_root on integers: (rational root or None, refined isolator as integer pairs) for the
    square-free primitive integer polynomial whose chain is given, over an isolating interval of one of its roots:
    refine below width 1 / (2 lc**2), then test the simplest fraction inside."""
    from .kernels import upoly as kernels
    lower, upper = interval
    if lower == upper:
        return _rational_pair(lower), interval
    bound = abs(integer[-1])
    limit = 2 * bound * bound
    while lower != upper and (upper[0] * lower[1] - lower[0] * upper[1]) * limit >= upper[1] * lower[1]:
        lower, upper = kernels.refine(chain, lower, upper)
    if lower == upper:
        return _rational_pair(lower), (lower, upper)
    candidate = _simplest_open_fraction(_rational_pair(lower), _rational_pair(upper))
    if candidate.denominator <= bound and not kernels._value(integer, candidate.numerator, candidate.denominator):
        return candidate, ((candidate.numerator, candidate.denominator),) * 2
    return None, (lower, upper)


def reconstruct_rational_root(poly, interval, *, limits=None):
    """Return (rational root or None, refined isolator), without heuristics."""
    with root_work(limits):
        lower, upper = interval
        if lower == upper:
            if evaluate_rational(poly, lower).numerator != 0:
                raise InvalidInput("A point isolator must be an exact root")
            return lower, (lower, upper)
        root, refined = _rational_root(_integer(poly), _chain(poly), (_point(lower), _point(upper)))
        if root is not None:
            return root, (root, root)
        return None, (_rational_pair(refined[0]), _rational_pair(refined[1]))


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
    """All real roots with multiplicities; rational roots are always recognized.

    The roots of all square-free factors are isolated together, as the roots of the square-free part, so that the
    records come out in increasing order. Each factor of degree at most two gives its closed forms; a larger one
    gives its rational roots, then its residual's closed forms or root descriptors, as in v0.7."""
    from .algebraic import trusted_real_algebraic_root, trusted_root_record
    from .kernels import upoly as kernels
    from .trusted import upoly as tp
    if type(poly) is not Polynomial:
        raise InvalidInput("real_roots requires a Polynomial")
    if poly.variable.domain is not Reals or poly.variable.assumptions:
        raise UnsupportedOperation("Root solving requires an unconstrained real variable")
    with root_work(limits) as budget:
        budget.inspect(poly)
        if poly.is_zero:
            raise InvalidInput("The zero polynomial has no finite root set", code="non_finite_root_set")
        integer = _integer(poly)
        if len(integer) <= 1:
            budget.evidence(0)
            return ()
        factors = kernels.square_free_integer(integer)
        part = (1,)
        for factor, _ in factors:
            part = tp.mul(part, factor)
        intervals = kernels.isolate(part)
        owners = []
        for lower, upper in intervals:
            for position, (factor, _) in enumerate(factors):
                if lower == upper:
                    if not kernels._value(factor, *lower):
                        break
                elif (kernels._sign(kernels._value(factor, *lower))
                      != kernels._sign(kernels._value(factor, *upper))):
                    break
            else:
                raise ArithmeticError("An isolated root belongs to no square-free factor")
            owners.append(position)
        values = [None] * len(intervals)
        for position, (factor, multiplicity) in enumerate(factors):
            places = [index for index, owner in enumerate(owners) if owner == position]
            if len(factor) <= 3:
                for place, root in zip(places, _small_roots(_as_monic(factor, poly.variable))):
                    values[place] = root
                continue
            chain = kernels.sturm_chain(factor)
            irrational = []
            residual = factor
            for place in places:
                root, _ = _rational_root(factor, chain, intervals[place])
                if root is None:
                    irrational.append(place)
                else:
                    values[place] = root
                    residual = tp.exact_quotient(residual, (-root.numerator, root.denominator))
            residual = tp.primitive(residual)
            if len(residual) <= 3:
                for place, root in zip(irrational, _small_roots(_as_monic(residual, poly.variable))):
                    values[place] = root
            else:
                for index, place in enumerate(irrational):
                    values[place] = trusted_real_algebraic_root(residual, index)
        if any(value is None for value in values):
            raise ArithmeticError("A real root was left without a value")
        records = tuple(trusted_root_record(value, factors[owner][1]) for value, owner in zip(values, owners))
        budget.evidence(len(records))
        return records


# ---------------------------------------------------------------------------------------------------------------
# Checkers (mathforge.certify.upoly)

def verify_root_records(poly, records, *, limits=None):
    """Independent membership, multiplicity, ordering and completeness check."""
    from .algebraic import RootRecord
    from .certify import upoly as certify
    with root_work(limits):
        if type(poly) is not Polynomial or poly.is_zero:
            return False
        records = tuple(records)
        if any(type(record) is not RootRecord for record in records):
            return False
        return certify.check_root_records(_integer(poly), records)


def verify_isolator(value, interval, *, limits=None):
    """Bind a point/open isolator to the actual descriptor's root index."""
    from .algebraic import RealAlgebraicRoot, as_endpoint
    from .certify import upoly as certify
    from .trusted import upoly as tp
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
        polynomial = certify.squarefree_part(tp.primitive(value.integer_coefficients))
        if lower == upper:
            return (tp.sign_at(polynomial, lower.numerator, lower.denominator) == 0 and
                    certify.count_roots(polynomial, None, lower) - 1 == value.real_index)
        return (tp.sign_at(polynomial, lower.numerator, lower.denominator) != 0 and
                tp.sign_at(polynomial, upper.numerator, upper.denominator) != 0 and
                certify.count_roots(polynomial, None, lower) == value.real_index and
                certify.count_roots(polynomial, lower, upper) == 1)
