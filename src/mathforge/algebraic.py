"""Immutable exact real root descriptors and bounded exact comparisons.

Equality and hashing remain structural. ``compare_real`` provides mathematical
comparison without promising general algebraic-number field arithmetic.
"""

from dataclasses import dataclass
from math import isqrt

from .errors import InvalidInput, UnsupportedOperation
from .model import Add, Expression, Mul, Pow, Rational, Sqrt, as_expression
from .polynomial import Polynomial


@dataclass(frozen=True)
class RealAlgebraicRoot(Expression):
    integer_coefficients: tuple[int, ...]
    real_index: int

    def __post_init__(self):
        from .root_isolation import (ROOT_VARIABLE, count_real_roots,
                                     primitive_integer_coefficients, root_work)
        coefficients = tuple(self.integer_coefficients)
        if not coefficients or any(type(coefficient) is not int for coefficient in coefficients):
            raise InvalidInput("Algebraic root coefficients must be exact integers")
        if type(self.real_index) is not int or self.real_index < 0:
            raise InvalidInput("Algebraic root index must be a nonnegative integer")
        with root_work() as budget:
            budget.nodes(len(coefficients))
            for coefficient in coefficients:
                budget.check_int(coefficient)
            polynomial = Polynomial(coefficients, ROOT_VARIABLE)
            if polynomial.degree <= 0:
                raise InvalidInput("An algebraic root requires a nonconstant polynomial")
            polynomial = polynomial.exact_div(polynomial.gcd(polynomial.derivative()))
            coefficients = primitive_integer_coefficients(polynomial)
            polynomial = Polynomial(coefficients, ROOT_VARIABLE)
            if self.real_index >= count_real_roots(polynomial):
                raise InvalidInput("Algebraic root index does not identify a real root")
            object.__setattr__(self, "integer_coefficients", coefficients)

    def __str__(self):
        return f"RootOf({self.integer_coefficients}, {self.real_index})"


@dataclass(frozen=True)
class RootRecord:
    root: Expression
    multiplicity: int

    def __post_init__(self):
        root = as_expression(self.root)
        # Scope validation precedes all structural equality shortcuts.
        as_endpoint(root)
        if type(self.multiplicity) is not int or self.multiplicity <= 0:
            raise InvalidInput("Root multiplicity must be a positive integer")
        object.__setattr__(self, "root", root)


def _exact_sqrt(value):
    if value.numerator < 0:
        return None
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return Rational(numerator, denominator)
    return None


def _quadratic_coordinates(expression):
    """Recognize a constant in one quadratic field as u + v sqrt(d)."""
    from .root_isolation import root_work
    with root_work() as budget:
        radicands = []
        pending = [expression]
        while pending:
            budget.nodes()
            node = pending.pop()
            if type(node) is Rational:
                continue
            if type(node) is Sqrt:
                if _exact_sqrt(node.radicand) is None:
                    radicands.append(node.radicand)
                continue
            if type(node) is Add:
                pending.extend(node.terms)
            elif type(node) is Mul:
                pending.extend(node.factors)
            elif type(node) is Pow:
                pending.append(node.base)
            else:
                raise UnsupportedOperation("Expected a rational, root descriptor or one quadratic-field constant")
        d = radicands[0] if radicands else Rational(0)
        zero, one = Rational(0), Rational(1)

        def product(a, b):
            return a[0] * b[0] + a[1] * b[1] * d, a[0] * b[1] + a[1] * b[0]

        def visit(node):
            budget.nodes()
            if type(node) is Rational:
                return node, zero
            if type(node) is Sqrt:
                rational = _exact_sqrt(node.radicand)
                if rational is not None:
                    return rational, zero
                ratio = _exact_sqrt(node.radicand / d)
                if ratio is None:
                    raise UnsupportedOperation("Independent quadratic fields are outside comparison scope")
                return zero, ratio
            if type(node) is Add:
                total = (zero, zero)
                for child in node.terms:
                    pair = visit(child)
                    total = total[0] + pair[0], total[1] + pair[1]
                return total
            if type(node) is Mul:
                total = (one, zero)
                for child in node.factors:
                    total = product(total, visit(child))
                return total
            base, exponent, result = visit(node.base), node.exponent, (one, zero)
            while exponent:
                if exponent & 1:
                    result = product(result, base)
                exponent >>= 1
                if exponent:
                    base = product(base, base)
            return result

        u, v = visit(expression)
        return u, v, d


def as_endpoint(value, *, limits=None):
    """Convert supported exact constants to Rational or root descriptor."""
    from .root_isolation import ROOT_VARIABLE, primitive_integer_coefficients, root_work
    with root_work(limits) as budget:
        budget.inspect(value)
        if type(value) is int:
            return Rational(value)
        if type(value) in (Rational, RealAlgebraicRoot):
            return value
        value = as_expression(value)
        u, v, d = _quadratic_coordinates(value)
        if v.numerator == 0:
            return u
        polynomial = Polynomial((u*u-v*v*d, -2*u, 1), ROOT_VARIABLE)
        return RealAlgebraicRoot(primitive_integer_coefficients(polynomial),
                                 0 if v.numerator < 0 else 1)


def _polynomial(value):
    from .root_isolation import ROOT_VARIABLE
    return Polynomial(value.integer_coefficients, ROOT_VARIABLE)


def interval_for(value, *, limits=None):
    from .root_isolation import isolate_real_roots, root_work
    with root_work(limits):
        endpoint = as_endpoint(value)
        if type(endpoint) is Rational:
            return endpoint, endpoint
        return isolate_real_roots(_polynomial(endpoint))[endpoint.real_index]


def rational_between(left, right, *, limits=None):
    """A rational strictly inside an exact open cell; None denotes infinity."""
    from .root_isolation import refine_interval, root_work
    with root_work(limits):
        left = None if left is None else as_endpoint(left)
        right = None if right is None else as_endpoint(right)
        if left is None and right is None:
            return Rational(0)
        if left is None:
            lower = interval_for(right)[0]
            return Rational(lower.numerator // lower.denominator - 1)
        if right is None:
            upper = interval_for(left)[1]
            return Rational(upper.numerator // upper.denominator + 1)
        if compare_real(left, right) >= 0:
            raise InvalidInput("An open cell requires strictly ordered endpoints")
        a, b = interval_for(left), interval_for(right)
        while a[1] >= b[0]:
            if a[0] != a[1]:
                a = refine_interval(_polynomial(left), a)
            if b[0] != b[1]:
                b = refine_interval(_polynomial(right), b)
        return (a[1] + b[0]) / 2


def compare_real(left, right, *, limits=None):
    """Return -1, 0 or 1; do not alter Python structural equality/hash."""
    from .root_isolation import (count_real_roots, evaluate_rational,
                                 refine_interval, root_work)
    with root_work(limits) as budget:
        left, right = as_endpoint(left), as_endpoint(right)
        if left == right:
            return 0
        if type(left) is Rational and type(right) is Rational:
            return (left > right) - (left < right)
        a, b = interval_for(left), interval_for(right)
        left_poly = None if type(left) is Rational else _polynomial(left)
        right_poly = None if type(right) is Rational else _polynomial(right)
        common = left_poly.gcd(right_poly) if left_poly is not None and right_poly is not None else None
        while True:
            budget.tick()
            if a[1] < b[0] or (a[1] == b[0] and a[0] != a[1] and b[0] != b[1]):
                return -1
            if b[1] < a[0] or (b[1] == a[0] and b[0] != b[1] and a[0] != a[1]):
                return 1
            if a[0] == a[1] and b[0] == b[1]:
                return (a[0] > b[0]) - (a[0] < b[0])
            if a[0] == a[1]:
                if b[0] < a[0] < b[1] and evaluate_rational(right_poly, a[0]).numerator == 0:
                    return 0
            elif b[0] == b[1]:
                if a[0] < b[0] < a[1] and evaluate_rational(left_poly, b[0]).numerator == 0:
                    return 0
            elif common is not None and common.degree > 0:
                lower, upper = max(a[0], b[0]), min(a[1], b[1])
                if lower < upper and count_real_roots(common, lower, upper) == 1:
                    return 0
            if a[0] != a[1]:
                a = refine_interval(left_poly, a)
            if b[0] != b[1]:
                b = refine_interval(right_poly, b)


def _interval_product(left, right):
    products = tuple(a*b for a in left for b in right)
    return min(products), max(products)


def _interval_evaluate(poly, interval):
    result = Rational(0), Rational(0)
    for coefficient in reversed(poly.coefficients):
        lower, upper = _interval_product(result, interval)
        result = lower + coefficient, upper + coefficient
    return result


def sign_at(poly, value, *, limits=None):
    """Exact sign of a rational polynomial at a supported exact real."""
    from .root_isolation import (ROOT_VARIABLE, count_real_roots, evaluate_rational,
                                 refine_interval, root_work)
    if type(poly) is not Polynomial:
        raise InvalidInput("sign_at requires a Polynomial")
    with root_work(limits) as budget:
        value = as_endpoint(value)
        if type(value) is Rational:
            numerator = evaluate_rational(poly, value).numerator
            return (numerator > 0) - (numerator < 0)
        if poly.is_zero:
            return 0
        defining = _polynomial(value)
        target = Polynomial(poly.coefficients, ROOT_VARIABLE)
        common = defining.gcd(target)
        interval = interval_for(value)
        if interval[0] == interval[1]:
            numerator = evaluate_rational(target, interval[0]).numerator
            return (numerator > 0) - (numerator < 0)
        if common.degree > 0 and count_real_roots(common, *interval) == 1:
            return 0
        while True:
            budget.tick()
            lower, upper = _interval_evaluate(target, interval)
            if lower > Rational(0):
                return 1
            if upper < Rational(0):
                return -1
            interval = refine_interval(defining, interval)
            if interval[0] == interval[1]:
                numerator = evaluate_rational(target, interval[0]).numerator
                return (numerator > 0) - (numerator < 0)
