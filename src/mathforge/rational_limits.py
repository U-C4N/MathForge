"""Bounded limits of real univariate rational functions and their certificates.

Evidence is ``("rational_limit_v1", function, point, side, certificate)``.
The source function retains all domain exclusions.  A finite-point certificate
is ``("regular",)`` or ``("pole", order, residual_denominator)``; at infinity
it is ``("infinity",)``.  The checker verifies the pole factorization rather
than calling the producer or replaying its repeated-division algorithm.
"""

from __future__ import annotations

from .analysis_types import LimitValue
from .errors import InvalidInput, UnsupportedOperation
from .limits import computation
from .model import Expression, Rational
from .polynomial import Polynomial
from .rational_function import as_rational_function, validate_variable


def _point_and_side(point, side):
    if type(side) is not str or side not in ("both", "left", "right"):
        raise InvalidInput("Limit side must be 'both', 'left' or 'right'")
    if type(point) is int:
        point = Rational(point)
    if type(point) is Rational:
        return point
    if type(point) is str:
        if point not in ("-inf", "+inf"):
            raise InvalidInput("Infinite limit points must be '-inf' or '+inf'")
        if side != "both":
            raise InvalidInput("Limits at infinity require side='both'")
        return point
    if isinstance(point, Expression):
        raise UnsupportedOperation("Finite limit points must be exact rationals")
    raise InvalidInput("A limit point must be an exact integer, rational or signed infinity")


def _signed_infinity(sign):
    return LimitValue("positive_infinity" if sign > 0 else "negative_infinity", None)


def limit_value(expression, variable, point, side="both", *, limits=None):
    """Produce a limit classification and source-bound exact evidence."""
    with computation(limits) as budget:
        budget.inspect((expression, variable, point, side))
        validate_variable(variable)
        point = _point_and_side(point, side)
        function = as_rational_function(expression, variable)
        numerator, denominator = function.numerator, function.denominator
        if type(point) is str:
            certificate = ("infinity",)
            if numerator.is_zero or numerator.degree < denominator.degree:
                value = LimitValue("finite", Rational(0))
            elif numerator.degree == denominator.degree:
                value = LimitValue("finite", numerator.leading_coefficient / denominator.leading_coefficient)
            else:
                sign = 1 if numerator.leading_coefficient.numerator > 0 else -1
                sign *= 1 if denominator.leading_coefficient.numerator > 0 else -1
                if point == "-inf" and (numerator.degree - denominator.degree) % 2:
                    sign = -sign
                value = _signed_infinity(sign)
        else:
            denominator_at = denominator.evaluate(point)
            numerator_at = numerator.evaluate(point)
            if denominator_at.numerator != 0:
                certificate = ("regular",)
                value = LimitValue("finite", numerator_at / denominator_at)
            else:
                factor = Polynomial((-point, Rational(1)), variable)
                residual, order = denominator, 0
                while residual.evaluate(point).numerator == 0:
                    budget.tick()
                    residual = residual.exact_div(factor)
                    order += 1
                certificate = ("pole", order, residual)
                coefficient = numerator_at / residual.evaluate(point)
                sign = 1 if coefficient.numerator > 0 else -1
                if side == "both" and order % 2:
                    value = LimitValue("does_not_exist", None)
                else:
                    if side == "left" and order % 2:
                        sign = -sign
                    value = _signed_infinity(sign)
        evidence = ("rational_limit_v1", function, point, side, certificate)
        budget.inspect((value, evidence), evidence=True)
        return value, evidence


def _at(polynomial, point):
    """Checker-only Horner evaluation using exact rational arithmetic."""
    result = Rational(0)
    for coefficient in reversed(polynomial.coefficients):
        result = result * point + coefficient
    return result


def _matches(value, kind, rational=None):
    return (type(value) is LimitValue and value.kind == kind
            and (type(value.value) is Rational and value.value == rational
                 if kind == "finite" else value.value is None))


def check_limit(expression, variable, point, side, value, evidence, *, limits=None):
    """Check a claimed limit without using the producer's limit algorithm.

    Malformed or incorrect claims return False. Resource exhaustion propagates
    to the caller, which must report an unknown verification rather than false.
    """
    with computation(limits) as budget:
        budget.inspect((expression, variable, point, side))
        budget.inspect((value, evidence), evidence=True)
        try:
            validate_variable(variable)
            point = _point_and_side(point, side)
            source = as_rational_function(expression, variable)
            if (type(evidence) is not tuple or len(evidence) != 5
                    or evidence[0] != "rational_limit_v1"
                    or type(evidence[1]) is not type(source) or evidence[1] != source
                    or type(evidence[2]) is not type(point) or evidence[2] != point
                    or type(evidence[3]) is not str or evidence[3] != side
                    or type(evidence[4]) is not tuple):
                return False
            certificate = evidence[4]
            numerator, denominator = source.numerator, source.denominator
            if type(point) is str:
                if certificate != ("infinity",):
                    return False
                if numerator.is_zero:
                    return _matches(value, "finite", Rational(0))
                # Inspect coefficient lengths and signs independently of the
                # producer's degree/leading-coefficient convenience properties.
                difference = len(numerator.coefficients) - len(denominator.coefficients)
                if difference < 0:
                    return _matches(value, "finite", Rational(0))
                leading_ratio = numerator.coefficients[-1] / denominator.coefficients[-1]
                if difference == 0:
                    return _matches(value, "finite", leading_ratio)
                negative = leading_ratio.numerator < 0
                if point == "-inf" and difference % 2:
                    negative = not negative
                return _matches(value, "negative_infinity" if negative else "positive_infinity")
            denominator_at = _at(denominator, point)
            numerator_at = _at(numerator, point)
            if denominator_at.numerator != 0:
                return (certificate == ("regular",)
                        and _matches(value, "finite", numerator_at / denominator_at))
            if (len(certificate) != 3 or certificate[0] != "pole"
                    or type(certificate[1]) is not int
                    or not 1 <= certificate[1] <= denominator.degree
                    or type(certificate[2]) is not Polynomial
                    or certificate[2].variable != variable
                    or numerator_at.numerator == 0):
                return False
            order, residual = certificate[1], certificate[2]
            residual_at = _at(residual, point)
            if residual_at.numerator == 0:
                return False
            factor = Polynomial((-point, Rational(1)), variable)
            if factor ** order * residual != denominator:
                return False
            if side == "both" and order % 2:
                return _matches(value, "does_not_exist")
            negative = (numerator_at.numerator < 0) != (residual_at.numerator < 0)
            if side == "left" and order % 2:
                negative = not negative
            return _matches(value, "negative_infinity" if negative else "positive_infinity")
        except (InvalidInput, UnsupportedOperation, AttributeError, TypeError, ValueError, ZeroDivisionError):
            return False
