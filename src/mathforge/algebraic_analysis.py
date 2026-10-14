"""Algebraic endpoint calculus and exact values of real local extrema."""

from __future__ import annotations

from . import algebraic_arithmetic as arithmetic
from .algebraic import RealAlgebraicRoot, sign_at
from .analysis_types import ExtremumValue, LimitValue
from .errors import InvalidInput, UnsupportedOperation
from .model import Rational, as_expression
from .polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from .rational_function import RationalFunction, as_polynomial, as_rational_function, validate_variable
from .root_isolation import root_work


ZERO, ONE = Rational(0), Rational(1)
OPERATIONS = frozenset(("limit", "integrate", "extrema_values"))


def _at_expression(polynomial, point):
    result = ZERO
    for coefficient in reversed(polynomial.coefficients):
        result = result * point + coefficient
    return result


def _quotient_arguments(function, point):
    return dict(left=_at_expression(function.numerator, point),
                right=_at_expression(function.denominator, point))


def _checker_at(polynomial, point):
    """Checker-only coefficient recurrence; never use producer evaluation."""
    from .limits import current_budget
    value = polynomial.coefficients[-1]
    for degree in range(len(polynomial.coefficients)-2, -1, -1):
        current_budget().tick()
        value = polynomial.coefficients[degree] + point*value
    return value


def _checker_polynomial(expression, variable):
    """Read source coefficients without the producer's polynomial converter."""
    from .limits import ResourceLimitError, current_budget
    from .multivariate import _checked_sources, _scan
    validate_variable(variable)
    if type(expression) is Polynomial:
        if (expression.variable != variable or type(expression.coefficients) is not tuple or
                not expression.coefficients or
                Polynomial(expression.coefficients, variable) != expression):
            raise InvalidInput("Malformed or mismatched source polynomial")
        return expression
    expression = as_expression(expression)
    _scan((expression,), {variable.uid: variable})
    coefficients = _checked_sources((expression,), (variable,))[0]
    degree = max((powers[0] for powers in coefficients), default=0)
    if degree > MAX_POLYNOMIAL_DEGREE:
        raise ResourceLimitError("Source polynomial exceeds the dense degree budget")
    current_budget().nodes(degree + 1)
    return Polynomial(tuple(coefficients.get((index,), ZERO) for index in range(degree+1)), variable)


def _checker_rational_source(expression, variable):
    if type(expression) is not RationalFunction:
        return RationalFunction(_checker_polynomial(expression, variable), Polynomial((ONE,), variable))
    if type(expression.excluded) is not tuple:
        raise InvalidInput("Malformed source domain exclusions")
    rebuilt = RationalFunction(_checker_polynomial(expression.numerator, variable),
                               _checker_polynomial(expression.denominator, variable),
                               tuple(_checker_polynomial(p, variable) for p in expression.excluded))
    if rebuilt != expression:
        raise InvalidInput("Noncanonical rational-function source")
    return rebuilt


def _side(side):
    if type(side) is not str or side not in ("left", "right", "both"):
        raise InvalidInput("Limit side must be 'left', 'right' or 'both'")
    return side


def _infinite(sign, order, side):
    if side == "both" and order % 2:
        return LimitValue("does_not_exist")
    if side == "left" and order % 2:
        sign = -sign
    return LimitValue("negative_infinity" if sign < 0 else "positive_infinity")


def _limit(args):
    from .limits import current_budget
    variable, side = args["variable"], _side(args["side"])
    validate_variable(variable)
    if type(args["point"]) is str:
        from .rational_limits import limit_value
        value, evidence = limit_value(args["expression"], variable, args["point"], side)
        return value, ("legacy_limit", evidence)
    point, point_evidence = arithmetic.produce("algebraic", dict(expression=args["point"]))
    function = as_rational_function(args["expression"], variable)
    denominator_sign = sign_at(function.denominator, point)
    if denominator_sign:
        scalar, scalar_evidence = arithmetic.produce("algebraic_div", _quotient_arguments(function, point))
        value, certificate = LimitValue("finite", scalar), ("regular", scalar_evidence)
    else:
        derivative, order = function.denominator, 0
        while sign_at(derivative, point) == 0:
            current_budget().tick()
            derivative = derivative.derivative()
            order += 1
        sign = sign_at(function.numerator, point) * sign_at(derivative, point)
        value, certificate = _infinite(sign, order, side), ("pole", order)
    return value, ("algebraic_limit_v1", point, point_evidence, certificate)


def _integral(args):
    variable = args["variable"]
    validate_variable(variable)
    if type(args["expression"]) is RationalFunction:
        raise UnsupportedOperation("Rational-function integration is not supported")
    polynomial = as_polynomial(args["expression"], variable)
    primitive = Polynomial((ZERO,) + tuple(c/(index+1) for index, c in enumerate(polynomial.coefficients)), variable)
    lower, low_evidence = arithmetic.produce("algebraic", dict(expression=args["lower"]))
    upper, high_evidence = arithmetic.produce("algebraic", dict(expression=args["upper"]))
    low, low_value_evidence = arithmetic.produce("algebraic", dict(expression=_at_expression(primitive, lower)))
    high, high_value_evidence = arithmetic.produce("algebraic", dict(expression=_at_expression(primitive, upper)))
    value, difference_evidence = arithmetic.produce("algebraic", dict(expression=high-low))
    return value, ("algebraic_integral_v1", primitive,
                   (lower, low_evidence, low, low_value_evidence),
                   (upper, high_evidence, high, high_value_evidence), difference_evidence)


def _extrema(args):
    from .function_analysis import analyze_function
    analysis, analysis_evidence = analyze_function(args["expression"], args["variable"])
    values, certificates = [], []
    for extremum in analysis.strict_extrema:
        eval_args = dict(expression=analysis.function, bindings=((args["variable"], extremum.point),))
        value, evidence = arithmetic.produce("evaluate_algebraic", eval_args)
        values.append(ExtremumValue(extremum.point, value, extremum.kind))
        certificates.append(evidence)
    return tuple(values), ("extrema_values_v1", analysis, analysis_evidence, tuple(certificates))


def produce(operation, args, *, limits=None):
    with root_work(limits) as budget:
        budget.inspect(args)
        arithmetic.validate_exact_scalars(args)
        if operation == "limit":
            value, evidence = _limit(args)
        elif operation == "integrate":
            value, evidence = _integral(args)
        elif operation == "extrema_values":
            value, evidence = _extrema(args)
        else:
            raise UnsupportedOperation("Unknown algebraic analysis operation")
        budget.inspect((value, evidence), evidence=True)
        return value, evidence


def _derivative_coefficients(polynomial, order):
    from .limits import current_budget
    result = []
    for degree in range(order, len(polynomial.coefficients)):
        coefficient = polynomial.coefficients[degree]
        for factor in range(degree-order+1, degree+1):
            current_budget().tick()
            coefficient *= factor
        result.append(coefficient)
    return Polynomial(tuple(result), polynomial.variable)


def _check_limit(args, value, evidence):
    if type(value) is not LimitValue or type(evidence) is not tuple:
        return False
    variable, side = args["variable"], _side(args["side"])
    validate_variable(variable)
    if type(args["point"]) is str:
        from .rational_limits import check_limit
        function = _checker_rational_source(args["expression"], variable)
        return (len(evidence) == 2 and evidence[0] == "legacy_limit" and
                check_limit(function, variable, args["point"], side, value, evidence[1]))
    if len(evidence) != 4 or evidence[0] != "algebraic_limit_v1":
        return False
    point, point_evidence, certificate = evidence[1:]
    if (not arithmetic.check_operation("algebraic", dict(expression=args["point"]), point, point_evidence)
            or type(certificate) is not tuple or len(certificate) != 2):
        return False
    function = _checker_rational_source(args["expression"], variable)
    if certificate[0] == "regular":
        arguments = dict(left=_checker_at(function.numerator, point),
                         right=_checker_at(function.denominator, point))
        return (sign_at(function.denominator, point) != 0 and value.kind == "finite" and
                arithmetic.check_operation("algebraic_div", arguments,
                                           value.value, certificate[1]))
    if certificate[0] != "pole":
        return False
    order = certificate[1]
    if type(order) is not int or not 1 <= order <= function.denominator.degree:
        return False
    if any(sign_at(_derivative_coefficients(function.denominator, k), point) != 0 for k in range(order)):
        return False
    leading_sign = sign_at(_derivative_coefficients(function.denominator, order), point)
    numerator_sign = sign_at(function.numerator, point)
    if leading_sign == 0 or numerator_sign == 0:
        return False
    if side == "both" and order % 2:
        return value.kind == "does_not_exist" and value.value is None
    negative = (leading_sign < 0) != (numerator_sign < 0)
    if side == "left" and order % 2:
        negative = not negative
    return (value.kind == ("negative_infinity" if negative else "positive_infinity")
            and value.value is None)


def _check_integral(args, value, evidence):
    if (type(evidence) is not tuple or len(evidence) != 5 or evidence[0] != "algebraic_integral_v1"
            or type(value) not in (Rational, RealAlgebraicRoot)):
        return False
    variable = args["variable"]
    validate_variable(variable)
    if type(args["expression"]) is RationalFunction:
        return False
    source = _checker_polynomial(args["expression"], variable)
    primitive, lower_record, upper_record, difference_evidence = evidence[1:]
    if (type(primitive) is not Polynomial or primitive.variable != variable or
            primitive.coefficients[0] != ZERO or _derivative_coefficients(primitive, 1) != source):
        return False
    for key, record in (("lower", lower_record), ("upper", upper_record)):
        if type(record) is not tuple or len(record) != 4:
            return False
        point, point_evidence, at, at_evidence = record
        if not arithmetic.check_operation("algebraic", dict(expression=args[key]), point, point_evidence):
            return False
        if not arithmetic.check_operation("algebraic", dict(expression=_checker_at(primitive, point)), at, at_evidence):
            return False
    return arithmetic.check_operation("algebraic", dict(expression=upper_record[2]-lower_record[2]), value, difference_evidence)


def _check_extrema(args, value, evidence):
    from .function_analysis import check_analysis
    if (type(value) is not tuple or type(evidence) is not tuple or len(evidence) != 4 or
            evidence[0] != "extrema_values_v1"):
        return False
    analysis, analysis_evidence, certificates = evidence[1:]
    source = _checker_rational_source(args["expression"], args["variable"])
    if not check_analysis(source, args["variable"], analysis, analysis_evidence):
        return False
    if type(certificates) is not tuple or len(value) != len(analysis.strict_extrema) or len(value) != len(certificates):
        return False
    for result, extremum, certificate in zip(value, analysis.strict_extrema, certificates):
        if type(result) is not ExtremumValue or result.point != extremum.point or result.kind != extremum.kind:
            return False
        eval_args = dict(expression=analysis.function, bindings=((args["variable"], extremum.point),))
        if not arithmetic.check_operation("evaluate_algebraic", eval_args, result.value, certificate):
            return False
    return True


def check_operation(operation, args, value, evidence, *, limits=None):
    with root_work(limits) as budget:
        budget.inspect(args)
        budget.inspect((value, evidence), evidence=True)
        try:
            arithmetic.validate_exact_scalars((args, value, evidence))
            if operation == "limit":
                return _check_limit(args, value, evidence)
            if operation == "integrate":
                return _check_integral(args, value, evidence)
            if operation == "extrema_values":
                return _check_extrema(args, value, evidence)
            return False
        except (InvalidInput, UnsupportedOperation, AttributeError, IndexError, KeyError,
                TypeError, ValueError, ZeroDivisionError):
            return False
