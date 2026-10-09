"""Exact polynomial calculus and domain-preserving rational derivatives."""
from .errors import InvalidInput, UnsupportedOperation
from .limits import computation
from .model import Rational, as_expression
from .polynomial import Polynomial
from .rational_function import RationalFunction, as_polynomial, validate_variable


def derivative(expression, variable, n=1, *, limits=None):
    with computation(limits) as budget:
        validate_variable(variable)
        if type(n) is not int or n < 0:
            raise InvalidInput("Derivative order must be a nonnegative integer")
        if type(expression) is RationalFunction:
            if expression.variable != variable:
                raise InvalidInput("Rational function variable does not match")
            value = expression
            for _ in range(n):
                budget.tick()
                value = value.derivative()
            return value
        polynomial = as_polynomial(expression, variable)
        if n == 0:
            return polynomial.to_expression() if type(expression) is Polynomial else as_expression(expression)
        for _ in range(min(n, max(0, polynomial.degree) + 1)):
            budget.tick()
            polynomial = polynomial.derivative()
        return polynomial.to_expression()


def antiderivative_value(expression, variable, *, limits=None):
    with computation(limits):
        validate_variable(variable)
        if type(expression) is RationalFunction:
            raise UnsupportedOperation("Rational-function integration is not supported")
        polynomial = as_polynomial(expression, variable)
        if polynomial.degree >= 1024:
            raise UnsupportedOperation("An antiderivative would exceed the polynomial degree limit")
        return Polynomial((Rational(0),) + tuple(
            coefficient / (index + 1) for index, coefficient in enumerate(polynomial.coefficients)), variable).to_expression()


def definite_integral(expression, variable, lower, upper, *, limits=None):
    with computation(limits) as budget:
        validate_variable(variable)
        if type(expression) is RationalFunction:
            raise UnsupportedOperation("Rational-function integration is not supported")
        lower, upper = as_expression(lower), as_expression(upper)
        if type(lower) is not Rational or type(upper) is not Rational:
            raise UnsupportedOperation("Integration bounds must be finite rationals")
        polynomial = as_polynomial(expression, variable)
        contributions = []
        low_power, high_power = Rational(1), Rational(1)
        value = Rational(0)
        for index, coefficient in enumerate(polynomial.coefficients):
            budget.tick()
            low_power, high_power = low_power * lower, high_power * upper
            contribution = coefficient * (high_power - low_power) / (index + 1)
            contributions.append(contribution)
            value = value + contribution
        return value, tuple(contributions)
