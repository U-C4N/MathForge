"""Univariate polynomials over MathForge's own rational arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidInput, UnsupportedOperation
from .model import Add, Mul, Pow, Rational, Sqrt, Symbol, add, as_expression, mul, power


# A deterministic scope limit prevents accidental enormous dense expansions.
MAX_POLYNOMIAL_DEGREE = 1024


def _trim(coefficients):
    result = list(coefficients)
    while len(result) > 1 and result[-1].numerator == 0:
        result.pop()
    return tuple(result) or (Rational(0),)


def _sum(left, right):
    size = max(len(left), len(right))
    return _trim(tuple((left[index] if index < len(left) else Rational(0)) +
                       (right[index] if index < len(right) else Rational(0)) for index in range(size)))


def _product(left, right):
    if left == (Rational(0),) or right == (Rational(0),):
        return (Rational(0),)
    degree = len(left) + len(right) - 2
    if degree > MAX_POLYNOMIAL_DEGREE:
        raise UnsupportedOperation(f"Dense polynomial degree exceeds {MAX_POLYNOMIAL_DEGREE}")
    values = [Rational(0)] * (degree + 1)
    for i, first in enumerate(left):
        if first.numerator:
            for j, second in enumerate(right):
                if second.numerator:
                    values[i + j] = values[i + j] + first * second
    return _trim(values)


@dataclass(frozen=True)
class Polynomial:
    """Constant-first, normalized rational coefficients and one explicit symbol."""

    coefficients: tuple[Rational, ...]
    variable: Symbol

    def __post_init__(self):
        if type(self.variable) is not Symbol:
            raise InvalidInput("Polynomial variable must be a Symbol")
        coefficients = tuple(as_expression(value) for value in self.coefficients)
        if any(not isinstance(value, Rational) for value in coefficients):
            raise UnsupportedOperation("Polynomial coefficients must be rational")
        coefficients = _trim(coefficients)
        if len(coefficients) - 1 > MAX_POLYNOMIAL_DEGREE:
            raise UnsupportedOperation(f"Dense polynomial degree exceeds {MAX_POLYNOMIAL_DEGREE}")
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def is_zero(self):
        return self.coefficients == (Rational(0),)

    @property
    def degree(self):
        return -1 if self.is_zero else len(self.coefficients) - 1

    @classmethod
    def from_expression(cls, expression, variable):
        if type(variable) is not Symbol:
            raise InvalidInput("Polynomial variable must be a Symbol")

        def convert(node):
            if isinstance(node, Rational):
                return (node,)
            if isinstance(node, Symbol):
                if node != variable:
                    raise UnsupportedOperation("Only one variable and rational coefficients are supported")
                return (Rational(0), Rational(1))
            if isinstance(node, Sqrt):
                raise UnsupportedOperation("Irrational coefficients are unsupported")
            if isinstance(node, Add):
                result = (Rational(0),)
                for term in node.terms:
                    result = _sum(result, convert(term))
                return result
            if isinstance(node, Mul):
                result = (Rational(1),)
                for factor in node.factors:
                    result = _product(result, convert(factor))
                return result
            if isinstance(node, Pow):
                if node.exponent == 0:
                    return (Rational(1),)
                base = convert(node.base)
                if (len(base) - 1) * node.exponent > MAX_POLYNOMIAL_DEGREE:
                    raise UnsupportedOperation(f"Dense polynomial degree exceeds {MAX_POLYNOMIAL_DEGREE}")
                exponent = node.exponent
                result = (Rational(1),)
                while exponent:
                    if exponent % 2:
                        result = _product(result, base)
                    exponent //= 2
                    if exponent:
                        base = _product(base, base)
                return result
            raise UnsupportedOperation("Unsupported polynomial expression")

        from .model import simplify_expression
        return cls(convert(simplify_expression(as_expression(expression))), variable)

    def to_expression(self):
        return add(*(mul(coefficient, power(self.variable, index))
                     for index, coefficient in enumerate(self.coefficients) if coefficient.numerator))

    def evaluate(self, value):
        from .model import _check_replacement, simplify_expression
        value = simplify_expression(as_expression(value))
        _check_replacement(self.variable, value)
        result = Rational(0)
        for coefficient in reversed(self.coefficients):
            result = result * value + coefficient
        return result

    def derivative(self):
        return Polynomial(tuple(coefficient * index for index, coefficient in
                                enumerate(self.coefficients) if index), self.variable)

    def __str__(self):
        return str(self.to_expression())
