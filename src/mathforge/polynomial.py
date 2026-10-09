"""Univariate polynomials over MathForge's own rational arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

from .errors import InvalidInput, UnsupportedOperation
from .model import Add, Mul, Pow, Rational, Sqrt, Symbol, add, as_expression, mul, power


# A deterministic scope limit prevents accidental enormous dense expansions.
MAX_POLYNOMIAL_DEGREE = 1024


def _budget():
    from .limits import current_budget
    return current_budget()


def _computation(limits=None):
    from .limits import computation
    return computation(limits)


def _visit(count=1):
    budget = _budget()
    if budget is not None:
        budget.nodes(count)


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
        _visit(len(coefficients))
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
    def from_expression(cls, expression, variable, *, limits=None):
        if type(variable) is not Symbol:
            raise InvalidInput("Polynomial variable must be a Symbol")

        def convert(node):
            _visit()
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
        with _computation(limits) if limits is not None else nullcontext():
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

    @property
    def leading_coefficient(self):
        return self.coefficients[-1]

    def _coerce(self, other):
        if type(other) is Polynomial:
            if self.variable != other.variable:
                raise InvalidInput("Polynomial operands require the same Symbol identity")
            return other
        if type(other) in (int, Rational):
            return Polynomial((other,), self.variable)
        raise TypeError("Expected a Polynomial or exact rational scalar")

    def __add__(self, other):
        if type(other) not in (Polynomial, int, Rational):
            return NotImplemented
        with _computation():
            other = self._coerce(other)
            return Polynomial(_sum(self.coefficients, other.coefficients), self.variable)

    __radd__ = __add__

    def __neg__(self):
        with _computation():
            return Polynomial(tuple(-c for c in self.coefficients), self.variable)

    def __sub__(self, other):
        if type(other) not in (Polynomial, int, Rational):
            return NotImplemented
        return self + -self._coerce(other)

    def __rsub__(self, other):
        if type(other) not in (Polynomial, int, Rational):
            return NotImplemented
        return self._coerce(other) - self

    def __mul__(self, other):
        if type(other) not in (Polynomial, int, Rational):
            return NotImplemented
        with _computation():
            other = self._coerce(other)
            return Polynomial(_product(self.coefficients, other.coefficients), self.variable)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if type(other) not in (Polynomial, int, Rational):
            return NotImplemented
        if type(other) not in (int, Rational):
            raise UnsupportedOperation("Use exact_div for polynomial division")
        other = as_expression(other)
        if other.numerator == 0:
            raise InvalidInput("Polynomial scalar divisor is zero")
        with _computation():
            return Polynomial(tuple(c / other for c in self.coefficients), self.variable)

    def __pow__(self, exponent):
        if type(exponent) is not int or exponent < 0:
            raise InvalidInput("Polynomial exponent must be a nonnegative integer")
        with _computation() as budget:
            budget.check_int(exponent)
            if self.degree > 0 and self.degree * exponent > MAX_POLYNOMIAL_DEGREE:
                raise UnsupportedOperation(f"Dense polynomial degree exceeds {MAX_POLYNOMIAL_DEGREE}")
            if exponent and self.is_zero:
                return self
            result, base = Polynomial((1,), self.variable), self
            while exponent:
                if exponent & 1:
                    result = result * base
                exponent >>= 1
                if exponent:
                    base = base * base
            return result

    def monic(self, *, limits=None):
        with _computation(limits):
            return self if self.is_zero else self / self.leading_coefficient

    def divmod(self, other, *, limits=None):
        with _computation(limits):
            other = self._coerce(other)
            if other.is_zero:
                raise InvalidInput("Polynomial divisor is zero")
            if self.degree < other.degree:
                return Polynomial((0,), self.variable), self
            remainder = list(self.coefficients)
            quotient = [Rational(0)] * (self.degree - other.degree + 1)
            while len(remainder) - 1 >= other.degree and any(c.numerator for c in remainder):
                shift = len(remainder) - len(other.coefficients)
                factor = remainder[-1] / other.leading_coefficient
                quotient[shift] = factor
                for index, coefficient in enumerate(other.coefficients):
                    remainder[shift + index] = remainder[shift + index] - factor * coefficient
                remainder = list(_trim(remainder))
            return Polynomial(tuple(quotient), self.variable), Polynomial(tuple(remainder), self.variable)

    def __divmod__(self, other):
        return self.divmod(other)

    def __floordiv__(self, other):
        return self.divmod(other)[0]

    def __mod__(self, other):
        return self.divmod(other)[1]

    def exact_div(self, other, *, limits=None):
        quotient, remainder = self.divmod(other, limits=limits)
        if not remainder.is_zero:
            raise InvalidInput("Polynomial division has a nonzero remainder", code="non_exact_division")
        return quotient

    def gcd(self, other, *, limits=None):
        with _computation(limits):
            a, b = self, self._coerce(other)
            while not b.is_zero:
                a, b = b, a.divmod(b)[1]
            return a.monic()

    def xgcd(self, other, *, limits=None):
        with _computation(limits):
            a, b = self, self._coerce(other)
            zero, one = Polynomial((0,), self.variable), Polynomial((1,), self.variable)
            if a.is_zero and b.is_zero:
                return zero, zero, zero
            old_s, s, old_t, t = one, zero, zero, one
            while not b.is_zero:
                quotient, remainder = a.divmod(b)
                a, b = b, remainder
                old_s, s = s, old_s - quotient * s
                old_t, t = t, old_t - quotient * t
            scale = a.leading_coefficient
            return a / scale, old_s / scale, old_t / scale

    def square_free(self, *, limits=None):
        with _computation(limits):
            if self.degree <= 0:
                return SquareFreeDecomposition(self.coefficients[0], ())
            coefficient = self.leading_coefficient
            f = self / coefficient
            repeated = f.gcd(f.derivative())
            remaining = f.exact_div(repeated)
            factors, multiplicity = [], 1
            while remaining.degree > 0:
                common = remaining.gcd(repeated)
                factor = remaining.exact_div(common)
                if factor.degree > 0:
                    factors.append(SquareFreeFactor(factor, multiplicity))
                remaining = common
                repeated = repeated.exact_div(common)
                multiplicity += 1
            return SquareFreeDecomposition(coefficient, tuple(factors))

    def __str__(self):
        return str(self.to_expression())


@dataclass(frozen=True)
class SquareFreeFactor:
    polynomial: Polynomial
    multiplicity: int

    def __post_init__(self):
        if type(self.polynomial) is not Polynomial or self.polynomial.degree <= 0:
            raise InvalidInput("A square-free factor must be a nonconstant Polynomial")
        if self.polynomial.leading_coefficient != Rational(1):
            raise InvalidInput("Square-free factors must be monic")
        if type(self.multiplicity) is not int or self.multiplicity <= 0:
            raise InvalidInput("Factor multiplicity must be a positive integer")


@dataclass(frozen=True)
class SquareFreeDecomposition:
    coefficient: Rational
    factors: tuple[SquareFreeFactor, ...]

    def __post_init__(self):
        coefficient = as_expression(self.coefficient)
        if type(coefficient) is not Rational:
            raise InvalidInput("Square-free coefficient must be rational")
        factors = tuple(self.factors)
        if any(type(factor) is not SquareFreeFactor for factor in factors):
            raise InvalidInput("Expected SquareFreeFactor entries")
        if coefficient.numerator == 0 and factors:
            raise InvalidInput("The zero square-free decomposition has no factors")
        if any(a.multiplicity >= b.multiplicity for a, b in zip(factors, factors[1:])):
            raise InvalidInput("Square-free factors must have strictly increasing multiplicities")
        if factors and any(f.polynomial.variable != factors[0].polynomial.variable for f in factors):
            raise InvalidInput("Square-free factors require the same variable")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "factors", factors)
