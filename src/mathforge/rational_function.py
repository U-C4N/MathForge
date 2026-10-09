"""Partial univariate rational functions with explicit, persistent exclusions."""

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import InvalidInput, UnsupportedOperation
from .limits import computation
from .model import Rational, Reals, Symbol, as_expression
from .polynomial import Polynomial


def validate_variable(variable):
    if type(variable) is not Symbol:
        raise InvalidInput("A Symbol variable is required")
    if variable.domain is not Reals or variable.assumptions:
        raise UnsupportedOperation("Only unconstrained real variables are supported")


def as_polynomial(value, variable):
    if type(value) is Polynomial:
        if value.variable != variable:
            raise InvalidInput("Polynomial variables must have the same identity")
        return value
    return Polynomial.from_expression(as_expression(value), variable)


def _key(p):
    return tuple((c.numerator, c.denominator) for c in p.coefficients)


def _square_free_part(p):
    if p.is_zero:
        raise InvalidInput("An identically zero exclusion has an empty domain")
    if p.degree <= 0:
        return None
    return p.exact_div(p.gcd(p.derivative())).monic()


def normalize_exclusions(values):
    normalized = []
    for p in values:
        part = _square_free_part(p)
        if part is not None and part not in normalized:
            normalized.append(part)
    return tuple(sorted(normalized, key=_key))


def exclusions_cover(source, target, *, limits=None):
    """Decide inclusion of real zero sets without multiplying exclusion factors."""
    from .root_isolation import count_real_roots
    with computation(limits):
        for polynomial in source:
            remainder = _square_free_part(polynomial)
            if remainder is None:
                continue
            for other in target:
                remainder = remainder.exact_div(remainder.gcd(other))
                if remainder.degree <= 0:
                    break
            if remainder.degree > 0 and count_real_roots(remainder) != 0:
                return False
        return True


def same_domain(left, right, *, limits=None):
    with computation(limits):
        return (left.variable == right.variable
                and exclusions_cover(left.excluded, right.excluded)
                and exclusions_cover(right.excluded, left.excluded))


@dataclass(frozen=True)
class RationalFunction:
    """Canonical value record. Use rational_function() for unreduced inputs."""

    numerator: Polynomial
    denominator: Polynomial
    excluded: tuple[Polynomial, ...] = ()

    def __post_init__(self):
        with computation() as budget:
            if type(self.numerator) is not Polynomial or type(self.denominator) is not Polynomial:
                raise InvalidInput("RationalFunction requires Polynomial numerator and denominator")
            validate_variable(self.variable)
            if self.denominator.variable != self.variable:
                raise InvalidInput("RationalFunction variables must agree")
            excluded = tuple(self.excluded)
            if any(type(p) is not Polynomial or p.variable != self.variable for p in excluded):
                raise InvalidInput("Exclusions must be polynomials in the same variable")
            budget.inspect((self.numerator, self.denominator, excluded))
            if self.denominator.is_zero or self.denominator.coefficients[-1] != Rational(1):
                raise InvalidInput("The canonical denominator must be nonzero and monic")
            if self.numerator.gcd(self.denominator).degree > 0:
                raise InvalidInput("The numerator and denominator must be relatively prime")
            if normalize_exclusions(excluded) != excluded:
                raise InvalidInput("Exclusions must be canonical monic square-free polynomials")
            if not exclusions_cover((self.denominator,), excluded):
                raise InvalidInput("Denominator roots must be included in the exclusions")
            object.__setattr__(self, "excluded", excluded)

    @property
    def variable(self):
        return self.numerator.variable

    @classmethod
    def _from_normal_form(cls, numerator, denominator, excluded, *, limits=None):
        with computation(limits):
            return cls(numerator, denominator, tuple(excluded))

    @classmethod
    def _normalize(cls, numerator, denominator, excluded, *, limits=None):
        with computation(limits):
            if denominator.is_zero:
                raise InvalidInput("Division by the zero rational function", code="division_by_zero")
            common = numerator.gcd(denominator)
            numerator = numerator.exact_div(common)
            denominator = denominator.exact_div(common)
            leading = denominator.coefficients[-1]
            numerator = numerator * (Rational(1) / leading)
            denominator = denominator * (Rational(1) / leading)
            return cls(numerator, denominator, normalize_exclusions(excluded))

    def _coerce(self, other):
        if type(other) is RationalFunction:
            if other.variable != self.variable:
                raise InvalidInput("RationalFunction variables must agree")
            return other
        p = as_polynomial(other, self.variable)
        return RationalFunction(p, Polynomial((1,), self.variable))

    def __add__(self, other):
        with computation():
            other = self._coerce(other)
            return self._normalize(self.numerator * other.denominator + other.numerator * self.denominator,
                                   self.denominator * other.denominator, self.excluded + other.excluded)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self._normalize(-self.numerator, self.denominator, self.excluded)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        with computation():
            other = self._coerce(other)
            return self._normalize(self.numerator * other.numerator, self.denominator * other.denominator,
                                   self.excluded + other.excluded)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        with computation():
            other = self._coerce(other)
            if other.numerator.is_zero:
                raise InvalidInput("Division by the zero rational function", code="division_by_zero")
            return self._normalize(self.numerator * other.denominator, self.denominator * other.numerator,
                                   self.excluded + other.excluded + (other.numerator,))

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, exponent):
        if type(exponent) is not int or exponent < 0:
            raise UnsupportedOperation("RationalFunction powers require a nonnegative integer")
        with computation():
            return self._normalize(self.numerator ** exponent, self.denominator ** exponent, self.excluded)

    def __bool__(self):
        raise TypeError("A rational function has no Python truth value")

    def __str__(self):
        restrictions = ", ".join(f"{p} != 0" for p in self.excluded)
        value = f"({self.numerator})/({self.denominator})"
        return value + (f" where {restrictions}" if restrictions else "")

    def is_defined_at(self, value, *, limits=None):
        from .algebraic import sign_at
        with computation(limits):
            return all(sign_at(p, value) != 0 for p in self.excluded)

    def evaluate(self, value, *, limits=None):
        with computation(limits):
            value = as_expression(value)
            if type(value) is not Rational:
                raise UnsupportedOperation("RationalFunction evaluation requires a rational value")
            if any(p.evaluate(value) == Rational(0) for p in self.excluded):
                raise InvalidInput("The rational function is undefined at this point", code="undefined_at_point")
            return self.numerator.evaluate(value) / self.denominator.evaluate(value)

    def substitute(self, bindings, *, limits=None):
        with computation(limits):
            if not isinstance(bindings, Mapping) or any(type(v) is not Symbol for v in bindings):
                raise InvalidInput("Bindings must map Symbol identities to exact values")
            # Validate even irrelevant entries so substitution does not hide bad input.
            for key, value in bindings.items():
                value = as_expression(value)
                if type(value) is Symbol:
                    validate_variable(value)
                elif type(value) is not Rational:
                    raise UnsupportedOperation("RationalFunction substitution supports rational values or symbol renaming")
                from .model import _check_replacement
                _check_replacement(key, value)
            if self.variable not in bindings:
                return self
            value = as_expression(bindings[self.variable])
            if type(value) is Rational:
                return self.evaluate(value)
            validate_variable(value)
            rename = lambda p: Polynomial(p.coefficients, value)
            return RationalFunction(rename(self.numerator), rename(self.denominator),
                                    tuple(rename(p) for p in self.excluded))

    def derivative(self, *, limits=None):
        with computation(limits):
            return self._normalize(self.numerator.derivative() * self.denominator
                                   - self.numerator * self.denominator.derivative(),
                                   self.denominator * self.denominator, self.excluded)


def rational_function(numerator, denominator, *, variable, limits=None):
    with computation(limits) as budget:
        validate_variable(variable)
        budget.inspect((numerator, denominator))
        n = as_polynomial(numerator, variable)
        d = as_polynomial(denominator, variable)
        return RationalFunction._normalize(n, d, (d,))


def as_rational_function(value, variable):
    if type(value) is RationalFunction:
        if value.variable != variable:
            raise InvalidInput("RationalFunction variable differs from the requested variable")
        return value
    return rational_function(value, 1, variable=variable)
