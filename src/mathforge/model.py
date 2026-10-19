"""Immutable exact numbers and real-valued expression trees.

Equality is structural; the deliberately small normalizer is not an equivalence
oracle. Every supported node denotes a total real-valued expression, which makes
the neutral-element rules below safe without discarding exceptional points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import gcd, isqrt
import sys
from typing import Iterable, Mapping
from uuid import UUID, uuid4

from .errors import InvalidInput, UnsupportedOperation
from .limits import _ACTIVE, ResourceLimitError, _register_rational, current_budget


_active_budget = _ACTIVE.get  # The active Budget or None, read once per fast operation.
_new_object = object.__new__


def _extension(value, module, name):
    """Exact classes only, without importing upward during core initialization."""
    loaded = sys.modules.get(__package__ + "." + module)
    return loaded is not None and type(value) is getattr(loaded, name, None)


def _root(value):
    return _extension(value, "algebraic", "RealAlgebraicRoot")


def _rf(value):
    return _extension(value, "rational_function", "RationalFunction")


class Domain(str, Enum):
    REALS = "reals"
    RATIONALS = "rationals"
    INTEGERS = "integers"


Reals = Domain.REALS
Rationals = Domain.RATIONALS
Integers = Domain.INTEGERS


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Assumption:
    name: str
    truth: Truth = Truth.UNKNOWN

    def __post_init__(self):
        if type(self.name) is not str or self.name not in {"nonzero", "positive", "nonnegative"}:
            raise InvalidInput("Supported assumptions: nonzero, positive, nonnegative")
        if type(self.truth) is not Truth:
            raise InvalidInput("Assumption truth must be a Truth value")


class Expression:
    """Base of the exact AST. Use Eq for equations, == for structure."""

    __slots__ = ()

    def __add__(self, other):
        if _rf(other):
            return NotImplemented
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        if _rf(other):
            return NotImplemented
        return add(self, mul(-1, other))

    def __rsub__(self, other):
        return add(other, mul(-1, self))

    def __mul__(self, other):
        if _rf(other) or _extension(other, "matrices", "Matrix"):
            return NotImplemented
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __neg__(self):
        return mul(-1, self)

    def __pos__(self):
        return self

    def __truediv__(self, other):
        if _rf(other):
            return NotImplemented
        denominator = simplify_expression(as_expression(other))
        if not isinstance(denominator, Rational):
            raise UnsupportedOperation("Only division by a rational constant is supported")
        if denominator.numerator == 0:
            raise ZeroDivisionError("division by zero")
        return mul(self, Rational(denominator.denominator, denominator.numerator))

    def __rtruediv__(self, other):
        return as_expression(other).__truediv__(self)

    def __pow__(self, exponent):
        return power(self, exponent)

    def __bool__(self):
        raise TypeError("A symbolic expression has no Python truth value")

    def subs(self, mapping: Mapping[Symbol, Expression | int]):
        return substitute_expression(self, mapping)


@dataclass(frozen=True, slots=True, init=False)
class Rational(Expression):
    """An exact rational number in lowest terms with a positive denominator.

    Arithmetic and ordering with another Rational or an int work on the integers directly: each result charges the
    active budget one work unit and keeps its integers within max_integer_bits. Any other operand, and a subclass
    receiver, take v0.7's generic Expression path unchanged.
    """

    numerator: int
    denominator: int = 1

    def __init__(self, numerator: int, denominator: int = 1) -> None:
        if type(numerator) is not int or type(denominator) is not int:
            raise TypeError("Rational requires exact Python integers, not float or bool")
        budget = _active_budget()
        if budget is not None:
            budget.charge_rational(numerator, denominator)
        if denominator != 1:
            if denominator == 0:
                raise ZeroDivisionError("rational denominator is zero")
            divisor = gcd(numerator, denominator)
            if denominator < 0:
                divisor = -divisor
            if divisor != 1:
                numerator //= divisor
                denominator //= divisor
        _set_numerator(self, numerator)
        _set_denominator(self, denominator)

    def __getstate__(self):
        # v0.7's pickled state was the attribute dict; keeping it lets a pickle load in either version.
        return {"numerator": self.numerator, "denominator": self.denominator}

    def __setstate__(self, state):
        _set_numerator(self, state["numerator"])
        _set_denominator(self, state["denominator"])

    def __str__(self):
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def __bool__(self):
        return self.numerator != 0

    def __abs__(self):
        return _exact(abs(self.numerator), self.denominator)

    def __neg__(self):
        if type(self) is not Rational:
            return Expression.__neg__(self)
        return _exact(-self.numerator, self.denominator)

    def __add__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                na, da, nb, db = self.numerator, self.denominator, other.numerator, other.denominator
                if da == 1:
                    return _exact(na + nb, 1) if db == 1 else _exact(na * db + nb, db)
                if db == 1:
                    return _exact(na + nb * da, da)
                divisor = gcd(da, db)
                if divisor == 1:
                    return _exact(na * db + nb * da, da * db)
                scale = da // divisor
                numerator = na * (db // divisor) + nb * scale
                common = gcd(numerator, divisor)
                if common == 1:
                    return _exact(numerator, scale * db)
                return _exact(numerator // common, scale * (db // common))
            if kind is int:
                da = self.denominator
                return _exact(self.numerator + other, 1) if da == 1 else _exact(self.numerator + other * da, da)
        return Expression.__add__(self, other)

    def __radd__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is int:
                da = self.denominator
                return _exact(other + self.numerator, 1) if da == 1 else _exact(other * da + self.numerator, da)
            if kind is Rational:
                return Rational.__add__(other, self)
        return Expression.__radd__(self, other)

    def __sub__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                na, da, nb, db = self.numerator, self.denominator, other.numerator, other.denominator
                if da == 1:
                    return _exact(na - nb, 1) if db == 1 else _exact(na * db - nb, db)
                if db == 1:
                    return _exact(na - nb * da, da)
                divisor = gcd(da, db)
                if divisor == 1:
                    return _exact(na * db - nb * da, da * db)
                scale = da // divisor
                numerator = na * (db // divisor) - nb * scale
                common = gcd(numerator, divisor)
                if common == 1:
                    return _exact(numerator, scale * db)
                return _exact(numerator // common, scale * (db // common))
            if kind is int:
                da = self.denominator
                return _exact(self.numerator - other, 1) if da == 1 else _exact(self.numerator - other * da, da)
        return Expression.__sub__(self, other)

    def __rsub__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is int:
                da = self.denominator
                return _exact(other - self.numerator, 1) if da == 1 else _exact(other * da - self.numerator, da)
            if kind is Rational:
                return Rational.__sub__(other, self)
        return Expression.__rsub__(self, other)

    def __mul__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                na, da, nb, db = self.numerator, self.denominator, other.numerator, other.denominator
                if da == 1 and db == 1:
                    return _exact(na * nb, 1)
                divisor = gcd(na, db)
                if divisor > 1:
                    na //= divisor
                    db //= divisor
                divisor = gcd(nb, da)
                if divisor > 1:
                    nb //= divisor
                    da //= divisor
                return _exact(na * nb, da * db)
            if kind is int:
                return _scale(self.numerator, self.denominator, other)
        return Expression.__mul__(self, other)

    def __rmul__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is int:
                return _scale(self.numerator, self.denominator, other)
            if kind is Rational:
                return Rational.__mul__(other, self)
        return Expression.__rmul__(self, other)

    def __truediv__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                return _quotient(self.numerator, self.denominator, other.numerator, other.denominator)
            if kind is int:
                return _quotient(self.numerator, self.denominator, other, 1)
        return Expression.__truediv__(self, other)

    def __rtruediv__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is int:
                return _quotient(other, 1, self.numerator, self.denominator)
            if kind is Rational:
                return _quotient(other.numerator, other.denominator, self.numerator, self.denominator)
        return Expression.__rtruediv__(self, other)

    def __pow__(self, exponent):
        if type(exponent) is not int or type(self) is not Rational:
            return Expression.__pow__(self, exponent)
        if exponent < 0:
            raise UnsupportedOperation("Negative powers and symbolic denominators are unsupported")
        numerator, denominator = self.numerator, self.denominator
        budget = _active_budget()
        if budget is not None:
            budget.tick()
            if exponent > 1:
                # v0.7's estimate: refuse a power before computing it if it could exceed the bound.
                bound = budget.limits.max_integer_bits
                for value in (numerator, denominator):
                    if abs(value) > 1 and value.bit_length() * exponent > bound:
                        raise ResourceLimitError("Computation limit exceeded: power integer bits")
        if exponent == 1:
            return self
        return trusted_rational(numerator ** exponent, denominator ** exponent)

    def __lt__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                return _ordered(self.numerator * other.denominator < other.numerator * self.denominator)
            if kind is int:
                return _ordered(self.numerator < other * self.denominator)
        difference = self._compare(other)
        return NotImplemented if difference is NotImplemented else difference < 0

    def __le__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                return _ordered(self.numerator * other.denominator <= other.numerator * self.denominator)
            if kind is int:
                return _ordered(self.numerator <= other * self.denominator)
        difference = self._compare(other)
        return NotImplemented if difference is NotImplemented else difference <= 0

    def __gt__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                return _ordered(self.numerator * other.denominator > other.numerator * self.denominator)
            if kind is int:
                return _ordered(self.numerator > other * self.denominator)
        difference = self._compare(other)
        return NotImplemented if difference is NotImplemented else difference > 0

    def __ge__(self, other):
        if type(self) is Rational:
            kind = type(other)
            if kind is Rational:
                return _ordered(self.numerator * other.denominator >= other.numerator * self.denominator)
            if kind is int:
                return _ordered(self.numerator >= other * self.denominator)
        difference = self._compare(other)
        return NotImplemented if difference is NotImplemented else difference >= 0

    def _compare(self, other):
        """v0.7's ordering, kept for the operands and receivers that the fast paths leave to it."""
        if type(other) is int:
            other = Rational(other)
        if not isinstance(other, Rational):
            return NotImplemented
        budget = current_budget()
        if budget is not None:
            budget.product(self.numerator, other.denominator)
            budget.product(other.numerator, self.denominator)
        return self.numerator * other.denominator - other.numerator * self.denominator


_set_numerator = Rational.__dict__["numerator"].__set__  # Slot setters: the frozen __setattr__ refuses writes.
_set_denominator = Rational.__dict__["denominator"].__set__
_register_rational(Rational)


def trusted_rational(numerator: int, denominator: int) -> Rational:
    """A Rational from integers already in lowest terms, with no validation and no budget charge.

    The caller guarantees exact ints, denominator > 0 and gcd(numerator, denominator) == 1; the result is then
    indistinguishable from Rational(numerator, denominator): equal, with the same hash and repr.
    """
    value = _new_object(Rational)
    _set_numerator(value, numerator)
    _set_denominator(value, denominator)
    return value


def _exact(numerator, denominator):
    """A fast-path result in lowest terms: one work unit of the active budget, integers within its bound."""
    budget = _active_budget()
    if budget is not None:
        budget.charge_rational(numerator, denominator)
    value = _new_object(Rational)
    _set_numerator(value, numerator)
    _set_denominator(value, denominator)
    return value


def _scale(numerator, denominator, factor):
    """numerator/denominator times an int, reduced as Fraction does: by gcd(factor, denominator) first."""
    if denominator == 1:
        return _exact(numerator * factor, 1)
    divisor = gcd(factor, denominator)
    if divisor > 1:
        factor //= divisor
        denominator //= divisor
    return _exact(numerator * factor, denominator)


def _quotient(na, da, nb, db):
    """(na/da) / (nb/db) for reduced operands with positive denominators, reduced as Fraction does."""
    if nb == 0:
        raise ZeroDivisionError("division by zero")
    divisor = gcd(na, nb)
    if divisor > 1:
        na //= divisor
        nb //= divisor
    divisor = gcd(db, da)
    if divisor > 1:
        da //= divisor
        db //= divisor
    numerator, denominator = na * db, nb * da
    if denominator < 0:
        return _exact(-numerator, -denominator)
    return _exact(numerator, denominator)


def _ordered(result):
    """A fast comparison's outcome, after charging the active budget one work unit."""
    budget = _active_budget()
    if budget is not None:
        budget.tick()
    return result


@dataclass(frozen=True)
class Symbol(Expression):
    name: str
    domain: Domain = Reals
    assumptions: tuple[Assumption, ...] = ()
    uid: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        if type(self.name) is not str or not self.name.strip():
            raise InvalidInput("Symbol name must be a nonempty string")
        if type(self.domain) is not Domain:
            raise InvalidInput("Symbol domain must be a Domain value")
        assumptions = tuple(self.assumptions)
        if any(type(item) is not Assumption for item in assumptions):
            raise InvalidInput("Symbol assumptions must contain Assumption objects")
        if len({item.name for item in assumptions}) != len(assumptions):
            raise InvalidInput("Duplicate assumption name")
        # Enumerate sign possibilities instead of silently admitting contradictory
        # restrictions. UNKNOWN does not narrow the domain.
        signs = {-1, 0, 1}
        for item in assumptions:
            if item.truth is Truth.UNKNOWN:
                continue
            predicate = {
                "positive": lambda sign: sign > 0,
                "nonnegative": lambda sign: sign >= 0,
                "nonzero": lambda sign: sign != 0,
            }[item.name]
            signs = {sign for sign in signs if predicate(sign) == (item.truth is Truth.TRUE)}
        if not signs:
            raise InvalidInput("Contradictory symbol assumptions")
        if type(self.uid) is not str:
            raise InvalidInput("Symbol uid must be a UUID string")
        try:
            uid = str(UUID(self.uid))
        except (ValueError, AttributeError) as exc:
            raise InvalidInput("Symbol uid must be a UUID string") from exc
        object.__setattr__(self, "uid", uid)
        object.__setattr__(self, "assumptions", tuple(sorted(assumptions, key=lambda item: item.name)))

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Add(Expression):
    terms: tuple[Expression, ...]

    def __post_init__(self):
        object.__setattr__(self, "terms", tuple(as_expression(value) for value in self.terms))
        if len(self.terms) < 2:
            raise InvalidInput("An Add node requires at least two terms; use add()")

    def __str__(self):
        result = str(self.terms[0])
        for term in self.terms[1:]:
            rendered = str(term)
            result += " - " + rendered[1:] if rendered.startswith("-") else " + " + rendered
        return result


@dataclass(frozen=True)
class Mul(Expression):
    factors: tuple[Expression, ...]

    def __post_init__(self):
        object.__setattr__(self, "factors", tuple(as_expression(value) for value in self.factors))
        if len(self.factors) < 2:
            raise InvalidInput("A Mul node requires at least two factors; use mul()")

    def __str__(self):
        factors = self.factors
        prefix = ""
        if factors[0] == Rational(-1):
            factors, prefix = factors[1:], "-"
        return prefix + "*".join(f"({value})" if isinstance(value, Add) else str(value) for value in factors)


@dataclass(frozen=True)
class Pow(Expression):
    base: Expression
    exponent: int

    def __post_init__(self):
        object.__setattr__(self, "base", as_expression(self.base))
        _validate_exponent(self.exponent)

    def __str__(self):
        simple = isinstance(self.base, (Symbol, Sqrt)) or (
            isinstance(self.base, Rational) and self.base.denominator == 1 and self.base.numerator >= 0
        )
        return f"{self.base if simple else '(' + str(self.base) + ')'}**{self.exponent}"


@dataclass(frozen=True)
class Sqrt(Expression):
    radicand: Rational

    def __post_init__(self):
        value = as_expression(self.radicand)
        if not isinstance(value, Rational):
            raise UnsupportedOperation("Square roots require a constant rational radicand")
        if value.numerator < 0:
            raise UnsupportedOperation("Negative square roots are outside the real expression model")
        object.__setattr__(self, "radicand", value)

    def __str__(self):
        return f"sqrt({self.radicand})"


def symbol(name: str, domain: Domain = Reals, assumptions: Iterable[Assumption] = ()) -> Symbol:
    return Symbol(name, domain, tuple(assumptions))


def as_expression(value):
    if type(value) is int:
        return Rational(value)
    if type(value) in (Rational, Symbol, Add, Mul, Pow, Sqrt):
        return value
    if _root(value):
        return value
    raise TypeError("Expected an exact integer or a supported MathForge expression; float and bool are not exact inputs")


def _validate_exponent(exponent):
    if type(exponent) is not int:
        raise UnsupportedOperation("Only nonnegative Python integer exponents are supported")
    if exponent < 0:
        raise UnsupportedOperation("Negative powers and symbolic denominators are unsupported")


def _radd(left: Rational, right: Rational) -> Rational:
    budget = current_budget()
    if budget is not None:
        budget.product(left.numerator, right.denominator)
        budget.product(right.numerator, left.denominator)
        budget.product(left.denominator, right.denominator)
    return Rational(left.numerator * right.denominator + right.numerator * left.denominator,
                    left.denominator * right.denominator)


def _rmul(left: Rational, right: Rational) -> Rational:
    budget = current_budget()
    if budget is not None:
        budget.product(left.numerator, right.numerator)
        budget.product(left.denominator, right.denominator)
    return Rational(left.numerator * right.numerator, left.denominator * right.denominator)


def _sort_key(expr):
    if _root(expr):
        return (6, expr.integer_coefficients, expr.real_index)
    if isinstance(expr, Rational):
        return (0, expr.numerator, expr.denominator)
    if isinstance(expr, Symbol):
        return (1, expr.uid)
    if isinstance(expr, Sqrt):
        return (2, _sort_key(expr.radicand))
    if isinstance(expr, Pow):
        return (3, _sort_key(expr.base), expr.exponent)
    if isinstance(expr, Mul):
        return (4, tuple(_sort_key(value) for value in expr.factors))
    return (5, tuple(_sort_key(value) for value in expr.terms))


def _fold_exact(values, operation, identity):
    """add() or mul() of exact Rationals and ints: one work unit per pairwise step, or one for a lone result."""
    if len(values) < 2:
        value = values[0] if values else identity
        return _exact(value, 1) if type(value) is int else _exact(value.numerator, value.denominator)
    total = values[0]
    if type(total) is int:
        total = trusted_rational(total, 1)
    for value in values[1:]:
        total = operation(total, value)
    return total


def add(*args):
    for value in args:
        if type(value) is not Rational and type(value) is not int:
            break
    else:
        return _fold_exact(args, Rational.__add__, 0)
    flat = []
    for value in args:
        value = simplify_expression(as_expression(value))
        flat.extend(value.terms if isinstance(value, Add) else (value,))
    constant = Rational(0)
    coefficients = {}
    for value in flat:
        if isinstance(value, Rational):
            constant = _radd(constant, value)
            continue
        coefficient = Rational(1)
        term = value
        if isinstance(value, Mul) and isinstance(value.factors[0], Rational):
            coefficient = value.factors[0]
            term = value.factors[1] if len(value.factors) == 2 else Mul(value.factors[1:])
        coefficients[term] = _radd(coefficients.get(term, Rational(0)), coefficient)
    terms = [mul(coefficient, term) for term, coefficient in coefficients.items() if coefficient.numerator]
    if constant.numerator:
        terms.append(constant)
    terms.sort(key=_sort_key)
    return Rational(0) if not terms else terms[0] if len(terms) == 1 else Add(tuple(terms))


def mul(*args):
    for value in args:
        if type(value) is not Rational and type(value) is not int:
            break
    else:
        return _fold_exact(args, Rational.__mul__, 1)
    flat = []
    for value in args:
        value = simplify_expression(as_expression(value))
        flat.extend(value.factors if isinstance(value, Mul) else (value,))
    coefficient = Rational(1)
    exponents = {}
    for value in flat:
        if isinstance(value, Rational):
            coefficient = _rmul(coefficient, value)
        else:
            base, exponent = (value.base, value.exponent) if isinstance(value, Pow) else (value, 1)
            exponents[base] = exponents.get(base, 0) + exponent
    if coefficient.numerator == 0:
        return coefficient
    factors = []
    for base, exponent in exponents.items():
        value = power(base, exponent)
        if isinstance(value, Rational):
            coefficient = _rmul(coefficient, value)
        elif isinstance(value, Mul) and isinstance(value.factors[0], Rational):
            coefficient = _rmul(coefficient, value.factors[0])
            factors.extend(value.factors[1:])
        else:
            factors.append(value)
    factors.sort(key=_sort_key)
    # Distributing a rational scalar permits additive cancellation without
    # expanding products or powers of symbolic sums.
    if len(factors) == 1 and isinstance(factors[0], Add) and coefficient != Rational(1):
        return add(*(mul(coefficient, term) for term in factors[0].terms))
    if coefficient != Rational(1) or not factors:
        factors.insert(0, coefficient)
    return factors[0] if len(factors) == 1 else Mul(tuple(factors))


def power(base, exponent):
    _validate_exponent(exponent)
    base = simplify_expression(as_expression(base))
    if exponent == 0:
        return Rational(1)
    if exponent == 1:
        return base
    if isinstance(base, Rational):
        budget = current_budget()
        if budget is not None:
            budget.tick()
            for value in (base.numerator, base.denominator):
                if abs(value) > 1 and value.bit_length() * exponent > budget.limits.max_integer_bits:
                    from .limits import ResourceLimitError
                    raise ResourceLimitError("Computation limit exceeded: power integer bits")
        return Rational(base.numerator ** exponent, base.denominator ** exponent)
    if isinstance(base, Pow):
        return power(base.base, base.exponent * exponent)
    if isinstance(base, Mul):
        # Nonnegative integer powers distribute over products of total real
        # expressions, including a negative rational times an exact radical.
        return mul(*(power(factor, exponent) for factor in base.factors))
    if isinstance(base, Sqrt):
        coefficient = power(base.radicand, exponent // 2)
        if exponent % 2 == 0:
            return coefficient
        return base if coefficient == Rational(1) else Mul((coefficient, base))
    return Pow(base, exponent)


def sqrt(value: Expression | int) -> Expression:
    value = simplify_expression(as_expression(value))
    if not isinstance(value, Rational):
        raise UnsupportedOperation("Symbolic square-root radicands are unsupported")
    if value.numerator < 0:
        raise UnsupportedOperation("Negative square roots are outside the real expression model")
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return Rational(numerator, denominator)
    return Sqrt(value)


def simplify_expression(expr):
    expr = as_expression(expr)
    if isinstance(expr, Add):
        return add(*expr.terms)
    if isinstance(expr, Mul):
        return mul(*expr.factors)
    if isinstance(expr, Pow):
        return power(expr.base, expr.exponent)
    if isinstance(expr, Sqrt):
        return sqrt(expr.radicand)
    return expr


def free_symbols(expr):
    expr = as_expression(expr)
    if isinstance(expr, Symbol):
        return frozenset((expr,))
    if isinstance(expr, (Rational, Sqrt)) or _root(expr):
        return frozenset()
    if isinstance(expr, Pow):
        return free_symbols(expr.base)
    children = expr.terms if isinstance(expr, Add) else expr.factors
    return frozenset().union(*(free_symbols(child) for child in children))


def _has_domain(expr, domain):
    if domain is Reals:
        return True
    if isinstance(expr, Rational):
        return domain is Rationals or expr.denominator == 1
    if isinstance(expr, Symbol):
        return expr.domain is Integers or (domain is Rationals and expr.domain is Rationals)
    if isinstance(expr, Sqrt) or _root(expr):
        return False
    if isinstance(expr, Pow):
        return _has_domain(expr.base, domain)
    children = expr.terms if isinstance(expr, Add) else expr.factors
    return all(_has_domain(child, domain) for child in children)


def _check_replacement(variable, value):
    if not _has_domain(value, variable.domain):
        if isinstance(value, (Rational, Sqrt)):
            raise InvalidInput(f"Replacement violates domain {variable.domain.value}")
        raise UnsupportedOperation("Cannot establish the replacement's membership in the symbol domain")
    predicates = {"positive": lambda sign: sign > 0, "nonnegative": lambda sign: sign >= 0,
                  "nonzero": lambda sign: sign != 0}
    for assumption in variable.assumptions:
        if assumption.truth is Truth.UNKNOWN:
            continue
        actual = Truth.UNKNOWN
        if isinstance(value, Rational):
            actual = Truth.TRUE if predicates[assumption.name](value.numerator) else Truth.FALSE
        elif isinstance(value, Sqrt):
            actual = Truth.TRUE if predicates[assumption.name](value.radicand.numerator) else Truth.FALSE
        elif isinstance(value, Symbol):
            actual = next((item.truth for item in value.assumptions if item.name == assumption.name), Truth.UNKNOWN)
        if actual is Truth.UNKNOWN:
            raise UnsupportedOperation("Cannot establish replacement assumptions")
        if actual is not assumption.truth:
            raise InvalidInput(f"Replacement violates assumption {assumption.name}")


def substitute_expression(expr, mapping):
    expr = as_expression(expr)
    if not isinstance(mapping, Mapping):
        raise InvalidInput("Substitution bindings must be a mapping keyed by Symbol")
    replacements = {}
    for variable, raw_value in mapping.items():
        if type(variable) is not Symbol:
            raise InvalidInput("Substitution bindings must be keyed by Symbol identity, not name")
        value = simplify_expression(as_expression(raw_value))
        _check_replacement(variable, value)
        replacements[variable] = value

    def visit(node):
        if isinstance(node, Symbol):
            return replacements.get(node, node)
        if isinstance(node, (Rational, Sqrt)) or _root(node):
            return simplify_expression(node)
        if isinstance(node, Add):
            return add(*(visit(term) for term in node.terms))
        if isinstance(node, Mul):
            return mul(*(visit(factor) for factor in node.factors))
        return power(visit(node.base), node.exponent)

    return visit(expr)


def evaluate_expression(expr, mapping=None):
    result = substitute_expression(expr, {} if mapping is None else mapping)
    if free_symbols(result):
        raise InvalidInput("Evaluation requires complete bindings; use substitute for a partial replacement")
    return simplify_expression(result)
