"""Substitution into exact expression trees, producing exactly what ``model.substitute_expression`` produces.

``model.substitute_expression`` rebuilds the tree bottom-up with ``add``, ``mul`` and ``power``, and each of those
first re-normalizes every argument with ``simplify_expression``, recursively. The arguments it receives are already
normal forms, the outputs of the same functions, and ``simplify_expression`` leaves a normal form unchanged, so this
module applies the same rules without the re-normalization: the same flattening, collection, coefficient extraction,
power rules, distribution of a rational coefficient over a single sum, and ordering by the key of
``model._sort_key`` (remembered for compound nodes during a call). Rational arithmetic and rational powers are the
model's own, so the active budget's limits apply as in v0.7.

``simplify_expression(e)`` is ``substitute_expression(e, {})``, so ``normalize`` serves for the bindings too. It is
a producer: the lean value path checks its value (mathforge.certify.substitution) before returning it.
"""

from __future__ import annotations

from ..model import Add, Mul, Pow, Rational, Sqrt, Symbol, power as model_power, sqrt as model_sqrt
from ..model import trusted_rational

_ZERO = trusted_rational(0, 1)
_ONE = trusted_rational(1, 1)
_new = object.__new__


class Unsupported(Exception):
    """A node this kernel does not rebuild: the caller falls back to the structured path."""


def _make_mul(factors):
    node = _new(Mul)
    node.__dict__["factors"] = factors
    return node


def _make_add(terms):
    node = _new(Add)
    node.__dict__["terms"] = terms
    return node


def _make_pow(base, exponent):
    node = _new(Pow)
    fields = node.__dict__
    fields["base"] = base
    fields["exponent"] = exponent
    return node


class _Rebuild:
    """One substitution: replacements by Symbol uid and the remembered sort keys and results of compound nodes."""

    __slots__ = ("replacements", "keys", "done")

    def __init__(self, replacements):
        self.replacements = replacements
        self.keys = {}  # id(node) -> (node, key); the node is kept, so its id is not reused during the call
        self.done = {}  # id(source node) -> (node, normal form)

    def key(self, node):
        """model._sort_key(node)."""
        kind = type(node)
        if kind is Rational:
            return (0, node.numerator, node.denominator)
        if kind is Symbol:
            return (1, node.uid)
        entry = self.keys.get(id(node))
        if entry is not None:
            return entry[1]
        if kind is Pow:
            key = (3, self.key(node.base), node.exponent)
        elif kind is Mul:
            key = (4, tuple([self.key(factor) for factor in node.factors]))
        elif kind is Add:
            key = (5, tuple([self.key(term) for term in node.terms]))
        elif kind is Sqrt:
            key = (2, (0, node.radicand.numerator, node.radicand.denominator))
        else:
            raise Unsupported(kind.__name__)
        self.keys[id(node)] = (node, key)
        return key

    def add(self, args):
        """model.add of normal forms."""
        for value in args:
            if type(value) is not Rational:
                break
        else:
            total = _ZERO
            for value in args:
                total = total + value
            return total
        constant = _ZERO
        coefficients = {}
        for value in args:
            for item in (value.terms if type(value) is Add else (value,)):
                if type(item) is Rational:
                    constant = constant + item
                    continue
                coefficient = _ONE
                term = item
                if type(item) is Mul and type(item.factors[0]) is Rational:
                    factors = item.factors
                    coefficient = factors[0]
                    term = factors[1] if len(factors) == 2 else _make_mul(factors[1:])
                previous = coefficients.get(term)
                coefficients[term] = coefficient if previous is None else previous + coefficient
        terms = []
        for term, coefficient in coefficients.items():
            if not coefficient.numerator:
                continue
            if coefficient.numerator == 1 and coefficient.denominator == 1:
                terms.append(term)
            elif type(term) is Mul:
                terms.append(_make_mul((coefficient,) + term.factors))
            else:
                terms.append(_make_mul((coefficient, term)))
        if constant.numerator:
            terms.append(constant)
        if not terms:
            return _ZERO
        if len(terms) == 1:
            return terms[0]
        terms.sort(key=self.key)
        return _make_add(tuple(terms))

    def mul(self, args):
        """model.mul of normal forms."""
        for value in args:
            if type(value) is not Rational:
                break
        else:
            total = _ONE
            for value in args:
                total = total * value
            return total
        coefficient = _ONE
        exponents = {}
        for value in args:
            for item in (value.factors if type(value) is Mul else (value,)):
                kind = type(item)
                if kind is Rational:
                    coefficient = coefficient * item
                elif kind is Pow:
                    base = item.base
                    exponents[base] = exponents.get(base, 0) + item.exponent
                else:
                    exponents[item] = exponents.get(item, 0) + 1
        if not coefficient.numerator:
            return coefficient
        factors = []
        for base, exponent in exponents.items():
            value = self.power(base, exponent)
            kind = type(value)
            if kind is Rational:
                coefficient = coefficient * value
            elif kind is Mul and type(value.factors[0]) is Rational:
                coefficient = coefficient * value.factors[0]
                factors.extend(value.factors[1:])
            else:
                factors.append(value)
        if len(factors) > 1:
            factors.sort(key=self.key)
        unit = coefficient.numerator == 1 and coefficient.denominator == 1
        if len(factors) == 1 and type(factors[0]) is Add and not unit:
            return self.add([self.mul([coefficient, term]) for term in factors[0].terms])
        if not unit or not factors:
            factors.insert(0, coefficient)
        return factors[0] if len(factors) == 1 else _make_mul(tuple(factors))

    def power(self, base, exponent):
        """model.power of a normal form and a nonnegative int."""
        if exponent == 0:
            return _ONE
        if exponent == 1:
            return base
        kind = type(base)
        if kind is Rational:
            return model_power(base, exponent)
        if kind is Pow:
            return self.power(base.base, base.exponent * exponent)
        if kind is Mul:
            return self.mul([self.power(factor, exponent) for factor in base.factors])
        if kind is Sqrt:
            coefficient = model_power(base.radicand, exponent // 2)
            if exponent % 2 == 0:
                return coefficient
            if coefficient.numerator == 1 and coefficient.denominator == 1:
                return base
            return _make_mul((coefficient, base))
        return _make_pow(base, exponent)

    def visit(self, node):
        kind = type(node)
        if kind is Symbol:
            return self.replacements.get(node.uid, node)
        if kind is Rational:
            return node
        entry = self.done.get(id(node))
        if entry is not None:
            return entry[1]
        if kind is Add:
            value = self.add([self.visit(term) for term in node.terms])
        elif kind is Mul:
            value = self.mul([self.visit(factor) for factor in node.factors])
        elif kind is Pow:
            exponent = node.exponent
            if type(exponent) is not int or exponent < 0:
                raise Unsupported("Pow exponent")
            value = self.power(self.visit(node.base), exponent)
        elif kind is Sqrt:
            value = model_sqrt(node.radicand)
        else:
            raise Unsupported(kind.__name__)
        self.done[id(node)] = (node, value)
        return value


def substitute(expression, replacements):
    """model.substitute_expression(expression, bindings) for an expression of Rational, Symbol, Add, Mul, Pow and
    Sqrt nodes, given the normalized replacement of each bound Symbol by its uid. The caller guarantees that one uid
    stands for one Symbol throughout (the verifier's symbol consistency), so a uid identifies the bound Symbol."""
    return _Rebuild(replacements).visit(expression)


def normalize(expression):
    """model.simplify_expression(expression), which is substitute_expression(expression, {})."""
    return _Rebuild({}).visit(expression)
