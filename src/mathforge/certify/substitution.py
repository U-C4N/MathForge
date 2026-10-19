"""Independent checker of substitute (and of evaluate) on exact expression trees: v0.7's criterion.

v0.7 checks a substitution by recomputing ``model.substitute_expression(source, bindings)`` and comparing it with the
value. That function re-normalizes every intermediate result at every level, which costs more than the operation
itself, so this checker computes the same criterion with its own implementation of the model's normalization: an
iterative post-order walk over the source that replaces the bound symbols and combines each node's normalized
children with the model's rules (flattening, collecting like terms and bases, rational powers, square roots of
rationals, distributing a rational over a single sum, and ordering by ``model._sort_key``). It shares no code with
the producer, mathforge.kernels.substitution; the tests compare both with ``model.substitute_expression``.
"""

from __future__ import annotations

from ..model import Add, Mul, Pow, Rational, Sqrt, Symbol, _sort_key, power, sqrt

_EMPTY = object()


def _is_one(value) -> bool:
    return value.numerator == 1 and value.denominator == 1


def _node(cls, name, value):
    node = object.__new__(cls)
    node.__dict__[name] = value
    return node


def _raise(base, exponent):
    """The normal form of base**exponent for a normal base and an int exponent >= 2."""
    while True:
        kind = type(base)
        if kind is Pow:
            base, exponent = base.base, base.exponent * exponent
            continue
        if kind is Rational:
            return power(base, exponent)
        if kind is Mul:
            return _product([_raise(factor, exponent) if exponent != 1 else factor for factor in base.factors])
        if kind is Sqrt:
            half = power(base.radicand, exponent // 2)
            if exponent % 2 == 0:
                return half
            return base if _is_one(half) else _node(Mul, "factors", (half, base))
        node = object.__new__(Pow)
        node.__dict__.update(base=base, exponent=exponent)
        return node


def _power(base, exponent):
    if exponent == 0:
        return Rational(1)
    if exponent == 1:
        return base
    return _raise(base, exponent)


def _product(values):
    """The normal form of the product of normal forms."""
    scalar = Rational(1)
    bases: dict = {}
    pending = list(values)
    pending.reverse()
    while pending:
        value = pending.pop()
        kind = type(value)
        if kind is Mul:
            pending.extend(reversed(value.factors))
        elif kind is Rational:
            scalar = scalar * value
        elif kind is Pow:
            bases[value.base] = bases.get(value.base, 0) + value.exponent
        else:
            bases[value] = bases.get(value, 0) + 1
    if not bases:
        return scalar
    if not scalar.numerator:
        return scalar
    factors = []
    for base, exponent in bases.items():
        value = _power(base, exponent)
        if type(value) is Rational:
            scalar = scalar * value
        elif type(value) is Mul and type(value.factors[0]) is Rational:
            scalar = scalar * value.factors[0]
            factors.extend(value.factors[1:])
        else:
            factors.append(value)
    factors.sort(key=_sort_key)
    if len(factors) == 1 and type(factors[0]) is Add and not _is_one(scalar):
        return _sum([_product([scalar, term]) for term in factors[0].terms])
    if not _is_one(scalar) or not factors:
        factors.insert(0, scalar)
    return factors[0] if len(factors) == 1 else _node(Mul, "factors", tuple(factors))


def _sum(values):
    """The normal form of the sum of normal forms."""
    constant = Rational(0)
    collected: dict = {}
    order = []
    pending = list(values)
    pending.reverse()
    symbolic = False
    while pending:
        value = pending.pop()
        kind = type(value)
        if kind is Add:
            pending.extend(reversed(value.terms))
            symbolic = True
            continue
        if kind is Rational:
            constant = constant + value
            continue
        symbolic = True
        scale, rest = Rational(1), value
        if kind is Mul and type(value.factors[0]) is Rational:
            scale = value.factors[0]
            rest = value.factors[1] if len(value.factors) == 2 else _node(Mul, "factors", value.factors[1:])
        known = collected.get(rest, _EMPTY)
        if known is _EMPTY:
            order.append(rest)
            collected[rest] = scale
        else:
            collected[rest] = known + scale
    if not symbolic:
        return constant
    terms = []
    for rest in order:
        scale = collected[rest]
        if not scale.numerator:
            continue
        if _is_one(scale):
            terms.append(rest)
        else:
            terms.append(_node(Mul, "factors", (scale, *rest.factors) if type(rest) is Mul else (scale, rest)))
    if constant.numerator:
        terms.append(constant)
    if not terms:
        return Rational(0)
    terms.sort(key=_sort_key)
    return terms[0] if len(terms) == 1 else _node(Add, "terms", tuple(terms))


def substituted(source, replacements):
    """model.substitute_expression(source, bindings), given each bound Symbol's normalized replacement; None for a
    node outside Rational, Symbol, Add, Mul, Pow and Sqrt."""
    results: dict = {}
    stack = [(source, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in results and not expanded:
            continue
        kind = type(node)
        if kind is Symbol:
            results[id(node)] = (node, replacements.get(node, node))
        elif kind is Rational:
            results[id(node)] = (node, node)
        elif kind is Sqrt:
            results[id(node)] = (node, sqrt(node.radicand))
        elif kind in (Add, Mul, Pow):
            children = node.terms if kind is Add else node.factors if kind is Mul else (node.base,)
            if not expanded:
                stack.append((node, True))
                stack.extend((child, False) for child in children)
                continue
            values = [results[id(child)][1] for child in children]
            if kind is Add:
                value = _sum(values)
            elif kind is Mul:
                value = _product(values)
            else:
                if type(node.exponent) is not int or node.exponent < 0:
                    return None
                value = _power(values[0], node.exponent)
            results[id(node)] = (node, value)
        else:
            return None
    return results[id(source)][1]


def check_substitution(source, bindings, value) -> bool:
    """v0.7's check of substitute: the value is model.substitute_expression(source, dict(bindings)), for bindings
    given as (Symbol, expression) pairs; each replacement is normalized first, as that function does."""
    replacements = {}
    for symbol, raw in bindings:
        replacement = substituted(raw, {})
        if replacement is None:
            return False
        replacements[symbol] = replacement
    expected = substituted(source, replacements)
    return expected is not None and type(value) is type(expected) and value == expected
