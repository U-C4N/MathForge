"""Lean function of the legacy substitute operation for exact expression trees.

An expression of Rational, Symbol, Add, Mul, Pow and Sqrt nodes with a dict of bindings keyed by unconstrained real
Symbols (whose replacements the model accepts unconditionally) is substituted by mathforge.kernels.substitution and
checked by mathforge.certify.substitution, which computes v0.7's criterion, ``model.substitute_expression(source,
bindings)``, with its own implementation. Every other input, a RationalFunction included, falls back.

The structured path inspects the arguments and its records (the request, the value, the step and the Result) under
the default limits. Here one walk sizes the arguments and the value as ``Budget.inspect`` counts them, with the
depth limit lowered by the depth at which a Result nests an argument and a node limit that leaves room for the
records, which repeat each about three times, so that the lean path falls back wherever a record could reach a
limit. The same walk requires one Symbol uid to carry one name, domain and assumptions throughout, as the verifier
does (verification_v2._symbol_consistency).
"""

from __future__ import annotations

from ..certify import substitution as certify
from ..kernels import substitution as kernels
from ..limits import ComputationLimits, computation, current_budget
from ..model import Add, Mul, Pow, Rational, Sqrt, Symbol
from . import FALLBACK, run_legacy
from ._poly import MAX_BITS, real_variable, small_int

_EXPRESSIONS = frozenset({Rational, Symbol, Add, Mul, Pow, Sqrt})
_DEPTH = ComputationLimits().max_depth - 10  # the Result nests a binding's value 8 levels down, with a margin
_NODES = ComputationLimits().max_nodes // 4


class _TooLarge(Exception):
    pass


class _Walk:
    """Sizes (nodes as Budget.inspect counts them, depth) of expression trees, and the metadata of each Symbol uid."""

    __slots__ = ("symbols", "sizes", "nodes")

    def __init__(self):
        self.symbols = {}
        self.sizes = {}  # id(node) -> (node, nodes, depth)
        self.nodes = 0

    def symbol(self, node):
        metadata = (node.name, node.domain, node.assumptions)
        known = self.symbols.setdefault(node.uid, metadata)
        if known is not metadata and known != metadata:
            raise _TooLarge("one uid with two Symbols")
        return 5 + 3 * len(node.assumptions)

    def measure(self, node):
        """(nodes, depth) of a tree; a shared subtree counts every time it occurs, as inspect walks it."""
        kind = type(node)
        if kind is Rational:
            numerator, denominator = node.numerator, node.denominator
            if (type(numerator) is not int or type(denominator) is not int or numerator.bit_length() > MAX_BITS
                    or denominator.bit_length() > MAX_BITS):
                raise _TooLarge("not a canonical Rational within the integer limit")
            return 3, 1
        if kind is Symbol:
            return self.symbol(node), 2 if not node.assumptions else 3
        entry = self.sizes.get(id(node))
        if entry is not None:
            return entry[1], entry[2]
        if kind is Add or kind is Mul:
            children = node.terms if kind is Add else node.factors
            if type(children) is not tuple:
                raise _TooLarge("not a tuple")
            nodes, depth = 2, 0
            for child in children:
                size, height = self.measure(child)
                nodes += size
                if height > depth:
                    depth = height
            depth += 2
        elif kind is Pow:
            nodes, depth = self.measure(node.base)
            nodes += 2
            depth += 1
        elif kind is Sqrt:
            if type(node.radicand) is not Rational:
                raise _TooLarge("not a Rational radicand")
            nodes, depth = 4, 2
        else:
            raise _TooLarge(kind.__name__)
        if depth > _DEPTH:
            raise _TooLarge("depth")
        self.sizes[id(node)] = (node, nodes, depth)
        return nodes, depth

    def add(self, node):
        self.nodes += self.measure(node)[0]
        if self.nodes > _NODES:
            raise _TooLarge("nodes")


def _exact(value):
    if type(value) is int:
        return Rational(value) if small_int(value) else None
    return value if type(value) in _EXPRESSIONS else None


def _substitute(expression: object, bindings: object) -> object:
    if current_budget() is not None or type(bindings) is not dict:
        return FALLBACK
    source = _exact(expression)
    if source is None:
        return FALLBACK
    pairs = []
    for key, raw in bindings.items():
        value = _exact(raw)
        if not real_variable(key) or value is None:
            return FALLBACK
        pairs.append((key, value))
    try:
        walk = _Walk()
        walk.add(source)
        for key, value in pairs:
            walk.symbol(key)
            walk.add(value)
        with computation(fresh=True):  # The model's rational arithmetic keeps its integer limit.
            replacements = {key.uid: kernels.normalize(value) for key, value in pairs}
            value = kernels.substitute(source, replacements)
            walk.add(value)
            if certify.check_substitution(source, pairs, value) is not True:
                return FALLBACK
        return value
    except Exception:
        return FALLBACK


def substitute(expression: object, bindings: object) -> object:
    """The fast path above, else the generic lean path of the legacy entry for the inputs it leaves (a
    RationalFunction, bindings of constrained symbols, other mappings)."""
    value = _substitute(expression, bindings)
    if value is FALLBACK:
        return run_legacy("substitute", {"expression": expression, "bindings": bindings})
    return value


LEAN = {"substitute": substitute}
