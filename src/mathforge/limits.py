"""Deterministic, operation-local resource budgets for exact computations."""

from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import wraps

from .errors import InvalidInput, MathForgeError


class ResourceLimitError(MathForgeError):
    code = "resource_limit_exceeded"


@dataclass(frozen=True)
class ComputationLimits:
    max_work: int = 2_000_000
    max_integer_bits: int = 65_536
    max_nodes: int = 100_000
    max_depth: int = 128
    max_matrix_entries: int = 4_096
    max_refinements: int = 100_000
    max_evidence_items: int = 100_000

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise InvalidInput(f"{field.name} must be a positive integer")


@dataclass(frozen=True)
class DecodeLimits:
    max_bytes: int = 16 * 1024 * 1024
    max_depth: int = 128
    max_nodes: int = 100_000
    max_integer_digits: int = 20_000
    max_workspace_objects: int = 10_000

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise InvalidInput(f"{field.name} must be a positive integer")


_DEFAULT_LIMITS = ComputationLimits()  # Immutable, so every budget without explicit limits can share it.

# How inspect walks a value, keyed by its exact type: a leaf, an int, a Rational (itself and its two integers), the
# items of a tuple or list, the keys and values of a dict, or, for a dataclass, the names of its walked fields.
_LEAF, _INT, _RATIONAL, _ITEMS, _PAIRS = "leaf", "int", "rational", "items", "pairs"
_SEED_KINDS: dict = {int: _INT, tuple: _ITEMS, list: _ITEMS, dict: _PAIRS, str: _LEAF, type(None): _LEAF,
                     bool: _LEAF, float: _LEAF}
_KINDS = dict(_SEED_KINDS)
_KIND_CACHE_LIMIT = 4_096  # Classes made at run time cannot grow the table without bound.


def _register_rational(cls):
    """Let inspect count model.Rational directly: model imports this module, so this module cannot import model."""
    _SEED_KINDS[cls] = _KINDS[cls] = _RATIONAL


def _kind(cls):
    """Classify a type once, by v0.7's tests in v0.7's order; each of them depends on the type alone.

    A seeded type keeps its seeded kind, whatever the cache holds. A full cache is replaced by a fresh copy of the
    seeds, never cleared in place: an inspect running in another thread meanwhile reads the table it started with,
    which still holds every seeded type, so it never classifies an int or a tuple as a leaf.
    """
    global _KINDS
    kind = _SEED_KINDS.get(cls)
    if kind is not None:
        return kind
    if issubclass(cls, Enum) or issubclass(cls, type) or not is_dataclass(cls):
        kind = _LEAF
    else:
        # A marked field, such as OperationRequest.version, names a format, not content.
        kind = tuple(f.name for f in fields(cls) if f.metadata.get("budget") != "skip")
    kinds = _KINDS
    if len(kinds) >= _KIND_CACHE_LIMIT:
        kinds = _KINDS = dict(_SEED_KINDS)
    kinds[cls] = kind
    return kind


class Budget:
    """Nested helpers share counters; verification may request a fresh budget.

    Every limit is also kept in a private attribute, refreshed whenever ``limits`` is set, so that a charge reads
    and writes this object only.
    """

    __slots__ = ("work", "node_count", "refinements", "evidence_count", "_limits", "_max_work", "_max_bits",
                 "_max_nodes", "_max_depth", "_max_matrix", "_max_refinements", "_max_evidence")

    def __init__(self, limits=None):
        if limits is None:
            limits = _DEFAULT_LIMITS
        elif type(limits) is not ComputationLimits:
            raise InvalidInput("limits must be ComputationLimits or None")
        self._limits = limits  # The limits setter's assignments, inline: every operation builds a budget.
        self._max_work = limits.max_work
        self._max_bits = limits.max_integer_bits
        self._max_nodes = limits.max_nodes
        self._max_depth = limits.max_depth
        self._max_matrix = limits.max_matrix_entries
        self._max_refinements = limits.max_refinements
        self._max_evidence = limits.max_evidence_items
        self.work = self.node_count = self.refinements = self.evidence_count = 0

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):  # Keep in step with __init__.
        self._limits = limits
        self._max_work = limits.max_work
        self._max_bits = limits.max_integer_bits
        self._max_nodes = limits.max_nodes
        self._max_depth = limits.max_depth
        self._max_matrix = limits.max_matrix_entries
        self._max_refinements = limits.max_refinements
        self._max_evidence = limits.max_evidence_items

    def tick(self, n=1):
        if type(n) is not int or n < 0:
            raise InvalidInput("Budget charge must be a nonnegative integer")
        work = self.work + n
        if work > self._max_work:
            raise ResourceLimitError(f"Computation limit exceeded: work ({self._max_work})")
        self.work = work

    def nodes(self, n=1):
        if type(n) is not int or n < 0:
            raise InvalidInput("Budget charge must be a nonnegative integer")
        count = self.node_count + n
        if count > self._max_nodes:
            raise ResourceLimitError(f"Computation limit exceeded: node_count ({self._max_nodes})")
        self.node_count = count

    def refine(self, n=1):
        if type(n) is not int or n < 0:
            raise InvalidInput("Budget charge must be a nonnegative integer")
        refinements = self.refinements + n
        if refinements > self._max_refinements:
            raise ResourceLimitError(f"Computation limit exceeded: refinements ({self._max_refinements})")
        self.refinements = refinements
        work = self.work + n
        if work > self._max_work:
            raise ResourceLimitError(f"Computation limit exceeded: work ({self._max_work})")
        self.work = work

    def evidence(self, n=1):
        if type(n) is not int or n < 0:
            raise InvalidInput("Budget charge must be a nonnegative integer")
        count = self.evidence_count + n
        if count > self._max_evidence:
            raise ResourceLimitError(f"Computation limit exceeded: evidence_count ({self._max_evidence})")
        self.evidence_count = count

    def check_int(self, value):
        if type(value) is not int:
            raise InvalidInput("An exact integer is required")
        if value.bit_length() > self._max_bits:
            raise ResourceLimitError("Computation limit exceeded: integer bits")

    def charge_rational(self, numerator, denominator):
        """Charge one exact rational result: one work unit, and both of its integers within max_integer_bits.

        The integers are trusted to be ints; check_int is the method that also checks their type.
        """
        work = self.work + 1
        if work > self._max_work:
            raise ResourceLimitError(f"Computation limit exceeded: work ({self._max_work})")
        self.work = work
        if numerator.bit_length() > self._max_bits or denominator.bit_length() > self._max_bits:
            raise ResourceLimitError("Computation limit exceeded: integer bits")

    def product(self, *values):
        work = self.work + 1
        if work > self._max_work:
            raise ResourceLimitError(f"Computation limit exceeded: work ({self._max_work})")
        self.work = work
        for value in values:
            if value == 0:
                return
        bits = 0
        for value in values:
            bits += abs(value).bit_length()
        if bits > self._max_bits + len(values) - 1:
            raise ResourceLimitError("Computation limit exceeded: product integer bits")

    def matrix(self, rows, cols):
        if rows * cols > self._max_matrix:
            raise ResourceLimitError("Computation limit exceeded: matrix entries")

    def inspect(self, value, *, evidence=False):
        """Bound traversal without recursion; inspect only known data containers.

        A dataclass field marked ``metadata={"budget": "skip"}`` is left out. Any other field is walked, whether or
        not it takes part in comparison. Every visited value counts one node (or one evidence item), and a Rational
        counts three: itself, its denominator and its numerator, visited in that order one level down, exactly as
        v0.7 walked its fields. The walk order is v0.7's, so the first limit to fail is the same.
        """
        if evidence:
            count, limit, counter = self.evidence_count, self._max_evidence, "evidence_count"
        else:
            count, limit, counter = self.node_count, self._max_nodes, "node_count"
        max_depth, max_bits, kinds = self._max_depth, self._max_bits, _KINDS
        stack = [(value, 0)]
        pop, extend = stack.pop, stack.extend
        try:
            while stack:
                node, depth = pop()
                if depth > max_depth:
                    raise ResourceLimitError("Computation limit exceeded: expression depth")
                if count >= limit:
                    raise ResourceLimitError(f"Computation limit exceeded: {counter} ({limit})")
                count += 1
                kind = kinds.get(type(node))
                if kind is None:
                    kind = _kind(type(node))
                if kind is _RATIONAL:
                    if depth == max_depth:
                        raise ResourceLimitError("Computation limit exceeded: expression depth")
                    for part in (node.denominator, node.numerator):
                        if count >= limit:
                            raise ResourceLimitError(f"Computation limit exceeded: {counter} ({limit})")
                        count += 1
                        if part.bit_length() > max_bits:
                            raise ResourceLimitError("Computation limit exceeded: integer bits")
                elif kind is _INT:
                    if node.bit_length() > max_bits:
                        raise ResourceLimitError("Computation limit exceeded: integer bits")
                elif kind is _LEAF:
                    continue
                elif kind is _ITEMS:
                    depth += 1
                    extend([(item, depth) for item in node])
                elif kind is _PAIRS:
                    depth += 1
                    extend([(item, depth) for pair in node.items() for item in pair])
                else:
                    depth += 1
                    extend([(getattr(node, name), depth) for name in kind])
        finally:
            if evidence:
                self.evidence_count = count
            else:
                self.node_count = count


_ACTIVE: ContextVar[Budget | None] = ContextVar("mathforge_budget", default=None)


def current_budget() -> Budget | None:
    return _ACTIVE.get()


class computation:
    """``with computation(limits=None, *, fresh=False) as budget``: share the active budget, or open a new one.

    Inside an active budget, a nested computation() yields that budget and ignores its own limits, so nested helpers
    cannot reset the counters; fresh=True always opens a new budget. The limits are validated on entry, and the
    object also decorates a function with one new context per call, as v0.7's generator-based manager did.
    """

    __slots__ = ("_limits", "_fresh", "_token")

    def __init__(self, limits=None, *, fresh=False):
        self._limits = limits
        self._fresh = fresh
        self._token = None

    def __enter__(self) -> Budget:
        limits = self._limits
        if limits is not None and type(limits) is not ComputationLimits:
            raise InvalidInput("limits must be ComputationLimits or None")
        if self._token is not None:
            raise RuntimeError("This computation() context is already active; nest a new computation() instead")
        active = _ACTIVE.get()
        if active is not None and not self._fresh:
            return active
        budget = Budget(limits)  # Looked up at entry, so a patched limits.Budget sees every budget.
        self._token = _ACTIVE.set(budget)
        return budget

    def __exit__(self, exc_type, exc, traceback):
        token = self._token
        if token is not None:
            self._token = None
            _ACTIVE.reset(token)

    def __call__(self, function):
        limits, fresh = self._limits, self._fresh

        @wraps(function)
        def within(*args, **kwargs):
            with computation(limits, fresh=fresh):
                return function(*args, **kwargs)
        return within
