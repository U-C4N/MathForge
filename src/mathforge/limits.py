"""Deterministic, operation-local resource budgets for exact computations."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from typing import Iterator

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


class Budget:
    """Nested helpers share counters; verification may request a fresh budget."""

    def __init__(self, limits=None):
        if limits is not None and type(limits) is not ComputationLimits:
            raise InvalidInput("limits must be ComputationLimits or None")
        self.limits = limits if limits is not None else ComputationLimits()
        self.work = self.node_count = self.refinements = self.evidence_count = 0

    def _add(self, name, amount, limit):
        if type(amount) is not int or amount < 0:
            raise InvalidInput("Budget charge must be a nonnegative integer")
        value = getattr(self, name) + amount
        if value > limit:
            raise ResourceLimitError(f"Computation limit exceeded: {name} ({limit})")
        setattr(self, name, value)

    def tick(self, n=1):
        self._add("work", n, self.limits.max_work)

    def nodes(self, n=1):
        self._add("node_count", n, self.limits.max_nodes)

    def refine(self, n=1):
        self._add("refinements", n, self.limits.max_refinements)
        self.tick(n)

    def evidence(self, n=1):
        self._add("evidence_count", n, self.limits.max_evidence_items)

    def check_int(self, value):
        if type(value) is not int:
            raise InvalidInput("An exact integer is required")
        if value.bit_length() > self.limits.max_integer_bits:
            raise ResourceLimitError("Computation limit exceeded: integer bits")

    def product(self, *values):
        self.tick()
        if any(value == 0 for value in values):
            return
        if sum(abs(value).bit_length() for value in values) > self.limits.max_integer_bits + len(values) - 1:
            raise ResourceLimitError("Computation limit exceeded: product integer bits")

    def matrix(self, rows, cols):
        if rows * cols > self.limits.max_matrix_entries:
            raise ResourceLimitError("Computation limit exceeded: matrix entries")

    def inspect(self, value, *, evidence=False):
        """Bound traversal without recursion; inspect only known data containers."""
        from dataclasses import is_dataclass
        from enum import Enum
        stack = [(value, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > self.limits.max_depth:
                raise ResourceLimitError("Computation limit exceeded: expression depth")
            (self.evidence if evidence else self.nodes)()
            if type(node) is int:
                self.check_int(node)
            elif isinstance(node, Enum):
                continue
            elif type(node) in (tuple, list):
                stack.extend((item, depth + 1) for item in node)
            elif type(node) is dict:
                stack.extend((item, depth + 1) for pair in node.items() for item in pair)
            elif is_dataclass(node) and not isinstance(node, type):
                stack.extend((getattr(node, f.name), depth + 1) for f in fields(node))


_ACTIVE: ContextVar[Budget | None] = ContextVar("mathforge_budget", default=None)


def current_budget() -> Budget | None:
    return _ACTIVE.get()


@contextmanager
def computation(limits=None, *, fresh=False) -> Iterator[Budget]:
    if limits is not None and type(limits) is not ComputationLimits:
        raise InvalidInput("limits must be ComputationLimits or None")
    active = current_budget()
    if active is not None and not fresh:
        yield active
        return
    budget = Budget(limits)
    token = _ACTIVE.set(budget)
    try:
        yield budget
    finally:
        _ACTIVE.reset(token)
