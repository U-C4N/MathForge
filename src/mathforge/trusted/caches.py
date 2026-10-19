"""Bounded, process-wide caches whose hits cost the active budget what the miss cost.

A kernel or an independent checker keeps what an exact computation produced (a Sturm chain, a table of
constants) in a module-level ``BoundedCache`` and asks it for the value with ``get(key, compute)``. The cache
makes a computation faster without making it observable:

* A miss runs ``compute()`` and records what it charged the active budget: the growth of the four counters
  ``work``, ``node_count``, ``refinements`` and ``evidence_count``.
* A hit returns the stored value and charges the active budget those four amounts again, the nominal cost, through
  the budget's own limit checks. Work counters, and the limit errors they raise, are the same whether or not an
  earlier call filled the cache. With no active budget, a hit charges nothing, as any exact operation outside
  ``computation()`` charges nothing.
* ``get`` is thread-safe. ``compute()`` runs outside the lock, so it may use this cache or any other. When two
  threads miss on one key at once, both compute and the first to finish wins: every caller then receives that one
  stored object.

What the owner of a cache promises: the key fully determines the value (``compute`` reads nothing else), and the
value is immutable (an int, a tuple of ints, a frozen record), because a hit returns the stored object itself. A
producer and its independent checker keep separate instances, so one side never serves the other.

Entries are also keyed by the active budget's limits, all of them, and by whether a budget is active at all. A
replay of counter growth cannot tell whether a computation would have passed ``check_int``, ``product``, ``matrix``,
``inspect`` or a cap read from ``budget.limits`` under other limits, so a value computed under one set of limits is
never served under another, and an entry made with no budget is never served to a budget, which would charge it
nothing. A hit therefore differs from a cold computation only in the counters' starting state, which the replay
applies. The cost is a second computation when callers differ in their limits, which only an explicit
``limits=`` argument does.

A failed ``compute`` stores nothing. When several limits would be exceeded by one replay, the error names the first
that the replay reaches, which can differ from the one a cold computation reaches; the class and code are the same.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import fields
from operator import attrgetter
import threading
from typing import Any, TypeVar
import weakref

from ..errors import InvalidInput
from ..limits import Budget, ComputationLimits, current_budget

__all__ = ["BoundedCache", "clear_all"]

T = TypeVar("T")

Counters = tuple[int, int, int, int]  # work, node_count, refinements, evidence_count

_CACHES: weakref.WeakSet[BoundedCache] = weakref.WeakSet()  # a dropped cache must not be kept alive by the registry
_CACHES_LOCK = threading.Lock()
_LIMITS_OF = attrgetter(*(field.name for field in fields(ComputationLimits)))  # Every limit, also one added later.


def _counters(budget: Budget) -> Counters:
    return (budget.work, budget.node_count, budget.refinements, budget.evidence_count)


def _charge(budget: Budget, deltas: Counters) -> None:
    """Charge a budget the recorded growth of its counters, enforcing its limits.

    ``Budget.refine(n)`` ticks ``work`` by ``n`` as well, so the work still to charge after the refinements is the
    rest of the recorded work. ``refine`` is the only charge that moves ``refinements``, so ``work`` never lags it.
    """
    work, nodes, refinements, evidence = deltas
    if nodes:
        budget.nodes(nodes)
    if evidence:
        budget.evidence(evidence)
    if refinements:
        budget.refine(refinements)
    if work > refinements:
        budget.tick(work - refinements)


class BoundedCache:
    """A least-recently-used cache of at most ``maxsize`` values that charges hits their nominal cost."""

    __slots__ = ("name", "maxsize", "_entries", "_lock", "__weakref__")

    def __init__(self, name: str, maxsize: int) -> None:
        if type(name) is not str or not name:
            raise InvalidInput("A cache name must be a nonempty string")
        if type(maxsize) is not int or maxsize <= 0:
            raise InvalidInput("A cache maxsize must be a positive integer")
        self.name = name
        self.maxsize = maxsize
        # (scope, key) -> (value, recorded counter growth); the growth is None exactly when the scope is None.
        self._entries: OrderedDict[Hashable, tuple[Any, Counters | None]] = OrderedDict()
        self._lock = threading.Lock()
        with _CACHES_LOCK:
            _CACHES.add(self)

    def get(self, key: Hashable, compute: Callable[[], T]) -> T:
        """Return the value stored for ``key``, computing it first on a miss.

        A hit charges the active budget the counter growth recorded by the miss, and may raise
        ``ResourceLimitError`` as that computation would have. Exceptions from ``compute`` propagate and store nothing.
        """
        budget = current_budget()
        scope = None if budget is None else _LIMITS_OF(budget.limits)
        slot = (scope, key)
        entries = self._entries
        with self._lock:
            entry = entries.get(slot)
            if entry is not None:
                entries.move_to_end(slot)
        if entry is not None:
            value, deltas = entry
            if budget is not None and deltas is not None:
                _charge(budget, deltas)
            return value
        if budget is None:
            value = compute()
            deltas = None
        else:
            before = _counters(budget)
            value = compute()
            after = _counters(budget)
            deltas = (max(0, after[0] - before[0]), max(0, after[1] - before[1]),
                      max(0, after[2] - before[2]), max(0, after[3] - before[3]))
        with self._lock:
            entry = entries.get(slot)
            if entry is not None:  # Another thread finished first: keep its object, so callers share one.
                entries.move_to_end(slot)
                return entry[0]
            entries[slot] = (value, deltas)
            while len(entries) > self.maxsize:
                entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"BoundedCache({self.name!r}, maxsize={self.maxsize}, size={len(self)})"


def clear_all() -> None:
    """Clear every BoundedCache created in this process, as the speed suite does before each sample."""
    with _CACHES_LOCK:
        caches = list(_CACHES)
    for cache in caches:
        cache.clear()
