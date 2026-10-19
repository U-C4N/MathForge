"""The speed-fixture interface: Fixture, ExemptFixture, EdgeCase and the symbol helper.

A fixture module (benchmarks/speed/fixtures/<area>.py) defines ``FIXTURES`` and ``EDGE_CASES`` from these classes.
The suite (benchmarks/speed_suite.py), the v0.7 value snapshot (tools/snapshot_v07_values.py) and the corpus test
(tests/v08/test_values_v07.py) all read them, so the field names and types below are fixed.

A MathForge builder returns ``(public value function, args, kwargs)`` and the caller makes the call: the same triple
is timed by the suite and run by the snapshot, so the function is the public one (``mathforge.integer_gcd``), never a
structured ``operations`` call. Builders are deterministic: a variant's inputs depend on ``i`` alone. SymPy is
imported only inside the ``sympy`` builders, so this module and every fixture module load without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    import mathforge


@dataclass(frozen=True)
class Fixture:
    """One operation on one input shape, timed against the fastest public SymPy call that returns the same object.

    ``size`` is ``"micro"`` (the smallest input that exercises the operation, with at least 1,000 variants of one
    shape) or ``"realistic"`` (the size a student, engineer or agent would plausibly use).
    """

    operation: str  # a name of tests/fixtures/v8/capabilities_v07.json, not "verify"
    size: str  # "micro" or "realistic"
    label: str  # unique within (operation, size)
    mathforge: Callable[[int], tuple[Callable[..., object], tuple, dict]]  # variant i -> (public value function, args, kwargs)
    sympy: Callable[[int], Callable[[], object]]  # variant i -> the timed zero-argument SymPy call, inputs prebuilt
    agree: Callable[[object, object], bool]  # (MathForge value, SymPy value) -> the same mathematical object
    reference: str  # the SymPy call chosen; the candidates measured, with their medians
    variants: int = 1  # distinct inputs of one shape; micro fixtures have at least 1,000
    v07: str = ""  # the benchmarks/fresh_comparison.py case this fixture carries over

    @property
    def id(self) -> str:
        return f"{self.operation}/{self.size}/{self.label}"


@dataclass(frozen=True)
class ExemptFixture:
    """A fixture of ``verify``, which has no SymPy counterpart: its time is held to a budget instead of a ratio."""

    operation: str
    size: str
    label: str
    timed: Callable[[], tuple[Callable[..., object], tuple, dict]]  # mf.verify(result), the result prebuilt
    budget: Callable[[], tuple[Callable[..., object], tuple, dict]]  # producing the same result through mathforge.operations
    reason: str


@dataclass(frozen=True)
class EdgeCase:
    """A value-corpus input that is never timed: an error, an input past a limit or a type edge."""

    operation: str
    label: str
    mathforge: Callable[[], tuple[Callable[..., object], tuple, dict]]


def symbol(name: str, k: int) -> "mathforge.Symbol":
    """A Symbol with the k-th fixed uid, ``uuid.UUID(int=k)``, never a random one.

    Every value built from it, and so every corpus entry that stores such a value, is the same in every process.
    Use one k per distinct symbol of a fixture.
    """
    import mathforge  # Here rather than at module level, so that describing fixtures does not import the package.

    return mathforge.Symbol(name, uid=str(uuid.UUID(int=k)))
