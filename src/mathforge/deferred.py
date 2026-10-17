"""The generic record of an unevaluated request (format v6) and the registry of deferral types.

A v6 operation that reaches no verified answer returns ``Unevaluated(kind, request, reason)`` instead of
a value; its Result has outcome NO_CONCLUSION and carries the record in ``deferred``. A candidate may
travel along, explicitly unclaimed: it is never verified and ``unwrap()`` still raises. The records
serialize as ``v6_unevaluated`` and ``v6_reason`` (``registry.record_types(6)``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import OperationRequest, _immutable_value


REASON_CODES = frozenset({"no_applicable_rule", "outside_decidable_class", "verification_undecided",
                          "branch_ambiguous", "constant_problem"})
KINDS = frozenset({"integral", "limit", "sum", "product", "series", "ode", "pde", "recurrence",
                   "solve", "simplify", "transform", "evaluate"})


@dataclass(frozen=True)
class Reason:
    """Why a request stays unevaluated: a code from REASON_CODES and free explanatory text."""
    code: str
    detail: str = ""

    def __post_init__(self):
        if type(self.code) is not str or type(self.detail) is not str:
            raise TypeError("A deferral reason has a text code and a text detail")
        if self.code not in REASON_CODES:
            raise ValueError("Unknown deferral reason code")


@dataclass(frozen=True)
class Unevaluated:
    """A request left unevaluated. At the top level of a Result, request is the Result's own request."""
    kind: str
    request: OperationRequest
    reason: Reason
    candidate: object | None = None  # Explicitly unclaimed.

    def __post_init__(self):
        if type(self.kind) is not str:
            raise TypeError("An unevaluated kind must be text")
        if self.kind not in KINDS:
            raise ValueError("Unknown unevaluated kind")
        if type(self.request) is not OperationRequest or type(self.reason) is not Reason:
            raise TypeError("An unevaluated record needs an OperationRequest and a Reason")
        object.__setattr__(self, "candidate", _immutable_value(self.candidate))


TYPES = {"reason": Reason, "unevaluated": Unevaluated}


def deferred_types() -> tuple[type, ...]:
    """The types a Result may carry in ``deferred``: the two v5 records, then Unevaluated."""
    from .functions import UnevaluatedIntegral, UnevaluatedLimit  # A producer module: imported on use.
    return UnevaluatedIntegral, UnevaluatedLimit, Unevaluated
