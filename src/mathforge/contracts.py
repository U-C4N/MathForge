"""Immutable problems, result axes, and auditable mathematical claims.

These types describe evidence, not formal proofs. Stored verification labels
can always be rechecked through :func:`mathforge.verify`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import OperationError
from .model import Assumption, Domain, Expression, Rational, Reals, Symbol, Truth, as_expression
from .polynomial import Polynomial


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"
    INVALID_INPUT = "invalid_input"
    ERROR = "error"


class Outcome(str, Enum):
    VALUE = "value"
    SOLUTIONS = "solutions"
    CANDIDATES = "candidates"
    NO_SOLUTION = "no_solution"
    NO_CONCLUSION = "no_conclusion"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class Exactness(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    MIXED = "mixed"
    NOT_APPLICABLE = "not_applicable"


class Completeness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class CheckStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNKNOWN = "unknown"
    UNCHECKED = "unchecked"


class VerificationLevel(str, Enum):
    NONE = "none"
    EXACT_SUBSTITUTION = "exact_substitution"
    ALGORITHMIC_CHECK = "algorithmic_check"


class StepRelation(str, Enum):
    EQUIVALENCE = "equivalence"
    IMPLICATION = "implication"
    APPROXIMATION = "approximation"
    DERIVATION = "derivation"
    EVALUATION = "evaluation"


def _enum(value, enum_type, name):
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be {enum_type.__name__}.")


def _text(value, name, *, empty=True):
    if type(value) is not str or (not empty and not value):
        raise TypeError(f"{name} must be {'nonempty ' if not empty else ''}text.")


def _typed_tuple(values, item_type, name):
    if isinstance(values, (str, bytes, dict)):
        raise TypeError(f"{name} must be an iterable of {item_type.__name__}.")
    result = tuple(values)
    if any(type(value) is not item_type for value in result):
        raise TypeError(f"{name} must contain only {item_type.__name__}.")
    return result


@dataclass(frozen=True)
class Eq:
    lhs: Expression
    rhs: Expression

    def __post_init__(self):
        object.__setattr__(self, "lhs", as_expression(self.lhs))
        object.__setattr__(self, "rhs", as_expression(self.rhs))

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"


@dataclass(frozen=True)
class FiniteSet:
    values: tuple[Expression, ...]
    domain: Domain = Reals

    def __post_init__(self):
        _enum(self.domain, Domain, "domain")
        values = tuple(dict.fromkeys(as_expression(v) for v in self.values))
        object.__setattr__(self, "values", values)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return "{" + ", ".join(map(str, self.values)) + "}"


@dataclass(frozen=True)
class EmptySet:
    domain: Domain = Reals

    def __post_init__(self):
        _enum(self.domain, Domain, "domain")

    def __str__(self):
        return "{}"


@dataclass(frozen=True)
class UniversalSet:
    domain: Domain = Reals

    def __post_init__(self):
        _enum(self.domain, Domain, "domain")

    def __str__(self):
        return f"All({self.domain.value})"


@dataclass(frozen=True)
class Condition:
    lhs: Expression
    relation: str
    rhs: Expression
    truth: Truth = Truth.UNKNOWN

    def __post_init__(self):
        object.__setattr__(self, "lhs", as_expression(self.lhs))
        object.__setattr__(self, "rhs", as_expression(self.rhs))
        if self.relation not in ("eq", "ne", "lt", "le", "gt", "ge"):
            raise ValueError("Unknown condition relation.")
        _enum(self.truth, Truth, "truth")


@dataclass(frozen=True)
class Check:
    claim: str
    status: CheckStatus
    level: VerificationLevel
    method: str
    subject: Expression | None = None
    details: str = ""

    def __post_init__(self):
        _text(self.claim, "claim", empty=False)
        _enum(self.status, CheckStatus, "status")
        _enum(self.level, VerificationLevel, "level")
        _text(self.method, "method")
        _text(self.details, "details")
        if self.subject is not None:
            object.__setattr__(self, "subject", as_expression(self.subject))


@dataclass(frozen=True)
class VerificationReport:
    checks: tuple[Check, ...]

    def __post_init__(self):
        object.__setattr__(self, "checks", _typed_tuple(self.checks, Check, "checks"))

    @property
    def verified(self) -> bool:
        return bool(self.checks) and all(c.status is CheckStatus.VERIFIED for c in self.checks)


@dataclass(frozen=True)
class Step:
    rule: str
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    preconditions: tuple[Condition, ...] = ()
    relation: StepRelation = StepRelation.EQUIVALENCE
    audit: tuple[Check, ...] = ()
    explanation: str | None = None

    def __post_init__(self):
        _text(self.rule, "rule", empty=False)
        object.__setattr__(self, "inputs", tuple(_immutable_value(v) for v in self.inputs))
        object.__setattr__(self, "outputs", tuple(_immutable_value(v) for v in self.outputs))
        object.__setattr__(self, "preconditions", _typed_tuple(self.preconditions, Condition, "preconditions"))
        _enum(self.relation, StepRelation, "relation")
        object.__setattr__(self, "audit", _typed_tuple(self.audit, Check, "audit"))
        if self.explanation is not None:
            _text(self.explanation, "explanation")


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    message: str

    def __post_init__(self):
        _text(self.code, "code", empty=False)
        _text(self.message, "message")


def _immutable_value(value):
    """Only supported immutable mathematical values enter evidence records."""
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, Expression):
        # The same concrete AST allowlist is used by operators and persistence.
        # An arbitrary subclass can introduce mutable fields and has no codec.
        return as_expression(value)
    if type(value) in (Polynomial, Eq, FiniteSet, EmptySet, UniversalSet,
                       Condition, Check, VerificationReport, ErrorInfo, Domain, Assumption):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    raise TypeError(f"Unsupported result/evidence value: {type(value).__name__}.")


@dataclass(frozen=True)
class Result:
    execution_status: ExecutionStatus
    outcome: Outcome
    value: Any = None
    exactness: Exactness = Exactness.NOT_APPLICABLE
    completeness: Completeness = Completeness.NOT_APPLICABLE
    verification: tuple[Check, ...] = ()
    method: str = ""
    domain: Domain | None = None
    assumptions: tuple[Assumption, ...] = ()
    steps: tuple[Step, ...] = ()
    problem: Eq | None = None
    expression: Expression | None = None
    for_: Symbol | None = None
    precision: int | None = None
    tolerance: Rational | None = None
    error_bound: Rational | None = None
    error: ErrorInfo | None = None

    def __post_init__(self):
        for name, enum_type in (("execution_status", ExecutionStatus), ("outcome", Outcome),
                                ("exactness", Exactness), ("completeness", Completeness)):
            _enum(getattr(self, name), enum_type, name)
        object.__setattr__(self, "value", _immutable_value(self.value))
        object.__setattr__(self, "verification", _typed_tuple(self.verification, Check, "verification"))
        object.__setattr__(self, "assumptions", _typed_tuple(self.assumptions, Assumption, "assumptions"))
        object.__setattr__(self, "steps", _typed_tuple(self.steps, Step, "steps"))
        _text(self.method, "method")
        for name, item_type in (("domain", Domain), ("problem", Eq), ("expression", Expression),
                                ("for_", Symbol), ("tolerance", Rational),
                                ("error_bound", Rational), ("error", ErrorInfo)):
            value = getattr(self, name)
            if value is not None:
                if item_type is Expression and isinstance(value, Expression):
                    as_expression(value)
                elif type(value) is not item_type:
                    raise TypeError(f"{name} must be a supported {item_type.__name__} or None.")
        if self.precision is not None and (type(self.precision) is not int or self.precision <= 0):
            raise ValueError("precision must be a positive integer or None.")
        for name in ("tolerance", "error_bound"):
            value = getattr(self, name)
            if value is not None and value.numerator < 0:
                raise ValueError(f"{name} cannot be negative.")
        if self.execution_status is not ExecutionStatus.COMPLETED:
            if (self.outcome is not Outcome.NOT_APPLICABLE or self.value is not None
                    or self.exactness is not Exactness.NOT_APPLICABLE
                    or self.completeness is not Completeness.NOT_APPLICABLE):
                raise ValueError("Uncompleted operations cannot claim a mathematical result.")
        if self.outcome is Outcome.NO_SOLUTION and type(self.value) is not EmptySet:
            raise ValueError("A no_solution outcome requires an EmptySet.")
        if self.outcome is Outcome.SOLUTIONS and not (
                type(self.value) is UniversalSet
                or (type(self.value) is FiniteSet and len(self.value) > 0)):
            raise ValueError("A solutions outcome requires a nonempty represented solution set.")
        if self.outcome is Outcome.CANDIDATES and not (
                type(self.value) is FiniteSet and len(self.value) > 0):
            raise ValueError("A candidates outcome requires a nonempty FiniteSet.")
        if self.outcome is Outcome.NO_CONCLUSION and self.value is not None:
            raise ValueError("A no_conclusion outcome cannot carry a mathematical answer.")

    @property
    def solution_set(self):
        """The represented solution set, or None if no set was established."""
        if (self.execution_status is ExecutionStatus.COMPLETED
                and self.outcome in (Outcome.SOLUTIONS, Outcome.CANDIDATES,
                                     Outcome.NO_SOLUTION, Outcome.PARTIAL)
                and type(self.value) in (FiniteSet, EmptySet, UniversalSet)):
            return self.value
        return None

    def unwrap(self):
        if self.execution_status is not ExecutionStatus.COMPLETED or self.value is None:
            raise OperationError(self)
        return self.value
