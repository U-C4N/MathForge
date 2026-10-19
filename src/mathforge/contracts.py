"""Immutable problems, result axes, and auditable mathematical claims.

These types describe evidence, not formal proofs. Stored verification labels
can always be rechecked through :func:`mathforge.verify`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TYPE_CHECKING, TypeVar

from .errors import OperationError
from .model import Assumption, Domain, Expression, Rational, Reals, Symbol, Truth, as_expression, _extension
from .polynomial import Polynomial

if TYPE_CHECKING:
    from .rational_function import RationalFunction
    from .matrices import LinearSystem, Matrix


def as_scalar(value):
    if _extension(value, "rational_function", "RationalFunction"):
        return value
    return as_expression(value)


# The v2-v4 record classes that evidence may hold, by class name: the module (below mathforge) that defines each.
# Computed once; _extended_value compares a value's exact class with the class that module defines now.
_EXTENSION_MODULES = {name: module for module, names in (
    ("polynomial", ("SquareFreeFactor", "SquareFreeDecomposition")),
    ("polynomial_exact", ("PolynomialFactor", "PolynomialFactorization")),
    ("exact_extensions", ("PartialFractionTerm", "PartialFractionDecomposition")),
    ("algebraic", ("RootRecord",)),
    ("rational_function", ("RationalFunction",)),
    ("matrices", ("Matrix", "RREFResult", "LinearSystem", "AffineSolutionSet")),
    ("intervals", ("Interval", "IntervalSet")),
    ("analysis_types", ("LimitValue", "ExcludedPoint", "MonotonicRegion", "LocalExtremum", "FunctionAnalysis",
                        "ExtremumValue")),
) for name in names}
_registry: Any = None  # mathforge.registry, imported on first use: contracts loads very early (no import cycles).


def _extended_value(value):
    global _registry
    cls = type(value)
    registry = _registry
    if registry is None:
        from . import registry
        _registry = registry
    if registry.record_module(cls.__module__):  # Static: other values never build the registry.
        return cls in registry.all_record_types()
    module = _EXTENSION_MODULES.get(cls.__name__)
    return module is not None and _extension(value, module, cls.__name__)


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


@dataclass(frozen=True, init=False)
class Eq:
    lhs: Expression | RationalFunction
    rhs: Expression | RationalFunction

    def __init__(self, lhs: Expression | RationalFunction | int,
                 rhs: Expression | RationalFunction | int):
        object.__setattr__(self, "lhs", as_scalar(lhs))
        object.__setattr__(self, "rhs", as_scalar(rhs))

    def __post_init__(self):
        object.__setattr__(self, "lhs", as_scalar(self.lhs))
        object.__setattr__(self, "rhs", as_scalar(self.rhs))

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"


@dataclass(frozen=True, init=False)
class Inequality:
    lhs: Expression | RationalFunction
    relation: str
    rhs: Expression | RationalFunction

    def __init__(self, lhs: Expression | RationalFunction | int, relation: str,
                 rhs: Expression | RationalFunction | int):
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "rhs", rhs)
        self.__post_init__()

    def __post_init__(self):
        object.__setattr__(self, "lhs", as_scalar(self.lhs))
        object.__setattr__(self, "rhs", as_scalar(self.rhs))
        if self.relation not in ("lt", "le", "gt", "ge"):
            raise ValueError("Inequality relation must be lt, le, gt or ge")


@dataclass(frozen=True)
class OperationRequest:
    operation: str
    arguments: tuple[tuple[str, Any], ...]
    # The format version of a request of format 6, as a document carried it; None otherwise. Format 6 is the only
    # request format this build writes, so no other version can be named, and to_data never writes one that this
    # build cannot read back. It names no mathematics: equality, hashing and repr ignore it, and the metadata keeps
    # Budget.inspect from counting it. Serialization checks it against the operation.
    version: int | None = field(default=None, compare=False, repr=False, metadata={"budget": "skip"})

    def __post_init__(self):
        if self.version is not None:
            if type(self.version) is not int:
                raise TypeError("An operation request version must be an integer.")
            if self.version != 6:
                raise ValueError(f"Unsupported operation request format version {self.version}: "
                                 "a request names format 6, the only one this build writes.")
        _text(self.operation, "operation", empty=False)
        pairs = tuple(self.arguments)
        if any(type(p) not in (tuple, list) or len(p) != 2 for p in pairs):
            raise TypeError("Arguments must contain name/value pairs")
        if any(type(name) is not str or not name for name, _ in pairs):
            raise TypeError("Argument names must be nonempty text")
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate operation argument")
        from .limits import current_budget
        budget = current_budget()
        if budget is not None:
            budget.inspect(pairs)
        object.__setattr__(self, "arguments", tuple(sorted(
            (name, _immutable_value(value)) for name, value in pairs)))

    def get(self, name):
        for key, value in self.arguments:
            if name == key:
                return value
        raise KeyError(name)


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
        from .limits import current_budget
        budget = current_budget()
        if budget is not None:
            budget.inspect((self.inputs, self.outputs, self.preconditions), evidence=True)
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


_CORE_RECORDS = frozenset((Polynomial, Eq, Inequality, OperationRequest, FiniteSet, EmptySet, UniversalSet,
                           Condition, Check, VerificationReport, ErrorInfo, Domain, Assumption, Truth))


def _immutable_value(value):
    """Only supported immutable mathematical values enter evidence records."""
    cls = type(value)
    if value is None or cls is str or cls is int or cls is bool:
        return value
    if cls is tuple or cls is list:  # Exact containers are no record class, so the record tests below miss them.
        return tuple([_immutable_value(item) for item in value])
    if isinstance(value, Expression):
        # The same concrete AST allowlist is used by operators and persistence.
        # An arbitrary subclass can introduce mutable fields and has no codec.
        return as_expression(value)
    if cls in _CORE_RECORDS:
        return value
    if _extended_value(value):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    raise TypeError(f"Unsupported result/evidence value: {cls.__name__}.")


T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    execution_status: ExecutionStatus
    outcome: Outcome
    value: T | None = None
    exactness: Exactness = Exactness.NOT_APPLICABLE
    completeness: Completeness = Completeness.NOT_APPLICABLE
    verification: tuple[Check, ...] = ()
    method: str = ""
    domain: Domain | None = None
    assumptions: tuple[Assumption, ...] = ()
    steps: tuple[Step, ...] = ()
    problem: Eq | Inequality | LinearSystem | None = None
    expression: Expression | Polynomial | RationalFunction | Matrix | None = None
    for_: Symbol | None = None
    precision: int | None = None
    tolerance: Rational | None = None
    error_bound: Rational | None = None
    error: ErrorInfo | None = None
    request: OperationRequest | None = None
    deferred: Any | None = None

    def __post_init__(self):
        for name, enum_type in (("execution_status", ExecutionStatus), ("outcome", Outcome),
                                ("exactness", Exactness), ("completeness", Completeness)):
            _enum(getattr(self, name), enum_type, name)
        object.__setattr__(self, "value", _immutable_value(self.value))
        if self.deferred is not None:
            object.__setattr__(self, "deferred", _immutable_value(self.deferred))
            from .deferred import Unevaluated, deferred_types  # Imported on use: deferred imports contracts.
            if type(self.deferred) not in deferred_types():
                raise TypeError("deferred must be an unevaluated record")
            if type(self.deferred) is Unevaluated and self.deferred.request != self.request:
                raise ValueError("An unevaluated record must carry its result's request")
            if (self.execution_status is not ExecutionStatus.COMPLETED
                    or self.outcome is not Outcome.NO_CONCLUSION or self.value is not None
                    or self.completeness is not Completeness.UNKNOWN
                    or self.exactness is not Exactness.NOT_APPLICABLE):
                raise ValueError("An unevaluated request cannot claim a computed answer")
        object.__setattr__(self, "verification", _typed_tuple(self.verification, Check, "verification"))
        object.__setattr__(self, "assumptions", _typed_tuple(self.assumptions, Assumption, "assumptions"))
        object.__setattr__(self, "steps", _typed_tuple(self.steps, Step, "steps"))
        _text(self.method, "method")
        _check_source(self.problem, self.expression, self.domain, self.for_)
        for name, item_type in (("tolerance", Rational),
                                ("error_bound", Rational), ("error", ErrorInfo),
                                ("request", OperationRequest)):
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
        _check_outcome(self.outcome, self.value)

    @property
    def solution_set(self):
        """The represented solution set, or None if no set was established."""
        if (self.execution_status is ExecutionStatus.COMPLETED
                and self.outcome in (Outcome.SOLUTIONS, Outcome.CANDIDATES,
                                     Outcome.NO_SOLUTION, Outcome.PARTIAL)
                and (type(self.value) in (FiniteSet, EmptySet, UniversalSet)
                     or _extension(self.value, "matrices", "AffineSolutionSet")
                     or _extension(self.value, "intervals", "IntervalSet"))):
            return self.value
        return None

    def unwrap(self) -> T:
        if self.execution_status is not ExecutionStatus.COMPLETED or self.value is None:
            raise OperationError(self)
        return self.value


# The expression classes that a Result keeps without as_scalar, besides Polynomial: by class name, their module.
_RESULT_EXPRESSIONS = {"Matrix": "matrices", "RealExpr": "functions", "RealFunction": "functions",
                       "RationalPrimitive": "rational_integration"}


def _check_source(problem, expression, domain, for_):
    """A Result's checks of its source fields problem, expression, domain and for_, in Result's order."""
    if problem is not None and not (type(problem) in (Eq, Inequality)
            or _extension(problem, "matrices", "LinearSystem")):
        raise TypeError("problem must be a supported mathematical problem")
    if expression is not None:
        cls = type(expression)
        module = _RESULT_EXPRESSIONS.get(cls.__name__)
        if not (cls is Polynomial or (module is not None and _extension(expression, module, cls.__name__))):
            as_scalar(expression)
    if domain is not None and type(domain) is not Domain:
        raise TypeError("domain must be a supported Domain or None.")
    if for_ is not None and type(for_) is not Symbol:
        raise TypeError("for_ must be a supported Symbol or None.")


def _check_outcome(outcome, value):
    """A Result's checks that its outcome fits its value."""
    if outcome is Outcome.NO_SOLUTION and type(value) is not EmptySet:
        raise ValueError("A no_solution outcome requires an EmptySet.")
    if outcome is Outcome.SOLUTIONS and not (
            type(value) is UniversalSet
            or (type(value) is FiniteSet and len(value) > 0)
            or _extension(value, "matrices", "AffineSolutionSet")
            or (_extension(value, "intervals", "IntervalSet") and len(value) > 0)):
        raise ValueError("A solutions outcome requires a nonempty represented solution set.")
    if outcome is Outcome.CANDIDATES and not (
            type(value) is FiniteSet and len(value) > 0):
        raise ValueError("A candidates outcome requires a nonempty FiniteSet.")
    if outcome is Outcome.NO_CONCLUSION and value is not None:
        raise ValueError("A no_conclusion outcome cannot carry a mathematical answer.")


# Trusted construction, for the structured pipelines only (operations, operations_v5 and the verifiers). A trusted
# builder takes fields that its caller has already validated and normalized as the public constructor would, from
# pieces that the active budget has already inspected, and runs no __post_init__: no check and no inspection is
# repeated. The public constructors keep all of their validation.

_new_record = object.__new__
_set_field = object.__setattr__
_RESULT_DEFAULTS = {name: field.default for name, field in Result.__dataclass_fields__.items()}


def _trusted_request(operation: str, arguments: tuple[tuple[str, Any], ...]) -> OperationRequest:
    """An OperationRequest without version: a nonempty operation name and the pairs of _request_arguments."""
    request = _new_record(OperationRequest)
    _set_field(request, "__dict__", {"operation": operation, "arguments": arguments, "version": None})
    return request


def _request_arguments(arguments: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """An OperationRequest's pairs for arguments by distinct nonempty names: each value normalized, then sorted."""
    return tuple(sorted([(name, _immutable_value(value)) for name, value in arguments.items()]))


def _trusted_step(rule: str, inputs: tuple[Any, ...], outputs: tuple[Any, ...], relation: StepRelation) -> Step:
    """A Step without preconditions, audit or explanation: a nonempty rule and normalized inputs and outputs."""
    step = _new_record(Step)
    _set_field(step, "__dict__", {"rule": rule, "inputs": inputs, "outputs": outputs, "preconditions": (),
                                  "relation": relation, "audit": (), "explanation": None})
    return step


def _trusted_result(fields: dict[str, Any]) -> Result[Any]:
    """A Result of the given fields, the others at their defaults: each field one that Result.__post_init__ keeps
    as it stands."""
    result = _new_record(Result)
    values = dict(_RESULT_DEFAULTS)
    values.update(fields)
    _set_field(result, "__dict__", values)
    return result


def _trusted_check(claim: str, status: CheckStatus, level: VerificationLevel, method: str,
                   details: str = "") -> Check:
    """A Check without subject: a nonempty claim, enum members and text."""
    check = _new_record(Check)
    _set_field(check, "__dict__", {"claim": claim, "status": status, "level": level, "method": method,
                                   "subject": None, "details": details})
    return check


def _trusted_report(checks: tuple[Check, ...]) -> VerificationReport:
    """A VerificationReport of a tuple of Checks."""
    report = _new_record(VerificationReport)
    _set_field(report, "__dict__", {"checks": checks})
    return report


def _attach(record: Any, name: str, value: Any) -> None:
    """Set a field of a record that its builder has not handed out yet: a Result's verification, a Step's audit."""
    _set_field(record, name, value)
