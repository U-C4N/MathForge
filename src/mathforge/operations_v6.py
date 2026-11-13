"""Source-bearing v6 operations: run a registered family's producer, then verify its value independently.

A v6 family module defines SPECS, TYPES, CHECKER (the import path of its independent checker, below
mathforge.certify), produce() and, optionally, RENDER: {record class of its TYPES: render(value, renderer,
child) -> str}, the signature of rendering_v5.render, and SUPERSEDES; the registry checks all of them at build
time. mathforge.areas.ntheory.continued_fraction, the pilot family, spells the contract out.

The generated wrappers call run() with every argument of the spec, defaults filled in. Every request that run()
builds carries its format version, 6. Before produce(), run() checks the argument names against the spec, then
the exact types that some argument kinds admit (_KINDS: an int kind takes an int, never a bool or a float); a
mismatch is invalid input. A producer returns (value, evidence), where value may be an Unevaluated record when it
reaches no answer. It validates the rest of its inputs first and raises InvalidInput for those it rejects. Once
the inputs are validated, an InvalidInput raised while building a record is a producer bug that run() cannot
tell from rejected input, so families build records only from validated data; the independent checker
re-validates the value anyway. A producer bug that run() can tell apart is an internal error: an exception that
produce() raises, other than an input, scope or budget error, and any failure to make its return into a Result,
other than a budget error. Validation before produce() and the independent verification keep the v5 codes.

A failure records the source context once the argument names are validated (ruling 8): the expression (or A) and
a Symbol variable as for_, with its domain; a failure before that records neither.

A family that supersedes v0.7 operations (SUPERSEDES) raises Declined from produce() for every input that it
leaves to the v0.7 implementation: run_dual() then answers with older(), the v0.7 record. With declinable=True,
run() also declines every input that fails its checks before produce(), so that v0.7's own errors stay.

The argument set of a released v6 operation never changes: run() and the decoder accept exactly the argument
names of the spec, and every stored request repeats them. Extend an operation by adding a new operation name.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from . import contracts as c
from . import operations_v5, registry
from .deferred import Unevaluated
from .errors import InvalidInput, MathForgeError, UnsupportedOperation
from .intervals import IntervalSet
from .limits import ComputationLimits, ResourceLimitError, computation
from .matrices import Matrix
from .model import Rational, Reals, Symbol
from .operations_v5 import _freeze
from .polynomial import Polynomial

REQUEST_VERSION = 6  # The format version that every request of run() carries.


class Declined(Exception):
    """Raised by a superseding family's produce() for an input that it leaves to the v0.7 implementation.

    It is no MathForgeError: run() reports it as the producer's internal error unless it runs declinable.
    """


def validate_arguments(operation, args):
    """The family of a registered v6 operation, once the argument names are exactly its spec's."""
    expected = registry.operation_arguments(6).get(operation)
    if expected is None:
        raise UnsupportedOperation("Unknown v6 operation")
    if type(args) is not dict or set(args) != expected:
        raise InvalidInput("Arguments do not match the v6 operation contract")
    return registry.family(operation, 6)


def source_context(args: Mapping[str, Any]) -> dict[str, Any]:
    """The source fields run() copies into a result: expression (or A), and a Symbol variable with Reals."""
    context: dict[str, Any] = {}
    if "expression" in args:
        context["expression"] = args["expression"]
    elif "A" in args:
        context["expression"] = args["A"]
    variable = args.get("variable")
    if type(variable) is Symbol:
        context.update(for_=variable, domain=Reals)
    return context


def _rational(value: object) -> bool:
    return type(value) is int or type(value) is Rational


_RELATIONS = ("=", "<", "<=", ">", ">=")  # The values of the relation kind; mcp.kinds enumerates them in this order.

# The argument kinds whose exact types run() checks before produce(), with the text of its error. Lists arrive
# frozen into tuples. The other kinds stay with the family.
_KINDS: dict[str, tuple[Callable[[object], bool], str]] = {
    "int": (lambda value: type(value) is int, "an exact integer"),
    "rational": (_rational, "an exact rational: an int or a Rational"),
    "int_list": (lambda value: type(value) is tuple and all(type(item) is int for item in value),
                 "a tuple or list of exact integers"),
    "rational_list": (lambda value: type(value) is tuple and all(_rational(item) for item in value),
                      "a tuple or list of exact rationals"),
    "matrix": (lambda value: type(value) is Matrix, "a Matrix"),
    "symbol": (lambda value: type(value) is Symbol, "a Symbol"),
    "polynomial": (lambda value: type(value) is Polynomial, "a Polynomial"),
    "interval_set": (lambda value: type(value) is IntervalSet, "an IntervalSet"),
    "relation": (lambda value: type(value) is str and value in _RELATIONS,
                 "a relation: '=', '<', '<=', '>' or '>='"),
}
# operation -> (the spec its checks were made for, (argument name, test, text) for each checked argument). A
# registry rebuilt with other families gives new spec objects, and so new checks.
_CHECKS: dict[str, tuple[Any, tuple[tuple[str, Callable[[object], bool], str], ...]]] = {}


def _check_kinds(operation: str, args: Mapping[str, object]) -> None:
    """Reject, as invalid input, an argument whose type its spec's kind does not admit."""
    spec = registry.spec(operation)
    planned = _CHECKS.get(operation)
    if planned is None or planned[0] is not spec:
        planned = _CHECKS[operation] = (spec, tuple((argument.name, *_KINDS[argument.kind])
                                                    for argument in spec.arguments if argument.kind in _KINDS))
    for name, fits, text in planned[1]:
        if not fits(args[name]):
            raise InvalidInput(f"Argument {name} of {operation} must be {text}")


@contextmanager
def _producer_failures(*passing: type[Exception]) -> Iterator[None]:
    """Report an exception other than the passing ones as the producer's internal error."""
    try:
        yield
    except passing:
        raise
    except Exception as exc:
        raise MathForgeError("The producer failed unexpectedly", code="internal_error") from exc


def run(operation: str, arguments: Mapping[str, object], *, limits: ComputationLimits | None = None,
        declinable: bool = False) -> c.Result[Any]:
    """The verified v6 Result of an operation, or its failure Result.

    Declined escapes only with declinable=True: from produce(), and for every input that fails the checks
    before it. Otherwise a Declined from produce() is the producer's internal error.
    """
    from .operations import _failure
    request = None
    context: dict[str, Any] = {}
    producing = False
    try:
        with computation(limits) as budget:
            if not isinstance(arguments, Mapping):
                raise InvalidInput("Operation arguments must be a mapping")
            args = {key: _freeze(value) for key, value in arguments.items()}
            budget.inspect(args)
            family = validate_arguments(operation, args)
            context = source_context(args)
            request = c.OperationRequest(operation, tuple(args.items()), REQUEST_VERSION)
            _check_kinds(operation, args)
            producing = True
            passing = (InvalidInput, UnsupportedOperation, ResourceLimitError, *((Declined,) if declinable else ()))
            with _producer_failures(*passing):
                produced = family.produce(operation, args)
            with _producer_failures(ResourceLimitError):  # Until the Result exists, a bad shape is the producer's.
                value, evidence = produced
                budget.inspect(value)
                budget.inspect(evidence, evidence=True)
                if type(value) is Unevaluated:
                    return c.Result(execution_status=c.ExecutionStatus.COMPLETED, outcome=c.Outcome.NO_CONCLUSION,
                                    completeness=c.Completeness.UNKNOWN, deferred=value, method=operation,
                                    request=request, **context)
                result: c.Result[Any] = c.Result(
                    execution_status=c.ExecutionStatus.COMPLETED, outcome=c.Outcome.VALUE, value=value,
                    exactness=c.Exactness.EXACT, completeness=c.Completeness.COMPLETE, method=operation,
                    request=request, **context,
                    steps=(c.Step("v6." + operation, (request,), (value, evidence),
                                  relation=c.StepRelation.EVALUATION),))
            from .verification_v6 import verify_result
            report = verify_result(result, limits=limits)
            if not report.verified:
                if any("resources" in check.claim for check in report.checks):
                    raise ResourceLimitError("Independent verification exceeded its budget")
                raise ArithmeticError("Independent verification rejected the produced result")
            return replace(result, verification=report.checks)
    except Exception as exc:
        if declinable and isinstance(exc, Declined):
            raise
        if declinable and not producing:  # The input fails a check before produce(): v0.7 reports it itself.
            raise Declined(f"{operation} leaves this input to its v0.7 implementation") from exc
        return _failure(exc, method=operation, context=context, request=request)


def run_dual(operation: str, arguments: Mapping[str, object], *, limits: ComputationLimits | None = None,
             older: Callable[[], c.Result[Any]] | None = None) -> c.Result[Any]:
    """The v6 Result of an operation that a v6 family supersedes, or older() when the family declines the input.

    older defaults to operations_v5.run(operation, arguments, limits=limits) for a v0.7 v5 operation; the public
    function of a superseded legacy operation, which mathforge.operations owns, passes its own.
    """
    if older is None:
        if operation not in registry.operation_arguments(5):
            raise TypeError(f"run_dual needs older for {operation!r}, which is not a v0.7 v5 operation")

        def older() -> c.Result[Any]:
            return operations_v5.run(operation, arguments, limits=limits)
    try:
        return run(operation, arguments, limits=limits, declinable=True)
    except Declined:
        return older()


def run_v07(operation: str, arguments: Mapping[str, object], *,
            limits: ComputationLimits | None = None) -> c.Result[Any]:
    """Run a v0.7 v5 operation as the registry holds it now: through run_dual while a v6 family supersedes it,
    else through operations_v5.run, as in v0.7.

    The generated wrappers of the v0.7 operations call it as run, so that a family taking one of them over, also
    one that a test registers, needs no regenerated wrapper; tools/generate_api.py writes run_dual for the
    operations that a production family supersedes.
    """
    if registry.is_v6_operation(operation):
        return run_dual(operation, arguments, limits=limits)
    return operations_v5.run(operation, arguments, limits=limits)
