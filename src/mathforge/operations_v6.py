"""Source-bearing v6 operations: run a registered family's producer, then verify its value independently.

A v6 family module defines SPECS, TYPES, CHECKER (the import path of its independent checker, below
mathforge.certify), produce() and, optionally, RENDER: {record class of its TYPES: render(value, renderer,
child) -> str}, the signature of rendering_v5.render; the registry checks all of them at build time.
mathforge.areas.ntheory.continued_fraction, the pilot family, spells the contract out.

The generated wrappers call run() with every argument of the spec, defaults filled in. A producer
returns (value, evidence), where value may be an Unevaluated record when it reaches no answer. It validates
its inputs first and raises InvalidInput for those it rejects. Once the inputs are validated, an InvalidInput
raised while building a record is a producer bug that run() cannot tell from rejected input, so families
build records only from validated data; the independent checker re-validates the value anyway.
A producer bug that run() can tell apart is an internal error: an exception that produce() raises, other
than an input, scope or budget error, and any failure to make its return into a Result, other than a budget
error. Validation before produce() and the independent verification keep the v5 codes.

The argument set of a released v6 operation never changes: run() and the decoder accept exactly the argument
names of the spec, and every stored request repeats them. Extend an operation by adding a new operation name.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from . import contracts as c
from . import registry
from .deferred import Unevaluated
from .errors import InvalidInput, MathForgeError, UnsupportedOperation
from .limits import ComputationLimits, ResourceLimitError, computation
from .model import Reals, Symbol
from .operations_v5 import _freeze


def validate_arguments(operation, args):
    """The family of a registered v6 operation, once the argument names are exactly its spec's."""
    expected = registry.operation_arguments(6).get(operation)
    if expected is None:
        raise UnsupportedOperation("Unknown v6 operation")
    if type(args) is not dict or set(args) != expected:
        raise InvalidInput("Arguments do not match the v6 operation contract")
    return registry.family(operation)


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


@contextmanager
def _producer_failures(*passing: type[Exception]) -> Iterator[None]:
    """Report an exception other than the passing ones as the producer's internal error."""
    try:
        yield
    except passing:
        raise
    except Exception as exc:
        raise MathForgeError("The producer failed unexpectedly", code="internal_error") from exc


def run(operation: str, arguments: Mapping[str, object], *,
        limits: ComputationLimits | None = None) -> c.Result[Any]:
    from .operations import _failure
    request = None
    context: dict[str, Any] = {}
    try:
        with computation(limits) as budget:
            if not isinstance(arguments, Mapping):
                raise InvalidInput("Operation arguments must be a mapping")
            args = {key: _freeze(value) for key, value in arguments.items()}
            budget.inspect(args)
            family = validate_arguments(operation, args)
            request = c.OperationRequest(operation, tuple(args.items()))
            context = source_context(args)
            with _producer_failures(InvalidInput, UnsupportedOperation, ResourceLimitError):
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
        return _failure(exc, method=operation, context=context, request=request)
