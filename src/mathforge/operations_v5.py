"""Source-bearing v0.7 operations and explicitly unevaluated requests."""
from collections.abc import Mapping
from dataclasses import replace

from . import contracts as c
from .dispatch_v5 import module_for, produce, validate_arguments
from .limits import ResourceLimitError, computation
from .model import Reals, Symbol


def _freeze(value):
    if isinstance(value, Mapping):
        return tuple((_freeze(k), _freeze(v)) for k, v in value.items())
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def partial(operation, args, value):
    test = getattr(module_for(operation), "is_partial", None)
    return bool(test(operation, args, value)) if test is not None else False


def run(operation, arguments, *, limits=None):
    from .operations import _failure
    request = None
    context = {}
    try:
        with computation(limits) as budget:
            args = {key: _freeze(value) for key, value in arguments.items()}
            budget.inspect(args)
            validate_arguments(operation, args)
            request = c.OperationRequest(operation, tuple(args.items()))
            if "expression" in args:
                context["expression"] = args["expression"]
            elif "A" in args:
                context["expression"] = args["A"]
            variable = args.get("variable")
            if type(variable) is Symbol:
                context.update(for_=variable, domain=Reals)
            value, evidence = produce(operation, args)
            budget.inspect(value)
            budget.inspect(evidence, evidence=True)
            from .functions import UnevaluatedIntegral, UnevaluatedLimit
            if type(value) in (UnevaluatedIntegral, UnevaluatedLimit):
                return c.Result(execution_status=c.ExecutionStatus.COMPLETED,
                    outcome=c.Outcome.NO_CONCLUSION, completeness=c.Completeness.UNKNOWN,
                    deferred=value, method=operation, request=request, **context)
            result = c.Result(execution_status=c.ExecutionStatus.COMPLETED,
                outcome=c.Outcome.VALUE, value=value, exactness=c.Exactness.EXACT,
                completeness=c.Completeness.PARTIAL if partial(operation, args, value) else c.Completeness.COMPLETE,
                method=operation, request=request, **context,
                steps=(c.Step("v5." + operation, (request,), (value, evidence),
                              relation=c.StepRelation.EVALUATION),))
            from .verification_v5 import verify_result
            report = verify_result(result, limits=limits)
            if not report.verified:
                if any("resources" in check.claim for check in report.checks):
                    raise ResourceLimitError("Independent verification exceeded its budget")
                raise ArithmeticError("Independent verification rejected the produced result")
            return replace(result, verification=report.checks)
    except Exception as exc:
        return _failure(exc, method=operation, context=context, request=request)
