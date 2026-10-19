"""Source-bearing v0.7 operations and explicitly unevaluated requests."""
from collections.abc import Mapping

from . import contracts as c
from .dispatch_v5 import module_for, validate_arguments
from .limits import ResourceLimitError, computation
from .model import Reals, Symbol

_SCALARS = frozenset((int, str, bool, type(None)))  # Exact classes that _freeze returns as they are.


def _freeze(value):
    if type(value) in _SCALARS:
        return value
    if isinstance(value, Mapping):
        return tuple((_freeze(k), _freeze(v)) for k, v in value.items())
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def partial(operation, args, value):
    test = getattr(module_for(operation), "is_partial", None)
    return bool(test(operation, args, value)) if test is not None else False


# The deferral classes and the verifier module, imported on first use: they import modules that import this one.
_late: dict = {}


def _deferrals_and_verifier():
    if not _late:
        from . import verification_v5
        from .functions import UnevaluatedIntegral, UnevaluatedLimit
        _late.update(deferrals=(UnevaluatedIntegral, UnevaluatedLimit), verifier=verification_v5)
    return _late["deferrals"], _late["verifier"]


def run(operation, arguments, *, limits=None):
    """Run a v0.7 operation into a verified Result.

    The arguments, the value and the evidence are each inspected once, in this budget, and the records are built
    from them by contracts' trusted builders, which repeat no inspection and no check; the verifier inspects the whole
    Result once more in its own fresh budget. The argument names are validated once, and the family module that the
    validation returns produces the value and decides its completeness.
    """
    request = None
    context = {}
    try:
        with computation(limits) as budget:
            args = {key: _freeze(value) for key, value in arguments.items()}
            budget.inspect(args)
            family = validate_arguments(operation, args)
            request = c._trusted_request(operation, c._request_arguments(args))
            if "expression" in args:
                context["expression"] = args["expression"]
            elif "A" in args:
                context["expression"] = args["A"]
            variable = args.get("variable")
            if type(variable) is Symbol:
                context.update(for_=variable, domain=Reals)
            value, evidence = family.produce(operation, args)
            budget.inspect(value)
            budget.inspect(evidence, evidence=True)
            deferrals, verifier = _deferrals_and_verifier()
            if type(value) in deferrals:
                return c.Result(execution_status=c.ExecutionStatus.COMPLETED,
                    outcome=c.Outcome.NO_CONCLUSION, completeness=c.Completeness.UNKNOWN,
                    deferred=value, method=operation, request=request, **context)
            test = getattr(family, "is_partial", None)
            partial_value = bool(test(operation, args, value)) if test is not None else False
            # What the Step and the Result check and normalize, in their order, once.
            value, evidence = c._immutable_value(value), c._immutable_value(evidence)
            c._check_source(None, context.get("expression"), context.get("domain"), context.get("for_"))
            step = c._trusted_step("v5." + operation, (request,), (value, evidence), c.StepRelation.EVALUATION)
            result = c._trusted_result({
                "execution_status": c.ExecutionStatus.COMPLETED, "outcome": c.Outcome.VALUE, "value": value,
                "exactness": c.Exactness.EXACT,
                "completeness": c.Completeness.PARTIAL if partial_value else c.Completeness.COMPLETE,
                "method": operation, "request": request, **context, "steps": (step,)})
            report = verifier.verify_result(result, limits=limits)
            if not report.verified:
                if any("resources" in check.claim for check in report.checks):
                    raise ResourceLimitError("Independent verification exceeded its budget")
                raise ArithmeticError("Independent verification rejected the produced result")
            c._attach(result, "verification", report.checks)
            return result
    except Exception as exc:
        from .operations import _failure
        return _failure(exc, method=operation, context=context, request=request)
