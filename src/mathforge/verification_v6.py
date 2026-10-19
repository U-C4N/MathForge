"""Independent checks of v6 results: source, contract, trace, and the family checker's own verdict.

An operation that this build does not register as v6 lies outside its verification scope: the report
is a single UNKNOWN check, never a FAILED one, so documents of later builds degrade gracefully. The
mathematics check is FAILED only when the family checker answers False; a checker that raises leaves it UNKNOWN.
"""
from . import contracts as c
from . import registry
from .deferred import KINDS, REASON_CODES, Reason, Unevaluated
from .errors import InvalidInput, MathForgeError
from .limits import ResourceLimitError, computation
from .operations import _safe_context
from .operations_v6 import source_context, validate_arguments
from .verification_v5 import _symbol_consistency


def _check(claim, passed, details=""):
    return c._trusted_check(claim, c.CheckStatus.VERIFIED if passed else c.CheckStatus.FAILED,
                            c.VerificationLevel.ALGORITHMIC_CHECK, "independent_v6_check", details)


def _unknown(claim, details):
    return c._trusted_check(claim, c.CheckStatus.UNKNOWN, c.VerificationLevel.NONE, "verification_scope", details)


def _source(result, request, args):
    """The result restates its request as run() records it; a failure keeps what operations._failure keeps."""
    context = source_context(args)
    if result.execution_status is not c.ExecutionStatus.COMPLETED:
        context = _safe_context(context)  # The variable's own domain and assumptions; a valid expression only.
    return (len(args) == len(request.arguments) and result.method == request.operation and result.problem is None
            and result.expression == context.get("expression") and result.for_ == context.get("for_")
            and result.domain is context.get("domain") and result.assumptions == context.get("assumptions", ())
            and _symbol_consistency(result))


def _no_error_or_bounds(result):
    """No error, precision, tolerance or error bound: a v6 result is exact or unevaluated."""
    return (result.error is None and result.precision is None and result.tolerance is None
            and result.error_bound is None)


def _unevaluated(result, request):
    deferred = result.deferred
    return (type(deferred) is Unevaluated and deferred.kind in KINDS and type(deferred.reason) is Reason
            and deferred.reason.code in REASON_CODES and deferred.request == request
            and result.outcome is c.Outcome.NO_CONCLUSION and result.value is None
            and result.completeness is c.Completeness.UNKNOWN and result.exactness is c.Exactness.NOT_APPLICABLE
            and not result.steps and _no_error_or_bounds(result))


def _mathematics(request, args, result):
    """The family checker's verdict: VERIFIED for True and FAILED for False only.

    A checker that raises, or answers anything but a bool, has a bug that says nothing about the result, so its
    check is UNKNOWN. A checker out of budget still ends the whole report as exhausted resources.
    """
    try:
        verdict = registry.checker(request.operation).check_operation(
            request.operation, args, result.value, result.steps[0].outputs[1])
    except (ResourceLimitError, RecursionError):
        raise
    except Exception as exc:
        return _unknown("result.mathematics", "The checker raised " + type(exc).__name__)
    if type(verdict) is not bool:
        return _unknown("result.mathematics", "The checker returned no verdict: " + type(verdict).__name__)
    return _check("result.mathematics", verdict)


def _trace(result, request):
    """Exactly one v6.<operation> step, from the request to the value and its evidence tuple."""
    if len(result.steps) != 1:
        return False
    step = result.steps[0]
    return (step.rule == "v6." + request.operation and step.inputs == (request,) and len(step.outputs) == 2
            and step.outputs[0] == result.value and type(step.outputs[1]) is tuple and not step.preconditions
            and step.relation is c.StepRelation.EVALUATION)


def verify_result(result, *, limits=None):
    if type(result) is not c.Result:
        raise InvalidInput("verify expects a Result")
    try:
        with computation(limits, fresh=True) as budget:
            budget.inspect(result, evidence=True)
            request = result.request
            if type(request) is not c.OperationRequest:
                return c._trusted_report((_check("result.request", False),))
            if request.operation not in registry.operation_arguments(6):
                return c._trusted_report((_unknown("result.operation", "unknown v6 operation"),))
            args = dict(request.arguments)
            validate_arguments(request.operation, args)
            checks = [_check("result.source", _source(result, request, args))]
            if result.execution_status is not c.ExecutionStatus.COMPLETED:
                checks.append(_unknown("result.execution", "No completed mathematical claim"))
            elif result.deferred is not None:
                checks.append(_check("result.unevaluated", _unevaluated(result, request)))
                checks.append(_unknown("result.mathematics", "The request remains unevaluated"))
            else:
                checks.append(_check("result.contract", result.outcome is c.Outcome.VALUE
                                     and result.exactness is c.Exactness.EXACT
                                     and result.completeness is c.Completeness.COMPLETE
                                     and _no_error_or_bounds(result)))
                connected = _trace(result, request)
                checks.append(_check("result.trace", connected))
                if connected:
                    checks.append(_mathematics(request, args, result))
            return c._trusted_report(tuple(checks))
    except (ResourceLimitError, RecursionError) as exc:
        return c._trusted_report((_unknown("result.resources", str(exc)),))
    except (MathForgeError, ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return c._trusted_report((_check("result.validity", False, str(exc)),))
