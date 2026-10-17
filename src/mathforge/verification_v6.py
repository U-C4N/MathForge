"""Independent checks of v6 results: source, contract, trace, and the family checker's own verdict.

An operation that this build does not register as v6 lies outside its verification scope: the report
is a single UNKNOWN check, never a FAILED one, so documents of later builds degrade gracefully.
"""
from . import contracts as c
from . import registry
from .deferred import KINDS, REASON_CODES, Reason, Unevaluated
from .errors import InvalidInput, MathForgeError
from .limits import ResourceLimitError, computation
from .model import Reals
from .operations_v6 import validate_arguments
from .verification_v2 import _symbol_consistency


def _check(claim, passed, details=""):
    return c.Check(claim, c.CheckStatus.VERIFIED if passed else c.CheckStatus.FAILED,
                   c.VerificationLevel.ALGORITHMIC_CHECK, "independent_v6_check", details=details)


def _unknown(claim, details):
    return c.Check(claim, c.CheckStatus.UNKNOWN, c.VerificationLevel.NONE, "verification_scope", details=details)


def _source(result, request, args):
    """The result restates its request: method, expression, variable and real domain, consistent symbols."""
    variable = args.get("variable")
    return (len(args) == len(request.arguments) and result.method == request.operation and result.problem is None
            and result.expression == args.get("expression", args.get("A"))
            and result.for_ == variable and result.domain is (Reals if variable is not None else None)
            and not result.assumptions and _symbol_consistency(result))


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
                return c.VerificationReport((_check("result.request", False),))
            if request.operation not in registry.operation_arguments(6):
                return c.VerificationReport((_unknown("result.operation", "unknown v6 operation"),))
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
                    verdict = registry.checker(request.operation).check_operation(
                        request.operation, args, result.value, result.steps[0].outputs[1])
                    checks.append(_check("result.mathematics", verdict is True))
            return c.VerificationReport(tuple(checks))
    except (ResourceLimitError, RecursionError) as exc:
        return c.VerificationReport((_unknown("result.resources", str(exc)),))
    except (MathForgeError, ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return c.VerificationReport((_check("result.validity", False, str(exc)),))
