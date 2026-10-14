"""Independent source, domain, mathematical identity and certificate checks."""
from . import contracts as c
from .dispatch_v5 import check_operation, validate_arguments
from .errors import InvalidInput, MathForgeError
from .limits import ResourceLimitError, computation
from .model import Reals
from .verification_v2 import _symbol_consistency


def _check(claim, passed, details=""):
    return c.Check(claim, c.CheckStatus.VERIFIED if passed else c.CheckStatus.FAILED,
                   c.VerificationLevel.ALGORITHMIC_CHECK, "independent_v5_check", details=details)


def _unknown(claim, details):
    return c.Check(claim, c.CheckStatus.UNKNOWN, c.VerificationLevel.NONE,
                   "verification_scope", details=details)


def _deferred_matches(operation, args, deferred):
    from .functions import (UnevaluatedIntegral, UnevaluatedLimit, real_function,
                            simplify_expr, as_real_expr)
    from .function_calculus import _source
    from .model import as_expression
    if operation not in ("function_antiderivative", "function_integrate",
                         "rational_integrate", "function_limit"):
        return False
    variable = args["variable"]
    source = (real_function(args["expression"]) if operation == "rational_integrate"
              else _source(args["expression"], variable))
    if deferred.variable != variable or deferred.expression != source: return False
    if operation == "function_limit":
        point = args["point"] if type(args["point"]) is str else simplify_expr(as_real_expr(args["point"]))
        return type(deferred) is UnevaluatedLimit and deferred.point == point and deferred.side == args["side"]
    if type(deferred) is not UnevaluatedIntegral: return False
    if operation == "function_antiderivative": return deferred.lower is None and deferred.upper is None
    # Deferred bounds retain the original exact input, so source linking needs
    # no algebraic computation or unsupported-normalization shortcut.
    return deferred.lower == as_expression(args["lower"]) and deferred.upper == as_expression(args["upper"])


def verify_result(result, *, limits=None):
    if type(result) is not c.Result:
        raise InvalidInput("verify expects a Result")
    try:
        with computation(limits, fresh=True) as budget:
            budget.inspect(result, evidence=True)
            request = result.request
            if type(request) is not c.OperationRequest:
                return c.VerificationReport((_check("result.request", False),))
            args = dict(request.arguments)
            validate_arguments(request.operation, args)
            variable = args.get("variable")
            source_ok = (len(args) == len(request.arguments)
                and result.method == request.operation and result.problem is None
                and result.expression == args.get("expression", args.get("A"))
                and result.for_ == variable and result.domain is (Reals if variable is not None else None)
                and not result.assumptions and _symbol_consistency(result))
            checks = [_check("result.source", source_ok)]
            if result.execution_status is not c.ExecutionStatus.COMPLETED:
                checks.append(_unknown("result.execution", "No completed mathematical claim"))
                return c.VerificationReport(tuple(checks))
            if result.deferred is not None:
                from .functions import UnevaluatedIntegral, UnevaluatedLimit
                correct = (type(result.deferred) in (UnevaluatedIntegral, UnevaluatedLimit)
                    and result.outcome is c.Outcome.NO_CONCLUSION and result.value is None
                    and result.completeness is c.Completeness.UNKNOWN
                    and result.exactness is c.Exactness.NOT_APPLICABLE and not result.steps
                    and result.error is None and result.precision is None
                    and result.tolerance is None and result.error_bound is None
                    and _deferred_matches(request.operation, args, result.deferred))
                checks.append(_check("result.unevaluated", correct))
                checks.append(_unknown("result.mathematics", "The request remains unevaluated"))
                return c.VerificationReport(tuple(checks))
            from .operations_v5 import partial
            expected = c.Completeness.PARTIAL if partial(request.operation, args, result.value) else c.Completeness.COMPLETE
            checks.append(_check("result.contract", result.outcome is c.Outcome.VALUE
                and result.exactness is c.Exactness.EXACT and result.completeness is expected
                and result.error is None and result.precision is None
                and result.tolerance is None and result.error_bound is None))
            connected = (len(result.steps) == 1 and result.steps[0].rule == "v5." + request.operation
                and result.steps[0].inputs == (request,) and len(result.steps[0].outputs) == 2
                and result.steps[0].outputs[0] == result.value
                and not result.steps[0].preconditions
                and result.steps[0].relation is c.StepRelation.EVALUATION
                and type(result.steps[0].outputs[1]) is tuple)
            checks.append(_check("result.trace", connected))
            if connected:
                checks.append(_check("result.mathematics", check_operation(
                    request.operation, args, result.value, result.steps[0].outputs[1])))
            return c.VerificationReport(tuple(checks))
    except (ResourceLimitError, RecursionError) as exc:
        return c.VerificationReport((_unknown("result.resources", str(exc)),))
    except (MathForgeError, ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return c.VerificationReport((_check("result.validity", False, str(exc)),))
