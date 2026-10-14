"""Source-linked checking of v3 analysis claims without producer dispatch."""

from . import contracts as c
from .errors import InvalidInput, UnsupportedOperation
from .limits import ResourceLimitError, computation
from .model import Reals
from .verification_v2 import _symbol_consistency, _variable


OPERATIONS = frozenset(("limit", "analyze_function"))
_ARGUMENTS = {"limit": {"expression", "variable", "point", "side"},
              "analyze_function": {"expression", "variable"}}


def _check(claim, passed, details=""):
    return c.Check(claim, c.CheckStatus.VERIFIED if passed else c.CheckStatus.FAILED,
                   c.VerificationLevel.ALGORITHMIC_CHECK, "independent_v3_check", details=details)


def _unknown(claim, details):
    return c.Check(claim, c.CheckStatus.UNKNOWN, c.VerificationLevel.NONE,
                   "verification_scope", details=details)


def _arguments(request):
    if type(request) is not c.OperationRequest or request.operation not in OPERATIONS:
        raise UnsupportedOperation("Unknown v3 operation")
    args = dict(request.arguments)
    if len(args) != len(request.arguments) or set(args) != _ARGUMENTS[request.operation]:
        raise InvalidInput("Recorded operation arguments do not match the v3 contract")
    _variable(args["variable"])
    return args


def _step(step):
    if type(step) is not c.Step or len(step.inputs) != 1 or len(step.outputs) != 2:
        return _check("step.validity", False)
    request = step.inputs[0]
    args = _arguments(request)
    if (step.rule != "v3." + request.operation or step.relation is not c.StepRelation.EVALUATION
            or step.preconditions):
        return _check("step.source", False)
    value, evidence = step.outputs
    if request.operation == "limit":
        from .rational_limits import check_limit
        passed = check_limit(args["expression"], args["variable"], args["point"], args["side"], value, evidence)
    else:
        from .function_analysis import check_analysis
        passed = check_analysis(args["expression"], args["variable"], value, evidence)
    return _check("step." + step.rule, passed)


def verify_step_v3(step):
    try:
        with computation() as budget:
            budget.inspect(step, evidence=True)
            if not _symbol_consistency(step):
                return _check("step.symbols", False)
            return _step(step)
    except ResourceLimitError as exc:
        return _unknown("step.resources", str(exc))
    except (InvalidInput, UnsupportedOperation, AttributeError, IndexError, KeyError,
            TypeError, ValueError, ZeroDivisionError) as exc:
        return _check("step.validity", False, str(exc))


def verify_result(result, *, limits=None):
    if type(result) is not c.Result:
        raise InvalidInput("verify expects a Result")
    try:
        with computation(limits) as budget:
            budget.inspect(result, evidence=True)
            args = _arguments(result.request)
            source_ok = (result.expression == args["expression"] and result.for_ == args["variable"]
                         and result.problem is None and result.domain is Reals and result.assumptions == ()
                         and result.method == result.request.operation and _symbol_consistency(result))
            checks = [_check("result.source", source_ok)]
            if result.execution_status is not c.ExecutionStatus.COMPLETED:
                checks.append(_unknown("result.execution", "An unsuccessful operation makes no mathematical claim"))
                return c.VerificationReport(tuple(checks))
            checks.append(_check("result.contract", result.outcome is c.Outcome.VALUE
                                 and result.exactness is c.Exactness.EXACT
                                 and result.completeness is c.Completeness.COMPLETE
                                 and result.error is None and result.precision is None
                                 and result.tolerance is None and result.error_bound is None))
            if len(result.steps) != 1:
                checks.append(_check("result.trace", False, "Exactly one v3 evidence step is required"))
            else:
                step = result.steps[0]
                connected = (step.rule == "v3." + result.request.operation
                             and step.inputs == (result.request,) and len(step.outputs) == 2
                             and step.outputs[0] == result.value)
                checks.append(_check("step.source", connected))
                if connected:
                    checks.append(verify_step_v3(step))
            return c.VerificationReport(tuple(checks))
    except ResourceLimitError as exc:
        return c.VerificationReport((_unknown("result.resources", str(exc)),))
    except (InvalidInput, UnsupportedOperation, AttributeError, IndexError, KeyError,
            TypeError, ValueError, ZeroDivisionError) as exc:
        return c.VerificationReport((_check("result.validity", False, str(exc)),))
