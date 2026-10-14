"""Source-linked checks for exact algebra, independent of SDK producer dispatch."""

from . import contracts as c
from .errors import InvalidInput, UnsupportedOperation
from .exact_dispatch import ARGUMENTS, check_operation, validate_arguments
from .limits import ResourceLimitError, computation
from .model import Reals, Symbol
from .verification_v2 import _symbol_consistency


def _check(claim, passed, details=""):
    return c.Check(claim, c.CheckStatus.VERIFIED if passed else c.CheckStatus.FAILED,
                   c.VerificationLevel.ALGORITHMIC_CHECK, "independent_v4_check", details=details)


def _unknown(claim, details):
    return c.Check(claim, c.CheckStatus.UNKNOWN, c.VerificationLevel.NONE,
                   "verification_scope", details=details)


def _arguments(request):
    if type(request) is not c.OperationRequest:
        raise InvalidInput("A v4 source request is required")
    args = dict(request.arguments)
    validate_arguments(request.operation, args)
    if len(args) != len(request.arguments):
        raise InvalidInput("Duplicate source argument")
    if "variable" in args:
        variable = args["variable"]
        if type(variable) is not Symbol or variable.domain is not Reals or variable.assumptions:
            raise InvalidInput("An unconstrained real Symbol is required")
    return args


def verify_step_v4(step):
    try:
        with computation() as budget:
            budget.inspect(step, evidence=True)
            if (type(step) is not c.Step or len(step.inputs) != 1 or len(step.outputs) != 2
                    or step.preconditions or step.relation is not c.StepRelation.EVALUATION
                    or not _symbol_consistency(step)):
                return _check("step.validity", False)
            request = step.inputs[0]
            args = _arguments(request)
            if step.rule != "v4." + request.operation:
                return _check("step.source", False)
            value, evidence = step.outputs
            if type(evidence) is not tuple:
                return _check("step.evidence", False)
            return _check("step." + step.rule,
                          check_operation(request.operation, args, value, evidence))
    except (ResourceLimitError, RecursionError) as exc:
        return _unknown("step.resources", str(exc))
    except (InvalidInput, UnsupportedOperation, AttributeError, IndexError, KeyError,
            TypeError, ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return _check("step.validity", False, str(exc))


def verify_result(result, *, limits=None):
    if type(result) is not c.Result:
        raise InvalidInput("verify expects a Result")
    try:
        with computation(limits) as budget:
            budget.inspect(result, evidence=True)
            args = _arguments(result.request)
            variable = args.get("variable")
            expected_expression = args.get("expression", args.get("A"))
            checks = [_check("result.source",
                result.expression == expected_expression and result.for_ == variable
                and result.problem is None and result.assumptions == ()
                and result.domain is (Reals if variable is not None else None)
                and result.method == result.request.operation and _symbol_consistency(result))]
            if result.execution_status is not c.ExecutionStatus.COMPLETED:
                checks.append(_unknown("result.execution", "An unsuccessful operation makes no mathematical claim"))
                return c.VerificationReport(tuple(checks))
            checks.append(_check("result.contract", result.outcome is c.Outcome.VALUE
                and result.exactness is c.Exactness.EXACT and result.completeness is c.Completeness.COMPLETE
                and result.error is None and result.precision is None
                and result.tolerance is None and result.error_bound is None))
            connected = (len(result.steps) == 1 and result.steps[0].rule == "v4." + result.request.operation
                and result.steps[0].inputs == (result.request,) and len(result.steps[0].outputs) == 2
                and result.steps[0].outputs[0] == result.value)
            checks.append(_check("result.trace", connected))
            if connected:
                checks.append(verify_step_v4(result.steps[0]))
            return c.VerificationReport(tuple(checks))
    except (ResourceLimitError, RecursionError) as exc:
        return c.VerificationReport((_unknown("result.resources", str(exc)),))
    except (InvalidInput, UnsupportedOperation, AttributeError, IndexError, KeyError,
            TypeError, ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return c.VerificationReport((_check("result.validity", False, str(exc)),))
