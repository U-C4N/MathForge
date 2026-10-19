"""Independent source, domain, mathematical identity and certificate checks."""
from dataclasses import fields, is_dataclass

from . import contracts as c
from .dispatch_v5 import validate_arguments
from .errors import InvalidInput, MathForgeError
from .limits import ResourceLimitError, computation
from .model import Assumption, Rational, Reals, Symbol


def _check(claim, passed, details=""):
    return c._trusted_check(claim, c.CheckStatus.VERIFIED if passed else c.CheckStatus.FAILED,
                            c.VerificationLevel.ALGORITHMIC_CHECK, "independent_v5_check", details)


def _unknown(claim, details):
    return c._trusted_check(claim, c.CheckStatus.UNKNOWN, c.VerificationLevel.NONE, "verification_scope", details)


# Class -> the names of the fields that _symbol_consistency walks: a dataclass's fields, () for any other class.
# The records' constructors restrict the fields left out below to text, enum members, integers, Rationals, ErrorInfo
# and Assumptions, which hold no Symbol, so walking them could change no answer.
_SEED_WALKED: dict = {
    c.Result: ("value", "verification", "steps", "problem", "expression", "for_", "request", "deferred"),
    c.Step: ("inputs", "outputs", "preconditions", "audit"),
    c.OperationRequest: ("arguments",),
    c.Check: ("subject",),
    c.ErrorInfo: (), Assumption: (), Rational: (),
}
_WALKED = dict(_SEED_WALKED)
_WALKED_LIMIT = 4_096  # Classes made at run time cannot grow the table without bound.


def _walked(cls):
    names = _WALKED.get(cls)
    if names is None:
        names = tuple(field.name for field in fields(cls)) if is_dataclass(cls) else ()
        if len(_WALKED) >= _WALKED_LIMIT:
            _WALKED.clear()
            _WALKED.update(_SEED_WALKED)
        _WALKED[cls] = names
    return names


def _symbol_consistency(value):
    """verification_v2._symbol_consistency, with the fields to walk of each class looked up once (_WALKED): one
    Symbol uid has one name, domain and assumptions throughout the tuples and dataclass fields below the value."""
    pending, seen, symbols = [value], set(), {}
    while pending:
        item = pending.pop()
        cls = type(item)
        if cls is Symbol:
            metadata = (item.name, item.domain, item.assumptions)
            if item.uid in symbols and symbols[item.uid] != metadata:
                return False
            symbols[item.uid] = metadata
        elif cls is tuple:
            if id(item) not in seen:
                seen.add(id(item))
                pending.extend(item)
        else:
            names = _WALKED.get(cls)
            if names is None:
                names = _walked(cls)  # A class (an instance of type) is no dataclass instance: its type has no fields.
            if names and id(item) not in seen:
                seen.add(id(item))
                pending.extend([getattr(item, name) for name in names])
    return True


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
                return c._trusted_report((_check("result.request", False),))
            args = dict(request.arguments)
            family = validate_arguments(request.operation, args)  # Once: its family checks the claim below.
            variable = args.get("variable")
            source_ok = (len(args) == len(request.arguments)
                and result.method == request.operation and result.problem is None
                and result.expression == args.get("expression", args.get("A"))
                and result.for_ == variable and result.domain is (Reals if variable is not None else None)
                and not result.assumptions and _symbol_consistency(result))
            checks = [_check("result.source", source_ok)]
            if result.execution_status is not c.ExecutionStatus.COMPLETED:
                checks.append(_unknown("result.execution", "No completed mathematical claim"))
                return c._trusted_report(tuple(checks))
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
                return c._trusted_report(tuple(checks))
            test = getattr(family, "is_partial", None)
            expected = (c.Completeness.PARTIAL if test is not None and test(request.operation, args, result.value)
                        else c.Completeness.COMPLETE)
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
                checks.append(_check("result.mathematics", family.check_operation(
                    request.operation, args, result.value, result.steps[0].outputs[1])))
            return c._trusted_report(tuple(checks))
    except (ResourceLimitError, RecursionError) as exc:
        return c._trusted_report((_unknown("result.resources", str(exc)),))
    except (MathForgeError, ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return c._trusted_report((_check("result.validity", False, str(exc)),))
