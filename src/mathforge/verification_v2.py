"""Checks of v2 source records and mathematical certificates.

This module never dispatches a producer operation. It checks identities, replays
elementary evidence and uses bounded exact arithmetic shared with the model.
"""

from dataclasses import fields, is_dataclass

from . import contracts as c
from .errors import InvalidInput, UnsupportedOperation
from .limits import ResourceLimitError, computation
from .model import (
    Expression, Rational, Reals, Symbol, as_expression, evaluate_expression,
    simplify_expression, substitute_expression,
)
from .polynomial import Polynomial, SquareFreeDecomposition


_ARGUMENTS = {
    "expand": ("expression", "variable"),
    "poly_divmod": ("f", "g"), "poly_gcd": ("f", "g"), "poly_xgcd": ("f", "g"),
    "square_free": ("f",), "real_roots": ("expression", "variable"),
    "compare_real": ("left", "right"),
    "rational_function": ("numerator", "denominator", "variable"),
    "differentiate": ("expression", "variable", "n"),
    "antiderivative": ("expression", "variable"),
    "integrate": ("expression", "variable", "lower", "upper"),
    "evaluate": ("expression", "bindings"), "substitute": ("expression", "bindings"),
    "simplify": ("expression",), "rref": ("A",), "rank": ("A",),
    "det": ("A",), "inverse": ("A",), "nullspace": ("A",),
    "solve": ("problem", "for_"), "solve_system": ("system_or_equations", "for_"),
}
_VARIABLE_OPERATIONS = frozenset(("expand", "real_roots", "differentiate", "antiderivative", "integrate"))
_EXPRESSION_OPERATIONS = _VARIABLE_OPERATIONS | {"evaluate", "substitute", "simplify"}
_MATRIX_OPERATIONS = frozenset(("rref", "rank", "det", "inverse", "nullspace"))


def _check(claim, passed, *, details=""):
    return c.Check(claim, c.CheckStatus.VERIFIED if passed else c.CheckStatus.FAILED,
                   c.VerificationLevel.ALGORITHMIC_CHECK, "independent_v2_check", details=details)


def _unknown(claim, details):
    return c.Check(claim, c.CheckStatus.UNKNOWN, c.VerificationLevel.NONE,
                   "verification_scope", details=details)


def _arguments(request):
    if type(request) is not c.OperationRequest:
        raise InvalidInput("A v2 source request is required")
    expected = _ARGUMENTS.get(request.operation)
    if expected is None:
        raise UnsupportedOperation("Unknown recorded operation")
    arguments = dict(request.arguments)
    if len(arguments) != len(request.arguments) or set(arguments) != set(expected):
        raise InvalidInput("Recorded operation arguments do not match its contract")
    return arguments


def _variable(value):
    if type(value) is not Symbol or value.domain is not Reals or value.assumptions:
        raise InvalidInput("The source variable must be an unconstrained real Symbol")
    return value


def _polynomial(value, variable):
    _variable(variable)
    if type(value) is Polynomial:
        if value.variable != variable:
            raise InvalidInput("Source polynomial variable does not match the request")
        return value
    return Polynomial.from_expression(value, variable)


def _derivative(p):
    return Polynomial(tuple(p.coefficients[k] * k for k in range(1, len(p.coefficients))), p.variable)


def _pgcd(a, b):
    """Euclidean identity helper, independent of the producer gcd function."""
    while not b.is_zero:
        _, r = a.divmod(b)
        a, b = b, r
    return a if a.is_zero else a / a.coefficients[-1]


def _at(p, x):
    value = Rational(0)
    for coefficient in reversed(p.coefficients):
        value = value * x + coefficient
    return value


def _rational(value):
    value = as_expression(value)
    if type(value) is not Rational:
        raise InvalidInput("An exact rational source value is required")
    return value


def _reduced_fraction(numerator, denominator):
    if denominator.is_zero:
        raise InvalidInput("The source denominator is identically zero")
    divisor = _pgcd(numerator, denominator)
    numerator = numerator.exact_div(divisor)
    denominator = denominator.exact_div(divisor)
    leading = denominator.coefficients[-1]
    return numerator / leading, denominator / leading


def _rf_identity(value, numerator, denominator, exclusions):
    from .rational_function import RationalFunction, exclusions_cover

    if type(value) is not RationalFunction or value.numerator.variable != numerator.variable:
        return False
    n, d = _reduced_fraction(numerator, denominator)
    return (value.numerator == n and value.denominator == d
            and exclusions_cover(exclusions, value.excluded)
            and exclusions_cover(value.excluded, exclusions))


def _square_free_part(p):
    if p.is_zero:
        raise InvalidInput("A zero polynomial cannot describe a domain exclusion")
    q = p.exact_div(_pgcd(p, _derivative(p)))
    return q / q.coefficients[-1]


def _check_division(a, b, answer):
    if type(a) is not Polynomial or type(b) is not Polynomial or a.variable != b.variable or b.is_zero:
        return False
    if type(answer) is not tuple or len(answer) != 2 or any(type(v) is not Polynomial for v in answer):
        return False
    q, r = answer
    return (q.variable == a.variable and r.variable == a.variable
            and (r.is_zero or r.degree < b.degree) and a == q * b + r)


def _check_gcd(a, b, g, s, t):
    if any(type(v) is not Polynomial for v in (a, b, g, s, t)):
        return False
    if any(v.variable != a.variable for v in (b, g, s, t)):
        return False
    if a.is_zero and b.is_zero:
        return g.is_zero and s.is_zero and t.is_zero
    return (not g.is_zero and g.coefficients[-1] == Rational(1)
            and a.divmod(g)[1].is_zero and b.divmod(g)[1].is_zero
            and s * a + t * b == g)


def _check_square_free(p, answer):
    if type(p) is not Polynomial or type(answer) is not SquareFreeDecomposition:
        return False
    if p.degree <= 0:
        return answer.coefficient == p.coefficients[0] and not answer.factors
    rebuilt = Polynomial((answer.coefficient,), p.variable)
    seen = []
    previous = 0
    for factor in answer.factors:
        q, exponent = factor.polynomial, factor.multiplicity
        if (q.variable != p.variable or q.degree <= 0 or q.coefficients[-1] != Rational(1)
                or type(exponent) is not int or exponent <= previous
                or _pgcd(q, _derivative(q)).degree != 0):
            return False
        if any(_pgcd(q, old).degree != 0 for old in seen):
            return False
        rebuilt = rebuilt * q**exponent
        seen.append(q)
        previous = exponent
    return rebuilt == p


def _check_calculus(operation, args, value, evidence):
    from .rational_function import RationalFunction

    source, variable = args["expression"], _variable(args["variable"])
    if operation == "differentiate":
        n = args["n"]
        if type(n) is not int or n < 0 or evidence != ():
            return False
        if n == 0:
            return (value == source if type(source) is RationalFunction
                    else _polynomial(value, variable) == _polynomial(source, variable))
        if type(source) is RationalFunction:
            if source.numerator.variable != variable:
                return False
            numerator, denominator = source.numerator, source.denominator
            for _ in range(n):
                numerator, denominator = _reduced_fraction(
                    _derivative(numerator) * denominator - numerator * _derivative(denominator),
                    denominator * denominator,
                )
            return _rf_identity(value, numerator, denominator, source.excluded)
        p = _polynomial(source, variable)
        if n > p.degree:
            expected = Polynomial((0,), variable)
        else:
            coefficients = []
            for k in range(n, len(p.coefficients)):
                coefficient = p.coefficients[k]
                for multiplier in range(k - n + 1, k + 1):
                    coefficient = coefficient * multiplier
                coefficients.append(coefficient)
            expected = Polynomial(tuple(coefficients), variable)
        return _polynomial(value, variable) == expected
    if type(source) is RationalFunction:
        return False
    p = _polynomial(source, variable)
    if operation == "antiderivative":
        if evidence != ():
            return False
        answer = _polynomial(value, variable)
        return answer.coefficients[0] == Rational(0) and _derivative(answer) == p
    lower, upper = _rational(args["lower"]), _rational(args["upper"])
    if type(value) is not Rational or type(evidence) is not tuple or len(evidence) != len(p.coefficients):
        return False
    expected = tuple(coefficient * (upper**(k + 1) - lower**(k + 1)) / (k + 1)
                     for k, coefficient in enumerate(p.coefficients))
    return evidence == expected and value == sum(expected, Rational(0))


def _bindings(value):
    if type(value) is not tuple:
        raise InvalidInput("Recorded bindings must be an immutable sequence")
    result = {}
    for entry in value:
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not Symbol:
            raise InvalidInput("Recorded bindings must be Symbol/value pairs")
        if entry[0] in result:
            raise InvalidInput("Recorded bindings cannot repeat a symbol")
        result[entry[0]] = entry[1]
    return result


def _check_rewrite(operation, args, value, evidence):
    from .rational_function import RationalFunction

    if evidence != ():
        return False
    source = args["expression"]
    if type(source) is not RationalFunction:
        if operation == "simplify":
            return value == simplify_expression(source)
        bindings = _bindings(args["bindings"])
        expected = (evaluate_expression(source, bindings) if operation == "evaluate"
                    else substitute_expression(source, bindings))
        return value == expected
    if operation == "simplify":
        return _rf_identity(value, source.numerator, source.denominator, source.excluded)
    bindings = _bindings(args["bindings"])
    from .model import _check_replacement
    normalized = {}
    for bound_variable, raw in bindings.items():
        replacement = as_expression(raw)
        if type(replacement) not in (Rational, Symbol):
            raise InvalidInput("RF bindings require exact rationals or real symbol renaming")
        if type(replacement) is Symbol:
            _variable(replacement)
        _check_replacement(bound_variable, replacement)
        normalized[bound_variable] = replacement
    bindings = normalized
    variable = source.numerator.variable
    if variable not in bindings:
        return operation == "substitute" and value == source
    replacement = as_expression(bindings[variable])
    if type(replacement) is Rational:
        if any(_at(p, replacement).numerator == 0 for p in source.excluded):
            return False
        return value == _at(source.numerator, replacement) / _at(source.denominator, replacement)
    if operation == "substitute" and type(replacement) is Symbol:
        _variable(replacement)
        rename = lambda p: Polynomial(p.coefficients, replacement)
        return _rf_identity(value, rename(source.numerator), rename(source.denominator),
                            tuple(rename(p) for p in source.excluded))
    return False


def _system(args):
    from .matrices import LinearSystem, linear_system_from_equations

    source, variables = args["system_or_equations"], args["for_"]
    if type(source) is LinearSystem:
        if variables is not None and variables != source.variables:
            raise InvalidInput("Explicit variables must match the LinearSystem order")
        return source
    if type(source) is not tuple or type(variables) is not tuple:
        raise InvalidInput("Equation systems require immutable equations and ordered variables")
    return linear_system_from_equations(source, variables)


def _check_operation(operation, args, value, evidence):
    if type(evidence) is not tuple:
        return False
    if operation == "poly_divmod":
        return evidence == () and _check_division(args["f"], args["g"], value)
    if operation in ("poly_gcd", "poly_xgcd"):
        if operation == "poly_gcd":
            if len(evidence) != 2:
                return False
            g, (s, t) = value, evidence
        else:
            if evidence != () or type(value) is not tuple or len(value) != 3:
                return False
            g, s, t = value
        return _check_gcd(args["f"], args["g"], g, s, t)
    if operation == "square_free":
        return evidence == () and _check_square_free(args["f"], value)
    if operation == "expand":
        return (evidence == () and _polynomial(value, args["variable"])
                == _polynomial(args["expression"], args["variable"]))
    if operation == "real_roots":
        from .root_certificates import check_root_certificate
        return check_root_certificate(_polynomial(args["expression"], args["variable"]), value, evidence)
    if operation == "compare_real":
        from .algebraic import compare_real
        return (evidence == () and type(value) is int and value in (-1, 0, 1)
                and value == compare_real(args["left"], args["right"]))
    if operation == "rational_function":
        if evidence != ():
            return False
        numerator = _polynomial(args["numerator"], args["variable"])
        denominator = _polynomial(args["denominator"], args["variable"])
        guard = _square_free_part(denominator)
        return _rf_identity(value, numerator, denominator, () if guard.degree == 0 else (guard,))
    if operation in ("differentiate", "antiderivative", "integrate"):
        return _check_calculus(operation, args, value, evidence)
    if operation in ("evaluate", "substitute", "simplify"):
        return _check_rewrite(operation, args, value, evidence)
    if operation in _MATRIX_OPERATIONS:
        from .matrices import (Matrix, RREFResult, check_determinant, check_nullspace,
                               check_rank, verify_row_reduction)
        A = args["A"]
        if type(A) is not Matrix:
            return False
        if operation == "rref":
            return (len(evidence) == 1 and type(value) is RREFResult
                    and verify_row_reduction(A, value, evidence[0]))
        if operation in ("rank", "nullspace"):
            if len(evidence) != 2 or type(evidence[0]) is not RREFResult:
                return False
            if not verify_row_reduction(A, evidence[0], evidence[1]):
                return False
            return check_rank(A, value) if operation == "rank" else check_nullspace(A, value)
        if operation == "det":
            return evidence == () and check_determinant(A, value)
        return (evidence == () and type(value) is Matrix and A.shape == value.shape
                and A.shape[0] == A.shape[1]
                and A @ value == value @ A == Matrix.identity(A.shape[0]))
    if operation == "solve_system":
        from .matrices import verify_linear_solution
        return verify_linear_solution(_system(args), value, evidence)
    if operation == "solve":
        from .real_sets import check_solution
        return check_solution(args["problem"], _variable(args["for_"]), value, evidence)
    raise UnsupportedOperation("This operation has no independent checker")


def _aliases(result, operation, args):
    expression = args["expression"] if operation in _EXPRESSION_OPERATIONS else None
    if operation in _MATRIX_OPERATIONS:
        expression = args["A"]
    variable = _variable(args["variable"]) if operation in _VARIABLE_OPERATIONS else None
    problem = None
    domain = Reals if variable is not None else None
    if operation == "solve":
        problem, variable, domain = args["problem"], _variable(args["for_"]), Reals
    elif operation == "solve_system":
        problem, domain = _system(args), Reals
    return (result.expression == expression and result.problem == problem
            and result.for_ == variable and result.domain is domain and result.assumptions == ())


def _symbol_consistency(value):
    pending, seen, symbols = [value], set(), {}
    while pending:
        item = pending.pop()
        if type(item) is Symbol:
            metadata = (item.name, item.domain, item.assumptions)
            if item.uid in symbols and symbols[item.uid] != metadata:
                return False
            symbols[item.uid] = metadata
        elif type(item) is tuple:
            if id(item) not in seen:
                seen.add(id(item))
                pending.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            if id(item) not in seen:
                seen.add(id(item))
                pending.extend(getattr(item, field.name) for field in fields(item))
    return True


def _verify_step_v2(step):
    """Validate one isolated v2 rule; source-result linkage is checked separately."""
    try:
        if type(step) is not c.Step or len(step.inputs) != 1 or len(step.outputs) != 2:
            return _check("step.validity", False)
        request = step.inputs[0]
        if type(request) is c.OperationRequest and request.operation not in _ARGUMENTS:
            return _unknown("step.scope", "Unknown recorded operation")
        args = _arguments(request)
        if step.rule != "v2." + request.operation:
            return _check("step.source", False)
        relation = (c.StepRelation.EQUIVALENCE if request.operation in ("solve", "solve_system")
                    else c.StepRelation.EVALUATION)
        passed = (step.relation is relation and not step.preconditions
                  and _check_operation(request.operation, args, step.outputs[0], step.outputs[1]))
        return _check("step." + step.rule, passed)
    except ResourceLimitError as exc:
        return _unknown("step.resources", str(exc))
    except UnsupportedOperation as exc:
        return _check("step.validity", False, details=str(exc))
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        return _check("step.validity", False, details=str(exc))


def verify_step_v2(step):
    """Bound standalone step checking and reuse an enclosing verifier budget."""
    try:
        with computation() as budget:
            budget.inspect(step, evidence=True)
            return _verify_step_v2(step)
    except ResourceLimitError as exc:
        return _unknown("step.resources", str(exc))


def verify_result(result, *, limits=None):
    if type(result) is not c.Result:
        raise InvalidInput("verify expects a Result")
    from .verification import _verify_legacy

    try:
        with computation(limits) as budget:
            budget.inspect(result, evidence=True)
            if result.request is None:
                return _verify_legacy(result)
            if result.request.operation not in _ARGUMENTS:
                return c.VerificationReport((_unknown("result.operation", "Unknown recorded operation"),))
            args = _arguments(result.request)
            operation = result.request.operation
            checks = [_check("result.source", _aliases(result, operation, args)
                             and _symbol_consistency(result))]
            if result.execution_status is not c.ExecutionStatus.COMPLETED:
                checks.append(_unknown("result.execution", "An unsuccessful operation makes no mathematical claim"))
                return c.VerificationReport(tuple(checks))
            solving = operation in ("solve", "solve_system")
            expected = (c.Outcome.NO_SOLUTION if type(result.value) is c.EmptySet else c.Outcome.SOLUTIONS) if solving else c.Outcome.VALUE
            checks.append(_check("result.contract", result.outcome is expected
                                 and result.exactness is c.Exactness.EXACT
                                 and result.completeness is c.Completeness.COMPLETE
                                 and result.error is None and result.precision is None
                                 and result.tolerance is None and result.error_bound is None))
            # v0.1 scalar solve rules remain checkable after adding a source request.
            if (operation == "solve" and type(result.problem) is c.Eq
                    and all(not step.rule.startswith("v2.") for step in result.steps)):
                legacy_scope = False
                if isinstance(result.problem.lhs, Expression) and isinstance(result.problem.rhs, Expression):
                    try:
                        polynomial = Polynomial.from_expression(result.problem.lhs - result.problem.rhs,
                                                                result.for_)
                        legacy_scope = polynomial.degree <= 2
                    except UnsupportedOperation:
                        pass
                if legacy_scope:
                    checks.extend(_verify_legacy(result).checks)
                    return c.VerificationReport(tuple(checks))
            if len(result.steps) != 1:
                checks.append(_check("result.trace", False, details="Exactly one v2 evidence step is required"))
                return c.VerificationReport(tuple(checks))
            step = result.steps[0]
            if step.rule != "v2." + operation:
                checks.append(_unknown("step.rule", "Unexpected or unknown operation rule"))
                return c.VerificationReport(tuple(checks))
            connected = (step.inputs == (result.request,) and len(step.outputs) == 2
                         and step.outputs[0] == result.value)
            checks.append(_check("step.source", connected))
            if connected:
                checks.append(verify_step_v2(step))
            return c.VerificationReport(tuple(checks))
    except ResourceLimitError as exc:
        return c.VerificationReport((_unknown("result.resources", str(exc)),))
    except UnsupportedOperation as exc:
        return c.VerificationReport((_check("result.validity", False, details=str(exc)),))
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        return c.VerificationReport((_check("result.validity", False, details=str(exc)),))
