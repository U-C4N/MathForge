"""Structured SDK operations with immutable source requests and bounded work."""
from collections.abc import Mapping
from dataclasses import replace
from . import algorithms, matrices
from .contracts import (Completeness, EmptySet, Eq, Inequality, ErrorInfo, Exactness,
    ExecutionStatus, OperationRequest, Outcome, Result, Step, StepRelation)
from .errors import InvalidInput, MathForgeError, UnsupportedOperation
from .limits import computation, ResourceLimitError
from .model import (Reals, Symbol, as_expression, evaluate_expression,
    simplify_expression, substitute_expression)
from .polynomial import Polynomial
from .rational_function import RationalFunction, as_polynomial, validate_variable


def _scalar_input(value):
    return as_expression(value) if type(value) is int else value


def _arguments(operation, arguments):
    args = dict(arguments)
    for key in ("expression", "numerator", "denominator", "lower", "upper", "left", "right"):
        if key in args:
            args[key] = _scalar_input(args[key])
    if "bindings" in args:
        bindings = {} if args["bindings"] is None and operation == "evaluate" else args["bindings"]
        if not isinstance(bindings, Mapping):
            raise InvalidInput("Bindings must map Symbols to exact values")
        args["bindings"] = tuple((key, _scalar_input(value)) for key, value in bindings.items())
    if operation == "solve_system":
        if type(args["system_or_equations"]) is not matrices.LinearSystem:
            args["system_or_equations"] = tuple(args["system_or_equations"])
        if args["for_"] is not None:
            args["for_"] = tuple(args["for_"])
    return args


def _context(operation, args):
    context = {}
    if "expression" in args:
        context["expression"] = args["expression"]
    if "A" in args:
        context["expression"] = args["A"]
    if "variable" in args and operation != "rational_function":
        context["for_"] = args["variable"]
    if operation == "solve":
        context.update(problem=args["problem"], for_=args["for_"])
    if operation == "solve_system":
        source = args["system_or_equations"]
        if type(source) is matrices.LinearSystem:
            context["problem"] = source
        context["domain"] = Reals
    variable = context.get("for_")
    if type(variable) is Symbol:
        context.update(domain=variable.domain, assumptions=variable.assumptions)
    return context


def _failure(exc, *, method, context, request=None):
    safe = {}
    problem = context.get("problem")
    if type(problem) in (Eq, Inequality, matrices.LinearSystem):
        safe["problem"] = problem
        if type(problem) is matrices.LinearSystem:
            safe["domain"] = Reals
    variable = context.get("for_")
    if type(variable) is Symbol:
        safe.update(for_=variable, domain=variable.domain, assumptions=variable.assumptions)
    expression = context.get("expression")
    try:
        if expression is not None:
            safe["expression"] = expression if type(expression) in (Polynomial, RationalFunction, matrices.Matrix) else as_expression(expression)
    except (TypeError, ValueError):
        pass
    if isinstance(exc, UnsupportedOperation):
        status, code = ExecutionStatus.UNSUPPORTED, "unsupported_operation"
    elif isinstance(exc, (InvalidInput, TypeError, ValueError, ZeroDivisionError)):
        status, code = ExecutionStatus.INVALID_INPUT, "invalid_input"
    else:
        status, code = ExecutionStatus.ERROR, "execution_error"
    if isinstance(exc, MathForgeError):
        code = exc.code
    return Result(execution_status=status, outcome=Outcome.NOT_APPLICABLE,
                  error=ErrorInfo(code, str(exc)), method=method, request=request, **safe)


def _run(operation, arguments, producer, *, limits=None, solutions=False):
    request = None
    context = _context(operation, arguments)
    try:
        with computation(limits) as budget:
            budget.inspect(arguments)
            args = _arguments(operation, arguments)
            context = _context(operation, args)
            request = OperationRequest(operation, tuple(args.items()))
            produced = producer(args)
            if type(produced) is Result:
                result = replace(produced, request=request)
            else:
                value, evidence = produced
                if operation == "solve_system":
                    context["problem"] = _system(args)
                outcome = ((Outcome.NO_SOLUTION if type(value) is EmptySet else Outcome.SOLUTIONS)
                           if solutions else Outcome.VALUE)
                result = Result(execution_status=ExecutionStatus.COMPLETED, outcome=outcome,
                    value=value, exactness=Exactness.EXACT, completeness=Completeness.COMPLETE,
                    method=operation, request=request, **context,
                    steps=(Step("v2." + operation, (request,), (value, evidence),
                                relation=StepRelation.EQUIVALENCE if solutions else StepRelation.EVALUATION),))
            from .verification import verify
            report = verify(result)
            if not report.verified:
                if any("resources" in check.claim for check in report.checks):
                    raise ResourceLimitError("Independent verification exceeded the computation budget")
                raise ArithmeticError("Independent verification rejected the produced result")
            if len(result.steps) == 1 and result.steps[0].rule.startswith("v2."):
                result = replace(result, steps=(replace(result.steps[0], audit=(report.checks[-1],)),))
            return replace(result, verification=report.checks)
    except Exception as exc:
        return _failure(exc, method=operation, context=context, request=request)


def _polynomial(value):
    if type(value) is not Polynomial:
        raise InvalidInput("This operation requires Polynomial operands")
    return value


def expand(expression, variable, *, limits=None) -> Result:
    def produce(a):
        validate_variable(a["variable"])
        return as_polynomial(a["expression"], a["variable"]).to_expression(), ()
    return _run("expand", dict(expression=expression, variable=variable), produce, limits=limits)


def poly_divmod(f, g, *, limits=None) -> Result:
    return _run("poly_divmod", dict(f=f, g=g), lambda a: (_polynomial(a["f"]).divmod(_polynomial(a["g"])), ()), limits=limits)


def poly_gcd(f, g, *, limits=None) -> Result:
    def produce(a):
        gcd, s, t = _polynomial(a["f"]).xgcd(_polynomial(a["g"]))
        return gcd, (s, t)
    return _run("poly_gcd", dict(f=f, g=g), produce, limits=limits)


def poly_xgcd(f, g, *, limits=None) -> Result:
    return _run("poly_xgcd", dict(f=f, g=g), lambda a: (_polynomial(a["f"]).xgcd(_polynomial(a["g"])), ()), limits=limits)


def square_free(f, *, limits=None) -> Result:
    return _run("square_free", dict(f=f), lambda a: (_polynomial(a["f"]).square_free(), ()), limits=limits)


def real_roots(expression, variable, *, limits=None) -> Result:
    from .root_isolation import real_roots as roots
    from .root_certificates import root_certificate
    def produce(a):
        validate_variable(a["variable"])
        polynomial = as_polynomial(a["expression"], a["variable"])
        records = roots(polynomial)
        return records, root_certificate(polynomial, records)
    return _run("real_roots", dict(expression=expression, variable=variable), produce, limits=limits)


def compare_real(left, right, *, limits=None) -> Result:
    from .algebraic import compare_real as compare
    return _run("compare_real", dict(left=left, right=right), lambda a: (compare(a["left"], a["right"]), ()), limits=limits)


def rational_function(numerator, denominator, *, variable, limits=None) -> Result:
    from .rational_function import rational_function as construct
    return _run("rational_function", dict(numerator=numerator, denominator=denominator, variable=variable),
                lambda a: (construct(a["numerator"], a["denominator"], variable=a["variable"]), ()), limits=limits)


def solve(problem, *, for_, limits=None) -> Result:
    def produce(a):
        p, variable = a["problem"], a["for_"]
        if type(p) not in (Eq, Inequality):
            raise InvalidInput("solve expects an Eq or Inequality")
        validate_variable(variable)
        if type(p) is Eq and type(p.lhs) is not RationalFunction and type(p.rhs) is not RationalFunction:
            poly = Polynomial.from_expression(p.lhs - p.rhs, variable)
            if poly.degree <= 2:
                return algorithms.solve_equation(p, variable)
        from .real_sets import solve_real_set
        return solve_real_set(p, variable)
    return _run("solve", dict(problem=problem, for_=for_), produce, limits=limits, solutions=True)


def _system(args):
    source, variables = args["system_or_equations"], args["for_"]
    if type(source) is matrices.LinearSystem:
        if variables is not None and variables != source.variables:
            raise InvalidInput("The explicit variable order must match LinearSystem.variables")
        return source
    if variables is None:
        raise InvalidInput("Equation sequences require an explicit variable order")
    return matrices.linear_system_from_equations(source, variables)


def solve_system(system_or_equations, *, for_=None, limits=None) -> Result:
    return _run("solve_system", dict(system_or_equations=system_or_equations, for_=for_),
                lambda a: matrices.solve_linear_system(_system(a)), limits=limits, solutions=True)


def _matrix_operation(operation, A, limits):
    def produce(a):
        matrix = a["A"]
        if operation in ("rref", "rank", "nullspace"):
            reduced, trace = matrices.reduce_with_trace(matrix)
            if operation == "rref":
                return reduced, (trace,)
            if operation == "rank":
                return len(reduced.pivot_columns), (reduced, trace)
            return matrices._nullspace_from_reduced(reduced.matrix, reduced.pivot_columns, matrix.ncols), (reduced, trace)
        return getattr(matrices, operation)(matrix), ()
    return _run(operation, dict(A=A), produce, limits=limits)


def rref(A, *, limits=None) -> Result:
    return _matrix_operation("rref", A, limits)


def rank(A, *, limits=None) -> Result:
    return _matrix_operation("rank", A, limits)


def det(A, *, limits=None) -> Result:
    return _matrix_operation("det", A, limits)


def inverse(A, *, limits=None) -> Result:
    return _matrix_operation("inverse", A, limits)


def nullspace(A, *, limits=None) -> Result:
    return _matrix_operation("nullspace", A, limits)


def differentiate(expression, variable, n=1, *, limits=None) -> Result:
    from .calculus import derivative
    return _run("differentiate", dict(expression=expression, variable=variable, n=n),
                lambda a: (derivative(a["expression"], a["variable"], a["n"]), ()), limits=limits)


def antiderivative(expression, variable, *, limits=None) -> Result:
    from .calculus import antiderivative_value
    return _run("antiderivative", dict(expression=expression, variable=variable),
                lambda a: (antiderivative_value(a["expression"], a["variable"]), ()), limits=limits)


def integrate(expression, variable, lower, upper, *, limits=None) -> Result:
    from .calculus import definite_integral
    return _run("integrate", dict(expression=expression, variable=variable, lower=lower, upper=upper),
                lambda a: definite_integral(a["expression"], a["variable"], a["lower"], a["upper"]), limits=limits)


def evaluate(expression, bindings=None, *, limits=None) -> Result:
    def produce(a):
        expression, bindings = a["expression"], dict(a["bindings"])
        if type(expression) is RationalFunction:
            value = expression.substitute(bindings)
            if type(value) is RationalFunction:
                raise InvalidInput("Evaluation requires a binding for the rational function variable")
        else:
            value = evaluate_expression(as_expression(expression), bindings)
        return value, ()
    return _run("evaluate", dict(expression=expression, bindings=bindings), produce, limits=limits)


def substitute(expression, bindings, *, limits=None) -> Result:
    def produce(a):
        expression, bindings = a["expression"], dict(a["bindings"])
        return (expression.substitute(bindings) if type(expression) is RationalFunction else
                substitute_expression(as_expression(expression), bindings)), ()
    return _run("substitute", dict(expression=expression, bindings=bindings), produce, limits=limits)


def simplify(expression, *, limits=None) -> Result:
    return _run("simplify", dict(expression=expression), lambda a: (
        a["expression"] if type(a["expression"]) is RationalFunction else
        simplify_expression(as_expression(a["expression"])), ()), limits=limits)
