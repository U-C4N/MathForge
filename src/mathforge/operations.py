"""Structured SDK operations with immutable source requests and bounded work."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Literal, TypeAlias, overload
from .analysis_types import FunctionAnalysis, LimitValue, ExtremumValue
from . import algorithms, matrices
from .contracts import (Completeness, EmptySet, FiniteSet, UniversalSet, Eq, Inequality, ErrorInfo, Exactness,
    ExecutionStatus, OperationRequest, Outcome, Result, Step, StepRelation)
from .errors import InvalidInput, MathForgeError, UnsupportedOperation
from .limits import ComputationLimits, computation, ResourceLimitError
from .model import (Expression, Rational, Reals, Symbol, as_expression, evaluate_expression,
    simplify_expression, substitute_expression)
from .polynomial import Polynomial, SquareFreeDecomposition
from .rational_function import RationalFunction, as_polynomial, validate_variable
from .algebraic import RootRecord, RealAlgebraicRoot
from .intervals import IntervalSet

from .polynomial_exact import PolynomialFactorization
from .exact_extensions import PartialFractionDecomposition

RealAlgebraic: TypeAlias = Rational | RealAlgebraicRoot

ScalarInput: TypeAlias = Expression | RationalFunction | int
PolynomialInput: TypeAlias = Expression | Polynomial | int
Bindings: TypeAlias = Mapping[Symbol, Expression | int]
EquationSolutionSet: TypeAlias = EmptySet | FiniteSet | UniversalSet | IntervalSet
LinearSolutionSet: TypeAlias = EmptySet | matrices.AffineSolutionSet
ExpressionVector: TypeAlias = tuple[Expression, ...]
ExpressionMatrix: TypeAlias = tuple[ExpressionVector, ...]


def _scalar_input(value):
    return as_expression(value) if type(value) is int else value


def _arguments(operation, arguments):
    args = dict(arguments)
    for key in ("expression", "numerator", "denominator", "lower", "upper", "left", "right", "base"):
        if key in args:
            args[key] = _scalar_input(args[key])
    if "point" in args and type(args["point"]) is int:
        args["point"] = Rational(args["point"])
    if operation in ("limit", "analyze_function"):
        # A rejected call must remain serializable without attaching a request
        # that violates the strict v3 wire contract.
        validate_variable(args["variable"])
        if not isinstance(args["expression"], (Expression, Polynomial, RationalFunction)):
            raise InvalidInput("Analysis requires an expression, polynomial or rational function")
        if operation == "limit":
            from .rational_limits import _point_and_side
            if type(args["point"]) in (Rational, str):
                args["point"] = _point_and_side(args["point"], args["side"])
            elif not isinstance(args["point"], Expression) or args["side"] not in ("both", "left", "right"):
                raise InvalidInput("A finite algebraic point and a valid side are required")
    if "bindings" in args:
        bindings = {} if args["bindings"] is None and operation in ("evaluate", "evaluate_algebraic") else args["bindings"]
        if not isinstance(bindings, Mapping):
            raise InvalidInput("Bindings must map Symbols to exact values")
        args["bindings"] = tuple((key, _scalar_input(value)) for key, value in bindings.items())
    if operation == "solve_system":
        if type(args["system_or_equations"]) is not matrices.LinearSystem:
            args["system_or_equations"] = tuple(args["system_or_equations"])
        if args["for_"] is not None:
            args["for_"] = tuple(args["for_"])
    if operation in ("expand_multivariate", "gradient", "jacobian", "hessian"):
        from .limits import current_budget
        for name in (("variables", "expressions") if operation == "jacobian" else ("variables",)):
            source = args[name]
            if isinstance(source, (str, bytes, Mapping)):
                raise InvalidInput(f"{name} must be an ordered iterable")
            values = []
            for value in source:
                current_budget().nodes()
                values.append(_scalar_input(value) if name == "expressions" else value)
            args[name] = tuple(values)
    from .exact_dispatch import NEW_OPERATIONS, validate_arguments
    if operation in NEW_OPERATIONS:
        validate_arguments(operation, args)
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


def _safe_context(context):
    """The context a failure result keeps: a typed problem, a Symbol with its own domain, a valid expression."""
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
    return safe


def _failure(exc, *, method, context, request=None):
    safe = _safe_context(context)
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


def _run(operation, arguments, producer, *, limits=None, solutions=False, _rule_version=None):
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
                    steps=(Step((f"v{_rule_version}." if _rule_version else "v3." if operation in ("limit", "analyze_function") else "v2.") + operation, (request,), (value, evidence),
                                relation=StepRelation.EQUIVALENCE if solutions else StepRelation.EVALUATION),))
            from .verification import verify
            report = verify(result)
            if not report.verified:
                if any("resources" in check.claim for check in report.checks):
                    raise ResourceLimitError("Independent verification exceeded the computation budget")
                raise ArithmeticError("Independent verification rejected the produced result")
            if len(result.steps) == 1 and result.steps[0].rule.startswith(("v2.", "v3.", "v4.")):
                result = replace(result, steps=(replace(result.steps[0], audit=(report.checks[-1],)),))
            return replace(result, verification=report.checks)
    except Exception as exc:
        return _failure(exc, method=operation, context=context, request=request)


def _polynomial(value):
    if type(value) is not Polynomial:
        raise InvalidInput("This operation requires Polynomial operands")
    return value


def expand_multivariate(expression: Expression | int, variables: Iterable[Symbol], *,
                        limits: ComputationLimits | None = None) -> Result[Expression]:
    """Expand a rational polynomial in an explicit, complete variable order."""
    from . import multivariate
    return _run("expand_multivariate", dict(expression=expression, variables=variables),
                lambda a: (multivariate.expand_multivariate(a["expression"], a["variables"]), ()),
                limits=limits)


def partial_diff(expression: Expression | int, variable: Symbol, n: int = 1, *,
                 limits: ComputationLimits | None = None) -> Result[Expression]:
    if _function_input(expression):
        return function_partial_diff(expression, variable, n, limits=limits)
    """Differentiate a multivariate rational polynomial, holding other symbols fixed."""
    from . import multivariate
    return _run("partial_diff", dict(expression=expression, variable=variable, n=n),
                lambda a: (multivariate.partial_diff(a["expression"], a["variable"], a["n"]), ()),
                limits=limits)


def gradient(expression: Expression | int, variables: Iterable[Symbol], *,
             limits: ComputationLimits | None = None) -> Result[ExpressionVector]:
    if _function_input(expression):
        return function_gradient(expression, tuple(variables), limits=limits)
    """Return partial derivatives in the supplied complete variable order."""
    from . import multivariate
    return _run("gradient", dict(expression=expression, variables=variables),
                lambda a: (multivariate.gradient(a["expression"], a["variables"]), ()), limits=limits)


def jacobian(expressions: Iterable[Expression | int], variables: Iterable[Symbol], *,
             limits: ComputationLimits | None = None) -> Result[ExpressionMatrix]:
    if type(expressions) in (tuple, list) and any(_function_input(e) for e in expressions):
        return function_jacobian(tuple(expressions), tuple(variables), limits=limits)
    """Return rows of functions and columns of variables as immutable expressions."""
    from . import multivariate
    return _run("jacobian", dict(expressions=expressions, variables=variables),
                lambda a: (multivariate.jacobian(a["expressions"], a["variables"]), ()), limits=limits)


def hessian(expression: Expression | int, variables: Iterable[Symbol], *,
            limits: ComputationLimits | None = None) -> Result[ExpressionMatrix]:
    if _function_input(expression):
        return function_hessian(expression, tuple(variables), limits=limits)
    """Return the symmetric matrix of second partial derivatives of a polynomial."""
    from . import multivariate
    return _run("hessian", dict(expression=expression, variables=variables),
                lambda a: (multivariate.hessian(a["expression"], a["variables"]), ()), limits=limits)


def limit(expression: PolynomialInput | RationalFunction, variable: Symbol,
          point: Expression | int | Literal["-inf", "+inf"], *,
          side: Literal["left", "right", "both"] = "both",
          limits: ComputationLimits | None = None) -> Result[LimitValue]:
    """Return a checked rational limit; nonexistence is a successful value."""
    if _function_input(expression):
        return function_limit(expression, variable, point, side, limits=limits)
    if isinstance(point, Expression) and type(point) is not Rational:
        return _run_exact("limit", dict(expression=expression, variable=variable, point=point, side=side), limits)
    from .rational_limits import limit_value
    return _run("limit", dict(expression=expression, variable=variable, point=point, side=side),
                lambda a: limit_value(a["expression"], a["variable"], a["point"], a["side"]), limits=limits)


def analyze_function(expression: PolynomialInput | RationalFunction, variable: Symbol, *,
                     limits: ComputationLimits | None = None) -> Result[FunctionAnalysis]:
    """Analyze the complete real domain, signs and local variation of a rational function."""
    from .function_analysis import analyze_function as produce_analysis
    return _run("analyze_function", dict(expression=expression, variable=variable),
                lambda a: produce_analysis(a["expression"], a["variable"]), limits=limits)


def expand(expression: PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[Expression]:
    def produce(a):
        validate_variable(a["variable"])
        return as_polynomial(a["expression"], a["variable"]).to_expression(), ()
    return _run("expand", dict(expression=expression, variable=variable), produce, limits=limits)


def poly_divmod(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Result[tuple[Polynomial, Polynomial]]:
    return _run("poly_divmod", dict(f=f, g=g), lambda a: (_polynomial(a["f"]).divmod(_polynomial(a["g"])), ()), limits=limits)


def poly_gcd(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Result[Polynomial]:
    def produce(a):
        gcd, s, t = _polynomial(a["f"]).xgcd(_polynomial(a["g"]))
        return gcd, (s, t)
    return _run("poly_gcd", dict(f=f, g=g), produce, limits=limits)


def poly_xgcd(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Result[tuple[Polynomial, Polynomial, Polynomial]]:
    return _run("poly_xgcd", dict(f=f, g=g), lambda a: (_polynomial(a["f"]).xgcd(_polynomial(a["g"])), ()), limits=limits)


def square_free(f: Polynomial, *, limits: ComputationLimits | None = None) -> Result[SquareFreeDecomposition]:
    return _run("square_free", dict(f=f), lambda a: (_polynomial(a["f"]).square_free(), ()), limits=limits)


def real_roots(expression: PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[tuple[RootRecord, ...]]:
    from .root_isolation import real_roots as roots
    from .root_certificates import root_certificate
    def produce(a):
        validate_variable(a["variable"])
        polynomial = as_polynomial(a["expression"], a["variable"])
        records = roots(polynomial)
        return records, root_certificate(polynomial, records)
    return _run("real_roots", dict(expression=expression, variable=variable), produce, limits=limits)


def compare_real(left: Expression | int, right: Expression | int, *, limits: ComputationLimits | None = None) -> Result[int]:
    from .algebraic import compare_real as compare
    result = _run("compare_real", dict(left=left, right=right), lambda a: (compare(a["left"], a["right"]), ()), limits=limits)
    if result.execution_status is ExecutionStatus.UNSUPPORTED:
        return _run_exact("compare_real", dict(left=left, right=right), limits)
    return result


def rational_function(numerator: PolynomialInput, denominator: PolynomialInput, *, variable: Symbol, limits: ComputationLimits | None = None) -> Result[RationalFunction]:
    from .rational_function import rational_function as construct
    return _run("rational_function", dict(numerator=numerator, denominator=denominator, variable=variable),
                lambda a: (construct(a["numerator"], a["denominator"], variable=a["variable"]), ()), limits=limits)


def solve(problem: Eq | Inequality, *, for_: Symbol, limits: ComputationLimits | None = None) -> Result[EquationSolutionSet]:
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


def solve_system(system_or_equations: matrices.LinearSystem | Iterable[Eq], *, for_: Iterable[Symbol] | None = None, limits: ComputationLimits | None = None) -> Result[LinearSolutionSet]:
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


def rref(A: matrices.Matrix, *, limits: ComputationLimits | None = None) -> Result[matrices.RREFResult]:
    return _matrix_operation("rref", A, limits)


def rank(A: matrices.Matrix, *, limits: ComputationLimits | None = None) -> Result[int]:
    return _matrix_operation("rank", A, limits)


def det(A: matrices.Matrix, *, limits: ComputationLimits | None = None) -> Result[Rational]:
    return _matrix_operation("det", A, limits)


def inverse(A: matrices.Matrix, *, limits: ComputationLimits | None = None) -> Result[matrices.Matrix]:
    return _matrix_operation("inverse", A, limits)


def nullspace(A: matrices.Matrix, *, limits: ComputationLimits | None = None) -> Result[tuple[tuple[Rational, ...], ...]]:
    return _matrix_operation("nullspace", A, limits)


def differentiate(expression: PolynomialInput | RationalFunction, variable: Symbol, n: int = 1, *, limits: ComputationLimits | None = None) -> Result[Expression | RationalFunction]:
    if _function_input(expression):
        return function_differentiate(expression, variable, n, limits=limits)
    from .calculus import derivative
    return _run("differentiate", dict(expression=expression, variable=variable, n=n),
                lambda a: (derivative(a["expression"], a["variable"], a["n"]), ()), limits=limits)


def antiderivative(expression: PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[Expression]:
    if _function_input(expression):
        return function_antiderivative(expression, variable, limits=limits)
    if type(expression) is RationalFunction:
        return rational_antiderivative(expression, variable, limits=limits)
    from .calculus import antiderivative_value
    return _run("antiderivative", dict(expression=expression, variable=variable),
                lambda a: (antiderivative_value(a["expression"], a["variable"]), ()), limits=limits)


@overload
def integrate(expression: PolynomialInput, variable: Symbol, lower: Rational | int, upper: Rational | int, *, limits: ComputationLimits | None = None) -> Result[Rational]: ...


@overload
def integrate(expression: PolynomialInput, variable: Symbol, lower: Expression | int, upper: Expression | int, *, limits: ComputationLimits | None = None) -> Result[RealAlgebraic]: ...


def integrate(expression: PolynomialInput, variable: Symbol, lower: Expression | int, upper: Expression | int, *, limits: ComputationLimits | None = None) -> Result[RealAlgebraic]:
    if _function_input(expression):
        return function_integrate(expression, variable, lower, upper, limits=limits)
    if type(expression) is RationalFunction:
        return rational_integrate(expression, variable, lower, upper, limits=limits)
    if any(isinstance(v, Expression) and type(v) is not Rational for v in (lower, upper)):
        return _run_exact("integrate", dict(expression=expression, variable=variable, lower=lower, upper=upper), limits)
    from .calculus import definite_integral
    return _run("integrate", dict(expression=expression, variable=variable, lower=lower, upper=upper),
                lambda a: definite_integral(a["expression"], a["variable"], a["lower"], a["upper"]), limits=limits)


def evaluate(expression: ScalarInput, bindings: Bindings | None = None, *, limits: ComputationLimits | None = None) -> Result[Expression]:
    if _function_input(expression):
        return function_evaluate(expression, {} if bindings is None else bindings, limits=limits)
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


def substitute(expression: ScalarInput, bindings: Bindings, *, limits: ComputationLimits | None = None) -> Result[Expression | RationalFunction]:
    def produce(a):
        expression, bindings = a["expression"], dict(a["bindings"])
        return (expression.substitute(bindings) if type(expression) is RationalFunction else
                substitute_expression(as_expression(expression), bindings)), ()
    return _run("substitute", dict(expression=expression, bindings=bindings), produce, limits=limits)


def simplify(expression: ScalarInput, *, limits: ComputationLimits | None = None) -> Result[Expression | RationalFunction]:
    return _run("simplify", dict(expression=expression), lambda a: (
        a["expression"] if type(a["expression"]) is RationalFunction else
        simplify_expression(as_expression(a["expression"])), ()), limits=limits)


def _run_exact(operation, arguments, limits):
    from .exact_dispatch import produce
    return _run(operation, arguments, lambda args: produce(operation, args), limits=limits, _rule_version=4)


def _function_input(value):
    from .functions import RealExpr, RealFunction
    return type(value) in (RealExpr, RealFunction)


def factor(f: Polynomial, *, limits: ComputationLimits | None = None) -> Result[PolynomialFactorization]:
    """Factor a nonzero rational polynomial completely over the rationals."""
    return _run_exact("factor", dict(f=f), limits)


def resultant(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Result[Rational]:
    return _run_exact("resultant", dict(f=f, g=g), limits)


def discriminant(f: Polynomial, *, limits: ComputationLimits | None = None) -> Result[Rational]:
    return _run_exact("discriminant", dict(f=f), limits)


def algebraic(expression: Expression | int, *, limits: ComputationLimits | None = None) -> Result[RealAlgebraic]:
    """Normalize a closed real algebraic expression, preserving the legacy AST."""
    return _run_exact("algebraic", dict(expression=expression), limits)


def algebraic_div(left: Expression | int, right: Expression | int, *, limits: ComputationLimits | None = None) -> Result[RealAlgebraic]:
    return _run_exact("algebraic_div", dict(left=left, right=right), limits)


def algebraic_pow(base: Expression | int, n: int, *, limits: ComputationLimits | None = None) -> Result[RealAlgebraic]:
    return _run_exact("algebraic_pow", dict(base=base, n=n), limits)


def minimal_polynomial(expression: Expression | int, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[Polynomial]:
    return _run_exact("minimal_polynomial", dict(expression=expression, variable=variable), limits)


def evaluate_algebraic(expression: PolynomialInput | RationalFunction, bindings: Bindings | None = None, *, limits: ComputationLimits | None = None) -> Result[RealAlgebraic]:
    return _run_exact("evaluate_algebraic", dict(expression=expression, bindings=bindings), limits)


def apart(expression: RationalFunction, *, limits: ComputationLimits | None = None) -> Result[PartialFractionDecomposition]:
    return _run_exact("apart", dict(expression=expression), limits)


def extrema_values(expression: PolynomialInput | RationalFunction, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[tuple[ExtremumValue, ...]]:
    return _run_exact("extrema_values", dict(expression=expression, variable=variable), limits)


def charpoly(A: matrices.Matrix, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[Polynomial]:
    return _run_exact("charpoly", dict(A=A, variable=variable), limits)


def real_eigenvalues(A: matrices.Matrix, *, limits: ComputationLimits | None = None) -> Result[tuple[RootRecord, ...]]:
    return _run_exact("real_eigenvalues", dict(A=A), limits)


from .api_v7 import (
    integer_gcd,
    integer_lcm,
    integer_xgcd,
    mod_inverse,
    crt,
    is_prime,
    factor_integer,
    divisors,
    totient,
    factorial,
    fibonacci,
    catalan,
    binomial,
    permutations_count,
    stirling,
    multinomial,
    complex_algebraic,
    complex_add,
    complex_sub,
    complex_mul,
    complex_div,
    complex_conjugate,
    complex_real,
    complex_imag,
    complex_norm_squared,
    complex_sqrt,
    complex_pow,
    quadratic_roots,
    bernoulli,
    binomial_distribution,
    discrete_uniform,
    hypergeometric,
    probability,
    condition,
    cdf,
    moment,
    expectation,
    distribution_variance,
    mean,
    median,
    variance,
    stddev,
    covariance,
    trace,
    rowspace,
    columnspace,
    lu,
    least_squares,
    rational_antiderivative,
    rational_integrate,
    function_differentiate,
    function_partial_diff,
    function_gradient,
    function_jacobian,
    function_hessian,
    taylor,
    function_evaluate,
    function_antiderivative,
    function_integrate,
    function_limit,
)
