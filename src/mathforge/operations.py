"""Structured SDK operations with immutable source requests and bounded work."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from functools import partial as _partial
from typing import Literal, NamedTuple, TypeAlias, overload
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


class LegacyOperation(NamedTuple):
    """The producer of a legacy operation and the mathematical check that verify applies to its record.

    ``produce(args)`` takes the arguments as ``_run`` normalizes them and returns ``(value, evidence)``; solve returns
    a whole Result for the equations that the v0.1 rules solve. ``check(args, value, evidence)`` is the check that
    ``verification.verify`` applies to the operation's step, called as the record calls it: with the request's
    arguments and the step's two outputs. The lean value path (mathforge.lean) runs both directly.
    """
    produce: Callable[[dict], tuple[object, tuple]]
    check: Callable[[dict, object, tuple], bool]


def _produce_expand_multivariate(a):
    from . import multivariate
    return multivariate.expand_multivariate(a["expression"], a["variables"]), ()


def _produce_partial_diff(a):
    from . import multivariate
    return multivariate.partial_diff(a["expression"], a["variable"], a["n"]), ()


def _produce_gradient(a):
    from . import multivariate
    return multivariate.gradient(a["expression"], a["variables"]), ()


def _produce_jacobian(a):
    from . import multivariate
    return multivariate.jacobian(a["expressions"], a["variables"]), ()


def _produce_hessian(a):
    from . import multivariate
    return multivariate.hessian(a["expression"], a["variables"]), ()


def _produce_limit(a):
    from .rational_limits import limit_value
    return limit_value(a["expression"], a["variable"], a["point"], a["side"])


def _produce_analyze_function(a):
    from .function_analysis import analyze_function as produce_analysis
    return produce_analysis(a["expression"], a["variable"])


def _produce_expand(a):
    validate_variable(a["variable"])
    return as_polynomial(a["expression"], a["variable"]).to_expression(), ()


def _produce_poly_divmod(a):
    return _polynomial(a["f"]).divmod(_polynomial(a["g"])), ()


def _produce_poly_gcd(a):
    gcd, s, t = _polynomial(a["f"]).xgcd(_polynomial(a["g"]))
    return gcd, (s, t)


def _produce_poly_xgcd(a):
    return _polynomial(a["f"]).xgcd(_polynomial(a["g"])), ()


def _produce_square_free(a):
    return _polynomial(a["f"]).square_free(), ()


def _produce_real_roots(a):
    from .root_isolation import real_roots as roots
    from .root_certificates import root_certificate
    validate_variable(a["variable"])
    polynomial = as_polynomial(a["expression"], a["variable"])
    records = roots(polynomial)
    return records, root_certificate(polynomial, records)


def _produce_compare_real(a):
    from .algebraic import compare_real as compare
    return compare(a["left"], a["right"]), ()


def _produce_rational_function(a):
    from .rational_function import rational_function as construct
    return construct(a["numerator"], a["denominator"], variable=a["variable"]), ()


def _produce_solve(a):
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


def _system(args):
    source, variables = args["system_or_equations"], args["for_"]
    if type(source) is matrices.LinearSystem:
        if variables is not None and variables != source.variables:
            raise InvalidInput("The explicit variable order must match LinearSystem.variables")
        return source
    if variables is None:
        raise InvalidInput("Equation sequences require an explicit variable order")
    return matrices.linear_system_from_equations(source, variables)


def _produce_solve_system(a):
    return matrices.solve_linear_system(_system(a))


def _produce_rref(a):
    reduced, trace = matrices.reduce_with_trace(a["A"])
    return reduced, (trace,)


def _produce_rank(a):
    reduced, trace = matrices.reduce_with_trace(a["A"])
    return len(reduced.pivot_columns), (reduced, trace)


def _produce_nullspace(a):
    matrix = a["A"]
    reduced, trace = matrices.reduce_with_trace(matrix)
    return matrices._nullspace_from_reduced(reduced.matrix, reduced.pivot_columns, matrix.ncols), (reduced, trace)


def _produce_det(a):
    return matrices.det(a["A"]), ()


def _produce_inverse(a):
    return matrices.inverse(a["A"]), ()


def _produce_differentiate(a):
    from .calculus import derivative
    return derivative(a["expression"], a["variable"], a["n"]), ()


def _produce_antiderivative(a):
    from .calculus import antiderivative_value
    return antiderivative_value(a["expression"], a["variable"]), ()


def _produce_integrate(a):
    from .calculus import definite_integral
    return definite_integral(a["expression"], a["variable"], a["lower"], a["upper"])


def _produce_evaluate(a):
    expression, bindings = a["expression"], dict(a["bindings"])
    if type(expression) is RationalFunction:
        value = expression.substitute(bindings)
        if type(value) is RationalFunction:
            raise InvalidInput("Evaluation requires a binding for the rational function variable")
    else:
        value = evaluate_expression(as_expression(expression), bindings)
    return value, ()


def _produce_substitute(a):
    expression, bindings = a["expression"], dict(a["bindings"])
    return (expression.substitute(bindings) if type(expression) is RationalFunction else
            substitute_expression(as_expression(expression), bindings)), ()


def _produce_simplify(a):
    return (a["expression"] if type(a["expression"]) is RationalFunction else
            simplify_expression(as_expression(a["expression"]))), ()


def _produce_exact(operation, args):
    from .exact_dispatch import produce
    return produce(operation, args)


def _check_v2(operation, args, value, evidence):
    from .verification_v2 import _check_operation
    return _check_operation(operation, args, value, evidence)


def _check_limit(args, value, evidence):
    from .rational_limits import check_limit
    return check_limit(args["expression"], args["variable"], args["point"], args["side"], value, evidence)


def _check_analysis(args, value, evidence):
    from .function_analysis import check_analysis
    return check_analysis(args["expression"], args["variable"], value, evidence)


def _check_exact(operation, args, value, evidence):
    from .exact_dispatch import check_operation
    return check_operation(operation, args, value, evidence)


def _v2(operation, produce):
    return LegacyOperation(produce, _partial(_check_v2, operation))


def _exact(operation):
    return LegacyOperation(_partial(_produce_exact, operation), _partial(_check_exact, operation))


# The v4 operations of exact_dispatch.ARGUMENTS, which _run_exact produces. compare_real, limit and integrate are
# also legacy v2/v3 operations: their public functions take the v4 path only for inputs the older one leaves out.
LEGACY_EXACT: dict[str, LegacyOperation] = {name: _exact(name) for name in (
    "factor", "resultant", "discriminant", "algebraic", "algebraic_div", "algebraic_pow", "minimal_polynomial",
    "evaluate_algebraic", "apart", "extrema_values", "charpoly", "real_eigenvalues", "compare_real", "limit",
    "integrate")}

# Every legacy operation's producer and check, on the path its public function takes through _run: verify and
# parse_function have none. The public functions below run their table entries.
LEGACY: dict[str, LegacyOperation] = {
    "expand_multivariate": _v2("expand_multivariate", _produce_expand_multivariate),
    "partial_diff": _v2("partial_diff", _produce_partial_diff),
    "gradient": _v2("gradient", _produce_gradient),
    "jacobian": _v2("jacobian", _produce_jacobian),
    "hessian": _v2("hessian", _produce_hessian),
    "limit": LegacyOperation(_produce_limit, _check_limit),
    "analyze_function": LegacyOperation(_produce_analyze_function, _check_analysis),
    "expand": _v2("expand", _produce_expand),
    "poly_divmod": _v2("poly_divmod", _produce_poly_divmod),
    "poly_gcd": _v2("poly_gcd", _produce_poly_gcd),
    "poly_xgcd": _v2("poly_xgcd", _produce_poly_xgcd),
    "square_free": _v2("square_free", _produce_square_free),
    "real_roots": _v2("real_roots", _produce_real_roots),
    "compare_real": _v2("compare_real", _produce_compare_real),
    "rational_function": _v2("rational_function", _produce_rational_function),
    "solve": _v2("solve", _produce_solve),
    "solve_system": _v2("solve_system", _produce_solve_system),
    "rref": _v2("rref", _produce_rref),
    "rank": _v2("rank", _produce_rank),
    "det": _v2("det", _produce_det),
    "inverse": _v2("inverse", _produce_inverse),
    "nullspace": _v2("nullspace", _produce_nullspace),
    "differentiate": _v2("differentiate", _produce_differentiate),
    "antiderivative": _v2("antiderivative", _produce_antiderivative),
    "integrate": _v2("integrate", _produce_integrate),
    "evaluate": _v2("evaluate", _produce_evaluate),
    "substitute": _v2("substitute", _produce_substitute),
    "simplify": _v2("simplify", _produce_simplify),
    **{name: LEGACY_EXACT[name] for name in (
        "factor", "resultant", "discriminant", "algebraic", "algebraic_div", "algebraic_pow", "minimal_polynomial",
        "evaluate_algebraic", "apart", "extrema_values", "charpoly", "real_eigenvalues")},
}


def expand_multivariate(expression: Expression | int, variables: Iterable[Symbol], *,
                        limits: ComputationLimits | None = None) -> Result[Expression]:
    """Expand a rational polynomial in an explicit, complete variable order."""
    return _run("expand_multivariate", dict(expression=expression, variables=variables),
                LEGACY["expand_multivariate"].produce, limits=limits)


def partial_diff(expression: Expression | int, variable: Symbol, n: int = 1, *,
                 limits: ComputationLimits | None = None) -> Result[Expression]:
    if _function_input(expression):
        return function_partial_diff(expression, variable, n, limits=limits)
    """Differentiate a multivariate rational polynomial, holding other symbols fixed."""
    return _run("partial_diff", dict(expression=expression, variable=variable, n=n),
                LEGACY["partial_diff"].produce, limits=limits)


def gradient(expression: Expression | int, variables: Iterable[Symbol], *,
             limits: ComputationLimits | None = None) -> Result[ExpressionVector]:
    if _function_input(expression):
        return function_gradient(expression, tuple(variables), limits=limits)
    """Return partial derivatives in the supplied complete variable order."""
    return _run("gradient", dict(expression=expression, variables=variables), LEGACY["gradient"].produce,
                limits=limits)


def jacobian(expressions: Iterable[Expression | int], variables: Iterable[Symbol], *,
             limits: ComputationLimits | None = None) -> Result[ExpressionMatrix]:
    if type(expressions) in (tuple, list) and any(_function_input(e) for e in expressions):
        return function_jacobian(tuple(expressions), tuple(variables), limits=limits)
    """Return rows of functions and columns of variables as immutable expressions."""
    return _run("jacobian", dict(expressions=expressions, variables=variables), LEGACY["jacobian"].produce,
                limits=limits)


def hessian(expression: Expression | int, variables: Iterable[Symbol], *,
            limits: ComputationLimits | None = None) -> Result[ExpressionMatrix]:
    if _function_input(expression):
        return function_hessian(expression, tuple(variables), limits=limits)
    """Return the symmetric matrix of second partial derivatives of a polynomial."""
    return _run("hessian", dict(expression=expression, variables=variables), LEGACY["hessian"].produce,
                limits=limits)


def limit(expression: PolynomialInput | RationalFunction, variable: Symbol,
          point: Expression | int | Literal["-inf", "+inf"], *,
          side: Literal["left", "right", "both"] = "both",
          limits: ComputationLimits | None = None) -> Result[LimitValue]:
    """Return a checked rational limit; nonexistence is a successful value."""
    if _function_input(expression):
        return function_limit(expression, variable, point, side, limits=limits)
    if isinstance(point, Expression) and type(point) is not Rational:
        return _run_exact("limit", dict(expression=expression, variable=variable, point=point, side=side), limits)
    return _run("limit", dict(expression=expression, variable=variable, point=point, side=side),
                LEGACY["limit"].produce, limits=limits)


def analyze_function(expression: PolynomialInput | RationalFunction, variable: Symbol, *,
                     limits: ComputationLimits | None = None) -> Result[FunctionAnalysis]:
    """Analyze the complete real domain, signs and local variation of a rational function."""
    return _run("analyze_function", dict(expression=expression, variable=variable),
                LEGACY["analyze_function"].produce, limits=limits)


def expand(expression: PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[Expression]:
    return _run("expand", dict(expression=expression, variable=variable), LEGACY["expand"].produce, limits=limits)


def poly_divmod(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Result[tuple[Polynomial, Polynomial]]:
    return _run("poly_divmod", dict(f=f, g=g), LEGACY["poly_divmod"].produce, limits=limits)


def poly_gcd(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Result[Polynomial]:
    return _run("poly_gcd", dict(f=f, g=g), LEGACY["poly_gcd"].produce, limits=limits)


def poly_xgcd(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Result[tuple[Polynomial, Polynomial, Polynomial]]:
    return _run("poly_xgcd", dict(f=f, g=g), LEGACY["poly_xgcd"].produce, limits=limits)


def square_free(f: Polynomial, *, limits: ComputationLimits | None = None) -> Result[SquareFreeDecomposition]:
    return _run("square_free", dict(f=f), LEGACY["square_free"].produce, limits=limits)


def real_roots(expression: PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[tuple[RootRecord, ...]]:
    return _run("real_roots", dict(expression=expression, variable=variable), LEGACY["real_roots"].produce,
                limits=limits)


def compare_real(left: Expression | int, right: Expression | int, *, limits: ComputationLimits | None = None) -> Result[int]:
    result = _run("compare_real", dict(left=left, right=right), LEGACY["compare_real"].produce, limits=limits)
    if result.execution_status is ExecutionStatus.UNSUPPORTED:
        return _run_exact("compare_real", dict(left=left, right=right), limits)
    return result


def rational_function(numerator: PolynomialInput, denominator: PolynomialInput, *, variable: Symbol, limits: ComputationLimits | None = None) -> Result[RationalFunction]:
    return _run("rational_function", dict(numerator=numerator, denominator=denominator, variable=variable),
                LEGACY["rational_function"].produce, limits=limits)


def solve(problem: Eq | Inequality, *, for_: Symbol, limits: ComputationLimits | None = None) -> Result[EquationSolutionSet]:
    return _run("solve", dict(problem=problem, for_=for_), LEGACY["solve"].produce, limits=limits, solutions=True)


def solve_system(system_or_equations: matrices.LinearSystem | Iterable[Eq], *, for_: Iterable[Symbol] | None = None, limits: ComputationLimits | None = None) -> Result[LinearSolutionSet]:
    return _run("solve_system", dict(system_or_equations=system_or_equations, for_=for_),
                LEGACY["solve_system"].produce, limits=limits, solutions=True)


def _matrix_operation(operation, A, limits):
    return _run(operation, dict(A=A), LEGACY[operation].produce, limits=limits)


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
    return _run("differentiate", dict(expression=expression, variable=variable, n=n),
                LEGACY["differentiate"].produce, limits=limits)


def antiderivative(expression: PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Result[Expression]:
    if _function_input(expression):
        return function_antiderivative(expression, variable, limits=limits)
    if type(expression) is RationalFunction:
        return rational_antiderivative(expression, variable, limits=limits)
    return _run("antiderivative", dict(expression=expression, variable=variable), LEGACY["antiderivative"].produce,
                limits=limits)


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
    return _run("integrate", dict(expression=expression, variable=variable, lower=lower, upper=upper),
                LEGACY["integrate"].produce, limits=limits)


def evaluate(expression: ScalarInput, bindings: Bindings | None = None, *, limits: ComputationLimits | None = None) -> Result[Expression]:
    if _function_input(expression):
        return function_evaluate(expression, {} if bindings is None else bindings, limits=limits)
    return _run("evaluate", dict(expression=expression, bindings=bindings), LEGACY["evaluate"].produce, limits=limits)


def substitute(expression: ScalarInput, bindings: Bindings, *, limits: ComputationLimits | None = None) -> Result[Expression | RationalFunction]:
    return _run("substitute", dict(expression=expression, bindings=bindings), LEGACY["substitute"].produce,
                limits=limits)


def simplify(expression: ScalarInput, *, limits: ComputationLimits | None = None) -> Result[Expression | RationalFunction]:
    return _run("simplify", dict(expression=expression), LEGACY["simplify"].produce, limits=limits)


def _run_exact(operation, arguments, limits):
    return _run(operation, arguments, LEGACY_EXACT[operation].produce, limits=limits, _rule_version=4)


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

# Every registry operation, also those added after v0.7: the generated __all__ lists them all, and the v0.7
# names above rebind to the very same wrappers. New families therefore never edit this module.
from ._generated.structured import *
