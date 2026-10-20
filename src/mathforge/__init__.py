"""MathForge: independent exact mathematics, with an optional workspace.

``solve`` returns a structured Result. Other convenience operations return
mathematical values, or raise OperationError carrying the same Result contract.
Use ``mathforge.operations`` for structured results from every operation.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from . import operations, registry
from .lean import FALLBACK as _FALLBACK, functions as _LEAN
from .contracts import (
    Check, CheckStatus, Completeness, Condition, EmptySet, Eq, ErrorInfo,
    Exactness, ExecutionStatus, FiniteSet, Outcome, Result, Step, StepRelation,
    UniversalSet, VerificationLevel, VerificationReport, Inequality, OperationRequest,
)
from .errors import (
    InvalidInput, MathForgeError, OperationError, SerializationError, UnsupportedOperation,
)
from .model import (
    Add, Assumption, Domain, Expression, Integers, Mul, Pow, Rational,
    Rationals, Reals, Sqrt, Symbol, Truth, sqrt, symbol,
)
from .operations import solve, solve_system
from .polynomial import Polynomial, SquareFreeFactor, SquareFreeDecomposition
from .polynomial_exact import PolynomialFactor, PolynomialFactorization
from .exact_extensions import PartialFractionTerm, PartialFractionDecomposition
from .algebraic import RealAlgebraicRoot, RootRecord
from .rational_function import RationalFunction
from .matrices import Matrix, RREFResult, LinearSystem, AffineSolutionSet
from .intervals import Interval, IntervalSet
from .limits import ComputationLimits, DecodeLimits, ResourceLimitError
from .serialization import dumps, from_data, loads, to_data
from .verification import verify
from .workspace import ObjectRef, OperationRecord, Workspace
from .rendering import RenderLimits, render_text, to_latex
from .parsing import ParseError, ParseLimits, parse_expression, parse_problem
from .capabilities import Capabilities, ToolCapability, capabilities
from .analysis_types import ExcludedPoint, FunctionAnalysis, LimitValue, LocalExtremum, MonotonicRegion, ExtremumValue
from typing import TYPE_CHECKING, Literal, overload
from collections.abc import Iterable

__version__ = "0.8.0"

RealAlgebraic = Rational | RealAlgebraicRoot


def factor(f: Polynomial, *, limits: ComputationLimits | None = None) -> PolynomialFactorization:
    """Return a complete irreducible factorization over the rational numbers."""
    if limits is None:
        _value = _LEAN.factor(f)
        if _value is not _FALLBACK:
            return _value
    return operations.factor(f, limits=limits).unwrap()


def resultant(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Rational:
    if limits is None:
        _value = _LEAN.resultant(f, g)
        if _value is not _FALLBACK:
            return _value
    return operations.resultant(f, g, limits=limits).unwrap()


def discriminant(f: Polynomial, *, limits: ComputationLimits | None = None) -> Rational:
    if limits is None:
        _value = _LEAN.discriminant(f)
        if _value is not _FALLBACK:
            return _value
    return operations.discriminant(f, limits=limits).unwrap()


def algebraic(expression: Expression | int, *, limits: ComputationLimits | None = None) -> RealAlgebraic:
    """Explicitly normalize a closed real algebraic expression."""
    if limits is None:
        _value = _LEAN.algebraic(expression)
        if _value is not _FALLBACK:
            return _value
    return operations.algebraic(expression, limits=limits).unwrap()


def algebraic_div(left: Expression | int, right: Expression | int, *, limits: ComputationLimits | None = None) -> RealAlgebraic:
    if limits is None:
        _value = _LEAN.algebraic_div(left, right)
        if _value is not _FALLBACK:
            return _value
    return operations.algebraic_div(left, right, limits=limits).unwrap()


def algebraic_pow(base: Expression | int, n: int, *, limits: ComputationLimits | None = None) -> RealAlgebraic:
    if limits is None:
        _value = _LEAN.algebraic_pow(base, n)
        if _value is not _FALLBACK:
            return _value
    return operations.algebraic_pow(base, n, limits=limits).unwrap()


def minimal_polynomial(expression: Expression | int, variable: Symbol, *, limits: ComputationLimits | None = None) -> Polynomial:
    if limits is None:
        _value = _LEAN.minimal_polynomial(expression, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.minimal_polynomial(expression, variable, limits=limits).unwrap()


def evaluate_algebraic(expression: operations.PolynomialInput | RationalFunction,
                       bindings: operations.Bindings | None = None, *,
                       limits: ComputationLimits | None = None) -> RealAlgebraic:
    if limits is None:
        _value = _LEAN.evaluate_algebraic(expression, bindings)
        if _value is not _FALLBACK:
            return _value
    return operations.evaluate_algebraic(expression, bindings, limits=limits).unwrap()


def apart(expression: RationalFunction, *, limits: ComputationLimits | None = None) -> PartialFractionDecomposition:
    if limits is None:
        _value = _LEAN.apart(expression)
        if _value is not _FALLBACK:
            return _value
    return operations.apart(expression, limits=limits).unwrap()


def extrema_values(expression: operations.PolynomialInput | RationalFunction, variable: Symbol, *,
                   limits: ComputationLimits | None = None) -> tuple[ExtremumValue, ...]:
    if limits is None:
        _value = _LEAN.extrema_values(expression, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.extrema_values(expression, variable, limits=limits).unwrap()


def charpoly(A: Matrix, variable: Symbol, *, limits: ComputationLimits | None = None) -> Polynomial:
    if limits is None:
        _value = _LEAN.charpoly(A, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.charpoly(A, variable, limits=limits).unwrap()


def real_eigenvalues(A: Matrix, *, limits: ComputationLimits | None = None) -> tuple[RootRecord, ...]:
    if limits is None:
        _value = _LEAN.real_eigenvalues(A)
        if _value is not _FALLBACK:
            return _value
    return operations.real_eigenvalues(A, limits=limits).unwrap()


def expand_multivariate(expression: Expression | int, variables: Iterable[Symbol], *,
                        limits: ComputationLimits | None = None) -> Expression:
    """Expand a rational polynomial using a sparse multivariate representation."""
    if limits is None:
        _value = _LEAN.expand_multivariate(expression, variables)
        if _value is not _FALLBACK:
            return _value
    return operations.expand_multivariate(expression, variables, limits=limits).unwrap()


def partial_diff(expression: Expression | int, variable: Symbol, n: int = 1, *,
                 limits: ComputationLimits | None = None) -> Expression:
    """Take an exact partial derivative while holding other symbols fixed."""
    if limits is None:
        _value = _LEAN.partial_diff(expression, variable, n)
        if _value is not _FALLBACK:
            return _value
    return operations.partial_diff(expression, variable, n, limits=limits).unwrap()


def gradient(expression: Expression | int, variables: Iterable[Symbol], *,
             limits: ComputationLimits | None = None) -> operations.ExpressionVector:
    if limits is None:
        _value = _LEAN.gradient(expression, variables)
        if _value is not _FALLBACK:
            return _value
    return operations.gradient(expression, variables, limits=limits).unwrap()


def jacobian(expressions: Iterable[Expression | int], variables: Iterable[Symbol], *,
             limits: ComputationLimits | None = None) -> operations.ExpressionMatrix:
    """Rows follow the functions; columns follow the supplied variables."""
    if limits is None:
        _value = _LEAN.jacobian(expressions, variables)
        if _value is not _FALLBACK:
            return _value
    return operations.jacobian(expressions, variables, limits=limits).unwrap()


def hessian(expression: Expression | int, variables: Iterable[Symbol], *,
            limits: ComputationLimits | None = None) -> operations.ExpressionMatrix:
    if limits is None:
        _value = _LEAN.hessian(expression, variables)
        if _value is not _FALLBACK:
            return _value
    return operations.hessian(expression, variables, limits=limits).unwrap()


def limit(expression: operations.PolynomialInput | RationalFunction, variable: Symbol,
          point: Expression | int | Literal["-inf", "+inf"], *,
          side: Literal["left", "right", "both"] = "both",
          limits: ComputationLimits | None = None) -> LimitValue:
    """Compute an exact one-sided, two-sided or infinite rational-function limit."""
    if limits is None:
        _value = _LEAN.limit(expression, variable, point, side)
        if _value is not _FALLBACK:
            return _value
    return operations.limit(expression, variable, point, side=side, limits=limits).unwrap()


def analyze_function(expression: operations.PolynomialInput | RationalFunction, variable: Symbol, *,
                     limits: ComputationLimits | None = None) -> FunctionAnalysis:
    """Inspect real domain, signs, stationary points and maximal monotonic regions."""
    if limits is None:
        _value = _LEAN.analyze_function(expression, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.analyze_function(expression, variable, limits=limits).unwrap()


def diff(expression: Expression | Polynomial | RationalFunction | int, variable: Symbol,
         n: int = 1, *, limits: ComputationLimits | None = None) -> Expression | RationalFunction:
    """Repeated exact derivative; rational functions retain their domain holes."""
    if limits is None:
        _value = _LEAN.differentiate(expression, variable, n)
        if _value is not _FALLBACK:
            return _value
    return operations.differentiate(expression, variable, n, limits=limits).unwrap()


def evaluate(expression: operations.ScalarInput, bindings: operations.Bindings | None = None,
             *, limits: ComputationLimits | None = None) -> Expression:
    """Evaluate with complete symbol bindings, retaining exact radical values."""
    if limits is None:
        _value = _LEAN.evaluate(expression, bindings)
        if _value is not _FALLBACK:
            return _value
    return operations.evaluate(expression, bindings, limits=limits).unwrap()


def substitute(expression: operations.ScalarInput, bindings: operations.Bindings,
               *, limits: ComputationLimits | None = None) -> Expression | RationalFunction:
    """Simultaneously replace symbols; unbound symbols may remain."""
    if limits is None:
        _value = _LEAN.substitute(expression, bindings)
        if _value is not _FALLBACK:
            return _value
    return operations.substitute(expression, bindings, limits=limits).unwrap()


def simplify(expression: operations.ScalarInput, *, limits: ComputationLimits | None = None) -> Expression | RationalFunction:
    """Apply only the documented, domain-preserving local rules."""
    if limits is None:
        _value = _LEAN.simplify(expression)
        if _value is not _FALLBACK:
            return _value
    return operations.simplify(expression, limits=limits).unwrap()


def expand(expression: operations.PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Expression:
    if limits is None:
        _value = _LEAN.expand(expression, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.expand(expression, variable, limits=limits).unwrap()


def poly_divmod(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> tuple[Polynomial, Polynomial]:
    if limits is None:
        _value = _LEAN.poly_divmod(f, g)
        if _value is not _FALLBACK:
            return _value
    return operations.poly_divmod(f, g, limits=limits).unwrap()


def poly_gcd(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> Polynomial:
    if limits is None:
        _value = _LEAN.poly_gcd(f, g)
        if _value is not _FALLBACK:
            return _value
    return operations.poly_gcd(f, g, limits=limits).unwrap()


def poly_xgcd(f: Polynomial, g: Polynomial, *, limits: ComputationLimits | None = None) -> tuple[Polynomial, Polynomial, Polynomial]:
    if limits is None:
        _value = _LEAN.poly_xgcd(f, g)
        if _value is not _FALLBACK:
            return _value
    return operations.poly_xgcd(f, g, limits=limits).unwrap()


def square_free(f: Polynomial, *, limits: ComputationLimits | None = None) -> SquareFreeDecomposition:
    if limits is None:
        _value = _LEAN.square_free(f)
        if _value is not _FALLBACK:
            return _value
    return operations.square_free(f, limits=limits).unwrap()


def real_roots(expression: operations.PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> tuple[RootRecord, ...]:
    if limits is None:
        _value = _LEAN.real_roots(expression, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.real_roots(expression, variable, limits=limits).unwrap()


def compare_real(left: Expression | int, right: Expression | int, *, limits: ComputationLimits | None = None) -> int:
    if limits is None:
        _value = _LEAN.compare_real(left, right)
        if _value is not _FALLBACK:
            return _value
    return operations.compare_real(left, right, limits=limits).unwrap()


def rational_function(numerator: operations.PolynomialInput, denominator: operations.PolynomialInput,
                      *, variable: Symbol, limits: ComputationLimits | None = None) -> RationalFunction:
    if limits is None:
        _value = _LEAN.rational_function(numerator, denominator, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.rational_function(numerator, denominator, variable=variable, limits=limits).unwrap()


def antiderivative(expression: operations.PolynomialInput, variable: Symbol, *, limits: ComputationLimits | None = None) -> Expression:
    if limits is None:
        _value = _LEAN.antiderivative(expression, variable)
        if _value is not _FALLBACK:
            return _value
    return operations.antiderivative(expression, variable, limits=limits).unwrap()


@overload
def integrate(expression: operations.PolynomialInput, variable: Symbol, lower: Rational | int,
              upper: Rational | int, *, limits: ComputationLimits | None = None) -> Rational: ...


@overload
def integrate(expression: operations.PolynomialInput, variable: Symbol, lower: Expression | int,
              upper: Expression | int, *, limits: ComputationLimits | None = None) -> RealAlgebraic: ...


def integrate(expression: operations.PolynomialInput, variable: Symbol, lower: Expression | int,
              upper: Expression | int, *, limits: ComputationLimits | None = None) -> RealAlgebraic:
    if limits is None:
        _value = _LEAN.integrate(expression, variable, lower, upper)
        if _value is not _FALLBACK:
            return _value
    return operations.integrate(expression, variable, lower, upper, limits=limits).unwrap()


def rref(A: Matrix, *, limits: ComputationLimits | None = None) -> RREFResult:
    if limits is None:
        _value = _LEAN.rref(A)
        if _value is not _FALLBACK:
            return _value
    return operations.rref(A, limits=limits).unwrap()


def rank(A: Matrix, *, limits: ComputationLimits | None = None) -> int:
    if limits is None:
        _value = _LEAN.rank(A)
        if _value is not _FALLBACK:
            return _value
    return operations.rank(A, limits=limits).unwrap()


def det(A: Matrix, *, limits: ComputationLimits | None = None) -> Rational:
    if limits is None:
        _value = _LEAN.det(A)
        if _value is not _FALLBACK:
            return _value
    return operations.det(A, limits=limits).unwrap()


def inverse(A: Matrix, *, limits: ComputationLimits | None = None) -> Matrix:
    if limits is None:
        _value = _LEAN.inverse(A)
        if _value is not _FALLBACK:
            return _value
    return operations.inverse(A, limits=limits).unwrap()


def nullspace(A: Matrix, *, limits: ComputationLimits | None = None) -> tuple[tuple[Rational, ...], ...]:
    if limits is None:
        _value = _LEAN.nullspace(A)
        if _value is not _FALLBACK:
            return _value
    return operations.nullspace(A, limits=limits).unwrap()


__all__ = [
    "PolynomialFactor", "PolynomialFactorization", "PartialFractionTerm", "PartialFractionDecomposition",
    "ExtremumValue", "RealAlgebraic", "factor", "resultant", "discriminant", "algebraic",
    "algebraic_div", "algebraic_pow", "minimal_polynomial", "evaluate_algebraic", "apart",
    "extrema_values", "charpoly", "real_eigenvalues",
    "Add", "Assumption", "Check", "CheckStatus", "Completeness", "Condition",
    "Domain", "EmptySet", "Eq", "ErrorInfo", "Exactness", "ExecutionStatus",
    "Expression", "FiniteSet", "Integers", "InvalidInput", "MathForgeError", "Mul",
    "ObjectRef", "OperationError", "OperationRecord", "Outcome", "Polynomial", "Pow",
    "Rational", "Rationals", "Reals", "Result", "SerializationError", "Sqrt", "Step",
    "StepRelation", "Symbol", "Truth", "UniversalSet", "UnsupportedOperation",
    "VerificationLevel", "VerificationReport", "Workspace", "diff", "dumps", "evaluate",
    "from_data", "loads", "operations", "simplify", "solve", "sqrt", "substitute",
    "symbol", "to_data", "verify",
    "AffineSolutionSet", "ComputationLimits", "DecodeLimits", "Inequality", "Interval",
    "IntervalSet", "LinearSystem", "Matrix", "OperationRequest", "RREFResult",
    "RationalFunction", "RealAlgebraicRoot", "ResourceLimitError", "RootRecord",
    "SquareFreeDecomposition", "SquareFreeFactor", "antiderivative", "compare_real",
    "det", "expand", "integrate", "inverse", "nullspace", "poly_divmod", "poly_gcd",
    "poly_xgcd", "rank", "rational_function", "real_roots", "rref", "solve_system", "square_free",
    "RenderLimits", "render_text", "to_latex",
    "ParseError", "ParseLimits", "parse_expression", "parse_problem",
    "Capabilities", "ToolCapability", "capabilities",
    "ExcludedPoint", "FunctionAnalysis", "LimitValue", "LocalExtremum", "MonotonicRegion",
    "limit", "analyze_function",
    "expand_multivariate", "partial_diff", "gradient", "jacobian", "hessian",
]

from . import functions
from .functions import (RealExpr, RealFunction, DomainPredicate, UnevaluatedIntegral,
                        UnevaluatedLimit, real_function, parse_function)
from .function_calculus import TaylorPolynomial, FunctionLimitValue
from .values_v7 import *
from .values_v7 import __all__ as _v7_names
from .number_theory import CRTResult, IntegerFactorization
from .complex_numbers import ComplexAlgebraic, ComplexRoot
from .probability import FiniteDistribution
from .linear_extensions import LUDecomposition, LeastSquaresResult
from .rational_integration import LogRootSum, RationalPrimitive

__all__ += _v7_names + ["functions", "RealExpr", "RealFunction", "DomainPredicate",
    "UnevaluatedIntegral", "UnevaluatedLimit", "real_function", "parse_function",
    "TaylorPolynomial", "FunctionLimitValue", "CRTResult", "IntegerFactorization", "ComplexAlgebraic", "ComplexRoot",
    "FiniteDistribution", "LUDecomposition", "LeastSquaresResult", "LogRootSum", "RationalPrimitive"]

# The area modules (mathforge.ntheory, mathforge.stats, ...) load on first attribute access (PEP 562), so
# that importing mathforge loads none of them. Type checkers see them through the hand-written import block
# below, which must name every entry of registry.AREAS (tests/v08/test_lazy_areas.py checks that). They
# never see __getattr__: a module-level one would make them accept any misspelled mathforge.<name>.
if TYPE_CHECKING:
    from . import (
        expr, algebra, sets, logic, equations, analysis, series, integrals, ode,
        special, linalg, stats, ntheory, discrete, geometry, vector, numeric, crypto,
    )
else:
    def __getattr__(name: str) -> ModuleType:
        if name in registry.AREAS:
            return importlib.import_module("." + name, __name__)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *registry.AREAS})
