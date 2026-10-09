"""MathForge: independent exact mathematics, with an optional workspace.

``solve`` returns a structured Result. Other convenience operations return
mathematical values, or raise OperationError carrying the same Result contract.
Use ``mathforge.operations`` for structured results from every operation.
"""

from . import operations
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
from .algebraic import RealAlgebraicRoot, RootRecord
from .rational_function import RationalFunction
from .matrices import Matrix, RREFResult, LinearSystem, AffineSolutionSet
from .intervals import Interval, IntervalSet
from .limits import ComputationLimits, DecodeLimits, ResourceLimitError
from .serialization import dumps, from_data, loads, to_data
from .verification import verify
from .workspace import ObjectRef, OperationRecord, Workspace

__version__ = "0.2.0"


def diff(expression: Expression | Polynomial | RationalFunction, variable: Symbol,
         n: int = 1, *, limits: ComputationLimits | None = None) -> Expression | RationalFunction:
    """Repeated exact derivative; rational functions retain their domain holes."""
    return operations.differentiate(expression, variable, n, limits=limits).unwrap()


def evaluate(expression: Expression | RationalFunction, bindings=None, *, limits=None) -> Expression:
    """Evaluate with complete symbol bindings, retaining exact radical values."""
    return operations.evaluate(expression, bindings, limits=limits).unwrap()


def substitute(expression: Expression | RationalFunction, bindings, *, limits=None) -> Expression | RationalFunction:
    """Simultaneously replace symbols; unbound symbols may remain."""
    return operations.substitute(expression, bindings, limits=limits).unwrap()


def simplify(expression: Expression | RationalFunction, *, limits=None) -> Expression | RationalFunction:
    """Apply only the documented, domain-preserving local rules."""
    return operations.simplify(expression, limits=limits).unwrap()


def expand(expression: Expression | Polynomial, variable: Symbol, *, limits=None) -> Expression:
    return operations.expand(expression, variable, limits=limits).unwrap()


def poly_divmod(f: Polynomial, g: Polynomial, *, limits=None) -> tuple[Polynomial, Polynomial]:
    return operations.poly_divmod(f, g, limits=limits).unwrap()


def poly_gcd(f: Polynomial, g: Polynomial, *, limits=None) -> Polynomial:
    return operations.poly_gcd(f, g, limits=limits).unwrap()


def poly_xgcd(f: Polynomial, g: Polynomial, *, limits=None) -> tuple[Polynomial, Polynomial, Polynomial]:
    return operations.poly_xgcd(f, g, limits=limits).unwrap()


def square_free(f: Polynomial, *, limits=None) -> SquareFreeDecomposition:
    return operations.square_free(f, limits=limits).unwrap()


def real_roots(expression: Expression | Polynomial, variable: Symbol, *, limits=None) -> tuple[RootRecord, ...]:
    return operations.real_roots(expression, variable, limits=limits).unwrap()


def compare_real(left: Expression | int, right: Expression | int, *, limits=None) -> int:
    return operations.compare_real(left, right, limits=limits).unwrap()


def rational_function(numerator, denominator, *, variable: Symbol, limits=None) -> RationalFunction:
    return operations.rational_function(numerator, denominator, variable=variable, limits=limits).unwrap()


def antiderivative(expression: Expression | Polynomial, variable: Symbol, *, limits=None) -> Expression:
    return operations.antiderivative(expression, variable, limits=limits).unwrap()


def integrate(expression: Expression | Polynomial, variable: Symbol, lower, upper, *, limits=None) -> Rational:
    return operations.integrate(expression, variable, lower, upper, limits=limits).unwrap()


def rref(A: Matrix, *, limits=None) -> RREFResult:
    return operations.rref(A, limits=limits).unwrap()


def rank(A: Matrix, *, limits=None) -> int:
    return operations.rank(A, limits=limits).unwrap()


def det(A: Matrix, *, limits=None) -> Rational:
    return operations.det(A, limits=limits).unwrap()


def inverse(A: Matrix, *, limits=None) -> Matrix:
    return operations.inverse(A, limits=limits).unwrap()


def nullspace(A: Matrix, *, limits=None) -> tuple[tuple[Rational, ...], ...]:
    return operations.nullspace(A, limits=limits).unwrap()


__all__ = [
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
]
