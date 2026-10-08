"""MathForge: independent exact mathematics, with an optional workspace.

``solve`` returns a structured Result. Other convenience operations return
mathematical values, or raise OperationError carrying the same Result contract.
Use ``mathforge.operations`` for structured results from every operation.
"""

from . import operations
from .contracts import (
    Check, CheckStatus, Completeness, Condition, EmptySet, Eq, ErrorInfo,
    Exactness, ExecutionStatus, FiniteSet, Outcome, Result, Step, StepRelation,
    UniversalSet, VerificationLevel, VerificationReport,
)
from .errors import (
    InvalidInput, MathForgeError, OperationError, SerializationError, UnsupportedOperation,
)
from .model import (
    Add, Assumption, Domain, Expression, Integers, Mul, Pow, Rational,
    Rationals, Reals, Sqrt, Symbol, Truth, sqrt, symbol,
)
from .operations import solve
from .polynomial import Polynomial
from .serialization import dumps, from_data, loads, to_data
from .verification import verify
from .workspace import ObjectRef, OperationRecord, Workspace

__version__ = "0.1.0"


def diff(expression: Expression, variable: Symbol) -> Expression:
    """Differentiate a rational-coefficient univariate polynomial expression."""
    return operations.differentiate(expression, variable).unwrap()


def evaluate(expression: Expression, bindings=None) -> Expression:
    """Evaluate with complete symbol bindings, retaining exact radical values."""
    return operations.evaluate(expression, bindings).unwrap()


def substitute(expression: Expression, bindings) -> Expression:
    """Simultaneously replace symbols; unbound symbols may remain."""
    return operations.substitute(expression, bindings).unwrap()


def simplify(expression: Expression) -> Expression:
    """Apply only the documented, domain-preserving local rules."""
    return operations.simplify(expression).unwrap()


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
]
