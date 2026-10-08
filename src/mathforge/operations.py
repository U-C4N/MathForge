"""Structured SDK operations shared with future transport adapters."""

from collections.abc import Mapping

from . import algorithms
from .contracts import (
    Completeness, Eq, ErrorInfo, Exactness, ExecutionStatus, Outcome, Result,
    Step, StepRelation,
)
from .errors import InvalidInput, UnsupportedOperation
from .model import (
    Expression, Reals, Symbol, as_expression, evaluate_expression,
    simplify_expression, substitute_expression,
)
from .polynomial import Polynomial


def _failure(exc, *, method, problem=None, expression=None, variable=None):
    problem = problem if type(problem) is Eq else None
    variable = variable if type(variable) is Symbol else None
    try:
        expression = as_expression(expression) if expression is not None else None
    except TypeError:
        expression = None
    if isinstance(exc, UnsupportedOperation):
        status, code = ExecutionStatus.UNSUPPORTED, "unsupported_operation"
    elif isinstance(exc, (InvalidInput, TypeError, ValueError, ZeroDivisionError)):
        status, code = ExecutionStatus.INVALID_INPUT, "invalid_input"
    else:
        status, code = ExecutionStatus.ERROR, "execution_error"
    return Result(
        execution_status=status, outcome=Outcome.NOT_APPLICABLE,
        error=ErrorInfo(code, str(exc)), method=method,
        problem=problem, expression=expression, for_=variable,
        domain=variable.domain if variable is not None else None,
        assumptions=variable.assumptions if variable is not None else (),
    )


def solve(problem, *, for_) -> Result:
    """Return exact solutions, an explicit unsupported result, or a typed error."""
    try:
        return algorithms.solve_equation(problem, for_)
    except Exception as exc:
        return _failure(exc, method="polynomial_real_solve", problem=problem, variable=for_)


def _value_result(value, expression, method, *, variable=None, inputs=None,
                  outputs=None, relation=StepRelation.EQUIVALENCE):
    return Result(
        execution_status=ExecutionStatus.COMPLETED, outcome=Outcome.VALUE,
        value=value, exactness=Exactness.EXACT, completeness=Completeness.COMPLETE,
        method=method, expression=expression, for_=variable,
        domain=variable.domain if variable is not None else None,
        assumptions=variable.assumptions if variable is not None else (),
        steps=(Step(rule=method, inputs=inputs or (expression,), outputs=outputs or (value,),
                    relation=relation),),
    )


def differentiate(expression, variable) -> Result:
    """Differentiate a rational univariate polynomial with respect to variable."""
    try:
        if type(variable) is not Symbol:
            raise InvalidInput("The differentiation variable must be a Symbol")
        if variable.domain != Reals:
            raise UnsupportedOperation("Polynomial differentiation is currently defined over the reals only")
        if variable.assumptions:
            raise UnsupportedOperation("Additional symbol assumptions are not supported by differentiation")
        expression = as_expression(expression)
        polynomial = Polynomial.from_expression(expression, variable)
        derivative = polynomial.derivative()
        return _value_result(derivative.to_expression(), expression, "polynomial.derivative",
                             variable=variable, inputs=(polynomial,), outputs=(derivative,),
                             relation=StepRelation.DERIVATION)
    except Exception as exc:
        return _failure(exc, method="polynomial.derivative", expression=expression, variable=variable)


def evaluate(expression, bindings=None) -> Result:
    """Evaluate with complete exact bindings; partial bindings are invalid input."""
    try:
        expression = as_expression(expression)
        if bindings is None:
            bindings = {}
        if not isinstance(bindings, Mapping):
            raise InvalidInput("Bindings must map Symbols to exact expressions")
        value = evaluate_expression(expression, bindings)
        return _value_result(value, expression, "expression.evaluate",
                             inputs=(expression, tuple(bindings.items())),
                             relation=StepRelation.EVALUATION)
    except Exception as exc:
        return _failure(exc, method="expression.evaluate", expression=expression)


def substitute(expression, bindings) -> Result:
    """Perform simultaneous symbol substitution while preserving exact objects."""
    try:
        expression = as_expression(expression)
        if not isinstance(bindings, Mapping):
            raise InvalidInput("Bindings must map Symbols to exact expressions")
        value = substitute_expression(expression, bindings)
        return _value_result(value, expression, "expression.substitute",
                             inputs=(expression, tuple(bindings.items())),
                             relation=StepRelation.EVALUATION)
    except Exception as exc:
        return _failure(exc, method="expression.substitute", expression=expression)


def simplify(expression) -> Result:
    """Apply the model's bounded, domain-preserving rewrite rules."""
    try:
        expression = as_expression(expression)
        return _value_result(simplify_expression(expression), expression, "expression.simplify")
    except Exception as exc:
        return _failure(exc, method="expression.simplify", expression=expression)
