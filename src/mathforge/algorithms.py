"""Exact, bounded algorithms; no transport or storage dependencies."""

from dataclasses import replace

from .contracts import (
    Completeness, Condition, EmptySet, Eq, Exactness,
    ExecutionStatus, FiniteSet, Outcome, Result, Step, StepRelation,
    UniversalSet,
)
from .errors import InvalidInput, UnsupportedOperation
from .model import Rational, Reals, Symbol, Truth, sqrt
from .polynomial import Polynomial


def solve_equation(problem: Eq, variable: Symbol) -> Result:
    """Solve rational univariate equations of degree at most two over R.

    The caller translates exceptions to the shared result contract.  Every
    returned complete answer has been checked against the original equation.
    """
    if type(problem) is not Eq or type(variable) is not Symbol:
        raise InvalidInput("solve expects an Eq and a Symbol")
    if variable.domain != Reals:
        raise UnsupportedOperation("The first solver supports the real domain only")
    if variable.assumptions:
        raise UnsupportedOperation("Additional symbol assumptions are not supported by this solver")

    polynomial = Polynomial.from_expression(problem.lhs - problem.rhs, variable)
    if polynomial.degree > 2:
        raise UnsupportedOperation("Only equations of degree at most two are supported")
    steps = [Step(
        rule="polynomial.normalize", inputs=(problem, variable),
        outputs=(polynomial,), relation=StepRelation.EQUIVALENCE,
        explanation="Collect the rational coefficients of lhs - rhs.",
    )]
    zero = Rational(0)
    if polynomial.is_zero:
        value = UniversalSet(Reals)
        rule = "polynomial.solve_zero"
        method = "zero_polynomial"
        preconditions = ()
        inputs = (polynomial,)
    elif polynomial.degree == 0:
        value = EmptySet(Reals)
        rule = "polynomial.solve_constant"
        method = "nonzero_constant"
        preconditions = (Condition(polynomial.coefficients[0], "ne", zero, Truth.TRUE),)
        inputs = (polynomial,)
    elif polynomial.degree == 1:
        c, b = polynomial.coefficients
        value = FiniteSet((-c / b,), Reals)
        rule = "polynomial.solve_linear"
        method = "linear_isolation"
        preconditions = (Condition(b, "ne", zero, Truth.TRUE),)
        inputs = (polynomial,)
    else:
        c, b, a = polynomial.coefficients
        discriminant = b * b - 4 * a * c
        steps.append(Step(
            rule="polynomial.discriminant", inputs=(polynomial,),
            outputs=(discriminant,), relation=StepRelation.EVALUATION,
            explanation="Compute b² - 4ac using exact rational arithmetic.",
        ))
        inputs = (polynomial, discriminant)
        if discriminant < zero:
            value = EmptySet(Reals)
            rule = "polynomial.solve_quadratic_none"
            condition = "lt"
        elif discriminant == zero:
            value = FiniteSet((-b / (2 * a),), Reals)
            rule = "polynomial.solve_quadratic_double"
            condition = "eq"
        else:
            center = -b / (2 * a)
            radius = sqrt(discriminant / (4 * a * a))
            value = FiniteSet((center - radius, center + radius), Reals)
            rule = "polynomial.solve_quadratic_two"
            condition = "gt"
        method = "quadratic_discriminant"
        preconditions = (
            Condition(a, "ne", zero, Truth.TRUE),
            Condition(discriminant, condition, zero, Truth.TRUE),
        )
    steps.append(Step(
        rule=rule, inputs=inputs, outputs=(value,), preconditions=preconditions,
        relation=StepRelation.EQUIVALENCE,
        explanation="Apply the degree and discriminant classification over the real numbers.",
    ))
    result = Result(
        execution_status=ExecutionStatus.COMPLETED,
        outcome=Outcome.NO_SOLUTION if isinstance(value, EmptySet) else Outcome.SOLUTIONS,
        value=value, exactness=Exactness.EXACT, completeness=Completeness.COMPLETE,
        method=method, domain=Reals, assumptions=variable.assumptions,
        steps=tuple(steps), problem=problem, for_=variable,
    )
    # This import keeps the verifier independent of the solving implementation.
    from .verification import verify, verify_step
    result = replace(result, steps=tuple(
        replace(step, audit=(verify_step(step),)) for step in result.steps
    ))
    report = verify(result)
    if not report.verified:
        # A failure of our bounded algorithm's own checks is an execution error,
        # never a mathematical claim that the original equation has no solution.
        raise ArithmeticError("Internal exact solution verification failed")
    return replace(result, verification=report.checks)
