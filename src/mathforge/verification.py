"""Independent exact substitution and degree-based completeness checks.

These are executable algebraic checks, not a formal proof kernel.  Stored
verification labels and explanatory text never serve as evidence.
"""

from dataclasses import dataclass
from math import isqrt

from .contracts import (
    Check, CheckStatus, Completeness, Condition, EmptySet, Eq, Exactness,
    ExecutionStatus, FiniteSet, Outcome, Result, Step, StepRelation,
    UniversalSet, VerificationLevel, VerificationReport,
)
from .errors import InvalidInput, UnsupportedOperation
from .model import Add, Expression, Mul, Pow, Rational, Reals, Sqrt, Symbol, Truth
from .polynomial import Polynomial


def _add(a: Rational, b: Rational) -> Rational:
    return Rational(a.numerator * b.denominator + b.numerator * a.denominator,
                    a.denominator * b.denominator)


def _neg(a: Rational) -> Rational:
    return Rational(-a.numerator, a.denominator)


def _mul(a: Rational, b: Rational) -> Rational:
    return Rational(a.numerator * b.numerator, a.denominator * b.denominator)


def _div(a: Rational, b: Rational) -> Rational:
    return Rational(a.numerator * b.denominator, a.denominator * b.numerator)


def _sqrt_rational(value: Rational) -> Rational | None:
    if value.numerator < 0:
        return None
    n, d = isqrt(value.numerator), isqrt(value.denominator)
    return Rational(n, d) if n * n == value.numerator and d * d == value.denominator else None


_ZERO = Rational(0)
_ONE = Rational(1)


@dataclass(frozen=True)
class _Field:
    """A rational pair u + v sqrt(d); multiplication is checked exactly."""

    d: Rational = _ZERO

    def normalize(self, pair: tuple[Rational, Rational]) -> tuple[Rational, Rational]:
        root = _sqrt_rational(self.d)
        if root is not None:
            return (_add(pair[0], _mul(pair[1], root)), _ZERO)
        return pair

    def add(self, a, b):
        return self.normalize((_add(a[0], b[0]), _add(a[1], b[1])))

    def mul(self, a, b):
        return self.normalize((
            _add(_mul(a[0], b[0]), _mul(_mul(a[1], b[1]), self.d)),
            _add(_mul(a[0], b[1]), _mul(a[1], b[0])),
        ))

    def power(self, base, exponent):
        result = (_ONE, _ZERO)
        while exponent:
            if exponent & 1:
                result = self.mul(result, base)
            exponent >>= 1
            if exponent:
                base = self.mul(base, base)
        return result

    def expression(self, expression: Expression, variable=None, value=None):
        if isinstance(expression, Rational):
            return (expression, _ZERO)
        if isinstance(expression, Symbol):
            if expression == variable and value is not None:
                return value
            raise UnsupportedOperation("Unbound symbol in exact verification")
        if isinstance(expression, Add):
            value_sum = (_ZERO, _ZERO)
            for term in expression.terms:
                value_sum = self.add(value_sum, self.expression(term, variable, value))
            return value_sum
        if isinstance(expression, Mul):
            value_product = (_ONE, _ZERO)
            for factor in expression.factors:
                value_product = self.mul(value_product, self.expression(factor, variable, value))
            return value_product
        if isinstance(expression, Pow):
            return self.power(self.expression(expression.base, variable, value), expression.exponent)
        if isinstance(expression, Sqrt):
            root = _sqrt_rational(expression.radicand)
            if root is not None:
                return (root, _ZERO)
            if self.d.numerator > 0:
                ratio_root = _sqrt_rational(_div(expression.radicand, self.d))
                if ratio_root is not None:
                    return self.normalize((_ZERO, ratio_root))
            raise UnsupportedOperation("The root is outside this equation's quadratic extension")
        raise UnsupportedOperation("Unknown expression node in exact verification")


def _classification(polynomial):
    """Return expected cardinality (-1 means all reals), field, discriminant."""
    if polynomial.is_zero:
        return -1, _Field(), None
    if polynomial.degree == 0:
        return 0, _Field(), None
    if polynomial.degree == 1:
        return 1, _Field(), None
    if polynomial.degree == 2:
        c, b, a = polynomial.coefficients
        discriminant = _add(_mul(b, b), _neg(_mul(Rational(4), _mul(a, c))))
        if discriminant.numerator < 0:
            return 0, _Field(), discriminant
        if discriminant.numerator == 0:
            return 1, _Field(), discriminant
        return 2, _Field(_div(discriminant, _mul(Rational(4), _mul(a, a)))), discriminant
    raise UnsupportedOperation("Verification supports degree at most two")


def _check(claim, passed, method, *, subject=None, details="", substitution=False):
    return Check(
        claim=claim, status=CheckStatus.VERIFIED if passed else CheckStatus.FAILED,
        level=(VerificationLevel.EXACT_SUBSTITUTION if substitution
               else VerificationLevel.ALGORITHMIC_CHECK),
        method=method, subject=subject, details=details,
    )


def _unknown(claim, method, details, subject=None):
    return Check(claim=claim, status=CheckStatus.UNKNOWN, level=VerificationLevel.NONE,
                 method=method, subject=subject, details=details)


def _condition_holds(condition: Condition) -> bool:
    if not isinstance(condition.lhs, Rational) or not isinstance(condition.rhs, Rational):
        return False
    difference = (condition.lhs.numerator * condition.rhs.denominator
                  - condition.rhs.numerator * condition.lhs.denominator)
    predicates = {
        "eq": difference == 0, "ne": difference != 0,
        "lt": difference < 0, "le": difference <= 0,
        "gt": difference > 0, "ge": difference >= 0,
    }
    return condition.truth == Truth.TRUE and predicates.get(condition.relation, False)


def _solution_checks(problem: Eq, variable: Symbol, polynomial: Polynomial, solution):
    count, field, _ = _classification(polynomial)
    checks = [_check("solution.domain", getattr(solution, "domain", None) == Reals,
                     "domain_check")]
    if isinstance(solution, UniversalSet):
        checks.append(_check("solution.completeness", count == -1,
                             "zero_polynomial_identity"))
        return checks
    if isinstance(solution, EmptySet):
        checks.append(_check("solution.completeness", count == 0,
                             "degree_and_discriminant"))
        return checks
    if not isinstance(solution, FiniteSet):
        checks.append(_check("solution.completeness", False, "solution_set_type"))
        return checks
    coordinates = []
    membership_verified = True
    for root in solution.values:
        try:
            root_pair = field.expression(root)
            coordinates.append(root_pair)
            lhs = field.expression(problem.lhs, variable, root_pair)
            rhs = field.expression(problem.rhs, variable, root_pair)
            valid = lhs == rhs
            membership_verified &= valid
            checks.append(_check("root.membership", valid, "exact_quadratic_substitution",
                                 subject=root, substitution=True))
        except (UnsupportedOperation, InvalidInput, ValueError, TypeError, ZeroDivisionError) as exc:
            membership_verified = False
            checks.append(_unknown("root.membership", "exact_quadratic_substitution",
                                   str(exc), root))
    # Degree/discriminant establishes how many distinct real roots exist;
    # substitution alone cannot establish this claim.
    complete = membership_verified and count >= 0 and len(set(coordinates)) == count
    checks.append(_check("solution.completeness", complete, "degree_and_discriminant",
                         details=f"Expected distinct real roots: {count}; checked: {len(set(coordinates))}"))
    return checks


def verify_step(step: Step) -> Check:
    """Recompute a supported rule, ignoring its recorded audit and explanation."""
    if type(step) is Step and step.rule.startswith("v2."):
        from .verification_v2 import verify_step_v2
        return verify_step_v2(step)
    try:
        if not isinstance(step, Step):
            return _check("step.validity", False, "typed_step")
        if not all(_condition_holds(c) for c in step.preconditions):
            return _check(f"step.{step.rule}", False, "precondition_check")
        if step.rule == "polynomial.normalize":
            problem, variable = step.inputs
            (output,) = step.outputs
            passed = (isinstance(problem, Eq) and isinstance(variable, Symbol)
                      and step.relation == StepRelation.EQUIVALENCE
                      and output == Polynomial.from_expression(problem.lhs - problem.rhs, variable))
        elif step.rule == "polynomial.discriminant":
            (polynomial,) = step.inputs
            (output,) = step.outputs
            passed = (isinstance(polynomial, Polynomial) and polynomial.degree == 2
                      and step.relation == StepRelation.EVALUATION
                      and output == _classification(polynomial)[2])
        elif step.rule.startswith("polynomial.solve_"):
            polynomial = step.inputs[0]
            (output,) = step.outputs
            count, _, discriminant = _classification(polynomial)
            expected_rule = (
                "zero" if polynomial.is_zero else
                "constant" if polynomial.degree == 0 else
                "linear" if polynomial.degree == 1 else
                "quadratic_none" if count == 0 else
                "quadratic_double" if count == 1 else "quadratic_two"
            )
            passed = (step.rule == f"polynomial.solve_{expected_rule}"
                      and step.relation == StepRelation.EQUIVALENCE
                      and len(step.inputs) == (2 if polynomial.degree == 2 else 1))
            if polynomial.degree == 2:
                passed &= step.inputs[1] == discriminant
            if passed:
                checks = _solution_checks(Eq(polynomial.to_expression(), 0), polynomial.variable,
                                          polynomial, output)
                passed &= all(check.status == CheckStatus.VERIFIED for check in checks)
        else:
            return _unknown(f"step.{step.rule}", "rule_registry", "This rule has no independent checker")
        return _check(f"step.{step.rule}", passed, "recomputed_rule")
    except (AttributeError, IndexError, TypeError, ValueError, UnsupportedOperation,
            InvalidInput, ZeroDivisionError) as exc:
        return _check(f"step.{getattr(step, 'rule', 'unknown')}", False,
                      "recomputed_rule", details=str(exc))


def _verify_legacy(result: Result) -> VerificationReport:
    """Check a solution independently, including completeness and recorded rules.

    Returns unknown for operations outside the supported verification scope.
    No solver is called, and no stored check status is trusted.
    """
    if type(result) is not Result:
        raise InvalidInput("verify expects a Result")
    if not isinstance(result.problem, Eq) or not isinstance(result.for_, Symbol):
        return VerificationReport((_unknown("result.verification", "verification_scope",
                                             "Only equation results are currently checked"),))
    variable = result.for_
    if variable.domain != Reals or variable.assumptions:
        return VerificationReport((_unknown("result.verification", "verification_scope",
                                             "Only unconstrained real equations are currently checked"),))
    try:
        polynomial = Polynomial.from_expression(result.problem.lhs - result.problem.rhs, variable)
        expected_count, _, _ = _classification(polynomial)
        checks = [_check("result.context", result.domain == Reals
                         and result.assumptions == variable.assumptions, "context_consistency")]
        expected_outcome = Outcome.NO_SOLUTION if expected_count == 0 else Outcome.SOLUTIONS
        checks.append(_check(
            "result.contract", result.execution_status == ExecutionStatus.COMPLETED
            and result.outcome == expected_outcome and result.exactness == Exactness.EXACT
            and result.completeness == Completeness.COMPLETE and result.error is None
            and result.precision is None and result.tolerance is None and result.error_bound is None,
            "result_axes_consistency",
        ))
        checks.extend(_solution_checks(result.problem, variable, polynomial, result.value))
        checks.extend(verify_step(step) for step in result.steps)
        # Independently valid rules must also belong to this source problem.
        for step in result.steps:
            if step.rule == "polynomial.normalize":
                connected = step.inputs == (result.problem, variable) and step.outputs == (polynomial,)
            elif step.rule.startswith("polynomial."):
                connected = bool(step.inputs) and step.inputs[0] == polynomial
                if step.rule.startswith("polynomial.solve_"):
                    connected &= step.outputs == (result.value,)
            else:
                connected = False
            checks.append(_check(f"step.context.{step.rule}", connected, "trace_consistency"))
        return VerificationReport(tuple(checks))
    except (UnsupportedOperation, InvalidInput, TypeError, ValueError, ZeroDivisionError) as exc:
        return VerificationReport((_unknown("result.verification", "verification_scope", str(exc)),))


def verify(result: Result, *, limits=None) -> VerificationReport:
    """Independently check the source, result, completeness and attached evidence."""
    from .verification_v2 import verify_result
    return verify_result(result, limits=limits)
