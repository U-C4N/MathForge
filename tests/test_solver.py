"""Mathematical and evidence checks for the bounded exact solver."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge.verification import verify_step


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x", domain=mf.Reals)

    def solve(self, expression, rhs=0):
        result = mf.solve(mf.Eq(expression, rhs), for_=self.x)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.COMPLETED, result.error)
        self.assertEqual(result.exactness, mf.Exactness.EXACT)
        self.assertEqual(result.completeness, mf.Completeness.COMPLETE)
        self.assertTrue(mf.verify(result).verified)
        self.assertTrue(result.steps)
        self.assertTrue(all(step.audit for step in result.steps))
        self.assertTrue(all(check.status == mf.CheckStatus.VERIFIED
                            for step in result.steps for check in step.audit))
        return result

    def test_linear_exact_rational_root(self):
        result = self.solve(3 * self.x + 2)
        self.assertEqual(result.solution_set.values, (mf.Rational(-2, 3),))
        self.assertEqual(result.method, "linear_isolation")

    def test_nonzero_constant_has_no_roots(self):
        result = self.solve(self.x - self.x + 1)
        self.assertEqual(result.outcome, mf.Outcome.NO_SOLUTION)
        self.assertEqual(result.solution_set, mf.EmptySet())

    def test_zero_polynomial_all_reals(self):
        result = self.solve(2 * self.x + 1, self.x + self.x + 1)
        self.assertEqual(result.solution_set, mf.UniversalSet())
        self.assertEqual(result.outcome, mf.Outcome.SOLUTIONS)

    def test_two_integer_roots(self):
        result = self.solve(self.x**2 - 5 * self.x + 6)
        self.assertEqual(set(result.solution_set.values), {mf.Rational(2), mf.Rational(3)})

    def test_double_root_counted_once(self):
        result = self.solve((self.x - 3)**2)
        self.assertEqual(result.solution_set.values, (mf.Rational(3),))

    def test_no_real_quadratic_roots(self):
        result = self.solve(self.x**2 + 1)
        self.assertEqual(result.solution_set, mf.EmptySet())
        self.assertEqual(result.outcome, mf.Outcome.NO_SOLUTION)

    def test_quadratic_coefficient_zero_reduces_to_linear(self):
        p = mf.Polynomial((6, -2, 0), self.x)
        result = self.solve(p.to_expression())
        self.assertEqual(result.solution_set.values, (mf.Rational(3),))
        self.assertEqual(result.method, "linear_isolation")

    def test_exact_square_root_two(self):
        result = self.solve(self.x**2 - 2)
        self.assertEqual(set(result.solution_set.values), {mf.sqrt(2), -mf.sqrt(2)})
        for root in result.solution_set:
            self.assertEqual(mf.evaluate(self.x**2 - 2, {self.x: root}), mf.Rational(0))
        checks = [c for c in result.verification if c.claim == "root.membership"]
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(c.level == mf.VerificationLevel.EXACT_SUBSTITUTION for c in checks))

    def test_negative_leading_coefficient(self):
        result = self.solve(-3 * self.x**2 + 6)
        self.assertEqual(set(result.solution_set.values), {mf.sqrt(2), -mf.sqrt(2)})

    def test_rational_coefficients_and_irrational_shifted_roots(self):
        result = self.solve(mf.Rational(1, 3) * self.x**2 + self.x - mf.Rational(1, 2))
        checks = [c for c in result.verification if c.claim == "root.membership"]
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(c.status == mf.CheckStatus.VERIFIED for c in checks))
        self.assertEqual(set(result.solution_set.values), {
            mf.Rational(-3, 2) - mf.sqrt(mf.Rational(15, 4)),
            mf.Rational(-3, 2) + mf.sqrt(mf.Rational(15, 4)),
        })

    def test_generated_known_rational_roots(self):
        # Factored inputs provide independently known roots. Vary signs, scales,
        # denominators, and repeated roots without a floating-point oracle.
        roots = [mf.Rational(-5, 3), mf.Rational(0), mf.Rational(2, 7), mf.Rational(4)]
        for scale in (mf.Rational(-3, 2), mf.Rational(1), mf.Rational(5, 7)):
            for first in roots:
                for second in roots:
                    with self.subTest(scale=scale, first=first, second=second):
                        result = self.solve(scale * (self.x - first) * (self.x - second))
                        self.assertEqual(set(result.solution_set.values), {first, second})

    def test_nonzero_right_hand_side(self):
        result = self.solve(self.x**2 + self.x, 6)
        self.assertEqual(set(result.solution_set.values), {mf.Rational(-3), mf.Rational(2)})

    def test_large_exact_coefficients(self):
        scale = 10**120 + 17
        result = self.solve(scale * self.x - (scale * 3))
        self.assertEqual(result.solution_set.values, (mf.Rational(3),))

    def test_unsupported_not_no_solution(self):
        result = mf.solve(mf.Eq(self.x**3 - mf.sqrt(2), 0), for_=self.x)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)
        self.assertEqual(result.outcome, mf.Outcome.NOT_APPLICABLE)
        self.assertIsNone(result.solution_set)
        self.assertNotEqual(result.outcome, mf.Outcome.NO_SOLUTION)

    def test_irrational_coefficient_unsupported(self):
        result = mf.solve(mf.Eq(self.x + mf.sqrt(2), 0), for_=self.x)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)

    def test_multivariate_unsupported(self):
        y = mf.symbol("y")
        result = mf.solve(mf.Eq(self.x + y, 0), for_=self.x)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)

    def test_nonreal_domain_unsupported(self):
        x = mf.symbol("x", domain=mf.Integers)
        result = mf.solve(mf.Eq(x - 2, 0), for_=x)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)

    def test_assumptions_never_ignored(self):
        for truth in mf.Truth:
            x = mf.symbol("x", assumptions=(mf.Assumption("positive", truth),))
            result = mf.solve(mf.Eq(x**2 - 1, 0), for_=x)
            self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)
            self.assertEqual(result.assumptions, x.assumptions)

    def test_invalid_input_separate_from_unsupported(self):
        for problem, variable in ((self.x, self.x), (mf.Eq(self.x, 0), "x")):
            result = mf.solve(problem, for_=variable)
            self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)
            self.assertIsNone(result.solution_set)

    def test_execution_failure_not_no_solution(self):
        with patch("mathforge.algorithms.solve_equation", side_effect=RuntimeError("test failure")):
            result = mf.solve(mf.Eq(self.x, 0), for_=self.x)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.ERROR)
        self.assertEqual(result.error.code, "execution_error")
        self.assertIsNone(result.solution_set)

    def test_membership_does_not_claim_completeness(self):
        result = self.solve(self.x**2 - 2)
        tampered = replace(result, value=mf.FiniteSet((mf.sqrt(2),)), steps=())
        report = mf.verify(tampered)
        self.assertFalse(report.verified)
        membership = [c for c in report.checks if c.claim == "root.membership"]
        completeness = [c for c in report.checks if c.claim == "solution.completeness"]
        self.assertTrue(all(c.status == mf.CheckStatus.VERIFIED for c in membership))
        self.assertEqual(completeness[0].status, mf.CheckStatus.FAILED)

    def test_wrong_root_is_rejected_despite_stored_verified_checks(self):
        result = self.solve(self.x**2 - 2)
        tampered = replace(result, value=mf.FiniteSet((mf.Rational(-1), mf.Rational(1))), steps=())
        report = mf.verify(tampered)
        self.assertFalse(report.verified)
        self.assertTrue(any(c.claim == "root.membership" and c.status == mf.CheckStatus.FAILED
                            for c in report.checks))

    def test_tampered_discriminant_step_is_rejected(self):
        result = self.solve(self.x**2 - 2)
        steps = tuple(replace(s, outputs=(mf.Rational(99),))
                      if s.rule == "polynomial.discriminant" else s for s in result.steps)
        report = mf.verify(replace(result, steps=steps))
        self.assertFalse(report.verified)
        self.assertTrue(any(c.claim == "step.polynomial.discriminant"
                            and c.status == mf.CheckStatus.FAILED for c in report.checks))

    def test_false_precondition_is_rejected(self):
        result = self.solve(self.x - 3)
        step = replace(result.steps[-1], preconditions=(
            mf.Condition(mf.Rational(0), "ne", mf.Rational(0), mf.Truth.TRUE),))
        self.assertEqual(verify_step(step).status, mf.CheckStatus.FAILED)

    def test_valid_step_from_another_problem_is_not_a_valid_trace(self):
        first = self.solve(self.x - 3)
        second = self.solve(self.x - 7)
        self.assertFalse(mf.verify(replace(first, steps=second.steps)).verified)

    def test_wrong_empty_or_universal_set_rejected(self):
        result = self.solve(self.x - 3)
        for value, outcome in ((mf.EmptySet(), mf.Outcome.NO_SOLUTION),
                               (mf.UniversalSet(), mf.Outcome.SOLUTIONS)):
            self.assertFalse(mf.verify(replace(result, value=value, outcome=outcome, steps=())).verified)

    def test_inconsistent_result_axes_are_rejected(self):
        result = self.solve(self.x - 3)
        invalid_contract_changes = (
            {"outcome": mf.Outcome.NO_SOLUTION},
            {"execution_status": mf.ExecutionStatus.ERROR},
            {"execution_status": mf.ExecutionStatus.UNSUPPORTED},
        )
        for change in invalid_contract_changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(result, **change)
        unsupported_claim_changes = (
            {"exactness": mf.Exactness.APPROXIMATE},
            {"completeness": mf.Completeness.UNKNOWN},
            {"precision": 16},
        )
        for change in unsupported_claim_changes:
            with self.subTest(change=change):
                self.assertFalse(mf.verify(replace(result, **change)).verified)

    def test_equivalent_radicals_do_not_forge_completeness(self):
        result = self.solve(self.x**2 - 2)
        # These are two structures but only one mathematical root.
        repeated = mf.FiniteSet((mf.sqrt(2), mf.sqrt(8) / 2))
        report = mf.verify(replace(result, value=repeated, steps=()))
        self.assertFalse(report.verified)
        self.assertEqual(next(c.status for c in report.checks
                              if c.claim == "solution.completeness"), mf.CheckStatus.FAILED)

    def test_equivalent_radical_representation_can_be_verified(self):
        result = self.solve(self.x**2 - 2)
        equivalent = mf.FiniteSet((mf.sqrt(8) / 2, -mf.sqrt(8) / 2))
        self.assertTrue(mf.verify(replace(result, value=equivalent, steps=())).verified)


class StructuredOperationsTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x")

    def test_derivative_and_evaluation(self):
        expression = mf.Rational(2, 3) * self.x**3 - 5 * self.x + 7
        derivative = mf.operations.differentiate(expression, self.x)
        self.assertEqual(derivative.execution_status, mf.ExecutionStatus.COMPLETED)
        self.assertEqual(mf.Polynomial.from_expression(derivative.value, self.x).coefficients,
                         (mf.Rational(-5), mf.Rational(0), mf.Rational(2)))
        for value in (-3, 0, 7):
            self.assertEqual(mf.evaluate(derivative.value, {self.x: value}), mf.Rational(2 * value**2 - 5))

    def test_substitution_and_evaluation_have_distinct_binding_rules(self):
        y = mf.symbol("y")
        expression = self.x + y
        partial = mf.operations.substitute(expression, {self.x: 2})
        self.assertEqual(partial.execution_status, mf.ExecutionStatus.COMPLETED)
        self.assertEqual(partial.value, 2 + y)
        evaluation = mf.operations.evaluate(expression, {self.x: 2})
        self.assertEqual(evaluation.execution_status, mf.ExecutionStatus.INVALID_INPUT)

    def test_convenience_failure_retains_result(self):
        with self.assertRaises(mf.OperationError) as captured:
            mf.evaluate(self.x)
        self.assertEqual(captured.exception.result.execution_status, mf.ExecutionStatus.INVALID_INPUT)

    def test_no_approximate_coercion(self):
        for value in (0.1, True):
            self.assertEqual(mf.operations.simplify(value).execution_status,
                             mf.ExecutionStatus.INVALID_INPUT)
            self.assertEqual(mf.operations.evaluate(self.x, {self.x: value}).execution_status,
                             mf.ExecutionStatus.INVALID_INPUT)

    def test_unknown_verification_is_not_a_proof(self):
        result = mf.operations.differentiate(self.x**2, self.x)
        report = mf.verify(replace(result, request=None))
        self.assertFalse(report.verified)
        self.assertEqual(report.checks[0].status, mf.CheckStatus.UNKNOWN)

    def test_non_mapping_bindings_are_invalid(self):
        for operation in (mf.operations.evaluate, mf.operations.substitute):
            result = operation(self.x, [(self.x, 2)])
            self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)

    def test_derivative_domain_is_explicit(self):
        for domain in (mf.Integers, mf.Rationals):
            x = mf.symbol("x", domain=domain)
            result = mf.operations.differentiate(x**2, x)
            self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
