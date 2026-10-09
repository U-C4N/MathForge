"""Failure boundaries and compatibility at the public operation layer."""
from unittest import TestCase, mock
import mathforge as m


class SDKV2Tests(TestCase):
    def test_scalar_and_system_failures_keep_safe_context(self):
        """REL-02: input/unsupported/execution failures do not become solutions."""
        x = m.symbol("x")
        problem = m.Eq(x**3, m.sqrt(2))
        result = m.solve(problem, for_=x)
        self.assertIs(result.execution_status, m.ExecutionStatus.UNSUPPORTED)
        self.assertEqual(result.problem, problem)
        self.assertEqual(result.request.get("problem"), problem)
        self.assertIsNone(result.solution_set)
        system = m.LinearSystem(m.Matrix(((1,),)), (1,), (x,))
        with mock.patch("mathforge.matrices.solve_linear_system", side_effect=RuntimeError("injected")):
            result = m.solve_system(system)
        self.assertIs(result.execution_status, m.ExecutionStatus.ERROR)
        self.assertEqual(result.problem, system)
        self.assertEqual(result.domain, m.Reals)
        self.assertEqual(result.request.get("system_or_equations"), system)
        self.assertIsNone(result.solution_set)

    def test_old_unsupported_case_is_now_supported_and_still_distinct_from_empty(self):
        """REL-03: cubic support extends behavior without hiding unsupported input."""
        x = m.symbol("x")
        cubic = m.solve(m.Eq(x**3, 2), for_=x)
        empty = m.solve(m.Eq(x**2, -1), for_=x)
        unsupported = m.solve(m.Eq(x**3, m.sqrt(2)), for_=x)
        self.assertIs(cubic.execution_status, m.ExecutionStatus.COMPLETED)
        self.assertTrue(m.verify(cubic).verified)
        self.assertIs(empty.outcome, m.Outcome.NO_SOLUTION)
        self.assertIs(unsupported.execution_status, m.ExecutionStatus.UNSUPPORTED)

    def test_foreign_or_inexact_arguments_do_not_break_the_error_path(self):
        """REL-16: malformed inputs return typed failures without an invented request."""
        x = m.symbol("x")
        class Foreign:
            def __init__(self):
                self.mutable = []
        for value in (0.5, Foreign()):
            for produce in (lambda: m.operations.evaluate(value),
                            lambda: m.operations.differentiate(value, x),
                            lambda: m.operations.rref(value),
                            lambda: m.solve(value, for_=x)):
                result = produce()
                self.assertIs(result.execution_status, m.ExecutionStatus.INVALID_INPUT)
                self.assertIsNone(result.request)
                self.assertIsNone(result.value)
                self.assertIsNone(result.solution_set)
                self.assertEqual(m.loads(m.dumps(result)), result)

    def test_failed_independent_gate_cannot_publish_a_complete_answer(self):
        """VER-02: producer success is insufficient when the independent check fails."""
        x = m.symbol("x")
        failed = m.VerificationReport((m.Check("injected", m.CheckStatus.FAILED,
                    m.VerificationLevel.ALGORITHMIC_CHECK, "test"),))
        with mock.patch("mathforge.verification.verify", return_value=failed):
            result = m.operations.differentiate(x**2, x)
        self.assertIs(result.execution_status, m.ExecutionStatus.ERROR)
        self.assertIsNone(result.value)
        self.assertIs(result.completeness, m.Completeness.NOT_APPLICABLE)
