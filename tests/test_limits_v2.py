"""Finite budgets remain explicit failures, never partial mathematical claims."""

from dataclasses import fields
import unittest
from unittest import mock

import mathforge as m
from mathforge import contracts as c
from mathforge import operations as op
from mathforge.limits import ComputationLimits, DecodeLimits, ResourceLimitError, computation
from mathforge.matrices import Matrix


class ComputationLimitTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000030")

    def test_all_limits_require_positive_exact_integers(self):
        """LIM-01: bool is not a computation or decoding budget."""
        for cls in (ComputationLimits, DecodeLimits):
            for field in fields(cls):
                for invalid in (0, -1, True, 1.5):
                    with self.subTest(cls=cls.__name__, field=field.name, invalid=invalid):
                        with self.assertRaises((TypeError, ValueError)):
                            cls(**{field.name: invalid})

    def test_nested_computations_reuse_budget_and_fresh_is_explicit(self):
        """LIM-04/LIM-10: nested arithmetic cannot silently reset the counter."""
        with computation(ComputationLimits(max_nodes=100), fresh=True) as outer:
            outer.inspect((m.Rational(1),))
            previous = outer.node_count
            with computation() as inner:
                self.assertIs(inner, outer)
                inner.inspect((m.Rational(2),))
            self.assertGreater(outer.node_count, previous)
            with computation(fresh=True) as separate:
                self.assertIsNot(separate, outer)
                self.assertEqual(separate.node_count, 0)

    def test_root_operation_limit_is_error_without_a_solution(self):
        """LIM-02: budget exhaustion does not mean that an equation is rootless."""
        result = op.solve(m.Eq(self.x**5 - self.x - 1, 0), for_=self.x,
                          limits=ComputationLimits(max_work=1))
        self.assertIs(result.execution_status, c.ExecutionStatus.ERROR)
        self.assertEqual(result.error.code, "resource_limit_exceeded")
        self.assertIsNone(result.value)
        self.assertIsNone(result.solution_set)
        self.assertIs(result.completeness, c.Completeness.NOT_APPLICABLE)

    def test_explicit_integer_bit_limit_is_bounded(self):
        """LIM-03: a valid large exact input never silently becomes approximate."""
        expression = (2**40) * self.x**3
        result = op.differentiate(expression, self.x,
                                  limits=ComputationLimits(max_integer_bits=16))
        self.assertIs(result.execution_status, c.ExecutionStatus.ERROR)
        self.assertEqual(result.error.code, "resource_limit_exceeded")
        self.assertIsNone(result.value)

    def test_matrix_constructor_honors_ambient_entry_budget(self):
        """LIM-06: matrix allocation inside an operation shares its entry limit."""
        with computation(ComputationLimits(max_matrix_entries=1), fresh=True):
            with self.assertRaises(ResourceLimitError):
                Matrix(((1, 2), (3, 4)))

    def test_verification_budget_is_unknown(self):
        """VER-15/LIM-10: exhausted checking is not a mathematical counterexample."""
        result = op.differentiate(self.x**3, self.x)
        self.assertIs(result.execution_status, c.ExecutionStatus.COMPLETED)
        report = m.verify(result, limits=ComputationLimits(max_nodes=1))
        self.assertFalse(report.verified)
        self.assertTrue(any(check.status is c.CheckStatus.UNKNOWN for check in report.checks))

    def test_finite_increased_limits_preserve_success(self):
        """LIM-09: explicit larger limits are local and supported."""
        limits = ComputationLimits(max_work=4_000_000, max_integer_bits=131_072)
        result = op.solve(m.Eq(self.x**3, 2), for_=self.x, limits=limits)
        self.assertIs(result.execution_status, c.ExecutionStatus.COMPLETED, result)
        self.assertTrue(m.verify(result, limits=limits).verified)
        self.assertEqual(ComputationLimits().max_work, 2_000_000)

    def test_unrelated_comparisons_do_not_change_canonical_results(self):
        """LIM-12: cache population order cannot affect persisted root evidence."""
        expression = self.x**3 - 2
        first = op.real_roots(expression, self.x)
        self.assertIs(first.execution_status, c.ExecutionStatus.COMPLETED, first)
        for left, right in ((m.sqrt(3), m.sqrt(2)), (m.Rational(2), m.sqrt(3)),
                            (m.sqrt(2), m.Rational(1))):
            result = op.compare_real(left, right)
            self.assertIs(result.execution_status, c.ExecutionStatus.COMPLETED, result)
        second = op.real_roots(expression, self.x)
        self.assertEqual(m.dumps(first), m.dumps(second))

    def test_process_interrupts_are_not_converted_to_results(self):
        """LIM-11: process control exceptions cross the actual operation boundary."""
        for interruption in (KeyboardInterrupt, SystemExit):
            with self.subTest(interruption=interruption.__name__):
                with mock.patch("mathforge.calculus.derivative", side_effect=interruption):
                    with self.assertRaises(interruption):
                        op.differentiate(self.x**2, self.x)


if __name__ == "__main__":
    unittest.main()
