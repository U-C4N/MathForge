"""Exact-output and deterministic-work regressions for v0.5 optimizations."""

import unittest
from unittest.mock import patch

from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.matrices import Matrix, reduce_with_trace, replay_row_operations, verify_row_reduction
from mathforge.model import Rational, Symbol
from mathforge.polynomial import Polynomial
from mathforge.root_isolation import count_real_roots, isolate_real_roots


class ExactPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.matrix = Matrix(((0, 0, 2, 4, 6), (0, 3, 6, 6, 0), (0, 0, 0, 5, 10)))
        self.x = Symbol("x", uid="00000000-0000-4000-8000-000000000050")
        self.polynomial = Polynomial((0, 4, 0, -5, 0, 1), self.x)

    def test_rref_preserves_prefixes_and_the_original_row_certificate(self):
        with computation(fresh=True) as budget:
            reduced, trace = reduce_with_trace(self.matrix)
            work = budget.work
        self.assertEqual(reduced.pivot_columns, (1, 2, 3))
        self.assertEqual(reduced.matrix, Matrix(((0, 1, 0, 0, -2),
                                                (0, 0, 1, 0, -1),
                                                (0, 0, 0, 1, 2))))
        self.assertEqual(trace, (("swap", 0, 1),
                                 ("scale", 0, Rational(1, 3)),
                                 ("scale", 1, Rational(1, 2)),
                                 ("add", 0, 1, Rational(-2)),
                                 ("scale", 2, Rational(1, 5)),
                                 ("add", 0, 2, Rational(2)),
                                 ("add", 1, 2, Rational(-2))))
        # The pre-optimization algorithm consumed 426 work units on this input.
        self.assertLessEqual(work, 300)
        self.assertEqual(replay_row_operations(self.matrix, trace), reduced.matrix)
        with patch("mathforge.matrices.reduce_with_trace", side_effect=AssertionError("producer called")):
            self.assertTrue(verify_row_reduction(self.matrix, reduced, trace))

    def test_root_subdivision_preserves_isolators_with_less_work(self):
        with computation(fresh=True) as budget:
            intervals = isolate_real_roots(self.polynomial)
            work = budget.work
        self.assertEqual(intervals, ((Rational(-7, 2), Rational(-7, 4)),
                                     (Rational(-7, 4), Rational(-7, 8)),
                                     (Rational(0), Rational(0)),
                                     (Rational(7, 8), Rational(7, 4)),
                                     (Rational(7, 4), Rational(7, 2))))
        # Frozen local input: 16,751 work units before endpoint state reuse.
        self.assertLessEqual(work, 6_000)
        with patch("mathforge.root_isolation.isolate_real_roots",
                   side_effect=AssertionError("producer called")):
            self.assertEqual(count_real_roots(self.polynomial), len(intervals))
            for lower, upper in intervals:
                point = lower == upper
                self.assertEqual(count_real_roots(self.polynomial, lower, upper,
                                                 include_lower=point, include_upper=point), 1)

    def test_midpoint_roots_are_emitted_once_even_with_multiplicity(self):
        polynomial = Polynomial((0, 4, -4, 1), self.x)  # x * (x - 2)^2
        self.assertEqual(isolate_real_roots(polynomial),
                         ((Rational(0), Rational(0)), (Rational(2), Rational(2))))

    def test_work_and_refinement_failures_remain_local(self):
        with self.assertRaises(ResourceLimitError):
            reduce_with_trace(self.matrix, limits=ComputationLimits(max_work=200))
        for limits in (ComputationLimits(max_work=4_000), ComputationLimits(max_refinements=10)):
            with self.subTest(limits=limits), self.assertRaises(ResourceLimitError):
                isolate_real_roots(self.polynomial, limits=limits)
        # A failed call cannot populate a cache that bypasses the next budget.
        for _ in range(2):
            with self.assertRaises(ResourceLimitError):
                isolate_real_roots(self.polynomial, limits=ComputationLimits(max_work=4_000))
        self.assertEqual(len(isolate_real_roots(self.polynomial,
                                              limits=ComputationLimits(max_work=6_000))), 5)


if __name__ == "__main__":
    unittest.main()
