"""P05/P06 partial-function domains and complete exact real solution sets."""

from dataclasses import FrozenInstanceError, replace
import random
import unittest

import mathforge as mf
from mathforge.errors import MathForgeError
from mathforge.limits import computation


class RationalFunctionV2Tests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x")

    def rf(self, numerator, denominator):
        return mf.rational_function(numerator, denominator, variable=self.x)

    def assertUndefined(self, function, value):
        result = mf.operations.evaluate(function, {function.variable: value})
        self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)
        self.assertEqual(result.error.code, "undefined_at_point")

    def test_cancellation_preserves_hole_and_exact_value(self):
        """RF-01/02/04: normalize value but preserve original denominator."""
        f = self.rf(self.x**2-1, self.x-1)
        self.assertEqual(f.numerator, mf.Polynomial((1, 1), self.x))
        self.assertEqual(f.denominator, mf.Polynomial((1,), self.x))
        self.assertUndefined(f, 1)
        self.assertEqual(mf.evaluate(f, {self.x: 2}), mf.Rational(3))
        negative = self.rf(2*self.x, -2*self.x-2)
        self.assertEqual(negative.denominator.leading_coefficient, mf.Rational(1))
        self.assertEqual(mf.evaluate(negative, {self.x: 2}), mf.Rational(-2, 3))

    def test_zero_one_cancellation_and_powers_keep_domain(self):
        """RF-05: total-AST neutral-element rewrites must not erase holes."""
        f = self.rf(1, self.x)
        for value in (0*f, f*0, f-f, f**0, f/f, mf.simplify(f-f)):
            with self.subTest(value=str(value)):
                self.assertIsInstance(value, mf.RationalFunction)
                self.assertUndefined(value, 0)
        self.assertEqual(mf.evaluate(f**0, {self.x: 2}), mf.Rational(1))
        self.assertEqual(mf.evaluate(0*f, {self.x: 2}), mf.Rational(0))

    def test_binary_domains_intersect_and_divisor_zeros_are_excluded(self):
        """RF-06/07: both source domains and divisor numerator survive."""
        f, g = self.rf(1, self.x), self.rf(self.x-2, self.x+1)
        result = f/g
        for hole in (-1, 0, 2):
            self.assertUndefined(result, hole)
        self.assertEqual(mf.evaluate(result, {self.x: 3}), mf.Rational(4, 3))
        zero = self.rf(0, self.x+1)*f
        self.assertUndefined(zero, -1)
        self.assertUndefined(zero, 0)
        self.assertEqual(mf.evaluate(zero, {self.x: 2}), mf.Rational(0))

    def test_mixed_operand_orders(self):
        """RF-08/09: Expression and Polynomial reflected operations work."""
        f = self.rf(1, self.x)
        p = mf.Polynomial((0, 1), self.x)
        pairs = ((f+self.x, self.x+f), (f*p, p*f), (f+2, 2+f), (f*2, 2*f))
        for left, right in pairs:
            self.assertEqual(mf.evaluate(left, {self.x: 2}), mf.evaluate(right, {self.x: 2}))
            self.assertUndefined(left, 0)
            self.assertUndefined(right, 0)
        self.assertEqual(mf.evaluate(self.x-f, {self.x: 2}), mf.Rational(3, 2))
        self.assertEqual(mf.evaluate(self.x/f, {self.x: 2}), mf.Rational(4))
        self.assertEqual(mf.evaluate(p/f, {self.x: 2}), mf.Rational(4))
        with self.assertRaises(MathForgeError):
            self.x/(self.x-1)

    def test_invalid_division_and_variable_identity(self):
        """RF-03/10: no empty-solution claim for invalid RF arithmetic."""
        for numerator, denominator in ((1, 0), (0, 0)):
            with self.assertRaises(MathForgeError):
                self.rf(numerator, denominator)
        with self.assertRaises(MathForgeError):
            self.rf(1, self.x)/self.rf(0, self.x+1)
        other = mf.symbol("x")
        with self.assertRaises(MathForgeError):
            self.rf(1, self.x)+mf.rational_function(1, other, variable=other)
        with self.assertRaises(MathForgeError):
            self.rf(1, self.x)**-1

    def test_substitution_rename_partial_and_exact_binding(self):
        """RF-11–14: partial substitution differs from complete evaluation."""
        f = self.rf(self.x**2-1, self.x-1)
        y = mf.symbol("y")
        self.assertEqual(mf.substitute(f, {}), f)
        self.assertEqual(mf.substitute(f, {self.x: 2}), mf.Rational(3))
        renamed = mf.substitute(f, {self.x: y})
        self.assertEqual(renamed.variable, y)
        self.assertTrue(all(p.variable == y for p in renamed.excluded))
        self.assertUndefined(renamed, 1)
        for bindings in ({}, {self.x: 0.5}, {self.x: True}):
            result = mf.operations.evaluate(f, bindings)
            self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)
        for value in (self.x+1, mf.sqrt(2)):
            self.assertEqual(mf.operations.substitute(f, {self.x: value}).execution_status,
                             mf.ExecutionStatus.UNSUPPORTED)
        self.assertEqual(mf.operations.evaluate(f, {self.x: mf.sqrt(2)}).execution_status,
                         mf.ExecutionStatus.UNSUPPORTED)

    def test_constructor_domain_and_assumption_scope(self):
        """RF-25: RF does not silently widen a restricted variable domain."""
        variables = (mf.symbol("n", domain=mf.Integers), mf.symbol("q", domain=mf.Rationals),
                     mf.symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),)))
        for variable in variables:
            result = mf.operations.rational_function(1, variable, variable=variable)
            self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)

    def test_rational_equation_filters_cancelled_roots(self):
        """RF-15/16/17: candidate filtering and identity domains are complete."""
        f = self.rf(self.x**2-1, self.x-1)
        empty = mf.solve(mf.Eq(f, 2), for_=self.x)
        self.assertIsInstance(empty.value, mf.EmptySet)
        self.assertTrue(mf.verify(empty).verified)
        single = mf.solve(mf.Eq(f, 3), for_=self.x)
        self.assertEqual(single.value, mf.FiniteSet((mf.Rational(2),)))
        self.assertTrue(mf.verify(single).verified)
        identity = mf.solve(mf.Eq(self.rf(self.x, self.x), 1), for_=self.x)
        self.assertIsInstance(identity.value, mf.IntervalSet)
        self.assertFalse(identity.value.contains(0))
        self.assertTrue(identity.value.contains(-1))
        self.assertTrue(identity.value.contains(1))
        self.assertTrue(mf.verify(identity).verified)

    def test_semantic_domains_with_redundant_or_nonreal_factors(self):
        """RF-18/19/24: domain equality concerns real zero unions."""
        x = self.x
        f = self.rf(x, x)
        g = self.rf(x*(x*x+1), x*(x*x+1))
        left = mf.solve(mf.Eq(f, 1), for_=x)
        right = mf.solve(mf.Eq(g, 1), for_=x)
        self.assertEqual(left.value, right.value)
        # Different structural guard presentations encode the same real domain.
        self.assertTrue(mf.verify(replace(left, value=right.value)).verified)
        separate = self.rf(x, x)*self.rf(x-1, x-1)
        together = self.rf(x*(x-1), x*(x-1))
        a = mf.solve(mf.Eq(separate, together), for_=x)
        self.assertFalse(a.value.contains(0))
        self.assertFalse(a.value.contains(1))
        self.assertTrue(a.value.contains(mf.Rational(1, 2)))
        self.assertTrue(mf.verify(a).verified)

    def test_erased_hole_and_wrong_value_fail_verification(self):
        """RF-22/VER-12: equal values do not excuse domain loss."""
        f = self.rf(self.x**2-1, self.x-1)
        result = mf.operations.simplify(f)
        filled_hole = self.rf(self.x+1, 1)
        self.assertFalse(mf.verify(replace(result, value=filled_hole)).verified)
        self.assertFalse(mf.verify(replace(result, value=f+1)).verified)

    def test_shared_budget_and_immutable_guard_records(self):
        """RF-23/LIM-04: RF construction and derived work share finite limits."""
        result = mf.operations.rational_function(self.x**5-self.x-1, self.x**4-1,
                                                 variable=self.x, limits=mf.ComputationLimits(max_work=2))
        self.assertEqual(result.execution_status, mf.ExecutionStatus.ERROR)
        self.assertEqual(result.error.code, "resource_limit_exceeded")
        f = self.rf(1, self.x)
        with self.assertRaises(FrozenInstanceError):
            f.excluded = ()

    def test_seeded_cancelled_domains_and_exact_evaluation(self):
        """RF-01/02: 100 known cancelled linear factors with independent values."""
        rng = random.Random(2026100905)
        for case in range(100):
            hole, root, sample = rng.sample(range(-20, 21), 3)
            f = self.rf((self.x-hole)*(self.x-root), self.x-hole)
            with self.subTest(case=case, hole=hole, root=root):
                self.assertUndefined(f, hole)
                self.assertEqual(mf.evaluate(f, {self.x: sample}), mf.Rational(sample-root))

    def test_zero_exclusion_is_rejected(self):
        """RF-20: a zero guard cannot silently create an empty-domain RF."""
        one, zero = mf.Polynomial((1,), self.x), mf.Polynomial((0,), self.x)
        with self.assertRaises(MathForgeError):
            mf.RationalFunction(one, one, (zero,))

    def test_many_independent_guards_do_not_require_a_large_product(self):
        """RF-23: guard union degree may exceed the individual Polynomial cap."""
        guards = tuple(mf.Polynomial((-index, 1), self.x) for index in reversed(range(1030)))
        zero, one = mf.Polynomial((0,), self.x), mf.Polynomial((1,), self.x)
        with computation(mf.ComputationLimits(max_work=5000000, max_nodes=5000000)):
            f = mf.RationalFunction(zero, one, guards)
            self.assertEqual(len(f.excluded), 1030)
            self.assertTrue(f.numerator.is_zero)
            self.assertEqual(f.evaluate(-1), mf.Rational(0))
            with self.assertRaises(MathForgeError) as captured:
                f.evaluate(1029)
            self.assertEqual(captured.exception.code, "undefined_at_point")


class IntervalAndInequalityV2Tests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x")

    def rf(self, numerator, denominator):
        return mf.rational_function(numerator, denominator, variable=self.x)

    def assertIntervals(self, value, expected):
        intervals = mf.IntervalSet.from_solution_set(value).intervals
        self.assertEqual(len(intervals), len(expected))
        for actual, (lower, upper, left_closed, right_closed) in zip(intervals, expected):
            if lower is None:
                self.assertIsNone(actual.lower)
            else:
                self.assertEqual(mf.compare_real(actual.lower, lower), 0)
            if upper is None:
                self.assertIsNone(actual.upper)
            else:
                self.assertEqual(mf.compare_real(actual.upper, upper), 0)
            self.assertEqual((actual.left_closed, actual.right_closed), (left_closed, right_closed))

    def solve(self, expression, relation):
        result = mf.solve(mf.Inequality(expression, relation, 0), for_=self.x)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.COMPLETED, result.error)
        self.assertTrue(mf.verify(result).verified)
        return result

    def test_polynomial_weak_and_singleton_solutions(self):
        """SET-01–03/07: repeated zeros need not change sign."""
        self.assertIntervals(self.solve(self.x**2-4, "ge").value,
                             ((None, -2, False, True), (2, None, True, False)))
        self.assertIntervals(self.solve((self.x-1)**2, "le").value, ((1, 1, True, True),))
        self.assertIntervals(self.solve((self.x**2-2)**2, "le").value,
                             ((-mf.sqrt(2), -mf.sqrt(2), True, True),
                              (mf.sqrt(2), mf.sqrt(2), True, True)))

    def test_rational_sign_chart_poles_and_cancelled_holes(self):
        """SET-04–06/10/18: denominator signs and holes determine cells."""
        self.assertIntervals(self.solve(self.rf(1, self.x), "lt").value,
                             ((None, 0, False, False),))
        self.assertIntervals(self.solve(self.rf(1, self.x**2), "gt").value,
                             ((None, 0, False, False), (0, None, False, False)))
        self.assertIntervals(self.solve(self.rf(self.x**2-4, self.x-1), "ge").value,
                             ((-2, 1, True, False), (2, None, True, False)))
        f = self.rf(self.x**2-1, self.x-1)
        self.assertFalse(mf.IntervalSet.from_solution_set(self.solve(f, "ge").value).contains(1))

    def test_zero_and_constant_functions(self):
        """SET-08/09: zero root enumeration must not be called for identities."""
        zero = self.rf(0, self.x)
        for relation in ("lt", "gt"):
            self.assertIsInstance(self.solve(zero, relation).value, mf.EmptySet)
        for relation in ("le", "ge"):
            self.assertIntervals(self.solve(zero, relation).value,
                                 ((None, 0, False, False), (0, None, False, False)))
        self.assertIsInstance(self.solve(self.x**2+1, "gt").value, mf.UniversalSet)
        self.assertIsInstance(self.solve(self.x**2+1, "lt").value, mf.EmptySet)

    def test_interval_touch_union_and_intersection(self):
        """SET-12/13/16: endpoint membership governs union and intersection."""
        left = mf.IntervalSet((mf.Interval(0, 1),))
        right = mf.IntervalSet((mf.Interval(1, 2),))
        self.assertIntervals(left.union(right), ((0, 1, False, False), (1, 2, False, False)))
        closed = mf.IntervalSet((mf.Interval(0, 1, right_closed=True),))
        self.assertIntervals(closed.union(right), ((0, 2, False, False),))
        other = mf.IntervalSet((mf.Interval(1, 2, left_closed=True),))
        self.assertIntervals(closed.intersection(other), ((1, 1, True, True),))
        self.assertIntervals(left.intersection(other), ())

    def test_degenerate_invalid_and_immutable_intervals(self):
        """SET-14/15: empty intervals normalize away, invalid bounds fail."""
        for flags in ((False, False), (False, True), (True, False)):
            value = mf.IntervalSet((mf.Interval(1, 1, *flags),))
            self.assertIntervals(value, ())
        self.assertIntervals(mf.IntervalSet((mf.Interval(1, 1, True, True),)), ((1, 1, True, True),))
        for args in ((2, 1, False, False), (None, 1, True, False), (0, None, False, True)):
            with self.assertRaises(MathForgeError):
                mf.Interval(*args)
        interval = mf.Interval(0, 1)
        with self.assertRaises(FrozenInstanceError):
            interval.lower = mf.Rational(-1)
        source = [interval]
        value = mf.IntervalSet(source)
        source.append(mf.Interval(2, 3))
        self.assertEqual(len(value.intervals), 1)

    def test_equivalent_algebraic_boundary_merge(self):
        """SET-11: endpoints with different descriptors represent one point."""
        a = mf.RealAlgebraicRoot((-2, 0, 1), 1)
        b = mf.RealAlgebraicRoot((6, -2, -3, 1), 1)
        value = mf.IntervalSet((mf.Interval(0, a, right_closed=True), mf.Interval(b, 2)))
        self.assertIntervals(value, ((0, 2, False, False),))

    def test_empty_and_universal_solution_bridge(self):
        """SET-19/20: all scalar solution categories support set combination."""
        empty, universal = mf.EmptySet(), mf.UniversalSet()
        finite_intervals = mf.IntervalSet((mf.Interval(0, 1, True, True),))
        self.assertIntervals(mf.IntervalSet.from_solution_set(empty), ())
        self.assertIntervals(mf.IntervalSet.from_solution_set(universal), ((None, None, False, False),))
        self.assertEqual(finite_intervals.intersection(universal), finite_intervals)
        self.assertEqual(finite_intervals.union(empty), finite_intervals)
        self.assertIntervals(finite_intervals.intersection(empty), ())
        self.assertIntervals(finite_intervals.union(universal), ((None, None, False, False),))
        for value in (mf.EmptySet(mf.Integers), mf.FiniteSet((mf.Rational(1),))):
            with self.assertRaises(MathForgeError):
                mf.IntervalSet.from_solution_set(value)

    def test_deleted_cell_and_wrong_endpoint_fail_completeness(self):
        """SET-17/VER-11: exact samples alone cannot certify the whole answer."""
        result = self.solve(self.x**2-4, "ge")
        missing = mf.IntervalSet((mf.Interval(2, None, True, False),))
        open_endpoint = mf.IntervalSet((mf.Interval(None, -2), mf.Interval(2, None, True, False)))
        for value in (missing, open_endpoint):
            self.assertFalse(mf.verify(replace(result, value=value)).verified)


if __name__ == "__main__":
    unittest.main()
