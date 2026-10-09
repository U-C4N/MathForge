"""P07 exact polynomial calculus and domain-preserving RF derivatives."""

from dataclasses import replace
import random
import unittest

import mathforge as mf


class CalculusV2Tests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x")

    def coefficients(self, expression):
        return mf.Polynomial.from_expression(expression, self.x).coefficients

    def test_derivative_zero_order_higher_order_and_input_validation(self):
        """CAL-01–03: n=0 identity, high-order zero, exact nonnegative n."""
        expression = (self.x+1)**3
        self.assertEqual(mf.diff(expression, self.x, n=0), expression)
        self.assertEqual(self.coefficients(mf.diff(expression, self.x, n=2)),
                         (mf.Rational(6), mf.Rational(6)))
        self.assertEqual(mf.diff(expression, self.x, n=1000000), mf.Rational(0))
        for order in (-1, True, 1.5):
            result = mf.operations.differentiate(expression, self.x, n=order)
            self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)

    def test_antiderivative_zero_constant_and_polynomial(self):
        """CAL-04/06: choose the unique zero-constant antiderivative."""
        expression = 3*self.x**2+2*self.x+1
        result = mf.operations.antiderivative(expression, self.x)
        self.assertEqual(self.coefficients(result.value), tuple(mf.Rational(v) for v in (0, 1, 1, 1)))
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(mf.antiderivative(0, self.x), mf.Rational(0))
        self.assertEqual(mf.antiderivative(7, self.x), 7*self.x)

    def test_antiderivative_constant_tampering(self):
        """CAL-05: derivative equality alone does not establish the chosen C."""
        result = mf.operations.antiderivative(3*self.x**2, self.x)
        self.assertFalse(mf.verify(replace(result, value=result.value+5)).verified)

    def test_definite_integral_exact_reversed_equal_and_degree_boundary(self):
        """CAL-07–09/15: degree-1024 definite integration needs no larger Polynomial."""
        expression = 3*self.x**2+2*self.x+1
        result = mf.operations.integrate(expression, self.x, 0, 1)
        self.assertEqual(result.value, mf.Rational(3))
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(mf.integrate(expression, self.x, 1, 0), mf.Rational(-3))
        self.assertEqual(mf.integrate(expression, self.x, 2, 2), mf.Rational(0))
        self.assertEqual(mf.integrate(self.x**1024, self.x, 0, 1), mf.Rational(1, 1025))
        large = mf.operations.antiderivative(self.x**1024, self.x)
        self.assertEqual(large.execution_status, mf.ExecutionStatus.UNSUPPORTED)

    def test_integral_scope_precedes_equal_bound_shortcut(self):
        """CAL-10: equal endpoints cannot conceal an unsupported integrand."""
        rf = mf.rational_function(self.x, 1, variable=self.x)
        self.assertEqual(mf.operations.integrate(rf, self.x, 1, 1).execution_status,
                         mf.ExecutionStatus.UNSUPPORTED)
        for lower, upper in ((0.0, 1), (0, True), (None, 1)):
            result = mf.operations.integrate(self.x, self.x, lower, upper)
            self.assertIn(result.execution_status, (mf.ExecutionStatus.INVALID_INPUT, mf.ExecutionStatus.UNSUPPORTED))
        for variable in (mf.symbol("n", domain=mf.Integers),
                         mf.symbol("q", domain=mf.Rationals),
                         mf.symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))):
            result = mf.operations.integrate(variable, variable, 0, 0)
            self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)

    def test_rational_derivative_preserves_cancelled_and_zero_holes(self):
        """CAL-11/12: derivative value simplification retains every hole."""
        f = mf.rational_function(self.x**2-1, self.x-1, variable=self.x)
        result = mf.operations.differentiate(f, self.x)
        self.assertIsInstance(result.value, mf.RationalFunction)
        self.assertEqual(mf.evaluate(result.value, {self.x: 2}), mf.Rational(1))
        hole = mf.operations.evaluate(result.value, {self.x: 1})
        self.assertEqual(hole.error.code, "undefined_at_point")
        self.assertTrue(mf.verify(result).verified)
        zero = mf.rational_function(0, self.x, variable=self.x)
        for order in (0, 1, 2):
            derivative = mf.diff(zero, self.x, n=order)
            self.assertEqual(mf.evaluate(derivative, {self.x: 2}), mf.Rational(0))
            self.assertEqual(mf.operations.evaluate(derivative, {self.x: 0}).error.code,
                             "undefined_at_point")

    def test_second_rational_derivative_and_filled_hole_tampering(self):
        """CAL-11/13: repeated quotient rule and domain-aware verification."""
        f = mf.rational_function(1, self.x, variable=self.x)
        result = mf.operations.differentiate(f, self.x, n=2)
        self.assertEqual(mf.evaluate(result.value, {self.x: 2}), mf.Rational(1, 4))
        self.assertTrue(mf.verify(result).verified)
        cancelled = mf.rational_function(self.x**2-1, self.x-1, variable=self.x)
        result = mf.operations.differentiate(cancelled, self.x)
        filled = mf.rational_function(1, 1, variable=self.x)
        self.assertFalse(mf.verify(replace(result, value=filled)).verified)

    def test_tampered_derivative_and_integral_result(self):
        """CAL-13/16: successful metadata cannot authenticate wrong values."""
        derivative = mf.operations.differentiate(self.x**3, self.x, n=2)
        self.assertFalse(mf.verify(replace(derivative, value=derivative.value+1)).verified)
        integral = mf.operations.integrate(self.x**2, self.x, 0, 2)
        self.assertFalse(mf.verify(replace(integral, value=mf.Rational(99))).verified)
        arguments = dict(derivative.request.arguments)
        self.assertIn("n", arguments)
        arguments["n"] = 1
        request = replace(derivative.request, arguments=tuple(sorted(arguments.items())))
        self.assertFalse(mf.verify(replace(derivative, request=request)).verified)

    def test_integral_contribution_certificate_tampering(self):
        """CAL-16: every source coefficient needs its own correct contribution."""
        result = mf.operations.integrate(3*self.x**2+2*self.x+1, self.x, 0, 1)
        step = result.steps[0]
        self.assertEqual(step.rule, "v2.integrate")
        value, contributions = step.outputs
        self.assertEqual(len(contributions), 3)
        omitted = replace(step, outputs=(value, contributions[:-1]))
        forged = replace(step, outputs=(value, (contributions[0]+1,)+contributions[1:]))
        for tampered in (omitted, forged):
            self.assertFalse(mf.verify(replace(result, steps=(tampered,))).verified)

    def test_calculus_limits_and_operation_convenience_agreement(self):
        """LIM-04/REL-01: nested calculus shares budget and structured behavior."""
        expression = (self.x+1)**8
        result = mf.operations.integrate(expression, self.x, 0, 1, limits=mf.ComputationLimits(max_work=2))
        self.assertEqual(result.execution_status, mf.ExecutionStatus.ERROR)
        self.assertEqual(result.error.code, "resource_limit_exceeded")
        structured = mf.operations.differentiate(expression, self.x, n=2)
        self.assertEqual(mf.diff(expression, self.x, n=2), structured.value)

    def test_seeded_coefficient_antiderivatives_and_integral_additivity(self):
        """CAL-14: 100 exact independent coefficient and interval identities."""
        rng = random.Random(2026100907)
        for case in range(100):
            coefficients = tuple(mf.Rational(rng.randrange(-7, 8), rng.randrange(1, 6))
                                 for _ in range(rng.randrange(1, 6)))
            polynomial = mf.Polynomial(coefficients, self.x)
            expression = polynomial.to_expression()
            expected = mf.Polynomial((mf.Rational(0),)+tuple(
                coefficient/(index+1) for index, coefficient in enumerate(polynomial.coefficients)), self.x)
            with self.subTest(case=case):
                actual = mf.antiderivative(expression, self.x)
                self.assertEqual(mf.Polynomial.from_expression(actual, self.x), expected)
                self.assertEqual(mf.Polynomial.from_expression(mf.diff(actual, self.x), self.x), polynomial)
                a, b, c = (mf.Rational(value, 3) for value in rng.sample(range(-9, 10), 3))
                whole = mf.integrate(expression, self.x, a, c)
                self.assertEqual(whole, mf.integrate(expression, self.x, a, b)+
                                 mf.integrate(expression, self.x, b, c))
                expected_integral = sum((coefficient*(c**(index+1)-a**(index+1))/(index+1)
                                         for index, coefficient in enumerate(polynomial.coefficients)), mf.Rational(0))
                self.assertEqual(whole, expected_integral)


if __name__ == "__main__":
    unittest.main()
