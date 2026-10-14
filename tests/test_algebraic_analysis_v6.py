"""Algebraic limits, integration bounds, extrema and their certificates."""

from dataclasses import replace
from unittest import TestCase, main, mock

import mathforge as mf
from mathforge import algebraic_analysis as analysis
from mathforge.algebraic import compare_real
from mathforge.analysis_types import ExtremumValue, LimitValue
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ResourceLimitError


class AlgebraicAnalysisTests(TestCase):
    def setUp(self):
        self.x = mf.symbol("x")
        self.root = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)

    def run_operation(self, operation, **args):
        value, evidence = analysis.produce(operation, args)
        self.assertTrue(analysis.check_operation(operation, args, value, evidence))
        return value, evidence

    def limit(self, function, point, side="both"):
        return self.run_operation("limit", expression=function, variable=self.x, point=point, side=side)[0]

    def test_finite_polynomial_and_rational_limits_at_algebraic_points(self):
        self.assertEqual(self.limit(self.x, self.root), LimitValue("finite", self.root))
        self.assertEqual(self.limit(self.x**3, self.root), LimitValue("finite", mf.Rational(2)))
        rational = mf.rational_function(self.x**3+2, self.x**3-1, variable=self.x)
        self.assertEqual(self.limit(rational, self.root), LimitValue("finite", mf.Rational(4)))

    def test_algebraic_hole_has_limit_but_remains_undefined(self):
        hole = mf.rational_function((self.x**2-2)*(self.x+1), self.x**2-2, variable=self.x)
        value = self.limit(hole, mf.sqrt(2))
        self.assertEqual(compare_real(value.value, 1+mf.sqrt(2)), 0)
        self.assertFalse(hole.is_defined_at(mf.sqrt(2)))

    def test_odd_and_even_order_poles_at_both_signs_of_sqrt_two(self):
        odd = mf.rational_function(1, self.x**2-2, variable=self.x)
        self.assertEqual(self.limit(odd, mf.sqrt(2), "left").kind, "negative_infinity")
        self.assertEqual(self.limit(odd, mf.sqrt(2), "right").kind, "positive_infinity")
        self.assertEqual(self.limit(odd, -mf.sqrt(2), "left").kind, "positive_infinity")
        self.assertEqual(self.limit(odd, -mf.sqrt(2), "right").kind, "negative_infinity")
        self.assertEqual(self.limit(odd, mf.sqrt(2)).kind, "does_not_exist")
        even = mf.rational_function(-1, (self.x**2-2)**2, variable=self.x)
        self.assertEqual(self.limit(even, mf.sqrt(2)).kind, "negative_infinity")

    def test_signed_infinity_uses_existing_rational_limit_checker(self):
        self.assertEqual(self.limit(self.x**2, "+inf").kind, "positive_infinity")
        with self.assertRaises(InvalidInput):
            self.limit(self.x, "+inf", "left")

    def test_polynomial_integrals_algebraic_and_cross_field_bounds(self):
        for lower, upper, expected in ((0, mf.sqrt(2), mf.Rational(1)),
                                       (mf.sqrt(2), 0, mf.Rational(-1)),
                                       (mf.sqrt(2), mf.sqrt(2), mf.Rational(0)),
                                       (mf.sqrt(2), mf.sqrt(3), mf.Rational(1, 2))):
            value, _ = self.run_operation("integrate", expression=self.x, variable=self.x, lower=lower, upper=upper)
            self.assertEqual(value, expected)
        value, _ = self.run_operation("integrate", expression=self.x**2, variable=self.x, lower=0, upper=self.root)
        self.assertEqual(value, mf.Rational(2, 3))

    def test_general_integral_algebraic_result(self):
        value, _ = self.run_operation("integrate", expression=1, variable=self.x, lower=mf.sqrt(2), upper=mf.sqrt(3))
        from mathforge.algebraic_arithmetic import produce
        expected, _ = produce("algebraic", dict(expression=mf.sqrt(3)-mf.sqrt(2)))
        self.assertEqual(value, expected)

    def test_integration_does_not_add_rational_antiderivatives_or_improper_integrals(self):
        rational = mf.rational_function(1, self.x, variable=self.x)
        with self.assertRaises(UnsupportedOperation):
            analysis.produce("integrate", dict(expression=rational, variable=self.x, lower=1, upper=mf.sqrt(2)))
        with self.assertRaises((UnsupportedOperation, TypeError)):
            analysis.produce("integrate", dict(expression=self.x, variable=self.x, lower=0, upper="+inf"))

    def test_extrema_return_complete_exact_points_and_values(self):
        rational = mf.rational_function(self.x**2+2, self.x, variable=self.x)
        values, _ = self.run_operation("extrema_values", expression=rational, variable=self.x)
        self.assertEqual(values, (
            ExtremumValue(mf.RealAlgebraicRoot((-2, 0, 1), 0), mf.RealAlgebraicRoot((-8, 0, 1), 0), "maximum"),
            ExtremumValue(mf.RealAlgebraicRoot((-2, 0, 1), 1), mf.RealAlgebraicRoot((-8, 0, 1), 1), "minimum")))
        self.assertEqual(self.run_operation("extrema_values", expression=self.x**4-4*self.x, variable=self.x)[0],
                         (ExtremumValue(mf.Rational(1), mf.Rational(-3), "minimum"),))

    def test_extrema_do_not_include_stationary_inflections_constants_or_holes(self):
        for expression in (self.x**3, 2, mf.rational_function(self.x**3, self.x, variable=self.x)):
            self.assertEqual(self.run_operation("extrema_values", expression=expression, variable=self.x)[0], ())

    def test_limit_certificate_rejects_wrong_order_direction_source_and_value(self):
        function = mf.rational_function(1, (self.x**2-2)**2, variable=self.x)
        args = dict(expression=function, variable=self.x, point=mf.sqrt(2), side="both")
        value, evidence = self.run_operation("limit", **args)
        self.assertFalse(analysis.check_operation("limit", args, LimitValue("negative_infinity"), evidence))
        self.assertFalse(analysis.check_operation("limit", args, value, (*evidence[:3], ("pole", 1))))
        self.assertFalse(analysis.check_operation("limit", dict(args, expression=-function), value, evidence))
        self.assertFalse(analysis.check_operation("limit", dict(args, point=mf.sqrt(3)), value, evidence))
        args = dict(expression=self.x, variable=self.x, point=self.root, side="both")
        value, evidence = self.run_operation("limit", **args)
        self.assertFalse(analysis.check_operation("limit", args, LimitValue("finite", mf.Rational(2)), evidence))

    def test_integral_certificate_checks_primitive_bounds_and_result(self):
        args = dict(expression=self.x**2, variable=self.x, lower=0, upper=self.root)
        value, evidence = self.run_operation("integrate", **args)
        self.assertFalse(analysis.check_operation("integrate", args, value+1, evidence))
        self.assertFalse(analysis.check_operation("integrate", dict(args, upper=mf.sqrt(2)), value, evidence))
        self.assertFalse(analysis.check_operation("integrate", args, value,
                                                (evidence[0], evidence[1]+1, *evidence[2:])))
        self.assertFalse(analysis.check_operation("integrate", args, value,
                                                (evidence[0], evidence[1], evidence[3], evidence[2], evidence[4])))

    def test_extrema_checker_rejects_missing_or_forged_coordinate_and_height(self):
        args = dict(expression=self.x**2, variable=self.x)
        values, evidence = self.run_operation("extrema_values", **args)
        for forged in ((), (replace(values[0], value=mf.Rational(1)),),
                       (replace(values[0], point=mf.Rational(1)),),
                       (replace(values[0], kind="maximum"),)):
            self.assertFalse(analysis.check_operation("extrema_values", args, forged, evidence))

    def test_analysis_checkers_never_call_producers(self):
        cases = (
            ("limit", dict(expression=self.x**3, variable=self.x, point=self.root, side="both")),
            ("integrate", dict(expression=self.x, variable=self.x, lower=0, upper=mf.sqrt(2))),
            ("extrema_values", dict(expression=self.x**2, variable=self.x)),
        )
        results = [(operation, args, *self.run_operation(operation, **args)) for operation, args in cases]
        blocked = AssertionError("producer called during checking")
        with (mock.patch.object(analysis, "produce", side_effect=blocked),
              mock.patch("mathforge.algebraic_arithmetic.produce", side_effect=blocked),
              mock.patch("mathforge.function_analysis.analyze_function", side_effect=blocked),
              mock.patch("mathforge.polynomial_exact.factor_with_evidence", side_effect=blocked)):
            for operation, args, value, evidence in results:
                self.assertTrue(analysis.check_operation(operation, args, value, evidence))

    def test_checker_does_not_share_evaluation_or_pole_classification_helpers(self):
        cases = (("limit", dict(expression=self.x**3, variable=self.x, point=self.root, side="both")),
                 ("limit", dict(expression=mf.rational_function(1, self.x**2-2, variable=self.x),
                                variable=self.x, point=mf.sqrt(2), side="right")),
                 ("integrate", dict(expression=self.x, variable=self.x, lower=0, upper=mf.sqrt(2))))
        results = [(operation, args, *self.run_operation(operation, **args)) for operation, args in cases]
        with (mock.patch.object(analysis, "_at_expression", side_effect=AssertionError("producer evaluation called")),
              mock.patch.object(analysis, "_quotient_arguments", side_effect=AssertionError("producer quotient called")),
              mock.patch.object(analysis, "_infinite", side_effect=AssertionError("producer pole classification called"))):
            for operation, args, value, evidence in results:
                self.assertTrue(analysis.check_operation(operation, args, value, evidence))

    def test_injected_endpoint_and_pole_producer_faults_are_detected(self):
        integral = dict(expression=self.x, variable=self.x, lower=0, upper=mf.sqrt(2))
        with mock.patch.object(analysis, "_at_expression", return_value=mf.Rational(0)):
            value, evidence = analysis.produce("integrate", integral)
            self.assertEqual(value, mf.Rational(0))
            self.assertFalse(analysis.check_operation("integrate", integral, value, evidence))
        finite = dict(expression=self.x, variable=self.x, point=self.root, side="both")
        with mock.patch.object(analysis, "_quotient_arguments", return_value=dict(left=mf.Rational(0), right=mf.Rational(1))):
            value, evidence = analysis.produce("limit", finite)
            self.assertFalse(analysis.check_operation("limit", finite, value, evidence))
        pole = dict(expression=mf.rational_function(1, self.x**2-2, variable=self.x),
                    variable=self.x, point=mf.sqrt(2), side="right")
        with mock.patch.object(analysis, "_infinite", return_value=LimitValue("negative_infinity")):
            value, evidence = analysis.produce("limit", pole)
            self.assertFalse(analysis.check_operation("limit", pole, value, evidence))

    def test_checker_reconstructs_polynomial_sources_independently(self):
        integral = dict(expression=self.x**2, variable=self.x, lower=0, upper=self.root)
        zero = mf.Polynomial((0,), self.x)
        with mock.patch.object(analysis, "as_polynomial", return_value=zero):
            value, evidence = analysis.produce("integrate", integral)
            self.assertFalse(analysis.check_operation("integrate", integral, value, evidence))
        zero_function = mf.rational_function(0, 1, variable=self.x)
        finite = dict(expression=self.x**3, variable=self.x, point=self.root, side="both")
        with mock.patch.object(analysis, "as_rational_function", return_value=zero_function):
            value, evidence = analysis.produce("limit", finite)
            self.assertFalse(analysis.check_operation("limit", finite, value, evidence))
        original = mf.Polynomial.from_expression

        def lose_symbolic_coefficients(expression, variable, **kwargs):
            if type(expression) in (int, mf.Rational):
                return original(expression, variable, **kwargs)
            return mf.Polynomial((0,), variable)

        with mock.patch.object(mf.Polynomial, "from_expression", side_effect=lose_symbolic_coefficients):
            for operation, args in (("integrate", integral), ("limit", finite),
                                    ("extrema_values", dict(expression=self.x**2, variable=self.x))):
                value, evidence = analysis.produce(operation, args)
                self.assertFalse(analysis.check_operation(operation, args, value, evidence))

    def test_producer_source_converters_are_unused_when_checking(self):
        cases = (("integrate", dict(expression=(self.x+1)**3, variable=self.x, lower=0, upper=self.root)),
                 ("limit", dict(expression=self.x**3, variable=self.x, point=self.root, side="both")),
                 ("extrema_values", dict(expression=self.x**2, variable=self.x)))
        results = [(operation, args, *self.run_operation(operation, **args)) for operation, args in cases]
        blocked = AssertionError("producer source converter called")
        with (mock.patch.object(analysis, "as_polynomial", side_effect=blocked),
              mock.patch.object(analysis, "as_rational_function", side_effect=blocked),
              mock.patch.object(mf.Polynomial, "from_expression", side_effect=blocked)):
            for operation, args, value, evidence in results:
                self.assertTrue(analysis.check_operation(operation, args, value, evidence))

    def test_analysis_budgets_bound_producer_and_certificate_checker(self):
        args = dict(expression=self.x, variable=self.x, lower=0, upper=mf.sqrt(2))
        value, evidence = self.run_operation("integrate", **args)
        for limits in (mf.ComputationLimits(max_work=1), mf.ComputationLimits(max_evidence_items=1)):
            with self.assertRaises(ResourceLimitError):
                analysis.produce("integrate", args, limits=limits)
            with self.assertRaises(ResourceLimitError):
                analysis.check_operation("integrate", args, value, evidence, limits=limits)


if __name__ == "__main__":
    main()
