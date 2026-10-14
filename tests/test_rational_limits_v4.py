"""Exact rational limits: domain holes, directional poles and certificates."""

from dataclasses import replace
import unittest
from unittest import mock

import mathforge as mf
from mathforge.analysis_types import LimitValue
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.rational_limits import check_limit, limit_value


class RationalLimitTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x")

    def rf(self, numerator, denominator):
        return mf.rational_function(numerator, denominator, variable=self.x)

    def assertLimit(self, expression, point, kind, rational=None, side="both"):
        value, evidence = limit_value(expression, self.x, point, side)
        self.assertEqual(value, LimitValue(kind, rational))
        self.assertTrue(check_limit(expression, self.x, point, side, value, evidence))
        return value, evidence

    def test_polynomials_constants_zero_and_rational_targets(self):
        self.assertLimit(self.x**2 + 2*self.x, mf.Rational(1, 2), "finite", mf.Rational(5, 4))
        for expression, expected in ((0, 0), (7, 7), (self.x-self.x, 0)):
            for point in (0, 3, "+inf", "-inf"):
                with self.subTest(expression=expression, point=point):
                    self.assertLimit(expression, point, "finite", mf.Rational(expected))

    def test_removable_and_zero_holes_do_not_block_limits(self):
        removable = self.rf(self.x**2-1, self.x-1)
        for side in ("both", "left", "right"):
            self.assertLimit(removable, 1, "finite", mf.Rational(2), side=side)
        self.assertFalse(removable.is_defined_at(mf.Rational(1)))
        self.assertLimit(self.rf(0, self.x), 0, "finite", mf.Rational(0))
        self.assertLimit(self.rf(self.x, self.x), 0, "finite", mf.Rational(1))

    def test_odd_poles_two_sided_nonexistence_and_one_sided_signs(self):
        for numerator, sign in ((1, 1), (-1, -1)):
            for order in (1, 3, 5):
                function = self.rf(numerator, self.x**order)
                with self.subTest(numerator=numerator, order=order):
                    self.assertLimit(function, 0, "does_not_exist")
                    self.assertLimit(function, 0, "positive_infinity" if sign > 0 else "negative_infinity", side="right")
                    self.assertLimit(function, 0, "negative_infinity" if sign > 0 else "positive_infinity", side="left")

    def test_even_poles_same_infinity_and_nontrivial_residual_sign(self):
        for side in ("both", "left", "right"):
            self.assertLimit(self.rf(3, (self.x-2)**4), 2, "positive_infinity", side=side)
            self.assertLimit(self.rf(3, (self.x-2)**2*(self.x-3)), 2, "negative_infinity", side=side)
        # A nonmonic factor at a nonintegral target exercises the residual.
        self.assertLimit(self.rf(1, (2*self.x-1)**3*(self.x+2)), mf.Rational(1, 2), "negative_infinity", side="left")

    def test_infinity_degree_difference_parity_and_leading_sign(self):
        for point in ("-inf", "+inf"):
            self.assertLimit(self.rf(2*self.x-1, 3*self.x**2+1), point, "finite", mf.Rational(0))
            self.assertLimit(self.rf(-2*self.x**2+1, 3*self.x**2+2), point, "finite", mf.Rational(-2, 3))
        for sign in (-1, 1):
            for difference in (1, 2, 3, 4):
                function = self.rf(sign*(self.x**(difference+1)+1), self.x-2)
                for point in ("+inf", "-inf"):
                    negative = sign < 0
                    if point == "-inf" and difference % 2:
                        negative = not negative
                    with self.subTest(sign=sign, difference=difference, point=point):
                        self.assertLimit(function, point, "negative_infinity" if negative else "positive_infinity")

    def test_scope_and_argument_validation_precede_shortcuts(self):
        for point in (True, False, 0.0, float("inf"), None, "inf", "0", [], {}):
            with self.subTest(point=point):
                with self.assertRaises(InvalidInput):
                    limit_value(0, self.x, point)
        for point in (mf.sqrt(2), self.x):
            with self.assertRaises(UnsupportedOperation):
                limit_value(0, self.x, point)
        for side in ("up", "", None, True, 1):
            with self.assertRaises(InvalidInput):
                limit_value(0, self.x, 0, side)
        for point in ("-inf", "+inf"):
            for side in ("left", "right"):
                with self.assertRaises(InvalidInput):
                    limit_value(0, self.x, point, side)
        for variable in (mf.symbol("n", domain=mf.Integers),
                         mf.symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))):
            with self.assertRaises(UnsupportedOperation):
                limit_value(0, variable, 0)
        with self.assertRaises(InvalidInput):
            limit_value(self.rf(1, self.x), mf.symbol("x"), 0)
        with self.assertRaises(UnsupportedOperation):
            limit_value(self.x + mf.symbol("y"), self.x, 0)

    def test_value_and_pole_evidence_tampering(self):
        function = self.rf(1, self.x**3*(self.x+1))
        value, evidence = self.assertLimit(function, 0, "does_not_exist")
        for bad_value in (LimitValue("finite", mf.Rational(0)), LimitValue("positive_infinity", None), None, "does_not_exist"):
            self.assertFalse(check_limit(function, self.x, 0, "both", bad_value, evidence))
        _, order, residual = evidence[4]
        for certificate in (("regular",), ("pole", True, residual), ("pole", 0, residual),
                            ("pole", order+1, residual), ("pole", order-1, residual),
                            ("pole", order, residual+1), ("pole", order, residual*mf.Polynomial((0, 1), self.x)),
                            ("pole", 10**20, residual), ("pole", order, 1), ("pole",)):
            with self.subTest(certificate=certificate):
                self.assertFalse(check_limit(function, self.x, 0, "both", value, evidence[:4]+(certificate,)))
        for malformed in (None, [], evidence[:-1], evidence+(0,), ("unknown",)+evidence[1:]):
            self.assertFalse(check_limit(function, self.x, 0, "both", value, malformed))

    def test_regular_and_infinity_claims_reject_wrong_values_and_certificates(self):
        for expression, point in ((self.x+1, 1), (self.x**2, "+inf"), (0, "-inf")):
            value, evidence = limit_value(expression, self.x, point)
            for wrong in (LimitValue("finite", mf.Rational(99)), LimitValue("does_not_exist", None)):
                self.assertFalse(check_limit(expression, self.x, point, "both", wrong, evidence))
            for wrong_certificate in (("unknown",), (), ("regular", 1), ("infinity", 1)):
                changed = evidence[:4] + (wrong_certificate,)
                self.assertFalse(check_limit(expression, self.x, point, "both", value, changed))

    def test_limit_value_kind_and_payload_are_consistent(self):
        for kind, payload in (("finite", None), ("finite", 1), ("finite", 0.1),
                              ("positive_infinity", mf.Rational(1)),
                              ("does_not_exist", mf.Rational(0)), ("unknown", None)):
            with self.subTest(kind=kind, payload=payload):
                with self.assertRaises(InvalidInput):
                    LimitValue(kind, payload)

    def test_source_point_side_and_domain_are_bound_even_for_equal_results(self):
        function = self.rf(0, self.x)
        value, evidence = self.assertLimit(function, 0, "finite", mf.Rational(0))
        self.assertFalse(check_limit(function, self.x, 1, "both", value, evidence))
        self.assertFalse(check_limit(function, self.x, 0, "left", value, evidence))
        self.assertFalse(check_limit(self.rf(0, 1), self.x, 0, "both", value, evidence))
        self.assertFalse(check_limit(self.rf(1, self.x), self.x, 0, "both", value, evidence))
        self.assertFalse(check_limit(function, mf.symbol("x"), 0, "both", value, evidence))
        self.assertTrue(check_limit(function, self.x, mf.Rational(0), "both", value, evidence))

    def test_checker_does_not_call_producer_or_producer_assembly(self):
        function = self.rf(-1, (self.x-2)**3)
        value, evidence = self.assertLimit(function, 2, "positive_infinity", side="left")
        with mock.patch("mathforge.rational_limits.limit_value", side_effect=AssertionError("producer called")), \
                mock.patch("mathforge.rational_limits._signed_infinity", side_effect=AssertionError("assembler called")), \
                mock.patch.object(mf.Polynomial, "evaluate", side_effect=AssertionError("producer evaluation called")):
            self.assertTrue(check_limit(function, self.x, 2, "left", value, evidence))

    def test_resource_limits_propagate_and_nested_work_shares_budget(self):
        function = self.rf(1, (self.x-1)**5)
        value, evidence = limit_value(function, self.x, 1)
        with self.assertRaises(ResourceLimitError):
            limit_value(function, self.x, 1, limits=ComputationLimits(max_work=1))
        with self.assertRaises(ResourceLimitError):
            check_limit(function, self.x, 1, "both", value, evidence, limits=ComputationLimits(max_evidence_items=1))
        with computation(ComputationLimits(max_work=1), fresh=True):
            with self.assertRaises(ResourceLimitError):
                limit_value(function, self.x, 1, limits=ComputationLimits(max_work=100000))

    def test_public_structured_result_verifies_and_tampered_source_fails(self):
        function = self.rf(1, self.x)
        result = mf.operations.limit(function, self.x, 0)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.COMPLETED)
        self.assertEqual(result.outcome, mf.Outcome.VALUE)
        self.assertEqual(result.value.kind, "does_not_exist")
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(mf.limit(function, self.x, 0), result.value)
        arguments = dict(result.request.arguments)
        arguments["side"] = "right"
        request = replace(result.request, arguments=tuple(arguments.items()))
        self.assertFalse(mf.verify(replace(result, request=request)).verified)


if __name__ == "__main__":
    unittest.main()
