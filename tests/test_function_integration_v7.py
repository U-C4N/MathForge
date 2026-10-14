"""Exact proper integrals, whole-interval domains and independent certificates."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge import function_integration as fi
from mathforge import rational_integration as ri
from mathforge import functions as f
from mathforge.function_checks import equivalent
from mathforge.limits import computation


class FunctionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000079")

    def function(self, text):
        return m.parse_function(text, symbols={"x": self.x})

    def arguments(self, source, lower, upper):
        return {"expression": source, "variable": self.x, "lower": lower, "upper": upper}

    def assertEquivalent(self, left, right):
        with computation():
            self.assertTrue(equivalent(left, right), (left, right))

    def integral(self, source, lower, upper, *, rational=False):
        module = ri if rational else fi
        operation = "rational_integrate" if rational else "function_integrate"
        args = self.arguments(source, lower, upper)
        value, evidence = module.produce(operation, args)
        self.assertNotIsInstance(value, f.UnevaluatedIntegral)
        self.assertTrue(module.check_operation(operation, args, value, evidence))
        return operation, args, value, evidence

    def test_exact_polynomial_exp_sin_and_log_integrals(self):
        cases = (
            ("x*x-2*x+1", -1, 2, f.as_real_expr(3)),
            ("exp(x)", 0, 1, f.simplify_expr(f.exp(1)-1)),
            ("sin(x)", 0, 1, f.simplify_expr(1-f.cos(1))),
            ("log(x)", 1, 2, f.simplify_expr(2*f.log(2)-2+1)),
        )
        for text, lower, upper, expected in cases:
            with self.subTest(source=text):
                _, _, value, _ = self.integral(self.function(text), lower, upper)
                self.assertEquivalent(value, expected)

    def test_reversal_and_zero_length_are_exact(self):
        source = self.function("exp(x)")
        _, _, forward, _ = self.integral(source, 0, 1)
        _, _, reverse, _ = self.integral(source, 1, 0)
        self.assertEquivalent(reverse, -forward)
        _, _, zero, _ = self.integral(source, 1, 1)
        self.assertEquivalent(zero, f.as_real_expr(0))

    def test_polynomial_domain_must_cover_entire_closed_interval(self):
        x = self.x
        source = f.real_function(x, (x,), domain=f.condition("positive", x*x-1))
        _, _, value, _ = self.integral(source, 2, 3)
        self.assertEqual(value, f.as_real_expr(m.Rational(5, 2)))
        for lower, upper in ((-2, 2), (1, 2)):
            with self.subTest(bounds=(lower, upper)):
                args = self.arguments(source, lower, upper)
                value, evidence = fi.produce("function_integrate", args)
                self.assertIsInstance(value, f.UnevaluatedIntegral)
                self.assertEqual(evidence, ())

    def test_improper_or_unknown_integrals_remain_deferred(self):
        cases = (("log(x)", -1, 1), ("log(x)", 0, 1),
                 ("1/(x-1)", 0, 2), ("exp(x*x)", 0, 1),
                 ("1/(x*x*x+x+1)", 1, 2))
        for text, lower, upper in cases:
            with self.subTest(source=text, bounds=(lower, upper)):
                args = self.arguments(self.function(text), lower, upper)
                value, evidence = fi.produce("function_integrate", args)
                self.assertIsInstance(value, f.UnevaluatedIntegral)
                self.assertEqual(evidence, ())
                self.assertFalse(fi.check_operation("function_integrate", args, value, evidence))

    def test_cancellation_holes_survive_function_and_rational_conversion(self):
        x = self.x
        sources = ((self.function("(x-1)/(x-1)"), False),
                   (m.rational_function(x-1, x-1, variable=x), True))
        for source, rational in sources:
            module = ri if rational else fi
            operation = "rational_integrate" if rational else "function_integrate"
            with self.subTest(rational=rational):
                _, _, value, _ = self.integral(source, 2, 3, rational=rational)
                self.assertEqual(value, f.as_real_expr(1))
                for lower, upper in ((0, 2), (1, 2), (1, 1)):
                    args = self.arguments(source, lower, upper)
                    value, evidence = module.produce(operation, args)
                    self.assertIsInstance(value, f.UnevaluatedIntegral)
                    self.assertEqual(evidence, ())

    def test_real_algebraic_bounds_and_algebraic_poles(self):
        x = self.x
        source = m.rational_function(1, x, variable=x)
        operation, args, value, evidence = self.integral(source, m.sqrt(2), m.sqrt(3), rational=True)
        self.assertEqual(len(evidence), 6)
        self.assertEqual(m.loads(m.dumps(value)), value)
        changed = dict(args, lower=m.sqrt(3))
        self.assertFalse(ri.check_operation(operation, changed, value, evidence))
        pole_source = m.rational_function(1, x*x-2, variable=x)
        for lower, upper in ((1, 2), (m.sqrt(2), 2)):
            args = self.arguments(pole_source, lower, upper)
            result, proof = ri.produce("rational_integrate", args)
            self.assertIsInstance(result, f.UnevaluatedIntegral)
            self.assertEqual(proof, ())

    def test_false_value_primitive_bounds_and_domain_proof_are_rejected(self):
        operation, args, value, evidence = self.integral(self.function("log(x)"), 1, 2)
        self.assertFalse(fi.check_operation(operation, args, value+1, evidence))
        primitive, formal, lower, upper, bounds_proof, domain_proof = evidence
        wrong_primitive = f.real_function(0, (self.x,))
        forged = (wrong_primitive, formal, lower, upper, bounds_proof, domain_proof)
        self.assertFalse(fi.check_operation(operation, args, f.as_real_expr(0), forged))
        self.assertFalse(fi.check_operation(operation, args, value,
                         (primitive, formal, lower, upper, (), domain_proof)))
        self.assertFalse(fi.check_operation(operation, args, value,
                         (primitive, formal, lower, upper, bounds_proof, ("true",))))
        crossing_args = dict(args, lower=-1)
        self.assertFalse(fi.check_operation(operation, crossing_args, value, evidence))

    def test_cancelling_extra_elementary_poles_are_not_valid_certificates(self):
        x = self.x
        source = m.rational_function(0, 1, variable=x)
        denominator = m.Polynomial((0, 1), x)
        one, minus_one = m.Polynomial((1,), x), m.Polynomial((-1,), x)
        forged = ri.RationalPrimitive(source, source, (),
                                     ((one, denominator), (minus_one, denominator)))
        self.assertFalse(ri.check_operation("rational_antiderivative",
                         {"expression": source, "variable": x}, forged, ()))

    def test_cancelling_extra_logarithmic_poles_are_not_valid_certificates(self):
        x, Q = self.x, m.Rational
        source = m.rational_function(0, 1, variable=x)
        h = ((Q(0),), (Q(1),))
        plus = ri.LogRootSum((Q(-1), Q(1)), h)
        minus = ri.LogRootSum((Q(1), Q(1)), h)
        forged = ri.RationalPrimitive(source, source, (plus, minus), ())
        self.assertFalse(ri.check_operation("rational_antiderivative",
                         {"expression": source, "variable": x}, forged, ()))

    def test_checkers_do_not_call_integral_domain_or_root_producers(self):
        x = self.x
        records = [
            (fi, self.integral(self.function("log(x)"), 1, 2)),
            (ri, self.integral(m.rational_function(1, x*x+1, variable=x), 0, 1, rational=True)),
            (ri, self.integral(m.rational_function(1, x, variable=x), m.sqrt(2), m.sqrt(3), rational=True)),
        ]
        targets = (
            "mathforge.function_integration.produce",
            "mathforge.function_integration.interval_proof",
            "mathforge.function_calculus.antiderivative",
            "mathforge.function_calculus._primitive",
            "mathforge.function_calculus._differentiate_expr",
            "mathforge.rational_integration.produce",
            "mathforge.rational_integration._primitive",
            "mathforge.rational_integration._lrt",
            "mathforge.rational_integration.RationalPrimitive.as_function",
            "mathforge.algebraic_arithmetic.produce",
            "mathforge.root_isolation.real_roots",
            "mathforge.real_sets.real_roots",
            "mathforge.real_sets.solve_real_set",
            "mathforge.exact_extensions._apart",
            "mathforge.polynomial_exact.factor_with_evidence",
        )
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch(target, side_effect=AssertionError(target)))
            for module, (operation, args, value, evidence) in records:
                with self.subTest(operation=operation, value=str(value)):
                    self.assertTrue(module.check_operation(operation, args, value, evidence))

    def test_corrupt_real_conversion_cannot_certify_zero_for_positive_integrand(self):
        x = self.x
        source = m.rational_function(1, x*x+1, variable=x)
        operation, args, value, evidence = self.integral(source, 0, 1, rational=True)
        wrong = f.real_function(0, (x,))
        with patch.object(ri.RationalPrimitive, "as_function", return_value=wrong):
            self.assertTrue(ri.check_operation(operation, args, value, evidence))
            self.assertFalse(ri.check_operation(operation, args, f.as_real_expr(0),
                                              (wrong,)+evidence[1:]))


if __name__ == "__main__":
    unittest.main()
