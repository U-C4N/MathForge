"""Hermite/LRT identities, proper-domain semantics, and corrupted primitives."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge import rational_integration as ri
from mathforge import functions as f


class RationalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000071")

    def source(self, denominator, numerator=1):
        return m.rational_function(numerator, denominator, variable=self.x)

    def primitive(self, source):
        args = {"expression": source, "variable": self.x}
        value, evidence = ri.produce("rational_antiderivative", args)
        self.assertTrue(ri.check_operation("rational_antiderivative", args, value, evidence))
        self.assertEqual(value.source, source)
        self.assertEqual(value.rational_part.excluded, source.excluded)
        return args, value, evidence

    def test_linear_repeated_quadratic_and_irreducible_quintic(self):
        x = self.x
        for denominator in (x, (x*x+1)**2, x**5-x+1):
            with self.subTest(denominator=str(denominator)):
                _, value, _ = self.primitive(self.source(denominator))
                self.assertTrue(value.log_terms)
        _, repeated, _ = self.primitive(self.source((x-1)**2))
        self.assertEqual(repeated.log_terms, ())
        self.assertEqual(repeated.rational_part.numerator.to_expression(), m.Rational(-1))
        self.assertEqual(repeated.rational_part.denominator.to_expression(), x-1)

    def test_canceled_source_holes_remain_excluded(self):
        x = self.x
        source = self.source(x*(x-1), x-1)
        _, primitive, _ = self.primitive(source)
        real = primitive.as_function()
        self.assertIs(real.domain.contains({x: m.Rational(1)}), m.Truth.FALSE)
        self.assertIs(real.domain.contains({x: m.Rational(2)}), m.Truth.TRUE)

    def test_proper_integral_reversal_and_pole_are_distinct(self):
        x = self.x
        source = self.source(x)
        args = {"expression": source, "variable": x, "lower": m.Rational(1), "upper": m.Rational(2)}
        value, evidence = ri.produce("rational_integrate", args)
        self.assertTrue(ri.check_operation("rational_integrate", args, value, evidence))
        self.assertEqual(value, f.simplify_expr(f.log(2)))
        reversed_args = dict(args, lower=m.Rational(2), upper=m.Rational(1))
        reverse, reverse_evidence = ri.produce("rational_integrate", reversed_args)
        self.assertTrue(ri.check_operation("rational_integrate", reversed_args, reverse, reverse_evidence))
        self.assertEqual(reverse, f.simplify_expr(-value))
        crossing = dict(args, lower=m.Rational(-1), upper=m.Rational(1))
        deferred, deferred_evidence = ri.produce("rational_integrate", crossing)
        self.assertIsInstance(deferred, f.UnevaluatedIntegral)
        self.assertFalse(ri.check_operation("rational_integrate", crossing, deferred, deferred_evidence))

    def test_modified_logarithmic_data_rejected(self):
        args, value, evidence = self.primitive(self.source(self.x))
        term = value.log_terms[0]
        changed = replace(term, residue_polynomial=(m.Rational(-2), m.Rational(1)))
        self.assertFalse(ri.check_operation("rational_antiderivative", args,
                                          replace(value, log_terms=(changed,)), evidence))
        self.assertFalse(ri.check_operation("rational_antiderivative", args,
                                          replace(value, log_terms=()), evidence))
        self.assertFalse(ri.check_operation("rational_antiderivative", args,
                                          replace(value, elementary_terms=()), evidence))

    def test_checker_does_not_run_hermite_lrt_or_factorization_producers(self):
        x = self.x
        records = [self.primitive(self.source(d)) for d in (x, (x*x+1)**2, x**5-x+1)]
        with patch.object(ri, "produce", side_effect=AssertionError("integral producer")), \
             patch.object(ri, "_primitive", side_effect=AssertionError("Hermite producer")), \
             patch.object(ri, "_lrt", side_effect=AssertionError("LRT producer")), \
             patch("mathforge.exact_extensions._apart", side_effect=AssertionError("partial fractions producer")), \
             patch("mathforge.polynomial_exact.factor_with_evidence", side_effect=AssertionError("factorization producer")):
            for args, value, evidence in records:
                self.assertTrue(ri.check_operation("rational_antiderivative", args, value, evidence))


if __name__ == "__main__":
    unittest.main()
