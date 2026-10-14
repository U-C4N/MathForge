"""Adversarial exact-algebra checks using constructor-bypassed records."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import exact_extensions as extensions, polynomial_exact as exact
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation


def forged(cls, **fields):
    value = object.__new__(cls)
    for name, field in fields.items():
        object.__setattr__(value, name, field)
    return value


class ExactAdversarialAuditV6Tests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x")

    def p(self, *coefficients):
        return mf.Polynomial(coefficients, self.x)

    def test_unreduced_rationals_cannot_hide_in_factors_or_characteristic_polynomials(self):
        one = forged(mf.Rational, numerator=2, denominator=2)
        factor = forged(mf.Polynomial, coefficients=(one, mf.Rational(1)), variable=self.x)
        result = exact.PolynomialFactorization(1, (exact.PolynomialFactor(factor, 1),))
        self.assertFalse(exact.check_factor(self.p(1, 1), result, ("qq-factorization-v1", (("linear",),))))
        polynomial = forged(mf.Polynomial, coefficients=(one, mf.Rational(-2), mf.Rational(1)), variable=self.x)
        self.assertFalse(extensions.check_operation("charpoly", {
            "A": mf.Matrix(((1, 0), (0, 1))), "variable": self.x}, polynomial, ()))

    def test_boolean_matrix_scalars_and_eigenvalue_multiplicities_are_rejected(self):
        one = forged(mf.Rational, numerator=True, denominator=1)
        matrix = forged(mf.Matrix, rows=((one,),), ncols=1)
        self.assertFalse(extensions.check_operation("charpoly", {"A": matrix, "variable": self.x}, self.p(-1, 1), ()))
        valid = mf.Matrix(((1,),))
        values, evidence = extensions.produce("real_eigenvalues", {"A": valid})
        record = forged(mf.RootRecord, root=values[0].root, multiplicity=True)
        self.assertFalse(extensions.check_operation("real_eigenvalues", {"A": valid}, (record,), evidence))

    def test_partial_fractions_reject_a_source_with_an_unexcluded_pole(self):
        function = mf.rational_function(1, self.x-1, variable=self.x)
        value, evidence = extensions.produce("apart", {"expression": function})
        source = forged(mf.RationalFunction, numerator=function.numerator,
                        denominator=function.denominator, excluded=())
        altered = replace(value, excluded=())
        self.assertFalse(extensions.check_operation("apart", {"expression": source}, altered, evidence))
        with self.assertRaises(InvalidInput):
            extensions.produce("apart", {"expression": source})

    def test_partial_fractions_cannot_drop_cancelled_holes_or_add_fake_exclusions(self):
        x = self.x
        function = mf.rational_function((x-2)*(x+1), (x-2)*(x-1)**2*(x*x+1), variable=x)
        value, evidence = extensions.produce("apart", {"expression": function})
        self.assertTrue(extensions.check_operation("apart", {"expression": function}, value, evidence))
        self.assertEqual(value.to_rational_function(), function)
        for excluded in ((), (self.p(-9, 1),)):
            altered = replace(value, excluded=excluded)
            self.assertFalse(extensions.check_operation("apart", {"expression": function}, altered, evidence))

    def test_improper_duplicate_and_overpowered_partial_fraction_terms_fail(self):
        function = mf.rational_function(self.x+1, (self.x-1)**2, variable=self.x)
        value, evidence = extensions.produce("apart", {"expression": function})
        term = value.terms[0]
        altered_terms = (
            value.terms + (term,),
            (forged(mf.PartialFractionTerm, numerator=term.factor, factor=term.factor, power=term.power),),
            (forged(mf.PartialFractionTerm, numerator=term.numerator, factor=term.factor, power=1000000),),
        )
        for terms in altered_terms:
            altered = forged(mf.PartialFractionDecomposition, polynomial=value.polynomial,
                             terms=terms, excluded=value.excluded)
            self.assertFalse(extensions.check_operation("apart", {"expression": function}, altered, evidence))

    def test_characteristic_checker_rejects_annihilators_and_sample_point_tricks(self):
        matrix = mf.Matrix(((1, 0), (0, 1)))
        args = {"A": matrix, "variable": self.x}
        # Both polynomials annihilate I, but only (x-1)^2 is characteristic.
        self.assertFalse(extensions.check_operation("charpoly", args, self.p(2, -3, 1), ()))
        true = self.p(1, -2, 1)
        sample_trick = true + self.p(0, 1)*self.p(-1, 1)*self.p(-2, 1)
        self.assertFalse(extensions.check_operation("charpoly", args, sample_trick, ()))
        wrong_identity = mf.Polynomial(true.coefficients, mf.symbol("x"))
        self.assertFalse(extensions.check_operation("charpoly", args, wrong_identity, ()))

    def test_real_eigenvalues_are_complete_with_repeated_and_empty_spectra(self):
        for rows, count in ((((2, 0), (0, 2)), 1), (((0, -1), (1, 0)), 0), ((), 0)):
            matrix = mf.Matrix(rows)
            values, evidence = extensions.produce("real_eigenvalues", {"A": matrix})
            self.assertEqual(len(values), count)
            self.assertTrue(extensions.check_operation("real_eigenvalues", {"A": matrix}, values, evidence))
            if values:
                self.assertEqual(values[0].multiplicity, 2)
                self.assertFalse(extensions.check_operation("real_eigenvalues", {"A": matrix}, (), evidence))
                altered = (replace(values[0], multiplicity=1),)
                self.assertFalse(extensions.check_operation("real_eigenvalues", {"A": matrix}, altered, evidence))

    def test_oversized_certificate_lengths_and_integer_growth_are_bounded(self):
        polynomial = self.p(1, 0, 0, 0, 1)
        value, evidence = exact.factor_with_evidence(polynomial)
        oversized = (evidence[0], evidence[1]*100)
        with self.assertRaises(ResourceLimitError):
            exact.check_factor(polynomial, value, oversized, limits=ComputationLimits(max_evidence_items=30))
        bad_integer = forged(mf.Rational, numerator=1 << 1000, denominator=(1 << 999)+1)
        source = forged(mf.Polynomial, coefficients=(bad_integer, mf.Rational(1)), variable=self.x)
        with self.assertRaises(ResourceLimitError):
            exact.factor_with_evidence(source, limits=ComputationLimits(max_integer_bits=64))
        with computation(ComputationLimits(max_evidence_items=30)):
            with self.assertRaises(ResourceLimitError):
                extensions.check_operation("charpoly", {"A": mf.Matrix(((1,),)), "variable": self.x},
                                           self.p(-1, 1), ((0,)*1000,))
        with self.assertRaises(ResourceLimitError):
            exact.resultant(self.p(1 << 20, 1, 0, 1), self.p(1, 1 << 20, 1),
                            limits=ComputationLimits(max_integer_bits=48))

    def test_subresultants_handle_degree_gaps_rational_content_and_repetition(self):
        pairs = (
            (self.p(1, 0, 0, 0, 0, 0, 1), self.p(2, 0, 1)),
            (self.p(-1, 1), self.p(2, 0, 0, 1)),
            (self.p(2, -3, 0, 5), self.p(7, 0, 2)),
            (self.p(mf.Rational(1, 7), 0, mf.Rational(-4, 3), mf.Rational(5, 2)),
             self.p(mf.Rational(2, 9), mf.Rational(-7, 5), 0, 0, mf.Rational(3, 2))),
            (self.p(-1, 1)**4*self.p(1, 0, 1), self.p(-1, 1)**2*self.p(2, 1)),
            (self.p(mf.Rational(-2, 3)), self.p(mf.Rational(3, 5), 2, 0, 1)),
            (self.p(7), self.p(-2)),
        )
        for f, g in pairs:
            value = exact.resultant(f, g)
            self.assertTrue(exact.check_resultant(f, g, value), (f, g, value))
            self.assertEqual(value, (-1)**(f.degree*g.degree)*exact.resultant(g, f))
        rng = random.Random(6127)
        for _ in range(70):
            f = self.p(*(mf.Rational(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(rng.randint(1, 8))), 1)
            g = self.p(*(mf.Rational(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(rng.randint(1, 6))), 2)
            self.assertTrue(exact.check_resultant(f, g, exact.resultant(f, g)))

    def test_subresultant_producer_uses_no_rational_long_division(self):
        f, g = self.p(2, -3, 0, 5), self.p(7, 0, 2)
        with patch.object(mf.Polynomial, "divmod", side_effect=AssertionError("rational division called")):
            value = exact.resultant(f, g)
        with patch.object(exact, "_subresultant_integer", side_effect=AssertionError("producer called")), \
             patch.object(exact, "_pseudo_remainder", side_effect=AssertionError("producer called")):
            self.assertTrue(exact.check_resultant(f, g, value))


if __name__ == "__main__":
    unittest.main()
