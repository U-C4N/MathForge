"""Exact factorization completeness, independent elimination and bounded proofs."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational, symbol
from mathforge.polynomial import Polynomial
from mathforge import polynomial_exact as exact


class PolynomialV6Tests(unittest.TestCase):
    def setUp(self):
        self.x = symbol("x")

    def p(self, *coefficients):
        return Polynomial(coefficients, self.x)

    def factor(self, f):
        value, evidence = exact.factor_with_evidence(f)
        self.assertTrue(exact.check_factor(f, value, evidence))
        return value, evidence

    def test_sophie_germain_has_no_rational_linear_factor(self):
        f = self.p(4, 0, 0, 0, 1)
        value, _ = self.factor(f)
        self.assertEqual(value, exact.PolynomialFactorization(1, (
            exact.PolynomialFactor(self.p(2, -2, 1), 1),
            exact.PolynomialFactor(self.p(2, 2, 1), 1))))

    def test_irreducible_without_any_irreducible_prime_reduction(self):
        # Phi_8 has Galois group V4: no good prime gives a degree-four
        # irreducible reduction. Complete recombination must still prove it.
        f = self.p(1, 0, 0, 0, 1)
        value, evidence = self.factor(f)
        self.assertEqual(value.factors, (exact.PolynomialFactor(f, 1),))
        self.assertGreater(len(evidence[1][0][3]), 1)

    def test_rational_nonmonic_multiplicities_and_normalization(self):
        a, b, c = self.p(Rational(1, 2), 1), self.p(-3, 1), self.p(2, 0, 1)
        f = a**2 * b**3 * c * Rational(-7, 15)
        value, _ = self.factor(f)
        self.assertEqual(value.coefficient, Rational(-7, 15))
        self.assertEqual({entry.polynomial: entry.multiplicity for entry in value.factors},
                         {a: 2, b: 3, c: 1})

    def test_zero_is_invalid_constants_have_empty_factor_lists(self):
        with self.assertRaises(InvalidInput):
            exact.factor_with_evidence(self.p(0))
        for value in (Rational(1), Rational(-5, 7), Rational(19)):
            actual, evidence = self.factor(self.p(value))
            self.assertEqual(actual, exact.PolynomialFactorization(value, ()))
            self.assertEqual(evidence, ("qq-factorization-v1", ()))
        self.assertFalse(exact.check_factor(self.p(0), exact.PolynomialFactorization(0, ()),
                                            ("qq-factorization-v1", ())))

    def test_zero_constant_term_and_degree_dropping_bad_prime(self):
        f = self.p(0, 2, 1) * self.p(1, 3)**2
        value, _ = self.factor(f)
        self.assertEqual(sum(v.polynomial.degree * v.multiplicity for v in value.factors), 4)
        nonmonic = self.p(1, 2) * self.p(1, 4)
        value, _ = self.factor(nonmonic)
        self.assertEqual(value.coefficient, Rational(8))
        self.assertEqual(len(value.factors), 2)

    def test_complete_product_is_not_a_proof_of_irreducibility(self):
        f = self.p(1, 0, 1) * self.p(2, 0, 1)
        false_value = exact.PolynomialFactorization(1, (exact.PolynomialFactor(f, 1),))
        with computation():
            proof = exact._irreducibility_evidence(f)
        self.assertFalse(exact.check_factor(f, false_value, ("qq-factorization-v1", (proof,))))

    def test_factor_checker_does_not_call_producer(self):
        f = self.p(1, 0, 0, 0, 1) * self.p(2, -2, 1)**2
        value, evidence = self.factor(f)
        names = ("factor_with_evidence", "_factor_square_free", "_berlekamp",
                 "_hensel", "_lift_data", "_candidate", "_irreducibility_evidence")
        patches = [patch.object(exact, name, side_effect=AssertionError("producer called")) for name in names]
        for item in patches:
            item.start()
        try:
            self.assertTrue(exact.check_factor(f, value, evidence))
        finally:
            for item in reversed(patches):
                item.stop()

    def test_factor_source_value_and_modular_proof_tampering(self):
        f = self.p(1, 0, 0, 0, 1)
        value, evidence = self.factor(f)
        proof = evidence[1][0]
        for corrupted in (
            (), ("linear",), (proof[0], 4, proof[2], proof[3]),
            (proof[0], proof[1], 0, proof[3]),
            (proof[0], proof[1], 1, proof[3]),
            (proof[0], proof[1], proof[2], proof[3][:-1]),
            (proof[0], True, proof[2], proof[3]),
        ):
            with self.subTest(corrupted=corrupted):
                self.assertFalse(exact.check_factor(f, value, (evidence[0], (corrupted,))))
        self.assertFalse(exact.check_factor(f + 1, value, evidence))
        self.assertFalse(exact.check_factor(f, replace(value, coefficient=Rational(2)), evidence))
        self.assertFalse(exact.check_factor(f, replace(value, factors=(replace(value.factors[0], multiplicity=2),)), evidence))
        lifts = list(proof[3])
        lifts[0] = (lifts[0][0] + 1,) + lifts[0][1:]
        self.assertFalse(exact.check_factor(f, value, (evidence[0], ((proof[0], proof[1], proof[2], tuple(lifts)),))))

    def test_frozen_records_reject_ambiguous_factorizations(self):
        with self.assertRaises(InvalidInput):
            exact.PolynomialFactor(self.p(2, 2), 1)
        for multiplicity in (0, -1, True, 1.5):
            with self.assertRaises(InvalidInput):
                exact.PolynomialFactor(self.p(1, 1), multiplicity)
        a, b = exact.PolynomialFactor(self.p(1, 1), 1), exact.PolynomialFactor(self.p(2, 1), 1)
        for factors in ((a, a), (b, a)):
            with self.assertRaises(InvalidInput):
                exact.PolynomialFactorization(1, factors)

    def test_checker_revalidates_frozen_record_canonicality(self):
        factor = exact.PolynomialFactor(self.p(1, 1), 1)
        forged = object.__new__(exact.PolynomialFactorization)
        object.__setattr__(forged, "coefficient", Rational(1))
        object.__setattr__(forged, "factors", (factor, factor))
        evidence = ("qq-factorization-v1", (("linear",), ("linear",)))
        self.assertFalse(exact.check_factor(self.p(1, 2, 1), forged, evidence))
        left, right = exact.PolynomialFactor(self.p(1, 1), 1), exact.PolynomialFactor(self.p(2, 1), 1)
        object.__setattr__(forged, "factors", (right, left))
        self.assertFalse(exact.check_factor(self.p(2, 3, 1), forged, evidence))

    def test_seeded_products_have_all_expected_irreducible_factors(self):
        randomizer = random.Random(2026101406)
        pool = [self.p(c, 1) for c in range(-3, 4)] + [self.p(c, 0, 1) for c in range(1, 5)]
        for case in range(20):
            chosen = randomizer.sample(pool, 3)
            multiplicities = [randomizer.randrange(1, 3) for _ in chosen]
            coefficient = Rational(randomizer.randrange(1, 9), randomizer.randrange(1, 9))
            f = self.p(coefficient)
            for factor, multiplicity in zip(chosen, multiplicities):
                f = f * factor**multiplicity
            with self.subTest(case=case):
                value, _ = self.factor(f)
                self.assertEqual(value.coefficient, coefficient)
                self.assertEqual({v.polynomial: v.multiplicity for v in value.factors},
                                 dict(zip(chosen, multiplicities)))

    def test_resultant_signs_shared_roots_and_constants(self):
        cases = (
            (self.p(1, 2), self.p(3, 4), 2),
            (self.p(3, 4), self.p(1, 2), -2),
            (self.p(1, 2, 3), self.p(3, 4), 19),
            (self.p(-1, 1), self.p(2, 0, 0, 1), 3),
            (self.p(0), self.p(5), 0),
            (self.p(7), self.p(5), 1),
            (self.p(7), self.p(1, 2, 3), 49),
            (self.p(1, 2, 3), self.p(5), 25),
            (self.p(-1, 0, 1), self.p(-1, 1), 0),
        )
        for f, g, expected in cases:
            with self.subTest(f=f, g=g):
                actual = exact.resultant(f, g)
                self.assertEqual(actual, Rational(expected))
                self.assertTrue(exact.check_resultant(f, g, actual))
                self.assertFalse(exact.check_resultant(f, g, actual + 1))

    def test_seeded_resultants_match_independent_sylvester_determinants(self):
        rng = random.Random(211)
        for case in range(50):
            f = self.p(*(Rational(rng.randrange(-5, 6), rng.randrange(1, 4)) for _ in range(rng.randrange(1, 6))))
            g = self.p(*(Rational(rng.randrange(-5, 6), rng.randrange(1, 4)) for _ in range(rng.randrange(1, 5))))
            with self.subTest(case=case):
                value = exact.resultant(f, g)
                self.assertTrue(exact.check_resultant(f, g, value))
                self.assertEqual(value, (-1)**(max(f.degree, 0)*max(g.degree, 0)) * exact.resultant(g, f))

    def test_discriminant_formulas_repetition_and_constant_convention(self):
        cases = ((self.p(0), 0), (self.p(8), 0), (self.p(3, 7), 1),
                 (self.p(2, 3, 4), -23), (self.p(-2, 0, 0, 1), -108),
                 (self.p(1, 1)**3, 0), (self.p(1, 0, 0, 0, 1), 256))
        for f, expected in cases:
            value = exact.discriminant(f)
            self.assertEqual(value, Rational(expected))
            self.assertTrue(exact.check_discriminant(f, value))
            self.assertFalse(exact.check_discriminant(f, value + 1))

    def test_elimination_checkers_do_not_call_producing_algorithms(self):
        f, g = self.p(1, 2, 0, 3), self.p(-1, 0, 1)
        r, d = exact.resultant(f, g), exact.discriminant(f)
        with patch.object(exact, "resultant", side_effect=AssertionError("producer called")), \
             patch.object(exact, "discriminant", side_effect=AssertionError("producer called")):
            self.assertTrue(exact.check_resultant(f, g, r))
            self.assertTrue(exact.check_discriminant(f, d))

    def test_shared_limits_bound_production_verification_and_evidence(self):
        f = self.p(1, 0, 0, 0, 1)
        value, evidence = self.factor(f)
        for action in (
            lambda: exact.factor_with_evidence(f, limits=ComputationLimits(max_work=2)),
            lambda: exact.check_factor(f, value, evidence, limits=ComputationLimits(max_work=2)),
            lambda: exact.check_factor(f, value, evidence, limits=ComputationLimits(max_evidence_items=2)),
            lambda: exact.factor_with_evidence(f, limits=ComputationLimits(max_matrix_entries=3)),
            lambda: exact.check_resultant(f, f.derivative(), Rational(1), limits=ComputationLimits(max_matrix_entries=3)),
            lambda: exact.resultant(self.p(1 << 40, 1), self.p(1, 1), limits=ComputationLimits(max_integer_bits=16)),
        ):
            with self.assertRaises(ResourceLimitError):
                action()
        with computation(ComputationLimits(max_work=2)):
            with self.assertRaises(ResourceLimitError):
                exact.factor_with_evidence(f, limits=ComputationLimits(max_work=100000))

    def test_operation_dispatch_and_variable_identity(self):
        f, g = self.p(1, 0, 1), self.p(-1, 1)
        for name, args in (("factor", {"f": f}), ("resultant", {"f": f, "g": g}),
                           ("discriminant", {"f": f})):
            value, evidence = exact.produce(name, args)
            self.assertTrue(exact.check_operation(name, args, value, evidence))
        other = Polynomial((1, 1), symbol("x"))
        with self.assertRaises(InvalidInput):
            exact.resultant(f, other)
        with self.assertRaises(InvalidInput):
            exact.factor_with_evidence(17)


if __name__ == "__main__":
    unittest.main()
