"""P03 root isolation, exact comparison and independent certificate checks."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

from mathforge.algebraic import (RealAlgebraicRoot, RootRecord, as_endpoint,
                                 compare_real, interval_for, rational_between, sign_at)
from mathforge.errors import InvalidInput, MathForgeError, UnsupportedOperation
from mathforge.limits import ComputationLimits
from mathforge.model import Rational, symbol, sqrt
from mathforge.polynomial import Polynomial
from mathforge.root_isolation import (ROOT_VARIABLE, _simplest_open_fraction,
    count_real_roots, isolate_real_roots, real_roots, sturm_sequence,
    reconstruct_rational_root, verify_isolator, verify_root_records)


class RootV2Tests(unittest.TestCase):
    def setUp(self):
        self.x = symbol("x")

    def p(self, expression):
        return Polynomial.from_expression(expression, self.x)

    def test_rational_cubic_and_repeated_multiplicities(self):
        """ROOT-02/05: rational output and independent multiplicities."""
        self.assertEqual(real_roots(self.p(self.x**3-self.x)),
                         tuple(RootRecord(Rational(i), 1) for i in (-1, 0, 1)))
        polynomial = self.p((self.x-1)**4*(self.x+2)**3)
        roots = real_roots(polynomial)
        self.assertEqual(roots, (RootRecord(Rational(-2), 3), RootRecord(Rational(1), 4)))
        self.assertTrue(verify_root_records(polynomial, roots))

    def test_irrational_cubic_quintic_and_no_real_roots(self):
        """ROOT-03/04/07/08: complete exact higher-degree and constant results."""
        for expression in (self.x**3-2, self.x**5-self.x-1):
            polynomial = self.p(expression)
            roots = real_roots(polynomial)
            self.assertEqual(len(roots), 1)
            self.assertIsInstance(roots[0].root, RealAlgebraicRoot)
            self.assertTrue(verify_root_records(polynomial, roots))
        self.assertEqual(real_roots(self.p(self.x**4+1)), ())
        self.assertEqual(real_roots(self.p(7)), ())
        with self.assertRaises(InvalidInput) as captured:
            real_roots(self.p(0))
        self.assertEqual(captured.exception.code, "non_finite_root_set")

    def test_sturm_sign_and_endpoint_conventions(self):
        """ROOT-10–12: signed remainders and roots on exact bounds."""
        polynomial = self.p(self.x**2+1)
        self.assertEqual(sturm_sequence(polynomial)[-1].coefficients, (Rational(-1),))
        self.assertEqual(count_real_roots(polynomial), 0)
        polynomial = self.p(self.x*(self.x-1))
        self.assertEqual(count_real_roots(polynomial, 0, 1), 0)
        self.assertEqual(count_real_roots(polynomial, 0, 1, include_lower=True), 1)
        self.assertEqual(count_real_roots(polynomial, 0, 1, include_upper=True), 1)
        self.assertEqual(count_real_roots(polynomial, 0, 1, include_lower=True, include_upper=True), 2)
        self.assertEqual(count_real_roots(polynomial, 0, 0, include_lower=True, include_upper=True), 1)
        self.assertEqual(count_real_roots(polynomial, 0, 0), 0)
        intervals = isolate_real_roots(self.p(self.x**3-self.x))
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[1], (Rational(0), Rational(0)))

    def test_large_denominator_reconstruction(self):
        """ROOT-14: rational recognition is not a midpoint heuristic."""
        polynomial = self.p((1000003*self.x-1)*(self.x**2-2))
        roots = real_roots(polynomial)
        self.assertEqual(roots[1].root, Rational(1, 1000003))
        self.assertEqual(compare_real(roots[0].root, -sqrt(2)), 0)
        self.assertEqual(compare_real(roots[2].root, sqrt(2)), 0)

    def test_same_root_different_polynomials_and_quadratic_fields(self):
        """ROOT-06/15/24: structural identity is not mathematical identity."""
        a = RealAlgebraicRoot((-2, 0, 1), 1)
        b = RealAlgebraicRoot((6, -2, -3, 1), 1)
        self.assertNotEqual(a, b)
        self.assertEqual(compare_real(a, b), 0)
        self.assertEqual(compare_real(a, sqrt(8)/2), 0)
        roots = real_roots(self.p(self.x**4-5*self.x**2+6))
        for root, expected in zip(roots, (-sqrt(3), -sqrt(2), sqrt(2), sqrt(3))):
            self.assertEqual(compare_real(root.root, expected), 0)

    def test_close_distinct_irrational_roots(self):
        """ROOT-13: overlapping brackets do not imply equal roots."""
        polynomial = self.p((self.x**2-2)*((100*self.x-1)**2-20000))
        roots = real_roots(polynomial)
        expected = (-sqrt(2), -sqrt(2)+Rational(1, 100), sqrt(2), sqrt(2)+Rational(1, 100))
        self.assertEqual(len(roots), 4)
        for actual, target in zip(roots, expected):
            self.assertEqual(compare_real(actual.root, target), 0)
        self.assertTrue(verify_root_records(polynomial, roots))

    def test_sign_and_singleton_descriptors(self):
        """ROOT-19: gcd membership plus exact nonzero sign."""
        root = RealAlgebraicRoot((-2, 0, 1), 1)
        for expression, expected in ((self.x**2-2, 0), (self.x-1, 1), (self.x-2, -1)):
            self.assertEqual(sign_at(self.p(expression), root), expected)
        rational_root = RealAlgebraicRoot((-1, 3), 0)
        self.assertEqual(compare_real(rational_root, Rational(1, 3)), 0)
        self.assertEqual(sign_at(self.p(3*self.x-1), rational_root), 0)
        self.assertEqual(as_endpoint(sqrt(2)), root)

    def test_descriptor_normalization_index_and_identity(self):
        """ROOT-16–18: immutable normalized descriptor and valid real index."""
        a = RealAlgebraicRoot((4, 0, -4, 0, 1), 1)
        b = RealAlgebraicRoot((-2, 0, 1), 1)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        original_hash = hash(a)
        interval_for(a)
        self.assertEqual(hash(a), original_hash)
        for index in (-1, True, 2):
            with self.assertRaises(InvalidInput):
                RealAlgebraicRoot((-2, 0, 1), index)
        with self.assertRaises(InvalidInput):
            RealAlgebraicRoot((1, 0, 1), 0)

    def test_comparison_checks_scope_before_equality(self):
        """ROOT-25: unsupported constants cannot pass via x == x."""
        for value in (self.x, sqrt(2)+sqrt(3)):
            with self.assertRaises(UnsupportedOperation):
                compare_real(value, value)

    def test_tampered_records_and_isolator_index(self):
        """ROOT-20/21/23: exact membership alone cannot prove completeness."""
        polynomial = self.p(self.x**3-self.x)
        records = real_roots(polynomial)
        self.assertFalse(verify_root_records(polynomial, records[:-1]))
        self.assertFalse(verify_root_records(polynomial, (records[0],)*3))
        self.assertFalse(verify_root_records(polynomial, (replace(records[0], multiplicity=2),)+records[1:]))
        negative = RealAlgebraicRoot((-2, 0, 1), 0)
        positive = RealAlgebraicRoot((-2, 0, 1), 1)
        self.assertTrue(verify_isolator(negative, interval_for(negative)))
        self.assertFalse(verify_isolator(positive, interval_for(negative)))
        with patch("mathforge.root_isolation.real_roots", side_effect=AssertionError("producer called")):
            self.assertTrue(verify_root_records(polynomial, records))

    def test_rational_samples_for_algebraic_and_infinite_cells(self):
        """SET-11/ROOT-13: samples lie strictly inside every exact cell."""
        for left, right in ((None, None), (None, -sqrt(2)), (sqrt(2), None),
                            (sqrt(2), sqrt(2)+Rational(1, 1000)), (1, 2)):
            sample = rational_between(left, right)
            self.assertIsInstance(sample, Rational)
            if left is not None:
                self.assertEqual(compare_real(left, sample), -1)
            if right is not None:
                self.assertEqual(compare_real(sample, right), -1)

    def test_seeded_minimum_denominator_reconstruction(self):
        """ROOT-14: independent brute oracle on 100 small intervals."""
        rng = random.Random(2026100903)
        for case in range(100):
            a = Rational(rng.randrange(-20, 21), rng.randrange(1, 16))
            b = Rational(rng.randrange(-20, 21), rng.randrange(1, 16))
            if a == b:
                b = b+1
            lower, upper = min(a, b), max(a, b)
            candidate = _simplest_open_fraction(lower, upper)
            with self.subTest(case=case, lower=lower, upper=upper):
                self.assertLess(lower, candidate)
                self.assertLess(candidate, upper)
                for denominator in range(1, candidate.denominator):
                    first_numerator = lower.numerator*denominator//lower.denominator+1
                    self.assertFalse(Rational(first_numerator, denominator) < upper)

    def test_seeded_real_root_counts(self):
        """ROOT-05: 100 known linear-factor root/multiplicity oracles."""
        rng = random.Random(2026100904)
        for case in range(100):
            values = sorted(rng.sample(range(-8, 9), rng.randrange(1, 5)))
            polynomial = self.p(rng.choice((-3, 2)))
            expected = []
            for value in values:
                multiplicity = rng.randrange(1, 4)
                polynomial = polynomial*self.p(self.x-value)**multiplicity
                expected.append(RootRecord(Rational(value), multiplicity))
            with self.subTest(case=case):
                actual = real_roots(polynomial)
                self.assertEqual(actual, tuple(expected))
                self.assertTrue(verify_root_records(polynomial, actual))

    def test_root_work_limit_is_not_an_empty_root_set(self):
        """ROOT-22/LIM-02: tiny root budgets fail without a mathematical claim."""
        with self.assertRaises(MathForgeError) as captured:
            real_roots(self.p(self.x**5-self.x-1), limits=ComputationLimits(max_work=3))
        self.assertEqual(captured.exception.code, "resource_limit_exceeded")

    def test_high_degree_root_count_and_sparse_producer(self):
        """POL-15/ROOT-04: no hidden small-degree cap in the new root core."""
        self.assertEqual(count_real_roots(self.p(self.x**1024+1)), 0)
        polynomial = self.p(self.x**17-2)
        records = real_roots(polynomial)
        self.assertEqual(len(records), 1)
        self.assertTrue(verify_root_records(polynomial, records))

    def test_independent_comparison_refinement_limit(self):
        """LIM-02: close roots cannot pass from an exhausted isolator."""
        left = RealAlgebraicRoot((-2, 0, 1), 1)
        right = sqrt(2)+Rational(1, 10**12)
        with self.assertRaises(MathForgeError) as captured:
            compare_real(left, right, limits=ComputationLimits(max_refinements=1))
        self.assertEqual(captured.exception.code, "resource_limit_exceeded")

    def test_reconstruction_open_rational_endpoints(self):
        """ROOT-12/14: CF reconstruction must not choose an excluded bound."""
        for lower, upper, expected in ((Rational(1, 3), Rational(1, 2), Rational(2, 5)),
                                        (Rational(-1, 2), Rational(-1, 3), Rational(-2, 5)),
                                        (Rational(0), Rational(1), Rational(1, 2)),
                                        (Rational(2), Rational(3), Rational(5, 2))):
            self.assertEqual(_simplest_open_fraction(lower, upper), expected)

    def test_reconstruction_itself_consumes_the_refinement_budget(self):
        """ROOT-22: reconstruction exhaustion cannot fall back to RootOf."""
        polynomial = self.p((1000003*self.x-1)*(self.x**2-2))
        interval = isolate_real_roots(polynomial)[1]
        self.assertNotEqual(interval[0], interval[1])
        with self.assertRaises(MathForgeError) as captured:
            reconstruct_rational_root(polynomial, interval, limits=ComputationLimits(max_refinements=1))
        self.assertEqual(captured.exception.code, "resource_limit_exceeded")

    def test_source_symbol_uuid_is_not_root_identity(self):
        """ROOT-16/17: different source variables and refinement preserve identity."""
        y = symbol("x")
        first = real_roots(self.p(self.x**3-2))[0].root
        second = real_roots(Polynomial.from_expression(y**3-2, y))[0].root
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        old_hash = hash(first)
        self.assertEqual(compare_real(first, Rational(126, 100)), -1)
        self.assertEqual(hash(first), old_hash)

    def test_legacy_small_degree_and_identity_equation_results(self):
        """ROOT-01/09: old root forms and zero/constant equation classifications."""
        import mathforge as mf
        self.assertEqual(real_roots(self.p(-3*self.x+3)), (RootRecord(Rational(1), 1),))
        self.assertEqual(tuple(r.root for r in real_roots(self.p(-self.x**2+2))), (-sqrt(2), sqrt(2)))
        for expression, expected_type in ((0, mf.UniversalSet), (3, mf.EmptySet)):
            result = mf.solve(mf.Eq(expression, 0), for_=self.x)
            self.assertIsInstance(result.value, expected_type)
            self.assertTrue(mf.verify(result).verified)


if __name__ == "__main__":
    unittest.main()
