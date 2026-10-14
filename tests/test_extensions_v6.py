"""Exact decomposition/spectrum identities, fault injection and bounded work."""

from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from random import Random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import exact_extensions as ex
from mathforge import multivariate as mv
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow
from mathforge.polynomial import Polynomial
from mathforge.root_isolation import ROOT_VARIABLE


def permutation_determinant(A, point):
    total = mf.Rational(0)
    for order in permutations(range(A.nrows)):
        inversions = sum(order[i] > order[j] for i in range(A.nrows) for j in range(i + 1, A.nrows))
        product = mf.Rational((-1) ** inversions)
        for row, column in enumerate(order):
            product *= (point if row == column else mf.Rational(0)) - A[row, column]
        total += product
    return total


class ExactExtensionTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000610")

    def polynomial(self, coefficients):
        return Polynomial(tuple(coefficients), self.x)

    def rf(self, numerator, denominator):
        from mathforge.rational_function import rational_function
        return rational_function(numerator, denominator, variable=self.x)

    def test_berkowitz_convention_empty_triangular_and_rational_matrices(self):
        cases = (
            (mf.Matrix(()), (1,)),
            (mf.Matrix(((3,),)), (-3, 1)),
            (mf.Matrix(((1, 5), (0, 2))), (2, -3, 1)),
            (mf.Matrix(((0, -1), (1, 0))), (1, 0, 1)),
            (mf.Matrix(((0, 1, 0), (0, 0, 1), (0, 0, 0))), (0, 0, 0, 1)),
            (mf.Matrix(((mf.Rational(1, 2), 1), (2, mf.Rational(1, 3)))),
             (mf.Rational(-11, 6), mf.Rational(-5, 6), 1)),
        )
        for A, expected in cases:
            with self.subTest(A=A):
                value, evidence = ex.produce("charpoly", {"A": A, "variable": self.x})
                self.assertEqual(value, self.polynomial(expected))
                self.assertEqual(evidence, ())
                self.assertTrue(ex.check_operation("charpoly", {"A": A, "variable": self.x}, value, evidence))

    def test_generated_charpolys_against_leibniz_identity(self):
        random = Random(2026101461)
        for index in range(30):
            n = random.randrange(5)
            A = mf.Matrix(tuple(tuple(mf.Rational(random.randint(-3, 3), random.randint(1, 3))
                                      for _ in range(n)) for _ in range(n)))
            value, evidence = ex.produce("charpoly", {"A": A, "variable": self.x})
            with self.subTest(index=index, A=A):
                self.assertEqual(value.degree, n)
                self.assertEqual(value.leading_coefficient, mf.Rational(1))
                for point in range(-2, n - 1):
                    self.assertEqual(value.evaluate(point), permutation_determinant(A, mf.Rational(point)))
                self.assertTrue(ex.check_operation("charpoly", {"A": A, "variable": self.x}, value, evidence))

    def test_charpoly_checker_is_independent_and_binds_matrix_and_variable(self):
        A = mf.Matrix(((2, 1), (3, 4)))
        args = {"A": A, "variable": self.x}
        value, evidence = ex.produce("charpoly", args)
        forbidden = AssertionError("Producer called by checker")
        with (patch.object(ex, "produce", side_effect=forbidden),
              patch.object(ex, "_charpoly", side_effect=forbidden),
              patch("mathforge.matrices.det", side_effect=forbidden),
              patch("mathforge.matrices.reduce_with_trace", side_effect=forbidden)):
            self.assertTrue(ex.check_operation("charpoly", args, value, evidence))
        self.assertFalse(ex.check_operation("charpoly", args, value + 1, evidence))
        self.assertFalse(ex.check_operation("charpoly", args, value, ("invented",)))
        self.assertFalse(ex.check_operation("charpoly", {**args, "A": mf.Matrix(((2, 0), (0, 4)))}, value, evidence))
        renamed = mf.Symbol("other", uid=self.x.uid)
        self.assertFalse(ex.check_operation("charpoly", {**args, "variable": renamed}, value, evidence))
        # Monic annihilating degree n is insufficient to establish a charpoly.
        zero = mf.Matrix(((0, 0), (0, 0)))
        self.assertFalse(ex.check_operation("charpoly", {"A": zero, "variable": self.x},
                                            self.polynomial((0, 1, 1)), ()))

    def test_real_spectrum_includes_multiplicity_and_can_be_empty(self):
        cases = ((mf.Matrix(()), ()), (mf.Matrix(((0, -1), (1, 0))), ()),
                 (mf.Matrix(((2, 1, 0), (0, 2, 0), (0, 0, -1))),
                  ((mf.Rational(-1), 1), (mf.Rational(2), 2))))
        for A, expected in cases:
            with self.subTest(A=A):
                value, evidence = ex.produce("real_eigenvalues", {"A": A})
                self.assertEqual(tuple((record.root, record.multiplicity) for record in value), expected)
                self.assertTrue(ex.check_operation("real_eigenvalues", {"A": A}, value, evidence))

    def test_irrational_spectrum_checker_rejects_missing_and_forged_roots(self):
        A = mf.Matrix(((0, 2), (1, 0)))
        value, evidence = ex.produce("real_eigenvalues", {"A": A})
        self.assertEqual(len(value), 2)
        self.assertEqual(mf.compare_real(value[0].root, -mf.sqrt(2)), 0)
        self.assertEqual(mf.compare_real(value[1].root, mf.sqrt(2)), 0)
        self.assertTrue(ex.check_operation("real_eigenvalues", {"A": A}, value, evidence))
        forbidden = AssertionError("Producer called by checker")
        with (patch.object(ex, "_charpoly", side_effect=forbidden),
              patch.object(ex, "_real_eigenvalues", side_effect=forbidden),
              patch("mathforge.root_isolation.real_roots", side_effect=forbidden),
              patch("mathforge.root_certificates.root_certificate", side_effect=forbidden)):
            self.assertTrue(ex.check_operation("real_eigenvalues", {"A": A}, value, evidence))
        for changed in (value[:-1], value[::-1], (replace(value[0], multiplicity=2), value[1])):
            self.assertFalse(ex.check_operation("real_eigenvalues", {"A": A}, changed, evidence))
        polynomial, root_evidence = evidence
        self.assertFalse(ex.check_operation("real_eigenvalues", {"A": A}, value,
                                            (polynomial + 1, root_evidence)))
        self.assertFalse(ex.check_operation("real_eigenvalues", {"A": A}, value,
                                            (Polynomial(polynomial.coefficients, self.x), root_evidence)))

    def test_apart_repeated_linear_and_irreducible_denominators(self):
        x = self.x
        source = self.rf(2*x + 3, (x+1)**2)
        value, evidence = ex.produce("apart", {"expression": source})
        factor = self.polynomial((1, 1))
        self.assertEqual(value.polynomial, self.polynomial((0,)))
        self.assertEqual(value.terms, (ex.PartialFractionTerm(self.polynomial((2,)), factor, 1),
                                       ex.PartialFractionTerm(self.polynomial((1,)), factor, 2)))
        self.assertEqual(value.to_rational_function(), source)
        self.assertTrue(ex.check_operation("apart", {"expression": source}, value, evidence))
        source = self.rf(x+2, (x*x+1)**2)
        value, evidence = ex.produce("apart", {"expression": source})
        self.assertEqual(value.terms, (ex.PartialFractionTerm(self.polynomial((2, 1)),
                                                              self.polynomial((1, 0, 1)), 2),))
        self.assertEqual(value.to_rational_function(), source)
        self.assertTrue(ex.check_operation("apart", {"expression": source}, value, evidence))

    def test_apart_preserves_holes_improper_parts_constants_and_zero(self):
        x = self.x
        cases = (self.rf((x-5)*(x**5+2), (x-5)*(x-1)**2*(x*x+1)),
                 self.rf((x-2)*(x+1), x-2), self.rf(0, x-2), self.rf(7, 1))
        for source in cases:
            with self.subTest(source=source):
                value, evidence = ex.produce("apart", {"expression": source})
                self.assertEqual(value.excluded, source.excluded)
                self.assertEqual(value.to_rational_function(), source)
                self.assertTrue(ex.check_operation("apart", {"expression": source}, value, evidence))
        source = cases[0]
        value, evidence = ex.produce("apart", {"expression": source})
        with self.assertRaises(FrozenInstanceError):
            value.excluded = ()
        self.assertFalse(ex.check_operation("apart", {"expression": source}, replace(value, excluded=()), evidence))

    def test_apart_checker_never_calls_factor_or_decomposition_producers(self):
        x = self.x
        source = self.rf(x**4 + 2*x + 3, (x-1)**2*(x*x+1))
        value, evidence = ex.produce("apart", {"expression": source})
        forbidden = AssertionError("Producer called by checker")
        with (patch.object(ex, "produce", side_effect=forbidden), patch.object(ex, "_apart", side_effect=forbidden),
              patch("mathforge.polynomial_exact.factor_with_evidence", side_effect=forbidden),
              patch.object(Polynomial, "xgcd", side_effect=forbidden)):
            self.assertTrue(ex.check_operation("apart", {"expression": source}, value, evidence))
        self.assertFalse(ex.check_operation("apart", {"expression": source},
                                            replace(value, polynomial=value.polynomial+1), evidence))
        self.assertFalse(ex.check_operation("apart", {"expression": source},
                                            replace(value, terms=value.terms[:-1]), evidence))
        self.assertFalse(ex.check_operation("apart", {"expression": source}, value, evidence[:-1]))

    def test_record_shapes_and_inputs_are_strict(self):
        factor = self.polynomial((1, 1))
        for numerator, power in ((self.polynomial((0,)), 1), (factor, 1), (self.polynomial((1,)), True)):
            with self.assertRaises(mf.InvalidInput):
                ex.PartialFractionTerm(numerator, factor, power)
        term = ex.PartialFractionTerm(self.polynomial((1,)), factor, 1)
        with self.assertRaises(mf.InvalidInput):
            ex.PartialFractionDecomposition(self.polynomial((0,)), (term, term), ())
        with self.assertRaises(mf.InvalidInput):
            ex.produce("apart", {"expression": self.x})
        for A in (mf.Matrix(((1, 2),)), (), [[1]]):
            with self.assertRaises(mf.InvalidInput):
                ex.produce("charpoly", {"A": A, "variable": self.x})
        self.assertFalse(ex.check_operation("unknown", {}, 0, ()))
        self.assertFalse(ex.check_operation("charpoly", {"A": mf.Matrix(()), "variable": self.x, "extra": 1},
                                            self.polynomial((1,)), ()))

    def test_work_and_dimension_limits_apply_to_producers_and_checkers(self):
        A = mf.Matrix(((1, 2), (3, 4)))
        source = self.rf(1, (self.x-1)*(self.x+1))
        for operation, args in (("apart", {"expression": source}),
                                ("charpoly", {"A": A, "variable": self.x}),
                                ("real_eigenvalues", {"A": A})):
            value, evidence = ex.produce(operation, args)
            with self.subTest(operation=operation):
                with computation(ComputationLimits(max_work=1), fresh=True), self.assertRaises(ResourceLimitError):
                    ex.produce(operation, args)
                with computation(ComputationLimits(max_work=1), fresh=True), self.assertRaises(ResourceLimitError):
                    ex.check_operation(operation, args, value, evidence)
        with computation(ComputationLimits(max_matrix_entries=3), fresh=True), self.assertRaises(ResourceLimitError):
            ex.produce("charpoly", {"A": A, "variable": self.x})


class IndependentMultivariateSourceTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000611")
        self.y = mf.Symbol("y", uid="00000000-0000-4000-8000-000000000612")

    def test_faulty_shared_converter_cannot_certify_zero_expansion(self):
        expression = (self.x + 1)**2
        args = {"expression": expression, "variables": (self.x,)}
        with patch.object(mv, "_convert_many", return_value=({},)):
            forged = mv.expand_multivariate(expression, (self.x,))
            self.assertEqual(forged, mf.Rational(0))
            self.assertFalse(mv.check_operation("expand_multivariate", args, forged, ()))
            result = mf.operations.expand_multivariate(expression, (self.x,))
            self.assertIs(result.execution_status, mf.ExecutionStatus.ERROR)
            self.assertIsNone(result.value)

    def test_source_identity_checker_uses_no_producer_conversion_helpers(self):
        expression = (mf.Rational(1, 3)*self.x+self.y)**5
        args = {"expression": expression, "variables": (self.x, self.y)}
        value = mv.expand_multivariate(expression, (self.x, self.y))
        forbidden = AssertionError("Producer conversion called by checker")
        with (patch.object(mv, "_convert_many", side_effect=forbidden),
              patch.object(mv, "_product", side_effect=forbidden),
              patch.object(mv, "_accumulate", side_effect=forbidden),
              patch.object(mv, "_to_expression", side_effect=forbidden),
              patch.object(mv, "_differentiate", side_effect=forbidden)):
            self.assertTrue(mv.check_operation("expand_multivariate", args, value, ()))

    def test_generated_nested_source_identities_match_independent_engine(self):
        random = Random(2026101462)
        axes = (self.x, self.y)

        def tree(depth):
            if depth == 0:
                return random.choice((self.x, self.y, mf.Rational(random.randrange(-3, 4), random.randrange(1, 5))))
            choice = random.randrange(3)
            if choice == 0:
                return Add((tree(depth-1), tree(depth-1)))
            if choice == 1:
                return Mul((tree(depth-1), tree(depth-1)))
            return Pow(tree(depth-1), random.randrange(4))

        for index in range(40):
            expression = tree(3)
            value = mv.expand_multivariate(expression, axes)
            with self.subTest(index=index):
                self.assertTrue(mv.check_operation("expand_multivariate",
                                                   {"expression": expression, "variables": axes}, value, ()))
                self.assertFalse(mv.check_operation("expand_multivariate",
                                                    {"expression": expression, "variables": axes}, value+1, ()))


if __name__ == "__main__":
    unittest.main()
