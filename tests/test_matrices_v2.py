"""Exact linear algebra acceptance tests; production algorithms are never oracles."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import permutations
from random import Random
import unittest
from unittest.mock import patch

from mathforge.contracts import EmptySet, Eq
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.matrices import (
    AffineSolutionSet, LinearSystem, Matrix, RREFResult,
    check_determinant, check_nullspace, check_rank, det, inverse, is_rref,
    linear_system_from_equations, nullspace, rank, reduce_with_trace,
    replay_row_operations, rref, solve_linear_system, verify_linear_solution,
    verify_row_reduction,
)
from mathforge.model import Assumption, Integers, Rational, Symbol, Truth, sqrt


def qvector(values):
    return tuple(value if type(value) is Rational else Rational(value) for value in values)


def leibniz(A):
    total = Fraction(0)
    for perm in permutations(range(A.nrows)):
        inversions = sum(perm[i] > perm[j] for i in range(A.nrows)
                         for j in range(i + 1, A.nrows))
        product = Fraction((-1) ** inversions)
        for i, j in enumerate(perm):
            value = A[i, j]
            product *= Fraction(value.numerator, value.denominator)
        total += product
    return Rational(total.numerator, total.denominator)


class MatrixValueTests(unittest.TestCase):
    def test_exact_normalized_immutable_rows(self):
        """LIN-01: Normalize integers; freeze nested rows; reject inexact entries."""
        source = [[1, Rational(2, 4)], [3, -4]]
        A = Matrix(source)
        source[0][0] = 100
        self.assertEqual(A.rows, (qvector((1, Rational(1, 2))), qvector((3, -4))))
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(A[0, 1], Rational(1, 2))
        self.assertEqual(A[-1], qvector((3, -4)))
        self.assertEqual(hash(A), hash(Matrix(A.rows)))
        with self.assertRaises(FrozenInstanceError):
            A.ncols = 5
        for values in (((1,), (1, 2)), ((0.1,),), ((True,),), ((Symbol("x"),),), "123"):
            with self.subTest(values=values), self.assertRaises((InvalidInput, TypeError)):
                Matrix(values)

    def test_empty_shapes_and_transpose(self):
        """LIN-02 LIN-05: All empty shapes retain dimensions under transpose."""
        for A, expected in ((Matrix(()), (0, 0)), (Matrix((), ncols=3), (0, 3)),
                            (Matrix(((), ())), (2, 0))):
            self.assertEqual(A.shape, expected)
            self.assertEqual(A.T.shape, expected[::-1])
            self.assertEqual(A.T.T, A)
        for columns in (-1, True, 1.5):
            with self.assertRaises(InvalidInput):
                Matrix((), ncols=columns)
        with self.assertRaises(InvalidInput):
            Matrix(((1, 2),), ncols=1)

    def test_arithmetic_and_scalar_directions(self):
        """LIN-04 LIN-05: @ is composition; both scalar orders remain exact."""
        A = Matrix(((1, 2), (3, 4)))
        B = Matrix(((2, 0), (1, 2)))
        self.assertEqual(A @ B, Matrix(((4, 4), (10, 8))))
        self.assertEqual(A + B, Matrix(((3, 2), (4, 6))))
        self.assertEqual(A - B, Matrix(((-1, 2), (2, 2))))
        self.assertEqual(2 * A, A * 2)
        self.assertEqual(Rational(1, 2) * A, A * Rational(1, 2))
        self.assertEqual(3 * (A + B), 3 * A + 3 * B)
        self.assertEqual(-A, A * -1)
        self.assertEqual(A.T.T, A)
        with self.assertRaises(InvalidInput):
            A * B

    def test_shape_mismatches(self):
        """LIN-03: Shape errors cannot be silently coerced or broadcast."""
        A = Matrix(((1, 2),))
        B = Matrix(((1,),))
        for operation in (lambda: A + B, lambda: A - B, lambda: A @ B):
            with self.assertRaises(InvalidInput):
                operation()
        self.assertEqual(Matrix(((), ())) @ Matrix((), ncols=3),
                         Matrix(((0, 0, 0), (0, 0, 0))))
        self.assertEqual((Matrix((), ncols=2) @ Matrix(((1, 2, 3), (4, 5, 6)))).shape,
                         (0, 3))

    def test_foreign_subclasses_do_not_enter_values(self):
        """LIN-01: Foreign mutable subclasses are not transport values."""
        class ForeignRational(Rational):
            pass

        class ForeignMatrix(Matrix):
            pass

        with self.assertRaises(InvalidInput):
            Matrix(((ForeignRational(1),),))
        with self.assertRaises(InvalidInput):
            rank(ForeignMatrix(((1,),)))


class EliminationTests(unittest.TestCase):
    def test_row_swaps_and_zero_leading_columns(self):
        """LIN-06 LIN-07: Pivot order is deterministic and includes zero columns."""
        A = Matrix(((0, 0, 2, 4), (0, 3, 1, 5), (0, 0, 0, 0)))
        reduction, trace = reduce_with_trace(A)
        self.assertEqual(reduction.pivot_columns, (1, 2))
        self.assertEqual(reduction.matrix,
                         Matrix(((0, 1, 0, 1), (0, 0, 1, 2), (0, 0, 0, 0))))
        self.assertEqual(rank(A), 2)
        self.assertEqual(trace[0], ("swap", 0, 1))
        self.assertTrue(verify_row_reduction(A, reduction, trace))
        self.assertEqual(replay_row_operations(A, trace), reduction.matrix)
        self.assertEqual(rref(A), reduction)

    def test_determinants_against_independent_formula(self):
        """LIN-08 LIN-09: Bareiss agrees with Leibniz, including row scaling."""
        examples = (
            Matrix(()), Matrix(((Rational(3, 7),),)),
            Matrix(((0, 2), (3, 4))), Matrix(((1, 2), (2, 4))),
            Matrix(((Rational(1, 2), Rational(1, 3)),
                    (Rational(3, 5), Rational(-2, 7)))),
            Matrix(((2, 4, 6), (0, -3, 9), (0, 0, Rational(5, 2)))),
            Matrix(((0, 2, 1), (3, 0, 4), (5, 6, 0))),
        )
        for A in examples:
            with self.subTest(A=A):
                expected = leibniz(A)
                self.assertEqual(det(A), expected)
                self.assertTrue(check_determinant(A, expected))
                self.assertFalse(check_determinant(A, expected + 1))
        self.assertEqual(det(examples[-2]), Rational(-15))

    def test_inverse_identity_and_singular_shape_errors(self):
        """LIN-10 LIN-11 LIN-13: Exact inverse is two-sided; singular is input error."""
        A = Matrix(((0, 2), (3, 4)))
        B = inverse(A)
        self.assertEqual(A @ B, Matrix.identity(2))
        self.assertEqual(B @ A, Matrix.identity(2))
        self.assertEqual(B, Matrix(((Rational(-2, 3), Rational(1, 3)),
                                    (Rational(1, 2), 0))))
        with self.assertRaises(InvalidInput) as caught:
            inverse(Matrix(((1, 2), (2, 4))))
        self.assertEqual(caught.exception.code, "singular_matrix")
        for operation in (det, inverse):
            with self.assertRaises(InvalidInput):
                operation(Matrix(((1, 2),)))

    def test_empty_matrix_results(self):
        """LIN-12 LIN-18: Empty determinants and nullspaces use standard conventions."""
        empty = Matrix(())
        self.assertEqual(det(empty), Rational(1))
        self.assertEqual(inverse(empty), empty)
        self.assertEqual(rank(Matrix((), ncols=3)), 0)
        self.assertEqual(nullspace(Matrix((), ncols=3)),
                         (qvector((1, 0, 0)), qvector((0, 1, 0)), qvector((0, 0, 1))))
        self.assertEqual(nullspace(Matrix(((), ()))), ())

    def test_rank_nullity_and_nullspace_membership(self):
        """LIN-26: All directions, not merely some homogeneous solutions, are returned."""
        A = Matrix(((1, 2, 3), (2, 4, 6)))
        basis = nullspace(A)
        self.assertEqual(basis, (qvector((-2, 1, 0)), qvector((-3, 0, 1))))
        self.assertEqual(rank(A) + len(basis), A.ncols)
        self.assertTrue(check_nullspace(A, basis))
        self.assertFalse(check_nullspace(A, basis[:1]))
        self.assertFalse(check_nullspace(A, (basis[0], basis[0])))
        self.assertTrue(check_rank(A, 1))
        self.assertFalse(check_rank(A, True))
        self.assertFalse(check_rank(A, 2))

    def test_illegal_and_unrelated_traces_fail(self):
        """LIN-27 LIN-28 LIN-29: Replay rejects forged pivots, rows, and source linkage."""
        A = Matrix(((0, 1), (2, 3)))
        reduction, trace = reduce_with_trace(A)
        for appended in (("scale", 0, Rational(0)), ("add", 0, 0, Rational(2)),
                         ("swap", -1, 0), ("swap", True, 0), ("bogus", 0, 1),
                         ("add", 0, 1, 0.5), ("swap", 0)):
            self.assertFalse(verify_row_reduction(A, reduction, trace + (appended,)))
        self.assertFalse(verify_row_reduction(A + Matrix.identity(2), reduction, trace))
        self.assertFalse(is_rref(Matrix(((1, 0), (1, 1))), (0, 1)))
        self.assertFalse(is_rref(Matrix(((0, 0), (1, 0))), (0,)))
        self.assertFalse(is_rref(Matrix.identity(2), (0,)))
        self.assertFalse(is_rref(Matrix.identity(2), (False, 1)))

    def test_compact_row_trace(self):
        """LIN-30: Row evidence contains operations instead of intermediate matrices."""
        reduction, trace = reduce_with_trace(Matrix(((1, 2, 3), (4, 5, 6), (7, 8, 10))))
        self.assertTrue(trace)
        self.assertTrue(all(type(operation) is tuple and len(operation) <= 4 for operation in trace))
        self.assertFalse(any(isinstance(item, Matrix) for operation in trace for item in operation))
        self.assertEqual(reduction.matrix, Matrix.identity(3))

    def test_checkers_do_not_call_production_algorithms(self):
        """VER-08: Matrix checking remains functional with all production entrypoints disabled."""
        A = Matrix(((1, 2), (3, 4)))
        reduction, trace = reduce_with_trace(A)
        with patch("mathforge.matrices.reduce_with_trace", side_effect=AssertionError("producer")), \
                patch("mathforge.matrices.det", side_effect=AssertionError("producer")), \
                patch("mathforge.matrices.rank", side_effect=AssertionError("producer")):
            self.assertTrue(check_determinant(A, Rational(-2)))
            self.assertTrue(check_rank(A, 2))
            self.assertTrue(check_nullspace(A, ()))
            self.assertTrue(verify_row_reduction(A, reduction, trace))


class LinearSystemTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = Symbol("x"), Symbol("y")

    def test_unique_and_affine_families(self):
        """LIN-14 LIN-15: Particular solutions and standard free-coordinate bases are exact."""
        unique = LinearSystem(Matrix(((1, 1), (1, -1))), (3, 1), (self.x, self.y))
        solution, evidence = solve_linear_system(unique)
        self.assertEqual(solution, AffineSolutionSet((self.x, self.y), (2, 1), ()))
        self.assertTrue(verify_linear_solution(unique, solution, evidence))
        affine = LinearSystem(Matrix(((1, 1),)), (1,), (self.x, self.y))
        solution, evidence = solve_linear_system(affine)
        self.assertEqual(solution.particular, qvector((1, 0)))
        self.assertEqual(solution.basis, (qvector((-1, 1)),))
        self.assertTrue(verify_linear_solution(affine, solution, evidence))

    def test_inconsistent_and_overdetermined(self):
        """LIN-16 LIN-17: Redundant equations and real contradictions are distinguished."""
        for rhs, consistent in (((1, 2), False), ((1, 1), True)):
            system = LinearSystem(Matrix(((1, 1), (1, 1))), rhs, (self.x, self.y))
            solution, evidence = solve_linear_system(system)
            self.assertEqual(type(solution) is AffineSolutionSet, consistent)
            self.assertTrue(verify_linear_solution(system, solution, evidence))
        system = LinearSystem(Matrix(((1, 1), (1, -1), (2, 0))), (3, 1, 4), (self.x, self.y))
        solution, evidence = solve_linear_system(system)
        self.assertEqual(solution.particular, qvector((2, 1)))
        self.assertTrue(verify_linear_solution(system, solution, evidence))

    def test_empty_system_and_zero_unknowns(self):
        """LIN-18 LIN-19: R^n and the unique empty tuple are distinct correct cases."""
        system = LinearSystem(Matrix((), ncols=2), (), (self.x, self.y))
        solution, evidence = solve_linear_system(system)
        self.assertEqual(solution, AffineSolutionSet((self.x, self.y), (0, 0), ((1, 0), (0, 1))))
        self.assertTrue(verify_linear_solution(system, solution, evidence))
        for rhs, consistent in (((), True), ((0,), True), ((0, 1), False)):
            system = LinearSystem(Matrix((() for _ in rhs), ncols=0), rhs, ())
            solution, evidence = solve_linear_system(system)
            if consistent:
                self.assertEqual(solution, AffineSolutionSet((), (), ()))
            else:
                self.assertIsInstance(solution, EmptySet)
            self.assertTrue(verify_linear_solution(system, solution, evidence))

    def test_ordered_variables_and_same_name_identities(self):
        """LIN-20 LIN-21: Coordinates follow supplied UUID order, not display names."""
        equations = (Eq(self.x + 2 * self.y, 5), Eq(self.x - self.y, 2))
        forward = linear_system_from_equations(equations, (self.x, self.y))
        reverse = linear_system_from_equations(equations, (self.y, self.x))
        self.assertEqual(solve_linear_system(forward)[0].particular, qvector((3, 1)))
        self.assertEqual(solve_linear_system(reverse)[0].particular, qvector((1, 3)))
        other_x = Symbol("x")
        system = linear_system_from_equations((Eq(self.x + other_x, 3),), (self.x, other_x))
        self.assertEqual(system.coefficients, Matrix(((1, 1),)))
        with self.assertRaises(InvalidInput):
            LinearSystem(Matrix(((1, 1),)), (3,), (self.x, self.x))

    def test_sparse_cancellation_before_affine_degree_check(self):
        """LIN-22: Effective affine equations are accepted after nonlinear cancellation."""
        lhs = (self.x + self.y) ** 2 - self.x ** 2 - 2 * self.x * self.y - self.y ** 2 + self.x + self.y
        system = linear_system_from_equations((Eq(lhs, 7),), (self.x, self.y))
        self.assertEqual(system.coefficients, Matrix(((1, 1),)))
        self.assertEqual(system.rhs, qvector((7,)))
        solution, evidence = solve_linear_system(system)
        self.assertTrue(verify_linear_solution(system, solution, evidence))

    def test_unsupported_equations_and_domains(self):
        """LIN-23 LIN-24 LIN-25: Nonlinear, irrational, undeclared, or constrained input fails honestly."""
        for expression in (self.x * self.y, self.x + sqrt(2), self.x + Symbol("z")):
            with self.assertRaises(UnsupportedOperation):
                linear_system_from_equations((Eq(expression, 0),), (self.x, self.y))
        for variable in (Symbol("z", domain=Integers),
                         Symbol("z", assumptions=(Assumption("positive", Truth.TRUE),))):
            with self.assertRaises(UnsupportedOperation):
                LinearSystem(Matrix(((1,),)), (1,), (variable,))
        with self.assertRaises(InvalidInput):
            linear_system_from_equations((self.x + 1,), (self.x,))

    def test_dimensions_and_conflicting_symbol_metadata(self):
        """LIN-01 LIN-21: Shape or UUID metadata mismatches cannot enter a system."""
        for rhs, variables in (((), (self.x,)), ((1,), ()), ((1, 2), (self.x,))):
            with self.assertRaises(InvalidInput):
                LinearSystem(Matrix(((1,),)), rhs, variables)
        with self.assertRaises(InvalidInput):
            AffineSolutionSet((self.x,), (), ())
        with self.assertRaises(InvalidInput):
            AffineSolutionSet((self.x,), (0,), ((1, 2),))
        renamed = replace(self.x, name="renamed")
        with self.assertRaises(InvalidInput):
            linear_system_from_equations((Eq(renamed, 1),), (self.x,))

    def test_forged_affine_solutions_and_source_evidence(self):
        """LIN-27 LIN-29 VER-10: Membership, independent directions, nullity, and source all matter."""
        system = LinearSystem(Matrix(((1, 1),)), (1,), (self.x, self.y))
        solution, evidence = solve_linear_system(system)
        for forged in (replace(solution, particular=(0, 0)), replace(solution, basis=()),
                       replace(solution, basis=solution.basis * 2),
                       replace(solution, basis=((1, 1),)), EmptySet()):
            self.assertFalse(verify_linear_solution(system, forged, evidence))
        other_system = replace(system, rhs=(2,))
        self.assertFalse(verify_linear_solution(other_system, solution, evidence))
        wrong_trace = evidence[:-1] + (evidence[-1] + (("scale", 0, Rational(0)),),)
        self.assertFalse(verify_linear_solution(system, solution, wrong_trace))
        self.assertFalse(verify_linear_solution(system, solution, evidence[:3]))

    def test_system_checker_avoids_producer(self):
        """VER-08: Affine checks remain independent when solver entrypoints are disabled."""
        system = LinearSystem(Matrix(((1, 1),)), (1,), (self.x, self.y))
        solution, evidence = solve_linear_system(system)
        with patch("mathforge.matrices.solve_linear_system", side_effect=AssertionError("producer")), \
                patch("mathforge.matrices.reduce_with_trace", side_effect=AssertionError("producer")):
            self.assertTrue(verify_linear_solution(system, solution, evidence))


class MatrixLimitAndGeneratedTests(unittest.TestCase):
    def test_augmented_and_output_shapes_obey_limits(self):
        """LIM-06: Augmented matrices and returned bases count toward matrix limits."""
        A = Matrix.identity(2)
        with self.assertRaises(ResourceLimitError):
            inverse(A, limits=ComputationLimits(max_matrix_entries=4))
        with self.assertRaises(ResourceLimitError):
            nullspace(Matrix((), ncols=3), limits=ComputationLimits(max_matrix_entries=8))

    def test_work_evidence_and_integer_limits(self):
        """LIM-03 LIM-05: Exact elimination and certificates respect the shared finite budget."""
        A = Matrix(((2, 3), (4, 7)))
        with self.assertRaises(ResourceLimitError):
            rref(A, limits=ComputationLimits(max_work=1))
        with self.assertRaises(ResourceLimitError):
            reduce_with_trace(A, limits=ComputationLimits(max_evidence_items=1))
        with self.assertRaises(ResourceLimitError):
            det(Matrix(((2 ** 100,),)), limits=ComputationLimits(max_integer_bits=32))

    def test_compiler_shared_node_budget(self):
        """LIM-04: Sparse normalization has one shared visit budget across subexpressions."""
        x, y = Symbol("x"), Symbol("y")
        with self.assertRaises(ResourceLimitError):
            linear_system_from_equations((Eq((x + y) ** 10, 0),), (x, y),
                                         limits=ComputationLimits(max_nodes=20))

    def test_one_hundred_deterministic_exact_examples(self):
        """LIN-08 LIN-26: 100 seeded small matrices exercise rank-nullity, determinants, and systems."""
        seed = 20404
        random = Random(seed)
        for case in range(100):
            n = random.randrange(5)
            rows = tuple(tuple(Rational(random.randint(-4, 4), random.randint(1, 5))
                               for _ in range(n)) for _ in range(n))
            A = Matrix(rows, ncols=n)
            with self.subTest(seed=seed, case=case, A=A):
                expected_det = leibniz(A)
                self.assertEqual(det(A), expected_det)
                self.assertTrue(check_determinant(A, expected_det))
                basis = nullspace(A)
                self.assertEqual(rank(A) + len(basis), n)
                self.assertTrue(check_nullspace(A, basis))
                variables = tuple(Symbol(f"x{i}") for i in range(n))
                known = tuple(Rational(random.randint(-3, 3)) for _ in range(n))
                rhs = tuple(sum((value * point for value, point in zip(row, known)), Rational(0))
                            for row in rows)
                system = LinearSystem(A, rhs, variables)
                solution, evidence = solve_linear_system(system)
                self.assertTrue(verify_linear_solution(system, solution, evidence))
                if expected_det.numerator:
                    self.assertEqual(solution.particular, known)
                    self.assertEqual(A @ inverse(A), Matrix.identity(n))


if __name__ == "__main__":
    unittest.main()
