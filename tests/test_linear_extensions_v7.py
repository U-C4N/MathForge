import unittest
from unittest.mock import patch

from mathforge import Matrix, Rational as Q
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
import mathforge.linear_extensions as l


class LinearExtensionsTests(unittest.TestCase):
    def checked(self, op, **args):
        value, evidence = l.produce(op, args)
        self.assertTrue(l.check_operation(op, args, value, evidence))
        return value, evidence

    def test_trace_and_spaces(self):
        A = Matrix(((1, 2, 3), (2, 4, 6), (0, 1, 1)))
        self.assertEqual(self.checked("trace", A=A)[0], Q(6))
        self.assertEqual(self.checked("rowspace", A=A)[0], ((Q(1), Q(0), Q(1)), (Q(0), Q(1), Q(1))))
        self.assertEqual(self.checked("columnspace", A=A)[0], ((Q(1), Q(2), Q(0)), (Q(2), Q(4), Q(1))))
        for shape in ((0, 3), (3, 0)):
            A = Matrix((() for _ in range(shape[0])), ncols=shape[1])
            self.assertEqual(self.checked("rowspace", A=A)[0], ())
            self.assertEqual(self.checked("columnspace", A=A)[0], ())

    def test_lu_singular_pivots_and_empty(self):
        for A in (Matrix(((0, 2), (3, 4))), Matrix(((0, 1), (0, 2))),
                  Matrix(((1, 2, 3), (2, 4, 6), (3, 7, 8))), Matrix(((0, 0), (0, 0))), Matrix(())):
            value, _ = self.checked("lu", A=A)
            self.assertEqual(value.P @ A, value.L @ value.U)

    def test_least_squares_full_and_deficient_rank(self):
        A = Matrix(((1, 0), (0, 1), (1, 1)))
        result, _ = self.checked("least_squares", A=A, b=(1, 1, 3))
        self.assertEqual(result.particular, (Q(4, 3), Q(4, 3)))
        self.assertEqual(result.basis, ())
        self.assertEqual(result.residual, (Q(-1, 3), Q(-1, 3), Q(1, 3)))
        A = Matrix(((1, 1), (2, 2)))
        result, _ = self.checked("least_squares", A=A, b=(1, 3))
        self.assertEqual(result.particular, (Q(7, 5), Q(0)))
        self.assertEqual(result.basis, ((Q(-1), Q(1)),))
        self.assertEqual(result.residual, (Q(-2, 5), Q(1, 5)))

    def test_empty_and_zero_systems(self):
        result, _ = self.checked("least_squares", A=Matrix((), ncols=2), b=())
        self.assertEqual(result.particular, (Q(0), Q(0)))
        self.assertEqual(len(result.basis), 2)
        result, _ = self.checked("least_squares", A=Matrix(((), ()), ncols=0), b=(2, 3))
        self.assertEqual(result.particular, ())
        self.assertEqual(result.residual, (Q(2), Q(3)))
        result, _ = self.checked("least_squares", A=Matrix(((0, 0),)), b=(7,))
        self.assertEqual(len(result.basis), 2)

    def test_independent_checks_and_forged_evidence(self):
        A = Matrix(((1, 2), (2, 4)))
        for op in ("trace", "rowspace", "columnspace", "lu", "least_squares"):
            args = {"A": A, "b": (1, 3)} if op == "least_squares" else {"A": A}
            value, evidence = self.checked(op, **args)
            with patch.object(l, op, side_effect=AssertionError("producer used")), \
                 patch.object(l, "reduce_with_trace", side_effect=AssertionError("eliminator used")), \
                 patch.object(l, "_space", side_effect=AssertionError("space producer used")), \
                 patch.object(l, "_least_squares", side_effect=AssertionError("LS producer used")):
                self.assertTrue(l.check_operation(op, args, value, evidence))
                self.assertFalse(l.check_operation(op, args, value, ("forged",)))
        result, evidence = self.checked("least_squares", A=A, b=(1, 3))
        missing = l.LeastSquaresResult(result.particular, (), result.residual)
        self.assertFalse(l.check_operation("least_squares", {"A": A, "b": (1, 3)}, missing, evidence))
        duplicate = l.LeastSquaresResult(result.particular, result.basis*2, result.residual)
        self.assertFalse(l.check_operation("least_squares", {"A": A, "b": (1, 3)}, duplicate, evidence))
        wrong = l.LeastSquaresResult(result.particular, result.basis, (Q(0), Q(0)))
        self.assertFalse(l.check_operation("least_squares", {"A": A, "b": (1, 3)}, wrong, evidence))
        lu, _ = self.checked("lu", A=A)
        wrong = l.LUDecomposition(lu.P, Matrix.identity(2), lu.U)
        self.assertFalse(l.check_operation("lu", {"A": A}, wrong, ()))

    def test_invalid_inputs_and_budget(self):
        A = Matrix(((1, 2),))
        for call in (lambda: l.trace(A), lambda: l.lu(A), lambda: l.least_squares(A, ())):
            with self.assertRaises(InvalidInput):
                call()
        A = Matrix.identity(4)
        with computation(ComputationLimits(max_work=10)):
            with self.assertRaises(ResourceLimitError):
                l.least_squares(A, (1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
