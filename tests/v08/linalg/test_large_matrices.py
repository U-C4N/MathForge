"""Large matrices on the value path: dense 40 x 40 and 64 x 64 integer matrices complete under the default limits."""

import random
import unittest

import mathforge as mf
from mathforge import lean

from .support import fraction_det


def dense(n, seed):
    rng = random.Random(seed)
    return mf.Matrix(tuple(tuple(rng.randint(-99, 99) for _ in range(n)) for _ in range(n)))


class LargeMatricesTest(unittest.TestCase):
    def check_matrix(self, matrix, invertible):
        n = matrix.nrows
        for operation in ("rref", "rank", "nullspace", "det") + (("inverse",) if invertible else ()):
            with self.subTest(n=n, operation=operation):
                self.assertIsNot(getattr(lean.functions, operation)(matrix), lean.FALLBACK)
        reduced = mf.rref(matrix)
        rank = mf.rank(matrix)
        basis = mf.nullspace(matrix)
        self.assertEqual(len(reduced.pivot_columns), rank)
        self.assertEqual(len(basis), n - rank)
        for vector in basis:
            self.assertTrue(all(sum((a * b for a, b in zip(row, vector)), mf.Rational(0)) == mf.Rational(0)
                                for row in matrix.rows))
        determinant = mf.det(matrix)
        if invertible:
            self.assertEqual(rank, n)
            self.assertEqual(reduced.matrix, mf.Matrix.identity(n))
            inverse = mf.inverse(matrix)
            self.assertEqual(inverse.shape, (n, n))
            first = tuple(sum((a * b for a, b in zip(matrix.rows[0], column)), mf.Rational(0))
                          for column in zip(*inverse.rows))
            self.assertEqual(first, (mf.Rational(1),) + (mf.Rational(0),) * (n - 1))
        else:
            self.assertEqual(determinant, mf.Rational(0))
            self.assertLess(rank, n)
        return determinant

    def test_40_and_64_square(self):
        for n, seed in ((40, 40), (64, 64)):
            matrix = dense(n, seed)
            determinant = self.check_matrix(matrix, invertible=True)
            if n == 40:
                exact = fraction_det(matrix)
                self.assertEqual((determinant.numerator, determinant.denominator),
                                 (exact.numerator, exact.denominator))
        rows = list(dense(40, 41).rows)
        rows[7] = rows[3]  # Two equal rows: rank deficient.
        self.check_matrix(mf.Matrix(tuple(rows)), invertible=False)

    def test_solve_system_40(self):
        """solve_system is structured only (mathforge.solve_system returns the Result): a dense 40 x 40 system
        still completes under the default limits."""
        matrix = dense(40, 42)
        unknowns = tuple(mf.Symbol(f"x{k}") for k in range(40))
        rhs = tuple(range(40))
        result = mf.solve_system(mf.LinearSystem(matrix, rhs, unknowns))
        solution = result.unwrap()
        self.assertEqual(solution.basis, ())
        for row, target in zip(matrix.rows, rhs):
            self.assertEqual(sum((a * b for a, b in zip(row, solution.particular)), mf.Rational(0)),
                             mf.Rational(target))


if __name__ == "__main__":
    unittest.main()
