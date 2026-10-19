"""The certify checkers of the linear algebra kernel accept honest results and reject forgeries."""

from fractions import Fraction
import unittest

import mathforge as mf
from mathforge.certify.linalg import determinant as det_checker, inverse as inverse_checker
from mathforge.certify.linalg._modular import det_mod, nonsingular_mod
from mathforge.certify.linalg.echelon import check_echelon
from mathforge.certify.linalg.lu import check_lu
from mathforge.certify.linalg.trace import check_trace
from mathforge.kernels import linalg as kernel
from mathforge.trusted.intmatrix import clear_row_denominators
from mathforge.trusted.primes import MODULI

from .support import R, fraction_det, matrices


def rational(fraction):
    return R(fraction.numerator, fraction.denominator)


class DeterminantCertificateTest(unittest.TestCase):
    def test_honest_determinants_pass(self):
        for matrix in matrices(21, 250, square=True, max_size=9):
            with self.subTest(matrix=matrix):
                self.assertTrue(det_checker.check_determinant(matrix, rational(fraction_det(matrix))))
                self.assertTrue(det_checker.check_determinant(matrix.rows, rational(fraction_det(matrix))))

    def test_forged_determinants_fail(self):
        for matrix in matrices(22, 150, square=True, max_size=9):
            value = fraction_det(matrix)
            for forged in (value + 1, value - 1, -value if value else 1, value * 2 if value else 2,
                           value + Fraction(1, 7)):
                with self.subTest(matrix=matrix, forged=forged):
                    self.assertFalse(det_checker.check_determinant(matrix, rational(forged)))
        self.assertFalse(det_checker.check_determinant(mf.Matrix(((1, 2), (3, 4))), -2))  # Not a Rational.
        self.assertFalse(det_checker.check_determinant(mf.Matrix(((1, 2, 3),)), R(0)))  # Not square.

    def test_hadamard_bound_and_residues(self):
        rows = tuple(tuple(R(50 if i == j else (3 * i + 5 * j) % 11 - 5) for j in range(8)) for i in range(8))
        int_rows, _ = clear_row_denominators(rows)
        true = kernel.determinant(int_rows)
        self.assertTrue(det_checker.check_integer_determinant(int_rows, true))
        # Agreeing modulo the first prime is not enough when the bound needs more: off by a multiple of it.
        self.assertFalse(det_checker.check_integer_determinant(int_rows, true + MODULI[0]))
        bound = 1
        for row in int_rows:
            bound *= sum(v * v for v in row)
        self.assertFalse(det_checker.check_integer_determinant(int_rows, bound))  # |D| > H.
        large = tuple(tuple((7 * i * i + 3 * j * j + i * j) % 199 - 99 for j in range(12)) for i in range(12))
        value = kernel.determinant(large)
        self.assertTrue(det_checker.check_integer_determinant(large, value))
        product = MODULI[0] * MODULI[1]
        self.assertFalse(det_checker.check_integer_determinant(large, value + product))
        # Within the Hadamard bound and right modulo the first prime, wrong modulo the second.
        bound = 1
        for row in large:
            bound *= sum(v * v for v in row)
        self.assertLessEqual((value + MODULI[0]) ** 2, bound)
        self.assertFalse(det_checker.check_integer_determinant(large, value + MODULI[0]))

    def test_modular_elimination(self):
        rows = ((2, 4), (1, 3))
        self.assertEqual(det_mod(rows, MODULI[0]), 2)
        self.assertTrue(nonsingular_mod(rows, MODULI[0]))
        self.assertFalse(nonsingular_mod(rows, 2))  # det 2 is 0 modulo 2.
        self.assertFalse(nonsingular_mod(((3, 0), (0, 5)), 15))  # 3 is not a unit modulo 15: no conclusion.
        self.assertTrue(nonsingular_mod(((2, 0), (0, 7)), 15))
        self.assertFalse(nonsingular_mod(rows, 1))
        self.assertEqual(det_mod((), MODULI[0]), 1)


class InverseCertificateTest(unittest.TestCase):
    def test_honest_and_forged_inverses(self):
        checked = 0
        for matrix in matrices(23, 200, square=True, max_size=7):
            inverse = kernel.inverse(matrix.rows)
            if inverse is None:
                continue
            checked += 1
            self.assertTrue(inverse_checker.check_inverse(matrix, inverse))
            self.assertTrue(inverse_checker.check_inverse(matrix, mf.Matrix(inverse, ncols=matrix.ncols)))
            if not inverse:
                continue
            for i, j in ((0, 0), (len(inverse) - 1, 0)):
                forged = [list(row) for row in inverse]
                forged[i][j] = forged[i][j] + R(1, 3)
                self.assertFalse(inverse_checker.check_inverse(matrix, tuple(tuple(row) for row in forged)))
            self.assertFalse(inverse_checker.check_inverse(matrix, inverse[:-1]))
        self.assertGreater(checked, 100)
        self.assertFalse(inverse_checker.check_inverse(mf.Matrix(((1, 2), (3, 4))), ((R(1), R(0)), (R(0), 1))))


class EchelonCertificateTest(unittest.TestCase):
    def certify(self, int_rows, ncols, certificate):
        return check_echelon(int_rows, ncols, certificate.numerators, certificate.denominator, certificate.pivots,
                             certificate.basis_rows, certificate.modulus)

    def test_honest_certificates_pass(self):
        for matrix in matrices(24, 300):
            int_rows, _ = clear_row_denominators(matrix.rows)
            with self.subTest(matrix=matrix):
                self.assertIs(self.certify(int_rows, matrix.ncols, kernel.echelon(int_rows, matrix.ncols)), True)

    def test_forgeries_fail(self):
        # Rank 2, pivots (0, 2): row 1 is twice row 0.
        int_rows = ((1, 2, 3, 4), (2, 4, 6, 8), (1, 2, 1, 0))
        honest = kernel.echelon(int_rows, 4)
        self.assertEqual(honest.pivots, (0, 2))
        self.assertTrue(self.certify(int_rows, 4, honest))
        numerators = [list(row) for row in honest.numerators]
        numerators[0][3] += 1  # One R entry changed.
        self.assertFalse(self.certify(int_rows, 4, honest._replace(numerators=tuple(map(tuple, numerators)))))
        self.assertFalse(self.certify(int_rows, 4, honest._replace(pivots=(0, 3))))  # A pivot moved.
        self.assertFalse(self.certify(int_rows, 4, honest._replace(pivots=(1, 2))))
        self.assertFalse(self.certify(int_rows, 4, honest._replace(basis_rows=(0, 1))))  # A dependent row.
        determinant = honest.denominator
        divisor = abs(determinant) if abs(determinant) > 1 else None
        if divisor is not None:  # A modulus dividing the minor's determinant.
            self.assertFalse(self.certify(int_rows, 4, honest._replace(modulus=divisor)))
        self.assertFalse(self.certify(int_rows, 4, honest._replace(modulus=1)))
        # Dropping a row of R: the rank claimed too small fails C2.
        self.assertFalse(self.certify(int_rows, 4, honest._replace(
            numerators=honest.numerators[:1], pivots=(0,), basis_rows=(0,))))
        # The identity claim for a singular square matrix fails C3.
        square = ((1, 2, 3), (2, 4, 6), (1, 0, 1))
        self.assertFalse(check_echelon(square, 3, None, 1, (0, 1, 2), (0, 1, 2), MODULI[0]))
        self.assertTrue(check_echelon(((2, 1), (1, 1)), 2, None, 1, (0, 1), (0, 1), MODULI[0]))
        self.assertFalse(check_echelon(((2, 1), (1, 1)), 2, ((2, 0), (0, 2.0)), 2, (0, 1), (0, 1), MODULI[0]))

    def test_composite_modulus_is_sound(self):
        int_rows = ((3, 0), (0, 5))
        honest = kernel.echelon(int_rows, 2)
        self.assertTrue(self.certify(int_rows, 2, honest))
        self.assertFalse(self.certify(int_rows, 2, honest._replace(modulus=15)))  # 15 is no unit modulus here.
        self.assertTrue(self.certify(int_rows, 2, honest._replace(modulus=7)))


class LUCertificateTest(unittest.TestCase):
    def test_honest_and_forged_factors(self):
        for matrix in matrices(25, 150, square=True, max_size=7):
            P, L, U = kernel.lu(matrix.rows)
            with self.subTest(matrix=matrix):
                self.assertTrue(check_lu(matrix, P, L, U))
                if len(P) < 2:
                    continue
                if len(set(matrix.rows)) == len(matrix.rows):  # Another permutation then gives another P A.
                    self.assertFalse(check_lu(matrix, P[::-1], L, U))
                forged = [list(row) for row in U]
                forged[0][-1] = forged[0][-1] + R(1, 2)
                self.assertFalse(check_lu(matrix, P, L, tuple(map(tuple, forged))))
                lower = [list(row) for row in L]
                lower[0][1] = R(1)  # Not lower triangular.
                self.assertFalse(check_lu(matrix, P, tuple(map(tuple, lower)), U))


class TraceCertificateTest(unittest.TestCase):
    def test_trace(self):
        rows = ((R(1, 2), R(3)), (R(4), R(-5, 3)))
        self.assertTrue(check_trace(rows, R(-7, 6)))
        self.assertFalse(check_trace(rows, R(-7, 5)))
        self.assertFalse(check_trace(rows, 1))


if __name__ == "__main__":
    unittest.main()
