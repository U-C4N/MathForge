"""The integer linear algebra kernel: the prime table, the producers and the lean value functions built on them."""

import math
import unittest

import mathforge as mf
from mathforge import lean, operations
from mathforge.kernels import linalg as kernel
from mathforge.trusted import primes
from mathforge.trusted.intmatrix import clear_row_denominators

from .support import as_fraction, fraction_det, fraction_rows, matrices


def outcome(call):
    try:
        value = call()
        if isinstance(value, mf.Result):
            value = value.unwrap()
        return ("value", value)
    except Exception as error:
        return ("error", type(error).__name__, getattr(error, "code", None))


class TrustedPrimesTest(unittest.TestCase):
    def test_moduli(self):
        table = primes.MODULI
        self.assertEqual(len(table), 64)
        self.assertTrue(all(a > b for a, b in zip(table, table[1:])))
        self.assertLess(table[0], 2**61)
        self.assertTrue(all(primes.is_prime(p) for p in table))
        # Nothing between two entries, nor between the first and 2**61, is prime.
        for high, low in zip((2**61, *table), table):
            for candidate in range(low + 1, high):
                self.assertFalse(primes.is_prime(candidate), candidate)

    def test_is_prime_small_numbers(self):
        sieve = [True] * 2000
        sieve[0] = sieve[1] = False
        for p in range(2, 2000):
            if sieve[p]:
                for multiple in range(p * p, 2000, p):
                    sieve[multiple] = False
        self.assertEqual([n for n in range(2000) if primes.is_prime(n)], [n for n in range(2000) if sieve[n]])
        for strong_pseudoprime in (2047, 1373653, 25326001, 3215031751, 2152302898747, 3474749660383,
                                   341550071728321, 3825123056546413051):
            self.assertFalse(primes.is_prime(strong_pseudoprime))

    def test_continuation_below_the_table(self):
        below = primes.primes_below(primes.MODULI[-1])
        first = next(below)
        self.assertTrue(primes.is_prime(first))
        self.assertTrue(all(not primes.is_prime(n) for n in range(first + 1, primes.MODULI[-1])))


class ProducerTest(unittest.TestCase):
    def test_clear_row_denominators(self):
        for matrix in matrices(11, 60):
            int_rows, scales = clear_row_denominators(matrix.rows)
            for row, int_row, scale in zip(fraction_rows(matrix), int_rows, scales):
                self.assertTrue(all(type(v) is int for v in int_row))
                self.assertEqual([scale * v for v in row], list(int_row))
                self.assertEqual(scale, math.lcm(*(v.denominator for v in row)) if row else 1)

    def test_determinant_and_inverse_equal_the_oracle(self):
        for matrix in matrices(12, 300, square=True):
            int_rows, scales = clear_row_denominators(matrix.rows)
            det = fraction_det(matrix)
            scale = 1
            for s in scales:
                scale *= s
            self.assertEqual(kernel.determinant(int_rows), det * scale)
            inverse = kernel.inverse(matrix.rows)
            if det == 0:
                self.assertIsNone(inverse)
                continue
            rows = fraction_rows(matrix)
            product = [[sum(a * as_fraction(b) for a, b in zip(row, column)) for column in zip(*inverse)]
                       for row in rows]
            n = matrix.nrows
            self.assertEqual(product, [[int(i == j) for j in range(n)] for i in range(n)])


class LeanValuesTest(unittest.TestCase):
    def assertLeanEqualsStructured(self, operation, *arguments):
        value = getattr(lean.functions, operation)(*arguments)
        expected = outcome(lambda: getattr(operations, operation)(*arguments))
        if expected[0] == "error":
            self.assertIs(value, lean.FALLBACK, (operation, arguments, expected))
            return
        self.assertIsNot(value, lean.FALLBACK, (operation, arguments))
        self.assertIs(type(value), type(expected[1]))
        self.assertEqual(value, expected[1])
        self.assertEqual(mf.to_data(value), mf.to_data(expected[1]))

    def test_trace_det_inverse_equal_v07(self):
        for matrix in matrices(13, 200):
            for operation in ("trace", "det", "inverse"):
                with self.subTest(operation=operation, matrix=matrix):
                    self.assertLeanEqualsStructured(operation, matrix)

    def test_inputs_that_fall_back(self):
        from mathforge.limits import ComputationLimits, computation
        with computation(ComputationLimits(max_integer_bits=300_000)):
            huge = mf.Matrix(((2**70_000, 1), (1, 3)))
        with computation(ComputationLimits(max_matrix_entries=65 * 65)):
            large = mf.Matrix(tuple(tuple((i + j) % 5 for j in range(65)) for i in range(65)))
        for operation in ("trace", "det", "inverse"):
            for argument in (huge, large, mf.Matrix(((1, 2, 3),)), ((1, 2), (3, 4)), None, True, 1.5):
                with self.subTest(operation=operation, argument=argument):
                    self.assertIs(getattr(lean.functions, operation)(argument), lean.FALLBACK)
        self.assertIs(lean.functions.inverse(mf.Matrix(((1, 2), (2, 4)))), lean.FALLBACK)
        with computation():
            self.assertIs(lean.functions.det(mf.Matrix(((1, 2), (3, 4)))), lean.FALLBACK)


if __name__ == "__main__":
    unittest.main()
