"""Matrices within the default limits whose integer forms explode (final review B, C1 b and f): the lean functions
fall back and the structured calls refuse with resource_limit_exceeded, each within a second, where det took about
110 s and trace 23 s. The inputs are the final review's repro inputs; building them is not timed."""

from __future__ import annotations

import time
import unittest

import mathforge as mf
from mathforge import lean, operations
from mathforge.errors import OperationError
from mathforge.limits import ResourceLimitError
from mathforge.model import Rational

SECONDS = 1.0
BASE = 2 ** 65535


def coprime_denominators(n):
    """An n x n matrix of entries 1/d with distinct odd d of 65,536 bits: each row's lcm has about n * 65,536 bits."""
    return mf.Matrix(tuple(tuple(Rational(1, BASE + 2 * (i * n + j) + 1) for j in range(n)) for i in range(n)))


def coprime_diagonal(n):
    """The diagonal matrix of the 1/d, d of 65,536 bits: the trace's common denominator has n * 65,536 bits."""
    diagonal = [Rational(1, BASE + 2 * k + 1) for k in range(n)]
    return mf.Matrix(tuple(tuple(diagonal[i] if i == j else Rational(0) for j in range(n)) for i in range(n)))


class ExplosiveMatricesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dense = coprime_denominators(32)
        cls.diagonal = coprime_diagonal(64)

    def assertRefusedQuickly(self, call, context=None):
        start = time.perf_counter()
        try:
            value = call()
        except (ResourceLimitError, OperationError) as error:
            self.assertEqual(error.code, "resource_limit_exceeded", context)
        else:
            self.assertIs(value, lean.FALLBACK, context)
        self.assertLessEqual(time.perf_counter() - start, SECONDS, context)

    def test_row_scales_past_the_integer_bound(self):
        for name in ("det", "inverse", "rank", "rref", "lu"):
            with self.subTest(name):
                self.assertRefusedQuickly(lambda: getattr(lean.functions, name)(self.dense), name)
                if name in ("det", "inverse", "rank"):
                    self.assertRefusedQuickly(lambda: getattr(operations, name)(self.dense).unwrap(), name)

    def test_running_denominators_past_the_integer_bound(self):
        self.assertRefusedQuickly(lambda: lean.functions.trace(self.diagonal))
        self.assertRefusedQuickly(lambda: operations.trace(self.diagonal).unwrap())
        self.assertRefusedQuickly(lambda: lean.functions.charpoly(self.diagonal, mf.Symbol("t")))
        self.assertRefusedQuickly(lambda: lean.functions.real_eigenvalues(self.diagonal))

    def test_a_row_scale_past_the_bound_keeps_v07_answers(self):
        """[[1/a, 1/b], [0, 1]] with coprime a and b of 40,000 bits: the first row's scale ab passes the integer bound,
        which v0.7's det refused there and v0.7's inverse did not (its Gauss-Jordan over Q never forms ab). The
        structured inverse keeps v0.7's answer, and the checks of a determinant and an inverse keep their verdicts."""
        from mathforge.matrices import check_determinant, check_inverse
        a, b = (1 << 40_000) + 1, (1 << 40_000) + 3
        A = mf.Matrix(((Rational(1, a), Rational(1, b)), (Rational(0), Rational(1))))
        inverse = operations.inverse(A).unwrap()
        self.assertEqual(inverse, mf.Matrix(((Rational(a), Rational(-a, b)), (Rational(0), Rational(1)))))
        self.assertEqual(mf.inverse(A), inverse)
        self.assertTrue(check_inverse(A, inverse))
        self.assertFalse(check_inverse(A, mf.Matrix(((Rational(a), Rational(a, b)), (Rational(0), Rational(1))))))
        self.assertTrue(check_determinant(A, Rational(1, a)))
        self.assertFalse(check_determinant(A, Rational(1, b)))
        with self.assertRaises(OperationError) as raised:
            operations.det(A).unwrap()
        self.assertEqual(raised.exception.code, "resource_limit_exceeded")


if __name__ == "__main__":
    unittest.main()
