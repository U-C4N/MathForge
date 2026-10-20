"""Surd expressions within the default limits whose powers explode (final review B, C1 d): nested powers of
1 + sqrt 2 would make field elements of millions of bits. The lean function declines them before any arithmetic and
the structured call refuses them, each within a second, where the lean one ran for minutes."""

from __future__ import annotations

import time
import unittest

from mathforge import lean, operations
from mathforge.errors import OperationError
from mathforge.limits import ResourceLimitError
from mathforge.model import Add, Pow, Rational, Sqrt

SECONDS = 1.0


def nested(levels, exponent):
    node = Add((Rational(1), Sqrt(Rational(2))))
    for _ in range(levels):
        node = Pow(node, exponent)
    return node


class ExplosiveSurdsTest(unittest.TestCase):
    def assertRefusedQuickly(self, call, context=None):
        start = time.perf_counter()
        try:
            value = call()
        except (ResourceLimitError, OperationError) as error:
            self.assertEqual(error.code, "resource_limit_exceeded", context)
        else:
            self.assertIs(value, lean.FALLBACK, context)
        self.assertLessEqual(time.perf_counter() - start, SECONDS, context)

    def test_nested_powers(self):
        for levels in (2, 3):
            expression = nested(levels, 256)
            with self.subTest(levels=levels):
                self.assertRefusedQuickly(lambda: lean.functions.complex_algebraic(expression, 0))
                self.assertRefusedQuickly(lambda: operations.complex_algebraic(expression, 0).unwrap())

    def test_the_scan_declines_before_any_arithmetic(self):
        from mathforge.lean import algebra_complex as module
        with self.assertRaises(module._Decline):
            module._scan(nested(2, 256))  # 256 * 256 * 2 bits for the leaf sqrt(2)
        self.assertEqual(module._scan(nested(1, 256)), [2])  # (1 + sqrt 2)**256 has 326-bit coefficients


if __name__ == "__main__":
    unittest.main()
