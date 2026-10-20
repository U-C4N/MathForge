"""Surd expressions within the default limits whose powers explode (final review B, C1 d): nested powers of
1 + sqrt 2 would make field elements of millions of bits. The lean function declines them before any arithmetic and
the structured call refuses them, each within a second, where the lean one ran for minutes.

Powers of products of many square roots (the scoped re-review's residual of C1 d): the interval check's ends had
64-bit-fraction denominators that multiplied across the factors, to 2 million bits for 16 factors (20 s) and more for
64; it now runs in fixed point. The lean function answers as the structured path does, or leaves it the refusal,
within a second, also when the value is exactly 1."""

from __future__ import annotations

import time
import unittest

from mathforge import lean, operations
from mathforge.errors import OperationError
from mathforge.limits import ResourceLimitError
from mathforge.model import Add, Mul, Pow, Rational, Sqrt
from mathforge.serialization import to_data

SECONDS = 1.0


def nested(levels, exponent):
    node = Add((Rational(1), Sqrt(Rational(2))))
    for _ in range(levels):
        node = Pow(node, exponent)
    return node


def powered_product(m, value_one=False):
    """((sqrt 2)**m)**2048 as Pow(Pow(Mul of m sqrt 2, 256), 8); with value_one, m/2 factors 1/2 make it exactly 1."""
    factors = (Sqrt(Rational(2)),) * m + ((Rational(1, 2),) * (m // 2) if value_one else ())
    return Pow(Pow(Mul(factors), 256), 8)


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

    def test_powers_of_long_products(self):
        for m in (16, 64):
            for value_one in (False, True):
                expression = powered_product(m, value_one)
                with self.subTest(m=m, value_one=value_one):
                    start = time.perf_counter()
                    try:
                        want = operations.complex_algebraic(expression, 0).unwrap()
                    except (ResourceLimitError, OperationError) as error:
                        self.assertEqual(error.code, "resource_limit_exceeded")
                        want = None
                    self.assertLessEqual(time.perf_counter() - start, SECONDS)
                    start = time.perf_counter()
                    got = lean.functions.complex_algebraic(expression, 0)
                    self.assertLessEqual(time.perf_counter() - start, SECONDS)
                    if want is None:
                        self.assertIs(got, lean.FALLBACK)
                    elif got is not lean.FALLBACK:
                        self.assertIs(type(got), type(want))
                        self.assertEqual(to_data(got), to_data(want))
                    if value_one:
                        self.assertEqual(to_data(want), to_data(operations.complex_algebraic(1, 0).unwrap()))

    def test_the_interval_check_stays_in_fixed_point(self):
        from mathforge.certify.complex import surd
        low, high = surd.interval(powered_product(16, value_one=True))
        self.assertLessEqual(max(low.denominator, high.denominator), 1 << 64)
        self.assertTrue(low <= 1 <= high and high - low < Rational(1, 1 << 30))
        with self.assertRaises(ValueError):
            surd.interval(powered_product(64))  # 2**65536: past max_integer_bits

    def test_the_scan_declines_before_any_arithmetic(self):
        from mathforge.lean import algebra_complex as module
        with self.assertRaises(module._Decline):
            module._scan(nested(2, 256))  # 256 * 256 * 2 bits for the leaf sqrt(2)
        self.assertEqual(module._scan(nested(1, 256)), [2])  # (1 + sqrt 2)**256 has 326-bit coefficients


if __name__ == "__main__":
    unittest.main()
