"""Indices within the default limits whose values have millions of bits (final review B, C1 e): the lean functions
fall back before math.comb or math.perm forms the value, and the structured calls refuse with
resource_limit_exceeded, each within a second, where the lean ones took minutes. The bounds that refuse them are the
last that the old size tests accepted (catalan up to 32,764)."""

from __future__ import annotations

import time
import unittest

from mathforge import lean, operations
from mathforge.errors import OperationError
from mathforge.limits import ResourceLimitError

SECONDS = 1.0


class ExplosiveCombinatoricsTest(unittest.TestCase):
    def assertRefusedQuickly(self, call, context=None):
        start = time.perf_counter()
        try:
            value = call()
        except (ResourceLimitError, OperationError) as error:
            self.assertEqual(error.code, "resource_limit_exceeded", context)
        else:
            self.assertIs(value, lean.FALLBACK, context)
        self.assertLessEqual(time.perf_counter() - start, SECONDS, context)

    def test_values_of_millions_of_bits(self):
        cases = (("binomial", (2 ** 65535, 3000)), ("binomial", (2 ** 20, 2 ** 19)), ("binomial", (10 ** 9, 70_000)),
                 ("permutations_count", (2 ** 65535, 3000)), ("permutations_count", (10 ** 6, 10 ** 6)),
                 ("multinomial", ((2 ** 65534, 3000),)), ("multinomial", ((2 ** 19, 2 ** 19),)),
                 ("catalan", (10 ** 6,)), ("catalan", (32_765,)))
        for index, (name, arguments) in enumerate(cases):
            with self.subTest(name=name, case=index):
                self.assertRefusedQuickly(lambda: getattr(lean.functions, name)(*arguments), name)
                self.assertRefusedQuickly(lambda: getattr(operations, name)(*arguments).unwrap(), name)

    def test_the_bounds_keep_the_answers_below_them(self):
        """The largest Catalan index within the bound, and binomials and arrangements just below the length bounds,
        still answer on the lean path, as the structured path does."""
        for name, arguments in (("catalan", (32_764,)), ("binomial", (2 ** 64, 1000)),
                                ("permutations_count", (2 ** 20, 3000)), ("multinomial", ((3, 2 ** 40),))):
            with self.subTest(name=name):
                value = getattr(lean.functions, name)(*arguments)
                self.assertIsNot(value, lean.FALLBACK)
                self.assertEqual(value, getattr(operations, name)(*arguments).unwrap())


if __name__ == "__main__":
    unittest.main()
