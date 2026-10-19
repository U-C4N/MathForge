"""The lean functions of the discrete operations (mathforge/lean/discrete_numbers.py): each returns FALLBACK or
exactly the structured result, and the public value functions keep the structured outcomes."""

import math
import random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.lean import FALLBACK
from mathforge.lean import discrete_numbers as module
from mathforge.limits import computation

BIG = 1 << 140_000  # Twice max_integer_bits and more.
TYPE_EDGES = (True, False, 1.5, "1", None, 2.0, -1, mf.Rational(1, 2))
OPERATIONS = ("factorial", "binomial", "multinomial", "permutations_count", "fibonacci", "catalan", "stirling")


def outcome(call):
    try:
        return "value", call()
    except mf.MathForgeError as error:
        return "error", getattr(error, "code", type(error).__name__)


def structured(operation, *args):
    return outcome(lambda: getattr(mf.operations, operation)(*args).unwrap())


class LeanDiscreteTest(unittest.TestCase):
    def setUp(self):
        self.counts = {}

    def check(self, operation, *args, public=True):
        """The lean value is FALLBACK or the structured value; the public function has the structured outcome."""
        value = getattr(lean.functions, operation)(*args)
        expected = structured(operation, *args)
        if value is not FALLBACK:
            self.assertEqual(expected, ("value", value), (operation, args))
            self.assertIs(type(value), int)
        if public:
            self.assertEqual(outcome(lambda: getattr(mf, operation)(*args)), expected, (operation, args))
        self.counts.setdefault(operation, [0, 0])[value is FALLBACK] += 1
        return value

    def test_lean_equals_structured(self):
        rng = random.Random(20261019)
        for _ in range(5000):
            n, k = rng.randrange(0, 70), rng.randrange(0, 72)
            self.check("binomial", n, k)
            self.check("permutations_count", n, k)
            self.check("stirling", rng.randrange(0, 30), rng.randrange(0, 32))
            self.check("factorial", rng.randrange(0, 400))
            self.check("fibonacci", rng.randrange(0, 4000))
            self.check("catalan", rng.randrange(0, 200))
            counts = tuple(rng.randrange(0, 25) for _ in range(rng.randrange(0, 6)))
            self.check("multinomial", counts if rng.random() < 0.8 else list(counts))
        for operation in OPERATIONS:  # The lean path answers every one of these inputs itself.
            self.assertEqual(self.counts[operation][1], 0, operation)

    def test_larger_inputs(self):
        for operation, arguments in (("factorial", (2000,)), ("factorial", (5000,)), ("binomial", (4000, 2000)),
                                     ("binomial", (200, 80)), ("binomial", (10 ** 6, 3)), ("catalan", (1500,)),
                                     ("permutations_count", (5000, 1000)), ("permutations_count", (10 ** 6, 3)),
                                     ("fibonacci", (20000,)), ("fibonacci", (90000,)), ("stirling", (120, 40)),
                                     ("stirling", (300, 150)), ("stirling", (2500, 3)), ("stirling", (65535, 2)),
                                     ("multinomial", ((500, 400, 300, 200, 100),)),
                                     ("multinomial", ((30, 25, 20, 15, 10),))):
            with self.subTest(operation=operation, arguments=arguments):
                self.assertIsNot(self.check(operation, *arguments), FALLBACK)

    def test_limit_boundaries(self):
        """At max_integer_bits the lean functions answer exactly where v0.7 does, or fall back."""
        self.assertEqual(module._FACTORIAL_MAX, 5910)
        self.assertLessEqual(math.factorial(5910).bit_length(), 65_536)
        self.assertGreater(math.factorial(5911).bit_length(), 65_536)
        self.assertIsNot(self.check("factorial", 5910, public=False), FALLBACK)
        self.assertIs(self.check("factorial", 5911, public=False), FALLBACK)
        self.assertEqual(structured("factorial", 5911), ("error", "resource_limit_exceeded"))
        self.assertIsNot(self.check("fibonacci", 94_400, public=False), FALLBACK)
        self.assertIs(self.check("fibonacci", 94_401, public=False), FALLBACK)
        self.assertEqual(structured("fibonacci", 94_401), ("error", "resource_limit_exceeded"))
        # binomial: v0.7 forms k * C(n, k) (k at most n/2); the largest central n that fits, by bisection.
        low, high = 60_000, 70_000
        while high - low > 1:
            middle = (low + high) // 2
            if (middle // 2 * math.comb(middle, middle // 2)).bit_length() <= 65_536:
                low = middle
            else:
                high = middle
        self.assertIsNot(self.check("binomial", low, low // 2, public=False), FALLBACK)
        self.assertIs(self.check("binomial", high, high // 2, public=False), FALLBACK)
        self.assertEqual(structured("binomial", high, high // 2), ("error", "resource_limit_exceeded"))
        for n in (1000, 3000):  # catalan: n * C(2n, n) within the bound or not
            fits = (n * math.comb(2 * n, n)).bit_length() <= 65_536
            self.assertIsNot(self.check("catalan", n, public=False), FALLBACK) if fits else None
        low, high = 1, 40000  # permutations: the value itself
        while high - low > 1:
            middle = (low + high) // 2
            if math.perm(40000, middle).bit_length() <= 65_536:
                low = middle
            else:
                high = middle
        self.assertIsNot(self.check("permutations_count", 40000, low, public=False), FALLBACK)
        self.assertIs(self.check("permutations_count", 40000, high, public=False), FALLBACK)
        self.assertEqual(structured("permutations_count", 40000, high), ("error", "resource_limit_exceeded"))
        # stirling(n, 2) = 2**(n - 1) - 1, and v0.7's check forms 2**n: 65,536 bits at n = 65,535.
        self.assertIsNot(self.check("stirling", 65_535, 2, public=False), FALLBACK)
        self.assertIs(self.check("stirling", 65_536, 2, public=False), FALLBACK)
        self.assertEqual(structured("stirling", 65_536, 2), ("error", "resource_limit_exceeded"))
        self.assertIs(self.check("multinomial", (100000, 100000, 100000), public=False), FALLBACK)
        self.assertIs(lean.functions.multinomial((0,) * 10_001), FALLBACK)

    def test_edges(self):
        for operation in OPERATIONS:
            for index, edge in enumerate((*TYPE_EDGES, BIG)):
                with self.subTest(operation=operation, edge=index):
                    if operation in ("binomial", "permutations_count", "stirling"):
                        self.check(operation, edge, 3)
                        self.check(operation, 10, edge)
                    elif operation == "multinomial":
                        self.check(operation, edge)
                        self.check(operation, (2, edge))
                    else:
                        self.check(operation, edge)
        for n, k in ((3, 10), (0, 0), (0, 1), (10, 0), (10, 10), (100, 97), (5, 1), (140000, 1), (10, 2), (10, 9),
                     (30, 15), (1, 0), (1, 1), (2, 1)):
            for operation in ("binomial", "permutations_count", "stirling"):
                self.check(operation, n, k)
        for counts in ((), (5,), (0, 0, 0), (2, -1, 3), (True, 2), (2.0, 3), ("2", 3), [2, 3, 4], ((1, 2), 3),
                       (2, 3, 4), range(3), iter((2, 3)), {2: 3}):
            self.check("multinomial", counts)
        for n in (0, 1, 2, 20, 30, 90):
            for operation in ("factorial", "fibonacci", "catalan"):
                self.check(operation, n)

    def test_values_need_an_accepting_check(self):
        with patch.object(module, "_balanced_product", lambda low, high: 7):
            self.assertIs(lean.functions.factorial(10), FALLBACK)
        with patch.object(module, "_check_choose", lambda *args: False):
            self.assertIs(lean.functions.binomial(10, 3), FALLBACK)
            self.assertIs(lean.functions.multinomial((2, 3)), FALLBACK)
        with patch.object(module, "_matrix_power_entry", lambda n: -1):
            self.assertIs(lean.functions.fibonacci(10), FALLBACK)
        with patch.object(module, "_check_stirling", lambda *args: False):
            self.assertIs(lean.functions.stirling(10, 3), FALLBACK)
        with patch.object(module, "perm", lambda n, k: 7):
            self.assertIs(lean.functions.permutations_count(10, 3), FALLBACK)
        with patch.object(module, "comb", lambda n, k: 7):
            self.assertIs(lean.functions.catalan(10), FALLBACK)
            self.assertIs(lean.functions.binomial(10, 3), FALLBACK)
        with patch.object(module, "_factorial", lambda n: 7):
            self.assertIs(lean.functions.factorial(10), FALLBACK)
        with patch.object(module, "_fibonacci_pair", lambda n: (7, 8)):
            self.assertIs(lean.functions.fibonacci(10), FALLBACK)
        with patch.object(module, "_stirling_band", lambda n, k: 7):
            self.assertIs(lean.functions.stirling(10, 3), FALLBACK)
        self.assertEqual(lean.functions.stirling(10, 3), 9330)

    def test_active_budget_falls_back(self):
        arguments = {"factorial": (12,), "binomial": (20, 8), "multinomial": ((2, 3, 1),),
                     "permutations_count": (10, 4), "fibonacci": (90,), "catalan": (30,), "stirling": (30, 15)}
        for operation, args in arguments.items():
            self.assertIsNot(getattr(lean.functions, operation)(*args), FALLBACK)
            with computation():
                self.assertIs(getattr(lean.functions, operation)(*args), FALLBACK, operation)

    def test_lean_functions_are_the_modules(self):
        for operation in OPERATIONS:
            self.assertIs(getattr(lean.functions, operation), module.LEAN[operation])

    def test_helpers(self):
        for low in range(0, 70):
            for high in range(low - 1, 200, 7):
                self.assertEqual(module._balanced_product(max(low, 1), high), math.prod(range(max(low, 1), high + 1)))
        for n in range(0, 300):
            self.assertEqual(module._fibonacci_pair(n)[0], module._matrix_power_entry(n))
        for n in range(0, 40):
            row = module._stirling_row(n, n)
            for k in range(0, n + 1):
                self.assertEqual(module._stirling_band(n, k), row[k])


if __name__ == "__main__":
    unittest.main()
