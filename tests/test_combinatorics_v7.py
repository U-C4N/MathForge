"""Combinatorial boundary conventions and independent exact identities."""
import math
import unittest
from unittest.mock import patch

from mathforge import combinatorics as co
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError


class CombinatoricsTests(unittest.TestCase):
    def checked(self, operation, **args):
        value, evidence = co.produce(operation, args)
        self.assertTrue(co.check_operation(operation, args, value, evidence), (operation, args, value))
        self.assertFalse(co.check_operation(operation, args, value+1, evidence))
        return value

    def test_factorial_binomial_and_permutations(self):
        for n in range(16):
            self.assertEqual(self.checked("factorial", n=n), math.factorial(n))
            for k in (0, n//2, n, n+1):
                self.assertEqual(self.checked("binomial", n=n, k=k), math.comb(n, k))
                self.assertEqual(self.checked("permutations_count", n=n, k=k), math.perm(n, k))

    def test_multinomial_sequences_and_boundary_conventions(self):
        self.assertEqual(self.checked("multinomial", counts=()), 1)
        self.assertEqual(self.checked("multinomial", counts=(0, 0, 0)), 1)
        self.assertEqual(self.checked("multinomial", counts=(2, 3, 1)), 60)
        self.assertEqual([self.checked("fibonacci", n=n) for n in range(11)], [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
        self.assertEqual(self.checked("fibonacci", n=100), 354224848179261915075)
        self.assertEqual([self.checked("catalan", n=n) for n in range(7)], [1, 1, 2, 5, 14, 42, 132])
        self.assertEqual(self.checked("stirling", n=0, k=0), 1)
        self.assertEqual(self.checked("stirling", n=6, k=0), 0)
        self.assertEqual(self.checked("stirling", n=5, k=3), 25)
        self.assertEqual(self.checked("stirling", n=12, k=4), 611501)
        self.assertEqual(self.checked("stirling", n=3, k=5), 0)

    def test_checker_does_not_call_producers(self):
        cases = [("factorial", {"n": 20}), ("binomial", {"n": 40, "k": 16}),
                 ("multinomial", {"counts": (4, 6, 3)}), ("permutations_count", {"n": 12, "k": 5}),
                 ("fibonacci", {"n": 200}), ("catalan", {"n": 15}), ("stirling", {"n": 15, "k": 6})]
        records = [(op, args, *co.produce(op, args)) for op, args in cases]
        with patch.object(co, "produce", side_effect=AssertionError("producer")), \
             patch.object(co, "_factorial", side_effect=AssertionError("factorial producer")), \
             patch.object(co, "_choose", side_effect=AssertionError("choose producer")), \
             patch.object(co, "_fib", side_effect=AssertionError("Fibonacci producer")), \
             patch.object(co, "_stirling", side_effect=AssertionError("Stirling producer")):
            for op, args, value, evidence in records:
                self.assertTrue(co.check_operation(op, args, value, evidence))
                self.assertFalse(co.check_operation(op, args, value, evidence+(1,)))

    def test_invalid_and_resource_limits(self):
        for bad in (-1, True, 1.0):
            with self.assertRaises(InvalidInput):
                co.produce("factorial", {"n": bad})
        for operation, args in (("factorial", {"n": 100}), ("fibonacci", {"n": 100}),
                                ("binomial", {"n": 100, "k": 50})):
            with self.assertRaises(ResourceLimitError):
                co.produce(operation, args, limits=ComputationLimits(max_integer_bits=16))
        with self.assertRaises(ResourceLimitError):
            co.produce("stirling", {"n": 10, "k": 5}, limits=ComputationLimits(max_work=1))


if __name__ == "__main__":
    unittest.main()
