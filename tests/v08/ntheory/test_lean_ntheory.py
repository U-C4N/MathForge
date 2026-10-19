"""The lean functions of the ntheory integer operations (mathforge/lean/ntheory_integers.py): each returns FALLBACK
or exactly the structured result, and the public value functions keep the structured outcomes."""

import random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.lean import FALLBACK
from mathforge.lean import ntheory_integers as module
from mathforge.limits import computation

BIG = 1 << 140_000  # Twice max_integer_bits and more.
LIMIT = 1 << 65_535  # The largest power of two within max_integer_bits (65,536 bits).
TYPE_EDGES = (True, False, 1.5, "1", None, 2.0, 3 + 0j, (), [], mf.Rational(1, 2))
OPERATIONS = ("integer_gcd", "integer_lcm", "integer_xgcd", "mod_inverse", "crt", "is_prime", "factor_integer",
              "divisors", "totient")


def outcome(call):
    try:
        return "value", call()
    except mf.MathForgeError as error:
        return "error", getattr(error, "code", type(error).__name__)


def structured(operation, *args):
    return outcome(lambda: getattr(mf.operations, operation)(*args).unwrap())


def same(left, right):
    """Equal values of the same classes, item by item for tuples."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        return len(left) == len(right) and all(map(same, left, right))
    return left == right


class LeanNtheoryTest(unittest.TestCase):
    def setUp(self):
        self.counts = {}

    def check(self, operation, *args):
        """The lean value is FALLBACK or the structured value; the public function has the structured outcome."""
        value = getattr(lean.functions, operation)(*args)
        expected = structured(operation, *args)
        if value is not FALLBACK:
            self.assertEqual(expected[0], "value", (operation, args, value, expected))
            self.assertTrue(same(value, expected[1]), (operation, args, value, expected))
        public = outcome(lambda: getattr(mf, operation)(*args))
        self.assertEqual(public[0], expected[0], (operation, args))
        self.assertTrue(public[1] == expected[1] if public[0] == "error" else same(public[1], expected[1]))
        self.counts.setdefault(operation, [0, 0])[value is FALLBACK] += 1
        return value

    def test_lean_equals_structured(self):
        rng = random.Random(20261019)

        def integer(bits):
            value = rng.getrandbits(rng.randrange(0, bits + 1))
            return -value if rng.random() < 0.3 else value

        for _ in range(5000):
            for operation in ("integer_gcd", "integer_lcm", "integer_xgcd"):
                bits = rng.choice((8, 16, 64, 64, 200))
                a, b = integer(bits), integer(bits)
                if rng.random() < 0.3:
                    shared = integer(40) or 1
                    a, b = a * shared, b * shared
                self.check(operation, a, b)
            self.check("mod_inverse", integer(rng.choice((8, 64, 100))), integer(rng.choice((4, 16, 64))))
            congruences = tuple((integer(40), integer(rng.choice((3, 8, 20)))) for _ in range(rng.randrange(0, 6)))
            self.check("crt", congruences if rng.random() < 0.8 else [list(pair) for pair in congruences])
            n = integer(rng.choice((4, 14, 20, 32, 40)))
            for operation in ("is_prime", "factor_integer", "divisors", "totient"):
                self.check(operation, n)
        for operation in OPERATIONS:  # The lean path answers these inputs itself, except where v0.7 raises.
            self.assertGreater(self.counts[operation][0], 1000, operation)

    def test_larger_inputs(self):
        rng = random.Random(5)
        for _ in range(40):
            bits = rng.choice((1000, 4000, 16000))
            a, b = rng.getrandbits(bits), rng.getrandbits(bits)
            for operation in ("integer_gcd", "integer_lcm", "integer_xgcd"):
                self.assertIsNot(self.check(operation, a, b), FALLBACK)
            self.assertIsNot(self.check("mod_inverse", a, (1 << 4423) - 1), FALLBACK)
        primes = (2 ** 61 - 1, 2 ** 89 - 1, 3224192153648326577750017, 10 ** 9 + 7, 1000003, 999983)
        for n in (*primes, 999983 * 1000003, 1000003 * 1000033, 600851475143, 2 ** 64 + 1, 3 ** 40 - 1,
                  10 ** 18 + 9, 2 ** 100, 720720, 73513440, -(2 ** 61 - 1) * 12):
            for operation in ("is_prime", "factor_integer", "divisors", "totient"):
                value = self.check(operation, n)
                if operation == "is_prime" or n > 0:
                    self.assertIsNot(value, FALLBACK, (operation, n))
        for count, bits in ((128, 900), (300, 3000)):
            number = rng.getrandbits(bits)
            primes_list = [p for p in range(2, 10 ** 5) if all(p % q for q in range(2, int(p ** 0.5) + 1))][:count]
            congruences = tuple((number % p, p) for p in primes_list)
            self.assertIsNot(self.check("crt", congruences), FALLBACK)

    def test_edges(self):
        for operation in OPERATIONS:
            arity = 1 if operation in ("crt", "is_prime", "factor_integer", "divisors", "totient") else 2
            valid = {"crt": ((2, 3), (3, 5)), "mod_inverse": 7}.get(operation, 12)
            for index, edge in enumerate((*TYPE_EDGES, BIG, -BIG)):
                with self.subTest(operation=operation, edge=index):
                    arguments = (edge,) if arity == 1 else (edge, valid)
                    self.check(operation, *arguments)
                    if arity == 2:
                        self.check(operation, valid, edge)
                    if edge in (BIG, -BIG):
                        self.assertIs(getattr(lean.functions, operation)(*arguments), FALLBACK)
        for a, b in ((0, 0), (0, 5), (5, 0), (-84, 30), (84, -30), (-84, -30), (7, 7), (35, 18), (1, 1), (-1, 0),
                     (0, -7), (LIMIT, 3), (3, LIMIT), (LIMIT - 1, LIMIT), (LIMIT, 2 * LIMIT), (2, 6), (6, 2)):
            for operation in ("integer_gcd", "integer_lcm", "integer_xgcd"):
                self.check(operation, a, b)
        for a, m in ((2, 4), (3, 0), (3, -7), (5, 1), (0, 1), (0, 7), (-3, 7), (10, 7), (LIMIT + 1, 3),
                     (3, LIMIT + 1), (-1, 2), (1, 2)):
            self.check("mod_inverse", a, m)
        for congruences in ((), ((2, 5),), [(2, 3), (3, 5)], ([2, 3], [3, 5]), ((0, 2), (1, 4)), ((1, 4), (3, 6)),
                            ((12, 5), (-4, 7)), ((1, 0),), ((1, -3),), ((True, 3),), ((1, 3.0),), ((1, 2, 3),),
                            (5,), ((1, 1), (0, 1)), ((5, 4), (5, 4)), {2: 3}, iter(((2, 3),)), ((BIG, 7),),
                            ((1, LIMIT + 1), (2, LIMIT + 3))):
            self.check("crt", congruences)
        for n in (0, 1, -1, 2, 3, 4, -7, 97, 561, 9409, 9973, 9999, 10007, 341, 2 ** 61 - 1, 1 << 20, 5040, -360,
                  -12, 41041, 3215031751, 2 ** 1024 - 1, 2 ** 1025 * 3, 3 * 2 ** 3000 + 1):
            for operation in ("is_prime", "factor_integer", "divisors", "totient"):
                self.check(operation, n)
        for n in (LIMIT, LIMIT - 1, LIMIT + 1, -LIMIT, (1 << 4096) + 1, (1 << 4097) + 1):
            self.check("is_prime", n)

    def test_limit_boundaries(self):
        """Where v0.7's checks exceed max_integer_bits, the lean functions fall back and v0.7's error stays."""
        a, b = (1 << 40_000) + 1, (1 << 40_000) + 3  # Coprime: s*a has about 80,000 bits.
        for operation in ("integer_gcd", "integer_lcm", "integer_xgcd"):
            self.assertIs(getattr(lean.functions, operation)(a, b), FALLBACK)
            self.assertEqual(structured(operation, a, b), ("error", "resource_limit_exceeded"))
        small = 3 ** 2000
        self.assertIsNot(lean.functions.integer_lcm(small, small + 2), FALLBACK)
        a, m = (1 << 65_000) + 7, (1 << 1000) + 1
        self.assertIs(lean.functions.mod_inverse(a, m), FALLBACK)
        self.assertEqual(structured("mod_inverse", a, m), ("error", "resource_limit_exceeded"))
        congruences = ((1, (1 << 40_000) + 1), (5, (1 << 30_000) + 3))
        self.assertIs(lean.functions.crt(congruences), FALLBACK)
        self.assertEqual(structured("crt", congruences), ("error", "resource_limit_exceeded"))

    def test_values_need_an_accepting_check(self):
        cases = (("is_prime", (2 ** 61 - 1,)), ("factor_integer", (10007 * 10009,)), ("divisors", (10007 * 12,)),
                 ("totient", (10009 * 3,)))
        with patch.object(module, "check_prime", lambda *args: False):
            for operation, arguments in cases:
                self.assertIs(getattr(lean.functions, operation)(*arguments), FALLBACK, operation)
        with patch.object(module, "trial_prime", lambda n: False):
            for operation, arguments in (("is_prime", (97,)), ("factor_integer", (360,)), ("divisors", (360,)),
                                         ("totient", (36,))):
                self.assertIs(getattr(lean.functions, operation)(*arguments), FALLBACK, operation)
        with patch.object(module.kernel, "factor", lambda n: ((2, 1), (3, 1))):
            for operation in ("factor_integer", "divisors", "totient"):
                self.assertIs(getattr(lean.functions, operation)(12), FALLBACK, operation)
        with patch.object(module.kernel, "prime_certificate", lambda n: (True, ("trial", n))):
            self.assertIs(lean.functions.is_prime(10007 * 10009), FALLBACK)
            self.assertIs(lean.functions.factor_integer(10007 * 10009 * 4), FALLBACK)
        with patch.object(module, "gcd", lambda *args: 7):
            for operation in ("integer_gcd", "integer_lcm", "integer_xgcd"):
                self.assertIs(getattr(lean.functions, operation)(84, 30), FALLBACK, operation)
        with patch.object(module, "pow", lambda *args: 2, create=True):
            for operation, arguments in (("integer_gcd", (35, 18)), ("integer_xgcd", (35, 18)),
                                         ("integer_lcm", (35, 18))):
                self.assertIs(getattr(lean.functions, operation)(*arguments), FALLBACK, operation)
        with patch.object(module, "_inverse", lambda *args: 2):
            self.assertIs(lean.functions.mod_inverse(3, 7), FALLBACK)
            big = (1 << 7000) + 1
            self.assertIs(lean.functions.integer_xgcd(big, 3 ** 4000), FALLBACK)
        with patch.object(module, "_euclid", lambda x, y: (1, 0, 0)):
            self.assertIs(lean.functions.crt(((2, 3), (3, 5))), FALLBACK)
        self.assertEqual(lean.functions.integer_gcd(84, 30), 6)

    def test_active_budget_falls_back(self):
        arguments = {"integer_gcd": (84, 30), "integer_lcm": (84, 30), "integer_xgcd": (84, 30),
                     "mod_inverse": (3, 7), "crt": (((2, 3), (3, 5)),), "is_prime": (97,), "factor_integer": (360,),
                     "divisors": (360,), "totient": (36,)}
        for operation, args in arguments.items():
            self.assertIsNot(getattr(lean.functions, operation)(*args), FALLBACK)
            with computation():
                self.assertIs(getattr(lean.functions, operation)(*args), FALLBACK, operation)

    def test_lean_functions_are_the_modules(self):
        for operation in OPERATIONS:
            self.assertIs(getattr(lean.functions, operation), module.LEAN[operation])


if __name__ == "__main__":
    unittest.main()
