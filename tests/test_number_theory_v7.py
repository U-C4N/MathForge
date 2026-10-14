"""Integer identities, certified primality, and generalized CRT failure proofs."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from mathforge import number_theory as nt
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError


class NumberTheoryTests(unittest.TestCase):
    def checked(self, operation, **args):
        value, evidence = nt.produce(operation, args)
        self.assertTrue(nt.check_operation(operation, args, value, evidence), (operation, value, evidence))
        return value, evidence

    def test_signed_gcd_bezout_lcm_and_zero(self):
        for a, b in ((0, 0), (0, -12), (-30, 21), (12, 18), (-7, -7)):
            g, _ = self.checked("integer_gcd", a=a, b=b)
            value, _ = self.checked("integer_xgcd", a=a, b=b)
            self.assertEqual(value[0], g)
            self.assertEqual(a*value[1]+b*value[2], g)
            lcm, _ = self.checked("integer_lcm", a=a, b=b)
            self.assertEqual(lcm*g, abs(a*b))

    def test_inverse_and_generalized_crt(self):
        self.assertEqual(self.checked("mod_inverse", a=-3, modulus=11)[0], 7)
        self.assertEqual(self.checked("mod_inverse", a=8, modulus=1)[0], 0)
        with self.assertRaises(InvalidInput):
            nt.produce("mod_inverse", {"a": 6, "modulus": 9})
        self.assertEqual(self.checked("crt", congruences=((2, 6), (8, 9)))[0], nt.CRTResult(8, 18))
        self.assertEqual(self.checked("crt", congruences=((0, 2), (1, 2)))[0], nt.CRTResult(None, None))
        self.assertEqual(self.checked("crt", congruences=())[0], nt.CRTResult(0, 1))
        self.assertEqual(self.checked("crt", congruences=((-1, 3), (2, 6)))[0], nt.CRTResult(2, 6))

    def test_primality_proofs_and_negative_inputs(self):
        for n, expected in ((-11, False), (0, False), (1, False), (2, True), (97, True),
                            (561, False), (10007, True), (65537, True), (10403, False)):
            value, evidence = self.checked("is_prime", n=n)
            self.assertIs(value, expected)
            if n == 65537:
                self.assertEqual(evidence[0][0], "pocklington")
            self.assertFalse(nt.check_operation("is_prime", {"n": n}, not value, evidence))

    def test_factorization_divisors_and_totient(self):
        result, _ = self.checked("factor_integer", n=-360)
        self.assertEqual(result, nt.IntegerFactorization(-1, ((2, 3), (3, 2), (5, 1))))
        self.assertEqual(self.checked("factor_integer", n=1)[0], nt.IntegerFactorization(1, ()))
        self.assertEqual(self.checked("divisors", n=-12)[0], (1, 2, 3, 4, 6, 12))
        self.assertEqual(self.checked("totient", n=36)[0], 12)
        self.assertEqual(self.checked("totient", n=1)[0], 1)
        for operation in ("factor_integer", "divisors", "totient"):
            with self.assertRaises(InvalidInput):
                nt.produce(operation, {"n": 0})

    def test_certificate_corruption_and_producer_independence(self):
        cases = [("integer_gcd", {"a": 30, "b": 21}),
                 ("crt", {"congruences": ((2, 6), (8, 9))}),
                 ("crt", {"congruences": ((0, 2), (1, 2))}),
                 ("is_prime", {"n": 65537}), ("factor_integer", {"n": 360}),
                 ("divisors", {"n": 24}), ("totient", {"n": 45})]
        prepared = [(op, args, *nt.produce(op, args)) for op, args in cases]
        with patch.object(nt, "produce", side_effect=AssertionError("producer")), \
             patch.object(nt, "_prime_certificate", side_effect=AssertionError("prime producer")), \
             patch.object(nt, "_trial_factors", side_effect=AssertionError("factor producer")), \
             patch.object(nt, "_xgcd", side_effect=AssertionError("Bezout producer")):
            for operation, args, value, evidence in prepared:
                self.assertTrue(nt.check_operation(operation, args, value, evidence))
                self.assertFalse(nt.check_operation(operation, args, value, evidence + ("forged",)))
        value, evidence = nt.produce("factor_integer", {"n": 12})
        self.assertFalse(nt.check_operation("factor_integer", {"n": 12}, replace(value, factors=((2, 1), (3, 1))), evidence))
        prime, evidence = nt.produce("is_prime", {"n": 65537})
        certificate = evidence[0]
        forged = certificate[:-1] + ((1,),)
        self.assertFalse(nt.check_operation("is_prime", {"n": 65537}, prime, (forged,)))
        self.assertFalse(nt.check_operation("divisors", {"n": 12}, (1, 2, 3, 4, 6, 6), nt.produce("divisors", {"n": 12})[1]))

    def test_exact_inputs_and_budgets(self):
        for bad in (True, 1.0, "1"):
            with self.assertRaises(InvalidInput):
                nt.produce("is_prime", {"n": bad})
        with self.assertRaises(ResourceLimitError):
            nt.produce("is_prime", {"n": 10007}, limits=ComputationLimits(max_work=1))
        with self.assertRaises(ResourceLimitError):
            nt.produce("is_prime", {"n": 65537}, limits=ComputationLimits(max_evidence_items=2))
        with self.assertRaises(InvalidInput):
            nt.produce("crt", {"congruences": ((1, 0),)})


if __name__ == "__main__":
    unittest.main()
