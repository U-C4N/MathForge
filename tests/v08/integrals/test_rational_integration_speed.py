"""Rational integration on coefficient tuples: v0.7's values, the same verdicts, and no gcd per Q(x) operation."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mathforge
from mathforge import rational_integration as ri
from mathforge.limits import computation
from mathforge.model import Rational
from mathforge.serialization import to_data

from . import seeded

PINNED = Path(__file__).resolve().parents[2] / "fixtures" / "v8" / "rational_integration_v07.json"
Q = Rational


class RationalIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.sources = seeded.sources(mathforge)

    def produce(self, source):
        args = {"expression": source, "variable": self.x}
        with computation(fresh=True):
            value, evidence = ri.produce("rational_antiderivative", args)
        return args, value, evidence

    def check(self, args, value, evidence):
        with computation(fresh=True):
            return ri.check_operation("rational_antiderivative", args, value, evidence)

    def test_values_unchanged(self):
        """200 seeded integrands of degree <= 6 with repeated and complex poles: the antiderivatives are v0.7's (their
        to_data digests, pinned from a v0.7 tree), and the checker accepts each record."""
        pinned = json.loads(PINNED.read_text(encoding="utf-8"))
        self.assertEqual((pinned["seed"], pinned["count"]), (seeded.SEED, seeded.COUNT))
        for index, (source, expected) in enumerate(zip(self.sources, pinned["antiderivatives"], strict=True)):
            with self.subTest(index=index, source=str(source)):
                args, value, evidence = self.produce(source)
                self.assertEqual(seeded.digest(to_data(value)), expected)
                self.assertTrue(self.check(args, value, evidence))
        # The public value path returns the same value.
        source = self.sources[0]
        self.assertEqual(seeded.digest(to_data(mathforge.rational_antiderivative(source, self.x))),
                         pinned["antiderivatives"][0])

    def test_tampered_primitives_rejected(self):
        x = self.x
        cases = [mathforge.rational_function(1, x * x + 1, variable=x),
                 mathforge.rational_function(x ** 3, (x * x + 1) ** 2, variable=x),
                 mathforge.rational_function(3 * x + 1, (x - 1) * (x + 2) * (x * x + x + 1), variable=x),
                 mathforge.rational_function(1, x ** 5 - x + 1, variable=x)]
        for source in cases:
            args, value, evidence = self.produce(source)
            self.assertTrue(self.check(args, value, evidence))
            forgeries = []
            if value.log_terms:
                term = value.log_terms[0]
                residue = term.residue_polynomial
                forgeries.append(("residue", replace(value, log_terms=(replace(
                    term, residue_polynomial=(residue[0] + 1,) + residue[1:]),) + value.log_terms[1:])))
                rows = term.logarithm_coefficients
                forgeries.append(("logarithm", replace(value, log_terms=(replace(
                    term, logarithm_coefficients=((rows[0][0] + 1,) + rows[0][1:],) + rows[1:]),)
                    + value.log_terms[1:])))
                forgeries.append(("no logarithms", replace(value, log_terms=())))
                forgeries.append(("doubled logarithms", replace(value, log_terms=value.log_terms * 2)))
            if value.elementary_terms:
                b, d = value.elementary_terms[0]
                forgeries.append(("residual", replace(value, elementary_terms=((b + 1, d),)
                                                      + value.elementary_terms[1:])))
                forgeries.append(("no residual", replace(value, elementary_terms=())))
            part = value.rational_part
            forgeries.append(("rational part", replace(value, rational_part=part + x)))
            for label, forged in forgeries:
                with self.subTest(source=str(source), forgery=label):
                    try:
                        verdict = self.check(args, forged, evidence)
                    except (mathforge.MathForgeError, ArithmeticError):
                        verdict = False
                    self.assertFalse(verdict)

    def test_no_gcd_per_operation(self):
        """Q(x) fractions with constant or shared linear denominators add without a polynomial gcd; the trace checker
        builds no Polynomial record per operation."""
        calls = []
        original = ri._pgcd

        def counting(a, b):
            calls.append((a, b))
            return original(a, b)

        x = (Q(0), Q(1))
        with computation(fresh=True), patch.object(ri, "_pgcd", counting), \
                patch.object(mathforge.Polynomial, "gcd", side_effect=AssertionError("Polynomial.gcd")):
            total = ri._F(())
            for k in range(1, 21):
                total = total + ri._F((Q(k), Q(1, k)), (Q(1), Q(2)))  # (k + x/k) / (1 + 2x)
            constant = ri._F(())
            for k in range(1, 21):
                constant = constant + ri._F((Q(k),) + x[1:])
        self.assertEqual(calls, [])
        self.assertEqual(total.d, (Q(1), Q(2)))
        self.assertEqual(total.n, (Q(210), sum((Q(1, k) for k in range(1, 21)), Q(0))))
        self.assertEqual(constant.n, (Q(210), Q(20)))

    def test_fraction_equality_is_independent_of_reduction(self):
        """Unreduced and reduced fractions of one value compare equal by cross-multiplication; zero is zero."""
        with computation(fresh=True):
            self._fractions()

    def _fractions(self):
        a = ri._F((Q(1), Q(1)), (Q(1), Q(2)))  # (1 + x) / (1 + 2x)
        b = a * ri._F((Q(3), Q(1)), (Q(3), Q(1)))  # The same value: (1 + x)(3 + x) / ((1 + 2x)(3 + x)), reduced.
        self.assertNotEqual(a.d, b.d)
        self.assertEqual(ri._pmul(a.n, b.d), ri._pmul(b.n, a.d))
        self.assertEqual((a - a).n, ())
        quotient = b / a
        self.assertEqual(quotient.n, quotient.d)
        with self.assertRaises(ZeroDivisionError):
            a / ri._F(())

    def test_residue_field_arithmetic(self):
        """Q[t]/(t^2 + 1): t * t = -1, 1/t = -t, and a noninvertible divisor is refused as in v0.7."""
        with computation(fresh=True):
            self._residues()

    def _residues(self):
        modulus = (Q(1), Q(0), Q(1))
        t = ri._K((Q(0), Q(1)), modulus)
        self.assertEqual((t * t).p, (Q(-1),))
        self.assertEqual((ri._K((Q(1),), modulus) / t).p, (Q(0), Q(-1)))
        with self.assertRaises(mathforge.InvalidInput):
            ri._K((Q(1),), (Q(-1), Q(0), Q(1))) / ri._K((Q(1), Q(1)), (Q(-1), Q(0), Q(1)))  # 1/(1 + t), t^2 = 1


if __name__ == "__main__":
    unittest.main()
