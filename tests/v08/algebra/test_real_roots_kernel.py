"""Sturm chains, root counts and isolation of the producer kernel (mathforge.kernels.upoly) and of the checker
(mathforge.certify.upoly), the real root records built on them, and the nominal charging of their caches."""

import random
import unittest

import mathforge as mf
from mathforge import lean, operations
from mathforge.certify import upoly as certify
from mathforge.kernels import upoly as kernels
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Rational
from mathforge.root_isolation import count_real_roots, real_roots, verify_root_records
from mathforge.trusted import caches
from mathforge.trusted import upoly as tp

try:
    import sympy
except ImportError:  # pragma: no cover - a development dependency
    sympy = None

X = mf.Symbol("x", uid="00000000-0000-4000-8000-0000000000e1")


def product(*factors):
    result = (1,)
    for factor in factors:
        result = tp.mul(result, factor)
    return result


def seeded_polynomials(count=300):
    """Degrees 1..20, repeated and rational roots, 2**200 coefficients, x**n - a, roots closer than 10**-30."""
    rng = random.Random(9401)
    cases = []
    for index in range(count):
        kind = index % 6
        if kind == 0:
            degree = rng.randint(1, 20)
            p = tuple(rng.randint(-20, 20) for _ in range(degree)) + (rng.choice((1, -3, 7)),)
        elif kind == 1:  # repeated and rational roots
            p = product(*((-rng.randint(-5, 5), 1) for _ in range(rng.randint(1, 4))),
                        *(((rng.randint(-9, 9), rng.randint(1, 4)),) * rng.randint(1, 3)))
        elif kind == 2:  # large coefficients
            degree = rng.randint(2, 8)
            p = tuple(rng.randint(-2**200, 2**200) for _ in range(degree)) + (rng.randint(1, 2**200),)
        elif kind == 3:  # x**n - a
            n = rng.randint(2, 12)
            p = (-rng.randint(1, 50),) + (0,) * (n - 1) + (1,)
        elif kind == 4:  # two roots closer than 10**-30: (x - a)(x - a - 10**-31) times a quadratic
            a = rng.randint(-5, 5)
            scale = 10**31
            p = product((-a * scale, scale), (-(a * scale + 1), scale), (rng.randint(-3, 3), 0, 1))
        else:  # products with irrational quadratic and cubic factors
            p = product((-2, 0, 1), (-3, 0, 0, 1) if rng.random() < 0.5 else (-1, -1, 0, 1),
                        (rng.randint(-4, 4), 1))
        while p and not p[-1]:
            p = p[:-1]
        if len(p) >= 2:
            cases.append(p)
    return cases


class RealRootsKernelTest(unittest.TestCase):
    def test_counts_agree(self):
        """Kernel and checker chains count the same distinct roots on every interval, and SymPy agrees."""
        rng = random.Random(9402)
        for p in seeded_polynomials():
            part = kernels.squarefree_part(p)
            self.assertEqual(part, certify.squarefree_part(p))
            chain = kernels.sturm_chain(part)
            total = kernels.count_open(chain, None, None)
            self.assertEqual(total, certify.count_roots(part, None, None), p)
            if sympy is not None and max(abs(value) for value in p) < 2**64:
                x = sympy.Symbol("x")
                self.assertEqual(total, sympy.Poly(list(reversed(p)), x).count_roots(), p)
            intervals = kernels.isolate(part)
            self.assertEqual(len(intervals), total)
            for lower, upper in intervals:
                if lower == upper:
                    self.assertEqual(tp.sign_at(part, *lower), 0)
                else:
                    self.assertEqual(kernels.count_open(chain, lower, upper), 1)
            for _ in range(5):
                lo = Rational(rng.randint(-40, 40), rng.randint(1, 7))
                hi = lo + Rational(rng.randint(0, 40), rng.randint(1, 7))
                expected = certify.count_roots(part, lo, hi)
                open_count = kernels.count_open(chain, (lo.numerator, lo.denominator), (hi.numerator, hi.denominator))
                at_hi = tp.sign_at(part, hi.numerator, hi.denominator) == 0
                self.assertEqual(open_count + at_hi, expected, (p, lo, hi))

    def test_records_check_and_values(self):
        for p in seeded_polynomials(120):
            polynomial = mf.Polynomial(p, X)
            records = real_roots(polynomial)
            with self.subTest(p=p):
                self.assertTrue(verify_root_records(polynomial, records))
                self.assertEqual(len({record.root for record in records}), len(records))
                self.assertEqual(len(records), count_real_roots(polynomial))
        # Wrong records are refused: a dropped root, a wrong multiplicity, a swapped order.
        polynomial = mf.Polynomial(product((-1, 1), (-1, 1), (2, 1), (-2, 0, 1)), X)
        records = real_roots(polynomial)
        self.assertTrue(verify_root_records(polynomial, records))
        self.assertFalse(verify_root_records(polynomial, records[1:]))
        self.assertFalse(verify_root_records(polynomial, tuple(reversed(records))))
        wrong = tuple(mf.RootRecord(record.root, record.multiplicity + (record is records[0]))
                      for record in records)
        self.assertFalse(verify_root_records(polynomial, wrong))

    def test_lean_real_roots_equal_the_structured_path(self):
        lean._clear_caches()
        for p in seeded_polynomials(60):
            expression = mf.Polynomial(p, X).to_expression()
            value = lean.functions.real_roots(expression, X)
            expected = operations.real_roots(expression, X).unwrap()
            self.assertIsNot(value, lean.FALLBACK)
            self.assertEqual(mf.to_data(value), mf.to_data(expected))

    def test_cache_hits_charge_the_nominal_cost(self):
        """The work counter is identical on a cold and a warm call."""
        polynomial = mf.Polynomial(product((-2, 0, 1), (-1, -1, 0, 0, 0, 1)), X)
        counters = []
        for warm in (False, True):
            if not warm:
                caches.clear_all()
            with computation(ComputationLimits(), fresh=True) as budget:
                real_roots(polynomial)
                counters.append((budget.work, budget.node_count, budget.refinements, budget.evidence_count))
        self.assertEqual(counters[0], counters[1])
        self.assertGreater(counters[0][0], 0)

    def test_compare_real_signs(self):
        rng = random.Random(9403)
        values = [mf.sqrt(2), mf.sqrt(3) + 1, mf.Rational(7, 5), mf.sqrt(2) + mf.sqrt(3), 2 * mf.sqrt(6) - 1,
                  mf.RealAlgebraicRoot((-2, 0, 0, 1), 0), mf.sqrt(5) * mf.sqrt(7), (mf.sqrt(2) + 1) ** 3]
        for _ in range(60):
            left, right = rng.choice(values), rng.choice(values)
            expected = operations.compare_real(left, right).unwrap()
            self.assertEqual(mf.compare_real(left, right), expected)
            self.assertEqual(lean.functions.compare_real(left, right), expected)
            if expected and type(left) is not mf.RealAlgebraicRoot and type(right) is not mf.RealAlgebraicRoot:
                sign, bits = kernels.separate(left, right)
                self.assertEqual(sign, expected)
                self.assertTrue(certify.check_sign(left, right, sign, bits))
                self.assertFalse(certify.check_sign(left, right, -sign, bits))


if __name__ == "__main__":
    unittest.main()
