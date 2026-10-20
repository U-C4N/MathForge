"""Forged values make the certify.upoly checkers answer False (final review B, I2).

check_definite_integral, check_rational_resultant and check_discriminant meet v0.7's criteria
(polynomial_exact.check_resultant and check_discriminant, a Sylvester determinant over Q) on the final review's 400
seeded cases, the true value and three forgeries each; check_factorization refuses a family of tampered
factorizations; compare gives the sign that an independent computation of the numbers gives (square roots by
isqrt, no Sturm chain), so the v2 check of compare_real refuses the two other signs.
"""

from __future__ import annotations

from math import isqrt
import random
import unittest

import mathforge as mf
from mathforge import polynomial_exact as exact
from mathforge.certify import upoly as C
from mathforge.kernels import upoly as kernels
from mathforge.limits import computation
from mathforge.model import Rational, trusted_rational
from mathforge.polynomial import Polynomial
from mathforge.trusted import upoly as tp

X = mf.Symbol("x", uid="00000000-0000-4000-8000-0000000000f2")


def review_cases(count=400):
    """The final review's cases (seed 11): f of degree 1 to 6, g of degree 0 to 5, small rational coefficients, and
    integration bounds."""
    rng = random.Random(11)
    cases = []
    for _ in range(count):
        def coefficients(low, high):
            degree = rng.randint(low, high)
            values = [Rational(rng.randint(-30, 30), rng.randint(1, 6)) for _ in range(degree)]
            return tuple(values) + (Rational(rng.choice((-7, -2, 1, 3)), rng.randint(1, 5)),)
        f, g = coefficients(1, 6), coefficients(0, 5)
        lower = Rational(rng.randint(-5, 5), rng.randint(1, 4))
        upper = Rational(rng.randint(-5, 5), rng.randint(1, 4))
        cases.append((f, g, lower, upper))
    return cases


class ResultantDiscriminantIntegralTest(unittest.TestCase):
    def test_verdicts_meet_v07_criteria(self):
        refused = 0
        for f, g, lower, upper in review_cases():
            F, G = Polynomial(f, X), Polynomial(g, X)
            with computation():
                resultant, discriminant = exact.resultant(F, G), exact.discriminant(F)
            for claim in (resultant, resultant + 1, -resultant if resultant else Rational(1),
                          resultant * 2 if resultant else Rational(2)):
                with computation():
                    v07 = exact.check_resultant(F, G, claim)
                self.assertIs(C.check_rational_resultant(f, g, claim), v07, (f, g, claim))
                refused += v07 is False
            for claim in (discriminant, discriminant + 1, -discriminant if discriminant else Rational(1),
                          discriminant / 3 if discriminant else Rational(1, 3)):
                with computation():
                    v07 = exact.check_discriminant(F, claim)
                self.assertIs(C.check_discriminant(f, claim), v07, (f, claim))
                refused += v07 is False
        # Every forgery differs from the value except -0 and 0 / 3, which are replaced by nonzero claims.
        self.assertEqual(refused, 400 * 6)

    def test_definite_integrals(self):
        for f, _, lower, upper in review_cases():
            with computation():
                value = mf.operations.integrate(Polynomial(f, X), X, lower, upper).unwrap()
            self.assertIs(C.check_definite_integral(f, lower, upper, value), True, (f, lower, upper))
            for forged in (value + Rational(1, 7), -value - 1, value * 2 + 1):
                self.assertIs(C.check_definite_integral(f, lower, upper, forged), False, (f, lower, upper, forged))

    def test_values_that_are_not_canonical_rationals(self):
        f, g = (Rational(-2), Rational(0), Rational(1)), (Rational(1), Rational(1))
        self.assertTrue(C.check_rational_resultant(f, g, Rational(-1)))
        self.assertTrue(C.check_discriminant(f, Rational(8)))
        self.assertTrue(C.check_definite_integral(f, Rational(0), Rational(3), Rational(3)))
        for forged in (-1, 8, 3, trusted_rational(-2, 2), trusted_rational(16, 2), trusted_rational(6, 2),
                       trusted_rational(3, -1), None):
            self.assertIs(C.check_rational_resultant(f, g, forged), False, forged)
            self.assertIs(C.check_discriminant(f, forged), False, forged)
            self.assertIs(C.check_definite_integral(f, Rational(0), Rational(3), forged), False, forged)


def factorization(values):
    """The lean producer's (monic factor, multiplicity, evidence) entries of a rational polynomial, sorted."""
    entries = []
    for part, multiplicity in kernels.square_free_integer(kernels.rational_primitive(values)):
        for factor, evidence in kernels.factor_square_free(part):
            entries.append((tp.monic_rationals(factor), multiplicity, evidence))
    entries.sort(key=lambda entry: (len(entry[0]) - 1, tuple((v.numerator, v.denominator) for v in entry[0])))
    return entries


def random_values(rng):
    """A product of two to four random integer polynomials, some squared, times a rational scale."""
    product = (1,)
    for _ in range(rng.randint(2, 4)):
        degree = rng.randint(1, 4)
        factor = tuple(rng.randint(-9, 9) for _ in range(degree)) + (rng.choice((1, 2, 3)),)
        for _ in range(rng.choice((1, 1, 2))):
            product = tp.mul(product, factor)
    scale = Rational(rng.choice((1, -1, 3)), rng.choice((1, 2, 5)))
    return tuple(Rational(value) * scale for value in product)


def shifted(values):
    """The monic f(x + 1) of a monic rational f, by Horner on its integer form."""
    result: tuple[int, ...] = ()
    for coefficient in reversed(tp.primitive(tp.from_rationals(values)[0])):
        result = tp.add(tp.mul(result, (1, 1)), (coefficient,))
    return tp.monic_rationals(tp.primitive(result))


def forgeries(entries, rng):
    """Tampered copies of a true factorization, each wrong in one respect."""
    forged = []
    first_values, first_multiplicity, first_evidence = entries[0]
    forged.append([(first_values, first_multiplicity + 1, first_evidence)] + entries[1:])  # A wrong multiplicity.
    forged.append([(first_values, 0, first_evidence)] + entries[1:])
    forged.append([(first_values, True, first_evidence)] + entries[1:])  # Not an int.
    forged.append([(tuple(value * 2 for value in first_values), first_multiplicity, first_evidence)] + entries[1:])
    forged.append(entries[1:])  # A factor left out.
    forged.append(entries + [entries[-1]])  # A factor twice.
    if len(entries) > 1:
        forged.append(entries[::-1])  # Out of order.
    moved = shifted(first_values)
    if moved != first_values:  # f(x + 1) with the evidence of f.
        forged.append([(moved, first_multiplicity, first_evidence)] + entries[1:])
    for index, (values, multiplicity, evidence) in enumerate(entries):
        if evidence == ("linear",):
            forged.append(entries[:index] + [(values, multiplicity, ("modular", 3, 1, ((1, 1),)))] + entries[index + 1:])
        else:
            _, prime, exponent, factors = evidence
            # (A lower exponent may still make a valid certificate; exponent 0 never does.)
            for wrong in (("linear",), ("modular", prime * prime, exponent, factors),
                          ("modular", prime, exponent, factors[1:]), ("modular", prime, exponent, ()),
                          ("modular", prime, 0, factors), ("modular", prime, exponent, factors + factors[:1])):
                forged.append(entries[:index] + [(values, multiplicity, wrong)] + entries[index + 1:])
    # Two factors of one multiplicity merged into a reducible one, with evidence of its own shape.
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i][1] == entries[j][1]:
                merged = tp.mul(tp.primitive(tp.from_rationals(entries[i][0])[0]),
                                tp.primitive(tp.from_rationals(entries[j][0])[0]))
                p, exponent, _, lifted = kernels.lift_data(merged)
                rest = [entry for k, entry in enumerate(entries) if k not in (i, j)]
                entry = (tp.monic_rationals(merged), entries[i][1], ("modular", p, exponent, tuple(lifted)))
                forged.append(sorted(rest + [entry], key=lambda e: (len(e[0]) - 1, tuple(
                    (v.numerator, v.denominator) for v in e[0]))))
                return forged
    return forged


class FactorizationTest(unittest.TestCase):
    def test_forged_factorizations_are_refused(self):
        rng = random.Random(4207)
        checked = 0
        for _ in range(40):
            values = random_values(rng)
            entries = factorization(values)
            with computation():
                self.assertIs(C.check_factorization(values, entries), True, values)
                for forged in forgeries(entries, rng):
                    with self.subTest(values=values, forged=forged):
                        self.assertIs(C.check_factorization(values, forged), False)
                    checked += 1
        self.assertGreater(checked, 400)


def surd(u: Rational, v: Rational, d: int):
    """The model expression u + v sqrt(d)."""
    return u + v * mf.sqrt(d)


_SCALE = 1 << 256


def scaled_sqrt(d: int) -> int:
    return isqrt(d * _SCALE * _SCALE)


def cubic_roots(a: int, b: int):
    """The real roots of (x**2 - a)(x - b), a not a square, in increasing order: (value * 2**256 rounded down, whether
    that is exact, the root as an expression)."""
    roots = [(-scaled_sqrt(a) - 1, False, -mf.sqrt(a)), (scaled_sqrt(a), False, mf.sqrt(a)),
             (b * _SCALE, True, Rational(b))]
    return sorted(roots, key=lambda root: root[0])


class CompareTest(unittest.TestCase):
    """certify.upoly.compare against the numbers computed to 256 bits by isqrt: rationals, quadratic surds
    u + v sqrt(d) and the roots of (x**2 - a)(x - b)."""

    def numbers(self, rng):
        found = []
        for _ in range(60):
            kind = rng.randint(0, 2)
            if kind == 0:
                r = Rational(rng.randint(-20, 20), rng.randint(1, 7))
                found.append((r, r.numerator * _SCALE // r.denominator, r.denominator == 1))
            elif kind == 1:
                u = Rational(rng.randint(-9, 9), rng.randint(1, 4))
                v = Rational(rng.choice((-3, -1, 1, 2)), rng.randint(1, 3))
                d = rng.choice((2, 3, 5, 6, 7, 10))
                scaled = (u.numerator * _SCALE // u.denominator) + v.numerator * scaled_sqrt(d) // v.denominator
                found.append((surd(u, v, d), scaled, False))
            else:
                a, b = rng.choice((2, 3, 5, 7, 11)), rng.randint(-4, 4)
                index = rng.randint(0, 2)
                value, exact_value, _ = cubic_roots(a, b)[index]
                found.append((mf.RealAlgebraicRoot(tp.mul((-a, 0, 1), (-b, 1)), index), value, exact_value))
        return found

    def test_signs_and_forged_signs(self):
        from mathforge.verification_v2 import _check_operation
        rng = random.Random(5150)
        numbers = self.numbers(rng)
        decided = 0
        for _ in range(200):
            (left, left_value, left_exact), (right, right_value, right_exact) = rng.sample(numbers, 2)
            difference = left_value - right_value
            if abs(difference) <= 8 and not (left_exact and right_exact):
                continue  # Within the 256-bit error of an isqrt: not decided here.
            truth = (difference > 0) - (difference < 0)
            with computation():
                self.assertEqual(C.compare(C.endpoint_of(left), C.endpoint_of(right)), truth, (left, right))
                for claim in (-1, 0, 1):
                    self.assertIs(_check_operation("compare_real", {"left": left, "right": right}, claim, ()),
                                  claim == truth, (left, right, claim))
            decided += 1
        self.assertGreater(decided, 150)

    def test_equal_numbers_in_two_forms(self):
        from mathforge.verification_v2 import _check_operation
        for a in (2, 3, 5, 7):
            for b in (-2, 0, 3):
                coefficients = tp.mul((-a, 0, 1), (-b, 1))
                pairs = [(expression, mf.RealAlgebraicRoot(coefficients, index))
                         for index, (_, _, expression) in enumerate(cubic_roots(a, b))]
                pairs.append((mf.RealAlgebraicRoot((-a, 0, 1), 1),
                              mf.RealAlgebraicRoot(coefficients, [root[2] for root in cubic_roots(a, b)].index(
                                  mf.sqrt(a)))))
                for left, right in pairs:
                    with computation():
                        self.assertEqual(C.compare(C.endpoint_of(left), C.endpoint_of(right)), 0, (left, right))
                        for claim in (-1, 1):
                            self.assertIs(_check_operation("compare_real", {"left": left, "right": right}, claim, ()),
                                          False)

    def test_isolating_intervals_are_never_points(self):
        """Roots met at bisection midpoints (0, 1, -1, 2 and 1/2 below) lie in half-open intervals (lo, hi]
        (final review B, M2)."""
        for roots in ((0, 1, -1, 2), (0, 3), (1, 2, 3, 4, 5), (-3, 0, 3)):
            p = (1,)
            for root in roots:
                p = tp.mul(p, (-root, 1))
            p = tp.mul(p, (-1, 2))  # 2x - 1
            intervals = C.isolate(p)
            self.assertEqual(len(intervals), len(roots) + 1)
            for lo, hi in intervals:
                self.assertLess(lo, hi)
            for root in (*roots, Rational(1, 2)):
                self.assertEqual(sum(1 for lo, hi in intervals if lo < root <= hi), 1, (roots, root))

    def test_invalid_descriptors_compare_to_nothing(self):
        self.assertIsNone(C.compare(None, ("rational", Rational(1))))
        self.assertIsNone(C.compare(("algebraic", ((-2, 0, 1), 2)), ("rational", Rational(1))))  # No third root.
        self.assertIsNone(C.endpoint_of(mf.Symbol("y")))


class RootRecordsTest(unittest.TestCase):
    def test_records_are_recognized_by_module_and_qualified_name(self):
        """check_root_records takes RootRecord by its module and qualified name, not by its name alone (final review
        B, M1), and refuses roots outside their isolating intervals."""
        from mathforge.algebraic import trusted_root_record
        p = tp.mul(tp.mul((-2, 0, 1), (-1, 1)), (-1, 1))  # (x**2 - 2)(x - 1)**2: -sqrt 2, 1 (twice), sqrt 2
        roots, multiplicities = (-mf.sqrt(2), Rational(1), mf.sqrt(2)), (1, 2, 1)
        records = tuple(trusted_root_record(root, m) for root, m in zip(roots, multiplicities))
        with computation():
            self.assertIs(C.check_root_records(p, records), True)

            class RootRecord:  # The same name in another module.
                def __init__(self, root, multiplicity):
                    self.root, self.multiplicity = root, multiplicity

            self.assertIs(C.check_root_records(p, tuple(RootRecord(r.root, r.multiplicity) for r in records)), False)
            for forged in ((roots[0], Rational(2), roots[2]), (roots[2], roots[1], roots[0]),
                           (roots[0], Rational(1), mf.sqrt(3))):
                self.assertIs(C.check_root_records(p, tuple(
                    trusted_root_record(root, m) for root, m in zip(forged, multiplicities))), False, forged)
            self.assertIs(C.check_root_records(p, tuple(
                trusted_root_record(root, 1) for root in roots)), False)  # A wrong multiplicity.


if __name__ == "__main__":
    unittest.main()
