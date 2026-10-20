"""Fast Rational arithmetic: values equal fractions.Fraction's, errors and the generic path equal v0.7's.

The error literals below were recorded on the unmodified v0.7 code (commit 007a4b3) before the fast paths existed.
"""

import copy
from fractions import Fraction
import itertools
import operator
import pickle
import random
import unittest
import weakref

from mathforge.errors import UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.matrices import Matrix
from mathforge.model import Add, Mul, Rational, Symbol, add, mul, trusted_rational


SAMPLES = (0, 1, -1, 7, -2**200 - 1, Rational(1, 2), Rational(-7, 3), Rational(2**64, 3**40), Rational(-1, 10**30))
ARITHMETIC = (operator.add, operator.sub, operator.mul, operator.truediv)
ORDERING = (operator.lt, operator.le, operator.gt, operator.ge)
# Denominators that share factors, so that random pairs reach every gcd branch of addition and multiplication.
DENOMINATORS = (1, 2, 3, 4, 6, 12, 35, 2**64, 3**40, 10**30)

# Step 1 records (v0.7): exception class and message.
NOT_EXACT = "Expected an exact integer or a supported MathForge expression; float and bool are not exact inputs"
NEGATIVE_POWER = "Negative powers and symbolic denominators are unsupported"
NOT_INTEGER_EXPONENT = "Only nonnegative Python integer exponents are supported"
X = Symbol("x", uid="00000000-0000-4000-8000-000000000801")


def fraction(value):
    return Fraction(value) if type(value) is int else Fraction(value.numerator, value.denominator)


def operand(rng):
    numerator = 0 if rng.random() < 0.05 else rng.randint(-2**rng.choice((3, 40, 130)), 2**rng.choice((3, 40, 130)))
    if rng.random() < 0.25:
        return numerator
    return Rational(numerator, rng.choice(DENOMINATORS) * rng.randint(1, 30))


def pairs():
    """Every ordered pair of SAMPLES, then 2,000 seeded random pairs; an int pair gets a Rational on the left."""
    rng = random.Random(20261019)
    for left, right in itertools.chain(itertools.product(SAMPLES, repeat=2),
                                       ((operand(rng), operand(rng)) for _ in range(2_000))):
        if type(left) is int and type(right) is int:
            left = Rational(left)
        yield left, right


class FastRationalTest(unittest.TestCase):
    def assertOutcome(self, thunk, error, message):
        with self.assertRaises(error) as caught:
            thunk()
        self.assertIs(type(caught.exception), error)
        self.assertEqual(str(caught.exception), message)

    def assertSameRational(self, result, expected):
        self.assertIs(type(result), Rational)
        self.assertEqual((result.numerator, result.denominator), (expected.numerator, expected.denominator))

    def test_arithmetic_matches_fraction(self):
        for left, right in pairs():
            for operation in ARITHMETIC:
                with self.subTest(left=left, right=right, operation=operation.__name__):
                    if operation is operator.truediv and fraction(right) == 0:
                        self.assertOutcome(lambda: operation(left, right), ZeroDivisionError, "division by zero")
                        continue
                    self.assertSameRational(operation(left, right), operation(fraction(left), fraction(right)))

    def test_results_are_ordinary_rationals(self):
        result = Rational(1, 6) + Rational(1, 3)
        self.assertEqual(result, Rational(1, 2))
        self.assertEqual(hash(result), hash(Rational(1, 2)))
        self.assertEqual(repr(result), "Rational(numerator=1, denominator=2)")

    def test_powers_match_fraction(self):
        rng = random.Random(20261019)
        bases = [Rational(value) if type(value) is int else value for value in SAMPLES]
        bases.extend(Rational(rng.randint(-50, 50), rng.randint(1, 50)) for _ in range(20))
        for base, exponent in itertools.product(bases, range(-3, 6)):
            with self.subTest(base=base, exponent=exponent):
                if exponent < 0:  # v0.7 refuses every negative exponent, a zero base included.
                    self.assertOutcome(lambda: base ** exponent, UnsupportedOperation, NEGATIVE_POWER)
                    continue
                self.assertSameRational(base ** exponent, fraction(base) ** exponent)

    def test_comparisons_match_fraction(self):
        for left, right in pairs():
            for operation in ORDERING:
                with self.subTest(left=left, right=right, operation=operation.__name__):
                    self.assertIs(operation(left, right), operation(fraction(left), fraction(right)))

    def test_other_operands_keep_the_generic_path(self):
        half = Rational(1, 2)
        with computation(fresh=True) as budget:
            self.assertEqual(half + X, Add((half, X)))
            self.assertEqual(budget.work, 16)  # v0.7's generic add charged 16 work units.
        self.assertEqual(half * X, Mul((half, X)))
        self.assertEqual(X - half, Add((Rational(-1, 2), X)))
        self.assertEqual(half * Matrix(((1, 2), (3, 4))), Matrix(((half, 1), (Rational(3, 2), 2))))
        self.assertEqual(Matrix(((1, 2), (3, 4))) * half, Matrix(((half, 1), (Rational(3, 2), 2))))
        # v0.7 raised here: only an int exponent is supported, so no Sqrt node is built.
        self.assertOutcome(lambda: half ** half, UnsupportedOperation, NOT_INTEGER_EXPONENT)
        self.assertOutcome(lambda: Rational(4) ** Rational(1, 2), UnsupportedOperation, NOT_INTEGER_EXPONENT)
        self.assertOutcome(lambda: half / X, UnsupportedOperation, "Only division by a rational constant is supported")

    def test_error_outcomes_unchanged(self):
        half = Rational(1, 2)
        cases = (
            (lambda: Rational(1) / 0, ZeroDivisionError, "division by zero"),
            (lambda: Rational(1) / Rational(0), ZeroDivisionError, "division by zero"),
            (lambda: 0 / Rational(0), ZeroDivisionError, "division by zero"),
            (lambda: 5 / Rational(0), ZeroDivisionError, "division by zero"),
            (lambda: Rational(0) ** -1, UnsupportedOperation, NEGATIVE_POWER),
            (lambda: Rational(2) ** -1, UnsupportedOperation, NEGATIVE_POWER),
            (lambda: half + True, TypeError, NOT_EXACT),
            (lambda: True + half, TypeError, NOT_EXACT),
            (lambda: half * True, TypeError, NOT_EXACT),
            (lambda: half / True, TypeError, NOT_EXACT),
            (lambda: half + 1.5, TypeError, NOT_EXACT),
            (lambda: 1.5 + half, TypeError, NOT_EXACT),
            (lambda: half + "a", TypeError, NOT_EXACT),
            (lambda: half - None, TypeError, NOT_EXACT),
            (lambda: half + Matrix(((1, 2), (3, 4))), TypeError, NOT_EXACT),
            (lambda: half ** True, UnsupportedOperation, NOT_INTEGER_EXPONENT),
            (lambda: half < "a", TypeError, "'<' not supported between instances of 'Rational' and 'str'"),
            (lambda: half < True, TypeError, "'<' not supported between instances of 'Rational' and 'bool'"),
            (lambda: half < 0.5, TypeError, "'<' not supported between instances of 'Rational' and 'float'"),
            (lambda: half >= None, TypeError, "'>=' not supported between instances of 'Rational' and 'NoneType'"),
            (lambda: 2 ** Rational(3), TypeError, "unsupported operand type(s) for ** or pow(): 'int' and 'Rational'"),
            (lambda: Rational(1.0), TypeError, "Rational requires exact Python integers, not float or bool"),
            (lambda: Rational(1, True), TypeError, "Rational requires exact Python integers, not float or bool"),
            (lambda: Rational(1, 0), ZeroDivisionError, "rational denominator is zero"),
        )
        for index, (thunk, error, message) in enumerate(cases):
            with self.subTest(case=index):
                self.assertOutcome(thunk, error, message)
        self.assertIs(Rational(1) == 1, False)  # Equality stays structural: an int is not a Rational.
        self.assertEqual(Rational(2, -4), Rational(-1, 2))
        self.assertIs(+half, half)

    def test_subclass_receivers_keep_the_v07_path(self):
        class ForeignRational(Rational):
            pass

        one, two = ForeignRational(1), ForeignRational(2)
        for thunk in (lambda: one + 1, lambda: 1 + one, lambda: one + Rational(1), lambda: Rational(1) + one,
                      lambda: one * 2, lambda: 2 * one, lambda: one - 1, lambda: 1 - one, lambda: one / 2,
                      lambda: 2 / one, lambda: Rational(1) / two, lambda: -one, lambda: two ** 2):
            with self.subTest(thunk=thunk):
                self.assertOutcome(thunk, TypeError, NOT_EXACT)
        self.assertSameRational(abs(ForeignRational(-1)), Fraction(1))
        with computation(fresh=True) as budget:
            self.assertIs(one < 2, True)  # v0.7 compared through a converted Rational(2) and two products.
            self.assertEqual(budget.work, 3)
            self.assertIs(Rational(1) < two, True)
            self.assertIs(2 > one, True)

    def test_add_and_mul_short_circuit(self):
        half = Rational(1, 2)
        records = (  # v0.7 values
            (add(), Fraction(0)), (add(5), Fraction(5)), (add(half), Fraction(1, 2)), (add(half, 1), Fraction(3, 2)),
            (add(half, half, half), Fraction(3, 2)), (add(1, 2, 3), Fraction(6)), (mul(), Fraction(1)),
            (mul(5), Fraction(5)), (mul(half), Fraction(1, 2)), (mul(half, 0), Fraction(0)),
            (mul(half, 2, half), Fraction(1, 2)), (mul(2, 3), Fraction(6)),
        )
        for result, expected in records:
            self.assertSameRational(result, expected)
        rng = random.Random(20261019)
        for _ in range(300):
            values = [operand(rng) for _ in range(rng.randint(0, 4))]
            fractions = [fraction(value) for value in values]
            with self.subTest(values=values):
                self.assertSameRational(add(*values), sum(fractions, Fraction(0)))
                product = Fraction(1)
                for value in fractions:
                    product *= value
                self.assertSameRational(mul(*values), product)
        self.assertOutcome(lambda: add(True), TypeError, NOT_EXACT)
        self.assertOutcome(lambda: add(half, True), TypeError, NOT_EXACT)
        self.assertOutcome(lambda: mul(1.5), TypeError, NOT_EXACT)
        with computation(fresh=True) as budget:  # A symbolic operand keeps v0.7's generic path and its charges.
            self.assertEqual(add(half, X), Add((half, X)))
            self.assertEqual(budget.work, 16)
            self.assertEqual(mul(half, X), Mul((half, X)))
            self.assertEqual(budget.work, 21)

    def test_pickles_keep_the_v07_state(self):
        # Rational(-3, 4) pickled by v0.7 with protocols 2 and 5: the state is the attribute dict.
        for blob in (b"\x80\x02cmathforge.model\nRational\nq\x00)\x81q\x01}q\x02(X\t\x00\x00\x00numeratorq\x03J\xfd\xff"
                     b"\xff\xffX\x0b\x00\x00\x00denominatorq\x04K\x04ub.",
                     b"\x80\x05\x95I\x00\x00\x00\x00\x00\x00\x00\x8c\x0fmathforge.model\x94\x8c\x08Rational\x94\x93\x94)"
                     b"\x81\x94}\x94(\x8c\tnumerator\x94J\xfd\xff\xff\xff\x8c\x0bdenominator\x94K\x04ub."):
            self.assertSameRational(pickle.loads(blob), Fraction(-3, 4))
        value = Rational(-2**70, 3)
        self.assertEqual(value.__getstate__(), {"numerator": -2**70, "denominator": 3})  # What v0.7 loads.
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertSameRational(pickle.loads(pickle.dumps(value, protocol)), Fraction(-2**70, 3))
        self.assertSameRational(copy.deepcopy([value])[0], Fraction(-2**70, 3))
        self.assertSameRational(copy.copy(value), Fraction(-2**70, 3))

    def test_weak_references_work_as_in_v07(self):
        value = Rational(3, 4)
        self.assertIs(weakref.ref(value)(), value)
        with self.assertRaises(TypeError):  # Slots: no attribute dict, unlike v0.7 (noted in the CHANGELOG).
            vars(value)

    def test_trusted_rational_is_the_public_object(self):
        self.assertEqual(trusted_rational(3, 4), Rational(3, 4))
        self.assertEqual(hash(trusted_rational(3, 4)), hash(Rational(3, 4)))
        self.assertEqual(repr(trusted_rational(3, 4)), "Rational(numerator=3, denominator=4)")
        self.assertIs(type(trusted_rational(-3, 4)), Rational)
        with computation(ComputationLimits(max_work=1, max_integer_bits=8)) as budget:
            value = trusted_rational(2**100, 3)  # No validation and no charge: the caller vouches for the integers.
            self.assertEqual((budget.work, value.numerator), (0, 2**100))

    def test_one_work_unit_per_operation(self):
        with computation() as budget:
            a, b = Rational(1, 2), Rational(1, 3)
            before = budget.work
            a + b
            a * b
            a - b
            a / b
            self.assertEqual(budget.work - before, 4)

    def test_every_fast_operation_charges_one_unit(self):
        a, b = Rational(1, 2), Rational(1, 3)
        operations = (lambda: a + 1, lambda: 1 + a, lambda: a - 1, lambda: 1 - a, lambda: a * 2, lambda: 2 * a,
                      lambda: a / 2, lambda: 2 / a, lambda: -a, lambda: abs(a), lambda: a ** 0,
                      lambda: a ** 3, lambda: a < b, lambda: a <= 1, lambda: 1 < a, lambda: a >= b, lambda: a > 2,
                      lambda: add(a, b), lambda: mul(a, b), lambda: add(a), lambda: mul(), lambda: Rational(3, 4))
        for index, thunk in enumerate(operations):
            with self.subTest(case=index), computation(fresh=True) as budget:
                thunk()
                self.assertEqual(budget.work, 1)
        with computation(fresh=True) as budget:  # add() and mul() charge one unit per pairwise step.
            add(a, b, 1)
            mul(a, b, 2, a)
            self.assertEqual(budget.work, 5)
        with computation(ComputationLimits(max_work=1), fresh=True) as budget:
            budget.tick()
            self.assertIs(a ** 1, a)  # As v0.7's power() returned the base: no charge, even at the work limit.
            self.assertEqual(budget.work, 1)

    def test_integer_bits_enforced(self):
        with computation(ComputationLimits(max_integer_bits=64)):
            with self.assertRaisesRegex(ResourceLimitError, "integer bits"):
                Rational(2**40) * Rational(2**40)
        cases = (
            (lambda: Rational(2**63) + Rational(2**63), "Computation limit exceeded: integer bits"),
            (lambda: Rational(1, 2**40) * Rational(1, 2**40), "Computation limit exceeded: integer bits"),
            (lambda: Rational(2**40) * 2**40, "Computation limit exceeded: integer bits"),
            (lambda: 2**40 / Rational(1, 2**40), "Computation limit exceeded: integer bits"),
            (lambda: add(2**63, 2**63), "Computation limit exceeded: integer bits"),
            (lambda: Rational(2) ** 63, "Computation limit exceeded: power integer bits"),  # v0.7's estimate
            (lambda: Rational(2**64), "Computation limit exceeded: integer bits"),
        )
        for index, (thunk, message) in enumerate(cases):
            with self.subTest(case=index), computation(ComputationLimits(max_integer_bits=64), fresh=True):
                self.assertOutcome(thunk, ResourceLimitError, message)
        with computation(ComputationLimits(max_integer_bits=64), fresh=True):
            self.assertEqual(Rational(2**40) + Rational(2**40), Rational(2**41))
        half = Rational(1, 2)
        with computation(ComputationLimits(max_work=2), fresh=True) as budget:
            half + 1
            half * 3
            self.assertOutcome(lambda: half - 1, ResourceLimitError, "Computation limit exceeded: work (2)")
            self.assertEqual(budget.work, 2)

    def test_no_limit_outside_computation(self):
        self.assertEqual((Rational(2**70) * 2**70).numerator, 2**140)
        self.assertEqual((Rational(2) ** 70_000).numerator, 2**70_000)


if __name__ == "__main__":
    unittest.main()
