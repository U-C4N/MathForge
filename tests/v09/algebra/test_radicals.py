"""The radicals family of mathforge.algebra: denest_sqrt.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register algebra radicals

Every expected value comes from the card or from a reference of this module, never from the producer:
- brute(): brute force over the candidates for x. With a = A/Da and b = B/Db in lowest terms, x + y = a and
  4*x*y = b^2*d make t = 2*Da*Db*x a root of the monic integer polynomial t^2 - 2*A*Db*t + B^2*Da^2*d, so t is an
  integer, and x > y > 0 puts it strictly between A*Db and 2*A*Db: every such t is tried.
- backward(): claims built from their values. y > 0 and a ratio w > 0 come first, then x0 = w^2*d*y, a = x0 + y and
  b = 2*sign*w*y, so that sqrt(x0*y) = w*y*sqrt(d) and (sqrt(x0) + sign*sqrt(y))^2 = a + b*sqrt(d).
- closed_form(): the card's formula with Fractions, x, y = (a +- c)/2 for c = sqrt(a^2 - b^2*d) when that is
  rational, for inputs too long for brute force; brute() and backward() confirm it.
- in_domain() decides the card's domain with Fractions, decimal_sign() the sign of a + b*sqrt(d) with Decimal
  arithmetic, and approximates() confirms to 60 digits that sqrt(x) + sign*sqrt(y) is sqrt(a + b*sqrt(d)).
tests/v09/algebra/test_radicals_checker.py tests the checker on its own.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from math import gcd, isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.algebra import radicals as family
from mathforge.areas.algebra.radicals import DenestedRoot
from mathforge.contracts import CheckStatus, Completeness, ErrorInfo, ExecutionStatus, Outcome
from mathforge.deferred import Reason, Unevaluated
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation

from .. import support

Q = mf.Rational
OP = "denest_sqrt"
FAST = 2.0  # Seconds: a full-size request completes, and a request beyond the budget fails, well within this.
MAX_BITS = ComputationLimits().max_integer_bits
NON_SQUARES = tuple(d for d in range(2, 80) if isqrt(d) * isqrt(d) != d)
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
NO_DENESTING = Reason("no_applicable_rule",
                      "a^2 - b^2*d is not a rational square: no denesting of the form sqrt(x) +- sqrt(y)")
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact number: a float, a bool, a str, None and two containers.
VERIFIED, FAILED, UNKNOWN = CheckStatus.VERIFIED, CheckStatus.FAILED, CheckStatus.UNKNOWN

# The known values of the card, (a, b, d) with (x, y, sign), and the inputs without a value and outside the domain.
KNOWN = (
    ((3, 2, 2), (2, 1, 1)),  # sqrt(2) + 1
    ((5, 2, 6), (3, 2, 1)),  # sqrt(3) + sqrt(2)
    ((7, -4, 3), (4, 3, -1)),  # 2 - sqrt(3)
    ((2, 1, 3), (Fraction(3, 2), Fraction(1, 2), 1)),  # sqrt(3/2) + sqrt(1/2)
)
CARD_UNEVALUATED = (1, 1, 2)  # a^2 - b^2*d = -1.
CARD_INVALID = ((-3, 2, 2), (1, -1, 2))  # a + b*sqrt(d) < 0.

# Further denestings: both signs, rational a and b, c = sqrt(a^2 - b^2*d) no integer, d with a square factor, x a
# rational square.
SPECIAL = (
    ((3, -2, 2), (2, 1, -1)),  # sqrt(2) - 1
    ((7, 4, 3), (4, 3, 1)),  # 2 + sqrt(3)
    ((5, -2, 6), (3, 2, -1)),  # sqrt(3) - sqrt(2)
    ((Fraction(5, 2), -1, 6), (Fraction(3, 2), 1, -1)),  # c = 1/2: sqrt(3/2) - 1
    ((Fraction(3, 2), Fraction(1, 2), 5), (Fraction(5, 4), Fraction(1, 4), 1)),
    ((Fraction(7, 3), Fraction(-2, 3), 10), (Fraction(5, 3), Fraction(2, 3), -1)),
    ((3, 1, 8), (2, 1, 1)),  # sqrt(8) = 2*sqrt(2): sqrt(2) + 1 again.
    ((19, -6, 2), (18, 1, -1)),  # 3*sqrt(2) - 1
    ((Fraction(11, 4), Fraction(-3, 2), 2), (Fraction(9, 4), Fraction(1, 2), -1)),  # 3/2 - sqrt(1/2)
)

# Inputs inside the domain without a denesting: a^2 - b^2*d negative or no rational square, a <= 0.
UNEVALUATED = (CARD_UNEVALUATED, (0, 1, 2), (-1, 2, 2), (Fraction(-1, 2), 1, 2), (4, 1, 3), (6, 1, 2), (2, 1, 2),
               (3, -1, 2), (Fraction(1, 2), Fraction(1, 3), 3), (Fraction(9, 4), Fraction(-1, 2), 5), (10 ** 40, 1, 2))

# Inputs outside the domain: b = 0, d < 2, d a perfect square, a + b*sqrt(d) < 0.
INVALID = (*CARD_INVALID, (4, 0, 2), (Fraction(1, 2), 0, 3), (-1, 0, 2), (3, 2, 1), (3, 2, 0), (3, 2, -2),
           (3, 2, -(2 ** 64)), (5, 2, 4), (10, 2, 9), (3, 2, 16), (10 ** 30 + 1, 2 * 10 ** 15, 10 ** 30),
           (3, 2, 2 ** 64), (-1, -1, 2), (0, -1, 2), (-5, 3, 2), (2, -2, 2), (Fraction(-1, 2), Fraction(1, 3), 2),
           (Fraction(1, 3), Fraction(-1, 2), 2), (-(10 ** 40), 1, 2),
           ((10 ** 40 + 1) ** 2 + 1, 2, (10 ** 40 + 1) ** 2))  # An odd square past a float square root's exact range.


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

def fraction(value: object) -> Fraction:
    """An exact number, an int, a Fraction or a Rational, as a Fraction."""
    return Fraction(value.numerator, value.denominator) if type(value) is Q else Fraction(value)


def exact(value: object) -> object:
    """An argument as a request holds it: an int for an integer, a Rational otherwise."""
    value = fraction(value)
    return value.numerator if value.denominator == 1 else Q(value.numerator, value.denominator)


def rational(value: object) -> Q:
    """A field of the record: always a Rational, an integral one included."""
    value = fraction(value)
    return Q(value.numerator, value.denominator)


def args_of(a: object, b: object, d: int) -> dict[str, object]:
    return {"a": exact(a), "b": exact(b), "d": d}


def request_of(args: dict[str, object]) -> mf.OperationRequest:
    """The request that operations_v6.run builds for the arguments, which an Unevaluated record carries."""
    return mf.OperationRequest(OP, tuple(args.items()), 6)


def record(x: object, y: object, sign: int) -> DenestedRoot:
    return DenestedRoot(rational(x), rational(y), sign)


def in_domain(a: object, b: object, d: int) -> bool:
    """The card's domain, with Fractions: b != 0, d >= 2 no perfect square, and a + b*sqrt(d) >= 0. Of two terms of
    opposite signs the one with the larger square, a^2 or b^2*d, decides (they differ, as d is no square)."""
    a, b = Fraction(a), Fraction(b)
    if b == 0 or d < 2 or isqrt(d) * isqrt(d) == d:
        return False
    if a >= 0 and b > 0:
        return True
    if a <= 0 and b < 0:
        return False
    return (a > 0) == (a * a > b * b * d)  # Nonzero a and b of opposite signs.


def decimal_sign(a: object, b: object, d: int) -> int:
    """The sign of a + b*sqrt(d) by Decimal arithmetic to 80 digits, for small arguments with d >= 2."""
    a, b = Fraction(a), Fraction(b)
    with localcontext() as context:
        context.prec = 80
        first, second = Decimal(a.numerator) / Decimal(a.denominator), Decimal(b.numerator) / Decimal(b.denominator)
        value = first + second * Decimal(d).sqrt()
        assert abs(value) > Decimal(10) ** -40, (a, b, d)  # Never 0: far from the precision for small arguments.
        return 1 if value > 0 else -1


def brute(a: object, b: object, d: int) -> tuple[Fraction, Fraction, int] | None:
    """The denesting of (a, b, d) inside the domain by brute force over t = 2*Da*Db*x (module docstring), None when
    there is none; the search takes about A*Db steps, so it serves small arguments only."""
    a, b = Fraction(a), Fraction(b)
    if a <= 0:
        return None  # x + y = a > 0 for x > y > 0.
    big_a, a_denominator, big_b, b_denominator = a.numerator, a.denominator, b.numerator, b.denominator
    total, product = 2 * big_a * b_denominator, big_b * big_b * a_denominator * a_denominator * d
    found = [t for t in range(big_a * b_denominator + 1, total) if t * (total - t) == product]
    assert len(found) <= 1, found  # The roots t and total - t of one quadratic, and t > total/2.
    if not found:
        return None
    scale = 2 * a_denominator * b_denominator
    return Fraction(found[0], scale), Fraction(total - found[0], scale), 1 if b > 0 else -1


def closed_form(a: object, b: object, d: int) -> tuple[Fraction, Fraction, int] | None:
    """The card's closed form with Fractions: ((a + c)/2, (a - c)/2, sign(b)) when a > 0 and a^2 - b^2*d = c^2 for a
    rational c > 0, None otherwise."""
    a, b = Fraction(a), Fraction(b)
    n = a * a - b * b * d
    if a <= 0 or n <= 0:
        return None
    p, q = isqrt(n.numerator), isqrt(n.denominator)
    if p * p != n.numerator or q * q != n.denominator:
        return None
    c = Fraction(p, q)
    return (a + c) / 2, (a - c) / 2, 1 if b > 0 else -1


def backward(y: Fraction, w: Fraction, d: int, sign: int) -> tuple[tuple[Fraction, Fraction, int],
                                                                     tuple[Fraction, Fraction, int]]:
    """((a, b, d), (x, y, sign)): a claim built from its value for y > 0, w > 0 and d no square. x0 = w^2*d*y gives
    sqrt(x0*y) = w*y*sqrt(d), so (sqrt(x0) + sign*sqrt(y))^2 = a + b*sqrt(d) for a = x0 + y and b = 2*sign*w*y;
    x0 = y would make d = 1/w^2 a square, and the canonical record orders x0 and y decreasingly."""
    x0 = w * w * d * y
    return (x0 + y, 2 * sign * w * y, d), (max(x0, y), min(x0, y), sign)


def at_the_size_bound(j: int, k: int, sign: int) -> tuple[tuple[tuple[Fraction, Fraction, int],
                                                                  tuple[Fraction, Fraction, int]], ...]:
    """Two true claims whose x has an integer of exactly bitlen(A) + bitlen(Da) + bitlen(Db) bits, the longest that the
    checker's size filter must admit (the checker's module docstring). With B = 2^j - 1, Da = 2^k - 1, n = B*Da,
    Db = n^2 + 1 and d = n^2 + 2: a = 1/Da and b = sign*B/Db give x = d/(2*Da*Db), the full denominator, and
    y = B^2*Da/(2*Db). With A = 2^k - 2 and Db' = 2^j - 1: a = A, b = sign/Db' and d' = 2*A*Db' - 1 (3 modulo 4, no
    square) give x = d'/(2*Db'), a numerator of almost 2*A*Db', and y = 1/(2*Db')."""
    big_b, a_denominator = (1 << j) - 1, (1 << k) - 1
    n = big_b * a_denominator
    b_denominator, d = n * n + 1, n * n + 2
    first = ((Fraction(1, a_denominator), Fraction(sign * big_b, b_denominator), d),
             (Fraction(d, 2 * a_denominator * b_denominator),
              Fraction(big_b * big_b * a_denominator, 2 * b_denominator), sign))
    big_a, b_denominator = (1 << k) - 2, (1 << j) - 1
    d = 2 * big_a * b_denominator - 1
    second = ((Fraction(big_a), Fraction(sign, b_denominator), d),
              (Fraction(d, 2 * b_denominator), Fraction(1, 2 * b_denominator), sign))
    return first, second


def approximates(a: object, b: object, d: int, x: object, y: object, sign: int) -> bool:
    """sqrt(x) + sign*sqrt(y) agrees with sqrt(a + b*sqrt(d)) to 60 significant digits (Decimal arithmetic to 80)."""
    def decimal(value: object) -> Decimal:
        number = fraction(value)
        return Decimal(number.numerator) / Decimal(number.denominator)

    with localcontext() as context:
        context.prec = 80
        right = (decimal(a) + decimal(b) * Decimal(d).sqrt()).sqrt()
        left = decimal(x).sqrt() + sign * decimal(y).sqrt()
        return abs(left - right) <= right * Decimal(10) ** -60


def beyond(bits: int) -> tuple[tuple[Fraction, Fraction, int], tuple[Fraction, Fraction]]:
    """((a, b, 2), (x, y)): arguments of about 2*bits bits whose value has y = 1/(z*Q) of about 4*bits bits. For
    u*s - 2*v*w = 1, Brahmagupta's identity gives 1 + 2*z^2 = Q*R with z = u*w + v*s, Q = u^2 + 2*v^2 and
    R = s^2 + 2*w^2; then x = 2*z/Q and y = 1/(z*Q) have x + y = R/z = a and 4*x*y = 8/Q^2 = b^2*2 for b = 2/Q."""
    rng = random.Random(bits)
    while True:
        u = rng.getrandbits(bits) | 1 << (bits - 1) | 1
        v = rng.getrandbits(bits) | 1 << (bits - 1)
        if gcd(u, 2 * v) != 1:
            continue
        s = pow(u, -1, 2 * v)  # u*s = 1 modulo 2v, 0 < s < 2v.
        w = (u * s - 1) // (2 * v)
        z, big_q, big_r = u * w + v * s, u * u + 2 * v * v, s * s + 2 * w * w
        assert 1 + 2 * z * z == big_q * big_r and z > 0
        x, y = Fraction(2 * z, big_q), Fraction(1, z * big_q)
        a, b = Fraction(big_r, z), Fraction(2, big_q)
        assert x + y == a and 4 * x * y == b * b * 2 and x > y > 0
        return (a, b, 2), (x, y)


def forged_rational(numerator: object, denominator: object = 1, *, missing: str = "") -> Q:
    """A Rational made with object.__new__, whose fields may break lowest terms, a positive denominator or exactness;
    the field named by missing stays unset."""
    value = object.__new__(Q)
    for name, item in (("numerator", numerator), ("denominator", denominator)):
        if name != missing:
            object.__setattr__(value, name, item)
    return value


class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class SubRational(Q):
    """A Rational subclass, equal to the right number but no Rational of the model."""


class RadicalsTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: DenestedRoot) -> None:
        """A completed v6 Result of that exact record, with the evidence () and its four checks, which mf.verify
        accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), DenestedRoot)
        self.assertEqual((type(result.value.x), type(result.value.y), type(result.value.sign)), (Q, Q, int))
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertUnevaluated(self, result: mf.Result) -> None:
        """A completed Result without a value: NO_CONCLUSION with the card's Unevaluated record, whose request is the
        Result's own; mf.verify verifies the source and the deferral and leaves the mathematics UNKNOWN, also after
        the JSON round trip."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertIs(result.outcome, Outcome.NO_CONCLUSION)
        self.assertIsNone(result.value)
        self.assertEqual(result.steps, ())
        self.assertIs(result.completeness, Completeness.UNKNOWN)
        self.assertEqual(result.deferred, Unevaluated("simplify", result.request, NO_DENESTING))
        expected = {"result.source": VERIFIED, "result.unevaluated": VERIFIED, "result.mathematics": UNKNOWN}
        self.assertEqual(support.statuses(mf.verify(result)), expected)
        decoded = mf.loads(mf.dumps(result))
        self.assertEqual(decoded, result)
        self.assertEqual(support.statuses(mf.verify(decoded)), expected)
        with self.assertRaises(mf.OperationError):
            result.unwrap()

    def assertProduces(self, args: dict[str, object], value: DenestedRoot) -> None:
        """produce() answers exactly that record with the evidence (), and the checker accepts the claim."""
        produced, evidence = family.produce(OP, dict(args))
        self.assertEqual((produced, evidence), (value, ()), args)
        self.assertEqual((type(produced.x), type(produced.y), type(produced.sign)), (Q, Q, int))
        self.assertIs(support.verdict(family, OP, args, produced, evidence), True, args)

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts: the card's four denestings and
        further ones, the card's input without a denesting (NO_CONCLUSION) and its two inputs outside the domain."""
        for (a, b, d), (x, y, sign) in KNOWN + SPECIAL:
            with self.subTest(a=a, b=b, d=d):
                self.assertTrue(in_domain(a, b, d))
                self.assertEqual(closed_form(a, b, d), (x, y, sign))
                self.assertEqual(brute(a, b, d), (x, y, sign))
                self.assertTrue(approximates(a, b, d, x, y, sign))
                result = support.run(OP, **args_of(a, b, d))
                self.assertVerified(result, record(x, y, sign))
                self.roundtrip(result)
        self.assertTrue(in_domain(*CARD_UNEVALUATED))
        self.assertIsNone(brute(*CARD_UNEVALUATED))
        self.assertUnevaluated(support.run(OP, **args_of(*CARD_UNEVALUATED)))
        for a, b, d in CARD_INVALID:
            with self.subTest(a=a, b=b, d=d):
                self.assertFalse(in_domain(a, b, d))
                self.assertEqual(decimal_sign(a, b, d), -1)
                result = support.run(OP, **args_of(a, b, d))
                self.assertInvalidInput(result)
                self.assertIn("a + b*sqrt(d) >= 0", result.error.message)
        # A Rational argument of denominator 1 is the same exact number as the int, in a request of its own.
        result = support.run(OP, a=Q(3), b=Q(2), d=2)
        self.assertVerified(result, record(2, 1, 1))
        self.assertEqual(dict(result.request.arguments), {"a": Q(3), "b": Q(2), "d": 2})
        self.roundtrip(result)

    def test_no_denesting_is_unevaluated(self):
        """Inside the domain, a^2 - b^2*d negative or no rational square, and a <= 0, give the card's Unevaluated
        record and no value; brute force confirms that no denesting exists."""
        for a, b, d in UNEVALUATED:
            with self.subTest(a=a, b=b, d=d):
                self.assertTrue(in_domain(a, b, d))
                self.assertIsNone(closed_form(a, b, d))
                if Fraction(a).numerator.bit_length() < 20:
                    self.assertIsNone(brute(a, b, d))
                result = support.run(OP, **args_of(a, b, d))
                self.assertUnevaluated(result)
                value, evidence = family.produce(OP, args_of(a, b, d))
                self.assertEqual((value, evidence), (result.deferred, ()))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the card's domain, and every Rational made with object.__new__ that
        breaks the canonical form while run() admits it. produce() tests the types itself too."""
        small = args_of(3, 2, 2)
        inexact = (*STAND_INS, Fraction(1, 2), Integer(3), SubRational(3), mf.sqrt(2))
        for name in small:
            for item in inexact + ((Q(5), Q(5, 2)) if name == "d" else ()):
                with self.subTest(argument=name, item=item):
                    self.assertInvalidInput(support.run(OP, **dict(small, **{name: item})))
        for a, b, d in INVALID:
            with self.subTest(a=a, b=b, d=d):
                self.assertFalse(in_domain(a, b, d))
                self.assertInvalidInput(support.run(OP, **args_of(a, b, d)))
                with self.assertRaises(InvalidInput):
                    family.produce(OP, args_of(a, b, d))
        for (a, b, d), text in (((4, 0, 2), "b != 0"), ((3, 2, 1), "d >= 2"), ((3, 2, -5), "d >= 2"),
                                ((5, 2, 4), "no perfect square"), ((3, 2, 2 ** 64), "no perfect square"),
                                ((-3, 2, 2), "a + b*sqrt(d) >= 0"), ((2, -2, 2), "a + b*sqrt(d) >= 0"),
                                ((0, -1, 2), "a + b*sqrt(d) >= 0")):
            with self.subTest(message=text, a=a, b=b, d=d):
                self.assertIn(text, support.run(OP, **args_of(a, b, d)).error.message)
        # Rationals made with object.__new__ outside lowest terms, with a denominator 0 or negative, or a bool field:
        # run() admits them by their class, and produce() refuses them.
        forgeries = ((forged_rational(10, 2), "lowest terms"), (forged_rational(-4, -2), "positive denominator"),
                     (forged_rational(6, 4), "lowest terms"), (forged_rational(5, -1), "positive denominator"),
                     (forged_rational(2, 0), "positive denominator"), (forged_rational(True, 1), "exact integers"))
        for item, text in forgeries:
            for name in ("a", "b"):
                with self.subTest(argument=name, forged=(item.numerator, item.denominator)):
                    result = support.run(OP, **dict(small, **{name: item}))
                    self.assertInvalidInput(result)
                    self.assertIn(text, result.error.message)
        # produce() itself refuses every stand-in, each forged Rational (a float field and a missing field too, which
        # run()'s inspection of the arguments meets first), other operations and other argument sets.
        for name in small:
            for item in (*STAND_INS, False, Fraction(3), Integer(3), SubRational(3), forged_rational(3.0),
                         forged_rational(3, missing="numerator"), forged_rational(3, missing="denominator"),
                         forged_rational(10, 2), forged_rational(3, -1)) + ((Q(5),) if name == "d" else ()):
                with self.subTest(argument=name, item=type(item).__name__, path="produce"):
                    with self.assertRaises(InvalidInput):
                        family.produce(OP, dict(small, **{name: item}))
        for args in (dict(small, c=1), {"a": 3, "b": 2}, {}, tuple(small.items()), None, [("a", 3)]):
            with self.subTest(arguments=args), self.assertRaises(InvalidInput):
                family.produce(OP, args)
        for operation in ("denest", "Denest_sqrt", "denest_sqrt ", "continued_fraction", "", None, 1):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(small))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OP, a=-3, b=2, d=2)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_the_domain_is_decided_exactly(self):
        """a + b*sqrt(2) within 10^-900 of 0, past any fixed precision, from solutions of p^2 - 2*q^2 = 1 and = -1 of
        about 3,000 bits (the recurrence (p, q) -> (3p + 4q, 2p + 3q) keeps the norm). For the norm 1, p - q*sqrt(2)
        is positive with the denesting sqrt((p + 1)/2) - sqrt((p - 1)/2), and -p + q*sqrt(2) is negative; for the norm
        -1, p - q*sqrt(2) is negative and -p + q*sqrt(2) positive without a denesting (a < 0)."""
        p_one, q_one, p_minus, q_minus = 3, 2, 1, 1
        for _ in range(1_200):
            p_one, q_one = 3 * p_one + 4 * q_one, 2 * p_one + 3 * q_one
            p_minus, q_minus = 3 * p_minus + 4 * q_minus, 2 * p_minus + 3 * q_minus
        self.assertEqual((p_one * p_one - 2 * q_one * q_one, p_minus * p_minus - 2 * q_minus * q_minus), (1, -1))
        self.assertGreater(min(p_one, p_minus).bit_length(), 3_000)
        for sign in (-1, 1):
            with self.subTest(norm=1, sign=sign):
                value = record(Fraction(p_one + 1, 2), Fraction(p_one - 1, 2), sign)
                self.assertVerified(support.run(OP, **args_of(p_one, sign * q_one, 2)), value)
        for a, b in ((-p_one, q_one), (p_minus, -q_minus)):
            with self.subTest(a_sign=1 if a > 0 else -1):
                result = support.run(OP, **args_of(a, b, 2))
                self.assertInvalidInput(result)
                self.assertIn("a + b*sqrt(d) >= 0", result.error.message)
        self.assertUnevaluated(support.run(OP, **args_of(-p_minus, q_minus, 2)))
        self.assertUnevaluated(support.run(OP, **args_of(p_minus, q_minus, 2)))  # a^2 - 2*b^2 = -1.

    def test_resource_limits(self):
        """An integer of 2**65536 in any argument, also inside a Rational, and max_work=1 exceed the budget. No
        argument takes a matrix, so support.oversized_matrix() has nothing to exceed here: in any argument it is no
        exact number, invalid input. A true value whose integers exceed max_integer_bits is beyond the budget too
        (test_values_beyond_max_integer_bits)."""
        small = args_of(3, 2, 2)
        for name in small:
            with self.subTest(argument=name):
                self.assertResourceLimit(support.run(OP, **dict(small, **{name: support.HUGE})))
                with self.assertRaises(ResourceLimitError):
                    family.produce(OP, dict(small, **{name: support.HUGE}))
        for name in ("a", "b"):
            for item in (Q(support.HUGE + 1, 3), Q(1, support.HUGE), Q(-support.HUGE - 1, support.HUGE)):
                with self.subTest(argument=name, rational=True):
                    self.assertResourceLimit(support.run(OP, **dict(small, **{name: item})))
                    with self.assertRaises(ResourceLimitError):
                        family.produce(OP, dict(small, **{name: item}))
        for a, b, d in (*(args for args, _ in KNOWN), CARD_UNEVALUATED):
            with self.subTest(a=a, b=b, d=d, max_work=1):
                self.assertResourceLimit(support.run(OP, limits=ComputationLimits(max_work=1), **args_of(a, b, d)))
        matrix = support.oversized_matrix()
        for name in small:
            with self.subTest(argument=name, matrix=True):
                self.assertInvalidInput(support.run(OP, **dict(small, **{name: matrix})))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The evidence is always () and the canonical value unique, so a true claim the producer did not make is one
        built by hand, here backwards from its value, or one whose arguments write the same numbers another way: an
        int as a Rational of denominator 1, or sqrt(d) with a square factor taken into b. The checker accepts each;
        the producer, asked, answers the same record."""
        rng = random.Random(20261201)
        for _ in range(60):
            y = Fraction(rng.randint(1, 30), rng.randint(1, 30))
            w = Fraction(rng.randint(1, 20), rng.randint(1, 20))
            (a, b, d), value = backward(y, w, rng.choice(NON_SQUARES), rng.choice((-1, 1)))
            claim = record(*value)
            with self.subTest(a=a, b=b, d=d):
                self.assertAccepts(OP, args_of(a, b, d), claim)
                self.assertEqual(family.produce(OP, args_of(a, b, d)), (claim, ()))
        for (a, b, d), value in KNOWN + SPECIAL:
            if Fraction(a).denominator == Fraction(b).denominator == 1:
                with self.subTest(a=a, b=b, d=d, rationals=True):
                    forms = ({"a": Q(int(a)), "b": int(b), "d": d}, {"a": int(a), "b": Q(int(b)), "d": d})
                    for args in forms:
                        self.assertAccepts(OP, args, record(*value))
                        self.assertEqual(family.produce(OP, args), (record(*value), ()))
        # b*sqrt(d) = 2*sqrt(2) = sqrt(8) = (2/3)*sqrt(18) = (1/5)*sqrt(200): one number, one value, four inputs.
        for b, d in ((2, 2), (1, 8), (Fraction(2, 3), 18), (Fraction(1, 5), 200)):
            with self.subTest(b=b, d=d):
                self.assertAccepts(OP, args_of(3, b, d), record(2, 1, 1))
                self.assertEqual(family.produce(OP, args_of(3, b, d)), (record(2, 1, 1), ()))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced, while its trace stays verified."""
        for args, x, y, sign in ((args_of(3, 2, 2), Q(2), Q(1), 1), (args_of(7, -4, 3), Q(4), Q(3), -1),
                                 (args_of(2, 1, 3), Q(3, 2), Q(1, 2), 1),
                                 (args_of(Fraction(5, 2), -1, 6), Q(3, 2), Q(1), -1)):
            result = support.run(OP, **args)
            value = record(x, y, sign)
            self.assertVerified(result, value)
            self.assertAccepts(OP, args, value)
            wrong = [record(x, y, -sign), record(x + 1, y, sign), record(x, y / 2, sign), record(x + y, y / 3, sign)]
            cases = [(f"wrong {item}", item, ()) for item in wrong]
            for name, field in (("x", x), ("y", y)):
                if field.denominator == 1:  # The right number as an int, which the canonical form refuses.
                    fields = {"x": x, "y": y, "sign": sign, name: field.numerator}
                    cases.append((f"{name} an int", support.forged(DenestedRoot, **fields), ()))
            cases += [
                ("x and y swapped", support.forged(DenestedRoot, x=y, y=x, sign=sign), ()),
                ("x a float", support.forged(DenestedRoot, x=x.numerator / x.denominator, y=y, sign=sign), ()),
                ("y a str", support.forged(DenestedRoot, x=x, y=str(y), sign=sign), ()),
                ("x a Rational subclass", support.forged(DenestedRoot, x=SubRational(x.numerator, x.denominator),
                                                         y=y, sign=sign), ()),
                ("x outside lowest terms", support.forged(DenestedRoot, x=forged_rational(2 * x.numerator,
                                                                                          2 * x.denominator),
                                                          y=y, sign=sign), ()),
                ("the sign a bool", support.forged(DenestedRoot, x=x, y=y, sign=sign == 1), ()),
                ("the sign a float", support.forged(DenestedRoot, x=x, y=y, sign=float(sign)), ()),
                ("the sign an int subclass", support.forged(DenestedRoot, x=x, y=y, sign=Integer(sign)), ()),
                ("the sign missing", support.forged(DenestedRoot, x=x, y=y), ()),
                ("an impostor", support.impostor(value), ()),
                ("a tuple of the fields", (x, y, sign), ()),
                ("an expression", mf.sqrt(x) + sign * mf.sqrt(y), ()),
                ("a giant x", support.forged(DenestedRoot, x=Q(GIANT), y=y, sign=sign), ()),
                ("a giant y denominator", support.forged(DenestedRoot, x=x, y=Q(1, GIANT), sign=sign), ()),
                ("evidence as a list", value, []),
                ("evidence None", value, None),
                ("an extra evidence item", value, (Q(1),)),
                ("evidence (c,)", value, (x - y,)),
                ("oversized evidence", value, (0,) * 100_000),
            ]
            for label, item, evidence in cases:
                with self.subTest(args=args, case=label):
                    self.assertRejects(OP, args, item, evidence)
            for operation in ("denest", "continued_fraction", "DENEST_SQRT", "", None):
                with self.subTest(args=args, operation=operation):
                    self.assertRejects(operation, args, value)
            report = support.statuses(mf.verify(support.tampered(result, wrong[0])))
            self.assertIs(report["result.trace"], VERIFIED)
            self.assertIs(report["result.mathematics"], FAILED)
            report = support.statuses(mf.verify(support.tampered(result, value, (Q(1),))))
            self.assertIs(report["result.mathematics"], FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text; an
        Unevaluated Result keeps its report as well."""
        requests = [args_of(a, b, d) for (a, b, d), _ in KNOWN + SPECIAL]
        (a, b, d), _ = backward(Fraction(1, 3 ** 4_000), Fraction(5 ** 2_000, 7 ** 1_500), 6, -1)
        requests.append(args_of(a, b, d))
        results = [support.run(OP, **args) for args in requests]
        texts = [mf.dumps(result) for result in results]
        unevaluated = support.run(OP, **args_of(*CARD_UNEVALUATED))
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for index, (result, text) in enumerate(zip(results, texts)):
                with self.subTest(request=index):  # The last request's integers are too long for a label.
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            self.assertUnevaluated(unevaluated)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(OP, **args_of(3, 2, 2))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: the legacy expression
        sqrt(x) + sign*sqrt(y), x first, a rational square root shown as a rational. An Unevaluated Result renders its
        record."""
        rendered = (
            ((3, 2, 2), "sqrt(2) + 1", r"\sqrt{2} + 1"),
            ((5, 2, 6), "sqrt(3) + sqrt(2)", r"\sqrt{3} + \sqrt{2}"),
            ((7, -4, 3), "2 - sqrt(3)", r"2 - \sqrt{3}"),
            ((2, 1, 3), "sqrt(3/2) + sqrt(1/2)", r"\sqrt{\frac{3}{2}} + \sqrt{\frac{1}{2}}"),
            ((3, -2, 2), "sqrt(2) - 1", r"\sqrt{2} - 1"),
            ((Fraction(5, 2), -1, 6), "sqrt(3/2) - 1", r"\sqrt{\frac{3}{2}} - 1"),
            ((19, -6, 2), "sqrt(18) - 1", r"\sqrt{18} - 1"),
            ((Fraction(11, 4), Fraction(-3, 2), 2), "3/2 - sqrt(1/2)", r"\frac{3}{2} - \sqrt{\frac{1}{2}}"),
        )
        for (a, b, d), text, latex in rendered:
            with self.subTest(text=text):
                result = support.run(OP, **args_of(a, b, d))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertIs(type(decoded.value), DenestedRoot)
                self.assertEqual((mf.render_text(result.value), mf.to_latex(result.value)), (text, latex))
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run(OP, **args_of(7, -4, 3)).value)["data"],
                         {"type": "v6_denested_root", "x": {"type": "rational", "numerator": "4", "denominator": "1"},
                          "y": {"type": "rational", "numerator": "3", "denominator": "1"},
                          "sign": {"type": "integer", "value": "-1"}})
        unevaluated = support.run(OP, **args_of(*CARD_UNEVALUATED))
        self.assertEqual(mf.loads(mf.dumps(unevaluated)), unevaluated)
        self.assertEqual(mf.render_text(unevaluated),
                         "completed / no_conclusion [not_applicable; unknown]: Unevaluated(simplify, denest_sqrt, "
                         f"Reason(no_applicable_rule, {NO_DENESTING.detail}))")
        self.assertIn(r"\text{Unevaluated}", mf.to_latex(unevaluated))

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path. Without a
        denesting the lean path falls back, and the value function raises the structured path's error."""
        rng = random.Random(20261202)
        cases = [args_of(a, b, d) for (a, b, d), _ in KNOWN + SPECIAL]
        for _ in range(20):
            y = Fraction(rng.randint(1, 10 ** 6), rng.randint(1, 10 ** 6))
            w = Fraction(rng.randint(1, 10 ** 4), rng.randint(1, 10 ** 4))
            (a, b, d), _ = backward(y, w, rng.choice(NON_SQUARES), rng.choice((-1, 1)))
            cases.append(args_of(a, b, d))
        for args in cases:
            with self.subTest(args=args):
                expected = support.run(OP, **args).value
                self.assertIs(type(expected), DenestedRoot)
                found = support.lean_value(OP, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(OP, **args), expected)
        for a, b, d in UNEVALUATED[:5]:
            with self.subTest(a=a, b=b, d=d, unevaluated=True):
                self.assertIs(support.lean_value(OP, **args_of(a, b, d)), lean.FALLBACK)
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(OP, **args_of(a, b, d))
                self.assertEqual(raised.exception.code, "operation_failed")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: rationals travel as
        JSON integers or as strings "p/q", integers from 2**53 on as strings of digits."""
        for request, args, value in (
                ({"a": 3, "b": 2, "d": 2}, args_of(3, 2, 2), record(2, 1, 1)),
                ({"a": "7", "b": "-4", "d": "3"}, args_of(7, -4, 3), record(4, 3, -1)),
                ({"a": "5/2", "b": -1, "d": 6}, args_of(Fraction(5, 2), -1, 6), record(Fraction(3, 2), 1, -1)),
                ({"a": "11/4", "b": "-3/2", "d": 2}, args_of(Fraction(11, 4), Fraction(-3, 2), 2),
                 record(Fraction(9, 4), Fraction(1, 2), -1)),
                ({"a": "3/1", "b": 2, "d": 2}, {"a": Q(3), "b": 2, "d": 2}, record(2, 1, 1))):
            with self.subTest(request=request):
                result = self.answer(support.call(OP, request))
                self.assertEqual(result, support.run(OP, **args))
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)
        d = 2 ** 200 + 3
        (a, b, dd), large = backward(Fraction(3 ** 90, 7 ** 40), Fraction(5, 11 ** 30), 3, -1)
        for args, value in ((args_of(d + 1, 2, d), record(d, 1, 1)), (args_of(a, b, dd), record(*large))):
            with self.subTest(large=True):
                result = self.answer(support.call_values(OP, **args))
                self.assertEqual(result, support.run(OP, **args))
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call(OP, {"a": 1, "b": 1, "d": 2}))
        self.assertEqual(result, support.run(OP, **args_of(*CARD_UNEVALUATED)))
        self.assertUnevaluated(result)
        for request, text in (({"a": -3, "b": 2, "d": 2}, "a + b*sqrt(d) >= 0"), ({"a": 4, "b": 0, "d": 2}, "b != 0"),
                              ({"a": 5, "b": 2, "d": 4}, "no perfect square")):
            with self.subTest(request=request):
                error = self.assertToolError(support.call(OP, request), "invalid_input")
                self.assertIn(text, error["message"])
        self.assertToolError(support.call(OP, {"a": 0.5, "b": 2, "d": 2}), "invalid_input")

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references on every small integer input (brute force, the domain by Decimal
        arithmetic), on 400 seeded rational claims built backwards, on 400 small ones that brute force confirms as
        well, on 1500 seeded small rational inputs (brute force) and on long claims built backwards. The checker
        accepts every value; at least 200 values come from each of the integer grid, the backward claims and brute
        force on rational inputs."""
        counts: Counter[str] = Counter()
        # Every integer input with -4 <= a <= 30, 1 <= |b| <= 10 and the first 20 non-squares d.
        for a in range(-4, 31):
            for b in range(-10, 11):
                for d in (NON_SQUARES[:20] if b else ()):
                    args = args_of(a, b, d)
                    if decimal_sign(a, b, d) < 0:
                        self.assertFalse(in_domain(a, b, d), args)
                        with self.assertRaises(InvalidInput, msg=args):
                            family.produce(OP, args)
                        counts["invalid"] += 1
                        continue
                    self.assertTrue(in_domain(a, b, d), args)
                    expected = brute(a, b, d)
                    self.assertEqual(closed_form(a, b, d), expected, args)
                    if expected is None:
                        self.assertEqual(family.produce(OP, args),
                                         (Unevaluated("simplify", request_of(args), NO_DENESTING), ()), args)
                        counts["unevaluated"] += 1
                        continue
                    self.assertProduces(args, record(*expected))
                    counts["integer"] += 1
                    counts[f"sign {expected[2]}"] += 1
        # Rational claims built backwards: x or y may be the square of a rational, and both signs occur.
        rng = random.Random(20261203)
        for _ in range(400):
            y = Fraction(rng.randint(1, 60), rng.randint(1, 60))
            if rng.random() < 0.3:
                y *= y
            w = Fraction(rng.randint(1, 40), rng.randint(1, 40))
            (a, b, d), value = backward(y, w, rng.choice(NON_SQUARES), rng.choice((-1, 1)))
            self.assertEqual(closed_form(a, b, d), value)
            self.assertProduces(args_of(a, b, d), record(*value))
            if counts["backward"] % 20 == 0:
                self.assertTrue(approximates(a, b, d, *value))
            counts["backward"] += 1
        # Small rational claims built backwards, which brute force confirms independently.
        for _ in range(400):
            y, w = Fraction(rng.randint(1, 8), rng.randint(1, 4)), Fraction(rng.randint(1, 4), rng.randint(1, 3))
            (a, b, d), value = backward(y, w, rng.choice(NON_SQUARES[:6]), rng.choice((-1, 1)))
            if a.numerator * b.denominator <= 3000:  # The length of brute()'s search.
                self.assertEqual(brute(a, b, d), value)
                self.assertProduces(args_of(a, b, d), record(*value))
                counts["rational brute"] += 1
        # Small rational inputs at random, decided by brute force.
        for _ in range(1500):
            a = Fraction(rng.randint(-10, 40), rng.randint(1, 6))
            b = Fraction(rng.choice((-1, 1)) * rng.randint(1, 12), rng.randint(1, 4))
            d = rng.choice(NON_SQUARES[:12])
            args = args_of(a, b, d)
            if not in_domain(a, b, d):
                self.assertEqual(decimal_sign(a, b, d), -1, args)
                with self.assertRaises(InvalidInput, msg=args):
                    family.produce(OP, args)
                continue
            expected = brute(a, b, d)
            if expected is None:
                self.assertIs(type(family.produce(OP, args)[0]), Unevaluated, args)
                counts["rational unevaluated"] += 1
            else:
                self.assertProduces(args, record(*expected))
                counts["rational brute"] += 1
        # Long claims built backwards, of hundreds to thousands of digits.
        for _ in range(30):
            y = Fraction(rng.getrandbits(rng.randrange(1, 3000)) | 1, rng.getrandbits(rng.randrange(1, 3000)) | 1)
            w = Fraction(rng.getrandbits(rng.randrange(1, 2000)) | 1, rng.getrandbits(rng.randrange(1, 2000)) | 1)
            d = rng.getrandbits(rng.randrange(2, 400)) | 2
            if isqrt(d) * isqrt(d) == d:
                continue
            (a, b, d), value = backward(y, w, d, rng.choice((-1, 1)))
            self.assertEqual(closed_form(a, b, d), value)
            self.assertProduces(args_of(a, b, d), record(*value))
            counts["long"] += 1
        # The structured path agrees on a sample, each Result verified.
        for (a, b, d), value in (((11, 6, 2), (9, 2, 1)), ((Fraction(7, 3), Fraction(-2, 3), 10),
                                                            (Fraction(5, 3), Fraction(2, 3), -1))):
            self.assertVerified(support.run(OP, **args_of(a, b, d)), record(*value))
        self.assertGreaterEqual(counts["integer"], 200, counts)
        self.assertEqual(counts["backward"], 400)
        self.assertGreaterEqual(counts["rational brute"], 200, counts)
        self.assertGreater(counts["rational unevaluated"], 500, counts)
        self.assertGreater(counts["unevaluated"], 1000, counts)
        self.assertGreater(counts["invalid"], 1000, counts)
        self.assertGreater(counts["long"], 20, counts)
        self.assertGreater(min(counts["sign 1"], counts["sign -1"]), 100, counts)

    # -----------------------------------------------------------------------------------------------------------
    # Sizes and work

    def test_full_size_inputs_fit_the_default_budget(self):
        """Requests with integers near max_integer_bits complete under the default limits, each within FAST seconds:
        sqrt(d + 1 +- 2*sqrt(d)) = sqrt(d) +- 1 for d of 65,536 bits, a claim with long numerators and denominators
        everywhere, and long arguments without a denesting."""
        d = 2 ** 65_535 + 3
        self.assertNotEqual(isqrt(d) ** 2, d)
        self.assertEqual((d + 1).bit_length(), MAX_BITS)
        cases = [(args_of(d + 1, 2 * sign, d), record(d, 1, sign)) for sign in (-1, 1)]
        (a, b, dd), value = backward(Fraction(1, 3 ** 12_000), Fraction(5 ** 6_000, 7 ** 5_000), 6, -1)
        self.assertGreater(Fraction(a).denominator.bit_length(), 40_000)
        cases.append((args_of(a, b, dd), record(*value)))
        for args, value in cases:
            with self.subTest(bits=max(fraction(item).numerator.bit_length() for item in args.values())):
                started = time.perf_counter()
                self.assertVerified(support.run(OP, **args), value)
                self.assertLess(time.perf_counter() - started, FAST)
        rng = random.Random(20261204)
        big_a, a_denominator = rng.getrandbits(MAX_BITS) | 1, rng.getrandbits(MAX_BITS) | 1
        big_b, b_denominator = rng.getrandbits(32_000) | 1, rng.getrandbits(MAX_BITS) | 1
        args = {"a": Q(big_a, a_denominator), "b": Q(big_b, b_denominator), "d": 3}
        started = time.perf_counter()
        result = support.run(OP, **args)
        self.assertLess(time.perf_counter() - started, FAST)
        self.assertIsNone(closed_form(Fraction(big_a, a_denominator), Fraction(big_b, b_denominator), 3))
        self.assertUnevaluated(result)

    def test_numerators_are_reduced_before_the_rationals_are_built(self):
        """For a = x + y with x = 8*k^2 and y = 1/4, b = 2*k/D and d = 2*D^2, b*sqrt(d) = 2*k*sqrt(2) and the value is
        (8*k^2, 1/4, 1), about as long as a. The card's unreduced numerator A*Db + m, with m = (x - y)*Da*Db, is far
        longer than max_integer_bits allows; the producer reduces it before it builds the Rational, so the request
        completes instead of exceeding the budget."""
        rng = random.Random(20261205)
        k, big_d = rng.getrandbits(32_000) | 1, rng.getrandbits(32_000) | 1 << 31_999 | 1
        while gcd(2 * k, big_d) != 1:
            big_d += 2
        x, y = Fraction(8 * k * k), Fraction(1, 4)
        a, b, d = x + y, Fraction(2 * k, big_d), 2 * big_d * big_d
        self.assertEqual((4 * x * y, x + y), (b * b * d, a))
        self.assertLessEqual(max(a.numerator.bit_length(), d.bit_length(), b.denominator.bit_length()), MAX_BITS)
        unreduced = a.numerator * b.denominator + (x - y) * a.denominator * b.denominator
        self.assertGreater(unreduced.numerator.bit_length(), MAX_BITS + 20_000)
        self.assertEqual(closed_form(a, b, d), (x, y, 1))
        started = time.perf_counter()
        self.assertVerified(support.run(OP, **args_of(a, b, d)), record(x, y, 1))
        self.assertLess(time.perf_counter() - started, FAST)

    def test_values_at_the_checkers_size_bound(self):
        """Values whose integers are as long as the checker's size filter must admit, bitlen(A) + bitlen(Da) +
        bitlen(Db) bits (at_the_size_bound), small and near max_integer_bits, both signs: produce() answers each one
        and the Result verifies, so the filter refuses no true value at its edge."""
        for j, k, sign in ((3, 40, 1), (4, 41, -1), (5, 21_000, -1), (3, 20_000, 1)):
            for (a, b, d), value in at_the_size_bound(j, k, sign):
                x, y, _ = value
                longest = max(item.bit_length() for item in (x.numerator, x.denominator, y.numerator, y.denominator))
                with self.subTest(j=j, k=k, sign=sign, a_denominator=a.denominator.bit_length()):
                    self.assertEqual(longest, sum(item.bit_length() for item in (a.numerator, a.denominator,
                                                                                 b.denominator)))
                    self.assertTrue(in_domain(a, b, d))
                    self.assertEqual(closed_form(a, b, d), value)
                    if k < 100 and sign == 1:  # For sign -1 and a = 1/Da, a + b*sqrt(d) cancels past 80 digits.
                        self.assertTrue(approximates(a, b, d, *value))
                    started = time.perf_counter()
                    self.assertVerified(support.run(OP, **args_of(a, b, d)), record(*value))
                    self.assertLess(time.perf_counter() - started, FAST)

    def test_values_beyond_max_integer_bits(self):
        """A true value whose own integers exceed max_integer_bits is beyond the budget, resource_limit_exceeded, while
        the arguments fit: y = 1/(z*Q) of about 68,000 bits for arguments of 34,000 bits (beyond()). Under a wider
        max_integer_bits the same request completes and verifies with those limits."""
        (a, b, d), (x, y) = beyond(17_000)
        self.assertLessEqual(max(a.numerator.bit_length(), a.denominator.bit_length(),
                                 b.denominator.bit_length()), MAX_BITS)
        self.assertGreater(y.denominator.bit_length(), MAX_BITS)
        self.assertEqual(closed_form(a, b, d), (x, y, 1))
        started = time.perf_counter()
        self.assertResourceLimit(support.run(OP, **args_of(a, b, d)))
        self.assertLess(time.perf_counter() - started, FAST)
        wide = ComputationLimits(max_integer_bits=80_000)
        result = support.run(OP, limits=wide, **args_of(a, b, d))
        with computation(wide, fresh=True):  # The record checks its integers in the active budget.
            expected = record(x, y, 1)
        self.assertEqual(result.value, expected, result.error)
        self.assertTrue(mf.verify(result, limits=wide).verified)
        self.assertIs(support.statuses(mf.verify(result))["result.resources"], UNKNOWN)  # Default limits: too long.
        # The same construction at a size that fits: a value and the checker's acceptance.
        (a, b, d), (x, y) = beyond(400)
        self.assertVerified(support.run(OP, **args_of(a, b, d)), record(x, y, 1))

    def test_work_is_charged_before_it_runs(self):
        """Every charge depends on the input alone: a budget one unit short of a request's work refuses it with
        ResourceLimitError, its exact work completes, and long integers cost more than short ones, within the
        default max_work. The record's own checks are charged to the same budget."""
        (a, b, d), value = backward(Fraction(1, 3 ** 12_000), Fraction(5 ** 6_000, 7 ** 5_000), 6, 1)
        args = args_of(a, b, d)
        with computation(fresh=True) as budget:
            produced = family.produce(OP, dict(args))
        work = budget.work
        self.assertEqual(produced, (record(*value), ()))
        with computation(ComputationLimits(max_work=work), fresh=True):
            again = family.produce(OP, dict(args))
        self.assertEqual(again, produced)
        with computation(ComputationLimits(max_work=work - 1), fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OP, dict(args))
        self.assertResourceLimit(support.run(OP, limits=ComputationLimits(max_work=work - 1), **args))
        self.assertVerified(support.run(OP, limits=ComputationLimits(max_work=work), **args), record(*value))
        with computation(fresh=True) as small:
            family.produce(OP, args_of(3, 2, 2))
        self.assertGreaterEqual(small.work, 20)  # One unit per step at least: 27 units at the time of writing.
        self.assertLess(small.work, 100)
        self.assertGreater(work, 100_000)  # 117,946 at the time of writing.
        d = 2 ** 65_535 + 3
        with computation(fresh=True) as full:
            family.produce(OP, args_of(d + 1, 2, d))
        # 73,739 at the time of writing, 16,369 of them for the square root of M = (d - 1)^2 of 131,071 bits.
        self.assertGreater(full.work, 60_000)
        self.assertLess(max(work, full.work), ComputationLimits().max_work // 4)
        # The costliest kind of request tried: x and y with numerators and denominators of about 64,000 bits. Its work
        # stays within the bound of the module docstring for any input under the default limits, 421,917 units.
        rng = random.Random(20261206)
        y = Fraction(rng.getrandbits(60_000) | 1, rng.getrandbits(60_000) | 1)
        (a, b, d), value = backward(y, Fraction(rng.getrandbits(2_000) | 1, rng.getrandbits(2_000) | 1), 3, 1)
        args, expected = args_of(a, b, d), record(*value)  # Built before the budget opens, which holds produce() only.
        with computation(fresh=True) as costly:
            produced = family.produce(OP, args)
        self.assertEqual(produced, (expected, ()))
        self.assertGreater(costly.work, 300_000)  # 336,400 at the time of writing.
        self.assertLessEqual(costly.work, 421_917)
