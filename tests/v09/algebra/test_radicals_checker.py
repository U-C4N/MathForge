"""Tests of the independent checker of mathforge.areas.algebra.radicals (mathforge.certify.algebra.radicals) alone,
first written before the family's producer existed: every claim is built here, never by the producer, and the checker
answers it as the verifier asks it (support.verdict, in a fresh budget of the default limits unless a test passes
others).

The true claims come from references of this module, never from a producer: the card's closed form with Fractions
(x, y = (a +- c)/2 for c = sqrt(a^2 - b^2*d) when that is rational), checked against brute force over every candidate
x = i/2 for integer arguments, and claims built backwards (y and a ratio w chosen first, then x = w^2*d*y, a = x + y and
b = +-2*w*y); Decimal arithmetic to 80 digits confirms that the claims and the forged values are what they say. The
evidence is always (), the layout of the card.

The checker answers False to every corrupted claim, never with an exception; only an argument longer than
max_integer_bits raises ResourceLimitError (run() and the verifier refuse such an argument before any checker runs),
as does a budget too small for an honest check. It refuses an oversized forged value or evidence within FAST seconds
under the default budget.
"""

from __future__ import annotations

import ast
import dataclasses
from decimal import Decimal, localcontext
import enum
from fractions import Fraction
import inspect
from math import isqrt
import random
import time
from types import MappingProxyType

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.algebra import radicals as family
from mathforge.areas.algebra.radicals import DenestedRoot
from mathforge.certify.algebra import radicals as checker
from mathforge.contracts import CheckStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.registry import Argument, OperationSpec

from .. import support

Q = mf.Rational
OP = "denest_sqrt"
FAST = 3.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
         "and decimal digits")
NON_SQUARES = tuple(d for d in range(2, 60) if isqrt(d) * isqrt(d) != d)
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.

# The known values of the card: (a, b, d) with (x, y, sign), and the inputs without a value or outside the domain.
KNOWN = (
    ((3, 2, 2), (2, 1, 1)),  # sqrt(2) + 1
    ((5, 2, 6), (3, 2, 1)),  # sqrt(3) + sqrt(2)
    ((7, -4, 3), (4, 3, -1)),  # 2 - sqrt(3)
    ((2, 1, 3), (Fraction(3, 2), Fraction(1, 2), 1)),  # sqrt(3/2) + sqrt(1/2)
)
UNEVALUATED = ((1, 1, 2),)  # a^2 - b^2*d = -1: no denesting, so no claim.
INVALID = ((-3, 2, 2), (1, -1, 2))  # a + b*sqrt(d) < 0.


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def exact(value: object) -> object:
    """An argument as a request holds it: an int for an integer, a Rational otherwise."""
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else Q(value.numerator, value.denominator)


def rational(value: object) -> Q:
    """A field of the record: always a Rational, an integer one included."""
    value = Fraction(value)
    return Q(value.numerator, value.denominator)


def args_of(a: object, b: object, d: object) -> dict[str, object]:
    return {"a": exact(a), "b": exact(b), "d": d}


def record(x: object, y: object, sign: int) -> DenestedRoot:
    return DenestedRoot(rational(x), rational(y), sign)


def sign_of(a: Fraction, b: Fraction, d: int) -> int:
    """The exact sign of a + b*sqrt(d) for a d that is no square (a = b*sqrt(d) is impossible for nonzero a, b)."""
    if a >= 0 and b >= 0:
        return 0 if a == b == 0 else 1
    if a <= 0 and b <= 0:
        return -1
    return (1 if a > 0 else -1) if a * a > b * b * d else (1 if b > 0 else -1)


def valid(a: object, b: object, d: int) -> bool:
    """Whether (a, b, d) lies in the card's domain: b != 0, d >= 2 no perfect square, a + b*sqrt(d) >= 0."""
    return b != 0 and d >= 2 and isqrt(d) * isqrt(d) != d and sign_of(Fraction(a), Fraction(b), d) >= 0


def reference(a: object, b: object, d: int) -> tuple[Fraction, Fraction, int] | None:
    """The card's closed form: (x, y, sign) = ((a + c)/2, (a - c)/2, sign(b)) when a > 0 and a^2 - b^2*d = c^2 for a
    rational c > 0; None when there is no denesting of the form sqrt(x) +- sqrt(y)."""
    a, b = Fraction(a), Fraction(b)
    n = a * a - b * b * d
    if a <= 0 or n <= 0:
        return None
    p, q = isqrt(n.numerator), isqrt(n.denominator)
    if p * p != n.numerator or q * q != n.denominator:
        return None
    c = Fraction(p, q)
    return (a + c) / 2, (a - c) / 2, 1 if b > 0 else -1


def brute(a: int, b: int, d: int) -> tuple[Fraction, Fraction, int] | None:
    """The denesting of integer a and b by brute force. x + y = a and 4*x*y = b^2*d make 2x and 2y the roots of
    t^2 - 2a*t + b^2*d, a monic integer polynomial, so they are integers: try every x = i/2 with a/2 < x < a and
    y = a - x, that is i*(2a - i) = b^2*d."""
    found = [(Fraction(i, 2), Fraction(2 * a - i, 2), 1 if b > 0 else -1) for i in range(a + 1, 2 * a)
             if i * (2 * a - i) == b * b * d]
    assert len(found) <= 1, found  # The roots of one quadratic.
    return found[0] if found else None


def backward(y: Fraction, w: Fraction, d: int, sign: int) -> tuple[tuple[Fraction, Fraction, int],
                                                                     tuple[Fraction, Fraction, int]]:
    """A claim built from its value, for y > 0, w > 0 and d no square: with x0 = w^2*d*y, sqrt(x0*y) = w*y*sqrt(d), so
    (sqrt(x0) + sign*sqrt(y))^2 = a + b*sqrt(d) with a = x0 + y and b = 2*sign*w*y. x0 = y would make d = 1/w^2 a
    square, and the canonical record orders x0 and y decreasingly (for sign -1, sqrt(a + b*sqrt(d)) is the positive
    difference)."""
    x0 = w * w * d * y
    return (x0 + y, 2 * sign * w * y, d), (max(x0, y), min(x0, y), sign)


def full_denominator(j: int, k: int, sign: int) -> tuple[tuple[Fraction, Fraction, int],
                                                          tuple[Fraction, Fraction, int]]:
    """A true claim whose x has the full denominator 2*Da*Db: B = 2^j - 1 and Da = 2^k - 1 (odd, and with leading
    ones, so that Da*Db loses no bit), n = B*Da, Db = n^2 + 1 and d = n^2 + 2 (strictly between two squares), a = 1/Da
    and b = sign*B/Db. Then x = d/(2*Da*Db) and y = B^2*Da/(2*Db): x + y = (2*n^2 + 2)/(2*Da*Db) = 1/Da and
    4*x*y = d*B^2/Db^2 = b^2*d, and d, odd, shares no factor with Da (d = 2 modulo Da) or Db (d = Db + 1)."""
    big_b, a_denominator = (1 << j) - 1, (1 << k) - 1
    n = big_b * a_denominator
    b_denominator, d = n * n + 1, n * n + 2
    return ((Fraction(1, a_denominator), Fraction(sign * big_b, b_denominator), d),
            (Fraction(d, 2 * a_denominator * b_denominator), Fraction(big_b * big_b * a_denominator, 2 * b_denominator),
             sign))


def long_numerator(k: int, j: int, sign: int) -> tuple[tuple[Fraction, Fraction, int],
                                                        tuple[Fraction, Fraction, int]]:
    """A true claim whose x has a numerator of almost 2*A*Db: A = 2^k - 2 (even) and Db = 2^j - 1, a = A,
    b = sign/Db and d = 2*A*Db - 1, which is 3 modulo 4 and so no square. Then x = (2*A*Db - 1)/(2*Db) and
    y = 1/(2*Db): x + y = A and 4*x*y = (2*A*Db - 1)/Db^2 = b^2*d."""
    big_a, b_denominator = (1 << k) - 2, (1 << j) - 1
    d = 2 * big_a * b_denominator - 1
    return (Fraction(big_a), Fraction(sign, b_denominator), d), (Fraction(d, 2 * b_denominator),
                                                                  Fraction(1, 2 * b_denominator), sign)


def number(value: object) -> Fraction:
    """An exact number as a Fraction: an int, a bool, a Fraction, or a Rational (also one made with object.__new__
    outside lowest terms, or of a subclass) by its numerator and denominator."""
    return Fraction(value.numerator, value.denominator)  # type: ignore[attr-defined]


def approximates(a: object, b: object, d: int, x: object, y: object, sign: object) -> bool:
    """sqrt(x) + sign*sqrt(y) agrees with sqrt(a + b*sqrt(d)) to 60 significant digits: Decimal arithmetic at 80, a
    numeric confirmation of a claim that is independent of every identity."""
    def decimal(value: object) -> Decimal:
        fraction = number(value)
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)

    with localcontext() as context:
        context.prec = 80
        right = (decimal(a) + decimal(b) * Decimal(d).sqrt()).sqrt()
        left = decimal(x).sqrt() + int(number(sign)) * decimal(y).sqrt()
        return abs(left - right) <= right * Decimal(10) ** -60


def claimed(args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, never by the producer."""
    request = contracts.OperationRequest(OP, tuple(args.items()), 6)
    step = contracts.Step("v6." + OP, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=OP, request=request, steps=(step,))


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values, arguments and evidence that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Sign(enum.IntEnum):
    """The signs as an int enumeration: equal to them but no exact ints."""
    MINUS = -1
    PLUS = 1


class SubRational(Q):
    """A Rational subclass: equal to the right number but no Rational of the model."""


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass, for an evidence tuple.
FakeRational = dataclasses.make_dataclass("Rational", ["numerator", "denominator"], frozen=True)  # Another module.


def forged_rational(numerator: object, denominator: object = 1, *, missing: str = "") -> Q:
    """A Rational made with object.__new__: its fields may break lowest terms, a positive denominator or exactness,
    and the field named by missing stays unset."""
    value = object.__new__(Q)
    for name, item in (("numerator", numerator), ("denominator", denominator)):
        if name != missing:
            object.__setattr__(value, name, item)
    return value


def label(item: object) -> str:
    """repr(item) for a subtest's label; a Rational made with object.__new__ without a field has no repr."""
    try:
        return repr(item)
    except AttributeError:
        return f"<{type(item).__name__} without a field>"


class RadicalsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, cases: list[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (name, args, value, evidence) case."""
        for name, args, value, evidence in cases:
            with self.subTest(case=name):
                self.assertRejects(OP, args, value, evidence)

    def assertRejectsFast(self, args: object, value: object, evidence: object = ()) -> None:
        started = time.perf_counter()
        self.assertRejects(OP, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_card(self):
        expected = OperationSpec(
            "denest_sqrt", "algebra",
            (Argument("a", "Rational | int", None, "rational"), Argument("b", "Rational | int", None, "rational"),
             Argument("d", "int", None, "int")),
            "DenestedRoot",
            "Denest sqrt(a + b*sqrt(d)) as sqrt(x) + sqrt(y) or sqrt(x) - sqrt(y) with rational x and y.",
            SCOPE, version=6)
        self.assertEqual(family.SPECS, (expected,))
        self.assertEqual(family.TYPES, {"denested_root": DenestedRoot})
        self.assertEqual(family.CHECKER, "mathforge.certify.algebra.radicals")
        self.assertEqual(set(family.RENDER), {DenestedRoot})
        self.assertEqual([field.name for field in dataclasses.fields(DenestedRoot)], ["x", "y", "sign"])
        self.assertEqual(registry.spec(OP), expected)
        self.assertIs(registry.checker(OP), checker)
        self.assertIs(support.checker(family), checker)
        self.assertEqual(checker.RECORD, (DenestedRoot.__module__, DenestedRoot.__qualname__))
        # The producer answers the card's first known value; tests/v09/algebra/test_radicals.py tests it in full.
        self.assertEqual(family.produce(OP, args_of(3, 2, 2)), (record(2, 1, 1), ()))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_accepts_the_known_values_of_the_card(self):
        for (a, b, d), (x, y, sign) in KNOWN:
            with self.subTest(a=a, b=b, d=d):
                self.assertTrue(valid(a, b, d))
                self.assertEqual(reference(a, b, d), (x, y, sign))
                self.assertEqual(brute(a, b, d), (x, y, sign))
                self.assertTrue(approximates(a, b, d, x, y, sign))
                self.assertAccepts(OP, args_of(a, b, d), record(x, y, sign))
                # A Rational argument of denominator 1 is the same exact number as the int.
                self.assertAccepts(OP, {"a": Q(a), "b": Q(b), "d": d}, record(x, y, sign))
                self.assertAccepts(OP, {"a": a, "b": Q(b), "d": d}, record(x, y, sign))
        for a, b, d in UNEVALUATED:
            self.assertTrue(valid(a, b, d))
            self.assertIsNone(reference(a, b, d))
            self.assertIsNone(brute(a, b, d))
        for a, b, d in INVALID:
            self.assertFalse(valid(a, b, d))

    def test_accepts_every_small_integer_claim(self):
        """Every valid (a, b, d) with 1 <= a <= 24, 1 <= |b| <= 10 and the first 15 non-squares d: the closed form
        agrees with the brute force, and each denesting it finds is accepted. Two neighbours of it are refused: the
        other sign, and the record (a - 1/3, 1/3), which meets x + y == a, so that only 4*x*y == b^2*d refuses it."""
        accepted = without = 0
        signs = set()
        for a in range(1, 25):
            for b in range(-10, 11):
                for d in NON_SQUARES[:15]:
                    if not b or not valid(a, b, d):
                        continue
                    found = reference(a, b, d)
                    self.assertEqual(found, brute(a, b, d), (a, b, d))
                    args = args_of(a, b, d)
                    third = (Fraction(a) - Fraction(1, 3), Fraction(1, 3), 1 if b > 0 else -1)
                    if found is None:
                        without += 1
                        self.assertRejects(OP, args, record(*third))
                        continue
                    x, y, sign = found
                    with self.subTest(a=a, b=b, d=d):
                        self.assertTrue(approximates(a, b, d, x, y, sign))
                        self.assertAccepts(OP, args, record(x, y, sign))
                        self.assertRejects(OP, args, record(x, y, -sign))
                        if third != found:
                            self.assertRejects(OP, args, record(*third))
                    accepted += 1
                    signs.add(sign)
        self.assertGreater(accepted, 100)
        self.assertGreater(without, 1000)
        self.assertEqual(signs, {-1, 1})

    def test_accepts_rational_claims_built_backwards(self):
        """400 seeded claims built from their values: rational a and b, x and y with denominators, both signs, and x
        or y the square of a rational. The closed form agrees, and the true value of a + 1 instead of a is refused."""
        rng = random.Random(20261112)
        squares = 0
        for _ in range(400):
            y = Fraction(rng.randint(1, 60), rng.randint(1, 60))
            if rng.random() < 0.3:
                y *= y  # sqrt(y) rational: the record shows a rational term.
            w = Fraction(rng.randint(1, 40), rng.randint(1, 40))
            d, sign = rng.choice(NON_SQUARES), rng.choice((-1, 1))
            (a, b, d), (x, y, sign) = backward(y, w, d, sign)
            with self.subTest(a=a, b=b, d=d):
                self.assertTrue(valid(a, b, d))
                self.assertEqual(reference(a, b, d), (x, y, sign))
                self.assertTrue(approximates(a, b, d, x, y, sign))
                self.assertAccepts(OP, args_of(a, b, d), record(x, y, sign))
                self.assertRejects(OP, args_of(a + 1, b, d), record(x, y, sign))
            squares += any(isqrt(v.numerator) ** 2 == v.numerator and isqrt(v.denominator) ** 2 == v.denominator
                           for v in (x, y))
        self.assertGreater(squares, 50)

    def test_accepts_the_special_cases(self):
        """Both signs with y a square, x a square or neither; x and y with denominators; rational a and b, with a
        rational c = sqrt(a^2 - b^2*d) that is no integer; d with square factors; and large numbers."""
        cases = (
            ((3, -2, 2), (2, 1, -1)),  # sqrt(2) - 1
            ((7, 4, 3), (4, 3, 1)),  # 2 + sqrt(3)
            ((5, -2, 6), (3, 2, -1)),  # sqrt(3) - sqrt(2)
            ((2, -1, 3), (Fraction(3, 2), Fraction(1, 2), -1)),
            ((Fraction(3, 2), Fraction(1, 2), 5), (Fraction(5, 4), Fraction(1, 4), 1)),  # c = 1
            ((Fraction(5, 2), -1, 6), (Fraction(3, 2), 1, -1)),  # c = 1/2: sqrt(3/2) - 1
            ((Fraction(7, 3), Fraction(-2, 3), 10), (Fraction(5, 3), Fraction(2, 3), -1)),  # c = 1
            ((3, 1, 8), (2, 1, 1)),  # sqrt(8) = 2*sqrt(2)
            ((4, 1, 12), (3, 1, 1)),  # sqrt(12) = 2*sqrt(3)
            ((11, 6, 2), (9, 2, 1)),  # 3 + sqrt(2)
            ((19, -6, 2), (18, 1, -1)),  # 3*sqrt(2) - 1
            ((10 ** 30 + 1, 2 * 10 ** 15, 10 ** 30), None),  # d a perfect square: invalid input, below.
        )
        for (a, b, d), expected in cases:
            with self.subTest(a=a, b=b, d=d):
                if expected is None:
                    self.assertFalse(valid(a, b, d))
                    self.assertRejects(OP, args_of(a, b, d), record(10 ** 30, 1, 1))
                    continue
                x, y, sign = (Fraction(item) for item in expected)
                self.assertEqual(reference(a, b, d), (x, y, sign))
                self.assertTrue(approximates(a, b, d, x, y, sign))
                self.assertAccepts(OP, args_of(a, b, d), record(x, y, int(sign)))
        # Large numbers, built backwards: integers of hundreds and thousands of digits.
        for y, w, d in ((Fraction(1), Fraction(10 ** 40 + 7), 2),
                        (Fraction(3 ** 200, 7 ** 100), Fraction(5, 11 ** 90), 3),
                        (Fraction(2 ** 3000 + 1, 3 ** 1500), Fraction(7 ** 900, 2 ** 1200 + 5), 1999),
                        (Fraction(1, 2 ** 20000 + 3), Fraction(3 ** 9000), 5)):
            for sign in (-1, 1):
                (a, b, d), (x, y_, sign) = backward(y, w, d, sign)
                with self.subTest(bits=Fraction(a).numerator.bit_length(), sign=sign):
                    self.assertEqual(reference(a, b, d), (x, y_, sign))
                    self.assertAccepts(OP, args_of(a, b, d), record(x, y_, sign))
                    self.assertRejects(OP, args_of(a, b, d), record(x + 1, y_, sign))

    def test_full_size_claims_fit_the_default_budget(self):
        """Arguments of nearly max_integer_bits: sqrt(d + 1 +- 2*sqrt(d)) = sqrt(d) +- 1 for a d of 65,536 bits, and a
        claim with long numerators and denominators everywhere, each verified within FAST seconds."""
        d = 2 ** 65_535 + 3
        self.assertNotEqual(isqrt(d) ** 2, d)
        self.assertEqual((d + 1).bit_length(), 65_536)
        for sign in (-1, 1):
            started = time.perf_counter()
            self.assertAccepts(OP, args_of(d + 1, 2 * sign, d), record(d, 1, sign))
            self.assertRejects(OP, args_of(d + 1, 2 * sign, d), record(d, 1, -sign))
            self.assertLess(time.perf_counter() - started, FAST)
        claim = backward(Fraction(1, 3 ** 12_000), Fraction(5 ** 6_000, 7 ** 5_000), 6, -1)
        (a, b, d), value = claim
        for number in (Fraction(a), Fraction(b), *(Fraction(item) for item in value[:2])):
            self.assertLessEqual(max(number.numerator.bit_length(), number.denominator.bit_length()), 65_536)
        self.assertGreater(Fraction(a).denominator.bit_length(), 40_000)
        self.assertEqual(reference(a, b, d), value)
        started = time.perf_counter()
        self.assertAccepts(OP, args_of(a, b, d), record(*value))
        self.assertRejects(OP, args_of(a, b, d), record(value[0], value[1], 1))
        self.assertLess(time.perf_counter() - started, FAST)

    def test_accepts_true_values_at_the_size_bound(self):
        """The size filter refuses a claimed integer longer than room = bitlen(A) + bitlen(Da) + bitlen(Db) + 2 bits.
        A true value has at most room - 2: its reduced denominators divide 2*Da*Db and its numerators are below
        2*A*Db (the checker's module docstring). These true claims reach exactly room - 2, small and near
        max_integer_bits, both signs: x with the full denominator 2*Da*Db (full_denominator), which a filter without
        the length of Da would refuse, and x with a numerator of almost 2*A*Db (long_numerator). A filter tighter by
        three bits refuses both."""
        claims = [full_denominator(3, 40, 1), full_denominator(5, 21_000, -1), long_numerator(40, 41, -1),
                  long_numerator(30_000, 30_000, 1)]
        for (a, b, d), value in claims:
            x, y, sign = value
            room = a.numerator.bit_length() + a.denominator.bit_length() + b.denominator.bit_length() + 2
            longest = max(item.bit_length() for item in (x.numerator, x.denominator, y.numerator, y.denominator))
            with self.subTest(room=room, sign=sign):
                self.assertEqual((x + y, 4 * x * y, x > y > 0), (a, b * b * d, True))
                self.assertTrue(valid(a, b, d))
                self.assertEqual(reference(a, b, d), value)
                if room < 200:
                    self.assertTrue(approximates(a, b, d, x, y, sign))
                self.assertEqual(longest, room - 2)
                self.assertLessEqual(max(longest, d.bit_length(), b.denominator.bit_length()), 65_536)
                if a.denominator > 1:  # The length of Da is needed: without it the room is too small.
                    self.assertGreater(longest, a.numerator.bit_length() + b.denominator.bit_length() + 2)
                started = time.perf_counter()
                self.assertAccepts(OP, args_of(a, b, d), record(x, y, sign))
                self.assertRejects(OP, args_of(a, b, d), record(x, y, -sign))
                self.assertLess(time.perf_counter() - started, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_rejects_wrong_values(self):
        args = args_of(3, 2, 2)  # sqrt(3 + 2*sqrt(2)) = sqrt(2) + 1
        self.assertAccepts(OP, args, record(2, 1, 1))
        for x, y, sign in ((2, 1, -1), (3, 1, 1), (2, Fraction(1, 2), 1), (Fraction(5, 2), Fraction(1, 2), 1),
                           (4, 2, 1), (1, Fraction(1, 2), 1), (Fraction(9, 4), Fraction(3, 4), 1), (3, 2, 1),
                           (8, 1, 1), (2, 1 + Fraction(1, 10 ** 30), 1)):
            with self.subTest(x=x, y=y, sign=sign):
                self.assertRejects(OP, args, record(x, y, sign))
        for (a, b, d), (x, y, sign) in KNOWN:  # A true value of one input is wrong for every other.
            for (other, _) in KNOWN:
                if other != (a, b, d):
                    self.assertRejects(OP, args_of(*other), record(x, y, sign))
            self.assertRejects(OP, args_of(a, -b, d), record(x, y, sign))
            self.assertRejects(OP, args_of(a + 1, b, d), record(x, y, sign))
            self.assertRejects(OP, args_of(a, b, d + 1 if d + 1 not in (4, 9) else d + 2), record(x, y, sign))
        # For sign -1 the swapped pair is wrong: sqrt(3) - 2 < 0.
        self.assertRejects(OP, args_of(7, -4, 3), support.forged(DenestedRoot, x=Q(3), y=Q(4), sign=-1))

    def test_rejects_claims_without_a_value(self):
        """Valid inputs that have no denesting, where the producer answers Unevaluated: a^2 - b^2*d < 0, a <= 0, or
        a^2 - b^2*d > 0 not a rational square. Candidates that meet x + y == a, 4*x*y == b^2*d or both are refused."""
        for a, b, d in (*UNEVALUATED, (-1, 1, 2), (0, 1, 2), (4, 1, 3), (Fraction(1, 2), Fraction(1, 3), 3), (6, 1, 2)):
            self.assertTrue(valid(a, b, d), (a, b, d))
            self.assertIsNone(reference(a, b, d), (a, b, d))
            for x, y in ((Fraction(3, 4), Fraction(1, 4)), (Fraction(2, 3), Fraction(1, 3)), (2, 1), (3, 1),
                         (Fraction(a) * 2, -Fraction(a)), (Fraction(a) - Fraction(1, 7), Fraction(1, 7))):
                if not 0 < y < x:
                    continue
                with self.subTest(a=a, b=b, d=d, x=x, y=y):
                    self.assertRejects(OP, args_of(a, b, d), record(x, y, 1 if b > 0 else -1))

    def test_rejects_claims_on_invalid_inputs(self):
        """No claim is true on input that the producer rejects. For d = 4, 9 or 1 the card's identities (1) to (4)
        alone accept a mathematically right equation such as sqrt(5 + 2*sqrt(4)) = sqrt(4) + sqrt(1); the checker
        refuses it because d must be >= 2 and no perfect square. b = 0 and a + b*sqrt(d) < 0 are refused as well."""
        square = (10 ** 40 + 1) ** 2  # Odd, past the exact range of a float square root.
        right_but_invalid = (((5, 2, 4), (4, 1, 1)), ((10, 2, 9), (9, 1, 1)), ((5, 4, 1), (4, 1, 1)),
                             ((5, -4, 1), (4, 1, -1)), ((13, 6, 4), (9, 4, 1)),
                             ((square + 1, 2, square), (square, 1, 1)), ((square + 1, -2, square), (square, 1, -1)))
        for (a, b, d), (x, y, sign) in right_but_invalid:
            with self.subTest(a=a, b=b, d=d):
                self.assertEqual((x + y, 4 * x * y, sign), (a, b * b * d, 1 if b > 0 else -1))  # (1), (2), (3) hold.
                self.assertTrue(approximates(a, b, d, x, y, sign))
                self.assertFalse(valid(a, b, d))
                self.assertRejects(OP, args_of(a, b, d), record(x, y, sign))
        cases = []
        for a, b, d in (*INVALID, (4, 0, 2), (3, 2, 0), (3, 2, -2), (3, 2, 1), (3, 2, 4), (-1, 1, 2 ** 64),
                        (3, 2, -(2 ** 64))):
            for value in (record(2, 1, 1), record(2, 1, -1), record(Fraction(3, 2), Fraction(1, 2), 1)):
                cases.append((f"{(a, b, d)} with {value}", args_of(a, b, d), value, ()))
        cases.append(("b = 0 with y = 0", args_of(4, 0, 2), support.forged(DenestedRoot, x=Q(4), y=Q(0), sign=1), ()))
        cases.append(("b = 0 with sign 0", args_of(4, 0, 2), support.forged(DenestedRoot, x=Q(3), y=Q(1), sign=0), ()))
        self.assertRejectsAll(cases)

    def test_rejects_noncanonical_values_that_are_mathematically_right(self):
        """Each forged value equals sqrt(a + b*sqrt(d)), checked numerically; only the canonical form that it breaks
        can refuse it: x and y swapped (for sign 1 the sum is the same number), an int for a Rational, a Rational
        outside lowest terms or with a negative denominator, a Fraction, a Rational subclass, a bool or an int
        enumeration for the sign."""
        args = args_of(3, 2, 2)
        forgeries = (
            ("x and y swapped", Q(1), Q(2), 1),
            ("x an int", 2, Q(1), 1),
            ("y an int", Q(2), 1, 1),
            ("both ints", 2, 1, 1),
            ("x outside lowest terms", forged_rational(4, 2), Q(1), 1),
            ("y outside lowest terms", Q(2), forged_rational(3, 3), 1),
            ("x with a negative denominator", forged_rational(-2, -1), Q(1), 1),
            ("x a Fraction", Fraction(2), Q(1), 1),
            ("y a Fraction", Q(2), Fraction(1), 1),
            ("x a Rational subclass", SubRational(2), Q(1), 1),
            ("the sign True", Q(2), Q(1), True),
            ("the sign an IntEnum", Q(2), Q(1), Sign.PLUS),
            ("the sign an int subclass", Q(2), Q(1), Integer(1)),
            ("the sign a Rational", Q(2), Q(1), Q(1)),
        )
        for case, x, y, sign in forgeries:
            with self.subTest(case=case):
                self.assertTrue(approximates(3, 2, 2, x, y, sign))
                self.assertRejects(OP, args, support.forged(DenestedRoot, x=x, y=y, sign=sign))
        self.assertAccepts(OP, args, support.forged(DenestedRoot, x=Q(2), y=Q(1), sign=1))  # The forge itself is fine.
        # sqrt(3/2) + sqrt(1/2) for (2, 1, 3), with a non-reduced 6/4.
        self.assertRejects(OP, args_of(2, 1, 3), support.forged(DenestedRoot, x=forged_rational(6, 4), y=Q(1, 2),
                                                                sign=1))
        self.assertRejects(OP, args_of(2, 1, 3), support.forged(DenestedRoot, x=Q(1, 2), y=Q(3, 2), sign=1))

    def test_rejects_impostors_and_forged_records(self):
        args, true = args_of(5, 2, 6), record(3, 2, 1)
        self.assertAccepts(OP, args, true)

        class Subclass(DenestedRoot):
            """A subclass is another class: its module and name differ."""

        cases = [
            ("an impostor", support.impostor(true)),
            ("a subclass", Subclass(Q(3), Q(2), 1)),
            ("a tuple of the fields", (Q(3), Q(2), 1)),
            ("a list of the fields", [Q(3), Q(2), 1]),
            ("None", None),
            ("the record in a tuple", (true,)),
            ("an expression", mf.sqrt(3) + mf.sqrt(2)),
            ("x missing", support.forged(DenestedRoot, y=Q(2), sign=1)),
            ("y missing", support.forged(DenestedRoot, x=Q(3), sign=1)),
            ("the sign missing", support.forged(DenestedRoot, x=Q(3), y=Q(2))),
            ("every field missing", support.forged(DenestedRoot)),
            ("x = y", support.forged(DenestedRoot, x=Q(2), y=Q(2), sign=1)),
            ("y = 0", support.forged(DenestedRoot, x=Q(5), y=Q(0), sign=1)),
            ("y < 0", support.forged(DenestedRoot, x=Q(6), y=Q(-1), sign=1)),
            ("x < y", support.forged(DenestedRoot, x=Q(2), y=Q(3), sign=1)),
            ("the sign 0", support.forged(DenestedRoot, x=Q(3), y=Q(2), sign=0)),
            ("the sign 2", support.forged(DenestedRoot, x=Q(3), y=Q(2), sign=2)),
            ("the sign -2", support.forged(DenestedRoot, x=Q(3), y=Q(2), sign=-2)),
            ("x a Rational of another module", support.forged(DenestedRoot, x=FakeRational(3, 1), y=Q(2), sign=1)),
            ("y a Rational of another module", support.forged(DenestedRoot, x=Q(3), y=FakeRational(2, 1), sign=1)),
            ("x without a numerator", support.forged(DenestedRoot, x=forged_rational(3, 1, missing="numerator"),
                                                     y=Q(2), sign=1)),
            ("y without a denominator", support.forged(DenestedRoot, x=Q(3),
                                                       y=forged_rational(2, 1, missing="denominator"), sign=1)),
            ("x with a zero denominator", support.forged(DenestedRoot, x=forged_rational(3, 0), y=Q(2), sign=1)),
            ("x with a float numerator", support.forged(DenestedRoot, x=forged_rational(3.0, 1), y=Q(2), sign=1)),
            ("y with a bool numerator", support.forged(DenestedRoot, x=Q(3), y=forged_rational(True, 1), sign=1)),
            ("x with an int subclass", support.forged(DenestedRoot, x=forged_rational(Integer(3), 1), y=Q(2),
                                                      sign=1)),
        ]
        self.assertRejectsAll([(name, args, value, ()) for name, value in cases])

    def test_rejects_inexact_numbers_and_lists(self):
        args, true = args_of(5, 2, 6), record(3, 2, 1)
        stand_ins = (True, False, 3.0, "3", None, [3], (3,), Fraction(3), Decimal(3), Integer(3))
        cases = [(f"x = {item!r}", args, support.forged(DenestedRoot, x=item, y=Q(2), sign=1), ())
                 for item in stand_ins]
        cases += [(f"y = {item!r}", args, support.forged(DenestedRoot, x=Q(3), y=item, sign=1), ())
                  for item in (True, 2.0, "2", None, [2], Fraction(2), Integer(2))]
        cases += [(f"sign = {item!r}", args, support.forged(DenestedRoot, x=Q(3), y=Q(2), sign=item), ())
                  for item in (True, 1.0, "1", None, [1], (1,), Q(1), Fraction(1), Integer(1), Sign.PLUS)]
        for name, items in (("a", (True, 5.0, "5", None, [5], (5,), Fraction(5), Decimal(5), Integer(5),
                                   FakeRational(5, 1), forged_rational(5.0, 1), forged_rational(10, 2),
                                   forged_rational(5, -1), forged_rational(5, 1, missing="denominator"))),
                            ("b", (True, 2.0, "2", None, [2], Fraction(2), Integer(2), forged_rational(4, 2),
                                   forged_rational(2, 0))),
                            ("d", (6.0, "6", None, [6], Q(6), Fraction(6), Integer(6), True))):
            cases += [(f"{name} = {label(item)}", dict(args, **{name: item}), true, ()) for item in items]
        cases += [("evidence as a list", args, true, []), ("evidence as a tuple subclass", args, true, Evidence()),
                  ("evidence None", args, true, None)]
        self.assertRejectsAll(cases)
        self.assertAccepts(OP, args, true)

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        """The evidence is (), always. Evidence that is consistent with the claim (c = x - y, the root of
        a^2 - b^2*d, or m with its denominator) is refused like any other: there is no place for it."""
        args, true = args_of(5, 2, 6), record(3, 2, 1)
        for evidence in ((0,), ((),), (true,), (Q(1),), (1,), ((1, 1),), ("c", 1), (Q(3), Q(2)), (None,), [], [()],
                         None, {}, 0, "", Evidence(), Evidence(((),)), (DenestedRoot(Q(3), Q(2), 1),)):
            with self.subTest(evidence=evidence):
                self.assertRejects(OP, args, true, evidence)

    def test_rejects_wrong_operations_and_arguments(self):
        args, true = args_of(3, 2, 2), record(2, 1, 1)
        self.assertAccepts(OP, args, true)
        for operation in ("denest", "Denest_sqrt", "denest_sqrt ", "DENEST_SQRT", "sqrt_denest", "continued_fraction",
                          "", None, 1, ("denest_sqrt",), b"denest_sqrt", ["denest_sqrt"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, true)
        self.assertRejectsAll([
            ("arguments as pairs", tuple(args.items()), true, ()),
            ("arguments as a list of pairs", list(args.items()), true, ()),
            ("arguments as a read-only mapping", MappingProxyType(args), true, ()),
            ("None as arguments", None, true, ()),
            ("no arguments", {}, true, ()),
            ("a missing", {"b": 2, "d": 2}, true, ()),
            ("d missing", {"a": 3, "b": 2}, true, ()),
            ("an extra argument", dict(args, c=1), true, ()),
            ("a renamed", {"A": 3, "b": 2, "d": 2}, true, ()),
            ("arguments of another operation", {"p": 3, "q": 2, "d": 2, "r": 1}, true, ()),
        ])

    def test_oversized_forgeries_fail_fast(self):
        """Values longer than any true value can be answer False before any arithmetic, whatever else they satisfy (a
        forged y = 1/GIANT and x = a - y meet x + y == a exactly); oversized evidence answers False at once. A value of
        ten million bits shows that nothing is computed on it: its gcd alone would charge far past max_work."""
        args = args_of(3, 2, 2)
        values = []
        for giant in (GIANT, 1 << 10_000_000):  # A shift, which builds the long integer in linear time.
            tiny = Q(1, giant)  # A denominator 1 or a numerator 1 keeps the Rational constructor's gcd trivial.
            values += [
                support.forged(DenestedRoot, x=Q(giant + 1), y=Q(1), sign=1),
                support.forged(DenestedRoot, x=Q(2), y=tiny, sign=1),
                support.forged(DenestedRoot, x=Q(3) - tiny, y=tiny, sign=1),
                support.forged(DenestedRoot, x=forged_rational(giant + 1, giant), y=tiny, sign=1),
                support.forged(DenestedRoot, x=forged_rational(2 * giant, giant), y=Q(1), sign=1),
                support.forged(DenestedRoot, x=Q(giant), y=Q(giant - 1), sign=1),
                support.forged(DenestedRoot, x=Q(2), y=Q(1), sign=giant),
                support.forged(DenestedRoot, x=Q(2), y=Q(1), sign=-giant),
            ]
        for index, value in enumerate(values):
            with self.subTest(value=index):
                self.assertRejectsFast(args, value)
        # Within max_integer_bits, but far longer than the true value of these arguments can be.
        long = 2 ** 60_000
        self.assertRejectsFast(args, support.forged(DenestedRoot, x=Q(3) - Q(1, long), y=Q(1, long), sign=1))
        self.assertRejectsFast(args, record(long + 1, long, 1))
        true = record(2, 1, 1)
        for evidence in ((GIANT,), (0,) * 100_000, ((),) * 100_000, (Q(GIANT),), ((GIANT,) * 1_000,)):
            with self.subTest(evidence=len(evidence)):
                self.assertRejectsFast(args, true, evidence)

    def test_arguments_beyond_max_integer_bits_exceed_the_budget(self):
        """An argument longer than max_integer_bits raises ResourceLimitError, as run() and the verifier refuse it
        before any checker runs: such a claim is beyond the budget, not wrong. The refusal is immediate."""
        true = record(2, 1, 1)
        for oversized, args in (("a", args_of(GIANT, 2, 2)), ("b", args_of(3, GIANT, 2)),
                                ("d", args_of(3, 2, GIANT + 1)),
                                ("the denominator of a", {"a": Q(1, GIANT), "b": 2, "d": 2}),
                                ("the numerator of b", {"a": 3, "b": Q(GIANT, 3), "d": 2})):
            with self.subTest(oversized=oversized):
                started = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, OP, args, true)
                self.assertLess(time.perf_counter() - started, FAST)
        limits = ComputationLimits(max_integer_bits=64)  # The same at other limits: 2**64 has 65 bits.
        self.assertAccepts(OP, args_of(2 ** 63 + 1, 2, 2 ** 63), record(2 ** 63, 1, 1), limits=limits)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, args_of(2 ** 64 + 1, 2, 2 ** 64 + 2), record(2, 1, 1), limits=limits)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_before_it_runs(self):
        """The charges depend on the claim alone: a budget one unit short of a check refuses it with
        ResourceLimitError, its exact size passes, and long integers cost more than short ones. Every claim is built
        before the budget opens, which then holds the checker's charges only (a record charges its own checks)."""
        (a, b, d), value = backward(Fraction(1, 3 ** 12_000), Fraction(5 ** 6_000, 7 ** 5_000), 6, 1)
        args, true = args_of(a, b, d), record(*value)
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation(OP, args, true, ()), True)
        work = budget.work
        # Four gcds of 19,000 to 47,000 bits charge about 22,400 units (trusted.upoly.charge_gcd) and the products
        # about 10,800 more (charge_product): each part alone stays below this bound.
        self.assertGreater(work, 25_000)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, args, true, limits=ComputationLimits(max_work=work - 1))
        self.assertAccepts(OP, args, true, limits=ComputationLimits(max_work=work))
        small_args, small_true = args_of(3, 2, 2), record(2, 1, 1)
        with computation(fresh=True) as small:
            self.assertIs(checker.check_operation(OP, small_args, small_true, ()), True)
        self.assertGreaterEqual(small.work, 10)  # One unit per product and square root: 15 at the time of writing.
        self.assertLess(small.work, 40)
        d = 2 ** 65_535 + 3  # isqrt(d) is charged like a product of d with itself: 4,097 units at 65,536 bits.
        full_args, full_true = args_of(d + 1, 2, d), record(d, 1, 1)
        with computation(fresh=True) as full:
            self.assertIs(checker.check_operation(OP, full_args, full_true, ()), True)
        self.assertGreater(full.work, 4_097)

    def test_no_power_and_no_producer_code(self):
        """The checker forms no power (squares are products, so nothing escapes the charges), imports only the
        standard library's math and the layers that a checker may use, and names no importlib, __import__, sys or
        builtins in its code."""
        tree = ast.parse(inspect.getsource(checker))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertEqual(names & {"pow", "importlib", "__import__", "sys", "builtins", "eval", "exec"}, set())
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imported, {"__future__", "math", "mathforge.limits", "mathforge.model",
                                    "mathforge.trusted.upoly"})

    # -----------------------------------------------------------------------------------------------------------
    # The verifier, the record and its presentation

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip, and a wrong value fails
        the mathematics check while the trace stays verified."""
        for (a, b, d), (x, y, sign) in KNOWN + (((Fraction(5, 2), -1, 6), (Fraction(3, 2), 1, -1)),):
            args = args_of(a, b, d)
            with self.subTest(args=args):
                decoded = self.roundtrip(claimed(args, record(x, y, sign)))
                self.assertIs(type(decoded.value), DenestedRoot)
                report = support.statuses(mf.verify(claimed(args, record(x, y, -sign))))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
                report = support.statuses(mf.verify(claimed(args, record(x, y, sign), (Q(1),))))
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
        report = support.statuses(mf.verify(claimed(args_of(5, 2, 4), record(4, 1, 1))))  # d = 4 is invalid input.
        self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_record_canonical_form(self):
        for x, y, sign in ((Q(2), Q(1), 1), (Q(3, 2), Q(1, 2), -1), (Q(10 ** 40), Q(1, 10 ** 40), 1),
                           (Q(1), Q(1, 2), 1)):
            with self.subTest(x=x, y=y, sign=sign):
                root = DenestedRoot(x, y, sign)
                self.assertEqual((root.x, root.y, root.sign), (x, y, sign))
                self.assertEqual(mf.from_data(mf.to_data(root)), root)
        for x, y, sign in ((Q(1), Q(2), 1), (Q(2), Q(2), 1), (Q(2), Q(0), 1), (Q(2), Q(-1), 1), (Q(-1), Q(-2), 1),
                           (Q(2), Q(1), 0), (Q(2), Q(1), 2), (Q(2), Q(1), True), (Q(2), Q(1), 1.0), (Q(2), Q(1), Q(1)),
                           (2, Q(1), 1), (Q(2), 1, 1), (2.0, Q(1), 1), ("2", Q(1), 1), (None, Q(1), 1),
                           (Fraction(2), Q(1), 1), (SubRational(2), Q(1), 1), (Q(2), Q(1), Sign.PLUS),
                           (forged_rational(4, 2), Q(1), 1), (forged_rational(2, -1), Q(1), 1),
                           (forged_rational(2, 1, missing="numerator"), Q(1), 1), (Q(2), forged_rational(1.0, 1), 1),
                           (FakeRational(2, 1), Q(1), 1)):
            with self.subTest(x=label(x), y=label(y), sign=label(sign)), self.assertRaises(mf.InvalidInput):
                DenestedRoot(x, y, sign)
        with self.assertRaises(ResourceLimitError):
            DenestedRoot(Q(GIANT), Q(1), 1)
        document = mf.to_data(DenestedRoot(Q(2), Q(1), 1))
        data = document["data"]
        data["x"], data["y"] = data["y"], data["x"]  # sqrt(1) + sqrt(2): the same number, not the canonical form.
        with self.assertRaisesRegex(mf.SerializationError, "^Invalid mathematical document$"):
            mf.from_data(document)
        document = mf.to_data(DenestedRoot(Q(2), Q(1), 1))
        document["data"]["x"] = {"type": "integer", "value": "2"}  # An int for the Rational x.
        with self.assertRaisesRegex(mf.SerializationError, "^Invalid mathematical document$"):
            mf.from_data(document)

    def test_rendering(self):
        """The legacy expression sqrt(x) + sign*sqrt(y), x first, a rational square root shown as a rational."""
        for value, text, latex in (
                (record(2, 1, 1), "sqrt(2) + 1", r"\sqrt{2} + 1"),
                (record(3, 2, 1), "sqrt(3) + sqrt(2)", r"\sqrt{3} + \sqrt{2}"),
                (record(4, 3, -1), "2 - sqrt(3)", r"2 - \sqrt{3}"),
                (record(Fraction(3, 2), Fraction(1, 2), 1), "sqrt(3/2) + sqrt(1/2)",
                 r"\sqrt{\frac{3}{2}} + \sqrt{\frac{1}{2}}"),
                (record(2, 1, -1), "sqrt(2) - 1", r"\sqrt{2} - 1"),
                (record(4, 3, 1), "2 + sqrt(3)", r"2 + \sqrt{3}"),
                (record(Fraction(9, 4), Fraction(1, 2), -1), "3/2 - sqrt(1/2)", r"\frac{3}{2} - \sqrt{\frac{1}{2}}")):
            with self.subTest(text=text):
                self.assertEqual((mf.render_text(value), mf.to_latex(value)), (text, latex))
        with self.assertRaisesRegex(mf.ResourceLimitError, "characters"):
            mf.render_text(record(10 ** 30, 3, 1), limits=mf.RenderLimits(max_chars=20))
