"""The independent checker of mathforge.areas.vector.products (mathforge.certify.vector.products), tested on its own
before the family's producer exists.

Every true claim is computed here, never by the producer: with fractions.Fraction and the textbook formulas (the sum
of the products, the cofactor formula of the cross product, (a.b / b.b) b and a.(b x c)), each checked against the
defining properties of its value and, for determinants, against the Leibniz expansion over every permutation. The
evidence is always (), the layout that the family module states. The checker answers False to every corrupted claim,
never with an exception, and refuses an oversized forgery within FAST seconds under the default budget.

Run this module by name with python -m unittest tests.v09.vector.test_products_checker
"""

from __future__ import annotations

import ast
from collections import namedtuple
from collections.abc import Iterable, Sequence
from fractions import Fraction
import inspect
from itertools import permutations
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.vector import products as family
from mathforge.certify.vector import products as checker
from mathforge.contracts import CheckStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
OPERATIONS = ("dot", "cross", "projection", "triple_product")

Number = int | Fraction | Rational


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def frac(number: Number) -> Fraction:
    """An exact number of the tests as a Fraction."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def exact(number: Number) -> int | Rational:
    """The number as a request holds it: an int when it is integral, else a Rational."""
    number = frac(number)
    return number.numerator if number.denominator == 1 else Rational(number.numerator, number.denominator)


def rational(number: Number) -> Rational:
    """The number as a value holds it: always a canonical Rational, Rational(n, 1) for an integer."""
    number = frac(number)
    return Rational(number.numerator, number.denominator)


def arguments(*vectors: Sequence[Number]) -> dict[str, object]:
    """The arguments a, b (and c) as a request holds them: tuples of ints and Rationals."""
    return {name: tuple(exact(x) for x in vector) for name, vector in zip(("a", "b", "c"), vectors)}


def values(numbers: Iterable[Number]) -> tuple[Rational, ...]:
    """A vector as a value holds it: a tuple of canonical Rationals."""
    return tuple(rational(x) for x in numbers)


def dot(a: Sequence[Number], b: Sequence[Number]) -> Fraction:
    """a.b, the sum of the products of the entries."""
    assert len(a) == len(b)
    return sum((frac(x) * frac(y) for x, y in zip(a, b)), Fraction(0))


def cross(a: Sequence[Number], b: Sequence[Number]) -> list[Fraction]:
    """a x b by the cofactor formula."""
    (a1, a2, a3), (b1, b2, b3) = [frac(x) for x in a], [frac(y) for y in b]
    return [a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1]


def projection(a: Sequence[Number], b: Sequence[Number]) -> list[Fraction]:
    """(a.b / b.b) b, the orthogonal projection of a onto b != 0."""
    t = dot(a, b) / dot(b, b)
    return [t * frac(y) for y in b]


def triple(a: Sequence[Number], b: Sequence[Number], c: Sequence[Number]) -> Fraction:
    """a.(b x c)."""
    return dot(a, cross(b, c))


def leibniz(rows: Sequence[Sequence[Number]]) -> Fraction:
    """The determinant by the Leibniz formula over every permutation, each sign from the number of inversions."""
    n = len(rows)
    total = Fraction(0)
    for order in permutations(range(n)):
        inversions = sum(1 for i in range(n) for j in range(i + 1, n) if order[i] > order[j])
        term = Fraction(1)
        for row, column in zip(rows, order):
            term *= frac(row[column])
        total += -term if inversions % 2 else term
    return total


def true_dot(a: Sequence[Number], b: Sequence[Number]) -> tuple[dict[str, object], Rational]:
    """(args, value) of a true dot claim."""
    return arguments(a, b), rational(dot(a, b))


def true_cross(a: Sequence[Number], b: Sequence[Number]) -> tuple[dict[str, object], tuple[Rational, ...]]:
    """(args, value) of a true cross claim; the reference value is orthogonal to a and b and positively oriented."""
    c = cross(a, b)
    assert dot(c, a) == 0 and dot(c, b) == 0 and leibniz([a, b, c]) == dot(c, c)
    return arguments(a, b), values(c)


def true_projection(a: Sequence[Number], b: Sequence[Number]) -> tuple[dict[str, object], tuple[Rational, ...]]:
    """(args, value) of a true projection claim; the reference value leaves a residual orthogonal to b."""
    p = projection(a, b)
    assert dot([frac(x) - y for x, y in zip(a, p)], b) == 0
    return arguments(a, b), values(p)


def true_triple(a: Sequence[Number], b: Sequence[Number], c: Sequence[Number]) -> tuple[dict[str, object], Rational]:
    """(args, value) of a true triple_product claim; the reference agrees with the Leibniz expansion."""
    v = triple(a, b, c)
    assert v == leibniz([a, b, c])
    return arguments(a, b, c), rational(v)


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values, arguments and evidence that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Fractional(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Entries(tuple):
    """A tuple subclass for a vector, a value or the evidence."""


Triple = namedtuple("Triple", "x y z")  # A tuple subclass with the shape of a cross product.


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def inexact(number: Number) -> list[tuple[str, object]]:
    """(label, stand-in) for an exact number that is neither an exact int nor a canonical Rational."""
    p, q = frac(number).numerator, frac(number).denominator
    return [("a bool", bool(p)), ("a float", p / q), ("a str", str(frac(number))), ("None", None),
            ("a Fraction", Fraction(p, q)), ("an int subclass", Integer(p)),
            ("a Rational subclass", Fractional(p, q)), ("a list", [exact(number)]), ("a tuple", (exact(number),)),
            ("a Rational of a bool", forged_rational(bool(p), 1)), ("a Rational of a float", forged_rational(p / q, 1)),
            ("a Rational with a float denominator", forged_rational(p, float(q))),
            ("a Rational in higher terms", forged_rational(2 * p, 2 * q)),
            ("a Rational with a negative denominator", forged_rational(-p, -q)),
            ("a Rational with a zero denominator", forged_rational(p, 0)),
            ("a Rational without a denominator", support.forged(Rational, numerator=p)),
            ("a Rational without a numerator", support.forged(Rational, denominator=q)),
            ("an impostor Rational", support.impostor(Rational(p, q)))]


class VectorProductsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Iterable[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case of the operation."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    def known_claims(self) -> dict[str, tuple[dict[str, object], object]]:
        """One true claim of each operation, with integral values where a value can be integral."""
        return {"dot": true_dot((1, 2, 3), (4, 5, 6)),  # 32
                "cross": true_cross((1, 2, 3), (4, 5, 6)),  # (-3, 6, -3)
                "projection": true_projection((1, 2), (3, 4)),  # (33/25, 44/25)
                "triple_product": true_triple((1, 2, 3), (4, 5, 6), (7, 8, 10))}  # -3

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        def vectors(*names: str) -> tuple[Argument, ...]:
            return tuple(Argument(name, "", None, "rational_list") for name in names)

        expected = (
            OperationSpec("dot", "vector", vectors("a", "b"), "Rational",
                          "Compute the dot product of two exact rational vectors.", SCOPE, version=6),
            OperationSpec("cross", "vector", vectors("a", "b"), "tuple[Rational, Rational, Rational]",
                          "Compute the cross product of two exact rational vectors in three dimensions.", SCOPE,
                          version=6),
            OperationSpec("projection", "vector", vectors("a", "b"), "tuple[Rational, ...]",
                          "Compute the orthogonal projection of an exact rational vector onto a nonzero rational "
                          "vector.", SCOPE, version=6),
            OperationSpec("triple_product", "vector", vectors("a", "b", "c"), "Rational",
                          "Compute the scalar triple product of three exact rational vectors in three dimensions.",
                          SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.SCOPE, SCOPE)
        self.assertEqual(family.TYPES, {})  # Every value is a Rational or a tuple of Rationals.
        self.assertEqual(getattr(family, "RENDER", {}), {})  # The shared renderer presents both.
        self.assertEqual(family.CHECKER, "mathforge.certify.vector.products")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
                self.assertEqual(checker.ARGUMENTS[spec.name], tuple(argument.name for argument in spec.arguments))
                # One sentence, as the catalog's summaries are.
                self.assertTrue(spec.summary[0].isupper() and spec.summary.endswith("."))
                self.assertNotIn(". ", spec.summary)
        self.assertEqual(tuple(checker.ARGUMENTS), OPERATIONS)
        self.assertEqual(checker.DIMENSIONS, {"dot": None, "cross": 3, "projection": None, "triple_product": 3})
        self.assertIs(support.checker(family), checker)

    def test_sarrus_is_the_leibniz_formula(self):
        """The checker's six terms are the six permutations of (0, 1, 2), each with the sign of its inversions."""
        signs = {}
        for i, j, k, sign in checker.SARRUS:
            order = (i, j, k)
            inversions = sum(1 for x in range(3) for y in range(x + 1, 3) if order[x] > order[y])
            self.assertEqual(sign, -1 if inversions % 2 else 1, order)
            signs[order] = sign
        self.assertEqual(set(signs), set(permutations(range(3))))
        self.assertEqual(len(checker.SARRUS), 6)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_accepts_the_known_values_of_the_card(self):
        for a, b, c in (((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((1, 2, 3), (4, 5, 6), (-3, 6, -3)),
                        ((1, 2, 3), (2, 4, 6), (0, 0, 0))):  # The card's cross products: parallel vectors give 0.
            with self.subTest(operation="cross", a=a, b=b):
                self.assertEqual(cross(a, b), [Fraction(x) for x in c])
                self.assertAccepts("cross", arguments(a, b), values(c))
        for a, b, v in (((1, 2, 3), (4, 5, 6), 32), ((1, 0), (0, 1), 0), ((-2,), (5,), -10),
                        ((Fraction(1, 2), Fraction(1, 3)), (3, 4), Fraction(17, 6))):
            with self.subTest(operation="dot", a=a, b=b):
                self.assertEqual(dot(a, b), v)
                self.assertAccepts("dot", arguments(a, b), rational(v))
        for a, b, p in (((1, 2), (3, 4), (Fraction(33, 25), Fraction(44, 25))), ((2, 0), (1, 0), (2, 0)),
                        ((0, 1), (1, 0), (0, 0)), ((3, 4, 5), (0, 0, 2), (0, 0, 5)), ((7,), (-2,), (7,))):
            with self.subTest(operation="projection", a=a, b=b):
                self.assertEqual(projection(a, b), [Fraction(x) for x in p])
                self.assertAccepts("projection", arguments(a, b), values(p))
        for a, b, c, v in (((1, 0, 0), (0, 1, 0), (0, 0, 1), 1), ((1, 0, 0), (0, 0, 1), (0, 1, 0), -1),
                           ((1, 2, 3), (4, 5, 6), (7, 8, 9), 0), ((1, 2, 3), (4, 5, 6), (7, 8, 10), -3)):
            with self.subTest(operation="triple_product", a=a, b=b, c=c):
                self.assertEqual(triple(a, b, c), v)
                self.assertEqual(leibniz([a, b, c]), v)
                self.assertAccepts("triple_product", arguments(a, b, c), rational(v))

    def test_accepts_special_cases(self):
        half, third, seventh = Fraction(1, 2), Fraction(1, 3), Fraction(1, 7)
        long_integer = 2 ** 300 + 7
        dots = [
            ("one entry", (5,), (Fraction(-3, 4),)),
            ("zero vectors", (0, 0, 0), (0, 0, 0)),
            ("orthogonal vectors", (1, 1, 0), (1, -1, 5)),
            ("a vector with itself", (third, -2, seventh), (third, -2, seventh)),
            ("different denominators", (half, third, seventh, Fraction(5, 11)), (Fraction(2, 9), 6, -7, 11)),
            ("cancelling denominators", (third, seventh), (3, 7)),
            ("negative entries", (-1, -2, -3), (-4, 5, -6)),
            ("a thousand entries", tuple(range(-500, 500)), tuple(Fraction(k, 1 + k % 5) for k in range(1000))),
            ("long integers", (long_integer, -long_integer, 1), (long_integer, long_integer, Fraction(1, 2 ** 200))),
        ]
        for label, a, b in dots:
            with self.subTest(operation="dot", case=label):
                self.assertAccepts("dot", *true_dot(a, b))
        crosses = [
            ("the zero vector", (0, 0, 0), (1, 2, 3)),
            ("a vector with itself", (half, 3, -seventh), (half, 3, -seventh)),
            ("parallel by a rational factor", (2, -4, 6), (-third, Fraction(2, 3), -1)),
            ("j x k = i", (0, 1, 0), (0, 0, 1)),
            ("k x i = j", (0, 0, 1), (1, 0, 0)),
            ("b x a", (4, 5, 6), (1, 2, 3)),
            ("rational entries", (half, -third, 2), (seventh, 0, Fraction(-9, 4))),
            ("long integers", (long_integer, 1, -long_integer), (3, long_integer, Fraction(1, long_integer))),
        ]
        for label, a, b in crosses:
            with self.subTest(operation="cross", case=label):
                self.assertAccepts("cross", *true_cross(a, b))
        projections = [
            ("a parallel to b", (2, 4, 6), (-1, -2, -3)),
            ("a orthogonal to b", (1, 1), (1, -1)),
            ("a = 0", (0, 0, 0), (1, 2, 3)),
            ("one entry", (Fraction(-5, 3),), (Fraction(2, 7),)),
            ("b with leading zeros", (1, 2, 3, 4), (0, 0, half, -1)),
            ("a = b", (third, 2), (third, 2)),
            ("rational entries", (half, -third, 2), (seventh, 0, Fraction(-9, 4))),
            ("two hundred entries", tuple(range(200)), tuple(Fraction(1, 1 + k % 3) for k in range(200))),
            ("long integers", (long_integer, 1), (1, long_integer)),
        ]
        for label, a, b in projections:
            with self.subTest(operation="projection", case=label):
                self.assertAccepts("projection", *true_projection(a, b))
        triples = [
            ("a repeated vector", (1, 2, 3), (4, 5, 6), (1, 2, 3)),
            ("coplanar vectors", (1, 2, 3), (4, 5, 6), (5, 7, 9)),
            ("a zero vector", (0, 0, 0), (4, 5, 6), (7, 8, 9)),
            ("cyclic order", (4, 5, 6), (7, 8, 10), (1, 2, 3)),
            ("odd order", (4, 5, 6), (1, 2, 3), (7, 8, 10)),
            ("rational entries", (half, -third, 2), (seventh, 0, Fraction(-9, 4)), (1, Fraction(5, 6), -seventh)),
            ("long integers", (long_integer, 0, 1), (0, long_integer, 0), (1, 0, Fraction(1, long_integer))),
        ]
        for label, a, b, c in triples:
            with self.subTest(operation="triple_product", case=label):
                self.assertAccepts("triple_product", *true_triple(a, b, c))
        # Ints and Rationals, also with denominator 1, are the same exact numbers in a request.
        mixed = {"a": (Rational(1), 2, Rational(3)), "b": (4, Rational(5), 6)}
        self.assertAccepts("dot", mixed, Rational(32))
        self.assertAccepts("cross", mixed, values((-3, 6, -3)))
        self.assertAccepts("triple_product", dict(mixed, c=(Rational(7), 8, 10)), Rational(-3))
        self.assertAccepts("projection", {"a": (Rational(1), 2), "b": (3, Rational(4))}, values(projection((1, 2),
                                                                                                        (3, 4))))

    def test_accepts_seeded_true_claims(self):
        rng = random.Random(20261112)

        def number(bits: int) -> Fraction:
            if rng.random() < 0.15:
                return Fraction(0)
            numerator = rng.randrange(-(2 ** bits), 2 ** bits + 1)
            return Fraction(numerator) if rng.random() < 0.4 else Fraction(numerator, rng.randrange(1, 2 ** bits + 1))

        def vector(n: int, bits: int) -> list[Fraction]:
            return [number(bits) for _ in range(n)]

        for case in range(150):
            n, bits = rng.randrange(1, 9), rng.choice((2, 5, 16, 64))
            a, b = vector(n, bits), vector(n, bits)
            if rng.random() < 0.2:  # Parallel vectors now and then.
                b = [number(3) * x for x in a]
            with self.subTest(operation="dot", case=case):
                self.assertAccepts("dot", *true_dot(a, b))
            if any(b):
                with self.subTest(operation="projection", case=case):
                    self.assertAccepts("projection", *true_projection(a, b))
        for case in range(150):
            bits = rng.choice((2, 5, 16, 64))
            a, b, c = vector(3, bits), vector(3, bits), vector(3, bits)
            if rng.random() < 0.2:  # Parallel vectors, a zero cross product and a zero triple product.
                b = [number(3) * x for x in a]
            with self.subTest(operation="cross", case=case):
                self.assertAccepts("cross", *true_cross(a, b))
            with self.subTest(operation="triple_product", case=case):
                self.assertAccepts("triple_product", *true_triple(a, b, c))

    def test_full_size_claims_fit_the_default_budget(self):
        """Values of about 64,000 bits, and vectors of thousands of entries, verify under the default limits."""
        rng = random.Random(20261113)

        def big(bits: int) -> int:
            return rng.choice((-1, 1)) * (rng.getrandbits(bits) | 1 << (bits - 1))

        def bigs(bits: int) -> list[int]:
            return [big(bits) for _ in range(3)]

        def small_rational() -> Fraction:
            return Fraction(rng.getrandbits(64) - 2 ** 63, 2 ** rng.randrange(0, 8))

        claims = [
            ("dot", *true_dot(bigs(32_000), bigs(32_000))),
            ("cross", *true_cross(bigs(32_000), bigs(32_000))),
            ("projection", *true_projection(bigs(20_000), bigs(20_000))),
            ("triple_product", *true_triple(bigs(21_000), bigs(21_000), bigs(21_000))),
            ("dot", *true_dot([rng.getrandbits(64) for _ in range(20_000)],
                              [small_rational() for _ in range(20_000)])),
            ("projection", *true_projection([rng.getrandbits(32) for _ in range(3_000)],
                                            [small_rational() for _ in range(3_000)])),
        ]
        for operation, args, value in claims:
            with self.subTest(operation=operation, length=len(args["a"])):
                with computation(fresh=True) as budget:
                    start = time.perf_counter()
                    self.assertIs(checker.check_operation(operation, args, value, ()), True)
                    elapsed = time.perf_counter() - start
                self.assertLess(budget.work, ComputationLimits().max_work)
                self.assertLess(elapsed, 2 * FAST)
        self.assertGreater(claims[1][2][0].numerator.bit_length(), 60_000)  # The values are of the full size.

    # -----------------------------------------------------------------------------------------------------------
    # Wrong and non-canonical values

    def test_rejects_wrong_values(self):
        dot_args = arguments((1, 2, 3), (4, 5, 6))  # 32; |a + b|^2 = 155, |a|^2 = 14, |b|^2 = 77.
        self.assertAccepts("dot", dot_args, Rational(32))
        self.assertRejectsAll("dot", [
            (label, dot_args, rational(v), ()) for label, v in (
                ("31", 31), ("33", 33), ("-32", -32), ("0", 0), ("twice the value", 64), ("half the value", 16),
                ("|a + b|^2", 155), ("|a|^2", 14), ("|b|^2", 77), ("|a + b|^2 - |a|^2 - |b|^2", 64),
                ("1/32", Fraction(1, 32)), ("32 + 1/7", 32 + Fraction(1, 7)),
                ("32 - 1/2**80", 32 - Fraction(1, 2 ** 80)))])
        rational_args = arguments((Fraction(1, 2), Fraction(1, 3)), (3, 4))  # 17/6
        self.assertRejectsAll("dot", [(label, rational_args, rational(v), ()) for label, v in (
            ("17/5", Fraction(17, 5)), ("6/17", Fraction(6, 17)), ("17", 17), ("17/36", Fraction(17, 36)))])

        args, value = true_cross((1, 2, 3), (4, 5, 6))  # (-3, 6, -3)
        self.assertEqual(value, values((-3, 6, -3)))
        cases = [(label, args, values(c), ()) for label, c in (
            ("b x a", (3, -6, 3)), ("twice a x b", (-6, 12, -6)), ("the zero vector", (0, 0, 0)),
            ("components rotated", (6, -3, -3)), ("components swapped", (-3, -3, 6)), ("a", (1, 2, 3)),
            ("b", (4, 5, 6)), ("half of a x b", (Fraction(-3, 2), 3, Fraction(-3, 2))),
            ("a seventh too much", (-3 + Fraction(1, 7), 6, -3)))]
        cases += [(f"component {k} off by one", args, values(c + (1 if i == k else 0) for i, c in
                                                             enumerate((-3, 6, -3))), ()) for k in range(3)]
        cases += [("two components", args, values((-3, 6)), ()), ("four components", args, values((-3, 6, -3, 0)), ()),
                  ("a dot product", args, Rational(32), ()), ("an empty tuple", args, (), ())]
        rational_cross = arguments((Fraction(1, 2), 0, 0), (0, Fraction(1, 3), 0))  # (0, 0, 1/6)
        self.assertAccepts("cross", rational_cross, values((0, 0, Fraction(1, 6))))
        cases += [(f"(0, 0, {v})", rational_cross, values((0, 0, v)), ())
                  for v in (Fraction(1, 3), Fraction(-1, 6), 6, Fraction(1, 5), Fraction(1, 36))]
        self.assertRejectsAll("cross", cases)

        args, value = true_projection((1, 2), (3, 4))  # (33/25, 44/25)
        self.assertEqual(value, values((Fraction(33, 25), Fraction(44, 25))))
        self.assertRejectsAll("projection", [(label, args, values(p), ()) for label, p in (
            ("twice the projection", (Fraction(66, 25), Fraction(88, 25))),
            ("plus a vector orthogonal to b", (Fraction(37, 25), Fraction(41, 25))),
            ("components swapped", (Fraction(44, 25), Fraction(33, 25))), ("a", (1, 2)), ("b", (3, 4)),
            ("the zero vector", (0, 0)), ("b onto a", (Fraction(11, 5), Fraction(22, 5))),
            ("a - p", (Fraction(-8, 25), Fraction(6, 25))), ("a denominator 26", (Fraction(33, 26), Fraction(44, 26))),
            ("three components", (Fraction(33, 25), Fraction(44, 25), 0)), ("one component", (Fraction(33, 25),)))]
            + [("the scalar t", args, Rational(11, 25), ()), ("an empty tuple", args, (), ())])

        args, value = true_triple((1, 2, 3), (4, 5, 6), (7, 8, 10))  # -3
        self.assertRejectsAll("triple_product", [(label, args, rational(v), ()) for label, v in (
            ("a.(c x b)", 3), ("-2", -2), ("-4", -4), ("0", 0), ("-3/2", Fraction(-3, 2)), ("-6", -6),
            ("-3 + 1/2**70", -3 + Fraction(1, 2 ** 70)))] + [("a tuple", args, (Rational(-3),), ())])

    def test_rejects_non_canonical_values(self):
        """Mathematically right values in a form that no canonical value has: an int, a list, a tuple subclass, or a
        Rational that is forged, of another class or not in lowest terms."""
        claims = self.known_claims()
        for operation in ("dot", "triple_product"):
            args, value = claims[operation]
            self.assertAccepts(operation, args, forged_rational(value.numerator, value.denominator))  # Forging alone.
            self.assertRejectsAll(operation, [(label, args, item, ()) for label, item in
                                              [("an int", value.numerator)] + inexact(value)])
        args, value = true_dot((Fraction(1, 2), Fraction(1, 3)), (3, 4))  # 17/6, not integral.
        self.assertRejectsAll("dot", [(label, args, item, ()) for label, item in inexact(value)])
        for operation, args, value in (("cross", *claims["cross"]), ("projection", *claims["projection"]),
                                       ("projection", *true_projection((2, 0), (1, 0)))):  # (2, 0), integral.
            cases: list[tuple[str, object, object, object]] = [
                ("a list", args, list(value), ()), ("a tuple subclass", args, Entries(value), ()),
                ("(numerator, denominator) pairs", args, tuple((x.numerator, x.denominator) for x in value), ())]
            if len(value) == 3:
                cases.append(("a namedtuple", args, Triple(*value), ()))
            if all(x.denominator == 1 for x in value):
                cases.append(("ints", args, tuple(x.numerator for x in value), ()))
            for index, number in enumerate(value):
                stand_ins = inexact(number) + ([("an int", number.numerator)] if number.denominator == 1 else [])
                for label, item in stand_ins:
                    cases.append((f"component {index}: {label}", args, value[:index] + (item,) + value[index + 1:],
                                  ()))
            self.assertRejectsAll(operation, cases)

    # -----------------------------------------------------------------------------------------------------------
    # Wrong arguments, domains, evidence and operations

    def test_rejects_corrupted_arguments(self):
        for operation, (args, value) in self.known_claims().items():
            cases: list[tuple[str, object, object, object]] = []
            for name, vector in args.items():
                for index, entry in enumerate(vector):
                    for label, item in inexact(entry):
                        changed = vector[:index] + (item,) + vector[index + 1:]
                        cases.append((f"{name}[{index}]: {label}", dict(args, **{name: changed}), value, ()))
                cases += [(f"{name} as a list", dict(args, **{name: list(vector)}), value, ()),
                          (f"{name} as a tuple subclass", dict(args, **{name: Entries(vector)}), value, ()),
                          (f"{name} one entry shorter", dict(args, **{name: vector[:-1]}), value, ()),
                          (f"{name} one entry longer", dict(args, **{name: vector + (1,)}), value, ()),
                          (f"{name} as None", dict(args, **{name: None}), value, ()),
                          (f"{name} as a Rational", dict(args, **{name: Rational(1)}), value, ())]
            first = next(iter(args))
            cases += [
                ("the arguments as pairs", tuple(args.items()), value, ()),
                ("the arguments as a list of pairs", list(args.items()), value, ()),
                ("the arguments as a read-only mapping", MappingProxyType(args), value, ()),
                ("None as the arguments", None, value, ()),
                ("an argument missing", {name: vector for name, vector in args.items() if name != first}, value, ()),
                ("an extra argument", dict(args, d=args[first]), value, ()),
                ("an argument renamed", {("x" if name == first else name): vector for name, vector in args.items()},
                 value, ()),
                ("a key that is no str", {(1 if name == first else name): vector for name, vector in args.items()},
                 value, ()),
            ]
            self.assertRejectsAll(operation, cases)

    def test_rejects_claims_outside_the_domain(self):
        zero, one = Rational(0), Rational(1)
        self.assertRejectsAll("dot", [
            ("empty vectors", arguments((), ()), zero, ()),
            ("different lengths", arguments((1, 2), (1, 2, 3)), Rational(5), ()),
            ("different lengths, padded value", arguments((1, 2), (1, 2, 3)), Rational(14), ()),
            ("an empty a", arguments((), (1,)), zero, ()),
            ("an empty b", arguments((1,), ()), zero, ()),
        ])
        self.assertRejectsAll("cross", [
            (f"lengths {len(a)} and {len(b)}", arguments(a, b), value, ())
            for a, b in (((1, 0), (0, 1)), ((1, 0, 0, 0), (0, 1, 0, 0)), ((1, 0, 0), (0, 1)), ((), ()),
                         ((1, 0), (0, 1, 0)), ((1,), (2,)))
            for value in (values((0, 0, 1)), values((0, 0)), (one,), (), values((0, 0, 1, 0)))])
        self.assertRejectsAll("projection", [
            ("b = 0", arguments((1, 2), (0, 0)), values((0, 0)), ()),
            ("b = 0, a as the value", arguments((1, 2), (0, 0)), values((1, 2)), ()),
            ("b = 0 in one dimension", arguments((5,), (0,)), values((0,)), ()),
            ("b = 0 as Rationals", {"a": (1, 2), "b": (zero, zero)}, values((0, 0)), ()),
            ("empty vectors", arguments((), ()), (), ()),
            ("different lengths", arguments((1, 2), (3, 4, 5)), values(projection((1, 2, 0), (3, 4, 5))), ()),
            ("different lengths, the other way", arguments((1, 2, 0), (3, 4)), values(projection((1, 2), (3, 4))),
             ()),
        ])
        self.assertRejectsAll("triple_product", [
            (f"lengths {len(a)}, {len(b)} and {len(c)}", arguments(a, b, c), value, ())
            for a, b, c in (((1, 0), (0, 1), (1, 1)), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
                            ((1, 0, 0), (0, 1, 0), (0, 0)), ((), (), ()))
            for value in (zero, one, -one)])

    def test_rejects_wrong_evidence(self):
        claims = self.known_claims()
        consistent = {  # Layouts that a producer might hand over, internally consistent, but not the family's ().
            "dot": [("the three squared norms", (155, 14, 77)), ("the norms as one item", ((155, 14, 77),)),
                    ("the norms as Rationals", (Rational(155), Rational(14), Rational(77)))],
            "cross": [("|c|^2", (54,)), ("det and |c|^2", ((54, 54),)), ("the value again", (claims["cross"][1],))],
            "projection": [("the scalar t", (Rational(11, 25),)), ("a.b and |b|^2", (Rational(11), Rational(25))),
                           ("the residual a - p", (values((Fraction(-8, 25), Fraction(6, 25))),))],
            "triple_product": [("the determinant", (Rational(-3),)), ("the determinant as an int", (-3,)),
                               ("b x c", (values(cross((4, 5, 6), (7, 8, 10))),))],
        }
        for operation, (args, value) in claims.items():
            self.assertRejectsAll(operation, [
                ("evidence None", args, value, None), ("evidence as a list", args, value, []),
                ("evidence as a tuple subclass", args, value, Entries()), ("an extra item", args, value, ((),)),
                ("a number", args, value, (0,)), ("a str", args, value, ("polarization",)),
                ("a str as the evidence", args, value, ""), ("a dict", args, value, {}),
                ("the value itself", args, value, (value,))]
                + [(label, args, value, evidence) for label, evidence in consistent[operation]])

    def test_rejects_wrong_operations(self):
        args, value = self.known_claims()["dot"]
        for operation in ("Dot", "dot ", "inner_product", "scalar_product", "continued_fraction", "powmod", "",
                          None, 7, ("dot",), b"dot", ["dot"], Rational(1)):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value)
        # A true claim of one operation, under the name of another with the same arguments.
        claims = self.known_claims()
        three_d = true_projection((1, 2, 3), (4, 5, 6))
        for operation, (claim_args, claim_value) in [
                ("cross", claims["dot"]), ("projection", claims["dot"]), ("triple_product", claims["dot"]),
                ("dot", claims["cross"]), ("projection", claims["cross"]), ("triple_product", claims["cross"]),
                ("dot", claims["projection"]), ("cross", claims["projection"]), ("cross", three_d),
                ("dot", claims["triple_product"]), ("cross", claims["triple_product"]),
                ("projection", claims["triple_product"])]:
            with self.subTest(operation=operation, value=claim_value):
                self.assertRejects(operation, claim_args, claim_value)

    def test_never_raises_on_junk(self):
        """Every slot of a claim filled with stand-ins of every kind: the checker answers False, never raising."""
        # None of these completes a true claim of the known claims (whose values are 32, (-3, 6, -3), (33/25, 44/25)
        # and -3).
        junk: list[object] = [None, 0, 1.5, "x", b"x", (), [], {}, Rational(1), (Rational(1),), object(), True,
                              support.forged(Rational), forged_rational(1, 0), Entries((1,)), MappingProxyType({}),
                              frozenset(), Fraction(1, 2), (None, None, None), ((1, 2), (3, 4)), [Rational(32)]]
        for operation, (args, value) in self.known_claims().items():
            for index, item in enumerate(junk):
                with self.subTest(operation=operation, item=index, kind=type(item).__name__):
                    self.assertRejects(item, args, value)
                    self.assertRejects(operation, item, value)
                    self.assertRejects(operation, args, item)
                    self.assertRejects(operation, args, value, item if item != () else (item,))
                    for name in args:
                        self.assertRejects(operation, dict(args, **{name: item}), value)

    # -----------------------------------------------------------------------------------------------------------
    # The conditions of the theorems

    def test_each_condition_refuses_on_its_own(self):
        """Values that fail exactly one condition of a theorem, each refused. For cross, -(a x b) fails only the
        orientation and 0 only Lagrange's identity; orthogonality can never be the only failure, since the other two
        already force C = A x B (|C - N|^2 = |C|^2 - 2 C.N + |N|^2 = 0), and the checker tests it anyway, as the card
        says. For projection, 2p fails only orthogonality and p + w with w orthogonal to b only parallelism."""
        a, b = (1, 0, 0), (0, 1, 0)
        n = cross(a, b)
        for label, c, holds in (("-(a x b)", [-x for x in n], (True, True, True, False)),
                                ("the zero vector", [0, 0, 0], (True, True, False, True))):
            conditions = (dot(c, a) == 0, dot(c, b) == 0, dot(c, c) == dot(a, a) * dot(b, b) - dot(a, b) ** 2,
                          leibniz([a, b, c]) == dot(c, c))
            self.assertEqual(conditions, holds, label)
            self.assertRejects("cross", arguments(a, b), values(c))
        a, b = (1, 2), (3, 4)
        p = projection(a, b)
        w = [Fraction(4, 25), Fraction(-3, 25)]  # Orthogonal to b.
        for label, q, holds in (("2p", [2 * x for x in p], (True, False)),
                                ("p + w", [x + y for x, y in zip(p, w)], (False, True))):
            parallel = q[0] * b[1] == q[1] * b[0]
            orthogonal = dot([frac(x) - y for x, y in zip(a, q)], b) == 0
            self.assertEqual((parallel, orthogonal), holds, label)
            self.assertRejects("projection", arguments(a, b), values(q))

    def test_accepts_exactly_the_true_value_among_its_neighbours(self):
        """Every candidate value on a grid around the true one, with the denominators that a true value could have:
        the checker accepts the true value and nothing else."""
        sixth = [Fraction(k, 6) for k in range(-4, 5)]
        for a, b, grid in (((1, 2, 3), (4, 5, 6), range(-6, 7)), ((1, 0, 0), (0, 1, 0), range(-2, 3)),
                           ((1, 2, 3), (2, 4, 6), range(-2, 3)),
                           ((Fraction(1, 2), 0, 1), (0, Fraction(1, 3), 1), sixth)):
            args, truth = true_cross(a, b)
            accepted = [c for c in ((x, y, z) for x in grid for y in grid for z in grid)
                        if support.verdict(family, "cross", args, values(c))]
            with self.subTest(operation="cross", a=a, b=b):
                self.assertEqual([values(c) for c in accepted], [truth])
        args, truth = true_projection((1, 2), (3, 4))  # (33/25, 44/25)
        accepted = [p for p in ((Fraction(i, 25), Fraction(j, 25)) for i in range(20, 51) for j in range(30, 61))
                    if support.verdict(family, "projection", args, values(p))]
        self.assertEqual([values(p) for p in accepted], [truth])
        args, truth = true_dot((Fraction(1, 2), Fraction(1, 3)), (3, 4))  # 17/6
        candidates = {Fraction(k, d) for k in range(-40, 41) for d in (1, 2, 3, 6, 36)}
        self.assertEqual([v for v in sorted(candidates) if support.verdict(family, "dot", args, rational(v))],
                         [Fraction(17, 6)])
        args, truth = true_triple((1, 2, 3), (4, 5, 6), (7, 8, 10))  # -3
        candidates = {Fraction(k, d) for k in range(-20, 21) for d in (1, 2, 3)}
        self.assertEqual([v for v in sorted(candidates) if support.verdict(family, "triple_product", args,
                                                                            rational(v))], [-3])

    def test_forged_denominators_are_refused_before_any_product(self):
        """A claimed cross product whose denominators do not divide D_a D_b is refused before a dot product is
        formed; so is a claimed vector of the wrong length, before an entry is read."""
        args = arguments((1, 2, 3), (4, 5, 6))  # D_a D_b = 1.
        with patch.object(checker, "_dot", side_effect=AssertionError("a dot product was formed")):
            self.assertRejects("cross", args, values((Fraction(-3, 7), 6, -3)))
            self.assertRejects("cross", args, (Rational(-3), Rational(6), forged_rational(-3, 2 ** 60_000)))
        with patch.object(checker, "_claimed", side_effect=AssertionError("an entry was read")):
            self.assertRejects("cross", args, values((-3, 6, -3, 0)))
            self.assertRejects("projection", arguments((1, 2), (3, 4)), tuple(Rational(k) for k in range(1000)))

    # -----------------------------------------------------------------------------------------------------------
    # Oversized forgeries and the budget

    def test_oversized_forgeries_fail_fast(self):
        rng = random.Random(20261114)
        claims = self.known_claims()
        n = 300
        a_long, b_long = tuple(range(n, 0, -1)), tuple(range(1, n + 1))
        # Pairwise different 60,000-bit numerators and denominators: none divides D_a |B|^2 or D_a D_b.
        heavy = tuple(forged_rational(rng.getrandbits(60_000) | 1, rng.getrandbits(60_000) | 1 << 59_999)
                      for _ in range(n))
        heavy_numerators = tuple(forged_rational(rng.getrandbits(60_000)) for _ in range(n))
        multiple = tuple(forged_rational((2 ** 60_000 + 1) * k, 3) for k in b_long)  # Parallel to b, giant.
        many = tuple(Rational(k) for k in range(100_000))
        cases = [
            ("dot", claims["dot"][0], forged_rational(GIANT), ()),
            ("dot", claims["dot"][0], forged_rational(1, GIANT), ()),
            ("dot", claims["dot"][0], forged_rational(GIANT, GIANT + 1), ()),
            ("dot", claims["dot"][0], many, ()),
            ("dot", claims["dot"][0], Rational(32), (0,) * 100_000),
            ("dot", claims["dot"][0], Rational(32), (GIANT,)),
            ("cross", claims["cross"][0], (forged_rational(GIANT), Rational(6), Rational(-3)), ()),
            ("cross", claims["cross"][0], many, ()),
            ("cross", claims["cross"][0], heavy[:3], ()),
            ("cross", claims["cross"][0], heavy_numerators[:3], ()),
            ("projection", arguments(a_long, b_long), heavy, ()),
            ("projection", arguments(a_long, b_long), heavy_numerators, ()),
            ("projection", arguments(a_long, b_long), multiple, ()),
            ("projection", arguments(a_long, b_long), many, ()),
            ("projection", claims["projection"][0], (forged_rational(1, GIANT), Rational(1)), ()),
            ("triple_product", claims["triple_product"][0], forged_rational(GIANT), ()),
            ("triple_product", claims["triple_product"][0], Rational(-3), (0,) * 100_000),
            ("triple_product", claims["triple_product"][0], tuple(range(100_000)), ()),
        ]
        for operation, args, value, evidence in cases:
            with self.subTest(operation=operation, value=type(value).__name__, length=len(args["a"])):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # A request integer over max_integer_bits raises ResourceLimitError, as the request's inspection does.
        for operation, args, value in (("dot", arguments((GIANT, 1), (1, 1)), Rational(1)),
                                       ("cross", arguments((1, 2, 3), (4, Fraction(1, GIANT), 6)), values((0, 0, 0))),
                                       ("triple_product", arguments((1, 0, 0), (0, 1, 0), (0, 0, -GIANT)),
                                        Rational(1))):
            with self.subTest(operation=operation, request="giant"):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_sums_of_squares_avoid_squaring_long_integer_forms(self):
        """Regression (wave M4 review): the checker squared every entry of the integer forms D_a a and D_b b, each as
        long as the common denominator D, which grows with the length n for distinct denominators, so its work was
        cubic in n: dot on 800 entries of 64-bit integers and rationals and projection on 400 (values of about 45,000
        bits, which the producer forms with about 57,000 and 130,000 units) ran out of the default budget, and so did
        any claim whose a holds many long denominators where b is 0. The polarization identity now runs over the
        coordinates where it can be nonzero, its squares (K / t)^2 are formed from K^2 when that is cheaper, and each
        projected entry is compared with the canonical pair of t b_j: measured about 163,000 and 113,000 units."""
        rng = random.Random(20261201)

        def integers(n: int) -> list[Fraction]:
            return [Fraction(rng.getrandbits(64) - 2 ** 63) for _ in range(n)]

        def rationals(n: int) -> list[Fraction]:
            return [Fraction(rng.getrandbits(64) - 2 ** 63, rng.getrandbits(64) | 1) for _ in range(n)]

        long_denominators = [Fraction(1, rng.getrandbits(60_000) | 1 << 59_999 | 1) for _ in range(30)]
        claims = [("dot", *true_dot(integers(800), rationals(800)), 400_000),
                  ("projection", *true_projection(integers(400), rationals(400)), 300_000),
                  # 30 coprime 60,000-bit denominators in a, where b is 0: their common denominator has 1.8 million
                  # bits, but no coordinate of the sums needs it.
                  ("dot", *true_dot(long_denominators + [3, 0], [0] * 30 + [Fraction(1, 7), Fraction(5, 2 ** 100)]),
                   5_000),
                  ("projection", *true_projection(long_denominators + [3, 0],
                                                  [0] * 30 + [Fraction(1, 7), Fraction(5, 2 ** 100)]), 5_000)]
        for operation, args, value, units in claims:
            with self.subTest(operation=operation, length=len(args["a"])):
                with computation(fresh=True) as budget:
                    start = time.perf_counter()
                    self.assertIs(checker.check_operation(operation, args, value, ()), True)
                    elapsed = time.perf_counter() - start
                self.assertLess(budget.work, units)
                self.assertLess(elapsed, FAST)
        self.assertGreater(claims[0][2].denominator.bit_length(), 40_000)  # Values of the size the producer forms.
        # The two routes to (K / t)^2 give the same integer: K^2 / t^2 for a short t, the square of K / t otherwise.
        common = 3 ** 40_000 * 2 ** 1_000 * 35
        with computation(fresh=True) as budget:
            for divisor in (1, 2, 35, 7 * 2 ** 999, 3 ** 20_000, 3 ** 39_999 * 2 ** 1_000 * 5, common):
                with self.subTest(divisor=divisor.bit_length()):
                    quotient = common // divisor
                    self.assertEqual(checker._Squares(common).of(budget, divisor), quotient * quotient)
        # The coordinates where a_i = 0 and b_i != 0 enter the projection's sums: wrong claims there are refused.
        args, value = true_projection((0, 1, 0), (1, 1, 2))  # (1/6, 1/6, 1/3)
        self.assertAccepts("projection", args, value)
        for wrong in ((0, 1, 0), (Fraction(1, 3), Fraction(1, 3), Fraction(2, 3)), (Fraction(1, 6), Fraction(1, 6), 0),
                      (0, 0, 0), (Fraction(1, 6), Fraction(1, 3), Fraction(1, 3))):
            with self.subTest(wrong=wrong):
                self.assertRejects("projection", args, values(wrong))

    def test_work_is_charged_before_it_runs(self):
        rng = random.Random(20261116)

        def vector(n: int) -> list[Fraction]:
            return [Fraction(rng.randrange(-10 ** 9, 10 ** 9), rng.randrange(1, 10 ** 3)) for _ in range(n)]

        claims = [("dot", *true_dot(vector(300), vector(300))), ("cross", *true_cross(vector(3), vector(3))),
                  ("projection", *true_projection(vector(300), vector(300))),
                  ("triple_product", *true_triple(vector(3), vector(3), vector(3)))]
        for operation, args, value in claims:
            with self.subTest(operation=operation):
                # The charges depend on the claim alone: a budget one unit short of the check refuses it, its exact
                # size passes.
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, ()), True)
                work = budget.work
                self.assertGreater(work, 3 * len(args["a"]))  # At least the reads and a product per entry.
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, value, limits=ComputationLimits(max_work=work))
        # A small budget stops a large claim at once.
        args, value = true_dot([rng.getrandbits(64) for _ in range(20_000)],
                               [rng.getrandbits(64) for _ in range(20_000)])
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "dot", args, value, limits=ComputationLimits(max_work=5_000))
        self.assertLess(time.perf_counter() - start, FAST)
        # Long integers cost more than short ones: the same shapes with 2,000-bit and 20,000-bit entries.
        charged = []
        for bits in (2_000, 20_000):
            entries = [[rng.getrandbits(bits) | 1 << (bits - 1) for _ in range(3)] for _ in range(3)]
            work = 0
            for operation, args, value in (("dot", *true_dot(entries[0], entries[1])),
                                           ("cross", *true_cross(entries[0], entries[1])),
                                           ("triple_product", *true_triple(*entries))):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, ()), True)
                work += budget.work
            charged.append(work)
        self.assertGreater(charged[1], 20 * charged[0])

    def test_no_power_is_formed_by_the_builtin(self):
        """The checker forms squares as products: no pow call and no ** at all, so nothing escapes the charges; it
        imports nothing but the allowed layers and the standard library."""
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertEqual(imported, {"__future__", "collections.abc", "math", "mathforge.limits", "mathforge.model",
                                    "mathforge.trusted.integers"})

    # -----------------------------------------------------------------------------------------------------------
    # The verifier

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip (an int and a Rational
        stay apart in the format), and a wrong value fails the mathematics check."""
        third = Fraction(1, 3)
        claims = [
            ("dot", *true_dot((1, 2, 3), (4, 5, 6)), Rational(33)),
            ("dot", *true_dot((third, 2), (3, Fraction(-1, 4))), Rational(1, 3)),  # The true value is 1/2.
            ("cross", *true_cross((1, 2, 3), (4, 5, third)), values((0, 0, 0))),
            ("projection", *true_projection((1, 2), (3, 4)), values((1, 2))),
            ("triple_product", *true_triple((1, 2, 3), (4, 5, 6), (7, 8, third)), Rational(0)),
        ]
        for operation, args, value, wrong in claims:
            with self.subTest(operation=operation, value=value):
                self.assertNotEqual(value, wrong)
                self.roundtrip(claimed(operation, args, value))
                report = support.statuses(mf.verify(claimed(operation, args, wrong)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
        # An int value where the family's value is a Rational fails too, also after the round trip.
        args, value = true_dot((1, 2, 3), (4, 5, 6))
        report = support.statuses(mf.verify(mf.loads(mf.dumps(claimed("dot", args, 32)))))
        self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
