"""The products family of mathforge.vector: dot, cross, projection, triple_product.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register vector products

Every expected value comes from the card or from a reference of this module, written here with fractions.Fraction and
independent of the producer: the sum of the products for dot, the Leibniz expansion of det[e_i, a, b] over every
permutation for the components of cross and of det[a, b, c] for triple_product, and (a.b / b.b) b for projection.
tests/v09/vector/test_products_checker.py tests the checker on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import itertools
from itertools import permutations
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.vector import products as family
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

from .. import support

FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
OPERATIONS = ("dot", "cross", "projection", "triple_product")
STAND_INS = (0.5, True, "1", None)  # No exact rational: a float, a bool, a str and None.

Number = int | Fraction | Rational

# One small request per operation.
SMALL = {
    "dot": dict(a=(1, 2, 3), b=(4, 5, 6)),
    "cross": dict(a=(1, 2, 3), b=(4, 5, 6)),
    "projection": dict(a=(1, 2), b=(3, 4)),
    "triple_product": dict(a=(1, 2, 3), b=(4, 5, 6), c=(7, 8, 10)),
}

HALF, THIRD = Rational(1, 2), Rational(1, 3)

# The known values of the card and of its examples: (operation, arguments, value).
KNOWN = (
    ("dot", dict(a=(1, 2, 3), b=(4, 5, 6)), 32),
    ("dot", dict(a=(1, 0), b=(0, 1)), 0),
    ("dot", dict(a=(-2,), b=(5,)), -10),
    ("dot", dict(a=(HALF, THIRD), b=(3, 4)), Fraction(17, 6)),
    ("cross", dict(a=(1, 0, 0), b=(0, 1, 0)), (0, 0, 1)),
    ("cross", dict(a=(1, 2, 3), b=(4, 5, 6)), (-3, 6, -3)),
    ("cross", dict(a=(1, 2, 3), b=(2, 4, 6)), (0, 0, 0)),  # Parallel vectors.
    ("cross", dict(a=(HALF, 0, 0), b=(0, THIRD, 0)), (0, 0, Fraction(1, 6))),
    ("projection", dict(a=(1, 2), b=(3, 4)), (Fraction(33, 25), Fraction(44, 25))),
    ("projection", dict(a=(2, 0), b=(1, 0)), (2, 0)),
    ("projection", dict(a=(0, 1), b=(1, 0)), (0, 0)),
    ("projection", dict(a=(3, 4, 5), b=(0, 0, 2)), (0, 0, 5)),
    ("projection", dict(a=(7,), b=(-2,)), (7,)),
    ("triple_product", dict(a=(1, 0, 0), b=(0, 1, 0), c=(0, 0, 1)), 1),
    ("triple_product", dict(a=(1, 0, 0), b=(0, 0, 1), c=(0, 1, 0)), -1),
    ("triple_product", dict(a=(1, 2, 3), b=(4, 5, 6), c=(7, 8, 9)), 0),
    ("triple_product", dict(a=(1, 2, 3), b=(4, 5, 6), c=(7, 8, 10)), -3),
)

# Inputs outside the domains of the card, each invalid input.
INVALID = (
    ("dot", dict(a=(1, 2), b=(1, 2, 3))),
    ("dot", dict(a=(1, 2, 3), b=(1, 2))),
    ("dot", dict(a=(), b=())),
    ("dot", dict(a=(), b=(1,))),
    ("dot", dict(a=(1,), b=())),
    ("cross", dict(a=(1, 0), b=(0, 1))),
    ("cross", dict(a=(1, 0, 0, 0), b=(0, 1, 0, 0))),
    ("cross", dict(a=(1, 0, 0), b=(0, 1))),
    ("cross", dict(a=(1, 0), b=(0, 1, 0))),
    ("cross", dict(a=(), b=())),
    ("cross", dict(a=(1,), b=(2,))),
    ("projection", dict(a=(1, 2), b=(0, 0))),
    ("projection", dict(a=(5,), b=(0,))),
    ("projection", dict(a=(1, 2), b=(Rational(0), Rational(0)))),
    ("projection", dict(a=(0, 0, 0), b=(0, 0, 0))),
    ("projection", dict(a=(), b=())),
    ("projection", dict(a=(1, 2), b=(3, 4, 5))),
    ("projection", dict(a=(1, 2, 0), b=(3, 4))),
    ("triple_product", dict(a=(1, 0), b=(0, 1), c=(1, 1))),
    ("triple_product", dict(a=(1, 0, 0, 0), b=(0, 1, 0, 0), c=(0, 0, 1, 0))),
    ("triple_product", dict(a=(1, 0, 0), b=(0, 1, 0), c=(0, 0))),
    ("triple_product", dict(a=(1, 0, 0), b=(0, 1), c=(0, 0, 1))),
    ("triple_product", dict(a=(), b=(), c=())),
)


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

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


def expected(value: Number | Sequence[Number]) -> Rational | tuple[Rational, ...]:
    """A value of the family from a number or a vector of the tests."""
    return tuple(rational(x) for x in value) if type(value) is tuple else rational(value)


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


def ref_dot(a: Sequence[Number], b: Sequence[Number]) -> Fraction:
    """a.b, the sum of the products of the entries."""
    return sum((frac(x) * frac(y) for x, y in zip(a, b, strict=True)), Fraction(0))


def ref_cross(a: Sequence[Number], b: Sequence[Number]) -> tuple[Fraction, ...]:
    """a x b by (a x b)_i = e_i.(a x b) = det[e_i, a, b], each determinant by the Leibniz formula."""
    unit = [[int(i == j) for j in range(3)] for i in range(3)]
    return tuple(leibniz([unit[i], a, b]) for i in range(3))


def ref_projection(a: Sequence[Number], b: Sequence[Number]) -> tuple[Fraction, ...]:
    """(a.b / b.b) b, for b != 0."""
    t = ref_dot(a, b) / ref_dot(b, b)
    return tuple(t * frac(y) for y in b)


def ref_triple(a: Sequence[Number], b: Sequence[Number], c: Sequence[Number]) -> Fraction:
    """det[a, b, c] by the Leibniz formula."""
    return leibniz([a, b, c])


def reference(operation: str, args: dict[str, Sequence[Number]]) -> Fraction | tuple[Fraction, ...]:
    """The value of a request by the references."""
    if operation == "dot":
        return ref_dot(args["a"], args["b"])
    if operation == "cross":
        return ref_cross(args["a"], args["b"])
    if operation == "projection":
        return ref_projection(args["a"], args["b"])
    return ref_triple(args["a"], args["b"], args["c"])


class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Fractional(Rational):
    """A Rational subclass, equal to the right number but no exact Rational."""


class Entries(tuple):
    """A tuple subclass for a vector or a value."""


class ProductsTests(support.FamilyTestCase):
    FAMILY = family

    def assertCanonical(self, value: object) -> None:
        """value is a Rational of the exact class, or a plain tuple of them; never an int."""
        if type(value) is tuple:
            self.assertTrue(all(type(item) is Rational for item in value), value)
        else:
            self.assertIs(type(value), Rational, value)

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with evidence () and its four checks, which mf.verify accepts."""
        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.value, value)
        self.assertCanonical(result.value)
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertAgrees(self, operation: str, args: dict[str, object], truth: object) -> None:
        """produce() answers the reference's value as canonical Rationals, with the evidence (), and the checker
        accepts it."""
        value, evidence = family.produce(operation, dict(args))
        self.assertEqual(value, expected(truth))
        self.assertCanonical(value)
        self.assertEqual(evidence, ())
        self.assertIs(support.verdict(family, operation, args, value, evidence), True)

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, args, value in KNOWN:
            with self.subTest(operation=operation, **args):
                self.assertEqual(reference(operation, args), value if type(value) is not tuple
                                 else tuple(Fraction(x) for x in value))  # The card's value, by the references.
                result = support.run(operation, **args)
                self.assertVerified(result, expected(value))
                self.roundtrip(result)
        # Ints and Rationals of denominator 1 are the same numbers, and a direct call takes lists as well as tuples.
        mixed = support.run("dot", a=(Rational(1), 2, Rational(3)), b=[4, Rational(5), 6])
        self.assertVerified(mixed, Rational(32))
        self.assertEqual(family.produce("dot", {"a": [1, 2, 3], "b": [4, 5, 6]}), (Rational(32), ()))
        self.assertEqual(family.produce("cross", {"a": [1, 2, 3], "b": (4, 5, 6)}),
                         (expected((-3, 6, -3)), ()))
        # Rational entries of all four operations, and a projection whose b has leading zeros.
        third, seventh = Fraction(1, 3), Fraction(1, 7)
        cases = (("dot", dict(a=(third, -2, seventh), b=(3, Fraction(-1, 4), 7))),
                 ("cross", dict(a=(Fraction(1, 2), -third, 2), b=(seventh, 0, Fraction(-9, 4)))),
                 ("projection", dict(a=(1, 2, 3, 4), b=(0, 0, Fraction(1, 2), -1))),
                 ("triple_product", dict(a=(Fraction(1, 2), -third, 2), b=(seventh, 0, Fraction(-9, 4)),
                                         c=(1, Fraction(5, 6), -seventh))))
        for operation, numbers in cases:
            args = {name: tuple(exact(x) for x in vector) for name, vector in numbers.items()}
            with self.subTest(operation=operation, rational=True):
                self.assertVerified(support.run(operation, **args), expected(reference(operation, numbers)))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument and in every entry; so is every input outside the domains of the card. produce() tests the
        types itself too."""
        containers = (Rational(1), 7, "123", {1, 2, 3}, {"x": 1}, ((1, 2), (3, 4)), mf.Matrix(((1, 2, 3),)),
                      support.oversized_matrix())  # No argument of the family is a matrix: a Matrix is no vector.
        for operation, args in SMALL.items():
            for name, vector in args.items():
                for item in STAND_INS + containers:
                    with self.subTest(operation=operation, argument=name, item=type(item).__name__):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: item})))
                for item in STAND_INS + (Fraction(1, 2), (1,), [1]):
                    with self.subTest(operation=operation, argument=name, entry=item):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: (item,) + vector[1:]})))
        for operation, args in INVALID:
            with self.subTest(operation=operation, **args):
                self.assertInvalidInput(support.run(operation, **args))
        messages = (("dot", dict(a=(1, 2), b=(1, 2, 3)), "dot requires a and b of the same length"),
                    ("dot", dict(a=(), b=()), "dot requires nonempty vectors a and b"),
                    ("cross", dict(a=(1, 0), b=(0, 1)), "cross requires vectors of length 3"),
                    ("projection", dict(a=(1, 2), b=(0, 0)), "projection requires a nonzero vector b"),
                    ("triple_product", dict(a=(1, 0, 0), b=(0, 1, 0), c=(0, 0)),
                     "triple_product requires vectors of length 3"))
        for operation, args, text in messages:
            with self.subTest(operation=operation, message=text):
                self.assertIn(text, support.run(operation, **args).error.message)
        # Rationals forged with object.__new__ outside the canonical form: the checker refuses a request that holds
        # one, so the producer refuses it as input, through run() and directly.
        forgeries = [support.forged(Rational, numerator=2, denominator=4),
                     support.forged(Rational, numerator=-1, denominator=-2),
                     support.forged(Rational, numerator=1, denominator=0),
                     support.forged(Rational, numerator=0, denominator=5),
                     support.forged(Rational, numerator=True, denominator=2),
                     support.forged(Rational, numerator=1, denominator=2.0),
                     support.forged(Rational, numerator=1),  # Without a denominator.
                     support.forged(Rational, denominator=2),  # Without a numerator.
                     Fractional(1, 2), Integer(3), Fraction(1, 2), 0.5, True, False, "1", None, support.impostor(HALF)]
        for operation, args in SMALL.items():
            for name, vector in args.items():
                for index, item in enumerate(forgeries):
                    changed = dict(args, **{name: (item,) + vector[1:]})
                    with self.subTest(operation=operation, argument=name, forgery=index, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, changed)
                    if index < 5:  # Slots of ints (a bool is one), so that run()'s inspection reaches the producer.
                        with self.subTest(operation=operation, argument=name, forgery=index, path="run"):
                            self.assertInvalidInput(support.run(operation, **changed))
                for item in (frozenset((1, 2)), Entries(vector), iter(vector), Rational(1), None, "12"):
                    with self.subTest(operation=operation, argument=name, container=type(item).__name__):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: item}))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(args, extra=(1,)))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, {name: vector for name, vector in list(args.items())[1:]})
                with self.assertRaises(InvalidInput):
                    family.produce(operation, tuple(args.items()))
        for operation, args in INVALID:
            with self.subTest(operation=operation, path="produce", **args), self.assertRaises(InvalidInput):
                family.produce(operation, dict(args))
        for operation in ("Dot", "inner_product", "continued_fraction", "", None, 7):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(a=(1, 2, 3), b=(4, 5, 6)))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("projection", a=(1, 2), b=(0, 0))
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in any entry, a value longer than max_integer_bits and max_work=1 exceed the budget.
        No operation of the family takes a matrix, so support.oversized_matrix() has nothing to exceed here (as a
        vector it is invalid input, test_invalid_inputs). Work is charged before it runs, so a request that needs
        more than its budget fails at once, and a value that the checker cannot afford within the request's limits
        is reported as resource_limit_exceeded, never as a value."""
        for operation, args in SMALL.items():
            for name, vector in args.items():
                for item in (support.HUGE, -support.HUGE, Rational(1, support.HUGE), Rational(support.HUGE, 3)):
                    with self.subTest(operation=operation, argument=name, item=item.bit_length()
                                      if type(item) is int else "rational"):
                        self.assertResourceLimit(support.run(operation, **dict(args, **{name: (item,) + vector[1:]})))
                with self.subTest(operation=operation, argument=name, path="produce"), self.assertRaises(
                        ResourceLimitError):
                    family.produce(operation, dict(args, **{name: (support.HUGE,) + vector[1:]}))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
        # Every entry read costs a unit and every term at least one more, also when the arithmetic is trivial.
        for n in (1, 10, 1000):
            with self.subTest(zeros=n), computation(fresh=True) as budget:
                self.assertEqual(family.produce("dot", {"a": (0,) * n, "b": (0,) * n}), (Rational(0), ()))
                self.assertGreaterEqual(budget.work, 3 * n)
        # Values longer than max_integer_bits: no computation under the budget may form them.
        big = 2 ** 40_000
        for operation, args in (("dot", dict(a=(big,), b=(big,))),
                                ("cross", dict(a=(big, 0, 0), b=(0, big, 0))),
                                ("projection", dict(a=(big, 0), b=(Rational(1, big), 1))),
                                ("triple_product", dict(a=(big, 0, 0), b=(0, big, 0), c=(0, 0, 2)))):
            with self.subTest(operation=operation, value="too long"):
                self.assertResourceLimit(support.run(operation, **args))
                with self.assertRaises(ResourceLimitError), computation(fresh=True):
                    family.produce(operation, args)
        # Charged before it runs: three products of 60,000 bits within 1,000 units, and 20,000 long entries within
        # 5,000 units, fail at once.
        rng = random.Random(20261130)
        huge = tuple(rng.getrandbits(60_000) | 1 << 59_999 for _ in range(3))
        many = tuple(rng.getrandbits(200) for _ in range(20_000))
        start = time.perf_counter()
        self.assertResourceLimit(support.run("cross", a=huge, b=huge[::-1], limits=ComputationLimits(max_work=1000)))
        self.assertResourceLimit(support.run("dot", a=many, b=many, limits=ComputationLimits(max_work=5000)))
        with computation(ComputationLimits(max_work=5000), fresh=True), self.assertRaises(ResourceLimitError):
            family.produce("dot", {"a": many, "b": many})
        self.assertLess(time.perf_counter() - start, FAST)
        # A claim that the checker cannot afford within the request's limits: the producer forms this cross product
        # with about 1,500 units, and the checker needs about 9,300.
        a = tuple(Rational(2 ** 3000 + i, 3 ** 1900 + 2 * i) for i in range(3))
        b = tuple(Rational(5 ** 1300 + i, 7 ** 1000 + i) for i in range(3))
        with computation(ComputationLimits(max_work=5000), fresh=True):
            family.produce("cross", {"a": a, "b": b})
        result = support.run("cross", a=a, b=b, limits=ComputationLimits(max_work=5000))
        self.assertResourceLimit(result)
        self.assertEqual(result.error.message, "Independent verification exceeded its budget")
        self.assertVerified(support.run("cross", a=a, b=b), expected(ref_cross(a, b)))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim built here from the references, never by the producer: on requests
        that hold Rationals of denominator 1 where the producer's own callers hold ints, long vectors and long
        integers. The family's evidence is always () and each value has exactly one canonical form, so no claim can
        differ from the producer's in its evidence: these are the claims the producer was not asked for."""
        rng = random.Random(20261124)
        claims = []
        for operation, args in SMALL.items():
            truth = expected(reference(operation, args))
            as_rationals = {name: tuple(Rational(x) for x in vector) for name, vector in args.items()}
            claims += [(operation, args, truth), (operation, as_rationals, truth)]
        long_a = tuple(Rational(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 50)) for _ in range(400))
        long_b = tuple(rng.randrange(-10 ** 6, 10 ** 6) for _ in range(400))
        claims += [("dot", dict(a=long_a, b=long_b), expected(ref_dot(long_a, long_b))),
                   ("projection", dict(a=long_a, b=long_b), expected(ref_projection(long_a, long_b)))]
        long_integer = 2 ** 300 + 7
        triple = dict(a=(long_integer, 0, 1), b=(0, long_integer, 0), c=(1, 0, Rational(1, long_integer)))
        claims += [("triple_product", triple, expected(ref_triple(**triple))),
                   ("cross", dict(a=triple["a"], b=triple["c"]), expected(ref_cross(triple["a"], triple["c"])))]
        for operation, args, value in claims:
            with self.subTest(operation=operation, length=len(args["a"])):
                self.assertAccepts(operation, args, value)
                self.assertEqual(family.produce(operation, dict(args)), (value, ()))  # The claim is the true one.

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a Rational
        forged with object.__new__, a bool, a float or a str where a Rational belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        wrong = {
            "dot": [Rational(31), Rational(33), Rational(-32), Rational(0), Rational(64), Rational(32, 7)],
            "cross": [expected((3, -6, 3)), expected((-3, 6, -2)), expected((0, 0, 0)), expected((-6, 12, -6)),
                      expected((6, -3, -3)), expected((-3, 6)), expected((-3, 6, -3, 0))],
            "projection": [expected((Fraction(66, 25), Fraction(88, 25))), expected((Fraction(44, 25),
                                                                                       Fraction(33, 25))),
                           expected((1, 2)), expected((0, 0)), expected((Fraction(33, 25),)), Rational(11, 25)],
            "triple_product": [Rational(3), Rational(-2), Rational(0), Rational(-3, 2), expected((-3,))],
        }
        for operation, args in SMALL.items():
            result = support.run(operation, **args)
            value = result.value
            self.assertVerified(result, expected(reference(operation, args)))
            self.assertAccepts(operation, args, value)
            cases = [(f"wrong {item!r}", item, ()) for item in wrong[operation]]
            cases += [("evidence None", value, None), ("evidence as a list", value, []),
                      ("an extra evidence item", value, ((),)), ("a number as evidence", value, (0,)),
                      ("oversized evidence", value, (0,) * 100_000), ("giant evidence", value, (GIANT,))]
            numbers = value if type(value) is tuple else (value,)
            for index, number in enumerate(numbers):
                stand_ins = [("an int", number.numerator), ("a float", number.numerator / number.denominator),
                             ("a str", str(number)), ("a bool", bool(number.numerator)), ("None", None),
                             ("a Fraction", Fraction(number.numerator, number.denominator)),
                             ("a Rational subclass", Fractional(number.numerator, number.denominator)),
                             ("an impostor", support.impostor(number)),
                             ("a Rational in higher terms",
                              support.forged(Rational, numerator=2 * number.numerator,
                                             denominator=2 * number.denominator)),
                             ("a giant forged Rational", support.forged(Rational, numerator=GIANT, denominator=1))]
                for label, item in stand_ins:
                    corrupted = numbers[:index] + (item,) + numbers[index + 1:] if type(value) is tuple else item
                    cases.append((f"entry {index}: {label}", corrupted, ()))
            if type(value) is tuple:
                cases += [("a list", list(value), ()), ("a tuple subclass", Entries(value), ())]
            else:
                cases += [("a one-entry tuple", (value,), ())]
            for label, item, evidence in cases:
                with self.subTest(operation=operation, case=label):
                    self.assertRejects(operation, args, item, evidence)
            for other in OPERATIONS + ("Dot", "inner_product", ""):
                if other != operation:
                    with self.subTest(operation=operation, renamed=other):
                        self.assertRejects(other, args, value)
            report = support.statuses(mf.verify(support.tampered(result, wrong[operation][0])))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [(operation, args) for operation, args, _ in KNOWN] + [
            ("projection", dict(a=(Rational(-5, 3), 2), b=(Rational(2, 7), Rational(1, 2)))),
            ("dot", dict(a=tuple(range(500)), b=tuple(Rational(1, k + 1) for k in range(500))))]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, _), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("dot", a=(1, 2), b=(3, 4))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: an integral
        Rational, a fraction, negative values, a vector of three, a vector of fractions, the zero vector and a
        vector of one entry."""
        rendered = (
            ("dot", SMALL["dot"], "32", "32"),
            ("dot", dict(a=(HALF, THIRD), b=(3, 4)), "17/6", r"\frac{17}{6}"),
            ("dot", dict(a=(HALF, THIRD), b=(-3, -4)), "-17/6", r"-\frac{17}{6}"),
            ("triple_product", SMALL["triple_product"], "-3", "-3"),
            ("cross", SMALL["cross"], "(-3, 6, -3)", r"\left(-3, 6, -3\right)"),
            ("cross", dict(a=(HALF, 2, -1), b=(3, Rational(-1, 7), 4)), "(55/7, -5, -85/14)",
             r"\left(\frac{55}{7}, -5, -\frac{85}{14}\right)"),
            ("cross", dict(a=(1, 2, 3), b=(2, 4, 6)), "(0, 0, 0)", r"\left(0, 0, 0\right)"),
            ("projection", SMALL["projection"], "(33/25, 44/25)", r"\left(\frac{33}{25}, \frac{44}{25}\right)"),
            ("projection", dict(a=(7,), b=(-2,)), "(7)", r"\left(7\right)"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(operation=operation, **args):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("dot", **SMALL["dot"]).value)["data"],
                         {"type": "rational", "numerator": "32", "denominator": "1"})
        self.assertEqual(mf.to_data(support.run("projection", a=(7,), b=(-2,)).value)["data"],
                         {"type": "tuple", "items": [{"type": "rational", "numerator": "7", "denominator": "1"}]})
        # An int and a Rational stay apart in the format: the value of the decoded Result is still a Rational.
        decoded = mf.loads(mf.dumps(support.run("triple_product", **SMALL["triple_product"])))
        self.assertIs(type(decoded.value), Rational)

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261125)

        def vector(n: int) -> tuple[int | Rational, ...]:
            return tuple(exact(Fraction(rng.randrange(-10 ** 6, 10 ** 6), rng.choice((1, 1, 2, 3, 10 ** 9))))
                         for _ in range(n))

        cases = []
        for _ in range(12):
            n = rng.randrange(1, 20)
            cases += [("dot", dict(a=vector(n), b=vector(n))), ("cross", dict(a=vector(3), b=vector(3))),
                      ("projection", dict(a=vector(n), b=vector(n))),
                      ("triple_product", dict(a=vector(3), b=vector(3), c=vector(3)))]
        cases.append(("dot", dict(a=[1, 2, 3], b=[4, 5, 6])))  # Lists, as a caller may pass them.
        for operation, args in cases:
            with self.subTest(operation=operation, **args):
                value = support.run(operation, **args).value
                self.assertIsNotNone(value)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, value)
                self.assertCanonical(found)
                self.assertEqual(support.value(operation, **args), value)
                self.assertEqual(value, expected(reference(operation, args)))

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        calls = {
            "dot": ({"a": [1, "1/2", "-3"], "b": ["2/3", 4, 5]}, dict(a=(1, HALF, -3), b=(Rational(2, 3), 4, 5))),
            "cross": ({"a": [1, 2, 3], "b": [4, 5, "6/7"]}, dict(a=(1, 2, 3), b=(4, 5, Rational(6, 7)))),
            "projection": ({"a": [1, 2], "b": ["3/1", 4]}, dict(a=(1, 2), b=(Rational(3), 4))),
            "triple_product": ({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 10]}, SMALL["triple_product"]),
        }
        for operation, (payload, args) in calls.items():
            with self.subTest(operation=operation):
                result = self.answer(support.call(operation, payload))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
                self.assertEqual(result.value, expected(reference(operation, args)))
        # Python values through their kinds: integers from 2**53 on travel as decimal strings, and Rational(5) as "5/1",
        # which decodes to a Rational again.
        args = dict(a=(2 ** 80 + 1, Rational(5), -HALF), b=(3, 2 ** 70, Rational(-7, 9)))
        for operation in ("dot", "cross", "projection"):
            with self.subTest(operation=operation, values=True):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("dot", {"a": [1, 2], "b": [3]}), "invalid_input")
        self.assertIn("dot requires a and b of the same length", error["message"])
        error = self.assertToolError(support.call("projection", {"a": [1, 2], "b": [0, "0/5"]}), "invalid_input")
        self.assertIn("nonzero vector b", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 200 seeded inputs per operation, with
        zeros, integers, fractions of many sizes, parallel and orthogonal vectors, and the checker accepts every
        answer with its evidence ()."""
        rng = random.Random(20261126)
        counts = dict.fromkeys(OPERATIONS, 0)

        def number(bits: int) -> Fraction:
            draw = rng.random()
            if draw < 0.15:
                return Fraction(0)
            numerator = rng.randrange(-(2 ** bits), 2 ** bits + 1)
            if draw < 0.5:
                return Fraction(numerator)
            return Fraction(numerator, rng.choice((rng.randrange(1, 2 ** bits + 1), 2 ** rng.randrange(1, 12), 6, 35)))

        def vector(n: int, bits: int) -> list[Fraction]:
            return [number(bits) for _ in range(n)]

        def request(*vectors: list[Fraction]) -> dict[str, tuple[int | Rational, ...]]:
            return {name: tuple(exact(x) for x in numbers) for name, numbers in zip(("a", "b", "c"), vectors)}

        refused = 0
        for case in range(260):
            n, bits = rng.randrange(1, 10), rng.choice((2, 5, 16, 64, 200))
            a, b = vector(n, bits), vector(n, bits)
            shape = rng.random()
            if shape < 0.15:  # Parallel vectors.
                b = [number(3) * x for x in a]
            elif shape < 0.25 and n > 1:  # b orthogonal to a in its first two entries.
                a[2:], b[2:] = [Fraction(0)] * (n - 2), [Fraction(0)] * (n - 2)
                b[0], b[1] = -a[1], a[0]
            with self.subTest(operation="dot", case=case):
                self.assertAgrees("dot", request(a, b), ref_dot(a, b))
                counts["dot"] += 1
            if not any(b):
                with self.subTest(operation="projection", case=case, b="zero"), self.assertRaises(InvalidInput):
                    family.produce("projection", request(a, b))
                refused += 1
                continue
            with self.subTest(operation="projection", case=case):
                self.assertAgrees("projection", request(a, b), ref_projection(a, b))
                counts["projection"] += 1
        for case in range(240):
            bits = rng.choice((2, 5, 16, 64, 200))
            a, b, c = vector(3, bits), vector(3, bits), vector(3, bits)
            shape = rng.random()
            if shape < 0.15:  # Parallel vectors: a zero cross product.
                b = [number(3) * x for x in a]
            elif shape < 0.25:  # c in the plane of a and b: a zero triple product.
                c = [number(3) * x + number(3) * y for x, y in zip(a, b)]
            with self.subTest(operation="cross", case=case):
                truth = ref_cross(a, b)
                self.assertEqual(ref_dot(truth, a), 0)  # The reference's own properties: orthogonal, ...
                self.assertEqual(ref_dot(truth, b), 0)
                self.assertEqual(ref_triple(a, b, truth), ref_dot(truth, truth))  # ... positively oriented.
                self.assertAgrees("cross", request(a, b), truth)
                counts["cross"] += 1
            with self.subTest(operation="triple_product", case=case):
                truth = ref_triple(a, b, c)
                self.assertEqual(truth, ref_dot(a, ref_cross(b, c)))
                self.assertAgrees("triple_product", request(a, b, c), truth)
                counts["triple_product"] += 1
        self.assertTrue(all(count >= 200 for count in counts.values()), counts)
        self.assertLess(refused, 60)

    def test_full_size_inputs_fit_the_default_budget(self):
        """Requests on long integers and long vectors complete and verify under the default limits: the sizes at
        which test_products_checker.py verifies its claims, as far as a request's node limits admit."""
        rng = random.Random(20261127)

        def big(bits: int) -> int:
            return rng.choice((-1, 1)) * (rng.getrandbits(bits) | 1 << (bits - 1))

        def bigs(bits: int, n: int = 3) -> tuple[int, ...]:
            return tuple(big(bits) for _ in range(n))

        def small_rational() -> Rational | int:
            return exact(Fraction(rng.getrandbits(64) - 2 ** 63, 2 ** rng.randrange(0, 8)))

        requests = [
            ("dot", dict(a=bigs(32_000), b=bigs(32_000)), 10_000),
            ("cross", dict(a=bigs(32_000), b=bigs(32_000)), 20_000),
            ("projection", dict(a=bigs(20_000), b=bigs(20_000)), 60_000),
            ("triple_product", dict(a=bigs(21_000), b=bigs(21_000), c=bigs(21_000)), 20_000),
            # 20,000 such entries exceed a request's max_nodes: the checker's test calls the checker directly.
            ("dot", dict(a=tuple(rng.getrandbits(64) for _ in range(10_000)),
                         b=tuple(small_rational() for _ in range(10_000))), 200_000),
            ("projection", dict(a=tuple(rng.getrandbits(32) for _ in range(3_000)),
                                b=tuple(small_rational() for _ in range(3_000))), 150_000),
        ]
        for operation, args, units in requests:
            with self.subTest(operation=operation, length=len(args["a"])):
                with computation(fresh=True) as budget:
                    value, _ = family.produce(operation, args)
                self.assertLess(budget.work, units)
                self.assertEqual(value, expected(reference(operation, args)))
                self.assertVerified(support.run(operation, **args), value)
        cross = support.run("cross", **requests[1][1]).value
        self.assertGreater(cross[0].numerator.bit_length(), 60_000)  # The values are of the full size.

    def test_sums_of_products_need_no_more_work_than_the_check(self):
        """Regression: a sum of products reduced to lowest terms after every term paid two gcds of the long
        denominators per term, and for vectors over one shared 10,000-bit denominator it needed four times the
        checker's work; over one common denominator of the terms it needs about as much (measured: 0.90 of the
        checker's work for dot, 1.01 for projection, after the review that made the checker's sums of squares cheaper).
        The bound leaves room for the checker's charges to change a little."""
        rng = random.Random(20261128)
        bits = 10_000
        shared = rng.getrandbits(bits) | 1 << (bits - 1) | 1
        a = tuple(Rational(rng.getrandbits(bits - 1), shared) for _ in range(120))
        b = tuple(Rational(rng.getrandbits(bits - 1), shared) for _ in range(120))
        for operation in ("dot", "projection"):
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    value, evidence = family.produce(operation, {"a": a, "b": b})
                with computation(fresh=True) as checking:
                    self.assertIs(support.checker(family).check_operation(operation, {"a": a, "b": b}, value,
                                                                          evidence), True)
                self.assertLess(budget.work, checking.work * 5 // 4)
        self.assertEqual(family.produce("dot", {"a": a, "b": b})[0], rational(ref_dot(a, b)))

    def test_long_rational_vectors_complete_through_run(self):
        """Regression (wave M4 review): run() reported resource_limit_exceeded ("Independent verification exceeded
        its budget") for dot on 800 entries of 64-bit integers and rationals and projection on 400, values of about
        45,000 bits that produce() forms with about 57,000 and 130,000 units, since the checker's work was cubic in the
        length; it now verifies them, as it does a request whose a holds long denominators only where b is 0."""
        rng = random.Random(20261202)

        def integers(n: int) -> tuple[int, ...]:
            return tuple(rng.getrandbits(64) - 2 ** 63 for _ in range(n))

        def rationals(n: int) -> tuple[int | Rational, ...]:
            return tuple(exact(Fraction(rng.getrandbits(64) - 2 ** 63, rng.getrandbits(64) | 1)) for _ in range(n))

        long_denominators = tuple(Rational(1, rng.getrandbits(60_000) | 1 << 59_999 | 1) for _ in range(30))
        requests = [("dot", dict(a=integers(800), b=rationals(800))),
                    ("projection", dict(a=integers(400), b=rationals(400))),
                    ("dot", dict(a=long_denominators + (3, 0), b=(0,) * 30 + (Rational(1, 7), Rational(5, 2 ** 100)))),
                    ("projection", dict(a=long_denominators + (3, 0),
                                        b=(0,) * 30 + (Rational(1, 7), Rational(5, 2 ** 100))))]
        for operation, args in requests:
            with self.subTest(operation=operation, length=len(args["a"])):
                self.assertVerified(support.run(operation, **args), expected(reference(operation, args)))

    def test_cost_helpers_restate_trusted_integers(self):
        """The family's cost helpers restate product_work and division_work of trusted.integers (the module states
        why) and add a unit per 64 words of a product's result."""
        sizes = (0, 1, 63, 64, 65, 127, 128, 500, 4_095, 4_096, 4_097, 20_000, 65_536, 131_072, 200_000)
        for left, right in itertools.product(sizes, repeat=2):
            with self.subTest(left=left, right=right):
                self.assertEqual(family._product_work(left, right), product_work(left, right) + ((left + right) >> 12))
                self.assertEqual(family._quotient_work(left, right), division_work(max(left - right, 0) + 64, right))
