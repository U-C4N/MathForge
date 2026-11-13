"""The order family of mathforge.stats: quantile, mode.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register stats order

Every expected value comes from the card or from the references of this module, written here and slow on purpose, never
from the producer and without any sort: an order statistic x_(k) is the item v with #{x < v} <= k < #{x <= v}, found
by counting against every item (quadratic), and the most frequent values by list.count over every distinct value. The
family has no deferred case: every valid input has a value, and every other input is invalid input.
tests/v09/stats/test_order_checker.py tests the checker on its own.

Run by module name: python -m unittest tests.v09.stats.test_order
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
import random
import time
from typing import Any
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.stats import order as family
from mathforge.contracts import CheckStatus, ErrorInfo, Outcome
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational

from .. import support

FAST = 1.0  # Seconds: a request that its budget or the domain refuses fails well within this.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
OPERATIONS = ("quantile", "mode")
STAND_INS = (0.5, True, "1", None)  # No exact rational: a float, a bool, a str and None.

Number = int | Fraction | Rational

# One small request per operation.
SMALL = {
    "quantile": dict(data=(3, 1, 4, 1, 5), p=Rational(1, 3)),
    "mode": dict(data=(3, 1, 4, 1, 5)),
}

# The known values of the card and special cases: (operation, arguments, value, evidence).
KNOWN = (
    ("quantile", dict(data=(1, 2, 3, 4), p=Rational(1, 2)), Rational(5, 2), ((1, 2), (2, 3))),
    ("quantile", dict(data=(1, 2, 3, 4), p=Rational(1, 4)), Rational(7, 4), ((0, 1), (1, 2))),
    ("quantile", dict(data=(5,), p=Rational(1, 3)), Rational(5), ((0, 5),)),
    ("quantile", dict(data=(3, 1, 2), p=0), Rational(1), ((0, 1),)),
    ("quantile", dict(data=(3, 1, 2), p=1), Rational(3), ((2, 3),)),
    # Special cases: the median of an odd sample, a repeated order statistic, fractions, negative data, p = 1 as a
    # Rational of denominator 1, and h integral although p is not (n - 1 = 4, p = 3/4).
    ("quantile", dict(data=(5, 1, 3), p=Rational(1, 2)), Rational(3), ((1, 3),)),
    ("quantile", dict(data=(2, 2, 2, 7), p=Rational(1, 2)), Rational(2), ((1, 2), (2, 2))),
    ("quantile", dict(data=(Rational(1, 2), Rational(1, 3), 2), p=Rational(1, 3)), Rational(4, 9),
     ((0, Rational(1, 3)), (1, Rational(1, 2)))),
    ("quantile", dict(data=(-4, 0, -1, 9), p=Rational(5, 6)), Rational(9, 2), ((2, 0), (3, 9))),
    ("quantile", dict(data=(4, 8, 6), p=Rational(1)), Rational(8), ((2, 8),)),
    ("quantile", dict(data=(10, 20, 30, 40, 50), p=Rational(3, 4)), Rational(40), ((3, 40),)),
    ("quantile", dict(data=(1, 2), p=Rational(1, 7)), Rational(8, 7), ((0, 1), (1, 2))),
    ("mode", dict(data=(1, 2, 2, 3)), (Rational(2),), (2,)),
    ("mode", dict(data=(1, 1, 2, 2, 3)), (Rational(1), Rational(2)), (2,)),
    ("mode", dict(data=(1, 2, 3)), (Rational(1), Rational(2), Rational(3)), (1,)),
    ("mode", dict(data=(Rational(1, 2), Rational(2, 4), 1)), (Rational(1, 2),), (2,)),
    # Special cases: one item, ints and Rationals of denominator 1 as one value, negative fractions ascending.
    ("mode", dict(data=(7,)), (Rational(7),), (1,)),
    ("mode", dict(data=(2, Rational(2), Rational(4, 2), 3, 3)), (Rational(2),), (3,)),
    ("mode", dict(data=(Rational(-1, 3), 5, Rational(-1, 2), Rational(-1, 3), Rational(-1, 2))),
     (Rational(-1, 2), Rational(-1, 3)), (2,)),
)

# Inputs outside the domains of the card, each invalid input: (operation, arguments, a part of the message).
INVALID = (
    ("quantile", dict(data=(), p=Rational(1, 2)), "nonempty data"),
    ("quantile", dict(data=[], p=0), "nonempty data"),
    ("quantile", dict(data=(1, 2, 3), p=-1), "0 <= p <= 1"),
    ("quantile", dict(data=(1, 2, 3), p=2), "0 <= p <= 1"),
    ("quantile", dict(data=(1, 2, 3), p=Rational(3, 2)), "0 <= p <= 1"),
    ("quantile", dict(data=(1, 2, 3), p=Rational(-1, 2)), "0 <= p <= 1"),
    ("quantile", dict(data=(1,), p=Rational(1000001, 1000000)), "0 <= p <= 1"),
    ("mode", dict(data=()), "nonempty data"),
    ("mode", dict(data=[]), "nonempty data"),
)


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

def frac(number: Number) -> Fraction:
    """An exact number of the tests as a Fraction."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def exact(number: Number) -> int | Rational:
    """The number as a request or an evidence holds it: an int when it is integral, else a Rational."""
    number = frac(number)
    return number.numerator if number.denominator == 1 else Rational(number.numerator, number.denominator)


def rational(number: Number) -> Rational:
    """The number as a value holds it: always a Rational."""
    number = frac(number)
    return Rational(number.numerator, number.denominator)


def order_statistic(data: Sequence[Number], rank: int) -> Fraction:
    """x_(rank) of the data without sorting: the item v with #{x < v} <= rank < #{x <= v}, by counting every item
    against every item (quadratic, slow on purpose)."""
    items = [frac(x) for x in data]
    for v in items:
        below = sum(1 for x in items if x < v)
        if below <= rank < below + sum(1 for x in items if x == v):
            return v
    raise AssertionError(f"no order statistic of rank {rank}")


def reference_quantile(data: Sequence[Number], p: Number) -> tuple[Rational, tuple] | None:
    """(value, evidence) of the type-7 quantile from order statistics found by counting; None outside the domain
    (empty data, p outside [0, 1])."""
    p = frac(p)
    if not data or not 0 <= p <= 1:
        return None
    h = (len(data) - 1) * p
    j = h.numerator // h.denominator
    low = order_statistic(data, j)
    if h == j:
        return rational(low), ((j, exact(low)),)
    high = order_statistic(data, j + 1)
    return rational(low + (h - j) * (high - low)), ((j, exact(low)), (j + 1, exact(high)))


def reference_mode(data: Sequence[Number]) -> tuple[tuple[Rational, ...], tuple[int]] | None:
    """(value, evidence) of mode: list.count for every distinct value, then the largest count's values in ascending
    order, found by repeatedly taking the least remaining one (no sort); None for empty data."""
    if not data:
        return None
    items = [frac(x) for x in data]
    distinct: list[Fraction] = []
    for x in items:
        if x not in distinct:
            distinct.append(x)
    counts = [items.count(x) for x in distinct]
    top = max(counts)
    winners = [x for x, count in zip(distinct, counts) if count == top]
    ascending = []
    while winners:
        least = min(winners)
        winners.remove(least)
        ascending.append(rational(least))
    return tuple(ascending), (top,)


def reference(operation: str, args: dict[str, Any]) -> tuple[object, tuple] | None:
    """(value, evidence) of either operation by the references; None outside the domain."""
    if operation == "quantile":
        return reference_quantile(args["data"], args["p"])
    return reference_mode(args["data"])


def seeded_number(rng: random.Random, bits: int) -> Fraction:
    """A random exact rational: an integer of up to bits bits, or a fraction of two such integers."""
    numerator = rng.randrange(-(2 ** bits), 2 ** bits + 1)
    return Fraction(numerator) if rng.random() < 0.5 else Fraction(numerator, rng.randrange(1, 2 ** bits + 1))


def seeded_data(rng: random.Random) -> list[Fraction]:
    """1 to 24 random exact rationals: with many repeats (small numerators over a few denominators), all equal, or
    spread out with long integers."""
    n = rng.randrange(1, 25)
    style = rng.random()
    if style < 0.5:
        spread = rng.choice((1, 2, 3, 6))
        return [Fraction(rng.randrange(-spread, spread + 1), rng.choice((1, 1, 2, 3))) for _ in range(n)]
    if style < 0.6:
        return [seeded_number(rng, 8)] * n
    return [seeded_number(rng, rng.choice((1, 3, 8, 40, 200))) for _ in range(n)]


def seeded_p(rng: random.Random) -> Fraction:
    """A random p in [0, 1]: 0, 1, a simple fraction, or one with a long denominator."""
    style = rng.random()
    if style < 0.1:
        return Fraction(rng.choice((0, 1)))
    denominator = rng.choice((2, 3, 4, 5, 10, 100)) if style < 0.6 else rng.randrange(1, 2 ** 64)
    return Fraction(rng.randrange(0, denominator + 1), denominator)


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class OrderTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object, evidence: tuple) -> None:
        """A completed v6 Result of that value, of its exact type down to every number, with that evidence and its
        four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), type(value))
        if type(value) is tuple:
            self.assertTrue(all(type(item) is Rational for item in result.value))
        self.assertIs(result.outcome, Outcome.VALUE)
        self.assertEqual(result.steps[0].outputs[1], evidence)
        for produced, expected in zip(result.steps[0].outputs[1], evidence):
            if type(expected) is tuple:  # A quantile's (rank, statistic): an int where integral, else a Rational.
                self.assertIs(type(produced[1]), type(expected[1]))
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, args, value, evidence in KNOWN:
            with self.subTest(operation=operation, **args):
                self.assertEqual(reference(operation, args), (value, evidence))  # The card's value, by the reference.
                result = support.run(operation, **args)
                self.assertVerified(result, value, evidence)
                self.roundtrip(result)
        # Lists are accepted as well as tuples, ints and Rationals of denominator 1 are the same numbers, and the order
        # of the data does not matter.
        for operation, args, value in (
                ("quantile", dict(data=[4, 3, 2, 1], p=Rational(1, 4)), Rational(7, 4)),
                ("quantile", dict(data=(Rational(1), 2, Rational(3), 4), p=Rational(1, 2)), Rational(5, 2)),
                ("mode", dict(data=[3, 2, 2, 1]), (Rational(2),)),
                ("mode", dict(data=(Rational(3), 1, Rational(1), 3, 2)), (Rational(1), Rational(3)))):
            with self.subTest(operation=operation, args=args):
                result = support.run(operation, **args)
                self.assertVerified(result, value, reference(operation, args)[1])

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument, as the argument itself and as an item of data; so is every input outside the domains of the
        card. produce() tests the types itself too."""
        containers = ({1, 2, 3}, {"a": 1}, "1, 2, 3", 3, Rational(3), Fraction(1, 2), support.oversized_matrix())
        items = (*STAND_INS, Fraction(1, 2), (1, 2), [1], 1j)
        for operation, args in SMALL.items():
            for item in (*STAND_INS, *containers):
                with self.subTest(operation=operation, argument="data", item=item):
                    self.assertInvalidInput(support.run(operation, **dict(args, data=item)))
            for item in items:
                changed = args["data"][:1] + (item,) + args["data"][2:]
                with self.subTest(operation=operation, argument="data", entry=item):
                    self.assertInvalidInput(support.run(operation, **dict(args, data=changed)))
        for item in (*STAND_INS, Fraction(1, 2), (1, 2), [Rational(1, 2)], {"p": 1}, "1/2", 1j):
            with self.subTest(operation="quantile", argument="p", item=item):
                self.assertInvalidInput(support.run("quantile", **dict(SMALL["quantile"], p=item)))
        for operation, args, text in INVALID:
            with self.subTest(operation=operation, **args):
                self.assertIsNone(reference(operation, args))
                result = support.run(operation, **args)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
        # operations_v6.run rejects a stand-in by its kind before produce() runs; produce() refuses one itself, also
        # a Rational made with object.__new__ that breaks the canonical form, and arguments of another shape.
        # Labels, not the items, name the subtests: a Rational without slots has no repr.
        entries = [(repr(item), item) for item in (*STAND_INS, Integer(3), Fraction(1, 2))]
        entries += [("2/4", forged_rational(2, 4)), ("-1/-2", forged_rational(-1, -2)), ("1/0", forged_rational(1, 0)),
                    ("True/1", forged_rational(True, 1)), ("1.5/1", forged_rational(1.5, 1)),
                    ("1/2.0", forged_rational(1, 2.0)), ("no denominator", support.forged(Rational, numerator=1)),
                    ("no slots", support.forged(Rational))]
        for operation, args in SMALL.items():
            for label, item in entries:
                changed = args["data"][:1] + (item,) + args["data"][2:]
                with self.subTest(operation=operation, argument="data", entry=label, path="produce"):
                    with self.assertRaises(InvalidInput):
                        family.produce(operation, dict(args, data=changed))
                if operation == "quantile":
                    with self.subTest(operation=operation, argument="p", item=label, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, p=item))
            for item in (*STAND_INS, {1, 2}, "12", 3, Rational(3), (x for x in (1, 2)), range(3)):
                with self.subTest(operation=operation, argument="data", item=item, path="produce"):
                    with self.assertRaises(InvalidInput):
                        family.produce(operation, dict(args, data=item))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                for shape in (dict(args, extra=1), {"values": args["data"]}, tuple(args.items()), None):
                    with self.assertRaises(InvalidInput):
                        family.produce(operation, shape)
        with self.assertRaises(InvalidInput):
            family.produce("mode", {"data": (1, 2), "p": 0})  # mode takes no p.
        with self.assertRaises(InvalidInput):
            family.produce("quantile", {"data": (1, 2)})  # quantile needs p.
        for operation in ("median", "Quantile", "modes", "", None, 7):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(SMALL["quantile"]))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("quantile", data=(1, 2, 3), p=2)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in either argument and max_work=1 exceed the budget. No operation of the family
        takes a matrix, so support.oversized_matrix() has nothing to exceed here (as data it is invalid input,
        test_invalid_inputs). Work is charged before it runs, so a request that needs more than its budget fails at
        once; a value integer over max_integer_bits exceeds the budget too."""
        for operation, args in SMALL.items():
            for item in (support.HUGE, -support.HUGE, Rational(1, support.HUGE), Rational(support.HUGE, 3)):
                changed = args["data"][:1] + (item,) + args["data"][2:]
                with self.subTest(operation=operation, argument="data", item=type(item).__name__):
                    self.assertResourceLimit(support.run(operation, **dict(args, data=changed)))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
        for item in (support.HUGE, Rational(1, support.HUGE), Rational(support.HUGE - 1, support.HUGE)):
            with self.subTest(operation="quantile", argument="p", item=item.bit_length() if type(item) is int else 0):
                self.assertResourceLimit(support.run("quantile", **dict(SMALL["quantile"], p=item)))
        rng = random.Random(20261201)
        long_ints = tuple(rng.getrandbits(65_536) for _ in range(300))
        long_fractions = tuple(Rational(rng.getrandbits(30_000) | 1, rng.getrandbits(30_000) | 1 << 29_999)
                               for _ in range(300))
        start = time.perf_counter()
        for operation in OPERATIONS:
            extra = {"p": Rational(1, 3)} if operation == "quantile" else {}
            with self.subTest(operation=operation, case="charged before it runs"):
                self.assertResourceLimit(support.run(operation, data=long_ints, limits=ComputationLimits(max_work=500),
                                                     **extra))
                self.assertResourceLimit(support.run(operation, data=long_fractions,
                                                     limits=ComputationLimits(max_work=5_000), **extra))
        self.assertLess(time.perf_counter() - start, FAST)
        # A value no Result can hold: x_(0) = 0, x_(1) = 1 / 2**65535 and p = 1 / 3**41000 give the quantile
        # 1 / (3**41000 * 2**65535), whose denominator has about 130,000 bits; the request itself fits.
        start = time.perf_counter()
        result = support.run("quantile", data=(0, Rational(1, 2 ** 65_535)), p=Rational(1, 3 ** 41_000))
        self.assertResourceLimit(result)
        self.assertIn("integer bits", result.error.message)
        self.assertLess(time.perf_counter() - start, FAST)
        # Within the limit: the median of 0 and 2**65536 - 2 (65,536 bits, the longest admitted) is 2**65535 - 1.
        result = support.run("quantile", data=(0, 2 ** 65_536 - 2), p=Rational(1, 2))
        self.assertVerified(result, Rational(2 ** 65_535 - 1), ((0, 0), (1, 2 ** 65_536 - 2)))
        # A mode whose value lists 20,000 distinct integers exceeds the nodes that a Result may hold.
        self.assertResourceLimit(support.run("mode", data=tuple(range(20_000))))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts true claims in forms the producer does not emit: evidence statistics written as
        Rationals of denominator 1 where the producer writes ints, the data shuffled or with ints written as
        Rationals, and data the producer never saw whose value follows from the producer's by hand (a positive
        scale and a shift)."""
        for operation, args, value, evidence in KNOWN:
            self.assertEqual(family.produce(operation, dict(args)), (value, evidence))
            widened = dict(args, data=tuple(Rational(x) if type(x) is int else x for x in args["data"]))
            permutation = list(range(len(args["data"])))
            random.Random(len(permutation)).shuffle(permutation)
            shuffled = dict(args, data=tuple(args["data"][i] for i in permutation))
            cases = [("Rationals for ints", widened, value, evidence), ("shuffled", shuffled, value, evidence),
                     ("reversed", dict(args, data=args["data"][::-1]), value, evidence)]
            if operation == "quantile":
                rational_evidence = tuple((rank, rational(statistic)) for rank, statistic in evidence)
                cases.append(("Rational statistics", args, value, rational_evidence))
            for label, claim_args, claim, proof in cases:
                with self.subTest(operation=operation, case=label, **args):
                    self.assertAccepts(operation, claim_args, claim, proof)
        # Data the producer never saw: the quantile and the modes of (s x + t) are s q + t and s m + t for s > 0.
        scale, shift = Fraction(3, 7), Fraction(-5, 2)
        for operation, args, value, evidence in KNOWN:
            moved = [scale * frac(x) + shift for x in args["data"]]
            claim_args = dict(args, data=tuple(exact(x) for x in moved))
            if operation == "quantile":
                claim = rational(scale * frac(value) + shift)
                proof = tuple((rank, exact(scale * frac(statistic) + shift)) for rank, statistic in evidence)
            else:
                claim, proof = tuple(rational(scale * frac(m) + shift) for m in value), evidence
            with self.subTest(operation=operation, case="moved", **args):
                self.assertEqual(reference(operation, claim_args), (claim, proof))
                self.assertAccepts(operation, claim_args, claim, proof)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        for operation, args in SMALL.items():
            result = support.run(operation, **args)
            value, evidence = result.value, result.steps[0].outputs[1]
            self.assertAccepts(operation, args, value, evidence)
            if operation == "quantile":  # data (3, 1, 4, 1, 5), p = 1/3: h = 4/3, x_(1) = 1, x_(2) = 3, value 5/3.
                self.assertEqual((value, evidence), (Rational(5, 3), ((1, 1), (2, 3))))
                wrong = [Rational(5, 2), Rational(1), Rational(3)]
                cases: list[tuple[str, object, object]] = [
                    ("an int value", 1, evidence), ("a float value", 5 / 3, evidence), ("a str value", "5/3", evidence),
                    ("a Fraction value", Fraction(5, 3), evidence), ("unreduced", forged_rational(10, 6), evidence),
                    ("a negative denominator", forged_rational(-5, -3), evidence),
                    ("the value in a tuple", (value,), evidence), ("a giant value", forged_rational(GIANT), evidence),
                    ("one statistic", value, evidence[:1]), ("a third statistic", value, evidence + ((3, 4),)),
                    ("ranks shifted", value, ((2, 3), (3, 4))), ("a bool rank", value, ((True, 1), (2, 3))),
                    ("a float statistic", value, ((1, 1.0), (2, 3))), ("a str rank", value, (("1", 1), (2, 3))),
                    ("evidence as a list", value, list(evidence)), ("a statistic as a list", value, ([1, 1], (2, 3))),
                    ("an int subclass rank", value, ((Integer(1), 1), (2, 3))), ("evidence None", value, None),
                    ("no evidence", value, ()), ("oversized evidence", value, ((1, 1),) * 100_000),
                    ("a giant statistic", value, ((1, GIANT), (2, 3)))]
            else:  # data (3, 1, 4, 1, 5): 1 twice.
                self.assertEqual((value, evidence), ((Rational(1),), (2,)))
                wrong = [(Rational(3),), (Rational(1), Rational(3)), (Rational(1), Rational(3), Rational(4),
                                                                      Rational(5))]
                cases = [("an int", (1,), evidence), ("a list", [Rational(1)], evidence),
                         ("a bare Rational", Rational(1), evidence),
                         ("empty", (), evidence), ("repeated", (Rational(1), Rational(1)), evidence),
                         ("a float", (1.0,), evidence), ("a str", ("1",), evidence),
                         ("forged unreduced", (forged_rational(2, 2),), evidence),
                         ("multiplicity 1", value, (1,)), ("multiplicity 3", value, (3,)),
                         ("a bool multiplicity", value, (True,)), ("multiplicity as a list", value, [2]),
                         ("no evidence", value, ()), ("extra evidence", value, (2, 2)),
                         ("a giant multiplicity", value, (GIANT,)), ("oversized value", (Rational(1),) * 100_000,
                                                                     evidence)]
            cases += [(f"wrong {item}", item, evidence) for item in wrong]
            for label, item, proof in cases:
                with self.subTest(operation=operation, case=label):
                    self.assertRejects(operation, args, item, proof)
            with self.subTest(operation=operation, case="data as a list"):
                self.assertRejects(operation, dict(args, data=list(args["data"])), value, evidence)
            other = "mode" if operation == "quantile" else "quantile"
            for name in (other, operation.upper(), "median", ""):
                with self.subTest(operation=operation, renamed=name):
                    self.assertRejects(name, args, value, evidence)
            report = support.statuses(mf.verify(support.tampered(result, wrong[0])))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [(operation, args) for operation, args, _, _ in KNOWN]
        rng = random.Random(20261202)
        big = tuple(Rational(rng.randrange(-999, 999), rng.randrange(1, 99)) for _ in range(500))
        requests += [("quantile", dict(data=big, p=Rational(2, 7))), ("mode", dict(data=big))]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, _), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, value=result.value):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("mode", data=(1, 2, 2))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a quantile that is an
        integer, a fraction or negative, and modes of one value and of several."""
        rendered = (
            ("quantile", dict(data=(3, 1, 2), p=0), "1", "1"),
            ("quantile", dict(data=(1, 2, 3, 4), p=Rational(1, 4)), "7/4", r"\frac{7}{4}"),
            ("quantile", dict(data=(-1, -2), p=Rational(1, 3)), "-5/3", r"-\frac{5}{3}"),
            ("mode", dict(data=(1, 2, 2, 3)), "(2)", r"\left(2\right)"),
            ("mode", dict(data=(-3,)), "(-3)", r"\left(-3\right)"),
            ("mode", dict(data=(1, 1, Rational(1, 2), Rational(1, 2))), "(1/2, 1)", r"\left(\frac{1}{2}, 1\right)"),
            ("mode", dict(data=(Rational(-1, 3), 2)), "(-1/3, 2)", r"\left(-\frac{1}{3}, 2\right)"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(operation=operation, **args):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("mode", data=(1, Rational(1, 2), Rational(1, 2), 1)).value)["data"],
                         {"type": "tuple", "items": [{"type": "rational", "numerator": "1", "denominator": "2"},
                                                     {"type": "rational", "numerator": "1", "denominator": "1"}]})
        self.assertEqual(mf.to_data(support.run("quantile", data=(5,), p=Rational(1, 3)).value)["data"],
                         {"type": "rational", "numerator": "5", "denominator": "1"})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261203)
        cases = []
        for _ in range(25):
            data = tuple(exact(x) for x in seeded_data(rng))
            cases += [("quantile", dict(data=data, p=exact(seeded_p(rng)))), ("mode", dict(data=data))]
        cases += [("quantile", dict(data=[4, 1, 3], p=Rational(1, 2))), ("mode", dict(data=[1, 2, 2]))]  # Lists.
        for operation, args in cases:
            with self.subTest(operation=operation, **args):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertIs(type(found), type(expected))
                self.assertEqual(support.value(operation, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, args in SMALL.items():
            with self.subTest(operation=operation):
                request = {"data": list(args["data"])}
                if operation == "quantile":
                    request["p"] = "1/3"
                result = self.answer(support.call(operation, request))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        # Rationals travel as "p/q" strings and integers from 2**53 on as decimal strings.
        data = (Rational(1, 2), -3, 2 ** 70, Rational(-7, 4), -3, 2 ** 70)
        for operation, args in (("quantile", dict(data=data, p=Rational(3, 5))), ("mode", dict(data=data))):
            with self.subTest(operation=operation, encoded=True):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertEqual((result.value, result.steps[0].outputs[1]), reference(operation, args))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("quantile", {"data": [1, 2], "p": "3/2"}), "invalid_input")
        self.assertIn("0 <= p <= 1", error["message"])
        error = self.assertToolError(support.call("mode", {"data": []}), "invalid_input")
        self.assertIn("nonempty data", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the slow counting references of this module on at least 200 seeded inputs per
        operation, values and evidence of their exact types, and the checker accepts every answer; lists give the
        values of tuples."""
        rng = random.Random(20261204)
        counts = dict.fromkeys(OPERATIONS, 0)
        ties = interpolated = 0
        for _ in range(240):  # A fixed number of datasets, so that a failing producer cannot keep the loop going.
            data = seeded_data(rng)
            p = seeded_p(rng)
            for operation, args in (("quantile", dict(data=tuple(exact(x) for x in data), p=exact(p))),
                                    ("mode", dict(data=tuple(exact(x) for x in data)))):
                expected = reference(operation, args)
                with self.subTest(operation=operation, **args):
                    value, evidence = family.produce(operation, dict(args))
                    self.assertEqual((value, evidence), expected)
                    if operation == "quantile":
                        self.assertIs(type(value), Rational)
                        self.assertEqual([type(s) for _, s in evidence], [type(s) for _, s in expected[1]])
                        interpolated += len(evidence) == 2
                    else:
                        self.assertIs(type(value), tuple)
                        self.assertTrue(all(type(item) is Rational for item in value))
                        ties += len(value) > 1
                    self.assertEqual(family.produce(operation, dict(args, data=list(args["data"]))), (value, evidence))
                    self.assertIs(support.verdict(family, operation, args, value, evidence), True)
                    counts[operation] += 1
        self.assertGreaterEqual(min(counts.values()), 200, counts)
        self.assertGreater(ties, 20)  # Several modes were met often,
        self.assertGreater(interpolated, 40)  # and so were quantiles between two order statistics.

    def test_full_size_inputs_fit_the_default_budget(self):
        """20,000 64-bit integers, or 10,000 fractions, complete and verify under the default limits, with the values of
        the references (here sorted Fractions and a Counter of their texts: the counting references are quadratic);
        data built so that every integer has the same hash stays fast, as the producer sorts and counts no dict."""
        from collections import Counter

        rng = random.Random(20261205)
        ints = tuple(rng.randrange(-(2 ** 64), 2 ** 64) for _ in range(20_000))
        repeated = tuple(rng.randrange(-300, 300) for _ in range(20_000))
        ratios = tuple(Rational(rng.randrange(-1000, 1000), rng.randrange(1, 256)) for _ in range(10_000))
        colliding = tuple((k % 2_000) * (2 ** 61 - 1) for k in range(20_000))  # Every hash is 0.
        for label, data in (("integers", ints), ("repeated", repeated), ("rationals", ratios),
                            ("colliding", colliding)):
            ordered = sorted(frac(x) for x in data)
            p = Fraction(2, 7)
            h = (len(ordered) - 1) * p
            j = h.numerator // h.denominator
            quantile = rational(ordered[j] + (h - j) * (ordered[j + 1] - ordered[j]))
            # Keyed by the text of each Fraction, whose hash Python randomizes: the colliding integers (and the
            # Fractions of them, which hash alike) would make a Counter of numbers quadratic.
            tally = Counter(str(x) for x in ordered)
            top = max(tally.values())
            modes = tuple(rational(x) for x in sorted(Fraction(text) for text, count in tally.items() if count == top))
            expected = {"quantile": (dict(data=data, p=Rational(2, 7)), quantile), "mode": (dict(data=data), modes)}
            for operation, (args, value) in expected.items():
                with self.subTest(data=label, operation=operation):
                    with computation(fresh=True) as budget:
                        self.assertEqual(family.produce(operation, dict(args))[0], value)
                    self.assertLess(budget.work, ComputationLimits().max_work // 4)
                    if operation == "mode" and len(modes) > 5_000:
                        continue  # A value of 20,000 integers exceeds the nodes of a Result (test_resource_limits).
                    start = time.perf_counter()
                    result = support.run(operation, **args)
                    self.assertLess(time.perf_counter() - start, 2 * FAST)
                    self.assertEqual(result.value, value, result.error)
                    self.assertTrue(mf.verify(result).verified)

    def test_work_is_charged_before_it_runs(self):
        """The producer's charges depend on the request alone: a budget one unit short refuses it, its exact size
        passes. Long integers cost more than short ones, by the products and comparisons of their length."""
        rng = random.Random(20261206)
        fractions = tuple(exact(Fraction(rng.randrange(-10 ** 6, 10 ** 6), rng.randrange(1, 50))) for _ in range(200))
        integers = tuple(rng.randrange(-10 ** 6, 10 ** 6) for _ in range(200))
        for label, data in (("fractions", fractions), ("integers", integers)):
            for operation, args in (("quantile", dict(data=data, p=Rational(3, 11))), ("mode", dict(data=data))):
                with self.subTest(data=label, operation=operation):
                    with computation(fresh=True) as budget:
                        answer = family.produce(operation, dict(args))
                    work = budget.work
                    self.assertGreater(work, 2 * 200)  # At least a unit per item and per comparison.
                    with self.assertRaises(ResourceLimitError), computation(ComputationLimits(max_work=work - 1),
                                                                            fresh=True):
                        family.produce(operation, dict(args))
                    with computation(ComputationLimits(max_work=work), fresh=True) as budget:
                        self.assertEqual(family.produce(operation, dict(args)), answer)
                    self.assertEqual(budget.work, work)
        # The same data scaled by a 20,000-bit factor: each comparison of a fraction pays for its two products by their
        # length, as trusted.integers.product_work prices them.
        from mathforge.trusted.integers import product_work

        scale = 2 ** 20_000 + 1
        short = fractions[:40]
        scaled = tuple(exact(frac(x) * scale) if type(x) is Rational else x for x in short)
        charged = []
        for data in (short, scaled):
            with computation(fresh=True) as budget:
                family.produce("mode", {"data": data})
            charged.append(budget.work)
        self.assertGreater(charged[1] - charged[0], 40 * product_work(20_000, 6))

    def test_family_module_imports(self):
        """The family module imports at its top only modules that `import mathforge` loads, and nowhere the checker
        layer, the lean path, the MCP layer or generated code."""
        package = family.__name__.split(".")[:-1]  # mathforge.areas.stats

        def absolute(node: ast.ImportFrom) -> str:
            """The module that an import names, a relative one resolved against the family's package."""
            base = package[:len(package) - node.level + 1] if node.level else []
            return ".".join([*base, node.module or ""])

        tree = ast.parse(inspect.getsource(family))
        top = {absolute(node) for node in tree.body if isinstance(node, ast.ImportFrom)}
        self.assertEqual(top, {"__future__", "functools", "math", "mathforge.errors", "mathforge.limits",
                               "mathforge.model", "mathforge.registry"})
        self.assertFalse([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
        everywhere = {absolute(node) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        for name in everywhere:
            self.assertFalse(any(part in name.split(".") for part in ("certify", "lean", "mcp", "_generated")), name)
        # The cost model, imported where it charges: import mathforge does not load it.
        self.assertEqual(everywhere - top, {"mathforge.trusted.integers"})
