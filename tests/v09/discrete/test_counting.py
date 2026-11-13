"""The counting family of mathforge.discrete: partitions, derangements, bell, stirling1.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register discrete counting

Every expected value comes from the card or from a reference of this module, written here and slow on purpose, never
from the producer: brute force over the partitions, the permutations and the set partitions; the partition numbers by
their largest part; derangements by inclusion and exclusion; Bell numbers as sums of Stirling numbers of the second
kind; Stirling rows as balanced products of polynomials. At the family's limits, where those are too slow, the values
are held to properties of their own: Ramanujan's congruences of the partition numbers, Touchard's congruence of the
Bell numbers and the rising factorial modulo primes. The card defers no input of this family: every valid request up
to the limits has a value, so no test expects an Unevaluated result. tests/v09/discrete/test_counting_checker.py
tests the checker on its own.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import functools
import inspect
from itertools import accumulate, permutations
from math import comb, factorial, isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.discrete import counting as family
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational

from .. import support

FAST = 1.0  # Seconds: a request that its budget or a limit refuses fails well within this.
LARGEST = {"partitions": 5000, "derangements": 5900, "bell": 1500, "stirling1": 1000}  # The documented limits of n.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
P1000 = 24061467864032622473692149727991  # p(1000), as Hardy and Ramanujan quote MacMahon's table.
STAND_INS = (0.5, True, False, "1", None, (1, 2), [1], Rational(3), Rational(1, 2))  # No exact int.

# One small request per operation.
SMALL = {
    "partitions": dict(n=10),
    "derangements": dict(n=10),
    "bell": dict(n=10),
    "stirling1": dict(n=5, k=3),
}

# The known values of the card, and the smallest arguments: (operation, arguments, value).
KNOWN = (
    ("partitions", dict(n=0), 1),
    ("partitions", dict(n=1), 1),
    ("partitions", dict(n=5), 7),
    ("partitions", dict(n=10), 42),
    ("partitions", dict(n=100), 190569292),
    ("derangements", dict(n=0), 1),
    ("derangements", dict(n=1), 0),
    ("derangements", dict(n=2), 1),
    ("derangements", dict(n=4), 9),
    ("derangements", dict(n=10), 1334961),
    ("bell", dict(n=0), 1),
    ("bell", dict(n=1), 1),
    ("bell", dict(n=5), 52),
    ("bell", dict(n=10), 115975),
    ("stirling1", dict(n=4, k=2), 11),
    ("stirling1", dict(n=5, k=3), 35),
    ("stirling1", dict(n=0, k=0), 1),
    ("stirling1", dict(n=1, k=0), 0),
    ("stirling1", dict(n=1, k=1), 1),
    ("stirling1", dict(n=3, k=5), 0),
)

# Inputs outside the domains of the card, each invalid input.
INVALID = (
    ("partitions", dict(n=-1)),
    ("partitions", dict(n=-10 ** 30)),
    ("derangements", dict(n=-1)),
    ("derangements", dict(n=-7)),
    ("bell", dict(n=-1)),
    ("bell", dict(n=-2 ** 70)),
    ("stirling1", dict(n=-1, k=0)),
    ("stirling1", dict(n=3, k=-1)),
    ("stirling1", dict(n=-1, k=-1)),
    ("stirling1", dict(n=0, k=-5)),
)


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

def primes_below(limit: int) -> list[int]:
    """The primes below limit, by Eratosthenes' sieve."""
    flags = bytearray([1]) * limit
    flags[:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return [n for n in range(limit) if flags[n]]


def partitions_brute(n: int, largest: int | None = None) -> int:
    """The partitions of n into parts of at most largest (any size by default), listed part by part."""
    if n == 0:
        return 1
    top = n if largest is None else min(n, largest)
    return sum(partitions_brute(n - part, part) for part in range(1, top + 1))


@functools.lru_cache(maxsize=None)
def partitions_by_largest_part(n: int) -> tuple[int, ...]:
    """(p(0), ..., p(n)) by the largest part: once the parts below size are counted, admitting the part size adds to
    the count of m the partitions of m - size into parts of at most size."""
    p = [1] + [0] * n
    for size in range(1, n + 1):
        for m in range(size, n + 1):
            p[m] += p[m - size]
    return tuple(p)


def partition_numbers(n: int) -> tuple[int, ...]:
    """(p(0), ..., p(n)) for n <= 1000, from the table by the largest part."""
    return partitions_by_largest_part(1000)[:n + 1]


@functools.lru_cache(maxsize=None)
def pentagonal_table(n: int) -> tuple[int, ...]:
    """(p(0), ..., p(n)) by Euler's recurrence as written here, with the terms of j = 1, 2, ...: the pentagonal numbers
    j(3j - 1)/2 and j(3j + 1)/2, both with the sign (-1)**(j + 1). For n beyond the table by the largest part, which
    test_references_agree holds it to."""
    table = [1]
    for m in range(1, n + 1):
        total, j = 0, 1
        while j * (3 * j - 1) // 2 <= m:
            sign = 1 if j % 2 else -1
            total += sign * table[m - j * (3 * j - 1) // 2]
            if j * (3 * j + 1) // 2 <= m:
                total += sign * table[m - j * (3 * j + 1) // 2]
            j += 1
        table.append(total)
    return tuple(table)


def ramanujan_congruences(table: tuple[int, ...]) -> bool:
    """Ramanujan's congruences p(5m + 4) = 0 mod 5, p(7m + 5) = 0 mod 7 and p(11m + 6) = 0 mod 11, at every index of
    a table of partition numbers."""
    return all(table[index] % modulus == 0 for modulus, start in ((5, 4), (7, 5), (11, 6))
               for index in range(start, len(table), modulus))


def derangements_brute(n: int) -> int:
    """The permutations of range(n) without a fixed point, counted one by one."""
    return sum(all(image != point for point, image in enumerate(p)) for p in permutations(range(n)))


def inclusion_exclusion(n: int) -> int:
    """D(n) = sum over i of (-1)**i n!/i!: the permutations that fix no point of any i-set, n!/i! = (i + 1)...n built
    downwards from i = n."""
    total, falling = 0, 1  # falling = n!/i!
    for i in range(n, -1, -1):
        total += falling if i % 2 == 0 else -falling
        falling *= i
    return total


def set_partitions_brute(n: int) -> int:
    """The set partitions of n items, as restricted growth strings: a_0 = 0 and a_i <= 1 + max(a_0, ..., a_(i - 1))."""
    count, pending = 0, [(0, -1)]  # (length so far, largest block number so far)
    while pending:
        length, largest = pending.pop()
        if length == n:
            count += 1
            continue
        for block in range(largest + 2):
            pending.append((length + 1, max(largest, block)))
    return count


@functools.lru_cache(maxsize=None)
def bell_by_stirling2(n: int) -> tuple[int, ...]:
    """(B(0), ..., B(n)), each B(m) the sum of the row S(m, 0), ..., S(m, m) of the Stirling numbers of the second kind,
    S(m, j) = j S(m - 1, j) + S(m - 1, j - 1): the item m joins one of the j blocks of the others or a block of its
    own."""
    row, numbers = [1], [1]
    for m in range(1, n + 1):
        row = [0] + [j * (row[j] if j < len(row) else 0) + row[j - 1] for j in range(1, m + 1)]
        numbers.append(sum(row))
    return tuple(numbers)


def bell_triangle(n: int) -> int:
    """B(n) by Bell's triangle, each row the running sums of the row before, started at its last entry: for the n
    beyond bell_by_stirling2's reach, which test_references_agree holds it to."""
    row = [1]
    for _ in range(n):
        row = list(accumulate(row, initial=row[-1]))
    return row[0]


def bell_touchard(value: int, n: int, primes: list[int]) -> bool:
    """Whether value agrees with B(n) modulo every prime p < n of primes by Touchard's congruence,
    B(p + m) = B(m) + B(m + 1) mod p, with B(m) and B(m + 1) from bell_by_stirling2."""
    small = bell_by_stirling2(max(n - p for p in primes) + 1)
    return all(value % p == (small[n - p] + small[n - p + 1]) % p for p in primes)


def cycle_count(permutation: tuple[int, ...]) -> int:
    seen, count = [False] * len(permutation), 0
    for start in range(len(permutation)):
        if not seen[start]:
            count += 1
            point = start
            while not seen[point]:
                seen[point] = True
                point = permutation[point]
    return count


@functools.lru_cache(maxsize=None)
def stirling_brute(n: int) -> tuple[int, ...]:
    """(c(n, 0), ..., c(n, n)): the permutations of range(n) counted by their number of cycles."""
    counts = [0] * (n + 1)
    for permutation in permutations(range(n)):
        counts[cycle_count(permutation)] += 1
    return tuple(counts)


def polynomial_product(left: list[int], right: list[int]) -> list[int]:
    """The coefficients of the product of two polynomials, lowest first, by schoolbook multiplication."""
    product = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    return product


def rising_factors(low: int, high: int) -> list[int]:
    """The coefficients of (x + low)(x + low + 1)...(x + high - 1), high > low, as a balanced product tree."""
    if high - low == 1:
        return [low, 1]
    middle = (low + high) // 2
    return polynomial_product(rising_factors(low, middle), rising_factors(middle, high))


@functools.lru_cache(maxsize=None)
def rising_row(n: int) -> tuple[int, ...]:
    """(c(n, 0), ..., c(n, n)), the coefficients of x(x + 1)...(x + n - 1), as a balanced product of its halves."""
    return tuple(rising_factors(0, n)) if n else (1,)


def rising_row_sequential(n: int) -> tuple[int, ...]:
    """The same row, multiplied out one factor at a time: for n beyond the product tree's reach (stirling1(1001, k),
    a claim that the producer does not make), which test_references_agree holds to the product tree."""
    row = [1]
    for m in range(n):
        row = [m * row[0]] + [row[i - 1] + m * row[i] for i in range(1, len(row))] + [row[-1]]
    return tuple(row)


def rising_row_mod(n: int, p: int) -> tuple[int, ...]:
    """x(x + 1)...(x + n - 1) modulo a prime p, as its n + 1 residues: for n = qp + r it is (x**p - x)**q times
    x(x + 1)...(x + r - 1), since its factors run q times through every residue modulo p and then through 0, ..., r - 1,
    and x(x + 1)...(x + p - 1) = x**p - x modulo p (Fermat's little theorem: its roots are the p residues)."""
    q, r = divmod(n, p)
    power = [0] * (p * q + 1)  # (x**p - x)**q = sum over j of C(q, j) x**(pj) (-x)**(q - j)
    for j in range(q + 1):
        power[p * j + q - j] = comb(q, j) * (-1) ** (q - j) % p
    tail = [1]
    for m in range(r):
        tail = [(low + m * high) % p for low, high in zip([0] + tail, tail + [0])]
    return tuple(c % p for c in polynomial_product(power, tail))


def reference(operation: str, args: dict[str, int]) -> int:
    """The value of a request with small arguments by the references (brute force where it is fast)."""
    n = args["n"]
    if operation == "partitions":
        return partitions_brute(n) if n <= 20 else partition_numbers(n)[n]
    if operation == "derangements":
        return derangements_brute(n) if n <= 7 else inclusion_exclusion(n)
    if operation == "bell":
        return set_partitions_brute(n) if n <= 8 else bell_by_stirling2(n)[n]
    row = stirling_brute(n) if n <= 6 else rising_row(n)
    return row[args["k"]] if args["k"] <= n else 0


@dataclass(frozen=True)
class Count:
    """A record that holds a right answer. No value of the counting family is a record, so the checker refuses every
    record: this one, an impostor of it and one forged with object.__new__."""
    value: object


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


class CountingTests(support.FamilyTestCase):
    FAMILY = family

    def assertAgrees(self, operation: str, args: dict[str, int], expected: int) -> tuple:
        """produce() answers expected, an exact int, and the checker accepts that answer with its evidence, which is
        returned."""
        value, evidence = family.produce(operation, dict(args))
        self.assertEqual(value, expected)
        self.assertIs(type(value), int)
        self.assertIs(type(evidence), tuple)
        self.assertIs(support.verdict(family, operation, args, value, evidence), True)
        return evidence

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that exact int, with its four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), int)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertLayout(self, operation: str, args: dict[str, int], evidence: object) -> None:
        """The evidence has the layout of the family's docstring, with the entries of the references: (table,) for
        partitions, (row,) for stirling1 (the full row, also for k > n), () otherwise; plain tuples of exact ints."""
        n = args["n"]
        if operation == "partitions":
            self.assertEqual(evidence, (partition_numbers(n),))
        elif operation == "stirling1":
            self.assertEqual(evidence, (rising_row(n),))
        else:
            self.assertEqual(evidence, ())
        self.assertIs(type(evidence), tuple)
        for inner in evidence:
            self.assertIs(type(inner), tuple)
            self.assertEqual(len(inner), n + 1)
            self.assertTrue(all(type(entry) is int for entry in inner))

    def test_references_agree(self):
        """The references of this module agree with brute force, with each other and with the congruences that the
        tests at the family's limits rely on."""
        table = partition_numbers(1000)
        self.assertEqual([partitions_brute(n) for n in range(26)], list(table[:26]))
        self.assertEqual(table[1000], P1000)
        self.assertEqual(pentagonal_table(1000), table)
        self.assertTrue(ramanujan_congruences(table))
        self.assertFalse(ramanujan_congruences(table[:6] + (table[6] + 1,) + table[7:]))  # p(6) = 11 is a multiple.
        self.assertEqual([derangements_brute(n) for n in range(9)], [inclusion_exclusion(n) for n in range(9)])
        self.assertEqual([set_partitions_brute(n) for n in range(10)], list(bell_by_stirling2(9)))
        self.assertEqual([bell_triangle(n) for n in range(0, 201, 25)], list(bell_by_stirling2(200)[::25]))
        primes = primes_below(100)
        numbers = bell_by_stirling2(200)
        for n in range(100, 201):
            self.assertTrue(bell_touchard(numbers[n], n, primes), n)
        self.assertFalse(bell_touchard(numbers[150] + 2 * 3 * 5, 150, primes))
        for n in range(8):
            self.assertEqual(stirling_brute(n), rising_row(n))
        for n in (0, 1, 7, 30, 64, 101):
            self.assertEqual(rising_row_sequential(n), rising_row(n))
            for p in (2, 3, 5, 7, 11, 13, 97):
                self.assertEqual(rising_row_mod(n, p), tuple(c % p for c in rising_row(n)), (n, p))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts, with the evidence of the
        family's docstring."""
        for operation, args, value in KNOWN:
            with self.subTest(operation=operation, **args):
                self.assertEqual(reference(operation, args), value)  # The card's value, by brute force.
                result = support.run(operation, **args)
                self.assertVerified(result, value)
                self.assertLayout(operation, args, result.steps[0].outputs[1])
                self.roundtrip(result)
        self.assertEqual(family.produce("partitions", {"n": 0}), (1, ((1,),)))
        self.assertEqual(family.produce("partitions", {"n": 5}), (7, ((1, 1, 2, 3, 5, 7),)))
        self.assertEqual(family.produce("derangements", {"n": 4}), (9, ()))
        self.assertEqual(family.produce("bell", {"n": 5}), (52, ()))
        self.assertEqual(family.produce("stirling1", {"n": 0, "k": 0}), (1, ((1,),)))
        self.assertEqual(family.produce("stirling1", {"n": 0, "k": 3}), (0, ((1,),)))
        self.assertEqual(family.produce("stirling1", {"n": 4, "k": 2}), (11, ((0, 6, 11, 6, 1),)))
        self.assertEqual(family.produce("stirling1", {"n": 3, "k": 5}), (0, ((0, 2, 3, 1),)))  # The row, never padded.
        self.assertEqual(family.produce("stirling1", {"n": 3, "k": 10 ** 40}), (0, ((0, 2, 3, 1),)))
        result = support.run("partitions", n=1000)
        self.assertVerified(result, P1000)
        self.assertEqual(result.steps[0].outputs[1], (partition_numbers(1000),))

    def test_invalid_inputs(self):
        """0.5, True, "1", None, a Rational and a container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the domains of the card. produce() tests the types itself too."""
        for operation, args in SMALL.items():
            for name in args:
                for item in STAND_INS:
                    with self.subTest(operation=operation, argument=name, item=item):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: item})))
        for operation, args in INVALID:
            with self.subTest(operation=operation, **args):
                result = support.run(operation, **args)
                self.assertInvalidInput(result)
                expected = "n >= 0 and k >= 0" if operation == "stirling1" else "n >= 0"
                self.assertIn(f"{operation} requires {expected}", result.error.message)
        # operations_v6.run rejects a stand-in by its kind before produce() runs; produce() refuses one itself.
        for operation, args in SMALL.items():
            for name in args:
                for item in (True, False, 0.5, "1", None, Rational(3), Integer(3), -1):
                    with self.subTest(operation=operation, argument=name, item=item, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: item}))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(args, extra=1))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, tuple(args.items()))
        with self.assertRaises(InvalidInput):
            family.produce("stirling1", {"n": 5})
        for operation in ("stirling", "Partitions", "partitions ", "", None, 7):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, {"n": 3})
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("partitions", n=-1)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 and max_work=1 exceed the budget, and an n above the operation's limit is refused at
        once, whatever the limits. No operation takes a matrix: an oversized one where an int belongs is invalid
        input. A small budget stops a large request early, since every step is charged before it runs."""
        for operation, args in SMALL.items():
            for name in args:
                with self.subTest(operation=operation, argument=name):
                    self.assertResourceLimit(support.run(operation, **dict(args, **{name: support.HUGE})))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
            with self.subTest(operation=operation, matrix="oversized"):
                self.assertInvalidInput(support.run(operation, **dict(args, n=support.oversized_matrix())))
        with self.assertRaises(ResourceLimitError):  # produce() holds k to max_integer_bits as well.
            family.produce("stirling1", {"n": 5, "k": support.HUGE})
        start = time.perf_counter()
        for operation, largest in LARGEST.items():
            extra = {"k": 0} if operation == "stirling1" else {}
            for n in (largest + 1, 10 ** 6, 2 ** 64):
                with self.subTest(operation=operation, n=n):
                    result = support.run(operation, n=n, **extra)
                    self.assertResourceLimit(result)
                    self.assertIn(f"{operation} takes n up to {largest}", result.error.message)
                    with computation(fresh=True) as budget, self.assertRaises(ResourceLimitError):
                        family.produce(operation, dict(n=n, **extra))
                    self.assertEqual(budget.work, 1)  # The call's unit alone: no step ran.
            with self.subTest(operation=operation, max_work="wide"):
                wide = ComputationLimits(max_work=10 ** 9)
                self.assertResourceLimit(support.run(operation, n=largest + 1, limits=wide, **extra))
        for operation, args in (("partitions", dict(n=5000)), ("derangements", dict(n=5900)), ("bell", dict(n=1500)),
                                ("stirling1", dict(n=1000, k=3))):
            with self.subTest(operation=operation, max_work=20_000):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=20_000), **args))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_results_longer_than_max_integer_bits_are_refused_exactly(self):
        """produce() refuses with ResourceLimitError exactly the requests whose value or evidence holds an integer
        longer than max_integer_bits, or whose check needs a product that Budget.product refuses (it admits factors
        of max_integer_bits + 1 bits together): the checker's n! B(n) for bell and (n - 1)! * n for stirling1, since
        the verifier checks under the request's own limits. At the bits needed it completes, one bit below it fails."""
        bell150, row120 = bell_by_stirling2(150)[150], rising_row(120)
        requests = (("partitions", dict(n=600), max(partition_numbers(600)).bit_length()),
                    ("derangements", dict(n=300), inclusion_exclusion(300).bit_length()),
                    ("bell", dict(n=150), bell150.bit_length() + factorial(150).bit_length() - 1),
                    *(("stirling1", dict(n=120, k=k),
                       max(max(row120).bit_length(), factorial(119).bit_length() + (120).bit_length() - 1))
                      for k in (2, 500)))
        self.assertGreater(requests[2][2], bell150.bit_length())  # The check needs more bits than B(150) has.
        self.assertGreater(requests[3][2], max(row120).bit_length())  # And more than the row's longest entry.
        for operation, args, bits in requests:
            with self.subTest(operation=operation, **args):
                with computation(ComputationLimits(max_integer_bits=bits), fresh=True):
                    value, evidence = family.produce(operation, dict(args))
                self.assertEqual(value, reference(operation, args))
                with self.assertRaises(ResourceLimitError), computation(ComputationLimits(max_integer_bits=bits - 1),
                                                                        fresh=True):
                    family.produce(operation, dict(args))
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_integer_bits=bits - 1),
                                                     **args))

    def test_producer_and_checker_run_out_of_integer_bits_together(self):
        """Under a lowered max_integer_bits, produce() completes exactly when the checker, in a budget of the same
        limits as operations_v6.run's verifier opens it, verifies the true claim: run() never computes a value and
        then reports "Independent verification exceeded its budget". Before the producer mirrored the checker's
        products, bell(108) to bell(214) did that under 1,000 bits (B(n) fits, n! B(n) does not), and stirling1 at
        the largest n that fits did for about a third of the limits (n! is a few bits longer than the row's longest
        entry). Every n is tried at the small limits; bell(4) needs one bit more than n! B(n) does (the square
        16 * 16 of its term 4**4)."""
        bells = bell_by_stirling2(130)
        rows = [rising_row(n) for n in range(111)]
        for operation, top, limits in (("bell", 130, (*range(1, 40), 333, 1000)),
                                       ("stirling1", 110, (*range(1, 40), 333, 701))):
            for bits in limits:
                restricted = ComputationLimits(max_integer_bits=bits)
                for n in range(top + 1):
                    if operation == "bell":
                        args, value, evidence = {"n": n}, bells[n], ()
                    else:
                        args, value, evidence = {"n": n, "k": 1}, rows[n][1] if n else 0, (rows[n],)
                    try:
                        with computation(restricted, fresh=True):
                            produced = family.produce(operation, dict(args))
                    except ResourceLimitError:
                        produced = None
                    integers = (value, *(evidence[0] if evidence else ()))
                    fits = max(integer.bit_length() for integer in integers) <= bits  # Else no Result may hold it.
                    try:
                        checked = fits and support.verdict(family, operation, args, value, evidence,
                                                           limits=restricted) is True
                    except ResourceLimitError:
                        checked = False
                    with self.subTest(operation=operation, bits=bits, n=n):
                        self.assertEqual(produced is not None, checked)
                        if produced is not None:
                            self.assertEqual(produced, (value, evidence))
                    if produced is None and not checked and n > 8:
                        break  # Both refuse every larger n: the integers only grow.
        row150 = rising_row(150)
        short = factorial(149).bit_length() + (150).bit_length() - 2  # One bit short of the check of stirling1(150, .).
        self.assertLessEqual(max(row150).bit_length(), short)  # The row itself fits.
        for operation, args, bits in (("bell", dict(n=150), 1000), ("stirling1", dict(n=150, k=3), short)):
            with self.subTest(operation=operation, bits=bits):
                result = support.run(operation, limits=ComputationLimits(max_integer_bits=bits), **args)
                self.assertResourceLimit(result)
                self.assertNotIn("Independent verification", result.error.message)

    def test_a_max_work_between_production_and_check_reports_the_resources(self):
        """operations_v6.run verifies in a fresh budget of the request's own limits, and the checker spends more
        work than the producer on derangements (about 5.5 times at n = 5900) and stirling1 (about 1.6 times): under a
        max_work between the two, produce() completes but run() reports resource_limit_exceeded, never a value and
        never internal_error. A max_work that covers the check verifies."""
        for operation, args, between, enough in (("derangements", dict(n=5900), 100_000, 300_000),
                                                 ("stirling1", dict(n=1000, k=7), 1_200_000, 1_800_000)):
            with self.subTest(operation=operation):
                with computation(ComputationLimits(max_work=between), fresh=True):
                    family.produce(operation, dict(args))
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=between), **args))
                result = support.run(operation, limits=ComputationLimits(max_work=enough), **args)
                self.assertTrue(mf.verify(result).verified, result.error)

    def test_no_term_of_the_bell_identity_has_the_bit_length_of_the_sum(self):
        """The bound that _bell_check_fits relies on: for 5 <= n, no term C(n, j) j**n D(n - j) of the checker's sum
        n! B(n) has the bit length of the sum, so no product of the check counts more bits than n! B(n) does. All n
        from 5 to 160 are checked here and a sample up to the limit 1500 (every n up to 1500 was checked once, which
        takes about a minute); for n <= 4 one term has it, which is why bell(4) is held one bit tighter."""
        derangements = [1]
        for m in range(1, LARGEST["bell"] + 1):
            derangements.append(m * derangements[-1] + (-1) ** m)

        def dominated(n: int) -> bool:
            terms = [comb(n, j) * j ** n * derangements[n - j] for j in range(1, n + 1)]
            total = sum(terms) + (1 if n == 0 else 0)
            return max(terms, default=1).bit_length() < total.bit_length()

        self.assertEqual([n for n in range(0, 161) if not dominated(n)], [0, 1, 2, 3, 4])
        for n in (500, 997, LARGEST["bell"]):
            self.assertTrue(dominated(n), n)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The evidence of a claim is determined by its arguments (the table and the row are unique), so the true
        claims that the producer does not make are those above its limits: the checker verifies them up to its own
        capacity under the default budget, which the limits leave a margin below."""
        table = pentagonal_table(5001)
        self.assertEqual(table[:1001], partition_numbers(1000))
        row = rising_row_sequential(1001)
        self.assertEqual((sum(row), row[1], row[1000]), (factorial(1001), factorial(1000), comb(1001, 2)))
        claims = [("partitions", {"n": 5001}, table[5001], (table,)),
                  ("derangements", {"n": 5910}, inclusion_exclusion(5910), ()),
                  ("bell", {"n": 1501}, bell_triangle(1501), ()),
                  *(("stirling1", {"n": 1001, "k": k}, row[k] if k <= 1001 else 0, (row,)) for k in (1, 500, 1002))]
        self.assertTrue(bell_touchard(claims[2][2], 1501, [p for p in primes_below(1501) if p > 1400]))
        for operation, args, value, evidence in claims:
            with self.subTest(operation=operation, **args):
                with self.assertRaises(ResourceLimitError):
                    family.produce(operation, dict(args))
                self.assertAccepts(operation, args, value, evidence)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, a record of any class (an
        impostor, one forged with object.__new__), a bool, a float or a str where an int belongs, a list for a
        tuple, missing or extra evidence, a changed table or row, a wrong operation name and oversized forged
        evidence; mf.verify fails the mathematics of a Result whose value was replaced."""
        claims = {"partitions": dict(n=30), "derangements": dict(n=12), "bell": dict(n=12), "stirling1": dict(n=8, k=3)}
        for operation, args in claims.items():
            result = support.run(operation, **args)
            value, evidence = result.value, result.steps[0].outputs[1]
            self.assertVerified(result, value)
            self.assertAccepts(operation, args, value, evidence)
            cases = [("one more", value + 1, evidence), ("one less", value - 1, evidence),
                     ("negated", -value, evidence), ("doubled", 2 * value, evidence),
                     ("a giant value", GIANT, evidence), ("a Rational", Rational(value), evidence),
                     ("a float", float(value), evidence), ("a str", str(value), evidence),
                     ("an int subclass", Integer(value), evidence), ("a bool", True, evidence),
                     ("in a tuple", (value,), evidence), ("a record", Count(value), evidence),
                     ("an impostor record", support.impostor(Count(value)), evidence),
                     ("a forged record", support.forged(Count, value=value), evidence),
                     ("evidence as a list", value, list(evidence)), ("evidence None", value, None),
                     ("an extra evidence item", value, evidence + ((),)),
                     ("oversized evidence", value, evidence + (0,) * 100_000)]
            if evidence:
                inner = evidence[0]
                cases += [("the evidence item missing", value, ()), ("the entries as a list", value, (list(inner),)),
                          ("an entry changed", value, (inner[:-2] + (inner[-2] + 1, inner[-1]),)),
                          ("an entry as a float", value, (inner[:-1] + (float(inner[-1]),),)),
                          ("one entry short", value, (inner[:-1],)), ("one entry long", value, (inner + (0,),)),
                          ("the entries unwrapped", value, inner)]
            for label, item, proof in cases:
                with self.subTest(operation=operation, case=label):
                    self.assertRejects(operation, args, item, proof)
            for other in claims:  # The true claim under the name of another operation.
                if other != operation:
                    with self.subTest(operation=operation, renamed=other):
                        self.assertRejects(other, args, value, evidence)
            report = support.statuses(mf.verify(support.tampered(result, value + 1)))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = list(SMALL.items()) + [("partitions", dict(n=1000)), ("derangements", dict(n=1000)),
                                          ("bell", dict(n=200)), ("stirling1", dict(n=60, k=7)),
                                          ("stirling1", dict(n=4, k=9))]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, args), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, **args):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("bell", n=5)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip with its evidence, and its value, an int, renders as text and as
        LaTeX: zero, small and long values."""
        d1000 = inclusion_exclusion(1000)  # 2,568 digits, within the default int-to-str limit of str().
        rendered = (("partitions", dict(n=10), "42"), ("partitions", dict(n=1000), str(P1000)),
                    ("derangements", dict(n=1), "0"), ("derangements", dict(n=1000), str(d1000)),
                    ("bell", dict(n=10), "115975"), ("stirling1", dict(n=5, k=3), "35"),
                    ("stirling1", dict(n=3, k=5), "0"))
        for operation, args, text in rendered:
            with self.subTest(operation=operation, **args):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), text)
                if len(text) < 20:
                    self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("bell", n=5).value)["data"], {"type": "integer", "value": "52"})
        self.assertEqual(mf.to_data(support.run("stirling1", n=3, k=5).value)["data"],
                         {"type": "integer", "value": "0"})
        # The longest value of the family, D(5900) of 65,401 bits, survives the round trip too.
        self.roundtrip(support.run("derangements", n=5900))

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261201)
        cases = [("partitions", dict(n=rng.randrange(0, 600))) for _ in range(8)]
        cases += [("derangements", dict(n=rng.randrange(0, 900))) for _ in range(8)]
        cases += [("bell", dict(n=rng.randrange(0, 200))) for _ in range(8)]
        cases += [("stirling1", dict(n=n, k=rng.randrange(0, n + 3))) for n in rng.sample(range(0, 150), 8)]
        cases += [("partitions", dict(n=0)), ("derangements", dict(n=1)), ("stirling1", dict(n=7, k=8)),
                  ("partitions", dict(n=5000)), ("derangements", dict(n=5900))]
        for operation, args in cases:
            with self.subTest(operation=operation, **args):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, **args), expected)
        # Past a limit the lean path falls back, and the value function raises the structured path's error.
        self.assertIs(support.lean_value("bell", n=1501), lean.FALLBACK)
        with self.assertRaises(mf.OperationError) as raised:
            support.value("bell", n=1501)
        self.assertEqual(raised.exception.code, "resource_limit_exceeded")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, args in SMALL.items():
            with self.subTest(operation=operation):
                result = self.answer(support.call(operation, args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        # Integers from 2**53 on travel as decimal strings: the table of p(0), ..., p(1000) and the row of c(60, .).
        for operation, args in (("partitions", dict(n=1000)), ("bell", dict(n=200)), ("stirling1", dict(n=60, k=7))):
            with self.subTest(operation=operation, large=True):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("partitions", {"n": -1}), "invalid_input")
        self.assertIn("partitions requires n >= 0", error["message"])
        error = self.assertToolError(support.call("stirling1", {"n": 1001, "k": 2}), "resource_limit_exceeded")
        self.assertIn("stirling1 takes n up to 1000", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 200 seeded inputs per operation, and the
        checker accepts every answer with its evidence; a few larger inputs agree with the references too."""
        rng = random.Random(20261202)
        counts = dict.fromkeys(LARGEST, 0)
        # partitions: brute force up to 20, else the table by the largest part, whose prefix is the evidence.
        for n in [*range(21), *(rng.randrange(21, 701) for _ in range(190))]:
            with self.subTest(operation="partitions", n=n):
                expected = partition_numbers(n)
                if n <= 20:
                    self.assertEqual(partitions_brute(n), expected[n])
                self.assertEqual(self.assertAgrees("partitions", {"n": n}, expected[n]), (expected,))
                counts["partitions"] += 1
        # derangements: brute force up to 7, else inclusion and exclusion.
        for n in [*range(8), *(rng.randrange(8, 401) for _ in range(200))]:
            with self.subTest(operation="derangements", n=n):
                self.assertEqual(self.assertAgrees("derangements", {"n": n}, reference("derangements", {"n": n})), ())
                counts["derangements"] += 1
        # bell: brute force up to 8, else sums of Stirling numbers of the second kind.
        for n in [*range(9), *(rng.randrange(9, 121) for _ in range(200))]:
            with self.subTest(operation="bell", n=n):
                self.assertEqual(self.assertAgrees("bell", {"n": n}, reference("bell", {"n": n})), ())
                counts["bell"] += 1
        # stirling1: brute force up to 6, else the product tree; k up to n + 2, and some far beyond n.
        for _ in range(240):
            n = rng.choice((rng.randrange(0, 7), rng.randrange(0, 80)))
            k = rng.choice((rng.randrange(0, n + 3), rng.randrange(0, 10 ** 30)))
            with self.subTest(operation="stirling1", n=n, k=k):
                evidence = self.assertAgrees("stirling1", {"n": n, "k": k}, reference("stirling1", {"n": n, "k": k}))
                self.assertEqual(evidence, (stirling_brute(n) if n <= 6 else rising_row(n),))
                counts["stirling1"] += 1
        self.assertTrue(all(count >= 200 for count in counts.values()), counts)
        # Larger inputs: the table by the largest part to 1000, Euler's recurrence as written here beyond it,
        # inclusion and exclusion, sums of Stirling numbers of the second kind and the product tree.
        for n in (1000, 2600):
            with self.subTest(operation="partitions", n=n):
                expected = partition_numbers(n) if n <= 1000 else pentagonal_table(n)
                self.assertEqual(self.assertAgrees("partitions", {"n": n}, expected[n]), (expected,))
        for n in (1000, 3001):
            with self.subTest(operation="derangements", n=n):
                self.assertAgrees("derangements", {"n": n}, inclusion_exclusion(n))
        with self.subTest(operation="bell", n=400):
            self.assertAgrees("bell", {"n": 400}, bell_by_stirling2(400)[400])
        for n, k in ((200, 0), (200, 1), (300, 37)):
            with self.subTest(operation="stirling1", n=n, k=k):
                self.assertEqual(self.assertAgrees("stirling1", {"n": n, "k": k}, rising_row(n)[k]), (rising_row(n),))

    def test_full_size_inputs_fit_the_default_budget(self):
        """The requests at the family's limits complete and verify under the default limits, with values that the
        references or their own congruences confirm: Euler's recurrence as written here and Ramanujan's congruences
        for the table of p(5000), inclusion and exclusion for D(5900), Touchard's congruence modulo 17 primes for
        B(1500), and for the row of c(1000, .) its sum 1000!, its alternating sum 0, c(1000, 1) = 999!,
        c(1000, 999) = C(1000, 2) and the rising factorial modulo 8 primes (Fermat's little theorem)."""
        n = LARGEST["stirling1"]
        primes = [p for p in primes_below(LARGEST["bell"]) if p > 1400]
        self.assertEqual(len(primes), 17)
        requests = (("partitions", dict(n=LARGEST["partitions"]), 400_000),
                    ("derangements", dict(n=LARGEST["derangements"]), 60_000),
                    ("bell", dict(n=LARGEST["bell"]), 1_600_000),
                    ("stirling1", dict(n=n, k=n // 2), 1_100_000))
        for operation, args, ceiling in requests:
            with self.subTest(operation=operation, **args):
                with computation(fresh=True) as budget:  # The default limits, as for a request.
                    value, evidence = family.produce(operation, dict(args))
                self.assertLess(budget.work, ceiling)  # Measured 385,078, 50,072, 1,496,470 and 1,070,813 units.
                if operation == "partitions":
                    self.assertEqual(evidence, (pentagonal_table(args["n"]),))
                    self.assertTrue(ramanujan_congruences(evidence[0]))
                elif operation == "derangements":
                    self.assertEqual(value, inclusion_exclusion(args["n"]))
                elif operation == "bell":
                    self.assertTrue(bell_touchard(value, args["n"], primes))
                    self.assertFalse(bell_touchard(value + 1, args["n"], primes))
                else:
                    (row,) = evidence
                    self.assertEqual(value, row[n // 2])
                    self.assertEqual((sum(row), sum(c if i % 2 == 0 else -c for i, c in enumerate(row))),
                                     (factorial(n), 0))
                    self.assertEqual((row[0], row[1], row[n - 1], row[n]), (0, factorial(n - 1), comb(n, 2), 1))
                    for p in (2, 3, 5, 7, 11, 13, 31, 997):
                        self.assertEqual(tuple(c % p for c in row), rising_row_mod(n, p), p)
                start = time.perf_counter()
                result = support.run(operation, **args)
                self.assertLess(time.perf_counter() - start, 2 * FAST)
                self.assertVerified(result, value)
                self.assertEqual(result.steps[0].outputs[1], evidence)

    def test_work_is_charged_before_it_runs(self):
        """The producer's charges depend on the request alone: a budget one unit short refuses it, its exact size
        passes. Every step pays at least a unit, and a request with n = 0 costs exactly two (the call and its start);
        long integers cost more than short ones."""
        requests = (("partitions", dict(n=600), 600 * 20), ("derangements", dict(n=900), 2 * 899),
                    ("bell", dict(n=300), 300 * 299 // 2), ("stirling1", dict(n=150, k=40), 150 * 151))
        for operation, args, least in requests:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    value, evidence = family.produce(operation, dict(args))
                work = budget.work
                self.assertGreater(work, least)  # A unit per term, per pass or per addition at least.
                with self.assertRaises(ResourceLimitError), computation(ComputationLimits(max_work=work - 1),
                                                                        fresh=True):
                    family.produce(operation, dict(args))
                with computation(ComputationLimits(max_work=work), fresh=True) as budget:
                    self.assertEqual(family.produce(operation, dict(args)), (value, evidence))
                self.assertEqual(budget.work, work)
        for operation, args in (("partitions", dict(n=0)), ("derangements", dict(n=0)), ("derangements", dict(n=1)),
                                ("bell", dict(n=0)), ("bell", dict(n=1)), ("stirling1", dict(n=0, k=0)),
                                ("stirling1", dict(n=0, k=4))):
            with self.subTest(operation=operation, **args):
                with computation(fresh=True) as budget:
                    family.produce(operation, dict(args))
                self.assertEqual(budget.work, 2)
        # D(m) has about m log2(m) bits: past 8,192 bits each pass pays for its length, so that the work per step
        # grows; below, a step costs its two units.
        charged = []
        for n in (500, 1000, 5900):
            with computation(fresh=True) as budget:
                family.produce("derangements", {"n": n})
            charged.append(budget.work)
        self.assertEqual(charged[0], 1 + 1 + 2 * 499)
        self.assertGreater(charged[2] - charged[1], 2 * 2 * (5900 - 1000))

    def test_family_module_imports_and_forms_no_power(self):
        """The family module imports at its top only modules that `import mathforge` loads, and nowhere the checker
        layer, the lean path, the MCP layer or generated code. It forms no power, with the builtin pow or with **:
        every product is a charged step of a loop."""
        package = family.__name__.split(".")[:-1]  # mathforge.areas.discrete

        def absolute(node: ast.ImportFrom) -> str:
            """The module that an import names, a relative one resolved against the family's package."""
            base = package[:len(package) - node.level + 1] if node.level else []
            return ".".join([*base, node.module or ""])

        tree = ast.parse(inspect.getsource(family))
        top = {absolute(node) for node in tree.body if isinstance(node, ast.ImportFrom)}
        self.assertEqual(top, {"__future__", "collections.abc", "mathforge.errors", "mathforge.limits",
                               "mathforge.registry"})
        self.assertFalse([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                name = absolute(node)
                self.assertFalse(any(part in name.split(".") for part in ("certify", "lean", "mcp", "_generated")),
                                 name)
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        self.assertEqual([node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                          and isinstance(node.op, ast.Pow)], [])
