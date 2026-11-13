"""The independent checker of mathforge.areas.discrete.counting (mathforge.certify.discrete.counting), tested on its own
before the family's producer exists.

Every true claim is computed here, never by the producer: by brute force on small inputs (partitions listed part by
part, permutations enumerated with their fixed points and their cycles, set partitions as restricted growth strings)
and by short references of this module that use other algorithms than the checker: partition numbers by the coin
recurrence (one part size at a time), which also checks Euler's recurrence as written here with the closed forms
k(3k - 1)/2 and k(3k + 1)/2; derangements by D(n) = (n - 1)(D(n - 1) + D(n - 2)) and by inclusion and exclusion; Bell
numbers by Bell's triangle; Stirling rows by multiplying out x(x + 1)...(x + n - 1). The evidence follows the layouts
of the family's docstring. The checker answers False to every corrupted claim, never with an exception, and refuses an
oversized forgery within FAST seconds under the default budget.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import enum
from functools import lru_cache
import inspect
from itertools import accumulate, permutations
from math import factorial
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.discrete import counting as family
from mathforge.certify.discrete import counting as checker
from mathforge.contracts import CheckStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
# The resource limits of the family's docstring: the card's for partitions and derangements, a margin below the
# checker's capacity under the default budget for bell and stirling1.
CAPS = {"partitions": 5000, "derangements": 5900, "bell": 1500, "stirling1": 1000}

Claim = tuple[str, dict[str, object], object, object]  # (operation, args, value, evidence)


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these, or from brute force.

def partitions_brute(n: int, largest: int | None = None) -> int:
    """The partitions of n into parts of at most largest (any size by default), listed part by part."""
    if n == 0:
        return 1
    top = n if largest is None else min(n, largest)
    return sum(partitions_brute(n - part, part) for part in range(1, top + 1))


@lru_cache(maxsize=None)
def coin_table(n: int) -> tuple[int, ...]:
    """(p(0), ..., p(n)) by the coin recurrence: admitting the part size k adds to the count of each m the count of
    m - k, a running sum along each residue class modulo k."""
    p = [1] + [0] * n
    for part in range(1, n + 1):
        for residue in range(part):
            p[residue::part] = accumulate(p[residue::part])
    return tuple(p)


def pentagonal_recurrence(n: int, sign: Callable[[int], int] | None = None, changes: dict[int, int] | None = None
                          ) -> tuple[int, ...]:
    """(t_0, ..., t_n) by Euler's recurrence with the pentagonal numbers from their closed forms k(3k - 1)/2 and
    k(3k + 1)/2 and the signs (-1)**(k + 1): the partition numbers. sign(index) replaces the sign of the index-th
    pentagonal number (1, 2, 5, 7, ... in increasing order), and changes adds to some t_m before the recurrence goes
    on: internally consistent tables that are not p's."""
    offsets = []
    k = 1
    while k * (3 * k - 1) // 2 <= n:
        offsets += [(k * (3 * k - 1) // 2, 1 if k % 2 else -1), (k * (3 * k + 1) // 2, 1 if k % 2 else -1)]
        k += 1
    if sign is not None:
        offsets = [(offset, sign(index)) for index, (offset, _) in enumerate(offsets)]
    table = [1]
    for m in range(1, n + 1):
        table.append(sum(s * table[m - offset] for offset, s in offsets if offset <= m) + (changes or {}).get(m, 0))
    return tuple(table)


@lru_cache(maxsize=None)
def euler_table(n: int) -> tuple[int, ...]:
    """(p(0), ..., p(n)) by pentagonal_recurrence; test_references_agree checks it against the coin recurrence."""
    return pentagonal_recurrence(n)


def distinct_table(n: int) -> tuple[int, ...]:
    """(q(0), ..., q(n)), the partitions into distinct parts: consistent with its own recurrence, and not p."""
    q = [1] + [0] * n
    for part in range(1, n + 1):
        for m in range(n, part - 1, -1):
            q[m] += q[m - part]
    return tuple(q)


def divisor_sum(k: int) -> int:
    """sigma(k), the sum of the divisors of k, by trial of every d up to k."""
    return sum(d for d in range(1, k + 1) if k % d == 0)


def derangements_brute(n: int) -> int:
    """The permutations of range(n) without a fixed point, counted one by one."""
    return sum(all(image != point for point, image in enumerate(p)) for p in permutations(range(n)))


@lru_cache(maxsize=None)
def derangement_numbers(n: int) -> tuple[int, ...]:
    """(D(0), ..., D(n)) by D(m) = (m - 1)(D(m - 1) + D(m - 2)), D(0) = 1, D(1) = 0."""
    numbers = [1, 0]
    for m in range(2, n + 1):
        numbers.append((m - 1) * (numbers[-1] + numbers[-2]))
    return tuple(numbers[:n + 1])


def inclusion_exclusion(n: int) -> int:
    """D(n) = sum over i of (-1)**i n!/i!: the permutations that fix no point of any i-set."""
    return sum((-1) ** i * (factorial(n) // factorial(i)) for i in range(n + 1))


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


@lru_cache(maxsize=None)
def bell_numbers(n: int) -> tuple[int, ...]:
    """(B(0), ..., B(n)) by Bell's triangle: each row starts with the last entry of the row before, and each entry
    adds its left neighbour and the entry above that neighbour; B(m) starts row m."""
    numbers, row = [1], [1]
    for _ in range(n):
        new = [row[-1]]
        for entry in row:
            new.append(new[-1] + entry)
        row = new
        numbers.append(row[0])
    return tuple(numbers)


def cycle_count(permutation: Sequence[int]) -> int:
    seen, count = [False] * len(permutation), 0
    for start in range(len(permutation)):
        if not seen[start]:
            count += 1
            point = start
            while not seen[point]:
                seen[point] = True
                point = permutation[point]
    return count


def stirling_brute(n: int) -> tuple[int, ...]:
    """(c(n, 0), ..., c(n, n)): the permutations of range(n) counted by their number of cycles."""
    counts = [0] * (n + 1)
    for permutation in permutations(range(n)):
        counts[cycle_count(permutation)] += 1
    return tuple(counts)


@lru_cache(maxsize=None)
def stirling_row(n: int) -> tuple[int, ...]:
    """(c(n, 0), ..., c(n, n)), the coefficients of x(x + 1)...(x + n - 1), multiplied out one factor at a time."""
    row = [1]
    for m in range(n):  # Times (x + m).
        row = [m * row[0]] + [row[i - 1] + m * row[i] for i in range(1, len(row))] + [row[-1]]
    return tuple(row)


def evaluate(coefficients: Sequence[int], x: int) -> int:
    return sum(c * x ** i for i, c in enumerate(coefficients))


def plus(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    """The coefficients of the sum of two polynomials of the same length."""
    return tuple(a + b for a, b in zip(left, right, strict=True))


def times_linear(coefficients: Sequence[int], root: int) -> list[int]:
    """The coefficients of P(x) * (x - root), lowest first."""
    shifted = [0, *coefficients]
    return [shifted[i] - root * (coefficients[i] if i < len(coefficients) else 0) for i in range(len(shifted))]


def vanishing(roots: Sequence[int], length: int) -> tuple[int, ...]:
    """The coefficients of (x - 1) * prod over roots of (x - root), padded with zeros to length: a polynomial that is 0
    at 1 and at every root, so adding it to a row keeps those values."""
    coefficients = [1]
    for root in (1, *roots):
        coefficients = times_linear(coefficients, root)
    return tuple(coefficients) + (0,) * (length - len(coefficients))


def rising_times_flat(n: int, t: int, flat: int, constant: int) -> tuple[int, ...]:
    """The n + 1 coefficients of x(x + 1)...(x + t - 1) * Q(x) for Q = x**(n - t) + flat * (1 + x + ... +
    x**(n - t - 1)) + constant, with 0 < t < n: by sums over windows of the rising factorial's coefficients, without a
    polynomial product. Entries >= 0 and a top entry 1 for flat, constant >= 0; the polynomial vanishes at 0, -1, ...,
    -(t - 1), so a checker's first t divisions succeed, and its quotients have long coefficients throughout."""
    rising, d = stirling_row(t), n - t
    prefix = (0, *accumulate(rising))  # prefix[i] = rising[0] + ... + rising[i - 1]
    return tuple(flat * (prefix[min(i, t) + 1] - prefix[max(i - d + 1, 0)])
                 + (rising[i - d] if i >= d else 0) + (constant * rising[i] if i <= t else 0) for i in range(n + 1))


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values and evidence that the checker must refuse.

@dataclass(frozen=True)
class Count:
    """A record that holds a right answer. No value of the counting family is a record, so the checker refuses every
    record: this one, an impostor of it and one forged with object.__new__."""
    value: object


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


class Small(enum.IntEnum):
    """Small counts as an int enumeration: equal to them but no exact ints."""
    ZERO = 0
    ONE = 1
    TWO = 2


class Hostile:
    """Fails the test on any comparison or use as a number: the checker must test exact types before it compares."""

    def __eq__(self, other: object) -> bool:
        raise AssertionError("the checker compared an object of unknown type")

    __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __eq__

    def __index__(self) -> int:
        raise AssertionError("the checker used an object of unknown type as an integer")

    __int__ = __index__


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass for an evidence tuple, a table or a row.


def inexact(number: int) -> list[tuple[str, object]]:
    """Stand-ins for an exact int that are no exact int, some of them equal to it: a bool, a float, a str, None, a
    Rational, an int subclass, a Hostile object and, for a small number, an IntEnum."""
    small = number.bit_length() < 1000
    items = [("a bool", bool(number)), ("a float", float(number) if small else 0.5),
             ("a str", str(number) if small else "1"), ("None", None), ("a Rational", Rational(number)),
             ("an int subclass", Integer(number)), ("a hostile object", Hostile())]
    if 0 <= number <= 2:
        items.append(("an IntEnum", Small(number)))
    return items


def claimed(operation: str, args: dict[str, int], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence),
                          relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


def partitions_claim(n: int) -> Claim:
    table = coin_table(n) if n <= 2000 else euler_table(n)
    return "partitions", {"n": n}, table[n], (table,)


def derangements_claim(n: int) -> Claim:
    return "derangements", {"n": n}, derangement_numbers(n)[n], ()


def bell_claim(n: int) -> Claim:
    return "bell", {"n": n}, bell_numbers(n)[n], ()


def stirling1_claim(n: int, k: int) -> Claim:
    row = stirling_row(n)
    return "stirling1", {"n": n, "k": k}, row[k] if k <= n else 0, (row,)


class CountingCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    def assertTypesMatter(self, operation: str, args: dict[str, int], value: int, evidence: tuple) -> None:
        """The checker accepts a true claim, and refuses it once an argument or the value is no exact int (equal to it
        or not), the value is a record of any class, the evidence is a list, a tuple subclass, None, one item longer or
        one item shorter, or, for a table or a row, that tuple is a list or a tuple subclass, one of its entries is no
        exact int, or it is one entry shorter or longer."""
        self.assertAccepts(operation, args, value, evidence)
        cases: list[tuple[str, object, object, object]] = []
        for name, number in args.items():
            cases += [(f"argument {name} as {label}", dict(args, **{name: item}), value, evidence)
                      for label, item in inexact(number)]
        cases += [(f"the value as {label}", args, item, evidence) for label, item in inexact(value)]
        cases += [("the value as a record", args, Count(value), evidence),
                  ("the value as an impostor record", args, support.impostor(Count(value)), evidence),
                  ("the value as a forged record", args, support.forged(Count, value=value), evidence),
                  ("the value as a forged record without its field", args, support.forged(Count), evidence),
                  ("the value in a tuple", args, (value,), evidence),
                  ("the evidence as a hostile object", args, value, Hostile()),
                  ("a hostile evidence item", args, value, (Hostile(),)),
                  ("evidence as a list", args, value, list(evidence)),
                  ("evidence as a tuple subclass", args, value, Evidence(evidence)),
                  ("evidence None", args, value, None),
                  ("an extra evidence item", args, value, evidence + ((),))]
        if evidence:
            cases.append(("the evidence item missing", args, value, evidence[:-1]))
            inner = evidence[0]
            cases += [("the table as a list", args, value, (list(inner),)),
                      ("the table as a tuple subclass", args, value, (Evidence(inner),)),
                      ("the table one entry short", args, value, (inner[:-1],)),
                      ("the table one entry long", args, value, (inner + (inner[-1],),))]
            for index in sorted({0, 1, len(inner) // 2, len(inner) - 1} & set(range(len(inner)))):
                cases += [(f"entry {index} as {label}", args, value, (inner[:index] + (item,) + inner[index + 1:],))
                          for label, item in inexact(inner[index])]
        self.assertRejectsAll(operation, cases)

    # -----------------------------------------------------------------------------------------------------------
    # The family module and the references

    def test_family_declares_the_specs_of_the_card(self):
        n, k = Argument("n", "int", None, "int"), Argument("k", "int", None, "int")
        expected = (
            OperationSpec("partitions", "discrete", (n,), "int",
                          "Count the integer partitions of a nonnegative integer.", SCOPE, version=6),
            OperationSpec("derangements", "discrete", (n,), "int",
                          "Count the permutations of n items with no fixed point.", SCOPE, version=6),
            OperationSpec("bell", "discrete", (n,), "int",
                          "Compute the Bell number B(n), the number of partitions of a set of n items.", SCOPE,
                          version=6),
            OperationSpec("stirling1", "discrete", (n, k), "int",
                          "Compute the unsigned Stirling number of the first kind c(n, k).", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {})  # Every value is an int.
        self.assertEqual(family.CHECKER, "mathforge.certify.discrete.counting")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)

    def test_references_agree(self):
        """The references of this module agree with brute force and with each other."""
        self.assertEqual([partitions_brute(n) for n in range(26)], list(coin_table(25)))
        self.assertEqual(euler_table(2000), coin_table(2000))
        self.assertEqual(coin_table(1000)[1000], 24061467864032622473692149727991)  # p(1000), Hardy and Ramanujan.
        self.assertEqual([derangements_brute(n) for n in range(9)], list(derangement_numbers(8)))
        self.assertEqual([inclusion_exclusion(n) for n in range(80)], list(derangement_numbers(79)))
        self.assertEqual([set_partitions_brute(n) for n in range(10)], list(bell_numbers(9)))
        for n in range(8):
            self.assertEqual(stirling_brute(n), stirling_row(n))
        self.assertEqual(distinct_table(10), (1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_partitions_accepts_true_claims(self):
        for n, value in ((0, 1), (1, 1), (5, 7), (10, 42), (100, 190569292)):  # The card's known values.
            with self.subTest(n=n):
                operation, args, found, evidence = partitions_claim(n)
                self.assertEqual(found, value)
                self.assertAccepts(operation, args, found, evidence)
        self.assertEqual(partitions_claim(0)[3], ((1,),))
        for n in range(301):
            with self.subTest(n=n):
                self.assertAccepts(*partitions_claim(n))
        for n in (999, 1000, 1500, 2000):
            with self.subTest(n=n):
                self.assertAccepts(*partitions_claim(n))

    def test_derangements_accepts_true_claims(self):
        for n, value in ((0, 1), (1, 0), (2, 1), (4, 9), (10, 1334961)):  # The card's known values.
            with self.subTest(n=n):
                self.assertEqual(derangement_numbers(10)[n], value)
                self.assertAccepts("derangements", {"n": n}, value)
        for n in range(401):
            with self.subTest(n=n):
                self.assertAccepts(*derangements_claim(n))
        for n in (1000, 3001, 4000):
            with self.subTest(n=n):
                self.assertAccepts(*derangements_claim(n))

    def test_bell_accepts_true_claims(self):
        for n, value in ((0, 1), (1, 1), (2, 2), (3, 5), (5, 52), (10, 115975)):  # bell(5), bell(10) from the card.
            with self.subTest(n=n):
                self.assertEqual(bell_numbers(10)[n], value)
                self.assertAccepts("bell", {"n": n}, value)
        for n in range(151):
            with self.subTest(n=n):
                self.assertAccepts(*bell_claim(n))
        for n in (400, 777):
            with self.subTest(n=n):
                self.assertAccepts(*bell_claim(n))

    def test_stirling1_accepts_true_claims(self):
        # The card's two known values first.
        for n, k, value in ((4, 2, 11), (5, 3, 35), (0, 0, 1), (1, 1, 1), (1, 0, 0), (3, 1, 2)):
            with self.subTest(n=n, k=k):
                self.assertEqual(stirling_row(n)[k], value)
                self.assertAccepts(*stirling1_claim(n, k))
        self.assertEqual(stirling1_claim(0, 0)[3], ((1,),))
        for n in range(41):  # Every k up to n, and k > n with the value 0 and the full row.
            for k in range(n + 3):
                with self.subTest(n=n, k=k):
                    self.assertAccepts(*stirling1_claim(n, k))
        for n, k in ((5, 10 ** 6), (0, 7), (200, 0), (200, 1), (200, 100), (200, 200), (400, 33)):
            with self.subTest(n=n, k=k):
                self.assertAccepts(*stirling1_claim(n, k))

    def test_full_size_claims_fit_the_default_budget(self):
        """The true claims at the family's resource limits verify under the default budget, within FAST seconds."""
        claims = [partitions_claim(CAPS["partitions"]), derangements_claim(CAPS["derangements"]),
                  bell_claim(CAPS["bell"]), stirling1_claim(CAPS["stirling1"], CAPS["stirling1"] // 2),
                  stirling1_claim(CAPS["stirling1"], 1)]
        limit = ComputationLimits().max_work
        for operation, args, value, evidence in claims:
            with self.subTest(operation=operation, args=args):
                start = time.perf_counter()
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, evidence), True)
                self.assertLess(time.perf_counter() - start, FAST)
                self.assertLess(budget.work, limit)
        # D(5900), the longest number of these claims, fits max_integer_bits, as a verified Result needs.
        self.assertLessEqual(derangement_numbers(5900)[5900].bit_length(), ComputationLimits().max_integer_bits)

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_rejects_claims_on_inputs_outside_the_domain(self):
        cases: list[tuple[str, dict[str, int], object, object]] = []
        for n in (-1, -2, -10, -GIANT):
            cases += [("partitions", {"n": n}, value, evidence) for value, evidence in
                      ((1, ((1,),)), (0, ((),)), (1, ((1, 1),)), (0, ()))]
            cases += [(operation, {"n": n}, value, ()) for operation in ("derangements", "bell")
                      for value in (-1, 0, 1, 2)]
            cases += [("stirling1", {"n": n, "k": 0}, value, evidence) for value, evidence in
                      ((0, ((1,),)), (1, ((1,),)), (0, ((),)), (0, ()))]
        for n, k in ((3, -1), (0, -1), (5, -GIANT), (-1, -1)):
            row = stirling_row(max(n, 0))
            cases += [("stirling1", {"n": n, "k": k}, value, (row,)) for value in (0, 1, row[-1])]
        for index, (operation, args, value, evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):  # No giant int in a label: str() would refuse it.
                self.assertRejects(operation, args, value, evidence)

    def test_partitions_rejects_corrupted_claims(self):
        args, table = {"n": 10}, coin_table(10)
        self.assertEqual(table[10], 42)
        self.assertTypesMatter("partitions", args, 42, (table,))
        self.assertTypesMatter("partitions", {"n": 0}, 1, ((1,),))
        self.assertTypesMatter("partitions", {"n": 1}, 1, ((1, 1),))
        self.assertTypesMatter("partitions", {"n": 100}, 190569292, (coin_table(100),))
        distinct = distinct_table(10)
        doubled = tuple(2 * entry for entry in table)
        shifted = coin_table(11)[1:]
        changed = pentagonal_recurrence(10, changes={7: 1})  # p(7) + 1, and the recurrence goes on from there.
        alternating = pentagonal_recurrence(10, sign=lambda index: 1 if index % 2 == 0 else -1)
        late = pentagonal_recurrence(100, changes={100: 1})  # Wrong at the last entry only.
        self.assertNotEqual(alternating, table)
        self.assertEqual(changed[:7], table[:7])
        self.assertRejectsAll("partitions", [
            ("a wrong value", args, 41, (table,)),
            ("p(n - 1)", args, 30, (table,)),
            ("p(n + 1)", args, 56, (table,)),
            ("the value as a Rational, equal and not canonical", args, Rational(42), (table,)),
            ("a wrong value with a matching last entry", args, 43, (table[:-1] + (43,),)),
            ("the partitions into distinct parts", args, distinct[10], (distinct,)),
            ("the table doubled, with p(0) = 2", args, 84, (doubled,)),
            ("p(0) = 0", args, 42, ((0,) + table[1:],)),
            ("a negative p(0)", args, 42, ((-1,) + table[1:],)),
            ("the table of n + 1 shifted down", args, shifted[10], (shifted,)),
            ("an entry changed and the recurrence continued", args, changed[10], (changed,)),
            ("Euler's recurrence with alternating signs", args, alternating[10], (alternating,)),
            ("the last of 101 entries changed", {"n": 100}, late[100], (late,)),
            ("an entry negated", args, 42, (table[:5] + (-table[5],) + table[6:],)),
            ("two entries swapped", args, 42, (table[:3] + (table[4], table[3]) + table[5:],)),
            ("the table reversed", args, 1, (table[::-1],)),
            ("the table of n - 1", args, 30, (table[:-1],)),
            ("the table of n + 1", args, 56, (coin_table(11),)),
            ("the table of n + 1 with p(n)", args, 42, (coin_table(11),)),
            ("no evidence", args, 42, ()),
            ("the table unwrapped", args, 42, table),
            ("the table unwrapped, n = 0", {"n": 0}, 1, (1,)),
            ("the table twice", args, 42, (table, table)),
            ("an empty table", args, 42, ((),)),
            ("a nested table", args, 42, ((table,),)),
            ("the table and the value", args, 42, (table, 42)),
            ("a str table", args, 42, (str(table),)),
            ("a dict table", args, 42, (dict(enumerate(table)),)),
        ])

    def test_derangements_rejects_corrupted_claims(self):
        args, value = {"n": 10}, 1334961
        self.assertTypesMatter("derangements", args, value, ())
        self.assertTypesMatter("derangements", {"n": 0}, 1, ())
        self.assertTypesMatter("derangements", {"n": 1}, 0, ())
        self.assertTypesMatter("derangements", {"n": 2}, 1, ())
        full = factorial(10)
        large = derangements_claim(3000)
        self.assertRejectsAll("derangements", [
            ("one more", args, value + 1, ()),
            ("one less", args, value - 1, ()),
            ("D(n - 1)", args, derangement_numbers(9)[9], ()),
            ("D(n + 1)", args, derangement_numbers(11)[11], ()),
            ("D(n) + n!, exact at every step", args, value + full, ()),
            ("D(n) - n!", args, value - full, ()),
            ("D(n) + 2 n!", args, value + 2 * full, ()),
            ("n!, every permutation", args, full, ()),
            ("0", args, 0, ()),
            ("negated", args, -value, ()),
            ("the value as a Rational", args, Rational(value), ()),
            ("D(0) = 0", {"n": 0}, 0, ()),
            ("D(0) = 2", {"n": 0}, 2, ()),
            ("D(1) = 1", {"n": 1}, 1, ()),
            ("D(1) = -1", {"n": 1}, -1, ()),
            ("D(2) = 0", {"n": 2}, 0, ()),
            ("D(3000) + 1", large[1], large[2] + 1, ()),
            ("D(3000) + 3000!", large[1], large[2] + factorial(3000), ()),
            ("evidence where none belongs", args, value, ((),)),
            ("the value as evidence", args, value, (value,)),
            ("the table of D as evidence", args, value, (derangement_numbers(10),)),
            ("a str as evidence", args, value, "()"),
        ])

    def test_bell_rejects_corrupted_claims(self):
        args, value = {"n": 10}, 115975
        self.assertTypesMatter("bell", args, value, ())
        self.assertTypesMatter("bell", {"n": 0}, 1, ())
        self.assertTypesMatter("bell", {"n": 1}, 1, ())
        self.assertTypesMatter("bell", {"n": 2}, 2, ())
        triangle_row = (115975, 137122, 162409)  # The start of row 10 of Bell's triangle.
        large = bell_claim(300)
        self.assertRejectsAll("bell", [
            ("one more", args, value + 1, ()),
            ("one less, the partitions into fewer than n blocks", args, value - 1, ()),
            ("B(n - 1)", args, bell_numbers(9)[9], ()),
            ("B(n + 1)", args, bell_numbers(11)[11], ()),
            ("2**(n - 1) - 1, below the size bound", args, 2 ** 9 - 1, ()),
            ("2**(n - 1), the partitions into at most two blocks", args, 2 ** 9, ()),
            ("n!", args, factorial(10), ()),
            ("n! + 1", args, factorial(10) + 1, ()),
            ("2**(n * bit_length(n)), above the size bound", args, 2 ** 40, ()),
            ("0", args, 0, ()),
            ("negated", args, -value, ()),
            ("the value as a Rational", args, Rational(value), ()),
            ("B(0) = 0", {"n": 0}, 0, ()),
            ("B(0) = 2", {"n": 0}, 2, ()),
            ("B(1) = 2", {"n": 1}, 2, ()),
            ("B(2) = 1", {"n": 2}, 1, ()),
            ("B(300) + 1", large[1], large[2] + 1, ()),
            ("B(300) - 1", large[1], large[2] - 1, ()),
            ("B(299)", large[1], bell_numbers(299)[299], ()),
            ("evidence where none belongs", args, value, ((),)),
            ("the value as evidence", args, value, (value,)),
            ("a row of Bell's triangle as evidence", args, value, (triangle_row,)),
            ("a str as evidence", args, value, "()"),
        ])

    def test_stirling1_rejects_corrupted_claims(self):
        n, k = 8, 3
        args, row = {"n": n, "k": k}, stirling_row(n)
        value = row[k]
        self.assertEqual(sum(row), factorial(n))
        self.assertTypesMatter("stirling1", args, value, (row,))
        self.assertTypesMatter("stirling1", {"n": 0, "k": 0}, 1, ((1,),))
        self.assertTypesMatter("stirling1", {"n": 1, "k": 0}, 0, ((0, 1),))
        self.assertTypesMatter("stirling1", {"n": 3, "k": 5}, 0, (stirling_row(3),))
        # Forged rows that keep P(1) = n! and the first values P(0), P(-1), ...: adding plus or minus (x - 1) times
        # x(x + 1)...(x + t - 1) keeps P(1) and the points 0, ..., -(t - 1), and changes P(-t); the sign is the one
        # that keeps every entry nonnegative.
        forged = {}
        for t in (0, 1, 3, n - 1):
            added = vanishing(tuple(-j for j in range(t)), n + 1)
            candidates = [plus(row, tuple(sign * entry for entry in added)) for sign in (1, -1)]
            candidate = next(c for c in candidates if all(entry >= 0 for entry in c))
            self.assertEqual(sum(candidate), factorial(n))
            self.assertTrue(all(evaluate(candidate, -j) == 0 for j in range(t)))
            self.assertNotEqual(evaluate(candidate, -t), 0)
            forged[t] = candidate
        signed = tuple((-1) ** (n - i) * entry for i, entry in enumerate(row))
        second_kind = (0, 1, 127, 966, 1701, 1050, 266, 28, 1)  # S(8, i), the Stirling numbers of the second kind.
        self.assertEqual(sum(second_kind), bell_numbers(8)[8])
        self.assertRejectsAll("stirling1", [
            ("c(n, k - 1)", args, row[k - 1], (row,)),
            ("c(n, k + 1)", args, row[k + 1], (row,)),
            ("one more", args, value + 1, (row,)),
            ("0", args, 0, (row,)),
            ("the value as a Rational", args, Rational(value), (row,)),
            ("S(n, k), the second kind, with its row", args, second_kind[k], (second_kind,)),
            ("S(n, k) with the true row", args, second_kind[k], (row,)),
            ("the signed row", args, signed[k], (signed,)),
            ("P(0) = 1, P(1) = n! kept", args, forged[0][k], (forged[0],)),
            ("P(0) = 0, P(-1) = 2", args, forged[1][k], (forged[1],)),
            ("right at 0, -1, -2, wrong at -3", args, forged[3][k], (forged[3],)),
            ("right at every point but -(n - 1)", args, forged[n - 1][k], (forged[n - 1],)),
            ("the row doubled, all the roots kept", args, 2 * value, (tuple(2 * entry for entry in row),)),
            ("the row reversed, P(1) = n! kept", args, row[n - k], (row[::-1],)),
            ("the row of n - 1 and a 0", args, stirling_row(n - 1)[k], (stirling_row(n - 1) + (0,),)),
            ("the row of n + 1", args, stirling_row(n + 1)[k], (stirling_row(n + 1),)),
            ("the row without its top entry", args, value, (row[:-1],)),
            ("the row with an extra 0", args, value, (row + (0,),)),
            ("a negative entry", args, value, (row[:5] + (-row[5],) + row[6:],)),
            ("an empty row", args, value, ((),)),
            ("no evidence", args, value, ()),
            ("the row unwrapped", args, value, row),
            ("the row twice", args, value, (row, row)),
            ("a nested row", args, value, ((row,),)),
            ("k > n, a value of 1", {"n": 3, "k": 5}, 1, (stirling_row(3),)),
            ("k > n, the row padded to k + 1", {"n": 3, "k": 5}, 0, (stirling_row(3) + (0, 0),)),
            ("k > n, no row", {"n": 3, "k": 5}, 0, ()),
            ("k > n, an empty row", {"n": 3, "k": 5}, 0, ((),)),
            ("k > n, a wrong row", {"n": 3, "k": 5}, 0, ((0, 2, 3, 2),)),
            ("n = 0, a value of 0", {"n": 0, "k": 0}, 0, ((1,),)),
            ("n = 0, the row (0,)", {"n": 0, "k": 0}, 0, ((0,),)),
            ("n = 0, the row (2,)", {"n": 0, "k": 0}, 2, ((2,),)),
        ])

    def test_rejects_wrong_operations_and_arguments(self):
        operation, args, value, evidence = stirling1_claim(5, 3)
        self.assertAccepts(operation, args, value, evidence)
        self.assertRejectsAll("stirling1", [
            ("arguments as pairs", tuple(args.items()), value, evidence),
            ("arguments as a list of pairs", list(args.items()), value, evidence),
            ("arguments as a read-only mapping", MappingProxyType(args), value, evidence),
            ("None as arguments", None, value, evidence),
            ("an argument missing", {"n": 5}, value, evidence),
            ("an extra argument", {"n": 5, "k": 3, "m": 1}, value, evidence),
            ("an argument renamed", {"n": 5, "j": 3}, value, evidence),
            ("the arguments swapped", {"n": 3, "k": 5}, value, evidence),
        ])
        self.assertRejects("stirling1", Hostile(), value, evidence)
        self.assertRejects("stirling1", {"n": Hostile(), "k": 3}, value, evidence)
        for name in ("stirling", "Stirling1", "stirling1 ", "stirling_1", "partitions", "bell", "", None, 7,
                     ("stirling1",), b"stirling1", ["stirling1"], Hostile()):
            with self.subTest(operation=name):
                self.assertRejects(name, args, value, evidence)
        # A true claim of one operation under the name of another with the same arguments.
        self.assertRejects("bell", *derangements_claim(10)[1:])
        self.assertRejects("derangements", *bell_claim(10)[1:])
        self.assertRejects("partitions", {"n": 10}, 1334961, ())
        self.assertRejects("derangements", *partitions_claim(10)[1:])
        self.assertRejects("bell", *partitions_claim(10)[1:])
        self.assertRejects("partitions", {"n": 5, "k": 3}, value, evidence)
        self.assertRejects("bell", {"n": 3}, 5, (stirling_row(3),))
        self.assertRejects("stirling1", {"n": 5}, 7, (coin_table(5),))
        self.assertRejects("stirling1", {"n": 5, "k": 5}, 7, (coin_table(5),))  # p(0..5) has six entries too.

    def test_oversized_forgeries_fail_fast(self):
        table, row = coin_table(10), stirling_row(5)
        prefix = euler_table(CAPS["partitions"])
        full = factorial(CAPS["derangements"])
        cases: list[Claim] = [
            ("partitions", {"n": 10}, GIANT, (table,)),
            ("partitions", {"n": 10}, GIANT, (table[:-1] + (GIANT,),)),
            ("partitions", {"n": 10}, 42, ((1,) + (GIANT,) * 10,)),
            ("partitions", {"n": 10}, 42, (table * 100_000,)),
            ("partitions", {"n": 10}, 42, (table,) * 100_000),
            ("partitions", {"n": 10}, 42, (table, (0,) * 100_000)),
            ("partitions", {"n": support.HUGE}, 1, ((1,),)),
            ("partitions", {"n": 99_999}, 0, ((1,) + (0,) * 99_999,)),
            ("partitions", {"n": 99_999}, 1, ((1,) * 100_000,)),
            # The true table up to the cap, then zeros: refused at the entry after the cap.
            ("partitions", {"n": 99_999}, 0, (prefix + (0,) * (99_999 - CAPS["partitions"]),)),
            ("derangements", {"n": 10}, GIANT, ()),
            ("derangements", {"n": 10}, -GIANT, ()),
            ("derangements", {"n": 10}, 1334961, (GIANT,)),
            ("derangements", {"n": 10}, 1334961, (0,) * 100_000),
            ("derangements", {"n": 5000}, support.HUGE, ()),
            ("derangements", {"n": 10 ** 6}, support.HUGE + 1, ()),
            ("derangements", {"n": GIANT}, 0, ()),
            ("derangements", {"n": GIANT}, 1, ()),
            # Exact at all 5,900 steps, ending at x_0 = 2.
            ("derangements", {"n": CAPS["derangements"]}, derangement_numbers(5900)[5900] + full, ()),
            ("bell", {"n": 10}, GIANT, ()),
            ("bell", {"n": 10}, -GIANT, ()),
            ("bell", {"n": 10}, 115975, (GIANT,)),
            ("bell", {"n": 10}, 115975, (0,) * 100_000),
            ("bell", {"n": GIANT}, 1, ()),
            ("bell", {"n": 10 ** 6}, 5, ()),
            ("bell", {"n": 10 ** 6}, GIANT, ()),
            # B(0) = B(1) = 1: refused by the size test, before the product n! v could meet max_integer_bits.
            ("bell", {"n": 0}, GIANT, ()),
            ("bell", {"n": 0}, support.HUGE, ()),
            ("bell", {"n": 0}, 2, ()),
            ("bell", {"n": 1}, GIANT, ()),
            ("stirling1", {"n": 5, "k": 3}, GIANT, (row,)),
            ("stirling1", {"n": 5, "k": 3}, 35, (row[:-1] + (GIANT,),)),
            ("stirling1", {"n": 5, "k": 3}, 35, ((GIANT,) * 6,)),
            ("stirling1", {"n": 5, "k": 3}, 35, (row * 200_000,)),
            ("stirling1", {"n": 5, "k": 3}, 35, (row,) * 100_000),
            ("stirling1", {"n": 5, "k": GIANT}, 1, (row,)),
            ("stirling1", {"n": GIANT, "k": 3}, 0, (row,)),
            ("stirling1", {"n": 99_999, "k": 3}, 0, ((0,) * 99_999 + (1,),)),
            ("stirling1", {"n": 99_999, "k": 99_999}, 1, ((0,) * 99_999 + (1,),)),
        ]
        for index, (operation, args, value, evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)

    def test_forged_rows_at_the_cap_cost_no_more_than_the_true_row(self):
        """Rows x(x + 1)...(x + t - 1) * Q(x) at the cap of stirling1 pass the checker's first t divisions with long
        quotients. With P(1) = n! kept, the checker refuses them within the default budget, at no more work than the
        true row (a unit or two per entry more for the pass over longer entries); with P(1) != n!, before any
        division."""
        n = CAPS["stirling1"]
        _, args, value, evidence = stirling1_claim(n, 1)
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation("stirling1", args, value, evidence), True)
        true_work, full = budget.work, factorial(n)
        for t in (n // 2, 9 * n // 10):
            flat, constant = divmod(full // factorial(t) - 1, n - t)  # Q(1) = n!/t!, so P(1) = n!.
            kept = rising_times_flat(n, t, flat, constant)
            # Q = x**(n - t) + M (1 + ... + x**(n - t - 1)) with the largest M that keeps every entry within the
            # checker's size test: P(1) is far above n!.
            longest = max(stirling_row(t)).bit_length()
            crude = rising_times_flat(n, t, 2 ** (n * n.bit_length() - longest - (n - t).bit_length() - 1) - 1, 0)
            for label, row in (("P(1) = n! kept", kept), ("P(1) far above n!", crude)):
                with self.subTest(t=t, case=label):
                    self.assertEqual(row[n], 1)
                    self.assertGreaterEqual(min(row), 0)
                    self.assertLessEqual(max(row).bit_length(), n * n.bit_length())
                    self.assertNotEqual(row, evidence[0])
                    self.assertEqual(sum(row) == full, row is kept)
                    start = time.perf_counter()
                    with computation(fresh=True) as budget:  # The default budget.
                        self.assertIs(checker.check_operation("stirling1", args, row[1], (row,)), False)
                    self.assertLess(time.perf_counter() - start, FAST)
                    if row is kept:
                        self.assertGreater(budget.work, n * n // 4)  # Some divisions ran.
                        self.assertLessEqual(budget.work, true_work + 2 * (n + 1))
                    else:
                        self.assertLess(budget.work, 20 * (n + 1))  # Refused before the first division.

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_before_it_runs(self):
        """Every check is charged, and the charges depend on the claim alone: a budget one unit short of a check
        refuses it with ResourceLimitError, its exact size passes. A large claim stops at once under a small budget."""
        # The least work of each claim: a unit per step of its main loops (for stirling1, the n(n + 1)/2 steps of
        # its divisions, which each division pays for before it runs).
        claims = [(partitions_claim(600), 16_000), (derangements_claim(900), 3 * 900), (bell_claim(300), 300),
                  (stirling1_claim(150, 40), 150 * 151 // 2)]
        for (operation, args, value, evidence), least in claims:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, evidence), True)
                work = budget.work
                self.assertGreater(work, least)  # Every step of every loop is charged.
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, value, evidence, limits=ComputationLimits(max_work=work))
        large = [partitions_claim(CAPS["partitions"]), derangements_claim(CAPS["derangements"]),
                 bell_claim(CAPS["bell"]), stirling1_claim(CAPS["stirling1"], CAPS["stirling1"] // 2)]
        start = time.perf_counter()
        for operation, args, value, evidence in large:
            with self.subTest(operation=operation, budget="small"):
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence,
                                    limits=ComputationLimits(max_work=2_000))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_products_respect_max_integer_bits(self):
        """The products of the bell and stirling1 checks pass Budget.product: under a small max_integer_bits a claim
        whose check needs longer integers runs out of the budget instead of forming them."""
        small = ComputationLimits(max_integer_bits=1_000)
        for operation, args, value, evidence in (bell_claim(200), stirling1_claim(200, 3)):
            with self.subTest(operation=operation):
                self.assertAccepts(operation, args, value, evidence)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence, limits=small)

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_divisor_identity_guards_the_pentagonal_generation(self):
        """A checker whose own pentagonal numbers or signs went wrong would accept the table of its wrong recurrence;
        the identity n p(n) = sum sigma(k) p(n - k) at m = n refuses that table."""
        original = checker._pentagonal

        def alternating(budget, n):  # The signs + - + - ... instead of + + - - ...
            offsets, signs = original(budget, n)
            return offsets, [1 if index % 2 == 0 else -1 for index in range(len(signs))]

        def skipping(budget, n):  # The pentagonal number 5 left out.
            offsets, signs = original(budget, n)
            return ([offset for offset in offsets if offset != 5],
                    [sign for offset, sign in zip(offsets, signs) if offset != 5])

        tables = {"alternating signs": (alternating, pentagonal_recurrence(30, sign=lambda i: 1 if i % 2 == 0 else -1)),
                  "a pentagonal number missing": (skipping, pentagonal_recurrence(
                      30, sign=lambda i: 0 if i == 2 else (1 if i // 2 % 2 == 0 else -1)))}
        for label, (generation, wrong) in tables.items():
            with self.subTest(case=label):
                self.assertNotEqual(wrong[30], euler_table(30)[30])
                self.assertNotEqual(30 * wrong[30], sum(divisor_sum(k) * wrong[30 - k] for k in range(1, 31)))
                with patch.object(checker, "_pentagonal", generation):
                    self.assertRejects("partitions", {"n": 30}, wrong[30], (wrong,))
                    with patch.object(checker, "_divisor_identity", lambda budget, n, table: True):
                        self.assertAccepts("partitions", {"n": 30}, wrong[30], (wrong,))  # The identity refused it.
        self.assertAccepts(*partitions_claim(30))

    def test_the_checker_forms_no_power_and_imports_no_producer(self):
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])  # Every power is a chain of charged products.
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imported, {"__future__", "collections.abc", "mathforge.limits", "mathforge.trusted.integers"})

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layouts
        survive the format, and a wrong value fails the mathematics check."""
        claims = [
            (*partitions_claim(100), 190569291),
            (*derangements_claim(10), 1334960),
            (*bell_claim(10), 115976),
            (*stirling1_claim(5, 3), 36),
            (*stirling1_claim(3, 5), 1),
        ]
        for operation, args, value, evidence, wrong in claims:
            with self.subTest(operation=operation, args=args):
                self.roundtrip(claimed(operation, args, value, evidence))
                report = support.statuses(mf.verify(claimed(operation, args, wrong, evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
