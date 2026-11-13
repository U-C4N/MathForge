"""The powers family of mathforge.linalg: matrix_power.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register linalg powers

Every expected value comes from the card or from a reference of this module, written here and slow on purpose
(repeated multiplication with fractions.Fraction, the plain Fibonacci recurrence) or from closed forms (Jordan blocks,
unitriangular, nilpotent, idempotent, permutation and diagonal matrices); never from the producer.
tests/v09/linalg/test_powers_checker.py tests the checker on its own. Matrix.__pow__ (M ** k), the producer's
value without the independent check, and the message of Matrix * Matrix, which names @, are tested here as well.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from math import comb
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.linalg import powers as family
from mathforge.contracts import CheckStatus, ErrorInfo, ExecutionStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.matrices import trusted_matrix
from mathforge.model import Rational

from .. import support

FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
BITS = ComputationLimits().max_integer_bits  # 65,536: the longest integer of a value, and the card's bound on B.
WIDE = ComputationLimits(max_integer_bits=300_000, max_matrix_entries=1_000_000)  # For values past the limits.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # A float, a bool, a str, None and two containers.
M_STAND_INS = STAND_INS + (((1, 0), (0, 1)), [[1, 0], [0, 1]], Rational(1), 1)  # Rows without the Matrix, numbers.
K_STAND_INS = STAND_INS + (False, 2.0, Rational(3))

Number = int | Fraction | Rational
Rows = list[list[Fraction]]

# The known values of the card: (M, k, M**k).
KNOWN = (
    ([[1, 1], [1, 0]], 10, [[89, 55], [55, 34]]),
    ([[2, 0], [0, 3]], 0, [[1, 0], [0, 1]]),
    ([[0, 1], [0, 0]], 2, [[0, 0], [0, 0]]),
    ([[Fraction(1, 2), 0], [0, 2]], 3, [[Fraction(1, 8), 0], [0, 8]]),
)


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

def frac(number: Number) -> Fraction:
    """An exact number of the tests as a Fraction."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def exact(number: Number) -> int | Rational:
    """The number as a Matrix entry is given: an int when it is integral, else a Rational."""
    number = frac(number)
    return number.numerator if number.denominator == 1 else Rational(number.numerator, number.denominator)


def fractions_of(rows: Iterable[Iterable[Number]]) -> Rows:
    """Rows of exact numbers as rows of Fractions."""
    return [[frac(number) for number in row] for row in rows]


def matrix(rows: Iterable[Iterable[Number]], ncols: int | None = None) -> mf.Matrix:
    """The Matrix of rows of exact numbers; ncols is the length of the first row by default, 0 without rows."""
    rows = [list(row) for row in rows]
    if ncols is None:
        ncols = len(rows[0]) if rows else 0
    return mf.Matrix(tuple(tuple(exact(number) for number in row) for row in rows), ncols=ncols)


def wide(rows: Iterable[Iterable[Number]], ncols: int | None = None) -> mf.Matrix:
    """matrix(rows), built inside a budget that admits integers and sizes past the default limits."""
    with computation(WIDE, fresh=True):
        return matrix(rows, ncols)


def identity(n: int) -> Rows:
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def multiply(left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]) -> Rows:
    """The product of two square matrices of Fractions of one size, entry by entry."""
    n = len(right)
    return [[sum((row[m] * right[m][j] for m in range(n)), Fraction(0)) for j in range(n)] for row in left]


def naive_power(rows: Rows, k: int) -> Rows:
    """rows**k as the identity times rows, k times over: slow on purpose, and no addition chain at all."""
    result = identity(len(rows))
    for _ in range(k):
        result = multiply(result, rows)
    return result


def integer_product(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> list[list[int]]:
    """The product of two square integer matrices, for the large integer references."""
    columns = list(zip(*right))
    return [[sum(a * b for a, b in zip(row, column)) for column in columns] for row in left]


def fibonacci_power(k: int) -> list[list[int]]:
    """[[1, 1], [1, 0]]**k = [[F(k + 1), F(k)], [F(k), F(k - 1)]] for k >= 1, by the plain recurrence
    F(m + 1) = F(m) + F(m - 1) from F(0) = 0 and F(1) = 1."""
    before, current = 0, 1  # F(m - 1), F(m) for m = 1.
    for _ in range(k - 1):
        before, current = current, before + current
    return [[before + current, current], [current, before]]


def forged_matrix(rows: object, ncols: object) -> mf.Matrix:
    """A Matrix made without its constructor, whose fields may break the canonical form."""
    return support.forged(mf.Matrix, rows=rows, ncols=ncols)


def forged_rational(numerator: object, denominator: object) -> Rational:
    """A Rational made without its constructor: it may be out of lowest terms or have a bad denominator."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def fits(rows: Rows, bits: int) -> bool:
    """Whether every numerator and denominator of rows of Fractions has at most bits bits."""
    return all(entry.numerator.bit_length() <= bits and entry.denominator.bit_length() <= bits
               for row in rows for entry in row)


class Integer(int):
    """An int subclass, equal to the right exponent but no exact int."""


class SubMatrix(mf.Matrix):
    """A Matrix subclass with the right rows: no transport value."""


class PowersTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: mf.Matrix) -> None:
        """A completed v6 Result of that value, with the evidence () and its four checks, which mf.verify accepts."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertEqual(result.value, value)
        self.assertIs(type(result.value), mf.Matrix)
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertCanonical(self, value: object, n: int) -> None:
        """value is a Matrix as its constructor builds it: n tuples of n Rationals in lowest terms, ncols n."""
        self.assertIs(type(value), mf.Matrix)
        self.assertIs(type(value.rows), tuple)
        self.assertEqual((len(value.rows), value.ncols), (n, n))
        for row in value.rows:
            self.assertIs(type(row), tuple)
            self.assertEqual(len(row), n)
            for entry in row:
                self.assertIs(type(entry), Rational)
                self.assertEqual(entry, Rational(entry.numerator, entry.denominator))  # Lowest terms already.

    def assertAgrees(self, rows: Iterable[Iterable[Number]], k: int, expected: Iterable[Iterable[Number]]) -> None:
        """produce() answers expected for M = rows, with the evidence (), and the checker accepts the answer."""
        rows = [list(row) for row in rows]
        args = {"M": matrix(rows, len(rows)), "k": k}
        value, evidence = family.produce("matrix_power", dict(args))
        self.assertEqual(value, matrix(expected, len(rows)))
        self.assertEqual(evidence, ())
        self.assertCanonical(value, len(rows))
        self.assertIs(support.verdict(family, "matrix_power", args, value, evidence), True)

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for rows, k, expected in KNOWN:
            with self.subTest(rows=rows, k=k):
                self.assertEqual(naive_power(fractions_of(rows), k), fractions_of(expected))  # The card, by reference.
                result = support.run("matrix_power", M=matrix(rows), k=k)
                self.assertVerified(result, matrix(expected))
                self.roundtrip(result)
        half = Fraction(1, 2)
        more = (([[1, 2], [2, 4]], 0, identity(2)),  # A singular M to the power 0.
                ([[Fraction(-2, 3), Fraction(1, 5)], [7, 0]], 1, [[Fraction(-2, 3), Fraction(1, 5)], [7, 0]]),
                ([[1, 1], [1, 0]], 100, fibonacci_power(100)),
                ([[half, 1], [0, half]], 5, [[half ** 5, 5 * half ** 4], [0, half ** 5]]),  # A Jordan block.
                ([[0, 1, 0], [0, 0, 1], [1, 0, 0]], 3, identity(3)))  # A cyclic permutation.
        for rows, k, expected in more:
            with self.subTest(rows=rows, k=k):
                self.assertEqual(naive_power(fractions_of(rows), k), fractions_of(expected))
                result = support.run("matrix_power", M=matrix(rows), k=k)
                self.assertVerified(result, matrix(expected))
                self.roundtrip(result)
        for k in (0, 1, 7, 10 ** 40):  # The 0 x 0 matrix is every power of itself.
            with self.subTest(rows=[], k=k):
                result = support.run("matrix_power", M=mf.Matrix(()), k=k)
                self.assertVerified(result, mf.Matrix((), ncols=0))
                self.roundtrip(result)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every invalid input of the card: a non-square M and k < 0. produce() tests the types
        itself too."""
        args = dict(M=matrix([[1, 1], [1, 0]]), k=10)
        for name, items in (("M", M_STAND_INS), ("k", K_STAND_INS)):
            for item in items:
                with self.subTest(argument=name, item=item):
                    self.assertInvalidInput(support.run("matrix_power", **dict(args, **{name: item})))
        flat, tall = matrix([[1, 2, 3], [4, 5, 6]]), matrix([[1, 2], [3, 4], [5, 6]])
        no_rows, no_columns = mf.Matrix((), ncols=3), mf.Matrix(((), ()))
        unimodular = matrix([[2, 1], [1, 1]])  # Invertible: still no negative power, which belongs to inverse.
        invalid = [(flat, 1, "square matrix M"), (flat, 0, "square matrix M"), (tall, 2, "square matrix M"),
                   (no_rows, 0, "square matrix M"), (no_columns, 1, "square matrix M"),
                   (unimodular, -1, "use inverse for negative powers"), (unimodular, -2, "use inverse"),
                   (matrix([[1]]), -1, "k >= 0"), (mf.Matrix(()), -1, "k >= 0"), (args["M"], -10 ** 30, "k >= 0")]
        for M, k, text in invalid:
            with self.subTest(shape=M.shape, k=k):
                result = support.run("matrix_power", M=M, k=k)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
        # A Matrix forged without its constructor, through run(): invalid input as well.
        one, nought = Rational(1), Rational(0)
        forged = {"a float entry": forged_matrix(((one, 0.5), (nought, one)), 2),
                  "an int entry": forged_matrix(((one, 1), (nought, one)), 2),
                  "a bool entry": forged_matrix(((one, True), (nought, one)), 2),
                  "a str entry": forged_matrix(((one, "1"), (nought, one)), 2),
                  "None as an entry": forged_matrix(((one, None), (nought, one)), 2),
                  "a Fraction entry": forged_matrix(((one, Fraction(1, 2)), (nought, one)), 2),
                  "a zero denominator": forged_matrix(((one, forged_rational(1, 0)), (nought, one)), 2),
                  "a negative denominator": forged_matrix(((one, forged_rational(1, -2)), (nought, one)), 2),
                  "a bool numerator": forged_matrix(((one, forged_rational(True, 2)), (nought, one)), 2),
                  "rows as a list": forged_matrix([(one, nought), (nought, one)], 2),
                  "a row as a list": forged_matrix(([one, nought], (nought, one)), 2),
                  "a ragged row": forged_matrix(((one, nought), (nought,)), 2),
                  "ncols None": forged_matrix(((one, nought), (nought, one)), None),
                  "ncols 2.0": forged_matrix(((one, nought), (nought, one)), 2.0)}
        for label, M in forged.items():
            with self.subTest(forged=label):
                self.assertInvalidInput(support.run("matrix_power", M=M, k=3))
        # operations_v6.run rejects a stand-in by its kind before produce() runs; produce() refuses one itself, and
        # also a Matrix forged without its rows or its ncols, which run() cannot inspect.
        M = args["M"]
        for item in (True, False, 0.5, "1", None, Rational(3), Integer(3), 2.0, (3,)):
            with self.subTest(path="produce", k=item), self.assertRaises(InvalidInput):
                family.produce("matrix_power", {"M": M, "k": item})
        stand_ins = list(forged.values()) + [SubMatrix(((1, 1), (1, 0))), M.rows, [list(row) for row in M.rows],
                                             support.impostor(M), support.forged(mf.Matrix, rows=M.rows),
                                             support.forged(mf.Matrix, ncols=2), 0.5, True, "1", None]
        for item in stand_ins:
            with self.subTest(path="produce", M=item), self.assertRaises(InvalidInput):
                family.produce("matrix_power", {"M": item, "k": 3})
        for arguments in (dict(args, n=2), {"M": M}, {"k": 3}, {"A": M, "k": 3}, tuple(args.items()),
                          MappingProxyType(args), None):
            with self.subTest(path="produce", arguments=arguments), self.assertRaises(InvalidInput):
                family.produce("matrix_power", arguments)
        for operation in ("Matrix_power", "matrix_powers", "matrix_power ", "power", "", None, ("matrix_power",)):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(args))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("matrix_power", M=flat, k=1)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536, a 65 x 65 matrix and max_work=1 exceed the budget; so do the card's bound
        B = ||M_int||_inf**k past max_integer_bits, a common denominator past it, a true value with a denominator past
        it, and a chain longer than the budget. Each request fails at once."""
        start = time.perf_counter()
        fib = matrix([[1, 1], [1, 0]])
        self.assertResourceLimit(support.run("matrix_power", M=fib, k=support.HUGE))
        self.assertResourceLimit(support.run("matrix_power", M=fib, k=-support.HUGE))
        oversized = support.oversized_matrix()
        for k in (0, 1, 2, -1):  # The entries of M are charged before k and the shape are judged.
            with self.subTest(oversized=True, k=k):
                self.assertResourceLimit(support.run("matrix_power", M=oversized, k=k))
        with self.assertRaises(ResourceLimitError):
            family.produce("matrix_power", {"M": oversized, "k": -1})
        with computation(ComputationLimits(max_matrix_entries=65 * 66), fresh=True):
            oblong = mf.Matrix(tuple(tuple((i + j) % 5 for j in range(66)) for i in range(65)))
        self.assertResourceLimit(support.run("matrix_power", M=oblong, k=1))  # Not square, but too large first.
        requests = (dict(M=fib, k=0), dict(M=fib, k=1), dict(M=fib, k=10), dict(M=mf.Matrix(()), k=0),
                    dict(M=mf.Matrix(()), k=5), dict(M=matrix([[Fraction(1, 2), 0], [0, 2]]), k=3))
        for request in requests:
            with self.subTest(max_work=1, **request):
                self.assertResourceLimit(support.run("matrix_power", limits=ComputationLimits(max_work=1), **request))
        # Integers of M past max_integer_bits: run() inspects the request first.
        self.assertResourceLimit(support.run("matrix_power", M=wide([[2 ** BITS]]), k=1))
        self.assertResourceLimit(support.run("matrix_power", M=wide([[Fraction(1, 2 ** BITS)]]), k=0))
        # The card's bound: B = N**k past 2**max_integer_bits; for a giant k decided from bits(N) alone.
        third, half = Fraction(1, 3), Fraction(1, 2)
        for rows, k in (([[2]], BITS + 1), ([[3]], 41_349), ([[3]], BITS), ([[1, 1], [1, 0]], BITS + 1),
                        ([[1, 1], [1, 0]], 10 ** 30), ([[half, half], [0, 1]], 2 * BITS),
                        ([[-1, 1], [1, 1]], BITS + 1)):
            with self.subTest(bound=rows, k=k):
                self.assertResourceLimit(support.run("matrix_power", M=matrix(rows), k=k))
        self.assertEqual((3 ** 41_349).bit_length(), BITS + 1)
        # A common denominator past max_integer_bits, for k >= 1; k = 0 needs none.
        p, q = 2 ** 40_000 - 1, 2 ** 40_000 + 1
        diagonal = matrix([[Fraction(1, p), 0], [0, Fraction(1, q)]])
        self.assertResourceLimit(support.run("matrix_power", M=diagonal, k=1))
        self.assertResourceLimit(support.run("matrix_power", M=diagonal, k=2))
        self.assertVerified(support.run("matrix_power", M=diagonal, k=0), matrix(identity(2)))
        # A true value with a denominator past max_integer_bits, within the bound B = 1: [[1/3]]**41349 holds 1/3**41349
        # (65,537 bits), and from k (bits(s) - 1) >= max_integer_bits + bits(B) on, s**k is never formed.
        for k in (41_349, 65_536, 10 ** 6, 2 ** 1000):
            with self.subTest(denominator=k):
                self.assertResourceLimit(support.run("matrix_power", M=matrix([[third]]), k=k))
        self.assertResourceLimit(support.run("matrix_power", M=matrix([[third, third], [0, third]]), k=41_349))
        # Chains longer than the budget stop at it: B = 1 for the identity, 64**1000 < 2**65536 for the 64 x 64 matrix.
        self.assertResourceLimit(support.run("matrix_power", M=mf.Matrix.identity(8), k=2 ** 60_000))
        rng = random.Random(20261130)
        big = matrix([[rng.randrange(-1, 2) for _ in range(64)] for _ in range(64)])
        self.assertResourceLimit(support.run("matrix_power", M=big, k=1000))
        self.assertResourceLimit(support.run("matrix_power", M=fib, k=4000, limits=ComputationLimits(max_work=100)))
        self.assertLess(time.perf_counter() - start, 3 * FAST)

    def test_bound_of_exactly_two_to_the_limit(self):
        """The card refuses B = ||M_int||_inf**k only when k log2(N) exceeds max_integer_bits: B = 2**65536 itself, from
        a power of two N, is computed (a regression: B >= 2**65536 was refused, also for [[1, 1], [1, 0]]**65536 whose
        integers have about 45,500 bits). Only an entry of +-2**65536, which such a B allows, is refused, as no value
        holds it: [[2]]**65536 and [[4/3]]**32768 = [[2**65536 / 3**32768]]."""
        fib = fibonacci_power(BITS)
        self.assertEqual(max(entry.bit_length() for row in fib for entry in row), 45_498)
        self.assertVerified(support.run("matrix_power", M=matrix([[1, 1], [1, 0]]), k=BITS), matrix(fib))
        power = 2 ** (BITS // 2)  # [[-1, 1], [1, 1]]**2 = 2 I, with N = 2.
        self.assertVerified(support.run("matrix_power", M=matrix([[-1, 1], [1, 1]]), k=BITS),
                            matrix([[power, 0], [0, power]]))
        half = Fraction(1, 2)  # s = 2 and M_int = [[1, 1], [-1, 1]], whose 8th power is 16 I: N = 2, B = 2**65536.
        scaled = Fraction(1, 2 ** (BITS // 2))
        self.assertVerified(support.run("matrix_power", M=matrix([[half, half], [-half, half]]), k=BITS),
                            matrix([[scaled, 0], [0, scaled]]))
        for rows, k in (([[2]], BITS), ([[Fraction(4, 3)]], BITS // 2), ([[2, 0], [0, 1]], BITS)):
            with self.subTest(rows=rows, k=k):
                self.assertResourceLimit(support.run("matrix_power", M=matrix(rows), k=k))
                with self.assertRaises(ResourceLimitError) as raised:
                    family.produce("matrix_power", {"M": matrix(rows), "k": k})
                self.assertIn("an entry of M**k", str(raised.exception))

    def test_producer_refuses_what_the_checker_refuses(self):
        """Under small integer limits, produce() raises ResourceLimitError on exactly the requests that the checker
        refuses (a common denominator or a bound B past max_integer_bits) and on those whose true value has a longer
        integer, which the checker answers False; every other request gives the reference value."""
        limits = ComputationLimits(max_integer_bits=64)
        third, half = Fraction(1, 3), Fraction(1, 2)
        cases = ([[3]], [[1, 1], [1, 0]], [[half, half], [0, 1]], [[third]], [[Fraction(2, 3), 0], [0, Fraction(1, 5)]],
                 [[0, half], [2, 0]], [[1, Fraction(1, 7)], [0, 1]], [[third, third], [0, third]], [[-1, 1], [1, 1]],
                 [[Fraction(1, 2 ** 33 - 9), 0], [0, Fraction(1, 2 ** 33 + 9)]], [[0, 0], [0, 0]], [[1, 0], [0, -1]])
        tally = {"refused": 0, "too long": 0, "computed": 0}
        for rows in cases:
            M = matrix(rows)
            rows = fractions_of(rows)
            expected = identity(len(rows))
            for k in range(0, 101):
                if k:
                    expected = multiply(expected, rows)
                args = {"M": M, "k": k}
                with self.subTest(rows=rows, k=k):
                    value = wide(expected)
                    try:
                        answer = support.verdict(family, "matrix_power", args, value, limits=limits)
                    except ResourceLimitError:
                        answer = None  # The checker refuses the request.
                    try:
                        with computation(limits, fresh=True):
                            produced, _ = family.produce("matrix_power", dict(args))
                    except ResourceLimitError:
                        produced = None
                    if answer is None:
                        self.assertIsNone(produced)
                        tally["refused"] += 1
                    elif fits(expected, limits.max_integer_bits):
                        self.assertEqual(produced, value)
                        self.assertIs(answer, True)
                        tally["computed"] += 1
                    else:  # [[1/3]]**41 = [[1/3**41]]: 3**41 has 65 bits.
                        self.assertIsNone(produced)
                        self.assertIs(answer, False)
                        tally["too long"] += 1
        self.assertTrue(all(count > 50 for count in tally.values()), tally)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also ones that the producer did not make: the evidence is always (),
        the only layout, and the values here are built by hand from the reference, as Matrix(rows), as
        Matrix.identity(n) and as trusted_matrix(rows, n), which are equal; the producer's value equals each of them.
        An M forged out of lowest terms (2/4 for 1/2) has the same powers, which the producer computes as well."""
        rng = random.Random(20261131)
        for _ in range(60):
            n, k = rng.randrange(0, 4), rng.randrange(0, 9)
            rows = [[Fraction(rng.randrange(-6, 7), rng.randrange(1, 6)) for _ in range(n)] for _ in range(n)]
            expected = naive_power(rows, k)
            args = {"M": matrix(rows, n), "k": k}
            exact_rows = tuple(tuple(exact(entry) for entry in row) for row in expected)
            canonical = tuple(tuple(Rational(entry.numerator, entry.denominator) for entry in row) for row in expected)
            built = [mf.Matrix(exact_rows, ncols=n), trusted_matrix(canonical, n)]
            if k == 0:
                built.append(mf.Matrix.identity(n))
            value, _ = family.produce("matrix_power", dict(args))
            for claim in built:
                with self.subTest(rows=rows, k=k, claim=claim):
                    self.assertEqual(claim, value)
                    self.assertAccepts("matrix_power", args, claim)
        # M forged out of lowest terms: [[1, 2/4], [0, 6/3]] is [[1, 1/2], [0, 2]], whose k-th power is
        # [[1, (2**k - 1)/2], [0, 2**k]].
        one, nought = Rational(1), Rational(0)
        forged = forged_matrix(((one, forged_rational(2, 4)), (nought, forged_rational(6, 3))), 2)
        for k in (0, 1, 2, 5, 30):
            expected = naive_power([[Fraction(1), Fraction(1, 2)], [Fraction(0), Fraction(2)]], k)
            self.assertEqual(expected, [[1, Fraction(2 ** k - 1, 2)], [0, 2 ** k]])
            args = {"M": forged, "k": k}
            with self.subTest(forged=True, k=k):
                value, evidence = family.produce("matrix_power", dict(args))
                self.assertEqual(value, matrix(expected))
                self.assertCanonical(value, 2)
                self.assertAccepts("matrix_power", args, matrix(expected), evidence)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        requests = ((matrix([[1, 1], [1, 0]]), 10), (matrix([[Fraction(1, 2), 0], [0, 2]]), 3),
                    (matrix([[1, 2], [3, 4]]), 3), (matrix([[5, 7], [1, 2]]), 0))
        for M, k in requests:
            result = support.run("matrix_power", M=M, k=k)
            value = result.value
            args = {"M": M, "k": k}
            self.assertVerified(result, value)
            self.assertAccepts("matrix_power", args, value)
            rows = value.rows
            true = fractions_of(rows)
            plus_one = [row[:] for row in true]
            plus_one[0][0] += 1
            transposed = [list(column) for column in zip(*true)]
            cases = [("an entry plus 1", matrix(plus_one), ()),
                     ("M**(k + 1)", matrix(naive_power(fractions_of(M.rows), k + 1)), ()),
                     ("M itself", M, ()),
                     ("twice the value", matrix([[2 * entry for entry in row] for row in true]), ()),
                     ("an int entry", forged_matrix(((rows[0][0].numerator,) + rows[0][1:], rows[1]), 2), ()),
                     ("an entry out of lowest terms",
                      forged_matrix(((forged_rational(2 * rows[0][0].numerator, 2 * rows[0][0].denominator),)
                                     + rows[0][1:], rows[1]), 2), ()),
                     ("a bool entry", forged_matrix(((True,) + rows[0][1:], rows[1]), 2), ()),
                     ("a float entry", forged_matrix(((float(true[0][0]),) + rows[0][1:], rows[1]), 2), ()),
                     ("a str entry", forged_matrix(((str(rows[0][0]),) + rows[0][1:], rows[1]), 2), ()),
                     ("rows as a list", forged_matrix(list(rows), 2), ()),
                     ("a row as a list", forged_matrix((list(rows[0]), rows[1]), 2), ()),
                     ("ncols 3", forged_matrix(rows, 3), ()), ("ncols True", forged_matrix(rows, True), ()),
                     ("ncols missing", support.forged(mf.Matrix, rows=rows), ()),
                     ("an impostor class", support.impostor(value), ()), ("a Matrix subclass", SubMatrix(rows), ()),
                     ("the rows alone", rows, ()), ("None", None, ()),
                     ("evidence None", value, None), ("evidence as a list", value, []),
                     ("an extra evidence item", value, ((),)), ("the evidence (0,)", value, (0,)),
                     ("oversized evidence", value, (0,) * 100_000)]
            if transposed != true:  # Only [[1, 2], [3, 4]]**3 = [[37, 54], [81, 118]] is not symmetric.
                cases.append(("the transposed value", matrix(transposed), ()))
            for label, item, evidence in cases:
                with self.subTest(k=k, case=label):
                    self.assertRejects("matrix_power", args, item, evidence, may_exhaust=label == "oversized evidence")
            for k_item in (True, float(k), str(k), Rational(k), Integer(k)):  # No exact int where k belongs.
                with self.subTest(k=k_item):
                    self.assertRejects("matrix_power", {"M": M, "k": k_item}, value)
            for name in ("matrix_powers", "Matrix_power", "power", "determinant", "", None):
                with self.subTest(operation=name):
                    self.assertRejects(name, args, value)
            report = support.statuses(mf.verify(support.tampered(result, matrix(plus_one))))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [(matrix(rows), k) for rows, k, _ in KNOWN]
        requests += [(matrix([[1, 1], [1, 0]]), 1000), (mf.Matrix(()), 3),
                     (matrix([[Fraction(-2, 3), 1], [0, Fraction(-2, 3)]]), 60)]
        results = [support.run("matrix_power", M=M, k=k) for M, k in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (M, k), result, text in zip(requests, results, texts):
                with self.subTest(shape=M.shape, k=k):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails, and so does M ** k, its sugar.
            failed = support.run("matrix_power", M=matrix([[1, 1], [1, 0]]), k=10)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))
            with self.assertRaises(AssertionError):
                matrix([[1, 1], [1, 0]]) ** 10

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: integer entries,
        rational entries, a negative rational, the zero matrix, the identity and the 0 x 0 matrix."""
        rendered = (
            ([[1, 1], [1, 0]], 10, "[[89, 55], [55, 34]]", r"\begin{bmatrix}89 & 55 \\ 55 & 34\end{bmatrix}"),
            ([[Fraction(1, 2), 0], [0, 2]], 3, "[[1/8, 0], [0, 8]]",
             r"\begin{bmatrix}\frac{1}{8} & 0 \\ 0 & 8\end{bmatrix}"),
            ([[Fraction(-3, 2)]], 3, "[[-27/8]]", r"\begin{bmatrix}-\frac{27}{8}\end{bmatrix}"),
            ([[0, 1], [0, 0]], 2, "[[0, 0], [0, 0]]", r"\begin{bmatrix}0 & 0 \\ 0 & 0\end{bmatrix}"),
            ([[5, 7], [1, 2]], 0, "[[1, 0], [0, 1]]", r"\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}"),
            ([], 4, "Matrix(0x0)", r"\text{Matrix(0x0)}"),
        )
        for rows, k, text, latex in rendered:
            with self.subTest(rows=rows, k=k):
                result = support.run("matrix_power", M=matrix(rows), k=k)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        eighth = support.run("matrix_power", M=matrix([[Fraction(1, 2), 0], [0, 2]]), k=3).value
        self.assertEqual(mf.to_data(eighth)["data"], {
            "type": "matrix", "ncols": "2",
            "rows": [[{"type": "rational", "numerator": "1", "denominator": "8"},
                      {"type": "rational", "numerator": "0", "denominator": "1"}],
                     [{"type": "rational", "numerator": "0", "denominator": "1"},
                      {"type": "rational", "numerator": "8", "denominator": "1"}]]})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261132)
        cases = [(matrix([[1, 1], [1, 0]]), 10), (mf.Matrix(()), 3), (matrix([[2, 0], [0, 3]]), 0)]
        for _ in range(30):
            n = rng.randrange(1, 5)
            rows = [[Fraction(rng.randrange(-5, 6), rng.choice((1, 1, 2, 3, 7))) for _ in range(n)] for _ in range(n)]
            cases.append((matrix(rows), rng.choice((rng.randrange(0, 12), rng.randrange(0, 200)))))
        for M, k in cases:
            with self.subTest(rows=M.rows, k=k):
                expected = support.run("matrix_power", M=M, k=k).value
                self.assertIsNotNone(expected)
                found = support.lean_value("matrix_power", M=M, k=k)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value("matrix_power", M=M, k=k), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        fib = matrix([[1, 1], [1, 0]])
        result = self.answer(support.call("matrix_power", {"M": [[1, 1], [1, 0]], "k": 10}))
        self.assertEqual(result, support.run("matrix_power", M=fib, k=10))
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(result.value, matrix([[89, 55], [55, 34]]))
        # Rationals travel as strings, the 0 x 0 matrix as [], and long integers of the value as decimal strings.
        eighth = matrix([[Fraction(1, 2), 0], [0, 2]])
        for arguments, M, k in (({"M": [["1/2", 0], [0, 2]], "k": 3}, eighth, 3), ({"M": [], "k": 5}, mf.Matrix(()), 5),
                                ({"M": [[1, 1], [1, 0]], "k": 1000}, fib, 1000)):
            with self.subTest(arguments=arguments):
                result = self.answer(support.call("matrix_power", arguments))
                self.assertEqual(result, support.run("matrix_power", M=M, k=k))
                self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call_values("matrix_power", M=eighth, k=3))
        self.assertEqual(result, support.run("matrix_power", M=eighth, k=3))
        self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("matrix_power", {"M": [[1, 2, 3]], "k": 1}), "invalid_input")
        self.assertIn("square matrix M", error["message"])
        error = self.assertToolError(support.call("matrix_power", {"M": [[2, 1], [1, 1]], "k": -1}), "invalid_input")
        self.assertIn("use inverse for negative powers", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with repeated multiplication on 300 seeded inputs (n from 0 to 5, integer and rational
        entries, k from 0 to 40), M ** k gives the same value, and the checker accepts every answer; closed forms
        cover large k."""
        rng = random.Random(20261133)
        entries = (lambda: Fraction(rng.randrange(-3, 4)), lambda: Fraction(rng.randrange(-20, 21)),
                   lambda: Fraction(rng.randrange(-9, 10), rng.randrange(1, 10)),
                   lambda: rng.choice((Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-2, 3))))
        count = 0
        while count < 300:
            n, k = rng.choice((0, 1, 2, 2, 3, 3, 4, 5)), rng.choice((rng.randrange(0, 6), rng.randrange(0, 41)))
            entry = rng.choice(entries)
            rows = [[entry() for _ in range(n)] for _ in range(n)]
            expected = naive_power(rows, k)
            with self.subTest(rows=rows, k=k):
                self.assertAgrees(rows, k, expected)
                self.assertEqual(matrix(rows, n) ** k, matrix(expected, n))
                count += 1
        # Closed forms for large k: the plain recurrence of the Fibonacci numbers, a Jordan block, binomial
        # coefficients, nilpotent, idempotent, permutation, rotation, diagonal and identity matrices.
        for k in (1, 2, 3, 20, 999, 3000):
            with self.subTest(form="Fibonacci", k=k):
                self.assertAgrees([[1, 1], [1, 0]], k, fibonacci_power(k))
        a = Fraction(-2, 3)
        for k in (2, 7, 60, 333):
            with self.subTest(form="Jordan block", k=k):
                self.assertAgrees([[a, 1], [0, a]], k, [[a ** k, k * a ** (k - 1)], [0, a ** k]])
        for k in (1, 5, 5000):
            with self.subTest(form="unitriangular", k=k):
                self.assertAgrees([[1, 1, 0], [0, 1, 1], [0, 0, 1]], k, [[1, k, comb(k, 2)], [0, 1, k], [0, 0, 1]])
        nilpotent = [[0, Fraction(1, 2), -3, 1], [0, 0, Fraction(5, 7), 2], [0, 0, 0, Fraction(-1, 4)], [0, 0, 0, 0]]
        for k in (4, 5, 1000):
            with self.subTest(form="nilpotent", k=k):
                self.assertAgrees(nilpotent, k, [[0] * 4] * 4)
        third = Fraction(1, 3)
        for k in (2, 50, 1000):  # J/3 is idempotent: s**k = 3**k, but the value stays J/3.
            with self.subTest(form="idempotent", k=k):
                self.assertAgrees([[third] * 3] * 3, k, [[third] * 3] * 3)
        self.assertAgrees([[0, 1], [1, 0]], 10 ** 20 + 1, [[0, 1], [1, 0]])
        self.assertAgrees([[0, -1], [1, 0]], 2 ** 100 + 3, [[0, 1], [-1, 0]])  # A rotation of order 4.
        cycle = [[int(j == (i + 1) % 5) for j in range(5)] for i in range(5)]
        self.assertAgrees(cycle, 10 ** 30 + 2, naive_power(fractions_of(cycle), 2))  # 10**30 = 0 modulo 5.
        self.assertAgrees(identity(6), 2 ** 1000 + 1, identity(6))
        diagonal = [Fraction(1, 2), Fraction(-3), Fraction(5, 7)]
        self.assertAgrees([[diagonal[i] if i == j else 0 for j in range(3)] for i in range(3)], 40,
                          [[diagonal[i] ** 40 if i == j else 0 for j in range(3)] for i in range(3)])
        self.assertAgrees([[Fraction(-3, 2)]], 1001, [[Fraction(-3, 2) ** 1001]])

    # -----------------------------------------------------------------------------------------------------------
    # The producer's construction and Matrix.__pow__

    def test_chain_is_right_to_left_binary(self):
        """The producer's chain is right-to-left binary, from the lowest bit of k up: bits(k) - 1 squarings of the
        base M_int**(2**i) and popcount(k) - 1 multiplications of the result by it. The checker's left-to-right
        chain multiplies by M_int instead, so the two compute different intermediate powers."""
        base = [[2, 1], [1, 1]]  # s = 1, and its powers are distinct: each product's factors are known by exponent.
        by_exponent, exponents, current = {}, {}, [[1, 0], [0, 1]]
        for j in range(1, 70):  # Repeated multiplication, no chain.
            current = integer_product(current, base)
            by_exponent[j] = current
            exponents[tuple(map(tuple, current))] = j
        products = []
        original = family._product

        def recorded(budget, left, right):
            products.append((exponents[tuple(map(tuple, left))], exponents[tuple(map(tuple, right))]))
            return original(budget, left, right)

        expected = {1: [], 2: [(1, 1)], 7: [(1, 1), (1, 2), (2, 2), (3, 4)], 12: [(1, 1), (2, 2), (4, 4), (4, 8)],
                    13: [(1, 1), (2, 2), (1, 4), (4, 4), (5, 8)],
                    64: [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]}
        with patch.object(family, "_product", recorded):
            for k in (1, 2, 7, 12, 13, 41, 63, 64):
                products.clear()
                with self.subTest(k=k):
                    value, _ = family.produce("matrix_power", {"M": matrix(base), "k": k})
                    self.assertEqual(value, matrix(by_exponent[k]))
                    if k in expected:
                        self.assertEqual(products, expected[k])
                    squarings = [pair for pair in products if pair[0] == pair[1]]
                    self.assertEqual(len(squarings), k.bit_length() - 1)
                    self.assertEqual(len(products) - len(squarings), k.bit_count() - 1)
                    self.assertTrue(all(pair[1] & (pair[1] - 1) == 0 for pair in products))  # Factors M_int**(2**i).

    def test_matrix_power_operator(self):
        """M ** k is produce()'s value, unchecked arithmetic like @: equal to the verified value, with the same errors
        (InvalidInput for a non-square M, k < 0 and a k that is no exact int) and the active budget's limits.
        Matrix * Matrix stays InvalidInput, with a message that names @; * by a scalar and @ are unchanged."""
        for rows, k, expected in KNOWN:
            with self.subTest(rows=rows, k=k):
                M = matrix(rows)
                self.assertEqual(M ** k, matrix(expected))
                self.assertEqual(M ** k, support.run("matrix_power", M=M, k=k).value)
                self.assertCanonical(M ** k, len(rows))
        fib = matrix([[1, 1], [1, 0]])
        self.assertEqual(fib ** 1, fib)
        self.assertEqual(fib ** 2, fib @ fib)
        self.assertEqual(fib ** 5, fib @ fib @ fib @ fib @ fib)
        self.assertEqual(mf.Matrix(()) ** 3, mf.Matrix(()))
        self.assertEqual(matrix([[1, 2], [2, 4]]) ** 0, mf.Matrix.identity(2))
        for M, k, text in ((matrix([[1, 2, 3]]), 2, "square matrix M"), (fib, -1, "use inverse for negative powers"),
                           (fib, 0.5, "exact integer k"), (fib, True, "exact integer k"), (fib, "2", "exact integer k"),
                           (fib, Rational(2), "exact integer k"), (SubMatrix(((1, 1), (1, 0))), 2, "Matrix M")):
            with self.subTest(shape=M.shape, k=k):
                with self.assertRaises(InvalidInput) as raised:
                    M ** k
                self.assertIn(text, str(raised.exception))
        with computation(ComputationLimits(max_work=10), fresh=True), self.assertRaises(ResourceLimitError):
            fib ** 1000
        with computation(ComputationLimits(max_integer_bits=64), fresh=True), self.assertRaises(ResourceLimitError):
            fib ** 100  # F(101) has 69 bits: B = 2**100 is past the bound.
        with self.assertRaises(ResourceLimitError):
            matrix([[2]]) ** BITS  # Without an active budget, the default limits.
        A, B = matrix([[1, 2], [3, 4]]), matrix([[2, 0], [1, 2]])
        with self.assertRaises(InvalidInput) as raised:  # The exception type is pinned by tests/test_matrices_v2.py.
            A * B
        self.assertIn("@", str(raised.exception))
        with self.assertRaises(InvalidInput) as raised:
            A * SubMatrix(((2, 0), (1, 2)))
        self.assertIn("@", str(raised.exception))
        self.assertEqual(A * 2, matrix([[2, 4], [6, 8]]))
        self.assertEqual(Rational(1, 2) * A, matrix([[Fraction(1, 2), 1], [Fraction(3, 2), 2]]))
        self.assertEqual(A @ B, matrix([[4, 4], [10, 8]]))
        with self.assertRaises(InvalidInput):
            A * 0.5

    def test_work_is_charged_before_it_runs(self):
        """produce() charges at least two units, and the same amount for the same request: a budget of exactly that
        amount computes the value and one unit less refuses the request."""
        rng = random.Random(20261134)
        requests = [(matrix([[1, 1], [1, 0]]), 1000), (matrix([[2, 0], [0, 3]]), 0), (mf.Matrix(()), 0),
                    (mf.Matrix(()), 9), (matrix([[Fraction(1, 2), Fraction(1, 3)], [5, Fraction(-7, 4)]]), 77),
                    (matrix([[Fraction(rng.randrange(-5, 6), rng.randrange(1, 4)) for _ in range(5)]
                             for _ in range(5)]), 37), (matrix([[Fraction(1, 3)]]), 1)]
        for M, k in requests:
            with self.subTest(shape=M.shape, k=k):
                with computation(fresh=True) as budget:
                    value, _ = family.produce("matrix_power", {"M": M, "k": k})
                work = budget.work
                self.assertGreaterEqual(work, 2)
                with computation(ComputationLimits(max_work=work), fresh=True):
                    self.assertEqual(family.produce("matrix_power", {"M": M, "k": k})[0], value)
                with computation(ComputationLimits(max_work=work - 1), fresh=True), \
                        self.assertRaises(ResourceLimitError):
                    family.produce("matrix_power", {"M": M, "k": k})

    def test_full_size_requests_fit_the_default_budget(self):
        """Requests at the edge of the limits verify under the default budget: a bound B of 65,536 bits (Fibonacci
        numbers and a power of 2), a denominator of 65,536 bits, an idempotent with s**k of 60,001 bits, and a 64 x 64
        matrix."""
        rng = random.Random(20261135)
        big = [[rng.randrange(-1, 2) for _ in range(64)] for _ in range(64)]
        half, third = Fraction(1, 2), Fraction(1, 3)
        requests = [([[1, 1], [1, 0]], BITS - 1, fibonacci_power(BITS - 1)), ([[2]], BITS - 1, [[2 ** (BITS - 1)]]),
                    ([[third]], 41_348, [[third ** 41_348]]),
                    ([[half, half], [half, half]], 60_000, [[half, half]] * 2),  # Idempotent: s**k = 2**60000.
                    (big, 3, integer_product(integer_product(big, big), big))]
        for rows, k, expected in requests:
            with self.subTest(n=len(rows), k=k):
                start = time.perf_counter()
                result = support.run("matrix_power", M=matrix(rows), k=k)
                self.assertVerified(result, matrix(expected))
                self.assertLess(time.perf_counter() - start, 3 * FAST)
        self.assertEqual((3 ** 41_348).bit_length(), BITS)  # The longest denominator that a value may hold.
