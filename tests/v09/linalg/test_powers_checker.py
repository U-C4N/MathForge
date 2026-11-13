"""The independent checker of mathforge.areas.linalg.powers (mathforge.certify.linalg.powers), tested on its own before
the family's producer exists.

Every true claim is computed here, never by the producer: by repeated multiplication with fractions.Fraction, by a
square-and-multiply on fractions that is checked against it, and by closed forms (Fibonacci numbers, Jordan blocks,
unitriangular, nilpotent and idempotent matrices, permutations, diagonal matrices). The evidence is always (), the
card's layout. The checker answers False to every corrupted claim, never with an exception, and refuses an oversized
forgery within FAST seconds under the default budget; ResourceLimitError comes only for the request limits that the
checker's docstring lists, the card's bound B = ||M_int||_inf**k among them.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from fractions import Fraction
import inspect
from math import comb
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.linalg import powers as family
from mathforge.certify.linalg import powers as checker
from mathforge.contracts import CheckStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.registry import Argument, OperationSpec
from mathforge.trusted.primes import MODULI, moduli

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"
SUMMARY = "Raise a square rational matrix to a nonnegative integer power."
BITS = ComputationLimits().max_integer_bits  # 65,536: the longest integer of a value, and the card's bound on B.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
WIDE = ComputationLimits(max_integer_bits=300_000, max_matrix_entries=1_000_000)  # For values past the limits.

Number = int | Fraction | Rational
Rows = list[list[Fraction]]


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

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


def zero(n: int) -> Rows:
    return [[Fraction(0)] * n for _ in range(n)]


def multiply(left: Rows, right: Rows) -> Rows:
    """The product of two square matrices of Fractions of one size, entry by entry."""
    n = len(right)
    return [[sum((row[m] * right[m][j] for m in range(n)), Fraction(0)) for j in range(n)] for row in left]


def naive_power(rows: Rows, k: int) -> Rows:
    """rows**k as the identity times rows, k times over."""
    result = identity(len(rows))
    for _ in range(k):
        result = multiply(result, rows)
    return result


def binary_power(rows: Rows, k: int) -> Rows:
    """rows**k by square-and-multiply on Fractions; checked against naive_power in test_accepts_random_claims."""
    result, base = identity(len(rows)), rows
    while k:
        if k & 1:
            result = multiply(result, base)
        k >>= 1
        if k:
            base = multiply(base, base)
    return result


def integer_product(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> list[list[int]]:
    """The product of two square integer matrices, for the large integer claims."""
    columns = list(zip(*right))
    return [[sum(a * b for a, b in zip(row, column)) for column in columns] for row in left]


def fibonacci(n: int) -> int:
    """F(n) by fast doubling, F(2m) = F(m) (2 F(m + 1) - F(m)) and F(2m + 1) = F(m)**2 + F(m + 1)**2; checked
    against the plain recurrence in test_accepts_closed_forms."""
    a, b = 0, 1  # F(0), F(1)
    for bit in bin(n)[2:]:
        a, b = a * (2 * b - a), a * a + b * b
        if bit == "1":
            a, b = b, a + b
    return a


def fibonacci_power(k: int) -> list[list[int]]:
    """[[1, 1], [1, 0]]**k = [[F(k + 1), F(k)], [F(k), F(k - 1)]] for k >= 1."""
    return [[fibonacci(k + 1), fibonacci(k)], [fibonacci(k), fibonacci(k - 1)]]


def prime_count(bound: int) -> int:
    """How many primes of trusted.primes.moduli() it takes for their product to exceed bound."""
    product, count = 1, 0
    for prime in moduli():
        product *= prime
        count += 1
        if product > bound:
            return count
    raise AssertionError("unreachable")


def forged_matrix(rows: object, ncols: object) -> mf.Matrix:
    """A Matrix made without its constructor, whose fields may break the canonical form."""
    return support.forged(mf.Matrix, rows=rows, ncols=ncols)


def forged_rational(numerator: object, denominator: object) -> Rational:
    """A Rational made without its constructor: it may be out of lowest terms or have a bad denominator."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def args_of(rows: Iterable[Iterable[Number]], k: int, ncols: int | None = None) -> dict[str, object]:
    """The arguments of matrix_power as a request holds them."""
    return {"M": matrix(rows, ncols), "k": k}


def claimed(args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest("matrix_power", tuple(args.items()), 6)
    step = contracts.Step("v6.matrix_power", (request,), (value, evidence),
                          relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method="matrix_power", request=request,
                            steps=(step,))


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values and arguments that the checker must refuse.

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class SubRational(Rational):
    """A Rational subclass: equal to the right entry but no exact Rational."""


class SubMatrix(mf.Matrix):
    """A Matrix subclass with the right rows."""


Matrix = type("Matrix", (mf.Matrix,), {})  # A subclass that even has the name, in this module.
Rows2 = type("Rows2", (tuple,), {})  # A tuple subclass for rows.


class PowersCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case of matrix_power."""
        for label, args, value, evidence in cases:
            with self.subTest(case=label):
                self.assertRejects("matrix_power", args, value, evidence)

    def assertPower(self, rows: Iterable[Iterable[Number]], k: int, expected: Iterable[Iterable[Number]]) -> None:
        """The checker accepts expected as rows**k, with the evidence ()."""
        rows = [list(row) for row in rows]
        self.assertAccepts("matrix_power", args_of(rows, k, len(rows)), matrix(expected, len(rows)), ())

    def assertExhausts(self, args: object, value: object, evidence: object = (), *,
                       limits: ComputationLimits | None = None) -> None:
        """The checker raises ResourceLimitError on the claim, within FAST seconds: unverified, never refuted."""
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "matrix_power", args, value, evidence, limits=limits)
        self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_spec_of_the_card(self):
        expected = OperationSpec("matrix_power", "linalg",
                                 (Argument("M", "Matrix", None, "matrix"), Argument("k", "int", None, "int")),
                                 "Matrix", SUMMARY, SCOPE, version=6)
        self.assertEqual(family.SPECS, (expected,))
        self.assertEqual(family.TYPES, {})  # The value is a Matrix, which no family record wraps.
        self.assertEqual(getattr(family, "RENDER", {}), {})
        self.assertEqual(family.CHECKER, "mathforge.certify.linalg.powers")
        self.assertEqual(registry.spec("matrix_power"), expected)
        self.assertIs(registry.checker("matrix_power"), checker)
        self.assertIs(support.checker(family), checker)
        self.assertEqual(checker.MATRIX, (mf.Matrix.__module__, mf.Matrix.__qualname__))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_known_values_of_the_card(self):
        half = Fraction(1, 2)
        cases = [([[1, 1], [1, 0]], 10, [[89, 55], [55, 34]]),
                 ([[2, 0], [0, 3]], 0, [[1, 0], [0, 1]]),
                 ([[0, 1], [0, 0]], 2, [[0, 0], [0, 0]]),
                 ([[half, 0], [0, 2]], 3, [[Fraction(1, 8), 0], [0, 8]])]
        for rows, k, expected in cases:
            with self.subTest(rows=rows, k=k):
                self.assertEqual(naive_power(fractions_of(rows), k), fractions_of(expected))
                self.assertPower(rows, k, expected)

    def test_accepts_random_claims(self):
        """Random small matrices of rationals, n from 0 to 4 and k from 0 to 11, against two references; each claim
        with one entry changed is refused."""
        rng = random.Random(20261112)

        def entry() -> Fraction:
            return rng.choice((Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(2), Fraction(-3),
                               Fraction(1, 2), Fraction(-2, 3), Fraction(5, 4),
                               Fraction(rng.randrange(-9, 10), rng.randrange(1, 8))))

        changes = (Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 3))
        refuted = 0
        for _ in range(250):
            n, k = rng.randrange(0, 5), rng.randrange(0, 12)
            rows = [[entry() for _ in range(n)] for _ in range(n)]
            expected = naive_power(rows, k)
            self.assertEqual(binary_power(rows, k), expected)
            args = args_of(rows, k, n)
            with self.subTest(rows=rows, k=k):
                self.assertAccepts("matrix_power", args, matrix(expected, n))
                if n:
                    i, j = rng.randrange(n), rng.randrange(n)
                    wrong = [row[:] for row in expected]
                    wrong[i][j] += rng.choice(changes)
                    self.assertRejects("matrix_power", args, matrix(wrong, n))
                    refuted += 1
        self.assertGreater(refuted, 150)

    def test_accepts_closed_forms(self):
        a = Fraction(-2, 3)
        for n in range(30):  # The fast doubling against the recurrence.
            self.assertEqual(fibonacci(n), fibonacci(n - 1) + fibonacci(n - 2) if n >= 2 else n)
        for k in (1, 2, 3, 20, 100, 1000, 4097):
            with self.subTest(form="Fibonacci", k=k):
                self.assertPower([[1, 1], [1, 0]], k, fibonacci_power(k))
        for k in (1, 2, 7, 60):  # A Jordan block with a rational eigenvalue.
            with self.subTest(form="Jordan block", k=k):
                expected = [[a ** k, k * a ** (k - 1)], [0, a ** k]]
                self.assertEqual(binary_power(fractions_of([[a, 1], [0, a]]), k), fractions_of(expected))
                self.assertPower([[a, 1], [0, a]], k, expected)
        for k in (1, 2, 5, 5000):  # Unitriangular: the binomial coefficients.
            with self.subTest(form="unitriangular", k=k):
                self.assertPower([[1, 1, 0], [0, 1, 1], [0, 0, 1]], k, [[1, k, comb(k, 2)], [0, 1, k], [0, 0, 1]])
        for k in (1, 3, 1000):  # A shear with a rational entry: s = 3, M_int = [[3, 1], [0, 3]], B = 4**k.
            with self.subTest(form="rational shear", k=k):
                self.assertPower([[1, Fraction(1, 3)], [0, 1]], k, [[1, Fraction(k, 3)], [0, 1]])
        nilpotent = [[0, Fraction(1, 2), -3, 1], [0, 0, Fraction(5, 7), 2], [0, 0, 0, Fraction(-1, 4)], [0, 0, 0, 0]]
        for k in range(1, 7):  # Strictly upper triangular: zero from k = 4 on.
            with self.subTest(form="nilpotent", k=k):
                expected = naive_power(fractions_of(nilpotent), k)
                self.assertEqual(expected == zero(4), k >= 4)
                self.assertPower(nilpotent, k, expected)
        third = Fraction(1, 3)
        for k in (1, 2, 50, 1000):  # J/3 is idempotent: s**k = 3**k, but the value stays J/3.
            with self.subTest(form="idempotent", k=k):
                self.assertPower([[third] * 3] * 3, k, [[third] * 3] * 3)
        swap = [[0, 1], [1, 0]]
        self.assertPower(swap, 10 ** 20, identity(2))  # Permutations: N = 1, so B = 1 for every k.
        self.assertPower(swap, 10 ** 20 + 1, swap)
        self.assertPower([[0, -1], [1, 0]], 2 ** 100 + 3, [[0, 1], [-1, 0]])  # A rotation of order 4.
        cycle = [[int(j == (i + 1) % 5) for j in range(5)] for i in range(5)]
        self.assertPower(cycle, 10 ** 30 + 2, naive_power(fractions_of(cycle), 2))  # 10**30 = 0 modulo 5.
        self.assertPower(identity(6), 2 ** 1000 + 1, identity(6))
        diagonal = [Fraction(1, 2), Fraction(-3), Fraction(5, 7)]
        self.assertPower([[diagonal[i] if i == j else 0 for j in range(3)] for i in range(3)], 40,
                         [[diagonal[i] ** 40 if i == j else 0 for j in range(3)] for i in range(3)])
        self.assertPower([[Fraction(-3, 2)]], 1001, [[Fraction(-3, 2) ** 1001]])
        self.assertPower([[Fraction(1, 3)]], 30_000, [[Fraction(1, 3 ** 30_000)]])
        self.assertPower([[1]], 2 ** 65_535, [[1]])  # A k of max_integer_bits bits: 65,535 squarings of [[1]].

    def test_accepts_special_cases(self):
        """k = 0 gives the identity, also for singular, zero and nilpotent M; k = 1 gives M; the zero matrix stays
        zero; the 0 x 0 matrix is its own power."""
        singular = [[1, 2], [2, 4]]
        for rows in (singular, zero(3), [[0, 1], [0, 0]], [[Fraction(7, 3)]], [[-1, Fraction(1, 2)], [3, 0]]):
            with self.subTest(rows=rows):
                self.assertPower(rows, 0, identity(len(rows)))
                self.assertPower(rows, 1, rows)
        for k in (1, 2, 5):  # Rank one: M**k = 5**(k - 1) M.
            self.assertPower(singular, k, [[5 ** (k - 1) * x for x in row] for row in singular])
        for k in (1, 2, 10 ** 9):  # The zero matrix: N = 0 and B = 0.
            with self.subTest(form="zero", k=k):
                self.assertPower(zero(3), k, zero(3))
        for k in (0, 1, 7, 10 ** 40):
            with self.subTest(form="0 x 0", k=k):
                self.assertAccepts("matrix_power", {"M": mf.Matrix((), ncols=0), "k": k}, mf.Matrix((), ncols=0))
                self.assertAccepts("matrix_power", {"M": mf.Matrix(()), "k": k}, mf.Matrix(()))
        # Large k with a small norm: the chain runs, the bound stays 1.
        self.assertPower([[0, Fraction(1, 2)], [0, 0]], 10 ** 6, zero(2))
        self.assertPower([[-1]], 10 ** 50 + 1, [[-1]])

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_rejects_claims_on_inputs_outside_the_domain(self):
        """A non-square M and a negative k have no matrix_power, whatever value is claimed (the inverse included:
        negative powers belong to inverse)."""
        flat = mf.Matrix(((1, 2, 3), (4, 5, 6)))
        tall = mf.Matrix(((1, 2), (3, 4), (5, 6)))
        no_rows, no_columns = mf.Matrix((), ncols=3), mf.Matrix(((), ()))
        unimodular = [[2, 1], [1, 1]]
        inverse = [[1, -1], [-1, 2]]
        self.assertEqual(multiply(fractions_of(unimodular), fractions_of(inverse)), identity(2))
        cases = [("2 x 3, k = 1", {"M": flat, "k": 1}, flat, ()),
                 ("2 x 3, k = 0, I2", {"M": flat, "k": 0}, matrix(identity(2)), ()),
                 ("2 x 3, k = 0, I3", {"M": flat, "k": 0}, matrix(identity(3)), ()),
                 ("3 x 2, k = 1", {"M": tall, "k": 1}, tall, ()),
                 ("0 x 3, k = 0", {"M": no_rows, "k": 0}, no_rows, ()),
                 ("0 x 3, k = 1, 0 x 0", {"M": no_rows, "k": 1}, mf.Matrix((), ncols=0), ()),
                 ("2 x 0, k = 1", {"M": no_columns, "k": 1}, no_columns, ()),
                 ("2 x 0, k = 0, I2", {"M": no_columns, "k": 0}, matrix(identity(2)), ()),
                 ("[[1]], k = -1", args_of([[1]], -1), matrix([[1]]), ()),
                 ("0 x 0, k = -1", {"M": mf.Matrix(()), "k": -1}, mf.Matrix(()), ())]
        for k, value in ((-1, inverse), (-2, naive_power(fractions_of(inverse), 2)), (-1, unimodular),
                         (-1, identity(2)), (-3, zero(2))):
            cases.append((f"k = {k}, {value}", args_of(unimodular, k), matrix(value), ()))
        self.assertRejectsAll(cases)

    def test_rejects_wrong_values(self):
        rows = [[1, 2], [3, 4]]
        args = args_of(rows, 3)
        true = naive_power(fractions_of(rows), 3)
        self.assertEqual(true, fractions_of([[37, 54], [81, 118]]))
        self.assertAccepts("matrix_power", args, matrix(true))
        bound = 7 ** 3  # N = ||M||_inf = 7.
        modulus = MODULI[0]  # The checker's P: the first prime already exceeds 2B.
        cases = []
        for i in range(2):
            for j in range(2):
                for change in (1, -1, Fraction(1, 2), Fraction(-1, 118), modulus, -modulus, 2 * bound + 1):
                    wrong = [row[:] for row in true]
                    wrong[i][j] += change
                    cases.append((f"entry ({i}, {j}) + {change}", args, matrix(wrong), ()))
        transposed = [list(column) for column in zip(*true)]
        for label, value in (("transposed", transposed), ("M**2", naive_power(fractions_of(rows), 2)),
                             ("M**4", naive_power(fractions_of(rows), 4)), ("M", fractions_of(rows)),
                             ("-V", [[-x for x in row] for row in true]),
                             ("2V", [[2 * x for x in row] for row in true]),
                             ("V/2", [[x / 2 for x in row] for row in true]), ("zero", zero(2)),
                             ("identity", identity(2)),
                             ("(M**T)**3", naive_power(fractions_of(list(zip(*rows))), 3))):
            cases.append((label, args, matrix(value), ()))
        cases += [("a 1 x 1 value", args, matrix([[37]]), ()),
                  ("a 3 x 3 value", args, matrix([[37, 54, 0], [81, 118, 0], [0, 0, 1]]), ()),
                  ("a 0 x 0 value", args, mf.Matrix(()), ()),
                  ("2 x 3 with the right rows and a zero column", args, matrix([[37, 54, 0], [81, 118, 0]]), ()),
                  ("one row", args, matrix([[37, 54]]), ())]
        # A rational matrix: wrong denominators.
        rational = [[Fraction(1, 2), 1], [0, Fraction(2, 3)]]
        rational_args = args_of(rational, 4)
        value = naive_power(fractions_of(rational), 4)
        self.assertAccepts("matrix_power", rational_args, matrix(value))
        for i, j, entry in ((0, 0, Fraction(1, 8)), (1, 1, Fraction(16, 27)), (0, 1, value[0][1] * 3),
                            (1, 0, Fraction(1, 3)), (0, 0, Fraction(1, 32))):
            wrong = [row[:] for row in value]
            wrong[i][j] = entry
            cases.append((f"rational, ({i}, {j}) = {entry}", rational_args, matrix(wrong), ()))
        self.assertRejectsAll(cases)

    def test_rejects_non_canonical_and_forged_values(self):
        """A value that equals M**k as numbers but is not the canonical Matrix: int entries, an entry out of lowest
        terms, a negative or zero denominator, lists, tuple subclasses, a wrong or missing ncols, ragged rows, an
        impostor class, subclasses, and the rows without the Matrix."""
        args = args_of([[Fraction(1, 2), 0], [0, 2]], 3)
        value = matrix([[Fraction(1, 8), 0], [0, 8]])
        self.assertAccepts("matrix_power", args, value)
        eighth, nought, eight = value.rows[0][0], value.rows[0][1], value.rows[1][1]
        rows = value.rows

        def with_entry(i: int, j: int, entry: object) -> mf.Matrix:
            changed = [list(row) for row in rows]
            changed[i][j] = entry
            return forged_matrix(tuple(tuple(row) for row in changed), 2)

        fibonacci_args = args_of([[1, 1], [1, 0]], 10)
        fibonacci_value = matrix([[89, 55], [55, 34]])
        self.assertAccepts("matrix_power", fibonacci_args, fibonacci_value)
        first, second = fibonacci_value.rows
        cases = [
            ("int entries", args, forged_matrix(((eighth, 0), (0, 8)), 2), ()),
            ("all int entries", fibonacci_args, forged_matrix(((89, 55), (55, 34)), 2), ()),
            ("one int entry", fibonacci_args, forged_matrix((first, (second[0], 34)), 2), ()),
            ("1/8 as 2/16", args, with_entry(0, 0, forged_rational(2, 16)), ()),
            ("8 as 16/2", args, with_entry(1, 1, forged_rational(16, 2)), ()),
            ("0 as 0/2", args, with_entry(0, 1, forged_rational(0, 2)), ()),
            ("1/8 as -1/-8", args, with_entry(0, 0, forged_rational(-1, -8)), ()),
            ("a zero denominator", args, with_entry(0, 1, forged_rational(0, 0)), ()),
            ("a missing denominator", args, with_entry(0, 0, support.forged(Rational, numerator=1)), ()),
            ("a missing numerator", args, with_entry(0, 0, support.forged(Rational, denominator=8)), ()),
            ("a bool numerator", args, with_entry(0, 0, forged_rational(True, 8)), ()),
            ("a float numerator", args, with_entry(0, 0, forged_rational(1.0, 8)), ()),
            ("an int-subclass denominator", args, with_entry(0, 0, forged_rational(1, Integer(8))), ()),
            ("a Rational-subclass entry", args, with_entry(0, 0, SubRational(1, 8)), ()),
            ("a bool entry", fibonacci_args, forged_matrix((first, (second[0], True)), 2), ()),
            ("a float entry", args, with_entry(1, 1, 8.0), ()),
            ("a str entry", args, with_entry(1, 1, "8"), ()),
            ("a Fraction entry", args, with_entry(0, 0, Fraction(1, 8)), ()),
            ("None as an entry", args, with_entry(0, 1, None), ()),
            ("rows as a list", args, forged_matrix(list(rows), 2), ()),
            ("a row as a list", args, forged_matrix((list(rows[0]), rows[1]), 2), ()),
            ("rows as a tuple subclass", args, forged_matrix(Rows2(rows), 2), ()),
            ("a row as a tuple subclass", args, forged_matrix((Rows2(rows[0]), rows[1]), 2), ()),
            ("ncols 3", args, forged_matrix(rows, 3), ()),
            ("ncols 1", args, forged_matrix(rows, 1), ()),
            ("ncols 2.0", args, forged_matrix(rows, 2.0), ()),
            ("ncols an int subclass", args, forged_matrix(rows, Integer(2)), ()),
            ("ncols None", args, forged_matrix(rows, None), ()),
            ("ncols missing", args, support.forged(mf.Matrix, rows=rows), ()),
            ("rows missing", args, support.forged(mf.Matrix, ncols=2), ()),
            ("a ragged row", args, forged_matrix(((eighth, nought), (eight,)), 2), ()),
            ("a long row", args, forged_matrix(((eighth, nought, nought), (nought, eight)), 2), ()),
            ("an impostor class", args, support.impostor(value), ()),
            ("a Matrix subclass", args, SubMatrix(rows), ()),
            ("a subclass named Matrix", args, Matrix(rows), ()),
            ("the rows alone", args, rows, ()),
            ("the rows as lists", args, [list(row) for row in rows], ()),
            ("None", args, None, ()),
            ("a str", args, "[[1/8, 0], [0, 8]]", ()),
            ("1 x 1: ncols True", args_of([[2]], 3), forged_matrix(((Rational(8),),), True), ()),
        ]
        self.assertRejectsAll(cases)
        self.assertAccepts("matrix_power", args_of([[2]], 3), matrix([[8]]))

    def test_rejects_wrong_arguments(self):
        rows = ((1, 1), (1, 0))
        M = matrix(rows)
        value = matrix([[89, 55], [55, 34]])
        args = {"M": M, "k": 10}
        self.assertAccepts("matrix_power", args, value)
        one = {"M": M, "k": 1}
        self.assertAccepts("matrix_power", one, M)
        self.assertAccepts("matrix_power", {"M": M, "k": 0}, matrix(identity(2)))
        r = M.rows
        cases = [(f"k = {k!r}", {"M": M, "k": k}, value, ())
                 for k in (10.0, "10", None, Rational(10), Integer(10), (10,), [10])]
        cases += [("k = True for M**1", {"M": M, "k": True}, M, ()),
                  ("k = False for M**0", {"M": M, "k": False}, matrix(identity(2)), ()),
                  ("k = 1.0", {"M": M, "k": 1.0}, M, ())]
        stand_ins = [
            ("M as tuples", rows), ("M as lists", [list(row) for row in rows]), ("M as the rows", r),
            ("an impostor M", support.impostor(M)), ("a Matrix-subclass M", SubMatrix(rows)),
            ("a subclass named Matrix", Matrix(rows)), ("M None", None), ("M an int", 1),
            ("M with int entries", forged_matrix(((1, 1), (1, 0)), 2)),
            ("M with a bool entry", forged_matrix(((r[0][0], r[0][1]), (r[1][0], False)), 2)),
            ("M with a float entry", forged_matrix(((1.0, r[0][1]), r[1]), 2)),
            ("M with a Fraction entry", forged_matrix(((Fraction(1), r[0][1]), r[1]), 2)),
            ("M with a Rational-subclass entry", forged_matrix(((SubRational(1), r[0][1]), r[1]), 2)),
            ("M with a zero denominator", forged_matrix(((forged_rational(1, 0), r[0][1]), r[1]), 2)),
            ("M with a negative denominator", forged_matrix(((forged_rational(-1, -1), r[0][1]), r[1]), 2)),
            ("M with a bool numerator", forged_matrix(((forged_rational(True, 1), r[0][1]), r[1]), 2)),
            ("M with rows as a list", forged_matrix(list(r), 2)),
            ("M with a row as a list", forged_matrix((list(r[0]), r[1]), 2)),
            ("M with a ragged row", forged_matrix((r[0], r[1][:1]), 2)),
            ("M with ncols 3", forged_matrix(r, 3)), ("M with ncols 2.0", forged_matrix(r, 2.0)),
            ("M with ncols missing", support.forged(mf.Matrix, rows=r)),
            ("M with rows missing", support.forged(mf.Matrix, ncols=2)),
        ]
        cases += [(label, {"M": stand_in, "k": 10}, value, ()) for label, stand_in in stand_ins]
        one_by_one = matrix([[1]])
        self.assertAccepts("matrix_power", {"M": one_by_one, "k": 10}, one_by_one)
        cases += [("1 x 1 M with ncols True", {"M": forged_matrix(one_by_one.rows, True), "k": 10}, one_by_one, ()),
                  ("1 x 1 M with an int entry", {"M": forged_matrix(((1,),), 1), "k": 10}, one_by_one, ())]
        cases += [
            ("arguments as pairs", tuple(args.items()), value, ()),
            ("arguments as a list of pairs", list(args.items()), value, ()),
            ("arguments as a read-only mapping", MappingProxyType(args), value, ()),
            ("None as arguments", None, value, ()),
            ("k missing", {"M": M}, value, ()),
            ("M missing", {"k": 10}, value, ()),
            ("an extra argument", {"M": M, "k": 10, "n": 2}, value, ()),
            ("A for M", {"A": M, "k": 10}, value, ()),
            ("n for k", {"M": M, "n": 10}, value, ()),
            ("no arguments", {}, value, ()),
        ]
        self.assertRejectsAll(cases)
        for operation in ("matrix_powers", "Matrix_power", "matrix_power ", "power", "pow", "matrix_exp", "",
                          None, 7, ("matrix_power",), b"matrix_power", ["matrix_power"], "continued_fraction",
                          "determinant", "inverse"):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, ())

    def test_rejects_any_evidence_but_the_empty_tuple(self):
        """The layout is (): data that a producer might hand over, internally consistent with the true claim (the
        common denominator, the bound, the modulus and the residues, the integer matrix s**k V), is refused like any
        other evidence."""
        args = args_of([[Fraction(1, 2), 1], [0, 3]], 2)
        value = matrix(naive_power(fractions_of([[Fraction(1, 2), 1], [0, 3]]), 2))
        self.assertAccepts("matrix_power", args, value, ())
        scaled = mf.Matrix(tuple(tuple(entry * 4 for entry in row) for row in value.rows))  # s**k V, s = 2, k = 2.
        self.assertTrue(all(entry.denominator == 1 for row in scaled.rows for entry in row))
        bound = 36  # M_int = [[1, 2], [0, 6]], so N = 6 and B = 6**2.
        residues = tuple(tuple(entry.numerator % MODULI[0] for entry in row) for row in scaled.rows)
        self.assertRejectsAll([
            ("None", args, value, None),
            ("a list", args, value, []),
            ("a str", args, value, "()"),
            ("an empty item", args, value, ((),)),
            ("a zero", args, value, (0,)),
            ("the value", args, value, (value,)),
            ("the request matrix", args, value, (args["M"],)),
            ("the common denominator", args, value, (2,)),
            ("the common denominator and k", args, value, (2, 2)),
            ("the bound", args, value, (bound,)),
            ("the modulus", args, value, (MODULI[0],)),
            ("the residues", args, value, (MODULI[0], residues)),
            ("the scaled integer matrix", args, value, (scaled,)),
            ("a list in a tuple", args, value, ([],)),
            ("a tuple subclass", args, value, Rows2(())),
        ])

    def test_oversized_forgeries_fail_fast(self):
        """Values far longer or larger than any true value, and giant evidence, are refused at once under the default
        budget: integers past max_integer_bits by their length, and within it by the bound B before any long
        arithmetic."""
        args = args_of([[1, 1], [1, 0]], 10)
        true = [[89, 55], [55, 34]]
        n64 = 64
        rng = random.Random(20261113)
        big_rows = [[rng.randrange(-1, 2) for _ in range(n64)] for _ in range(n64)]
        big_args = args_of(big_rows, 2)
        giant_entry = Rational(2 ** 60_000)  # One object, shared by every entry of the forged 64 x 64 value.
        third = args_of([[Fraction(1, 3)]], 30_000)
        cases = [
            ("an entry 2**70000", args, wide([[GIANT, 55], [55, 34]]), ()),
            ("an entry -2**70000", args, wide([[89, 55], [55, -GIANT]]), ()),
            ("an entry 1/2**70000", args, wide([[89, 55], [55, Fraction(1, GIANT)]]), ()),
            ("an entry 2**70000/3", args, wide([[89, 55], [55, Fraction(GIANT + 1, 3)]]), ()),
            ("an entry 2**60000", args, matrix([[2 ** 60_000, 55], [55, 34]]), ()),
            ("an entry 1/2**60000", args, matrix([[89, Fraction(1, 2 ** 60_000)], [55, 34]]), ()),
            ("an entry 3**37000/2**5000", args, matrix([[Fraction(3 ** 37_000, 2 ** 5_000), 55], [55, 34]]), ()),
            ("100,000 rows", args, forged_matrix(tuple(matrix(true).rows) * 50_000, 2), ()),
            ("a row of 100,000 entries", args, forged_matrix((matrix(true).rows[0] * 50_000,) * 2, 2), ()),
            ("ncols 10**100", args, forged_matrix(matrix(true).rows, 10 ** 100), ()),
            ("evidence of 100,000 zeros", args, matrix(true), (0,) * 100_000),
            ("evidence holding 100,000 zeros", args, matrix(true), ((0,) * 100_000,)),
            ("evidence 2**70000", args, matrix(true), (GIANT,)),
            ("a 64 x 64 value of 60,000-bit entries", big_args,
             forged_matrix(tuple((giant_entry,) * n64 for _ in range(n64)), n64), ()),
            # [[1/3]]**30000 = [[1/3**30000]]: q of 47,549 bits passes the screen, and the exact division refutes.
            ("1/(3**30000 + 2)", third, matrix([[Fraction(1, 3 ** 30_000 + 2)]]), ()),
            ("2/3**30000", third, matrix([[Fraction(2, 3 ** 30_000)]]), ()),
            ("1/3**29999", third, matrix([[Fraction(1, 3 ** 29_999)]]), ()),
            ("[[2]] for [[1]]**(2**65535)", args_of([[1]], 2 ** 65_535), matrix([[2]]), ()),
        ]
        for label, case_args, value, evidence in cases:
            with self.subTest(case=label):
                start = time.perf_counter()
                self.assertRejects("matrix_power", case_args, value, evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Resource limits and work

    def test_resource_limits_of_the_request(self):
        """ResourceLimitError, the claim unverified and never refuted, exactly for the request limits of the module:
        too many entries in M, a request integer past max_integer_bits, a common denominator past it (k >= 1), and
        the card's bound B = ||M_int||_inf**k of more than max_integer_bits bits; also for a true value."""
        oversized = support.oversized_matrix()
        self.assertExhausts({"M": oversized, "k": 0}, oversized)
        self.assertExhausts({"M": oversized, "k": 1}, oversized)
        # The card's bound, refused when k log2(N) exceeds max_integer_bits: B > 2**65536.
        self.assertExhausts(args_of([[2]], BITS + 1), wide([[2 ** (BITS + 1)]]))
        self.assertExhausts(args_of([[2]], BITS + 1), matrix([[1]]))
        self.assertExhausts(args_of([[3]], BITS), matrix([[1]]))  # k (bits(N) - 1) = 65536 with N no power of two.
        self.assertExhausts(args_of([[1, 1], [1, 0]], BITS + 1), matrix(identity(2)))  # N = 2.
        self.assertExhausts(args_of([[1, 1], [1, 0]], 10 ** 30), matrix(identity(2)))  # Decided from bits(N).
        self.assertExhausts(args_of([[Fraction(1, 2), Fraction(1, 2)], [0, 1]], 2 * BITS), matrix(identity(2)))
        limit = BITS * 10_000_000 // 15_849_625  # Near BITS / log2(3), then made exact.
        while (3 ** (limit + 1)).bit_length() <= BITS:
            limit += 1
        while (3 ** limit).bit_length() > BITS:
            limit -= 1
        self.assertEqual(limit, 41_348)
        self.assertExhausts(args_of([[3]], limit + 1), wide([[3 ** (limit + 1)]]))
        self.assertAccepts("matrix_power", args_of([[3]], limit), matrix([[3 ** limit]]))
        # A common denominator past max_integer_bits: (2**40000 - 1)(2**40000 + 1) has 80,000 bits. k = 0 needs none.
        p, q = 2 ** 40_000 - 1, 2 ** 40_000 + 1
        diagonal = matrix([[Fraction(1, p), 0], [0, Fraction(1, q)]])
        self.assertExhausts({"M": diagonal, "k": 1}, diagonal)
        self.assertAccepts("matrix_power", {"M": diagonal, "k": 0}, matrix(identity(2)))
        # Request integers past max_integer_bits.
        self.assertExhausts({"M": wide([[2 ** BITS]]), "k": 0}, matrix([[1]]))
        self.assertExhausts({"M": wide([[Fraction(1, 2 ** BITS)]]), "k": 0}, matrix([[1]]))
        self.assertExhausts(args_of([[1]], 2 ** BITS), matrix([[1]]))
        # Small limits.
        self.assertExhausts(args_of([[1, 1], [1, 0]], 10), matrix([[89, 55], [55, 34]]),
                            limits=ComputationLimits(max_work=1))
        self.assertExhausts(args_of([[1, 1], [1, 0]], 10), matrix([[89, 55], [55, 34]]),
                            limits=ComputationLimits(max_matrix_entries=3))
        self.assertExhausts(args_of([[1, 1], [1, 0]], 10), matrix([[89, 55], [55, 34]]),
                            limits=ComputationLimits(max_integer_bits=6))  # B = 2**10 has 11 bits.

    def test_bound_of_exactly_two_to_the_limit(self):
        """B = 2**65536 itself, from a power of two N with k log2(N) = 65536, is no resource limit: the card refuses
        only a k log2(N) that exceeds max_integer_bits (a regression: B >= 2**65536 was refused). The claim is
        decided: true values verify, wrong ones are refuted, and a true value holding +-2**65536, which no value may
        hold, answers False like a denominator past max_integer_bits."""
        half = Fraction(1, 2)
        scaled = Fraction(1, 2 ** (BITS // 2))  # (M_int / 2)**65536, M_int = [[1, 1], [-1, 1]] with M_int**8 = 16 I.
        for args, value in ((args_of([[1, 1], [1, 0]], BITS), matrix(fibonacci_power(BITS))),
                            (args_of([[half, half], [-half, half]], BITS), matrix([[scaled, 0], [0, scaled]]))):
            with self.subTest(k=args["k"]):
                start = time.perf_counter()
                self.assertAccepts("matrix_power", args, value)
                self.assertLess(time.perf_counter() - start, FAST)
        self.assertRejects("matrix_power", args_of([[1, 1], [1, 0]], BITS), matrix(identity(2)))
        self.assertRejects("matrix_power", args_of([[1, 1], [1, 0]], BITS), matrix(fibonacci_power(BITS - 1)))
        self.assertRejects("matrix_power", args_of([[2]], BITS), wide([[2 ** BITS]]))  # True, but past the limit.
        self.assertRejects("matrix_power", args_of([[2]], BITS), matrix([[1]]))
        self.assertRejects("matrix_power", args_of([[Fraction(4, 3)]], BITS // 2),
                           wide([[Fraction(2 ** BITS, 3 ** (BITS // 2))]]))

    def test_work_is_charged_before_it_runs(self):
        """The charges depend on the claim alone: a budget one unit short of a check refuses it, and its exact size
        passes. The search for a prime past the table is charged, and a long chain stops at the budget."""
        rng = random.Random(20261114)
        rows = [[Fraction(rng.randrange(-5, 6), rng.randrange(1, 4)) for _ in range(5)] for _ in range(5)]
        claims = [(args_of(rows, 37), matrix(binary_power(rows, 37))),
                  (args_of([[1, 1], [1, 0]], 5000), matrix(fibonacci_power(5000)))]  # 82 primes, 18 past the table.
        for args, value in claims:
            with computation(fresh=True) as budget:
                self.assertIs(checker.check_operation("matrix_power", args, value, ()), True)
            work = budget.work
            with self.assertRaises(ResourceLimitError):
                support.verdict(family, "matrix_power", args, value, limits=ComputationLimits(max_work=work - 1))
            self.assertAccepts("matrix_power", args, value, limits=ComputationLimits(max_work=work))
        # Each prime past the table of 64 is charged before its search: 2B = 2**5001 takes 82 primes.
        self.assertEqual(prime_count(2 ** 5001), 82)
        self.assertGreater(work, (82 - len(MODULI)) * checker.EXTRA_PRIME_WORK)
        # A long chain runs out of the default budget within FAST seconds, and of a small one at once. The claims
        # pass the bound and the screen (B = 1 for the identity; 64**1000 < 2**65536 for the 64 x 64 matrix), so the
        # budget runs out inside the chain.
        self.assertExhausts({"M": matrix(identity(8)), "k": 2 ** 60_000}, matrix(identity(8)))
        big = [[rng.randrange(-1, 2) for _ in range(64)] for _ in range(64)]
        self.assertExhausts(args_of(big, 1000), matrix(big))
        self.assertExhausts(args_of([[1, 1], [1, 0]], 4000), matrix(fibonacci_power(4000)),
                            limits=ComputationLimits(max_work=1_000))  # It takes about 3,000 units.

    def test_full_size_claims_fit_the_default_budget(self):
        """Claims at the edge of the request limits verify under the default budget: B of 65,536 bits, which takes
        1,075 primes, a value integer of 65,536 bits, a denominator s**k of 64,984 bits and a 64 x 64 matrix."""
        self.assertEqual(prime_count(2 ** (BITS - 1) * 2), 1_075)
        for args, value in ((args_of([[1, 1], [1, 0]], BITS - 1), matrix(fibonacci_power(BITS - 1))),
                            (args_of([[2]], BITS - 1), matrix([[2 ** (BITS - 1)]])),
                            (args_of([[Fraction(1, 3)]], 41_000), matrix([[Fraction(1, 3 ** 41_000)]]))):
            with self.subTest(k=args["k"]):
                start = time.perf_counter()
                self.assertAccepts("matrix_power", args, value)
                self.assertLess(time.perf_counter() - start, FAST)
        self.assertEqual(value.rows[0][0].denominator.bit_length(), 64_984)
        self.assertRejects("matrix_power", args_of([[1, 1], [1, 0]], BITS - 1),
                           matrix([[fibonacci(BITS), fibonacci(BITS - 1)], [fibonacci(BITS - 1), fibonacci(BITS - 1)]]))
        rng = random.Random(20261115)
        big = [[rng.randrange(-1, 2) for _ in range(64)] for _ in range(64)]
        cube = integer_product(integer_product(big, big), big)
        self.assertAccepts("matrix_power", args_of(big, 3), matrix(cube))
        cube[63][0] += 1
        self.assertRejects("matrix_power", args_of(big, 3), matrix(cube))

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_the_exact_steps_decide_without_the_screen(self):
        """The screen by bit lengths only saves time: without it, the exact steps (the division of s**k, the bound
        and the residues) still refute values that it would have refused, and wrong values within the bound."""
        args = args_of([[1, 1], [1, 0]], 10)
        with patch.object(checker, "_screened", lambda *arguments: True):
            self.assertAccepts("matrix_power", args, matrix([[89, 55], [55, 34]]))
            for value in ([[89 + MODULI[0], 55], [55, 34]], [[89 - MODULI[0], 55], [55, 34]],
                          [[2 ** 60_000, 55], [55, 34]], [[89, Fraction(1, 2 ** 60_000)], [55, 34]],
                          [[89, 55], [55, Fraction(69, 2)]], [[90, 55], [55, 34]], [[89, 55], [55, 35]]):
                with self.subTest(value=value):
                    self.assertRejects("matrix_power", args, matrix(value))
            third = args_of([[Fraction(1, 3)]], 300)
            self.assertAccepts("matrix_power", third, matrix([[Fraction(1, 3 ** 300)]]))
            self.assertRejects("matrix_power", third, matrix([[Fraction(1, 3 ** 299)]]))
            self.assertRejects("matrix_power", third, matrix([[Fraction(2, 3 ** 300)]]))

    def test_lowest_terms_are_checked(self):
        """8 as 16/2 and 0 as 0/2 have the right values as numbers, and their denominators divide s**k = 8: only the
        test of lowest terms refuses them. (1/8 as 2/16 fails the division of s**k by 16 as well.)"""
        args = args_of([[Fraction(1, 2), 0], [0, 2]], 3)
        (eighth, nought), (_, eight) = matrix([[Fraction(1, 8), 0], [0, 8]]).rows
        for value in (forged_matrix(((eighth, nought), (nought, forged_rational(16, 2))), 2),
                      forged_matrix(((eighth, forged_rational(0, 2)), (nought, eight)), 2)):
            self.assertRejects("matrix_power", args, value)
            with patch.object(checker, "_canonical", lambda *arguments: True):
                self.assertAccepts("matrix_power", args, value)
        self.assertRejects("matrix_power", args, forged_matrix(((forged_rational(2, 16), nought), (nought, eight)), 2))

    def test_no_builtin_pow_and_no_double_star(self):
        """Every power is a charged square-and-multiply loop: the checker calls no pow and writes no **."""
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
        self.assertEqual(imported, {"__future__", "math", "operator", "mathforge.limits", "mathforge.model",
                                    "mathforge.trusted.integers", "mathforge.trusted.primes"})

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip, while produce() would
        fail: the Matrix value and the request survive the format, and a wrong value fails the mathematics check."""
        claims = [
            (args_of([[1, 1], [1, 0]], 10), matrix([[89, 55], [55, 34]]), matrix([[89, 55], [55, 33]])),
            (args_of([[Fraction(1, 2), 0], [0, 2]], 3), matrix([[Fraction(1, 8), 0], [0, 8]]),
             matrix([[Fraction(1, 4), 0], [0, 8]])),
            (args_of([[1, 2], [2, 4]], 0), matrix(identity(2)), matrix([[1, 2], [2, 4]])),
            ({"M": mf.Matrix(()), "k": 3}, mf.Matrix(()), matrix([[0]])),
            (args_of([[1, 1], [1, 0]], 1000), matrix(fibonacci_power(1000)), matrix(fibonacci_power(999))),
        ]
        with patch.object(family, "produce", side_effect=AssertionError("the producer ran")):
            for args, value, wrong in claims:
                with self.subTest(k=args["k"]):
                    self.roundtrip(claimed(args, value))
                    report = support.statuses(mf.verify(claimed(args, wrong)))
                    self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                    self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
                    report = support.statuses(mf.verify(claimed(args, value, ((),))))
                    self.assertIs(report["result.mathematics"], CheckStatus.FAILED)  # Evidence where none belongs.

