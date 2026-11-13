"""The forms family of mathforge.linalg: definiteness, gram_schmidt.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register linalg forms

The references of this module are short and slow, over fractions.Fraction, and share nothing with the producer's
fraction-free eliminations:
- definiteness: the counts of positive, negative and zero eigenvalues by Descartes' rule of signs on the characteristic
  polynomial from the Faddeev-LeVerrier recurrence (exact for a symmetric matrix, whose eigenvalues are all real); the
  evidence (S, D) is checked by recomputing S^T M S over Fractions and the determinant of S by Gaussian elimination.
- gram_schmidt: the textbook recurrence q_i = m_i - sum_{j<i} mu_ij q_j over Fractions, which gives both Q and mu.

Run this module by name with

    python -m unittest tests.v09.linalg.test_forms
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.linalg import forms as family
from mathforge.areas.linalg.forms import Inertia
from mathforge.contracts import ErrorInfo, ExecutionStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import Budget, ComputationLimits, ResourceLimitError, computation
from mathforge.model import trusted_rational

from .. import support

Q = mf.Rational
F = Fraction
DEF = "definiteness"
GS = "gram_schmidt"
OPERATIONS = (DEF, GS)
FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
WIDE = ComputationLimits(max_integer_bits=300_000, max_matrix_entries=1_000_000)  # For matrices past the limits.
STAND_INS = (0.5, True, "1", None, 1, Q(1), (1, 2), [1], ((1, 0), (0, 1)), [[1, 0], [0, 1]])
Rows = list[list[Fraction]]


# ---------------------------------------------------------------------------------------------------------------
# References over fractions.Fraction.

def fraction(value: object) -> Fraction:
    if type(value) is Q:
        return F(value.numerator, value.denominator)
    return F(value)


def rational(value: Fraction) -> mf.Rational:
    return Q(value.numerator, value.denominator)


def fractions(rows: Sequence[Sequence[object]]) -> Rows:
    return [[fraction(x) for x in row] for row in rows]


def matrix(rows: Sequence[Sequence[object]], ncols: int | None = None) -> mf.Matrix:
    """The Matrix of rows of exact numbers (ints, Fractions or Rationals); [] is the matrix with no row."""
    if not rows:
        return mf.Matrix((), ncols=0 if ncols is None else ncols)
    return mf.Matrix(tuple(tuple(rational(fraction(x)) for x in row) for row in rows))


def wide(rows: Sequence[Sequence[object]]) -> mf.Matrix:
    """matrix(rows), built inside a budget that admits integers past the default limits."""
    with computation(WIDE, fresh=True):
        return matrix(rows)


def multiply(a: Rows, b: Rows) -> Rows:
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0)) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a: Rows) -> Rows:
    return [list(column) for column in zip(*a)]


def determinant(a: Rows) -> Fraction:
    """By Gaussian elimination over Fractions."""
    work = [row[:] for row in a]
    n, result = len(work), F(1)
    for column in range(n):
        pivot = next((i for i in range(column, n) if work[i][column]), None)
        if pivot is None:
            return F(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result = -result
        result *= work[column][column]
        for i in range(column + 1, n):
            factor = work[i][column] / work[column][column]
            work[i] = [x - factor * y for x, y in zip(work[i], work[column])]
    return result


def charpoly(m: Rows) -> list[Fraction]:
    """det(t I - M), lowest degree first, by the Faddeev-LeVerrier recurrence."""
    n = len(m)
    coefficients = [F(0)] * n + [F(1)]
    current = [[F(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        current = multiply(m, current)
        for i in range(n):
            current[i][i] += coefficients[n - k + 1]
        product = multiply(m, current)
        coefficients[n - k] = -sum(product[i][i] for i in range(n)) / k
    return coefficients


def sign_changes(values: Sequence[Fraction]) -> int:
    signs = [value > 0 for value in values if value]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def kind_of(positive: int, negative: int, zero: int) -> str:
    n = positive + negative + zero
    if positive == n:
        return "positive_definite"
    if negative == n:
        return "negative_definite"
    if negative == 0:
        return "positive_semidefinite"
    if positive == 0:
        return "negative_semidefinite"
    return "indefinite"


def reference_inertia(rows: Sequence[Sequence[object]]) -> Inertia:
    """The inertia of a symmetric M by Descartes' rule on its characteristic polynomial."""
    m = fractions(rows)
    if not m:
        return Inertia(0, 0, 0, "positive_definite")
    p = charpoly(m)
    zero = next(index for index, value in enumerate(p) if value)
    rest = p[zero:]
    mirrored = [value if index % 2 == 0 else -value for index, value in enumerate(rest)]
    counts = (sign_changes(rest), sign_changes(mirrored), zero)
    return Inertia(*counts, kind_of(*counts))


def reference_gram_schmidt(rows: Sequence[Sequence[object]]) -> tuple[Rows, list[list[Fraction]]] | None:
    """(Q, mu) of the textbook recurrence, or None for dependent rows."""
    q: Rows = []
    mu: list[list[Fraction]] = []
    for row in fractions(rows):
        coefficients = [sum((a * b for a, b in zip(row, other)), F(0)) / sum((b * b for b in other), F(0))
                        for other in q]
        current = list(row)
        for coefficient, other in zip(coefficients, q):
            current = [a - coefficient * b for a, b in zip(current, other)]
        if not any(current):
            return None
        q.append(current)
        mu.append(coefficients)
    return q, mu


def random_symmetric(rng: random.Random, n: int, rank: int | None = None) -> Rows:
    """A random symmetric rational matrix; with rank, B^T diag(+-1) B for a rank x n B, so of rank at most rank."""
    if rank is None:
        m = [[F(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                m[i][j] = m[j][i] = F(rng.randint(-9, 9), rng.choice((1, 1, 1, 2, 3)))
        return m
    if not rank:
        return [[F(0)] * n for _ in range(n)]
    b = [[F(rng.randint(-3, 3), rng.choice((1, 1, 2))) for _ in range(n)] for _ in range(rank)]
    d = [[F(rng.choice((-1, 1)) if i == j else 0) for j in range(rank)] for i in range(rank)]
    return multiply(multiply(transpose(b), d), b)


def sparse_symmetric(rng: random.Random, n: int) -> Rows:
    """A symmetric matrix with many zeros, also on the diagonal: zero pivots, swaps and the 2 x 2 congruence step."""
    m = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if rng.random() < 0.3:
                m[i][j] = m[j][i] = F(rng.randint(-3, 3))
    return m


def forged_matrix(rows: object, ncols: object) -> mf.Matrix:
    return support.forged(mf.Matrix, rows=rows, ncols=ncols)


def forged_rational(numerator: object, denominator: object) -> mf.Rational:
    return support.forged(Q, numerator=numerator, denominator=denominator)


def items(value: object) -> int:
    """The items of a value as Budget.inspect counts them, in a budget wide enough for any of them."""
    counter = Budget(ComputationLimits(max_evidence_items=10 ** 9, max_integer_bits=10 ** 7, max_depth=1_000))
    counter.inspect(value, evidence=True)
    return counter.evidence_count


class SubMatrix(mf.Matrix):
    """A Matrix subclass with the right rows: no transport value."""


# The card's known values.
KNOWN_DEFINITENESS = (
    ([[2, 1], [1, 2]], (2, 0, 0, "positive_definite")),
    ([[1, 2], [2, 1]], (1, 1, 0, "indefinite")),
    ([[0, 0], [0, 0]], (0, 0, 2, "positive_semidefinite")),
    ([[0, 1], [1, 0]], (1, 1, 0, "indefinite")),
    ([[-1]], (0, 1, 0, "negative_definite")),
    ([[1, 1], [1, 1]], (1, 0, 1, "positive_semidefinite")),
)
KNOWN_GRAM_SCHMIDT = (
    ([[3, 1], [2, 2]], [[3, 1], [F(-2, 5), F(6, 5)]], [[], [F(4, 5)]]),
    ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[], [0]]),
    ([[1, 1, 0], [1, 0, 1]], [[1, 1, 0], [F(1, 2), F(-1, 2), 1]], [[], [F(1, 2)]]),
)
# More inputs: the 0 x 0 matrix, zero pivots, the 2 x 2 congruence step, rows that leave the elimination, rationals.
SPECIAL_DEFINITENESS = {
    "empty": ([], (0, 0, 0, "positive_definite")),
    "zero_1x1": ([[0]], (0, 0, 1, "positive_semidefinite")),
    "negative_pair": ([[-2, 0], [0, -3]], (0, 2, 0, "negative_definite")),
    "negative_semidefinite": ([[-1, 0], [0, 0]], (0, 1, 1, "negative_semidefinite")),
    "zero_first": ([[0, 0], [0, -1]], (0, 1, 1, "negative_semidefinite")),
    "congruence_past_zero_row": ([[0, 0, 1], [0, 0, 0], [1, 0, 0]], (1, 1, 1, "indefinite")),
    "rank_one": ([[1, 2, 3], [2, 4, 6], [3, 6, 9]], (1, 0, 2, "positive_semidefinite")),
    "hilbert": ([[F(1, i + j + 1) for j in range(4)] for i in range(4)], (4, 0, 0, "positive_definite")),
    "mixed_denominators": ([[F(1, 2), F(1, 3)], [F(1, 3), F(-1, 5)]], (1, 1, 0, "indefinite")),
    "zero_3x3": ([[0] * 3 for _ in range(3)], (0, 0, 3, "positive_semidefinite")),
}
SPECIAL_GRAM_SCHMIDT = {
    "no_row_no_column": ([], 0),
    "no_row_three_columns": ([], 3),
    "one_row": ([[0, -2, 0]], 3),
    "orthogonal": ([[1, 1], [1, -1]], 2),
    "rational": ([[F(1, 2), F(1, 3)], [F(-1, 7), 2]], 2),
    "square_basis": ([[1, 2, 3], [4, 5, 6], [7, 8, 10]], 3),
    "triangular": ([[0, 0, 1], [0, 1, 1], [1, 1, 1]], 3),
    "row_denominators": ([[F(1, 3), F(2, 3), 0, 1], [F(5, 7), 0, F(1, 2), 1], [1, 1, 1, F(1, 11)]], 4),
}


class FormsTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object = None) -> None:
        """A completed v6 Result, of that value when one is given, with its four checks, which mf.verify accepts."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        if value is not None:
            self.assertEqual(result.value, value)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertDefinitenessAgrees(self, rows: Sequence[Sequence[object]]) -> Inertia:
        """produce() answers the reference inertia with evidence of the layout that holds over Fractions, and the
        checker accepts it; returns the value."""
        args = {"M": matrix(rows)}
        value, evidence = family.produce(DEF, dict(args))
        self.assertIs(type(value), Inertia)
        self.assertEqual(value, reference_inertia(rows))
        n = len(rows)
        self.assertIs(type(evidence), tuple)
        self.assertEqual(len(evidence), 2)
        s_rows, diagonal = evidence
        self.assertIs(type(s_rows), tuple)
        self.assertIs(type(diagonal), tuple)
        self.assertEqual([type(row) for row in s_rows], [tuple] * n)
        self.assertTrue(all(type(x) is Q and len(row) == n for row in s_rows for x in row))
        self.assertTrue(all(type(x) is Q for x in diagonal))
        self.assertEqual(len(diagonal), n)
        if n:
            s = fractions(s_rows)
            congruent = multiply(multiply(transpose(s), fractions(rows)), s)
            self.assertEqual(congruent, [[fraction(diagonal[i]) if i == j else F(0) for j in range(n)]
                                         for i in range(n)])
            self.assertNotEqual(determinant(s), 0)
        self.assertIs(support.verdict(family, DEF, args, value, evidence), True)
        return value

    def assertGramSchmidtAgrees(self, rows: Sequence[Sequence[object]], ncols: int | None = None) -> mf.Matrix:
        """produce() answers the reference Q with the reference mu, and the checker accepts it; returns the value."""
        args = {"M": matrix(rows, ncols)}
        value, evidence = family.produce(GS, dict(args))
        q, mu = reference_gram_schmidt(rows)
        self.assertIs(type(value), mf.Matrix)
        self.assertEqual(value, matrix(q, args["M"].ncols))
        self.assertEqual(value.ncols, args["M"].ncols)
        self.assertIs(type(evidence), tuple)
        self.assertEqual([tuple(fraction(x) for x in row) for row in evidence], [tuple(row) for row in mu])
        self.assertTrue(all(type(row) is tuple and all(type(x) is Q for x in row) for row in evidence))
        self.assertIs(support.verdict(family, GS, args, value, evidence), True)
        return value

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for rows, fields in KNOWN_DEFINITENESS + tuple(SPECIAL_DEFINITENESS.values()):
            with self.subTest(operation=DEF, rows=rows):
                expected = Inertia(*fields)
                self.assertEqual(reference_inertia(rows), expected)
                result = support.run(DEF, M=matrix(rows))
                self.assertVerified(result, expected)
                self.roundtrip(result)
                self.assertDefinitenessAgrees(rows)
        for rows, expected, mu in KNOWN_GRAM_SCHMIDT:
            with self.subTest(operation=GS, rows=rows):
                result = support.run(GS, M=matrix(rows))
                self.assertVerified(result, matrix(expected))
                self.assertEqual(result.steps[0].outputs[1], tuple(tuple(rational(F(x)) for x in row) for row in mu))
                self.roundtrip(result)
        for name, (rows, ncols) in SPECIAL_GRAM_SCHMIDT.items():
            with self.subTest(operation=GS, name=name):
                result = support.run(GS, M=matrix(rows, ncols))
                self.assertVerified(result)
                self.assertEqual(result.value.ncols, ncols)
                self.roundtrip(result)
                self.assertGramSchmidtAgrees(rows, ncols)
        # The evidence layout on the card's first matrices and on the empty ones.
        result = support.run(DEF, M=matrix([[2, 1], [1, 2]]))
        # The producer's S = T U^T is an integer matrix (module docstring of the family).
        self.assertEqual(result.steps[0].outputs[1], (((Q(1), Q(-1)), (Q(0), Q(2))), (Q(2), Q(6))))
        self.assertEqual(support.run(DEF, M=matrix([])).steps[0].outputs[1], ((), ()))
        result = support.run(GS, M=mf.Matrix((), ncols=3))
        self.assertEqual(result.value, mf.Matrix((), ncols=3))
        self.assertEqual(result.steps[0].outputs[1], ())

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error; so are
        a non-square and a non-symmetric M for definiteness, dependent rows for gram_schmidt and a Matrix forged
        without its constructor. produce() tests the types itself too."""
        for operation in OPERATIONS:
            for item in STAND_INS:
                with self.subTest(operation=operation, M=item):
                    self.assertInvalidInput(support.run(operation, M=item))
        for M in (matrix([[1, 2, 3], [2, 5, 6]]), matrix([[1, 2], [2, 3], [5, 6]]), mf.Matrix((), ncols=3),
                  mf.Matrix(((), ())), matrix([[1, 2, 3]])):
            with self.subTest(operation=DEF, shape=M.shape):
                result = support.run(DEF, M=M)
                self.assertInvalidInput(result)
                self.assertIn("square matrix M", result.error.message)
        for rows in ([[1, 2], [3, 1]], [[0, 1], [0, 0]], [[1, 0, 0], [0, 1, 0], [F(1, 2), 0, 1]],
                     [[1, F(1, 2)], [F(1, 3), 1]]):
            with self.subTest(operation=DEF, rows=rows):
                result = support.run(DEF, M=matrix(rows))
                self.assertInvalidInput(result)
                self.assertIn("symmetric", result.error.message)
        dependent = ([[0, 0]], [[1, 2], [2, 4]], [[1, 0], [0, 1], [1, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[1, 1, 0], [0, 0, 0], [0, 1, 1]], [[F(1, 2), 1], [1, 2]], [[1], [2]], [[1, 0], [1, 0]],
                     [[1, 2], [3, 4], [5, 6]])
        for rows in dependent:
            with self.subTest(operation=GS, rows=rows):
                result = support.run(GS, M=matrix(rows))
                self.assertInvalidInput(result)
                self.assertIn("linearly independent rows", result.error.message)
        self.assertInvalidInput(support.run(GS, M=mf.Matrix(((), ()))))  # Two zero rows of no entry.
        one, nought = Q(1), Q(0)
        forged = {"a float entry": forged_matrix(((one, 0.5), (0.5, one)), 2),
                  "an int entry": forged_matrix(((one, 1), (1, one)), 2),
                  "a bool entry": forged_matrix(((one, True), (True, one)), 2),
                  "a str entry": forged_matrix(((one, "1"), ("1", one)), 2),
                  "None as an entry": forged_matrix(((one, None), (None, one)), 2),
                  "a Fraction entry": forged_matrix(((one, F(1, 2)), (F(1, 2), one)), 2),
                  "a zero denominator": forged_matrix(((one, nought), (nought, forged_rational(1, 0))), 2),
                  "a negative denominator": forged_matrix(((one, nought), (nought, forged_rational(1, -2))), 2),
                  "a bool numerator": forged_matrix(((one, nought), (nought, forged_rational(True, 2))), 2),
                  "4/2": forged_matrix(((one, nought), (nought, forged_rational(4, 2))), 2),
                  "0/2": forged_matrix(((one, nought), (nought, forged_rational(0, 2))), 2),
                  "rows as a list": forged_matrix([(one, nought), (nought, one)], 2),
                  "a row as a list": forged_matrix(([one, nought], (nought, one)), 2),
                  "a ragged row": forged_matrix(((one, nought), (nought,)), 2),
                  "ncols None": forged_matrix(((one, nought), (nought, one)), None),
                  "ncols 2.0": forged_matrix(((one, nought), (nought, one)), 2.0),
                  "ncols True": forged_matrix(((one,),), True),
                  "ncols -1": forged_matrix((), -1)}
        for operation in OPERATIONS:
            for label, M in forged.items():
                with self.subTest(operation=operation, forged=label):
                    self.assertInvalidInput(support.run(operation, M=M))
        M = matrix([[2, 1], [1, 2]])
        # A Rational forged with a float or without a denominator: run()'s inspection of the request reads both fields
        # before produce() runs (that is run()'s error, not this family's), so only produce() is asked here.
        unreadable = [forged_matrix(((one, nought), (nought, forged_rational(1, 2.0))), 2),
                      forged_matrix(((one, nought), (nought, support.forged(Q, numerator=1))), 2)]
        stand_ins = list(forged.values()) + unreadable + list(STAND_INS) + [
            SubMatrix(((2, 1), (1, 2))), M.rows, support.impostor(M), support.forged(mf.Matrix, rows=M.rows),
            support.forged(mf.Matrix, ncols=2)]
        for operation in OPERATIONS:
            for item in stand_ins:
                with self.subTest(path="produce", operation=operation, M=item), self.assertRaises(InvalidInput):
                    family.produce(operation, {"M": item})
            for arguments in ({"M": M, "k": 2}, {}, {"A": M}, {"m": M}, tuple({"M": M}.items()),
                              MappingProxyType({"M": M}), None, [M]):
                with self.subTest(path="produce", operation=operation, arguments=arguments):
                    with self.assertRaises(InvalidInput):
                        family.produce(operation, arguments)
        for operation in ("Definiteness", "gram-schmidt", "definiteness ", "inertia", "", None, (DEF,)):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, {"M": M})
        with self.assertRaises(InvalidInput):
            family.produce(DEF, {"M": matrix([[1, 2], [3, 4]])})
        with self.assertRaises(InvalidInput):
            family.produce(GS, {"M": matrix([[1, 2], [2, 4]])})
        for operation, bad in ((DEF, matrix([[1, 2, 3]])), (GS, matrix([[1, 1], [1, 1]]))):
            with self.subTest(value_function=operation), self.assertRaises(mf.OperationError) as raised:
                support.value(operation, M=bad)
            self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536, a 65 x 65 matrix and max_work=1 exceed the budget; so do an integer of the
        evidence or the value past a smaller max_integer_bits (the checker's refusals), long entries whose work passes
        max_work and a Result that the verifier would not admit under max_evidence_items. Each request fails at
        once."""
        start = time.perf_counter()
        oversized = support.oversized_matrix()
        with computation(ComputationLimits(max_matrix_entries=65 * 66), fresh=True):
            oblong = mf.Matrix(tuple(tuple((i + j) % 5 for j in range(66)) for i in range(65)))
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                self.assertResourceLimit(support.run(operation, M=wide([[support.HUGE]])))
                self.assertResourceLimit(support.run(operation, M=wide([[1, 0], [0, F(1, support.HUGE)]])))
                self.assertResourceLimit(support.run(operation, M=oversized))
                self.assertResourceLimit(support.run(operation, M=oblong))  # Not square, but too large first.
                for M in (oversized, oblong):
                    with self.assertRaises(ResourceLimitError):
                        family.produce(operation, {"M": M})
                for rows in ([[2, 1], [1, 2]], [], [[F(1, 2)]], [[0, 1], [1, 0]]):
                    with self.subTest(max_work=1, rows=rows):
                        self.assertResourceLimit(support.run(operation, M=matrix(rows),
                                                             limits=ComputationLimits(max_work=1)))
        # Under 64-bit integers: D = (2**40, 2**40 (2**80 - 1)) has an integer past the limit, so the checker would
        # answer False; for gram_schmidt q_1 has the denominator 2**80 + 1.
        small = ComputationLimits(max_integer_bits=64)
        self.assertResourceLimit(support.run(DEF, M=matrix([[2 ** 40, 1], [1, 2 ** 40]]), limits=small))
        self.assertResourceLimit(support.run(GS, M=matrix([[2 ** 40, 1], [1, 0]]), limits=small))
        for operation, rows in ((DEF, [[2 ** 40, 1], [1, 2 ** 40]]), (GS, [[2 ** 40, 1], [1, 0]])):
            with computation(small, fresh=True), self.assertRaises(ResourceLimitError):
                family.produce(operation, {"M": matrix(rows)})
        # The same requests fit the default limits.
        self.assertVerified(support.run(DEF, M=matrix([[2 ** 40, 1], [1, 2 ** 40]])))
        self.assertVerified(support.run(GS, M=matrix([[2 ** 40, 1], [1, 0]])))
        # Long entries: max_work refuses them before the long-integer work runs.
        rng = random.Random(20261113)
        long_rows = [[0] * 30 for _ in range(30)]
        for i in range(30):
            for j in range(i, 30):
                long_rows[i][j] = long_rows[j][i] = rng.getrandbits(3_000) - rng.getrandbits(3_000)
        long_gs = [[rng.getrandbits(3_000) - rng.getrandbits(3_000) for _ in range(30)] for _ in range(30)]
        self.assertResourceLimit(support.run(DEF, M=matrix(long_rows)))
        self.assertResourceLimit(support.run(GS, M=matrix(long_gs)))
        self.assertLess(time.perf_counter() - start, 4 * FAST)
        # max_evidence_items: the Result's own items, the request twice, the value twice and the evidence once.
        for operation, rows in ((DEF, [[2, 1], [1, 2]]), (DEF, []), (DEF, SPECIAL_DEFINITENESS["hilbert"][0]),
                                (GS, [[3, 1], [2, 2]]), (GS, SPECIAL_GRAM_SCHMIDT["row_denominators"][0])):
            with self.subTest(operation=operation, rows=rows, evidence_items=True):
                M = matrix(rows)
                result = support.run(operation, M=M)
                self.assertVerified(result)
                request = family._REQUEST_ITEMS + items(M)
                self.assertEqual(items(result.request), request)
                total = (family._RESULT_ITEMS + 2 * request + 2 * items(result.value)
                         + items(result.steps[0].outputs[1]))
                self.assertEqual(items(result), total)
                self.assertEqual(items(mf.loads(mf.dumps(result))), total)
                limits = ComputationLimits(max_evidence_items=total)
                at_limit = support.run(operation, M=M, limits=limits)
                self.assertVerified(at_limit, result.value)
                self.assertTrue(mf.verify(at_limit, limits=limits).verified)
                below = support.run(operation, M=M, limits=ComputationLimits(max_evidence_items=total - 1))
                self.assertResourceLimit(below)
                self.assertIn("evidence_count", below.error.message)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: S with
        permuted, scaled or eigenvector columns, and Rationals that are not in lowest terms."""
        args = {"M": matrix([[2, 1], [1, 2]])}
        value, (s_rows, diagonal) = family.produce(DEF, dict(args))
        self.assertAccepts(DEF, args, value, ((rows_of([[1, 1], [1, -1]])), (Q(6), Q(2))))
        swapped = tuple((row[1], row[0]) for row in s_rows)
        self.assertAccepts(DEF, args, value, (swapped, (diagonal[1], diagonal[0])))
        halved = tuple((row[0] / 2, row[1] * 3) for row in s_rows)
        self.assertAccepts(DEF, args, value, (halved, (diagonal[0] / 4, diagonal[1] * 9)))
        self.assertAccepts(DEF, args, value, (((trusted_rational(2, 2), Q(-1, 2)), (Q(0), Q(1))), (Q(2), Q(3, 2))))
        args = {"M": matrix([[3, 1], [2, 2]])}
        q, mu = family.produce(GS, dict(args))
        self.assertAccepts(GS, args, q, mu)
        self.assertAccepts(GS, args, q, ((), (trusted_rational(8, 10),)))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        args = {"M": matrix([[2, 1], [1, 2]])}
        value, evidence = family.produce(DEF, dict(args))
        s_rows, diagonal = evidence
        corruptions = {
            "a wrong inertia": (DEF, Inertia(1, 1, 0, "indefinite"), evidence),
            "a wrong kind": (DEF, support.forged(Inertia, positive=2, negative=0, zero=0, kind="indefinite"),
                             evidence),
            "a bool count": (DEF, support.forged(Inertia, positive=True, negative=0, zero=1,
                                                 kind="positive_semidefinite"), evidence),
            "a float count": (DEF, support.forged(Inertia, positive=2.0, negative=0, zero=0,
                                                  kind="positive_definite"), evidence),
            "a str count": (DEF, support.forged(Inertia, positive="2", negative=0, zero=0, kind="positive_definite"),
                            evidence),
            "a missing field": (DEF, support.forged(Inertia, positive=2, negative=0, zero=0), evidence),
            "an impostor": (DEF, support.impostor(value), evidence),
            "a wrong D": (DEF, value, (s_rows, (Q(2), Q(2)))),
            "an int in S": (DEF, value, (((1, Q(-1)), s_rows[1]), diagonal)),
            "a float in D": (DEF, value, (s_rows, (2.0, diagonal[1]))),
            "S as a list": (DEF, value, (list(s_rows), diagonal)),
            "a singular S": (DEF, Inertia(0, 0, 2, "positive_semidefinite"), (((Q(0), Q(0)), (Q(0), Q(0))),
                                                                             (Q(0), Q(0)))),
            "missing evidence": (DEF, value, (s_rows,)),
            "extra evidence": (DEF, value, evidence + ((),)),
            "an oversized D": (DEF, value, (s_rows, (trusted_rational(2 ** 70_000, 1), diagonal[1]))),
        }
        gs_args = {"M": matrix([[3, 1], [2, 2]])}
        q, mu = family.produce(GS, dict(gs_args))
        corruptions.update({
            "a wrong Q": ("gram_schmidt", matrix([[3, 1], [-2, 6]]), mu),
            "Q not in lowest terms": (GS, forged_matrix(((Q(3), Q(1)), (trusted_rational(-4, 10), Q(6, 5))), 2), mu),
            "a Q impostor": (GS, support.impostor(q), mu),
            "a wrong mu": (GS, q, ((), (Q(1, 5),))),
            "mu as a list": (GS, q, [(), (Q(4, 5),)]),
            "an int in mu": (GS, q, ((), (1,))),
            "missing mu": (GS, q, ()),
            "an oversized mu": (GS, q, ((), (trusted_rational(2 ** 70_000, 1),))),
        })
        for label, (operation, claimed, claimed_evidence) in corruptions.items():
            with self.subTest(label):
                self.assertRejects(operation, gs_args if operation == GS else args, claimed, claimed_evidence)
        for operation in ("Definiteness", "inertia", "", None):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = ([(DEF, matrix(rows)) for rows, _ in KNOWN_DEFINITENESS]
                    + [(DEF, matrix(SPECIAL_DEFINITENESS[name][0])) for name in ("empty", "congruence_past_zero_row")]
                    + [(GS, matrix(rows)) for rows, _, _ in KNOWN_GRAM_SCHMIDT] + [(GS, mf.Matrix((), ncols=2))])
        results = [support.run(operation, M=M) for operation, M in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, M), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, rows=M.rows):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            for operation in OPERATIONS:  # The patch is live: a new run reaches it.
                failed = support.run(operation, M=matrix([[2, 1], [1, 2]]))
                self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: every kind of
        inertia, and Gram-Schmidt matrices with integer, rational and no rows."""
        rendered = (
            (DEF, [[2, 1], [1, 2]], "positive definite (2, 0, 0)", r"\text{positive definite}\ \left(2, 0, 0\right)"),
            (DEF, [[-1]], "negative definite (0, 1, 0)", r"\text{negative definite}\ \left(0, 1, 0\right)"),
            (DEF, [[1, 1], [1, 1]], "positive semidefinite (1, 0, 1)",
             r"\text{positive semidefinite}\ \left(1, 0, 1\right)"),
            (DEF, [[-1, 0], [0, 0]], "negative semidefinite (0, 1, 1)",
             r"\text{negative semidefinite}\ \left(0, 1, 1\right)"),
            (DEF, [[1, 2], [2, 1]], "indefinite (1, 1, 0)", r"\text{indefinite}\ \left(1, 1, 0\right)"),
            (DEF, [], "positive definite (0, 0, 0)", r"\text{positive definite}\ \left(0, 0, 0\right)"),
            (GS, [[3, 1], [2, 2]], "[[3, 1], [-2/5, 6/5]]",
             r"\begin{bmatrix}3 & 1 \\ -\frac{2}{5} & \frac{6}{5}\end{bmatrix}"),
            (GS, [[1, 0], [0, 1]], "[[1, 0], [0, 1]]", r"\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}"),
        )
        for operation, rows, text, latex in rendered:
            with self.subTest(operation=operation, rows=rows):
                result = support.run(operation, M=matrix(rows))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        empty = support.run(GS, M=mf.Matrix((), ncols=3))
        self.roundtrip(empty)
        self.assertEqual(mf.render_text(empty.value), "Matrix(0x3)")
        self.assertEqual(mf.to_latex(empty.value), r"\text{Matrix(0x3)}")
        data = mf.to_data(support.run(DEF, M=matrix([[2, 1], [1, 2]])).value)["data"]
        self.assertEqual(data["type"], "v6_inertia")
        self.assertEqual(data["kind"], {"type": "text", "value": "positive_definite"})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261114)
        cases = ([(DEF, matrix(rows)) for rows, _ in KNOWN_DEFINITENESS] + [(DEF, matrix([]))]
                 + [(GS, matrix(rows)) for rows, _, _ in KNOWN_GRAM_SCHMIDT] + [(GS, mf.Matrix((), ncols=2))])
        for _ in range(15):
            n = rng.randrange(1, 5)
            cases.append((DEF, matrix(random_symmetric(rng, n))))
            k = rng.randrange(1, n + 1)
            cases.append((GS, matrix([[F(rng.randrange(-4, 5), rng.choice((1, 1, 2, 3))) for _ in range(n)]
                                      for _ in range(k)])))
        for operation, M in cases:
            with self.subTest(operation=operation, rows=M.rows):
                result = support.run(operation, M=M)
                if operation == GS and result.execution_status is not ExecutionStatus.COMPLETED:
                    self.assertInvalidInput(result)  # Dependent seeded rows.
                    continue
                expected = result.value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, M=M)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, M=M), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, arguments, rows in ((DEF, {"M": [[2, 1], [1, 2]]}, [[2, 1], [1, 2]]),
                                           (DEF, {"M": [["1/2", "1/3"], ["1/3", "-1/5"]]},
                                            SPECIAL_DEFINITENESS["mixed_denominators"][0]),
                                           (DEF, {"M": []}, []),
                                           (GS, {"M": [[3, 1], [2, 2]]}, [[3, 1], [2, 2]]),
                                           (GS, {"M": [[1, 1, 0], [1, 0, 1]]}, [[1, 1, 0], [1, 0, 1]])):
            with self.subTest(operation=operation, arguments=arguments):
                result = self.answer(support.call(operation, arguments))
                self.assertEqual(result, support.run(operation, M=matrix(rows)))
                self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call(DEF, {"M": [[2, 1], [1, 2]]}))
        self.assertEqual(result.value, Inertia(2, 0, 0, "positive_definite"))
        for operation, rows in ((DEF, SPECIAL_DEFINITENESS["hilbert"][0]),
                                (GS, SPECIAL_GRAM_SCHMIDT["row_denominators"][0])):
            with self.subTest(operation=operation, call_values=True):
                result = self.answer(support.call_values(operation, M=matrix(rows)))
                self.assertEqual(result, support.run(operation, M=matrix(rows)))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call(DEF, {"M": [[1, 2], [3, 4]]}), "invalid_input")
        self.assertIn("symmetric", error["message"])
        error = self.assertToolError(support.call(GS, {"M": [[1, 2], [2, 4]]}), "invalid_input")
        self.assertIn("linearly independent rows", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references on 240 seeded inputs per operation, its evidence holds over
        Fractions, and the checker accepts every answer. definiteness: dense, low-rank and sparse symmetric matrices
        (zero pivots, swaps, the 2 x 2 congruence step, rows that leave the elimination), n from 1 to 6;
        gram_schmidt: integer and rational rows, k <= n <= 6, near-dependent rows, only inputs whose rows the
        reference finds independent."""
        rng = random.Random(20261115)
        kinds: dict[str, int] = {}
        for count in range(240):
            n = rng.randint(1, 6)
            shape = count % 3
            if shape == 0:
                rows = random_symmetric(rng, n)
            elif shape == 1:
                rows = random_symmetric(rng, n, rank=rng.randint(0, n))
            else:
                rows = sparse_symmetric(rng, n)
            with self.subTest(operation=DEF, index=count, rows=rows):
                value = self.assertDefinitenessAgrees(rows)
                kinds[value.kind] = kinds.get(value.kind, 0) + 1
        self.assertEqual(set(kinds), {"positive_definite", "negative_definite", "positive_semidefinite",
                                      "negative_semidefinite", "indefinite"}, kinds)
        self.assertTrue(all(tally >= 10 for tally in kinds.values()), kinds)
        done = refused = 0
        while done < 240:
            n = rng.randint(1, 6)
            k = rng.randint(1, n)
            if done % 3 == 2 and k >= 2:  # A row that is a combination of the others plus a small change.
                rows = [[F(rng.randint(-5, 5)) for _ in range(n)] for _ in range(k - 1)]
                combination = [sum((F(rng.randint(-2, 2)) * row[j] for row in rows), F(0)) for j in range(n)]
                combination[rng.randrange(n)] += rng.choice((0, 0, 1, F(1, 3)))
                rows.insert(rng.randrange(k), combination)
            else:
                rows = [[F(rng.randint(-6, 6), rng.choice((1, 1, 2, 5))) for _ in range(n)] for _ in range(k)]
            if reference_gram_schmidt(rows) is None:
                refused += 1
                with self.subTest(operation=GS, dependent=rows):
                    self.assertInvalidInput(support.run(GS, M=matrix(rows)))
                continue
            with self.subTest(operation=GS, index=done, rows=rows):
                self.assertGramSchmidtAgrees(rows)
            done += 1
        self.assertGreater(refused, 5)

    # -----------------------------------------------------------------------------------------------------------
    # The producer's limits and costs

    def test_largest_matrices_verify(self):
        """A 64 x 64 matrix, the largest the default limits admit, gives a verified Result for each operation in
        well under the module's time budget, and its Result reloads and verifies (catalogue section 4.1: verify on the
        largest accepted inputs)."""
        rng = random.Random(64)
        n = 64
        symmetric = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                symmetric[i][j] = symmetric[j][i] = rng.randint(-9, 9)
        tridiagonal = [[2 if i == j else (-1 if abs(i - j) == 1 else 0) for j in range(n)] for i in range(n)]
        dense = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        for operation, rows, expected in ((DEF, symmetric, None), (DEF, tridiagonal, Inertia(64, 0, 0,
                                                                                             "positive_definite")),
                                          (GS, dense, None)):
            with self.subTest(operation=operation):
                started = time.perf_counter()
                result = support.run(operation, M=matrix(rows))
                self.assertVerified(result, expected)
                self.assertLess(time.perf_counter() - started, 3 * FAST)
                if rows is not symmetric:  # Its round trip verifies twice more; the other two keep the test short.
                    self.roundtrip(result)

    def test_rational_symmetric_64_verifies(self):
        """Regression: a 64 x 64 symmetric matrix of small rationals (a/b with |a|, b <= 9) verifies. Its S = T U^T
        holds ~960-bit minors, and the checker's exact Bareiss determinant of S alone cost some 10^7 units (five times
        max_work), so run() answered resource_limit_exceeded; the checker now proves det S != 0 modulo a word-sized
        prime first (some 4,000 units)."""
        rng = random.Random(7)
        n = 64
        half = [[F(rng.randint(-9, 9), rng.randint(1, 9)) for _ in range(n)] for _ in range(n)]
        rows = [[half[i][j] + half[j][i] for j in range(n)] for i in range(n)]
        started = time.perf_counter()
        self.assertVerified(support.run(DEF, M=matrix(rows)), None)
        self.assertLess(time.perf_counter() - started, 3 * FAST)

    def test_work_is_charged(self):
        """Every step charges the budget: the work grows with the size of the matrix and with the length of its
        entries, and a budget a little below a request's own cost refuses it."""
        def work(operation: str, rows: Sequence[Sequence[object]]) -> int:
            with computation(ComputationLimits(), fresh=True) as budget:
                family.produce(operation, {"M": matrix(rows)})
                return budget.work

        small = work(DEF, [[2, 1], [1, 2]])
        larger = work(DEF, [[2 if i == j else 1 for j in range(8)] for i in range(8)])
        longer = work(DEF, [[2 ** 3000 if i == j else 1 for j in range(8)] for i in range(8)])
        self.assertLess(small, larger)
        self.assertLess(larger, longer)
        gs_small = work(GS, [[3, 1], [2, 2]])
        gs_larger = work(GS, [[(i * j + 1) % 7 + (i == j) * 9 for j in range(8)] for i in range(8)])
        self.assertLess(gs_small, gs_larger)
        for operation, rows in ((DEF, [[2 if i == j else 1 for j in range(8)] for i in range(8)]),
                                (GS, [[(i * j + 1) % 7 + (i == j) * 9 for j in range(8)] for i in range(8)])):
            with self.subTest(operation=operation):
                cost = work(operation, rows)
                with computation(ComputationLimits(max_work=cost - 1), fresh=True):
                    with self.assertRaises(ResourceLimitError):
                        family.produce(operation, {"M": matrix(rows)})


def rows_of(rows: Sequence[Sequence[object]]) -> tuple[tuple[mf.Rational, ...], ...]:
    return tuple(tuple(rational(fraction(x)) for x in row) for row in rows)
