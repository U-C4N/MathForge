"""The lattice family of mathforge.linalg: hermite_normal_form, smith_normal_form.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register linalg lattice

Every expected value comes from the card or from a reference of this module, written here and slow on purpose: the
Hermite normal form by Euclidean row reduction down each column (no transform, no extended gcd), the Smith normal
form from the determinantal divisors (the gcd of all j x j minors, each determinant by elimination over Fractions);
never from the producer. tests/v09/linalg/test_lattice_checker.py tests the checker on its own. The family has no
deferred (Unevaluated) case: every integer matrix has both normal forms, so no Result here is NO_CONCLUSION.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from math import gcd
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.linalg import lattice as family
from mathforge.contracts import ErrorInfo, ExecutionStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational

from .. import support

HNF, SNF = "hermite_normal_form", "smith_normal_form"
OPERATIONS = (HNF, SNF)
FAST = 3.0  # Seconds: a request at the edge of the default limits, or one its budget refuses, ends well within this.
WIDE = ComputationLimits(max_integer_bits=300_000, max_matrix_entries=1_000_000)  # For requests past the limits.
STAND_INS = (0.5, True, "1", None, (1, 2), [1], ((1, 0), (0, 1)), [[1, 0], [0, 1]], Rational(1), 1)

Rows = list[list[int]]

# The known values of the card: (M, ncols, value). HNF values are integer rows, SNF values tuples of ints.
KNOWN_HNF = (
    ([[2, 4], [3, 6]], [[1, 2], [0, 0]]),
    ([[4, 6], [2, 3]], [[2, 3], [0, 0]]),
    ([[2, 3], [5, 7]], [[1, 0], [0, 1]]),
    ([[6, 4], [0, 2]], [[6, 0], [0, 2]]),
)
KNOWN_SNF = (
    ([[2, 4], [6, 8]], (2, 4)),
    ([[1, 2, 3], [4, 5, 6]], (1, 3)),
    ([[0, 0], [0, 0]], (0, 0)),
    ([[2, 0], [0, 3]], (1, 6)),
)


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

def reference_hnf(rows: Sequence[Sequence[int]], ncols: int) -> Rows:
    """The row Hermite normal form by Euclidean reduction down each column: the row of least absolute entry becomes
    the pivot row, every row below is reduced by the floor quotient, until the pivot alone remains; then the pivot is
    made positive and the rows above are reduced into [0, pivot)."""
    a = [list(row) for row in rows]
    m, r = len(a), 0
    for c in range(ncols):
        if r == m:
            break
        while True:
            nonzero = [i for i in range(r, m) if a[i][c]]
            if not nonzero:
                break
            k = min(nonzero, key=lambda i: abs(a[i][c]))
            a[r], a[k] = a[k], a[r]
            for i in range(r + 1, m):
                q = a[i][c] // a[r][c]
                a[i] = [x - q * y for x, y in zip(a[i], a[r])]
            if not any(a[i][c] for i in range(r + 1, m)):
                break
        if not a[r][c]:
            continue
        if a[r][c] < 0:
            a[r] = [-x for x in a[r]]
        for i in range(r):
            q = a[i][c] // a[r][c]
            a[i] = [x - q * y for x, y in zip(a[i], a[r])]
        r += 1
    return a


def determinant(rows: Sequence[Sequence[int]]) -> int:
    """The determinant by Gaussian elimination over Fractions."""
    a = [[Fraction(x) for x in row] for row in rows]
    n, result = len(a), Fraction(1)
    for c in range(n):
        k = next((i for i in range(c, n) if a[i][c]), None)
        if k is None:
            return 0
        if k != c:
            a[c], a[k] = a[k], a[c]
            result = -result
        result *= a[c][c]
        for i in range(c + 1, n):
            f = a[i][c] / a[c][c]
            a[i] = [x - f * y for x, y in zip(a[i], a[c])]
    return int(result)


def reference_snf(rows: Sequence[Sequence[int]], ncols: int) -> tuple[int, ...]:
    """The invariant factors from the determinantal divisors D_j (gcd of all j x j minors): d_j = D_j / D_(j-1), and
    0 once D_j = 0."""
    m = len(rows)
    divisors = [1]
    for j in range(1, min(m, ncols) + 1):
        g = 0
        for chosen_rows in combinations(range(m), j):
            for chosen_columns in combinations(range(ncols), j):
                g = gcd(g, determinant([[rows[r][c] for c in chosen_columns] for r in chosen_rows]))
        divisors.append(g)
    return tuple(divisors[j] // divisors[j - 1] if divisors[j] else 0 for j in range(1, len(divisors)))


def multiply(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]], width: int) -> Rows:
    """left (r x s) times right (s x width)."""
    return [[sum(row[k] * right[k][j] for k in range(len(right))) for j in range(width)] for row in left]


def identity(n: int) -> Rows:
    return [[int(i == j) for j in range(n)] for i in range(n)]


def frozen(rows: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Integer rows in the evidence layout."""
    return tuple(tuple(row) for row in rows)


def matrix(rows: Sequence[Sequence[object]], ncols: int | None = None) -> mf.Matrix:
    """The Matrix of rows; ncols is the length of the first row by default, 0 without rows."""
    rows = [list(row) for row in rows]
    return mf.Matrix(tuple(tuple(row) for row in rows), ncols=(len(rows[0]) if rows else 0) if ncols is None else ncols)


def wide(rows: Sequence[Sequence[object]], ncols: int | None = None) -> mf.Matrix:
    """matrix(rows), built inside a budget that admits integers and sizes past the default limits."""
    with computation(WIDE, fresh=True):
        return matrix(rows, ncols)


def integers(value: mf.Matrix) -> Rows:
    """The integer entries of a Matrix of integer Rationals."""
    return [[entry.numerator for entry in row] for row in value.rows]


def forged_matrix(rows: object, ncols: object) -> mf.Matrix:
    return support.forged(mf.Matrix, rows=rows, ncols=ncols)


def forged_rational(numerator: object, denominator: object) -> Rational:
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def seeded_rows(rng: random.Random, max_rows: int = 4, max_cols: int = 4) -> tuple[Rows, int]:
    """A seeded integer matrix: empty, zero, small, large-entry, rank-deficient (one row a combination of two
    others), negated and single-row or single-column shapes all occur."""
    m, n = rng.randrange(0, max_rows + 1), rng.randrange(0, max_cols + 1)
    bound = rng.choice((1, 2, 5, 30, 10 ** 6))
    rows = [[rng.randrange(-bound, bound + 1) for _ in range(n)] for _ in range(m)]
    if m >= 3 and rng.random() < 0.3:
        x, y = rng.randrange(-3, 4), rng.randrange(-3, 4)
        rows[-1] = [x * a + y * b for a, b in zip(rows[0], rows[1])]
    if m and rng.random() < 0.1:
        rows[rng.randrange(m)] = [0] * n
    return rows, n


class Integer(int):
    """An int subclass: no exact int."""


class SubMatrix(mf.Matrix):
    """A Matrix subclass with the right rows: no transport value."""


class LatticeTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value with its four checks, which mf.verify accepts."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertEqual(result.value, value)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertLayout(self, operation: str, value: object, evidence: object, m: int, n: int) -> None:
        """The card's layout: H an m x n Matrix of integer Rationals or a tuple of min(m, n) ints, and evidence
        matrices as tuples of tuples of exact ints of the right sizes."""
        if operation == HNF:
            self.assertIs(type(value), mf.Matrix)
            self.assertEqual((len(value.rows), value.ncols), (m, n))
            for row in value.rows:
                self.assertIs(type(row), tuple)
                for entry in row:
                    self.assertIs(type(entry), Rational)
                    self.assertEqual(entry.denominator, 1)
            sizes = (m, m)
        else:
            self.assertIs(type(value), tuple)
            self.assertEqual(len(value), min(m, n))
            self.assertTrue(all(type(item) is int for item in value))
            sizes = (m, m, n, n)
        self.assertIs(type(evidence), tuple)
        self.assertEqual(len(evidence), len(sizes))
        for item, size in zip(evidence, sizes):
            self.assertIs(type(item), tuple)
            self.assertEqual(len(item), size)
            for row in item:
                self.assertIs(type(row), tuple)
                self.assertEqual(len(row), size)
                self.assertTrue(all(type(entry) is int for entry in row))

    def assertAgrees(self, operation: str, rows: Rows, ncols: int) -> tuple[object, tuple]:
        """produce() answers the reference's value in the card's layout, and the checker accepts the claim."""
        args = {"M": matrix(rows, ncols)}
        with computation(fresh=True):
            value, evidence = family.produce(operation, dict(args))
        self.assertLayout(operation, value, evidence, len(rows), ncols)
        if operation == HNF:
            self.assertEqual(integers(value), reference_hnf(rows, ncols))
        else:
            self.assertEqual(value, reference_snf(rows, ncols))
        self.assertIs(support.verdict(family, operation, args, value, evidence), True)
        return value, evidence

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts; also the empty shapes, a zero
        matrix and negative entries."""
        for rows, expected in KNOWN_HNF:
            with self.subTest(operation=HNF, rows=rows):
                self.assertEqual(reference_hnf(rows, 2), expected)  # The card, by reference.
                result = support.run(HNF, M=matrix(rows))
                self.assertVerified(result, matrix(expected))
                self.roundtrip(result)
        for rows, expected in KNOWN_SNF:
            with self.subTest(operation=SNF, rows=rows):
                self.assertEqual(reference_snf(rows, len(rows[0])), expected)
                result = support.run(SNF, M=matrix(rows))
                self.assertVerified(result, expected)
                self.roundtrip(result)
        more_hnf = (([[-3]], 1, [[3]]), ([[0, -2, 4]], 3, [[0, 2, -4]]), ([[0, 0], [0, 0], [0, 0]], 2, [[0, 0]] * 3),
                    ([[2], [3], [4]], 1, [[1], [0], [0]]), ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3,
                                                            [[1, 2, 3], [0, 3, 6], [0, 0, 0]]),
                    ([[0, 3], [5, 0]], 2, [[5, 0], [0, 3]]), ([], 0, []), ([], 3, []), ([[], []], 0, [[], []]))
        for rows, ncols, expected in more_hnf:
            with self.subTest(operation=HNF, rows=rows, ncols=ncols):
                self.assertEqual(reference_hnf(rows, ncols), expected)
                result = support.run(HNF, M=matrix(rows, ncols))
                self.assertVerified(result, matrix(expected, ncols))
                self.roundtrip(result)
        more_snf = (([[-6]], 1, (6,)), ([[0, 0], [0, 3]], 2, (3, 0)), ([[-2, 0], [0, 3]], 2, (1, 6)),
                    ([[4, 0, 0], [0, 6, 0], [0, 0, 10]], 3, (2, 2, 60)), ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3,
                                                                          (1, 3, 0)),
                    ([[2], [4], [6]], 1, (2,)), ([], 0, ()), ([], 3, ()), ([[], []], 0, ()))
        for rows, ncols, expected in more_snf:
            with self.subTest(operation=SNF, rows=rows, ncols=ncols):
                self.assertEqual(reference_snf(rows, ncols), expected)
                result = support.run(SNF, M=matrix(rows, ncols))
                self.assertVerified(result, expected)
                self.roundtrip(result)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error; so is
        every invalid input of the card, a non-integer entry, and every Matrix forged with entries that are no integer
        Rationals. produce() tests the types itself too."""
        for operation in OPERATIONS:
            for item in STAND_INS:
                with self.subTest(operation=operation, M=item):
                    self.assertInvalidInput(support.run(operation, M=item))
            for rows in ([[Rational(1, 2), 1]], [[1, 2], [3, Rational(-7, 3)]], [[Rational(5, 4)]]):
                with self.subTest(operation=operation, rows=rows):
                    result = support.run(operation, M=matrix(rows))
                    self.assertInvalidInput(result)
                    self.assertIn("integer entries", result.error.message)
            one, nought = Rational(1), Rational(0)
            forged = {"a float entry": forged_matrix(((one, 0.5), (nought, one)), 2),
                      "an int entry": forged_matrix(((one, 1), (nought, one)), 2),
                      "a bool entry": forged_matrix(((one, True), (nought, one)), 2),
                      "a str entry": forged_matrix(((one, "1"), (nought, one)), 2),
                      "None as an entry": forged_matrix(((one, None), (nought, one)), 2),
                      "a Fraction entry": forged_matrix(((one, Fraction(1, 2)), (nought, one)), 2),
                      "4/2 out of lowest terms": forged_matrix(((one, forged_rational(4, 2)), (nought, one)), 2),
                      "a zero denominator": forged_matrix(((one, forged_rational(1, 0)), (nought, one)), 2),
                      "a bool numerator": forged_matrix(((one, forged_rational(True, 1)), (nought, one)), 2),
                      "rows as a list": forged_matrix([(one, nought), (nought, one)], 2),
                      "a row as a list": forged_matrix(([one, nought], (nought, one)), 2),
                      "a ragged row": forged_matrix(((one, nought), (nought,)), 2),
                      "ncols None": forged_matrix(((one, nought), (nought, one)), None),
                      "ncols 2.0": forged_matrix(((one, nought), (nought, one)), 2.0),
                      "ncols -1": forged_matrix((), -1)}
            for label, M in forged.items():
                with self.subTest(operation=operation, forged=label):
                    self.assertInvalidInput(support.run(operation, M=M))
                    with self.assertRaises(InvalidInput):
                        family.produce(operation, {"M": M})
            M = matrix([[2, 4], [3, 6]])
            # A float numerator: run() fails while it inspects the request (Budget.inspect reads its bit length), before
            # produce(), which refuses it itself.
            with self.assertRaises(InvalidInput):
                family.produce(operation, {"M": forged_matrix(((one, forged_rational(1.0, 1)), (nought, one)), 2)})
            for item in (SubMatrix(((2, 4), (3, 6))), M.rows, [list(row) for row in M.rows], support.impostor(M),
                         support.forged(mf.Matrix, rows=M.rows), support.forged(mf.Matrix, ncols=2)) + STAND_INS:
                with self.subTest(path="produce", operation=operation, M=item), self.assertRaises(InvalidInput):
                    family.produce(operation, {"M": item})
            for arguments in ({"M": M, "k": 2}, {}, {"A": M}, tuple({"M": M}.items()), MappingProxyType({"M": M}),
                              None):
                with self.subTest(path="produce", arguments=arguments), self.assertRaises(InvalidInput):
                    family.produce(operation, arguments)
            with self.assertRaises(mf.OperationError) as raised:  # The value function raises the structured error.
                support.value(operation, M=matrix([[Rational(1, 2)]]))
            self.assertEqual(raised.exception.code, "invalid_input")
        for operation in ("Hermite_normal_form", "smith_normal_forms", "hnf", "", None, (HNF,), "eigenvectors"):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, {"M": matrix([[1]])})

    def test_resource_limits(self):
        """An integer of 2**65536, a 65 x 65 matrix and max_work=1 exceed the budget; so do evidence matrices past
        max_matrix_entries (an M of 65 rows, or of 65 columns for smith_normal_form, which the checker cannot verify)
        and a value or evidence integer past max_integer_bits. Each request fails at once."""
        start = time.perf_counter()
        oversized = support.oversized_matrix()
        tall = wide([[i % 3 - 1] for i in range(65)], 1)
        flat = wide([[i % 3 - 1 for i in range(65)]], 65)
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                self.assertResourceLimit(support.run(operation, M=wide([[support.HUGE]])))
                self.assertResourceLimit(support.run(operation, M=wide([[1, 0], [0, -support.HUGE]])))
                self.assertResourceLimit(support.run(operation, M=oversized))
                with self.assertRaises(ResourceLimitError):
                    family.produce(operation, {"M": oversized})
                for M in (matrix([[2, 4], [3, 6]]), matrix([]), matrix([[7]])):
                    self.assertResourceLimit(support.run(operation, M=M, limits=ComputationLimits(max_work=1)))
                self.assertResourceLimit(support.run(operation, M=tall))  # U is 65 x 65.
                # Under limits that admit 65 x 65 evidence the same request verifies.
                admitting = ComputationLimits(max_matrix_entries=65 * 65)
                result = support.run(operation, M=tall, limits=admitting)
                self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
                self.assertTrue(mf.verify(result, limits=admitting).verified)
                self.assertFalse(mf.verify(result).verified)  # The default limits refuse its 65 x 65 evidence.
        # A 1 x 65 M: U is 1 x 1 for hermite_normal_form, but V is 65 x 65 for smith_normal_form.
        self.assertVerified(support.run(HNF, M=flat), wide([[1 - i % 3 for i in range(65)]], 65))  # Negated.
        self.assertResourceLimit(support.run(SNF, M=flat))
        self.assertEqual(support.run(SNF, M=flat, limits=ComputationLimits(max_matrix_entries=65 * 65)).value, (1,))
        # A true value longer than max_integer_bits: the lcm of two coprime 40-bit entries under 64-bit limits.
        p, q = 2 ** 40 - 87, 2 ** 40 - 167
        self.assertEqual(gcd(p, q), 1)
        small = ComputationLimits(max_integer_bits=64)
        self.assertResourceLimit(support.run(SNF, M=matrix([[p, 0], [0, q]]), limits=small))
        with computation(small, fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(SNF, {"M": matrix([[p, 0], [0, q]])})
        self.assertVerified(support.run(HNF, M=matrix([[p, 0], [0, q]]), limits=small), matrix([[p, 0], [0, q]]))
        # The default limits: 2**65535 + 1 is prime to 5, and its product with 5 has 65,538 bits.
        self.assertResourceLimit(support.run(SNF, M=matrix([[2 ** 65535 + 1, 0], [0, 5]])))
        self.assertLess(time.perf_counter() - start, 2 * FAST)

    def test_small_limits_refuse_or_verify(self):
        """Under small integer limits every seeded request either verifies under those limits or is refused with
        ResourceLimitError by produce(): never a claim that the checker rejects."""
        rng = random.Random(20261113)
        limits = ComputationLimits(max_integer_bits=24)
        tally = {"verified": 0, "refused": 0}
        for _ in range(150):
            rows, ncols = seeded_rows(rng)
            rows = [[entry % (1 << 20) - (1 << 19) for entry in row] for row in rows]
            for operation in OPERATIONS:
                args = {"M": matrix(rows, ncols)}
                with self.subTest(operation=operation, rows=rows):
                    try:
                        with computation(limits, fresh=True):
                            value, evidence = family.produce(operation, dict(args))
                    except ResourceLimitError:
                        tally["refused"] += 1
                        continue
                    self.assertIs(support.verdict(family, operation, args, value, evidence, limits=limits), True)
                    tally["verified"] += 1
        self.assertGreater(tally["verified"], 100, tally)
        self.assertGreater(tally["refused"], 5, tally)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts true claims with evidence built by hand, and with the producer's evidence changed by
        unimodular matrices that fix the value: a sign change of the zero rows of H (the left kernel) for
        hermite_normal_form, S U and V S' with diagonal signs S, S' that agree on the diagonal for
        smith_normal_form."""
        # [[2, 4], [3, 6]]: U = [[-1, 1], [-3, 2]], V = U^-1; and with the kernel row negated.
        args = {"M": matrix([[2, 4], [3, 6]])}
        h = matrix([[1, 2], [0, 0]])
        self.assertAccepts(HNF, args, h, (((-1, 1), (-3, 2)), ((2, -1), (3, -1))))
        self.assertAccepts(HNF, args, h, (((-1, 1), (3, -2)), ((2, 1), (3, 1))))
        # diag(2, 3) -> (1, 6): L = (s, t; -b/g, a/g), R = (1, -t b/g; 1, s a/g) with s = -1, t = 1.
        self.assertAccepts(SNF, {"M": matrix([[2, 0], [0, 3]])}, (1, 6),
                           (((-1, 1), (-3, 2)), ((2, -1), (3, -1)), ((1, -3), (1, -2)), ((-2, 3), (-1, 1))))
        rng = random.Random(20261114)
        for _ in range(60):
            rows, ncols = seeded_rows(rng)
            m = len(rows)
            args = {"M": matrix(rows, ncols)}
            with self.subTest(rows=rows, ncols=ncols):
                h, (u, v) = family.produce(HNF, dict(args))
                rank = sum(1 for row in h.rows if any(entry.numerator for entry in row))
                signs = [1 if i < rank else rng.choice((1, -1)) for i in range(m)]
                u2 = frozen([[signs[i] * x for x in u[i]] for i in range(m)])
                v2 = frozen([[x * signs[j] for j, x in enumerate(v[i])] for i in range(m)])
                self.assertAccepts(HNF, args, h, (u2, v2))
                d, (u, u_inv, w, w_inv) = family.produce(SNF, dict(args))
                left = [rng.choice((1, -1)) for _ in range(m)]
                right = [left[i] if i < min(m, ncols) else rng.choice((1, -1)) for i in range(ncols)]
                self.assertAccepts(SNF, args, d, (
                    frozen([[left[i] * x for x in u[i]] for i in range(m)]),
                    frozen([[x * left[j] for j, x in enumerate(u_inv[i])] for i in range(m)]),
                    frozen([[x * right[j] for j, x in enumerate(w[i])] for i in range(ncols)]),
                    frozen([[right[i] * x for x in w_inv[i]] for i in range(ncols)])))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical with consistent evidence, an
        impostor class, a record forged with object.__new__, a bool, a float or a str where an int belongs, a list for
        a tuple, missing or extra evidence, a wrong operation name and oversized forged evidence."""
        args = {"M": matrix([[6, 4], [0, 2]])}
        h, (u, v) = family.produce(HNF, dict(args))
        self.assertEqual(h, matrix([[6, 0], [0, 2]]))
        one, zero = Rational(1), Rational(0)
        i2 = ((1, 0), (0, 1))
        hnf_cases = (
            ("wrong value", matrix([[6, 0], [0, 1]]), (u, v)),
            ("not reduced above the pivot, true lattice", matrix([[6, 4], [0, 2]]), (i2, i2)),
            ("negative pivot", matrix([[-6, 0], [0, 2]]), (((-1, 2), (0, 1)), ((-1, 2), (0, 1)))),
            ("impostor value", support.impostor(h), (u, v)),
            ("rows as a list", forged_matrix(list(h.rows), 2), (u, v)),
            ("int entries", forged_matrix(((6, 0), (0, 2)), 2), (u, v)),
            ("missing evidence", h, (u,)),
            ("extra evidence", h, (u, v, ())),
            ("bool in evidence", h, (tuple(tuple(bool(x) if x in (0, 1) else x for x in row) for row in u), v)),
            ("float in evidence", h, (tuple(tuple(float(x) for x in row) for row in u), v)),
            ("str in evidence", h, (tuple(tuple(str(x) for x in row) for row in u), v)),
            ("list for a tuple", h, ([list(row) for row in u], v)),
            ("Matrix evidence", h, (matrix(u), matrix(v))),
            ("non-unimodular U with H = U M", matrix([[12, 0], [0, 2]]), (((2, -4), (0, 1)), ((1, 4), (0, 2)))),
        )
        for label, value, evidence in hnf_cases:
            with self.subTest(operation=HNF, corruption=label):
                self.assertRejects(HNF, args, value, evidence)
        diag = {"M": matrix([[2, 0], [0, 3]])}
        d, evidence = family.produce(SNF, dict(diag))
        self.assertEqual(d, (1, 6))
        swap = ((0, 1), (1, 0))
        u, u_inv, w, w_inv = evidence
        swapped = (frozen(multiply(swap, u, 2)), frozen(multiply(u_inv, swap, 2)), frozen(multiply(w, swap, 2)),
                   frozen(multiply(swap, w_inv, 2)))
        self.assertAccepts(SNF, diag, d, evidence)
        snf_cases = (
            ("broken chain with consistent evidence", (6, 1), swapped),
            ("the diagonal itself", (2, 3), (i2, i2, i2, i2)),
            ("a wrong factor", (1, 5), evidence),
            ("negative factors", (-1, -6), evidence),
            ("bool factors", (True, 6), evidence),
            ("float factor", (1.0, 6), evidence),
            ("str factor", ("1", 6), evidence),
            ("list value", [1, 6], evidence),
            ("Rational factor", (Rational(1), 6), evidence),
            ("int subclass", (Integer(1), 6), evidence),
            ("too few factors", (1,), evidence),
            ("missing evidence", d, evidence[:3]),
            ("extra evidence", d, evidence + ((),)),
            ("list evidence", d, list(evidence)),
        )
        for label, value, corrupted in snf_cases:
            with self.subTest(operation=SNF, corruption=label):
                self.assertRejects(SNF, diag, value, corrupted)
        self.assertRejects(SNF, args, h, (u, v))  # A Hermite claim under the other name.
        self.assertRejects("hermite", args, h, (u, v))
        self.assertRejects(HNF, {"M": forged_matrix(((one, zero), (zero, forged_rational(1, 2))), 2)}, h, (u, v))
        # Oversized forged evidence fails fast: integers past max_integer_bits answer False, and long products run
        # out of budget.
        start = time.perf_counter()
        huge = support.HUGE
        self.assertRejects(HNF, args, h, (((huge, 0), (0, 1)), v), may_exhaust=True)
        self.assertRejects(SNF, diag, (1, huge), evidence, may_exhaust=True)
        big = 64
        long_entry = 2 ** 60_000 + 1
        filled = tuple(tuple(long_entry for _ in range(big)) for _ in range(big))
        rows = [[(i * j) % 5 for j in range(big)] for i in range(big)]
        self.assertRejects(HNF, {"M": matrix(rows)}, matrix(rows), (filled, filled), may_exhaust=True)
        self.assertLess(time.perf_counter() - start, FAST)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [(HNF, matrix(rows)) for rows, _ in KNOWN_HNF] + [(SNF, matrix(rows)) for rows, _ in KNOWN_SNF]
        requests += [(HNF, matrix([], 3)), (SNF, matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]]))]
        results = [support.run(operation, M=M) for operation, M in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, M), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, shape=M.shape):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(HNF, M=matrix([[1, 2], [3, 4]]))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a Matrix of integers,
        an empty-column Matrix, a tuple of ints and the empty tuple."""
        rendered = (
            (HNF, [[2, 4], [3, 6]], 2, "[[1, 2], [0, 0]]", r"\begin{bmatrix}1 & 2 \\ 0 & 0\end{bmatrix}"),
            (HNF, [[-5]], 1, "[[5]]", r"\begin{bmatrix}5\end{bmatrix}"),
            (HNF, [[], [], []], 0, "Matrix(3x0)", r"\text{Matrix(3x0)}"),
            (HNF, [], 0, "Matrix(0x0)", r"\text{Matrix(0x0)}"),
            (SNF, [[2, 4], [6, 8]], 2, "(2, 4)", r"\left(2, 4\right)"),
            (SNF, [[0, 0], [0, 0]], 2, "(0, 0)", r"\left(0, 0\right)"),
            (SNF, [[12]], 1, "(12)", r"\left(12\right)"),
            (SNF, [], 3, "()", r"\left(\right)"),
        )
        for operation, rows, ncols, text, latex in rendered:
            with self.subTest(operation=operation, rows=rows):
                result = support.run(operation, M=matrix(rows, ncols))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        value = support.run(SNF, M=matrix([[2, 4], [6, 8]])).value
        self.assertEqual(mf.to_data(value)["data"], {"type": "tuple", "items": [{"type": "integer", "value": "2"},
                                                                              {"type": "integer", "value": "4"}]})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261115)
        cases = [(operation, matrix(rows)) for operation in OPERATIONS for rows, _ in KNOWN_HNF + KNOWN_SNF]
        for _ in range(20):
            rows, ncols = seeded_rows(rng)
            cases.append((rng.choice(OPERATIONS), matrix(rows, ncols)))
        for operation, M in cases:
            with self.subTest(operation=operation, rows=M.rows):
                expected = support.run(operation, M=M).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, M=M)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, M=M), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, arguments, M, value in (
                (HNF, {"M": [[2, 4], [3, 6]]}, matrix([[2, 4], [3, 6]]), matrix([[1, 2], [0, 0]])),
                (SNF, {"M": [[1, 2, 3], [4, 5, 6]]}, matrix([[1, 2, 3], [4, 5, 6]]), (1, 3)),
                (SNF, {"M": []}, mf.Matrix(()), ())):
            with self.subTest(operation=operation, arguments=arguments):
                result = self.answer(support.call(operation, arguments))
                self.assertEqual(result, support.run(operation, M=M))
                self.assertTrue(mf.verify(result).verified)
                self.assertEqual(result.value, value)
        for operation in OPERATIONS:
            M = matrix([[6, 4], [0, 2]])
            with self.subTest(operation=operation, path="call_values"):
                result = self.answer(support.call_values(operation, M=M))
                self.assertEqual(result, support.run(operation, M=M))
                self.assertTrue(mf.verify(result).verified)
            with self.subTest(operation=operation, path="error"):
                error = self.assertToolError(support.call(operation, {"M": [["1/2", 1]]}), "invalid_input")
                self.assertIn("integer entries", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the slow references on at least 500 seeded inputs per operation (shapes 0 x 0 to 4 x 4,
        small and large entries, rank-deficient and zero rows), and the checker accepts every answer."""
        rng = random.Random(20261116)
        counts = dict.fromkeys(OPERATIONS, 0)
        while min(counts.values()) < 500:
            rows, ncols = seeded_rows(rng)
            for operation in OPERATIONS:
                with self.subTest(operation=operation, rows=rows, ncols=ncols):
                    self.assertAgrees(operation, rows, ncols)
                    counts[operation] += 1
        # Larger shapes against the Hermite reference, and structured Smith forms with known factors.
        for _ in range(20):
            m, n = rng.randrange(5, 9), rng.randrange(5, 9)
            rows = [[rng.randrange(-9, 10) for _ in range(n)] for _ in range(m)]
            with self.subTest(operation=HNF, rows=rows):
                self.assertAgrees(HNF, rows, n)
        for factors in ((1, 2, 6, 0), (3, 3, 3), (1, 1, 10 ** 30), (5, 0, 0)):
            k = len(factors)
            diagonal = [[factors[i] if i == j else 0 for j in range(k)] for i in range(k)]
            rows = multiply(multiply([[1, 2, 0, 1][:k]] + identity(k)[1:], diagonal, k),
                            [[1] * k] + identity(k)[1:], k)  # Unimodular on both sides: the same factors.
            with self.subTest(operation=SNF, factors=factors):
                value, _ = self.assertAgrees(SNF, rows, k)
                self.assertEqual(value, factors)

    # -----------------------------------------------------------------------------------------------------------
    # The producer's work and the edge of the default limits

    def test_work_is_charged_before_it_runs(self):
        """produce() charges at least two units and the same amount for the same request: a budget of exactly that
        amount computes the value, one unit less refuses the request."""
        requests = [matrix([[2, 4], [3, 6]]), matrix([]), matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]]),
                    matrix([[2 ** 70 + 1, 3], [5, 2 ** 65 - 1]]), matrix([[0, 0], [0, 0]], 2)]
        for operation in OPERATIONS:
            for M in requests:
                with self.subTest(operation=operation, rows=M.rows):
                    with computation(fresh=True) as budget:
                        value, _ = family.produce(operation, {"M": M})
                    work = budget.work
                    self.assertGreaterEqual(work, 2)
                    with computation(ComputationLimits(max_work=work), fresh=True):
                        self.assertEqual(family.produce(operation, {"M": M})[0], value)
                    with computation(ComputationLimits(max_work=work - 1), fresh=True), \
                            self.assertRaises(ResourceLimitError):
                        family.produce(operation, {"M": M})

    def test_full_size_requests_fit_the_default_budget(self):
        """64 x 64 matrices of small entries (dense, rank-deficient), 64 x 1 and 1 x 64 verify under the default
        budget, each within FAST seconds; entries of 40 bits on a 64 x 64 matrix exceed it and are refused at once."""
        rng = random.Random(20261117)
        dense = [[rng.randrange(-3, 4) for _ in range(64)] for _ in range(64)]
        left = [[rng.randrange(-2, 3) for _ in range(8)] for _ in range(64)]
        right = [[rng.randrange(-2, 3) for _ in range(64)] for _ in range(8)]
        deficient = multiply(left, right, 64)  # Rank at most 8.
        column = [[rng.randrange(-10 ** 9, 10 ** 9)] for _ in range(64)]
        requests = [(dense, 64), (deficient, 64), (column, 1), ([[row[0] for row in column]], 64)]
        for rows, ncols in requests:
            for operation in OPERATIONS:
                with self.subTest(operation=operation, shape=(len(rows), ncols)):
                    start = time.perf_counter()
                    result = support.run(operation, M=matrix(rows, ncols))
                    self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
                    self.assertTrue(mf.verify(result).verified)
                    self.assertLess(time.perf_counter() - start, FAST)
        self.assertEqual(integers(support.run(HNF, M=matrix(column, 1)).value)[0],
                         [gcd(*(row[0] for row in column))])
        self.assertEqual(support.run(SNF, M=matrix(column, 1)).value, (gcd(*(row[0] for row in column)),))
        large = [[rng.randrange(-2 ** 40, 2 ** 40) for _ in range(64)] for _ in range(64)]
        for operation in OPERATIONS:
            with self.subTest(operation=operation, entries="40 bits"):
                start = time.perf_counter()
                self.assertResourceLimit(support.run(operation, M=matrix(large)))
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Regressions of the review

    def test_comparator_network_sorts(self):
        """The comparators of the divisibility pass (Batcher's odd-even merge sort) sort every 0-1 input of up to 12
        items and seeded inputs of up to 64: by the 0-1 principle, on a distributive lattice such as divisibility
        (gcd and lcm for min and max) they make a chain."""
        rng = random.Random(20261113)
        for count in range(0, 65):
            pairs = family._batcher_pairs(count)
            self.assertTrue(all(0 <= i < j < count for i, j in pairs), count)
            inputs = ([[(bits >> k) & 1 for k in range(count)] for bits in range(1 << count)] if count <= 12 else
                      [[rng.randrange(0, 40) for _ in range(count)] for _ in range(40)])
            for items in inputs:
                for i, j in pairs:
                    if items[i] > items[j]:
                        items[i], items[j] = items[j], items[i]
                self.assertEqual(items, sorted(items), count)

    def test_diagonal_smith_forms_at_full_size(self):
        """64 x 64 diagonals verify under the default limits. Before the review, the reversed chain
        diag(2^64, ..., 2^1) cost 2.1M units of producer work (a gcd step at each of its 2016 pairs) and was refused,
        and the diagonals of the first 64 primes and of the cubes 2^3, ..., 65^3 came with evidence entries of 11,568
        and 2,779 bits, whose checks (10.3M and 5.5M units) the verification budget refused."""
        primes = [p for p in range(2, 320) if all(p % q for q in range(2, p))][:64]
        cases = ([2 ** (64 - i) for i in range(64)], primes, [(i + 2) ** 3 for i in range(64)])
        for entries in cases:
            with self.subTest(entries=entries[:3]):
                start = time.perf_counter()
                rows = [[entries[i] if i == j else 0 for j in range(64)] for i in range(64)]
                result = support.run(SNF, M=matrix(rows))
                self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
                self.assertEqual(result.value, diagonal_invariants(entries))
                self.assertTrue(mf.verify(result).verified)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_permuted_diagonals_agree(self):
        """Seeded diagonals in any order, zeros among them, agree with the prime-by-prime reference."""
        rng = random.Random(20261114)
        for _ in range(60):
            k = rng.randrange(1, 13)
            entries = [rng.choice((0, 1, 2, 3, 4, 6, 8, 9, 12, 25, 27, 30, 49, 64, 210)) for _ in range(k)]
            rows = [[entries[i] if i == j else 0 for j in range(k)] for i in range(k)]
            with self.subTest(entries=entries):
                if k <= 5:  # The determinantal reference as well.
                    value, _ = self.assertAgrees(SNF, rows, k)
                else:
                    with computation(fresh=True):
                        value, evidence = family.produce(SNF, {"M": matrix(rows)})
                    self.assertIs(support.verdict(family, SNF, {"M": matrix(rows)}, value, evidence), True)
                self.assertEqual(value, diagonal_invariants(entries))


def diagonal_invariants(entries: Sequence[int]) -> tuple[int, ...]:
    """The invariant factors of the square diagonal matrix of these entries, prime by prime: each prime's exponents
    in the nonzero entries, sorted, go to the last places (trial division; for small entries only)."""
    nonzero = [entry for entry in entries if entry]
    factors = [1] * len(nonzero)
    exponents: dict[int, list[int]] = {}
    for entry in nonzero:
        rest, p = abs(entry), 2
        while rest > 1:
            count = 0
            while rest % p == 0:
                rest, count = rest // p, count + 1
            if count:
                exponents.setdefault(p, []).append(count)
            p += 1
    for p, counts in exponents.items():
        for place, count in zip(range(len(nonzero) - 1, -1, -1), sorted(counts, reverse=True)):
            factors[place] *= p ** count
    return tuple(factors) + (0,) * (len(entries) - len(nonzero))
