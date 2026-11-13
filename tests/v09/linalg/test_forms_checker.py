"""Tests of the independent checker of mathforge.areas.linalg.forms (mathforge.certify.linalg.forms) alone, written
before the family's producer exists: every claim is built here, and the checker answers it as the verifier asks it
(support.verdict, in a fresh budget of the default limits unless a test passes others).

The true claims come from short references of this module over fractions.Fraction, never from a producer or a kernel:
- definiteness: a congruence diagonalization by elementary column operations (S starts as I; a zero pivot is replaced
  by a later nonzero diagonal entry, or else by e_k + e_j for an off-diagonal A[k][j] != 0; then every later column is
  cleared against the pivot), with S^T M S recomputed and asserted diagonal. The counts are checked against a second
  reference, Descartes' rule of signs on the characteristic polynomial from the Faddeev-LeVerrier recurrence, exact
  for a symmetric matrix since all its roots are real. Other valid evidence (eigenbases, scaled and permuted
  columns) shows that the checker accepts any (S, D) and not the producer's.
- gram_schmidt: the textbook recurrence q_i = m_i - sum_{j<i} mu_ij q_j over Fractions.

The evidence follows the layout of the family's docstring: (S, D) with S a tuple of row tuples of Rationals and D a
tuple of Rationals; for gram_schmidt mu itself, ((), (mu_10,), (mu_20, mu_21), ...).

The checker answers False to every corrupted claim, never with an exception, and refuses an oversized forgery within
FAST seconds under the default budget. It raises ResourceLimitError only for the request caps of its docstring: an M
with more entries than max_matrix_entries (the card's matrix rule) and an integer of M past max_integer_bits.

Run this module by name with

    python -m unittest tests.v09.linalg.test_forms_checker
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import random
import time

import mathforge as mf
from mathforge.areas.linalg import forms as family
from mathforge.areas.linalg.forms import Inertia
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import trusted_rational

from .. import support

Q = mf.Rational
F = Fraction
DEF = "definiteness"
GS = "gram_schmidt"
FAST = 3.0  # Seconds: the bound for every oversized forgery, far above what each one takes.
GIANT = 2 ** 70_000  # Longer than max_integer_bits.
BIG = 2 ** 65_000 + 1  # Within max_integer_bits, far beyond anything a small matrix needs.
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


def to_matrix(rows: Sequence[Sequence[object]], ncols: int | None = None) -> mf.Matrix:
    if not rows:
        return mf.Matrix((), ncols=0 if ncols is None else ncols)
    return mf.Matrix(tuple(tuple(rational(fraction(x)) for x in row) for row in rows))


def rationals(rows: Sequence[Sequence[object]]) -> tuple[tuple[mf.Rational, ...], ...]:
    return tuple(tuple(rational(fraction(x)) for x in row) for row in rows)


def vector(values: Sequence[object]) -> tuple[mf.Rational, ...]:
    return tuple(rational(fraction(x)) for x in values)


def identity(n: int) -> Rows:
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def transpose(a: Rows) -> Rows:
    return [list(column) for column in zip(*a)]


def multiply(a: Rows, b: Rows) -> Rows:
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0)) for j in range(len(b[0]))] for i in range(len(a))]


def congruent(m: Rows, s: Rows) -> Rows:
    """S^T M S."""
    if not m:
        return []
    return multiply(multiply(transpose(s), m), s)


def diagonalize(m: Rows) -> tuple[Rows, list[Fraction]]:
    """(S, D) with S invertible and S^T M S = diag(D), by elementary column operations on S."""
    n = len(m)
    s = identity(n)

    def add_column(target: int, source: int, factor: Fraction) -> None:
        for row in s:
            row[target] += factor * row[source]

    for k in range(n):
        a = congruent(m, s)
        if not a[k][k]:
            j = next((j for j in range(k + 1, n) if a[j][j]), None)
            if j is not None:
                for row in s:
                    row[k], row[j] = row[j], row[k]
            else:
                j = next((j for j in range(k + 1, n) if a[k][j]), None)
                if j is None:
                    continue
                add_column(k, j, F(1))  # e_k + e_j: A[k][k] becomes 2 A[k][j] != 0.
            a = congruent(m, s)
        for i in range(k + 1, n):
            if a[k][i]:
                add_column(i, k, -a[k][i] / a[k][k])
    a = congruent(m, s)
    assert all(not a[i][j] for i in range(n) for j in range(n) if i != j), a
    return s, [a[i][i] for i in range(n)]


def charpoly(m: Rows) -> list[Fraction]:
    """det(t I - M), lowest degree first, by the Faddeev-LeVerrier recurrence."""
    n = len(m)
    coefficients = [F(0)] * n + [F(1)]
    current = [[F(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        current = multiply(m, current) if n else []
        for i in range(n):
            current[i][i] += coefficients[n - k + 1]
        product = multiply(m, current)
        coefficients[n - k] = -sum(product[i][i] for i in range(n)) / k
    return coefficients


def sign_changes(values: Sequence[Fraction]) -> int:
    signs = [value > 0 for value in values if value]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def descartes_inertia(m: Rows) -> tuple[int, int, int]:
    """(positive, negative, zero) eigenvalue counts of a symmetric M by Descartes' rule, exact as every root is
    real."""
    p = charpoly(m)
    zero = next(index for index, value in enumerate(p) if value)
    rest = p[zero:]
    mirrored = [value if index % 2 == 0 else -value for index, value in enumerate(rest)]
    return sign_changes(rest), sign_changes(mirrored), zero


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


def inertia_of(diagonal: Sequence[Fraction]) -> Inertia:
    counts = (sum(1 for x in diagonal if x > 0), sum(1 for x in diagonal if x < 0), sum(1 for x in diagonal if not x))
    return Inertia(*counts, kind_of(*counts))


def definiteness_claim(rows: Sequence[Sequence[object]]) -> tuple[dict, Inertia, tuple]:
    """args, value and evidence of a true definiteness claim built by the references."""
    m = fractions(rows)
    s, d = diagonalize(m)
    value = inertia_of(d)
    if m:
        assert (value.positive, value.negative, value.zero) == descartes_inertia(m)
    return {"M": to_matrix(m)}, value, (rationals(s), vector(d))


def gram_schmidt(rows: Rows) -> tuple[Rows, list[list[Fraction]]]:
    """(Q, mu) of the textbook recurrence; AssertionError for dependent rows."""
    q: Rows = []
    mu: list[list[Fraction]] = []
    for row in rows:
        coefficients = [sum((a * b for a, b in zip(row, other)), F(0)) / sum((b * b for b in other), F(0))
                        for other in q]
        current = list(row)
        for coefficient, other in zip(coefficients, q):
            current = [a - coefficient * b for a, b in zip(current, other)]
        assert any(current)
        q.append(current)
        mu.append(coefficients)
    return q, mu


def gs_claim(rows: Sequence[Sequence[object]], ncols: int | None = None) -> tuple[dict, mf.Matrix, tuple]:
    m = fractions(rows)
    q, mu = gram_schmidt(m)
    width = len(m[0]) if m else ncols
    return {"M": to_matrix(m, width)}, to_matrix(q, width), tuple(vector(row) for row in mu)


def replaced(items: tuple, index: int, item: object) -> tuple:
    return items[:index] + (item,) + items[index + 1:]


def random_symmetric(rng: random.Random, n: int, rank: int | None = None) -> Rows:
    """A random symmetric rational matrix; with rank, B^T diag(+-1) B for a rank x n B, so of rank at most rank."""
    if rank is None:
        m = [[F(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                m[i][j] = m[j][i] = F(rng.randint(-9, 9), rng.choice((1, 1, 2, 3)))
        return m
    if not rank:
        return [[F(0)] * n for _ in range(n)]
    b = [[F(rng.randint(-4, 4), rng.choice((1, 2))) for _ in range(n)] for _ in range(rank)]
    d = [[F(rng.choice((-1, 1)) if i == j else 0) for j in range(rank)] for i in range(rank)]
    return multiply(multiply(transpose(b), d), b)


class FormsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsFast(self, operation: object, args: object, value: object, evidence: object, *,
                          may_exhaust: bool = False) -> None:
        started = time.perf_counter()
        self.assertRejects(operation, args, value, evidence, may_exhaust=may_exhaust)
        self.assertLess(time.perf_counter() - started, FAST)

    # definiteness: true claims.

    def test_definiteness_known_values_of_the_card(self):
        known = (([[2, 1], [1, 2]], (2, 0, 0, "positive_definite")),
                 ([[1, 2], [2, 1]], (1, 1, 0, "indefinite")),
                 ([[0, 0], [0, 0]], (0, 0, 2, "positive_semidefinite")),
                 ([[0, 1], [1, 0]], (1, 1, 0, "indefinite")),
                 ([[-1]], (0, 1, 0, "negative_definite")),
                 ([[1, 1], [1, 1]], (1, 0, 1, "positive_semidefinite")))
        for rows, fields in known:
            with self.subTest(rows=rows):
                args, value, evidence = definiteness_claim(rows)
                self.assertEqual(value, Inertia(*fields))
                self.assertAccepts(DEF, args, value, evidence)

    def test_definiteness_special_cases(self):
        cases = ([], [[0]], [[5]], [[Q(-1, 3)]], [[-2, 0], [0, -3]], [[-1, 0], [0, 0]], [[0, 0], [0, -1]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                 [[0, 0, 1], [0, 0, 0], [1, 0, 0]],  # A zero pivot, the 2x2 congruence step past a zero row.
                 [[1, 2, 3], [2, 4, 6], [3, 6, 9]],  # Rank 1.
                 [[Q(1, 2), Q(1, 3)], [Q(1, 3), Q(1, 4)]])  # Hilbert-like, positive definite.
        expected = {0: (0, 0, 0, "positive_definite"), 1: (0, 0, 1, "positive_semidefinite"),
                    5: (0, 1, 1, "negative_semidefinite"), 6: (0, 1, 1, "negative_semidefinite"),
                    10: (1, 0, 2, "positive_semidefinite")}
        for index, rows in enumerate(cases):
            with self.subTest(rows=rows):
                args, value, evidence = definiteness_claim(rows)
                if index in expected:
                    self.assertEqual(value, Inertia(*expected[index]))
                self.assertAccepts(DEF, args, value, evidence)
        # The 0 x 0 matrix: the card's order gives positive_definite, with the evidence ((), ()).
        self.assertAccepts(DEF, {"M": mf.Matrix((), ncols=0)}, Inertia(0, 0, 0, "positive_definite"), ((), ()))

    def test_definiteness_seeded_matrices(self):
        rng = random.Random(20261113)
        for trial in range(40):
            n = rng.randint(1, 6)
            rank = None if trial % 3 else rng.randint(0, n)
            rows = random_symmetric(rng, n, rank)
            with self.subTest(trial=trial, rows=rows):
                args, value, evidence = definiteness_claim(rows)
                self.assertAccepts(DEF, args, value, evidence)

    def test_definiteness_larger_matrix_quickly(self):
        rng = random.Random(7)
        rows = random_symmetric(rng, 16)
        args, value, evidence = definiteness_claim(rows)
        started = time.perf_counter()
        self.assertAccepts(DEF, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    def test_definiteness_other_valid_evidence(self):
        m = {"M": to_matrix([[2, 1], [1, 2]])}
        pd = Inertia(2, 0, 0, "positive_definite")
        # Eigenbasis columns (1, 1) and (1, -1): S^T M S = diag(6, 2).
        self.assertAccepts(DEF, m, pd, (rationals([[1, 1], [1, -1]]), vector([6, 2])))
        # Permuted columns permute D; scaled columns scale D by squares.
        self.assertAccepts(DEF, m, pd, (rationals([[1, 1], [-1, 1]]), vector([2, 6])))
        self.assertAccepts(DEF, m, pd, (rationals([[Q(1, 2), -3], [Q(1, 2), 3]]), vector([Q(3, 2), 18])))
        # Orthonormal-free basis of the indefinite [[0, 1], [1, 0]]: columns (1, 1) and (1, -1) give diag(2, -2).
        self.assertAccepts(DEF, {"M": to_matrix([[0, 1], [1, 0]])}, Inertia(1, 1, 0, "indefinite"),
                           (rationals([[1, 1], [1, -1]]), vector([2, -2])))
        # Evidence Rationals need not be in lowest terms (forged 2/2 for 1).
        two_halves = trusted_rational(2, 2)
        self.assertAccepts(DEF, m, pd, (((two_halves, Q(-1, 2)), (Q(0), Q(1))), (Q(2), Q(3, 2))))
        # Random congruences of a diagonal matrix.
        rng = random.Random(3)
        for trial in range(10):
            n = rng.randint(1, 5)
            d = [F(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(n)]
            while True:
                t = [[F(rng.randint(-3, 3)) for _ in range(n)] for _ in range(n)]
                if charpoly(t)[0]:  # det(t) != 0, so S = t^-1 below exists.
                    break
            # M = t^T diag(d) t, and S = t^-1 gives S^T M S = diag(d).
            diag = [[d[i] if i == j else F(0) for j in range(n)] for i in range(n)]
            m_rows = multiply(multiply(transpose(t), diag), t)
            s = inverse(t)
            with self.subTest(trial=trial):
                self.assertAccepts(DEF, {"M": to_matrix(m_rows)}, inertia_of(d), (rationals(s), vector(d)))

    # gram_schmidt: true claims.

    def test_gram_schmidt_known_values_of_the_card(self):
        known = (([[3, 1], [2, 2]], [[3, 1], [Q(-2, 5), Q(6, 5)]]),
                 ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
                 ([[1, 1, 0], [1, 0, 1]], [[1, 1, 0], [Q(1, 2), Q(-1, 2), 1]]))
        for rows, expected in known:
            with self.subTest(rows=rows):
                args, value, evidence = gs_claim(rows)
                self.assertEqual(value, to_matrix(expected))
                self.assertAccepts(GS, args, value, evidence)
        self.assertAccepts(GS, {"M": to_matrix([[3, 1], [2, 2]])}, to_matrix([[3, 1], [Q(-2, 5), Q(6, 5)]]),
                           ((), (Q(4, 5),)))

    def test_gram_schmidt_special_cases(self):
        # No row at all, with 0 and with 3 columns; one row; already orthogonal rows; rational entries; a square basis.
        self.assertAccepts(GS, {"M": mf.Matrix((), ncols=0)}, mf.Matrix((), ncols=0), ())
        self.assertAccepts(GS, {"M": mf.Matrix((), ncols=3)}, mf.Matrix((), ncols=3), ())
        for rows in ([[5]], [[0, -2, 0]], [[1, 1], [1, -1]], [[Q(1, 2), Q(1, 3)], [Q(-1, 7), 2]],
                     [[1, 2, 3], [4, 5, 6], [7, 8, 10]], [[0, 0, 1], [0, 1, 1], [1, 1, 1]]):
            with self.subTest(rows=rows):
                self.assertAccepts(GS, *gs_claim(rows))
        # Evidence Rationals need not be in lowest terms.
        self.assertAccepts(GS, {"M": to_matrix([[3, 1], [2, 2]])}, to_matrix([[3, 1], [Q(-2, 5), Q(6, 5)]]),
                           ((), (trusted_rational(8, 10),)))

    def test_gram_schmidt_seeded_matrices(self):
        rng = random.Random(1113)
        done = 0
        while done < 30:
            n = rng.randint(1, 6)
            k = rng.randint(1, n)
            rows = [[F(rng.randint(-6, 6), rng.choice((1, 1, 2, 5))) for _ in range(n)] for _ in range(k)]
            try:
                claim = gs_claim(rows)
            except AssertionError:  # Dependent rows: no Gram-Schmidt value.
                continue
            done += 1
            with self.subTest(rows=rows):
                self.assertAccepts(GS, *claim)

    def test_gram_schmidt_larger_matrix_quickly(self):
        rng = random.Random(11)
        rows = [[F(rng.randint(-9, 9)) for _ in range(14)] for _ in range(12)]
        claim = gs_claim(rows)
        started = time.perf_counter()
        self.assertAccepts(GS, *claim)
        self.assertLess(time.perf_counter() - started, FAST)

    # Request caps: the only ResourceLimitError of the checker.

    def test_request_caps(self):
        big = support.oversized_matrix()
        for operation in (DEF, GS):
            with self.subTest(operation=operation):
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, {"M": big}, Inertia(65, 0, 0, "positive_definite"), ((), ()))
        with computation(ComputationLimits(max_integer_bits=80_000), fresh=True):
            huge = mf.Matrix(((Q(support.HUGE),),))
        for operation, value, evidence in ((DEF, Inertia(1, 0, 0, "positive_definite"), (((Q(1),),), (Q(1),))),
                                           (GS, to_matrix([[1]]), ((),))):
            with self.subTest(operation=operation, huge=True):
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, {"M": huge}, value, evidence)

    # definiteness: corrupted claims.

    def test_definiteness_rejects_wrong_values(self):
        args, value, evidence = definiteness_claim([[2, 1], [1, 2]])
        for wrong in (Inertia(1, 1, 0, "indefinite"), Inertia(1, 0, 1, "positive_semidefinite"),
                      Inertia(0, 2, 0, "negative_definite"), Inertia(3, 0, 0, "positive_definite"),
                      Inertia(2, 0, 1, "positive_semidefinite")):
            with self.subTest(wrong=wrong):
                self.assertRejects(DEF, args, wrong, evidence)
        # A wrong D with the right signs, and a wrong D whose signs match the wrong value.
        s, d = evidence
        self.assertRejects(DEF, args, value, (s, vector([2, 2])))
        self.assertRejects(DEF, args, Inertia(1, 1, 0, "indefinite"), (s, vector([2, Q(-3, 2)])))
        # A non-symmetric M: no claim passes.
        identity_s = rationals([[1, 0], [0, 1]])
        for claimed, d_ in ((Inertia(2, 0, 0, "positive_definite"), vector([1, 1])),
                            (Inertia(1, 0, 1, "positive_semidefinite"), vector([1, 0]))):
            with self.subTest(claimed=claimed):
                self.assertRejects(DEF, {"M": to_matrix([[1, 2], [0, 1]])}, claimed, (identity_s, d_))
        # A symmetric M with the identity for S and a D that is not S^T M S.
        self.assertRejects(DEF, {"M": to_matrix([[1, 0], [0, 0]])}, Inertia(2, 0, 0, "positive_definite"),
                           (identity_s, vector([1, 1])))
        # A non-square M.
        self.assertRejects(DEF, {"M": to_matrix([[1, 0, 0], [0, 1, 0]])}, Inertia(2, 0, 0, "positive_definite"),
                           (rationals([[1, 0], [0, 1]]), vector([1, 1])))

    def test_definiteness_rejects_lower_asymmetry(self):
        # Regression: the checker computes S^T M S on and above the diagonal only, so its symmetry test is what
        # rejects an M whose asymmetry lies below the diagonal: with S = I the upper part of S^T M S is diagonal.
        identity_s = rationals(identity(2))
        self.assertRejects(DEF, {"M": to_matrix([[1, 0], [2, 1]])}, Inertia(2, 0, 0, "positive_definite"),
                           (identity_s, vector([1, 1])))
        self.assertRejects(DEF, {"M": to_matrix([[1, 0, 0], [5, 2, 0], [7, 3, -1]])}, Inertia(2, 1, 0, "indefinite"),
                           (rationals(identity(3)), vector([1, 2, -1])))
        # Equal values in different (non-canonical) forms are symmetric.
        forged_m = support.forged(mf.Matrix, rows=((Q(1), trusted_rational(2, 4)), (Q(1, 2), Q(1))), ncols=2)
        args, value, evidence = definiteness_claim([[1, Q(1, 2)], [Q(1, 2), 1]])
        self.assertAccepts(DEF, {"M": forged_m}, value, evidence)

    def test_definiteness_invertibility_beyond_the_moduli(self):
        # Regression for the modular invertibility test: a determinant that every one of its moduli divides leaves
        # the decision to the exact elimination, which accepts a nonzero one and rejects 0.
        from mathforge.certify.linalg import forms as checker
        product = 1
        for modulus in checker._MODULI:
            product *= modulus
        identity_m = {"M": to_matrix(identity(2))}
        pd = Inertia(2, 0, 0, "positive_definite")
        self.assertAccepts(DEF, identity_m, pd, (rationals([[product, 0], [0, 1]]), vector([product * product, 1])))
        self.assertAccepts(DEF, identity_m, pd, (rationals([[product, product], [product, -product]]),
                                                 vector([2 * product * product] * 2)))
        self.assertAccepts(DEF, {"M": to_matrix([[1]])}, Inertia(1, 0, 0, "positive_definite"),
                           (rationals([[F(1, product)]]), vector([F(1, product * product)])))
        # Singular S, every entry a multiple of the moduli: zero modulo each, and Bareiss finds no pivot.
        self.assertRejects(DEF, identity_m, Inertia(1, 0, 1, "positive_semidefinite"),
                           (rationals([[product, 0], [0, 0]]), vector([product * product, 0])))
        self.assertRejects(DEF, {"M": to_matrix([[1, 0], [0, -1]])}, Inertia(0, 0, 2, "positive_semidefinite"),
                           (rationals([[product, product], [product, product]]), vector([0, 0])))

    def test_definiteness_rejects_noncanonical_values(self):
        args, _, evidence = definiteness_claim([[0, 0], [0, 0]])
        # The zero matrix is also negative semidefinite, but the card's order names it positive_semidefinite.
        for kind in ("negative_semidefinite", "positive_definite", "indefinite", "Positive_semidefinite",
                     "positive semidefinite"):
            with self.subTest(kind=kind):
                self.assertRejects(DEF, args, support.forged(Inertia, positive=0, negative=0, zero=2, kind=kind),
                                   evidence)
        self.assertRejects(DEF, {"M": mf.Matrix((), ncols=0)},
                           support.forged(Inertia, positive=0, negative=0, zero=0, kind="negative_definite"), ((), ()))

    def test_definiteness_rejects_impostors_and_forged_records(self):
        args, value, evidence = definiteness_claim([[1, 2], [2, 1]])
        self.assertRejects(DEF, args, support.impostor(value), evidence)
        self.assertRejects(DEF, args, (1, 1, 0, "indefinite"), evidence)
        self.assertRejects(DEF, args, to_matrix([[1]]), evidence)
        self.assertRejects(DEF, args, None, evidence)
        for fields in ({"positive": 1, "negative": 1, "zero": 0}, {"positive": 1, "negative": 1, "kind": "indefinite"},
                       {}, {"positive": 1, "negative": 1, "zero": -1, "kind": "indefinite"},
                       {"positive": 2, "negative": 1, "zero": -1, "kind": "indefinite"}):
            with self.subTest(fields=fields):
                self.assertRejects(DEF, args, support.forged(Inertia, **fields), evidence)

    def test_definiteness_rejects_inexact_numbers_and_lists(self):
        args, value, evidence = definiteness_claim([[2, 1], [1, 2]])
        s, d = evidence
        for bad in (True, 1.0, "1", Q(1), None):
            with self.subTest(count=bad):
                self.assertRejects(DEF, args, support.forged(Inertia, positive=bad, negative=0, zero=1,
                                                             kind="positive_semidefinite"), evidence)
        self.assertRejects(DEF, args, support.forged(Inertia, positive=True, negative=False, zero=False,
                                                     kind="positive_definite"), evidence)
        self.assertRejects(DEF, args, support.forged(Inertia, positive=2, negative=0, zero=0, kind=None), evidence)
        for bad in (2, True, 2.0, "2", F(2)):
            with self.subTest(entry=bad):
                self.assertRejects(DEF, args, value, (s, replaced(d, 0, bad)))
                self.assertRejects(DEF, args, value, (replaced(s, 0, replaced(s[0], 0, bad)), d))
        self.assertRejects(DEF, args, value, [s, d])
        self.assertRejects(DEF, args, value, (list(s), d))
        self.assertRejects(DEF, args, value, ((list(s[0]), s[1]), d))
        self.assertRejects(DEF, args, value, (s, list(d)))
        # Forged Rationals: a zero or negative denominator, a missing slot.
        for bad in (trusted_rational(1, 0), trusted_rational(1, -1), trusted_rational(1, True),
                    trusted_rational(2.0, 1)):
            with self.subTest(forged=bad):
                self.assertRejects(DEF, args, value, (s, replaced(d, 0, bad)))
                self.assertRejects(DEF, args, value, (replaced(s, 1, replaced(s[1], 1, bad)), d))

    def test_definiteness_rejects_missing_extra_and_misshapen_evidence(self):
        args, value, evidence = definiteness_claim([[2, 1], [1, 2]])
        s, d = evidence
        for bad in ((), (s,), (s, d, ()), (d, s), (s, d[:1]), (s, d + (Q(1),)), (s[:1], d), (s + (s[0],), d),
                    ((s[0][:1], s[1]), d), ((s[0] + (Q(0),), s[1]), d), None, s, ((), ())):
            with self.subTest(evidence=bad):
                self.assertRejects(DEF, args, value, bad)

    def test_definiteness_rejects_forged_consistent_evidence(self):
        m = {"M": to_matrix([[2, 1], [1, 2]])}
        # Singular S whose S^T M S is diagonal: columns (1, 1) and 0 give diag(6, 0), a false (1, 0, 1).
        self.assertRejects(DEF, m, Inertia(1, 0, 1, "positive_semidefinite"),
                           (rationals([[1, 0], [1, 0]]), vector([6, 0])))
        # S = 0 claims the zero inertia of any matrix.
        self.assertRejects(DEF, m, Inertia(0, 0, 2, "positive_semidefinite"),
                           (rationals([[0, 0], [0, 0]]), vector([0, 0])))
        # Proportional columns: (1, -1) twice gives diag(2, 2) off-diagonal 2, and with columns (1, 1), (2, 2)
        # S^T M S = [[6, 12], [12, 24]], not diagonal.
        self.assertRejects(DEF, m, Inertia(2, 0, 0, "positive_definite"),
                           (rationals([[1, 2], [1, 2]]), vector([6, 24])))
        # A 3 x 3 singular S with a zero column and a diagonal product.
        m3 = {"M": to_matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])}
        self.assertRejects(DEF, m3, Inertia(2, 0, 1, "positive_semidefinite"),
                           (rationals([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), vector([1, 0, 1])))
        # The right inertia with a D that matches a different S.
        self.assertRejects(DEF, m, Inertia(2, 0, 0, "positive_definite"),
                           (rationals([[1, 1], [1, -1]]), vector([2, 6])))

    def test_definiteness_rejects_wrong_operations_and_arguments(self):
        args, value, evidence = definiteness_claim([[2, 1], [1, 2]])
        for operation in (GS, "Definiteness", "inertia", "", None, b"definiteness"):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)
        for bad in ({"A": args["M"]}, {"M": args["M"], "k": 1}, {}, [("M", args["M"])], None,
                    {"M": ((Q(2), Q(1)), (Q(1), Q(2)))}, {"M": support.impostor(args["M"])},
                    {"M": support.forged(mf.Matrix, rows=[[Q(2), Q(1)], [Q(1), Q(2)]], ncols=2)},
                    {"M": support.forged(mf.Matrix, rows=((Q(2), Q(1)), (Q(1), Q(2))), ncols=True)},
                    {"M": support.forged(mf.Matrix, rows=((Q(2), Q(1)), (Q(1), Q(2))))},
                    {"M": support.forged(mf.Matrix, rows=((2, Q(1)), (Q(1), Q(2))), ncols=2)},
                    {"M": support.forged(mf.Matrix, rows=(), ncols=-1)}):
            with self.subTest(args=bad):
                self.assertRejects(DEF, bad, value, evidence)

    def test_definiteness_rejects_oversized_forgeries_fast(self):
        args, value, evidence = definiteness_claim([[2, 1], [1, 2]])
        s, d = evidence
        giant = trusted_rational(GIANT, 1)
        self.assertRejectsFast(DEF, args, value, (s, replaced(d, 0, giant)))
        self.assertRejectsFast(DEF, args, value, (s, replaced(d, 0, trusted_rational(1, GIANT))))
        self.assertRejectsFast(DEF, args, value, (replaced(s, 0, replaced(s[0], 1, giant)), d))
        self.assertRejectsFast(DEF, args, support.forged(Inertia, positive=GIANT, negative=0, zero=0,
                                                         kind="positive_definite"), evidence)
        self.assertRejectsFast(DEF, args, value, (s,) * 100_000)
        self.assertRejectsFast(DEF, args, value, (s, d + (Q(1),) * 100_000))
        # Many long Rationals within max_integer_bits: refused (or out of budget) fast.
        n = 12
        m = {"M": to_matrix(identity(n))}
        long_s = tuple(tuple(Q(BIG + i + j, BIG - i) for j in range(n)) for i in range(n))
        long_d = tuple(Q(BIG + i) for i in range(n))
        self.assertRejectsFast(DEF, m, Inertia(n, 0, 0, "positive_definite"), (long_s, long_d), may_exhaust=True)

    # gram_schmidt: corrupted claims.

    def test_gram_schmidt_rejects_wrong_values(self):
        args, value, evidence = gs_claim([[3, 1], [2, 2]])
        for rows, mu in (([[3, 1], [-2, 6]], ((), (Q(4, 5),))),  # q_1 scaled by 5: orthogonal, M != L Q.
                         ([[3, 1], [Q(-2, 5), Q(6, 5)]], ((), (Q(3, 5),))),  # Wrong mu.
                         ([[3, 1], [2, 2]], ((), (Q(0),))),  # M itself: not orthogonal.
                         ([[3, 1], [Q(2, 5), Q(-6, 5)]], ((), (Q(4, 5),))),
                         ([[Q(3, 10), Q(1, 10)], [Q(-2, 5), Q(6, 5)]], ((), (Q(8),))),  # q_0 rescaled.
                         ([[2, 2], [1, -1]], ((), (Q(1),)))):  # The Gram-Schmidt rows of the swapped rows.
            with self.subTest(rows=rows, mu=mu):
                self.assertRejects(GS, args, to_matrix(rows), mu)
        self.assertRejects(GS, args, to_matrix([[3, 1]]), ((),))
        self.assertRejects(GS, args, to_matrix([[3, 1, 0], [Q(-2, 5), Q(6, 5), 0]]), evidence)
        self.assertRejects(GS, args, mf.Matrix((), ncols=2), ())
        self.assertRejects(GS, {"M": mf.Matrix((), ncols=2)}, mf.Matrix((), ncols=3), ())

    def test_gram_schmidt_rejects_noncanonical_values(self):
        args, value, evidence = gs_claim([[3, 1], [2, 2]])
        (q00, q01), (q10, q11) = value.rows
        for row in ((trusted_rational(-4, 10), q11), (q10, trusted_rational(-6, -5)), (-Q(2, 5), 1.2)):
            with self.subTest(row=row):
                self.assertRejects(GS, args, support.forged(mf.Matrix, rows=((q00, q01), row), ncols=2), evidence)
        args, value, evidence = gs_claim([[1, 0], [0, 1]])
        for row in ((trusted_rational(0, 2), Q(1)), (Q(0), 1), (False, Q(1))):
            with self.subTest(row=row):
                self.assertRejects(GS, args, support.forged(mf.Matrix, rows=(value.rows[0], row), ncols=2), evidence)

    def test_gram_schmidt_rejects_impostors_and_forged_records(self):
        args, value, evidence = gs_claim([[1, 1, 0], [1, 0, 1]])
        self.assertRejects(GS, args, support.impostor(value), evidence)
        self.assertRejects(GS, args, value.rows, evidence)
        self.assertRejects(GS, args, Inertia(2, 0, 0, "positive_definite"), evidence)
        for fields in ({"rows": value.rows}, {"ncols": 3}, {"rows": list(value.rows), "ncols": 3},
                       {"rows": (list(value.rows[0]), value.rows[1]), "ncols": 3}, {"rows": value.rows, "ncols": 2},
                       {"rows": value.rows, "ncols": True}, {"rows": value.rows, "ncols": 3.0},
                       {"rows": value.rows[:1], "ncols": 3}, {"rows": value.rows + (value.rows[0],), "ncols": 3}):
            with self.subTest(fields=fields):
                self.assertRejects(GS, args, support.forged(mf.Matrix, **fields), evidence)

    def test_gram_schmidt_reads_the_column_count_once(self):
        # Regression: an object named like Matrix whose ncols answers 0 and then 3 was read twice (the type test, then
        # the comparison) and passed as the 0 x 3 Matrix; read once, it is a 0 x 0 claim for a 0 x 3 M.
        class Matrix:
            rows = ()

            def __init__(self) -> None:
                self.reads = 0

            @property
            def ncols(self) -> int:
                self.reads += 1
                return 0 if self.reads % 2 else 3

        Matrix.__module__, Matrix.__qualname__ = "mathforge.matrices", "Matrix"
        self.assertRejects(GS, {"M": mf.Matrix((), ncols=3)}, Matrix(), ())
        self.assertAccepts(GS, {"M": mf.Matrix((), ncols=3)}, mf.Matrix((), ncols=3), ())

    def test_gram_schmidt_rejects_inexact_numbers_lists_and_misshapen_evidence(self):
        args, value, evidence = gs_claim([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        for bad in (1, True, 0.5, "1", F(1, 2), None, trusted_rational(1, 0), trusted_rational(1, -2)):
            with self.subTest(entry=bad):
                self.assertRejects(GS, args, value, replaced(evidence, 2, replaced(evidence[2], 1, bad)))
        for bad in ((), evidence[:2], evidence + ((),), list(evidence), (list(evidence[0]),) + evidence[1:],
                    replaced(evidence, 1, list(evidence[1])), replaced(evidence, 1, ()),
                    replaced(evidence, 0, (Q(0),)), replaced(evidence, 2, evidence[2][:1]),
                    replaced(evidence, 2, evidence[2] + (Q(0),)), (evidence,), None, value.rows):
            with self.subTest(evidence=bad):
                self.assertRejects(GS, args, value, bad)

    def test_gram_schmidt_rejects_forged_consistent_evidence(self):
        # Dependent rows: Q with a zero row is orthogonal to everything and M = L Q holds, but q_1 = 0.
        m = {"M": to_matrix([[1, 1], [2, 2]])}
        self.assertRejects(GS, m, to_matrix([[1, 1], [0, 0]]), ((), (Q(2),)))
        m = {"M": to_matrix([[1, 0], [0, 1], [1, 1]])}
        self.assertRejects(GS, m, to_matrix([[1, 0], [0, 1], [0, 0]]), ((), (Q(0),), (Q(1), Q(1))))
        # Zero rows of M: never independent.
        self.assertRejects(GS, {"M": to_matrix([[0, 0]])}, to_matrix([[0, 0]]), ((),))
        self.assertRejects(GS, {"M": mf.Matrix(((), ()), ncols=0)}, mf.Matrix(((), ()), ncols=0), ((), (Q(0),)))
        # Orthogonal rows whose dot products with M are right, but M's rows leave their span: Bessel's equality
        # fails (|m_0|^2 = 2, while (m_0 . q_0)^2 / N_0 = 1).
        self.assertRejects(GS, {"M": to_matrix([[1, 0, 1]])}, to_matrix([[1, 0, 0]]), ((),))
        self.assertRejects(GS, {"M": to_matrix([[1, 0, 0], [1, 1, 1]])}, to_matrix([[1, 0, 0], [0, 1, 0]]),
                           ((), (Q(1),)))
        # An orthogonal basis of the same span with a lower triangular L that is not unit.
        m = {"M": to_matrix([[3, 1], [2, 2]])}
        self.assertRejects(GS, m, to_matrix([[1, 0], [0, 1]]), ((), (Q(2),)))

    def test_gram_schmidt_rejects_wrong_operations_and_arguments(self):
        args, value, evidence = gs_claim([[3, 1], [2, 2]])
        for operation in (DEF, "Gram_schmidt", "gram-schmidt", None):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)
        for bad in ({"A": args["M"]}, {"M": args["M"], "x": 1}, {}, None, {"M": args["M"].rows},
                    {"M": support.impostor(args["M"])},
                    {"M": support.forged(mf.Matrix, rows=((Q(3), Q(1)), [Q(2), Q(2)]), ncols=2)},
                    {"M": support.forged(mf.Matrix, rows=((Q(3), Q(1)), (Q(2), 2.0)), ncols=2)}):
            with self.subTest(args=bad):
                self.assertRejects(GS, bad, value, evidence)

    def test_gram_schmidt_rejects_oversized_forgeries_fast(self):
        args, value, evidence = gs_claim([[3, 1], [2, 2]])
        giant = trusted_rational(GIANT, 1)
        self.assertRejectsFast(GS, args, value, ((), (giant,)))
        self.assertRejectsFast(GS, args, value, ((), (trusted_rational(1, GIANT),)))
        self.assertRejectsFast(GS, args, support.forged(mf.Matrix, rows=((giant, Q(1)), value.rows[1]), ncols=2),
                               evidence)
        self.assertRejectsFast(GS, args, value, evidence + ((),) * 100_000)
        self.assertRejectsFast(GS, args, support.forged(mf.Matrix, rows=value.rows * 50_000, ncols=2), evidence)
        # Many long Rationals within max_integer_bits.
        n = 12
        m = {"M": to_matrix(identity(n))}
        long_q = support.forged(mf.Matrix, rows=tuple(tuple(Q(BIG + 2 * j, BIG - 2 * i) for j in range(n))
                                                      for i in range(n)), ncols=n)
        long_mu = tuple(tuple(Q(BIG + j, BIG + 2 * i) for j in range(i)) for i in range(n))
        self.assertRejectsFast(GS, m, long_q, long_mu, may_exhaust=True)
        self.assertRejectsFast(GS, m, to_matrix(identity(n)), long_mu, may_exhaust=True)

    # The record and its rendering.

    def test_inertia_refuses_noncanonical_forms(self):
        for fields in ((2, 0, 0, "positive_semidefinite"), (0, 0, 2, "negative_semidefinite"),
                       (0, 0, 0, "negative_definite"), (1, 1, 0, "positive_definite"), (-1, 0, 0, "positive_definite"),
                       (True, 0, 0, "positive_definite"), (1.0, 0, 0, "positive_definite"), (1, 0, 0, None),
                       ("1", 0, 0, "positive_definite"), (support.HUGE, 0, 0, "positive_definite")):
            with self.subTest(fields=fields):
                with self.assertRaises((InvalidInput, ResourceLimitError)):
                    Inertia(*fields)
        self.assertEqual(Inertia(0, 0, 0, "positive_definite").kind, "positive_definite")

    def test_rendering(self):
        cases = ((Inertia(3, 0, 0, "positive_definite"), "positive definite (3, 0, 0)",
                  r"\text{positive definite}\ \left(3, 0, 0\right)"),
                 (Inertia(1, 1, 0, "indefinite"), "indefinite (1, 1, 0)", r"\text{indefinite}\ \left(1, 1, 0\right)"),
                 (Inertia(0, 1, 1, "negative_semidefinite"), "negative semidefinite (0, 1, 1)",
                  r"\text{negative semidefinite}\ \left(0, 1, 1\right)"))
        for value, text, latex in cases:
            with self.subTest(value=value):
                self.assertEqual(mf.render_text(value), text)
                self.assertEqual(mf.to_latex(value), latex)


def inverse(a: Rows) -> Rows:
    n = len(a)
    work = [row[:] + identity(n)[i] for i, row in enumerate(a)]
    for column in range(n):
        pivot = next(i for i in range(column, n) if work[i][column])
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [x / scale for x in work[column]]
        for i in range(n):
            if i != column and work[i][column]:
                factor = work[i][column]
                work[i] = [x - factor * y for x, y in zip(work[i], work[column])]
    return [row[n:] for row in work]
