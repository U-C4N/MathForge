"""The independent checker of mathforge.areas.linalg.lattice (mathforge.certify.linalg.lattice), tested on its own:
no true claim here comes from the family's producer.

Every true claim is computed here, never by the producer: by the card's known values, and by short reference
reductions written below (integer row and column operations that track a unimodular matrix and its inverse at every
step), which are themselves checked against brute force: the Smith normal form against the determinantal divisors
(gcds of all j x j minors), the Hermite normal form against the shape rules and the lattice it generates. The evidence
always has the card's layout: (U, V) for hermite_normal_form and (U, U_inv, V, V_inv) for smith_normal_form, each an
integer matrix as a tuple of tuples of ints. The checker answers False to every corrupted claim, never with an
exception, and refuses an oversized forgery within FAST seconds under the default budget; ResourceLimitError comes
only where the checker's docstring lists it (sizes and work through the budget).
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
import inspect
from itertools import combinations, permutations
from math import gcd
import random
import time

import mathforge as mf
from mathforge.areas.linalg import lattice as family
from mathforge.certify.linalg import lattice as checker
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.model import Rational

from .. import support

FAST = 3.0  # Seconds: an oversized forgery is refused well within this under the default budget.
BITS = ComputationLimits().max_integer_bits
HNF, SNF = "hermite_normal_form", "smith_normal_form"

Rows = list[list[int]]


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def identity(n: int) -> Rows:
    return [[int(i == j) for j in range(n)] for i in range(n)]


def multiply(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]], width: int) -> Rows:
    """left (r x s) times right (s x width)."""
    return [[sum(row[k] * right[k][j] for k in range(len(right))) for j in range(width)] for row in left]


def frozen(rows: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Integer rows in the evidence layout: a tuple of tuples of ints."""
    return tuple(tuple(row) for row in rows)


class Tracked:
    """An integer matrix A under row operations (A <- E A) and column operations (A <- A F), with U, U_inv, V, V_inv
    such that A = U A_0 V, U U_inv = I and V V_inv = I at every step."""

    def __init__(self, rows: Sequence[Sequence[int]], ncols: int) -> None:
        self.a = [list(row) for row in rows]
        self.m, self.n = len(self.a), ncols
        self.u, self.u_inv = identity(self.m), identity(self.m)
        self.v, self.v_inv = identity(self.n), identity(self.n)

    def row_swap(self, i: int, j: int) -> None:
        for rows in (self.a, self.u):
            rows[i], rows[j] = rows[j], rows[i]
        for row in self.u_inv:  # U_inv E^-1, E^-1 = E: swap columns.
            row[i], row[j] = row[j], row[i]

    def row_add(self, i: int, j: int, c: int) -> None:
        """row i += c * row j; U_inv's column j -= c * its column i."""
        for rows in (self.a, self.u):
            rows[i] = [x + c * y for x, y in zip(rows[i], rows[j])]
        for row in self.u_inv:
            row[j] -= c * row[i]

    def row_negate(self, i: int) -> None:
        for rows in (self.a, self.u):
            rows[i] = [-x for x in rows[i]]
        for row in self.u_inv:
            row[i] = -row[i]

    def col_swap(self, i: int, j: int) -> None:
        for rows in (self.a, self.v):
            for row in rows:
                row[i], row[j] = row[j], row[i]
        self.v_inv[i], self.v_inv[j] = self.v_inv[j], self.v_inv[i]

    def col_add(self, i: int, j: int, c: int) -> None:
        """column i += c * column j; V_inv's row j -= c * its row i."""
        for rows in (self.a, self.v):
            for row in rows:
                row[i] += c * row[j]
        self.v_inv[j] = [x - c * y for x, y in zip(self.v_inv[j], self.v_inv[i])]


def reference_hnf(rows: Sequence[Sequence[int]], ncols: int) -> tuple[Rows, Rows, Rows]:
    """(H, U, V): the row Hermite normal form by gcd elimination down each column, then reduction above the pivot."""
    t = Tracked(rows, ncols)
    r = 0
    for c in range(ncols):
        if r == t.m:
            break
        while True:
            nonzero = [i for i in range(r, t.m) if t.a[i][c]]
            if not nonzero:
                break
            t.row_swap(r, min(nonzero, key=lambda i: abs(t.a[i][c])))
            for i in range(r + 1, t.m):
                if t.a[i][c]:
                    t.row_add(i, r, -(t.a[i][c] // t.a[r][c]))
            if not any(t.a[i][c] for i in range(r + 1, t.m)):
                break
        if not t.a[r][c]:
            continue
        if t.a[r][c] < 0:
            t.row_negate(r)
        for i in range(r):
            t.row_add(i, r, -(t.a[i][c] // t.a[r][c]))  # Floor division: the entry lands in [0, pivot).
        r += 1
    return t.a, t.u, t.u_inv


def reference_snf(rows: Sequence[Sequence[int]], ncols: int) -> tuple[tuple[int, ...], Rows, Rows, Rows, Rows]:
    """(d, U, U_inv, V, V_inv): the Smith normal form by the classical pivot reduction."""
    t = Tracked(rows, ncols)
    k = min(t.m, ncols)
    for s in range(k):
        while True:
            entries = [(abs(t.a[i][j]), i, j) for i in range(s, t.m) for j in range(s, ncols) if t.a[i][j]]
            if not entries:
                break
            _, i, j = min(entries)
            t.row_swap(s, i)
            t.col_swap(s, j)
            pivot = t.a[s][s]
            for i in range(s + 1, t.m):
                t.row_add(i, s, -(t.a[i][s] // pivot))
            for j in range(s + 1, ncols):
                t.col_add(j, s, -(t.a[s][j] // pivot))
            if any(t.a[i][s] for i in range(s + 1, t.m)) or any(t.a[s][j] for j in range(s + 1, ncols)):
                continue
            bad = [i for i in range(s + 1, t.m) for j in range(s + 1, ncols) if t.a[i][j] % pivot]
            if bad:
                t.row_add(s, bad[0], 1)
                continue
            break
        if t.a[s][s] < 0:
            t.row_negate(s)
    return tuple(t.a[i][i] for i in range(k)), t.u, t.u_inv, t.v, t.v_inv


def determinant(rows: Sequence[Sequence[int]]) -> int:
    """The determinant by the Leibniz formula: brute force for the small minors of the tests."""
    n = len(rows)
    total = 0
    for perm in permutations(range(n)):
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    sign = -sign
        product = sign
        for i in range(n):
            product *= rows[i][perm[i]]
        total += product
    return total


def brute_snf(rows: Sequence[Sequence[int]], ncols: int) -> tuple[int, ...]:
    """The invariant factors from the determinantal divisors D_j, the gcd of all j x j minors: d_j = D_j / D_(j-1)."""
    m = len(rows)
    divisors = [1]
    for j in range(1, min(m, ncols) + 1):
        g = 0
        for chosen_rows in combinations(range(m), j):
            for chosen_cols in combinations(range(ncols), j):
                g = gcd(g, determinant([[rows[r][c] for c in chosen_cols] for r in chosen_rows]))
        divisors.append(g)
    return tuple(divisors[j] // divisors[j - 1] if divisors[j] else 0 for j in range(1, len(divisors)))


def matrix(rows: Sequence[Sequence[object]], ncols: int | None = None) -> mf.Matrix:
    rows = [list(row) for row in rows]
    return mf.Matrix(tuple(tuple(row) for row in rows), ncols=(len(rows[0]) if rows else 0) if ncols is None else ncols)


def args_of(rows: Sequence[Sequence[object]], ncols: int | None = None) -> dict[str, object]:
    return {"M": matrix(rows, ncols)}


def forged_matrix(rows: object, ncols: object) -> mf.Matrix:
    return support.forged(mf.Matrix, rows=rows, ncols=ncols)


def forged_rational(numerator: object, denominator: object) -> Rational:
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def hnf_claim(rows: Sequence[Sequence[int]], ncols: int | None = None) -> tuple[dict, mf.Matrix, tuple]:
    """(args, H, (U, V)) computed by reference_hnf."""
    ncols = (len(rows[0]) if rows else 0) if ncols is None else ncols
    h, u, v = reference_hnf(rows, ncols)
    return args_of(rows, ncols), matrix(h, ncols), (frozen(u), frozen(v))


def snf_claim(rows: Sequence[Sequence[int]], ncols: int | None = None) -> tuple[dict, tuple, tuple]:
    """(args, d, (U, U_inv, V, V_inv)) computed by reference_snf."""
    ncols = (len(rows[0]) if rows else 0) if ncols is None else ncols
    d, u, ui, v, vi = reference_snf(rows, ncols)
    return args_of(rows, ncols), d, (frozen(u), frozen(ui), frozen(v), frozen(vi))


def random_rows(rng: random.Random, m: int, n: int, low: int = -6, high: int = 6) -> Rows:
    return [[rng.randint(low, high) for _ in range(n)] for _ in range(m)]


def rank_deficient(rng: random.Random, m: int, n: int, rank: int) -> Rows:
    """An m x n integer matrix of rank at most `rank`: a product of random m x rank and rank x n factors."""
    return multiply(random_rows(rng, m, rank, -3, 3), random_rows(rng, rank, n, -3, 3), n)


HNF_KNOWN = (
    ([[2, 4], [3, 6]], [[1, 2], [0, 0]]),
    ([[4, 6], [2, 3]], [[2, 3], [0, 0]]),
    ([[2, 3], [5, 7]], [[1, 0], [0, 1]]),
    ([[6, 4], [0, 2]], [[6, 0], [0, 2]]),
)
SNF_KNOWN = (
    ([[2, 4], [6, 8]], (2, 4)),
    ([[1, 2, 3], [4, 5, 6]], (1, 3)),
    ([[0, 0], [0, 0]], (0, 0)),
    ([[2, 0], [0, 3]], (1, 6)),
)


class ReferenceTests(support.FamilyTestCase):
    """The references of these tests agree with brute force and with the card."""

    FAMILY = family

    def test_references_on_seeded_matrices(self):
        rng = random.Random(9071)
        for _ in range(60):
            m, n = rng.randint(0, 4), rng.randint(0, 4)
            rows = random_rows(rng, m, n) if rng.random() < 0.6 else rank_deficient(rng, m, n, rng.randint(0, 2))
            h, u, v = reference_hnf(rows, n)
            self.assertEqual(multiply(u, v, m), identity(m))
            self.assertEqual(multiply(u, rows, n), h)
            d, u, ui, w, wi = reference_snf(rows, n)
            self.assertEqual(d, brute_snf(rows, n), rows)
            self.assertEqual(multiply(u, ui, m), identity(m))
            self.assertEqual(multiply(w, wi, n), identity(n))
            diagonal = [[d[i] if i == j else 0 for j in range(n)] for i in range(m)]
            self.assertEqual(multiply(multiply(u, rows, n), w, n), diagonal)

    def test_references_on_known_values(self):
        for rows, expected in HNF_KNOWN:
            self.assertEqual(reference_hnf(rows, len(rows[0]))[0], expected)
        for rows, expected in SNF_KNOWN:
            self.assertEqual(reference_snf(rows, len(rows[0]))[0], expected)
            self.assertEqual(brute_snf(rows, len(rows[0])), expected)


class CheckerAcceptsTests(support.FamilyTestCase):
    FAMILY = family

    def test_module_shape(self):
        self.assertEqual(family.CHECKER, "mathforge.certify.linalg.lattice")
        self.assertEqual([spec.name for spec in family.SPECS], [HNF, SNF])
        self.assertEqual(family.TYPES, {})

    def test_known_values(self):
        for rows, expected in HNF_KNOWN:
            args, h, evidence = hnf_claim(rows)
            self.assertEqual(h, matrix(expected))
            self.assertAccepts(HNF, args, matrix(expected), evidence)
        for rows, expected in SNF_KNOWN:
            args, d, evidence = snf_claim(rows)
            self.assertEqual(d, expected)
            self.assertAccepts(SNF, args, expected, evidence)

    def test_hand_written_evidence(self):
        """Evidence that a producer would hardly choose: any unimodular U that maps M to its HNF works."""
        # [[2, 3], [5, 7]] has determinant -1, so its HNF is I with U = M^-1 = [[-7, 3], [5, -2]], V = M.
        self.assertAccepts(HNF, args_of([[2, 3], [5, 7]]), matrix([[1, 0], [0, 1]]),
                           (((-7, 3), (5, -2)), ((2, 3), (5, 7))))
        # [[2, 4], [3, 6]]: row 2 - row 1 gives [1, 2]; then row 1 - 2 row 2 is 0; swap.
        u = [[-1, 1], [3, -2]]  # u M = [[1, 2], [0, 0]].
        v = [[2, 1], [3, 1]]  # Its inverse: det u = -1.
        self.assertEqual(multiply(u, v, 2), identity(2))
        self.assertAccepts(HNF, args_of([[2, 4], [3, 6]]), matrix([[1, 2], [0, 0]]), (frozen(u), frozen(v)))
        # Another U for the same H: adding 5 times the zero row of H to its first row fixes H, so e u maps M to H too.
        e = [[1, 5], [0, 1]]
        e_inv = [[1, -5], [0, 1]]
        self.assertAccepts(HNF, args_of([[2, 4], [3, 6]]), matrix([[1, 2], [0, 0]]),
                           (frozen(multiply(e, u, 2)), frozen(multiply(v, e_inv, 2))))
        # SNF of diag(2, 3) = (1, 6) by hand: U = [[1, 1], [-3, -2]], V = [[-1, -3], [1, 2]].
        u = [[1, 1], [-3, -2]]
        v = [[-1, -3], [1, 2]]
        m = [[2, 0], [0, 3]]
        self.assertEqual(multiply(multiply(u, m, 2), v, 2), [[1, 0], [0, 6]])
        ui = [[-2, -1], [3, 1]]
        vi = [[2, 3], [-1, -1]]
        self.assertAccepts(SNF, args_of(m), (1, 6), (frozen(u), frozen(ui), frozen(v), frozen(vi)))

    def test_special_shapes(self):
        """Empty shapes, zero matrices, a row, a column, rank deficiency, negative entries, units, the identity."""
        cases = [
            ([], 0), ([], 3), ([[], [], []], 0), ([[0, 0, 0]], 3), ([[0], [0]], 1), ([[0, 0], [0, 0]], 2),
            ([[-5]], 1), ([[0]], 1), ([[7]], 1), ([[4, -6, 10]], 3), ([[4], [-6], [10]], 1),
            ([[0, 0, 3], [0, 0, 5]], 3), ([[0, 2, 4], [0, 0, 0], [0, 3, 6]], 3), (identity(4), 4),
            ([[1, 0, 0], [0, 0, 0], [0, 0, 1]], 3), ([[-1, -1], [-1, 1]], 2), ([[12, 18, 30], [8, 12, 20]], 3),
        ]
        for rows, ncols in cases:
            with self.subTest(rows=rows, ncols=ncols):
                self.assertAccepts(HNF, *hnf_claim(rows, ncols))
                self.assertAccepts(SNF, *snf_claim(rows, ncols))
        self.assertAccepts(HNF, args_of([], 3), matrix([], 3), ((), ()))
        self.assertAccepts(SNF, args_of([], 3), (), ((), (), frozen(identity(3)), frozen(identity(3))))
        self.assertAccepts(SNF, args_of([[], []], 0), (), (frozen(identity(2)), frozen(identity(2)), (), ()))
        self.assertAccepts(SNF, args_of([[-5]]), (5,), (((-1,),), ((-1,),), ((1,),), ((1,),)))

    def test_seeded_claims(self):
        rng = random.Random(4417)
        for _ in range(80):
            m, n = rng.randint(1, 6), rng.randint(1, 6)
            rows = random_rows(rng, m, n) if rng.random() < 0.6 else rank_deficient(rng, m, n, rng.randint(1, 3))
            with self.subTest(rows=rows):
                self.assertAccepts(HNF, *hnf_claim(rows, n))
                self.assertAccepts(SNF, *snf_claim(rows, n))

    def test_large_claims_under_the_default_budget(self):
        """A 64 x 64 claim (the largest evidence size of the default limits) and long entries verify in time."""
        rng = random.Random(77)
        # Unitriangular plus a small permutation-free perturbation: HNF evidence of moderate size.
        rows = [[(1 if i == j else (rng.randint(-2, 2) if j > i else 0)) for j in range(64)] for i in range(64)]
        rows[63][0] = 2
        for operation, claim in ((HNF, hnf_claim(rows)), (SNF, snf_claim(rows))):
            start = time.perf_counter()
            self.assertAccepts(operation, *claim)
            self.assertLess(time.perf_counter() - start, FAST)
        big = 2 ** 3000 + 17
        rows = [[big, 3 * big + 1], [5, 7], [big - 1, 2]]
        self.assertAccepts(HNF, *hnf_claim(rows))
        self.assertAccepts(SNF, *snf_claim(rows))

    def test_resource_limits_where_the_docstring_lists_them(self):
        """A 65 x 65 M, a request integer past max_integer_bits, and an M whose true evidence matrices exceed
        max_matrix_entries (65 rows; 65 columns for smith_normal_form) leave the claim unverified."""
        big_args = {"M": support.oversized_matrix()}
        for operation in (HNF, SNF):
            with self.assertRaises(ResourceLimitError):
                support.verdict(family, operation, big_args, None, ())
        huge_args = {"M": forged_matrix(((Rational(support.HUGE),),), 1)}
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, HNF, huge_args, matrix([[1]]), (((1,),), ((1,),)))
        tall = [[i + 1] for i in range(65)]
        args, h, evidence = hnf_claim(tall)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, HNF, args, h, evidence)
        args, d, evidence = snf_claim(tall)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, SNF, args, d, evidence)
        args, d, evidence = snf_claim([[i + 1 for i in range(65)]])
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, SNF, args, d, evidence)
        # Under wider limits the same claims verify.
        wide = ComputationLimits(max_matrix_entries=65 * 65)
        self.assertAccepts(HNF, *hnf_claim(tall), limits=wide)
        self.assertAccepts(SNF, *snf_claim(tall), limits=wide)


class CheckerRejectsTests(support.FamilyTestCase):
    FAMILY = family

    def assertAllRejected(self, operation: str, table: Sequence[tuple[str, object, object, object]]) -> None:
        for label, args, value, evidence in table:
            with self.subTest(label):
                self.assertRejects(operation, args, value, evidence)

    def test_hnf_corruptions(self):
        args, h, (u, v) = hnf_claim([[2, 4], [3, 6]])  # H = [[1, 2], [0, 0]].
        self.assertAccepts(HNF, args, h, (u, v))
        one, two, zero = Rational(1), Rational(2), Rational(0)
        good_rows = ((one, two), (zero, zero))
        ident = ((1, 0), (0, 1))
        table = [
            ("wrong value", args, matrix([[1, 3], [0, 0]]), (u, v)),
            ("wrong value, zero", args, matrix([[0, 0], [0, 0]]), (u, v)),
            ("wrong shape value", args, matrix([[1, 2]]), (u, v)),
            ("wrong ncols", args, forged_matrix(good_rows, 3), (u, v)),
            ("impostor value", args, support.impostor(h), (u, v)),
            ("impostor argument", {"M": support.impostor(args["M"])}, h, (u, v)),
            ("value rows as list", args, forged_matrix(list(good_rows), 2), (u, v)),
            ("value row as list", args, forged_matrix(([one, two], (zero, zero)), 2), (u, v)),
            ("int entry in value", args, forged_matrix(((1, two), (zero, zero)), 2), (u, v)),
            ("bool numerator", args, forged_matrix(((forged_rational(True, 1), two), (zero, zero)), 2), (u, v)),
            ("float numerator", args, forged_matrix(((forged_rational(1.0, 1), two), (zero, zero)), 2), (u, v)),
            ("str numerator", args, forged_matrix(((forged_rational("1", 1), two), (zero, zero)), 2), (u, v)),
            ("denominator 2", args, forged_matrix(((forged_rational(2, 2), two), (zero, zero)), 2), (u, v)),
            ("non-integer value", args, forged_matrix(((Rational(1, 2), two), (zero, zero)), 2), (u, v)),
            ("missing slot", args, forged_matrix(((support.forged(Rational, numerator=1), two), (zero, zero)), 2),
             (u, v)),
            ("missing rows", args, support.forged(mf.Matrix, ncols=2), (u, v)),
            ("bool ncols", args, forged_matrix(good_rows, True), (u, v)),
            ("None value", args, None, (u, v)),
            ("value as tuple", args, ((1, 2), (0, 0)), (u, v)),
            ("no evidence", args, h, ()),
            ("one item", args, h, (u,)),
            ("extra item", args, h, (u, v, v)),
            ("evidence list", args, h, [u, v]),
            ("U as list", args, h, (list(u), v)),
            ("U row as list", args, h, ((list(u[0]), u[1]), v)),
            ("U wrong size", args, h, (((1,),), v)),
            ("U short row", args, h, (((u[0][0],), u[1]), v)),
            ("U bool", args, h, (((True, 0), (0, True)), ident)),
            ("U float", args, h, ((tuple(float(x) for x in u[0]), u[1]), v)),
            ("U str", args, h, ((tuple(str(x) for x in u[0]), u[1]), v)),
            ("U Rational", args, h, ((tuple(Rational(x) for x in u[0]), u[1]), v)),
            ("U as Matrix", args, h, (matrix(u), matrix(v))),
            ("swapped U and V", args, h, (v, u)),
            ("V not inverse", args, h, (u, ident)),
            ("identity evidence", args, h, (ident, ident)),
            ("V None", args, h, (u, None)),
            ("non-integer argument", args_of([[Rational(1, 2), 4], [3, 6]]), h, (u, v)),
            ("args with extra key", {"M": args["M"], "k": 1}, h, (u, v)),
            ("args as tuple", (args["M"],), h, (u, v)),
            ("args wrong key", {"A": args["M"]}, h, (u, v)),
            ("forged argument row list", {"M": forged_matrix([[two, Rational(4)], [Rational(3), Rational(6)]], 2)},
             h, (u, v)),
            ("bool argument entry", {"M": forged_matrix(((True, 4), (3, 6)), 2)}, h, (u, v)),
            ("negative ncols", {"M": forged_matrix((), -1)}, matrix([]), ((), ())),
        ]
        self.assertAllRejected(HNF, table)

    def test_hnf_non_canonical_but_correct(self):
        """U M with U unimodular, consistent evidence, but not in the canonical shape."""
        rows = [[2, 3], [5, 7]]  # HNF I.
        args, h, (u, v) = hnf_claim(rows)
        base_u, base_v = [list(r) for r in u], [list(r) for r in v]

        def transformed(e: Rows, e_inv: Rows) -> tuple[mf.Matrix, tuple]:
            new_u = multiply(e, base_u, 2)
            return matrix(multiply(new_u, rows, 2)), (frozen(new_u), frozen(multiply(base_v, e_inv, 2)))

        for label, e, e_inv in (
            ("entry above pivot too large", [[1, 5], [0, 1]], [[1, -5], [0, 1]]),
            ("entry above pivot negative", [[1, -1], [0, 1]], [[1, 1], [0, 1]]),
            ("negative pivot", [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]),
            ("negative second pivot", [[1, 0], [0, -1]], [[1, 0], [0, -1]]),
            ("rows swapped", [[0, 1], [1, 0]], [[0, 1], [1, 0]]),
            ("lower entry", [[1, 0], [3, 1]], [[1, 0], [-3, 1]]),
        ):
            with self.subTest(label):
                value, evidence = transformed(e, e_inv)
                self.assertNotEqual(value, h)
                self.assertRejects(HNF, args, value, evidence)
        # A zero row first, and pivots not strictly increasing.
        args, h, (u, v) = hnf_claim([[2, 4], [3, 6]])  # H = [[1, 2], [0, 0]].
        swap = [[0, 1], [1, 0]]
        self.assertRejects(HNF, args, matrix([[0, 0], [1, 2]]),
                           (frozen(multiply(swap, u, 2)), frozen(multiply(v, swap, 2))))
        args, h, (u, v) = hnf_claim([[6, 4], [0, 2]])  # H = [[6, 0], [0, 2]]; [[6, 4], [0, 2]] has 4 >= 2 above.
        self.assertRejects(HNF, args, matrix([[6, 4], [0, 2]]), (frozen(identity(2)), frozen(identity(2))))
        self.assertRejects(HNF, args, matrix([[6, -2], [0, 2]]), ((((1, -1), (0, 1))), (((1, 1), (0, 1)))))

    def test_hnf_forged_consistent_evidence(self):
        """H = U M holds and H has the canonical shape, but U is not unimodular (det 2): H is the HNF of an index-2
        sublattice. Evidence with U V = I, but U M != H, is refused as well."""
        args = args_of([[2, 4], [3, 6]])
        u = ((1, 0), (-3, 2))  # u M = [[2, 4], [0, 0]], det u = 2.
        self.assertEqual(multiply(u, [[2, 4], [3, 6]], 2), [[2, 4], [0, 0]])
        for v in (((1, 0), (0, 1)), ((1, 0), (2, 1)), ((1, 0), (3, 1))):
            self.assertRejects(HNF, args, matrix([[2, 4], [0, 0]]), (u, v))
        # [[2, 0], [0, 2]] claimed as the HNF of [[2, 0], [0, 1]] by U = diag(1, 2): no integer inverse.
        self.assertRejects(HNF, args_of([[2, 0], [0, 1]]), matrix([[2, 0], [0, 2]]),
                           (((1, 0), (0, 2)), ((1, 0), (0, 1))))
        # A unimodular pair that maps M elsewhere.
        args, h, (u, v) = hnf_claim([[2, 3], [5, 7]])
        self.assertRejects(HNF, args, h, (((1, 1), (0, 1)), ((1, -1), (0, 1))))

    def test_snf_corruptions(self):
        args, d, (u, ui, v, vi) = snf_claim([[2, 4], [6, 8]])  # (2, 4).
        self.assertAccepts(SNF, args, d, (u, ui, v, vi))
        ev = (u, ui, v, vi)
        ident = ((1, 0), (0, 1))
        table = [
            ("wrong value", args, (2, 8), ev),
            ("wrong value 1", args, (1, 8), ev),
            ("too short", args, (2,), ev),
            ("too long", args, (2, 4, 0), ev),
            ("value as list", args, [2, 4], ev),
            ("bool factor", args, (True, 4), ev),
            ("float factor", args, (2.0, 4), ev),
            ("str factor", args, ("2", 4), ev),
            ("Rational factor", args, (Rational(2), 4), ev),
            ("value as Matrix", args, matrix([[2, 0], [0, 4]]), ev),
            ("None value", args, None, ev),
            ("no evidence", args, d, ()),
            ("three items", args, d, (u, ui, v)),
            ("five items", args, d, (u, ui, v, vi, vi)),
            ("evidence list", args, d, [u, ui, v, vi]),
            ("U as list", args, d, (list(u), ui, v, vi)),
            ("V row as list", args, d, (u, ui, (list(v[0]), v[1]), vi)),
            ("U wrong size", args, d, (((1,),), ui, v, vi)),
            ("V wrong size", args, d, (u, ui, frozen(identity(3)), vi)),
            ("U bool", args, d, (tuple(tuple(bool(x) if x in (0, 1) else x for x in r) for r in u), ui, v, vi)),
            ("V float", args, d, (u, ui, tuple(tuple(float(x) for x in r) for r in v), vi)),
            ("U_inv str", args, d, (u, tuple(tuple(str(x) for x in r) for r in ui), v, vi)),
            ("U as Matrix", args, d, (matrix(u), matrix(ui), matrix(v), matrix(vi))),
            ("U_inv wrong", args, d, (u, ident, v, vi)),
            ("V_inv wrong", args, d, (u, ui, v, ident)),
            ("U and V swapped", args, d, (v, vi, u, ui)),
            ("identity evidence", args, d, (ident, ident, ident, ident)),
            ("impostor argument", {"M": support.impostor(args["M"])}, d, ev),
            ("non-integer argument", args_of([[Rational(1, 2), 4], [6, 8]]), d, ev),
            ("args with extra key", {"M": args["M"], "x": 0}, d, ev),
            ("args as list", [args["M"]], d, ev),
        ]
        self.assertAllRejected(SNF, table)

    def test_snf_non_canonical_but_correct(self):
        """Diagonal forms equivalent to M with consistent evidence that break the canonical form."""
        i2 = frozen(identity(2))
        neg = ((-1, 0), (0, 1))
        swap = ((0, 1), (1, 0))
        # diag(2, 3) itself: a diagonal form, but 2 does not divide 3.
        self.assertRejects(SNF, args_of([[2, 0], [0, 3]]), (2, 3), (i2, i2, i2, i2))
        # (6, 1): diag(6, 1) = P diag(1, 6) P with the swap P.
        args, d, (u, ui, v, vi) = snf_claim([[2, 0], [0, 3]])
        self.assertEqual(d, (1, 6))
        new = (frozen(multiply(swap, u, 2)), frozen(multiply(ui, swap, 2)), frozen(multiply(v, swap, 2)),
               frozen(multiply(swap, vi, 2)))
        self.assertRejects(SNF, args, (6, 1), new)
        # A negative factor: (-1, 6) with the first row of U negated.
        new = (frozen(multiply(neg, u, 2)), frozen(multiply(ui, neg, 2)), v, vi)
        self.assertRejects(SNF, args, (-1, 6), new)
        # Zeros first: [[0, 0], [0, 1]] is (1, 0); (0, 1) with the identity evidence.
        self.assertRejects(SNF, args_of([[0, 0], [0, 1]]), (0, 1), (i2, i2, i2, i2))
        self.assertAccepts(SNF, args_of([[0, 0], [0, 1]]), (1, 0), (swap, swap, swap, swap))

    def test_snf_forged_consistent_evidence(self):
        """U M V = D holds with D in the canonical shape, but U is not unimodular: (1, 12) for diag(2, 3)."""
        args, d, (u, ui, v, vi) = snf_claim([[2, 0], [0, 3]])
        scaled = multiply([[1, 0], [0, 2]], u, 2)
        m = [[2, 0], [0, 3]]
        self.assertEqual(multiply(multiply(scaled, m, 2), v, 2), [[1, 0], [0, 12]])
        self.assertRejects(SNF, args, (1, 12), (frozen(scaled), ui, v, vi))
        # (1, 6) with a unimodular pair that does not diagonalize M.
        self.assertRejects(SNF, args, (1, 6), (frozen(identity(2)),) * 4)

    def test_wrong_operation(self):
        args, h, evidence = hnf_claim([[2, 4], [3, 6]])
        sargs, d, sevidence = snf_claim([[2, 4], [6, 8]])
        for operation in ("hermite_normal_forms", "HERMITE_NORMAL_FORM", "", None, 1, ("hermite_normal_form",)):
            self.assertRejects(operation, args, h, evidence)
        self.assertRejects(SNF, args, h, evidence)
        self.assertRejects(HNF, sargs, d, sevidence)
        self.assertRejects("matrix_power", args, h, evidence)

    def test_oversized_forgeries_fail_fast(self):
        """Values and evidence past max_integer_bits answer False; long consistent-looking evidence answers False or
        exhausts the budget, within FAST seconds."""
        args, h, (u, v) = hnf_claim([[2, 3], [5, 7]])
        huge = support.HUGE

        def timed(operation: str, args: object, value: object, evidence: object, may_exhaust: bool = False) -> None:
            start = time.perf_counter()
            self.assertRejects(operation, args, value, evidence, may_exhaust=may_exhaust)
            self.assertLess(time.perf_counter() - start, FAST)

        timed(HNF, args, forged_matrix(((Rational(huge), Rational(0)), (Rational(0), Rational(1))), 2), (u, v))
        timed(HNF, args, h, (((huge, 0), (0, 1)), v))
        timed(HNF, args, h, (u, ((1, 0), (0, -huge))))
        timed(SNF, args, (1, huge), (u, v, u, v))
        timed(SNF, args, (huge,) * 2, (u, v, u, v))
        # Consistent unimodular evidence with 60,000-bit entries that does not map M to H.
        x = 2 ** 60_000 + 1
        bad_u = frozen(multiply([[1, x], [0, 1]], u, 2))
        bad_v = frozen(multiply(v, [[1, -x], [0, 1]], 2))
        timed(HNF, args, h, (bad_u, bad_v))
        # Large 64 x 64 evidence of long entries: refused or out of budget, fast.
        n = 64
        rng = random.Random(5)
        long_rows = tuple(tuple(rng.getrandbits(60_000) for _ in range(n)) for _ in range(n))
        big_args = args_of(identity(n))
        timed(HNF, big_args, matrix(identity(n)), (long_rows, long_rows), may_exhaust=True)
        timed(SNF, big_args, (1,) * n, (long_rows, long_rows, long_rows, long_rows), may_exhaust=True)
        # A very long chain value: k = 64 factors of 65,536 bits, divisibility charged.
        timed(SNF, big_args, tuple(2 ** (BITS - 1) for _ in range(n)), (long_rows,) * 4, may_exhaust=True)

    def test_never_raises_on_garbage(self):
        args, h, evidence = hnf_claim([[2, 4], [3, 6]])
        garbage = (None, 0, 1.5, "x", (), [], {}, object(), (None,), ((None,),), mf.Matrix(()))
        for item in garbage:
            for operation in (HNF, SNF):
                self.assertRejects(operation, args, item, evidence)
                self.assertRejects(operation, args, h, (item, item))
                self.assertRejects(operation, args, h, (item,) * 4)
                self.assertRejects(operation, {"M": item}, h, evidence)
                self.assertRejects(operation, item, h, evidence)


class CheckerSourceTests(support.FamilyTestCase):
    FAMILY = family

    def test_checker_states_the_theorems_and_imports(self):
        doc = checker.__doc__ or ""
        for phrase in ("hermite_normal_form(M) = H", "smith_normal_form(M) = (d_1, ..., d_k)", "unique",
                       "determinantal divisors", "unimodular"):
            self.assertIn(phrase, doc)
        tree = ast.parse(inspect.getsource(checker))
        allowed = ("mathforge.certify", "mathforge.trusted", "mathforge.errors", "mathforge.limits", "mathforge.model")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mathforge"):
                self.assertTrue(node.module.startswith(allowed), node.module)
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith("mathforge") for alias in node.names))
            if isinstance(node, ast.Name):
                self.assertNotEqual(node.id, "pow")  # No builtin pow: nothing in this checker needs one.

