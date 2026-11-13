"""Tests of the independent checker of mathforge.areas.linalg.eigen (mathforge.certify.linalg.eigen) alone, written
before the family's producer exists: every claim is built here, and the checker answers it as the verifier asks it
(support.verdict, in a fresh budget of the default limits unless a test passes others).

The true claims come from short references of this module, never from a producer or a kernel: the characteristic
polynomial by the Faddeev-LeVerrier recurrence over fractions.Fraction (a third algorithm, beside the producer's
Berkowitz and the checker's Hessenberg form); the rational eigenvalues and their multiplicities by the rational root
theorem (every candidate a/b, a dividing the constant term and b the leading coefficient of the primitive integer form,
tried exactly and divided out as often as it is a root); the factorization of the rest from the construction of each
matrix (companion blocks of chosen irreducible polynomials, Jordan blocks, conjugation), checked against the reference
polynomial, or, for a rest of degree at most 3 without a rational root, the rest itself; the irreducibility
certificates by brute force (the least prime that divides no leading coefficient and modulo which the factor is
square-free, its factors modulo that prime by trial division with every monic polynomial, and Hensel lifting by trying
every correction of each p-adic digit); and each eigenspace by Gauss-Jordan elimination over Fractions, the rows of a
nonsingular minor chosen greedily and the first prime of PRIMES that does not divide its determinant. The evidence
follows the layout of the family's docstring.

The checker answers False to every corrupted claim, never with an exception, and refuses an oversized forgery within
FAST seconds under the default budget, also one made of many long Rationals within max_integer_bits. It raises
ResourceLimitError only for the caps of its docstring, which depend on the matrix alone: more entries than
max_matrix_entries (the card's matrix rule), and a common denominator or a coefficient of the characteristic
polynomial past max_integer_bits. It tests the irreducibility certificates, the one check whose work can grow
exponentially, after everything else.

Run this module by name with

    python -m unittest tests.v09.linalg.test_eigen_checker
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import functools
from itertools import product as cartesian
from math import gcd, isqrt, lcm
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge.areas.linalg import eigen as family
from mathforge.areas.linalg.eigen import Eigenspace, RationalEigenvectors
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError

from .. import support

Q = mf.Rational
F = Fraction
OP = "eigenvectors"
FAST = 3.0  # Seconds: the bound for every oversized forgery, far above what each one takes.
PRIMES = (2**61 - 1, 2**31 - 1, 1_000_000_007, 998_244_353)  # Echelon moduli: the first that the minor allows.
GIANT = 2 ** 70_000  # Longer than max_integer_bits.
BIG = 2 ** 65_000 + 1  # Within max_integer_bits, far beyond anything a small matrix needs.
Rows = list[list[Fraction]]


# ---------------------------------------------------------------------------------------------------------------
# References over fractions.Fraction: matrices as lists of rows, polynomials as lists lowest degree first.

def fraction(value: object) -> Fraction:
    if type(value) is Q:
        return F(value.numerator, value.denominator)
    return F(value)


def rational(value: Fraction) -> mf.Rational:
    return Q(value.numerator, value.denominator)


def fractions(rows: Sequence[Sequence[object]]) -> Rows:
    return [[fraction(x) for x in row] for row in rows]


def to_matrix(rows: Rows) -> mf.Matrix:
    if not rows:
        return mf.Matrix((), ncols=0)
    return mf.Matrix(tuple(tuple(rational(x) for x in row) for row in rows))


def identity(n: int) -> Rows:
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def multiply(a: Rows, b: Rows) -> Rows:
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0)) for j in range(len(b[0]))] for i in range(len(a))]


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


def conjugate(s: Rows, b: Rows) -> Rows:
    """s b s^-1: the same characteristic polynomial and eigenvalues, other eigenvectors."""
    return multiply(multiply(s, b), inverse(s))


def block_diagonal(*blocks: Rows) -> Rows:
    n = sum(len(block) for block in blocks)
    rows = [[F(0)] * n for _ in range(n)]
    offset = 0
    for block in blocks:
        for i, row in enumerate(block):
            for j, x in enumerate(row):
                rows[offset + i][offset + j] = fraction(x)
        offset += len(block)
    return rows


def companion(*tail: int) -> Rows:
    """The companion matrix of t^n + tail[n-1] t^(n-1) + ... + tail[0]: that polynomial is its characteristic one."""
    n = len(tail)
    rows = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        if i:
            rows[i][i - 1] = F(1)
        rows[i][n - 1] = -F(tail[i])
    return rows


def jordan(value: object, size: int) -> Rows:
    return [[fraction(value) if i == j else F(int(j == i + 1)) for j in range(size)] for i in range(size)]


def unimodular(n: int, rng: random.Random, steps: int = 12) -> Rows:
    """A product of elementary row additions with small multipliers: an integer matrix of determinant 1."""
    s = identity(n)
    for _ in range(steps):
        i, j = rng.sample(range(n), 2)
        factor = rng.choice((-2, -1, 1, 2))
        s[i] = [x + factor * y for x, y in zip(s[i], s[j])]
    return s


def charpoly(rows: Rows) -> list[Fraction]:
    """det(t I - A), lowest degree first, by the Faddeev-LeVerrier recurrence: M_0 = 0, c_n = 1, and for k = 1..n,
    M_k = A M_(k-1) + c_(n-k+1) I and c_(n-k) = -tr(A M_k)/k."""
    n = len(rows)
    c = [F(0)] * n + [F(1)]
    m = [[F(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        am = multiply(rows, m)
        m = [[am[i][j] + (c[n - k + 1] if i == j else 0) for j in range(n)] for i in range(n)]
        c[n - k] = -sum(multiply(rows, m)[i][i] for i in range(n)) / k
    return c


def poly_multiply(a: Sequence[Fraction], b: Sequence[Fraction]) -> list[Fraction]:
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += F(x) * y
    return out


def evaluate(p: Sequence[Fraction], x: Fraction) -> Fraction:
    value = F(0)
    for c in reversed(p):
        value = value * x + c
    return value


def deflate(p: Sequence[Fraction], root: Fraction) -> list[Fraction]:
    """p / (t - root) by synthetic division, for a root of p."""
    out, carry = [F(0)] * (len(p) - 1), F(0)
    for i in range(len(p) - 1, 0, -1):
        carry = carry * root + p[i]
        out[i - 1] = carry
    assert carry * root + p[0] == 0
    return out


def primitive_integer(p: Sequence[Fraction]) -> list[int]:
    """The primitive integer polynomial with a positive leading coefficient of a nonzero rational one."""
    scale = lcm(*(F(c).denominator for c in p))
    values = [int(F(c) * scale) for c in p]
    divisor = gcd(*values) * (1 if values[-1] > 0 else -1)
    return [v // divisor for v in values]


def divisors(n: int) -> list[int]:
    n = abs(n)
    small = [d for d in range(1, isqrt(n) + 1) if n % d == 0]
    return sorted(set(small + [n // d for d in small]))


def rational_roots(p: Sequence[Fraction]) -> tuple[dict[Fraction, int], list[Fraction]]:
    """Every rational root of p with its multiplicity, and the rest p / prod (t - root)^multiplicity: the rational
    root theorem on the primitive integer form, each candidate tried exactly and divided out while it is a root."""
    roots: dict[Fraction, int] = {}
    rest = [F(c) for c in p]
    while len(rest) > 1 and rest[0] == 0:
        roots[F(0)] = roots.get(F(0), 0) + 1
        rest = rest[1:]
    if len(rest) > 1:
        integer = primitive_integer(rest)
        candidates = {F(sign * a, b) for a in divisors(integer[0]) for b in divisors(integer[-1]) for sign in (1, -1)}
        for candidate in sorted(candidates):
            while len(rest) > 1 and evaluate(rest, candidate) == 0:
                roots[candidate] = roots.get(candidate, 0) + 1
                rest = deflate(rest, candidate)
    return roots, rest


def rref(rows: Rows, ncols: int) -> tuple[Rows, list[int]]:
    """The nonzero rows of the reduced row echelon form and the pivot columns, by Gauss-Jordan elimination."""
    work = [row[:] for row in rows]
    pivots: list[int] = []
    for column in range(ncols):
        r = len(pivots)
        found = next((i for i in range(r, len(work)) if work[i][column]), None)
        if found is None:
            continue
        work[r], work[found] = work[found], work[r]
        scale = work[r][column]
        work[r] = [x / scale for x in work[r]]
        for i in range(len(work)):
            if i != r and work[i][column]:
                factor = work[i][column]
                work[i] = [x - factor * y for x, y in zip(work[i], work[r])]
        pivots.append(column)
    return work[:len(pivots)], pivots


def nullspace(rows: Rows) -> tuple[list[tuple[Fraction, ...]], list[int]]:
    """The canonical nullspace basis read from the RREF R (1 at each free column f, 0 at the other free columns,
    -R[i][f] at the pivot of row i), and the pivot columns."""
    n = len(rows)
    reduced, pivots = rref(rows, n)
    basis = []
    for free in range(n):
        if free in pivots:
            continue
        vector = [F(0)] * n
        vector[free] = F(1)
        for row, pivot in zip(reduced, pivots):
            vector[pivot] = -row[free]
        basis.append(tuple(vector))
    return basis, pivots


def row_scaled(rows: Rows) -> list[list[int]]:
    """Each row times the least common multiple of its denominators (trusted.intmatrix.clear_row_denominators)."""
    scaled = []
    for row in rows:
        scale = lcm(*(x.denominator for x in row))
        scaled.append([int(x * scale) for x in row])
    return scaled


def determinant(rows: Rows) -> Fraction:
    work, value = [row[:] for row in rows], F(1)
    for column in range(len(work)):
        found = next((i for i in range(column, len(work)) if work[i][column]), None)
        if found is None:
            return F(0)
        if found != column:
            work[column], work[found] = work[found], work[column]
            value = -value
        value *= work[column][column]
        for i in range(column + 1, len(work)):
            factor = work[i][column] / work[column][column]
            work[i] = [x - factor * y for x, y in zip(work[i], work[column])]
    return value


def echelon_pair(rows: Rows, pivots: list[int]) -> tuple[tuple[int, ...], int]:
    """(basis_rows, modulus): rows of the row-scaled integer matrix whose minor at the pivot columns is nonsingular,
    chosen greedily, and the first prime of PRIMES that does not divide that minor's determinant."""
    integer = row_scaled(rows)
    chosen: list[int] = []
    for i in range(len(integer)):
        if len(chosen) == len(pivots):
            break
        trial = [[F(integer[k][c]) for c in pivots] for k in chosen + [i]]
        if len(rref(trial, len(pivots))[1]) == len(trial):
            chosen.append(i)
    d = determinant([[F(integer[k][c]) for c in pivots] for k in chosen])
    assert d and d.denominator == 1
    return tuple(chosen), next(m for m in PRIMES if int(d) % m)


# Brute force modulo primes: integer polynomials as lists lowest degree first, without trailing zeros.

def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, isqrt(n) + 1))


def trim(values: object) -> list[int]:
    out = list(values)
    while out and out[-1] == 0:
        out.pop()
    return out


def divide_monic(a: Sequence[int], b: Sequence[int], m: int) -> tuple[list[int], list[int]]:
    """(quotient, remainder) of a by the monic b, the coefficients modulo m."""
    work = trim(x % m for x in a)
    quotient = [0] * max(len(work) - len(b) + 1, 0)
    while len(work) >= len(b):
        top, shift = work[-1], len(work) - len(b)
        quotient[shift] = top
        for i, x in enumerate(b):
            work[shift + i] = (work[shift + i] - top * x) % m
        work = trim(work)
    return quotient, work


def monic_mod(p: Sequence[int], m: int) -> list[int]:
    inverse_lead = pow(p[-1], -1, m)
    return trim(x * inverse_lead % m for x in p)


def factor_mod(p: Sequence[int], prime: int) -> list[list[int]]:
    """The monic irreducible factors of p modulo a prime not dividing lc(p), with repetitions: trial division by every
    monic polynomial of degree 1, 2, ... in turn, a factor divided out as often as it divides."""
    rest = monic_mod(p, prime)
    factors: list[list[int]] = []
    degree = 1
    while 2 * degree <= len(rest) - 1:
        for tail in cartesian(range(prime), repeat=degree):
            candidate = list(tail) + [1]
            while len(rest) > len(candidate) - 1:
                quotient, remainder = divide_monic(rest, candidate, prime)
                if remainder:
                    break
                factors.append(candidate)
                rest = trim(quotient)
        degree += 1
    if len(rest) > 1:
        factors.append(rest)
    return factors


def coefficient_bound(p: Sequence[int]) -> int:
    """v0.7's bound of check_irreducible: |lc(p)| 2^deg(p) ceil(||p||_2); the modulus must exceed twice it."""
    total = sum(c * c for c in p)
    norm = isqrt(total)
    return abs(p[-1]) * 2 ** (len(p) - 1) * (norm if norm * norm == total else norm + 1)


def lift(p: Sequence[int], factors: list[list[int]], prime: int, exponent: int) -> tuple[tuple[int, ...], ...]:
    """The factors modulo prime lifted to prime**exponent one digit at a time: a factor g modulo prime**k becomes the
    monic g + prime**k d, d of lower degree with digits in [0, prime), that divides p / lc(p) modulo prime**(k+1);
    Hensel's lemma makes it unique, the factors being pairwise coprime modulo the prime. Every d is tried."""
    if len(factors) == 1:  # The one factor of p / lc(p): its lift is p / lc(p) itself.
        return (tuple(monic_mod(p, prime ** exponent)),)
    lifted = [list(g) for g in factors]
    for k in range(1, exponent):
        low, high = prime ** k, prime ** (k + 1)
        target = monic_mod(p, high)
        following = []
        for g in lifted:
            for digits in cartesian(range(prime), repeat=len(g) - 1):
                candidate = [c + low * d for c, d in zip(g, digits)] + [1]
                if not divide_monic(target, candidate, high)[1]:
                    following.append(candidate)
                    break
            else:
                raise AssertionError("no Hensel lift")
        lifted = following
    return tuple(tuple(g) for g in lifted)


def certificate(p: Sequence[int], prime: int | None = None, extra: int = 0) -> tuple:
    """check_irreducible's evidence for an irreducible primitive integer polynomial: ("linear",) for degree 1, else
    ("modular", prime, exponent, lifted) with the least prime that divides no leading coefficient and modulo which p
    is square-free (unless one is given), and the least exponent with prime**exponent > 2 coefficient_bound(p) (plus
    extra)."""
    if len(p) == 2:
        return ("linear",)
    if prime is None:
        prime = 2
        while not (is_prime(prime) and p[-1] % prime and len({tuple(f) for f in factor_mod(p, prime)})
                   == len(factor_mod(p, prime))):
            prime += 1
    factors = factor_mod(p, prime)
    assert len({tuple(f) for f in factors}) == len(factors), "not square-free modulo the prime"
    exponent, modulus, bound = 1, prime, coefficient_bound(p)
    while modulus <= 2 * bound:
        modulus *= prime
        exponent += 1
    exponent += extra
    return "modular", prime, exponent, lift(p, factors, prime, exponent)


# ---------------------------------------------------------------------------------------------------------------
# True claims

def key(coefficients: tuple[mf.Rational, ...]) -> tuple:
    """The order of the factorization entries: degree, then (numerator, denominator) of each coefficient."""
    return len(coefficients) - 1, tuple((c.numerator, c.denominator) for c in coefficients)


def charpoly_evidence(rows: Rows, p: list[Fraction]) -> tuple[int, ...]:
    """(1, C_1, ..., C_n): det(t I - s M) highest power first, s the lcm of every denominator: C_k = c_(n-k) s^k."""
    n = len(rows)
    s = lcm(1, *(x.denominator for row in rows for x in row))
    values = [p[n - k] * s ** k for k in range(n + 1)]
    assert all(v.denominator == 1 for v in values)
    return tuple(int(v) for v in values)


def true_claim(rows: Sequence[Sequence[object]], factors: Sequence[tuple[Sequence[int], int]] | None = None):
    """(args, value, evidence) of the true claim about the matrix of these rows. factors are the irreducible factors of
    degree >= 2 of the characteristic polynomial with their multiplicities, as the construction of the matrix gives
    them (monic integer coefficients, lowest degree first); None when the rest after the rational roots has degree at
    most 3, so that it is irreducible itself or 1."""
    rows = fractions(rows)
    p = charpoly(rows)
    roots, rest = rational_roots(p)
    if factors is None:
        assert len(rest) <= 4
        parts = [] if len(rest) == 1 else [(rest, 1)]
    else:
        parts = [([F(c) for c in q], e) for q, e in factors]
    product = [F(1)]
    for q, e in parts:
        assert not rational_roots(q)[0], "a factor with a rational root"
        for _ in range(e):
            product = poly_multiply(product, q)
    assert product == rest, (product, rest)
    entries = [((rational(-root), Q(1)), m, ("linear",)) for root, m in roots.items()]
    entries += [(tuple(rational(c) for c in q), e, certificate(primitive_integer(q))) for q, e in parts]
    entries.sort(key=lambda entry: key(entry[0]))
    spaces, echelons = [], []
    for root in sorted(roots):
        shifted = [[x - root if i == j else x for j, x in enumerate(row)] for i, row in enumerate(rows)]
        basis, pivots = nullspace(shifted)
        spaces.append(Eigenspace(rational(root), roots[root], tuple(tuple(rational(x) for x in v) for v in basis)))
        echelons.append(echelon_pair(shifted, pivots))
    value = RationalEigenvectors(tuple(spaces), tuple(rational(c) for c in rest))
    return {"M": to_matrix(rows)}, value, (charpoly_evidence(rows, p), tuple(entries), tuple(echelons))


SQRT2 = (-2, 0, 1)
CASES: dict[str, tuple[Rows, list | None]] = {
    # The card's known values, and the probe of benchmarks/probes/linalg.py.
    "symmetric": (fractions([[2, 1], [1, 2]]), []),
    "sqrt2": (fractions([[0, 2], [1, 0]]), [(SQRT2, 1)]),
    "jordan": (fractions([[1, 1], [0, 1]]), []),
    "identity": (identity(3), []),
    "probe": (fractions([[2, 1, 0], [1, 3, 1], [0, 1, 4]]), [((6, -6, 1), 1)]),
    # Special cases.
    "empty": ([], []),
    "zero": (fractions([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), []),
    "one_by_one": (fractions([[Q(-5, 7)]]), []),
    "rational": (fractions([[Q(1, 2), Q(1, 3)], [0, Q(-2, 5)]]), []),
    "key_order": (fractions([[1, 0], [0, Q(1, 2)]]), []),  # Entries (t - 1), (t - 1/2); eigenspaces 1/2, 1.
    "scaled": (fractions([[3, 0], [0, 0]]), []),
    "x4_plus_1": (companion(1, 0, 0, 0), [((1, 0, 0, 0, 1), 1)]),  # Reducible modulo every prime.
    "swinnerton_dyer": (companion(1, 0, -10, 0), [((1, 0, -10, 0, 1), 1)]),  # The minimal polynomial of sqrt2+sqrt3.
    "repeated_factor": (block_diagonal(companion(-2, 0), companion(-2, 0)), [(SQRT2, 2)]),
    "two_factors": (conjugate(fractions([[1, 1, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 0, 1, 0],
                                         [0, 1, 0, 0, 1]]),
                              block_diagonal(companion(-2, 0), companion(-3, 0), [[5]])),
                    [(SQRT2, 1), ((-3, 0, 1), 1)]),
    "cubic": (block_diagonal(companion(-2, 0, 0), [[-1]]), [((-2, 0, 0, 1), 1)]),
    "defective": (conjugate(unimodular(5, random.Random(5)), block_diagonal(jordan(2, 3), [[2]], [[-3]])), []),
    "rational_dense": (conjugate(fractions([[1, Q(1, 2), 0], [0, 1, Q(-2, 3)], [Q(1, 4), 0, 1]]),
                                 block_diagonal(jordan(Q(1, 3), 2), [[-1]])), []),
}


@functools.cache
def claim(name: str):
    rows, factors = CASES[name]
    return true_claim(rows, factors)


def large_claim():
    """A 12 x 12 rational matrix: a defective eigenvalue, a repeated full one, a non-integer one, 0, 3, and the
    residual (t^2 - 2)(t^3 - t - 1), conjugated by a unimodular matrix."""
    blocks = block_diagonal(jordan(1, 2), [[-2]], [[-2]], [[Q(1, 2)]], companion(-2, 0), companion(-1, -1, 0),
                            [[0]], [[3]])
    rows = conjugate(unimodular(12, random.Random(12), steps=30), blocks)
    return true_claim(rows, [(SQRT2, 1), ((-1, -1, 0, 1), 1)])


@functools.cache
def nilpotent_claim():
    """The 24 x 24 matrix [[0, 0], [I, 0]] of 12 x 12 blocks: p = t^24, the eigenvalue 0 with multiplicity 24 and an
    eigenspace of dimension 12, the standard vectors of columns 12, ..., 23, whose 12 pivot entries come before their
    free column."""
    return true_claim([[F(int(i >= 12 and j == i - 12)) for j in range(24)] for i in range(24)])


def raw_rational(numerator: object, denominator: object) -> mf.Rational:
    """A Rational made with object.__new__: its fields need not be in lowest terms, nor exact ints."""
    value = object.__new__(Q)
    object.__setattr__(value, "numerator", numerator)
    object.__setattr__(value, "denominator", denominator)
    return value


def forged_value(value: RationalEigenvectors, **fields: object) -> RationalEigenvectors:
    return support.forged(RationalEigenvectors, **{"eigenspaces": value.eigenspaces, "residual": value.residual,
                                                   **fields})


def forged_space(space: Eigenspace, **fields: object) -> Eigenspace:
    return support.forged(Eigenspace, **{"value": space.value, "multiplicity": space.multiplicity,
                                         "basis": space.basis, **fields})


def with_space(value: RationalEigenvectors, index: int, space: object) -> RationalEigenvectors:
    spaces = list(value.eigenspaces)
    spaces[index] = space
    return forged_value(value, eigenspaces=tuple(spaces))


def replaced(items: tuple, index: int, item: object) -> tuple:
    return items[:index] + (item,) + items[index + 1:]


def vectors(*rows: Sequence[object]) -> tuple[tuple[mf.Rational, ...], ...]:
    return tuple(tuple(Q(x) if type(x) is int else x for x in row) for row in rows)


class EigenCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsFast(self, args: object, value: object, evidence: object, operation: object = OP) -> None:
        started = time.perf_counter()
        self.assertRejects(operation, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    def assertExhausts(self, args: object, value: object, evidence: object,
                       limits: ComputationLimits | None = None) -> None:
        started = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, args, value, evidence, limits=limits)
        self.assertLess(time.perf_counter() - started, FAST)

    # True claims, each built by the test.

    def test_accepts_the_known_values_of_the_card(self):
        expected = {
            "symmetric": ([(Q(1), 1, vectors((-1, 1))), (Q(3), 1, vectors((1, 1)))], (Q(1),)),
            "sqrt2": ([], (Q(-2), Q(0), Q(1))),
            "jordan": ([(Q(1), 2, vectors((1, 0)))], (Q(1),)),
            "identity": ([(Q(1), 3, vectors((1, 0, 0), (0, 1, 0), (0, 0, 1)))], (Q(1),)),
            "probe": ([(Q(3), 1, vectors((-1, -1, 1)))], (Q(6), Q(-6), Q(1))),
        }
        for name, (spaces, residual) in expected.items():
            args, value, evidence = claim(name)
            with self.subTest(name=name):
                self.assertEqual(value, RationalEigenvectors(tuple(Eigenspace(*space) for space in spaces), residual))
                self.assertAccepts(OP, args, value, evidence)

    def test_accepts_special_cases(self):
        """The empty matrix, a zero matrix, 1 x 1, rational entries, eigenvalues 0, negative and non-integer, the
        entries of the factorization in another order than the eigenspaces, residuals that split modulo every prime,
        with a repeated factor, with two factors, of degree 3, defective and full eigenspaces, dense matrices."""
        names = ("empty", "zero", "one_by_one", "rational", "key_order", "scaled", "x4_plus_1", "swinnerton_dyer",
                 "repeated_factor", "two_factors", "cubic", "defective", "rational_dense")
        for name in names:
            args, value, evidence = claim(name)
            with self.subTest(name=name):
                self.assertAccepts(OP, args, value, evidence)
        self.assertEqual(claim("empty")[1:], (RationalEigenvectors((), (Q(1),)), ((1,), (), ())))
        _, value, evidence = claim("key_order")
        self.assertEqual([space.value for space in value.eigenspaces], [Q(1, 2), Q(1)])
        self.assertEqual([entry[0][0] for entry in evidence[1]], [Q(-1), Q(-1, 2)])
        self.assertEqual(len(claim("x4_plus_1")[2][1][0][2][3]), 2)  # Two lifted factors modulo 3^4.
        defective = claim("defective")[1].eigenspaces
        self.assertEqual([(s.value, s.multiplicity, len(s.basis)) for s in defective], [(Q(-3), 1, 1), (Q(2), 4, 2)])

    def test_accepts_seeded_matrices(self):
        """Random 3 x 3 integer matrices, and 4 x 4 ones with a planted eigenvalue conjugated by a unimodular matrix;
        a claim with one eigenvalue moved is refused."""
        rng = random.Random(20261112)
        for index in range(36):
            if index % 3:
                rows = [[F(rng.randint(-4, 4)) for _ in range(3)] for _ in range(3)]
            else:
                inner = [[F(rng.randint(-3, 3)) for _ in range(3)] for _ in range(3)]
                top = [F(rng.randint(-3, 3)) for _ in range(4)]
                rows = conjugate(unimodular(4, rng), [top] + [[F(0)] + row for row in inner])
            args, value, evidence = true_claim(rows)
            with self.subTest(index=index, rows=rows):
                self.assertAccepts(OP, args, value, evidence)
                if value.eigenspaces:
                    space = value.eigenspaces[-1]
                    moved = with_space(value, -1, forged_space(space, value=space.value + 1))
                    self.assertRejects(OP, args, moved, evidence)

    def test_accepts_a_larger_matrix_quickly(self):
        args, value, evidence = large_claim()
        self.assertEqual([(s.value, s.multiplicity, len(s.basis)) for s in value.eigenspaces],
                         [(Q(-2), 2, 2), (Q(0), 1, 1), (Q(1, 2), 1, 1), (Q(1), 2, 1), (Q(3), 1, 1)])
        started = time.perf_counter()
        self.assertAccepts(OP, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    def test_accepts_other_valid_evidence(self):
        """Evidence that the producer would not choose: another prime and a longer lifting for an irreducibility
        certificate, other rows and another order for a minor, other moduli (a composite one too)."""
        args, value, evidence = claim("sqrt2")
        q, e, _ = evidence[1][0]
        for prime, extra in ((3, 1), (5, 0), (11, 2)):
            with self.subTest(prime=prime, extra=extra):
                other = (evidence[0], ((q, e, certificate([-2, 0, 1], prime, extra)),), evidence[2])
                self.assertAccepts(OP, args, value, other)
        args, value, evidence = claim("symmetric")
        for pairs in ((((1,), PRIMES[0]), ((0,), PRIMES[1])), (((0,), 3), ((1,), 2)),
                      (((0,), 3 * 1_000_000_007), ((1,), 2**60))):
            with self.subTest(pairs=pairs):
                self.assertAccepts(OP, args, value, evidence[:2] + (pairs,))
        args, value, evidence = claim("probe")  # rank 2: the minor on rows (1, 0) has the opposite determinant.
        (rows, modulus), = evidence[2]
        self.assertAccepts(OP, args, value, evidence[:2] + (((rows[::-1], modulus),),))

    def test_caps(self):
        """The card's matrix rule: more entries than max_matrix_entries raise ResourceLimitError, whatever the claim.
        A common denominator s past max_integer_bits, and a coefficient of p past it, raise it too, for true claims
        that wider limits accept."""
        args, value, evidence = claim("symmetric")
        self.assertExhausts({"M": support.oversized_matrix()}, value, evidence)
        # s = d1 d2 has about 80,000 bits, while the evidence stays short: det(t I - s M) = t (t - d2).
        d1, d2 = 2**40_000 + 1, 2**40_000 + 3
        rows = [[F(1, d1), F(1, d2)], [F(0), F(0)]]
        wide = ComputationLimits(max_integer_bits=200_000)
        long_claim = (to_matrix(rows),
                      RationalEigenvectors((Eigenspace(Q(0), 1, vectors((Q(-d1, d2), 1))),
                                            Eigenspace(Q(1, d1), 1, vectors((1, 0)))), (Q(1),)),
                      ((1, -d2, 0), (((Q(-1, d1), Q(1)), 1, ("linear",)), ((Q(0), Q(1)), 1, ("linear",))),
                       (((0,), PRIMES[0]), ((0,), PRIMES[0]))))
        self.assertExhausts({"M": long_claim[0]}, *long_claim[1:])
        self.assertAccepts(OP, {"M": long_claim[0]}, *long_claim[1:], limits=wide)
        # s = d fits, but p = (t - 1/d)^2 has the constant term 1/d^2 of about 80,000 bits.
        d = 2**40_000 + 1
        square = (to_matrix([[F(1, d), F(0)], [F(0), F(1, d)]]),
                  RationalEigenvectors((Eigenspace(Q(1, d), 2, vectors((1, 0), (0, 1))),), (Q(1),)),
                  ((1, -2, 1), (((Q(-1, d), Q(1)), 2, ("linear",)),), (((), 2),)))
        self.assertExhausts({"M": square[0]}, *square[1:])
        self.assertAccepts(OP, {"M": square[0]}, *square[1:], limits=wide)

    # Corrupted claims: each answers False, never raising.

    def test_rejects_wrong_values(self):
        args, value, evidence = claim("symmetric")
        one, three = value.eigenspaces
        wrong = (
            RationalEigenvectors((one, Eigenspace(Q(4), 1, three.basis)), (Q(1),)),
            RationalEigenvectors((Eigenspace(Q(1), 1, vectors((1, 1))), three), (Q(1),)),
            RationalEigenvectors((Eigenspace(Q(1), 1, vectors((0, 1))), three), (Q(1),)),
            RationalEigenvectors((Eigenspace(Q(1), 2, one.basis),), (Q(1),)),
            RationalEigenvectors((Eigenspace(Q(3), 2, three.basis),), (Q(1),)),
            RationalEigenvectors((), (Q(3), Q(-4), Q(1))),
        )
        for candidate in wrong:
            with self.subTest(value=candidate):
                self.assertRejects(OP, args, candidate, evidence)
        args, value, evidence = claim("sqrt2")
        for residual in ((Q(-3), Q(0), Q(1)), (Q(2), Q(0), Q(1)), (Q(-2), Q(1), Q(1))):
            with self.subTest(residual=residual):
                self.assertRejects(OP, args, RationalEigenvectors((), residual), evidence)
        self.assertRejects(OP, args, claim("symmetric")[1], claim("symmetric")[2])
        args, value, evidence = claim("jordan")
        self.assertRejects(OP, args, RationalEigenvectors((Eigenspace(Q(1), 2, vectors((1, 0), (0, 1))),), (Q(1),)),
                           (evidence[0], evidence[1], (((), 2),)))
        self.assertRejects(OP, args, RationalEigenvectors((Eigenspace(Q(1), 2, vectors((0, 1))),), (Q(1),)), evidence)
        args, value, evidence = claim("identity")
        short = RationalEigenvectors((Eigenspace(Q(1), 3, vectors((1, 0, 0), (0, 1, 0))),), (Q(1),))
        for echelons in (evidence[2], (((0,), PRIMES[0]),), (((1,), PRIMES[0]),)):
            with self.subTest(echelons=echelons):
                self.assertRejects(OP, args, short, evidence[:2] + (echelons,))
        args, value, evidence = claim("probe")
        (space,) = value.eigenspaces
        for candidate in (RationalEigenvectors((space,), (Q(6), Q(-5), Q(1))),
                          RationalEigenvectors((Eigenspace(Q(2), 1, space.basis),), value.residual),
                          RationalEigenvectors((Eigenspace(Q(3), 1, vectors((1, -1, 1))),), value.residual)):
            with self.subTest(value=candidate):
                self.assertRejects(OP, args, candidate, evidence)
        # The same characteristic polynomial as [[2, 1], [1, 2]], other eigenvectors.
        self.assertRejects(OP, claim("symmetric")[0], *true_claim([[1, 0], [0, 3]])[1:])

    def test_rejects_noncanonical_values_that_are_mathematically_right(self):
        """Each forged value names the right eigenvalues, multiplicities and eigenspaces, or the right polynomial; only
        the canonical form that it breaks can refuse it."""
        args, value, evidence = claim("symmetric")
        one, three = value.eigenspaces
        for basis in (vectors((-2, 2)), vectors((1, -1)), vectors((Q(-1, 2), Q(1, 2)))):
            with self.subTest(basis=basis):
                self.assertRejects(OP, args, with_space(value, 0, forged_space(one, basis=basis)), evidence)
        self.assertRejects(OP, args, forged_value(value, eigenspaces=(three, one)), evidence)
        self.assertRejects(OP, args, with_space(value, 0, forged_space(one, value=raw_rational(2, 2))), evidence)
        self.assertRejects(OP, args, with_space(value, 1, forged_space(three, value=raw_rational(6, 2))), evidence)
        self.assertRejects(OP, args, with_space(value, 0, forged_space(one, basis=((raw_rational(-2, 2), Q(1)),))),
                           evidence)
        self.assertRejects(OP, args, with_space(value, 0, forged_space(one, basis=((Q(-1), raw_rational(3, 3)),))),
                           evidence)
        self.assertRejects(OP, args, with_space(value, 0, forged_space(one, multiplicity=1, value=1)), evidence)
        args, value, evidence = claim("identity")
        (space,) = value.eigenspaces
        for basis in (vectors((1, 0, 0), (1, 1, 0), (0, 0, 1)), vectors((0, 0, 1), (0, 1, 0), (1, 0, 0)),
                      vectors((1, 0, 0), (0, 1, 0), (0, 1, 1)), vectors((1, 0, 0), (0, 2, 0), (0, 0, 1))):
            with self.subTest(basis=basis):
                self.assertRejects(OP, args, with_space(value, 0, forged_space(space, basis=basis)), evidence)
        args, value, evidence = claim("defective")
        two = value.eigenspaces[1]
        first, second = two.basis
        mixed = (first, tuple(a + b for a, b in zip(first, second)))
        self.assertRejects(OP, args, with_space(value, 1, forged_space(two, basis=mixed)), evidence)
        args, value, evidence = claim("sqrt2")
        for residual in ((Q(-4), Q(0), Q(2)), (Q(-2), Q(0), Q(1), Q(0)), (raw_rational(-4, 2), Q(0), Q(1)),
                         (Q(-2), raw_rational(0, 5), Q(1)), (Q(-2), Q(0), raw_rational(2, 2)), (-2, 0, 1)):
            with self.subTest(residual=residual):
                self.assertRejects(OP, args, forged_value(value, residual=residual), evidence)
        # The whole characteristic polynomial as the residual, hiding the eigenvalue 3, with the true factorization
        # and with one that calls the cubic irreducible.
        args, value, evidence = claim("probe")
        hidden = RationalEigenvectors((), (Q(-18), Q(24), Q(-9), Q(1)))
        self.assertRejects(OP, args, hidden, evidence[:2] + ((),))
        cubic = ((Q(-18), Q(24), Q(-9), Q(1)), 1, certificate([-18, 24, -9, 1], 5))
        self.assertRejects(OP, args, hidden, (evidence[0], (cubic,), ()))

    def test_rejects_impostors_and_forged_records(self):
        args, value, evidence = claim("symmetric")
        one, three = value.eigenspaces

        class Subclass(RationalEigenvectors):
            """A subclass is another class: its module and name differ."""

        class Space(Eigenspace):
            pass

        class Tuple(tuple):
            pass

        class Integer(int):
            pass

        class Number(Q):
            pass

        candidates = (
            support.impostor(value), Subclass(value.eigenspaces, value.residual), None, (value.eigenspaces,
                                                                                         value.residual),
            forged_value(value, eigenspaces=(support.impostor(one), three)),
            forged_value(value, eigenspaces=(Space(one.value, one.multiplicity, one.basis), three)),
            support.forged(RationalEigenvectors, eigenspaces=value.eigenspaces),
            support.forged(RationalEigenvectors, residual=value.residual), support.forged(RationalEigenvectors),
            forged_value(value, eigenspaces=(support.forged(Eigenspace, value=Q(1), multiplicity=1), three)),
            forged_value(value, eigenspaces=(support.forged(Eigenspace), three)),
            with_space(value, 0, forged_space(one, multiplicity=0)),
            with_space(value, 0, forged_space(one, basis=())),
            with_space(value, 0, forged_space(one, basis=vectors((-1, 1), (-1, 1)))),
            with_space(value, 0, forged_space(one, basis=vectors((-1, 1, 0)))),
            with_space(value, 0, forged_space(one, basis=vectors((1,)))),
            forged_value(value, eigenspaces=(one, one)),
            forged_value(value, eigenspaces=(one, three, three)),
            forged_value(value, residual=()),
            forged_value(value, eigenspaces=Tuple(value.eigenspaces)),
            forged_value(value, residual=Tuple(value.residual)),
            with_space(value, 0, forged_space(one, basis=Tuple(one.basis))),
            with_space(value, 0, forged_space(one, basis=(Tuple(one.basis[0]),))),
            with_space(value, 0, forged_space(one, multiplicity=Integer(1))),
            with_space(value, 0, forged_space(one, value=Number(1))),
            with_space(value, 0, forged_space(one, basis=((Number(-1), Q(1)),))),
            with_space(value, 0, forged_space(one, value=raw_rational(1, 0))),
            with_space(value, 0, forged_space(one, value=raw_rational(-1, -1))),
            with_space(value, 0, forged_space(one, value=raw_rational(1.0, 1))),
            with_space(value, 0, forged_space(one, value=object.__new__(Q))),
        )
        for candidate in candidates:
            with self.subTest(value=candidate):
                self.assertRejects(OP, args, candidate, evidence)

    def test_rejects_inexact_numbers_and_lists(self):
        args, value, evidence = claim("symmetric")
        one, three = value.eigenspaces
        values = []
        for bad in (True, 1.0, "1", Q(1), None):
            values.append(with_space(value, 0, forged_space(one, multiplicity=bad)))
        for bad in (1.0, True, "1", 1, None):
            values.append(with_space(value, 0, forged_space(one, value=bad)))
        for bad in (-1.0, -1, "-1", None):
            values.append(with_space(value, 0, forged_space(one, basis=((bad, Q(1)),))))
        values.append(with_space(value, 0, forged_space(one, basis=((Q(-1), True),))))
        values.append(forged_value(value, residual=(1,)))
        values.append(forged_value(value, residual=(1.0,)))
        values.append(forged_value(value, residual=(True,)))
        values.append(forged_value(value, eigenspaces=list(value.eigenspaces)))
        values.append(forged_value(value, residual=[Q(1)]))
        values.append(with_space(value, 0, forged_space(one, basis=list(one.basis))))
        values.append(with_space(value, 0, forged_space(one, basis=([Q(-1), Q(1)],))))
        for candidate in values:
            with self.subTest(value=candidate):
                self.assertRejects(OP, args, candidate, evidence)
        charpoly, factorization, echelons = evidence
        (q1, e1, c1), entry3 = factorization
        bad_evidence = [
            (replaced(charpoly, 1, True), factorization, echelons), (replaced(charpoly, 1, -4.0), factorization,
                                                                     echelons),
            (replaced(charpoly, 1, Q(-4)), factorization, echelons), (replaced(charpoly, 0, True), factorization,
                                                                      echelons),
            (replaced(charpoly, 2, "3"), factorization, echelons),
            (charpoly, ((q1, True, c1), entry3), echelons), (charpoly, ((q1, 1.0, c1), entry3), echelons),
            (charpoly, (((-3, 1), e1, c1), entry3), echelons), (charpoly, (((Q(-3), 1.0), e1, c1), entry3), echelons),
            (charpoly, (((Q(-3), True), e1, c1), entry3), echelons),
            (charpoly, ((q1, e1, ["linear"]), entry3), echelons), (charpoly, ((q1, e1, "linear"), entry3), echelons),
            (charpoly, factorization, (((True,), PRIMES[0]), echelons[1])),
            (charpoly, factorization, (((0.0,), PRIMES[0]), echelons[1])),
            (charpoly, factorization, (((Q(0),), PRIMES[0]), echelons[1])),
            (charpoly, factorization, (((0,), True), echelons[1])),
            (charpoly, factorization, (((0,), float(PRIMES[1])), echelons[1])),
            (charpoly, factorization, (((0,), str(PRIMES[0])), echelons[1])),
            (charpoly, factorization, (((0,), Q(PRIMES[0])), echelons[1])),
            # Lists for tuples.
            list(evidence), (list(charpoly), factorization, echelons), (charpoly, list(factorization), echelons),
            (charpoly, ([q1, e1, c1], entry3), echelons), (charpoly, ((list(q1), e1, c1), entry3), echelons),
            (charpoly, factorization, list(echelons)), (charpoly, factorization, ([(0,), PRIMES[0]], echelons[1])),
            (charpoly, factorization, (([0], PRIMES[0]), echelons[1])),
        ]
        for candidate in bad_evidence:
            with self.subTest(evidence=candidate):
                self.assertRejects(OP, args, value, candidate)
        args, value, evidence = claim("sqrt2")
        charpoly, ((q, e, (tag, prime, exponent, lifted)),), echelons = evidence
        for bad in ((tag, float(prime), exponent, lifted), (tag, True, exponent, lifted), (tag, Q(prime), exponent,
                                                                                            lifted),
                    (tag, prime, float(exponent), lifted), (tag, prime, True, lifted),
                    (tag, prime, exponent, ((float(lifted[0][0]),) + lifted[0][1:],)),
                    (tag, prime, exponent, ((True,) + lifted[0][1:],)), (tag, prime, exponent, list(lifted)),
                    (tag, prime, exponent, (list(lifted[0]),)), [tag, prime, exponent, lifted]):
            with self.subTest(certificate=bad):
                self.assertRejects(OP, args, value, (charpoly, ((q, e, bad),), echelons))

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        args, value, evidence = claim("symmetric")
        charpoly, factorization, echelons = evidence
        (q1, e1, c1), (q3, e3, c3) = factorization
        shapes = [
            (), (charpoly,), (charpoly, factorization), evidence + ((),), None, [], {}, 0,
            (charpoly[:-1], factorization, echelons), (charpoly + (0,), factorization, echelons),
            ((2,) + charpoly[1:], factorization, echelons), ((-1,) + charpoly[1:], factorization, echelons),
            ((0,) + charpoly[1:], factorization, echelons),
            (charpoly, factorization[:1], echelons), (charpoly, factorization[::-1], echelons),
            (charpoly, factorization + factorization[:1], echelons), (charpoly, (factorization[0],) * 2, echelons),
            (charpoly, ((q1, 2, c1),), echelons), (charpoly, ((q1, e1, c1), (q3, e3, c3), ((Q(0), Q(1)), 1, c1)),
                                                   echelons),
            (charpoly, ((q1, e1), (q3, e3, c3)), echelons), (charpoly, ((q1, e1, c1, ()), (q3, e3, c3)), echelons),
            (charpoly, ((q1, e1, ()), (q3, e3, c3)), echelons), (charpoly, ((q1, e1, ("modular",)), (q3, e3, c3)),
                                                                 echelons),
            (charpoly, ((q1, e1, ("linear", "linear")), (q3, e3, c3)), echelons),
            (charpoly, ((q1, e1, ("Linear",)), (q3, e3, c3)), echelons),
            (charpoly, ((q1 + (Q(0),), e1, c1), (q3, e3, c3)), echelons),
            (charpoly, factorization, echelons[:1]), (charpoly, factorization, echelons + echelons[:1]),
            (charpoly, factorization, ()), (charpoly, factorization, (echelons[0], ((0,),))),
            (charpoly, factorization, (echelons[0], ((0,), PRIMES[0], 1))),
            (charpoly, factorization, (echelons[0], ((), PRIMES[0]))),
            (charpoly, factorization, (echelons[0], ((0, 1), PRIMES[0]))),
            (charpoly, factorization, (echelons[0], ((5,), PRIMES[0]))),
            (charpoly, factorization, (echelons[0], ((-1,), PRIMES[0]))),
            (charpoly, factorization, (echelons[0], ((0,), 1))), (charpoly, factorization, (echelons[0], ((0,), 0))),
            (charpoly, factorization, (echelons[0], ((0,), -PRIMES[0]))),
        ]
        for candidate in shapes:
            with self.subTest(evidence=candidate):
                self.assertRejects(OP, args, value, candidate)
        args, value, evidence = claim("sqrt2")
        charpoly, ((q, e, (tag, prime, exponent, lifted)),), echelons = evidence
        for bad in (("linear",), ("modular", prime, exponent), (tag, prime, exponent, lifted, ()), ("Modular", prime,
                                                                                                    exponent, lifted),
                    (tag, prime, exponent, ()), (tag, prime, exponent, lifted + lifted),
                    (tag, prime, exponent, ((lifted[0][0], 1),)), (tag, prime, exponent - 1, lifted),
                    (tag, prime, 0, lifted), (tag, 4, exponent, lifted), (tag, 1, exponent, lifted)):
            with self.subTest(certificate=bad):
                self.assertRejects(OP, args, value, (charpoly, ((q, e, bad),), echelons))
        self.assertRejects(OP, args, value, (charpoly, (), echelons))
        self.assertRejects(OP, args, value, (charpoly, ((q, e, (tag, prime, exponent, lifted)),) * 2, echelons))

    def test_rejects_forged_consistent_evidence(self):
        """Evidence that holds together but is not true of the matrix: a characteristic polynomial of another matrix,
        a reducible polynomial with an irreducibility certificate (one factor modulo 3, and the true modular
        factorization), a lifted factor of another polynomial, a minor that is singular or not a unit modulo its
        modulus, and eigenspaces that claim too much or too little with matching certificates."""
        args = claim("symmetric")[0]
        self.assertRejects(OP, args, *true_claim([[1, 0], [0, 4]])[1:])
        flip = {"M": to_matrix(fractions([[0, 1], [1, 0]]))}
        self.assertAccepts(OP, flip, *true_claim([[0, 1], [1, 0]])[1:])
        hidden = RationalEigenvectors((), (Q(-1), Q(0), Q(1)))
        charpoly = (1, 0, -1)
        for lifted in (((80, 0, 1),), ((1, 1), (80, 1))):  # t^2 - 1 modulo 3^4, as one factor and as two.
            with self.subTest(lifted=lifted):
                self.assertRejects(OP, flip, hidden, (charpoly, (((Q(-1), Q(0), Q(1)), 1, ("modular", 3, 4, lifted)),),
                                                      ()))
        args, value, evidence = claim("sqrt2")
        charpoly, ((q, e, (tag, prime, exponent, lifted)),), echelons = evidence
        for other in (((24, 0, 1),), ((25, 1, 1),)):  # t^2 + 3 and t^2 + t - 2 lifted from t^2 + 1 modulo 3.
            with self.subTest(lifted=other):
                self.assertRejects(OP, args, value, (charpoly, ((q, e, (tag, prime, exponent, other)),), echelons))
        self.assertRejects(OP, args, value, (charpoly, ((q, e, (tag, 2, 6, ((62, 0, 1),))),), echelons))
        args, value, evidence = true_claim([[1, 0], [0, 2]])
        self.assertAccepts(OP, args, value, evidence)
        self.assertEqual(evidence[2][0][0], (1,))
        self.assertRejects(OP, args, value, evidence[:2] + ((((0,), PRIMES[0]),) + evidence[2][1:],))
        args, value, evidence = claim("scaled")  # [[3, 0], [0, 0]]: for 0 the minor is (3), for 3 it is (-3).
        for modulus in (3, 6, 3 * PRIMES[2]):
            with self.subTest(modulus=modulus):
                self.assertRejects(OP, args, value, evidence[:2] + ((((0,), modulus), evidence[2][1]),))
                self.assertRejects(OP, args, value, evidence[:2] + ((evidence[2][0], ((1,), modulus)),))
        args, value, evidence = claim("jordan")
        self.assertRejects(OP, args, RationalEigenvectors((Eigenspace(Q(1), 2, vectors((1, 0), (0, 1))),), (Q(1),)),
                           evidence[:2] + ((((), 2),),))
        args, value, evidence = claim("defective")
        two = value.eigenspaces[1]
        self.assertRejects(OP, args, with_space(value, 1, forged_space(two, basis=two.basis[:1])),
                           evidence[:2] + ((evidence[2][0], (evidence[2][1][0] + (4,), PRIMES[0])),))

    def test_rejects_oversized_forgeries_fast(self):
        """Every oversized forgery answers False within FAST seconds under the default budget: none of them is a true
        claim, and the screens of the checker's docstring refuse them before the work that they would make large."""
        args, value, evidence = claim("symmetric")
        charpoly, factorization, echelons = evidence
        (q1, e1, c1), entry3 = factorization
        one, three = value.eigenspaces
        many = 100_000
        forgeries_of_evidence = [
            (replaced(charpoly, 1, GIANT), factorization, echelons),
            (replaced(charpoly, 1, BIG), factorization, echelons),
            (charpoly + (0,) * many, factorization, echelons),
            (charpoly, factorization * many, echelons),
            (charpoly, (((Q(BIG), Q(1)), e1, c1), entry3), echelons),
            (charpoly, (((Q(-1, BIG), Q(1)), e1, c1), entry3), echelons),
            (charpoly, (((Q(-1, 2**65_000 + 3), Q(1)), e1, c1), entry3), echelons),
            (charpoly, ((q1 + (Q(0),) * many + (Q(1),), e1, c1), entry3), echelons),
            (charpoly, factorization, (((0,) * many, PRIMES[0]), echelons[1])),
            (charpoly, factorization, (((0,), 2**100_000), echelons[1])),
            (charpoly, factorization, (((0,), 2**61), echelons[1])),
            (charpoly, factorization, (((0,), GIANT), echelons[1])),
        ]
        for index, candidate in enumerate(forgeries_of_evidence):  # str() of a giant int would raise ValueError.
            with self.subTest(evidence=index):
                self.assertRejectsFast(args, value, candidate)
        forged_values = [
            forged_value(value, residual=(Q(0),) * many + (Q(1),)),
            forged_value(value, eigenspaces=value.eigenspaces * many),
            with_space(value, 0, forged_space(one, basis=((Q(0),) * many + (Q(1),),))),
            with_space(value, 0, forged_space(one, value=Q(GIANT))),
            with_space(value, 0, forged_space(one, value=Q(1, GIANT))),
            with_space(value, 0, forged_space(one, multiplicity=10**100)),
            with_space(value, 0, forged_space(one, basis=((Q(GIANT), Q(1)),))),
            forged_value(value, residual=(Q(GIANT),)),
            with_space(value, 0, forged_space(one, basis=vectors((-1, 1)) * many)),
        ]
        for index, candidate in enumerate(forged_values):
            with self.subTest(value=index):
                self.assertRejectsFast(args, candidate, evidence)
        # Long Rationals within max_integer_bits where the matrix allows only short ones: in the basis (as numerator and
        # as denominator, both past the Hadamard bound of the minors) and in the residual (past Gauss and Mignotte).
        for index, basis in enumerate((((Q(-BIG), Q(1)),), ((Q(1, BIG), Q(1)),))):
            with self.subTest(basis=index):
                self.assertRejectsFast(args, with_space(value, 0, forged_space(one, basis=basis)), evidence)
        args, value, evidence = claim("sqrt2")
        charpoly, ((q, e, (tag, prime, exponent, lifted)),), echelons = evidence
        self.assertRejectsFast(args, RationalEigenvectors((), (Q(-BIG), Q(0), Q(1))), evidence)
        self.assertRejectsFast(args, RationalEigenvectors((), (Q(-1, BIG), Q(0), Q(1))), evidence)
        certificates = (
            (tag, 2**127 - 1, exponent, lifted), (tag, 2**61 - 1, 2, lifted), (tag, GIANT, 1, lifted),
            (tag, 10**9 + 7, 1, ((10**9 + 5, 0, 1),)), (tag, prime, 10**6, lifted), (tag, prime, 20_000, lifted),
            (tag, prime, 200, lifted), (tag, prime, exponent, lifted * many),
            (tag, prime, exponent, ((GIANT, 0, 1),)), (tag, prime, exponent, ((25,) + (0,) * many + (1,),)),
        )
        for index, bad in enumerate(certificates):
            with self.subTest(certificate=index):
                self.assertRejectsFast(args, value, (charpoly, ((q, e, bad),), echelons))

    def test_rejects_many_long_entries_fast(self):
        """Many long Rationals within max_integer_bits: their lowest terms are tested only after the screens that bound
        them by the matrix, so the gcds of a forgery cannot exhaust the budget (144 gcds of 65,000-bit integers would
        pass max_work)."""
        args, value, evidence = nilpotent_claim()
        self.assertAccepts(OP, args, value, evidence)
        (space,) = value.eigenspaces
        self.assertEqual((space.value, space.multiplicity, len(space.basis)), (Q(0), 24, 12))
        long = Q(BIG - 2, BIG)  # Both integers of about 65,000 bits, in lowest terms.
        for entry in (long, Q(BIG), Q(1, BIG)):
            basis = tuple((entry,) * 12 + vector[12:] for vector in space.basis)
            with self.subTest(entry=entry.denominator.bit_length()):
                self.assertRejectsFast(args, with_space(value, 0, forged_space(space, basis=basis)), evidence)
        shorter = forged_space(space, multiplicity=1, basis=space.basis[:1])
        residual = (long,) * 23 + (Q(1),)
        self.assertRejectsFast(args, forged_value(value, eigenspaces=(shorter,), residual=residual), evidence)

    def test_certificate_modulus_against_the_integer_bound(self):
        """check_irreducible raises ResourceLimitError for a modulus prime**exponent longer than max_integer_bits and
        for exponent * (bitlen(prime) - 1) past it: there the checker answers False, and it accepts every modulus within
        the bound, also one with exponent * bitlen(prime) past it. Under max_integer_bits=5, 3**3 = 27 and 5**2 = 25
        have 5 bits, 3**4 = 81 has 7."""
        args, value, evidence = claim("sqrt2")
        charpoly, ((q, e, _),), echelons = evidence
        narrow = ComputationLimits(max_integer_bits=5)
        three, five = certificate([-2, 0, 1], 3), certificate([-2, 0, 1], 5)
        self.assertEqual((three, five), (("modular", 3, 3, ((25, 0, 1),)), ("modular", 5, 2, ((23, 0, 1),))))
        for good in (three, five):
            with self.subTest(certificate=good):
                self.assertAccepts(OP, args, value, (charpoly, ((q, e, good),), echelons), limits=narrow)
                self.assertAccepts(OP, args, value, (charpoly, ((q, e, good),), echelons))
        for bad in (certificate([-2, 0, 1], 3, 1), ("modular", 3, 6, ((727, 0, 1),)), ("modular", 2, 6, ((62, 0, 1),))):
            with self.subTest(certificate=bad):
                self.assertRejects(OP, args, value, (charpoly, ((q, e, bad),), echelons), limits=narrow)
        self.assertAccepts(OP, args, value, (charpoly, ((q, e, certificate([-2, 0, 1], 3, 1)),), echelons))

    def test_irreducibility_is_checked_last(self):
        """check_factorization, whose irreducibility certificates are the one check whose work can grow exponentially,
        runs only for a factorization of p that matches the value in every other respect: each corruption below is
        refused before it, and its verdict is the checker's verdict."""
        checker = support.checker(family)
        symmetric_args, symmetric, symmetric_evidence = claim("symmetric")
        one, three = symmetric.eigenspaces
        sqrt2_args, sqrt2, sqrt2_evidence = claim("sqrt2")
        charpoly, factorization, _ = sqrt2_evidence
        t2_minus_3 = ((Q(-3), Q(0), Q(1)), 1, certificate([-3, 0, 1]))
        jordan_args, jordan, jordan_evidence = claim("jordan")
        doubled = (jordan_evidence[0], (((Q(-1), Q(1)), 1, ("linear",)),) * 2, jordan_evidence[2])  # t - 1 twice.
        diagonal_args, diagonal, diagonal_evidence = true_claim([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
        low, high = diagonal.eigenspaces
        swapped = RationalEigenvectors((Eigenspace(Q(1), 1, low.basis[:1]), Eigenspace(Q(2), 2, high.basis)), (Q(1),))
        scaled_args, scaled, scaled_evidence = claim("scaled")
        corrupted = (
            (symmetric_args, RationalEigenvectors((one, Eigenspace(Q(4), 1, three.basis)), (Q(1),)),
             symmetric_evidence),
            (symmetric_args, with_space(symmetric, 0, forged_space(one, basis=vectors((1, 1)))), symmetric_evidence),
            (symmetric_args, forged_value(symmetric, eigenspaces=(three, one)), symmetric_evidence),
            (symmetric_args, symmetric, ((1, -4, 4),) + symmetric_evidence[1:]),
            (diagonal_args, swapped, diagonal_evidence),
            (sqrt2_args, RationalEigenvectors((), (Q(-3), Q(0), Q(1))), sqrt2_evidence),
            (sqrt2_args, RationalEigenvectors((), (Q(-3), Q(0), Q(1))), (charpoly, (t2_minus_3,), ())),
            (sqrt2_args, forged_value(sqrt2, residual=(Q(-2), raw_rational(0, 5), Q(1))), sqrt2_evidence),
            (jordan_args, jordan, doubled),
            (scaled_args, scaled, scaled_evidence[:2] + ((((0,), 3), scaled_evidence[2][1]),)),
        )
        with patch.object(checker, "check_factorization", side_effect=AssertionError("check_factorization ran")):
            for index, (args, value, evidence) in enumerate(corrupted):
                with self.subTest(claim=index):
                    self.assertRejects(OP, args, value, evidence)
        for verdict in (True, False):
            with patch.object(checker, "check_factorization", return_value=verdict) as mocked:
                self.assertIs(support.verdict(family, OP, sqrt2_args, sqrt2, sqrt2_evidence), verdict)
                mocked.assert_called_once_with((Q(-2), Q(0), Q(1)), factorization)

    def test_rejects_invalid_arguments(self):
        args, value, evidence = claim("symmetric")
        matrix = args["M"]
        impostor = dataclasses.make_dataclass("Matrix", ["rows", "ncols"], frozen=True)

        def forged_matrix(rows: object, ncols: object) -> mf.Matrix:
            return support.forged(mf.Matrix, rows=rows, ncols=ncols)

        bad = (
            {"M": mf.Matrix(((2, 1, 0), (1, 2, 0)))}, {"M": matrix.rows}, {"M": impostor(matrix.rows, 2)},
            {"A": matrix}, {"M": matrix, "k": 1}, {}, None, [("M", matrix)], {"M": None},
            {"M": forged_matrix(list(matrix.rows), 2)}, {"M": forged_matrix(matrix.rows, 3)},
            {"M": forged_matrix(matrix.rows, True)}, {"M": forged_matrix(((2.0, Q(1)), (Q(1), Q(2))), 2)},
            {"M": forged_matrix(((2, 1), (1, 2)), 2)}, {"M": forged_matrix(((raw_rational(2, 0), Q(1)),
                                                                             (Q(1), Q(2))), 2)},
            {"M": forged_matrix(((raw_rational(-2, -1), Q(1)), (Q(1), Q(2))), 2)},
            {"M": forged_matrix(((Q(GIANT), Q(1)), (Q(1), Q(2))), 2)}, {"M": forged_matrix(([Q(2), Q(1)],
                                                                                            (Q(1), Q(2))), 2)},
            {"M": support.forged(mf.Matrix, rows=matrix.rows)},
            # The value 2 written 4/2, which no Matrix holds: s, and so the evidence, would depend on how it is written.
            {"M": forged_matrix(((raw_rational(4, 2), Q(1)), (Q(1), Q(2))), 2)},
            {"M": forged_matrix(((Q(2), Q(1)), (Q(1), raw_rational(0, 3))), 2)},
        )
        for index, candidate in enumerate(bad):  # str() of a giant int would raise ValueError.
            with self.subTest(args=index):
                self.assertRejects(OP, candidate, value, evidence)
        self.assertAccepts(OP, {"M": forged_matrix(((Q(2), Q(1)), (Q(1), Q(2))), 2)}, value, evidence)

    def test_rejects_wrong_operation_names(self):
        args, value, evidence = claim("symmetric")
        for operation in ("eigenvalues", "Eigenvectors", "eigenvectors ", "", None, 1, "matrix_power",
                          "real_eigenvalues", b"eigenvectors"):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)

    # The records and their rendering, which the family module defines in this step.

    def test_records_refuse_noncanonical_forms(self):
        one = Eigenspace(Q(1), 1, vectors((-1, 1)))
        three = Eigenspace(Q(3), 1, vectors((1, 1)))
        for fields in ((1, 1, vectors((-1, 1))), (Q(1), 0, vectors((-1, 1))), (Q(1), True, vectors((-1, 1))),
                       (Q(1), 1, ()), (Q(1), 1, vectors((-1, 1), (0, 1))), (Q(1), 2, vectors((1, 0), (1, 1)))
                       , (Q(1), 2, vectors((0, 1), (1, 0))), (Q(1), 1, vectors((-2, 2))), (Q(1), 1, vectors((0, 0))),
                       (Q(1), 1, ((Q(-1), 1),)), (Q(1), 1, [(Q(-1), Q(1))]), (Q(1), 2, vectors((1, 0), (0, 1, 0))),
                       (Q(1), 1, ((),)), (1.0, 1, vectors((-1, 1)))):
            with self.subTest(fields=fields):
                with self.assertRaises(InvalidInput):
                    Eigenspace(*fields)
        for fields in (((three, one), (Q(1),)), ((one, one), (Q(1),)), ((one,), (Q(1),)), ((one, three), (Q(2),)),
                       ((), (Q(-2), Q(1))), ((), (Q(0), Q(0), Q(1))), ((), (Q(-4), Q(0), Q(2))), ((), ()),
                       ((), (1,)), ([one, three], (Q(1),)), ((one, three), [Q(1)]), ((one.basis,), (Q(1),))):
            with self.subTest(fields=fields):
                with self.assertRaises(InvalidInput):
                    RationalEigenvectors(*fields)
        self.assertEqual(RationalEigenvectors((), (Q(1),)).residual, (Q(1),))

    def test_rendering(self):
        texts = {
            "symmetric": ("1 (multiplicity 1): [(-1, 1)]; 3 (multiplicity 1): [(1, 1)]",
                          r"1\ \left(\text{multiplicity }1\right)\colon\ \left[\left(-1, 1\right)\right];\quad "
                          r"3\ \left(\text{multiplicity }1\right)\colon\ \left[\left(1, 1\right)\right]"),
            "sqrt2": ("no rational eigenvalue; residual x^2 - 2",
                      r"\text{no rational eigenvalue};\quad \text{residual }x^{2} - 2"),
            "probe": ("3 (multiplicity 1): [(-1, -1, 1)]; residual x^2 - 6*x + 6",
                      r"3\ \left(\text{multiplicity }1\right)\colon\ \left[\left(-1, -1, 1\right)\right];\quad "
                      r"\text{residual }x^{2} - 6\, x + 6"),
            "empty": ("no rational eigenvalue", r"\text{no rational eigenvalue}"),
            "rational": ("-2/5 (multiplicity 1): [(-10/27, 1)]; 1/2 (multiplicity 1): [(1, 0)]",
                         r"-\frac{2}{5}\ \left(\text{multiplicity }1\right)\colon\ \left[\left(-\frac{10}{27}, "
                         r"1\right)\right];\quad \frac{1}{2}\ \left(\text{multiplicity }1\right)\colon\ "
                         r"\left[\left(1, 0\right)\right]"),
        }
        for name, (text, latex) in texts.items():
            value = claim(name)[1]
            with self.subTest(name=name):
                self.assertEqual(mf.render_text(value), text)
                self.assertEqual(mf.to_latex(value), latex)
        quartic = RationalEigenvectors((Eigenspace(Q(1, 2), 1, vectors((1,) + (0,) * 4)),),
                                       (Q(1, 3), Q(-1), Q(0), Q(-5, 2), Q(1)))
        self.assertEqual(mf.render_text(quartic),
                         "1/2 (multiplicity 1): [(1, 0, 0, 0, 0)]; residual x^4 - 5/2*x^3 - x + 1/3")
