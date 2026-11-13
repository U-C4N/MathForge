"""The eigen family of mathforge.linalg: eigenvectors.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register linalg eigen

Every expected value comes from the card or from a reference of this module, written here and slow on purpose, never
from the producer: the characteristic polynomial by interpolation (det(t I - M) at t = 0, ..., n by Gaussian
elimination over fractions.Fraction, then the Vandermonde system solved over Fractions; neither Berkowitz, as the
producer computes it, nor the checker's Hessenberg form nor the checker test's Faddeev-LeVerrier recurrence), the
rational eigenvalues and their multiplicities by the rational root theorem, the residual by synthetic division, and each
eigenspace by Gauss-Jordan elimination over Fractions. The factorization evidence is compared with
polynomial_exact.factor_with_evidence, v0.7's factorization that the checker's layout names as its direct source, and
each echelon certificate is tested for what it claims (a minor that is a unit modulo a prime below 2**61).
tests/v09/linalg/test_eigen_checker.py tests the checker on its own. Run this module by name with

    python -m unittest tests.v09.linalg.test_eigen
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from fractions import Fraction
from math import gcd, isqrt, lcm
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.linalg import eigen as family
from mathforge.areas.linalg.eigen import Eigenspace, RationalEigenvectors
from mathforge.contracts import CheckStatus, ErrorInfo, ExecutionStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import Budget, ComputationLimits, ResourceLimitError, computation
from mathforge.polynomial import Polynomial
from mathforge.polynomial_exact import factor_with_evidence

from .. import support

Q = mf.Rational
F = Fraction
OP = "eigenvectors"
FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
WIDE = ComputationLimits(max_integer_bits=300_000, max_matrix_entries=1_000_000)  # For matrices past the limits.
STAND_INS = (0.5, True, "1", None, (1, 2), [1], ((1, 0), (0, 1)), [[1, 0], [0, 1]], Q(1), 1)
X = mf.Symbol("x")
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


def matrix(rows: Sequence[Sequence[object]]) -> mf.Matrix:
    """The Matrix of rows of exact numbers (ints, Fractions or Rationals); [] is the 0 x 0 matrix."""
    if not rows:
        return mf.Matrix((), ncols=0)
    return mf.Matrix(tuple(tuple(rational(fraction(x)) for x in row) for row in rows))


def wide(rows: Sequence[Sequence[object]]) -> mf.Matrix:
    """matrix(rows), built inside a budget that admits integers and sizes past the default limits."""
    with computation(WIDE, fresh=True):
        return matrix(rows)


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
    """s b s^-1: the eigenvalues of b, with other eigenvectors."""
    return multiply(multiply(s, b), inverse(s))


def unimodular(n: int, rng: random.Random, steps: int = 10) -> Rows:
    """A product of elementary row additions: an integer matrix of determinant 1."""
    s = identity(n)
    for _ in range(steps):
        i, j = rng.sample(range(n), 2)
        factor = rng.choice((-2, -1, 1, 2))
        s[i] = [x + factor * y for x, y in zip(s[i], s[j])]
    return s


def block_diagonal(*blocks: Sequence[Sequence[object]]) -> Rows:
    n = sum(len(block) for block in blocks)
    rows = [[F(0)] * n for _ in range(n)]
    offset = 0
    for block in blocks:
        for i, row in enumerate(block):
            for j, x in enumerate(row):
                rows[offset + i][offset + j] = fraction(x)
        offset += len(block)
    return rows


def companion(*tail: object) -> Rows:
    """The companion matrix of t^n + tail[n-1] t^(n-1) + ... + tail[0], its characteristic polynomial."""
    n = len(tail)
    rows = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        if i:
            rows[i][i - 1] = F(1)
        rows[i][n - 1] = -fraction(tail[i])
    return rows


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


def solve(system: Rows, rhs: list[Fraction]) -> list[Fraction]:
    """The solution of a nonsingular square system over Fractions, by Gauss-Jordan elimination."""
    n = len(system)
    work = [row[:] + [value] for row, value in zip(system, rhs)]
    for column in range(n):
        pivot = next(i for i in range(column, n) if work[i][column])
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [x / scale for x in work[column]]
        for i in range(n):
            if i != column and work[i][column]:
                factor = work[i][column]
                work[i] = [x - factor * y for x, y in zip(work[i], work[column])]
    return [row[n] for row in work]


def charpoly(rows: Rows) -> list[Fraction]:
    """det(t I - M), lowest degree first, by interpolation: its values at t = 0, ..., n, each a determinant over
    Fractions, and the Vandermonde system of those n + 1 points."""
    n = len(rows)
    points = list(range(n + 1))
    values = [determinant([[F(t) * (i == j) - x for j, x in enumerate(row)] for i, row in enumerate(rows)])
              for t in points]
    return solve([[F(t) ** j for j in range(n + 1)] for t in points], values)


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


def divisors(n: int) -> list[int]:
    small = [d for d in range(1, isqrt(abs(n)) + 1) if n % d == 0]
    return sorted(set(small + [abs(n) // d for d in small]))


def rational_roots(p: Sequence[Fraction]) -> tuple[dict[Fraction, int], list[Fraction]]:
    """Every rational root of p with its multiplicity, and the rest p / prod (t - root)^multiplicity: the rational
    root theorem on the integer form of the rest, each candidate a/b tried exactly and divided out while it is a
    root."""
    roots: dict[Fraction, int] = {}
    rest = [F(c) for c in p]
    while len(rest) > 1 and rest[0] == 0:
        roots[F(0)] = roots.get(F(0), 0) + 1
        rest = rest[1:]
    if len(rest) > 1:
        scale = lcm(*(c.denominator for c in rest))
        integer = [int(c * scale) for c in rest]
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


def nullspace(rows: Rows) -> list[tuple[Fraction, ...]]:
    """The canonical nullspace basis read from the RREF R: 1 at each free column f, 0 at the other free columns and
    -R[i][f] at the pivot of row i."""
    n = len(rows)
    reduced, pivots = rref(rows, n)
    basis = []
    for free in range(n):
        if free not in pivots:
            vector = [F(0)] * n
            vector[free] = F(1)
            for row, pivot in zip(reduced, pivots):
                vector[pivot] = -row[free]
            basis.append(tuple(vector))
    return basis


def reference(rows: Sequence[Sequence[object]]) -> RationalEigenvectors:
    """The value of eigenvectors(M), from the references alone."""
    rows = fractions(rows)
    roots, rest = rational_roots(charpoly(rows))
    spaces = []
    for root in sorted(roots):
        shifted = [[x - root if i == j else x for j, x in enumerate(row)] for i, row in enumerate(rows)]
        basis = tuple(tuple(rational(x) for x in vector) for vector in nullspace(shifted))
        spaces.append(Eigenspace(rational(root), roots[root], basis))
    return RationalEigenvectors(tuple(spaces), tuple(rational(c) for c in rest))


def charpoly_evidence(rows: Sequence[Sequence[object]]) -> tuple[int, ...]:
    """(1, C_1, ..., C_n): det(t I - s M) highest power first, s the lcm of every denominator: C_k = c_(n-k) s^k."""
    rows = fractions(rows)
    n = len(rows)
    s = lcm(1, *(x.denominator for row in rows for x in row))
    p = charpoly(rows) if n else [F(1)]
    values = [p[n - k] * s ** k for k in range(n + 1)]
    assert all(v.denominator == 1 for v in values)
    return tuple(int(v) for v in values)


def v07_factorization(rows: Sequence[Sequence[object]]) -> tuple:
    """The factorization entries (coefficients, multiplicity, irreducibility) of the characteristic polynomial, read
    from polynomial_exact.factor_with_evidence of the reference polynomial."""
    p = charpoly(fractions(rows))
    if len(p) == 1:
        return ()
    value, evidence = factor_with_evidence(Polynomial(tuple(rational(c) for c in p), X))
    return tuple((factor.polynomial.coefficients, factor.multiplicity, certificate)
                 for factor, certificate in zip(value.factors, evidence[1]))


def is_prime(n: int) -> bool:
    """Miller-Rabin with the first twelve prime bases: deterministic below 3.3 * 10**24."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in bases:
        return True
    if any(n % b == 0 for b in bases):
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for b in bases:
        x = pow(b, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def row_scaled(rows: Rows) -> list[list[int]]:
    return [[int(x * lcm(*(y.denominator for y in row))) for x in row] for row in rows]


def echelon_holds(rows: Sequence[Sequence[object]], value: RationalEigenvectors, echelons: tuple) -> bool:
    """Whether each (basis_rows, modulus) names rows of the row-scaled M - lambda I whose minor at the pivot columns
    (the columns without a free column of the basis) is a unit modulo a prime below 2**61, in the order of the
    eigenspaces."""
    rows = fractions(rows)
    n = len(rows)
    if len(echelons) != len(value.eigenspaces):
        return False
    for space, (basis_rows, modulus) in zip(value.eigenspaces, echelons):
        root = fraction(space.value)
        shifted = row_scaled([[x - root if i == j else x for j, x in enumerate(row)] for i, row in enumerate(rows)])
        free = {max(j for j, x in enumerate(vector) if x.numerator) for vector in space.basis}
        pivots = [j for j in range(n) if j not in free]
        if type(basis_rows) is not tuple or len(basis_rows) != len(pivots) or len(set(basis_rows)) != len(pivots):
            return False
        if not (type(modulus) is int and 2 <= modulus < 2 ** 61 and is_prime(modulus)):
            return False
        minor = determinant([[F(shifted[i][j]) for j in pivots] for i in basis_rows])
        if int(minor) % modulus == 0:
            return False
    return True


def lift_passes(factorization: tuple, max_bits: int) -> bool:
    """Whether a square-free part of degree >= 2 (the product of the factors of one multiplicity) has a primitive
    integer form f with 2 |lc(f)| 2**deg(f) ceil(||f||_2) of more than max_bits bits: v0.7's Hensel lifting of that
    part then needs a modulus past the limit."""
    parts: dict[int, list[Fraction]] = {}
    for coefficients, multiplicity, _ in factorization:
        q = [fraction(c) for c in coefficients]
        part = parts.get(multiplicity, [F(1)])
        product = [F(0)] * (len(part) + len(q) - 1)
        for i, x in enumerate(part):
            for j, y in enumerate(q):
                product[i + j] += x * y
        parts[multiplicity] = product
    for part in parts.values():
        if len(part) < 3:
            continue
        scale = lcm(*(c.denominator for c in part))
        integer = [int(c * scale) for c in part]
        content = gcd(*integer)
        integer = [c // content for c in integer]
        total = sum(c * c for c in integer)
        norm = isqrt(total) + (isqrt(total) ** 2 < total)
        if (2 * abs(integer[-1]) * 2 ** (len(integer) - 1) * norm).bit_length() > max_bits:
            return True
    return False


def forged_matrix(rows: object, ncols: object) -> mf.Matrix:
    return support.forged(mf.Matrix, rows=rows, ncols=ncols)


def forged_rational(numerator: object, denominator: object) -> mf.Rational:
    return support.forged(Q, numerator=numerator, denominator=denominator)


def forged_value(value: RationalEigenvectors, **fields: object) -> RationalEigenvectors:
    return support.forged(RationalEigenvectors, **{"eigenspaces": value.eigenspaces, "residual": value.residual,
                                                   **fields})


def forged_space(space: Eigenspace, **fields: object) -> Eigenspace:
    return support.forged(Eigenspace, **{"value": space.value, "multiplicity": space.multiplicity,
                                         "basis": space.basis, **fields})


def vectors(*rows: Sequence[object]) -> tuple[tuple[mf.Rational, ...], ...]:
    return tuple(tuple(rational(fraction(x)) for x in row) for row in rows)


def items(value: object) -> int:
    """The items of a value as Budget.inspect counts them, in a budget wide enough for any of them."""
    counter = Budget(ComputationLimits(max_evidence_items=10 ** 9, max_integer_bits=10 ** 7, max_depth=1_000))
    counter.inspect(value, evidence=True)
    return counter.evidence_count


class SubMatrix(mf.Matrix):
    """A Matrix subclass with the right rows: no transport value."""


# The card's known values, and the probe of benchmarks/probes/linalg.py: (rows, eigenspaces, residual).
KNOWN = (
    ([[2, 1], [1, 2]], ((1, 1, ((-1, 1),)), (3, 1, ((1, 1),))), (1,)),
    ([[0, 2], [1, 0]], (), (-2, 0, 1)),
    ([[1, 1], [0, 1]], ((1, 2, ((1, 0),)),), (1,)),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ((1, 3, ((1, 0, 0), (0, 1, 0), (0, 0, 1))),), (1,)),
    ([[2, 1, 0], [1, 3, 1], [0, 1, 4]], ((3, 1, ((-1, -1, 1),)),), (6, -6, 1)),
)


def known_value(spaces: tuple, residual: tuple) -> RationalEigenvectors:
    return RationalEigenvectors(tuple(Eigenspace(Q(value), multiplicity, vectors(*basis))
                                      for value, multiplicity, basis in spaces), tuple(Q(c) for c in residual))


# More matrices with values from the references: special cases and every shape of the factorization.
SPECIAL = {
    "zero": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "one_by_one": [[F(-5, 7)]],
    "rational": [[F(1, 2), F(1, 3)], [0, F(-2, 5)]],
    "key_order": [[1, 0], [0, F(1, 2)]],  # Factorization entries t - 1, t - 1/2; eigenspaces 1/2, 1.
    "x4_plus_1": companion(1, 0, 0, 0),  # Irreducible, reducible modulo every prime.
    "swinnerton_dyer": companion(1, 0, -10, 0),
    "repeated_factor": block_diagonal(companion(-2, 0), companion(-2, 0)),
    "split_part": block_diagonal(companion(-2, 0), companion(-3, 0)),  # (t^2 - 2)(t^2 - 3): one square-free part.
    "residual_fraction": [[0, F(1, 2)], [1, 0]],  # t^2 - 1/2.
    "defective": conjugate(unimodular(5, random.Random(5)),
                           block_diagonal([[2, 1, 0], [0, 2, 1], [0, 0, 2]], [[2]], [[-3]])),
    "rational_dense": conjugate(fractions([[1, F(1, 2), 0], [0, 1, F(-2, 3)], [F(1, 4), 0, 1]]),
                                block_diagonal([[F(1, 3), 1], [0, F(1, 3)]], [[-1]])),
    "hilbert": [[F(1, i + j + 1) for j in range(4)] for i in range(4)],
    "mixed": block_diagonal([[1, 1], [0, 1]], [[F(-1, 2)]], companion(-1, -1, 0), [[0]]),
}


class EigenTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: RationalEigenvectors | None = None) -> None:
        """A completed v6 Result, of that value when one is given, with its four checks, which mf.verify accepts."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertIs(type(result.value), RationalEigenvectors)
        if value is not None:
            self.assertEqual(result.value, value)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertAgrees(self, rows: Sequence[Sequence[object]]) -> RationalEigenvectors:
        """produce() answers the reference value with the evidence of the layout, and the checker accepts both;
        returns the value."""
        args = {"M": matrix(rows)}
        value, evidence = family.produce(OP, dict(args))
        self.assertEqual(value, reference(rows))
        self.assertIs(type(evidence), tuple)
        self.assertEqual(len(evidence), 3)
        self.assertEqual(evidence[0], charpoly_evidence(rows))
        self.assertEqual(evidence[1], v07_factorization(rows))
        self.assertTrue(echelon_holds(rows, value, evidence[2]), evidence[2])
        self.assertIs(support.verdict(family, OP, args, value, evidence), True)
        return value

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for rows, spaces, residual in KNOWN:
            with self.subTest(rows=rows):
                expected = known_value(spaces, residual)
                self.assertEqual(reference(rows), expected)  # The card, by reference.
                result = support.run(OP, M=matrix(rows))
                self.assertVerified(result, expected)
                self.roundtrip(result)
        # The layout of the evidence on two of them, and on the 0 x 0 matrix.
        result = support.run(OP, M=matrix([[2, 1], [1, 2]]))
        charpoly_part, factorization, echelons = result.steps[0].outputs[1]
        self.assertEqual(charpoly_part, (1, -4, 3))
        self.assertEqual(factorization, (((Q(-3), Q(1)), 1, ("linear",)), ((Q(-1), Q(1)), 1, ("linear",))))
        self.assertEqual([len(rows) for rows, _ in echelons], [1, 1])
        result = support.run(OP, M=matrix([[0, 2], [1, 0]]))
        self.assertEqual(result.steps[0].outputs[1][:2], ((1, 0, -2), (((Q(-2), Q(0), Q(1)), 1,
                                                                         ("modular", 3, 3, ((25, 0, 1),))),)))
        result = support.run(OP, M=mf.Matrix(()))
        self.assertVerified(result, RationalEigenvectors((), (Q(1),)))
        self.assertEqual(result.steps[0].outputs[1], ((1,), (), ()))
        self.roundtrip(result)
        for name, rows in SPECIAL.items():
            with self.subTest(name=name):
                result = support.run(OP, M=matrix(rows))
                self.assertVerified(result, reference(rows))
                self.roundtrip(result)
        # Some shapes of those values, from the references.
        self.assertEqual([s.value for s in reference(SPECIAL["key_order"]).eigenspaces], [Q(1, 2), Q(1)])
        defective = reference(SPECIAL["defective"]).eigenspaces
        self.assertEqual([(s.value, s.multiplicity, len(s.basis)) for s in defective], [(Q(-3), 1, 1), (Q(2), 4, 2)])
        self.assertEqual(reference(SPECIAL["residual_fraction"]).residual, (Q(-1, 2), Q(0), Q(1)))
        self.assertEqual(reference(SPECIAL["repeated_factor"]).residual, (Q(4), Q(0), Q(-4), Q(0), Q(1)))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error; so are a
        non-square M and a Matrix forged without its constructor (an entry that is no canonical Rational, among them
        one out of lowest terms, which the checker refuses). produce() tests the types itself too."""
        for item in STAND_INS:
            with self.subTest(M=item):
                self.assertInvalidInput(support.run(OP, M=item))
        flat, tall = matrix([[1, 2, 3], [4, 5, 6]]), matrix([[1, 2], [3, 4], [5, 6]])
        for M in (flat, tall, mf.Matrix((), ncols=3), mf.Matrix(((), ())), matrix([[1, 2, 3]])):
            with self.subTest(shape=M.shape):
                result = support.run(OP, M=M)
                self.assertInvalidInput(result)
                self.assertIn("square matrix M", result.error.message)
        one, nought = Q(1), Q(0)
        forged = {"a float entry": forged_matrix(((one, 0.5), (nought, one)), 2),
                  "an int entry": forged_matrix(((one, 1), (nought, one)), 2),
                  "a bool entry": forged_matrix(((one, True), (nought, one)), 2),
                  "a str entry": forged_matrix(((one, "1"), (nought, one)), 2),
                  "None as an entry": forged_matrix(((one, None), (nought, one)), 2),
                  "a Fraction entry": forged_matrix(((one, F(1, 2)), (nought, one)), 2),
                  "a zero denominator": forged_matrix(((one, forged_rational(1, 0)), (nought, one)), 2),
                  "a negative denominator": forged_matrix(((one, forged_rational(1, -2)), (nought, one)), 2),
                  "a bool numerator": forged_matrix(((one, forged_rational(True, 2)), (nought, one)), 2),
                  "4/2": forged_matrix(((one, forged_rational(4, 2)), (nought, one)), 2),
                  "0/2": forged_matrix(((one, forged_rational(0, 2)), (nought, one)), 2),
                  "rows as a list": forged_matrix([(one, nought), (nought, one)], 2),
                  "a row as a list": forged_matrix(([one, nought], (nought, one)), 2),
                  "a ragged row": forged_matrix(((one, nought), (nought,)), 2),
                  "ncols None": forged_matrix(((one, nought), (nought, one)), None),
                  "ncols 2.0": forged_matrix(((one, nought), (nought, one)), 2.0),
                  "ncols True": forged_matrix(((one,),), True)}
        for label, M in forged.items():
            with self.subTest(forged=label):
                self.assertInvalidInput(support.run(OP, M=M))
        M = matrix([[2, 1], [1, 2]])
        # A Rational forged with a float or without a denominator: operations_v6.run's inspection of the request reads
        # the bit length of both fields before produce() runs (that is run()'s, not this family's, error), so only
        # produce() is asked here, which refuses them as invalid input.
        unreadable = [forged_matrix(((one, forged_rational(1, 2.0)), (nought, one)), 2),
                      forged_matrix(((one, support.forged(Q, numerator=1)), (nought, one)), 2)]
        stand_ins = list(forged.values()) + unreadable + list(STAND_INS) + [
            SubMatrix(((2, 1), (1, 2))), M.rows, support.impostor(M), support.forged(mf.Matrix, rows=M.rows),
            support.forged(mf.Matrix, ncols=2), flat]
        for item in stand_ins:
            with self.subTest(path="produce", M=item), self.assertRaises(InvalidInput):
                family.produce(OP, {"M": item})
        for arguments in ({"M": M, "k": 2}, {}, {"A": M}, {"m": M}, tuple({"M": M}.items()), MappingProxyType({"M": M}),
                          None, [M]):
            with self.subTest(path="produce", arguments=arguments), self.assertRaises(InvalidInput):
                family.produce(OP, arguments)
        for operation in ("Eigenvectors", "eigenvector", "eigenvectors ", "eigenvalues", "", None, (OP,)):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, {"M": M})
        with self.assertRaises(mf.OperationError) as raised:  # The value function raises the structured path's error.
            support.value(OP, M=flat)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536, a 65 x 65 matrix and max_work=1 exceed the budget; so do a common denominator and
        a coefficient of the characteristic polynomial past max_integer_bits (the checker's refusals), long entries
        whose long-integer work passes max_work, a matrix whose Berkowitz ticks alone pass it, and a Result that the
        verifier would not admit under max_evidence_items. Each request fails at once."""
        start = time.perf_counter()
        self.assertResourceLimit(support.run(OP, M=wide([[support.HUGE]])))
        self.assertResourceLimit(support.run(OP, M=wide([[1, F(1, support.HUGE)], [0, 1]])))
        oversized = support.oversized_matrix()
        self.assertResourceLimit(support.run(OP, M=oversized))
        with self.assertRaises(ResourceLimitError):
            family.produce(OP, {"M": oversized})
        with computation(ComputationLimits(max_matrix_entries=65 * 66), fresh=True):
            oblong = mf.Matrix(tuple(tuple((i + j) % 5 for j in range(66)) for i in range(65)))
        self.assertResourceLimit(support.run(OP, M=oblong))  # Not square, but too large first.
        with self.assertRaises(ResourceLimitError):
            family.produce(OP, {"M": oblong})
        for rows in ([[2, 1], [1, 2]], [], [[0, 2], [1, 0]], [[F(1, 2)]]):
            with self.subTest(max_work=1, rows=rows):
                self.assertResourceLimit(support.run(OP, M=matrix(rows), limits=ComputationLimits(max_work=1)))
        # A common denominator past max_integer_bits: s = d1 d2 of about 80,000 bits.
        d1, d2 = 2 ** 40_000 + 1, 2 ** 40_000 + 3
        self.assertResourceLimit(support.run(OP, M=matrix([[F(1, d1), 0], [0, F(1, d2)]])))
        # Under 64-bit integers: a coefficient of p past the limit (2**80), and one of the evidence (C_2 = 2**80).
        small = ComputationLimits(max_integer_bits=64)
        self.assertResourceLimit(support.run(OP, M=matrix([[2 ** 40, 1], [0, 2 ** 40]]), limits=small))
        self.assertResourceLimit(support.run(OP, M=matrix([[F(1, 2 ** 40), 0], [0, F(1, 2 ** 40)]]), limits=small))
        with computation(small, fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OP, {"M": matrix([[2 ** 40, 1], [0, 2 ** 40]])})
        # Long entries: a 30 x 30 matrix of 2,000-bit entries is refused before Berkowitz runs.
        rng = random.Random(20261113)
        long_rows = [[rng.getrandbits(2_000) for _ in range(30)] for _ in range(30)]
        self.assertResourceLimit(support.run(OP, M=matrix(long_rows)))
        # Berkowitz's ticks: 53 x 53 passes max_work whatever its entries; the 52 x 52 identity fits.
        self.assertResourceLimit(support.run(OP, M=mf.Matrix.identity(53)))
        self.assertResourceLimit(support.run(OP, M=mf.Matrix.identity(8), limits=ComputationLimits(max_work=100)))
        self.assertLess(time.perf_counter() - start, 3 * FAST)
        result = support.run(OP, M=mf.Matrix.identity(52))
        self.assertVerified(result)
        self.assertEqual([(s.value, s.multiplicity, len(s.basis)) for s in result.value.eigenspaces], [(Q(1), 52, 52)])

    def test_result_items_fit_max_evidence_items(self):
        """The verifier walks the whole stored Result: _RESULT_ITEMS for its own fields and checks, the request twice,
        the value twice and the evidence once. produce() refuses a Result past max_evidence_items, and a Result at the
        limit verifies under it."""
        for rows in ([[2, 1], [1, 2]], [], [[0, 2], [1, 0]], SPECIAL["mixed"], SPECIAL["rational"]):
            with self.subTest(rows=rows):
                result = support.run(OP, M=matrix(rows))
                self.assertVerified(result)
                request = family._REQUEST_ITEMS + items(matrix(rows))
                self.assertEqual(items(result.request), request)
                total = (family._RESULT_ITEMS + 2 * request + 2 * items(result.value)
                         + items(result.steps[0].outputs[1]))
                self.assertEqual(items(result), total)
                self.assertEqual(items(mf.loads(mf.dumps(result))), total)
                limits = ComputationLimits(max_evidence_items=total)
                at_limit = support.run(OP, M=matrix(rows), limits=limits)
                self.assertVerified(at_limit, result.value)
                self.assertTrue(mf.verify(at_limit, limits=limits).verified)
                below = support.run(OP, M=matrix(rows), limits=ComputationLimits(max_evidence_items=total - 1))
                self.assertResourceLimit(below)
                self.assertIn("evidence_count", below.error.message)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: other
        rows, another order of the rows and other moduli for a minor, the lifted factors of a whole square-free part
        for each of its factors instead of each factor's own lift data; and values built by hand with the Rational
        constructor, equal to the producer's."""
        rows = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]  # The eigenvalue 3: M - 3I has rank 2.
        args = {"M": matrix(rows)}
        value, evidence = family.produce(OP, dict(args))
        (basis_rows, modulus), = evidence[2]
        for pair in ((basis_rows[::-1], modulus), (basis_rows, 2 ** 31 - 1), (basis_rows, 1_000_000_007),
                     ((1, 2), modulus), ((2, 0), 998_244_353)):
            with self.subTest(pair=pair):
                self.assertTrue(echelon_holds(rows, value, (pair,)))
                self.assertAccepts(OP, args, value, evidence[:2] + ((pair,),))
        # (t^2 - 2)(t^2 - 3) is one square-free part: factor_square_free certifies both factors with its lifted
        # factors, produce() with each factor's own lift data, as factor_with_evidence does. Both are accepted.
        from mathforge.kernels import upoly as polynomials
        rows = SPECIAL["split_part"]
        args = {"M": matrix(rows)}
        value, evidence = family.produce(OP, dict(args))
        with computation(fresh=True):
            parts = polynomials.factor_square_free((6, 0, -5, 0, 1))
        by_factor = {factor: certificate for factor, certificate in parts}
        other = tuple((q, e, by_factor[tuple(c.numerator for c in q)]) for q, e, _ in evidence[1])
        self.assertNotEqual(other, evidence[1])
        self.assertAccepts(OP, args, value, (evidence[0], other, evidence[2]))
        self.assertAccepts(OP, args, value, evidence)
        # Values built by hand from the references, with the public constructors.
        for name in ("rational", "defective", "mixed", "hilbert"):
            rows = SPECIAL[name]
            args = {"M": matrix(rows)}
            value, evidence = family.produce(OP, dict(args))
            built = reference(rows)
            with self.subTest(name=name):
                self.assertEqual(built, value)
                self.assertAccepts(OP, args, built, evidence)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        rows = SPECIAL["mixed"]  # Eigenvalues -1/2, 0, 1 (multiplicity 2, one vector) and the residual t^3 - t - 1.
        args = {"M": matrix(rows)}
        result = support.run(OP, M=args["M"])
        self.assertVerified(result)
        value, evidence = result.value, result.steps[0].outputs[1]
        self.assertAccepts(OP, args, value, evidence)
        spaces = value.eigenspaces
        first = spaces[0]

        def with_space(index: int, space: object) -> RationalEigenvectors:
            return forged_value(value, eigenspaces=spaces[:index] + (space,) + spaces[index + 1:])

        vector = first.basis[0]
        doubled = tuple(Q(2) * x for x in vector)
        cases = {
            "an eigenvalue moved": with_space(0, forged_space(first, value=first.value + 1)),
            "a multiplicity plus one": with_space(0, forged_space(first, multiplicity=first.multiplicity + 1)),
            "a doubled basis vector": with_space(0, forged_space(first, basis=(doubled,))),
            "a missing eigenspace": forged_value(value, eigenspaces=spaces[1:]),
            "the eigenspaces reversed": forged_value(value, eigenspaces=spaces[::-1]),
            "the residual moved": forged_value(value, residual=(Q(2),) + value.residual[1:]),
            "the residual 1": forged_value(value, residual=(Q(1),)),
            "an impostor value": support.impostor(value),
            "an impostor eigenspace": with_space(0, support.impostor(first)),
            "a multiplicity True": with_space(0, forged_space(first, multiplicity=True)),
            "a multiplicity 1.0": with_space(0, forged_space(first, multiplicity=1.0)),
            "a multiplicity '1'": with_space(0, forged_space(first, multiplicity="1")),
            "an int eigenvalue": with_space(0, forged_space(first, value=first.value.numerator)),
            "a zero written 0/2": with_space(0, forged_space(first, basis=((forged_rational(0, 2),) + vector[1:],))),
            "a basis as a list": with_space(0, forged_space(first, basis=list(first.basis))),
            "eigenspaces as a list": forged_value(value, eigenspaces=list(spaces)),
            "a residual as a list": forged_value(value, residual=list(value.residual)),
            "None": None,
        }
        self.assertEqual(vector[0], Q(0))
        for label, item in cases.items():
            with self.subTest(case=label):
                self.assertRejects(OP, args, item, evidence)
        evidence_cases = {"None": None, "a list": list(evidence), "two parts": evidence[:2],
                          "an extra part": evidence + ((),), "no echelons": evidence[:2] + ((),),
                          "a charpoly as a list": (list(evidence[0]),) + evidence[1:],
                          "a bool in the charpoly": ((True,) + evidence[0][1:],) + evidence[1:],
                          "a float in the charpoly": ((1.0,) + evidence[0][1:],) + evidence[1:],
                          "a str modulus": evidence[:2] + (((evidence[2][0][0], str(evidence[2][0][1])),)
                                                           + evidence[2][1:],),
                          "oversized": (0,) * 100_000}
        for label, item in evidence_cases.items():
            with self.subTest(evidence=label):
                self.assertRejects(OP, args, value, item, may_exhaust=label == "oversized")
        for name in ("Eigenvectors", "eigenvalues", "matrix_power", "", None):
            with self.subTest(operation=name):
                self.assertRejects(name, args, value, evidence)
        self.assertRejects(OP, {"M": matrix([[1, 1], [0, 1]])}, value, evidence)
        report = support.statuses(mf.verify(support.tampered(result, cases["the residual moved"])))
        self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
        self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        matrices = [matrix(rows) for rows, _, _ in KNOWN] + [mf.Matrix(()), matrix(SPECIAL["mixed"]),
                                                              matrix(SPECIAL["rational_dense"])]
        results = [support.run(OP, M=M) for M in matrices]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for M, result, text in zip(matrices, results, texts):
                with self.subTest(shape=M.shape):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            failed = support.run(OP, M=matrix([[2, 1], [1, 2]]))  # The patch is live: a new run reaches it.
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: both records, an
        integer and a rational eigenvalue, a residual with integer and with rational coefficients, no eigenvalue, and
        the 0 x 0 matrix."""
        rendered = (
            ([[2, 1], [1, 2]], "1 (multiplicity 1): [(-1, 1)]; 3 (multiplicity 1): [(1, 1)]",
             r"1\ \left(\text{multiplicity }1\right)\colon\ \left[\left(-1, 1\right)\right];\quad "
             r"3\ \left(\text{multiplicity }1\right)\colon\ \left[\left(1, 1\right)\right]"),
            ([[0, 2], [1, 0]], "no rational eigenvalue; residual x^2 - 2",
             r"\text{no rational eigenvalue};\quad \text{residual }x^{2} - 2"),
            ([[2, 1, 0], [1, 3, 1], [0, 1, 4]], "3 (multiplicity 1): [(-1, -1, 1)]; residual x^2 - 6*x + 6",
             r"3\ \left(\text{multiplicity }1\right)\colon\ \left[\left(-1, -1, 1\right)\right];\quad "
             r"\text{residual }x^{2} - 6\, x + 6"),
            (SPECIAL["rational"], "-2/5 (multiplicity 1): [(-10/27, 1)]; 1/2 (multiplicity 1): [(1, 0)]",
             r"-\frac{2}{5}\ \left(\text{multiplicity }1\right)\colon\ \left[\left(-\frac{10}{27}, 1\right)\right];"
             r"\quad \frac{1}{2}\ \left(\text{multiplicity }1\right)\colon\ \left[\left(1, 0\right)\right]"),
            (SPECIAL["residual_fraction"], "no rational eigenvalue; residual x^2 - 1/2",
             r"\text{no rational eigenvalue};\quad \text{residual }x^{2} - \frac{1}{2}"),
            (SPECIAL["x4_plus_1"], "no rational eigenvalue; residual x^4 + 1",
             r"\text{no rational eigenvalue};\quad \text{residual }x^{4} + 1"),
            ([[1, 1], [0, 1]], "1 (multiplicity 2): [(1, 0)]",
             r"1\ \left(\text{multiplicity }2\right)\colon\ \left[\left(1, 0\right)\right]"),
            ([], "no rational eigenvalue", r"\text{no rational eigenvalue}"),
        )
        for rows, text, latex in rendered:
            with self.subTest(rows=rows):
                result = support.run(OP, M=matrix(rows))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        space = support.run(OP, M=matrix(SPECIAL["defective"])).value.eigenspaces[1]
        self.assertEqual(mf.render_text(space), "2 (multiplicity 4): [" + ", ".join(
            "(" + ", ".join(mf.render_text(x) for x in vector) + ")" for vector in space.basis) + "]")
        self.assertTrue(mf.to_latex(space).startswith(r"2\ \left(\text{multiplicity }4\right)\colon\ \left["))
        data = mf.to_data(support.run(OP, M=matrix([[1, 1], [0, 1]])).value)["data"]
        self.assertEqual(data["type"], "v6_rational_eigenvectors")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261114)
        cases = [matrix(rows) for rows, _, _ in KNOWN] + [mf.Matrix(()), matrix(SPECIAL["mixed"])]
        for _ in range(25):
            n = rng.randrange(1, 5)
            cases.append(matrix([[F(rng.randrange(-4, 5), rng.choice((1, 1, 2, 3))) for _ in range(n)]
                                 for _ in range(n)]))
        for M in cases:
            with self.subTest(rows=M.rows):
                expected = support.run(OP, M=M).value
                self.assertIsNotNone(expected)
                found = support.lean_value(OP, M=M)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(OP, M=M), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        symmetric = matrix([[2, 1], [1, 2]])
        result = self.answer(support.call(OP, {"M": [[2, 1], [1, 2]]}))
        self.assertEqual(result, support.run(OP, M=symmetric))
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(result.value, known_value(*KNOWN[0][1:]))
        rational_rows = SPECIAL["rational"]
        for arguments, M in (({"M": [["1/2", "1/3"], [0, "-2/5"]]}, matrix(rational_rows)), ({"M": []}, mf.Matrix(())),
                             ({"M": [[0, 2], [1, 0]]}, matrix([[0, 2], [1, 0]]))):
            with self.subTest(arguments=arguments):
                result = self.answer(support.call(OP, arguments))
                self.assertEqual(result, support.run(OP, M=M))
                self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call_values(OP, M=matrix(SPECIAL["mixed"])))
        self.assertEqual(result, support.run(OP, M=matrix(SPECIAL["mixed"])))
        self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call(OP, {"M": [[1, 2, 3]]}), "invalid_input")
        self.assertIn("square matrix M", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references on 240 seeded matrices (n from 1 to 5: small integer and rational
        entries, triangular matrices with planted and repeated eigenvalues conjugated by unimodular matrices, blocks
        with irreducible companions), its factorization evidence equals factor_with_evidence's, its echelon
        certificates hold, and the checker accepts every answer."""
        rng = random.Random(20261115)
        irreducible = ((-2, 0), (1, 1), (-3, 0, 0), (-1, -1, 0), (F(-1, 2), 0), (1, 0, 0, 0))

        def entry() -> Fraction:
            return F(rng.randrange(-6, 7), rng.choice((1, 1, 1, 2, 3, 4)))

        count = 0
        shapes = {"defective": 0, "residual": 0, "no eigenvalue": 0, "rational eigenvalue": 0}
        while count < 240:
            kind = count % 4
            if kind == 0:
                n = rng.randrange(1, 5)
                rows = [[F(rng.randrange(-4, 5)) for _ in range(n)] for _ in range(n)]
            elif kind == 1:
                n = rng.randrange(1, 4)
                rows = [[entry() for _ in range(n)] for _ in range(n)]
            elif kind == 2:  # Upper triangular with eigenvalues from a short list: repeated, often defective.
                n = rng.randrange(2, 6)
                diagonal = [rng.choice((F(0), F(1), F(-1), F(1, 2), F(2))) for _ in range(n)]
                upper = [[diagonal[i] if i == j else (F(rng.randrange(-1, 2)) if j > i else F(0)) for j in range(n)]
                         for i in range(n)]
                rows = conjugate(unimodular(n, rng, steps=6), upper)
            else:  # An irreducible companion block beside a few rational eigenvalues.
                block = companion(*rng.choice(irreducible))
                rest = [[[rng.choice((F(1), F(-2), F(1, 3)))]] for _ in range(rng.randrange(0, 3))]
                rows = block_diagonal(block, *rest)
                n = len(rows)
                rows = conjugate(unimodular(n, rng, steps=5), rows)
            with self.subTest(index=count, rows=rows):
                value = self.assertAgrees(rows)
                shapes["defective"] += any(len(s.basis) < s.multiplicity for s in value.eigenspaces)
                shapes["residual"] += len(value.residual) > 1
                shapes["no eigenvalue"] += not value.eigenspaces
                shapes["rational eigenvalue"] += any(s.value.denominator > 1 for s in value.eigenspaces)
            count += 1
        self.assertTrue(all(tally >= 20 for tally in shapes.values()), shapes)

    # -----------------------------------------------------------------------------------------------------------
    # The producer's limits

    def test_producer_refuses_what_the_checker_refuses(self):
        """Under 64-bit integers, produce() raises ResourceLimitError on the requests that the checker refuses (a
        common denominator, a coefficient of p, a row scale of M - lambda I past the limit) and on those whose true
        claim has a longer integer, which the checker answers False; every other request gives the value of the
        default limits. One refusal is the factorization's own (lift_passes), for claims that the checker accepts.
        The entries are too long for the rational root theorem of the references (its candidates
        are the divisors of a constant term of a hundred bits and more): the checker under the default limits vouches
        for the value of the default limits instead."""
        limits = ComputationLimits(max_integer_bits=64)
        rng = random.Random(20261116)
        tally = {"refused": 0, "computed": 0, "lift": 0}
        cases = [[[F(1, 2 ** 33 - 9), 0], [0, F(1, 2 ** 33 + 9)]], [[2 ** 40, 1], [0, 2 ** 40]],
                 [[2 ** 30, 1], [1, 2 ** 30]], [[1, 2 ** 60], [0, 2]], [[F(1, 2 ** 62), 0], [0, 1]]]
        for _ in range(40):
            n = rng.randrange(1, 4)
            bits = rng.choice((4, 16, 24, 40))
            cases.append([[F(rng.randrange(-2 ** bits, 2 ** bits), rng.choice((1, 3, 2 ** rng.randrange(0, 30))))
                           for _ in range(n)] for _ in range(n)])
        for rows in cases:
            args = {"M": matrix(rows)}
            with self.subTest(rows=rows):
                try:
                    with computation(limits, fresh=True):
                        produced = family.produce(OP, dict(args))
                except ResourceLimitError:
                    produced = None
                wide_value, wide_evidence = family.produce(OP, dict(args))  # Under the default limits.
                self.assertIs(support.verdict(family, OP, args, wide_value, wide_evidence), True)
                try:
                    answer = support.verdict(family, OP, args, wide_value, wide_evidence, limits=limits)
                except ResourceLimitError:
                    answer = None
                if produced is None and answer is True:
                    # The one refusal of a claim that the checker accepts: v0.7's factorization lifts each square-free
                    # part of degree >= 2 to a modulus above twice its coefficient bound, which may pass the limit
                    # although every factor is linear: (t - 2**-62)(t - 1) needs a modulus of about 2**127.
                    self.assertTrue(lift_passes(wide_evidence[1], limits.max_integer_bits), rows)
                    tally["lift"] += 1
                elif produced is None:
                    tally["refused"] += 1
                else:
                    self.assertEqual(produced, (wide_value, wide_evidence))
                    self.assertIs(answer, True)
                    tally["computed"] += 1
        self.assertTrue(tally["refused"] >= 10 and tally["computed"] >= 10 and tally["lift"] >= 1, tally)

    def test_tight_integer_limits_keep_claims_that_fit(self):
        """Regressions: produce() refused claims whose every integer fits max_integer_bits, which the checker accepts
        under those limits. The kernel's echelon modulus is a word prime of 61 bits, which the verifier refuses as an
        int of the evidence under max_integer_bits < 61: a prime of at most that many bits replaces it. The square-free
        decomposition's pseudo-remainder sequence refused a square-free characteristic polynomial: a coprimality test
        modulo a word prime decides that case. The kernel's fraction-free elimination refused a pivot (the whole minor)
        past the limit although the reduced rows are 0s and 1s: Gauss-Jordan elimination over Q takes over."""
        a, b = 2_959_912_470, 2_189_200_054  # 32 bits each: the kernel's second pivot a b has 62.
        cases = (
            (32, [[1, 0], [0, 2]]),
            (32, [[-2, F(2, 3)], [0, 1]]),
            (8, [[1, 1], [0, 1]]),
            (48, [[-110, -1, 3, -2], [0, F(-183, 256), 1, -3], [0, 0, 0, -1], [0, 0, 0, 160]]),
            (32, [[-134, -166, 0, 0], [208, -253, 0, 0], [0, 0, 0, -55], [0, 0, 0, 218]]),
            (48, [[0, 0, a], [b, 0, -3_897_297_295], [0, 0, 0]]),
        )
        for bits, rows in cases:
            limits = ComputationLimits(max_integer_bits=bits)
            with self.subTest(bits=bits, rows=rows):
                result = support.run(OP, M=matrix(rows), limits=limits)
                self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
                self.assertEqual(result.value, reference(rows))
                self.assertTrue(mf.verify(result, limits=limits).verified)
                value, evidence = result.value, result.steps[0].outputs[1]
                self.assertTrue(echelon_holds(rows, value, evidence[2]), evidence[2])
                self.assertTrue(all(modulus.bit_length() <= bits for _, modulus in evidence[2]), evidence[2])
                self.assertEqual(evidence[1], v07_factorization(rows))
        # The fallback elimination over Q chooses the kernel's pivot rows and, under the default limits, its modulus:
        # with the kernel made to refuse every elimination, produce() answers the same value and evidence.
        from mathforge.kernels import linalg as kernel

        def refuse(*_: object) -> None:
            raise ResourceLimitError("Computation limit exceeded: integer bits")

        for name, rows in SPECIAL.items():
            with self.subTest(fallback=name):
                expected = family.produce(OP, {"M": matrix(rows)})
                with patch.object(kernel, "echelon", refuse):
                    self.assertEqual(family.produce(OP, {"M": matrix(rows)}), expected)
        # Under the default limits the kernel's word prime stays.
        evidence = support.run(OP, M=matrix([[-2, F(2, 3)], [0, 1]])).steps[0].outputs[1]
        self.assertTrue(all(modulus > 2 ** 60 for _, modulus in evidence[2]), evidence[2])
        # The square-free shortcut answers what square_free_integer answers, with and without repeated factors.
        from mathforge.kernels import upoly as polynomials
        rng = random.Random(20261118)
        for _ in range(60):
            f: tuple[int, ...] = (1,)
            for _ in range(rng.randrange(1, 4)):
                factor = tuple(rng.randrange(-9, 10) for _ in range(rng.randrange(1, 4))) + (rng.randrange(1, 4),)
                for _ in range(rng.randrange(1, 3)):
                    f = tuple(sum(f[i] * factor[k - i] for i in range(len(f)) if 0 <= k - i < len(factor))
                              for k in range(len(f) + len(factor) - 1))
            content = gcd(*f)
            f = tuple(c // content for c in f)
            if len(f) < 2:
                continue
            with self.subTest(f=f), computation(fresh=True):
                self.assertEqual(family._square_free_parts(f), polynomials.square_free_integer(f))

    def test_work_is_charged(self):
        """Every valid request charges its budget; a request with long entries charges their long products before
        the kernels run, so that its budget refuses it quickly; the records' own budget charges stay within it."""
        for rows in ([[2, 1], [1, 2]], [], SPECIAL["mixed"]):
            with computation(fresh=True) as budget:
                family.produce(OP, {"M": matrix(rows)})
            self.assertGreaterEqual(budget.work, 2)
        rng = random.Random(20261117)
        rows = [[rng.getrandbits(1_500) for _ in range(12)] for _ in range(12)]
        with computation(fresh=True) as budget:
            family._charge_berkowitz(budget, tuple(tuple(row) for row in rows), 12)
        self.assertGreater(budget.work, 0)
        with computation(fresh=True) as budget:
            family._charge_berkowitz(budget, ((1, 2), (3, 4)), 2)
        self.assertEqual(budget.work, 0)  # Short entries cost nothing beyond the kernel's ticks.
        started = time.perf_counter()
        self.assertResourceLimit(support.run(OP, M=matrix(rows), limits=ComputationLimits(max_work=50_000)))
        self.assertLess(time.perf_counter() - started, FAST)
        # Regression: Hensel lifting goes one p-adic digit at a time and trusted.upoly charges each of its products one
        # unit whatever its size; its long products are charged before it starts. A 10 x 10 matrix of 1,000-bit entries
        # (over 4,000 digits modulo 5) and a 6 x 6 one of 6,000-bit entries took 4 and 40 seconds without that charge.
        for size, bits in ((10, 1_000), (6, 6_000)):
            long_rows = [[rng.getrandbits(bits) for _ in range(size)] for _ in range(size)]
            started = time.perf_counter()
            with self.subTest(size=size, bits=bits):
                self.assertResourceLimit(support.run(OP, M=matrix(long_rows)))
                self.assertLess(time.perf_counter() - started, FAST)
        with computation(fresh=True) as budget:
            family._charge_lifting(budget, (rng.getrandbits(3_000), 0, 0, 0, 0, 0, 1))  # x^6 + c: 3,000-bit c.
        self.assertGreater(budget.work, 100_000)
        with computation(fresh=True) as budget:
            family._charge_lifting(budget, (-2, 0, 1))
        self.assertLess(budget.work, 100)  # Short coefficients: good_prime's own ticks, no long products.
        # A record's fields are already exact: dataclasses.replace with a non-canonical field is refused.
        value = support.run(OP, M=matrix([[2, 1], [1, 2]])).value
        with self.assertRaises(InvalidInput):
            dataclasses.replace(value.eigenspaces[0], multiplicity=0)
