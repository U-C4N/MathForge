"""Seeded matrices and a Fraction oracle for the linear algebra kernel tests."""

from fractions import Fraction
import random

import mathforge as mf

R = mf.Rational


def rational_matrix(rng, m, n, *, denominators=1, zeros=0.0, low=-9, high=9):
    """An m x n Matrix of seeded rationals p/q, q <= denominators, about `zeros` of the entries zero."""
    rows = []
    for _ in range(m):
        rows.append(tuple(R(0) if rng.random() < zeros else R(rng.randint(low, high), rng.randint(1, denominators))
                          for _ in range(n)))
    return mf.Matrix(tuple(rows), ncols=n)


def with_dependent_rows(rng, matrix):
    """The matrix with a row replaced by a combination of two others (when it has three rows or more)."""
    rows = list(matrix.rows)
    if len(rows) >= 3:
        a, b = rng.randint(-3, 3), rng.randint(-3, 3)
        rows[-1] = tuple(a * x + b * y for x, y in zip(rows[0], rows[1]))
    return mf.Matrix(tuple(rows), ncols=matrix.ncols)


def matrices(seed, count, *, square=False, max_size=8):
    """count seeded matrices: shapes 0..max_size, every rank, denominators up to 12, zero rows and columns."""
    rng = random.Random(seed)
    result = []
    for index in range(count):
        m = rng.randint(0, max_size)
        n = m if square else rng.randint(0, max_size)
        matrix = rational_matrix(rng, m, n, denominators=rng.choice((1, 1, 3, 12)),
                                 zeros=rng.choice((0.0, 0.0, 0.3, 0.7)))
        if index % 5 == 1:
            matrix = with_dependent_rows(rng, matrix)
        if index % 7 == 2 and m:
            rows = list(matrix.rows)
            rows[rng.randrange(m)] = (R(0),) * n
            matrix = mf.Matrix(tuple(rows), ncols=n)
        result.append(matrix)
    return result


def fraction_rows(matrix):
    return [[Fraction(v.numerator, v.denominator) for v in row] for row in matrix.rows]


def fraction_det(matrix):
    """det by Fraction Gaussian elimination: the tests' oracle."""
    rows = fraction_rows(matrix)
    n = len(rows)
    det = Fraction(1)
    for k in range(n):
        pivot = next((i for i in range(k, n) if rows[i][k]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != k:
            rows[k], rows[pivot] = rows[pivot], rows[k]
            det = -det
        det *= rows[k][k]
        for i in range(k + 1, n):
            factor = rows[i][k] / rows[k][k]
            rows[i] = [a - factor * b for a, b in zip(rows[i], rows[k])]
    return det


def as_fraction(value):
    return Fraction(value.numerator, value.denominator)
