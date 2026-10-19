"""Independent checker of a determinant: a cofactor expansion for small matrices, else a Hadamard bound and residues
modulo primes; never an elimination over the integers.

With A_int = diag(s) A (mathforge.trusted.intmatrix), det(A) = det(A_int) / prod(s), and a claimed value stands for
D = value * prod(s), which must be an integer. Up to 4 x 4, D is compared with det(A_int) by Laplace expansion. Larger
matrices need D**2 <= H2 = prod_i |row_i(A_int)|**2 (Hadamard: det(A_int)**2 <= H2) and D = det(A_int) modulo primes of
trusted.primes whose product P satisfies P**2 > 4 H2: then |D - det(A_int)| <= 2 sqrt(H2) < P, so D = det(A_int). The
residues come from this package's own lazy elimination, modulo the product of those primes at once (a composite
modulus whose pivots must be units), or, when a pivot is not a unit there, modulo each prime alone.
"""

from __future__ import annotations

from math import prod
from operator import mul

from mathforge.model import Rational
from mathforge.trusted.intmatrix import clear_row_denominators
from mathforge.trusted.primes import moduli

from . import square_rows
from ._modular import det_mod


def _laplace(rows):
    """det of a square integer matrix of size at most 4 by cofactor expansion along the first row."""
    n = len(rows)
    if n == 0:
        return 1
    if n == 1:
        return rows[0][0]
    if n == 2:
        (a, b), (c, d) = rows
        return a * d - b * c
    if n == 3:
        (a, b, c), (d, e, f), (g, h, i) = rows
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    total = 0
    for column, entry in enumerate(rows[0]):
        if entry:
            minor = [row[:column] + row[column + 1:] for row in rows[1:]]
            term = entry * _laplace(minor)
            total = total - term if column & 1 else total + term
    return total


def check_integer_determinant(int_rows, target) -> bool:
    """Whether target = det of the square integer matrix int_rows (a tuple of int tuples)."""
    if len(int_rows) <= 4:
        return target == _laplace(int_rows)
    bound = 1
    for row in int_rows:
        bound *= sum(map(mul, row, row))
    if target * target > bound:
        return False
    if not bound:  # A zero row: the determinant is 0, and so is target by the bound.
        return True
    limit = 4 * bound
    primes, product = [], 1
    for modulus in moduli():
        primes.append(modulus)
        product *= modulus
        if product * product > limit:
            break
    residue = det_mod(int_rows, product)
    if residue is not None:
        return residue == target % product
    for modulus in primes:  # A pivot was not a unit modulo the product: one prime at a time.
        if det_mod(int_rows, modulus) != target % modulus:
            return False
    return True


def check_determinant(A, value) -> bool:
    """Whether value = det(A), for a square rational matrix A (a Matrix or a tuple of row tuples)."""
    rows = square_rows(A)
    if rows is None or type(value) is not Rational:
        return False
    int_rows, scales = clear_row_denominators(rows)
    scale = prod(scales)
    if scale % value.denominator:
        return False
    return check_integer_determinant(int_rows, value.numerator * (scale // value.denominator))
