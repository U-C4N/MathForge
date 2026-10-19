"""Independent checker of a characteristic polynomial: a coefficient bound and a Hessenberg reduction modulo primes,
never Berkowitz.

For a square integer matrix A (A_int scaled by one common denominator), the coefficient c_k of t**(n-k) in
det(t I - A) is (-1)**k times the sum of the principal k x k minors, each at most the product of the norms of its rows
(Hadamard), so |c_k| <= e_k(rho) <= M = prod_i (1 + rho_i) with rho_i >= |row_i(A)|. A claim (1, c_1, ..., c_n) is
accepted when every |c_k| <= M and the claim agrees modulo primes of trusted.primes whose product exceeds 2 M with the
characteristic polynomial modulo each prime, computed by reducing A to upper Hessenberg form by similarity and
expanding the Hessenberg determinant along its last column.
"""

from __future__ import annotations

from math import isqrt
from operator import mul

from mathforge.limits import current_budget
from mathforge.trusted.primes import moduli


def _charpoly_mod(int_rows, modulus):
    """det(t I - A) modulo a prime, lowest power first."""
    budget = current_budget()
    n = len(int_rows)
    H = [[value % modulus for value in row] for row in int_rows]
    for k in range(n - 2):
        pivot = k + 1
        while not H[pivot][k]:
            pivot += 1
            if pivot == n:
                break
        if pivot == n:
            continue
        below = k + 1
        if pivot != below:
            H[pivot], H[below] = H[below], H[pivot]
            for row in H:
                row[pivot], row[below] = row[below], row[pivot]
        inverse = pow(H[below][k], -1, modulus)
        source = H[below]
        for i in range(k + 2, n):
            target = H[i]
            factor = target[k] * inverse % modulus
            if factor:  # row_i -= factor row_below, then column_below += factor column_i (a similarity)
                target[k:] = [(a - factor * b) % modulus for a, b in zip(target[k:], source[k:])]
                for row in H:
                    row[below] = (row[below] + factor * row[i]) % modulus
        if budget is not None:
            budget.tick(1 + 2 * n * (n - k))
    polynomials = [[1]]
    for m in range(1, n + 1):
        previous = polynomials[m - 1]
        diagonal = H[m - 1][m - 1]
        current = [0, *previous]
        for index, coefficient in enumerate(previous):
            current[index] = (current[index] - diagonal * coefficient) % modulus
        product = 1
        for i in range(m - 2, -1, -1):
            product = product * H[i + 1][i] % modulus
            if not product:
                break
            weight = H[i][m - 1] * product % modulus
            if weight:
                for index, coefficient in enumerate(polynomials[i]):
                    current[index] = (current[index] - weight * coefficient) % modulus
        polynomials.append(current)
    return polynomials[n]


def check_charpoly(int_rows, coefficients) -> bool:
    """Whether coefficients (a list or tuple of ints, highest power first) are those of det(t I - A) for the square
    integer matrix int_rows."""
    n = len(int_rows)
    if (type(coefficients) not in (list, tuple) or len(coefficients) != n + 1 or coefficients[0] != 1
            or any(type(value) is not int for value in coefficients)):
        return False
    bound = 1
    for row in int_rows:
        if len(row) != n:
            return False
        squares = sum(map(mul, row, row))
        bound *= 2 + isqrt(squares)  # 1 + rho_i with rho_i = isqrt(squares) + 1 >= |row_i|.
    if any(abs(value) > bound for value in coefficients):
        return False
    product = 1
    for modulus in moduli():
        expected = _charpoly_mod(int_rows, modulus)
        if any((claimed - expected[n - k]) % modulus for k, claimed in enumerate(coefficients)):
            return False
        product *= modulus
        if product > 2 * bound:
            return True
    return False  # Unreachable: the primes never run out below the bound of a valid matrix.
