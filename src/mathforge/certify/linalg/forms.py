"""Independent checker of the family mathforge.areas.linalg.forms: definiteness and gram_schmidt.

The checker never reruns the producer's algorithms (a symmetric elimination for definiteness, the Gram-Schmidt
recurrence for gram_schmidt). It verifies the certificates of the card docs/v09/cards/m7-extended.md in exact integer
arithmetic. The theorems that make True imply the claim:

definiteness(M) = Inertia(positive, negative, zero, kind), evidence (S, D).
    M is an n x n matrix of exact rationals, n >= 0, and symmetric (checked first, entry by entry). S is an n x n
    rational matrix given by its rows and D a tuple of n rationals. The checker computes S^T M S exactly and requires
    it to equal diag(D), and proves S invertible from an integer form of S (below): its determinant is a unit modulo
    one of three word-sized moduli (certify.linalg._modular.nonsingular_mod; a unit is not 0), or else its own exact
    determinant (Bareiss's fraction-free elimination) is not 0. Sylvester's law of inertia: congruent real symmetric matrices, M and S^T M S with S
    invertible, have the same numbers of positive, negative and zero eigenvalues; those of diag(D) are the numbers
    of positive, negative and zero entries of D. So the counts of the signs of D are the inertia of M, and the
    checker requires them to equal positive, negative and zero. Every symmetric M has such a pair (S, D) (a
    symmetric elimination finds one), so the check is complete. kind is a function of the counts, derived in the
    card's order: positive == n gives "positive_definite", negative == n "negative_definite", negative == 0
    "positive_semidefinite" (the zero matrix included), positive == 0 "negative_semidefinite", anything else
    "indefinite"; each matches its definition through the eigenvalues (a symmetric matrix is positive definite iff
    every eigenvalue is positive, and so on). For n = 0 the derivation gives Inertia(0, 0, 0, "positive_definite")
    with the evidence ((), ()).
    The integer form: with m >= 1 the least common multiple of the denominators of M, M_int = m M, and with c_j >= 1
    that of the denominators of column j of S, S_int = S diag(c) (column j scaled by c_j), an integer matrix with
    det(S_int) = det(S) prod(c_j). Then (S^T M S)[i][j] = (S_int^T M_int S_int)[i][j] / (m c_i c_j), so S^T M S =
    diag(D) iff P = S_int^T M_int S_int has P[i][j] = 0 for i != j and P[j][j] * den(D_j) = m c_j^2 num(D_j). P is
    symmetric because M_int is, so the checker computes P[i][j] for i <= j only.
    Bareiss's elimination: with row exchanges, the pivot of step k is (up to sign) the leading (k+1) x (k+1) minor of
    the permuted matrix, and every division by the previous pivot is exact; a step whose column holds no nonzero
    entry at or below the diagonal means a zero column of a Schur complement, so det = 0. det(S_int) != 0 exactly
    when every step finds a pivot.

gram_schmidt(M) = Q, evidence mu.
    M is a k x n matrix of exact rationals (k, n >= 0) with rows m_0, ..., m_(k-1); Q a k x n Matrix of canonical
    Rationals with rows q_0, ..., q_(k-1); mu the evidence itself, a tuple of k tuples, mu[i] = (mu_i0, ...,
    mu_i(i-1)) of Rationals (mu[0] == ()). Write N_j = q_j . q_j. The checker requires
    (a) every q_i != 0 and q_i . q_j = 0 for i < j (the card's orthogonality);
    (b) m_i . q_j = 0 for j > i, m_i . q_i = N_i, and m_i . q_j = mu_ij N_j for j < i;
    (c) Bessel's equality m_i . m_i = sum_{j <= i} (m_i . q_j)^2 / N_j, for every i.
    These are the card's "M == L Q exactly" (L the unit lower triangular matrix of mu), checked without forming L Q,
    whose common denominators grow with the product of all the Gram determinants: by (a) the q_j are an orthogonal
    basis of their span W (rational vectors are real, so N_j > 0), the orthogonal projection of m_i on W is
    p_i = sum_j (m_i . q_j / N_j) q_j, which (b) makes q_i + sum_{j<i} mu_ij q_j, and m_i . m_i = p_i . p_i +
    |m_i - p_i|^2 with p_i . p_i = sum_j (m_i . q_j)^2 / N_j, so (c) gives m_i = p_i. Conversely M = L Q with (a)
    implies (b) and (c). So the checks hold exactly when (a) holds and M = L Q.
    Uniqueness: if M = L Q = L' Q' with L, L' unit lower triangular and Q, Q' with nonzero pairwise orthogonal rows,
    then Q' = T Q with T = L'^-1 L unit lower triangular, and Q' Q'^T = T (Q Q^T) T^T where Q Q^T and Q' Q'^T are
    diagonal with positive diagonals. The factorization LDL^T of a positive definite matrix with L unit lower
    triangular is unique, so T = I and Q' = Q, L' = L. The Gram-Schmidt rows q_i = m_i - sum_{j<i} mu_ij q_j with
    mu_ij = (m_i . q_j)/(q_j . q_j) satisfy (a) and M = L Q whenever the rows of M are linearly independent (by
    induction, q_i is m_i minus its orthogonal projection on span(m_0, ..., m_(i-1)) = span(q_0, ..., q_(i-1)),
    nonzero as m_i lies outside that span), so the unique Q that passes is the claimed value, with the claimed mu,
    and the check is complete. Linearly dependent rows admit no passing claim: L is invertible, so Q would have the
    rank of M, below k, while k nonzero orthogonal rows are independent. k = 0 gives the Matrix with no row and M's
    column count, with the evidence ().
    The integer form: c_j >= 1 the least common multiple of the denominators of q_j, Q_int[j] = c_j q_j and
    G_j = Q_int[j] . Q_int[j]; r_i that of the denominators of m_i and M_int[i] = r_i m_i; x_ij = M_int[i] . Q_int[j]
    = r_i c_j (m_i . q_j). Then (a) is Q_int[i] . Q_int[j] = 0, (b) is x_ij = 0 for j > i, x_ii c_i = r_i G_i and
    x_ij den(mu_ij) c_j = r_i num(mu_ij) G_j for j < i, and (c) is M_int[i] . M_int[i] = sum_{j <= i} x_ij^2 / G_j,
    a sum of fractions kept in lowest terms (each partial sum is the squared length of a projection of M_int[i]).

Shapes. The operation is the str "definiteness" or "gram_schmidt" and args a dict of exactly {"M": M}. M and the
gram_schmidt value are identified by (type(x).__module__, type(x).__qualname__) == ("mathforge.matrices", "Matrix"),
the definiteness value by ("mathforge.areas.linalg.forms", "Inertia"), and all are read as data (rows, a tuple of row
tuples; ncols, an exact int >= 0; positive, negative and zero exact ints >= 0; kind a str). Every Rational is of the
exact class mathforge.model.Rational with exact int numerator and denominator and a positive denominator. The entries
of the value Q must also be canonical: lowest terms (0 is 0/1). The evidence Rationals (S, D, mu) need not be in lowest
terms, as no step relies on it. A bool, a float, a str, an int where a Rational belongs, a subclass, a list for a tuple,
a row or a tuple of the wrong length, missing or extra evidence and a wrong operation name answer False, never an
exception.

Resource limits. ResourceLimitError leaves a claim unverified, never verified. It is raised, before any claim is read,
for an M with more entries than max_matrix_entries (budget.matrix, which produce() calls first) and for an integer of M
longer than max_integer_bits (the verifier's inspection of the request raises it before a checker runs); and for work
past max_work. A claimed integer (in the value or the evidence) longer than max_integer_bits answers False at once, as
no computation under the budget forms it; the producer keeps its evidence within that bound.

Budget. Every loop is charged before it runs, on the cost model of trusted.integers: product_work for a product, a
long division by the length of its quotient (division_work), a gcd by a first remainder and then quadratically in the
shorter operand, one unit per entry read. The only builtin pow is nonsingular_mod's modular inverse of a pivot below a
64-bit modulus (an extended gcd of word-sized integers, charged with its pivot step), after S_int is reduced modulo it
at a charged division per entry.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.limits import Budget, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

from ._modular import nonsingular_mod

MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of the class of M and of the gram_schmidt value.
INERTIA = ("mathforge.areas.linalg.forms", "Inertia")  # Module and name of the class of the definiteness value.

Pair = tuple[int, int]  # The numerator and the positive denominator of a Rational.
# Word-sized primes for the invertibility of S (_invertible). Soundness does not depend on their primality; a prime
# modulus only makes a nonzero determinant that it does not divide always give a unit.
_MODULI = (2 ** 61 - 1, 2 ** 64 - 59, 2 ** 63 - 25)


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the answer of operation for args = {"M": M}, by the theorems of the module.
    False for anything but the operations' exact shapes; ResourceLimitError for a request past the limits that the
    module lists and for work past the budget."""
    if type(operation) is not str or type(args) is not dict or len(args) != 1 or "M" not in args:
        return False
    if operation == "definiteness":
        check = _check_definiteness
    elif operation == "gram_schmidt":
        check = _check_gram_schmidt
    else:
        return False
    with computation() as budget:  # One budget for every step: without an active budget, the call opens one.
        budget.tick()
        return check(budget, args["M"], value, evidence)


# ---------------------------------------------------------------------------------------------------------------
# definiteness

def _kind(positive: int, negative: int, zero: int) -> str:
    """The card's kind of the counts, in the card's order of tests."""
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


def _check_definiteness(budget: Budget, matrix: object, value: object, evidence: object) -> bool:
    request = _request_rows(budget, matrix)
    if request is None:
        return False
    rows, ncols = request
    n = len(rows)
    if ncols != n or not _symmetric(budget, rows):
        return False
    counts = _inertia(value, n)
    if counts is None:
        return False
    limit = budget.limits.max_integer_bits
    if type(evidence) is not tuple or len(evidence) != 2:
        return False
    s_rows = _claimed_rows(budget, evidence[0], n, n, limit, canonical=False)
    diagonal = _claimed_vector(budget, evidence[1], n, limit, canonical=False)
    if s_rows is None or diagonal is None:
        return False
    # Sylvester: the signs of D are the inertia. Compared first, as it costs one pass.
    signs = (sum(1 for numerator, _ in diagonal if numerator > 0), sum(1 for numerator, _ in diagonal if numerator < 0),
             sum(1 for numerator, _ in diagonal if not numerator))
    if signs != counts:
        return False
    common = _common_denominator(budget, [pair for row in rows for pair in row])
    m_int = [_scaled(budget, row, common) for row in rows]
    columns = []  # Column j of S_int = S diag(c), and c_j.
    for j in range(n):
        budget.tick(1 + n)
        column = [s_rows[i][j] for i in range(n)]
        scale = _common_denominator(budget, column)
        columns.append((_scaled(budget, column, scale), scale))
    # T = M_int S_int column by column, then P = S_int^T T on and above the diagonal.
    t_columns = []
    for s_column, _ in columns:
        budget.tick(1 + n)
        t_columns.append([_dot(budget, m_row, s_column) for m_row in m_int])
    for j in range(n):
        t_column = t_columns[j]
        for i in range(j):
            if _dot(budget, columns[i][0], t_column):
                return False
        entry = _dot(budget, columns[j][0], t_column)
        numerator, denominator = diagonal[j]
        scale = columns[j][1]
        if not entry and not numerator:  # Both sides 0; spares the products below.
            continue
        bits = entry.bit_length() + denominator.bit_length()
        budget.tick(3 + product_work(entry.bit_length(), denominator.bit_length())
                    + 2 * product_work(bits, bits) + product_work(bits, numerator.bit_length()))
        if entry * denominator != common * scale * scale * numerator:
            return False
    return _invertible(budget, [[column[i] for column, _ in columns] for i in range(n)])


def _invertible(budget: Budget, int_rows: list[list[int]]) -> bool:
    """Whether a square integer matrix has a nonzero determinant: True as soon as its determinant is a unit modulo
    one of _MODULI, else Bareiss's exact elimination decides.

    Sound: nonsingular_mod answers True only for a determinant that is a unit modulo the modulus, and 0 is a unit
    modulo no modulus >= 2, so the determinant is not 0 (this holds for any modulus, prime or not). Complete: the exact
    elimination still decides every matrix whose determinant the moduli all divide (or that is 0). The moduli come
    first because Bareiss's intermediate entries are minors of S_int: for the S = T U^T of a symmetric elimination they
    reach the sum of the lengths of all its pivots, and a 64 x 64 matrix of small rationals cost some 10^7 units there
    (five times max_work) against some 10^5 units modulo a word-sized modulus."""
    n = len(int_rows)
    for modulus in _MODULI:
        modulus_bits = modulus.bit_length()
        work = 1 + n
        for row in int_rows:
            for entry in row:
                work += 1 + _quotient_work(entry.bit_length(), modulus_bits)
        budget.tick(work)
        reduced = [[entry % modulus for entry in row] for row in int_rows]
        budget.tick(1 + n)  # det_mod charges each of its pivot steps to the active budget, this one.
        if nonsingular_mod(reduced, modulus):
            return True
    return _nonsingular(budget, int_rows)


def _inertia(value: object, n: int) -> tuple[int, int, int] | None:
    """(positive, negative, zero) of an Inertia record read as data, when its counts are exact ints >= 0 adding up to
    n and its kind is the card's kind of the counts; None otherwise."""
    if (type(value).__module__, type(value).__qualname__) != INERTIA:
        return None
    fields = (getattr(value, "positive", None), getattr(value, "negative", None), getattr(value, "zero", None))
    kind = getattr(value, "kind", None)  # A record forged with object.__new__ may lack a field.
    if any(type(field) is not int or field < 0 for field in fields) or type(kind) is not str:
        return None
    positive, negative, zero = fields
    if positive + negative + zero != n or kind != _kind(positive, negative, zero):
        return None
    return positive, negative, zero


def _symmetric(budget: Budget, rows: list[list[Pair]]) -> bool:
    """Whether M[i][j] = M[j][i] for every i < j, compared by cross products (the request's entries need not be in
    lowest terms)."""
    n = len(rows)
    for i in range(n):
        budget.tick(1 + n)
        for j in range(i + 1, n):
            (a, b), (c, d) = rows[i][j], rows[j][i]
            if a == c and b == d:
                continue
            budget.tick(2 + product_work(a.bit_length(), d.bit_length()) + product_work(c.bit_length(), b.bit_length()))
            if a * d != c * b:
                return False
    return True


def _nonsingular(budget: Budget, int_rows: list[list[int]]) -> bool:
    """Whether a square integer matrix has a nonzero determinant, by Bareiss's fraction-free elimination with row
    exchanges: every step must find a pivot. True for the 0 x 0 matrix (determinant 1)."""
    n = len(int_rows)
    a = [list(row) for row in int_rows]
    budget.tick(1 + n * n)
    previous = 1
    for k in range(n):
        if not a[k][k]:
            budget.tick(n - k)
            for i in range(k + 1, n):
                if a[i][k]:
                    a[k], a[i] = a[i], a[k]
                    break
            else:
                return False
        pivot_row = a[k]
        pivot = pivot_row[k]
        pivot_bits, previous_bits = pivot.bit_length(), previous.bit_length()
        for i in range(k + 1, n):
            row = a[i]
            factor = row[k]
            factor_bits = factor.bit_length()
            work = 1 + n - k
            for j in range(k + 1, n):
                entry_bits = row[j].bit_length()
                work += (product_work(entry_bits, pivot_bits) + product_work(factor_bits, pivot_row[j].bit_length())
                         + _quotient_work(max(entry_bits + pivot_bits, factor_bits + pivot_row[j].bit_length()) + 1,
                                          previous_bits))
            budget.tick(work)
            for j in range(k + 1, n):
                row[j] = (row[j] * pivot - factor * pivot_row[j]) // previous  # Exact (Sylvester's identity).
        previous = pivot
    return True


# ---------------------------------------------------------------------------------------------------------------
# gram_schmidt

def _check_gram_schmidt(budget: Budget, matrix: object, value: object, evidence: object) -> bool:
    request = _request_rows(budget, matrix)
    if request is None:
        return False
    rows, ncols = request
    k = len(rows)
    limit = budget.limits.max_integer_bits
    if (type(value).__module__, type(value).__qualname__) != MATRIX:
        return False
    q_rows = _claimed_rows(budget, getattr(value, "rows", None), k, ncols, limit, canonical=True)
    claimed_ncols = getattr(value, "ncols", None)  # Read once: an impostor's property may answer differently twice.
    if q_rows is None or type(claimed_ncols) is not int or claimed_ncols != ncols:
        return False
    if type(evidence) is not tuple or len(evidence) != k:
        return False
    mu = []
    for i, row in enumerate(evidence):
        coefficients = _claimed_vector(budget, row, i, limit, canonical=False)
        if coefficients is None:
            return False
        mu.append(coefficients)
    # Q_int[j] = c_j q_j, every row nonzero, and G_j = Q_int[j] . Q_int[j] > 0.
    q_int, scales, norms = [], [], []
    for row in q_rows:
        budget.tick(1 + ncols)
        if not any(numerator for numerator, _ in row):
            return False
        scale = _common_denominator(budget, row)
        scaled = _scaled(budget, row, scale)
        q_int.append(scaled)
        scales.append(scale)
        norms.append(_dot(budget, scaled, scaled))
    # (a) Orthogonality.
    for i in range(k):
        for j in range(i + 1, k):
            if _dot(budget, q_int[i], q_int[j]):
                return False
    for i in range(k):
        budget.tick(1 + k)
        common = _common_denominator(budget, rows[i])
        m_int = _scaled(budget, rows[i], common)  # M_int[i] = r_i m_i.
        # (b) x_ij = M_int[i] . Q_int[j]: 0 for j > i, x_ii c_i = r_i G_i, and x_ij den(mu_ij) c_j = r_i num(mu_ij) G_j
        # for j < i.
        for j in range(i + 1, k):
            if _dot(budget, m_int, q_int[j]):
                return False
        products = []  # x_ij for j <= i.
        for j in range(i + 1):
            product = _dot(budget, m_int, q_int[j])
            numerator, denominator = mu[i][j] if j < i else (1, 1)
            left = (product, denominator, scales[j])
            right = (common, numerator, norms[j])
            if not _equal_products(budget, left, right):
                return False
            products.append(product)
        # (c) Bessel's equality H_i = sum_{j <= i} x_ij^2 / G_j, with H_i = M_int[i] . M_int[i].
        if not _bessel(budget, _dot(budget, m_int, m_int), products, norms):
            return False
    return True


def _equal_products(budget: Budget, left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    """Whether the products of two triples of ints are equal, each product charged before it runs."""
    sides = []
    for a, b, c in (left, right):
        a_bits, b_bits, c_bits = a.bit_length(), b.bit_length(), c.bit_length()
        budget.tick(3 + product_work(a_bits, b_bits) + product_work(a_bits + b_bits, c_bits))
        sides.append(a * b * c)
    return sides[0] == sides[1]


def _bessel(budget: Budget, target: int, products: list[int], norms: list[int]) -> bool:
    """Whether target = sum_j products[j]**2 / norms[j] (norms > 0), summed as a fraction kept in lowest terms."""
    numerator, denominator = 0, 1
    for product, norm in zip(products, norms):
        if not product:
            budget.tick()
            continue
        product_bits, norm_bits = product.bit_length(), norm.bit_length()
        numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
        budget.tick(6 + product_work(product_bits, product_bits) + product_work(numerator_bits, norm_bits)
                    + product_work(2 * product_bits, denominator_bits) + product_work(denominator_bits, norm_bits))
        numerator = numerator * norm + product * product * denominator
        denominator *= norm
        numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
        budget.tick(_gcd_work(numerator_bits, denominator_bits))
        divisor = gcd(numerator, denominator)
        if divisor != 1:
            budget.tick(2 + _quotient_work(numerator_bits, divisor.bit_length())
                        + _quotient_work(denominator_bits, divisor.bit_length()))
            numerator //= divisor
            denominator //= divisor
    target_bits, denominator_bits = target.bit_length(), denominator.bit_length()
    budget.tick(1 + product_work(target_bits, denominator_bits))
    return numerator == target * denominator


# ---------------------------------------------------------------------------------------------------------------
# Reading and integer arithmetic

def _request_rows(budget: Budget, matrix: object) -> tuple[list[list[Pair]], int] | None:
    """The entries of M as (numerator, denominator) pairs and its column count when M is a Matrix of exact Rationals
    with positive denominators, read as data; None otherwise. budget.matrix(rows, cols) comes first, as in produce(),
    and a request integer over max_integer_bits raises ResourceLimitError, as the verifier's request inspection
    does."""
    if (type(matrix).__module__, type(matrix).__qualname__) != MATRIX:
        return None
    rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
    if type(rows) is not tuple or type(ncols) is not int or ncols < 0:
        return None
    budget.matrix(len(rows), ncols)  # Past max_matrix_entries: ResourceLimitError, before the shape is read.
    pairs = []
    for row in rows:
        budget.tick(1 + ncols)
        if type(row) is not tuple or len(row) != ncols:
            return None
        items = []
        for entry in row:
            if type(entry) is not Rational:
                return None
            numerator, denominator = getattr(entry, "numerator", None), getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int or denominator < 1:
                return None
            budget.check_int(numerator)
            budget.check_int(denominator)
            items.append((numerator, denominator))
        pairs.append(items)
    return pairs, ncols


def _claimed_rows(budget: Budget, rows: object, nrows: int, ncols: int, limit: int, *,
                  canonical: bool) -> list[list[Pair]] | None:
    """A claimed tuple of nrows tuples of ncols Rationals as pairs (see _claimed); None for any other shape."""
    if type(rows) is not tuple or len(rows) != nrows:
        return None
    pairs = []
    for row in rows:
        budget.tick()
        items = _claimed_vector(budget, row, ncols, limit, canonical=canonical)
        if items is None:
            return None
        pairs.append(items)
    return pairs


def _claimed_vector(budget: Budget, entries: object, length: int, limit: int, *, canonical: bool) -> list[Pair] | None:
    """A claimed tuple of length Rationals as pairs (see _claimed); None for any other shape."""
    if type(entries) is not tuple or len(entries) != length:
        return None
    budget.tick(1 + length)
    items = []
    for entry in entries:
        pair = _claimed(budget, entry, limit, canonical)
        if pair is None:
            return None
        items.append(pair)
    return items


def _claimed(budget: Budget, entry: object, limit: int, canonical: bool) -> Pair | None:
    """(numerator, denominator) of a claimed Rational of the exact class, with exact int fields, a positive
    denominator and both integers within limit bits, and, when canonical, in lowest terms (0 as 0/1); None
    otherwise. A longer integer answers None before any arithmetic touches it."""
    if type(entry) is not Rational:  # An int is no Rational here, nor is a bool, a float or a subclass.
        return None
    numerator, denominator = getattr(entry, "numerator", None), getattr(entry, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        return None
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    if numerator_bits > limit or denominator_bits > limit:
        return None
    if canonical and denominator != 1:
        budget.tick(_gcd_work(numerator_bits, denominator_bits))
        if gcd(numerator, denominator) != 1:
            return None
    return numerator, denominator


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd (Lehmer's algorithm) of integers of the given lengths: a first remainder of the longer
    by the shorter one (division_work), then one unit per 64 squared 64-bit words of the shorter one."""
    longer, shorter = max(left_bits, right_bits), min(left_bits, right_bits)
    words = shorter >> 6
    return 1 + division_work(longer, shorter) + (words * words >> 6)


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient."""
    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _lcm(budget: Budget, left: int, right: int) -> int:
    """The least common multiple of two positive ints, its gcd, quotient and product charged."""
    if right == 1:
        return left
    if left == 1:
        return right
    left_bits, right_bits = left.bit_length(), right.bit_length()
    budget.tick(2 + _gcd_work(left_bits, right_bits) + _quotient_work(left_bits, right_bits)
                + product_work(left_bits, right_bits))
    return left // gcd(left, right) * right


def _common_denominator(budget: Budget, pairs: list[Pair]) -> int:
    """The least common multiple of the denominators of some pairs, one lcm at a time."""
    budget.tick(1 + len(pairs))
    common = 1
    for _, denominator in pairs:
        if denominator == 1 or denominator == common:
            continue
        budget.tick(1 + _quotient_work(common.bit_length(), denominator.bit_length()))
        if common % denominator:
            common = _lcm(budget, common, denominator)
    return common


def _scaled(budget: Budget, pairs: list[Pair], common: int) -> list[int]:
    """The integers numerator * (common / denominator) of pairs whose denominators divide common."""
    budget.tick(1 + len(pairs))
    common_bits = common.bit_length()
    result = []
    for numerator, denominator in pairs:
        if denominator == common:
            result.append(numerator)
            continue
        budget.tick(2 + _quotient_work(common_bits, denominator.bit_length())
                    + product_work(common_bits, numerator.bit_length()))
        result.append(numerator * (common // denominator))
    return result


def _dot(budget: Budget, left: list[int], right: list[int]) -> int:
    """The dot product of two integer vectors of one length, each product charged before the sum runs."""
    work = 1 + len(left)
    for a, b in zip(left, right):
        a_bits, b_bits = a.bit_length(), b.bit_length()
        work += product_work(a_bits, b_bits) + ((a_bits + b_bits) >> 12)
    budget.tick(work)
    return sum(a * b for a, b in zip(left, right))
