"""Independent checker of the family mathforge.areas.linalg.lattice: hermite_normal_form, smith_normal_form.

The checker never reduces a matrix: it multiplies the integer matrices of the evidence exactly and reads the canonical
shapes off the claimed value (docs/v09/cards/m7-extended.md). The theorems that make True imply the claim:

hermite_normal_form(M) = H, evidence (U, V).
    M is an m x n integer matrix, H an m x n integer matrix, U and V m x m integer matrices. The checker verifies
    U V = I, H = U M and the shape of H: every nonzero row has its leading nonzero entry (pivot) positive and strictly
    to the right of the pivot of the nonzero row above it, no nonzero row follows a zero row, and every entry above a
    pivot lies in [0, pivot). From U V = I, det(U) det(V) = 1 with integer determinants, so det(U) = +-1 and U is
    unimodular (its inverse V is an integer matrix). Hence the rows of H = U M generate the same lattice of Z^n as the
    rows of M, and H is a row Hermite normal form of that lattice. The Hermite normal form is unique: two matrices in
    that shape whose rows generate the same lattice are equal (the nonzero rows of each are the unique echelon basis
    with reduced entries above the pivots; the zero rows number m minus the rank in both). So H is THE Hermite normal
    form of M.

smith_normal_form(M) = (d_1, ..., d_k), evidence (U, U_inv, V, V_inv), k = min(m, n).
    The checker verifies that the d_i are ints with d_i >= 0 and d_i | d_(i+1) (0 divides only 0, so the zeros come
    last: the card's "zeros last" follows from the chain and is tested with it), U U_inv = I_m, V V_inv = I_n, and
    U M V = D, the m x n matrix with d_1, ..., d_k on its diagonal and zeros elsewhere. As above, U and V are
    unimodular. By the Cauchy-Binet formula the gcd of the j x j minors (the j-th determinantal divisor) of U M V
    divides that of M, and conversely through U_inv and V_inv, so D and M have the same determinantal divisors. For a
    diagonal D whose entries form a divisibility chain, the j-th determinantal divisor is d_1 ... d_j. Hence the
    nonzero d_j are the quotients of consecutive determinantal divisors of M, their number is the rank of M, and the
    rest are 0: the value is the unique Smith normal form of M (its invariant factors).

Shapes. The operation must be one of the two names (a str), args a dict of exactly M, and the evidence a tuple of
exactly 2 (hermite_normal_form) or 4 (smith_normal_form) items. M and the claimed H are identified by
(type(x).__module__, type(x).__qualname__) == ("mathforge.matrices", "Matrix") and read as data: rows, a tuple of
m tuples of n entries, and ncols, an exact int n >= 0 (H must have M's shape). Every entry is of the exact class
mathforge.model.Rational with an exact int numerator and the denominator 1 (an integer): a non-integer entry of M,
which the producer refuses as invalid input, answers False. An evidence matrix of size s is a tuple of s rows, each a
tuple of s exact ints (the 0 x 0 matrix is ()); a Matrix record there answers False. The value of smith_normal_form
is a tuple of exactly k exact ints. A bool, a float, a str, an int or Rational subclass, a list for a tuple, a wrong
shape or count, a negative factor, a broken chain and any other evidence answer False, never an exception.

Resource limits. ResourceLimitError leaves a claim unverified, never verified. It is raised for: an M with more
entries than max_matrix_entries (budget.matrix, which produce() calls first); an integer of M longer than
max_integer_bits (the request's inspection raises before a checker runs); an evidence matrix of the right length whose
s x s entries exceed max_matrix_entries (s = m for U and V, and s = n for V and V_inv of smith_normal_form: under the
default limits an M with more than 64 rows, or more than 64 columns for smith_normal_form, cannot be verified, so the
producer refuses those requests too); and work past max_work, which long evidence integers can take. An integer of
the value or of the evidence longer than max_integer_bits answers False: no computation under the budget forms it.

Budget. One computation budget for the whole check (the verifier's, or one of the default limits opened here). Every
loop is charged before it runs: one unit per entry read, and for each row of a matrix product, before it is formed,
n_inner * width * (1 + product_work(a, b) + (a + b) / 1024) for entries of at most a and b bits (the products and the
additions, on the scale of trusted.integers). A product row is compared at once, so a wrong claim usually stops at its
first row. No builtin pow runs.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Iterator
from operator import mul

from mathforge.limits import Budget, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of the class of M and of the value of hermite_normal_form.

IntRows = list[list[int]]


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the claimed normal form of args = {"M": M}, by the theorems of the module.
    False for anything but the operation's exact shapes; ResourceLimitError for a request or an evidence matrix past
    the limits that the module lists and for work past the budget."""
    if type(operation) is not str or operation not in _CHECKS:
        return False
    if type(args) is not dict or len(args) != 1 or "M" not in args or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every step: without an active budget, the call opens one.
        budget.tick()
        request = _request_rows(budget, args["M"])
        if request is None:
            return False
        rows, ncols = request
        return _CHECKS[operation](budget, rows, ncols, value, evidence)


def _check_hermite(budget: Budget, rows: IntRows, ncols: int, value: object, evidence: tuple) -> bool:
    """hermite_normal_form: H in the canonical shape, U V = I and U M = H, in that order."""
    size = len(rows)
    claimed = _claimed_rows(budget, value, size, ncols)
    if claimed is None or not _hermite_shape(budget, claimed, ncols) or len(evidence) != 2:
        return False
    left = _square(budget, evidence[0], size)
    right = _square(budget, evidence[1], size) if left is not None else None
    if right is None or not _inverse_pair(budget, left, right):
        return False
    for index, row in enumerate(_product_rows(budget, left, rows, ncols)):
        budget.tick(1 + ncols)
        if row != claimed[index]:
            return False
    return True


def _check_smith(budget: Budget, rows: IntRows, ncols: int, value: object, evidence: tuple) -> bool:
    """smith_normal_form: the divisibility chain, U U_inv = I, V V_inv = I and U M V = D, in that order."""
    size = len(rows)
    factors = _factors(budget, value, min(size, ncols))
    if factors is None or len(evidence) != 4:
        return False
    left, left_inverse = _square(budget, evidence[0], size), _square(budget, evidence[1], size)
    if left is None or left_inverse is None:
        return False
    right, right_inverse = _square(budget, evidence[2], ncols), _square(budget, evidence[3], ncols)
    if right is None or right_inverse is None:
        return False
    if not _inverse_pair(budget, left, left_inverse) or not _inverse_pair(budget, right, right_inverse):
        return False
    middle = list(_product_rows(budget, left, rows, ncols))  # U M, m x n.
    for index, row in enumerate(_product_rows(budget, middle, right, ncols)):
        budget.tick(1 + ncols)
        # Row i of D: d_(i+1) at column i when i < n (then i < k, since i < m), zeros elsewhere.
        expected = [factors[index] if column == index else 0 for column in range(ncols)]
        if row != expected:
            return False
    return True


_CHECKS = {"hermite_normal_form": _check_hermite, "smith_normal_form": _check_smith}


def _request_rows(budget: Budget, matrix: object) -> tuple[IntRows, int] | None:
    """The integer entries of M and its column count when M is a Matrix of integer Rationals, read as data; None
    otherwise. budget.matrix(rows, cols) comes first, as in produce(), and an integer of M longer than
    max_integer_bits raises ResourceLimitError."""
    if (type(matrix).__module__, type(matrix).__qualname__) != MATRIX:
        return None
    rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
    if type(rows) is not tuple or type(ncols) is not int or ncols < 0:
        return None
    budget.matrix(len(rows), ncols)  # Past max_matrix_entries: ResourceLimitError, before the shape is read.
    result = []
    for row in rows:
        budget.tick(1 + ncols)
        if type(row) is not tuple or len(row) != ncols:
            return None
        items = []
        for entry in row:
            numerator = _integer_entry(entry)
            if numerator is None:
                return None
            budget.check_int(numerator)
            items.append(numerator)
        result.append(items)
    return result, ncols


def _integer_entry(entry: object) -> int | None:
    """The int n of an entry that is an exact Rational n/1, None for any other entry (a non-integer Rational, an int,
    a bool, a float, a Rational subclass or a forged Rational with slots of other types)."""
    if type(entry) is not Rational:
        return None
    numerator = getattr(entry, "numerator", None)  # A forged Rational may lack a slot.
    denominator = getattr(entry, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator != 1:
        return None
    return numerator


def _claimed_rows(budget: Budget, value: object, size: int, ncols: int) -> IntRows | None:
    """The integer entries of the claimed H when it is a Matrix of M's shape whose entries are exact Rationals n/1
    with n within max_integer_bits, read as data; None otherwise."""
    if (type(value).__module__, type(value).__qualname__) != MATRIX:
        return None
    rows, width = getattr(value, "rows", None), getattr(value, "ncols", None)
    if type(rows) is not tuple or len(rows) != size or type(width) is not int or width != ncols:
        return None
    limit = budget.limits.max_integer_bits
    result = []
    for row in rows:
        budget.tick(1 + ncols)
        if type(row) is not tuple or len(row) != ncols:
            return None
        items = []
        for entry in row:
            numerator = _integer_entry(entry)
            if numerator is None or numerator.bit_length() > limit:
                return None
            items.append(numerator)
        result.append(items)
    return result


def _hermite_shape(budget: Budget, claimed: IntRows, ncols: int) -> bool:
    """Whether the integer rows are in row Hermite normal form: each nonzero row's pivot positive and strictly right of
    the pivot above, no nonzero row after a zero row, every entry above a pivot in [0, pivot)."""
    pivots = []  # (row, column) of each pivot, top to bottom.
    previous, ended = -1, False
    for index, row in enumerate(claimed):
        budget.tick(1 + ncols)
        column = next((place for place, entry in enumerate(row) if entry), -1)
        if column < 0:
            ended = True
            continue
        if ended or column <= previous or row[column] < 0:
            return False
        pivots.append((index, column))
        previous = column
    for index, column in pivots:
        pivot = claimed[index][column]
        budget.tick(1 + index * (1 + (pivot.bit_length() >> 10)))  # The comparisons: linear in the lengths.
        for above in range(index):
            entry = claimed[above][column]
            if entry < 0 or entry >= pivot:
                return False
    return True


def _factors(budget: Budget, value: object, count: int) -> list[int] | None:
    """The claimed invariant factors when value is a tuple of count exact ints d_i >= 0 within max_integer_bits with
    d_i | d_(i+1) (so the zeros come last); None otherwise."""
    if type(value) is not tuple or len(value) != count:
        return None
    limit = budget.limits.max_integer_bits
    budget.tick(1 + count)
    for factor in value:
        if type(factor) is not int or factor < 0 or factor.bit_length() > limit:
            return None
    for index in range(count - 1):
        current, following = value[index], value[index + 1]
        if not current:  # 0 divides only 0: a zero is followed by zeros alone.
            if following:
                return None
            continue
        budget.tick(1 + division_work(following.bit_length(), current.bit_length()))
        if following % current:
            return None
    return list(value)


def _square(budget: Budget, item: object, size: int) -> IntRows | None:
    """An evidence matrix of size x size exact ints within max_integer_bits, as integer rows; None otherwise. Its
    length is judged before budget.matrix(size, size), so that only a matrix of the right length past
    max_matrix_entries raises ResourceLimitError."""
    if type(item) is not tuple or len(item) != size:
        return None
    budget.matrix(size, size)
    limit = budget.limits.max_integer_bits
    result = []
    for row in item:
        budget.tick(1 + size)
        if type(row) is not tuple or len(row) != size:
            return None
        for entry in row:
            if type(entry) is not int or entry.bit_length() > limit:
                return None
        result.append(list(row))
    return result


def _longest(budget: Budget, rows: IntRows) -> int:
    """The greatest bit length of the entries of integer rows, 0 for none; one unit per entry."""
    longest = 0
    for row in rows:
        budget.tick(1 + len(row))
        for entry in row:
            bits = entry.bit_length()
            if bits > longest:
                longest = bits
    return longest


def _product_rows(budget: Budget, left: IntRows, right: IntRows, width: int) -> Iterator[list[int]]:
    """The rows of left * right, one at a time, for integer rows left (r x s) and right (s x width), each row charged
    before it is formed: s * width products of entries of at most a and b bits with their additions."""
    inner = len(right)
    right_bits = _longest(budget, right)
    budget.tick(1 + inner * width)
    columns = list(zip(*right)) if inner else [()] * width
    for row in left:
        budget.tick(1 + len(row))
        row_bits = max((entry.bit_length() for entry in row), default=0)
        budget.tick(width * inner * (1 + product_work(row_bits, right_bits) + ((row_bits + right_bits) >> 10)))
        yield [sum(map(mul, row, column)) for column in columns]


def _inverse_pair(budget: Budget, left: IntRows, right: IntRows) -> bool:
    """Whether left * right is the identity, for square integer rows of one size; stops at the first wrong row."""
    size = len(left)
    for index, row in enumerate(_product_rows(budget, left, right, size)):
        budget.tick(1 + size)
        if row != [1 if column == index else 0 for column in range(size)]:
            return False
    return True
