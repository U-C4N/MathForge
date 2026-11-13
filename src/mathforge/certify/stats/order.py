"""Independent checker of the family mathforge.areas.stats.order: quantile and mode.

The checker never sorts the data. It proves each claimed order statistic by counting, and the multiplicities of mode
by one counting pass.

The data, for both operations. The request holds data, a nonempty tuple (a request holds a list frozen into a tuple)
whose items are exact rationals: an int, or a Rational of two exact ints in lowest terms with a positive denominator,
as the Rational constructor makes it. Each item is read as the pair (numerator, denominator), (k, 1) for an int k:
this pair is the unique reduced form of the number, so two items are equal as numbers exactly when their pairs are
equal (2 and Rational(2) are one value), and x < y exactly when x.numerator * y.denominator < y.numerator *
x.denominator (positive denominators). Let x_(0) <= ... <= x_(n-1) be the data sorted.

The theorem that makes True imply the claim, for each operation:

quantile(data, p) = q, evidence ((j, v),) or ((j, v), (j + 1, w)).
    p is an exact rational with 0 <= p <= 1 (an int, or a Rational in lowest terms), else every claim answers False.
    The checker forms h = (n - 1) p = t / d (d the denominator of p) and j = floor(t / d) with the remainder
    r = t - j d, so h - j = r / d. The evidence lists exactly the ranks (j,) when r == 0 and (j, j + 1) when r != 0,
    in this order, each rank an exact int and each order statistic v an exact int or a canonical Rational (within
    max_integer_bits). For every listed (k, v) one counting pass forms L = #{i : x_i < v} and U = #{i : x_i <= v} and
    requires L <= k < U.
    Lemma (as statistics.median_certificate argues). In the sorted data the items below v take exactly the positions
    0, ..., L - 1 and the items equal to v the positions L, ..., U - 1; so x_(k) = v exactly when L <= k < U (for a v
    that is no item, L = U and no rank passes). The evidence therefore holds the true order statistics x_(j) and, when
    r != 0, x_(j+1). The value must be a Rational in canonical form (exact ints in lowest terms, a positive
    denominator, both within max_integer_bits; never an int) equal to x_(j) when r == 0, and when r != 0 equal to
    x_(j) + (r / d)(x_(j+1) - x_(j)), compared by cross-multiplication with positive denominators:
        value.numerator * (d b e) == value.denominator * (a d e + r (c b - a e))
    for x_(j) = a / b and x_(j+1) = c / e. That is the type-7 quantile of the card, and a number has exactly one
    canonical Rational: each input has exactly one accepted value.

mode(data) = (m_1, ..., m_s), evidence (c,).
    The evidence is a tuple of one exact int c >= 1; the value a nonempty tuple of Rationals in canonical form (never
    an int), strictly ascending. One counting pass over the data maps every reduced pair to its multiplicity (keys
    encoded as bytes, whose hash Python randomizes per process, so that data built to collide in the integer hash
    cannot make the dict quadratic beyond the budget). True requires: the largest multiplicity M equals c; every m_i
    has multiplicity M; and the number of distinct values of multiplicity M equals s.
    Theorem. The pairs are the unique reduced forms, so the dict counts the multiplicity of each number exactly and M
    is the maximal multiplicity. The m_i are distinct (strictly ascending) members of the set S of the values of
    multiplicity M, and there are |S| of them, so {m_1, ..., m_s} = S; strictly ascending, they are S in ascending
    order: the card's value, in its one canonical form.

Shapes and budget. The operation must be the string "quantile" or "mode", args a dict of exactly its arguments (data
and p, or data), data a nonempty tuple. A bool, a float, a str, an int subclass or a Rational subclass is no exact
number, a list no tuple, a record of another class with the same fields no Rational, and anything else answers
False, never an exception. A value or evidence integer over max_integer_bits answers False, as no computation under
the budget forms it; a quantile evidence of more than two entries, a mode value longer than the data and a
multiplicity above the data's length are refused before any arithmetic, so an oversized forgery answers False at
once. A request integer over max_integer_bits raises ResourceLimitError, as operations_v6.run and the verifier's
inspection raise before a checker runs, and so does work past max_work: a claim is then unverified, never verified.
Every loop is charged before it runs, each long integer operation on the scale of the cost model of trusted.integers
(product_work for a product, division_work for the length of a quotient, a gcd quadratic in its shorter operand, one
unit per 256 words of a linear pass). No builtin pow runs, so certify.ntheory.factorization.charged_pow is not needed.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

from mathforge.limits import Budget, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

# The argument names of each operation.
ARGUMENTS = {"quantile": frozenset(("data", "p")), "mode": frozenset(("data",))}
RATIONAL = ("mathforge.model", "Rational")  # Module and name of the class of every value number.

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1, in lowest terms.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, malformed arguments, a malformed or non-canonical value and malformed or wrong
    evidence; ResourceLimitError for a request integer over max_integer_bits and for work past the budget."""
    if type(operation) is not str:
        return False
    check = _CHECKS.get(operation)
    if check is None:
        return False
    names = ARGUMENTS[operation]
    if (type(args) is not dict or len(args) != len(names) or any(type(name) is not str for name in args)
            or not names.issuperset(args)):
        return False
    data = args["data"]
    if type(data) is not tuple or not data or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every step: the active one, or one of the default limits.
        return check(budget, args, value, evidence)


# ---------------------------------------------------------------------------------------------------------------
# The two operations

def _quantile(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    """The quantile theorem of the module: the ranks j (and j + 1) from p, each order statistic proved by counting,
    then the value by the type-7 formula."""
    claim = _value_number(budget, value)
    if claim is None or not 1 <= len(evidence) <= 2:
        return False
    entries: list[tuple[int, Pair]] = []
    for entry in evidence:
        budget.tick()
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not int:
            return False
        number = _evidence_number(budget, entry[1])
        if number is None:
            return False
        entries.append((entry[0], number))
    p = _request_number(budget, args["p"])
    if p is None:
        return False
    p_numerator, p_denominator = p
    if p_numerator < 0 or p_numerator > p_denominator:  # 0 <= p <= 1.
        return False
    pairs = _pairs(budget, args["data"])
    if pairs is None:
        return False
    t = _product(budget, len(pairs) - 1, p_numerator)  # h = (n - 1) p = t / d.
    budget.tick(1 + _quotient_work(t.bit_length(), p_denominator.bit_length()))
    j, remainder = divmod(t, p_denominator)  # j = floor(h), h - j = remainder / d.
    ranks = (j,) if remainder == 0 else (j, j + 1)
    if len(entries) != len(ranks) or any(rank != expected for (rank, _), expected in zip(entries, ranks)):
        return False
    for rank, number in entries:
        if not _order_statistic(budget, pairs, rank, number):
            return False
    if remainder == 0:
        return claim == entries[0][1]  # Both reduced pairs: equal exactly when the numbers are.
    (a, b), (c, e) = entries[0][1], entries[1][1]  # x_(j) = a / b and x_(j+1) = c / e.
    # x_(j) + (r / d)(c / e - a / b) = (a d e + r (c b - a e)) / (d b e).
    denominator = _product(budget, _product(budget, p_denominator, b), e)
    gap = _difference(budget, _product(budget, c, b), _product(budget, a, e))
    numerator = _sum(budget, _product(budget, _product(budget, a, p_denominator), e), _product(budget, remainder, gap))
    return _product(budget, claim[0], denominator) == _product(budget, claim[1], numerator)


def _mode(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    """The mode theorem of the module: one counting pass, the maximal count against the evidence, the value against
    the values of maximal count."""
    data = args["data"]
    if len(evidence) != 1 or type(evidence[0]) is not int:
        return False
    multiplicity = evidence[0]
    if not 1 <= multiplicity <= len(data):  # Refused before any arithmetic, however long the integer.
        return False
    if type(value) is not tuple or not value or len(value) > len(data):
        return False
    claims: list[Pair] = []
    for number in value:
        budget.tick()
        pair = _value_number(budget, number)
        if pair is None:
            return False
        if claims and not _less(budget, claims[-1], pair):  # Strictly ascending, so distinct.
            return False
        claims.append(pair)
    pairs = _pairs(budget, data)
    if pairs is None:
        return False
    counts: dict[tuple[bytes, bytes], int] = {}
    for pair in pairs:
        key = _key(budget, pair)
        counts[key] = counts.get(key, 0) + 1
    top = 0
    for count in counts.values():
        budget.tick()
        top = max(top, count)
    if top != multiplicity:
        return False
    winners = 0
    for count in counts.values():
        budget.tick()
        winners += count == top
    if winners != len(claims):
        return False
    for pair in claims:
        if counts.get(_key(budget, pair)) != top:
            return False
    return True


_CHECKS: dict[str, Callable[[Budget, dict, object, tuple], bool]] = {
    "quantile": _quantile,
    "mode": _mode,
}


# ---------------------------------------------------------------------------------------------------------------
# Order statistics and counting

def _order_statistic(budget: Budget, pairs: list[Pair], rank: int, number: Pair) -> bool:
    """Whether number is the order statistic x_(rank) of the data: #{x < v} <= rank < #{x <= v}, in one pass."""
    numerator, denominator = number
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    below = equal = 0
    for x_numerator, x_denominator in pairs:
        x_numerator_bits, x_denominator_bits = x_numerator.bit_length(), x_denominator.bit_length()
        budget.tick(2 + product_work(x_numerator_bits, denominator_bits) + product_work(numerator_bits,
                                                                                         x_denominator_bits)
                    + _linear(x_numerator_bits + x_denominator_bits + numerator_bits + denominator_bits))
        left, right = x_numerator * denominator, numerator * x_denominator
        if left < right:
            below += 1
        elif left == right:
            equal += 1
    return below <= rank < below + equal


def _key(budget: Budget, pair: Pair) -> tuple[bytes, bytes]:
    """The reduced pair as two byte strings, an injective encoding (the length follows from the bit length) whose hash
    Python randomizes per process: the counting dict then cannot be driven quadratic by colliding integer hashes."""
    numerator, denominator = pair
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    budget.tick(1 + 2 * _linear(numerator_bits + denominator_bits))  # The encoding, then the hash.
    return (numerator.to_bytes((numerator_bits + 8) >> 3, "little", signed=True),
            denominator.to_bytes((denominator_bits + 8) >> 3, "little", signed=True))


def _less(budget: Budget, left: Pair, right: Pair) -> bool:
    """Whether left < right, for reduced pairs with positive denominators."""
    return _product(budget, left[0], right[1]) < _product(budget, right[0], left[1])


# ---------------------------------------------------------------------------------------------------------------
# Shapes of the numbers

def _named(value: object, name: tuple[str, str]) -> bool:
    """Whether the value's class has this module and name, both exact strings (a class may set __module__ to any
    object, whose comparison could answer anything)."""
    cls = type(value)
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


def _value_number(budget: Budget, number: object) -> Pair | None:
    """(p, s) of a value number: exactly a Rational of exact ints in lowest terms with s >= 1, both within
    max_integer_bits; None for anything else: an int, a Rational subclass or impostor, and a Rational made with
    object.__new__ that lacks a slot or breaks the form."""
    if type(number) is not Rational or not _named(number, RATIONAL):
        return None
    return _reduced(budget, getattr(number, "numerator", None), getattr(number, "denominator", None))


def _evidence_number(budget: Budget, number: object) -> Pair | None:
    """(p, s) of an evidence number: an exact int (s = 1) or a value number; None for anything else, an integer over
    max_integer_bits included."""
    if type(number) is int:
        if number.bit_length() > budget.limits.max_integer_bits:
            return None
        budget.tick()
        return number, 1
    return _value_number(budget, number)


def _reduced(budget: Budget, numerator: object, denominator: object) -> Pair | None:
    """(numerator, denominator) when both are exact ints within max_integer_bits, the denominator >= 1 and the pair
    in lowest terms; None otherwise."""
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        return None
    limit = budget.limits.max_integer_bits
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    if numerator_bits > limit or denominator_bits > limit:
        return None  # No computation under the budget forms it.
    budget.tick()
    if denominator > 1:
        budget.tick(_gcd_work(numerator_bits, denominator_bits))
        if gcd(numerator, denominator) != 1:
            return None
    return numerator, denominator


def _request_number(budget: Budget, number: object) -> Pair | None:
    """(p, s) of a number of the request: an exact int, or a Rational of exact ints with a positive denominator in
    lowest terms; None for anything else. An integer over max_integer_bits raises ResourceLimitError, as the request's
    inspection does."""
    budget.tick()
    kind = type(number)
    if kind is int:
        budget.check_int(number)
        return number, 1
    if kind is not Rational or not _named(number, RATIONAL):
        return None
    numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int:
        return None
    budget.check_int(numerator)
    budget.check_int(denominator)
    return _reduced(budget, numerator, denominator)


def _pairs(budget: Budget, numbers: tuple[object, ...]) -> list[Pair] | None:
    """The items of data as reduced pairs; None when an item is no exact rational in canonical form. The types and
    lengths of every item first (a linear pass), then the gcds."""
    raw: list[tuple[int, int]] = []
    for number in numbers:
        budget.tick()
        kind = type(number)
        if kind is int:
            budget.check_int(number)
            raw.append((number, 1))
            continue
        if kind is not Rational or not _named(number, RATIONAL):
            return None
        numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int:
            return None
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator < 1:
            return None
        raw.append((numerator, denominator))
    for numerator, denominator in raw:
        if _reduced(budget, numerator, denominator) is None:
            return None
    return raw


# ---------------------------------------------------------------------------------------------------------------
# Charged integer steps

def _linear(bits: int) -> int:
    """The units of a linear pass over integers of this many bits in all: one per 256 64-bit words."""
    return bits >> 14


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient."""
    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: one, a division of the longer by the shorter with a pass
    over the longer, then Lehmer's algorithm on the shorter, quadratic, one unit per 64 squared 64-bit words as
    trusted.upoly.charge_gcd charges it."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + _quotient_work(large, small) + (large >> 12) + (words * words >> 6)


def _product(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs."""
    left_bits, right_bits = left.bit_length(), right.bit_length()
    budget.tick(1 + product_work(left_bits, right_bits) + _linear(left_bits + right_bits))
    return left * right


def _sum(budget: Budget, left: int, right: int) -> int:
    """left + right, charged by the length of its operands."""
    budget.tick(1 + _linear(left.bit_length() + right.bit_length()))
    return left + right


def _difference(budget: Budget, left: int, right: int) -> int:
    """left - right, charged by the length of its operands."""
    budget.tick(1 + _linear(left.bit_length() + right.bit_length()))
    return left - right
