"""Independent checker of the family mathforge.areas.series.laurent: residue.

residue(function, variable, point) = s, evidence (k, R, c).

The function is a record of mathforge.rational_function.RationalFunction, read as data: its numerator N and denominator
D, records of mathforge.polynomial.Polynomial, both in the Symbol variable (equal name, domain, assumptions and uid).
Every class is identified by its module and qualified name, both exact strings; the checker imports no producer code.
Canonical means: a coefficient tuple of 1 to MAX_LENGTH Rationals, lowest degree first, without a trailing zero; D not
zero and monic, as the record holds it; a Rational with exact int fields, a denominator >= 1 and gcd 1. The point a is
an int or a canonical Rational (kind "rational"). The value s is a canonical Rational, never an int. The evidence is
exactly the tuple (k, R, c): k an int with 0 <= k <= deg D, R a tuple of exactly deg D - k + 1 canonical Rationals
(never ints) without a trailing zero, c a tuple of exactly k canonical Rationals, (c_{-k}, ..., c_{-1}). Anything else
(a list for a tuple, a bool, float or str for a number, an impostor or a forged record) answers False. The excluded
polynomials of the record are not read: they mark points where the function is undefined but finite, and leave the
Laurent expansion of N/D unchanged. The checker does not test gcd(N, D) = 1 either: the theorem below holds for every
N and every D != 0, so a claim about a forged non-reduced record is still a true statement about N/D.

The theorem that makes True imply the claim (docs/v09/cards/m7-extended.md). Write y = x - a. True requires:
(1) D = (x - a)^k R as polynomials, exactly;
(2) R(a) != 0;
(3) y^k divides N - R P, where P = sum_{i=0..k-1} c_i y^i and c_i = c[i] = c_{-(k-i)} (that is, P = sum_{j=1..k}
    c_{-j} y^(k-j)): the coefficients of y^0, ..., y^(k-1) in the Taylor expansion at a vanish;
(4) s = c_{-1} for k >= 1, and s = 0 for k = 0.
Then N = R P + y^k H for a polynomial H, so N/D = N/(y^k R) = P/y^k + H/R = sum_{j=1..k} c_{-j} (x - a)^(-j) + H/R,
and H/R is holomorphic at a since R(a) != 0. The Laurent expansion of N/D at a is unique, so c_{-1} is its coefficient
of (x - a)^(-1), the residue; for k = 0, N/D = N/R is holomorphic at a and the residue is 0. By (1) and (2), k is the
multiplicity of a in D and R = D/(x - a)^k, so k and R are unique, and the c are the unique principal part: the true
claim is the only one that passes. (With N and D coprime, N(a) != 0 for k >= 1, so k is the pole order.)

How each condition is verified, in integers. a = p/q in lowest terms (q >= 1). N = N'/G, D = D'/F and R = R'/E with
integer coefficient vectors N', D', R' and their least common denominators G, F, E.
(1) With t_j = binomial(k, j) q^j (-p)^(k-j), (q x - p)^k = sum t_j x^j and (x - a)^k = (q x - p)^k / q^k, so (1) is
    the coefficientwise identity D'_i q^k E = W_i F for W = R' * (t_0, ..., t_k), a convolution.
(2), (3) The Taylor coefficients at a: for an integer vector A' of degree n, T_A(z) = q^n A'((z + p)/q) =
    sum_i A'_i q^(n-i) (z + p)^i is an integer polynomial in z = q y (the Taylor shift by p of sum_i A'_i q^(n-i) w^i,
    by synthetic division), and A(x) = T_A(z)/(den(A) q^n). Its first L coefficients come from L passes. R(a) != 0 iff
    T_R(0) != 0. Multiplying (3) by G E q^(deg N + deg R + k), it reads, for m = 0, ..., k - 1:
        alpha T_N[m] = beta sum_{i=0..m} T_R[m-i] f_i,  alpha = E q^(deg R + k), beta = G q^(deg N), f_i = c_i q^(k-i),
    which, with c_i = e_i/H over the least common denominator H of the c, is an identity of integers.
(4) compares the pairs (numerator, denominator) of s and of c_{-1} (or of 0); equal exact ints make s canonical too.

Sizes, before the expensive steps, so that oversized forged evidence answers False quickly; each bound only refuses
claims that cannot be true. Let n = deg D, m = deg R = n - k and bits(x) the bit length of |x|.
- R: by (1), (q x - p)^k divides D' in Q[x]; q x - p is primitive, so by Gauss's lemma D' = (q x - p)^k S with S in
  Z[x], and R = q^k S / F. Every reduced denominator of R divides F, and Mignotte's bound for a factor in Z[x]
  (|S_j| <= binomial(m, j) ||D'||_2 <= 2^m sqrt(n + 1) max|D'_i|) bounds every reduced numerator by
  q^k 2^m (n + 1) max|D'_i|: an entry beyond bits(F), or beyond k bits(q) + m + bits(n + 1) + bits(max|D'_i|), is
  refused before its gcd runs. E divides F.
- c: solving (3) from m = 0 up, f_m = g_m / b^(m+1) with b = |beta T_R[0]| >= 1 and integers |g_m| <= A (2B)^m, where
  A = max_m |alpha T_N[m]| and B = max_{i<k} |beta T_R[i]| >= b (induction: |g_m| <= A b^m + B sum_{i<m} |g_i| b^(m-1-i)
  <= A B^m (1 + sum_{i<m} 2^i)). So c_m = g_m / (b^(m+1) q^(k-m)): a numerator beyond bits(A) + m (bits(B) + 1) bits or
  a denominator beyond (m + 1) bits(b) + (k - m) bits(q) bits is refused before its gcd runs, and H divides b^k q^k.

Work. An argument holding an integer longer than max_integer_bits raises ResourceLimitError, and so does an argument
whose common denominator is longer: the claim is then beyond the budget, not wrong. A value or evidence holding such an
integer answers False. Every product, quotient and gcd is charged before it runs, on the scale of
mathforge.trusted.integers (product_work, division_work; a gcd as a long division plus a quadratic term of the shorter
operand, as certify.series.sums charges it), every loop charges the budget, and work past max_work raises
ResourceLimitError, which leaves a claim unverified, never verified. No builtin pow runs: powers are chains of charged
products. The cost is about (k + 1)(m + 1) products for (1), (k + 1) n shift steps for T_R and k (deg N + 1) for T_N,
and k(k + 1)/2 products for (3), fewer where T_R or the t_j vanish (a = 0: (1) costs m + 1 products and no shift runs).

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead.
"""

from __future__ import annotations

from math import gcd

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Assumption, Domain, Rational, Symbol, Truth
from mathforge.trusted.integers import division_work, product_work

RATIONAL_FUNCTION = ("mathforge.rational_function", "RationalFunction")  # Module and name of the argument's class.
POLYNOMIAL = ("mathforge.polynomial", "Polynomial")  # Module and name of its numerator's and denominator's class.
MAX_LENGTH = 1025  # mathforge.polynomial.MAX_POLYNOMIAL_DEGREE + 1: no Polynomial record holds more coefficients.
MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.

_ARGUMENTS = ("function", "variable", "point")  # The argument names of residue, in the order of its spec.
_ZERO, _ONE = (0, 1), (1, 1)  # The pairs of the rationals 0 and 1.
_INTEGER_BITS = "Computation limit exceeded: integer bits"

Pair = tuple[int, int]  # (numerator, denominator) of a canonical rational.
Form = tuple[list[int], int, int]  # (integer coefficients, their common denominator, their largest bit length).
Key = tuple[object, ...]  # The fields of a Symbol that Symbol equality compares.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified residue of the function at the point, by the theorem of the
    module. False for an unknown operation, for arguments other than residue's and for any malformed value or
    evidence; ResourceLimitError for an argument beyond the budget and for work past it."""
    if type(operation) is not str or operation != "residue" or type(args) is not dict:
        return False
    if len(args) != len(_ARGUMENTS) or any(type(name) is not str for name in args) or any(
            name not in args for name in _ARGUMENTS):
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, open one of the defaults.
        return _residue(budget, args["function"], args["variable"], args["point"], value, evidence)


# ---------------------------------------------------------------------------------------------------------------
# Charged integer arithmetic

def _times(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs: one unit and what the lengths cost beyond it (product_work)."""
    budget.tick(1 + product_work(left.bit_length(), right.bit_length()))
    return left * right


def _quotient(budget: Budget, dividend: int, divisor: int) -> int:
    """dividend // divisor for a divisor >= 1, its long division charged before it runs (division_work)."""
    budget.tick(1 + division_work(dividend.bit_length(), divisor.bit_length()))
    return dividend // divisor


def _gcd(budget: Budget, left: int, right: int) -> int:
    """math.gcd, charged before it runs as certify.series.sums charges it: the long division of the longer operand by
    the shorter (division_work) and one unit per 64 squared 64-bit words of the shorter."""
    longer, shorter = max(left.bit_length(), right.bit_length()), min(left.bit_length(), right.bit_length())
    words = shorter >> 6
    budget.tick(1 + (words * words >> 6) + division_work(longer - shorter + 64, shorter))
    return gcd(left, right)


def _powers(budget: Budget, base: int, count: int) -> list[int]:
    """[base^0, base^1, ..., base^count], each a charged product; base = 1 costs one unit per entry."""
    powers = [1]
    for _ in range(count):
        budget.tick()
        powers.append(powers[-1] if base == 1 else _times(budget, powers[-1], base))
    return powers


def _bits(values: list[int]) -> int:
    """The largest bit length among the integers (0 for none)."""
    return max((value.bit_length() for value in values), default=0)


def _integers(budget: Budget, pairs: list[Pair], bound: int) -> Form | None:
    """(P, E, bits): the least common denominator E of the canonical pairs, the integers P[i] = pairs[i] * E and their
    largest bit length; None as soon as E would have more than bound bits."""
    denominator = 1
    for _, part in pairs:
        budget.tick()
        if part != 1:
            common = _gcd(budget, denominator, part)
            denominator = _times(budget, _quotient(budget, denominator, common), part)
            if denominator.bit_length() > bound:
                return None
    integers = []
    for numerator, part in pairs:
        budget.tick()
        integers.append(numerator if denominator == 1 else _times(budget, numerator, _quotient(budget, denominator,
                                                                                                 part)))
    return integers, denominator, _bits(integers)


def _taylor(budget: Budget, integers: list[int], p: int, q: int, qpowers: list[int], count: int) -> list[int]:
    """The first count coefficients (zeros beyond the degree) of T(z) = q^n A((z + p)/q) = sum_i A_i q^(n-i) (z + p)^i
    for the integer vector A of degree n = len(integers) - 1, qpowers holding q^0 .. q^n at least.

    The vector B_i = A_i q^(n-i) is shifted by p with synthetic division: pass i (j from n - 1 down to i, B_j += p
    B_(j+1)) makes the coefficient of z^i final, and the leading one never changes, so min(count, n) passes give the
    first count. Every value stays below (n + 1) max|B| (1 + |p|)^n in absolute value (the weights of B_k in an entry
    are at most binomial(k, j) |p|^(k-j)), the bound by which each pass is charged before it runs: per step a product
    by p and an addition."""
    n = len(integers) - 1
    values = []
    for i, coefficient in enumerate(integers):
        budget.tick()
        values.append(coefficient if q == 1 or not coefficient else _times(budget, coefficient, qpowers[n - i]))
    if p != 0:
        bound = _bits(values) + (n + 1).bit_length() + n * (abs(p) + 1).bit_length()
        step = 1 + product_work(p.bit_length(), bound) + (bound >> 11)
        for i in range(min(count, n)):
            budget.tick(1 + (n - i) * step)
            for j in range(n - 1, i - 1, -1):
                values[j] += p * values[j + 1]
    values = values[:count]
    return values + [0] * (count - len(values))


# ---------------------------------------------------------------------------------------------------------------
# Reading arguments, values and evidence

def _named(item: object, name: tuple[str, str]) -> bool:
    """Whether the item's class has this module and name, both exact strings (a class may set __module__ to any
    object, whose comparison could answer anything, and a metaclass may compute it)."""
    cls = type(item)
    try:
        module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    except Exception:  # Foreign code raised: no record class of mathforge does.
        return False
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


def _field(item: object, name: str) -> object:
    """A field of a record read as data; None where it is missing or foreign code raises (a class of the same module
    and name but not the record's may compute its fields)."""
    try:
        return getattr(item, name, None)
    except Exception:
        return None


def _symbol_key(symbol: object) -> Key | None:
    """(name, domain, assumptions, uid) of a Symbol whose fields have their exact types: what Symbol equality compares,
    in a form whose comparison runs no foreign code. None for anything else, a forged Symbol included."""
    if type(symbol) is not Symbol:
        return None
    name, domain = getattr(symbol, "name", None), getattr(symbol, "domain", None)
    assumptions, uid = getattr(symbol, "assumptions", None), getattr(symbol, "uid", None)
    if (type(name) is not str or type(domain) is not Domain or type(uid) is not str or type(assumptions) is not tuple
            or len(assumptions) > MAX_ASSUMPTIONS):
        return None
    facts = []
    for assumption in assumptions:  # At most MAX_ASSUMPTIONS items: no charge needed.
        if type(assumption) is not Assumption:
            return None
        fact = getattr(assumption, "name", None), getattr(assumption, "truth", None)
        if type(fact[0]) is not str or type(fact[1]) is not Truth:
            return None
        facts.append(fact)
    return name, domain, tuple(facts), uid


def _fields(item: object) -> Pair | None:
    """(numerator, denominator) of a Rational with exact int fields and a denominator >= 1, lowest terms not tested;
    None for anything else, an int and a forged Rational without its slots included."""
    if type(item) is not Rational:
        return None
    numerator, denominator = getattr(item, "numerator", None), getattr(item, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        return None
    return numerator, denominator


def _pair(budget: Budget, item: object, *, argument: bool = False, integer: bool = False,
          numerators: int | None = None, denominators: int | None = None) -> Pair | None:
    """(numerator, denominator) of a canonical Rational, or of an int where integer is True (the kind "rational" takes
    one); None for anything else, and for a numerator of more than numerators bits or a denominator of more than
    denominators bits, tested before the gcd runs. An integer longer than max_integer_bits raises ResourceLimitError in
    an argument and answers None in a value or evidence."""
    if type(item) is int:
        if not integer:
            return None
        pair = item, 1
    else:
        pair = _fields(item)
        if pair is None:
            return None
    numerator, denominator = pair
    limit = budget.limits.max_integer_bits
    if numerator.bit_length() > limit or denominator.bit_length() > limit:
        if argument:
            raise ResourceLimitError(_INTEGER_BITS)
        return None
    if ((numerators is not None and numerator.bit_length() > numerators)
            or (denominators is not None and denominator.bit_length() > denominators)):
        return None
    if denominator != 1 and _gcd(budget, numerator, denominator) != 1:
        return None
    return pair


def _read(budget: Budget, values: object, length: int | None = None, *, argument: bool = False,
          numerators: int | None = None, denominators: int | None = None) -> list[Pair] | None:
    """The pairs of a canonical coefficient tuple: Rationals, lowest degree first, without a trailing zero, exactly
    length of them (1 to MAX_LENGTH when length is None), the length tested before any entry is read; None for
    anything else."""
    if type(values) is not tuple or not 0 < len(values) <= MAX_LENGTH or (length is not None and len(values) != length):
        return None
    pairs = []
    for item in values:
        budget.tick()
        pair = _pair(budget, item, argument=argument, numerators=numerators, denominators=denominators)
        if pair is None:
            return None
        pairs.append(pair)
    if len(pairs) > 1 and pairs[-1][0] == 0:
        return None
    return pairs


def _polynomial(budget: Budget, item: object, key: Key) -> list[Pair] | None:
    """The pairs of a canonical Polynomial argument in the variable of that key; None for anything else."""
    if not _named(item, POLYNOMIAL) or _symbol_key(_field(item, "variable")) != key:
        return None
    return _read(budget, _field(item, "coefficients"), argument=True)


def _function(budget: Budget, item: object, key: Key) -> tuple[list[Pair], list[Pair]] | None:
    """(N, D) of a RationalFunction argument in the variable of that key, D not zero and monic; None otherwise."""
    if not _named(item, RATIONAL_FUNCTION):
        return None
    numerator = _polynomial(budget, _field(item, "numerator"), key)
    denominator = _polynomial(budget, _field(item, "denominator"), key)
    if numerator is None or denominator is None or denominator[-1] != _ONE:  # Monic, hence not zero.
        return None
    return numerator, denominator


def _source(budget: Budget, pairs: list[Pair]) -> Form:
    """_integers of a polynomial argument; ResourceLimitError when its common denominator exceeds max_integer_bits."""
    form = _integers(budget, pairs, budget.limits.max_integer_bits)
    if form is None:
        raise ResourceLimitError(_INTEGER_BITS)
    return form


# ---------------------------------------------------------------------------------------------------------------
# The operation

def _residue(budget: Budget, function: object, variable: object, point: object, value: object,
             evidence: object) -> bool:
    key = _symbol_key(variable)
    source = None if key is None else _function(budget, function, key)
    a = _pair(budget, point, argument=True, integer=True)
    claim = _fields(value)  # Compared below with a canonical pair: equal exact ints make it canonical too.
    if source is None or a is None or claim is None or type(evidence) is not tuple or len(evidence) != 3:
        return False
    k, cofactor, principal = evidence
    n_pairs, d_pairs = source
    degree = len(d_pairs) - 1  # n = deg D >= 0.
    if (type(k) is not int or not 0 <= k <= degree or type(cofactor) is not tuple or len(cofactor) != degree - k + 1
            or type(principal) is not tuple or len(principal) != k):
        return False
    p, q = a
    cofactor_degree = degree - k  # m = deg R.
    n_integers, n_denominator, _ = _source(budget, n_pairs)  # N = N'/G.
    d_integers, d_denominator, d_bits = _source(budget, d_pairs)  # D = D'/F.

    # (1) D = (x - a)^k R. The bounds of the module first, then R = R'/E and the convolution with the t_j.
    numerators = k * q.bit_length() + cofactor_degree + (degree + 1).bit_length() + d_bits
    r_pairs = _read(budget, cofactor, cofactor_degree + 1, numerators=numerators,
                    denominators=d_denominator.bit_length())
    if r_pairs is None:
        return False
    form = _integers(budget, r_pairs, d_denominator.bit_length())  # E divides F.
    if form is None:
        return False
    r_integers, r_denominator, _ = form
    qpowers = _powers(budget, q, max(degree, len(n_pairs) - 1, k))
    if not _divides(budget, d_integers, d_denominator, r_integers, r_denominator, p, q, qpowers, k):
        return False

    # (2) R(a) != 0, with the Taylor coefficients of R at a that (3) needs.
    r_taylor = _taylor(budget, r_integers, p, q, qpowers, max(k, 1))
    if r_taylor[0] == 0:
        return False
    if k == 0:  # N/D = N/R is holomorphic at a: the residue is 0.
        return claim == _ZERO

    # (3) The principal part: alpha T_N[m] = beta sum_{i<=m} T_R[m-i] f_i for m < k.
    n_taylor = _taylor(budget, n_integers, p, q, qpowers, k)
    alpha = _times(budget, r_denominator, qpowers[cofactor_degree + k])  # E q^(deg R + k).
    beta = _times(budget, n_denominator, qpowers[len(n_pairs) - 1])  # G q^(deg N).
    big_a = alpha.bit_length() + _bits(n_taylor)  # At least bits(A).
    big_b = beta.bit_length() + _bits(r_taylor)  # At least bits(B).
    small_b = beta.bit_length() + r_taylor[0].bit_length()  # At least bits(b).
    c_pairs = []
    for m, item in enumerate(principal):
        budget.tick()
        pair = _pair(budget, item, numerators=big_a + m * (big_b + 1),
                     denominators=(m + 1) * small_b + (k - m) * q.bit_length())
        if pair is None:
            return False
        c_pairs.append(pair)
    form = _integers(budget, c_pairs, k * (small_b + q.bit_length()))  # H divides b^k q^k.
    if form is None:
        return False
    c_integers, c_denominator, _ = form
    scaled = []  # e_i q^(k-i).
    for i, coefficient in enumerate(c_integers):
        budget.tick()
        scaled.append(coefficient if q == 1 or not coefficient else _times(budget, coefficient, qpowers[k - i]))
    left_scale = _times(budget, alpha, c_denominator)  # alpha H.
    nonzero = [j for j in range(k) if r_taylor[j]]  # The nonzero T_R[j]; the loop below runs over them alone.
    budget.tick(k)
    for m in range(k):
        budget.tick()
        total = 0
        for j in nonzero:
            if j > m:
                break
            budget.tick()
            if scaled[m - j]:
                total += _times(budget, r_taylor[j], scaled[m - j])
        if _times(budget, left_scale, n_taylor[m]) != _times(budget, beta, total):
            return False

    # (4) The residue is c_{-1}, the last entry of c.
    return claim == c_pairs[-1]


def _divides(budget: Budget, d_integers: list[int], d_denominator: int, r_integers: list[int], r_denominator: int,
             p: int, q: int, qpowers: list[int], k: int) -> bool:
    """Whether D'/F = (x - p/q)^k R'/E, as D'_i q^k E = W_i F for W = R' * (q x - p)^k, len(D') = len(R') + k."""
    # t_j = binomial(k, j) q^j (-p)^(k-j), the coefficients of (q x - p)^k; only t_k = q^k for p = 0.
    t = [0] * (k + 1)
    if p == 0:
        t[k] = qpowers[k]
    else:
        negative = _powers(budget, -p, k)
        binomial = 1
        for j in range(k + 1):
            budget.tick()
            t[j] = _times(budget, _times(budget, binomial, qpowers[j]), negative[k - j])
            if j < k:
                binomial = _quotient(budget, _times(budget, binomial, k - j), j + 1)
    w = [0] * len(d_integers)
    for i, coefficient in enumerate(r_integers):
        budget.tick()
        if coefficient:
            for j, term in enumerate(t):
                budget.tick()
                if term:
                    w[i + j] += _times(budget, coefficient, term)
    left = _times(budget, qpowers[k], r_denominator)  # q^k E.
    for i, coefficient in enumerate(d_integers):
        budget.tick()
        if _times(budget, coefficient, left) != _times(budget, w[i], d_denominator):
            return False
    return True
