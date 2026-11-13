"""Independent checker of the family mathforge.areas.series.sums: sum_polynomial, geometric_sum, geometric_series and
polynomial_geometric_series.

The checker never reruns the producer's algorithms (Newton interpolation of partial sums, the closed form of a geometric
sum, the triangular solve for Q). It verifies each polynomial claim by an exact polynomial identity, compared
coefficient by coefficient after a Taylor shift (repeated synthetic division by x + 1 or x - 1, in integers), and a
geometric sum by binary splitting without a division. The card (docs/v09/cards/m3-analysis.md) tests the two identities
at deg p + 1 integer points; the coefficient identity implies those point tests, and with the card's degree tests they
imply it, so the conditions are the card's. The shift is chosen for its cost: its integers grow by one bit per degree,
where the values at the points 1, ..., d + 1 grow by log2(d + 1) bits per degree. On the sum of x^1000 it takes 0.08
seconds and 390,000 units, where the point tests took 0.6 seconds, beyond the default budget at an honest charge.

The polynomial argument and value are records of mathforge.polynomial.Polynomial, which the checker identifies by its
module and name (both exact strings) and reads as data. Canonical means a tuple of 1 to MAX_LENGTH Rationals, lowest
degree first, without a trailing zero ((0,) is the zero polynomial), and a Symbol whose fields have their exact types.
A Rational is canonical when its numerator and denominator are exact ints, the denominator >= 1 and their gcd 1; an int
counts as a rational only as the argument first or ratio (kind "rational"). Every argument must have its exact type
(type(x) is int: a bool, a float or a str is no number), the value its canonical form (a Rational, never an int, for
the three rational operations) and the evidence exactly the layout below (a list is no tuple); anything else answers
False, and so does a claim on an input outside the operation's domain, which has no value at all.

The theorem that makes True imply the claim, for each operation. Below, p = polynomial = P/D with P in Z[x] of degree
d, D >= 1 the least common denominator of its coefficients and M = max |P_i|; bits(n) is the bit length of |n|.

sum_polynomial(polynomial) = F, evidence ().
    F is a polynomial in the same Symbol (equal name, domain, assumptions and uid). True requires F = 0 when p = 0;
    otherwise deg F = d + 1, F(0) = 0 and the polynomial identity F(x) - F(x - 1) = p(x). Then F(n) = F(0) +
    sum_{k=1..n} (F(k) - F(k - 1)) = p(1) + ... + p(n) for every integer n >= 0; a polynomial is determined by its
    values at the integers n >= 0, so F is the closed form of the sum. For p = 0 every sum is 0. The true F passes.
    The identity implies the card's point test F(k) - F(k - 1) = p(k), k = 1, ..., d + 1, which needs the degree test:
    F + c x(x - 1)...(x - d - 1), of degree d + 2, has the values of F at 0, ..., d + 1. Here the identity alone
    refuses it, and the degree test refuses it early. (The card's example, F = 0 for p = x - 1, fails the point 2.)
    Sizes. In Newton's form F(x) = sum_{j=1..d+1} a_j binomial(x, j), with a_j = Delta^j F(0) =
    sum_{i<j} (-1)^(j-1-i) binomial(j - 1, i) p(i + 1), as Delta F(x) = p(x + 1). Every F(k) lies in Z/D and
    binomial(x, j) in Z[x]/j!, so the common denominator E of F divides D (d + 1)!. |a_j| <= 2^(j-1) max |p(k)| over
    k = 1, ..., d + 1, |p(k)| <= (d + 1)^d (d + 1) M/D there, and the coefficients of binomial(x, j) are at most 1 in
    absolute value (the unsigned Stirling numbers of the first kind sum to j!), so every coefficient of E F is below
    (d + 1)! 2^(d+1) (d + 1)^d (d + 1) M in absolute value. A claim whose E has more than bits(D) + sum_{j=2..d+1}
    bits(j) bits, or a coefficient of E F more than sum_{j=2..d+1} bits(j) + (d + 1) + d bits(d + 1) + bits(d + 1) +
    bits(M) bits, is refused before the shift runs.

geometric_sum(first, ratio, count) = s, evidence ().
    a = first, r = ratio and n = count >= 0. True requires s = S_n, the sum of the n terms a*r^k, k < n, computed by
    binary splitting without any division: S_0 = 0, S_1 = a, and along the bits of n, S_(2m) = S_m + r^m S_m and
    S_(m+1) = S_m + r^m S_1. Splitting the range of the index proves S_(m+j) = S_m + r^m S_j for the finite sums, so
    by induction over the bits S_n is the sum, for every ratio, also r = 1, where the producer's closed form
    a(1 - r^n)/(1 - r) and the identity S (1 - r) = a (1 - r^n) are void. In integers, with a = an/ad and r = rn/rd:
    S_m = an U_m / (ad rd^(m-1)), where U_1 = 1, U_(2m) = U_m (rn^m + rd^m) and U_(m+1) = U_m rd + rn^m, and s is
    compared with S_n by cross-multiplication. a = 0 or n = 0 gives 0 without computing. The card's resource limit: a
    product that would exceed SPLITTING_BITS * max_integer_bits raises ResourceLimitError before it is formed. A sum
    whose own integers fit max_integer_bits needs no intermediate beyond about 3 * max_integer_bits: U_n is congruent
    to rn^(n-1) modulo rd, so in lowest terms rd^(n-1) can cancel only against an and U_n only against ad, which keeps
    both within twice max_integer_bits and the powers within about as much. The limit thus refuses only sums that do
    not fit, and the producer, which refuses count * bits(ratio) above max_integer_bits, never makes such a claim.

geometric_series(first, ratio) = s, evidence ().
    True requires first = 0 and s = 0, or |ratio| < 1 and s (1 - ratio) = first. For |r| < 1 the partial sums
    a(1 - r^n)/(1 - r) converge to a/(1 - r), the only s with s (1 - r) = a, as 1 - r != 0. For a = 0 every term is 0,
    whatever the ratio. For a != 0 and |r| >= 1, |a r^k| >= |a| > 0, so the terms do not tend to 0 and the series
    diverges: no value passes (the formal a/(1 - r) = -1 of a = 1, r = 2 is refused).

polynomial_geometric_series(polynomial, ratio) = s, evidence (Q,).
    Q is a tuple of canonical Rationals with as many entries as p has coefficients and no trailing zero (deg Q = deg p;
    (0,) for p = 0). True requires, for p = 0: Q = (0,) and s = 0, for every ratio (every term is 0); otherwise
    |r| < 1, s = -Q(0) and the polynomial identity r Q(x + 1) - Q(x) = p(x). Then p(k) r^k = r^(k+1) Q(k + 1) -
    r^k Q(k), the partial sums telescope to sum_{k<n} p(k) r^k = r^n Q(n) - Q(0), and as |r| < 1, r^n Q(n) tends to 0:
    the series converges to -Q(0) = s. For r != 1 the map Q -> r Q(x + 1) - Q(x) keeps the degree, so the true Q has
    degree d and passes. For p != 0 and |r| >= 1 the terms p(k) r^k do not tend to 0 (|p(k)| >= c > 0 for large k), so
    the series diverges and no claim passes: Q = 1 satisfies the identity for p = 1, r = 2, and its formal value -1 is
    refused. The identity implies the card's test at the d + 1 integer points 0, ..., d (Q + E, with E of degree d + 1
    through the points (k, c r^(-k)), passes those points; the degree test and the identity refuse it).
    Sizes. With r = rn/rd and u = rd - rn > 0, the identity solved from the top, Q_i = (rd p_i - rn sum_{j>i}
    binomial(j, i) Q_j)/(rn - rd), puts the common denominator E of Q into D u^(d+1). With t = r/(1 - r), r(1 + Delta)
    - 1 = (r - 1)(1 - t Delta) and Delta^(d+1) p = 0, so Q = -(1/(1 - r)) sum_{j=0..d} t^j Delta^j p; in Newton's
    form p = sum_i c_i binomial(x, i), Delta^j p = sum_{i>=j} c_i binomial(x, i - j) and |c_i| <= 2^i max |p(k)| over
    k = 0, ..., d, where |p(k)| <= max(1, d)^d (d + 1) M/D. As 1/(1 - r) = rd/u and |t| = |rn|/u, every coefficient
    of E Q is below rd (d + 1) max(u, |rn|)^d 2^(d+1) max(1, d)^d (d + 1) M in absolute value. A claim whose E has
    more than bits(D) + (d + 1) bits(u) bits, or a coefficient of E Q more than bits(rd) + bits(d + 1) +
    d bits(max(u, |rn|)) + (d + 1) + d bits(max(1, d)) + bits(d + 1) + bits(M) bits, is refused before the shift runs.

Sizes and work. An argument holding an integer longer than max_integer_bits raises ResourceLimitError, as the
producer's budget refuses it: such a claim is beyond the budget, not wrong, and so is a polynomial argument whose
coefficients have a common denominator longer than max_integer_bits. A value or evidence holding such an integer
answers False: no record built within the budget holds one, and mf.verify refuses such a result before it calls the
checker. The degree tests bound the claimed polynomials by the argument's degree before their coefficients are read,
the denominator bound applies to each coefficient before its gcd runs, and the size bounds apply before the shift, so
an oversized forgery fails fast. Every product, quotient, gcd and shift is charged to the budget before it runs, on the
scale of mathforge.trusted.integers (product_work, division_work; a gcd as the long division of the longer operand by
the shorter plus trusted.upoly.charge_gcd's quadratic term of the shorter; a shift as _shift states), and every loop
charges the budget; work past max_work raises ResourceLimitError, which leaves a claim unverified, never verified. No
builtin pow runs: every power is a chain of charged products.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Assumption, Domain, Rational, Symbol, Truth
from mathforge.trusted.integers import division_work, product_work

POLYNOMIAL = ("mathforge.polynomial", "Polynomial")  # Module and name of the class of polynomial arguments and values.
MAX_LENGTH = 1025  # mathforge.polynomial.MAX_POLYNOMIAL_DEGREE + 1: no Polynomial record holds more coefficients.
MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.
SPLITTING_BITS = 4  # geometric_sum refuses a product longer than this many times max_integer_bits.

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "sum_polynomial": ("polynomial",),
    "geometric_sum": ("first", "ratio", "count"),
    "geometric_series": ("first", "ratio"),
    "polynomial_geometric_series": ("polynomial", "ratio"),
}
_ZERO = (0, 1)  # The pair of the rational 0.
_INTEGER_BITS = "Computation limit exceeded: integer bits"

Pair = tuple[int, int]  # (numerator, denominator) of a canonical rational.
Form = tuple[list[int], int, int]  # (integer coefficients, their common denominator, their largest bit length).
Key = tuple[object, ...]  # The fields of a Symbol that Symbol equality compares.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for arguments other than the operation's and for any malformed value or evidence;
    ResourceLimitError for an argument beyond the budget and for work past it."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = _ARGUMENTS.get(operation)
    if (names is None or len(args) != len(names) or any(type(name) is not str for name in args)
            or any(name not in args for name in names)):
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, open one of the defaults.
        return _CHECKS[operation](budget, *(args[name] for name in names), value, evidence)


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
    """math.gcd, charged before it runs. Lehmer's algorithm first reduces the longer operand modulo the shorter, a long
    division with a quotient of their difference in length (division_work), and then runs quadratically in the
    shorter one: one unit per 64 squared 64-bit words, trusted.upoly.charge_gcd's scale. Measured: 20 microseconds for
    65,536 bits against 64 (65 units), 2 milliseconds for 65,536 against 65,536 (16,417 units)."""
    longer, shorter = max(left.bit_length(), right.bit_length()), min(left.bit_length(), right.bit_length())
    words = shorter >> 6
    budget.tick(1 + (words * words >> 6) + division_work(longer - shorter + 64, shorter))
    return gcd(left, right)


def _shift(budget: Budget, integers: list[int], bits: int, step: int) -> list[int]:
    """The coefficients of P(x + step), step = 1 or -1, for the integer polynomial P (lowest degree first, coefficients
    of at most bits bits), by repeated synthetic division: pass i makes the coefficient of x^i final.

    After any number of passes the entry of x^j is sum_k P_k w(j, k) step^(k - j) with integer weights 0 <= w(j, k) <=
    binomial(k, j) (a pass only adds the next entry, and the final weights are the binomial coefficients), so every
    value stays below 2^(bits + length) in absolute value. The whole double loop is charged before it runs, a quarter
    unit per step and half a unit per step for each 4,096 bits of that bound: measured, a step takes about 50
    nanoseconds on short integers, 0.42 microseconds at 16,384 bits and 1.5 at 65,536 bits."""
    values = list(integers)
    length = len(values)
    steps = length * (length - 1) >> 1
    budget.tick(1 + (steps * (1 + 2 * ((bits + length) >> 12)) >> 2))
    last = length - 1
    if step > 0:
        for i in range(last):
            for j in range(last - 1, i - 1, -1):
                values[j] += values[j + 1]
    else:
        for i in range(last):
            for j in range(last - 1, i - 1, -1):
                values[j] -= values[j + 1]
    return values


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
    integers, bits = [], 0
    for numerator, part in pairs:
        budget.tick()
        integer = numerator if denominator == 1 else _times(budget, numerator, _quotient(budget, denominator, part))
        integers.append(integer)
        bits = max(bits, integer.bit_length())
    return integers, denominator, bits


def _source(budget: Budget, pairs: list[Pair]) -> Form:
    """_integers of a polynomial argument; ResourceLimitError when its common denominator exceeds max_integer_bits:
    the sums and the claim then need integers beyond the budget."""
    form = _integers(budget, pairs, budget.limits.max_integer_bits)
    if form is None:
        raise ResourceLimitError(_INTEGER_BITS)
    return form


# ---------------------------------------------------------------------------------------------------------------
# Reading arguments, values and evidence

def _named(item: object, name: tuple[str, str]) -> bool:
    """Whether the item's class has this module and name, both exact strings (a class may set __module__ to any
    object, whose comparison could answer anything, and a metaclass may compute it)."""
    cls = type(item)
    try:
        module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    except Exception:  # Foreign code raised: no record of the family's classes does.
        return False
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


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


def _record(item: object) -> tuple[object, Key] | None:
    """(coefficients, key of the variable) of an object of the class named POLYNOMIAL whose variable is a Symbol; the
    coefficients are not read yet. None for anything else (a forged record may lack a field, which reads as None)."""
    if not _named(item, POLYNOMIAL):
        return None
    try:  # A class of the same module and name but not the record's may compute its fields.
        variable, coefficients = getattr(item, "variable", None), getattr(item, "coefficients", None)
    except Exception:  # Foreign code raised: the record's plain fields never do.
        return None
    key = _symbol_key(variable)
    return None if key is None else (coefficients, key)


def _pair(budget: Budget, item: object, *, argument: bool = False, integer: bool = False,
          bound: int | None = None) -> Pair | None:
    """(numerator, denominator) of a canonical Rational, or of an int where integer is True (the kind "rational" takes
    one); None for anything else, and for a denominator of more than bound bits, tested before the gcd runs. An
    integer longer than max_integer_bits raises ResourceLimitError in an argument and answers None in a value or
    evidence."""
    if type(item) is int:
        if not integer:
            return None
        numerator, denominator = item, 1
    elif type(item) is Rational:
        numerator = getattr(item, "numerator", None)  # A forged Rational may lack its slots.
        denominator = getattr(item, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            return None
    else:
        return None
    limit = budget.limits.max_integer_bits
    if numerator.bit_length() > limit or denominator.bit_length() > limit:
        if argument:
            raise ResourceLimitError(_INTEGER_BITS)
        return None
    if bound is not None and denominator.bit_length() > bound:
        return None
    if denominator != 1 and _gcd(budget, numerator, denominator) != 1:
        return None
    return numerator, denominator


def _read(budget: Budget, values: object, length: int | None = None, *, argument: bool = False,
          bound: int | None = None) -> list[Pair] | None:
    """The pairs of a canonical coefficient tuple: Rationals, lowest degree first, without a trailing zero, exactly
    length of them (1 to MAX_LENGTH when length is None), the length tested before any entry is read; None for
    anything else. bound limits the bits of every denominator."""
    if type(values) is not tuple or not 0 < len(values) <= MAX_LENGTH or (length is not None and len(values) != length):
        return None
    pairs = []
    for item in values:
        budget.tick()
        pair = _pair(budget, item, argument=argument, bound=bound)
        if pair is None:
            return None
        pairs.append(pair)
    if len(pairs) > 1 and pairs[-1][0] == 0:
        return None
    return pairs


def _polynomial(budget: Budget, item: object) -> tuple[list[Pair], Key] | None:
    """(pairs, key of the variable) of a canonical Polynomial argument; None for anything else."""
    record = _record(item)
    if record is None:
        return None
    pairs = _read(budget, record[0], argument=True)
    return None if pairs is None else (pairs, record[1])


def _factorial_bits(budget: Budget, degree: int) -> int:
    """sum_{j=2..degree+1} bits(j), at least the bit length of (degree + 1)!."""
    budget.tick(degree + 1)
    return sum(j.bit_length() for j in range(2, degree + 2))


# ---------------------------------------------------------------------------------------------------------------
# The operations

def _sum_polynomial(budget: Budget, polynomial: object, value: object, evidence: object) -> bool:
    if type(evidence) is not tuple or evidence:
        return False
    source = _polynomial(budget, polynomial)
    claim = _record(value)
    if source is None or claim is None or claim[1] != source[1]:  # The same Symbol, its uid included.
        return False
    p, cells = source[0], claim[0]
    if p == [_ZERO]:  # Every sum of zeros is 0: F = 0.
        return _read(budget, cells, 1) == [_ZERO]
    degree = len(p) - 1
    if type(cells) is not tuple or len(cells) != degree + 2:  # deg F = d + 1, tested before F is read.
        return False
    p_integers, p_denominator, p_bits = _source(budget, p)
    factorial = _factorial_bits(budget, degree)
    denominators = p_denominator.bit_length() + factorial  # E divides D (d + 1)!.
    sizes = factorial + degree + 1 + degree * (degree + 1).bit_length() + (degree + 1).bit_length() + p_bits
    f = _read(budget, cells, degree + 2, bound=denominators)
    if f is None or f[0] != _ZERO:  # F(0) = 0.
        return False
    form = _integers(budget, f, denominators)
    if form is None or form[2] > sizes:
        return False
    f_integers, f_denominator, f_bits = form
    shifted = _shift(budget, f_integers, f_bits, -1)  # E F(x - 1).
    for j in range(degree + 1):  # F(x) - F(x - 1) = p(x) at x^j, times D E; x^(d+1) cancels in the shift.
        budget.tick()
        if (_times(budget, p_denominator, f_integers[j] - shifted[j])
                != _times(budget, f_denominator, p_integers[j])):
            return False
    return True


def _geometric_sum(budget: Budget, first: object, ratio: object, count: object, value: object,
                   evidence: object) -> bool:
    a = _pair(budget, first, argument=True, integer=True)
    r = _pair(budget, ratio, argument=True, integer=True)
    if a is None or r is None or type(count) is not int or type(evidence) is not tuple or evidence:
        return False
    budget.check_int(count)
    claim = _pair(budget, value)
    if claim is None or count < 0:  # A negative count has no sum.
        return False
    (an, ad), (rn, rd) = a, r
    if an == 0 or count == 0:  # S_1 = 0 makes every S_m 0, and the empty sum is 0.
        return claim == _ZERO
    cap = SPLITTING_BITS * budget.limits.max_integer_bits

    def grown(left: int, right: int) -> int:
        if left.bit_length() + right.bit_length() - 1 > cap:  # The product would be longer than cap.
            raise ResourceLimitError(_INTEGER_BITS)
        return _times(budget, left, right)

    # S_m = an U_m / (ad rd^(m-1)), carried as U_m, power = rn^m and base = rd^m from m = 1: S_1 = a.
    u, power, base = 1, rn, rd
    for bit in bin(count)[3:]:  # The bits after the leading 1, from the top.
        budget.tick()
        u = grown(u, power + base)  # S_(2m) = S_m + r^m S_m.
        power, base = grown(power, power), grown(base, base)
        if bit == "1":  # S_(m+1) = S_m + r^m S_1.
            u = grown(u, rd) + power
            power, base = grown(power, rn), grown(base, rd)
    # s = S_n = an U_n rd / (ad rd^n), cross-multiplied with s = vn / vd (both denominators positive).
    vn, vd = claim
    return _times(budget, _times(budget, vn, ad), base) == _times(budget, _times(budget, vd, an), _times(budget, u, rd))


def _geometric_series(budget: Budget, first: object, ratio: object, value: object, evidence: object) -> bool:
    a = _pair(budget, first, argument=True, integer=True)
    r = _pair(budget, ratio, argument=True, integer=True)
    claim = _pair(budget, value)
    if a is None or r is None or claim is None or type(evidence) is not tuple or evidence:
        return False
    (an, ad), (rn, rd), (vn, vd) = a, r, claim
    if an == 0:  # Every term is 0, whatever the ratio.
        return claim == _ZERO
    if not -rd < rn < rd:  # a != 0 and |r| >= 1: the series diverges.
        return False
    # s (1 - r) = a, that is vn (rd - rn) ad = an vd rd.
    return _times(budget, _times(budget, vn, rd - rn), ad) == _times(budget, _times(budget, an, vd), rd)


def _polynomial_geometric_series(budget: Budget, polynomial: object, ratio: object, value: object,
                                 evidence: object) -> bool:
    source = _polynomial(budget, polynomial)
    r = _pair(budget, ratio, argument=True, integer=True)
    claim = _pair(budget, value)
    if source is None or r is None or claim is None or type(evidence) is not tuple or len(evidence) != 1:
        return False
    p, cells = source[0], evidence[0]
    if type(cells) is not tuple or len(cells) != len(p):  # deg Q = deg p, tested before Q is read.
        return False
    if p == [_ZERO]:  # Every term is 0, whatever the ratio; Q = 0 telescopes to 0.
        return claim == _ZERO and _read(budget, cells, 1) == [_ZERO]
    rn, rd = r
    if not -rd < rn < rd:  # p != 0 and |r| >= 1: the series diverges.
        return False
    degree = len(p) - 1
    p_integers, p_denominator, p_bits = _source(budget, p)
    gap = rd - rn  # u = rd (1 - r) > 0.
    denominators = p_denominator.bit_length() + (degree + 1) * gap.bit_length()  # E divides D u^(d+1).
    sizes = (rd.bit_length() + (degree + 1).bit_length() + degree * max(gap, rn, -rn).bit_length() + degree + 1
             + degree * max(1, degree).bit_length() + (degree + 1).bit_length() + p_bits)
    q = _read(budget, cells, degree + 1, bound=denominators)
    if q is None or claim != (-q[0][0], q[0][1]):  # s = -Q(0).
        return False
    form = _integers(budget, q, denominators)
    if form is None or form[2] > sizes:
        return False
    q_integers, q_denominator, q_bits = form
    shifted = _shift(budget, q_integers, q_bits, 1)  # E Q(x + 1).
    scale = _times(budget, rd, q_denominator)
    for j in range(degree + 1):  # r Q(x + 1) - Q(x) = p(x) at x^j, times rd D E.
        budget.tick()
        difference = _times(budget, rn, shifted[j]) - _times(budget, rd, q_integers[j])
        if _times(budget, p_denominator, difference) != _times(budget, scale, p_integers[j]):
            return False
    return True


_CHECKS: dict[str, Callable[..., bool]] = {
    "sum_polynomial": _sum_polynomial,
    "geometric_sum": _geometric_sum,
    "geometric_series": _geometric_series,
    "polynomial_geometric_series": _polynomial_geometric_series,
}
