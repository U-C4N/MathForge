"""Producers over dense univariate polynomials: calculus and division over Q, gcds, subresultants, square-free
decomposition, Sturm chains and real root isolation.

Rational coefficients come and go as tuples of canonical Rationals (``(Rational(0),)`` for zero); the work is done
on integer polynomials (mathforge.trusted.upoly). Nothing here is trusted: every lean function checks a result with
mathforge.certify before returning it. Work is charged to the active budget in batches (trusted.upoly.charge).
"""

from __future__ import annotations

from math import gcd, isqrt

from ..limits import ResourceLimitError, current_budget
from ..model import Rational, trusted_rational
from ..trusted import caches
from ..trusted import upoly as tp

_ZERO = trusted_rational(0, 1)
_ONE_R = trusted_rational(1, 1)


def _max_bits() -> int:
    budget = current_budget()
    return 65_536 if budget is None else budget.limits.max_integer_bits


def _too_long() -> ResourceLimitError:
    return ResourceLimitError("Computation limit exceeded: integer bits")


# ---------------------------------------------------------------------------------------------------------------
# Calculus over Q

def derivative(coefficients, n: int, widest: int | None = None):
    """The coefficients of the n-th derivative (n >= 0): c_k k (k-1) ... (k-n+1), the falling factorial updated
    from one k to the next. ``widest`` bounds the bit length of the input numerators, when the caller knows it."""
    degree = len(coefficients) - 1
    if n == 0:
        return coefficients
    if n > degree or not coefficients[-1].numerator:
        return (_ZERO,)
    tp.charge(degree + n)
    limit = _max_bits()
    falling = 1
    for factor in range(2, n + 1):
        falling *= factor
    if widest is None:
        widest = max(value.numerator.bit_length() for value in coefficients)
    # No result is longer than the longest input numerator times the largest falling factorial, below degree**n.
    checked = widest + n * degree.bit_length() > limit
    result = []
    append = result.append
    new, set_numerator, set_denominator = tp.RATIONAL_SLOTS
    k = n
    for value in coefficients[n:]:
        if k > n:
            falling = falling * k // (k - n)
        k += 1
        numerator = value.numerator
        if not numerator:
            append(_ZERO)
            continue
        denominator = value.denominator
        if denominator == 1:
            numerator *= falling
        else:
            divisor = gcd(falling, denominator)
            numerator *= falling // divisor
            denominator //= divisor
        if checked and numerator.bit_length() > limit:
            raise _too_long()
        item = new(Rational)
        set_numerator(item, numerator)
        set_denominator(item, denominator)
        append(item)
    return tuple(result)


def antiderivative(coefficients):
    """The coefficients of the antiderivative with constant term 0: c_k / (k + 1) after c_0."""
    if not coefficients[-1].numerator:
        return (_ZERO,)
    tp.charge(len(coefficients))
    limit = _max_bits()
    result = [_ZERO]
    append = result.append
    for k, value in enumerate(coefficients, 1):
        numerator = value.numerator
        if not numerator:
            append(_ZERO)
            continue
        divisor = gcd(numerator, k)
        denominator = value.denominator * (k // divisor)
        if denominator.bit_length() > limit:
            raise _too_long()
        append(trusted_rational(numerator // divisor, denominator))
    return tuple(result)


def definite_integral(coefficients, lower, upper):
    """The integral of the polynomial from lower to upper (Rationals): A(upper) - A(lower) for the antiderivative A,
    each value by integer Horner on the homogenized form."""
    limit = _max_bits()
    widest = max(lower.numerator.bit_length(), lower.denominator.bit_length(),
                 upper.numerator.bit_length(), upper.denominator.bit_length())
    if (len(coefficients) + 1) * widest > limit:
        raise _too_long()  # Powers of the bounds alone would outgrow the integer limit.
    numerators, denominator = tp.from_rationals(antiderivative(coefficients))
    value = tp.value_at(numerators, upper) - tp.value_at(numerators, lower)
    if denominator != 1:
        value = value / denominator
    if value.numerator.bit_length() > limit or value.denominator.bit_length() > limit:
        raise _too_long()
    return value


# ---------------------------------------------------------------------------------------------------------------
# Greatest common divisors over Z

def _digits(value: int, bits: int) -> tuple[int, ...]:
    """The polynomial P with P(2**bits) = value and every coefficient in [-2**(bits-1), 2**(bits-1))."""
    result = []
    mask, half, full = (1 << bits) - 1, 1 << (bits - 1), 1 << bits
    while value:
        digit = value & mask
        value >>= bits
        if digit >= half:
            digit -= full
            value += 1
        result.append(digit)
    return tuple(result)


def _identity(product: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    """product == left * right, by one Kronecker evaluation."""
    if not left or not right:
        return not product
    if len(product) != len(left) + len(right) - 1:
        return False
    bits = tp.kronecker_bits(tp.max_bits(product),
                             tp.max_bits(left) + tp.max_bits(right) + min(len(left), len(right)).bit_length())
    return tp.kronecker(product, bits) == tp.kronecker(left, bits) * tp.kronecker(right, bits)


def heuristic_gcd(f: tuple[int, ...], g: tuple[int, ...]):
    """(h, u, v) with f = h u and g = h v, h the primitive gcd with a positive leading coefficient, for nonconstant
    primitive f and g with positive leading coefficients; None when the heuristic does not settle.

    The heuristic gcd of Char, Geddes and Gonnet at the points 2**k: the integer gcd of f(2**k) and g(2**k), read as
    a polynomial in balanced base 2**k, is the polynomial gcd once 2**k is large enough; the cofactors follow from
    the quotients. Each candidate is kept only if both product identities hold."""
    norm_f = max(max(f), -min(f))
    norm_g = max(max(g), -min(g))
    bound = 2 * min(norm_f, norm_g) + 29
    start = max(min(bound, 99 * isqrt(bound)), 2 * min(norm_f // f[-1], norm_g // g[-1]) + 4)
    bits = start.bit_length() + 1
    for _ in range(6):
        tp.charge(len(f) + len(g))
        value_f, value_g = tp.kronecker(f, bits), tp.kronecker(g, bits)
        if value_f and value_g:
            common = gcd(value_f, value_g)
            h = tp.primitive(_digits(common, bits))
            if h:
                value_h = tp.kronecker(h, bits)
                quotient_f, rest_f = divmod(value_f, value_h)
                quotient_g, rest_g = divmod(value_g, value_h)
                if not rest_f and not rest_g:
                    u, v = _digits(quotient_f, bits), _digits(quotient_g, bits)
                    if _identity(f, h, u) and _identity(g, h, v):
                        return h, u, v
        bits = bits * 3 // 2 + 8
    return None


def euclid_gcd(f: tuple[int, ...], g: tuple[int, ...]) -> tuple[int, ...]:
    """The primitive gcd (positive leading coefficient) of nonzero integer polynomials, by the primitive
    pseudo-remainder sequence."""
    a, b = tp.primitive(f), tp.primitive(g)
    if len(a) < len(b):
        a, b = b, a
    while b:
        remainder = tp.pseudo_remainder(a, b)
        a, b = b, tp.primitive(remainder)
        for value in (a[-1] if a else 0,):
            if value.bit_length() > _max_bits():
                raise _too_long()
    return tp.primitive(a)


def gcd_cofactors(f: tuple[int, ...], g: tuple[int, ...]):
    """(h, u, v): h the primitive gcd of the nonzero primitive integer polynomials f and g (positive leading
    coefficients), with f = h u and g = h v."""
    if len(f) == 1 or len(g) == 1:
        return (1,), f, g
    found = heuristic_gcd(f, g)
    if found is not None:
        return found
    h = euclid_gcd(f, g)
    return h, tp.exact_quotient(f, h), tp.exact_quotient(g, h)


def rational_primitive(coefficients) -> tuple[int, ...]:
    """The primitive integer polynomial (positive leading coefficient) of a nonzero rational one."""
    return tp.primitive(tp.from_rationals(coefficients)[0])


def square_free_integer(f: tuple[int, ...]) -> list[tuple[tuple[int, ...], int]]:
    """[(a_i, i), ...] with f = c * prod a_i**i, the a_i primitive, square-free, pairwise coprime and nonconstant, i
    increasing: v0.7's Polynomial.square_free on primitive integer polynomials (f primitive, positive leading
    coefficient, degree >= 1)."""
    repeated, remaining, _ = gcd_cofactors(f, tp.primitive(tp.derivative(f)))
    factors = []
    multiplicity = 1
    while len(remaining) > 1:
        common, factor, rest = gcd_cofactors(remaining, repeated)
        if len(factor) > 1:
            factors.append((factor, multiplicity))
        remaining, repeated = common, rest
        multiplicity += 1
    return factors


def _rational_divmod(a: list, b: list) -> tuple[list, list]:
    """Quotient and remainder of Rational coefficient lists without trailing zeros ([] is zero), b nonzero."""
    if len(a) < len(b):
        return [], a
    remainder = list(a)
    quotient = [_ZERO] * (len(a) - len(b) + 1)
    inverse = 1 / b[-1]
    lower = b[:-1]
    for shift in range(len(a) - len(b), -1, -1):
        top = remainder.pop()
        if top.numerator:
            factor = top * inverse
            quotient[shift] = factor
            for index, value in enumerate(lower, shift):
                if value.numerator:
                    remainder[index] = remainder[index] - factor * value
    while remainder and not remainder[-1].numerator:
        remainder.pop()
    return quotient, remainder


def _rational_sub_product(a: list, q: list, s: list) -> list:
    """a - q * s for Rational coefficient lists."""
    result = list(a) + [_ZERO] * max(0, len(q) + len(s) - 1 - len(a))
    for i, x in enumerate(q):
        if x.numerator:
            for j, y in enumerate(s, i):
                if y.numerator:
                    result[j] = result[j] - x * y
    while result and not result[-1].numerator:
        result.pop()
    return result


def xgcd_rational(f, g):
    """(h, s, t) of v0.7's Polynomial.xgcd for nonzero Rational coefficient tuples: the extended Euclidean algorithm
    over Q, scaled so that h is monic. Coefficient lists, [] for zero."""
    a, b = list(f), list(g)
    old_s, s, old_t, t = [_ONE_R], [], [], [_ONE_R]
    tp.charge(len(f) * len(g))
    while b:
        quotient, remainder = _rational_divmod(a, b)
        a, b = b, remainder
        old_s, s = s, _rational_sub_product(old_s, quotient, s)
        old_t, t = t, _rational_sub_product(old_t, quotient, t)
    inverse = 1 / a[-1]
    return ([value * inverse for value in a], [value * inverse for value in old_s],
            [value * inverse for value in old_t])


# ---------------------------------------------------------------------------------------------------------------
# Resultants

def subresultant(f: tuple[int, ...], g: tuple[int, ...]) -> int:
    """Res(f, g) of nonzero integer polynomials by Brown's subresultant PRS, as v0.7's
    polynomial_exact._subresultant_integer computes it: pseudo-remainders and exact scalar divisions in Z."""
    n, m = len(f) - 1, len(g) - 1
    sign = 1
    if n < m:
        f, g, n, m = g, f, m, n
        sign = -1 if n * m % 2 else 1
    if m == 0:
        return sign * g[0] ** n
    limit = _max_bits()
    drop = n - m
    remainder = tp.pseudo_remainder(f, g)
    if (drop + 1) % 2:
        remainder = tp.neg(remainder)
    leading = g[-1]
    previous = -(leading ** drop)
    while remainder:
        next_degree = len(remainder) - 1
        f, g, m, drop = g, remainder, next_degree, m - next_degree
        divisor = -leading * previous ** drop
        remainder = tp.exact_quotient(tp.pseudo_remainder(f, g), (divisor,)) if divisor != 1 else \
            tp.pseudo_remainder(f, g)
        leading = g[-1]
        if leading.bit_length() > limit:
            raise _too_long()
        if drop > 1:
            numerator = (-leading) ** drop
            denominator = previous ** (drop - 1)
            previous, rest = divmod(numerator, denominator)
            if rest:
                raise ArithmeticError("Nonexact scalar subresultant division")
        else:
            previous = -leading
    return sign * -previous if m == 0 else 0


def resultant_rational(f, g):
    """Res(f, g) of rational polynomials as v0.7's polynomial_exact.resultant defines it: the subresultant of the
    primitive integer polynomials, times (lc f / lc F)**deg g (lc g / lc G)**deg f. Returns (value, F, G, Res(F, G))."""
    integer_f, integer_g = rational_primitive(f), rational_primitive(g)
    value = subresultant(integer_f, integer_g)
    scale_f = f[-1] / integer_f[-1]
    scale_g = g[-1] / integer_g[-1]
    result = trusted_rational(value, 1) * scale_f ** (len(g) - 1) * scale_g ** (len(f) - 1)
    return result, integer_f, integer_g, value


# ---------------------------------------------------------------------------------------------------------------
# Division over Q

def pseudo_divmod(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """(Q, R, e) with lc(q)**e p = Q q + R over Z and deg R < deg q, e = deg p - deg q + 1 (q nonzero, p at
    least as long as q)."""
    m = len(q)
    lead = q[-1]
    exponent = len(p) - m + 1
    remainder = list(p)
    quotient = [0] * exponent
    lower = q[:-1]
    tp.charge(exponent * len(p))
    for shift in range(len(p) - m, -1, -1):
        top = remainder.pop()
        if lead != 1:
            quotient = [lead * value for value in quotient]
            remainder = [lead * value for value in remainder]
        quotient[shift] = top
        if top:
            for index, value in enumerate(lower, shift):
                remainder[index] -= top * value
    while remainder and not remainder[-1]:
        remainder.pop()
    return tuple(quotient), tuple(remainder), exponent


def divmod_rational(f, g):
    """(quotient, remainder) coefficients of f by g over Q (g nonzero): Rationals, zero as (0,)."""
    numerators_f, denominator_f = tp.from_rationals(f)
    numerators_g, denominator_g = tp.from_rationals(g)
    if len(numerators_f) < len(numerators_g):
        return (_ZERO,), f
    quotient, remainder, exponent = pseudo_divmod(numerators_f, numerators_g)
    lead = numerators_g[-1]
    # lead**e f = Q G' / dg ... with f = F/df and g = G/dg: q = Q dg / (lead**e df), r = R / (lead**e df).
    scale = lead ** exponent * denominator_f
    numerator_scale = denominator_g
    if scale < 0:
        scale, numerator_scale, sign = -scale, -numerator_scale, -1
    else:
        sign = 1
    if numerator_scale != 1:
        quotient = tuple([value * numerator_scale for value in quotient])
    if sign < 0:
        remainder = tuple([-value for value in remainder])
    return tp.rationals(quotient, scale), tp.rationals(remainder, scale)
