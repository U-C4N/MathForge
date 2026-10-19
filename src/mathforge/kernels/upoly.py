"""Producers over dense univariate polynomials: calculus and division over Q, gcds, subresultants, square-free
decomposition, Sturm chains and real root isolation.

Rational coefficients come and go as tuples of canonical Rationals (``(Rational(0),)`` for zero); the work is done
on integer polynomials (mathforge.trusted.upoly). Nothing here is trusted: every lean function checks a result with
mathforge.certify before returning it. Work is charged to the active budget in batches (trusted.upoly.charge).
"""

from __future__ import annotations

from fractions import Fraction
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
        remainder = tp.pseudo_remainder(f, g)
        if divisor != 1:
            quotient = tuple([value // divisor for value in remainder])
            if any(value % divisor for value in remainder):
                raise ArithmeticError("Nonexact subresultant scaling division")
            remainder = quotient
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
    numerator, denominator = resultant_scale(f, g, integer_f, integer_g)
    return tp.scaled_rational(value, numerator, denominator), integer_f, integer_g, value


def discriminant_rational(f):
    """disc(f) of a rational polynomial of degree n >= 1 as v0.7 defines it, (-1)**(n(n-1)/2) Res(f, f') / lc(f),
    computed on the primitive integer form F with f = c F: Res(f, f') = c**(2n-1) Res(F, F'), so
    disc(f) = (-1)**(n(n-1)/2) c**(2n-2) Res(F, F') / lc(F)."""
    integer = rational_primitive(f)
    value = subresultant(integer, tp.derivative(integer))
    degree = len(f) - 1
    lead = f[-1]
    numerator, denominator = lead.numerator, lead.denominator * integer[-1]  # c = lc(f) / lc(F)
    if max(numerator.bit_length(), denominator.bit_length()) * (2 * degree - 2) > _max_bits():
        raise _too_long()
    if degree * (degree - 1) // 2 % 2:
        value = -value
    return tp.scaled_rational(value, numerator ** (2 * degree - 2), denominator ** (2 * degree - 2) * integer[-1])


def resultant_scale(f, g, integer_f, integer_g) -> tuple[int, int]:
    """(numerator, denominator) of (lc f / lc F)**deg g (lc g / lc G)**deg f, refused (as v0.7's rational powers
    are) when a power would outgrow the integer limit."""
    lead_f, lead_g = f[-1], g[-1]
    degree_f, degree_g = len(f) - 1, len(g) - 1
    limit = _max_bits()
    for lead, integer, exponent in ((lead_f, integer_f[-1], degree_g), (lead_g, integer_g[-1], degree_f)):
        if exponent > 1 and max(lead.numerator.bit_length(), (lead.denominator * integer).bit_length()) * exponent > limit:
            raise _too_long()
    return (lead_f.numerator ** degree_g * lead_g.numerator ** degree_f,
            (lead_f.denominator * integer_f[-1]) ** degree_g * (lead_g.denominator * integer_g[-1]) ** degree_f)


# ---------------------------------------------------------------------------------------------------------------
# Sturm chains and real root isolation, with v0.7's normalization and bisection (root_isolation.py)

_STURM = caches.BoundedCache("kernels.sturm", 512)


def squarefree_part(p: tuple[int, ...]) -> tuple[int, ...]:
    """The primitive square-free part (positive leading coefficient) of a nonzero integer polynomial; (1,) for a
    constant."""
    if len(p) <= 1:
        return (1,)
    f = tp.primitive(p)
    common, part, _ = gcd_cofactors(f, tp.primitive(tp.derivative(f)))
    return part


def sturm_chain(p: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """v0.7's signed Sturm chain of a primitive square-free integer polynomial with positive leading coefficient:
    p, p', then each next member the primitive part, sign kept, of minus the remainder of the two before it
    (pseudo-remainders, with the sign of lc**k undone), until a remainder vanishes."""
    return _STURM.get(p, lambda: _sturm_chain(p))


def _sturm_chain(p):
    if len(p) <= 1:
        return (p,)
    chain = [p, tp.derivative(p)]
    limit = _max_bits()
    while True:
        a, b = chain[-2], chain[-1]
        remainder = tp.pseudo_remainder(a, b)
        if not remainder:
            break
        if b[-1] > 0 or (len(a) - len(b) + 1) % 2 == 0:
            remainder = tp.neg(remainder)  # -rem up to the positive factor lc(b)**k
        divisor = gcd(*remainder)
        if divisor != 1:
            remainder = tuple([value // divisor for value in remainder])
        if max(remainder[-1].bit_length(), remainder[0].bit_length()) > limit:
            raise _too_long()
        chain.append(remainder)
        if len(remainder) == 1:
            break
    return tuple(chain)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _value(p: tuple[int, ...], numerator: int, denominator: int) -> int:
    """p(numerator / denominator) * denominator**deg p, an integer with the sign of p at the point."""
    if denominator == 1:
        value = 0
        for coefficient in reversed(p):
            value = value * numerator + coefficient
        return value
    value, power = 0, 1
    for coefficient in reversed(p):
        value = value * numerator + coefficient * power
        power *= denominator
    return value


def endpoint(chain, numerator: int, denominator: int) -> tuple[int, bool]:
    """(sign variations of the chain at the point, whether the point is a root of chain[0]), zeros skipped and the
    first member's zero not counted, as root_isolation's endpoint and _variations count them."""
    tp.charge(sum(len(member) for member in chain))
    first = _sign(_value(chain[0], numerator, denominator))
    previous, variations = first, 0
    for member in chain[1:]:
        sign = _sign(_value(member, numerator, denominator))
        if sign:
            if previous and previous != sign:
                variations += 1
            previous = sign
    return variations, first == 0


def variations_at_infinity(chain, positive: bool) -> int:
    signs = []
    for member in chain:
        sign = 1 if member[-1] > 0 else -1
        if not positive and (len(member) - 1) % 2:
            sign = -sign
        signs.append(sign)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def count_open(chain, lower, upper) -> int:
    """The number of distinct roots of chain[0] in the open interval (lower, upper); a bound is (numerator,
    denominator) or None for an infinite end."""
    low = variations_at_infinity(chain, False) if lower is None else endpoint(chain, *lower)[0]
    if upper is None:
        return low - variations_at_infinity(chain, True)
    high, zero = endpoint(chain, *upper)
    return low - high - zero


def root_bound(p: tuple[int, ...]) -> int:
    """root_isolation._root_bound: 2 + max ceil(|c_i / lc|) over the lower coefficients."""
    lead = abs(p[-1])
    largest = 0
    for value in p[:-1]:
        ceiling = -(-abs(value) // lead)
        if ceiling > largest:
            largest = ceiling
    return largest + 2


def isolate(p: tuple[int, ...]) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """root_isolation.isolate_real_roots of the polynomial whose square-free primitive part is p (positive leading
    coefficient): the same bisection from (-bound, bound), so the same intervals, as ((num, den), (num, den)) pairs
    in lowest terms, sorted; a point interval for a root met at a midpoint."""
    if len(p) <= 1:
        return ()
    chain = sturm_chain(p)
    budget = current_budget()
    bound = root_bound(p)
    lower_state = ((-bound, 1),) + endpoint(chain, -bound, 1)
    upper_state = ((bound, 1),) + endpoint(chain, bound, 1)
    pending = [(lower_state, upper_state)]
    intervals = []
    while pending:
        if budget is not None:
            budget.refine()
        lower_state, upper_state = pending.pop()
        lower, lower_variations, lower_zero = lower_state
        upper, upper_variations, upper_zero = upper_state
        count = lower_variations - upper_variations - upper_zero
        if not count:
            continue
        if count == 1 and not lower_zero and not upper_zero:
            intervals.append((lower, upper))
            continue
        numerator = lower[0] * upper[1] + upper[0] * lower[1]
        denominator = 2 * lower[1] * upper[1]
        divisor = gcd(numerator, denominator)
        middle = (numerator // divisor, denominator // divisor)
        middle_variations, middle_zero = endpoint(chain, *middle)
        middle_state = (middle, middle_variations, middle_zero)
        if middle_zero:
            intervals.append((middle, middle))
        left_count = lower_variations - middle_variations - middle_zero
        right_count = middle_variations - upper_variations - upper_zero
        if left_count + right_count + middle_zero != count:
            raise ArithmeticError("Sturm subdivision did not preserve the root count")
        if right_count:
            pending.append((middle_state, upper_state))
        if left_count:
            pending.append((lower_state, middle_state))
    intervals.sort(key=_rational_key)
    return tuple(intervals)


def _rational_key(interval):
    """Sort key of an interval by its lower end (numerator, denominator > 0), ordered as the rational."""
    return Fraction(*interval[0])


def midpoint(lower, upper):
    numerator = lower[0] * upper[1] + upper[0] * lower[1]
    denominator = 2 * lower[1] * upper[1]
    divisor = gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def refine(chain, lower, upper):
    """root_isolation.refine_interval: one bisection of an isolating interval of chain[0]'s root."""
    budget = current_budget()
    if budget is not None:
        budget.refine()
    if lower == upper:
        return lower, upper
    middle = midpoint(lower, upper)
    if not _value(chain[0], *middle):
        return middle, middle
    return (lower, middle) if count_open(chain, lower, middle) else (middle, upper)


# ---------------------------------------------------------------------------------------------------------------
# Signs of closed real algebraic expressions by fixed-point interval arithmetic

class OutOfScope(Exception):
    """An expression node the interval evaluator does not handle."""


def _enclosure(expression, bits: int):
    """Integers (lo, hi) with lo <= expression * 2**bits <= hi, for a tree of Rational, Sqrt, Add, Mul and Pow
    nodes (outward rounding at every step)."""
    from ..model import Add, Mul, Pow, Sqrt
    kind = type(expression)
    if kind is Rational:
        scaled = expression.numerator << bits
        low = scaled // expression.denominator
        return low, low + (1 if scaled % expression.denominator else 0)
    if kind is Sqrt:
        radicand = expression.radicand
        if type(radicand) is not Rational or radicand.numerator < 0:
            raise OutOfScope("radicand")
        low = isqrt((radicand.numerator << (2 * bits)) // radicand.denominator)
        return low, low + 1
    if kind is Add:
        low = high = 0
        for term in expression.terms:
            a, b = _enclosure(term, bits)
            low += a
            high += b
        return low, high
    if kind is Mul:
        low = high = 1 << bits
        for factor in expression.factors:
            a, b = _enclosure(factor, bits)
            products = (low * a, low * b, high * a, high * b)
            low, high = min(products) >> bits, -((-max(products)) >> bits)
        return low, high
    if kind is Pow:
        exponent = expression.exponent
        if type(exponent) is not int or exponent < 0 or exponent > 4096:
            raise OutOfScope("exponent")
        a, b = _enclosure(expression.base, bits)
        low = high = 1 << bits
        for _ in range(exponent):
            products = (low * a, low * b, high * a, high * b)
            low, high = min(products) >> bits, -((-max(products)) >> bits)
        return low, high
    raise OutOfScope(kind.__name__)


def separate(left, right, *, start: int = 64, limit: int = 4096):
    """(sign of left - right, the precision in bits that separates their enclosures), doubling the precision from
    start; None when the enclosures still overlap at the limit (the numbers may be equal)."""
    bits = start
    while bits <= limit:
        tp.charge(bits)
        a, b = _enclosure(left, bits)
        c, d = _enclosure(right, bits)
        if b < c:
            return -1, bits
        if d < a:
            return 1, bits
        bits *= 2
    return None


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
