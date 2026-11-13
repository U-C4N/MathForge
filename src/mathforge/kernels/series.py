"""Truncated power series over Q: the kernel of series_coefficients (mathforge.areas.series.power_series).

A series of length n is a tuple of n canonical Rationals (lowest terms, a positive denominator), s[k] the coefficient
of t^k, where t = x - a for the center a of an expansion. Every function returns a series of the length of its
operands (derivative one coefficient shorter, integral one longer), truncating every product.

Nothing here is a proof. The family's checker, mathforge.certify.series.power_series, recomputes each coefficient with
the first-order ODE recurrences of mathforge.certify.series.taylor (an N-version check), so this kernel computes them
by other means, and never by those recurrences:

- Composition with Maclaurin series. When an argument u is u0 + c t^v, its value u0 at the center and one monomial,
  f(u) = sum_k m_k c^k t^(v k), where m_k is the k-th Taylor coefficient of f about u0, given by a closed form:
  1/k! for exp; (-1)^j/(2j + 1)! and (-1)^j/(2j)! for sin and cos, without the signs for sinh and cosh;
  (-1)^j/(2j + 1) for atan; binom(2j, j)/(4^j (2j + 1)) for asin; (-1)^(k+1)/k for log about 1; q binom(1/2, k)
  q^(-2k) for sqrt about q^2; binom(m, k) u0^(m - k) for the int power m, the reciprocal included (m = -1); and
  T_j/(2j - 1)! for tan (tanh with alternating signs), T_j the tangent numbers of Brent and Harvey's algorithm
  (tangent_numbers). Each table but tan's goes from one coefficient to the next by a small rational factor, so it
  costs one or two rational operations per coefficient: exp(x) to order 4096 is about 8,000 of them.
- Newton iteration for every other argument, each step doubling the number of exact coefficients: the reciprocal
  y <- y + y (1 - u y); the inverse square root z <- z + z (1 - u z^2)/2, and sqrt(u) = u z; exp, y <- y (1 + u -
  log y); tan and tanh, y <- y - (g(y) - u)(1 + y^2) with g = atan, and y <- y - (g(y) - u)(1 - y^2) with g = atanh.
  The half-angle formulas give the others from w = tan(u/2) and w = tanh(u/2): sin u = 2w/(1 + w^2) and
  cos u = 2/(1 + w^2) - 1, sinh u = 2w/(1 - w^2) and cosh u = 2/(1 - w^2) - 1. log, atan, atanh and asin are
  integrals: log u = int u'/u, atan u = int u'/(1 + u^2), atanh u = int u'/(1 - u^2), asin u = int u' (1 - u^2)^(-1/2).
  An int power squares and multiplies.
- Products: a schoolbook loop when one factor has at most SPARSE nonzero coefficients, otherwise Kronecker
  substitution: each factor over the least common denominator of its coefficients, packed into one integer with a
  slot of fixed width per coefficient, the two integers multiplied by CPython (Karatsuba), the slots read back as
  signed integers, and each coefficient reduced to lowest terms.

Work. The functions charge the active computation budget, if any (the producer always opens one), before each
operation on Python integers runs, by what product_work, gcd_work and division_work estimate for its operands, in
units of about 0.2 microseconds measured with CPython (the checkers' scale): a product of two integers of 65,536 bits
about 4,200 units (0.84 ms measured), of 1 Mbit 340,000 (68 ms), of 16 Mbit 27.7 million (5.8 s). The rational
operations (plus, times, quotient, ...) are written here so that each of their gcds, products and divisions is
charged by its length; a gcd first takes Euclid's remainder step, a long division as cheap as its quotient is short,
so that the nested denominators of factorial-like coefficients (k! and (k + 1)!) cost what they take, while long
coprime ones are charged quadratically. Long coefficients are therefore what limits an expansion: under the default
max_work (2,000,000) a product of two series with coefficients of about 4,000 bits each (factorials near 500!) is near
the limit at a length of 450, while series with short coefficients multiply cheaply at every length up to 4097. A
coefficient whose integers exceed max_integer_bits raises ResourceLimitError.
"""

from __future__ import annotations

from math import gcd

from ..limits import ResourceLimitError, current_budget
from ..model import Rational, trusted_rational

ZERO = trusted_rational(0, 1)
ONE = trusted_rational(1, 1)
_HALF = trusted_rational(1, 2)
_TWO = trusted_rational(2, 1)
SPARSE = 4  # A factor with at most this many nonzero coefficients is multiplied by the schoolbook loop.
_POWERS_OF_THREE = tuple(3 ** k for k in range(64))  # Karatsuba: three half-size products per doubling.
_TOO_LONG = "Computation limit exceeded: integer bits"

Series = tuple  # A tuple of canonical Rationals.


# ---------------------------------------------------------------------------------------------------------------
# Work estimates

def _charge(units: int) -> None:
    """Charge the active budget, if any, before the work runs."""
    budget = current_budget()
    if budget is not None and units:
        budget.tick(units)


def _max_bits() -> int:
    budget = current_budget()
    return 65_536 if budget is None else budget.limits.max_integer_bits


def _words(bits: int) -> int:
    return (bits >> 6) + 1


def product_work(left_bits: int, right_bits: int) -> int:
    """Units of one product of integers of these bit lengths by CPython: schoolbook while the shorter one has at most
    32 words of 64 bits (about 64 word pairs a unit); Karatsuba from there on, 3^k/14 units for two integers of 2^k
    words, interpolated linearly up to 2^(k+1) and charged at the longer length while the shorter one has more than
    half of its words; below that, CPython multiplies the longer one block by block, each block as long as the
    shorter one. Calibrated on 65,536-bit operands (0.84 ms) and checked from 4,096 bits (0.01 ms) to 4 Mbit
    (0.65 s)."""
    large, small = _words(left_bits), _words(right_bits)
    if large < small:
        large, small = small, large
    if small <= 32:
        return 1 + (large * small >> 6)
    if 2 * small > large:
        return 1 + _karatsuba(large)
    return 1 + -(-large // small) * _karatsuba(small)


def _karatsuba(words: int) -> int:
    """Units of a Karatsuba product of two integers of this many 64-bit words."""
    k = words.bit_length() - 1
    if k >= len(_POWERS_OF_THREE):
        raise ResourceLimitError("Computation limit exceeded: work of an integer product")
    return (_POWERS_OF_THREE[k] * (2 * words - (1 << k)) >> k) // 14


def gcd_work(left_bits: int, right_bits: int) -> int:
    """Units of math.gcd: its first remainder is linear in the longer integer, Lehmer's steps after it quadratic in
    the shorter one (measured: 2.1 ms, about 10,000 units, for two integers of 65,536 bits; this charges 16,500)."""
    large, small = _words(left_bits), _words(right_bits)
    if large < small:
        large, small = small, large
    return 1 + ((large + small) >> 4) + (small * small >> 6)


def division_work(dividend_bits: int, divisor_bits: int) -> int:
    """Units of a long division: the words of the quotient times those of the divisor, 32 pairs a unit."""
    dividend, divisor = _words(dividend_bits), _words(divisor_bits)
    return 1 + (max(dividend - divisor + 1, 1) * divisor >> 5)


# ---------------------------------------------------------------------------------------------------------------
# Rational arithmetic, charged

def _gcd(a: int, b: int) -> int:
    """gcd(a, b), charged step by step: Euclid's first remainder (a long division, cheap when the quotient is short,
    as between nested factorials), then Lehmer's algorithm on the divisor and that remainder."""
    a, b = abs(a), abs(b)
    if a < b:
        a, b = b, a
    if not b:
        return a
    _charge(division_work(a.bit_length(), b.bit_length()))
    remainder = a % b
    if not remainder:
        return b
    _charge(gcd_work(b.bit_length(), remainder.bit_length()))
    return gcd(b, remainder)


def _made(numerator: int, denominator: int) -> Rational:
    """The Rational of integers already in lowest terms, denominator > 0; ResourceLimitError when either one exceeds
    max_integer_bits (no coefficient of a value may)."""
    if not numerator:
        return ZERO
    limit = _max_bits()
    if numerator.bit_length() > limit or denominator.bit_length() > limit:
        raise ResourceLimitError(_TOO_LONG)
    return trusted_rational(numerator, denominator)


def reduced(numerator: int, denominator: int) -> Rational:
    """numerator/denominator in lowest terms for denominator > 0."""
    if not numerator:
        return ZERO
    divisor = _gcd(numerator, denominator) if denominator != 1 else 1
    if divisor != 1:
        _charge(division_work(numerator.bit_length(), divisor.bit_length())
                + division_work(denominator.bit_length(), divisor.bit_length()))
        numerator //= divisor
        denominator //= divisor
    return _made(numerator, denominator)


def plus(x: Rational, y: Rational) -> Rational:
    """x + y by Knuth's algorithm (fractions.Fraction adds so): with g = gcd(dx, dy), s = dx/g and r = dy/g, the sum
    is (nx r + ny s)/(s dy), reduced only by the gcd of its numerator with g."""
    nx, dx, ny, dy = x.numerator, x.denominator, y.numerator, y.denominator
    if not ny:
        return x
    if not nx:
        return y
    g = _gcd(dx, dy)
    if g == 1:
        _charge(product_work(nx.bit_length(), dy.bit_length()) + product_work(ny.bit_length(), dx.bit_length())
                + product_work(dx.bit_length(), dy.bit_length()))
        return _made(nx * dy + ny * dx, dx * dy)
    _charge(division_work(dx.bit_length(), g.bit_length()) + division_work(dy.bit_length(), g.bit_length()))
    s, r = dx // g, dy // g
    _charge(product_work(nx.bit_length(), r.bit_length()) + product_work(ny.bit_length(), s.bit_length()))
    t = nx * r + ny * s
    if not t:
        return ZERO
    g2 = _gcd(t, g)
    if g2 != 1:
        _charge(division_work(t.bit_length(), g2.bit_length()) + division_work(dy.bit_length(), g2.bit_length()))
        t //= g2
        dy //= g2
    _charge(product_work(s.bit_length(), dy.bit_length()))
    return _made(t, s * dy)


def negative(x: Rational) -> Rational:
    _charge(1)
    return trusted_rational(-x.numerator, x.denominator) if x.numerator else ZERO


def minus(x: Rational, y: Rational) -> Rational:
    return plus(x, negative(y))


def times(x: Rational, y: Rational) -> Rational:
    """x y with each numerator reduced against the other denominator first, as fractions.Fraction multiplies."""
    nx, dx, ny, dy = x.numerator, x.denominator, y.numerator, y.denominator
    if not nx or not ny:
        return ZERO
    if dy != 1:
        g = _gcd(nx, dy)
        if g != 1:
            _charge(division_work(nx.bit_length(), g.bit_length()) + division_work(dy.bit_length(), g.bit_length()))
            nx, dy = nx // g, dy // g
    if dx != 1:
        g = _gcd(ny, dx)
        if g != 1:
            _charge(division_work(ny.bit_length(), g.bit_length()) + division_work(dx.bit_length(), g.bit_length()))
            ny, dx = ny // g, dx // g
    _charge(product_work(nx.bit_length(), ny.bit_length()) + product_work(dx.bit_length(), dy.bit_length()))
    return _made(nx * ny, dx * dy)


def inverse(x: Rational) -> Rational:
    """1/x for x != 0."""
    _charge(1)
    numerator, denominator = x.numerator, x.denominator
    return trusted_rational(-denominator, -numerator) if numerator < 0 else trusted_rational(denominator, numerator)


def quotient(x: Rational, y: Rational) -> Rational:
    """x/y for y != 0."""
    return times(x, inverse(y))


def times_int(x: Rational, k: int) -> Rational:
    return times(x, trusted_rational(k, 1))


def over_int(x: Rational, k: int) -> Rational:
    """x/k for an int k >= 1."""
    return times(x, trusted_rational(1, k))


def rational_power(value: Rational, exponent: int) -> Rational:
    """value^exponent for an int exponent (value != 0 when it is negative); ResourceLimitError, before any power is
    formed, when the result's integers would exceed max_integer_bits (a power cannot be interrupted)."""
    if exponent == 0:
        return ONE
    numerator, denominator = value.numerator, value.denominator
    if exponent < 0:
        if not numerator:
            raise ZeroDivisionError("a negative power of zero")
        numerator, denominator, exponent = denominator, numerator, -exponent
        if denominator < 0:
            numerator, denominator = -numerator, -denominator
    limit = _max_bits()
    for part in (numerator, denominator):
        # |part| >= 2 has at least (bit_length - 1) * exponent + 1 bits in its power.
        if abs(part) > 1 and (part.bit_length() - 1) * exponent >= limit:
            raise ResourceLimitError(_TOO_LONG)
    _charge(1)
    for part in (numerator, denominator):
        if abs(part) > 1:
            half = part.bit_length() * exponent >> 1
            _charge(exponent.bit_length() * product_work(half, half))  # Its squarings, the last one the longest.
    # The powers of coprime integers are coprime: the result is in lowest terms.
    return _made(numerator ** exponent, denominator ** exponent)


# ---------------------------------------------------------------------------------------------------------------
# Elementary series

def constant(value: Rational, length: int) -> Series:
    """(value, 0, ..., 0)."""
    return (value,) + (ZERO,) * (length - 1)


def identity(center: Rational, length: int) -> Series:
    """The variable x = center + t."""
    return (center,) if length == 1 else (center, ONE) + (ZERO,) * (length - 2)


def valuation(a: Series) -> int:
    """The index of the first nonzero coefficient, the length for the zero series."""
    for k, value in enumerate(a):
        if value.numerator:
            return k
    return len(a)


def _nonzero(a: Series) -> int:
    return sum(1 for value in a if value.numerator)


def add(a: Series, b: Series) -> Series:
    return tuple([plus(x, y) for x, y in zip(a, b)])


def sub(a: Series, b: Series) -> Series:
    return tuple([minus(x, y) for x, y in zip(a, b)])


def neg(a: Series) -> Series:
    return tuple([negative(x) for x in a])


def scale(a: Series, factor: Rational) -> Series:
    if factor == ONE:
        return a
    return tuple([times(x, factor) for x in a])


def shift_constant(a: Series, value: Rational) -> Series:
    """a + value."""
    return (plus(a[0], value),) + a[1:]


def derivative(a: Series) -> Series:
    """The derivative, one coefficient shorter: k a[k] at k - 1."""
    return tuple([times_int(a[k], k) for k in range(1, len(a))])


def integral(a: Series) -> Series:
    """The primitive with constant term 0, one coefficient longer: a[k - 1]/k at k."""
    return (ZERO,) + tuple([over_int(a[k], k + 1) for k in range(len(a))])


def _padded(a, length: int) -> Series:
    """The first length coefficients of a, with zeros after its end."""
    return tuple(a[:length]) + (ZERO,) * (length - len(a))


# ---------------------------------------------------------------------------------------------------------------
# Products

def mul(a: Series, b: Series) -> Series:
    """The product, truncated to the common length."""
    length = len(a)
    if length == 1:
        return (times(a[0], b[0]),)
    _charge(length)  # The scans for the valuations and the counts.
    low_a, low_b = valuation(a), valuation(b)
    if low_a + low_b >= length:
        return (ZERO,) * length
    count_a, count_b = _nonzero(a), _nonzero(b)
    if min(count_a, count_b) <= SPARSE:
        return _schoolbook(a, b) if count_a <= count_b else _schoolbook(b, a)
    size = length - low_a - low_b  # Coefficients low_a + low_b, ..., length - 1 of the product.
    return (ZERO,) * (low_a + low_b) + _kronecker(a[low_a:low_a + size], b[low_b:low_b + size], size)


def _schoolbook(sparse: Series, dense: Series) -> Series:
    """The truncated product, looping over the few nonzero coefficients of sparse."""
    length = len(dense)
    out = [ZERO] * length
    for i, x in enumerate(sparse):
        if x.numerator:
            _charge(length - i)
            for j in range(length - i):
                y = dense[j]
                if y.numerator:
                    out[i + j] = plus(out[i + j], times(x, y))
    return tuple(out)


def _integer_form(series: Series) -> tuple[list[int], int]:
    """(numerators, d) with series[k] = numerators[k]/d and d the least common denominator."""
    common = 1
    for value in series:
        d = value.denominator
        if d == 1 or d == common:
            continue
        _charge(division_work(d.bit_length(), common.bit_length()))
        if not d % common:  # A multiple of every denominator so far, as factorials are of each other.
            common = d
            continue
        _charge(division_work(common.bit_length(), d.bit_length()))
        if common % d:
            divisor = _gcd(common, d)
            _charge(division_work(common.bit_length(), divisor.bit_length())
                    + product_work(common.bit_length() - divisor.bit_length() + 1, d.bit_length()))
            common = common // divisor * d
    numerators = []
    for value in series:
        if not value.numerator:
            numerators.append(0)
            continue
        d = value.denominator
        if d == common:
            numerators.append(value.numerator)
            continue
        _charge(division_work(common.bit_length(), d.bit_length())
                + product_work(value.numerator.bit_length(), common.bit_length() - d.bit_length() + 1))
        numerators.append(value.numerator * (common // d))
    return numerators, common


def _pack(values: list[int], width: int) -> int:
    """sum values[k] 2^(8 width k) for integers below 2^(8 width - 1) in absolute value: each one written in its
    slot of width bytes, a negative one in two's complement, which borrows one from the slot above it."""
    mask = (1 << (width << 3)) - 1
    packed = int.from_bytes(b"".join([(value & mask).to_bytes(width, "little") for value in values]), "little")
    if any(value < 0 for value in values):
        zero, one = bytes(width), (1).to_bytes(width, "little")
        borrows = int.from_bytes(b"".join([one if value < 0 else zero for value in values]), "little")
        packed -= borrows << (width << 3)
    return packed


def _unpack(product: int, width: int, count: int) -> list[int]:
    """The first count signed slots of an integer sum p_k 2^(8 width k) whose every |p_k| is below 2^(8 width - 1):
    adding 2^(8 width - 1) to every slot makes each one a digit in [0, 2^(8 width)), read from the bytes."""
    bits = width << 3
    mask = (1 << (bits * count)) - 1
    half = 1 << (bits - 1)
    offset = int.from_bytes((bytes(width - 1) + b"\x80") * count, "little")
    raw = (((product & mask) + offset) & mask).to_bytes(width * count, "little")
    return [int.from_bytes(raw[i:i + width], "little") - half for i in range(0, width * count, width)]


def _kronecker(a: Series, b: Series, count: int) -> Series:
    """The first count coefficients of the product of a and b (each of length count), by Kronecker substitution."""
    left, left_den = _integer_form(a)
    right, right_den = _integer_form(b)
    while left and not left[-1]:
        left.pop()
    while right and not right[-1]:
        right.pop()
    if not left or not right:
        return (ZERO,) * count
    left_bits = max(max(left), -min(left)).bit_length()
    right_bits = max(max(right), -min(right)).bit_length()
    # |p_k| <= min(len) * max|left| * max|right| < 2^(slot - 1): every slot of the product holds its coefficient.
    slot = left_bits + right_bits + min(len(left), len(right)).bit_length() + 1
    width = (slot + 7) >> 3
    bits = width << 3
    _charge(product_work(bits * len(left), bits * len(right))
            + 2 * (len(left) + len(right) + count) + ((len(left) + len(right) + count) * bits >> 13))
    product = _pack(left, width) * _pack(right, width)
    _charge(product_work(left_den.bit_length(), right_den.bit_length()))
    return _from_integers(_unpack(product, width, count), left_den * right_den)


def _from_integers(numerators: list[int], denominator: int) -> Series:
    """The Rationals numerators[k]/denominator in lowest terms: their content is divided out first, so that each
    coefficient's own gcd works on the shorter integers."""
    common = denominator
    for value in numerators:
        if common == 1:
            break
        if value:
            common = _gcd(common, value)
    if common != 1:
        _charge(len(numerators) * division_work(max(max(numerators), -min(numerators)).bit_length(),
                                                common.bit_length())
                + division_work(denominator.bit_length(), common.bit_length()))
        numerators = [value // common for value in numerators]
        denominator //= common
    return tuple([reduced(value, denominator) for value in numerators])


# ---------------------------------------------------------------------------------------------------------------
# Newton iterations

def _newton_steps(length: int):
    """The precisions (m, 2m capped at length) of a Newton iteration that starts with one exact coefficient."""
    known = 1
    while known < length:
        target = min(2 * known, length)
        yield known, target
        known = target


def reciprocal(u: Series) -> Series:
    """1/u for u[0] != 0: y <- y + y (1 - u y), which doubles the exact coefficients of y. When u y = 1 - e with
    e = O(t^m), u (y + y e) = 1 - e^2 = 1 + O(t^(2m)); y e starts at t^m, so the first m coefficients stay."""
    length = len(u)
    monomial = monomial_of(u)
    if monomial is not None:
        return compose_power(u[0], monomial, -1, length)
    y = [inverse(u[0])]
    for known, target in _newton_steps(length):
        product = mul(u[:target], _padded(y, target))  # 1 + O(t^known)
        error = (ZERO,) * known + neg(product[known:target])  # 1 - u y
        y += mul(_padded(y, target), error)[known:target]
    return tuple(y)


def inverse_sqrt(u: Series, root: Rational) -> Series:
    """u^(-1/2) for u[0] = root^2, root > 0: z <- z + z (1 - u z^2)/2. When u z^2 = 1 - e with e = O(t^m),
    u (z (1 + e/2))^2 = (1 - e)(1 + e + e^2/4) = 1 + O(t^(2m))."""
    length = len(u)
    z = [inverse(root)]
    for known, target in _newton_steps(length):
        padded = _padded(z, target)
        product = mul(u[:target], mul(padded, padded))  # 1 + O(t^known)
        error = (ZERO,) * known + neg(product[known:target])
        z += scale(mul(padded, error)[known:target], _HALF)
    return tuple(z)


def log(u: Series) -> Series:
    """log u for u[0] = 1: int u'/u, the reciprocal one coefficient shorter than u."""
    length = len(u)
    if length == 1:
        return (ZERO,)
    monomial = monomial_of(u)
    if monomial is not None:
        return compose("log", monomial, length)
    return integral(mul(derivative(u), reciprocal(u[:length - 1])))


def exp(u: Series) -> Series:
    """exp u for u[0] = 0: y <- y (1 + u - log y). When y = exp(u) (1 + e) with e = O(t^m), log y = u + e + O(e^2),
    so y (1 + u - log y) = exp(u)(1 + e)(1 - e + O(e^2)) = exp(u)(1 + O(t^(2m)))."""
    length = len(u)
    monomial = monomial_of(u)
    if monomial is not None:
        return compose("exp", monomial, length)
    y = [ONE]
    for known, target in _newton_steps(length):
        padded = _padded(y, target)
        logarithm = log(padded)
        difference = (ZERO,) * known + sub(u[known:target], logarithm[known:target])  # u - log y = O(t^known)
        y += mul(padded, difference)[known:target]
    return tuple(y)


def _inverse_tangent(y: Series, sign: int) -> tuple[Series, Series]:
    """(g(y), 1 + sign y^2) for y[0] = 0, where g = atan for sign 1 and atanh for sign -1:
    g(y) = int y'/(1 + sign y^2)."""
    length = len(y)
    square = mul(y, y)
    factor = shift_constant(square if sign > 0 else neg(square), ONE)
    if length == 1:
        return (ZERO,), factor
    return integral(mul(derivative(y), reciprocal(factor[:length - 1]))), factor


def atan(u: Series) -> Series:
    """atan u for u[0] = 0: int u'/(1 + u^2)."""
    monomial = monomial_of(u)
    if monomial is not None:
        return compose("atan", monomial, len(u))
    return _inverse_tangent(u, 1)[0]


def asin(u: Series) -> Series:
    """asin u for u[0] = 0: int u' (1 - u^2)^(-1/2)."""
    length = len(u)
    monomial = monomial_of(u)
    if monomial is not None:
        return compose("asin", monomial, length)
    head = u[:length - 1]
    return integral(mul(derivative(u), inverse_sqrt(shift_constant(neg(mul(head, head)), ONE), ONE)))


def sqrt(u: Series, root: Rational) -> Series:
    """sqrt u for u[0] = root^2, root > 0: u (u^(-1/2))."""
    monomial = monomial_of(u)
    if monomial is not None:
        return compose_sqrt(root, monomial, len(u))
    return mul(u, inverse_sqrt(u, root))


def tangent(u: Series, sign: int) -> Series:
    """tan u for sign 1 and tanh u for sign -1, u[0] = 0: Newton's method on g(y) = u with g = atan, atanh,
    y <- y - (g(y) - u)(1 + sign y^2), as g'(y) = 1/(1 + sign y^2). When y = y* + e with e = O(t^m),
    g(y) - u = g'(y*) e + O(e^2), and the step leaves y* + O(e^2)."""
    length = len(u)
    monomial = monomial_of(u)
    if monomial is not None:
        return compose("tan" if sign > 0 else "tanh", monomial, length)
    y = [ZERO]
    for known, target in _newton_steps(length):
        padded = _padded(y, target)
        inverse_value, factor = _inverse_tangent(padded, sign)
        difference = (ZERO,) * known + sub(inverse_value[known:target], u[known:target])  # g(y) - u = O(t^known)
        y += neg(mul(difference, factor)[known:target])
    return tuple(y)


def sin_cos(u: Series, hyperbolic: bool = False) -> tuple[Series, Series]:
    """(sin u, cos u), or (sinh u, cosh u), for u[0] = 0. A monomial argument composes the Maclaurin series; any other
    takes the half-angle formulas with w = tan(u/2) (tanh(u/2)): sin u = 2w r and cos u = 2r - 1 with
    r = 1/(1 + w^2), sinh u = 2w r and cosh u = 2r - 1 with r = 1/(1 - w^2)."""
    length = len(u)
    monomial = monomial_of(u)
    if monomial is not None:
        names = ("sinh", "cosh") if hyperbolic else ("sin", "cos")
        return compose(names[0], monomial, length), compose(names[1], monomial, length)
    w = tangent(scale(u, _HALF), -1 if hyperbolic else 1)
    square = mul(w, w)
    two_r = scale(reciprocal(shift_constant(neg(square) if hyperbolic else square, ONE)), _TWO)
    return mul(w, two_r), shift_constant(two_r, negative(ONE))


def power(u: Series, exponent: int) -> Series:
    """u^exponent for an int exponent; a negative one needs u[0] != 0. A monomial argument composes the binomial
    series; any other squares and multiplies (after a reciprocal for a negative exponent)."""
    length = len(u)
    if exponent == 0:
        return constant(ONE, length)
    monomial = monomial_of(u)
    if monomial is not None:
        return compose_power(u[0], monomial, exponent, length)
    if exponent < 0:
        u, exponent = reciprocal(u), -exponent
    low = valuation(u)
    if low and exponent >= -(-length // low):  # low * exponent >= length: every coefficient vanishes.
        return (ZERO,) * length
    result = None
    base = u
    while True:
        if exponent & 1:
            result = base if result is None else mul(result, base)
        exponent >>= 1
        if not exponent:
            return result
        base = mul(base, base)


# ---------------------------------------------------------------------------------------------------------------
# Composition with Maclaurin series

def monomial_of(u: Series) -> tuple[int, Rational] | None:
    """(v, c) when u - u[0] is the monomial c t^v (c = 0 and v = 1 for a constant u), None otherwise."""
    _charge(1 + (len(u) >> 2))
    found = None
    for k in range(1, len(u)):
        if u[k].numerator:
            if found is not None:
                return None
            found = (k, u[k])
    return (1, ZERO) if found is None else found


def _placed(terms: list[Rational], step: int, length: int) -> Series:
    """The series sum terms[k] t^(step k), truncated to length."""
    out = [ZERO] * length
    for k, value in enumerate(terms):
        if step * k >= length:
            break
        out[step * k] = value
    return tuple(out)


def compose(name: str, monomial: tuple[int, Rational], length: int) -> Series:
    """f(c t^v) for f = exp, sin, cos, sinh, cosh, atan, asin, tan or tanh (expanded about 0), and log(1 + c t^v):
    the Maclaurin coefficients of f, the k-th times c^k, at t^(v k)."""
    step, c = monomial
    count = (length - 1) // step + 1  # Terms 0, ..., count - 1 of the Maclaurin series reach the series.
    terms = [ZERO] * count
    if not c.numerator:
        terms[0] = ONE if name in ("exp", "cos", "cosh") else ZERO
        return _placed(terms, step, length)
    square = times(c, c)
    if name == "exp":  # c^k/k!
        terms[0] = term = ONE
        for k in range(1, count):
            term = over_int(times(term, c), k)
            terms[k] = term
    elif name in ("sin", "cos", "sinh", "cosh"):  # c^k/k! on odd or even k, signs alternating for sin and cos.
        start = 1 if name in ("sin", "sinh") else 0
        term = c if start else ONE
        ratio = negative(square) if name in ("sin", "cos") else square
        for k in range(start, count, 2):
            if k > start:
                term = over_int(times(term, ratio), (k - 1) * k)
            terms[k] = term
    elif name == "atan":  # (-1)^j c^(2j+1)/(2j + 1)
        odd = c
        for k in range(1, count, 2):
            if k > 1:
                odd = times(odd, square)
            terms[k] = over_int(odd if k % 4 == 1 else negative(odd), k)
    elif name == "asin":  # binom(2j, j)/(4^j (2j + 1)) c^(2j+1): (2j - 1)^2/((2j)(2j + 1)) c^2 times the last term
        term = c
        for k in range(1, count, 2):
            if k > 1:
                term = over_int(times_int(times(term, square), (k - 2) * (k - 2)), (k - 1) * k)
            terms[k] = term
    elif name == "log":  # log(1 + s) = sum (-1)^(k+1) s^k/k
        power_k = ONE
        for k in range(1, count):
            power_k = times(power_k, c)
            terms[k] = over_int(power_k if k % 2 else negative(power_k), k)
    elif name in ("tan", "tanh"):  # T_j/(2j - 1)! c^(2j-1), alternating in sign for tanh
        numbers = tangent_numbers(count // 2)
        factorial = 1
        odd = c
        for j, number in enumerate(numbers, 1):
            k = 2 * j - 1
            if j > 1:
                _charge(product_work(factorial.bit_length(), 2 * k.bit_length()))
                factorial *= (k - 1) * k
                odd = times(odd, square)
            terms[k] = times(reduced(number if name == "tan" or j % 2 else -number, factorial), odd)
    else:
        raise ValueError(f"no Maclaurin table for {name}")
    return _placed(terms, step, length)


def compose_sqrt(root: Rational, monomial: tuple[int, Rational], length: int) -> Series:
    """sqrt(root^2 + c t^v) = root sum binom(1/2, k) (c/root^2)^k t^(v k), root > 0."""
    step, c = monomial
    count = (length - 1) // step + 1
    terms = [ZERO] * count
    terms[0] = term = root
    if c.numerator:
        ratio = quotient(c, times(root, root))
        for k in range(1, count):
            # binom(1/2, k)/binom(1/2, k - 1) = (1/2 - k + 1)/k = (3 - 2k)/(2k)
            term = over_int(times_int(times(term, ratio), 3 - 2 * k), 2 * k)
            terms[k] = term
    return _placed(terms, step, length)


def compose_power(base: Rational, monomial: tuple[int, Rational], exponent: int, length: int) -> Series:
    """(base + c t^v)^exponent = sum binom(exponent, k) base^(exponent - k) c^k t^(v k) for an int exponent; a
    negative exponent needs base != 0. base = 0 leaves the one term c^exponent t^(v exponent)."""
    step, c = monomial
    count = (length - 1) // step + 1
    terms = [ZERO] * count
    if not base.numerator:
        if exponent < count:  # exponent >= 1 here
            terms[exponent] = rational_power(c, exponent)
        return _placed(terms, step, length)
    terms[0] = term = rational_power(base, exponent)
    if c.numerator:
        ratio = quotient(c, base)
        last = count if exponent < 0 else min(count, exponent + 1)  # binom(m, k) = 0 for k > m >= 0
        for k in range(1, last):
            term = over_int(times_int(times(term, ratio), exponent - k + 1), k)
            terms[k] = term
    return _placed(terms, step, length)


def tangent_numbers(count: int) -> list[int]:
    """T_1, ..., T_count with tan x = sum T_j x^(2j - 1)/(2j - 1)! (1, 2, 16, 272, 7936, ...), by Brent and Harvey's
    algorithm (Fast computation of Bernoulli, Tangent and Secant numbers, 2011): T_j = (j - 1)! to start, then for
    k = 2, ..., count and j = k, ..., count, T_j <- (j - k) T_(j-1) + (j - k + 2) T_j, count^2/2 steps of integer
    arithmetic in all. Each row is charged before it runs, by the length of its longest number."""
    numbers = [0, 1] + [0] * max(count - 1, 0)
    for k in range(2, count + 1):
        numbers[k] = (k - 1) * numbers[k - 1]
    _charge(count * (1 + (numbers[count].bit_length() >> 12)))
    for k in range(2, count + 1):
        _charge((count - k + 1) * (1 + (numbers[count].bit_length() >> 12)))
        for j in range(k, count + 1):
            numbers[j] = (j - k) * numbers[j - 1] + (j - k + 2) * numbers[j]
    return numbers[1:count + 1]


# ---------------------------------------------------------------------------------------------------------------
# What the check will charge

def _rational_words(value: Rational) -> int:
    """The 64-bit words of a Rational's numerator and denominator together, rounded down: the size by which the
    checker's series arithmetic charges an operation (trusted.series.words, restated so that this kernel imports
    nothing the checker's evaluator uses)."""
    return (value.numerator.bit_length() + value.denominator.bit_length()) >> 6


def convolution_work(a: Series, b: Series, length: int, slack: int = 0) -> int:
    """A lower bound of the units that the checker's evaluator (mathforge.certify.series.taylor over
    mathforge.trusted.series) charges for the products a[i] b[j] with i + j < length and both nonzero, each followed by
    its addition to a running sum, as in its truncated products, reciprocals, square roots and ODE loops. Each such
    pair costs it at least its two Rational results (two units) plus trusted.series.operation_work of the product's
    operands and of the sum's, of which rounding down loses less than two units; so at least A B/64 + (A + B)/4
    units for operands of A and B words (trusted.series.words), the sum's own term left out. slack lowers every A by
    that many words (at least 0): the evaluator multiplies j a[j] rather than a[j] for j <= 4096, a Rational at most
    13 bits, hence one word, shorter. Charges the active budget one unit per 16 coefficients read."""
    count_a, count_b = min(len(a), length), min(len(b), length)
    _charge(1 + ((count_a + count_b) >> 4))
    sums, counts = [0], [0]  # Over b[0..m-1]: the words of its nonzero coefficients, and their number.
    for value in b[:count_b]:
        if value.numerator:
            sums.append(sums[-1] + _rational_words(value))
            counts.append(counts[-1] + 1)
        else:
            sums.append(sums[-1])
            counts.append(counts[-1])
    total = 0
    for i in range(count_a):
        value = a[i]
        if value.numerator:
            m = min(count_b, length - i)
            size = max(_rational_words(value) - slack, 0)
            total += size * sums[m] + 16 * (size * counts[m] + sums[m])
    return total >> 6

