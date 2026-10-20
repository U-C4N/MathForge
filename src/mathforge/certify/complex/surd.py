"""Checks of answers whose parts are quadratic surds: elements of a field K = Q(sqrt b_1, ..., sqrt b_t)
(mathforge.trusted.surd.Field), each part named by a Rational or a RealAlgebraicRoot as the structured path names it.

* ``represents``: a descriptor denotes its element: its polynomial vanishes at the element and at every conjugate,
  its degree is the number of conjugates, and its index is the number of conjugates below the element, decided by
  interval enclosures of the square roots (the producer decides it exactly, by the signs of elements).
* ``check_product``, ``check_quotient``, ``check_power``: a product against the three-multiplication product, a
  quotient multiplied back, a power recomputed left to right (the producer's runs right to left) and, for a negative
  exponent, multiplied back.
* ``check_quadratic_roots``: each root substituted into the polynomial, the roots in strictly increasing order.
* ``check_algebraic``: an expression read again on its own (square roots recognized as rational multiples of basis
  elements that square to the radicand and are not negative, quadratic roots checked against their descriptors,
  sums and products folded from the right, powers as repeated products) and by interval arithmetic on the
  expression alone.

A wrong field multiplication could confirm a product computed with it, so every product, quotient, power and root is
also checked by ``consistent_*``: fixed-point interval arithmetic (64 fractional bits, rounded outward) on rectangles
of the operands' 64-bit enclosures (Field.enclosure, a sum of coefficients times square-root bounds) must meet the
answer's own rectangle. These never multiply two field elements.
"""

from __future__ import annotations

from math import gcd, isqrt

from ...limits import ComputationLimits
from ...model import Add, Mul, Pow, Rational, Sqrt

_BITS = ComputationLimits().max_integer_bits
_FRACTION = 64  # Fractional bits of the fixed-point rectangles.
_RECTANGLE_BITS = 1 << 17  # A rectangle whose integers grow past this is refused rather than followed.


def _is_root(value) -> bool:
    """A RealAlgebraicRoot, recognized by its module and qualified name (this package does not import it)."""
    kind = type(value)
    return kind.__module__ == "mathforge.algebraic" and kind.__qualname__ == "RealAlgebraicRoot"


def _canonical(value, limit: int = _BITS) -> bool:
    if type(value) is not Rational:
        return False
    n, d = value.numerator, value.denominator
    return (type(n) is int and type(d) is int and d > 0 and n.bit_length() <= limit and d.bit_length() <= limit
            and (d == 1 or gcd(n, d) == 1))


# ---------------------------------------------------------------------------------------------------------------
# Names of elements

def represents(field, x, part) -> bool:
    """Whether the Rational or RealAlgebraicRoot ``part`` denotes the element x of the field."""
    if type(part) is Rational:
        return _canonical(part) and x[0] == part and not any(c.numerator for c in x[1:])
    if not _is_root(part):
        return False
    coefficients, index = part.integer_coefficients, part.real_index
    if (type(coefficients) is not tuple or len(coefficients) < 3 or type(index) is not int
            or any(type(c) is not int for c in coefficients) or coefficients[-1] <= 0 or gcd(*coefficients) != 1):
        return False
    orbit = field.orbit(x)
    if len(orbit) != len(coefficients) - 1 or not 0 <= index < len(orbit):
        return False
    below = 0
    for y in orbit:
        if field.horner(coefficients, y) != field.zero:
            return False
        order = field.compare(x, y)
        if order is None:
            return False
        below += order > 0
    return below == index


# ---------------------------------------------------------------------------------------------------------------
# Products, quotients and powers

def three_product(field, z, w):
    """(a + b i)(c + d i) by three multiplications of field elements: c (a + b), a (d - c) and b (c + d)."""
    (a, b), (c, d) = z, w
    k1, k2, k3 = field.mul(c, field.add(a, b)), field.mul(a, field.sub(d, c)), field.mul(b, field.add(c, d))
    return field.sub(k1, k3), field.add(k1, k2)


def check_product(field, z, w, product) -> bool:
    """product = z w, against the three-multiplication product and the interval product."""
    return tuple(product) == three_product(field, z, w) and consistent_product(field, z, w, product)


def check_quotient(field, z, w, quotient) -> bool:
    """quotient = z / w: the quotient times w is z, exactly and in interval arithmetic."""
    (a, b), (c, d) = z, w
    real, imag = quotient
    return (field.sub(field.mul(real, c), field.mul(imag, d)) == a
            and field.add(field.mul(real, d), field.mul(imag, c)) == b
            and consistent_product(field, quotient, w, z))


def check_power(field, z, n: int, answer) -> bool:
    """answer = z**n for an int n != 0: the power recomputed by left-to-right binary exponentiation with the
    three-multiplication product, times the answer for a negative n; and the same in interval arithmetic."""
    if type(n) is not int or n == 0:
        return False
    power = tuple(z)
    for bit in bin(abs(n))[3:]:
        power = three_product(field, power, power)
        if bit == "1":
            power = three_product(field, power, z)
    if n > 0:
        return tuple(answer) == power and consistent_power(field, z, n, answer)
    return (three_product(field, answer, power) == (field.one, field.zero)
            and consistent_power(field, z, n, answer))


def check_quadratic_roots(field, a, b, c, roots) -> bool:
    """roots, each (real, imag) in the field, are the two roots of a x**2 + b x + c (a, b, c field constants) in
    strictly increasing order (by real part, then imaginary part): each is substituted exactly and in interval
    arithmetic."""
    if len(roots) != 2:
        return False
    previous = None
    for real, imag in roots:
        # (a x + b) x + c is zero at x = real + imag i.
        high_real, high_imag = field.add(field.mul(a, real), b), field.mul(a, imag)
        if (field.add(field.sub(field.mul(high_real, real), field.mul(high_imag, imag)), c) != field.zero
                or field.add(field.mul(high_real, imag), field.mul(high_imag, real)) != field.zero):
            return False
        if not consistent_root(field, a, b, c, (real, imag)):
            return False
        if previous is not None:
            order = field.compare(previous[0], real)
            if order is None or order > 0 or (order == 0 and field.compare(previous[1], imag) != -1):
                return False
        previous = real, imag
    return True


# ---------------------------------------------------------------------------------------------------------------
# Fixed-point interval arithmetic on the operands' enclosures

def _fixed(field, x):
    """Integers (lo, hi) with lo <= x * 2**64 <= hi, from the element's 64-bit enclosure."""
    low, high = field.enclosure(x, _FRACTION)
    lo = (low.numerator << _FRACTION) // low.denominator
    hi = -((-high.numerator << _FRACTION) // high.denominator)
    return lo, hi


def _times(p, q):
    """The outward-rounded product of two fixed-point intervals."""
    products = (p[0] * q[0], p[0] * q[1], p[1] * q[0], p[1] * q[1])
    return min(products) >> _FRACTION, -((-max(products)) >> _FRACTION)


def _plus(p, q):
    return p[0] + q[0], p[1] + q[1]


def _minus(p, q):
    return p[0] - q[1], p[1] - q[0]


def _rectangle(field, z):
    return _fixed(field, z[0]), _fixed(field, z[1])


def _complex_times(z, w):
    """The rectangle around the products of two rectangles' points; ValueError past _RECTANGLE_BITS."""
    (a, b), (c, d) = z, w
    real, imag = _minus(_times(a, c), _times(b, d)), _plus(_times(a, d), _times(b, c))
    if max(abs(value) for value in (*real, *imag)).bit_length() > _RECTANGLE_BITS:
        raise ValueError("rectangle bits")
    return real, imag


def _meet(z, w) -> bool:
    """Whether two rectangles share a point."""
    return all(max(p[0], q[0]) <= min(p[1], q[1]) for p, q in zip(z, w))


def consistent_product(field, z, w, product) -> bool:
    """The rectangle of z times that of w meets the rectangle of the product."""
    try:
        return _meet(_complex_times(_rectangle(field, z), _rectangle(field, w)), _rectangle(field, product))
    except ValueError:
        return False


def consistent_power(field, z, n: int, answer) -> bool:
    """The rectangle of z to the |n|-th power (right-to-left binary exponentiation) meets the answer's rectangle,
    or, for a negative n, times the answer's rectangle meets 1."""
    one = ((1 << _FRACTION, 1 << _FRACTION), (0, 0))
    try:
        base, power, m = _rectangle(field, z), one, abs(n)
        while m:
            if m & 1:
                power = _complex_times(power, base)
            m >>= 1
            if m:
                base = _complex_times(base, base)
        if n > 0:
            return _meet(power, _rectangle(field, answer))
        return _meet(_complex_times(power, _rectangle(field, answer)), one)
    except ValueError:
        return False


def consistent_root(field, a, b, c, root) -> bool:
    """(a x + b) x + c, in interval arithmetic on the rectangle of the root x, contains 0."""
    zero = (field.zero, field.zero)
    try:
        x = _rectangle(field, root)
        high = _complex_times(_rectangle(field, (a, field.zero)), x)
        high = (_plus(high[0], _fixed(field, b)), high[1])
        value = _complex_times(high, x)
        value = (_plus(value[0], _fixed(field, c)), value[1])
        return _meet(value, _rectangle(field, zero))
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------------------------
# Expressions of square roots

def _sqrt_element(field, radicand):
    """The element sqrt(radicand) of the field, for a Rational radicand >= 0: the non-negative rational multiple
    g e_m of the basis element e_m whose square, g**2 weight[m], is the radicand; None when there is none."""
    for mask, weight in enumerate(field.weight):
        ratio = Rational(radicand.numerator, radicand.denominator * weight)
        p, q = isqrt(ratio.numerator), isqrt(ratio.denominator)
        if p * p == ratio.numerator and q * q == ratio.denominator:
            x = [field.zero[0]] * field.size
            x[mask] = Rational(p, q)
            return tuple(x)
    return None


def _quadratic_element(field, part):
    """The element of a quadratic RealAlgebraicRoot c2 x**2 + c1 x + c0: (-c1 +- g sqrt(weight[m]))/(2 c2) with
    g**2 weight[m] the discriminant, the sign by the index; None when the discriminant is no such product."""
    coefficients = part.integer_coefficients
    if type(coefficients) is not tuple or len(coefficients) != 3 or any(type(c) is not int for c in coefficients):
        return None
    c0, c1, c2 = coefficients
    if c2 <= 0 or part.real_index not in (0, 1):
        return None
    root = _sqrt_element(field, Rational(c1 * c1 - 4 * c0 * c2))
    if root is None or not any(c.numerator for c in root[1:]):
        return None
    shift = field.constant(Rational(-c1, 2 * c2))
    half = field.scale(root, Rational(1, 2 * c2))
    return field.add(shift, half) if part.real_index else field.sub(shift, half)


def recompute(field, node):
    """The value of an expression of Rationals, Sqrt, quadratic RealAlgebraicRoots, Add, Mul and Pow as an element of
    the field, read on its own: ValueError for anything that does not check."""
    kind = type(node)
    if kind is Rational:
        return field.constant(node)
    if kind is Sqrt:
        radicand = node.radicand
        if type(radicand) is not Rational or radicand.numerator < 0:
            raise ValueError("radicand")
        x = _sqrt_element(field, radicand)
        if x is None or field.mul(x, x) != field.constant(radicand) or field.compare(x, field.zero) not in (0, 1):
            raise ValueError("square root")
        return x
    if _is_root(node):
        x = _quadratic_element(field, node)
        if x is None or not represents(field, x, node):
            raise ValueError("quadratic root")
        return x
    if kind is Add:
        total = field.zero
        for term in reversed(node.terms):
            total = field.add(recompute(field, term), total)
        return total
    if kind is Mul:
        product = field.one
        for factor in reversed(node.factors):
            product = field.mul(recompute(field, factor), product)
        return product
    if kind is Pow and type(node.exponent) is int and node.exponent >= 0:
        base, result = recompute(field, node.base), field.one
        for _ in range(node.exponent):
            result = field.mul(base, result)
        return result
    raise ValueError(kind.__name__)


_INTERVAL_BITS = _BITS + _FRACTION  # A fixed-point end this long holds a value past 2**max_integer_bits.


def _bounded(low: int, high: int):
    """The fixed-point interval (low, high); ValueError when an end holds a value past 2**max_integer_bits."""
    if max(abs(low), abs(high)).bit_length() > _INTERVAL_BITS:
        raise ValueError("interval bits")
    return low, high


def _fixed_sqrt(r):
    """Integers (lo, lo + 1) around sqrt(r) * 2**64 for a Rational r >= 0."""
    if type(r) is not Rational or r.numerator < 0:
        raise ValueError("radicand")
    low = isqrt((r.numerator << 2 * _FRACTION) // r.denominator)
    return low, low + 1


def _fixed_interval(node):
    """Integers (lo, hi) with lo <= value * 2**64 <= hi for an expression, by fixed-point interval arithmetic,
    rounded outward at every product, on the expression alone. Fixed point keeps the denominators at 2**64 however
    many factors meet (exact rational ends would multiply theirs); an end past 2**max_integer_bits raises ValueError."""
    kind = type(node)
    if kind is Rational:
        scaled = node.numerator << _FRACTION
        low = scaled // node.denominator
        return _bounded(low, low + (1 if scaled % node.denominator else 0))
    if kind is Sqrt:
        return _bounded(*_fixed_sqrt(node.radicand))
    if _is_root(node):
        c0, c1, c2 = node.integer_coefficients
        if type(c2) is not int or c2 <= 0:
            raise ValueError("leading coefficient")
        root_low, root_high = _fixed_sqrt(Rational(c1 * c1 - 4 * c0 * c2))
        shift, divisor = -c1 << _FRACTION, 2 * c2  # (-c1 +- sqrt(disc)) / (2 c2)
        if node.real_index:
            low, high = shift + root_low, shift + root_high
        else:
            low, high = shift - root_high, shift - root_low
        return _bounded(low // divisor, -(-high // divisor))
    if kind is Add:
        low = high = 0
        for term in node.terms:
            term_low, term_high = _fixed_interval(term)
            low, high = low + term_low, high + term_high
        return _bounded(low, high)
    if kind is Mul:
        result = (1 << _FRACTION, 1 << _FRACTION)
        for factor in node.factors:
            result = _bounded(*_times(result, _fixed_interval(factor)))
        return result
    if kind is Pow and type(node.exponent) is int and node.exponent >= 0:
        base, result = _fixed_interval(node.base), (1 << _FRACTION, 1 << _FRACTION)
        for _ in range(node.exponent):
            result = _bounded(*_times(result, base))
        return result
    raise ValueError(kind.__name__)


def interval(node):
    """A rational interval around the value of an expression, by interval arithmetic on the expression alone: its
    ends are multiples of 2**-64 (_fixed_interval)."""
    low, high = _fixed_interval(node)
    scale = 1 << _FRACTION
    return Rational(low, scale), Rational(high, scale)


def check_algebraic(field, node, x) -> bool:
    """x is the value of the expression: read again on its own (recompute) and consistent with the interval
    arithmetic of the expression (interval) at 64 bits."""
    try:
        if recompute(field, node) != x:
            return False
        low, high = interval(node)
    except (ValueError, ArithmeticError, RecursionError):
        return False
    x_low, x_high = field.enclosure(x, 64)
    return max(low, x_low) <= min(high, x_high)
