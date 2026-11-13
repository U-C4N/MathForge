"""Truncated power series over Q: tuples of n + 1 Rationals, s[k] the coefficient of (x - a)^k.

Every function keeps the length of its operands (they must agree), except ``derivative``, which drops the last
coefficient, and ``integral``, which adds one; ``mul`` truncates the product to that length. A caller outside
``computation()`` runs unmetered, as any exact operation.

Work. Each Rational result charges the active budget one unit (the model's Rational arithmetic), and the loops of
``mul`` and ``reciprocal`` charge one unit per pair of indices up front, so runaway orders stop at the work limit. A
Rational result costs one unit whatever its length, while its gcds and products grow with the length of its operands,
quadratically where both are long. So every operation on two Rationals of a series is also charged, before it runs,
``operation_work(words(left), words(right))`` units: one per 64 pairs of the 64-bit words of their four integers and
one per 4 words, mostly the gcds of the result (measured with CPython 3.15: a product or difference of two Rationals of
62 words each takes 35 to 42 us, 91 units; of 125 words each 72 to 120 us, 306 units; of 500 words each 0.6 to 1 ms,
4,156 units; of a 2-word and a 500-word one 23 to 27 us, 140 units; about 0.2 to 0.45 microsecond a unit, the scale of
the other checkers. The earlier rate, one unit per 256 pairs and per 16 words, let sqrt(1 + x) at order 2000 run 4 s
within the default budget).
mathforge.certify.series.taylor charges its own loops with the same two functions, and the producer of
series_coefficients (mathforge.kernels.series) bounds with them what its check will charge.
"""

from __future__ import annotations

from mathforge.errors import InvalidInput
from mathforge.limits import current_budget
from mathforge.model import Rational

ZERO, ONE = Rational(0), Rational(1)


def words(value: Rational) -> int:
    """The 64-bit words of a Rational's numerator and denominator together, rounded down (0 below 64 bits)."""
    return (value.numerator.bit_length() + value.denominator.bit_length()) >> 6


def operation_work(left: int, right: int) -> int:
    """The units, beyond the one unit of its result, of one addition, subtraction, product or quotient of two
    Rationals of left and right words (``words``): one per 64 pairs of words and one per 4 words. 0 below 4 words."""
    return (left * right >> 6) + ((left + right) >> 2)


def _tick(n: int) -> None:
    budget = current_budget()
    if budget is not None:
        budget.tick(n)


def _charge_pairs(a: tuple, b: tuple) -> None:
    """Charge the term-by-term operations of a and b by their lengths."""
    budget = current_budget()
    if budget is not None:
        units = 0
        for u, v in zip(a, b):
            units += operation_work(words(u), words(v))
        budget.tick(units)


def _charge_each(a: tuple) -> None:
    """Charge one operation of each term of a with a short operand (a sign, a small int or factor)."""
    budget = current_budget()
    if budget is not None:
        budget.tick(sum([words(u) for u in a]) >> 2)


def constant(value: Rational, length: int) -> tuple[Rational, ...]:
    """(value, 0, ..., 0) of the given length."""
    return (value,) + (ZERO,) * (length - 1)


def add(a: tuple, b: tuple) -> tuple:
    _charge_pairs(a, b)  # Operands of different lengths raise ValueError below, as zip(strict=True) does.
    return tuple([u + v for u, v in zip(a, b, strict=True)])


def sub(a: tuple, b: tuple) -> tuple:
    _charge_pairs(a, b)  # Operands of different lengths raise ValueError below, as zip(strict=True) does.
    return tuple([u - v for u, v in zip(a, b, strict=True)])


def neg(a: tuple) -> tuple:
    _charge_each(a)
    return tuple([-u for u in a])


def scale(a: tuple, factor: Rational) -> tuple:
    budget = current_budget()
    if budget is not None:
        size = words(factor)
        budget.tick(sum([operation_work(words(u), size) for u in a]))
    return tuple([u * factor for u in a])


def mul(a: tuple, b: tuple) -> tuple:
    """The product truncated to the common length. Each term's product and its addition to the sum are charged by
    their lengths before they run."""
    size = len(a)
    if len(b) != size:
        raise InvalidInput("Truncated series of different lengths")
    _tick(size * (size + 1) // 2)
    out = [ZERO] * size
    budget = current_budget()
    if budget is None:
        for i, u in enumerate(a):
            if u.numerator:
                for j in range(size - i):
                    v = b[j]
                    if v.numerator:
                        out[i + j] = out[i + j] + u * v
        return tuple(out)
    tick = budget.tick
    sizes = [words(v) for v in b]
    for i, u in enumerate(a):
        if u.numerator:
            left = words(u)
            for j in range(size - i):
                v = b[j]
                if v.numerator:
                    right = sizes[j]
                    units = (left * right >> 6) + ((left + right) >> 2)
                    if units:
                        tick(units)
                    term = u * v
                    total = out[i + j]
                    right = (term.numerator.bit_length() + term.denominator.bit_length()) >> 6
                    left_sum = (total.numerator.bit_length() + total.denominator.bit_length()) >> 6
                    units = (left_sum * right >> 6) + ((left_sum + right) >> 2)
                    if units:
                        tick(units)
                    out[i + j] = total + term
    return tuple(out)


def derivative(a: tuple) -> tuple:
    """The derivative, one coefficient shorter: k s[k] at k - 1."""
    _charge_each(a)
    return tuple([a[k] * k for k in range(1, len(a))])


def integral(a: tuple) -> tuple:
    """The primitive with constant term 0, one coefficient longer: s[k - 1] / k at k."""
    _charge_each(a)
    return (ZERO,) + tuple([a[k - 1] / k for k in range(1, len(a) + 1)])


def reciprocal(a: tuple) -> tuple:
    """1 / s for s[0] != 0, truncated to the same length. Each operation is charged by its lengths before it runs."""
    if not a or not a[0].numerator:
        raise InvalidInput("A truncated series without a constant term has no reciprocal")
    size = len(a)
    _tick(size * (size + 1) // 2)
    budget = current_budget()
    inverse = ONE / a[0]
    out = [inverse]
    if budget is None:
        for k in range(1, size):
            total = ZERO
            for j in range(1, k + 1):
                if a[j].numerator:
                    total = total + a[j] * out[k - j]
            out.append(-total * inverse)
        return tuple(out)
    tick = budget.tick
    sizes = [words(v) for v in a]
    found = [words(inverse)]  # words(out[k]) for each k so far
    for k in range(1, size):
        total = ZERO
        for j in range(1, k + 1):
            if a[j].numerator:
                left, right = sizes[j], found[k - j]
                units = (left * right >> 6) + ((left + right) >> 2)
                if units:
                    tick(units)
                term = a[j] * out[k - j]
                left = (total.numerator.bit_length() + total.denominator.bit_length()) >> 6
                right = (term.numerator.bit_length() + term.denominator.bit_length()) >> 6
                units = (left * right >> 6) + ((left + right) >> 2)
                if units:
                    tick(units)
                total = total + term
        left = (total.numerator.bit_length() + total.denominator.bit_length()) >> 6
        units = (left >> 2) + (left * found[0] >> 6) + ((left + found[0]) >> 2)  # The negation and the product.
        if units:
            tick(units)
        value = -total * inverse
        out.append(value)
        found.append((value.numerator.bit_length() + value.denominator.bit_length()) >> 6)
    return tuple(out)


def power(a: tuple, exponent: int) -> tuple:
    """s ** exponent for an int exponent >= 0, truncated (square and multiply)."""
    if type(exponent) is not int or exponent < 0:
        raise InvalidInput("A truncated series power needs an int exponent >= 0")
    result, base = constant(ONE, len(a)), a
    while exponent:
        if exponent & 1:
            result = mul(result, base)
        exponent >>= 1
        if exponent:
            base = mul(base, base)
    return result
