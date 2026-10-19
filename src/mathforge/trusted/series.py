"""Truncated power series over Q: tuples of n + 1 Rationals, s[k] the coefficient of (x - a)^k.

Every function keeps the length of its operands (they must agree), except ``derivative``, which drops the last
coefficient, and ``integral``, which adds one; ``mul`` truncates the product to that length. Each multiplication
step charges the active budget one work unit through Rational arithmetic, and the loops tick it in batches, so
runaway orders stop at the work limit. A caller outside ``computation()`` runs unmetered, as any exact operation.
"""

from __future__ import annotations

from mathforge.errors import InvalidInput
from mathforge.limits import current_budget
from mathforge.model import Rational

ZERO, ONE = Rational(0), Rational(1)


def _tick(n: int) -> None:
    budget = current_budget()
    if budget is not None:
        budget.tick(n)


def constant(value: Rational, length: int) -> tuple[Rational, ...]:
    """(value, 0, ..., 0) of the given length."""
    return (value,) + (ZERO,) * (length - 1)


def add(a: tuple, b: tuple) -> tuple:
    return tuple([u + v for u, v in zip(a, b, strict=True)])


def sub(a: tuple, b: tuple) -> tuple:
    return tuple([u - v for u, v in zip(a, b, strict=True)])


def neg(a: tuple) -> tuple:
    return tuple([-u for u in a])


def scale(a: tuple, factor: Rational) -> tuple:
    return tuple([u * factor for u in a])


def mul(a: tuple, b: tuple) -> tuple:
    """The product truncated to the common length."""
    size = len(a)
    if len(b) != size:
        raise InvalidInput("Truncated series of different lengths")
    _tick(size * (size + 1) // 2)
    out = [ZERO] * size
    for i, u in enumerate(a):
        if u.numerator:
            for j in range(size - i):
                v = b[j]
                if v.numerator:
                    out[i + j] = out[i + j] + u * v
    return tuple(out)


def derivative(a: tuple) -> tuple:
    """The derivative, one coefficient shorter: k s[k] at k - 1."""
    return tuple([a[k] * k for k in range(1, len(a))])


def integral(a: tuple) -> tuple:
    """The primitive with constant term 0, one coefficient longer: s[k - 1] / k at k."""
    return (ZERO,) + tuple([a[k - 1] / k for k in range(1, len(a) + 1)])


def reciprocal(a: tuple) -> tuple:
    """1 / s for s[0] != 0, truncated to the same length."""
    if not a or not a[0].numerator:
        raise InvalidInput("A truncated series without a constant term has no reciprocal")
    size = len(a)
    _tick(size * (size + 1) // 2)
    inverse = ONE / a[0]
    out = [inverse]
    for k in range(1, size):
        total = ZERO
        for j in range(1, k + 1):
            if a[j].numerator:
                total = total + a[j] * out[k - j]
        out.append(-total * inverse)
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
