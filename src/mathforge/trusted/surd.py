"""Exact arithmetic in a multiquadratic field K = Q(sqrt b_1, ..., sqrt b_t): the surd kernel that the lean complex
surd producers (mathforge.lean.algebra_complex) and their independent checks (mathforge.certify.complex) share.

The base b_1, ..., b_t is a list of pairwise coprime squarefree integers above 1. For such a base K has degree 2**t
over Q, the products of the sqrt b_j over the subsets of {1, ..., t} are a basis of K, and changing the signs of any
of the sqrt b_j is an automorphism of K. This is Besicovitch's theorem (the square roots of pairwise coprime
squarefree integers above 1 are linearly independent over Q) with the Galois theory of a compositum of quadratic
fields; everything that names an element by its minimal polynomial (the product of X - y over the distinct images y
of the element under the sign changes) rests on it. Its hypotheses are not assumed: the producer establishes them by
exact computation, a radicand being split into a square and a squarefree part by trial division (declined when the
part that is left cannot be shown squarefree) and the radicands refined by gcds into pairwise coprime numbers above 1.

An element is the tuple of its 2**t Rational coefficients, that of the subset with bit mask m at index m. Products
follow (e_s)(e_u) = weight[s & u] e_(s xor u), weight[m] the product of the b_j in the subset m. Nothing here decides
what a value of an operation is; tests/v08/complex checks this arithmetic against the structured path's numbers.
"""

from __future__ import annotations

from math import isqrt

from ..model import Rational, trusted_rational

_ZERO, _ONE = trusted_rational(0, 1), trusted_rational(1, 1)


class Field:
    """K = Q(sqrt b_1, ..., sqrt b_t) for a pairwise coprime base; elements are tuples of 2**t Rationals."""

    __slots__ = ("base", "size", "weight", "zero", "one", "lower")

    def __init__(self, base):
        self.base = tuple(base)
        self.size = 1 << len(base)
        weight = [1] * self.size  # The product of the b_j of a subset: (e_s)(e_u) = weight[s & u] e_(s xor u).
        for mask in range(1, self.size):
            low = mask & -mask
            weight[mask] = weight[mask ^ low] * self.base[low.bit_length() - 1]
        self.weight = tuple(weight)
        self.zero = (_ZERO,) * self.size
        self.one = (_ONE,) + (_ZERO,) * (self.size - 1)
        self.lower = Field(self.base[:-1]) if base else None  # The field without the last square root.

    def constant(self, value):
        return (value,) + (_ZERO,) * (self.size - 1)

    def mask(self, n):
        """The subset of the base whose product is the squarefree number n; ValueError when there is none."""
        mask = 0
        for j, b in enumerate(self.base):
            if n % b == 0:
                n //= b
                mask |= 1 << j
        if n != 1:
            raise ValueError("not a product of the base")
        return mask

    def quadratic(self, coefficients, index, g, n):
        """The root of c2 x**2 + c1 x + c0 with the given index, where c1**2 - 4 c0 c2 = g**2 n."""
        c0, c1, c2 = coefficients
        x = [_ZERO] * self.size
        x[0] = Rational(-c1, 2 * c2)
        x[self.mask(n)] = Rational(g if index else -g, 2 * c2)
        return tuple(x)

    def add(self, x, y):
        return tuple(a + b for a, b in zip(x, y))

    def sub(self, x, y):
        return tuple(a - b for a, b in zip(x, y))

    def neg(self, x):
        return tuple(-a for a in x)

    def scale(self, x, factor):
        return tuple(a * factor for a in x)

    def mul(self, x, y):
        out = [_ZERO] * self.size
        weight = self.weight
        for s, a in enumerate(x):
            if a.numerator:
                for u, b in enumerate(y):
                    if b.numerator:
                        product = a * b
                        w = weight[s & u]
                        out[s ^ u] = out[s ^ u] + (product if w == 1 else product * w)
        return tuple(out)

    def inverse(self, x):
        """1/x, by (u + v sqrt b)(u - v sqrt b) = u*u - b*v*v in the field below; ZeroDivisionError for zero."""
        if self.size == 1:
            return (_ONE / x[0],)
        half, low = self.size // 2, self.lower
        u, v = x[:half], x[half:]
        norm = low.sub(low.mul(u, u), low.scale(low.mul(v, v), self.base[-1]))
        inverse = low.inverse(norm)
        return low.mul(u, inverse) + low.neg(low.mul(v, inverse))

    def sign(self, x):
        """The sign of the real number x, exactly: from the signs of u and v in x = u + v sqrt b, and, when they
        differ, that of u*u - b*v*v."""
        if self.size == 1:
            return (x[0].numerator > 0) - (x[0].numerator < 0)
        half, low = self.size // 2, self.lower
        u, v = x[:half], x[half:]
        su, sv = low.sign(u), low.sign(v)
        if su == sv or sv == 0:
            return su
        if su == 0:
            return sv
        difference = low.sign(low.sub(low.mul(u, u), low.scale(low.mul(v, v), self.base[-1])))
        if difference == 0:
            raise ArithmeticError("sqrt b is in the field below")
        return su if difference > 0 else sv

    def enclosure(self, x, bits):
        """Rational (low, high) with low <= x <= high, from sqrt b within 2**-bits: a sum of each coefficient times
        the product of the square roots' bounds, which never multiplies two elements."""
        scale = 1 << bits
        roots = []
        for b in self.base:
            r = isqrt(b * scale * scale)
            roots.append((Rational(r, scale), Rational(r + 1, scale)))
        low = high = _ZERO
        for mask, c in enumerate(x):
            if c.numerator:
                product_low = product_high = _ONE
                for j, (l, h) in enumerate(roots):
                    if mask >> j & 1:
                        product_low, product_high = product_low * l, product_high * h
                if c.numerator > 0:
                    low, high = low + c * product_low, high + c * product_high
                else:
                    low, high = low + c * product_high, high + c * product_low
        return low, high

    def compare(self, x, y):
        """-1, 0 or 1 as x is below, equal to or above y, decided by interval enclosures; None when 1,024 bits do
        not separate them."""
        difference = self.sub(x, y)
        if not any(c.numerator for c in difference):
            return 0
        for bits in (64, 256, 1024):
            low, high = self.enclosure(difference, bits)
            if low.numerator > 0:
                return 1
            if high.numerator < 0:
                return -1
        return None

    def orbit(self, x):
        """The distinct images of x under the sign changes, x first."""
        seen, images = set(), []
        for flip in range(self.size):
            image = tuple(-c if (mask & flip).bit_count() & 1 else c for mask, c in enumerate(x))
            if image not in seen:
                seen.add(image)
                images.append(image)
        return images

    def horner(self, coefficients, x):
        """The integer polynomial with the given coefficients (constant term first) at x."""
        result = self.zero
        for c in reversed(coefficients):
            result = self.add(self.mul(result, x), self.constant(trusted_rational(c, 1)))
        return result

    def bounded(self, bits, *elements):
        """Whether every coefficient of the elements has a numerator and a denominator of at most ``bits`` bits."""
        for x in elements:
            for c in x:
                if c.numerator.bit_length() > bits or c.denominator.bit_length() > bits:
                    return False
        return True
