"""Lean complex scalar operations: Gaussian rationals and quadratic surds.

``mathforge.complex_*`` and ``mathforge.quadratic_roots`` called without ``limits`` first call these functions. An
operand that is an int, a ``Rational`` or a ``ComplexAlgebraic`` whose two parts are ``Rational`` is a Gaussian
rational, an element of Q(i). On such operands the function computes the answer with plain integer arithmetic,
checks it by a second computation of a different shape and builds the ``ComplexAlgebraic`` from trusted Rationals.
The structured path builds the same value from certified real-algebraic arithmetic, which for Gaussian operands
only ever meets rational numbers, so the two values are equal (``tests/v08/complex``).

A product, quotient or power of operands whose irrational parts are quadratic surds (a ``RealAlgebraicRoot`` of
degree two), the roots of a rational quadratic with irrational roots, and ``complex_algebraic`` of expressions of
square roots take the second path, described in front of ``_Field``: exact arithmetic in a field Q(sqrt b_1, ...,
sqrt b_t), each irrational part named as the structured path names it and checked against its element.

Everything else, an operand with another irrational part or a call that is no valid one, is the generic runner's
(``run_value``: the family's producer and its v0.7 checker, as before) or, for a call that the structured path
refuses, ``FALLBACK``.

The checks of the Gaussian path, one per operation, recompute the answer another way and compare by
cross-multiplication, so that no reduced fraction is compared with a fraction produced by the same code:

* add and subtract: the other operand subtracted from, or added to, the answer gives the first operand back;
* multiply: the three-multiplication (Karatsuba) expansion of the product of the two numerators;
* divide: the answer times the divisor gives the dividend back;
* conjugate, real and imaginary part, algebraic: the parts against the operands';
* squared modulus: (a + b)**2 - 2ab for a**2 + b**2;
* square root: the root squared gives the operand back, on the principal branch (non-negative real part, and a
  non-negative imaginary part when the real part is zero);
* power: the product by left-to-right binary exponentiation, where the producer's runs right to left; a negative
  power is checked by multiplying back;
* quadratic roots: each root substituted into the polynomial (and, for a double root, into its derivative), the
  roots in strictly increasing order, and the discriminant's sign against the number of roots.

Every integer in an operand and in an answer stays within ``max_integer_bits``; the multiplicative operations take
operands of at most a sixteenth of that, so that no step of theirs holds an integer that the structured path would
refuse. A larger operand, a result past the limit, a zero divisor and a call made inside an active budget are left
to the structured path (``FALLBACK``), which charges that budget or raises v0.7's error. Exponents are limited to
256 bits for the same reason: the structured path's work and evidence for them stay far from their limits.
"""

from __future__ import annotations

from math import gcd, isqrt, lcm

from ..algebraic import RealAlgebraicRoot
from ..complex_numbers import ComplexAlgebraic, ComplexRoot
from ..limits import ComputationLimits, current_budget
from ..model import Add, Mul, Pow, Rational, Sqrt, trusted_rational
from . import FALLBACK, run_value

_BITS = ComputationLimits().max_integer_bits
_SMALL = _BITS // 16  # Operands of the products, quotients, roots and powers.
_EXPONENT_BITS = 256

_new = object.__new__
_set = object.__setattr__


# ---------------------------------------------------------------------------------------------------------------
# Operands and answers

def _valid(n, d, limit):
    """Whether n/d is a canonical rational with integers of at most ``limit`` bits."""
    return (type(n) is int and type(d) is int and d > 0 and n.bit_length() <= limit and d.bit_length() <= limit
            and (d == 1 or gcd(n, d) == 1))


def _gauss(value, limit):
    """(an, ad, bn, bd) with an/ad + (bn/bd) i the value of a Gaussian-rational operand, or None."""
    kind = type(value)
    if kind is ComplexAlgebraic:
        real, imag = value.real, value.imag
        if type(real) is Rational and type(imag) is Rational:
            an, ad, bn, bd = real.numerator, real.denominator, imag.numerator, imag.denominator
            if _valid(an, ad, limit) and _valid(bn, bd, limit):
                return an, ad, bn, bd
    elif kind is int:
        if value.bit_length() <= limit:
            return value, 1, 0, 1
    elif kind is Rational:
        n, d = value.numerator, value.denominator
        if _valid(n, d, limit):
            return n, d, 0, 1
    return None


def _scalar(value, limit):
    """(n, d) of an int or a Rational operand, or None."""
    kind = type(value)
    if kind is int:
        if value.bit_length() <= limit:
            return value, 1
    elif kind is Rational:
        n, d = value.numerator, value.denominator
        if _valid(n, d, limit):
            return n, d
    return None


def _common(an, ad, bn, bd):
    """(a, b, d) with (a + b i)/d the value of the parts an/ad + (bn/bd) i, for d > 0."""
    if ad == bd:
        return an, bn, ad
    g = gcd(ad, bd)
    return an * (bd // g), bn * (ad // g), ad // g * bd


def _reduced(n, d):
    """n/d in lowest terms, for d > 0."""
    if d != 1:
        g = gcd(n, d)
        if g != 1:
            return n // g, d // g
    return n, d


def _parts(x, y, d):
    """The parts x/d and y/d in lowest terms (xn, xd, yn, yd); None when an integer passes the limit."""
    xn, xd = _reduced(x, d)
    yn, yd = _reduced(y, d)
    if (xn.bit_length() > _BITS or xd.bit_length() > _BITS or yn.bit_length() > _BITS
            or yd.bit_length() > _BITS):
        return None
    return xn, xd, yn, yd


def _complex(xn, xd, yn, yd):
    value = _new(ComplexAlgebraic)
    _set(value, "real", trusted_rational(xn, xd))
    _set(value, "imag", trusted_rational(yn, yd))
    return value


def _generic(operation, arguments):
    return run_value(operation, arguments)


# ---------------------------------------------------------------------------------------------------------------
# Construction and the linear operations

def complex_algebraic(real, imag):
    if current_budget() is not None:
        return FALLBACK
    try:
        if type(real) is int and type(imag) is int:
            if real.bit_length() > _BITS or imag.bit_length() > _BITS:
                return FALLBACK
            return _complex(real, 1, imag, 1)
        a, b = _scalar(real, _BITS), _scalar(imag, _BITS)
        if a is None or b is None:
            return _surd(_surd_algebraic, "complex_algebraic", {"real": real, "imag": imag}, real, imag)
        return _complex(a[0], a[1], b[0], b[1])  # _scalar has checked both parts to be canonical rationals.
    except Exception:
        return FALLBACK


def complex_add(left, right):
    if current_budget() is not None:
        return FALLBACK
    try:
        z, w = _gauss(left, _BITS), _gauss(right, _BITS)
        if z is None or w is None:
            return _generic("complex_add", {"left": left, "right": right})
        za, zb, zd = _common(*z)
        wa, wb, wd = _common(*w)
        if zd == wd:
            x, y, d = za + wa, zb + wb, zd
        else:
            x, y, d = za * wd + wa * zd, zb * wd + wb * zd, zd * wd
        parts = _parts(x, y, d)
        if parts is None:
            return FALLBACK
        xn, xd, yn, yd = parts
        # The answer less the right operand is the left one.
        if not (_valid(xn, xd, _BITS) and _valid(yn, yd, _BITS)
                and (xn * wd - wa * xd) * zd == za * xd * wd and (yn * wd - wb * yd) * zd == zb * yd * wd):
            return FALLBACK
        return _complex(xn, xd, yn, yd)
    except Exception:
        return FALLBACK


def complex_sub(left, right):
    if current_budget() is not None:
        return FALLBACK
    try:
        z, w = _gauss(left, _BITS), _gauss(right, _BITS)
        if z is None or w is None:
            return _generic("complex_sub", {"left": left, "right": right})
        za, zb, zd = _common(*z)
        wa, wb, wd = _common(*w)
        if zd == wd:
            x, y, d = za - wa, zb - wb, zd
        else:
            x, y, d = za * wd - wa * zd, zb * wd - wb * zd, zd * wd
        parts = _parts(x, y, d)
        if parts is None:
            return FALLBACK
        xn, xd, yn, yd = parts
        # The answer plus the right operand is the left one.
        if not (_valid(xn, xd, _BITS) and _valid(yn, yd, _BITS)
                and (xn * wd + wa * xd) * zd == za * xd * wd and (yn * wd + wb * yd) * zd == zb * yd * wd):
            return FALLBACK
        return _complex(xn, xd, yn, yd)
    except Exception:
        return FALLBACK


def complex_conjugate(value):
    if current_budget() is not None:
        return FALLBACK
    try:
        z = _gauss(value, _BITS)
        if z is None:
            return _generic("complex_conjugate", {"value": value})
        an, ad, bn, bd = z
        result = _complex(an, ad, -bn, bd)
        real, imag = result.real, result.imag
        if not (real.numerator == an and real.denominator == ad and imag.numerator + bn == 0
                and imag.denominator == bd):
            return FALLBACK
        return result
    except Exception:
        return FALLBACK


def complex_real(value):
    if current_budget() is not None:
        return FALLBACK
    try:
        kind = type(value)
        if kind is ComplexAlgebraic:
            if _gauss(value, _BITS) is not None:  # Both parts are checked, as the structured path checks them.
                return value.real
        elif kind is Rational:
            if _valid(value.numerator, value.denominator, _BITS):
                return value
        elif kind is int:
            if value.bit_length() <= _BITS:
                return trusted_rational(value, 1)
        return _generic("complex_real", {"value": value})
    except Exception:
        return FALLBACK


def complex_imag(value):
    if current_budget() is not None:
        return FALLBACK
    try:
        kind = type(value)
        if kind is ComplexAlgebraic:
            if _gauss(value, _BITS) is not None:
                return value.imag
        elif kind is Rational:
            if _valid(value.numerator, value.denominator, _BITS):
                return trusted_rational(0, 1)
        elif kind is int:
            if value.bit_length() <= _BITS:
                return trusted_rational(0, 1)
        return _generic("complex_imag", {"value": value})
    except Exception:
        return FALLBACK


# ---------------------------------------------------------------------------------------------------------------
# Products and quotients

def complex_mul(left, right):
    if current_budget() is not None:
        return FALLBACK
    try:
        z, w = _gauss(left, _SMALL), _gauss(right, _SMALL)
        if z is None or w is None:
            return _surd(_surd_mul, "complex_mul", {"left": left, "right": right}, left, right)
        za, zb, zd = _common(*z)
        wa, wb, wd = _common(*w)
        d = zd * wd
        parts = _parts(za * wa - zb * wb, za * wb + zb * wa, d)
        if parts is None:
            return FALLBACK
        xn, xd, yn, yd = parts
        # The same product by three multiplications.
        k1, k2, k3 = wa * (za + zb), za * (wb - wa), zb * (wa + wb)
        if not (_valid(xn, xd, _BITS) and _valid(yn, yd, _BITS)
                and xn * d == (k1 - k3) * xd and yn * d == (k1 + k2) * yd):
            return FALLBACK
        return _complex(xn, xd, yn, yd)
    except Exception:
        return FALLBACK


def complex_div(left, right):
    if current_budget() is not None:
        return FALLBACK
    try:
        z, w = _gauss(left, _SMALL), _gauss(right, _SMALL)
        if z is None or w is None:
            return _surd(_surd_div, "complex_div", {"left": left, "right": right}, left, right)
        za, zb, zd = _common(*z)
        wa, wb, wd = _common(*w)
        norm = wa * wa + wb * wb
        if norm == 0:
            return FALLBACK  # A zero divisor: the structured path raises.
        parts = _parts((za * wa + zb * wb) * wd, (zb * wa - za * wb) * wd, zd * norm)
        if parts is None:
            return FALLBACK
        xn, xd, yn, yd = parts
        # The answer times the divisor is the dividend.
        if not (_valid(xn, xd, _BITS) and _valid(yn, yd, _BITS)
                and (xn * wa * yd - yn * wb * xd) * zd == za * xd * yd * wd
                and (xn * wb * yd + yn * wa * xd) * zd == zb * xd * yd * wd):
            return FALLBACK
        return _complex(xn, xd, yn, yd)
    except Exception:
        return FALLBACK


def complex_norm_squared(value):
    if current_budget() is not None:
        return FALLBACK
    try:
        z = _gauss(value, _SMALL)
        if z is None:
            return _generic("complex_norm_squared", {"value": value})
        a, b, d = _common(*z)
        n, m = _reduced(a * a + b * b, d * d)
        if n.bit_length() > _BITS or m.bit_length() > _BITS:
            return FALLBACK
        # a**2 + b**2 as (a + b)**2 - 2ab.
        if not (_valid(n, m, _BITS) and n * d * d == ((a + b) * (a + b) - 2 * a * b) * m):
            return FALLBACK
        return trusted_rational(n, m)
    except Exception:
        return FALLBACK


# ---------------------------------------------------------------------------------------------------------------
# Square roots

def _exact_root(n, d):
    """The square root of the canonical rational n/d >= 0 as a canonical (p, q), or None when it is irrational."""
    p, q = isqrt(n), isqrt(d)
    if p * p == n and q * q == d:
        return p, q
    return None


def complex_sqrt(value):
    if current_budget() is not None:
        return FALLBACK
    try:
        z = _gauss(value, _SMALL)
        if z is None:
            return _generic("complex_sqrt", {"value": value})
        a, b, d = _common(*z)
        norm = a * a + b * b
        radius = isqrt(norm)
        if radius * radius != norm:
            return _generic("complex_sqrt", {"value": value})  # |z| is irrational.
        # With r = radius/d, the real part is the root of (r + a)/2 and the imaginary part, with the sign of b, the
        # root of (r - a)/2; both are non-negative.
        u = _exact_root(*_reduced(radius + a, 2 * d))
        v = _exact_root(*_reduced(radius - a, 2 * d))
        if u is None or v is None:
            return _generic("complex_sqrt", {"value": value})  # A half-sum is not a rational square.
        (up, uq), (vp, vq) = u, v
        if b < 0:
            vp = -vp
        # The principal root squared is the operand: real part u*u - v*v, imaginary part 2uv.
        if not (_valid(up, uq, _BITS) and _valid(vp, vq, _BITS) and up >= 0 and (up > 0 or vp >= 0)
                and (up * up * vq * vq - vp * vp * uq * uq) * d == a * uq * uq * vq * vq
                and 2 * up * vp * d == b * uq * vq):
            return FALLBACK
        return _complex(up, uq, vp, vq)
    except Exception:
        return FALLBACK


# ---------------------------------------------------------------------------------------------------------------
# Powers

def _gaussian_power(a, b, m):
    """(a + b i)**m for m > 0 by right-to-left binary exponentiation; None when an integer passes the limit."""
    ra, rb, pa, pb = 1, 0, a, b
    while m:
        if m & 1:
            ra, rb = ra * pa - rb * pb, ra * pb + rb * pa
            if ra.bit_length() > _BITS or rb.bit_length() > _BITS:
                return None
        m >>= 1
        if m:
            pa, pb = pa * pa - pb * pb, 2 * pa * pb
            if pa.bit_length() > _BITS or pb.bit_length() > _BITS:
                return None
    return ra, rb


def _gaussian_power_left(a, b, m):
    """The same power by left-to-right binary exponentiation."""
    ra, rb = a, b
    for bit in bin(m)[3:]:
        ra, rb = ra * ra - rb * rb, 2 * ra * rb
        if bit == "1":
            ra, rb = ra * a - rb * b, ra * b + rb * a
    return ra, rb


def complex_pow(base, n):
    if current_budget() is not None:
        return FALLBACK
    try:
        z = _gauss(base, _SMALL)
        if z is None or type(n) is not int:
            return _surd(_surd_pow, "complex_pow", {"base": base, "n": n}, base, n)
        if n.bit_length() > _EXPONENT_BITS:
            return FALLBACK
        if n == 0:
            return _complex(1, 1, 0, 1)  # Also for a zero base, as the structured path does.
        a, b, d = _common(*z)
        m = abs(n)
        if (d.bit_length() - 1) * m > _BITS:
            return FALLBACK  # d**m alone passes the limit.
        power = _gaussian_power(a, b, m)
        if power is None:
            return FALLBACK
        pa, pb = power
        e = d ** m
        if n > 0:
            parts = _parts(pa, pb, e)
            if parts is None:
                return FALLBACK
            xn, xd, yn, yd = parts
            if not (_valid(xn, xd, _BITS) and _valid(yn, yd, _BITS) and _gaussian_power_left(a, b, m) == power
                    and xn * e == pa * xd and yn * e == pb * yd):
                return FALLBACK
            return _complex(xn, xd, yn, yd)
        # z**-m = e (pa - pb i)/(pa**2 + pb**2).
        norm = pa * pa + pb * pb
        if norm == 0:
            return FALLBACK  # A zero base to a negative power: the structured path raises.
        parts = _parts(e * pa, -e * pb, norm)
        if parts is None:
            return FALLBACK
        xn, xd, yn, yd = parts
        # The answer times z**m is one: (xn/xd + (yn/yd) i) (pa + pb i)/e = 1.
        if not (_valid(xn, xd, _BITS) and _valid(yn, yd, _BITS) and _gaussian_power_left(a, b, m) == power
                and (xn * pa * yd - yn * pb * xd) == xd * yd * e and xn * pb * yd + yn * pa * xd == 0):
            return FALLBACK
        return _complex(xn, xd, yn, yd)
    except Exception:
        return FALLBACK


# ---------------------------------------------------------------------------------------------------------------
# Quadratic roots

def quadratic_roots(a, b, c):
    if current_budget() is not None:
        return FALLBACK
    try:
        fa, fb, fc = _scalar(a, _SMALL), _scalar(b, _SMALL), _scalar(c, _SMALL)
        if fa is None or fb is None or fc is None:
            return _generic("quadratic_roots", {"a": a, "b": b, "c": c})
        (an, ad), (bn, bd), (cn, cd) = fa, fb, fc
        if an == 0:
            return FALLBACK  # No leading coefficient: the structured path raises.
        # The integer polynomial qa x**2 + qb x + qc with the same roots, its leading coefficient positive.
        scale = lcm(ad, bd, cd)
        qa, qb, qc = an * (scale // ad), bn * (scale // bd), cn * (scale // cd)
        if qa < 0:
            qa, qb, qc = -qa, -qb, -qc
        disc = qb * qb - 4 * qa * qc
        s = isqrt(abs(disc))
        if s * s != abs(disc):  # Irrational roots.
            return _surd(_surd_quadratic, "quadratic_roots", {"a": a, "b": b, "c": c},
                         qa, qb, disc, an, ad, bn, bd, cn, cd)
        # The roots (-qb +- sqrt(disc))/(2 qa): real for disc >= 0, otherwise -qb/(2 qa) +- (s/(2 qa)) i. Since
        # 2 qa > 0 the sign - comes first in the increasing order that the structured path sorts the roots into.
        den = 2 * qa
        if disc == 0:
            roots = ((_reduced(-qb, den), (0, 1)),)
            multiplicity = 2
        elif disc > 0:
            roots = tuple((_reduced(-qb + sign * s, den), (0, 1)) for sign in (-1, 1))
            multiplicity = 1
        else:
            real = _reduced(-qb, den)
            roots = tuple((real, _reduced(sign * s, den)) for sign in (-1, 1))
            multiplicity = 1
        if not _quadratic_check(an, ad, bn, bd, cn, cd, roots, multiplicity):
            return FALLBACK
        return tuple(_root_record(re, im, multiplicity) for re, im in roots)
    except Exception:
        return FALLBACK


def _root_of(value, multiplicity):
    root = _new(ComplexRoot)
    _set(root, "root", value)
    _set(root, "multiplicity", multiplicity)
    return root


def _root_record(real, imag, multiplicity):
    return _root_of(_complex(real[0], real[1], imag[0], imag[1]), multiplicity)


def _quadratic_check(an, ad, bn, bd, cn, cd, roots, multiplicity):
    """Each root satisfies a x**2 + b x + c = 0 for a = an/ad, b = bn/bd, c = cn/cd; a double root also satisfies
    2 a x + b = 0; the roots increase strictly; the discriminant's sign agrees with their number."""
    # sign(b**2 - 4ac) from the cleared denominators bd**2 ad cd > 0.
    sign = bn * bn * ad * cd - 4 * an * cn * bd * bd
    if (sign == 0) != (multiplicity == 2) or len(roots) != (1 if multiplicity == 2 else 2):
        return False
    previous = None
    for (un, ud), (vn, vd) in roots:
        if not (_valid(un, ud, _BITS) and _valid(vn, vd, _BITS)):
            return False
        # Real part: a (u**2 - v**2) + b u + c = 0 and imaginary part: v (2 a u + b) = 0, over ad bd cd ud**2 vd**2.
        real = (an * bd * cd * (un * un * vd * vd - vn * vn * ud * ud) + bn * ad * cd * un * ud * vd * vd
                + cn * ad * bd * ud * ud * vd * vd)
        if real != 0 or vn * (2 * an * bd * un + bn * ad * ud) != 0:
            return False
        if multiplicity == 2 and (vn != 0 or 2 * an * bd * un + bn * ad * ud != 0):
            return False
        if previous is not None:
            (pun, pud), (pvn, pvd) = previous
            if un * pud != pun * ud:
                # Different real parts: the second must be larger.
                if not un * pud > pun * ud:
                    return False
            elif not vn * pvd > pvn * vd:
                return False
        previous = (un, ud), (vn, vd)
    return True


# ---------------------------------------------------------------------------------------------------------------
# Quadratic surds: the fields Q(sqrt b_1, ..., sqrt b_t)
#
# An irrational part of an operand that is a quadratic RealAlgebraicRoot is p + q sqrt(n) for a squarefree n. The
# squarefree numbers of one call are refined into pairwise coprime numbers b_1, ..., b_t, and every part is then an
# element of K = Q(sqrt b_1, ..., sqrt b_t). For pairwise coprime squarefree b_j > 1 the field K has degree 2**t over
# Q, the products of the sqrt b_j over the subsets of {1, ..., t} are a basis of K, and changing the signs of any of
# the sqrt b_j is an automorphism of K (Besicovitch's theorem; Galois theory of the compositum of quadratic fields).
# An element is the tuple of its 2**t coefficients, that of the subset with bit mask m at index m.
#
# The minimal polynomial of an element x over Q is therefore the product of (X - y) over the distinct images y of x
# under the sign changes, and the roots of it are all real. The structured path names an irrational real algebraic
# number by that polynomial, with primitive integer coefficients and a positive leading one, and the number of
# its roots below x; so does _encode. Everything the surd functions return is checked against its element by
# _represents: the polynomial vanishes at the element and at its conjugates, its degree is the number of
# conjugates, and the index is the number of conjugates below the element, which is decided with rational interval
# enclosures of the square roots (the producer decides it exactly, by the signs of elements).

_ZERO, _ONE = trusted_rational(0, 1), trusted_rational(1, 1)
_TRIAL = 2_000  # Trial division bound of _squarefree.
_ROOTS = 3  # Square roots in one field: degree at most 8.
_SURD_EXPONENT_BITS = 64


class _Decline(Exception):
    """The surd path leaves this call to the generic runner."""


def _squarefree(value):
    """(g, n) with value = g*g*n and n squarefree and above 1, for an integer value above 1; None when value is a
    square, or when what trial division leaves cannot be shown to be squarefree."""
    g, n, m, p = 1, 1, value, 2
    while p <= _TRIAL and p * p <= m:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            g *= p ** (e // 2)
            if e & 1:
                n *= p
        p += 1 if p == 2 else 2
    if m > 1:
        if p * p > m:  # No prime factor up to the square root: m is prime.
            n *= m
        elif m < p ** 3:  # Its prime factors are at least p: one, two equal or two different ones.
            root = isqrt(m)
            if root * root == m:
                g *= root
            else:
                n *= m
        else:
            return None
    return (g, n) if n > 1 else None


def _coprime_base(numbers):
    """Pairwise coprime squarefree numbers above 1 of which every given squarefree number is a product."""
    base, pending = [], [n for n in numbers if n > 1]
    while pending:
        n = pending.pop()
        if n == 1:
            continue
        for index, b in enumerate(base):
            g = gcd(n, b)
            if g > 1:
                del base[index]
                pending.extend((b // g, g, n // g))
                break
        else:
            base.append(n)
    return sorted(base)


class _Field:
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
        self.lower = _Field(self.base[:-1]) if base else None  # The field without the last square root.

    def constant(self, value):
        return (value,) + (_ZERO,) * (self.size - 1)

    def mask(self, n):
        """The subset of the base whose product is the squarefree number n."""
        mask = 0
        for j, b in enumerate(self.base):
            if n % b == 0:
                n //= b
                mask |= 1 << j
        if n != 1:
            raise _Decline
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
        """Rational (low, high) with low <= x <= high, from sqrt b within 2**-bits."""
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

    def bounded(self, *elements):
        for x in elements:
            for c in x:
                if c.numerator.bit_length() > _SMALL or c.denominator.bit_length() > _SMALL:
                    raise _Decline


def _quadratic_data(part):
    """(coefficients, index, g, n) of an irreducible quadratic RealAlgebraicRoot with Delta = g*g*n, else _Decline."""
    if type(part) is not RealAlgebraicRoot:
        raise _Decline
    coefficients, index = part.integer_coefficients, part.real_index
    if type(coefficients) is not tuple or len(coefficients) != 3 or type(index) is not int or index not in (0, 1):
        raise _Decline
    c0, c1, c2 = coefficients
    if (type(c0) is not int or type(c1) is not int or type(c2) is not int or c2 <= 0 or gcd(c0, c1, c2) != 1
            or max(c0.bit_length(), c1.bit_length(), c2.bit_length()) > _SMALL):
        raise _Decline
    delta = c1 * c1 - 4 * c0 * c2
    split = _squarefree(delta) if delta > 1 else None
    if split is None:
        raise _Decline  # No two real roots, a reducible quadratic, or a discriminant that is too hard to split.
    return coefficients, index, split[0], split[1]


def _operand_parts(value):
    kind = type(value)
    if kind is ComplexAlgebraic:
        return value.real, value.imag
    if kind is Rational:
        return value, _ZERO
    if kind is int and value.bit_length() <= _SMALL:
        return trusted_rational(value, 1), _ZERO
    raise _Decline


def _decode(values):
    """The field of the operands and their parts in it: (field, [((re, re_part), (im, im_part)), ...]); _Decline when
    an operand is neither a Gaussian rational nor a quadratic surd, or the field is Q or has too many square roots."""
    parsed, radicands = [], []
    for value in values:
        pair = []
        for part in _operand_parts(value):
            if type(part) is Rational:
                if not _valid(part.numerator, part.denominator, _SMALL):
                    raise _Decline
                pair.append((part, None))
            else:
                data = _quadratic_data(part)
                radicands.append(data[3])
                pair.append((part, data))
        parsed.append(pair)
    base = _coprime_base(radicands)
    if not base or len(base) > _ROOTS:
        raise _Decline
    field = _Field(base)
    return field, [tuple((field.constant(part) if data is None else field.quadratic(*data), part)
                         for part, data in pair) for pair in parsed]


def _represents(field, x, part):
    """Whether the Rational or RealAlgebraicRoot ``part`` denotes the element x of the field."""
    if type(part) is Rational:
        return _valid(part.numerator, part.denominator, _BITS) and x[0] == part and not any(x[1:])
    if type(part) is not RealAlgebraicRoot:
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


def _encode(field, x):
    """The Rational or RealAlgebraicRoot of the element x, checked against it."""
    if not any(x[1:]):
        part = x[0]
    else:
        orbit = field.orbit(x)
        polynomial = [field.one]
        for y in orbit:  # The product of X - y.
            product = [field.zero] * (len(polynomial) + 1)
            for i, c in enumerate(polynomial):
                product[i + 1] = field.add(product[i + 1], c)
                product[i] = field.sub(product[i], field.mul(y, c))
            polynomial = product
        if any(any(c[1:]) for c in polynomial):
            raise _Decline  # Not rational: no Galois orbit.
        rationals = [c[0] for c in polynomial]
        scale = lcm(*(c.denominator for c in rationals))
        integers = [c.numerator * (scale // c.denominator) for c in rationals]
        content = gcd(*integers)
        part = _new(RealAlgebraicRoot)
        _set(part, "integer_coefficients", tuple(c // content for c in integers))
        _set(part, "real_index", sum(field.sign(field.sub(x, y)) > 0 for y in orbit))
    if not _represents(field, x, part):
        raise _Decline
    return part


def _surd_value(field, real, imag):
    field.bounded(real, imag)
    value = _new(ComplexAlgebraic)
    _set(value, "real", _encode(field, real))
    _set(value, "imag", _encode(field, imag))
    return value


def _checked(field, operands):
    """The operands' parts as field elements, every one checked against the part it came from."""
    for operand in operands:
        for x, part in operand:
            if not _represents(field, x, part):
                raise _Decline
    return [[x for x, _ in operand] for operand in operands]


def _surd(attempt, operation, arguments, *values):
    """The surd attempt's value; the generic runner's when the attempt declines or fails."""
    try:
        value = attempt(*values)
    except Exception:
        value = None
    return _generic(operation, arguments) if value is None else value


def _complex_product(field, z, w):
    (a, b), (c, d) = z, w
    return field.sub(field.mul(a, c), field.mul(b, d)), field.add(field.mul(a, d), field.mul(b, c))


def _complex_product_checked(field, z, w):
    """The same product by three multiplications of field elements."""
    (a, b), (c, d) = z, w
    k1, k2, k3 = field.mul(c, field.add(a, b)), field.mul(a, field.sub(d, c)), field.mul(b, field.add(c, d))
    return field.sub(k1, k3), field.add(k1, k2)


def _surd_mul(left, right):
    field, operands = _decode((left, right))
    (a, b), (c, d) = _checked(field, operands)
    real, imag = _complex_product(field, (a, b), (c, d))
    if (real, imag) != _complex_product_checked(field, (a, b), (c, d)):
        raise _Decline
    return _surd_value(field, real, imag)


def _surd_div(left, right):
    field, operands = _decode((left, right))
    (a, b), (c, d) = _checked(field, operands)
    norm = field.add(field.mul(c, c), field.mul(d, d))
    inverse = field.inverse(norm)  # ZeroDivisionError for a zero divisor.
    real = field.mul(field.add(field.mul(a, c), field.mul(b, d)), inverse)
    imag = field.mul(field.sub(field.mul(b, c), field.mul(a, d)), inverse)
    # The answer times the divisor is the dividend.
    if (field.sub(field.mul(real, c), field.mul(imag, d)) != a
            or field.add(field.mul(real, d), field.mul(imag, c)) != b):
        raise _Decline
    return _surd_value(field, real, imag)


def _surd_pow(base, n):
    if type(n) is not int or n.bit_length() > _SURD_EXPONENT_BITS:
        raise _Decline
    field, operands = _decode((base,))
    ((a, b),) = _checked(field, operands)
    if n == 0:
        return _complex(1, 1, 0, 1)  # The structured path's answer for every valid base.
    m = abs(n)
    result, power = (field.one, field.zero), (a, b)
    while m:
        if m & 1:
            result = _complex_product(field, result, power)
            field.bounded(*result)
        m >>= 1
        if m:
            power = _complex_product(field, power, power)
            field.bounded(*power)
    # The same power by left-to-right binary exponentiation, with the three-multiplication product.
    check = (a, b)
    for bit in bin(abs(n))[3:]:
        check = _complex_product_checked(field, check, check)
        if bit == "1":
            check = _complex_product_checked(field, check, (a, b))
    if check != result:
        raise _Decline
    if n < 0:
        x, y = result
        inverse = field.inverse(field.add(field.mul(x, x), field.mul(y, y)))  # ZeroDivisionError for a zero base.
        answer = field.mul(x, inverse), field.neg(field.mul(y, inverse))
        # The answer times the power is one.
        if _complex_product_checked(field, answer, result) != (field.one, field.zero):
            raise _Decline
        result = answer
    return _surd_value(field, *result)


# The parts of complex_algebraic are expressions. The lean path reads those built from Rationals, Sqrt, quadratic
# RealAlgebraicRoots, Add, Mul and Pow with a small exponent; the limits keep it far below the depth and the node
# count at which the structured path's records refuse an expression, so that an expression of the structured path's
# limits is never answered here.
_EXPRESSION_NODES = 200
_EXPRESSION_DEPTH = 16
_EXPRESSION_POWER = 256


def _rational_root(r):
    """(g, q, n) with sqrt(r) = (g/q) sqrt(n) for a canonical Rational r >= 0: n = 1 for a rational root, else
    squarefree; _Decline when the squarefree part cannot be found."""
    product = r.numerator * r.denominator
    if product == 0:
        return 0, 1, 1
    root = isqrt(product)
    if root * root == product:
        return root, r.denominator, 1
    split = _squarefree(product)
    if split is None:
        raise _Decline
    return split[0], r.denominator, split[1]


def _scan(expression):
    """The squarefree radicands of the square roots and quadratic roots of an expression; _Decline for a shape that
    the lean path does not read, or one that is large."""
    radicands, pending, nodes = [], [(expression, 0)], 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        kind = type(node)
        if nodes > _EXPRESSION_NODES or depth > _EXPRESSION_DEPTH:
            raise _Decline
        if kind is Rational:
            if not _valid(node.numerator, node.denominator, _SMALL):
                raise _Decline
        elif kind is Sqrt:
            radicand = node.radicand
            if (type(radicand) is not Rational or radicand.numerator < 0
                    or not _valid(radicand.numerator, radicand.denominator, _SMALL)):
                raise _Decline
            n = _rational_root(radicand)[2]
            if n > 1:
                radicands.append(n)
        elif kind is RealAlgebraicRoot:
            radicands.append(_quadratic_data(node)[3])
        elif kind is Add:
            pending.extend((term, depth + 1) for term in node.terms)
        elif kind is Mul:
            pending.extend((factor, depth + 1) for factor in node.factors)
        elif kind is Pow:
            if type(node.exponent) is not int or not 0 <= node.exponent <= _EXPRESSION_POWER:
                raise _Decline
            pending.append((node.base, depth + 1))
        else:
            raise _Decline
    return radicands


def _element(field, node):
    """The value of a scanned expression as an element of the field: sums and products folded from the left, powers
    by repeated squaring."""
    kind = type(node)
    if kind is Rational:
        return field.constant(node)
    if kind is Sqrt:
        g, q, n = _rational_root(node.radicand)
        x = [_ZERO] * field.size
        x[field.mask(n) if n > 1 else 0] = Rational(g, q)
        return tuple(x)
    if kind is RealAlgebraicRoot:
        return field.quadratic(*_quadratic_data(node))
    if kind is Add:
        total = field.zero
        for term in node.terms:
            total = field.add(total, _element(field, term))
        return total
    if kind is Mul:
        product = field.one
        for factor in node.factors:
            product = field.mul(product, _element(field, factor))
        return product
    base, result = _element(field, node.base), field.one
    for bit in bin(node.exponent)[2:]:
        result = field.mul(result, result)
        if bit == "1":
            result = field.mul(result, base)
    return result


def _recompute(field, node):
    """The value again, as the check: every square root squares to its radicand and is not negative, every
    quadratic root is checked against its descriptor, sums and products are folded from the right and powers are
    repeated products."""
    kind = type(node)
    if kind is Rational:
        return field.constant(node)
    if kind is Sqrt:
        x = _element(field, node)
        order = field.compare(x, field.zero)
        if field.mul(x, x) != field.constant(node.radicand) or order is None or order < 0:
            raise _Decline
        return x
    if kind is RealAlgebraicRoot:
        x = field.quadratic(*_quadratic_data(node))
        if not _represents(field, x, node):
            raise _Decline
        return x
    if kind is Add:
        total = field.zero
        for term in reversed(node.terms):
            total = field.add(_recompute(field, term), total)
        return total
    if kind is Mul:
        product = field.one
        for factor in reversed(node.factors):
            product = field.mul(_recompute(field, factor), product)
        return product
    base, result = _recompute(field, node.base), field.one
    for _ in range(node.exponent):
        result = field.mul(base, result)
    return result


def _sqrt_interval(r):
    """Rational (low, high) around sqrt(r) for a Rational r >= 0, within 2**-64."""
    scale = 1 << 64
    low = isqrt(r.numerator * scale * scale // r.denominator)
    return Rational(low, scale), Rational(low + 1, scale)


def _interval(node):
    """A rational interval around the value of a scanned expression, by interval arithmetic on the expression alone."""
    kind = type(node)
    if kind is Rational:
        return node, node
    if kind is Sqrt:
        return _sqrt_interval(node.radicand)
    if kind is RealAlgebraicRoot:
        c0, c1, c2 = node.integer_coefficients
        low, high = _sqrt_interval(Rational(c1 * c1 - 4 * c0 * c2))
        shift, half = Rational(-c1, 2 * c2), Rational(1, 2 * c2)
        if node.real_index:
            return shift + low * half, shift + high * half
        return shift - high * half, shift - low * half
    if kind is Add:
        low = high = _ZERO
        for term in node.terms:
            term_low, term_high = _interval(term)
            low, high = low + term_low, high + term_high
        return low, high
    if kind is Mul:
        low = high = _ONE
        for factor in node.factors:
            factor_low, factor_high = _interval(factor)
            products = low * factor_low, low * factor_high, high * factor_low, high * factor_high
            low, high = min(products), max(products)
        return low, high
    base_low, base_high = _interval(node.base)
    low = high = _ONE
    for _ in range(node.exponent):
        products = low * base_low, low * base_high, high * base_low, high * base_high
        low, high = min(products), max(products)
    return low, high


def _surd_algebraic(real, imag):
    parts = []
    for part in (real, imag):
        if type(part) is int:
            if part.bit_length() > _SMALL:
                raise _Decline
            part = trusted_rational(part, 1)
        parts.append(part)
    radicands = []
    for part in parts:
        radicands += _scan(part)
    base = _coprime_base(radicands)
    if len(base) > _ROOTS:
        raise _Decline
    field = _Field(base)
    elements = [_element(field, part) for part in parts]
    for part, x in zip(parts, elements):
        # The value read again another way, and consistent with the interval arithmetic of the expression.
        if _recompute(field, part) != x:
            raise _Decline
        low, high = _interval(part)
        x_low, x_high = field.enclosure(x, 64)
        if max(low, x_low) > min(high, x_high):
            raise _Decline
    return _surd_value(field, *elements)


def _surd_quadratic(qa, qb, disc, an, ad, bn, bd, cn, cd):
    """The roots of qa x**2 + qb x + qc, for qa > 0 and a discriminant that is no square and no minus a square, as
    elements of Q(sqrt n) where g*g*n = |disc| with n squarefree: -qb/(2 qa) +- (g/(2 qa)) sqrt(n) for a positive
    discriminant and -qb/(2 qa) +- (g/(2 qa)) sqrt(n) i for a negative one, in increasing order. They are checked
    against the polynomial with the coefficients an/ad, bn/bd and cn/cd, and against each other."""
    split = _squarefree(abs(disc))
    if split is None:
        raise _Decline
    g, n = split
    field = _Field([n])
    den = 2 * qa
    shift = field.constant(Rational(-qb, den))
    roots = []
    for sign in (-1, 1):
        radical = (_ZERO, Rational(sign * g, den))  # The coefficient of sqrt(n) is the one of the subset {1}.
        roots.append((field.add(shift, radical), field.zero) if disc > 0 else (shift, radical))
    a, b, c = (field.constant(Rational(x, y)) for x, y in ((an, ad), (bn, bd), (cn, cd)))
    previous = None
    for real, imag in roots:
        # (a x + b) x + c is zero at x = real + imag i.
        high_real, high_imag = field.add(field.mul(a, real), b), field.mul(a, imag)
        if (field.add(field.sub(field.mul(high_real, real), field.mul(high_imag, imag)), c) != field.zero
                or field.add(field.mul(high_real, imag), field.mul(high_imag, real)) != field.zero):
            raise _Decline
        if previous is not None:
            order = field.compare(previous[0], real)  # The roots increase strictly: by real part, then imaginary part.
            if order is None or order > 0 or (order == 0 and field.compare(previous[1], imag) != -1):
                raise _Decline
        previous = real, imag
    return tuple(_root_of(_surd_value(field, real, imag), 1) for real, imag in roots)


LEAN = {
    "complex_algebraic": complex_algebraic,
    "complex_add": complex_add,
    "complex_sub": complex_sub,
    "complex_mul": complex_mul,
    "complex_div": complex_div,
    "complex_conjugate": complex_conjugate,
    "complex_real": complex_real,
    "complex_imag": complex_imag,
    "complex_norm_squared": complex_norm_squared,
    "complex_sqrt": complex_sqrt,
    "complex_pow": complex_pow,
    "quadratic_roots": quadratic_roots,
}
