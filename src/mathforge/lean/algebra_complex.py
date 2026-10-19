"""Lean complex scalar operations: the Gaussian-rational path.

``mathforge.complex_*`` and ``mathforge.quadratic_roots`` called without ``limits`` first call these functions. An
operand that is an int, a ``Rational`` or a ``ComplexAlgebraic`` whose two parts are ``Rational`` is a Gaussian
rational, an element of Q(i). On such operands the function computes the answer with plain integer arithmetic,
checks it by a second computation of a different shape and builds the ``ComplexAlgebraic`` from trusted Rationals.
The structured path builds the same value from certified real-algebraic arithmetic, which for Gaussian operands
only ever meets rational numbers, so the two values are equal (``tests/v08/complex``).

Everything else, an operand with an irrational part or a call that is no valid Gaussian one, is the generic runner's
(``run_value``: the family's producer and its v0.7 checker, as before) or, for a call that the structured path
refuses, ``FALLBACK``.

The checks, one per operation, recompute the answer another way and compare by cross-multiplication, so that no
reduced fraction is compared with a fraction produced by the same code:

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

from ..complex_numbers import ComplexAlgebraic, ComplexRoot
from ..limits import ComputationLimits, current_budget
from ..model import Rational, trusted_rational
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
            return _generic("complex_algebraic", {"real": real, "imag": imag})
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
            part = value.real
            if type(part) is Rational and _valid(part.numerator, part.denominator, _BITS):
                return part
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
            part = value.imag
            if type(part) is Rational and _valid(part.numerator, part.denominator, _BITS):
                return part
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
            return _generic("complex_mul", {"left": left, "right": right})
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
            return _generic("complex_div", {"left": left, "right": right})
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
            return _generic("complex_pow", {"base": base, "n": n})
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
        if s * s != abs(disc):
            return _generic("quadratic_roots", {"a": a, "b": b, "c": c})  # Irrational roots.
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


def _root_record(real, imag, multiplicity):
    root = _new(ComplexRoot)
    _set(root, "root", _complex(real[0], real[1], imag[0], imag[1]))
    _set(root, "multiplicity", multiplicity)
    return root


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
