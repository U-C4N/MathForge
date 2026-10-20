"""Lean complex scalar operations: Gaussian rationals and quadratic surds.

``mathforge.complex_*`` and ``mathforge.quadratic_roots`` called without ``limits`` first call these functions. An
operand that is an int, a ``Rational`` or a ``ComplexAlgebraic`` whose two parts are ``Rational`` is a Gaussian
rational, an element of Q(i). On such operands the function computes the answer with plain integer arithmetic, has
it checked by mathforge.certify.complex.gaussian, which recomputes it another way, and builds the
``ComplexAlgebraic`` from trusted Rationals. The structured path builds the same value from certified real-algebraic
arithmetic, which for Gaussian operands only ever meets rational numbers, so the two values are equal
(``tests/v08/complex``).

A product, quotient or power of operands whose irrational parts are quadratic surds (a ``RealAlgebraicRoot`` of
degree two), the roots of a rational quadratic with irrational roots, and ``complex_algebraic`` of expressions of
square roots take the second path, described in front of ``_squarefree``: exact arithmetic in a field
Q(sqrt b_1, ..., sqrt b_t) (mathforge.trusted.surd), each irrational part named as the structured path names it, and
every part, product, quotient, power and root checked by mathforge.certify.complex.surd.

This module holds the producers and their wiring only: every check it calls lives in mathforge.certify.complex.

Everything else, an operand with another irrational part or a call that is no valid one, is the generic runner's
(``run_value``: the family's producer and its v0.7 checker, as before) or, for a call that the structured path
refuses, ``FALLBACK``.

Every integer in an operand and in an answer stays within ``max_integer_bits``; the multiplicative operations take
operands of at most a sixteenth of that, so that no step of theirs holds an integer that the structured path would
refuse. A larger operand, a result past the limit, a zero divisor and a call made inside an active budget are left
to the structured path (``FALLBACK``), which charges that budget or raises v0.7's error. Exponents are limited to
256 bits for the same reason: the structured path's work and evidence for them stay far from their limits.
"""

from __future__ import annotations

from math import gcd, isqrt, lcm

from ..algebraic import RealAlgebraicRoot
from ..certify.complex import gaussian as gaussian_checks
from ..certify.complex import surd as surd_checks
from ..complex_numbers import ComplexAlgebraic, ComplexRoot
from ..limits import ComputationLimits, current_budget
from ..model import Add, Mul, Pow, Rational, Sqrt, trusted_rational
from ..trusted.surd import Field as _Field
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
        if not gaussian_checks.check_sum((za, zb, zd), (wa, wb, wd), parts):
            return FALLBACK
        return _complex(*parts)
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
        if not gaussian_checks.check_sum((za, zb, zd), (wa, wb, wd), parts, subtract=True):
            return FALLBACK
        return _complex(*parts)
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
        if not gaussian_checks.check_conjugate(z, result):
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
        if not gaussian_checks.check_product((za, zb, zd), (wa, wb, wd), parts):
            return FALLBACK
        return _complex(*parts)
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
        if not gaussian_checks.check_quotient((za, zb, zd), (wa, wb, wd), parts):
            return FALLBACK
        return _complex(*parts)
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
        if not gaussian_checks.check_norm_squared((a, b, d), n, m):
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
        if not gaussian_checks.check_sqrt((a, b, d), (up, uq, vp, vq)):
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
            if not gaussian_checks.check_power((a, b, d), n, parts):
                return FALLBACK
            return _complex(*parts)
        # z**-m = e (pa - pb i)/(pa**2 + pb**2).
        norm = pa * pa + pb * pb
        if norm == 0:
            return FALLBACK  # A zero base to a negative power: the structured path raises.
        parts = _parts(e * pa, -e * pb, norm)
        if parts is None:
            return FALLBACK
        if not gaussian_checks.check_power((a, b, d), n, parts):
            return FALLBACK
        return _complex(*parts)
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
        if not gaussian_checks.check_quadratic_roots(an, ad, bn, bd, cn, cd, roots, multiplicity):
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


# ---------------------------------------------------------------------------------------------------------------
# Quadratic surds: the fields Q(sqrt b_1, ..., sqrt b_t)
#
# An irrational part of an operand that is a quadratic RealAlgebraicRoot is p + q sqrt(n) for a squarefree n. The
# squarefree numbers of one call are refined into pairwise coprime numbers b_1, ..., b_t, and every part is then an
# element of K = Q(sqrt b_1, ..., sqrt b_t), computed with mathforge.trusted.surd.Field. That K has degree 2**t, with
# the sign changes of the sqrt b_j as automorphisms, is Besicovitch's theorem; its hypotheses (each b_j squarefree,
# pairwise coprime, above 1) are established here by exact computation: _squarefree splits a radicand by trial
# division and declines when what is left cannot be shown squarefree, and _coprime_base refines the radicands by gcds.
#
# The minimal polynomial of an element x over Q is therefore the product of (X - y) over the distinct images y of x
# under the sign changes, and the roots of it are all real. The structured path names an irrational real algebraic
# number by that polynomial, with primitive integer coefficients and a positive leading one, and the number of
# its roots below x; so does _encode, which decides the index exactly by the signs of elements. Everything the surd
# functions return is checked by mathforge.certify.complex.surd: each name against its element, and each product,
# quotient, power and root by another formula and by interval arithmetic on the operands' enclosures.

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
    if not surd_checks.represents(field, x, part):
        raise _Decline
    return part


def _surd_value(field, real, imag):
    if not field.bounded(_SMALL, real, imag):
        raise _Decline
    value = _new(ComplexAlgebraic)
    _set(value, "real", _encode(field, real))
    _set(value, "imag", _encode(field, imag))
    return value


def _checked(field, operands):
    """The operands' parts as field elements, every one checked against the part it came from."""
    for operand in operands:
        for x, part in operand:
            if not surd_checks.represents(field, x, part):
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


def _surd_mul(left, right):
    field, operands = _decode((left, right))
    (a, b), (c, d) = _checked(field, operands)
    real, imag = _complex_product(field, (a, b), (c, d))
    if not surd_checks.check_product(field, (a, b), (c, d), (real, imag)):
        raise _Decline
    return _surd_value(field, real, imag)


def _surd_div(left, right):
    field, operands = _decode((left, right))
    (a, b), (c, d) = _checked(field, operands)
    norm = field.add(field.mul(c, c), field.mul(d, d))
    inverse = field.inverse(norm)  # ZeroDivisionError for a zero divisor.
    real = field.mul(field.add(field.mul(a, c), field.mul(b, d)), inverse)
    imag = field.mul(field.sub(field.mul(b, c), field.mul(a, d)), inverse)
    if not surd_checks.check_quotient(field, (a, b), (c, d), (real, imag)):
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
            if not field.bounded(_SMALL, *result):
                raise _Decline
        m >>= 1
        if m:
            power = _complex_product(field, power, power)
            if not field.bounded(_SMALL, *power):
                raise _Decline
    if n < 0:
        x, y = result
        inverse = field.inverse(field.add(field.mul(x, x), field.mul(y, y)))  # ZeroDivisionError for a zero base.
        result = field.mul(x, inverse), field.neg(field.mul(y, inverse))
    if not surd_checks.check_power(field, (a, b), n, result):
        raise _Decline
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
    the lean path does not read, or one that is large. Large includes a leaf whose length (bits), times the product
    of the exponents of the powers above it, passes _SMALL: its power alone would hold integers of about that length,
    past what _surd_value accepts, and nested powers would reach millions of bits before any check."""
    radicands, pending, nodes = [], [(expression, 0, 1)], 0
    while pending:
        node, depth, weight = pending.pop()
        nodes += 1
        kind = type(node)
        if nodes > _EXPRESSION_NODES or depth > _EXPRESSION_DEPTH:
            raise _Decline
        if kind is Rational:
            if (not _valid(node.numerator, node.denominator, _SMALL)
                    or weight * max(node.numerator.bit_length(), node.denominator.bit_length()) > _SMALL):
                raise _Decline
        elif kind is Sqrt:
            radicand = node.radicand
            if (type(radicand) is not Rational or radicand.numerator < 0
                    or not _valid(radicand.numerator, radicand.denominator, _SMALL)
                    or weight * max(radicand.numerator.bit_length(), radicand.denominator.bit_length()) > _SMALL):
                raise _Decline
            n = _rational_root(radicand)[2]
            if n > 1:
                radicands.append(n)
        elif kind is RealAlgebraicRoot:
            data = _quadratic_data(node)
            if weight * (1 + max(c.bit_length() for c in data[0])) > _SMALL:
                raise _Decline
            radicands.append(data[3])
        elif kind is Add:
            pending.extend((term, depth + 1, weight) for term in node.terms)
        elif kind is Mul:
            pending.extend((factor, depth + 1, weight) for factor in node.factors)
        elif kind is Pow:
            if type(node.exponent) is not int or not 0 <= node.exponent <= _EXPRESSION_POWER:
                raise _Decline
            pending.append((node.base, depth + 1, weight * max(node.exponent, 1)))
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
        if not surd_checks.check_algebraic(field, part, x):
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
    if not surd_checks.check_quadratic_roots(field, a, b, c, roots):
        raise _Decline
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
