"""Checks of answers on Gaussian rationals, Q(i): each recomputes the answer another way and compares by
cross-multiplication, so that no reduced fraction is compared with a fraction produced by the same code.

An operand comes as (a, b, d), its value (a + b i)/d with d > 0; an answer as its parts (xn, xd, yn, yd), the real part
xn/xd and the imaginary part yn/yd, which must be canonical rationals within the default max_integer_bits.

* sum and difference: the other operand subtracted from, or added to, the answer gives the first operand back;
* product: the three-multiplication (Karatsuba) expansion of the product of the two numerators;
* quotient: the answer times the divisor gives the dividend back;
* conjugate: the parts against the operand's;
* squared modulus: (a + b)**2 - 2ab for a**2 + b**2;
* square root: the root squared gives the operand back, on the principal branch (non-negative real part, and a
  non-negative imaginary part when the real part is zero);
* power: the power recomputed by left-to-right binary exponentiation (the producer's runs right to left); a negative
  power is checked by multiplying back;
* quadratic roots: each root substituted into the polynomial (and, for a double root, into its derivative), the
  roots in strictly increasing order, and the discriminant's sign against the number of roots.
"""

from __future__ import annotations

from math import gcd

from ...limits import ComputationLimits

_BITS = ComputationLimits().max_integer_bits


def canonical(n, d, limit: int = _BITS) -> bool:
    """Whether n/d is a canonical rational with integers of at most ``limit`` bits."""
    return (type(n) is int and type(d) is int and d > 0 and n.bit_length() <= limit and d.bit_length() <= limit
            and (d == 1 or gcd(n, d) == 1))


def _parts(answer) -> bool:
    xn, xd, yn, yd = answer
    return canonical(xn, xd) and canonical(yn, yd)


def check_sum(z, w, answer, subtract: bool = False) -> bool:
    """answer = z + w (z - w when subtract): the answer less (plus) w is z."""
    if not _parts(answer):
        return False
    (za, zb, zd), (wa, wb, wd), (xn, xd, yn, yd) = z, w, answer
    if subtract:
        wa, wb = -wa, -wb
    return (xn * wd - wa * xd) * zd == za * xd * wd and (yn * wd - wb * yd) * zd == zb * yd * wd


def check_product(z, w, answer) -> bool:
    """answer = z w, against the product by three multiplications."""
    if not _parts(answer):
        return False
    (za, zb, zd), (wa, wb, wd), (xn, xd, yn, yd) = z, w, answer
    d = zd * wd
    k1, k2, k3 = wa * (za + zb), za * (wb - wa), zb * (wa + wb)
    return xn * d == (k1 - k3) * xd and yn * d == (k1 + k2) * yd


def check_quotient(z, w, answer) -> bool:
    """answer = z / w: the answer times w is z."""
    if not _parts(answer):
        return False
    (za, zb, zd), (wa, wb, wd), (xn, xd, yn, yd) = z, w, answer
    return ((xn * wa * yd - yn * wb * xd) * zd == za * xd * yd * wd
            and (xn * wb * yd + yn * wa * xd) * zd == zb * xd * yd * wd)


def check_conjugate(parts, value) -> bool:
    """value (a ComplexAlgebraic, read through its parts) is the conjugate of an/ad + (bn/bd) i."""
    an, ad, bn, bd = parts
    real, imag = getattr(value, "real", None), getattr(value, "imag", None)
    return (canonical(getattr(real, "numerator", None), getattr(real, "denominator", None))
            and canonical(getattr(imag, "numerator", None), getattr(imag, "denominator", None))
            and real.numerator == an and real.denominator == ad and imag.numerator + bn == 0
            and imag.denominator == bd)


def check_norm_squared(z, n: int, m: int) -> bool:
    """n/m = |z|**2, as (a + b)**2 - 2ab over d**2."""
    a, b, d = z
    return canonical(n, m) and n * d * d == ((a + b) * (a + b) - 2 * a * b) * m


def check_sqrt(z, answer) -> bool:
    """answer is the principal square root of z: its square is z, its real part is non-negative, and its imaginary
    part is non-negative when its real part is zero."""
    if not _parts(answer):
        return False
    (a, b, d), (up, uq, vp, vq) = z, answer
    return (up >= 0 and (up > 0 or vp >= 0)
            and (up * up * vq * vq - vp * vp * uq * uq) * d == a * uq * uq * vq * vq
            and 2 * up * vp * d == b * uq * vq)


def _power_left(a: int, b: int, m: int) -> tuple[int, int]:
    """(a + b i)**m for m > 0 by left-to-right binary exponentiation."""
    ra, rb = a, b
    for bit in bin(m)[3:]:
        ra, rb = ra * ra - rb * rb, 2 * ra * rb
        if bit == "1":
            ra, rb = ra * a - rb * b, ra * b + rb * a
    return ra, rb


def check_power(z, n: int, answer) -> bool:
    """answer = z**n for an int n != 0 and a nonzero z: against the power recomputed left to right, times the
    answer for a negative n."""
    if not _parts(answer) or type(n) is not int or n == 0:
        return False
    (a, b, d), (xn, xd, yn, yd) = z, answer
    m = abs(n)
    pa, pb = _power_left(a, b, m)
    e = d ** m
    if n > 0:
        return xn * e == pa * xd and yn * e == pb * yd
    # (xn/xd + (yn/yd) i) (pa + pb i)/e = 1.
    return (xn * pa * yd - yn * pb * xd) == xd * yd * e and xn * pb * yd + yn * pa * xd == 0


def check_quadratic_roots(an, ad, bn, bd, cn, cd, roots, multiplicity) -> bool:
    """Each root satisfies a x**2 + b x + c = 0 for a = an/ad, b = bn/bd, c = cn/cd; a double root also satisfies
    2 a x + b = 0; the roots increase strictly; the discriminant's sign agrees with their number. A root comes as
    ((un, ud), (vn, vd)), its value un/ud + (vn/vd) i."""
    # sign(b**2 - 4ac) from the cleared denominators bd**2 ad cd > 0.
    sign = bn * bn * ad * cd - 4 * an * cn * bd * bd
    if (sign == 0) != (multiplicity == 2) or len(roots) != (1 if multiplicity == 2 else 2):
        return False
    previous = None
    for (un, ud), (vn, vd) in roots:
        if not (canonical(un, ud) and canonical(vn, vd)):
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
