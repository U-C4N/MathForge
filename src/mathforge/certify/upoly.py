"""Independent checkers over dense univariate polynomials.

They verify a producer's claim without its algorithm: a derivative against the falling factorials of math.perm, an
antiderivative by differentiating it, a division, a gcd or a square-free decomposition through product identities
decided by one Kronecker evaluation (trusted.upoly.kronecker), coprimality modulo a prime, a resultant by the
Euclidean algorithm modulo a prime larger than twice its Hadamard bound, and real root counts by their own Sturm
chains (subresultant remainders, normalized differently from the producer's). The irreducibility check reimplements
v0.7's ``polynomial_exact._check_irreducible`` and ``_rabin`` on trusted.upoly.

Rational coefficient tuples are canonical Rationals without trailing zeros, ``(Rational(0),)`` for zero. Every
function returns a bool and never trusts its arguments' types beyond what it checks.
"""

from __future__ import annotations

from math import gcd, perm

from ..model import Rational
from ..trusted import upoly as tp


def canonical(value) -> bool:
    """A Rational in lowest terms with a positive denominator."""
    if type(value) is not Rational:
        return False
    denominator = value.denominator
    return denominator == 1 or (denominator > 1 and gcd(value.numerator, denominator) == 1)


def _is_zero(coefficients) -> bool:
    return len(coefficients) == 1 and type(coefficients[0]) is Rational and not coefficients[0].numerator


def canonical_tuple(coefficients) -> bool:
    """A tuple of canonical Rationals without a trailing zero, or (0,)."""
    if type(coefficients) is not tuple or not coefficients:
        return False
    if len(coefficients) > 1 and (type(coefficients[-1]) is not Rational or not coefficients[-1].numerator):
        return False
    for value in coefficients:
        if type(value) is not Rational:
            return False
        denominator = value.denominator
        if denominator != 1 and (denominator < 1 or gcd(value.numerator, denominator) != 1):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# Calculus

def check_derivative(coefficients, n: int, result) -> bool:
    """result is the n-th derivative: result_j = c_(j+n) * (j+n)! / j!, the factor from math.perm, each a canonical
    Rational."""
    degree = len(coefficients) - 1
    if n == 0:
        return result == coefficients and canonical_tuple(result)
    if _is_zero(coefficients) or n > degree:
        return canonical_tuple(result) and _is_zero(result)
    if type(result) is not tuple or len(result) != degree - n + 1 or not result[-1].numerator:
        return False
    tp.charge(len(result))
    for value, source, k in zip(result, coefficients[n:], range(n, degree + 1)):
        if type(value) is not Rational:
            return False
        denominator = value.denominator
        if denominator == 1:
            if value.numerator * source.denominator != source.numerator * perm(k, n):
                return False
        elif (denominator < 1 or gcd(value.numerator, denominator) != 1
              or value.numerator * source.denominator != source.numerator * perm(k, n) * denominator):
            return False
    return True


def check_antiderivative(coefficients, result) -> bool:
    """result has constant term 0 and derivative coefficients: (k + 1) result_(k+1) = c_k."""
    if not canonical_tuple(result):
        return False
    if _is_zero(coefficients):
        return _is_zero(result)
    if len(result) != len(coefficients) + 1 or result[0].numerator:
        return False
    tp.charge(len(result))
    for k, source in enumerate(coefficients):
        value = result[k + 1]
        if value.numerator * (k + 1) * source.denominator != source.numerator * value.denominator:
            return False
    return True


def check_definite_integral(coefficients, lower, upper, value) -> bool:
    """value = sum of c_k (upper**(k+1) - lower**(k+1)) / (k + 1), v0.7's contributions, summed exactly."""
    if not canonical(value):
        return False
    tp.charge(2 * len(coefficients))
    total = Rational(0)
    low, high = Rational(1), Rational(1)
    for k, source in enumerate(coefficients, 1):
        low, high = low * lower, high * upper
        if source.numerator:
            total = total + source * (high - low) / k
    return total == value


# ---------------------------------------------------------------------------------------------------------------
# Coprimality and gcds over Z

# Mersenne primes 2**k - 1 (Lucas-Lehmer): moduli of the coprimality tests, and of the resultant check, which needs
# one larger than twice the Hadamard bound. Only units are ever inverted, so a wrong entry could only make a check
# fail (pow raises ValueError), never pass.
_MERSENNE_EXPONENTS = (61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937,
                       21701, 23209, 44497, 86243)
_SMALL_PRIMES = ((1 << 61) - 1, (1 << 31) - 1, (1 << 89) - 1, (1 << 107) - 1, (1 << 127) - 1)


def coprime(u: tuple[int, ...], v: tuple[int, ...]) -> bool:
    """No nonconstant common factor over Q of the nonzero integer polynomials u and v.

    Sufficient: gcd(u mod p, v mod p) = 1 for a prime p dividing neither leading coefficient. A common factor w over
    Q may be taken primitive in Z[x] (Gauss); p divides neither its leading coefficient nor, so w mod p keeps its
    degree and divides both residues. An unlucky prime only makes the test fail; five primes are tried."""
    if not u or not v:
        return False
    if len(u) == 1 or len(v) == 1:
        return True
    for prime in _SMALL_PRIMES:
        if u[-1] % prime and v[-1] % prime:
            if tp.gcd_mod(tp.mod(u, prime), tp.mod(v, prime), prime) == (1,):
                return True
    return False


def product_identity(product: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    """product == left * right over Z, by one Kronecker evaluation."""
    if not left or not right:
        return not product
    if len(product) != len(left) + len(right) - 1:
        return False
    bits = tp.kronecker_bits(tp.max_bits(product),
                             tp.max_bits(left) + tp.max_bits(right) + min(len(left), len(right)).bit_length())
    tp.charge(len(product) + len(left) + len(right))
    return tp.kronecker(product, bits) == tp.kronecker(left, bits) * tp.kronecker(right, bits)


def check_gcd(f: tuple[int, ...], g: tuple[int, ...], h: tuple[int, ...], u: tuple[int, ...],
              v: tuple[int, ...]) -> bool:
    """h is a gcd of the nonzero integer polynomials f and g over Q: f = h u, g = h v and u, v coprime."""
    return bool(h) and product_identity(f, h, u) and product_identity(g, h, v) and coprime(u, v)


def monic_tuple(values) -> bool:
    """A canonical rational coefficient tuple of a nonzero polynomial with leading coefficient 1."""
    return canonical_tuple(values) and values[-1].numerator == 1 and values[-1].denominator == 1


def integer_form(values) -> tuple[int, ...]:
    """The primitive integer polynomial with positive leading coefficient of a nonzero rational one."""
    return tp.primitive(tp.from_rationals(values)[0])


def _linear_identity(terms, target) -> bool:
    """sum of a_i * b_i == target for rational coefficient tuples (pairs (a_i, b_i), target), by clearing every
    denominator and one Kronecker evaluation."""
    integers = [(tp.from_rationals(a), tp.from_rationals(b)) for a, b in terms]
    target_values, target_denominator = tp.from_rationals(target)
    denominators = [da * db for (_, da), (_, db) in integers]
    common = target_denominator
    for value in denominators:
        common *= value
    bounds = [tp.max_bits(target_values) + (common // target_denominator).bit_length()]
    for ((a, da), (b, db)), denominator in zip(integers, denominators):
        if a and b:
            bounds.append(tp.max_bits(a) + tp.max_bits(b) + min(len(a), len(b)).bit_length()
                          + (common // denominator).bit_length())
    bits = tp.kronecker_bits(*bounds)
    total = 0
    for ((a, _), (b, _)), denominator in zip(integers, denominators):
        if a and b:
            total += tp.kronecker(a, bits) * tp.kronecker(b, bits) * (common // denominator)
    return total == tp.kronecker(target_values, bits) * (common // target_denominator)


def _degree_of(values) -> int:
    return -1 if _is_zero(values) else len(values) - 1


def check_xgcd(f, g, h, s, t) -> bool:
    """(h, s, t) is v0.7's Polynomial.xgcd of nonzero f and g: h monic, s f + t g = h, h divides f and g, and the
    cofactor degrees that make (s, t) unique: with u = f/h and v = g/h, s = 0 when v is constant and deg s < deg v
    otherwise; t = 0 when u is constant and v is not, and deg t < deg u when neither is constant."""
    if not (monic_tuple(h) and canonical_tuple(s) and canonical_tuple(t)):
        return False
    if not _linear_identity(((s, f), (t, g)), h):
        return False
    divisor = integer_form(h)
    quotient_f = tp.divmod_exact(integer_form(f), divisor)
    quotient_g = tp.divmod_exact(integer_form(g), divisor)
    if quotient_f is None or quotient_g is None or quotient_f[1] or quotient_g[1]:
        return False
    degree_u, degree_v = len(quotient_f[0]) - 1, len(quotient_g[0]) - 1
    degree_s, degree_t = _degree_of(s), _degree_of(t)
    if degree_v == 0:
        return degree_s == -1
    if degree_s >= degree_v:
        return False
    return degree_t == -1 if degree_u == 0 else degree_t < degree_u


def check_reduced_fraction(numerator, denominator, reduced_numerator, reduced_denominator) -> bool:
    """reduced_numerator / reduced_denominator is numerator / denominator in lowest terms with a monic
    denominator: the cross products agree and the reduced parts are coprime (denominator nonzero)."""
    if not (canonical_tuple(reduced_numerator) and monic_tuple(reduced_denominator)) or _is_zero(denominator):
        return False
    if not _cross_identity(reduced_numerator, denominator, numerator, reduced_denominator):
        return False
    if _is_zero(reduced_numerator):
        return len(reduced_denominator) == 1
    return coprime(integer_form(reduced_numerator), integer_form(reduced_denominator))


def _cross_identity(a, b, c, d) -> bool:
    """a * b == c * d for rational coefficient tuples: A B dc dd == C D da db after clearing denominators."""
    (A, da), (B, db), (C, dc), (D, dd) = (tp.from_rationals(values) for values in (a, b, c, d))
    left_scale, right_scale = dc * dd, da * db
    bits = tp.kronecker_bits(
        tp.max_bits(A) + tp.max_bits(B) + min(len(A), len(B)).bit_length() + left_scale.bit_length(),
        tp.max_bits(C) + tp.max_bits(D) + min(len(C), len(D)).bit_length() + right_scale.bit_length())
    tp.charge(len(A) + len(B) + len(C) + len(D))
    kronecker = tp.kronecker
    return (kronecker(A, bits) * kronecker(B, bits) * left_scale
            == kronecker(C, bits) * kronecker(D, bits) * right_scale)


def check_square_free(f, factors) -> bool:
    """v0.7's check of a square-free decomposition of the nonconstant rational polynomial f, given as (monic
    factor, multiplicity) pairs: nonconstant monic factors, strictly increasing positive multiplicities, the product
    of the distinct factors square-free (so each factor is square-free and any two are coprime) and
    f = lc(f) * prod q_i**e_i. Both are decided on the primitive integer forms Q_i: square-freeness modulo a prime
    dividing no leading coefficient, and F * prod lc(Q_i)**e_i = lc(F) * prod Q_i**e_i by one Kronecker
    evaluation (the coefficients of a product are bounded by the product of the 1-norms)."""
    previous = 0
    forms = []
    for values, multiplicity in factors:
        if (type(multiplicity) is not int or multiplicity <= previous or not monic_tuple(values)
                or len(values) < 2):
            return False
        previous = multiplicity
        forms.append((integer_form(values), multiplicity))
    if not forms:
        return False
    target = integer_form(f)
    scale = 1
    product_bits = target[-1].bit_length()
    distinct = (1,)
    for form, multiplicity in forms:
        scale *= form[-1] ** multiplicity
        product_bits += multiplicity * tp.norm1_bits(form)
        distinct = tp.mul(distinct, form)
    if sum((len(form) - 1) * multiplicity for form, multiplicity in forms) != len(target) - 1:
        return False
    if not coprime(distinct, tp.derivative(distinct)):
        return False
    bits = tp.kronecker_bits(tp.max_bits(target) + scale.bit_length(), product_bits)
    tp.charge(sum(len(form) * multiplicity for form, multiplicity in forms))
    product = target[-1]
    for form, multiplicity in forms:
        product *= tp.kronecker(form, bits) ** multiplicity
    return tp.kronecker(target, bits) * scale == product


def check_rational_resultant(f, g, integer_f, integer_g, integer_value, value) -> bool:
    """value = Res(F, G) (lc f / lc F)**deg g (lc g / lc G)**deg f with F and G the primitive integer forms of f and
    g, as v0.7's polynomial_exact.resultant scales the integer resultant."""
    if integer_form(f) != integer_f or integer_form(g) != integer_g or not canonical(value):
        return False
    expected = (Rational(integer_value) * (f[-1] / integer_f[-1]) ** (len(g) - 1)
                * (g[-1] / integer_g[-1]) ** (len(f) - 1))
    return expected == value


def check_discriminant(f, resultant_value, value) -> bool:
    """v0.7's criterion: value * lc(f) = (-1)**(n(n-1)/2) Res(f, f') for f of degree n >= 1."""
    if not canonical(value) or not canonical(resultant_value):
        return False
    degree = len(f) - 1
    sign = -1 if degree * (degree - 1) // 2 % 2 else 1
    return value * f[-1] == resultant_value * sign


def check_square_free_part(f: tuple[int, ...], part: tuple[int, ...], common: tuple[int, ...],
                           rest: tuple[int, ...]) -> bool:
    """part = f / gcd(f, f'), the square-free part of the nonconstant integer polynomial f: common is that gcd,
    with f = common * part and primitive(f') = common * rest, part and rest coprime."""
    return check_gcd(f, tp.primitive(tp.derivative(f)), common, part, rest)


# ---------------------------------------------------------------------------------------------------------------
# Resultants

def _hadamard_bits(f: tuple[int, ...], g: tuple[int, ...]) -> int:
    """Bits of a bound on |Res(f, g)|: ||f||_2**deg g * ||g||_2**deg f (Hadamard on the Sylvester matrix)."""
    norm_f = sum(value * value for value in f).bit_length() // 2 + 1
    norm_g = sum(value * value for value in g).bit_length() // 2 + 1
    return (len(g) - 1) * norm_f + (len(f) - 1) * norm_g


def _resultant_mod(f: tuple[int, ...], g: tuple[int, ...], m: int) -> int:
    """Res(f, g) mod m by the Euclidean algorithm in Z/m: Res(a, b) = (-1)**(deg a deg b) lc(b)**(deg a - deg r)
    Res(b, r) with r = a mod b, Res(a, c) = c**deg a for a constant c, and 0 when a remainder vanishes. Every leading
    coefficient divided by must be a unit mod m (pow raises ValueError otherwise)."""
    a, b = tp.mod(f, m), tp.mod(g, m)
    if len(a) != len(f) or len(b) != len(g):
        raise ValueError("A leading coefficient vanishes modulo m")
    result = 1
    while True:
        degree_a, degree_b = len(a) - 1, len(b) - 1
        if degree_b == 0:
            return result * pow(b[0], degree_a, m) % m
        if degree_a == 0:
            return result * pow(a[0], degree_b, m) % m
        if degree_a < degree_b:
            a, b = b, a
            if degree_a * degree_b % 2:
                result = -result
            continue
        remainder = tp.divmod_mod(a, b, m)[1]
        if not remainder:
            return 0
        if degree_a * degree_b % 2:
            result = -result
        result = result * pow(b[-1], degree_a - (len(remainder) - 1), m) % m
        a, b = b, remainder


def check_resultant(f: tuple[int, ...], g: tuple[int, ...], value: int) -> bool:
    """value = Res(f, g) for nonzero integer polynomials: equal modulo a prime larger than twice the Hadamard bound,
    which no other integer of absolute value within the bound is."""
    if type(value) is not int:
        return False
    if len(f) == 1 or len(g) == 1:
        return value == (g[0] ** (len(f) - 1) if len(g) == 1 else f[0] ** (len(g) - 1))
    bits = _hadamard_bits(f, g)
    if value.bit_length() > bits:
        return False
    for exponent in _MERSENNE_EXPONENTS:
        if exponent >= bits + 2:
            modulus = (1 << exponent) - 1
            break
    else:
        return False
    tp.charge(len(f) * len(g))
    try:
        return _resultant_mod(f, g, modulus) == value % modulus
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------------------------
# Product identities

def _integer(coefficients) -> tuple[tuple[int, ...], int]:
    return tp.from_rationals(coefficients)


def _degree(coefficients) -> int:
    return -1 if _is_zero(coefficients) else len(coefficients) - 1


def check_division(f, g, quotient, remainder) -> bool:
    """f = quotient * g + remainder over Q with deg remainder < deg g (g nonzero), by one Kronecker evaluation of
    F dq dg dr = Q G df dr + R df dq dg, the integer form with every denominator cleared."""
    if not (canonical_tuple(quotient) and canonical_tuple(remainder)) or _is_zero(g):
        return False
    if _degree(remainder) >= _degree(g):
        return False
    numerators_f, df = _integer(f)
    numerators_g, dg = _integer(g)
    numerators_q, dq = _integer(quotient)
    numerators_r, dr = _integer(remainder)
    left_scale, product_scale, remainder_scale = dq * dg * dr, df * dr, df * dq * dg
    bits = tp.kronecker_bits(
        tp.max_bits(numerators_f) + left_scale.bit_length(),
        tp.max_bits(numerators_q) + tp.max_bits(numerators_g) + min(len(numerators_q), len(numerators_g)).bit_length()
        + product_scale.bit_length(),
        tp.max_bits(numerators_r) + remainder_scale.bit_length())
    tp.charge(len(numerators_f) + len(numerators_q) + len(numerators_g) + len(numerators_r))
    kronecker = tp.kronecker
    return (kronecker(numerators_f, bits) * left_scale
            == kronecker(numerators_q, bits) * kronecker(numerators_g, bits) * product_scale
            + kronecker(numerators_r, bits) * remainder_scale)
