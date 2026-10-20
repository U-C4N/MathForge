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

from ..limits import ResourceLimitError
from ..model import Rational
from ..trusted import upoly as tp
from ..trusted.caches import BoundedCache


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

# Mersenne primes 2**k - 1 (Lucas-Lehmer), the moduli of the coprimality tests. The resultant check works modulo a
# power of the first, P**j larger than twice the Hadamard bound: Euclid's identities for resultants hold over any
# commutative ring as long as every leading coefficient divided by is a unit, and a unit's inverse modulo P**j
# lifts by Newton's iteration from its inverse modulo P. Agreement modulo m decides an integer bounded by m / 2.
_SMALL_PRIMES = ((1 << 61) - 1, (1 << 31) - 1, (1 << 89) - 1, (1 << 107) - 1, (1 << 127) - 1)
_P = (1 << 61) - 1


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
    distinct = (1,)
    for form, _ in forms:
        distinct = tp.mul(distinct, form)
    if sum((len(form) - 1) * multiplicity for form, multiplicity in forms) != len(target) - 1:
        return False
    if not coprime(distinct, tp.derivative(distinct)):
        return False
    return _monic_product_identity(target, forms)


def _monic_product_identity(target: tuple[int, ...], forms) -> bool:
    """monic(target) = prod (Q / lc Q)**e over (Q, e) in forms, as target * prod lc(Q)**e = lc(target) * prod Q**e
    by one Kronecker evaluation (the coefficients of a product are bounded by the product of the 1-norms)."""
    scale = 1
    product_bits = target[-1].bit_length()
    for form, multiplicity in forms:
        scale *= form[-1] ** multiplicity
        product_bits += multiplicity * tp.norm1_bits(form)
    bits = tp.kronecker_bits(tp.max_bits(target) + scale.bit_length(), product_bits)
    tp.charge(sum(len(form) * multiplicity for form, multiplicity in forms))
    product = target[-1]
    for form, multiplicity in forms:
        product *= tp.kronecker(form, bits) ** multiplicity
    return tp.kronecker(target, bits) * scale == product


def check_factorization(f, entries) -> bool:
    """v0.7's check_factor criterion for a nonconstant rational polynomial f and entries (monic factor,
    multiplicity, irreducibility evidence): canonical monic nonconstant factors in strictly increasing order of
    (degree, coefficients), positive multiplicities, each factor irreducible (check_irreducible on its primitive
    integer form), and f = lc(f) prod q**e."""
    previous = None
    forms = []
    for values, multiplicity, evidence in entries:
        if type(multiplicity) is not int or multiplicity < 1 or not monic_tuple(values) or len(values) < 2:
            return False
        key = (len(values) - 1, tuple((value.numerator, value.denominator) for value in values))
        if previous is not None and key <= previous:
            return False
        previous = key
        form = integer_form(values)
        if not check_irreducible(form, evidence):
            return False
        forms.append((form, multiplicity))
    target = integer_form(f)
    if sum((len(form) - 1) * multiplicity for form, multiplicity in forms) != len(target) - 1:
        return False
    return _monic_product_identity(target, forms)


def check_rational_resultant(f, g, value) -> bool:
    """value = Res(f, g) for nonzero rational polynomials: with F and G their primitive integer forms,
    Res(f, g) = Res(F, G) (lc f / lc F)**deg g (lc g / lc G)**deg f, so value divided by those scales must be the
    integer Res(F, G), which check_resultant decides."""
    if not canonical(value):
        return False
    integer_f, integer_g = integer_form(f), integer_form(g)
    degree_f, degree_g = len(f) - 1, len(g) - 1
    lead_f, lead_g = f[-1], g[-1]
    if (max(lead_f.numerator.bit_length(), (lead_f.denominator * integer_f[-1]).bit_length()) * degree_g
            + max(lead_g.numerator.bit_length(), (lead_g.denominator * integer_g[-1]).bit_length()) * degree_f
            > 4 * 65_536):
        return False
    # value = Res(F, G) * n / d, so Res(F, G) = value d / n must be an integer.
    n = lead_f.numerator ** degree_g * lead_g.numerator ** degree_f
    d = (lead_f.denominator * integer_f[-1]) ** degree_g * (lead_g.denominator * integer_g[-1]) ** degree_f
    top, bottom = value.numerator * d, value.denominator * n
    if top % bottom:
        return False
    return check_resultant(integer_f, integer_g, top // bottom)


def check_discriminant(f, value) -> bool:
    """v0.7's criterion, value * lc(f) = (-1)**(n(n-1)/2) Res(f, f') for f of degree n >= 1, decided on the
    primitive integer form: with f = c F, Res(f, f') = c**(2n-1) Res(F, F'), so value * lc(f) / (sign c**(2n-1))
    must be the integer Res(F, F'), which check_resultant decides."""
    if not canonical(value):
        return False
    integer = integer_form(f)
    degree = len(f) - 1
    lead = f[-1]
    c_numerator, c_denominator = lead.numerator, lead.denominator * integer[-1]
    if max(c_numerator.bit_length(), c_denominator.bit_length()) * (2 * degree - 1) > 4 * 65_536:
        return False
    # value * lc(f) = sign * c**(2n-1) * R  =>  R = sign * value * lc(f) * c_den**(2n-1) / c_num**(2n-1)
    top = value.numerator * lead.numerator * c_denominator ** (2 * degree - 1)
    bottom = value.denominator * lead.denominator * c_numerator ** (2 * degree - 1)
    if top % bottom:
        return False
    integer_value = top // bottom
    if degree * (degree - 1) // 2 % 2:
        integer_value = -integer_value
    return check_resultant(integer, tp.derivative(integer), integer_value)


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


def _inverse(value: int, m: int) -> int:
    """The inverse of value modulo m = P**j: from the inverse modulo P (ValueError when P divides value), by
    Newton's iteration x <- x (2 - value x), which doubles the power of P that a value x - 1/value is divisible by."""
    inverse = pow(value % _P, -1, _P)
    modulus = _P
    while modulus < m:
        modulus = min(modulus * modulus, m)
        inverse = inverse * (2 - value * inverse) % modulus
    return inverse


def _resultant_mod(f: tuple[int, ...], g: tuple[int, ...], m: int) -> int:
    """Res(f, g) mod m by the Euclidean algorithm in Z/m: Res(a, b) = (-1)**(deg a deg b) lc(b)**(deg a - deg r)
    Res(b, r) with r = a mod b, Res(a, c) = c**deg a for a constant c, and 0 when a remainder vanishes. Every leading
    coefficient divided by must be a unit mod m = P**j (ValueError otherwise)."""
    a, b = [value % m for value in f], [value % m for value in g]
    if not a[-1] or not b[-1]:
        raise ValueError("A leading coefficient vanishes modulo m")
    tp.charge(len(f) * len(g))
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
        # a mod b in (Z/m)[x]: lc(b) must be a unit; entries below the current top are reduced once, at the end.
        n = degree_b
        inverse = _inverse(b[-1], m)
        remainder = a[:]
        lower = b[:-1]
        for end in range(degree_a, n - 1, -1):
            top = remainder[end] % m
            if top:
                digit = top * inverse % m
                shift = end - n
                remainder[shift:end] = [value - digit * other for value, other in zip(remainder[shift:end], lower)]
        remainder = [value % m for value in remainder[:n]]
        while remainder and not remainder[-1]:
            remainder.pop()
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
    modulus = _P ** ((bits + 2) // 60 + 1)  # P**j > 2**(60 j) >= 2**(bits + 2)
    tp.charge(len(f) * len(g))
    try:
        return _resultant_mod(f, g, modulus) == value % modulus
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------------------------
# Sturm chains of the checker: subresultant remainders, signed as a Sturm sequence

_STURM = BoundedCache("certify.sturm", 512)


def _limit() -> int:
    budget = tp._active()
    return 65_536 if budget is None else budget.limits.max_integer_bits


def _too_long() -> ResourceLimitError:
    return ResourceLimitError("Computation limit exceeded: integer bits")


def euclid_gcd(f: tuple[int, ...], g: tuple[int, ...]) -> tuple[int, ...]:
    """The primitive gcd (positive leading coefficient) of nonzero integer polynomials, by the primitive
    pseudo-remainder sequence (the checker's own; the producer uses a heuristic gcd). A divisor whose leading
    coefficient passes max_integer_bits is refused before it divides (ResourceLimitError), as the producer's Euclid
    refuses it."""
    a, b = tp.primitive(f), tp.primitive(g)
    if len(a) < len(b):
        a, b = b, a
    limit = _limit()
    while b:
        if b[-1].bit_length() > limit:
            raise _too_long()
        remainder = tp.pseudo_remainder(a, b)
        tp.charge_gcd(tp.max_bits(remainder), len(remainder))  # Its content: up to one gcd per coefficient.
        a, b = b, tp.primitive(remainder)
    return tp.primitive(a)


def squarefree_part(p: tuple[int, ...]) -> tuple[int, ...]:
    """The primitive square-free part (positive leading coefficient) of a nonzero integer polynomial: p / gcd(p, p')
    by Euclid; (1,) for a constant."""
    if len(p) <= 1:
        return (1,)
    f = tp.primitive(p)
    return tp.primitive(tp.exact_quotient(f, euclid_gcd(f, tp.derivative(f))))


def sturm_chain(p: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """A Sturm sequence of the square-free integer polynomial p (degree >= 1): the subresultant PRS of p and p'
    (Brown and Collins' scalars, as SymPy's dup_inner_subresultants computes them), each member multiplied by the
    sign that makes it a positive multiple of minus the remainder of the two before it. With R_(i+1) =
    s_i lc(R_i)**k_i rem(R_(i-1), R_i), the sign is sigma_(i+1) = -sigma_(i-1) sign(s_i) sign(lc R_i)**k_i.

    A member whose leading or constant coefficient passes max_integer_bits is refused (ResourceLimitError) before it
    divides, as the producer's chain refuses its own members."""
    return _STURM.get(p, lambda: _sturm_chain(p))


def _sturm_chain(p):
    if len(p) <= 1:
        return (p,)
    f, g = p, tp.derivative(p)
    members = [f, g]
    signs = [1, 1]
    degree_f, degree_g = len(f) - 1, len(g) - 1
    drop = degree_f - degree_g
    scale = -1 if (drop + 1) % 2 else 1  # s_1 = (-1)**(d+1)
    h = tp.pseudo_remainder(f, g)
    if scale < 0:
        h = tp.neg(h)
    lead = g[-1]
    c = lead ** drop
    c = -c
    scale_sign = scale
    exponent = drop + 1
    limit = _limit()
    while h:
        if max(h[-1].bit_length(), h[0].bit_length()) > limit:
            raise _too_long()
        sign_lead = 1 if lead > 0 else -1
        signs.append(-signs[-2] * scale_sign * (sign_lead if exponent % 2 else 1))
        members.append(h)
        if len(h) == 1:
            break  # A nonzero constant ends the chain: every remainder by it vanishes.
        k = len(h) - 1
        f, g, drop = g, h, len(g) - 1 - k
        tp.charge_power(c.bit_length(), drop)
        b = -lead * c ** drop
        remainder = tp.pseudo_remainder(f, g)
        exponent = len(f) - len(g) + 1
        quotient = tuple([value // b for value in remainder])
        if any(value % b for value in remainder):
            raise ArithmeticError("Nonexact subresultant division")
        h = quotient
        scale_sign = 1 if b > 0 else -1  # s_i = 1 / b
        lead = g[-1]
        if drop > 1:
            tp.charge_power(max(c.bit_length(), lead.bit_length()), drop)
            q = c ** (drop - 1)
            c, rest = divmod((-lead) ** drop, q)
            if rest:
                raise ArithmeticError("Nonexact subresultant scalar")
        else:
            c = -lead
    return tuple(member if sign > 0 else tp.neg(member) for member, sign in zip(members, signs))


def _sign_at(p: tuple[int, ...], numerator: int, denominator: int) -> int:
    return tp.sign_at(p, numerator, denominator)


def _variations(chain, point) -> int:
    """Sign variations of the chain at a rational point (num, den), or at -infinity (None, False) and +infinity
    (None, True); zeros are skipped."""
    signs = []
    if point[0] is None:
        positive = point[1]
        for member in chain:
            sign = 1 if member[-1] > 0 else -1
            if not positive and (len(member) - 1) % 2:
                sign = -sign
            signs.append(sign)
    else:
        numerator, denominator = point
        for member in chain:
            sign = _sign_at(member, numerator, denominator)
            if sign:
                signs.append(sign)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def count_roots(p: tuple[int, ...], lo, hi) -> int:
    """The distinct real roots of the square-free integer polynomial p in (lo, hi], lo and hi Rationals or None for
    an infinite end (Sturm's theorem on the checker's chain)."""
    if len(p) <= 1:
        return 0
    chain = sturm_chain(p)
    low = (None, False) if lo is None else (lo.numerator, lo.denominator)
    high = (None, True) if hi is None else (hi.numerator, hi.denominator)
    tp.charge(sum(len(member) for member in chain))
    return _variations(chain, low) - _variations(chain, high)


def isolate(p: tuple[int, ...]):
    """Disjoint half-open intervals (lo, hi] with lo < hi, increasing, each holding exactly one real root of the
    square-free integer polynomial p, by plain bisection from the Cauchy bound. A root met at a midpoint lies in the
    interval to its left, (lo, middle], so no interval is a point."""
    if len(p) <= 1:
        return ()
    lead = abs(p[-1])
    bound = Rational(1 + max(abs(value) for value in p[:-1]) // lead + 1)
    pending = [(-bound, bound)]
    result = []
    while pending:
        lo, hi = pending.pop()
        count = count_roots(p, lo, hi)
        if count == 0:
            continue
        if count == 1:
            result.append((lo, hi))
            continue
        middle = (lo + hi) / 2
        pending.append((middle, hi))
        pending.append((lo, middle))
    result.sort(key=lambda interval: interval[0])
    return tuple(result)


# ---------------------------------------------------------------------------------------------------------------
# Exact real scalars: the checker's reading of what algebraic.as_endpoint accepts, and an exact comparison

def _square_root(value):
    """The rational square root of a nonnegative Rational, or None."""
    from math import isqrt
    if value.numerator < 0:
        return None
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return Rational(numerator, denominator)
    return None


def field_coordinates(expression):
    """(u, v, d) with expression = u + v sqrt(d), for a tree of Rational, Sqrt, Add, Mul and Pow nodes that lies in
    one quadratic field (d the first non-square radicand met, None when there is none); None otherwise. This is the
    scope of algebraic._quadratic_coordinates, computed here on its own."""
    from ..model import Add, Mul, Pow, Sqrt
    radicand = None
    pending = [expression]
    while pending:
        node = pending.pop()
        kind = type(node)
        if kind is Rational:
            continue
        if kind is Sqrt:
            if type(node.radicand) is not Rational:
                return None
            if radicand is None and _square_root(node.radicand) is None:
                radicand = node.radicand
            continue
        if kind is Add:
            pending.extend(node.terms)
        elif kind is Mul:
            pending.extend(node.factors)
        elif kind is Pow:
            if type(node.exponent) is not int or node.exponent < 0:
                return None
            pending.append(node.base)
        else:
            return None
    zero = Rational(0)
    d = radicand if radicand is not None else zero

    def product(a, b):
        return a[0] * b[0] + a[1] * b[1] * d, a[0] * b[1] + a[1] * b[0]

    def visit(node):
        kind = type(node)
        if kind is Rational:
            return node, zero
        if kind is Sqrt:
            root = _square_root(node.radicand)
            if root is not None:
                return root, zero
            ratio = _square_root(node.radicand / d) if d.numerator else None
            if ratio is None:
                raise ValueError("two quadratic fields")
            return zero, ratio
        if kind is Add:
            total = (zero, zero)
            for child in node.terms:
                pair = visit(child)
                total = total[0] + pair[0], total[1] + pair[1]
            return total
        if kind is Mul:
            total = (Rational(1), zero)
            for child in node.factors:
                total = product(total, visit(child))
            return total
        base, exponent, result = visit(node.base), node.exponent, (Rational(1), zero)
        while exponent:
            if exponent & 1:
                result = product(result, base)
            exponent >>= 1
            if exponent:
                base = product(base, base)
        return result

    try:
        u, v = visit(expression)
    except (ValueError, RecursionError):
        return None
    return u, v, (radicand if v.numerator else None)


def _is_class(value, name: str) -> bool:
    """Whether value is an instance of mathforge.algebraic's class of that name, recognized by its module and
    qualified name (this package does not import the record classes)."""
    kind = type(value)
    return kind.__module__ == "mathforge.algebraic" and kind.__qualname__ == name


def endpoint_of(value):
    """("rational", Rational) or ("algebraic", (primitive square-free integer polynomial, real index)) for what
    algebraic.as_endpoint accepts (an int, a Rational, a RealAlgebraicRoot, a closed quadratic-field expression);
    None otherwise."""
    if type(value) is int:
        return "rational", Rational(value)
    if type(value) is Rational:
        return "rational", value
    if _is_class(value, "RealAlgebraicRoot"):
        coefficients, index = value.integer_coefficients, value.real_index
        if (type(coefficients) is not tuple or len(coefficients) < 2 or any(type(c) is not int for c in coefficients)
                or type(index) is not int or index < 0):
            return None
        part = squarefree_part(coefficients)
        if len(part) < 2 or index >= count_roots(part, None, None):
            return None
        return "algebraic", (part, index)
    found = field_coordinates(value)
    if found is None:
        return None
    u, v, d = found
    if not v.numerator:
        return "rational", u
    # u + v sqrt(d) is a root of t**2 - 2 u t + (u**2 - v**2 d): the larger one when v > 0.
    return "algebraic", (tp.primitive(tp.from_rationals((u * u - v * v * d, -2 * u, Rational(1)))[0]),
                         0 if v.numerator < 0 else 1)


def _isolator(p, index):
    """The index-th of the checker's isolating intervals of the square-free p."""
    intervals = isolate(p)
    return intervals[index] if index < len(intervals) else None


def _bisect(p, interval):
    lo, hi = interval
    middle = (lo + hi) / 2
    return (lo, middle) if count_roots(p, lo, middle) else (middle, hi)


def compare(left, right) -> int | None:
    """-1, 0 or 1 for two endpoint_of values, by the checker's own isolating intervals (lo, hi]: refined until they
    separate; two algebraic numbers are equal exactly when the gcd of their polynomials has a root in both
    intervals. None for an invalid descriptor."""
    if left is None or right is None:
        return None
    if left[0] == "rational" and right[0] == "rational":
        return (left[1] > right[1]) - (left[1] < right[1])
    if left[0] == "rational":
        result = compare(right, left)
        return None if result is None else -result
    p, index = left[1]
    a = _isolator(p, index)
    if a is None:
        return None
    if right[0] == "rational":
        r = right[1]
        if tp.sign_at(p, r.numerator, r.denominator) == 0 and a[0] < r <= a[1]:
            return 0
        while a[0] < r <= a[1]:
            a = _bisect(p, a)
        return -1 if a[1] < r else 1
    q, other = right[1]
    b = _isolator(q, other)
    if b is None:
        return None
    common = euclid_gcd(p, q)
    if len(common) > 1:
        low, high = max(a[0], b[0]), min(a[1], b[1])
        if low < high and count_roots(squarefree_part(common), low, high) >= 1:
            return 0
    while not (a[1] <= b[0] or b[1] <= a[0]):
        a, b = _bisect(p, a), _bisect(q, b)
    return -1 if a[1] <= b[0] else 1


def _interval_product(low, high, a, b):
    """[low, high] * [a, b] for Rational ends, the four products charged by the size of their integers (each runs
    two gcds)."""
    tp.charge_gcd(max(low.numerator.bit_length(), low.denominator.bit_length(), high.numerator.bit_length(),
                      high.denominator.bit_length(), a.numerator.bit_length(), a.denominator.bit_length(),
                      b.numerator.bit_length(), b.denominator.bit_length()), 8)
    products = (low * a, low * b, high * a, high * b)
    return min(products), max(products)


def _rational_enclosure(expression, bits: int):
    """Rationals (lo, hi) enclosing a tree of Rational, Sqrt, Add, Mul and Pow nodes: exact rational arithmetic,
    with sqrt(p/q) between s / (q 2**bits) and (s + 1) / (q 2**bits) for s = isqrt(p q 4**bits)."""
    from math import isqrt
    from ..model import Add, Mul, Pow, Sqrt
    kind = type(expression)
    if kind is Rational:
        return expression, expression
    if kind is Sqrt:
        radicand = expression.radicand
        if type(radicand) is not Rational or radicand.numerator < 0:
            raise ValueError("radicand")
        root = isqrt(radicand.numerator * radicand.denominator << (2 * bits))
        scale = radicand.denominator << bits
        return Rational(root, scale), Rational(root + 1, scale)
    if kind is Add:
        low = high = Rational(0)
        for term in expression.terms:
            a, b = _rational_enclosure(term, bits)
            low, high = low + a, high + b
        return low, high
    if kind is Mul:
        low = high = Rational(1)
        for factor in expression.factors:
            a, b = _rational_enclosure(factor, bits)
            low, high = _interval_product(low, high, a, b)
        return low, high
    if kind is Pow:
        exponent = expression.exponent
        if type(exponent) is not int or exponent < 0:
            raise ValueError("exponent")
        a, b = _rational_enclosure(expression.base, bits)
        low = high = Rational(1)
        for _ in range(exponent):
            low, high = _interval_product(low, high, a, b)
        return low, high
    raise ValueError(kind.__name__)


def check_sign(left, right, sign: int, bits: int) -> bool:
    """sign = sign(left - right) for closed real expressions of Rational, Sqrt, Add, Mul and Pow nodes, decided by
    rational enclosures at the given precision that do not overlap: the claim is checked, never trusted."""
    if sign not in (-1, 1) or type(bits) is not int or not 0 < bits <= 1 << 16:
        return False
    try:
        a, b = _rational_enclosure(left, bits)
        c, d = _rational_enclosure(right, bits)
    except (ValueError, RecursionError):
        return False
    return b < c if sign < 0 else d < a


# ---------------------------------------------------------------------------------------------------------------
# Real root records: v0.7's root_isolation.verify_root_records criterion

def _root_data(root):
    """(kind, payload) of a record's root: ("rational", Rational) or ("algebraic", (polynomial, index))."""
    return endpoint_of(root)


def check_root_records(p: tuple[int, ...], records) -> bool:
    """v0.7's verify_root_records criterion for the nonzero integer polynomial p: as many records as distinct real
    roots, their roots strictly increasing, each a root of p of the recorded multiplicity.

    The checker isolates the roots of p's square-free part S with its own Sturm chain; record j must hold the j-th
    root: a Rational inside the j-th interval with S = 0 there, or the index-th root of a polynomial c dividing S that
    lies in the j-th interval (Sturm counts on c). The multiplicity of the j-th root is 1 plus the number of the
    iterated gcds G_1 = gcd(p, p'), G_(k+1) = gcd(G_k, G_k') that vanish there."""
    if type(records) is not tuple or not p:
        return False
    part = squarefree_part(p)
    intervals = isolate(part)
    if len(intervals) != len(records):
        return False
    gcds = []
    current = tp.primitive(p)
    while len(current) > 1:
        current = euclid_gcd(current, tp.derivative(current))
        if len(current) > 1:
            gcds.append(squarefree_part(current))
    for record, (lo, hi) in zip(records, intervals):
        multiplicity = getattr(record, "multiplicity", None)
        if not _is_class(record, "RootRecord") or type(multiplicity) is not int or multiplicity < 1:
            return False
        data = _root_data(record.root)
        if data is None:
            return False
        kind, payload = data
        if kind == "rational":
            r = payload
            if not lo < r <= hi:  # isolate's intervals are half-open, never points.
                return False
            if _sign_at(part, r.numerator, r.denominator) != 0:
                return False
            vanishing = sum(1 for g in gcds if _sign_at(g, r.numerator, r.denominator) == 0)
        else:
            c, index = payload
            c = tp.primitive(c)
            division = tp.divmod_exact(part, c)
            if division is None or division[1] or len(c) < 2:
                return False
            if not (count_roots(c, None, lo) <= index < count_roots(c, None, hi)):
                return False
            vanishing = 0
            for g in gcds:
                if count_roots(g, lo, hi) >= 1:
                    vanishing += 1
        if vanishing + 1 != multiplicity:
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# Irreducibility over Q: polynomial_exact._check_irreducible and _rabin on integer tuples

_TRIAL_BATCH = 256  # Trial divisions charged at a time.


def _prime(n) -> bool:
    """v0.7's primality test of the evidence's prime: trial division by 3, 5, 7, ..., charged one unit of work per
    division as v0.7 ticked them (in batches), so that a large prime stops at max_work as it did there."""
    if type(n) is not int or n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    divisor = 3
    while divisor * divisor <= n:
        tp.charge(_TRIAL_BATCH)
        for divisor in range(divisor, divisor + 2 * _TRIAL_BATCH, 2):
            if divisor * divisor > n:
                return True
            if n % divisor == 0:
                return False
        divisor += 2
    return True


def rabin(f: tuple[int, ...], p: int) -> bool:
    """Rabin's criterion: the monic f (reduced mod p) is irreducible over GF(p) when x**(p**n) = x mod f and
    gcd(f, x**(p**(n/q)) - x) = 1 for every prime q dividing n = deg f (independent of Berlekamp)."""
    degree = len(f) - 1
    if degree <= 0 or f[-1] != 1:
        return False
    prime_divisors, remaining, divisor = [], degree, 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            prime_divisors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor += 1
    if remaining > 1:
        prime_divisors.append(remaining)
    checkpoints = {degree // divisor for divisor in prime_divisors}
    x = tp.divmod_mod((0, 1), f, p)[1]
    power = x
    for i in range(1, degree + 1):
        power = tp.pow_mod(power, p, f, p)
        if i in checkpoints and tp.gcd_mod(f, tp.mod(tp.sub(power, x), p), p) != (1,):
            return False
    return power == x


def _coefficient_bound(f: tuple[int, ...]) -> int:
    from math import isqrt
    total = sum(value * value for value in f)
    norm = isqrt(total)
    if norm * norm < total:
        norm += 1
    return abs(f[-1]) * (1 << (len(f) - 1)) * norm


def check_irreducible(p: tuple[int, ...], evidence) -> bool:
    """The integer polynomial p (any nonzero multiple of the factor) is irreducible over Q, given the evidence that
    polynomial_exact._irreducibility_evidence produces: ("linear",) for degree one, or ("modular", prime,
    exponent, lifted) with the monic factors of p modulo prime**exponent. As v0.7's _check_irreducible: the prime
    divides no leading coefficient and the modulus exceeds twice the coefficient bound; the factors are monic,
    irreducible mod the prime (Rabin), pairwise coprime there, and multiply to p; and no product of a subset of at
    most half the degree, read symmetrically and made primitive, divides p over Q."""
    from itertools import combinations
    if type(p) is not tuple or len(p) < 2 or any(type(value) is not int for value in p) or not p[-1]:
        return False
    degree = len(p) - 1
    if degree == 1:
        return evidence == ("linear",)
    if type(evidence) is not tuple or len(evidence) != 4 or evidence[0] != "modular":
        return False
    _, prime, exponent, factors = evidence
    if (type(prime) is not int or type(exponent) is not int or exponent <= 0
            or type(factors) is not tuple or not factors or not _prime(prime)):
        return False
    limit = _limit()
    if exponent > limit:
        return False
    # v0.7 built prime**exponent one checked product at a time, refusing it past max_integer_bits; a power surely
    # past the bound is refused before it is formed.
    if (prime.bit_length() - 1) * exponent > limit:
        raise _too_long()
    modulus = prime ** exponent
    if modulus.bit_length() > limit:
        raise _too_long()
    source = tp.primitive(p)
    if source[-1] % prime == 0 or modulus <= 2 * _coefficient_bound(source):
        return False
    for factor in factors:
        if (type(factor) is not tuple or len(factor) <= 1 or factor[-1] != 1
                or any(type(c) is not int or c < 0 or c >= modulus for c in factor)):
            return False
    if sum(len(factor) - 1 for factor in factors) != degree:
        return False
    modular = [tp.mod(factor, prime) for factor in factors]
    for i, factor in enumerate(modular):
        if not rabin(factor, prime):
            return False
        if any(tp.gcd_mod(factor, prior, prime) != (1,) for prior in modular[:i]):
            return False
    product: tuple[int, ...] = (source[-1] % modulus,)
    for factor in factors:
        product = tp.mul_mod(product, factor, modulus)
    if product != tp.mod(source, modulus):
        return False
    half = degree // 2
    for size in range(1, len(factors) + 1):
        for chosen in combinations(range(len(factors)), size):
            subset_degree = sum(len(factors[index]) - 1 for index in chosen)
            if subset_degree > half:
                continue
            tp.charge(degree)
            coefficients: tuple[int, ...] = (source[-1] % modulus,)
            for index in chosen:
                coefficients = tp.mul_mod(coefficients, factors[index], modulus)
            signed = tp.trim(c if 2 * c <= modulus else c - modulus for c in coefficients)
            if len(signed) - 1 != subset_degree:
                return False
            division = tp.divmod_exact(source, tp.primitive(signed))
            if division is not None and not division[1]:
                return False
    return True


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
