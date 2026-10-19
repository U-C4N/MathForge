"""Dense univariate polynomials over Z, and the exact conversions around them.

A polynomial is a tuple of Python ints, lowest degree first, with no trailing zeros; ``()`` is the zero polynomial.
Producers (mathforge.kernels) and independent checkers (mathforge.certify) share these functions, so they are kept
plain: schoolbook arithmetic, Horner evaluation, Euclid modulo an integer, and the Kronecker evaluation that turns a
polynomial identity into one integer comparison. tests/v08/algebra/test_upoly.py checks them against
``mathforge.polynomial.Polynomial`` and against each other.

Work is charged to the active budget, when there is one, in batches: one ``tick`` per call, proportional to the
number of coefficient operations, so that a runaway computation stops at ``max_work`` without a charge per
multiplication. Results whose integers outgrow ``max_integer_bits`` are refused by the callers that can create them
(``check_bits``).

The module also builds the two exact objects that the lean value path returns from coefficient lists without
re-validating them: Rationals in lowest terms (``rationals``) and the expression tree that
``Polynomial.to_expression`` builds (``expression``).
"""

from __future__ import annotations

from math import gcd

from ..errors import InvalidInput
from ..limits import ResourceLimitError, current_budget
from ..model import Add, Mul, Pow, Rational, trusted_rational

__all__ = [
    "trim", "add", "sub", "mul", "scale", "neg", "derivative", "pseudo_remainder", "divmod_exact", "exact_quotient",
    "content", "primitive", "sign_at", "value_at", "mod", "mul_mod", "divmod_mod", "gcd_mod", "pow_mod",
    "monic_mod", "max_bits", "norm1_bits", "kronecker_bits", "kronecker", "check_bits", "charge",
    "from_rationals", "rationals", "monic_rationals", "expression",
]

_new = object.__new__


# ---------------------------------------------------------------------------------------------------------------
# Budget

def charge(n: int) -> None:
    """Charge the active budget, if any, n units of work (one batched tick)."""
    budget = current_budget()
    if budget is not None and n:
        budget.tick(n)


def check_bits(value: int) -> None:
    """Refuse an integer longer than the active budget's max_integer_bits (65,536 without a budget)."""
    budget = current_budget()
    limit = 65_536 if budget is None else budget.limits.max_integer_bits
    if value.bit_length() > limit:
        raise ResourceLimitError("Computation limit exceeded: integer bits")


# ---------------------------------------------------------------------------------------------------------------
# Arithmetic over Z

def trim(values) -> tuple[int, ...]:
    """The tuple of values without trailing zeros."""
    values = list(values)
    while values and not values[-1]:
        values.pop()
    return tuple(values)


def add(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    if len(p) < len(q):
        p, q = q, p
    result = list(p)
    for index, value in enumerate(q):
        result[index] += value
    charge(len(p))
    while result and not result[-1]:
        result.pop()
    return tuple(result)


def neg(p: tuple[int, ...]) -> tuple[int, ...]:
    return tuple([-value for value in p])


def sub(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    if len(p) >= len(q):
        result = list(p)
        for index, value in enumerate(q):
            result[index] -= value
    else:
        result = [-value for value in q]
        for index, value in enumerate(p):
            result[index] += value
    charge(len(result))
    while result and not result[-1]:
        result.pop()
    return tuple(result)


def scale(p: tuple[int, ...], c: int) -> tuple[int, ...]:
    if not c:
        return ()
    charge(len(p))
    return tuple([c * value for value in p])


def mul(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """The product; schoolbook for short factors, Kronecker substitution (one big-integer product) otherwise."""
    if not p or not q:
        return ()
    if len(p) < len(q):
        p, q = q, p
    charge(len(p) * len(q))
    if len(q) <= 12:
        result = [0] * (len(p) + len(q) - 1)
        for j, b in enumerate(q):
            if b:
                for i, a in enumerate(p, j):
                    result[i] += a * b
        return tuple(result)
    bits = max_bits(p) + max_bits(q) + len(q).bit_length() + 1
    return _unpack(kronecker(p, bits) * kronecker(q, bits), bits, len(p) + len(q) - 1)


def derivative(p: tuple[int, ...]) -> tuple[int, ...]:
    charge(len(p))
    return tuple([index * value for index, value in enumerate(p)][1:])


def pseudo_remainder(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """prem(p, q) = lc(q)**(deg p - deg q + 1) * p mod q, computed in Z (q nonzero)."""
    if not q:
        raise InvalidInput("Pseudo-division by the zero polynomial")
    m = len(q)
    if len(p) < m:
        return p
    lead = q[-1]
    exponent = len(p) - m + 1
    remainder = list(p)
    lower = q[:-1]
    charge((len(p) - m + 1) * len(p))
    while len(remainder) >= m:
        top = remainder.pop()  # lead * top - top * lead cancels the leading term
        shift = len(remainder) - m + 1
        if lead != 1:
            remainder = [lead * value for value in remainder]
        for index, value in enumerate(lower, shift):
            remainder[index] -= top * value
        exponent -= 1
        while remainder and not remainder[-1]:
            remainder.pop()
    if remainder and exponent and lead != 1:
        factor = lead ** exponent
        remainder = [factor * value for value in remainder]
    return tuple(remainder)


def divmod_exact(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """(quotient, remainder) of p by q over Z when every quotient digit is an integer, else None (q nonzero)."""
    if not q:
        raise InvalidInput("Division by the zero polynomial")
    m = len(q)
    if len(p) < m:
        return (), p
    lead = q[-1]
    remainder = list(p)
    quotient = [0] * (len(p) - m + 1)
    charge(len(quotient) * m)
    for shift in range(len(p) - m, -1, -1):
        top = remainder[shift + m - 1]
        if top:
            digit, rest = divmod(top, lead)
            if rest:
                return None
            quotient[shift] = digit
            for index in range(m - 1):
                remainder[shift + index] -= digit * q[index]
            remainder[shift + m - 1] = 0
    del remainder[m - 1:]
    while remainder and not remainder[-1]:
        remainder.pop()
    return tuple(quotient), tuple(remainder)


def exact_quotient(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """p / q over Z; ArithmeticError when q does not divide p there."""
    result = divmod_exact(p, q)
    if result is None or result[1]:
        raise ArithmeticError("The divisor does not divide the polynomial over Z")
    return result[0]


def content(p: tuple[int, ...]) -> int:
    """The gcd of the coefficients, nonnegative; 0 for the zero polynomial."""
    return gcd(*p) if p else 0


def primitive(p: tuple[int, ...]) -> tuple[int, ...]:
    """p divided by its content, with a positive leading coefficient; () stays ()."""
    if not p:
        return ()
    divisor = gcd(*p)
    if p[-1] < 0:
        divisor = -divisor
    if divisor == 1:
        return p
    return tuple([value // divisor for value in p])


# ---------------------------------------------------------------------------------------------------------------
# Evaluation

def sign_at(p: tuple[int, ...], num: int, den: int) -> int:
    """The sign of p(num/den) for den > 0: Horner on the homogenized form sum c_i num**i den**(n-i)."""
    if not p:
        return 0
    charge(len(p))
    if den == 1:
        value = 0
        for coefficient in reversed(p):
            value = value * num + coefficient
    else:
        value, power = 0, 1
        for coefficient in reversed(p):
            value = value * num + coefficient * power
            power *= den
    return (value > 0) - (value < 0)


def value_at(p: tuple[int, ...], x: Rational) -> Rational:
    """p(x) as a Rational."""
    num, den = x.numerator, x.denominator
    if not p:
        return trusted_rational(0, 1)
    charge(len(p))
    value, power = 0, 1
    for coefficient in reversed(p):
        value = value * num + coefficient * power
        power *= den
    power //= den  # den ** deg p
    divisor = gcd(value, power)
    return trusted_rational(value // divisor, power // divisor)


# ---------------------------------------------------------------------------------------------------------------
# Modulo an integer m >= 2 (a prime wherever an inverse is needed; pow raises ValueError otherwise)

def mod(p: tuple[int, ...], m: int) -> tuple[int, ...]:
    charge(len(p))
    result = [value % m for value in p]
    while result and not result[-1]:
        result.pop()
    return tuple(result)


def mul_mod(p: tuple[int, ...], q: tuple[int, ...], m: int) -> tuple[int, ...]:
    if not p or not q:
        return ()
    if len(p) < len(q):
        p, q = q, p
    charge(len(p) * len(q))
    if len(q) <= 12:
        result = [0] * (len(p) + len(q) - 1)
        for j, b in enumerate(q):
            if b:
                for i, a in enumerate(p, j):
                    result[i] += a * b
        result = [value % m for value in result]
    else:
        bits = max_bits(p) + max_bits(q) + len(q).bit_length() + 1
        result = [value % m for value in _unpack(kronecker(p, bits) * kronecker(q, bits), bits,
                                                   len(p) + len(q) - 1)]
    while result and not result[-1]:
        result.pop()
    return tuple(result)


def divmod_mod(p: tuple[int, ...], q: tuple[int, ...], m: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(quotient, remainder) modulo m; p and q reduced mod m, q nonzero with an invertible leading coefficient."""
    if not q:
        raise InvalidInput("Zero modular polynomial divisor")
    n = len(q)
    if len(p) < n:
        return (), p
    inverse = pow(q[-1], -1, m)
    remainder = list(p)
    quotient = [0] * (len(p) - n + 1)
    charge(len(quotient) * n)
    lower = q[:-1]
    for shift in range(len(p) - n, -1, -1):
        top = remainder[shift + n - 1]
        if top:
            digit = top * inverse % m
            quotient[shift] = digit
            for index, value in enumerate(lower, shift):
                remainder[index] = (remainder[index] - digit * value) % m
    del remainder[n - 1:]
    while remainder and not remainder[-1]:
        remainder.pop()
    while quotient and not quotient[-1]:
        quotient.pop()
    return tuple(quotient), tuple(remainder)


def monic_mod(p: tuple[int, ...], m: int) -> tuple[int, ...]:
    if not p or p[-1] == 1:
        return p
    inverse = pow(p[-1], -1, m)
    charge(len(p))
    return tuple([value * inverse % m for value in p])


def gcd_mod(p: tuple[int, ...], q: tuple[int, ...], m: int) -> tuple[int, ...]:
    """The monic gcd modulo a prime m of p and q, both reduced mod m; () when both are zero."""
    while q:
        p, q = q, divmod_mod(p, q, m)[1]
    return monic_mod(p, m)


def pow_mod(p: tuple[int, ...], e: int, f: tuple[int, ...], m: int) -> tuple[int, ...]:
    """p**e modulo (f, m) for e >= 0, f nonconstant with an invertible leading coefficient."""
    result: tuple[int, ...] = divmod_mod((1,), f, m)[1]
    base = divmod_mod(mod(p, m), f, m)[1]
    while e:
        if e & 1:
            result = divmod_mod(mul_mod(result, base, m), f, m)[1]
        e >>= 1
        if e:
            base = divmod_mod(mul_mod(base, base, m), f, m)[1]
    return result


# ---------------------------------------------------------------------------------------------------------------
# Kronecker evaluation: a polynomial identity as one comparison of integers

def max_bits(p: tuple[int, ...]) -> int:
    """The bit length of the largest absolute coefficient (0 for the zero polynomial)."""
    if not p:
        return 0
    return max(max(p), -min(p)).bit_length()


def norm1_bits(p: tuple[int, ...]) -> int:
    """The bit length of the sum of the absolute coefficients: a bound for every coefficient of a product."""
    return sum(abs(value) for value in p).bit_length()


def kronecker(p: tuple[int, ...], bits: int) -> int:
    """p(2**bits), by Horner with shifts."""
    value = 0
    for coefficient in reversed(p):
        value = (value << bits) + coefficient
    return value


def kronecker_bits(*bounds: int) -> int:
    """A shift that makes evaluation at 2**shift injective on polynomials whose coefficients are bounded in absolute
    value by the sum of 2**b over the given bit lengths b: every coefficient then stays below 2**(shift - 1) - 1.

    If P(2**s) = 0 for an integer polynomial P whose coefficients are below 2**s - 1 in absolute value, then P = 0:
    the top nonzero term exceeds the sum of all lower ones."""
    return max(bounds, default=0) + len(bounds).bit_length() + 2


def _unpack(value: int, bits: int, length: int) -> tuple[int, ...]:
    """The coefficients of the polynomial P with P(2**bits) = value, all below 2**(bits - 1) in absolute value."""
    result = []
    mask, half, full = (1 << bits) - 1, 1 << (bits - 1), 1 << bits
    for _ in range(length):
        digit = value & mask
        value >>= bits
        if digit >= half:
            digit -= full
            value += 1
        result.append(digit)
    if value:
        raise ArithmeticError("Kronecker unpacking left a carry: the coefficient bound was too small")
    while result and not result[-1]:
        result.pop()
    return tuple(result)


# ---------------------------------------------------------------------------------------------------------------
# Rational coefficients

def from_rationals(coefficients) -> tuple[tuple[int, ...], int]:
    """(P, d) with coefficients == P / d, d > 0 the least common denominator, for canonical Rationals."""
    denominator = 1
    for value in coefficients:
        den = value.denominator
        if den != 1 and denominator % den:
            denominator = denominator // gcd(denominator, den) * den
    if denominator == 1:
        values = [value.numerator for value in coefficients]
    else:
        values = [value.numerator * (denominator // value.denominator) for value in coefficients]
    while values and not values[-1]:
        values.pop()
    return tuple(values), denominator


def rationals(p: tuple[int, ...], denominator: int = 1) -> tuple[Rational, ...]:
    """The Rationals p_i / denominator in lowest terms (denominator > 0); the zero polynomial gives (0,)."""
    if not p:
        return (trusted_rational(0, 1),)
    if denominator == 1:
        return tuple([trusted_rational(value, 1) for value in p])
    result = []
    for value in p:
        divisor = gcd(value, denominator)
        result.append(trusted_rational(value // divisor, denominator // divisor))
    return tuple(result)


def monic_rationals(p: tuple[int, ...]) -> tuple[Rational, ...]:
    """The Rationals of p / lc(p) in lowest terms (p nonzero)."""
    lead = p[-1]
    if lead == 1:
        return tuple([trusted_rational(value, 1) for value in p])
    sign = -1 if lead < 0 else 1
    lead *= sign
    result = []
    for value in p:
        value *= sign
        divisor = gcd(value, lead)
        result.append(trusted_rational(value // divisor, lead // divisor))
    return tuple(result)


# ---------------------------------------------------------------------------------------------------------------
# The expression tree of Polynomial.to_expression

def expression(coefficients, variable):
    """The Expression that ``Polynomial(coefficients, variable).to_expression()`` builds, for canonical Rational
    coefficients without trailing zeros: add(*(mul(c, power(variable, i)) for each nonzero c)).

    That sum puts the constant first, then the monomials with coefficient 1 (the variable, then its powers by
    exponent), then the terms c * monomial by the key of c, (numerator, denominator), then by exponent: the order of
    model._sort_key. One term is returned itself, and no term gives Rational(0). The nodes are made without their
    constructors, which would only repeat what the coefficients already are."""
    constant = None
    units = []
    scaled = []
    for index, coefficient in enumerate(coefficients):
        numerator = coefficient.numerator
        if not numerator:
            continue
        if index > 1:
            monomial = _new(Pow)
            fields = monomial.__dict__
            fields["base"] = variable
            fields["exponent"] = index
        elif index:
            monomial = variable
        else:
            constant = coefficient
            continue
        if numerator == 1 and coefficient.denominator == 1:
            units.append(monomial)
        else:
            term = _new(Mul)
            term.__dict__["factors"] = (coefficient, monomial)
            scaled.append((numerator, coefficient.denominator, index, term))
    if scaled:
        if len(scaled) > 1:
            scaled.sort()  # The first three items differ between any two entries: no node is compared.
        units.extend([item[3] for item in scaled])
    if constant is not None:
        units.insert(0, constant)
    if not units:
        return trusted_rational(0, 1)
    if len(units) == 1:
        return units[0]
    node = _new(Add)
    node.__dict__["terms"] = tuple(units)
    return node
