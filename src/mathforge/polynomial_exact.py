"""Bounded exact factorization and elimination over the rational numbers.

Factor production uses Berlekamp, digit Hensel lifting and Zassenhaus
recombination. Its checker verifies modular factors with Rabin's criterion and
independently enumerates all possible proper factors. Resultants use a
fraction-free subresultant PRS; their checker uses a Sylvester determinant. These are algorithmic
checks, not formal proofs. Every loop shares the caller's computation budget.
"""

from dataclasses import dataclass
from itertools import combinations
from math import gcd, isqrt

from .errors import InvalidInput, UnsupportedOperation
from .limits import computation, current_budget
from .model import Rational, Symbol, as_expression
from .polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial


def _key(poly):
    return poly.degree, tuple((c.numerator, c.denominator) for c in poly.coefficients)


@dataclass(frozen=True)
class PolynomialFactor:
    polynomial: Polynomial
    multiplicity: int

    def __post_init__(self):
        if type(self.polynomial) is not Polynomial or self.polynomial.degree <= 0:
            raise InvalidInput("A factor must be a nonconstant Polynomial")
        if self.polynomial.leading_coefficient != Rational(1):
            raise InvalidInput("Polynomial factors must be monic")
        if type(self.multiplicity) is not int or self.multiplicity <= 0:
            raise InvalidInput("Factor multiplicity must be a positive integer")


@dataclass(frozen=True)
class PolynomialFactorization:
    coefficient: Rational
    factors: tuple[PolynomialFactor, ...]

    def __post_init__(self):
        coefficient = as_expression(self.coefficient)
        factors = tuple(self.factors)
        if type(coefficient) is not Rational:
            raise InvalidInput("Factorization coefficient must be rational")
        if any(type(f) is not PolynomialFactor for f in factors):
            raise InvalidInput("Expected PolynomialFactor entries")
        if coefficient.numerator == 0 and factors:
            raise InvalidInput("A zero factorization cannot contain factors")
        if factors and any(f.polynomial.variable != factors[0].polynomial.variable for f in factors):
            raise InvalidInput("Factors must have the same variable identity")
        if any(_key(a.polynomial) >= _key(b.polynomial) for a, b in zip(factors, factors[1:])):
            raise InvalidInput("Factors must be distinct and canonically sorted")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "factors", factors)


def _tick(n=1):
    current_budget().tick(n)


def _mul_int(a, b):
    budget = current_budget()
    budget.product(a, b)
    result = a * b
    budget.check_int(result)
    return result


def _trim(values):
    values = list(values)
    while len(values) > 1 and values[-1] == 0:
        _tick()
        values.pop()
    return tuple(values) or (0,)


def _mod(values, modulus):
    _tick(len(values))
    return _trim(v % modulus for v in values)


def _add_mod(a, b, p, sign=1):
    values = []
    for i in range(max(len(a), len(b))):
        _tick()
        value = (a[i] if i < len(a) else 0) + sign * (b[i] if i < len(b) else 0)
        current_budget().check_int(value)
        values.append(value % p)
    return _trim(values)


def _mul_mod(a, b, p):
    if a == (0,) or b == (0,):
        return (0,)
    current_budget().nodes(len(a) + len(b) - 1)
    values = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product = _mul_int(x, y)
            value = values[i + j] + product
            current_budget().check_int(value)
            values[i + j] = value % p
    return _trim(values)


def _divmod_mod(a, b, p):
    a, b = _mod(a, p), _mod(b, p)
    if b == (0,):
        raise InvalidInput("Zero modular polynomial divisor")
    if len(a) < len(b):
        return (0,), a
    q, r = [0] * (len(a) - len(b) + 1), list(a)
    inverse = pow(b[-1], -1, p)
    while r != [0] and len(r) >= len(b):
        _tick()
        shift = len(r) - len(b)
        scale = _mul_int(r[-1], inverse) % p
        q[shift] = scale
        for i, value in enumerate(b):
            r[i + shift] = (r[i + shift] - _mul_int(scale, value)) % p
        r = list(_trim(r))
    return _trim(q), tuple(r)


def _gcd_mod(a, b, p):
    while b != (0,):
        _tick()
        a, b = b, _divmod_mod(a, b, p)[1]
    if a == (0,):
        return a
    inverse = pow(a[-1], -1, p)
    return _mod(tuple(_mul_int(c, inverse) for c in a), p)


def _inverse_mod(a, f, p):
    old_r, r, old_s, s = f, a, (0,), (1,)
    while r != (0,):
        q, rem = _divmod_mod(old_r, r, p)
        old_r, r = r, rem
        old_s, s = s, _add_mod(old_s, _mul_mod(q, s, p), p, -1)
    if len(old_r) != 1 or old_r == (0,):
        raise InvalidInput("Modular inverse does not exist")
    inverse = pow(old_r[0], -1, p)
    return _divmod_mod(tuple(_mul_int(c, inverse) % p for c in old_s), f, p)[1]


def _pow_mod(base, exponent, f, p):
    result = (1,)
    base = _divmod_mod(base, f, p)[1]
    while exponent:
        _tick()
        if exponent & 1:
            result = _divmod_mod(_mul_mod(result, base, p), f, p)[1]
        exponent >>= 1
        if exponent:
            base = _divmod_mod(_mul_mod(base, base, p), f, p)[1]
    return result


def _prime(n):
    if type(n) is not int or n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d <= n // d:
        _tick()
        if n % d == 0:
            return False
        d += 2
    return True


def _primitive(poly):
    from .root_isolation import primitive_integer_coefficients
    return primitive_integer_coefficients(poly)


def _good_prime(f):
    p = 2
    while True:
        _tick()
        current_budget().check_int(p)
        if _prime(p) and f[-1] % p:
            fp = _mod(f, p)
            derivative = _mod(tuple(_mul_int(i, c) for i, c in enumerate(fp) if i), p)
            if _gcd_mod(fp, derivative, p) == (1,):
                inverse = pow(fp[-1], -1, p)
                return p, _mod(tuple(_mul_int(c, inverse) for c in fp), p)
        p += 1


def _nullspace_mod(matrix, p):
    n = len(matrix)
    current_budget().matrix(n, n)
    rows, pivots, row = [list(r) for r in matrix], [], 0
    for col in range(n):
        pivot = next((j for j in range(row, n) if rows[j][col] % p), None)
        _tick(n - row)
        if pivot is None:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        inverse = pow(rows[row][col], -1, p)
        rows[row] = [_mul_int(v, inverse) % p for v in rows[row]]
        for j in range(n):
            _tick()
            if j != row and rows[j][col]:
                scale = rows[j][col]
                rows[j] = [(a - _mul_int(scale, b)) % p for a, b in zip(rows[j], rows[row])]
        pivots.append(col)
        row += 1
    basis = []
    for col in range(n):
        if col not in pivots:
            vector = [0] * n
            vector[col] = 1
            for j, pivot in enumerate(pivots):
                vector[pivot] = -rows[j][col] % p
            basis.append(_trim(vector))
    return tuple(basis)


def _berlekamp(f, p):
    n = len(f) - 1
    current_budget().matrix(n, n)
    xp, power = _pow_mod((0, 1), p, f, p), (1,)
    matrix = [[0] * n for _ in range(n)]
    for col in range(n):
        for row, coefficient in enumerate(power):
            matrix[row][col] = coefficient
        matrix[col][col] = (matrix[col][col] - 1) % p
        power = _divmod_mod(_mul_mod(power, xp, p), f, p)[1]
    basis = _nullspace_mod(matrix, p)
    factors = [f]
    for vector in basis:
        if len(factors) == len(basis):
            break
        for a in range(p):
            split = []
            for h in factors:
                _tick()
                g = _gcd_mod(h, _add_mod(vector, (a,), p, -1), p)
                if 1 < len(g) < len(h):
                    split.extend((g, _divmod_mod(h, g, p)[0]))
                else:
                    split.append(h)
            factors = split
            if len(factors) == len(basis):
                break
    if len(factors) != len(basis):
        raise InvalidInput("Modular factorization failed its dimension invariant")
    return tuple(sorted(factors, key=lambda h: (len(h), h)))


def _coefficient_bound(f):
    """Conservative bound for lc(f) times every primitive divisor of f."""
    budget = current_budget()
    total = 0
    for c in f:
        total += _mul_int(c, c)
        budget.check_int(total)
    norm = isqrt(total)
    if norm * norm < total:
        norm += 1
    degree = len(f) - 1
    budget.check_int(1 << degree)
    return _mul_int(_mul_int(abs(f[-1]), 1 << degree), norm)


def _hensel(f, p, factors, bound):
    """Lift one base-p digit at a time; CRT corrections keep factors monic."""
    inverses = []
    for i, factor in enumerate(factors):
        cofactor = (1,)
        for j, other in enumerate(factors):
            if i != j:
                cofactor = _mul_mod(cofactor, other, p)
        inverses.append(_inverse_mod(cofactor, factor, p))
    lifted, modulus, exponent = tuple(factors), p, 1
    while modulus <= 2 * bound:
        _tick()
        next_modulus = _mul_int(modulus, p)
        inverse_lc = pow(f[-1], -1, next_modulus)
        target = _mod(tuple(_mul_int(c, inverse_lc) for c in f), next_modulus)
        product = (1,)
        for factor in lifted:
            product = _mul_mod(product, factor, next_modulus)
        error = []
        for a, b in zip(target + (0,) * (len(f) - len(target)),
                        product + (0,) * (len(f) - len(product))):
            difference = (a - b) % next_modulus
            if difference % modulus:
                raise InvalidInput("Hensel factor congruence failed")
            error.append((difference // modulus) % p)
        updated = []
        for old, factor, inverse in zip(lifted, factors, inverses):
            correction = _divmod_mod(_mul_mod(_trim(error), inverse, p), factor, p)[1]
            updated.append(_add_mod(old, tuple(_mul_int(modulus, c) for c in correction), next_modulus))
        lifted, modulus, exponent = tuple(updated), next_modulus, exponent + 1
    return modulus, exponent, lifted


def _lift_data(f):
    p, modular = _good_prime(f)
    factors = _berlekamp(modular, p)
    modulus, exponent, lifted = _hensel(f, p, factors, _coefficient_bound(f))
    return p, exponent, modulus, lifted


def _candidate(f, factors, subset, modulus, variable):
    product = (1,)
    for index in subset:
        product = _mul_mod(product, factors[index], modulus)
    coefficients = []
    for coefficient in product:
        value = _mul_int(f[-1], coefficient) % modulus
        coefficients.append(value - modulus if value > modulus // 2 else value)
    content = 0
    for value in coefficients:
        _tick()
        content = gcd(content, abs(value))
    if not content:
        return Polynomial((0,), variable)
    sign = 1 if coefficients[-1] > 0 else -1
    return Polynomial(tuple(sign * c // content for c in coefficients), variable)


def _factor_square_free(f, variable):
    if len(f) == 2:
        return (Polynomial(f, variable).monic(),)
    _, _, modulus, lifted = _lift_data(f)
    stack, result = [(f, lifted)], []
    while stack:
        coefficients, factors = stack.pop()
        source = Polynomial(coefficients, variable)
        found = False
        for size in range(1, len(factors)):
            for subset in combinations(range(len(factors)), size):
                _tick()
                degree = sum(len(factors[i]) - 1 for i in subset)
                if degree > source.degree // 2:
                    continue
                candidate = _candidate(coefficients, factors, subset, modulus, variable)
                if candidate.degree != degree:
                    continue
                quotient, remainder = source.divmod(candidate)
                if remainder.is_zero:
                    chosen = set(subset)
                    stack.append((_primitive(candidate), tuple(factors[i] for i in subset)))
                    stack.append((_primitive(quotient), tuple(v for i, v in enumerate(factors) if i not in chosen)))
                    found = True
                    break
            if found:
                break
        if not found:
            result.append(source.monic())
    return tuple(result)


def _irreducibility_evidence(poly):
    if poly.degree == 1:
        return ("linear",)
    p, exponent, _, lifted = _lift_data(_primitive(poly))
    return "modular", p, exponent, lifted


def factor_with_evidence(f, *, limits=None):
    with computation(limits) as budget:
        _validate(f)
        budget.inspect(f)
        if f.is_zero:
            raise InvalidInput("The zero polynomial has no irreducible factorization")
        if f.degree <= 0:
            value = PolynomialFactorization(f.coefficients[0], ())
        else:
            entries = []
            for group in f.square_free().factors:
                for factor in _factor_square_free(_primitive(group.polynomial), f.variable):
                    entries.append(PolynomialFactor(factor, group.multiplicity))
            value = PolynomialFactorization(f.leading_coefficient,
                                             tuple(sorted(entries, key=lambda v: _key(v.polynomial))))
        evidence = ("qq-factorization-v1", tuple(_irreducibility_evidence(v.polynomial) for v in value.factors))
        budget.inspect(evidence, evidence=True)
        return value, evidence


def _rabin(f, p):
    """Independent finite-field irreducibility criterion, without Berlekamp."""
    degree = len(f) - 1
    if degree <= 0 or f[-1] != 1:
        return False
    prime_divisors, remaining, divisor = [], degree, 2
    while divisor <= remaining // divisor:
        _tick()
        if remaining % divisor == 0:
            prime_divisors.append(divisor)
            while remaining % divisor == 0:
                _tick()
                remaining //= divisor
        divisor += 1
    if remaining > 1:
        prime_divisors.append(remaining)
    checkpoints = {degree // divisor for divisor in prime_divisors}
    x = _divmod_mod((0, 1), f, p)[1]
    power = x
    for i in range(1, degree + 1):
        power = _pow_mod(power, p, f, p)
        if i in checkpoints and _gcd_mod(f, _add_mod(power, x, p, -1), p) != (1,):
            return False
    return power == x


def _check_irreducible(poly, evidence):
    if poly.degree == 1:
        return evidence == ("linear",)
    if type(evidence) is not tuple or len(evidence) != 4 or evidence[0] != "modular":
        return False
    _, p, exponent, factors = evidence
    if (type(p) is not int or type(exponent) is not int or exponent <= 0
            or type(factors) is not tuple or not factors or not _prime(p)):
        return False
    budget = current_budget()
    # Charge before exponentiation; malformed certificates cannot allocate huge ints.
    if exponent > budget.limits.max_integer_bits:
        return False
    modulus = 1
    for _ in range(exponent):
        modulus = _mul_int(modulus, p)
    source = _primitive(poly)
    if source[-1] % p == 0 or modulus <= 2 * _coefficient_bound(source):
        return False
    if any(type(f) is not tuple or len(f) <= 1 or any(type(c) is not int or c < 0 or c >= modulus for c in f)
           or f[-1] != 1 for f in factors):
        return False
    if sum(len(f) - 1 for f in factors) != poly.degree:
        return False
    modular = tuple(_mod(f, p) for f in factors)
    for i, factor in enumerate(modular):
        if not _rabin(factor, p):
            return False
        if any(_gcd_mod(factor, prior, p) != (1,) for prior in modular[:i]):
            return False
    product = (source[-1],)
    for factor in factors:
        product = _mul_mod(product, factor, modulus)
    if product != _mod(source, modulus):
        return False
    # Deliberately independent enumeration and reconstruction, not _candidate
    # or the producer's recombination routine. Degree <= n/2 is exhaustive:
    # every reducible polynomial has a factor of at most half its degree.
    indices = list(range(len(factors)))
    pending = [(0, (), 0)]
    while pending:
        position, chosen, degree = pending.pop()
        _tick()
        if position < len(indices):
            pending.append((position + 1, chosen, degree))
            next_degree = degree + len(factors[position]) - 1
            if next_degree <= poly.degree // 2:
                pending.append((position + 1, chosen + (position,), next_degree))
            continue
        if not chosen:
            continue
        coefficients = (source[-1],)
        for index in chosen:
            coefficients = _mul_mod(coefficients, factors[index], modulus)
        signed = tuple(c if 2 * c <= modulus else c - modulus for c in coefficients)
        divisor_poly = Polynomial(signed, poly.variable)
        if divisor_poly.degree != degree:
            return False
        divisor_poly = Polynomial(_primitive(divisor_poly), poly.variable)
        if poly.divmod(divisor_poly)[1].is_zero:
            return False
    return True


def check_factor(f, value, evidence, *, limits=None):
    with computation(limits) as budget:
        _validate(f)
        budget.inspect((f, value))
        budget.inspect(evidence, evidence=True)
        if f.is_zero:
            return False
        if (type(value) is not PolynomialFactorization or type(evidence) is not tuple
                or len(evidence) != 2 or evidence[0] != "qq-factorization-v1"
                or type(evidence[1]) is not tuple or len(evidence[1]) != len(value.factors)):
            return False
        # A claimed certificate may contain records forged by bypassing frozen
        # constructors. Product reconstruction alone cannot certify canonical
        # distinct factors, positive multiplicities or immutable record shape.
        try:
            if PolynomialFactorization(value.coefficient, value.factors) != value:
                return False
            if any(PolynomialFactor(entry.polynomial, entry.multiplicity) != entry
                   for entry in value.factors):
                return False
            for entry in value.factors:
                _validate(entry.polynomial)
        except (InvalidInput, AttributeError, TypeError, ValueError):
            return False
        if f.degree <= 0:
            return not value.factors and value.coefficient == f.coefficients[0] and evidence[1] == ()
        if value.coefficient != f.leading_coefficient:
            return False
        if sum(v.polynomial.degree * v.multiplicity for v in value.factors) != f.degree:
            return False
        product = Polynomial((value.coefficient,), f.variable)
        for entry, proof in zip(value.factors, evidence[1]):
            if entry.polynomial.variable != f.variable or not _check_irreducible(entry.polynomial, proof):
                return False
            product = product * entry.polynomial ** entry.multiplicity
        return product == f


def _validate(f, g=None):
    with computation() as budget:
        for value in (f,) if g is None else (f, g):
            if (type(value) is not Polynomial or type(getattr(value, "variable", None)) is not Symbol
                    or type(getattr(value, "coefficients", None)) is not tuple or not value.coefficients):
                raise InvalidInput("Expected a canonical Polynomial operand")
            if len(value.coefficients) > MAX_POLYNOMIAL_DEGREE + 1:
                raise InvalidInput("Polynomial exceeds the supported dense degree")
            budget.nodes(len(value.coefficients))
            for coefficient in value.coefficients:
                budget.tick()
                if not _canonical_rational(coefficient):
                    raise InvalidInput("Polynomial coefficients must be canonical exact rationals")
                budget.check_int(coefficient.numerator)
                budget.check_int(coefficient.denominator)
            if len(value.coefficients) > 1 and value.coefficients[-1].numerator == 0:
                raise InvalidInput("Polynomial coefficients must have no trailing zeros")
        if g is not None and f.variable != g.variable:
            raise InvalidInput("Polynomial operands require the same Symbol identity")


def _canonical_rational(value):
    if (type(value) is not Rational or type(getattr(value, "numerator", None)) is not int
            or type(getattr(value, "denominator", None)) is not int or value.denominator <= 0):
        return False
    budget = current_budget()
    if budget is not None:
        budget.check_int(value.numerator)
        budget.check_int(value.denominator)
    return gcd(value.numerator, value.denominator) == 1


def _integer_power(base, exponent):
    result = 1
    while exponent:
        _tick()
        if exponent & 1:
            result = _mul_int(result, base)
        exponent >>= 1
        if exponent:
            base = _mul_int(base, base)
    return result


def _integer_scale(poly, scalar):
    current_budget().nodes(len(poly))
    return _trim(_mul_int(coefficient, scalar) for coefficient in poly)


def _integer_quotient(poly, divisor):
    """All subresultant scaling divisions must be exact in Z."""
    current_budget().nodes(len(poly))
    result = []
    for coefficient in poly:
        _tick()
        quotient, remainder = divmod(coefficient, divisor)
        if remainder:
            raise ArithmeticError("Nonexact subresultant scaling division")
        result.append(quotient)
    return _trim(result)


def _pseudo_remainder(f, g):
    """lc(g) ** (deg(f)-deg(g)+1) f modulo g, without fractions."""
    if len(f) < len(g):
        return f
    exponent, leading = len(f) - len(g) + 1, g[-1]
    remainder = f
    while remainder != (0,) and len(remainder) >= len(g):
        _tick()
        shift, coefficient = len(remainder) - len(g), remainder[-1]
        scaled = list(_integer_scale(remainder, leading))
        for index, value in enumerate(g):
            scaled[index + shift] -= _mul_int(coefficient, value)
            current_budget().check_int(scaled[index + shift])
        remainder = _trim(scaled)
        exponent -= 1
    if remainder == (0,):
        return remainder
    return _integer_scale(remainder, _integer_power(leading, exponent))


def _subresultant_integer(f, g):
    """Brown's subresultant PRS, retaining scalar subresultants.

Pseudo-remainders and exact scalar divisions keep every intermediate in Z.
The scaling recurrence handles abnormal degree drops without introducing
fractions or incorrectly returning the final constant remainder itself.
"""
    n, m = len(f)-1, len(g)-1
    sign = 1
    if n < m:
        f, g, n, m = g, f, m, n
        sign = -1 if n*m % 2 else 1
    if m == 0:
        return sign * _integer_power(g[0], n)
    drop = n-m
    remainder = _integer_scale(_pseudo_remainder(f, g), -1 if (drop+1) % 2 else 1)
    leading = g[-1]
    scalar = _integer_power(leading, drop)
    previous = -scalar
    while remainder != (0,):
        _tick()
        next_degree = len(remainder)-1
        f, g, m, drop = g, remainder, next_degree, m-next_degree
        divisor = -_mul_int(leading, _integer_power(previous, drop))
        remainder = _integer_quotient(_pseudo_remainder(f, g), divisor)
        leading = g[-1]
        if drop > 1:
            numerator = _integer_power(-leading, drop)
            denominator = _integer_power(previous, drop-1)
            previous, rest = divmod(numerator, denominator)
            if rest:
                raise ArithmeticError("Nonexact scalar subresultant division")
        else:
            previous = -leading
        scalar = -previous
    return sign * scalar if m == 0 else 0


def resultant(f, g, *, limits=None):
    with computation(limits) as budget:
        _validate(f, g)
        budget.inspect((f, g))
        if f.is_zero or g.is_zero:
            return Rational(0)
        # Clear denominators and content once. Res(cF,dG)=c^deg(G)d^deg(F)Res(F,G).
        integer_f, integer_g = _primitive(f), _primitive(g)
        scale_f = f.leading_coefficient / Rational(integer_f[-1])
        scale_g = g.leading_coefficient / Rational(integer_g[-1])
        value = Rational(_subresultant_integer(integer_f, integer_g))
        return value * scale_f ** g.degree * scale_g ** f.degree


def _sylvester_resultant(f, g):
    """Independent rational Gaussian determinant of the Sylvester matrix."""
    if f.is_zero or g.is_zero:
        return Rational(0)
    m, n = f.degree, g.degree
    size = m + n
    budget = current_budget()
    budget.matrix(size, size)
    rows = [[Rational(0)] * size for _ in range(size)]
    for i in range(n):
        rows[i][i:i + m + 1] = reversed(f.coefficients)
    for i in range(m):
        rows[n + i][i:i + n + 1] = reversed(g.coefficients)
    determinant = Rational(1)
    for column in range(size):
        budget.tick(size - column)
        pivot = next((i for i in range(column, size) if rows[i][column].numerator), None)
        if pivot is None:
            return Rational(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        diagonal = rows[column][column]
        determinant = determinant * diagonal
        for i in range(column + 1, size):
            if rows[i][column].numerator:
                ratio = rows[i][column] / diagonal
                for j in range(column + 1, size):
                    rows[i][j] = rows[i][j] - ratio * rows[column][j]
                rows[i][column] = Rational(0)
    return determinant


def check_resultant(f, g, value, *, limits=None):
    with computation(limits) as budget:
        _validate(f, g)
        budget.inspect((f, g, value))
        return _canonical_rational(value) and value == _sylvester_resultant(f, g)


def discriminant(f, *, limits=None):
    with computation(limits) as budget:
        _validate(f)
        budget.inspect(f)
        if f.degree <= 0:
            return Rational(0)
        sign = -1 if f.degree * (f.degree - 1) // 2 % 2 else 1
        return sign * resultant(f, f.derivative()) / f.leading_coefficient


def check_discriminant(f, value, *, limits=None):
    with computation(limits) as budget:
        _validate(f)
        budget.inspect((f, value))
        if not _canonical_rational(value):
            return False
        if f.degree <= 0:
            return value == Rational(0)
        sign = -1 if f.degree * (f.degree - 1) // 2 % 2 else 1
        return value * f.leading_coefficient == sign * _sylvester_resultant(f, f.derivative())


def produce(operation, args):
    if operation == "factor":
        return factor_with_evidence(args["f"])
    if operation == "resultant":
        return resultant(args["f"], args["g"]), ()
    if operation == "discriminant":
        return discriminant(args["f"]), ()
    raise UnsupportedOperation("Unknown exact polynomial operation")


def check_operation(operation, args, value, evidence):
    if operation == "factor":
        return check_factor(args["f"], value, evidence)
    if evidence != ():
        return False
    if operation == "resultant":
        return check_resultant(args["f"], args["g"], value)
    if operation == "discriminant":
        return check_discriminant(args["f"], value)
    return False
