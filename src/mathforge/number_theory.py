"""Bounded exact integer arithmetic and independently checked prime certificates."""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt

from .errors import InvalidInput
from .limits import computation, current_budget, ResourceLimitError


ARGUMENTS = {
    "integer_gcd": {"a", "b"}, "integer_lcm": {"a", "b"},
    "integer_xgcd": {"a", "b"}, "mod_inverse": {"a", "modulus"},
    "crt": {"congruences"}, "is_prime": {"n"}, "factor_integer": {"n"},
    "divisors": {"n"}, "totient": {"n"},
}


def _integer(value, *, positive=False):
    if type(value) is not int or positive and value <= 0:
        raise InvalidInput("An exact positive integer is required" if positive else "An exact integer is required")
    current_budget().check_int(value)
    return value


@dataclass(frozen=True)
class IntegerFactorization:
    sign: int
    factors: tuple[tuple[int, int], ...]

    def __post_init__(self):
        with computation() as budget:
            if type(self.sign) is not int or self.sign not in (-1, 1):
                raise InvalidInput("A nonzero integer factorization has sign -1 or 1")
            if type(self.factors) is not tuple:
                raise InvalidInput("Factors must be an immutable tuple")
            previous = 1
            for item in self.factors:
                budget.tick()
                if type(item) is not tuple or len(item) != 2:
                    raise InvalidInput("Each factor is a (prime, multiplicity) pair")
                prime, exponent = item
                _integer(prime, positive=True)
                _integer(exponent, positive=True)
                if prime <= previous:
                    raise InvalidInput("Factors must be strictly increasing and at least two")
                previous = prime


@dataclass(frozen=True)
class CRTResult:
    """A residue class; two None fields denote an inconsistent system."""
    residue: int | None
    modulus: int | None

    def __post_init__(self):
        with computation():
            if self.residue is None and self.modulus is None:
                return
            _integer(self.residue)
            _integer(self.modulus, positive=True)
            if not 0 <= self.residue < self.modulus:
                raise InvalidInput("A CRT residue must be canonical modulo its positive modulus")


TYPES = {"integer_factorization": IntegerFactorization, "crt_result": CRTResult}


def _product(a, b):
    current_budget().product(a, b)
    result = a * b
    current_budget().check_int(result)
    return result


def _xgcd(a, b):
    r0, r1, s0, s1, t0, t1 = abs(a), abs(b), 1, 0, 0, 1
    while r1:
        current_budget().tick()
        q, remainder = divmod(r0, r1)
        r0, r1 = r1, remainder
        s0, s1 = s1, s0 - _product(q, s1)
        t0, t1 = t1, t0 - _product(q, t1)
    return r0, s0 if a >= 0 else -s0, t0 if b >= 0 else -t0


def _powmod(a, exponent, modulus):
    result, a = 1 % modulus, a % modulus
    while exponent:
        current_budget().tick()
        if exponent & 1:
            result = _product(result, a) % modulus
        exponent //= 2
        if exponent:
            a = _product(a, a) % modulus
    return result


def _trial_factors(n):
    """The (prime, exponent) pairs of n >= 0, sorted (() below 2), from the integer kernel: trial division by the
    primes below 2**10, then Pollard-Brent rho, charging the active budget once per batch. v0.7 divided by every
    odd number up to the square root; the pairs are the same, the factorization of an integer being unique, and every
    prime is still certified by _prime_certificate."""
    if n < 2:
        return ()
    from .kernels.integers import factor  # On first use: importing mathforge loads no kernel.
    pairs = factor(n)
    current_budget().nodes(len(pairs))
    return pairs


def _prime_certificate(n, depth=0):
    budget = current_budget()
    budget.tick()
    if depth > budget.limits.max_depth:
        raise ResourceLimitError("Prime certificate exceeds the depth budget")
    if n < 2:
        return False, ("below", n)
    if n < 10_000:
        for divisor in range(2, isqrt(n) + 1):
            budget.tick()
            if n % divisor == 0:
                return False, ("factor", n, divisor)
        return True, ("trial", n)
    for divisor in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        budget.tick()
        if n % divisor == 0:
            return False, ("factor", n, divisor)
    # A failing Fermat test is a proof of compositeness, never a probable-prime claim.
    for witness in (2, 3, 5, 7):
        if _powmod(witness, n - 1, n) != 1:
            return False, ("fermat", n, witness)
    factors = _trial_factors(n - 1)
    certified = []
    for prime, exponent in factors:
        proven, certificate = _prime_certificate(prime, depth + 1)
        if not proven:
            raise ArithmeticError("Internal factorization produced a composite factor")
        certified.append((prime, exponent, certificate))
    witnesses = []
    for prime, _ in factors:
        witness = 2
        while witness < n:
            budget.tick()
            fermat = _powmod(witness, n - 1, n)
            if fermat != 1:
                return False, ("fermat", n, witness)
            divisor = gcd(_powmod(witness, (n - 1) // prime, n) - 1, n)
            if divisor == 1:
                witnesses.append(witness)
                break
            if 1 < divisor < n:
                return False, ("factor", n, divisor)
            witness += 1
        else:
            raise ArithmeticError("No Pocklington witness found")
    return True, ("pocklington", n, tuple(certified), tuple(witnesses))


def _check_prime(n, claimed, certificate, depth=0):
    """Check a trial/divisor/Fermat/Pocklington proof without factoring n or n-1."""
    budget = current_budget()
    budget.tick()
    if depth > budget.limits.max_depth:
        raise ResourceLimitError("Prime certificate exceeds the depth budget")
    if type(claimed) is not bool or type(certificate) is not tuple or len(certificate) < 2:
        return False
    kind, recorded = certificate[:2]
    if type(recorded) is not int or recorded != n:
        return False
    if kind == "below":
        return len(certificate) == 2 and not claimed and n < 2
    if kind == "factor":
        if len(certificate) != 3 or type(certificate[2]) is not int:
            return False
        divisor = certificate[2]
        return not claimed and 1 < divisor < n and n % divisor == 0
    if kind == "fermat":
        if len(certificate) != 3 or type(certificate[2]) is not int:
            return False
        witness = certificate[2]
        return not claimed and n > 2 and 1 < witness < n and _powmod(witness, n - 1, n) != 1
    if kind == "trial":
        if len(certificate) != 2 or not claimed or not 2 <= n < 10_000:
            return False
        for divisor in range(2, isqrt(n) + 1):
            budget.tick()
            if n % divisor == 0:
                return False
        return True
    if kind != "pocklington" or len(certificate) != 4 or not claimed or n < 2:
        return False
    factors, witnesses = certificate[2:]
    if type(factors) is not tuple or type(witnesses) is not tuple or len(factors) != len(witnesses) or not factors:
        return False
    total, previous = 1, 1
    for factor, witness in zip(factors, witnesses):
        if type(factor) is not tuple or len(factor) != 3:
            return False
        prime, exponent, child = factor
        if type(prime) is not int or prime <= previous or type(exponent) is not int or exponent < 1:
            return False
        if not _check_prime(prime, True, child, depth + 1):
            return False
        if exponent > (n - 1).bit_length():
            return False
        for _ in range(exponent):
            total = _product(total, prime)
            if total > n - 1:
                return False
        if type(witness) is not int or not 1 < witness < n:
            return False
        if _powmod(witness, n - 1, n) != 1 or gcd(_powmod(witness, (n - 1) // prime, n) - 1, n) != 1:
            return False
        previous = prime
    return total == n - 1


def _factor(n):
    if n == 0:
        raise InvalidInput("Zero has no finite prime factorization")
    pairs = _trial_factors(abs(n))
    value = IntegerFactorization(-1 if n < 0 else 1, pairs)
    certificates = tuple(_prime_certificate(prime)[1] for prime, _ in pairs)
    return value, certificates


def _check_factor(n, value, certificates):
    if n == 0 or type(value) is not IntegerFactorization or type(certificates) is not tuple:
        return False
    if value.sign != (-1 if n < 0 else 1) or type(value.sign) is not int or type(value.factors) is not tuple or len(value.factors) != len(certificates):
        return False
    product, previous = 1, 1
    for pair, certificate in zip(value.factors, certificates):
        if type(pair) is not tuple or len(pair) != 2:
            return False
        prime, exponent = pair
        if type(prime) is not int or prime <= previous or type(exponent) is not int or not 1 <= exponent <= abs(n).bit_length():
            return False
        if not _check_prime(prime, True, certificate):
            return False
        for _ in range(exponent):
            product = _product(product, prime)
            if product > abs(n):
                return False
        previous = prime
    return product == abs(n)


def _congruences(values):
    if type(values) is not tuple:
        raise InvalidInput("Congruences must be immutable (residue, positive modulus) pairs")
    for pair in values:
        current_budget().tick()
        if type(pair) is not tuple or len(pair) != 2:
            raise InvalidInput("Each congruence requires a residue and modulus")
        _integer(pair[0])
        _integer(pair[1], positive=True)
    return values


def _validate(operation, args):
    if operation not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[operation]:
        raise InvalidInput("Invalid integer operation arguments")
    if operation == "crt":
        _congruences(args["congruences"])
    else:
        for name, value in args.items():
            _integer(value, positive=name == "modulus" or operation == "totient")


def _produce(operation, args, *, limits=None):
    with computation(limits) as budget:
        budget.inspect(args)
        _validate(operation, args)
        if operation in ("integer_gcd", "integer_xgcd", "integer_lcm", "mod_inverse"):
            a, b = args["a"], args.get("b", args.get("modulus"))
            g, s, t = _xgcd(a, b)
            if operation == "integer_gcd":
                return g, (s, t)
            if operation == "integer_xgcd":
                return (g, s, t), ()
            if operation == "integer_lcm":
                return (0 if not g else abs(_product(a // g, b))), (g, s, t)
            if g != 1:
                raise InvalidInput("The modular inverse does not exist")
            return s % b, ((1 - _product(a, s % b)) // b,)
        if operation == "crt":
            residue, modulus, trace = 0, 1, []
            for index, (target, next_modulus) in enumerate(args["congruences"]):
                g, s, t = _xgcd(modulus, next_modulus)
                delta = target - residue
                trace.append((g, s, t))
                if delta % g:
                    return CRTResult(None, None), (tuple(trace), index)
                step = _product(delta // g, s) % (next_modulus // g)
                residue += _product(modulus, step)
                modulus = _product(modulus, next_modulus // g)
                residue %= modulus
            return CRTResult(residue, modulus), (tuple(trace), None)
        n = args["n"]
        if operation == "is_prime":
            value, certificate = _prime_certificate(n)
            return value, (certificate,)
        factorization, certificates = _factor(n)
        if operation == "factor_integer":
            return factorization, certificates
        if operation == "totient":
            value = n
            for prime, _ in factorization.factors:
                value = _product(value // prime, prime - 1)
            return value, (factorization, certificates)
        values = [1]
        for prime, exponent in factorization.factors:
            original, power = tuple(values), 1
            for _ in range(exponent):
                power = _product(power, prime)
                budget.nodes(len(original))
                values.extend(_product(divisor, power) for divisor in original)
        return tuple(sorted(values)), (factorization, certificates)


def produce(operation, args, *, limits=None):
    with computation(limits) as budget:
        value, evidence = _produce(operation, args)
        budget.inspect(value)
        budget.inspect(evidence, evidence=True)
        return value, evidence


def _bezout(a, b, g, s, t):
    if any(type(v) is not int for v in (g, s, t)) or g < 0:
        return False
    if g == 0:
        return a == b == 0 and _product(a, s) + _product(b, t) == 0
    return a % g == b % g == 0 and _product(a, s) + _product(b, t) == g


def check_operation(operation, args, value, evidence, *, limits=None):
    with computation(limits) as budget:
        try:
            budget.inspect(args)
            budget.inspect((value, evidence), evidence=True)
            _validate(operation, args)
            if type(evidence) is not tuple:
                return False
            if operation == "integer_xgcd":
                return evidence == () and type(value) is tuple and len(value) == 3 and _bezout(args["a"], args["b"], *value)
            if operation == "integer_gcd":
                return len(evidence) == 2 and _bezout(args["a"], args["b"], value, *evidence)
            if operation == "integer_lcm":
                if type(value) is not int or len(evidence) != 3 or not _bezout(args["a"], args["b"], *evidence):
                    return False
                g = evidence[0]
                return value == (abs(_product(args["a"] // g, args["b"])) if g else 0)
            if operation == "mod_inverse":
                return (type(value) is int and 0 <= value < args["modulus"] and len(evidence) == 1
                        and type(evidence[0]) is int
                        and _product(args["a"], value) + _product(args["modulus"], evidence[0]) == 1)
            if operation == "crt":
                if type(value) is not CRTResult or len(evidence) != 2 or type(evidence[0]) is not tuple:
                    return False
                trace, failure = evidence
                inputs = args["congruences"]
                if failure is not None and (type(failure) is not int or not 0 <= failure < len(inputs)):
                    return False
                if len(trace) != (len(inputs) if failure is None else failure + 1):
                    return False
                residue, modulus = 0, 1
                for index, ((target, next_modulus), entry) in enumerate(zip(inputs, trace)):
                    if type(entry) is not tuple or len(entry) != 3 or not _bezout(modulus, next_modulus, *entry):
                        return False
                    g, s, _ = entry
                    delta = target - residue
                    if delta % g:
                        return failure == index and value.residue is None and value.modulus is None
                    if failure == index:
                        return False
                    residue += _product(modulus, _product(delta // g, s) % (next_modulus // g))
                    modulus = _product(modulus, next_modulus // g)
                    residue %= modulus
                return failure is None and type(value.residue) is int and type(value.modulus) is int and (value.residue, value.modulus) == (residue, modulus)
            n = args["n"]
            if operation == "is_prime":
                return len(evidence) == 1 and _check_prime(n, value, evidence[0])
            if operation == "factor_integer":
                return _check_factor(n, value, evidence)
            if len(evidence) != 2 or not _check_factor(n, *evidence):
                return False
            factorization = evidence[0]
            if operation == "totient":
                if type(value) is not int:
                    return False
                # Product of phi(p**e), rather than the producer's successive n/p updates.
                expected = 1
                for prime, exponent in factorization.factors:
                    term = prime - 1
                    for _ in range(exponent - 1):
                        term = _product(term, prime)
                    expected = _product(expected, term)
                return value == expected
            if type(value) is not tuple or not value or any(type(d) is not int or d <= 0 for d in value):
                return False
            count = 1
            for _, exponent in factorization.factors:
                count = _product(count, exponent + 1)
            if len(value) != count:
                return False
            previous = 0
            for divisor in value:
                budget.tick()
                if divisor <= previous or abs(n) % divisor:
                    return False
                previous = divisor
            return True
        except (InvalidInput, TypeError, ValueError, AttributeError, KeyError, IndexError, ZeroDivisionError):
            return False
