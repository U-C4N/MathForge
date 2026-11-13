"""Discrete logarithms modulo a prime and sums of two squares (mathforge.ntheory).

Every argument is an exact int (MCP kind and annotation "int", so operations_v6.run already rejects a bool, a float,
a str and None as invalid input), and no argument has a default. For each operation, its domain, the canonical form of
its value and the evidence that its checker, mathforge.certify.ntheory.discrete, verifies
(docs/v09/cards/m7-extended.md):

- discrete_log(base, target, p): the least k >= 0 with base**k = target modulo a prime p. p must be prime (a p below 2
  or composite is invalid input, decided with a proof) and base % p != 0 (invalid input otherwise); target is any int,
  reduced modulo p. Value DiscreteLogarithm(exponent, order): order is the multiplicative order d of base modulo p, and
  exponent the least solution k, or None when there is none; every solution is then exponent + t*order. Canonical
  form: order >= 1 (the record cannot see p, so order | p - 1 is the checker's test) and 0 <= exponent < order when it
  is an int. Evidence (factors, certificate): factors is the certified factorization of order, () for order = 1, and
  certificate the prime certificate of p, () below psi_13 and a Pocklington certificate from there on.
- two_squares(n): r2(n), the number of pairs (x, y) in Z^2 with x**2 + y**2 = n >= 0 (n < 0 is invalid input), and
  one representation as an example. Value SumOfTwoSquares(count, a, b): count = r2(n) (r2(0) = 1, r2(1) = 4,
  r2(25) = 12) and (a, b) one representation n = a**2 + b**2 with 0 <= a <= b, for example the one the producer's
  deterministic method finds (the value does not claim which one it is), or a and b both None when count = 0.
  Canonical form: count >= 0, a and b both None exactly when count = 0, else ints with 0 <= a <= b. Evidence
  (factors,): the certified factorization of n, () for n in {0, 1}, so ((),) then.

A prime certificate is what mathforge.kernels.certificates.prime_evidence builds; a certified factorization is
((q, e, certificate), ...) with strictly increasing primes, as factorization_evidence builds it.
mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/ntheory/test_discrete.py is the definition of done.

How produce() computes (docs/v09/cards/m7-extended.md). Nothing here is a proof: the checker verifies every value with
modular powers, prime proofs and certified factorizations of its own.
- discrete_log refuses base % p == 0 (invalid input for every p, so it comes first: one division, before any
  primality proof), then asks kernels.certificates.prime_evidence for the certificate of p, which refuses a p that is
  not prime (invalid input). p = 2 has the trivial group: order 1, exponent 0 for an odd target and None otherwise.
  Otherwise p - 1 = prod q**e is read from p's Pocklington certificate, which holds it from psi_13 on (factoring
  p - 1 a second time would double the dearest step), and below psi_13 the integer kernel factors p - 1; for each
  prime q the power
  y = base**((p - 1)/q**e) has the order q**f, q's exact power in the order d of base (the cofactor holds every other
  prime of p - 1 to its full power), so raising y to the q-th power until it is 1 finds f, and d = prod q**f. Then
  target**d != 1 (or target = 0) means target is no power of base (the powers of base are exactly the d-th roots of
  unity of the cyclic group): exponent None. Otherwise Pohlig and Hellman: for each q**f of d, g = base**(d/q**f) of
  order q**f and h = target**(d/q**f), and x = log_g h modulo q**f by halving f (_prime_power_log: the logarithm
  modulo q**f1 in the subgroup of order q**f1, then the rest in the subgroup of order q**f2, f = f1 + f2) down to
  logarithms in subgroups of the prime order q, each by Shanks's baby-step giant-step (m = isqrt(q - 1) + 1 baby
  steps gamma**j in a table, then up to m giant steps by gamma**-m); the Chinese remainder theorem joins the x modulo
  the q**f into the exponent modulo d, which lies in [0, d) and is therefore the least solution. The factors of d are
  certified prime by prime (prime_evidence).
- two_squares answers (1, 0, 0) for n = 0 and (4, 0, 1) for n = 1. Otherwise factorization_evidence(n) gives the
  certified factorization, the evidence, and r2(n) follows from it: 0 when a prime q = 3 (mod 4) has an odd exponent,
  else 4 * prod (e + 1) over the primes p = 1 (mod 4). The representation is a Gaussian integer of norm n: for each
  prime p = 1 (mod 4), p = x**2 + y**2 by Hermite and Serret (Brillhart's form of Cornacchia's algorithm): t = c**((p -
  1)/4) for the least c = 2, 3, ... with t*t = -1 (mod p), which holds exactly for the non-residues c, then the
  Euclidean algorithm on p and t until the first remainder x with x*x < p, and y = isqrt(p - x*x). The product of
  the (x + yi)**e (Brahmagupta and Fibonacci's identity is the multiplicativity of the norm), times (1 + i) for an odd
  power of 2, times the integer 2**(e // 2) * prod q**(e/2) over the primes q = 3 (mod 4), has the norm n; (a, b) are
  its absolute coordinates, sorted.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call, refuses an argument longer than max_integer_bits
with ResourceLimitError, and charges every further step before it runs, on the scale of trusted.integers: a long
division division_work, a square-and-multiply power multiplication_work per squaring and per multiplication (all
charged before its loop starts; nothing here calls the builtin pow), a baby-step giant-step search
2 * m * (multiplication_work + 1) units before it starts (a multiplication and a table operation per step), a product
of Gaussian integers four product_work charges, and a step of the Euclidean algorithm a unit plus one per 4,096 bits.
The integer kernel and kernels.certificates charge their own work, and a certified factorization is refused with
ResourceLimitError when check_factorization would need more than max_work, the rule of kernels.certificates: the
verifier checks in a fresh budget. With the default max_work of 2,000,000 the search fits a prime q of the order up
to about 2**36 for p of at most 383 bits (m = 2**18, about a million units) and fails at once from about 2**38 on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isqrt

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...registry import Argument, OperationSpec

# The producer rules:
# - Imports at the top come only from modules that `import mathforge` already loads; kernels are imported inside
#   produce(). Never import mathforge.certify, mathforge.lean, mathforge.mcp, mathforge._generated or a generated
#   area module (mathforge.ntheory, ...).
# - produce() validates its inputs first, with tests like `type(x) is int` (a bool, a float or a str is never an
#   exact number), and raises InvalidInput. It builds records only from validated data. For a Matrix argument it
#   calls current_budget().matrix(rows, cols) before it validates the shape.
# - The value is never None. The evidence is a tuple of ints, Rationals, tuples, str and records: no list, dict or
#   float. Under the verifier, 23 + 2 * request + 2 * value + evidence nodes stay within max_evidence_items.
# - Every operation takes at least one argument. Only an Expression or a Matrix that Result.expression should hold
#   is named `expression` or `A`.
# - Operation names are new, and nothing is superseded (no SUPERSEDES): tools/new_family.py checked these names.
# - No speed-suite fixture and no hand-written lean module: the generic lean runner serves v6 operations.

# The scope text of the ntheory families of v0.9 (areas/ntheory/modular.py), naming this family's objects as well.
_SCOPE = ("exact integers; certified primes, orders, residues, discrete logarithms and sums of two squares; "
          "canonical integer solution families")

SPECS = (
    OperationSpec("discrete_log", "ntheory",
                  (Argument("base", "int", None, "int"), Argument("target", "int", None, "int"),
                   Argument("p", "int", None, "int")),
                  "DiscreteLogarithm", "Solve base**k = target modulo a prime p for the least k >= 0.", _SCOPE,
                  version=6),
    OperationSpec("two_squares", "ntheory",
                  (Argument("n", "int", None, "int"),),
                  "SumOfTwoSquares", "Count the representations of n as a sum of two squares and give one.", _SCOPE,
                  version=6),
)


@dataclass(frozen=True)
class DiscreteLogarithm:
    """The solutions k >= 0 of base**k = target modulo a prime p: exactly exponent + t*order for t >= 0, where order is
    the multiplicative order of base modulo p and exponent the least solution; no solution when exponent is None.

    The canonical form: order is an exact int >= 1, and exponent is None or an exact int with 0 <= exponent < order.
    (order divides p - 1, which the record cannot see: the checker tests it.)"""
    exponent: int | None
    order: int

    def __post_init__(self):
        with computation() as budget:
            budget.tick()
            if type(self.order) is not int:  # Neither bool nor float is an exact integer.
                raise InvalidInput("The order of a discrete logarithm is an exact integer")
            budget.check_int(self.order)
            if self.order < 1:
                raise InvalidInput("The order of a discrete logarithm is at least 1")
            if self.exponent is None:
                return
            if type(self.exponent) is not int:
                raise InvalidInput("The exponent of a discrete logarithm is an exact integer or None")
            budget.check_int(self.exponent)
            if not 0 <= self.exponent < self.order:
                raise InvalidInput("The exponent of a discrete logarithm lies in [0, order)")


@dataclass(frozen=True)
class SumOfTwoSquares:
    """r2(n), the number of pairs (x, y) of integers with x**2 + y**2 = n, and, for example, one of them with
    0 <= a <= b; a and b are None when there is none.

    The canonical form: count is an exact int >= 0; a and b are both None when count is 0, and otherwise exact ints
    with 0 <= a <= b. Which representation (a, b) is, the value does not claim."""
    count: int
    a: int | None
    b: int | None

    def __post_init__(self):
        with computation() as budget:
            budget.tick()
            if type(self.count) is not int:  # Neither bool nor float is an exact integer.
                raise InvalidInput("The count of a sum of two squares is an exact integer")
            budget.check_int(self.count)
            if self.count < 0:
                raise InvalidInput("The count of a sum of two squares is at least 0")
            if self.count == 0:
                if self.a is not None or self.b is not None:
                    raise InvalidInput("A sum of two squares without representation has a = b = None")
                return
            if type(self.a) is not int or type(self.b) is not int:
                raise InvalidInput("A representation as a sum of two squares has two exact integers a and b")
            budget.check_int(self.a)
            budget.check_int(self.b)
            if not 0 <= self.a <= self.b:
                raise InvalidInput("A representation as a sum of two squares has 0 <= a <= b")


TYPES: dict[str, type] = {"discrete_logarithm": DiscreteLogarithm, "sum_of_two_squares": SumOfTwoSquares}
CHECKER = "mathforge.certify.ntheory.discrete"


# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
_NOT_PRIME = "discrete_log requires a prime p"
_NOT_A_UNIT = "discrete_log requires base % p != 0: base is not invertible modulo p"
_UNCHECKABLE = "Computation limit exceeded: work ({}) for the independent check of the evidence"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name and
    kind; the module states the algorithms and the evidence. InvalidInput for an argument that is no exact int or
    lies outside the operation's domain, ResourceLimitError past the budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the discrete family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    values = [args[name] for name in names]
    if any(type(value) is not int for value in values):  # Neither a bool nor a float nor a str is an exact integer.
        raise InvalidInput(f"{operation} requires exact integers: {', '.join(names)}")
    with computation() as budget:
        budget.tick()  # The call itself; on a valid input every operation charges at least one unit more.
        for value in values:
            budget.check_int(value)
        return _PRODUCERS[operation](budget, *values)


def _discrete_log(budget: Budget, base: int, target: int, p: int) -> tuple[DiscreteLogarithm, tuple]:
    """DiscreteLogarithm(exponent, order) of base**k = target modulo the prime p, and the evidence
    (factors of order, certificate of p)."""
    if p < 2:
        raise InvalidInput(_NOT_PRIME)
    from ...kernels.certificates import prime_evidence
    from ...trusted.integers import division_work, factorization_entry_work, multiplication_work

    # base % p == 0 is invalid input whether or not p is prime, so it is decided first, by one division: a prime p
    # that the budget cannot certify then still gives invalid input for such a base, not resource_limit_exceeded.
    residue = _reduced(budget, base, p)
    if not residue:
        raise InvalidInput(_NOT_A_UNIT)
    try:
        certificate = prime_evidence(p)
    except InvalidInput as error:  # p is an exact int, so prime_evidence refuses only a p that is not prime.
        raise InvalidInput(_NOT_PRIME) from error
    goal = _reduced(budget, target, p)
    if p == 2:  # (Z/2)^* = {1}: base = 1 has the order 1, and only an odd target is a power of it.
        return DiscreteLogarithm(0 if goal else None, 1), ((), certificate)
    work = multiplication_work(p.bit_length())
    group = p - 1
    # The order d of base: q's exact power q**f in d, for each q**e of p - 1, is the order of base**((p - 1)/q**e).
    order, pairs = 1, []
    for q, e in _group_factors(budget, group, certificate):  # Increasing primes.
        budget.tick(factorization_entry_work(0, q.bit_length() * e) + 1
                    + division_work(group.bit_length(), q.bit_length() * e))
        y = _power(budget, residue, group // q ** e, p, work)
        f = 0
        while y != 1:
            if f == e:  # base**(p - 1) = 1 for the prime p: only a wrong factorization of p - 1 gets here.
                raise ArithmeticError("The order of a unit does not divide p - 1")
            y = _power(budget, y, q, p, work)
            f += 1
        if f:
            budget.tick(factorization_entry_work(order.bit_length(), q.bit_length() * f))
            order *= q ** f
            pairs.append((q, f))
    evidence = (_certified(budget, tuple(pairs)), certificate)
    if not goal or _power(budget, goal, order, p, work) != 1:  # No d-th root of unity: no power of base.
        return DiscreteLogarithm(None, order), evidence
    # Pohlig and Hellman: the logarithm modulo each q**f of the order, joined by the Chinese remainder theorem.
    exponent, modulus = 0, 1
    for q, f in pairs:
        budget.tick(factorization_entry_work(0, q.bit_length() * f) + 1
                    + division_work(order.bit_length(), q.bit_length() * f))
        prime_power = q ** f
        cofactor = order // prime_power
        g = _power(budget, residue, cofactor, p, work)  # Of the order q**f.
        h = _power(budget, goal, cofactor, p, work)  # In the subgroup that g generates, since goal**d = 1.
        x = _prime_power_log(budget, g, h, q, f, p, work)
        exponent, modulus = _crt(budget, exponent, modulus, x, prime_power)
    if _power(budget, residue, exponent, p, work) != goal:  # Pohlig-Hellman solves every solvable equation.
        raise ArithmeticError("Pohlig-Hellman produced no solution")
    return DiscreteLogarithm(exponent, order), evidence


def _group_factors(budget: Budget, group: int, certificate: tuple) -> tuple[tuple[int, int], ...]:
    """The prime factorization ((q, e), ...) of group = p - 1, increasing primes. From psi_13 on the Pocklington
    certificate of p, ("pocklington", p, ((q, e, certificate_q), ...), witnesses), already holds it (Pocklington's
    theorem needs p - 1 factored completely): reading it spares a second factorization of p - 1, which costs as much
    as the first and took orders with primes of about 2**34 past the default budget. Below psi_13 the certificate is
    (), and the integer kernel factors p - 1, charging its work to the budget."""
    from ...kernels.integers import factor
    from ...trusted.integers import factorization_entry_work

    if not certificate:
        return factor(group)
    pairs, product = tuple((q, e) for q, e, _ in certificate[2]), 1
    for q, e in pairs:  # The kernel built the certificate from a factorization of p - 1: a mismatch is its bug.
        budget.tick(factorization_entry_work(product.bit_length(), q.bit_length() * e))
        product *= q ** e
    if product != group:
        raise ArithmeticError("The Pocklington certificate does not factor p - 1")
    return pairs


def _prime_power_log(budget: Budget, g: int, h: int, q: int, f: int, p: int, work: int) -> int:
    """x in [0, q**f) with g**x = h modulo p, for g of the order q**f and h a power of g. With f = f1 + f2 and
    x = x1 + q**f1 * x2: x1 is the logarithm of h**(q**f2) to the base g**(q**f2), of the order q**f1, and x2 that of
    h * g**-x1 = (g**(q**f1))**x2 to the base g**(q**f1), of the order q**f2; halving f costs O(f log f) powers
    by q in all, where the digit-by-digit loop costs O(f**2) (p = 13 * 2**1000 + 1 took it past the default budget).
    A prime order, f = 1, is the baby-step giant-step search."""
    from ...trusted.integers import factorization_entry_work

    if f == 1:
        return _baby_giant(budget, g, h, q, p, work)
    low = f >> 1
    high = f - low
    budget.tick(factorization_entry_work(0, q.bit_length() * high) + factorization_entry_work(0, q.bit_length() * f))
    up_high, up_low = q ** high, q ** low  # Both below q**f, the order.
    x1 = _prime_power_log(budget, _power(budget, g, up_high, p, work), _power(budget, h, up_high, p, work), q, low,
                          p, work)
    budget.tick(1 + work)  # h * g**-x1, with g**-x1 = g**(q**f - x1).
    rest = h * _power(budget, g, (up_high * up_low - x1) if x1 else 0, p, work) % p
    x2 = _prime_power_log(budget, _power(budget, g, up_low, p, work), rest, q, high, p, work)
    budget.tick(2 + factorization_entry_work(0, q.bit_length() * f))  # x1 + q**f1 * x2, below q**f.
    return x1 + up_low * x2


def _baby_giant(budget: Budget, gamma: int, h: int, q: int, p: int, work: int) -> int:
    """t in [0, q) with gamma**t = h modulo p, for gamma of the prime order q and h a power of gamma, by Shanks's
    baby-step giant-step algorithm. The whole search is charged before it starts, so a q too large for the budget
    fails at once."""
    if h == 1:
        budget.tick()
        return 0
    m = isqrt(q - 1) + 1  # m * m >= q, and m <= q.
    budget.tick(1 + 2 * m * (work + 1))  # m baby steps and up to m giant steps: a product and a table operation each.
    table: dict[int, int] = {}
    power = 1
    for j in range(m):
        table.setdefault(power, j)
        power = power * gamma % p
    step = _power(budget, gamma, q - m, p, work)  # gamma**-m, as gamma**q = 1.
    y = h
    for i in range(m):
        j = table.get(y)
        if j is not None:
            return i * m + j
        y = y * step % p
    raise ArithmeticError("The baby-step giant-step search found no logarithm")


def _crt(budget: Budget, x1: int, m1: int, x2: int, m2: int) -> tuple[int, int]:
    """(x, m1 * m2) with x = x1 (mod m1), x = x2 (mod m2) and 0 <= x < m1 * m2, for coprime m1 and m2 >= 1."""
    from ...trusted.integers import division_work, product_work

    budget.tick(1 + division_work(m1.bit_length(), m2.bit_length()))
    inverse = _inverse(budget, m1 % m2, m2)
    budget.tick(3 + 2 * product_work(m1.bit_length() + m2.bit_length(), m2.bit_length())
                + division_work(2 * m2.bit_length() + m1.bit_length(), m2.bit_length()))
    return x1 + m1 * ((x2 - x1) * inverse % m2), m1 * m2


def _inverse(budget: Budget, a: int, m: int) -> int:
    """The inverse of a modulo m in [0, m), for gcd(a, m) = 1 and m >= 1, by the extended Euclidean algorithm; each
    step is charged before it runs, a unit and one per 4,096 bits of m (its division and its update of the
    coefficient are linear in the length of m)."""
    step = 1 + (m.bit_length() >> 12)
    r0, r1, s0, s1 = m, a, 0, 1  # r_i = s_i * a modulo m throughout.
    while r1:
        budget.tick(step)
        q, r = divmod(r0, r1)
        r0, r1, s0, s1 = r1, r, s1, s0 - q * s1
    if r0 != 1:  # The moduli of the Chinese remainder theorem are powers of distinct primes.
        raise ArithmeticError("The moduli of the Chinese remainder theorem are not coprime")
    return s0 % m


def _two_squares(budget: Budget, n: int) -> tuple[SumOfTwoSquares, tuple]:
    """SumOfTwoSquares(r2(n), a, b) and the evidence (factors of n,)."""
    if n < 0:
        raise InvalidInput("two_squares requires n >= 0")
    if n < 2:  # r2(0) = 1, (0, 0) alone; r2(1) = 4, (+-1, 0) and (0, +-1). Neither has a prime to hand over.
        budget.tick()
        return SumOfTwoSquares(1 if n == 0 else 4, 0, n), ((),)
    from ...kernels.certificates import factorization_evidence
    from ...trusted.integers import factorization_entry_work, multiplication_work, product_work

    factors = factorization_evidence(n)
    count = 4
    for prime, exponent, _ in factors:
        budget.tick(1 + product_work(count.bit_length(), exponent.bit_length()))
        if prime % 4 == 3 and exponent % 2:  # Jacobi: no representation at all.
            return SumOfTwoSquares(0, None, None), (factors,)
        if prime % 4 == 1:
            count *= exponent + 1
    # A Gaussian integer re + im*i of norm n / scale**2, and the integer scale.
    re, im, scale = 1, 0, 1
    for prime, exponent, _ in factors:
        if prime % 4 == 1:
            x, y = _prime_as_squares(budget, prime, multiplication_work(prime.bit_length()))
            re, im = _gaussian_times(budget, (re, im), _gaussian_power(budget, (x, y), exponent))
            continue
        half = exponent >> 1  # 2**e = (1 + i)**e up to a unit; q**e = (q**(e/2))**2 for q = 3 (mod 4), e even.
        budget.tick(factorization_entry_work(scale.bit_length(), prime.bit_length() * half))
        scale *= prime ** half
        if exponent & 1:  # Only for the prime 2: an odd power of a prime 3 (mod 4) has returned above.
            budget.tick(2 + (max(re.bit_length(), im.bit_length()) >> 12))
            re, im = re - im, re + im  # (re + im*i)(1 + i).
    budget.tick(1 + 2 * product_work(scale.bit_length(), max(re.bit_length(), im.bit_length())))
    a, b = sorted((abs(re) * scale, abs(im) * scale))
    return SumOfTwoSquares(count, a, b), (factors,)


def _prime_as_squares(budget: Budget, p: int, work: int) -> tuple[int, int]:
    """(x, y) with x*x + y*y = p for a prime p = 1 (mod 4), by Hermite and Serret: a square root t of -1 modulo p,
    then the Euclidean algorithm on p and t down to the first remainder x with x*x < p."""
    exponent = (p - 1) >> 2
    c = 2
    while True:
        if c >= p:  # A prime p = 1 (mod 4) has non-residues below it: only a composite p gets here.
            raise ArithmeticError("No square root of -1 modulo a prime 1 (mod 4)")
        t = _power(budget, c, exponent, p, work)
        budget.tick(1 + work)
        if t * t % p == p - 1:  # c**((p - 1)/2) = -1: c is a non-residue, and t a square root of -1.
            break
        c += 1
    step = 1 + (p.bit_length() >> 12)
    r0, r1 = p, t
    while True:
        budget.tick(step + work)  # The comparison r1*r1 < p and, past it, a long division.
        if r1 * r1 < p:
            break
        r0, r1 = r1, r0 % r1
    budget.tick(1 + work)
    y = isqrt(p - r1 * r1)
    if r1 * r1 + y * y != p:
        raise ArithmeticError("The Hermite-Serret algorithm found no representation")
    return r1, y


def _gaussian_times(budget: Budget, z: tuple[int, int], w: tuple[int, int]) -> tuple[int, int]:
    """The product of the Gaussian integers z and w, as (real part, imaginary part), charged before it runs."""
    from ...trusted.integers import product_work

    left = max(z[0].bit_length(), z[1].bit_length())
    right = max(w[0].bit_length(), w[1].bit_length())
    budget.tick(2 + 4 * product_work(left, right))
    return z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0]


def _gaussian_power(budget: Budget, z: tuple[int, int], exponent: int) -> tuple[int, int]:
    """z**exponent for a Gaussian integer z and an exponent >= 1, by left-to-right square-and-multiply; each product
    is charged before it runs."""
    result = z
    for bit in bin(exponent)[3:]:  # The bits after the leading 1, from the top.
        result = _gaussian_times(budget, result, result)
        if bit == "1":
            result = _gaussian_times(budget, result, z)
    return result


_PRODUCERS: dict[str, Callable[..., tuple[object, tuple]]] = {
    "discrete_log": _discrete_log,
    "two_squares": _two_squares,
}


def _reduced(budget: Budget, number: int, modulus: int) -> int:
    """number % modulus, in [0, modulus) for a modulus >= 1, its long division charged before it runs."""
    from ...trusted.integers import division_work

    budget.tick(1 + division_work(number.bit_length(), modulus.bit_length()))
    return number % modulus


def _power(budget: Budget, base: int, exponent: int, modulus: int, work: int) -> int:
    """base**exponent modulo modulus, for 0 <= base < modulus, exponent >= 0 and modulus >= 1, by left-to-right binary
    square-and-multiply: never the builtin pow. work is multiplication_work of the modulus's bit length, the charge of
    each squaring and each multiplication, all charged before the loop starts."""
    if not exponent:
        budget.tick()
        return 1 % modulus
    budget.tick(1 + (exponent.bit_length() + exponent.bit_count() - 2) * work)
    result = base
    for bit in bin(exponent)[3:]:  # The bits after the leading 1, from the top.
        result = result * result % modulus
        if bit == "1":
            result = result * base % modulus
    return result


def _certified(budget: Budget, pairs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, tuple], ...]:
    """((q, e, certificate), ...) for a prime factorization ((q, e), ...) with increasing primes that the integer
    kernel found, each certificate from prime_evidence. ResourceLimitError when check_factorization would need more
    than max_work, as factorization_evidence refuses it: the verifier checks in a fresh budget of the same limits."""
    from ...kernels.certificates import prime_evidence
    from ...trusted.integers import factorization_work

    entries = []
    for q, e in pairs:
        try:
            certificate = prime_evidence(q)
        except InvalidInput as error:  # factor() took a composite for a prime: a producer bug, never an input error.
            raise ArithmeticError("Internal factorization produced a composite factor") from error
        entries.append((q, e, certificate))
    factors = tuple(entries)
    allowance = budget.limits.max_work  # The check runs in a budget of its own (kernels.certificates).
    if factorization_work(factors, allowance) > allowance:
        raise ResourceLimitError(_UNCHECKABLE.format(budget.limits.max_work))
    return factors


def _render_logarithm(logarithm, renderer, child):
    r"""k = 8 (mod 10), or: no solution. In LaTeX k \equiv 8 \pmod{10}, or \text{no solution}."""
    if logarithm.exponent is None:
        return renderer.label("no solution")
    exponent, order = child(logarithm.exponent), child(logarithm.order)
    if renderer.latex:
        return renderer.join((r"k \equiv ", exponent, r" \pmod{", order, "}"))
    return renderer.join(("k = ", exponent, " (mod ", order, ")"))


def _render_squares(squares, renderer, child):
    r"""12 representations, e.g. 25 = 0^2 + 5^2 (1 representation for n = 0), or: no representation. In LaTeX
    12\ \text{representations, e.g.}\ 25 = 0^{2} + 5^{2}, or \text{no representation}."""
    if squares.a is None:
        return renderer.label("no representation")
    words = "representation, e.g." if squares.count == 1 else "representations, e.g."
    a, b = squares.a, squares.b
    n = child(a * a + b * b)  # The record holds a and b of at most max_integer_bits each: a short product.
    square = "^{2}" if renderer.latex else "^2"
    if renderer.latex:
        return renderer.join((child(squares.count), r"\ ", renderer.label(words), r"\ ", n, " = ", child(a), square,
                              " + ", child(b), square))
    return renderer.join((child(squares.count), " ", words, " ", n, " = ", child(a), square, " + ", child(b), square))


RENDER = {DiscreteLogarithm: _render_logarithm, SumOfTwoSquares: _render_squares}
