"""Independent checker of the family mathforge.areas.algebra.sequences: sturm_sequence and cyclotomic_polynomial.

The checker never reruns the producer's algorithms (Euclidean remainders, cyclotomic division). It reads every record as
plain data, recognized by the module and name of its class: a Polynomial (coefficients, canonical Rationals lowest
degree first without a trailing zero, at most 1,025 of them; variable, a Symbol), each field from the record itself.
Every number must have its exact type (type(x) is int: a bool, a float or a str is no integer; a Rational subclass is no
Rational; a list is no tuple) and the canonical form stated below, and the evidence exactly the layout below; anything
else answers False, and so does a claim on an input outside the operation's domain, which has no value at all (the zero
polynomial, n < 1), and a claim whose arguments are not a plain dict of exactly the operation's argument names. Work is
charged to the active budget before it runs, in every loop; work past the budget raises ResourceLimitError, which leaves
a claim unverified, never verified. The checker uses no builtin pow and no ** at all.

The theorem that makes True imply the claim, for each operation:

sturm_sequence(polynomial) = (p0, ..., p(L-1)), evidence (q1, ..., q(L-1)).
    p is the request's polynomial, nonzero, of degree n, with canonical coefficients and a Symbol variable. The value is
    a tuple of L >= 1 canonical nonzero Polynomial records, each in p's variable itself (the same name, domain,
    assumptions and identity); the evidence is a tuple of L - 1 quotients, each a tuple of canonical Rationals lowest
    degree first without a trailing zero (an int is refused: the card's layout is Rationals). The checker verifies:
    p0 == p; when n = 0, L = 1 (p' = 0, so the sequence is (p,)); when n >= 1, L >= 2 and p1 == p', the derivative that
    the checker forms itself; deg p(i+1) < deg p(i) for 1 <= i <= L - 2; and, with p(L) = 0, for every 1 <= i <= L - 1
    the exact identity p(i-1) + p(i+1) == q(i) * p(i) over Q. Then p(i-1) = q(i) p(i) + (-p(i+1)) with
    deg(-p(i+1)) < deg p(i) (or -p(i+1) = 0 for i = L - 1), so by the uniqueness of polynomial division over Q, q(i) is
    the quotient and -p(i+1) the remainder of p(i-1) by p(i): p(i+1) = -rem(p(i-1), p(i)). Every listed member is
    nonzero and the remainder after the last member is zero, so the sequence runs exactly as long as the card's
    definition makes it: the value is the canonical Sturm sequence, and no other tuple passes (the evidence is then
    determined too). The lengths are tested before any arithmetic, as necessary conditions: len(p0) = n + 1,
    len(p1) = n, strictly decreasing member lengths, and len(q(i)) = deg p(i-1) - deg p(i) + 1, which a true claim
    meets (the degree of a quotient), so L <= n + 1 and the evidence is no longer than the value.
    The modular filter only ever refuses. Before the exact identities, which need common denominators, the checker
    evaluates every member and every quotient at the point POINT modulo the prime PRIME = 2**61 - 1, each rational
    polynomial as a fraction N / D of residues (Horner's rule on N/D * t + a/b = (N t b + a D) / (D b), with no
    inversion), and refuses a claim for which (N_a D_c + N_c D_a) D_q D_b - N_q N_b D_a D_c is not 0 modulo PRIME for
    some i: that integer is the exact identity at the integer POINT multiplied by the products of the denominators,
    so it vanishes when the identity holds. A True never rests on the filter.

cyclotomic_polynomial(n, variable) = G_n, evidence ((d, G_d), ...).
    n is an exact int with 1 <= n <= MAX_ORDER = 100,000 (the card's resource limit: a larger n raises
    ResourceLimitError, which leaves the claim unverified, as the producer refuses it); variable is a Symbol. The value
    is a Polynomial in that variable whose coefficients are Rationals with denominator 1; the evidence lists, for every
    proper divisor d of n in increasing order (the checker finds the divisors itself, by trial division), the pair
    (d, G_d) with G_d a tuple of exact ints, lowest degree first, without a trailing zero. The checker factors n into
    primes and, for every divisor d of n (n itself with G_n the value), verifies the identity in Z[x]
        G_d * prod_{s | d, mu(s) = -1} (x^(d/s) - 1) == prod_{s | d, mu(s) = +1} (x^(d/s) - 1),
    s running over the square-free divisors of d (mu the Moebius function), by one exact comparison at x = 2**w
    (Kronecker substitution): with |every coefficient of G_d| < 2**b and k- and k+ binomials on each side, every
    coefficient of the left side is below 2**(b + k-) in absolute value (each binomial at most doubles the largest
    one), every one of the right side at most 2**(k+), so with w = max(b + k-, k+) + 3 every coefficient of the
    difference is below 2**w - 1 in absolute value, and a polynomial with such coefficients that vanishes at 2**w is
    zero (its top term exceeds the sum of all lower ones). Z[x] has no zero divisors and the binomials are nonzero, so
    G_d = prod_{e | d} (x^e - 1)^mu(d/e) in Q(x), which is Phi_d: Moebius inversion of x^d - 1 = prod_{e | d} Phi_e in
    the multiplicative group of Q(x). In particular the card's identities x^d - 1 == prod_{e | d} G_e hold for every
    d | n (substitute and collect: the exponent of x^f - 1 is the sum of mu over the divisors of d/f, 1 for f = d and 0
    otherwise), and those identities determine every G_d by induction on d, which is the card's theorem: so
    G_n = Phi_n and the evidence lists Phi_d for every proper divisor d. Necessary conditions tested first, which
    every true claim meets: the evidence lists exactly the proper divisors; len(G_d) = phi(d) + 1 <= 1,025 (no
    Polynomial record holds a polynomial of a larger degree, so phi(n) > 1,024 answers False); every coefficient of
    G_d has at most phi(d) + 1 bits (Mignotte's bound for a divisor of degree phi(d) of x^d - 1, whose 2-norm is
    sqrt 2, gives |c| <= 2**phi(d) * sqrt 2).

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Assumption, Domain, Rational, Symbol, Truth
from mathforge.trusted import upoly as tp
from mathforge.trusted.integers import division_work, product_work

POLYNOMIAL = ("mathforge.polynomial", "Polynomial")  # Module and name of the record class read as data.
MAX_DEGREE = 1024  # polynomial.MAX_POLYNOMIAL_DEGREE, which a checker cannot import; a test compares them.
MAX_ORDER = 100_000  # The card's resource limit on n for cyclotomic_polynomial.
MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.
PRIME = (1 << 61) - 1  # The Mersenne prime of the modular filter of sturm_sequence.
PRIME_BITS = PRIME.bit_length()
POINT = 0x1D5B_9F3A_C6E2_4817 % PRIME  # The evaluation point of the filter; any residue serves.

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "sturm_sequence": ("polynomial",),
    "cyclotomic_polynomial": ("n", "variable"),
}
_MISSING = object()  # A field that a record forged with object.__new__ lacks.

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1.
IntPoly = tuple[int, ...]  # Integer coefficients, lowest degree first, no trailing zero; () is zero.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for arguments outside the operation's domain and for any malformed value or
    evidence; ResourceLimitError past the budget, for a request integer over max_integer_bits and for n > MAX_ORDER."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = _ARGUMENTS.get(operation)
    if (names is None or len(args) != len(names) or any(type(name) is not str for name in args)
            or any(name not in args for name in names)):
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        budget.tick()
        if operation == "sturm_sequence":
            return _sturm_sequence(budget, args["polynomial"], value, evidence)
        return _cyclotomic_polynomial(budget, args["n"], args["variable"], value, evidence)


# ---------------------------------------------------------------------------------------------------------------
# Reading the data

def _class(value: object) -> tuple[object, object]:
    kind = type(value)
    return getattr(kind, "__module__", None), getattr(kind, "__qualname__", None)


def _named(value: object, name: tuple[str, str]) -> bool:
    """Whether the value's class has this module and name, both exact strings (a class may set __module__ to any
    object, whose comparison could answer anything)."""
    module, qualname = _class(value)
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


def _field(record: object, name: str) -> object:
    """A field of a record read as data, _MISSING when the record itself does not hold it: a record with a __dict__ is
    read there and only there (getattr could answer a class-level attribute for a field that a record forged with
    object.__new__ left unset); a record without one keeps its fields in slots (Rational)."""
    try:
        fields = vars(record)
    except TypeError:  # No __dict__: slots.
        return getattr(record, name, _MISSING)
    if type(fields) is not dict:
        return _MISSING
    return fields.get(name, _MISSING)


def _symbol_key(symbol: object) -> tuple[object, ...] | None:
    """(name, domain, assumptions, uid) of a Symbol whose fields have their exact types: what Symbol equality compares,
    in a form whose comparison runs no foreign code. None for anything else, a forged Symbol whose fields break their
    types included; fields are read from the record itself (_field), never from the class defaults of Symbol, so a
    Symbol forged with object.__new__ that leaves its domain or assumptions unset is refused too."""
    if type(symbol) is not Symbol:
        return None
    name, domain = _field(symbol, "name"), _field(symbol, "domain")
    assumptions, uid = _field(symbol, "assumptions"), _field(symbol, "uid")
    if (type(name) is not str or type(domain) is not Domain or type(uid) is not str or type(assumptions) is not tuple
            or len(assumptions) > MAX_ASSUMPTIONS):
        return None
    facts = []
    for assumption in assumptions:  # At most MAX_ASSUMPTIONS items: no charge needed.
        if type(assumption) is not Assumption:
            return None
        fact = _field(assumption, "name"), _field(assumption, "truth")
        if type(fact[0]) is not str or type(fact[1]) is not Truth:
            return None
        facts.append(fact)
    return name, domain, tuple(facts), uid


def _coefficient_tuple(record: object, key: tuple[object, ...]) -> tuple[object, ...] | None:
    """The coefficients field of a Polynomial record in the variable of this key, unread: a tuple of 1 to
    MAX_DEGREE + 1 items. None for anything else."""
    if not _named(record, POLYNOMIAL) or _symbol_key(_field(record, "variable")) != key:
        return None
    coefficients = _field(record, "coefficients")
    if type(coefficients) is not tuple or not 1 <= len(coefficients) <= MAX_DEGREE + 1:
        return None
    return coefficients


def _pairs(budget: Budget, coefficients: tuple[object, ...]) -> list[Pair] | None:
    """The coefficients as (numerator, denominator) pairs when each is a Rational of exact ints within max_integer_bits
    with a positive denominator, and the last one is nonzero (no trailing zero, and not the zero polynomial); None
    otherwise. Lowest terms are tested later (_lowest_terms), after the cheap tests."""
    limit = budget.limits.max_integer_bits
    pairs = []
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not Rational:
            return None
        numerator, denominator = _field(coefficient, "numerator"), _field(coefficient, "denominator")
        if (type(numerator) is not int or type(denominator) is not int or denominator < 1
                or numerator.bit_length() > limit or denominator.bit_length() > limit):
            return None
        pairs.append((numerator, denominator))
    if not pairs or not pairs[-1][0]:
        return None
    return pairs


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: a division of the longer by the shorter, then Lehmer's
    algorithm, quadratic in the shorter, one unit per 64 squared 64-bit words as trusted.upoly.charge_gcd charges it."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + division_work(large, small) + (words * words >> 6)


def _lowest_terms(budget: Budget, pairs: list[Pair]) -> bool:
    """Whether every pair (with a positive denominator) is in lowest terms: 0 only as 0/1."""
    for numerator, denominator in pairs:
        budget.tick()
        if denominator == 1:
            continue
        budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
        if gcd(numerator, denominator) != 1:
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# sturm_sequence

def _sturm_sequence(budget: Budget, polynomial: object, value: object, evidence: object) -> bool:
    key = _symbol_key(_field(polynomial, "variable"))
    if key is None:
        return False
    coefficients = _coefficient_tuple(polynomial, key)
    if coefficients is None:
        return False
    base = _pairs(budget, coefficients)
    if base is None or not _lowest_terms(budget, base):
        return False  # Not a canonical nonzero polynomial: outside the domain, no value at all.
    degree = len(base) - 1
    if type(value) is not tuple or not 1 <= len(value) <= degree + 1:
        return False
    if type(evidence) is not tuple or len(evidence) != len(value) - 1:
        return False
    # The shapes and lengths of every member and quotient, before any number is read.
    members = []
    for member in value:
        budget.tick()
        items = _coefficient_tuple(member, key)
        if items is None:
            return False
        members.append(items)
    if len(members[0]) != degree + 1 or (degree and (len(members) < 2 or len(members[1]) != degree)):
        return False  # p0 has p's degree; p1 = p' has degree n - 1, and is there exactly when n >= 1.
    for index in range(2, len(members)):
        budget.tick()
        if len(members[index]) >= len(members[index - 1]):
            return False  # deg p(i+1) < deg p(i).
    quotients = []
    for index, quotient in enumerate(evidence, 1):
        budget.tick()
        if type(quotient) is not tuple or len(quotient) != len(members[index - 1]) - len(members[index]) + 1:
            return False  # deg q(i) = deg p(i-1) - deg p(i).
        quotients.append(quotient)
    # The numbers: types, lengths and no trailing zero, then p0 == p and p1 == p'.
    polynomials = []
    for items in members + quotients:
        pairs = _pairs(budget, items)
        if pairs is None:
            return False
        polynomials.append(pairs)
    sequence, divisions = polynomials[:len(members)], polynomials[len(members):]
    budget.tick(len(base))
    if sequence[0] != base:
        return False
    if degree and sequence[1] != _derivative(budget, base):
        return False
    # Every member and quotient as a fraction of residues at POINT; a wrong identity is refused here.
    values = [_residue_value(budget, pairs) for pairs in polynomials]
    residues, quotient_residues = values[:len(members)], values[len(members):]
    for index in range(1, len(sequence)):
        budget.tick()
        n_a, d_a = residues[index - 1]
        n_b, d_b = residues[index]
        n_c, d_c = residues[index + 1] if index + 1 < len(sequence) else (0, 1)
        n_q, d_q = quotient_residues[index - 1]
        if ((n_a * d_c + n_c * d_a) % PRIME * (d_q * d_b % PRIME) - n_q * n_b % PRIME * (d_a * d_c % PRIME)) % PRIME:
            return False
    # Lowest terms of the members after p0 and p1 (those two equal canonical pairs) and of the quotients.
    for pairs in sequence[2:] + divisions:
        if not _lowest_terms(budget, pairs):
            return False
    # The exact identities over Z: p(i-1) + p(i+1) == q(i) * p(i), denominators cleared.
    forms = [_integer_form(budget, pairs) for pairs in sequence]
    for index in range(1, len(sequence)):
        budget.tick()
        a, d_a = forms[index - 1]
        b, d_b = forms[index]
        c, d_c = forms[index + 1] if index + 1 < len(sequence) else ((), 1)
        q, d_q = _integer_form(budget, divisions[index - 1])
        product = tp.mul(q, b)  # Charged by trusted.upoly.
        factor = _times(budget, d_q, d_b)
        left = _add(budget, _scale(budget, a, _times(budget, d_c, factor)),
                    _scale(budget, c, _times(budget, d_a, factor)))
        right = _scale(budget, product, _times(budget, d_a, d_c))
        budget.tick(1 + len(left))
        if left != right:
            return False
    return True


def _derivative(budget: Budget, pairs: list[Pair]) -> list[Pair]:
    """The canonical pairs of the derivative of a polynomial of degree >= 1 with canonical pairs: k * a/b reduced is
    (k/g) * a over b/g with g = gcd(k, b), already in lowest terms as gcd(a, b) = 1 and gcd(k/g, b/g) = 1."""
    result = []
    for index in range(1, len(pairs)):
        numerator, denominator = pairs[index]
        bits = denominator.bit_length()
        budget.tick(2 + division_work(bits, index.bit_length()) + product_work(numerator.bit_length(), 11))
        divisor = gcd(index, denominator)
        result.append(((index // divisor) * numerator, denominator // divisor))
    return result


def _residue_value(budget: Budget, pairs: list[Pair]) -> tuple[int, int]:
    """(N, D) with N / D the polynomial's value at POINT modulo PRIME, by Horner's rule on fractions of residues."""
    total, denominator = 0, 1
    for numerator, bottom in reversed(pairs):
        budget.tick(1 + division_work(numerator.bit_length(), PRIME_BITS) + division_work(bottom.bit_length(),
                                                                                           PRIME_BITS))
        a, b = numerator % PRIME, bottom % PRIME
        total, denominator = (total * POINT % PRIME * b + a * denominator) % PRIME, denominator * b % PRIME
    return total, denominator


def _integer_form(budget: Budget, pairs: list[Pair]) -> tuple[IntPoly, int]:
    """(P, D) with the polynomial equal to P / D: D the least common multiple of the denominators, P integers."""
    common = 1
    for _, denominator in pairs:
        common_bits, bits = common.bit_length(), denominator.bit_length()
        budget.tick(1 + division_work(common_bits, bits))
        remainder = common % denominator
        if remainder:
            budget.tick(_gcd_work(bits, remainder.bit_length()) + product_work(common_bits, bits))
            common *= denominator // gcd(denominator, remainder)  # gcd(d, D % d) = gcd(d, D).
    common_bits = common.bit_length()
    integers = []
    for numerator, denominator in pairs:
        budget.tick(1 + division_work(common_bits, denominator.bit_length())
                    + product_work(numerator.bit_length(), common_bits))
        integers.append(numerator * (common // denominator))
    return tp.trim(integers), common


def _times(budget: Budget, left: int, right: int) -> int:
    budget.tick(1 + product_work(left.bit_length(), right.bit_length()))
    return left * right


def _scale(budget: Budget, polynomial: IntPoly, factor: int) -> IntPoly:
    """factor * polynomial, every product charged before it runs."""
    bits = factor.bit_length()
    result = []
    for coefficient in polynomial:
        budget.tick(1 + product_work(coefficient.bit_length(), bits))
        result.append(coefficient * factor)
    return tuple(result)


def _add(budget: Budget, left: IntPoly, right: IntPoly) -> IntPoly:
    """left + right without trailing zeros, each sum charged at the length of its operands."""
    if len(left) < len(right):
        left, right = right, left
    result = list(left)
    for index, coefficient in enumerate(right):
        budget.tick(1 + ((result[index].bit_length() + coefficient.bit_length()) >> 14))
        result[index] += coefficient
    return tp.trim(result)


# ---------------------------------------------------------------------------------------------------------------
# cyclotomic_polynomial

def _cyclotomic_polynomial(budget: Budget, n: object, variable: object, value: object, evidence: object) -> bool:
    if type(n) is not int:
        return False
    budget.check_int(n)  # A request integer over max_integer_bits raises, as the request's inspection does.
    if n < 1:
        return False  # Outside the domain: no value at all.
    if n > MAX_ORDER:
        raise ResourceLimitError("Computation limit exceeded: cyclotomic_polynomial takes n <= 100000")
    key = _symbol_key(variable)
    if key is None:
        return False
    coefficients = _coefficient_tuple(value, key)
    if coefficients is None:
        return False
    primes = _prime_factors(budget, n)
    divisors = _divisors(budget, n, primes)  # Increasing, n last.
    if type(evidence) is not tuple or len(evidence) != len(divisors) - 1:
        return False
    phis = [_phi(budget, d, primes) for d in divisors]
    if phis[-1] + 1 != len(coefficients):
        return False  # deg Phi_n = phi(n); a record holds at most MAX_DEGREE + 1 coefficients.
    # The value's integers: Rationals with denominator 1.
    integers = []
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not Rational:
            return False
        numerator, denominator = _field(coefficient, "numerator"), _field(coefficient, "denominator")
        if type(numerator) is not int or type(denominator) is not int or denominator != 1:
            return False
        integers.append(numerator)
    polynomials = []
    for index, entry in enumerate(evidence):
        budget.tick()
        if type(entry) is not tuple or len(entry) != 2:
            return False
        d, items = entry
        if type(d) is not int or d != divisors[index] or type(items) is not tuple or len(items) != phis[index] + 1:
            return False
        polynomials.append(items)
    polynomials.append(tuple(integers))
    for index, items in enumerate(polynomials):
        bound = phis[index] + 1  # Mignotte's bound, in bits.
        for item in items:
            budget.tick()
            if type(item) is not int or item.bit_length() > bound:
                return False
        if not items[-1]:
            return False  # A trailing zero.
    for index, d in enumerate(divisors):
        if not _moebius_identity(budget, d, [p for p in primes if not d % p], polynomials[index]):
            return False
    return True


def _prime_factors(budget: Budget, n: int) -> list[int]:
    """The distinct primes of 1 <= n <= MAX_ORDER, increasing, by trial division."""
    primes, rest, candidate = [], n, 2
    while candidate * candidate <= rest:
        budget.tick()
        if not rest % candidate:
            primes.append(candidate)
            while not rest % candidate:
                budget.tick()
                rest //= candidate
        candidate += 1
    if rest > 1:
        primes.append(rest)
    return primes


def _divisors(budget: Budget, n: int, primes: list[int]) -> list[int]:
    """Every divisor of n, increasing (n last), from its distinct primes: the products of their powers that divide n."""
    divisors = [1]
    for p in primes:
        powers, power = [], p
        while not n % power:
            budget.tick()
            powers.append(power)
            power *= p
        budget.tick(len(divisors) * len(powers))
        divisors += [d * q for d in divisors for q in powers]
    budget.tick(len(divisors))
    return sorted(divisors)


def _phi(budget: Budget, d: int, primes: list[int]) -> int:
    """Euler's phi of a divisor d of n, from the primes of n that divide d."""
    result = d
    for p in primes:
        budget.tick()
        if not d % p:
            result -= result // p
    return result


def _moebius_identity(budget: Budget, d: int, primes: list[int], coefficients: tuple[int, ...]) -> bool:
    """Whether G * prod_{mu(s) = -1} (x^(d/s) - 1) == prod_{mu(s) = +1} (x^(d/s) - 1), s over the square-free divisors
    of d, by one comparison at x = 2**w (module docstring)."""
    minus, plus = [], []
    subsets = [(1, 0)]  # (product of chosen primes, how many).
    for p in primes:
        budget.tick(len(subsets))
        subsets += [(s * p, k + 1) for s, k in subsets]
    for s, k in subsets:
        (minus if k & 1 else plus).append(d // s)
    budget.tick(1 + len(coefficients))
    bits = max(abs(item).bit_length() for item in coefficients)
    width = max(bits + len(minus), len(plus)) + 3
    left = tp.kronecker(coefficients, width)  # Charged by trusted.upoly.
    length = len(coefficients) * width
    for e in minus:
        length += e * width
        budget.tick(1 + (length >> 12))  # A shift and a subtraction, linear in the length.
        left = (left << (e * width)) - left
    right, length = 1, 1
    for e in plus:
        length += e * width
        budget.tick(1 + (length >> 12))
        right = (right << (e * width)) - right
    budget.tick(1 + (length >> 12))
    return left == right
