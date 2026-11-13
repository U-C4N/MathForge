"""Independent checker of the family mathforge.areas.algebra.polynomial_tools: interpolate.

The checker never runs the producer's divided differences. It evaluates the claimed polynomial at every node, exactly,
with integers only, which is the card's uniqueness test.

The theorem that makes True imply the claim:

interpolate(xs, ys, variable) = P, evidence ().
    The request holds n >= 1 abscissas x_i = p_i/q_i and as many ordinates y_i = u_i/v_i, exact ints or canonical
    Rationals (lowest terms, positive denominator), and the abscissas are pairwise distinct: the reduced pairs
    (p_i, q_i) differ, so the numbers do. P is a mathforge.polynomial.Polynomial record, read as data, whose
    coefficients c_0, ..., c_d (constant first) are canonical Rationals c_k = n_k/d_k without a trailing zero
    ((Rational(0),) for zero), with d + 1 <= n coefficients, so deg P < n, and whose variable is the request's Symbol
    (the same name, domain, assumptions and identity). With D the least common multiple of the d_k, the integers
    a_k = n_k * (D / d_k) and, at each node, S_i = sum_k a_k * p_i**k * q_i**(d - k) (Horner's rule on the homogenized
    form), the checker verifies v_i * S_i == u_i * D * q_i**d. Since P(x_i) = S_i / (D * q_i**d), that is P(x_i) = y_i.
    Uniqueness of interpolation: another polynomial Q of degree < n through the n points would make P - Q a polynomial
    of degree < n with n distinct roots, hence zero. So P is the only polynomial of degree < n through the points,
    and the polynomial of least degree through them: one of degree <= deg P < n through them is P itself. Its
    canonical record is unique, and it is the one claimed. A request with no point, lists of different lengths or a
    repeated abscissa has no value, and every claim on it answers False.

The modular filter only ever refuses. Before anything quadratic in the length of an integer runs (a gcd, the common
denominator D, the exact evaluation), the checker reduces modulo the prime PRIME = 2**61 - 1 the integer identity
    F_i = v_i * sum_k n_k * (Pi / d_k) * p_i**k * q_i**(d - k) - u_i * Pi * q_i**d = Pi * q_i**d * v_i * (P(x_i) - y_i),
with Pi the product of every d_k, of which only residues are formed (Pi / d_k, the product of the other d_j, from prefix
and suffix products). The identity holds for any nonzero denominators, reduced or not, and a true claim has F_i = 0,
so a nonzero residue at any node proves the claim wrong. A wrong claim is thus refused after word-sized operations
and one division per long integer, even when its coefficients have huge pairwise coprime denominators. A True never
rests on the filter: a claim that agrees modulo PRIME at every node still has to pass the exact evaluation. A wrong
claim crafted to agree modulo PRIME, with long pairwise coprime denominators, needs a D longer than the budget allows
and ends unverified (ResourceLimitError) within the budget's time, never verified: no fixed modulus refuses every such
claim, and the exact identity is the only test left once P(x_i) - y_i is nonzero but agrees with 0 modulo PRIME.

Shapes and budget. The operation must be the string "interpolate", args a dict of exactly xs, ys and variable, xs
and ys tuples (a request holds lists frozen into tuples), the value of the class named POLYNOMIAL and the evidence
the empty tuple; a bool, a float, a str, an int subclass or a Rational subclass is no exact number, a Polynomial has
Rational coefficients only, and anything else answers False, never an exception. A value integer over
max_integer_bits answers False, as no computation under the budget forms it. A request integer over max_integer_bits
raises ResourceLimitError, as operations_v6.run and the verifier's inspection raise before a checker runs, and so
does work past max_work: a claim is then unverified, never verified. Every loop is charged before it runs, a long
integer operation on the scale of the cost model of trusted.integers (product_work for a product, division_work for
the length of a quotient, a gcd quadratic in its shorter operand as trusted.upoly.charge_gcd charges it, one unit per
256 words of a linear pass), and no builtin pow is used: the powers of q_i build up one product at a time inside
Horner's rule. Under the default limits a claim of the record's full size, 1,025 integer nodes and degree 1,024,
verifies in about 1.6 million of the 2 million units with 20-bit integer coefficients. Long rational coefficients cost
more: random one-digit ordinates at 600 consecutive integers (coefficients of about 4,600 bits) take about 620,000
units, and at 1,025 of them (about 8,800 bits) the default max_work runs out, so that the claim stays unverified.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.limits import Budget, computation
from mathforge.model import Assumption, Domain, Rational, Symbol, Truth
from mathforge.trusted.integers import division_work, product_work

OPERATION = "interpolate"
ARGUMENTS = frozenset(("xs", "ys", "variable"))  # The argument names of the spec.
POLYNOMIAL = ("mathforge.polynomial", "Polynomial")  # Module and name of the value's class.
MAX_LENGTH = 1025  # A Polynomial record holds at most MAX_POLYNOMIAL_DEGREE + 1 = 1,025 coefficients.
MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.
PRIME = (1 << 61) - 1  # The Mersenne prime of the filter.
PRIME_BITS = PRIME.bit_length()

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorem of the module.
    False for an unknown operation, malformed arguments, a malformed or non-canonical value and any evidence but ();
    ResourceLimitError for a request integer over max_integer_bits and for work past the budget."""
    if type(operation) is not str or operation != OPERATION:
        return False
    if (type(args) is not dict or len(args) != len(ARGUMENTS) or any(type(name) is not str for name in args)
            or not ARGUMENTS.issuperset(args)):
        return False
    xs, ys, variable = args["xs"], args["ys"], args["variable"]
    if type(xs) is not tuple or type(ys) is not tuple or not xs or len(xs) != len(ys):
        return False
    if type(evidence) is not tuple or evidence:
        return False
    # The value's shape, each field read once: a forged record may lack a field, which reads as None.
    if not _named(value, POLYNOMIAL):
        return False
    coefficients = getattr(value, "coefficients", None)
    if type(coefficients) is not tuple or not 1 <= len(coefficients) <= min(len(xs), MAX_LENGTH):
        return False  # The degree is below the number of points, and within the record's bound.
    key = _symbol_key(variable)
    if key is None or _symbol_key(getattr(value, "variable", None)) != key:
        return False
    with computation() as budget:  # One budget for every step: the active one, or one of the default limits.
        # Linear steps first: shapes, types and lengths of every number, the value's (at most MAX_LENGTH) first.
        pairs = _coefficients(budget, coefficients)
        if pairs is None:
            return False
        nodes = _request_pairs(budget, xs)
        values = None if nodes is None else _request_pairs(budget, ys)  # Malformed xs: ys need no reading.
        if values is None:
            return False
        if not _filter(budget, pairs, nodes, values):
            return False
        # Lowest terms (a gcd per number), then distinct abscissas and the exact evaluation.
        if not (_lowest_terms(budget, pairs) and _lowest_terms(budget, nodes) and _lowest_terms(budget, values)):
            return False
        return _distinct(budget, nodes) and _exact(budget, pairs, nodes, values)


# ---------------------------------------------------------------------------------------------------------------
# Shapes

def _named(value: object, name: tuple[str, str]) -> bool:
    """Whether the value's class has this module and name, both exact strings (a class may set __module__ to any
    object, whose comparison could answer anything)."""
    cls = type(value)
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


def _symbol_key(symbol: object) -> tuple[object, ...] | None:
    """(name, domain, assumptions, uid) of a Symbol whose fields have their exact types: what Symbol equality compares,
    in a form whose comparison runs no foreign code. None for anything else, a forged Symbol included."""
    if type(symbol) is not Symbol:
        return None
    name, domain = getattr(symbol, "name", None), getattr(symbol, "domain", None)
    assumptions, uid = getattr(symbol, "assumptions", None), getattr(symbol, "uid", None)
    if (type(name) is not str or type(domain) is not Domain or type(uid) is not str or type(assumptions) is not tuple
            or len(assumptions) > MAX_ASSUMPTIONS):
        return None
    facts = []
    for assumption in assumptions:  # At most MAX_ASSUMPTIONS items: no charge needed.
        if type(assumption) is not Assumption:
            return None
        fact = getattr(assumption, "name", None), getattr(assumption, "truth", None)
        if type(fact[0]) is not str or type(fact[1]) is not Truth:
            return None
        facts.append(fact)
    return name, domain, tuple(facts), uid


def _coefficients(budget: Budget, coefficients: tuple[object, ...]) -> list[Pair] | None:
    """The coefficients as pairs when each is a Rational of exact ints within max_integer_bits with a positive
    denominator and the last of several is not zero; None otherwise. A Polynomial record never holds an int: its
    constructor makes every coefficient a Rational and trims trailing zeros. Lowest terms are checked later."""
    limit = budget.limits.max_integer_bits
    pairs = []
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not Rational:
            return None
        numerator = getattr(coefficient, "numerator", None)  # A forged Rational may lack a slot.
        denominator = getattr(coefficient, "denominator", None)
        if (type(numerator) is not int or type(denominator) is not int or denominator < 1
                or numerator.bit_length() > limit or denominator.bit_length() > limit):
            return None
        pairs.append((numerator, denominator))
    if len(pairs) > 1 and pairs[-1][0] == 0:
        return None
    return pairs


def _request_pairs(budget: Budget, numbers: tuple[object, ...]) -> list[Pair] | None:
    """The exact rationals of xs or ys as pairs: each an int, or a Rational of exact ints with a positive denominator.
    None for anything else. An integer over max_integer_bits raises ResourceLimitError, as the request's inspection
    does. Lowest terms are checked later."""
    pairs = []
    for number in numbers:
        budget.tick()
        kind = type(number)
        if kind is int:
            budget.check_int(number)
            pairs.append((number, 1))
            continue
        if kind is not Rational:
            return None
        numerator = getattr(number, "numerator", None)
        denominator = getattr(number, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int:
            return None
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator < 1:
            return None
        pairs.append((numerator, denominator))
    return pairs


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient (division_work charges the words of the
    dividend, right for a remainder after a product; here the quotient may be far shorter)."""
    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: one, a division of the longer by the shorter with a pass
    over the longer, then Lehmer's algorithm on the shorter, quadratic, one unit per 64 squared 64-bit words as
    trusted.upoly.charge_gcd charges it (16,384 at 65,536 bits; measured about 1.8 ms at 60,000 bits)."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + _quotient_work(large, small) + (large >> 12) + (words * words >> 6)


def _lowest_terms(budget: Budget, pairs: list[Pair]) -> bool:
    """Whether every pair (with a positive denominator) is in lowest terms: 0 only as 0/1."""
    for numerator, denominator in pairs:
        if denominator == 1:
            budget.tick()
            continue
        budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
        if gcd(numerator, denominator) != 1:
            return False
    return True


def _distinct(budget: Budget, nodes: list[Pair]) -> bool:
    """Whether the abscissas are pairwise distinct: lowest terms make equal numbers equal pairs."""
    seen: set[Pair] = set()
    for node in nodes:
        budget.tick(1 + ((node[0].bit_length() + node[1].bit_length()) >> 14))  # Hashing reads both integers.
        if node in seen:
            return False
        seen.add(node)
    return True


# ---------------------------------------------------------------------------------------------------------------
# The modular filter: it refuses wrong claims cheaply and never accepts anything

def _residue(budget: Budget, number: int) -> int:
    """number modulo PRIME, its long division charged before it runs."""
    budget.tick(1 + division_work(number.bit_length(), PRIME_BITS))
    return number % PRIME


def _filter(budget: Budget, pairs: list[Pair], nodes: list[Pair], values: list[Pair]) -> bool:
    """Whether F_i (module docstring) vanishes modulo PRIME at every node; False proves the claim wrong."""
    length = len(pairs)
    numerators = [_residue(budget, numerator) for numerator, _ in pairs]
    denominators = [_residue(budget, denominator) for _, denominator in pairs]
    budget.tick(1 + 3 * length)  # The prefix products, then the suffix products and the reduced coefficients.
    prefix = [1] * (length + 1)
    for k, denominator in enumerate(denominators):
        prefix[k + 1] = prefix[k] * denominator % PRIME
    reduced = [0] * length  # n_k * (Pi / d_k) modulo PRIME.
    suffix = 1
    for k in range(length - 1, -1, -1):
        reduced[k] = numerators[k] * (prefix[k] * suffix % PRIME) % PRIME
        suffix = suffix * denominators[k] % PRIME
    product = prefix[length]  # Pi modulo PRIME.
    top = length - 1
    for (p, q), (u, v) in zip(nodes, values):
        p, q, u, v = (_residue(budget, number) for number in (p, q, u, v))
        budget.tick(1 + (top >> 1))  # Word-sized steps, two per unit.
        total, power = reduced[top], 1
        for k in range(top - 1, -1, -1):
            power = power * q % PRIME
            total = (total * p + reduced[k] * power) % PRIME
        if (v * total - u * product % PRIME * power) % PRIME:  # power is q**top modulo PRIME.
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# The exact evaluation

def _common_denominator(budget: Budget, pairs: list[Pair]) -> int:
    """D, the least common multiple of the denominators: D % d == 0 leaves D, else D * (d / gcd(d, D % d))."""
    common = 1
    for _, denominator in pairs:
        common_bits, bits = common.bit_length(), denominator.bit_length()
        budget.tick(1 + _quotient_work(common_bits, bits))
        remainder = common % denominator
        if remainder:
            budget.tick(_gcd_work(bits, remainder.bit_length()))
            divisor = gcd(denominator, remainder)  # gcd(d, D), at most d.
            divisor_bits = divisor.bit_length()
            factor_bits = bits - divisor_bits + 1
            budget.tick(1 + _quotient_work(bits, divisor_bits) + product_work(common_bits, factor_bits)
                        + ((common_bits + factor_bits) >> 14))
            common *= denominator // divisor
    return common


def _exact(budget: Budget, pairs: list[Pair], nodes: list[Pair], values: list[Pair]) -> bool:
    """Whether v_i * S_i == u_i * D * q_i**d at every node (module docstring), S_i by Horner's rule."""
    common = _common_denominator(budget, pairs)
    common_bits = common.bit_length()
    integers = []
    for numerator, denominator in pairs:
        bits = denominator.bit_length()
        budget.tick(1 + _quotient_work(common_bits, bits) + product_work(numerator.bit_length(), common_bits - bits + 1)
                    + (common_bits >> 14))
        integers.append(numerator * (common // denominator))
    budget.tick(len(integers))
    lengths = [integer.bit_length() for integer in integers]
    top = len(integers) - 1
    for (p, q), (u, v) in zip(nodes, values):
        budget.tick()
        p_bits, q_bits = p.bit_length(), q.bit_length()
        total, power = integers[top], 1
        if q == 1:  # An integer abscissa: no powers of q to form.
            for k in range(top - 1, -1, -1):
                total_bits = total.bit_length()
                budget.tick(1 + ((total_bits + lengths[k]) >> 14) + product_work(total_bits, p_bits))
                total = total * p + integers[k]
        else:
            for k in range(top - 1, -1, -1):
                total_bits, power_bits = total.bit_length(), power.bit_length() + q_bits
                budget.tick(1 + ((total_bits + lengths[k] + power_bits) >> 14) + product_work(total_bits, p_bits)
                            + product_work(power_bits - q_bits, q_bits) + product_work(lengths[k], power_bits))
                power *= q
                total = total * p + integers[k] * power
        # total = sum_k a_k * p**k * q**(top - k) and power = q**top.
        total_bits, power_bits, u_bits = total.bit_length(), power.bit_length(), u.bit_length()
        budget.tick(1 + product_work(v.bit_length(), total_bits) + product_work(u_bits, common_bits)
                    + product_work(u_bits + common_bits, power_bits) + ((total_bits + common_bits + power_bits) >> 13))
        if v * total != u * common * power:
            return False
    return True
