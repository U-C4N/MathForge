"""Independent checker of the family mathforge.areas.ntheory.arithmetic_functions: moebius, divisor_sigma and
integer_nth_root.

Every value of the family is a plain int, tested with ``type(value) is int``: a bool, a float, a Rational, an int
subclass or a record of any class is refused, so no record class is named here. The arguments are tested again, as
exact ints inside the domain of their operation (moebius: n >= 1; divisor_sigma: n >= 1 and k >= 0;
integer_nth_root: n >= 0 and k >= 1). A claim outside that domain is False even where its arithmetic would hold:
0 % (2*2) == 0 does not make moebius(0) = 0 a true claim, and 0**-1 is never formed.

moebius(n) = mu(n), with one of two evidences:

* ``("square", s)``: the value is 0, s is an int > 1 and s*s divides n. A prime p that divides s has p*p | s*s | n, so
  n is not squarefree and mu(n) = 0; no primality is needed. s*s is formed only when 2*(bits(s) - 1) < bits(n): else
  s*s >= 2**bits(n) > n >= 1, which s*s cannot divide.
* ``("factorization", factors)``: check_factorization(n, factors) (certify.ntheory.factorization) proves that factors
  is the prime factorization of n, which the fundamental theorem of arithmetic makes unique. When every exponent is
  1, n is the product of r = len(factors) distinct primes and mu(n) = (-1)**r; mu(1) = 1 with the empty
  factorization. A factorization with an exponent above 1 is refused: a zero value takes the square evidence. r
  distinct primes multiply to at least 2**r, so r < bits(n), and a longer tuple is refused before it is read.

divisor_sigma(n, k) = sigma_k(n), the sum of d**k over the positive divisors d of n, with evidence ``(factors,)``.
check_factorization(n, factors) proves n = prod p**e over distinct primes; sigma_k is multiplicative and
sigma_k(p**e) = 1 + p**k + ... + p**(k*e), so sigma_k(n) is the product of those sums, which this checker evaluates by
Horner's rule (s = 1, then e times s = s * p**k + 1), not by the producer's closed quotient. Before the factorization
is read, the length of the value is screened: n**k <= sigma_k(n) <= tau(n) * n**k <= n**(k + 1), so a true value has
between k*(bits(n) - 1) + 1 and (k + 1)*bits(n) bits, and any other value is False at once. A k with
k*bits(n) > max_integer_bits, the producer's documented cap, raises ResourceLimitError once value and evidence have
passed: p**k may then have more bits than the budget admits.

integer_nth_root(n, k) = floor(n**(1/k)), with evidence ``()``. x -> x**k is strictly increasing on x >= 0, so the
value r is the only int >= 0 with r**k <= n < (r + 1)**k, which is what is checked. 0 and 1 are their own powers, and
a power of x >= 2 is formed only after a bit-length screen: x >= 2**(bits(x) - 1), so (bits(x) - 1)*k >= bits(n)
proves x**k >= 2**bits(n) > n without computing it (r is then too large, and r + 1 large enough). A power that is
computed has x >= 2 and (bits(x) - 1)*k < bits(n), so k < bits(n) and x**k < 2**(bits(x)*k) <= 2**(2*bits(n)).

Work. The whole check runs in one budget (the active one, or one of the default limits). The integers of the request
must fit max_integer_bits, else ResourceLimitError, as operations_v6.run and the verifier's inspection raise before a
checker runs. Every loop charges the budget, and every power is charged before the builtin computes it: charged_pow
of certify.ntheory.factorization takes a modulus, and these powers are exact, so _power charges them on the same
scale (a product of the result's length per bit of the exponent). Malformed evidence and values answer False and
never raise; ResourceLimitError (never a verdict) is left to the request's size, the divisor_sigma cap, and work past
max_work, which check_factorization also raises for expensive certificates.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer. tools/check_layering.py follows every import, and tests/v08/test_layering.py rejects every
way around it.
"""

from __future__ import annotations

from mathforge.limits import Budget, ResourceLimitError, computation

from .factorization import check_factorization

# The argument names of each operation, as its spec in the family's SPECS declares them (all of kind "int").
_ARGUMENTS = {"moebius": ("n",), "divisor_sigma": ("n", "k"), "integer_nth_root": ("n", "k")}
SQUARE, FACTORIZATION = "square", "factorization"  # The two tags of the evidence of moebius.


def _product_work(left_bits: int, right_bits: int) -> int:
    """The work units of a product of integers of the given bit lengths: 1, and one per 256 pairs of their 64-bit
    words, the scale of trusted.upoly.charge_product and certify.ntheory.factorization."""
    return 1 + ((left_bits >> 6) * (right_bits >> 6) >> 8)


def _power(budget: Budget, base: int, exponent: int) -> int:
    """base**exponent for ints base >= 2 and exponent >= 1, charged before the builtin runs: base**exponent is below
    2**(bits(base) * exponent), and binary exponentiation makes at most two products of at most that length per bit
    of the exponent. The callers bound bits(base) * exponent first."""
    bits = base.bit_length() * exponent
    budget.tick(2 * exponent.bit_length() * _product_work(bits, bits))
    return base ** exponent


def _multiply(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it is formed."""
    budget.tick(_product_work(left.bit_length(), right.bit_length()))
    return left * right


def _moebius(budget: Budget, n: int, value: object, evidence: object) -> bool:
    if n < 1 or type(value) is not int or value not in (-1, 0, 1):
        return False
    if type(evidence) is not tuple or len(evidence) != 2 or type(evidence[0]) is not str:
        return False
    kind, data = evidence
    if kind == SQUARE:
        if value != 0 or type(data) is not int or data < 2:
            return False
        if 2 * (data.bit_length() - 1) >= n.bit_length():  # s*s >= 2**bits(n) > n >= 1, so s*s does not divide n.
            return False
        square = _multiply(budget, data, data)
        budget.tick(_product_work(n.bit_length(), square.bit_length()))  # The long division of n by s*s.
        return n % square == 0
    # r distinct primes multiply to at least 2**r, so a factorization of n has fewer than bits(n) of them.
    if kind != FACTORIZATION or value == 0 or type(data) is not tuple or len(data) >= n.bit_length():
        return False
    for item in data:  # Exponent 1 everywhere, before any prime proof runs.
        budget.tick()
        if type(item) is not tuple or len(item) != 3 or type(item[1]) is not int or item[1] != 1:
            return False
    if not check_factorization(n, data):
        return False
    return value == (1 if len(data) % 2 == 0 else -1)


def _divisor_sigma(budget: Budget, n: int, k: int, value: object, evidence: object) -> bool:
    if n < 1 or k < 0 or type(value) is not int or value < 1:
        return False
    bits = n.bit_length()
    # n**k <= sigma_k(n) <= n**(k + 1), and 2**((bits - 1)*k) <= n**k: the window of a true value's length.
    if not k * (bits - 1) + 1 <= value.bit_length() <= (k + 1) * bits:
        return False
    if type(evidence) is not tuple or len(evidence) != 1:
        return False
    factors = evidence[0]
    if not check_factorization(n, factors):
        return False
    if k * bits > budget.limits.max_integer_bits:
        raise ResourceLimitError("Computation limit exceeded: divisor_sigma takes k * bits(n) <= max_integer_bits")
    total = 1
    for prime, exponent, _ in factors:
        budget.tick()
        step = _power(budget, prime, k) if k else 1  # p**k < 2**(k * bits(p)) <= 2**(k * bits(n)): within the cap.
        part = 1
        for _ in range(exponent):  # Horner: 1 + p**k + ... + p**(k*e).
            part = _multiply(budget, part, step) + 1
        total = _multiply(budget, total, part)
    return total == value


def _integer_nth_root(budget: Budget, n: int, k: int, value: object, evidence: object) -> bool:
    if n < 0 or k < 1 or type(value) is not int or value < 0 or type(evidence) is not tuple or len(evidence) != 0:
        return False
    bits = n.bit_length()
    # value**k <= n.
    if value >= 2:
        if (value.bit_length() - 1) * k >= bits:  # value**k >= 2**((bits(value) - 1)*k) >= 2**bits(n) > n.
            return False
        if _power(budget, value, k) > n:
            return False
    elif value > n:  # 0 and 1 are their own powers: here value = 1 and n = 0.
        return False
    # n < (value + 1)**k.
    above = value + 1
    if (above.bit_length() - 1) * k >= bits:  # (value + 1)**k >= 2**bits(n) > n, uncomputed.
        return True
    if above == 1:  # value = 0 with n >= 1, where 1**k = 1 <= n.
        return False
    return _power(budget, above, k) > n


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether the evidence proves that value is the answer of operation to args; the module states the theorems."""
    if type(operation) is not str or operation not in _ARGUMENTS or type(args) is not dict:
        return False
    names = _ARGUMENTS[operation]
    if len(args) != len(names) or any(name not in args or type(args[name]) is not int for name in names):
        return False
    with computation() as budget:
        for name in names:
            budget.check_int(args[name])  # A request integer over max_integer_bits: ResourceLimitError, as in run().
        if operation == "moebius":
            return _moebius(budget, args["n"], value, evidence)
        if operation == "divisor_sigma":
            return _divisor_sigma(budget, args["n"], args["k"], value, evidence)
        return _integer_nth_root(budget, args["n"], args["k"], value, evidence)
