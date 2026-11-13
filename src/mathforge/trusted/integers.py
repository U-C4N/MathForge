"""Trusted integer arithmetic shared by the integer kernel and the prime-certificate checker, and the cost model of the
prime proofs and certified factorizations of mathforge.certify.ntheory.factorization.

The checker charges its budget with the functions below, and the producers of mathforge.kernels.certificates, which
must not import the checker, compute with the same functions what the checker will charge for the evidence they
build, so that both sides run out of a budget at the same point.

Work units. A multiplication modulo an integer of w 64-bit words is 1 + w*w/32 units (multiplication_work), on the
scale of trusted.upoly.size_factor, where a plain product of that size is 1 + w*w/256 (product_work): the reduction
is a long division (division_work), which stays quadratic where the product is Karatsuba's. An exponentiation is one
such multiplication per bit of its exponent. Measured with CPython on one machine, the default max_work (2,000,000)
then pays for about 0.2 seconds of exponentiations modulo 64-bit integers and 0.3 to 0.5 seconds at every modulus size
from 256 to 65,536 bits; less below 64 bits, and less with small bases.
"""

from math import prod

from ..limits import ComputationLimits
from .primes import DETERMINISTIC_BELOW

TRIAL_BELOW = 10_000  # prime_proof's trial criterion is exact below it; from it on, strong tests or a certificate.
_STRONG_TESTS = 13  # trusted.primes.is_prime runs one exponentiation modulo n for each of its thirteen bases.
_MAX_DEPTH = ComputationLimits().max_depth  # certify.ntheory.primes.MAX_DEPTH: check_prime refuses deeper nodes.


def powers_by_prime(base, exponent, modulus, primes):
    """{p: pow(base, exponent // p, modulus) for each p of primes}, for distinct primes that all divide exponent; {}
    for no primes.

    One exponentiation by exponent // prod(primes), then a product-tree descent: a group's power raised to the
    product of the other half of the group is the power of that half. For k primes this takes about log2(k)
    exponentiations by the product of the primes in all, rather than k exponentiations by exponent // p."""
    primes = tuple(primes)
    if not primes:  # An empty group would split into two empty groups for ever.
        return {}
    result = {}
    pending = [(pow(base, exponent // prod(primes), modulus), primes)]
    while pending:
        value, group = pending.pop()
        if len(group) == 1:
            result[group[0]] = value
            continue
        half = len(group) // 2
        left, right = group[:half], group[half:]
        pending.append((pow(value, prod(right), modulus), left))
        pending.append((pow(value, prod(left), modulus), right))
    return result


def multiplication_work(bits: int) -> int:
    """The work units of one multiplication modulo an integer of the given bit length: 1, and one more per 32 pairs of
    its 64-bit words (1 up to 383 bits, 9 at 1,024 bits, 129 at 4,096 and 32,769 at 65,536)."""
    words = bits >> 6
    return 1 + (words * words >> 5)


def product_work(left_bits: int, right_bits: int) -> int:
    """What a product of integers of the given bit lengths costs beyond one unit: one unit per 256 pairs of their
    64-bit words, as trusted.upoly.charge_product charges it."""
    return (left_bits >> 6) * (right_bits >> 6) >> 8


def division_work(dividend_bits: int, divisor_bits: int) -> int:
    """What a long division of an integer of dividend_bits bits by one of divisor_bits bits costs beyond one unit: one
    unit per 32 pairs of their 64-bit words, the divisor counted one word longer, on multiplication_work's scale."""
    return (dividend_bits >> 6) * (1 + (divisor_bits >> 6)) >> 5


def prime_proof_work(n, certificate, limit: int) -> int:
    """The work that certify.ntheory.factorization.prime_proof(n, certificate) charges before it proves anything, when
    that is at most limit >= 0; otherwise some number above limit, reached without walking the rest of the
    certificate, so that a certificate whose subtrees are shared many times over costs at most about limit steps.

    0 where prime_proof needs no budget: for n below TRIAL_BELOW, and for every argument that it refuses by its type
    or shape alone. For 10,000 <= n < psi_13 (trusted.primes.DETERMINISTIC_BELOW) and the certificate (): the strong
    tests of trusted.primes.is_prime, a multiplication per bit of n for each of its thirteen bases (one unit each,
    since n has at most 82 bits). From psi_13 on, for a certificate ("pocklington", ...): one unit per node of the
    certificate and, for each Pocklington node, the exponentiations and gcds that certify.ntheory.primes.check_prime
    can run on it.

    check_prime works on a node only when the node is a Pocklington certificate of the very number it has to prove,
    at most MAX_DEPTH levels down, and only once the p of the node are distinct primes whose powers multiply to that
    number minus 1. For such a node of a b-bit number with k factors and g distinct witnesses it runs, modulo that
    number, one exponentiation per witness (at most b bits of exponent each), the product trees of powers_by_prime
    (ceil(log2 k) levels, at most b + g + k bits of exponents a level, since the products of the groups multiply to a
    divisor of the number minus 1), one exponentiation by a p per witness (b + g bits in all) and k gcds. Every other
    node is refused, or accepted ("trial"), without an exponentiation; walking it costs its unit, and its children are
    still walked, since check_prime reads them in order and may reach some of them before it refuses the node.
    """
    if type(n) is not int or type(certificate) is not tuple or n < TRIAL_BELOW:
        return 0
    if n < DETERMINISTIC_BELOW:
        return 0 if certificate else _STRONG_TESTS * n.bit_length()
    if len(certificate) != 4 or certificate[0] != "pocklington":
        return 0
    total, pending = 0, [(n, certificate, 0)]
    while pending and total <= limit:
        number, node, depth = pending.pop()
        total += 1
        if (depth > _MAX_DEPTH or type(node) is not tuple or len(node) != 4 or node[0] != "pocklington"
                or type(node[1]) is not int or node[1] != number or number < 2):
            continue
        factors, witnesses = node[2], node[3]
        if type(factors) is not tuple or type(witnesses) is not tuple or len(factors) != len(witnesses):
            continue
        bits, count = number.bit_length(), len(factors)
        if 0 < count <= bits:  # More distinct primes than bits cannot multiply to the number minus 1.
            groups = len({witness for witness in witnesses if type(witness) is int})
            levels = (count - 1).bit_length()
            exponent_bits = bits * (groups + levels + 1) + (levels + 1) * (groups + count)
            words = bits >> 6
            total += exponent_bits * multiplication_work(bits) + count * (1 + (words * words >> 6))
        for factor in factors:
            if type(factor) is tuple and len(factor) == 3 and type(factor[0]) is int:
                pending.append((factor[0], factor[2], depth + 1))
    return total


def factorization_entry_work(product_bits: int, power_bits: int) -> int:
    """What certify.ntheory.factorization.check_factorization charges for one entry (p, e) of a factorization, before
    it builds p**e: one unit, the power, of at most power_bits = bit_length(p) * e bits (about its last squaring), and
    its product with the product of the entries before it, of product_bits bits."""
    return 1 + product_work(power_bits >> 1, power_bits >> 1) + product_work(product_bits, power_bits)


def factorization_work(factors, limit: int) -> int:
    """The work that certify.ntheory.factorization.check_factorization(n, factors) charges when it accepts factors,
    ((p, e, certificate), ...) with the p**e multiplying to n, when that is at most limit >= 0; otherwise some number
    above limit. The entries in turn (factorization_entry_work), then the prime proofs (prime_proof_work); the check
    of a factorization that it refuses stops charging where it refuses it."""
    total, product = 0, 1
    for prime, exponent, _ in factors:
        total += factorization_entry_work(product.bit_length(), prime.bit_length() * exponent)
        if total > limit:
            return total
        product *= prime ** exponent
    for prime, _, certificate in factors:
        total += prime_proof_work(prime, certificate, limit - total)
        if total > limit:
            break
    return total
