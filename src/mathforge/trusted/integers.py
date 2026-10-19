"""Trusted integer arithmetic shared by the integer kernel and the prime-certificate checker."""

from math import prod


def powers_by_prime(base, exponent, modulus, primes):
    """{p: pow(base, exponent // p, modulus) for each p of primes}, for distinct primes that all divide exponent.

    One exponentiation by exponent // prod(primes), then a product-tree descent: a group's power raised to the
    product of the other half of the group is the power of that half. For k primes this takes about log2(k)
    exponentiations by the product of the primes in all, rather than k exponentiations by exponent // p."""
    result = {}
    pending = [(pow(base, exponent // prod(primes), modulus), tuple(primes))]
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
