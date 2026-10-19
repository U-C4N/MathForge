"""Speed fixtures and v0.7 value-corpus inputs of the ntheory area (integer_gcd ... totient).

Shapes. The fixture interface has two sizes, ``micro`` and ``realistic``; every operation has one micro fixture and two
realistic ones, a size a student, an engineer or an agent would use and a scaled one.

* ``micro``: the smallest inputs that still make 2,048 *distinct* calls of one shape, for example
  ``integer_gcd(84 + 6*i, 30 + 6*i)``. Distinct inputs matter: a suite sample makes k consecutive calls, and a repeated
  input would be answered from a cache (SymPy's ``cacheit``, the ``lru_cache`` of ``igcd``) instead of being computed.
  ``igcd`` keeps 1,024 results that ``clear_cache()`` does not empty, and ``ilcm`` ends in ``igcd``, so the gcd and
  lcm fixtures have more than 1,024 variants even when they are not micro fixtures.
* ``realistic``, plausible: RSA-sized moduli, a CRT over 128 primes, a Mersenne modulus, the v0.7 case
  ``factor_integer(600851475143)``.
* ``realistic``, scaled: the same operation near the largest size the default limits allow (65,536 bits) and that v0.7
  still finishes in well under a second.

Big integers are built from SHA-256 blocks (``_noise``), so they are the same in every process and on every Python.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib

import mathforge as mf
from benchmarks.speed.fixture import EdgeCase, Fixture

MICRO = 2048
BIG = 1 << 140000  # 140,001 bits: more than twice max_integer_bits (65,536), so clearly over the limit
TINY = {"max_integer_bits": 1}  # every integer of two or more bits is over this explicit limit

# The SymPy call each fixture is timed against, then the other candidates measured on the same inputs. A candidate is
# a SymPy-level public function or method that a user calls for the operation and that returns the same mathematical
# object (checked with agree): igcd, factorial(Integer), the sympy.stats functions. Scalar domain-element primitives
# (ZZ.gcd, ZZ.factorial, QQ and PythonMPQ arithmetic) are ground arithmetic and never count, like dup_*, dmp_* and
# gf_*. Where SymPy has no call for the operation, the reference is the shortest expression a SymPy user would write
# with SymPy objects (Rational, Integer, sqrt), inputs prebuilt. The fastest candidate is chosen, but the documented
# top-level function wins any tie within 30%. Medians are microseconds per call over three interleaved passes of 11 to
# 100 samples (k consecutive variants per sample, SymPy caches cleared before every sample, SYMPY_GROUND_TYPES=python)
# on a machine that other jobs share: differences under about 30% are noise.
# "not counted" marks a formula, an alias or a domain-element primitive that is not a candidate.
_REFERENCE = {
    "integer_gcd/micro/pairs": (
        "igcd(a, b): 0.645 us; other candidates: igcd(Integer, Integer) 0.924 us; Integer.gcd 2.07 us; "
        "gcd(Integer, Integer) 71.4 us; not counted: igcd2 / sympy.external.gmpy.gcd (math.gcd alias) "
        "0.0875 us; ZZ.gcd (domain-element primitive) 0.119 us; igcd.__wrapped__ (igcd without its lru; "
        "not what a user calls) 0.495 us. igcd keeps an lru_cache of 1,024 results that clear_cache() "
        "leaves alone (emptied before every sample here), so the fixtures have more than 1,024 variants; "
        "ZZ.gcd and igcd2 are math.gcd, ground arithmetic, not counted"),
    "integer_gcd/realistic/rsa_moduli": (
        "igcd(a, b): 9.82 us; other candidates: igcd(Integer, Integer) 10.2 us; Integer.gcd 13.8 us; "
        "gcd(Integer, Integer) 100 us; not counted: igcd2 / sympy.external.gmpy.gcd (math.gcd alias) 8.66 "
        "us; ZZ.gcd (domain-element primitive) 8.72 us; igcd.__wrapped__ (igcd without its lru; not what "
        "a user calls) 9.12 us. igcd keeps an lru_cache of 1,024 results that clear_cache() leaves alone "
        "(emptied before every sample here), so the fixtures have more than 1,024 variants; ZZ.gcd and "
        "igcd2 are math.gcd, ground arithmetic, not counted"),
    "integer_gcd/realistic/bits30000": (
        "igcd(a, b): 1.17 ms; other candidates: Integer.gcd 1.2 ms; igcd(Integer, Integer) 1.21 ms; "
        "gcd(Integer, Integer) 1.35 ms; not counted: igcd.__wrapped__ (igcd without its lru; not what a "
        "user calls) 1.15 ms; ZZ.gcd (domain-element primitive) 1.15 ms; igcd2 / sympy.external.gmpy.gcd "
        "(math.gcd alias) 1.16 ms. igcd keeps an lru_cache of 1,024 results that clear_cache() leaves "
        "alone (emptied before every sample here), so the fixtures have more than 1,024 variants; ZZ.gcd "
        "and igcd2 are math.gcd, ground arithmetic, not counted"),
    "integer_lcm/micro/pairs": (
        "ilcm(a, b): 0.538 us; other candidates: ilcm(Integer, Integer) 0.668 us; Integer.lcm 3.32 us; "
        "lcm(Integer, Integer) 79.2 us; not counted: ZZ.lcm (domain-element primitive) 0.572 us. ilcm "
        "calls igcd, whose lru_cache holds 1,024 results, so the fixtures have more than 1,024 variants; "
        "ZZ.lcm is a domain-element primitive, not counted"),
    "integer_lcm/realistic/rsa_moduli": (
        "ilcm(a, b): 13 us; other candidates: ilcm(Integer, Integer) 13.1 us; Integer.lcm 18 us; "
        "lcm(Integer, Integer) 106 us; not counted: ZZ.lcm (domain-element primitive) 13.1 us. ilcm calls "
        "igcd, whose lru_cache holds 1,024 results, so the fixtures have more than 1,024 variants; ZZ.lcm "
        "is a domain-element primitive, not counted"),
    "integer_lcm/realistic/bits20000": (
        "ilcm(a, b): 738 us; other candidates: ilcm(Integer, Integer) 740 us; Integer.lcm 765 us; "
        "lcm(Integer, Integer) 861 us; not counted: ZZ.lcm (domain-element primitive) 814 us. ilcm calls "
        "igcd, whose lru_cache holds 1,024 results, so the fixtures have more than 1,024 variants; ZZ.lcm "
        "is a domain-element primitive, not counted"),
    "integer_xgcd/micro/pairs": (
        "igcdex(a, b): 0.837 us; other candidates: igcdex(Integer, Integer) 0.928 us; gcdex(Integer, "
        "Integer) 81 us; not counted: ZZ.gcdex (domain-element primitive) 0.929 us. ZZ.gcdex is a "
        "domain-element primitive, not counted"),
    "integer_xgcd/realistic/bits4096": (
        "igcdex(a, b): 1.87 ms; other candidates: igcdex(Integer, Integer) 1.85 ms; gcdex(Integer, "
        "Integer) 1.95 ms; not counted: ZZ.gcdex (domain-element primitive) 1.83 ms. ZZ.gcdex is a "
        "domain-element primitive, not counted"),
    "integer_xgcd/realistic/bits16000": (
        "igcdex(a, b): 21.7 ms; other candidates: gcdex(Integer, Integer) 21.4 ms; igcdex(Integer, "
        "Integer) 22.3 ms; not counted: ZZ.gcdex (domain-element primitive) 22.1 ms. ZZ.gcdex is a "
        "domain-element primitive, not counted"),
    "mod_inverse/micro/prime_modulus": (
        "mod_inverse(a, m): 1.71 us; other candidates: mod_inverse(Integer, Integer) 1.84 us; "
        "invert(Integer, Integer) 76.2 us; not counted: ZZ.invert (domain-element primitive) 1.57 us. "
        "ZZ.invert is a domain-element primitive, not counted"),
    "mod_inverse/realistic/mersenne4423": (
        "mod_inverse(a, m): 2.12 ms; other candidates: mod_inverse(Integer, Integer) 2.12 ms; "
        "invert(Integer, Integer) 2.27 ms; not counted: ZZ.invert (domain-element primitive) 2.16 ms. "
        "ZZ.invert is a domain-element primitive, not counted"),
    "mod_inverse/realistic/mersenne19937": (
        "mod_inverse(a, m): 31.5 ms; other candidates: mod_inverse(Integer, Integer) 31.6 ms; "
        "invert(Integer, Integer) 31.9 ms; not counted: ZZ.invert (domain-element primitive) 31.6 ms. "
        "ZZ.invert is a domain-element primitive, not counted"),
    "crt/micro/three_moduli": (
        "crt(m, v, check=False): 3.65 us; other candidates: crt(m, v, check=True) 5.67 us; "
        "solve_congruence(*pairs) 8.35 us; crt(Integer lists, check=False) 23 us"),
    "crt/realistic/primes128": (
        "crt(m, v, check=False): 319 us; other candidates: crt(m, v, check=True) 387 us; "
        "solve_congruence(*pairs) 460 us; crt(Integer lists, check=False) 1.2 ms"),
    "crt/realistic/primes512": (
        "crt(m, v, check=False): 3.97 ms; other candidates: solve_congruence(*pairs) 4.59 ms; crt(m, v, "
        "check=True) 4.71 ms; crt(Integer lists, check=False) 9.87 ms"),
    "is_prime/micro/odd_numbers": (
        "isprime(n): 0.458 us; other candidates: isprime(Integer) 0.533 us; not counted: Integer.is_prime "
        "0.121 us. Integer.is_prime is cached on the Integer object, which clear_cache() cannot empty, so "
        "it does not compute; isprime proves primality below 3.3e24 and SymPy has nothing faster"),
    "is_prime/realistic/mersenne61": (
        "isprime(n): 95.4 us; other candidates: isprime(Integer) 96.5 us; not counted: Integer.is_prime "
        "0.25 us. Integer.is_prime is cached on the Integer object, which clear_cache() cannot empty, so "
        "it does not compute; isprime proves primality below 3.3e24 and SymPy has nothing faster"),
    "is_prime/realistic/prime_near_3e24": (
        "isprime(n): 238 us; other candidates: isprime(Integer) 238 us; not counted: Integer.is_prime "
        "0.263 us. Integer.is_prime is cached on the Integer object, which clear_cache() cannot empty, so "
        "it does not compute; isprime proves primality below 3.3e24 and SymPy has nothing faster"),
    "factor_integer/micro/small_numbers": (
        "factorint(n): 2.62 us; other candidates: factorint(Integer) 2.83 us; Integer.factors() 6.01 us"),
    "factor_integer/realistic/euler3": (
        "factorint(n): 60.4 us; other candidates: factorint(Integer) 60.9 us; Integer.factors() 67.2 us"),
    "factor_integer/realistic/semiprime_12_digits": (
        "factorint(n): 174 us; other candidates: factorint(Integer) 174 us; Integer.factors() 182 us"),
    "divisors/micro/small_numbers": (
        "divisors(n): 7.89 us; other candidates: sorted(divisors(n, generator=True)) 7.89 us; "
        "divisors(Integer) 8.21 us"),
    "divisors/realistic/n720720": (
        "divisors(n): 55.2 us; other candidates: sorted(divisors(n, generator=True)) 55.1 us; "
        "divisors(Integer) 57 us"),
    "divisors/realistic/n73513440": (
        "divisors(n): 128 us; other candidates: divisors(Integer) 129 us; sorted(divisors(n, "
        "generator=True)) 129 us"),
    "totient/micro/small_numbers": (
        "totient(Integer(n)): 13 us; other candidates: totient(int) 44.3 us; not counted: "
        "prod(p**(e-1)*(p-1) over factorint(n)) (composite) 3.69 us. the factorint formula is a "
        "composite, not a totient call"),
    "totient/realistic/euler3": (
        "totient(Integer(n)): 48 us; other candidates: totient(int) 79.5 us; not counted: "
        "prod(p**(e-1)*(p-1) over factorint(n)) (composite) 64.4 us. the factorint formula is a "
        "composite, not a totient call"),
    "totient/realistic/toy_rsa_modulus": (
        "totient(Integer(n)): 142 us; other candidates: totient(int) 172 us; not counted: "
        "prod(p**(e-1)*(p-1) over factorint(n)) (composite) 179 us. the factorint formula is a composite, "
        "not a totient call"),
}


def _fixture(operation, size, label, mathforge, sympy, agree, variants=1, v07=""):
    return Fixture(operation, size, label, mathforge, sympy, agree, _REFERENCE[f"{operation}/{size}/{label}"],
                   variants, v07)


# ---------------------------------------------------------------------------------------------------------------------
# Inputs. Every function depends on the variant number i alone.


@lru_cache(maxsize=None)
def _noise(seed: str, bits: int) -> int:
    """An odd integer of exactly ``bits`` bits that depends on the arguments only (SHA-256 blocks)."""
    blocks, counter = [], 0
    while 256 * len(blocks) < bits:
        blocks.append(hashlib.sha256(f"{seed}/{counter}".encode()).digest())
        counter += 1
    value = int.from_bytes(b"".join(blocks), "big") >> (256 * len(blocks) - bits)
    return value | (1 << (bits - 1)) | 1


@lru_cache(maxsize=None)
def _primes(count: int) -> tuple[int, ...]:
    found: list[int] = []
    candidate = 2
    while len(found) < count:
        if all(candidate % p for p in found if p * p <= candidate):
            found.append(candidate)
        candidate += 1
    return tuple(found)


def _small_pair(i):
    return 84 + 6 * i, 30 + 6 * i


def _rsa_pair(i):
    """Two 2,048-bit moduli that share a 1,024-bit factor; the batch-gcd attack on RSA keys computes their gcd."""
    shared = _noise("rsa/shared", 1024) + 2 * i
    return shared * (_noise("rsa/first", 1024) + 2 * i), shared * (_noise("rsa/second", 1024) + 2 * i)


def _lcm_big_pair(i):
    return _noise("lcm/first", 20000) + 2 * i, _noise("lcm/second", 20000) + 2 * i


def _gcd_big_pair(i):
    return _noise("gcd/first", 30000) + 2 * i, _noise("gcd/second", 30000) + 2 * i


def _xgcd_pair(i):
    return _noise("xgcd/first", 4096) + 2 * i, _noise("xgcd/second", 4096) + 2 * i


def _xgcd_big_pair(i):
    return _noise("xgcd/big-first", 16000) + 2 * i, _noise("xgcd/big-second", 16000) + 2 * i


MERSENNE_4423 = (1 << 4423) - 1  # a Mersenne prime
MERSENNE_19937 = (1 << 19937) - 1  # a Mersenne prime


def _inverse_micro(i):
    return 2 + i, 4099  # 4099 is prime and larger than every a, so every a is invertible


def _inverse_realistic(i):
    return _noise("inverse/a", 4400) + 2 * i, MERSENNE_4423


def _inverse_scaled(i):
    return _noise("inverse/big-a", 19900) + 2 * i, MERSENNE_19937


def _crt_micro(i):
    """The residues of i modulo three coprime moduli; the solution is i itself, modulo 47,027."""
    return tuple((i % m, m) for m in (31, 37, 41))


def _crt_primes(count, bits):
    """Residues of one number of ``bits`` bits modulo the first ``count`` primes (their product exceeds the number)."""
    number = _noise(f"crt/{count}", bits)
    return tuple((number % p, p) for p in _primes(count))


def _crt_realistic(i):
    return _crt_primes(128, 900)


def _crt_scaled(i):
    return _crt_primes(512, 5000)


def _odd(i):
    return 101 + 2 * i


MERSENNE_61 = (1 << 61) - 1
PRIME_3E24 = 3224192153648326577750017  # 2**40 * 3**20 * 29**2 + 1, below 3.317e24 where SymPy's isprime is a proof


def _small(i):
    return 360 + i


def _phi_small(i):
    return 36 + i


# ---------------------------------------------------------------------------------------------------------------------
# agree: (MathForge value, SymPy value) -> the same mathematical object. No SymPy import is needed: SymPy integers are
# compared through ==, which is exact between int and Integer.


def _same_int(m, s):
    return type(m) is int and (type(s) is int or getattr(s, "is_Integer", False)) and m == s


def _same_xgcd(m, s):
    """MathForge returns (g, x, y) and SymPy's igcdex (x, y, g); both are the extended Euclidean triple."""
    return type(m) is tuple and len(m) == 3 and len(s) == 3 and all(map(_same_int, m, (s[2], s[0], s[1])))


def _same_crt(m, s):
    if s is None:
        return type(m) is mf.CRTResult and m.residue is None and m.modulus is None
    return type(m) is mf.CRTResult and _same_int(m.residue, s[0]) and _same_int(m.modulus, s[1])


def _same_bool(m, s):
    return type(m) is bool and type(s) is bool and m is s


def _same_factorization(m, s):
    return (type(m) is mf.IntegerFactorization and m.sign == 1
            and dict(m.factors) == {int(p): int(e) for p, e in s.items()})


def _same_divisors(m, s):
    return type(m) is tuple and len(m) == len(s) and all(map(_same_int, m, s))


# ---------------------------------------------------------------------------------------------------------------------
# SymPy builders: variant i -> the timed zero-argument call, inputs prebuilt. SymPy is imported here and nowhere else.


def _sp_igcd(inputs):
    def build(i):
        from sympy import igcd
        a, b = inputs(i)
        return lambda: igcd(a, b)
    return build


def _sp_ilcm(inputs):
    def build(i):
        from sympy import ilcm
        a, b = inputs(i)
        return lambda: ilcm(a, b)
    return build


def _sp_igcdex(inputs):
    def build(i):
        from sympy.core.intfunc import igcdex
        a, b = inputs(i)
        return lambda: igcdex(a, b)
    return build


def _sp_mod_inverse(inputs):
    def build(i):
        from sympy import mod_inverse
        a, m = inputs(i)
        return lambda: mod_inverse(a, m)
    return build


def _sp_crt(inputs):
    def build(i):
        from sympy.ntheory.modular import crt
        pairs = inputs(i)
        residues, moduli = [r for r, _ in pairs], [m for _, m in pairs]
        return lambda: crt(moduli, residues, check=False)
    return build


def _sp_isprime(number):
    def build(i):
        from sympy import isprime
        n = number(i)
        return lambda: isprime(n)
    return build


def _sp_factorint(number):
    def build(i):
        from sympy import factorint
        n = number(i)
        return lambda: factorint(n)
    return build


def _sp_divisors(number):
    def build(i):
        from sympy import divisors
        n = number(i)
        return lambda: divisors(n)
    return build


def _sp_totient(number):
    def build(i):
        from sympy import Integer, totient
        n = Integer(number(i))
        return lambda: totient(n)
    return build


def _mf_pair(function, inputs):
    return lambda i: (function, inputs(i), {})


def _mf_one(function, number):
    return lambda i: (function, (number(i),), {})


# ---------------------------------------------------------------------------------------------------------------------

FIXTURES = (
    # integer_gcd -----------------------------------------------------------------------------------------------------
    _fixture("integer_gcd", "micro", "pairs", _mf_pair(mf.integer_gcd, _small_pair), _sp_igcd(_small_pair),
             _same_int, variants=MICRO),
    _fixture("integer_gcd", "realistic", "rsa_moduli", _mf_pair(mf.integer_gcd, _rsa_pair), _sp_igcd(_rsa_pair),
             _same_int, variants=MICRO),
    _fixture("integer_gcd", "realistic", "bits30000", _mf_pair(mf.integer_gcd, _gcd_big_pair), _sp_igcd(_gcd_big_pair),
             _same_int, variants=1100),
    # integer_lcm -----------------------------------------------------------------------------------------------------
    _fixture("integer_lcm", "micro", "pairs", _mf_pair(mf.integer_lcm, _small_pair), _sp_ilcm(_small_pair),
             _same_int, variants=MICRO),
    _fixture("integer_lcm", "realistic", "rsa_moduli", _mf_pair(mf.integer_lcm, _rsa_pair), _sp_ilcm(_rsa_pair),
             _same_int, variants=MICRO),
    _fixture("integer_lcm", "realistic", "bits20000", _mf_pair(mf.integer_lcm, _lcm_big_pair), _sp_ilcm(_lcm_big_pair),
             _same_int, variants=1100),
    # integer_xgcd ----------------------------------------------------------------------------------------------------
    _fixture("integer_xgcd", "micro", "pairs", _mf_pair(mf.integer_xgcd, _small_pair), _sp_igcdex(_small_pair),
             _same_xgcd, variants=MICRO),
    _fixture("integer_xgcd", "realistic", "bits4096", _mf_pair(mf.integer_xgcd, _xgcd_pair), _sp_igcdex(_xgcd_pair),
             _same_xgcd),
    _fixture("integer_xgcd", "realistic", "bits16000", _mf_pair(mf.integer_xgcd, _xgcd_big_pair),
             _sp_igcdex(_xgcd_big_pair), _same_xgcd),
    # mod_inverse -----------------------------------------------------------------------------------------------------
    _fixture("mod_inverse", "micro", "prime_modulus", _mf_pair(mf.mod_inverse, _inverse_micro),
             _sp_mod_inverse(_inverse_micro), _same_int, variants=MICRO),
    _fixture("mod_inverse", "realistic", "mersenne4423", _mf_pair(mf.mod_inverse, _inverse_realistic),
             _sp_mod_inverse(_inverse_realistic), _same_int),
    _fixture("mod_inverse", "realistic", "mersenne19937", _mf_pair(mf.mod_inverse, _inverse_scaled),
             _sp_mod_inverse(_inverse_scaled), _same_int),
    # crt -------------------------------------------------------------------------------------------------------------
    _fixture("crt", "micro", "three_moduli", lambda i: (mf.crt, (_crt_micro(i),), {}), _sp_crt(_crt_micro), _same_crt,
             variants=MICRO),
    _fixture("crt", "realistic", "primes128", lambda i: (mf.crt, (_crt_realistic(i),), {}), _sp_crt(_crt_realistic),
             _same_crt),
    _fixture("crt", "realistic", "primes512", lambda i: (mf.crt, (_crt_scaled(i),), {}), _sp_crt(_crt_scaled),
             _same_crt),
    # is_prime --------------------------------------------------------------------------------------------------------
    _fixture("is_prime", "micro", "odd_numbers", _mf_one(mf.is_prime, _odd), _sp_isprime(_odd), _same_bool,
             variants=MICRO),
    _fixture("is_prime", "realistic", "mersenne61", _mf_one(mf.is_prime, lambda i: MERSENNE_61),
             _sp_isprime(lambda i: MERSENNE_61), _same_bool),
    _fixture("is_prime", "realistic", "prime_near_3e24", _mf_one(mf.is_prime, lambda i: PRIME_3E24),
             _sp_isprime(lambda i: PRIME_3E24), _same_bool),
    # factor_integer --------------------------------------------------------------------------------------------------
    _fixture("factor_integer", "micro", "small_numbers", _mf_one(mf.factor_integer, _small), _sp_factorint(_small),
             _same_factorization, variants=MICRO),
    _fixture("factor_integer", "realistic", "euler3", _mf_one(mf.factor_integer, lambda i: 600851475143),
             _sp_factorint(lambda i: 600851475143), _same_factorization, v07="factor_integer"),
    _fixture("factor_integer", "realistic", "semiprime_12_digits", _mf_one(mf.factor_integer, lambda i: 999983 * 1000003),
             _sp_factorint(lambda i: 999983 * 1000003), _same_factorization),
    # divisors --------------------------------------------------------------------------------------------------------
    _fixture("divisors", "micro", "small_numbers", _mf_one(mf.divisors, _small), _sp_divisors(_small),
             _same_divisors, variants=MICRO),
    _fixture("divisors", "realistic", "n720720", _mf_one(mf.divisors, lambda i: 720720),
             _sp_divisors(lambda i: 720720), _same_divisors),
    _fixture("divisors", "realistic", "n73513440", _mf_one(mf.divisors, lambda i: 73513440),
             _sp_divisors(lambda i: 73513440), _same_divisors),
    # totient ---------------------------------------------------------------------------------------------------------
    _fixture("totient", "micro", "small_numbers", _mf_one(mf.totient, _phi_small), _sp_totient(_phi_small),
             _same_int, variants=MICRO),
    _fixture("totient", "realistic", "euler3", _mf_one(mf.totient, lambda i: 600851475143),
             _sp_totient(lambda i: 600851475143), _same_int),
    _fixture("totient", "realistic", "toy_rsa_modulus", _mf_one(mf.totient, lambda i: 1000003 * 1000033),
             _sp_totient(lambda i: 1000003 * 1000033), _same_int),
)


# ---------------------------------------------------------------------------------------------------------------------
# Edge cases: never timed. For every operation: the four type edges in the first argument, the error classes the
# operation can raise (v0.7 has invalid_input and resource_limit_exceeded; none of these operations raises
# unsupported_operation), one input far over the default limits (twice max_integer_bits or more), one explicit limit
# that the input exceeds, and the values at the borders of each domain.


def _case(operation, label, function, *args, **kwargs):
    return EdgeCase(operation, label, lambda: (function, args, kwargs))


def _type_edges(operation, function, *rest):
    return tuple(EdgeCase(operation, f"type_{name}", lambda v=value: (function, (v, *rest), {}))
                 for name, value in (("true", True), ("float", 1.5), ("str", "1"), ("none", None)))


def _limit(**limits):
    return {"limits": mf.ComputationLimits(**limits)}


def _two_integers(operation, function, borders):
    return (
        *_type_edges(operation, function, 4),
        _case(operation, "type_second_str", function, 4, "1"),
        _case(operation, "type_second_float", function, 4, 1.5),
        _case(operation, "over_limit", function, BIG, 4),
        _case(operation, "over_limit_second", function, 4, BIG),
        _case(operation, "explicit_limits", function, 84, 30, **_limit(**TINY)),
        *(_case(operation, label, function, *args) for label, args in borders),
    )


_SIGNS = (("zero_zero", (0, 0)), ("zero_left", (0, 5)), ("zero_right", (5, 0)), ("negative_left", (-84, 30)),
          ("negative_right", (84, -30)), ("negative_both", (-84, -30)), ("equal", (7, 7)), ("coprime", (35, 18)))

EDGE_CASES = (
    *_two_integers("integer_gcd", mf.integer_gcd, _SIGNS),
    *_two_integers("integer_lcm", mf.integer_lcm, _SIGNS),
    *_two_integers("integer_xgcd", mf.integer_xgcd, _SIGNS),
    # mod_inverse: the modulus must be positive, and the inverse must exist
    *_type_edges("mod_inverse", mf.mod_inverse, 7),
    _case("mod_inverse", "type_second_str", mf.mod_inverse, 3, "1"),
    _case("mod_inverse", "type_second_float", mf.mod_inverse, 3, 1.5),
    _case("mod_inverse", "over_limit", mf.mod_inverse, BIG, 7),
    _case("mod_inverse", "over_limit_modulus", mf.mod_inverse, 3, BIG),
    _case("mod_inverse", "explicit_limits", mf.mod_inverse, 3, 7, **_limit(**TINY)),
    _case("mod_inverse", "no_inverse", mf.mod_inverse, 2, 4),
    _case("mod_inverse", "modulus_zero", mf.mod_inverse, 3, 0),
    _case("mod_inverse", "modulus_negative", mf.mod_inverse, 3, -7),
    _case("mod_inverse", "modulus_one", mf.mod_inverse, 5, 1),
    _case("mod_inverse", "a_zero", mf.mod_inverse, 0, 7),
    _case("mod_inverse", "a_negative", mf.mod_inverse, -3, 7),
    _case("mod_inverse", "a_above_modulus", mf.mod_inverse, 10, 7),
    # crt: tuples of (residue, positive modulus)
    *_type_edges("crt", mf.crt),
    _case("crt", "over_limit", mf.crt, ((BIG, 7),)),
    _case("crt", "over_limit_modulus", mf.crt, ((1, BIG),)),
    _case("crt", "explicit_limits", mf.crt, ((2, 3), (3, 5)), **_limit(**TINY)),
    _case("crt", "empty", mf.crt, ()),
    _case("crt", "single", mf.crt, ((2, 5),)),
    _case("crt", "list_of_pairs", mf.crt, [(2, 3), (3, 5)]),
    _case("crt", "pair_as_list", mf.crt, ([2, 3], [3, 5])),
    _case("crt", "inconsistent", mf.crt, ((0, 2), (1, 4))),
    _case("crt", "consistent_not_coprime", mf.crt, ((1, 4), (3, 6))),
    _case("crt", "residue_out_of_range", mf.crt, ((12, 5), (-4, 7))),
    _case("crt", "modulus_zero", mf.crt, ((1, 0),)),
    _case("crt", "modulus_negative", mf.crt, ((1, -3),)),
    _case("crt", "bool_residue", mf.crt, ((True, 3),)),
    _case("crt", "float_modulus", mf.crt, ((1, 3.0),)),
    _case("crt", "triple", mf.crt, ((1, 2, 3),)),
    _case("crt", "not_a_pair", mf.crt, (5,)),
    # is_prime: every integer is accepted, the answer is False below 2
    *_type_edges("is_prime", mf.is_prime),
    _case("is_prime", "over_limit", mf.is_prime, BIG),
    _case("is_prime", "explicit_limits", mf.is_prime, 97, **_limit(**TINY)),
    *(_case("is_prime", label, mf.is_prime, n) for label, n in (
        ("zero", 0), ("negative", -7), ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("carmichael_561", 561),
        ("square_of_prime", 9409), ("prime_9973", 9973), ("below_trial_bound", 9999), ("above_trial_bound", 10007),
        ("mersenne_61", MERSENNE_61), ("fermat_pseudoprime_base_two", 341))),
    # factor_integer / divisors / totient
    *_type_edges("factor_integer", mf.factor_integer),
    _case("factor_integer", "over_limit", mf.factor_integer, BIG),
    _case("factor_integer", "explicit_limits", mf.factor_integer, 360, **_limit(**TINY)),
    *(_case("factor_integer", label, mf.factor_integer, n) for label, n in (
        ("zero", 0), ("one", 1), ("minus_one", -1), ("negative", -360), ("prime", 97), ("prime_power", 1 << 20),
        ("square_of_prime", 9409), ("two_primes", 91))),
    *_type_edges("divisors", mf.divisors),
    _case("divisors", "over_limit", mf.divisors, BIG),
    _case("divisors", "explicit_limits", mf.divisors, 360, **_limit(**TINY)),
    *(_case("divisors", label, mf.divisors, n) for label, n in (
        ("zero", 0), ("one", 1), ("minus_one", -1), ("negative", -12), ("prime", 97), ("prime_power", 1 << 10),
        ("highly_composite", 5040))),
    *_type_edges("totient", mf.totient),
    _case("totient", "over_limit", mf.totient, BIG),
    _case("totient", "explicit_limits", mf.totient, 36, **_limit(**TINY)),
    *(_case("totient", label, mf.totient, n) for label, n in (
        ("zero", 0), ("negative", -9), ("one", 1), ("two", 2), ("prime", 97), ("prime_power", 1 << 20),
        ("composite", 360))),
    # explicit limits of the wrong type are an invalid input; a raised limit admits what the defaults refuse
    _case("integer_gcd", "type_limits_int", mf.integer_gcd, 84, 30, limits=3),
    _case("integer_lcm", "type_limits_int", mf.integer_lcm, 84, 30, limits=3),
    _case("integer_xgcd", "type_limits_int", mf.integer_xgcd, 84, 30, limits=3),
    _case("mod_inverse", "type_limits_int", mf.mod_inverse, 3, 7, limits=3),
    _case("crt", "type_limits_int", mf.crt, ((2, 3), (3, 5)), limits=3),
    _case("is_prime", "type_limits_int", mf.is_prime, 97, limits=3),
    _case("factor_integer", "type_limits_int", mf.factor_integer, 360, limits=3),
    _case("divisors", "type_limits_int", mf.divisors, 360, limits=3),
    _case("totient", "type_limits_int", mf.totient, 36, limits=3),
    _case("integer_gcd", "raised_limits", mf.integer_gcd, 1 << 70000, 4, **_limit(max_integer_bits=80000)),
    _case("integer_xgcd", "raised_limits", mf.integer_xgcd, 1 << 70000, 4, **_limit(max_integer_bits=80000)),
    _case("is_prime", "raised_limits", mf.is_prime, 1 << 70000, **_limit(max_integer_bits=80000)),
)
