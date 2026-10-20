"""Speed fixtures and v0.7 value-corpus inputs of the discrete area (factorial ... stirling).

Shapes. The fixture interface has two sizes, ``micro`` and ``realistic``; every operation has one micro fixture and two
realistic ones, a size a student, an engineer or an agent would use and a scaled one.

* ``micro``: at least 1,000 *distinct* small inputs of one shape, so that no call of a suite sample repeats an input
  that SymPy's ``cacheit`` (cleared before every sample) or a MathForge cache could answer from memory. Two-argument
  operations walk the (n, k) pairs of a triangle; one-argument operations (factorial, fibonacci, catalan) take n over a
  range of 1,000 consecutive values.
* ``realistic``, plausible: the v0.7 case ``binomial(200, 80)``, a multinomial of five counts, ``fibonacci(20000)``.
* ``realistic``, scaled: the same operation near the largest size the default limits allow (65,536 bits of output) or
  that v0.7 still finishes in well under a second.
"""

from __future__ import annotations

from functools import lru_cache

import mathforge as mf
from benchmarks.speed.fixture import EdgeCase, Fixture

MICRO = 1024
BIG = 1 << 140000  # 140,001 bits: more than twice max_integer_bits (65,536), so clearly over the limit
TINY = {"max_integer_bits": 1}  # every integer of two or more bits is over this explicit limit

# The SymPy call each fixture is timed against, then the other candidates measured on the same inputs. A candidate is
# a SymPy-level public function or method that a user calls for the operation and that returns the same mathematical
# object (checked with agree): igcd, factorial(Integer), the sympy.stats functions. Scalar domain-element primitives
# (ZZ.gcd, ZZ.factorial, QQ and PythonMPQ arithmetic) are ground arithmetic and never count, like dup_*, dmp_* and
# gf_*. Where SymPy has no call for the operation, the reference is the fastest expression a SymPy user would write
# with SymPy objects (Rational, Integer, sqrt), inputs prebuilt. The reference is the fastest same-object call, the one
# with the smallest median: no function wins a tie, and the spelling that takes ints or SymPy Integers is whichever
# measured faster. Medians are microseconds per call over three interleaved passes of 11 to 100 samples (k consecutive
# variants per sample, SymPy caches cleared before every sample, SYMPY_GROUND_TYPES=python). A reference that was
# re-pointed to a faster call was measured again with the suite's own sampling, together with the call it replaced: 31
# to 127 interleaved samples of k calls, caches reset before each.
# "not counted" marks a formula, an alias or a domain-element primitive that is not a candidate.
_REFERENCE = {
    "factorial/micro/range_5_to_1004": (
        "factorial(Integer(n)): 88 us; other candidates: ff(n, n) (FallingFactorial, n! as n(n-1)...1) "
        "98.8 us; factorial(int) 203 us; not counted: ZZ.factorial(n) (domain-element primitive: mpmath's "
        "ifac, memoized up to 1,000) 0.289 us. ZZ.factorial is a domain-element primitive (mpmath's ifac, "
        "a table up to 1,000), not counted"),
    "factorial/realistic/n2000": (
        "factorial(Integer(n)): 458 us; other candidates: ff(n, n) (FallingFactorial, n! as n(n-1)...1) "
        "468 us; factorial(int) 579 us; not counted: ZZ.factorial(n) (domain-element primitive: mpmath's "
        "ifac, memoized up to 1,000) 582 us. ZZ.factorial is a domain-element primitive (mpmath's ifac, a "
        "table up to 1,000), not counted"),
    "factorial/realistic/n5000": (
        "factorial(Integer(n)): 1.04 ms; other candidates: ff(n, n) (FallingFactorial, n! as n(n-1)...1) "
        "1.05 ms; factorial(int) 1.18 ms; not counted: ZZ.factorial(n) (domain-element primitive: "
        "mpmath's ifac, memoized up to 1,000) 4.32 ms. ZZ.factorial is a domain-element primitive "
        "(mpmath's ifac, a table up to 1,000), not counted"),
    "binomial/micro/triangle": (
        "binomial(Integer(n), Integer(k)): 174 us; other candidates: nC(int, int) 194 us; binomial(int, "
        "int) 216 us"),
    "binomial/realistic/c200_80": (
        "binomial(Integer(n), Integer(k)): 303 us; other candidates: nC(int, int) 448 us; binomial(int, "
        "int) 456 us"),
    "binomial/realistic/c4000_2000": (
        "binomial(Integer(n), Integer(k)): 1.8 ms; other candidates: nC(int, int) 1.86 ms; binomial(int, "
        "int) 1.86 ms"),
    "multinomial/micro/three_counts": (
        "multinomial_coefficients(3, n)[counts]: 123 us; other candidates: prod binomial(partial sum, c) "
        "with Integer() built inside the call 190 us; (sum c)! / prod c! with Integer() built inside the "
        "call 199 us. SymPy has no function for one multinomial coefficient; multinomial_coefficients "
        "computes all coefficients of a total"),
    "multinomial/realistic/five_counts": (
        "factorial(Integer(sum c)) / prod factorial(Integer(c)) (no single-call function exists): 203 us; "
        "other candidates: (sum c)! / prod c! with Integer() built inside the call 822 us; prod "
        "binomial(partial sum, c) with Integer() built inside the call 1.32 ms. SymPy has no function for "
        "one multinomial coefficient; multinomial_coefficients computes all coefficients of a total"),
    "multinomial/realistic/five_counts_1500": (
        "prod binomial(partial sum, c) (no single-call function exists): 1.54 ms; other candidates: prod "
        "binomial(partial sum, c) with Integer() built inside the call 1.55 ms; (sum c)! / prod c! with "
        "Integer() built inside the call 1.73 ms. SymPy has no function for one multinomial coefficient; "
        "multinomial_coefficients computes all coefficients of a total"),
    "permutations_count/micro/triangle": (
        "nP(n, k): 1.98 us; other candidates: nP(Integer, Integer) 5.77 us; ff(n, k) 25.4 us; not "
        "counted: factorial(n) / factorial(n - k) (composite) 105 us"),
    "permutations_count/realistic/p1000_300": (
        "nP(n, k): 43.5 us; other candidates: nP(Integer, Integer) 57.3 us; ff(n, k) 624 us; not counted: "
        "factorial(n) / factorial(n - k) (composite) 676 us"),
    "permutations_count/realistic/p5000_1000": (
        "nP(n, k): 290 us; other candidates: nP(Integer, Integer) 303 us; ff(n, k) 2.62 ms; not counted: "
        "factorial(n) / factorial(n - k) (composite) 2.87 ms"),
    "fibonacci/micro/range_10_to_1009": (
        "fibonacci(Integer(n)): 9.65 us; other candidates: fibonacci(int) 10.7 us"),
    "fibonacci/realistic/n20000": (
        "fibonacci(Integer(n)): 223 us; other candidates: fibonacci(int) 224 us"),
    "fibonacci/realistic/n90000": (
        "fibonacci(Integer(n)): 2.44 ms; other candidates: fibonacci(int) 2.44 ms"),
    "catalan/micro/range_5_to_1004": (
        "catalan(Integer(n)): 1.2 ms; other candidates: catalan(int) 1.21 ms; not counted: binomial(2n, "
        "n) / (n + 1) (composite) 292 us. binomial(2n, n) / (n + 1) is a formula of two calls, not a "
        "catalan call"),
    "catalan/realistic/n300": (
        "catalan(Integer(n)): 1.58 ms; other candidates: catalan(int) 1.63 ms; not counted: binomial(2n, "
        "n) / (n + 1) (composite) 376 us. binomial(2n, n) / (n + 1) is a formula of two calls, not a "
        "catalan call"),
    "catalan/realistic/n1500": (
        "catalan(Integer(n)): 2.4 ms; other candidates: catalan(int) 2.46 ms; not counted: binomial(2n, "
        "n) / (n + 1) (composite) 1.2 ms. binomial(2n, n) / (n + 1) is a formula of two calls, not a "
        "catalan call"),
    "stirling/micro/triangle": (
        "stirling(n, k): 11.2 us; other candidates: stirling(Integer, Integer) 11.3 us"),
    "stirling/realistic/s120_40": (
        "stirling(n, k): 262 us; other candidates: stirling(Integer, Integer) 265 us"),
    "stirling/realistic/s300_150": (
        "stirling(Integer, Integer): 2.68 ms; other candidates: stirling(n, k) 2.68 ms"),
}


def _fixture(operation, size, label, mathforge, sympy, agree, variants=1, v07=""):
    return Fixture(operation, size, label, mathforge, sympy, agree, _REFERENCE[f"{operation}/{size}/{label}"],
                   variants, v07)


# ---------------------------------------------------------------------------------------------------------------------
# Inputs. Every function depends on the variant number i alone.


@lru_cache(maxsize=None)
def _triangle(low: int, high: int, margin: int) -> tuple[tuple[int, int], ...]:
    """The pairs (n, k) with low <= n < high and margin <= k <= n - margin, in order of n and then k."""
    return tuple((n, k) for n in range(low, high) for k in range(margin, n - margin + 1))


def _pair(i):
    return _triangle(10, 60, 1)[i]  # 1,675 pairs; the first 1,024 reach n = 47


def _stirling_pair(i):
    return _triangle(5, 51, 3)[i]  # 3 <= k <= n - 2: SymPy has closed forms for k = 1, 2, n - 1 and n


def _counts(i):
    return (1 + i % 13, 1 + i // 13 % 13, 1 + i // 169 % 13)


def _range_n(first):
    return lambda i: first + i


# ---------------------------------------------------------------------------------------------------------------------
# agree


def _same_int(m, s):
    return type(m) is int and (type(s) is int or getattr(s, "is_Integer", False)) and m == s


# ---------------------------------------------------------------------------------------------------------------------
# SymPy builders: variant i -> the timed zero-argument call, inputs prebuilt. SymPy is imported here and nowhere else.


def _sp_factorial(number):
    def build(i):
        from sympy import Integer, factorial
        n = Integer(number(i))
        return lambda: factorial(n)
    return build


def _sp_binomial(inputs):
    def build(i):
        from sympy import Integer, binomial
        n, k = map(Integer, inputs(i))
        return lambda: binomial(n, k)
    return build


def _sp_multinomial_table(inputs):
    """SymPy's documented multinomial_coefficients(m, n) builds every coefficient of total n; one is looked up."""
    def build(i):
        from sympy.ntheory.multinomial import multinomial_coefficients
        counts = inputs(i)
        m, n = len(counts), sum(counts)
        return lambda: multinomial_coefficients(m, n)[counts]
    return build


def _sp_multinomial_factorials(inputs):
    """SymPy has no single-coefficient multinomial function: (sum c)! / prod c!, the fastest formula for 5 counts."""
    def build(i):
        from sympy import Integer, factorial
        counts = inputs(i)
        total, parts = Integer(sum(counts)), [Integer(c) for c in counts]

        def multinomial():
            value = factorial(total)
            for part in parts:
                value = value / factorial(part)
            return value
        return multinomial
    return build


def _sp_multinomial_binomials(inputs):
    """The same coefficient as a product of binomials C(c1, c1) C(c1 + c2, c2) ..., the fastest formula at 1,500."""
    def build(i):
        from sympy import Integer, binomial
        counts = inputs(i)

        def multinomial():
            total, value = 0, Integer(1)
            for count in counts:
                total += count
                value *= binomial(total, count)
            return value
        return multinomial
    return build


def _sp_nP(inputs):
    def build(i):
        from sympy.functions.combinatorial.numbers import nP
        n, k = inputs(i)
        return lambda: nP(n, k)
    return build


def _sp_fibonacci(number):
    def build(i):
        from sympy import Integer, fibonacci
        n = Integer(number(i))
        return lambda: fibonacci(n)
    return build


def _sp_catalan(number):
    def build(i):
        from sympy import Integer, catalan
        n = Integer(number(i))
        return lambda: catalan(n)
    return build


def _sp_stirling(inputs):
    def build(i):
        from sympy.functions.combinatorial.numbers import stirling
        n, k = inputs(i)
        return lambda: stirling(n, k)
    return build


def _sp_stirling_integers(inputs):
    """stirling on SymPy Integers, the spelling that measured faster than Python ints on the (300, 150) pair."""
    def build(i):
        from sympy import Integer
        from sympy.functions.combinatorial.numbers import stirling
        n, k = map(Integer, inputs(i))
        return lambda: stirling(n, k)
    return build


def _mf_one(function, number):
    return lambda i: (function, (number(i),), {})


def _mf_pair(function, inputs):
    return lambda i: (function, inputs(i), {})


FIXTURES = (
    # factorial -------------------------------------------------------------------------------------------------------
    _fixture("factorial", "micro", "range_5_to_1004", _mf_one(mf.factorial, _range_n(5)),
             _sp_factorial(_range_n(5)), _same_int, variants=1000),
    _fixture("factorial", "realistic", "n2000", _mf_one(mf.factorial, lambda i: 2000),
             _sp_factorial(lambda i: 2000), _same_int),
    _fixture("factorial", "realistic", "n5000", _mf_one(mf.factorial, lambda i: 5000),
             _sp_factorial(lambda i: 5000), _same_int),
    # binomial --------------------------------------------------------------------------------------------------------
    _fixture("binomial", "micro", "triangle", _mf_pair(mf.binomial, _pair), _sp_binomial(_pair), _same_int,
             variants=MICRO),
    _fixture("binomial", "realistic", "c200_80", _mf_pair(mf.binomial, lambda i: (200, 80)),
             _sp_binomial(lambda i: (200, 80)), _same_int, v07="binomial"),
    _fixture("binomial", "realistic", "c4000_2000", _mf_pair(mf.binomial, lambda i: (4000, 2000)),
             _sp_binomial(lambda i: (4000, 2000)), _same_int),
    # multinomial -----------------------------------------------------------------------------------------------------
    _fixture("multinomial", "micro", "three_counts", lambda i: (mf.multinomial, (_counts(i),), {}),
             _sp_multinomial_table(_counts), _same_int, variants=MICRO),
    _fixture("multinomial", "realistic", "five_counts", lambda i: (mf.multinomial, ((30, 25, 20, 15, 10),), {}),
             _sp_multinomial_factorials(lambda i: (30, 25, 20, 15, 10)), _same_int),
    _fixture("multinomial", "realistic", "five_counts_1500",
             lambda i: (mf.multinomial, ((500, 400, 300, 200, 100),), {}),
             _sp_multinomial_binomials(lambda i: (500, 400, 300, 200, 100)), _same_int),
    # permutations_count ----------------------------------------------------------------------------------------------
    _fixture("permutations_count", "micro", "triangle", _mf_pair(mf.permutations_count, _pair), _sp_nP(_pair),
             _same_int, variants=MICRO),
    _fixture("permutations_count", "realistic", "p1000_300", _mf_pair(mf.permutations_count, lambda i: (1000, 300)),
             _sp_nP(lambda i: (1000, 300)), _same_int),
    _fixture("permutations_count", "realistic", "p5000_1000", _mf_pair(mf.permutations_count, lambda i: (5000, 1000)),
             _sp_nP(lambda i: (5000, 1000)), _same_int),
    # fibonacci -------------------------------------------------------------------------------------------------------
    _fixture("fibonacci", "micro", "range_10_to_1009", _mf_one(mf.fibonacci, _range_n(10)),
             _sp_fibonacci(_range_n(10)), _same_int, variants=1000),
    _fixture("fibonacci", "realistic", "n20000", _mf_one(mf.fibonacci, lambda i: 20000),
             _sp_fibonacci(lambda i: 20000), _same_int),
    _fixture("fibonacci", "realistic", "n90000", _mf_one(mf.fibonacci, lambda i: 90000),
             _sp_fibonacci(lambda i: 90000), _same_int),
    # catalan ---------------------------------------------------------------------------------------------------------
    _fixture("catalan", "micro", "range_5_to_1004", _mf_one(mf.catalan, _range_n(5)), _sp_catalan(_range_n(5)),
             _same_int, variants=1000),
    _fixture("catalan", "realistic", "n300", _mf_one(mf.catalan, lambda i: 300), _sp_catalan(lambda i: 300),
             _same_int),
    _fixture("catalan", "realistic", "n1500", _mf_one(mf.catalan, lambda i: 1500), _sp_catalan(lambda i: 1500),
             _same_int),
    # stirling --------------------------------------------------------------------------------------------------------
    _fixture("stirling", "micro", "triangle", _mf_pair(mf.stirling, _stirling_pair), _sp_stirling(_stirling_pair),
             _same_int, variants=MICRO),
    _fixture("stirling", "realistic", "s120_40", _mf_pair(mf.stirling, lambda i: (120, 40)),
             _sp_stirling(lambda i: (120, 40)), _same_int),
    _fixture("stirling", "realistic", "s300_150", _mf_pair(mf.stirling, lambda i: (300, 150)),
             _sp_stirling_integers(lambda i: (300, 150)), _same_int),
)


# ---------------------------------------------------------------------------------------------------------------------
# Edge cases: never timed. For every operation: the four type edges in the first argument, the error classes the
# operation can raise (v0.7 has invalid_input and resource_limit_exceeded; none of these operations raises
# unsupported_operation), an index far over the default limits (twice max_integer_bits or more), a result far over
# them (at least twice 65,536 bits), one explicit limit that the input exceeds, and the borders of each domain.


def _case(operation, label, function, *args, **kwargs):
    return EdgeCase(operation, label, lambda: (function, args, kwargs))


def _type_edges(operation, function, *rest):
    return tuple(EdgeCase(operation, f"type_{name}", lambda v=value: (function, (v, *rest), {}))
                 for name, value in (("true", True), ("float", 1.5), ("str", "1"), ("none", None)))


def _limit(**limits):
    return {"limits": mf.ComputationLimits(**limits)}


def _pair_edges(operation, function, valid, output, borders):
    return (
        *_type_edges(operation, function, valid[1]),
        _case(operation, "type_second_str", function, valid[0], "1"),
        _case(operation, "type_second_float", function, valid[0], 1.5),
        _case(operation, "over_limit", function, BIG, valid[1]),
        _case(operation, "over_limit_second", function, valid[0], BIG),
        _case(operation, "over_limit_output", function, *output),
        _case(operation, "explicit_limits", function, *valid, **_limit(**TINY)),
        _case(operation, "negative_first", function, -1, valid[1]),
        _case(operation, "negative_second", function, valid[0], -1),
        *(_case(operation, label, function, *args) for label, args in borders),
    )


EDGE_CASES = (
    # factorial
    *_type_edges("factorial", mf.factorial),
    _case("factorial", "over_limit", mf.factorial, BIG),
    _case("factorial", "over_limit_output", mf.factorial, 30000),
    _case("factorial", "explicit_limits", mf.factorial, 20, **_limit(**TINY)),
    *(_case("factorial", label, mf.factorial, n) for label, n in (
        ("negative", -1), ("zero", 0), ("one", 1), ("two", 2), ("twenty", 20))),
    # binomial
    *_pair_edges("binomial", mf.binomial, (10, 3), (200000, 100000), (
        ("k_above_n", (3, 10)), ("zero_zero", (0, 0)), ("n_zero_k_one", (0, 1)), ("k_zero", (10, 0)),
        ("k_equals_n", (10, 10)), ("symmetric", (100, 97)), ("large_n_small_k", (10 ** 6, 3)))),
    # multinomial
    *_type_edges("multinomial", mf.multinomial),
    _case("multinomial", "over_limit", mf.multinomial, (BIG, 1)),
    _case("multinomial", "over_limit_output", mf.multinomial, (100000, 100000, 100000)),
    _case("multinomial", "explicit_limits", mf.multinomial, (2, 3, 4), **_limit(**TINY)),
    *(_case("multinomial", label, mf.multinomial, counts) for label, counts in (
        ("empty", ()), ("single", (5,)), ("zeros", (0, 0, 0)), ("negative_count", (2, -1, 3)),
        ("bool_count", (True, 2)), ("float_count", (2.0, 3)), ("str_count", ("2", 3)), ("list_of_counts", [2, 3, 4]),
        ("nested", ((1, 2), 3)), ("trinomial", (2, 3, 4)))),
    # permutations_count
    *_pair_edges("permutations_count", mf.permutations_count, (10, 3), (100000, 50000), (
        ("k_above_n", (3, 10)), ("zero_zero", (0, 0)), ("k_zero", (10, 0)), ("k_equals_n", (10, 10)),
        ("large_n_small_k", (10 ** 6, 3)))),
    # fibonacci
    *_type_edges("fibonacci", mf.fibonacci),
    _case("fibonacci", "over_limit", mf.fibonacci, BIG),
    _case("fibonacci", "over_limit_output", mf.fibonacci, 300000),
    _case("fibonacci", "explicit_limits", mf.fibonacci, 20, **_limit(**TINY)),
    *(_case("fibonacci", label, mf.fibonacci, n) for label, n in (
        ("negative", -1), ("zero", 0), ("one", 1), ("two", 2), ("ninety", 90))),
    # catalan
    *_type_edges("catalan", mf.catalan),
    _case("catalan", "over_limit", mf.catalan, BIG),
    _case("catalan", "over_limit_output", mf.catalan, 100000),
    _case("catalan", "explicit_limits", mf.catalan, 20, **_limit(**TINY)),
    *(_case("catalan", label, mf.catalan, n) for label, n in (
        ("negative", -1), ("zero", 0), ("one", 1), ("two", 2), ("thirty", 30))),
    # stirling
    *_pair_edges("stirling", mf.stirling, (10, 3), (140000, 2), (
        ("k_above_n", (3, 10)), ("zero_zero", (0, 0)), ("n_positive_k_zero", (5, 0)), ("k_one", (5, 1)),
        ("k_one_large_n", (140000, 1)), ("k_equals_n", (5, 5)), ("k_two", (10, 2)), ("n_minus_one", (10, 9)),
        ("middle", (30, 15)))),
    # explicit limits of the wrong type are an invalid input; a raised limit admits what the defaults refuse
    _case("factorial", "type_limits_int", mf.factorial, 20, limits=3),
    _case("binomial", "type_limits_int", mf.binomial, 10, 3, limits=3),
    _case("multinomial", "type_limits_int", mf.multinomial, (2, 3, 4), limits=3),
    _case("permutations_count", "type_limits_int", mf.permutations_count, 10, 3, limits=3),
    _case("fibonacci", "type_limits_int", mf.fibonacci, 20, limits=3),
    _case("catalan", "type_limits_int", mf.catalan, 20, limits=3),
    _case("stirling", "type_limits_int", mf.stirling, 10, 3, limits=3),
    _case("binomial", "raised_limits", mf.binomial, 70000, 1, **_limit(max_integer_bits=80000)),
    _case("permutations_count", "raised_limits", mf.permutations_count, 70000, 1, **_limit(max_integer_bits=80000)),
)
