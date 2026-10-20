"""Speed fixtures and v0.7 value-corpus inputs of the stats area (bernoulli ... covariance).

Shapes. The fixture interface has two sizes, ``micro`` and ``realistic``; every operation but bernoulli (whose result
has two outcomes whatever the input) has one micro fixture and two realistic ones, a size a student, an engineer or an
agent would use and a scaled one.

* ``micro``: at least 1,000 *distinct* inputs of one tiny shape (5-outcome distributions, 5-element samples, 4-outcome
  uniforms), so that no call of a suite sample repeats an input that SymPy's ``cacheit`` or a MathForge cache could
  answer from memory. Distinct means distinct: a weight vector is the base-9 digits of the variant number, a sample the
  base-11 digits.
* ``realistic``, plausible: the binomial PMF of the v0.7 case (n = 40, p = 2/7, 41 outcomes), the eight-outcome uniform
  of the v0.7 variance case, samples of 200 measurements.
* ``realistic``, scaled: the same operation on 120 outcomes, 100 binomial trials, samples of 500 values. The sizes are
  the largest that keep the corpus entries small (a returned distribution is stored in full), that keep SymPy's own
  ``sympy.stats`` calls (which grow with the support) below a few hundred milliseconds, and that v0.7 handles in a
  fraction of a second: its median checker is quadratic (0.7 s for 500 values).

SymPy has no sample statistics, so ``mean``, ``median``, ``variance``, ``stddev`` and ``covariance`` are timed against
the fastest expression a SymPy user would write (the fastest of the formulas measured) on prebuilt ``sympy.Rational``
objects; the standard deviation is ``sqrt(variance, evaluate=False)``, the unreduced root that ``mathforge.stddev``
returns (``sympy.sqrt`` also simplifies the radical, which is most of its time and not part of that object); every
distribution operation is timed against the fastest same-object ``sympy.stats`` call on a prebuilt ``FiniteRV`` that
carries the same PMF: ``P`` or ``FinitePSpace.probability``, ``E`` or ``FinitePSpace.compute_expectation``, ``moment``
or ``compute_expectation(X**n)``, ``cdf`` or ``compute_cdf``.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

import mathforge as mf
from benchmarks.speed.fixture import EdgeCase, Fixture

MICRO = 1024
BIG = 1 << 140000  # 140,001 bits: more than twice max_integer_bits (65,536), so clearly over the limit
TINY = {"max_integer_bits": 1, "max_nodes": 1, "max_work": 1}  # an explicit limit that every input below exceeds
R = mf.Rational

# The SymPy call each fixture is timed against, then the other candidates measured on the same inputs. A candidate is
# a SymPy-level public function or method that a user calls for the operation and that returns the same mathematical
# object (checked with agree): igcd, factorial(Integer), the sympy.stats functions. Scalar domain-element primitives
# (ZZ.gcd, ZZ.factorial, QQ and PythonMPQ arithmetic) are ground arithmetic and never count, like dup_*, dmp_* and
# gf_*. Where SymPy has no call for the operation, the reference is the fastest expression a SymPy user would write
# with SymPy objects (Rational, Integer, sqrt), inputs prebuilt. The reference is the fastest same-object call, the one
# with the smallest median: no function wins a tie, and a method of an object (``FinitePSpace.compute_expectation``)
# is as good a call as a top-level function (``E``). Medians are microseconds per call over three interleaved passes of
# 11 to 100 samples (k consecutive variants per sample, SymPy caches cleared before every sample,
# SYMPY_GROUND_TYPES=python). A reference that was re-pointed to a faster call was measured again with the suite's own
# sampling, together with the call it replaced: 31 to 127 interleaved samples of k calls, caches reset before each.
# "not counted" marks a formula, an alias or a domain-element primitive that is not a candidate.
_REFERENCE = {
    "bernoulli/micro/p_over_1031": (
        "density(Bernoulli('X', p)).dict: 52.4 us; not counted: BernoulliDistribution(p, 1, 0).dict "
        "(internal class, not exported) 25 us. density(X) alone returns a lazy distribution object; .dict "
        "materializes the PMF, the object MathForge returns"),
    "bernoulli/realistic/p_21_digits": (
        "density(Bernoulli('X', p)).dict: 195 us; not counted: BernoulliDistribution(p, 1, 0).dict "
        "(internal class, not exported) 108 us. density(X) alone returns a lazy distribution object; "
        ".dict materializes the PMF, the object MathForge returns"),
    "binomial_distribution/micro/small_n": (
        "density(Binomial('X', n, p)).dict: 988 us; not counted: BinomialDistribution(n, p, 1, 0).dict "
        "(internal class, not exported) 917 us; sum of pmf(k) over range (composite) 483 us. density(X) "
        "alone returns a lazy distribution object; .dict materializes the PMF, the object MathForge "
        "returns; Binomial(...).pspace.density is lazy too"),
    "binomial_distribution/realistic/n40": (
        "density(Binomial('X', n, p)).dict: 14.9 ms; not counted: BinomialDistribution(n, p, 1, 0).dict "
        "(internal class, not exported) 14.7 ms; sum of pmf(k) over range (composite) 9.15 ms. density(X) "
        "alone returns a lazy distribution object; .dict materializes the PMF, the object MathForge "
        "returns; Binomial(...).pspace.density is lazy too"),
    "binomial_distribution/realistic/n100": (
        "density(Binomial('X', n, p)).dict: 37.8 ms; not counted: BinomialDistribution(n, p, 1, 0).dict "
        "(internal class, not exported) 37 ms; sum of pmf(k) over range (composite) 23.3 ms. density(X) "
        "alone returns a lazy distribution object; .dict materializes the PMF, the object MathForge "
        "returns; Binomial(...).pspace.density is lazy too"),
    "discrete_uniform/micro/four_outcomes": (
        "density(DiscreteUniform('X', values)).dict: 36.7 us. density(X) alone returns a lazy "
        "distribution object; .dict materializes the PMF, the object MathForge returns; "
        "DiscreteUniform(...).pspace.density and FiniteRV(...) are not the PMF"),
    "discrete_uniform/realistic/fifty_outcomes": (
        "density(DiscreteUniform('X', values)).dict: 226 us. density(X) alone returns a lazy distribution "
        "object; .dict materializes the PMF, the object MathForge returns; "
        "DiscreteUniform(...).pspace.density and FiniteRV(...) are not the PMF"),
    "discrete_uniform/realistic/two_hundred_outcomes": (
        "density(DiscreteUniform('X', values)).dict: 601 us. density(X) alone returns a lazy distribution "
        "object; .dict materializes the PMF, the object MathForge returns; "
        "DiscreteUniform(...).pspace.density and FiniteRV(...) are not the PMF"),
    "hypergeometric/micro/small_urn": (
        "density(Hypergeometric('X', N, m, n)).dict: 270 us; not counted: HypergeometricDistribution(N, "
        "m, n).dict (internal class, not exported) 178 us. density(X) alone returns a lazy distribution "
        "object; .dict materializes the PMF, the object MathForge returns"),
    "hypergeometric/realistic/urn_200": (
        "density(Hypergeometric('X', N, m, n)).dict: 13.9 ms; not counted: HypergeometricDistribution(N, "
        "m, n).dict (internal class, not exported) 13.8 ms. density(X) alone returns a lazy distribution "
        "object; .dict materializes the PMF, the object MathForge returns"),
    "hypergeometric/realistic/urn_600": (
        "density(Hypergeometric('X', N, m, n)).dict: 35.3 ms; not counted: HypergeometricDistribution(N, "
        "m, n).dict (internal class, not exported) 35.2 ms. density(X) alone returns a lazy distribution "
        "object; .dict materializes the PMF, the object MathForge returns"),
    "probability/micro/five_outcomes": (
        "P(Or(Eq(X, v) ...)): 903 us; other candidates: X.pspace.probability(Contains(X, FiniteSet)) 1.27 ms; "
        "P(Contains(X, FiniteSet(event))) 1.31 ms; not counted: sum(density(X)[v] for v in event) (dict lookups) 66.2"
        " us. the dict lookups are not a SymPy probability call"),
    "probability/realistic/binomial41": (
        "X.pspace.probability(Contains(X, FiniteSet)): 12.8 ms; other candidates: P(Contains(X, FiniteSet(event))) "
        "13.1 ms; first pass, an earlier machine state: P(Or(Eq(X, v) ...)) 25.5 ms; not counted: sum(density(X)[v] "
        "for v in event) (dict lookups) 994 us. the dict lookups are not a SymPy probability call"),
    "probability/realistic/outcomes_120": (
        "X.pspace.probability(Contains(X, FiniteSet)): 37.7 ms; other candidates: P(Contains(X, FiniteSet(event))) "
        "38.2 ms; first pass, an earlier machine state: P(Or(Eq(X, v) ...)) 98.5 ms; not counted: sum(density(X)[v] "
        "for v in event) (dict lookups) 3.71 ms. the dict lookups are not a SymPy probability call"),
    "condition/micro/five_outcomes": (
        "density(given(X, Or(Eq(X, v) ...))).dict: 2.61 ms; other candidates: density(given(X, Contains(X, "
        "FiniteSet(event)))).dict 2.99 ms; not counted: density(given(X, Contains(X, FiniteSet))) (no .dict) 6.07 ms "
        "(first pass). conditional_space(...).density returns a lazy object, not the conditional PMF"),
    "condition/realistic/binomial41": (
        "density(given(X, Contains(X, FiniteSet(event)))).dict: 37.2 ms; other candidates: "
        "density(given(X, Contains(X, FiniteSet))) (no .dict) 36.1 ms; density(given(X, Or(Eq...))) 47.6 "
        "ms. conditional_space(...).density returns a lazy object, not the conditional PMF"),
    "condition/realistic/outcomes_120": (
        "density(given(X, Contains(X, FiniteSet(event)))).dict: 146 ms; other candidates: "
        "density(given(X, Contains(X, FiniteSet))) (no .dict) 149 ms; density(given(X, Or(Eq...))) 213 "
        "ms. conditional_space(...).density returns a lazy object, not the conditional PMF"),
    "cdf/micro/five_outcomes": (
        "cdf(X)[x]: 738 us; other candidates: X.pspace.compute_cdf(X)[x] 740 us; P(X <= x) 1.1 ms. cdf(X) "
        "returns the whole table of a finite variable; indexing it is the documented way to evaluate it"),
    "cdf/realistic/binomial41": (
        "cdf(X)[x]: 4.16 ms; other candidates: X.pspace.compute_cdf(X)[x] 4.16 ms; P(X <= x) 5.93 ms. "
        "cdf(X) returns the whole table of a finite variable; indexing it is the documented way to "
        "evaluate it"),
    "cdf/realistic/outcomes_120": (
        "X.pspace.compute_cdf(X)[x]: 10.8 ms; other candidates: cdf(X)[x] 10.8 ms; first pass, an earlier machine "
        "state: P(X <= x) 21.6 ms. cdf(X) returns the whole table of a finite variable; indexing it is the documented"
        " way to evaluate it"),
    "moment/micro/five_outcomes": (
        "X.pspace.compute_expectation(X**n): 558 us; other candidates: moment(X, n) 750 us; first pass, an earlier "
        "machine state: E(X**n) 1.19 ms"),
    "moment/realistic/binomial41": (
        "X.pspace.compute_expectation(X**n): 3.81 ms; other candidates: moment(X, n) 4.62 ms; first pass, an earlier "
        "machine state: E(X**n) 5.49 ms"),
    "moment/realistic/outcomes_120": (
        "X.pspace.compute_expectation(X**n): 15 ms; other candidates: moment(X, n) 17.8 ms; first pass, an earlier "
        "machine state: E(X**n) 18.3 ms"),
    "expectation/micro/five_outcomes": (
        "X.pspace.compute_expectation(X): 364 us; other candidates: E(X) 437 us; first pass, an earlier machine "
        "state: moment(X, 1) 954 us"),
    "expectation/realistic/binomial41": (
        "X.pspace.compute_expectation(X): 2.02 ms; other candidates: E(X) 2.36 ms; first pass, an earlier machine "
        "state: moment(X, 1) 5.07 ms"),
    "expectation/realistic/outcomes_120": (
        "X.pspace.compute_expectation(X): 7.9 ms; other candidates: E(X) 9.06 ms; first pass, an earlier machine "
        "state: moment(X, 1) 18 ms"),
    "distribution_variance/micro/five_outcomes": (
        "variance(X): 3.9 ms; not counted: moment(X, 2) - E(X)**2 (composite) 2.29 ms; "
        "X.pspace.compute_expectation((X - E)**2) (composite) 2.54 ms"),
    "distribution_variance/realistic/finite_variance": (
        "variance(X): 5.64 ms; not counted: moment(X, 2) - E(X)**2 (composite) 3.63 ms; "
        "X.pspace.compute_expectation((X - E)**2) (composite) 3.89 ms"),
    "distribution_variance/realistic/outcomes_120": (
        "variance(X): 51.6 ms; not counted: moment(X, 2) - E(X)**2 (composite) 36.3 ms; "
        "X.pspace.compute_expectation((X - E)**2) (composite) 43 ms"),
    "mean/micro/five_values": (
        "sum(data) / n on sympy.Rational: 4.98 us; other candidates: Add(*data) / n 25.4 us; "
        "Mul(Add(*data), Rational(1, n)) 41.8 us; not counted: sum(data) / n on PythonMPQ (ground "
        "arithmetic) 2.6 us. SymPy has no sample statistics: the fastest of the expressions a SymPy user "
        "would write on prebuilt sympy.Rational; PythonMPQ (QQ) arithmetic is ground arithmetic, not "
        "counted"),
    "mean/realistic/values_200": (
        "sum(data) / n on sympy.Rational: 530 us; other candidates: Add(*data) / n 740 us; "
        "Mul(Add(*data), Rational(1, n)) 778 us; not counted: sum(data) / n on PythonMPQ (ground "
        "arithmetic) 105 us. SymPy has no sample statistics: the fastest of the expressions a SymPy user "
        "would write on prebuilt sympy.Rational; PythonMPQ (QQ) arithmetic is ground arithmetic, not "
        "counted"),
    "mean/realistic/values_500": (
        "sum(data) / n on sympy.Rational: 1.25 ms; other candidates: Add(*data) / n 1.72 ms; "
        "Mul(Add(*data), Rational(1, n)) 1.75 ms; not counted: sum(data) / n on PythonMPQ (ground "
        "arithmetic) 246 us. SymPy has no sample statistics: the fastest of the expressions a SymPy user "
        "would write on prebuilt sympy.Rational; PythonMPQ (QQ) arithmetic is ground arithmetic, not "
        "counted"),
    "median/micro/five_values": (
        "sorted(data), then the middle element or the mean of the two, on sympy.Rational: 11.5 us; not "
        "counted: sorted(data) on PythonMPQ (ground arithmetic) 1.29 us. SymPy has no sample statistics: "
        "the expression a SymPy user would write on prebuilt sympy.Rational; PythonMPQ (QQ) arithmetic is "
        "ground arithmetic, not counted"),
    "median/realistic/values_200": (
        "sorted(data), then the middle element or the mean of the two, on sympy.Rational: 2.69 ms; not "
        "counted: sorted(data) on PythonMPQ (ground arithmetic) 248 us. SymPy has no sample statistics: "
        "the expression a SymPy user would write on prebuilt sympy.Rational; PythonMPQ (QQ) arithmetic is "
        "ground arithmetic, not counted"),
    "median/realistic/values_500": (
        "sorted(data), then the middle element or the mean of the two, on sympy.Rational: 8.85 ms; not "
        "counted: sorted(data) on PythonMPQ (ground arithmetic) 733 us. SymPy has no sample statistics: "
        "the expression a SymPy user would write on prebuilt sympy.Rational; PythonMPQ (QQ) arithmetic is "
        "ground arithmetic, not counted"),
    "variance/micro/five_values": (
        "(sum(x*x) - sum(x)**2 / n) / (n - ddof) on sympy.Rational: 39.5 us; other candidates: two-pass "
        "(x - m)*(x - m) 63 us; one-pass Add(*[x*x]) - Add(*data)**2 / n 203 us; two-pass (x - m)**2 556 "
        "us; not counted: one-pass on PythonMPQ (ground arithmetic) 9.49 us. SymPy has no sample "
        "statistics: the fastest of the expressions a SymPy user would write on prebuilt sympy.Rational; "
        "PythonMPQ (QQ) arithmetic is ground arithmetic, not counted"),
    "variance/realistic/values_200": (
        "(sum(x*x) - sum(x)**2 / n) / (n - ddof) on sympy.Rational: 1.65 ms; other candidates: one-pass "
        "Add(*[x*x]) - Add(*data)**2 / n 2.23 ms; two-pass (x - m)*(x - m) 2.33 ms; two-pass (x - m)**2 "
        "20.8 ms; not counted: one-pass on PythonMPQ (ground arithmetic) 338 us. SymPy has no sample "
        "statistics: the fastest of the expressions a SymPy user would write on prebuilt sympy.Rational; "
        "PythonMPQ (QQ) arithmetic is ground arithmetic, not counted"),
    "variance/realistic/values_500": (
        "(sum(x*x) - sum(x)**2 / n) / (n - ddof) on sympy.Rational: 4.16 ms; other candidates: one-pass "
        "Add(*[x*x]) - Add(*data)**2 / n 5.11 ms; two-pass (x - m)*(x - m) 6.06 ms; two-pass (x - m)**2 "
        "52.6 ms; not counted: one-pass on PythonMPQ (ground arithmetic) 811 us. SymPy has no sample "
        "statistics: the fastest of the expressions a SymPy user would write on prebuilt sympy.Rational; "
        "PythonMPQ (QQ) arithmetic is ground arithmetic, not counted"),
    "stddev/micro/five_values": (
        "sqrt((sum(x*x) - sum(x)**2 / n) / (n - ddof), evaluate=False) on sympy.Rational: 31.5 us; other candidates: "
        "sympy.sqrt((sum(x*x) - sum(x)**2 / n) / (n - ddof)) on sympy.Rational, which also simplifies the radical 254"
        " us; first pass, an earlier machine state: sympy.sqrt(one-pass) 515 us; sympy.sqrt(two-pass (x - m)*(x - m))"
        " 517 us; sympy.sqrt(one-pass Add) 675 us; not counted: sqrt of the one-pass variance on PythonMPQ (ground "
        "arithmetic) 468 us. SymPy has no sample statistics: the unreduced root, sqrt(variance, evaluate=False), of "
        "the fastest variance expression on prebuilt sympy.Rational is the number that mathforge.stddev returns "
        "(sympy.sqrt(variance) also simplifies the radical, most of its time, and is measured as a rival); PythonMPQ "
        "(QQ) arithmetic is ground arithmetic, not counted"),
    "stddev/realistic/values_200": (
        "sqrt((sum(x*x) - sum(x)**2 / n) / (n - ddof), evaluate=False) on sympy.Rational: 984 us; other candidates: "
        "sympy.sqrt((sum(x*x) - sum(x)**2 / n) / (n - ddof)) on sympy.Rational, which also simplifies the radical "
        "1.54 ms; first pass, an earlier machine state: sympy.sqrt(one-pass) 2.75 ms; sympy.sqrt(one-pass Add) 3.21 "
        "ms; sympy.sqrt(two-pass (x - m)*(x - m)) 3.46 ms; not counted: sqrt of the one-pass variance on PythonMPQ "
        "(ground arithmetic) 1.36 ms. SymPy has no sample statistics: the unreduced root, sqrt(variance, "
        "evaluate=False), of the fastest variance expression on prebuilt sympy.Rational is the number that "
        "mathforge.stddev returns (sympy.sqrt(variance) also simplifies the radical, most of its time, and is "
        "measured as a rival); PythonMPQ (QQ) arithmetic is ground arithmetic, not counted"),
    "stddev/realistic/values_500": (
        "sqrt((sum(x*x) - sum(x)**2 / n) / (n - ddof), evaluate=False) on sympy.Rational: 2.35 ms; other candidates: "
        "sympy.sqrt((sum(x*x) - sum(x)**2 / n) / (n - ddof)) on sympy.Rational, which also simplifies the radical 2.9"
        " ms; first pass, an earlier machine state: sympy.sqrt(one-pass) 5.21 ms; sympy.sqrt(one-pass Add) 6.21 ms; "
        "sympy.sqrt(two-pass (x - m)*(x - m)) 7.04 ms; not counted: sqrt of the one-pass variance on PythonMPQ "
        "(ground arithmetic) 1.76 ms. SymPy has no sample statistics: the unreduced root, sqrt(variance, "
        "evaluate=False), of the fastest variance expression on prebuilt sympy.Rational is the number that "
        "mathforge.stddev returns (sympy.sqrt(variance) also simplifies the radical, most of its time, and is "
        "measured as a rival); PythonMPQ (QQ) arithmetic is ground arithmetic, not counted"),
    "covariance/micro/five_pairs": (
        "(sum(a*b) - sum(a)*sum(b) / n) / (n - ddof) on sympy.Rational: 51.9 us; other candidates: "
        "two-pass (a - mx)*(b - my) 85.9 us; one-pass Add(*[a*b]) - Add(*x)*Add(*y) / n 101 us; not "
        "counted: one-pass on PythonMPQ (ground arithmetic) 12.2 us. SymPy has no sample statistics: the "
        "fastest of the expressions a SymPy user would write on prebuilt sympy.Rational; PythonMPQ (QQ) "
        "arithmetic is ground arithmetic, not counted"),
    "covariance/realistic/pairs_200": (
        "(sum(a*b) - sum(a)*sum(b) / n) / (n - ddof) on sympy.Rational: 2.18 ms; other candidates: "
        "one-pass Add(*[a*b]) - Add(*x)*Add(*y) / n 2.76 ms; two-pass (a - mx)*(b - my) 3.19 ms; not "
        "counted: one-pass on PythonMPQ (ground arithmetic) 441 us. SymPy has no sample statistics: the "
        "fastest of the expressions a SymPy user would write on prebuilt sympy.Rational; PythonMPQ (QQ) "
        "arithmetic is ground arithmetic, not counted"),
    "covariance/realistic/pairs_500": (
        "(sum(a*b) - sum(a)*sum(b) / n) / (n - ddof) on sympy.Rational: 5.53 ms; other candidates: "
        "one-pass Add(*[a*b]) - Add(*x)*Add(*y) / n 6.94 ms; two-pass (a - mx)*(b - my) 8.47 ms; not "
        "counted: one-pass on PythonMPQ (ground arithmetic) 1.05 ms. SymPy has no sample statistics: the "
        "fastest of the expressions a SymPy user would write on prebuilt sympy.Rational; PythonMPQ (QQ) "
        "arithmetic is ground arithmetic, not counted"),
}


def _fixture(operation, size, label, mathforge, sympy, agree, variants=1, v07=""):
    return Fixture(operation, size, label, mathforge, sympy, agree, _REFERENCE[f"{operation}/{size}/{label}"],
                   variants, v07)


# ---------------------------------------------------------------------------------------------------------------------
# Inputs. Numbers are (numerator, denominator) pairs, so that each side builds its own Rational type from one source.
# Every function depends on the variant number i alone.


def _digit(i, base, position):
    return i // base ** position % base


def _micro_pmf(i):
    """Outcomes 1..5; the weights are 1 + the base-9 digits of i, so 59,049 distinct vectors."""
    weights = [1 + _digit(i, 9, j) for j in range(5)]
    total = sum(weights)
    return tuple((j + 1, 1) for j in range(5)), tuple((w, total) for w in weights)


_MASKS_ANY = tuple(range(1, 32))
_MASKS_TWO_OR_MORE = tuple(m for m in range(1, 32) if bin(m).count("1") >= 2)


def _subset(mask):
    return tuple((j + 1, 1) for j in range(5) if mask >> j & 1)


def _micro_event(i):
    """A nonempty subset of the five outcomes, as a tuple of pairs; the 31 masks cycle."""
    return _subset(_MASKS_ANY[i * 7 % 31])


def _micro_event_pair(i):
    """At least two outcomes: SymPy's given() cannot condition on a single value."""
    return _subset(_MASKS_TWO_OR_MORE[i * 7 % len(_MASKS_TWO_OR_MORE)])


def _micro_point(i):
    return (1 + i % 5, 1)


def _micro_order(i):
    return 2 + i % 4


@lru_cache(maxsize=None)
def _binomial41_pmf():
    """Binomial(n=40, p=2/7): the PMF of the v0.7 binomial_pmf case, 41 outcomes."""
    return (tuple((k, 1) for k in range(41)),
            tuple((comb(40, k) * 2 ** k * 5 ** (40 - k), 7 ** 40) for k in range(41)))


def _binomial41_event(i):
    return tuple((k, 1) for k in (5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 21))


@lru_cache(maxsize=None)
def _eight_pmf():
    """The finite_variance case of benchmarks/fresh_comparison.py: uniform on eight rationals."""
    support = ((-7, 3), (-2, 1), (-1, 5), (0, 1), (2, 7), (4, 3), (3, 1), (11, 2))
    return support, ((1, 8),) * 8


@lru_cache(maxsize=None)
def _scaled_pmf():
    """120 outcomes k/7 with weights proportional to k + 1."""
    total = 120 * 121 // 2
    return tuple((k, 7) for k in range(120)), tuple((k + 1, total) for k in range(120))


def _scaled_event(i):
    return tuple((k, 7) for k in range(0, 120, 6))  # 20 outcomes


def _small_sample(i):
    """Five rationals; the numerators are base-11 digits of i, the denominators 2..6."""
    return tuple((3 + _digit(i, 11, j), 2 + j) for j in range(5))


def _small_sample_y(i):
    return tuple((3 + _digit(i * 13 + 5, 11, j), 2 + j) for j in range(5))


def _measurements(count, salt):
    """count decimals with two places, 10.00 to 99.99, in a fixed pseudo-random order."""
    return tuple((1000 + (j * 7919 + salt * 104729 + 13) % 9000, 100) for j in range(count))


def _values_200(i):
    return _measurements(200, 1)


def _values_200_y(i):
    return _measurements(200, 2)


def _values_500(i):
    return _measurements(500, 1)


def _values_500_y(i):
    return _measurements(500, 2)


def _p_micro(i):
    return (i + 1, 1031)  # 1031 is prime and above every i + 1 below 1,024, so 0 < p < 1


def _binomial_small(i):
    """n in 3..13 and p = k/98 for k in 1..97: 1,067 distinct pairs."""
    return 3 + i // 97 % 11, (1 + i % 97, 98)


def _four_outcomes(i):
    return tuple((i + d, 1) for d in (0, 1, 3, 7))


def _fifty_outcomes(i):
    return tuple((k * k + 3, 7) for k in range(50))


def _two_hundred_outcomes(i):
    return tuple((k * k + 3, 7) for k in range(200))


@lru_cache(maxsize=None)
def _urns():
    """(population, successes, draws) with 6 <= population < 16 and 1 <= successes, draws <= population."""
    return tuple((n, k, d) for n in range(6, 16) for k in range(1, n + 1) for d in range(1, n + 1))


def _urn(i):
    return _urns()[i]


# ---------------------------------------------------------------------------------------------------------------------
# MathForge and SymPy objects built from the shared inputs.


def _mf_pairs(pairs):
    return tuple(R(p, q) for p, q in pairs)


def _mf_point(pair):
    return R(*pair)


def _mf_distribution(pmf):
    outcomes, weights = pmf
    return mf.FiniteDistribution(_mf_pairs(outcomes), _mf_pairs(weights))


def _sp_rv(pmf):
    from sympy import Rational
    from sympy.stats import FiniteRV
    outcomes, weights = pmf
    return FiniteRV("X", {Rational(*o): Rational(*w) for o, w in zip(outcomes, weights)})


def _sp_event(rv, event):
    from sympy import Contains, FiniteSet, Rational
    return Contains(rv, FiniteSet(*[Rational(*e) for e in event]))


def _sp_data(pairs):
    from sympy import Rational
    return [Rational(p, q) for p, q in pairs]


# ---------------------------------------------------------------------------------------------------------------------
# agree: (MathForge value, SymPy value) -> the same mathematical object. SymPy rationals expose .p and .q.


def _pq(number):
    """(numerator, denominator) of a Python int, a SymPy Rational or Integer (.p, .q) or a PythonMPQ."""
    if type(number) is int:
        return number, 1
    if getattr(number, "is_Rational", False):
        return number.p, number.q
    return number.numerator, number.denominator


def _same_rational(m, s):
    return type(m) is R and (m.numerator, m.denominator) == _pq(s)


def _same_pmf(m, s):
    """A FiniteDistribution and a {outcome: probability} dict of SymPy rationals (zero probabilities left out).

    The SymPy value must be a materialized dict: SymPy's distribution objects (BinomialDistribution, ...) answer
    .items() lazily, which would let a call that builds nothing pass for one that builds the whole PMF.
    """
    if type(m) is not mf.FiniteDistribution or not isinstance(s, dict):
        return False
    expected = {(o.numerator, o.denominator): (w.numerator, w.denominator)
                for o, w in zip(m.outcomes, m.probabilities)}
    actual = {_pq(k): _pq(v) for k, v in s.items() if v != 0}
    return expected == actual


def _same_sqrt(m, s):
    """mathforge.stddev gives a Rational (a perfect square variance) or Sqrt(radicand), unreduced; SymPy gives the
    unevaluated root of the variance (``sqrt(variance, evaluate=False)``) or its simplified form.

    m is the same non-negative number as s when s >= 0 and s**2, which SymPy evaluates to the Rational variance, is
    the radicand of a Sqrt, or the square of a Rational.
    """
    square = s ** 2
    if not (bool(s.is_nonnegative) and square.is_Rational):
        return False
    if type(m) is R:
        return m.numerator >= 0 and (m * m).numerator == square.p and (m * m).denominator == square.q
    if type(m) is not mf.Sqrt:
        return False
    return (square.p, square.q) == (m.radicand.numerator, m.radicand.denominator)


# ---------------------------------------------------------------------------------------------------------------------
# SymPy builders: variant i -> the timed zero-argument call, inputs prebuilt. SymPy is imported here and nowhere else.


def _sp_bernoulli(parameter):
    def build(i):
        from sympy import Rational
        from sympy.stats import Bernoulli, density
        p = Rational(*parameter(i))
        return lambda: density(Bernoulli("X", p)).dict
    return build


def _sp_binomial(parameters):
    def build(i):
        from sympy import Integer, Rational
        from sympy.stats import Binomial, density
        n, p = parameters(i)
        n, p = Integer(n), Rational(*p)
        return lambda: density(Binomial("X", n, p)).dict
    return build


def _sp_uniform(outcomes):
    def build(i):
        from sympy import Rational
        from sympy.stats import DiscreteUniform, density
        values = [Rational(*o) for o in outcomes(i)]
        return lambda: density(DiscreteUniform("X", values)).dict
    return build


def _sp_hypergeometric(urn):
    def build(i):
        from sympy import Integer
        from sympy.stats import Hypergeometric, density
        n, k, d = map(Integer, urn(i))
        return lambda: density(Hypergeometric("X", n, k, d)).dict
    return build


def _sp_equalities(rv, event):
    """The event as an Or of equalities, ``Or(Eq(X, v) ...)``: what P and given evaluate fastest for a small event."""
    from sympy import Eq, Or, Rational
    return Or(*[Eq(rv, Rational(*e)) for e in event])


def _sp_probability_pspace(pmf, event):
    """``FinitePSpace.probability(Contains(X, FiniteSet))``: the call that P makes, without the dispatch around it."""
    def build(i):
        rv = _sp_rv(pmf(i))
        condition = _sp_event(rv, event(i))
        return lambda: rv.pspace.probability(condition)
    return build


def _sp_probability_equalities(pmf, event):
    def build(i):
        from sympy.stats import P
        rv = _sp_rv(pmf(i))
        condition = _sp_equalities(rv, event(i))
        return lambda: P(condition)
    return build


def _sp_condition(pmf, event):
    def build(i):
        from sympy.stats import density, given
        rv = _sp_rv(pmf(i))
        condition = _sp_event(rv, event(i))
        return lambda: density(given(rv, condition)).dict
    return build


def _sp_condition_equalities(pmf, event):
    def build(i):
        from sympy.stats import density, given
        rv = _sp_rv(pmf(i))
        condition = _sp_equalities(rv, event(i))
        return lambda: density(given(rv, condition)).dict
    return build


def _sp_cdf(pmf, point):
    def build(i):
        from sympy import Rational
        from sympy.stats import cdf
        rv = _sp_rv(pmf(i))
        x = Rational(*point(i))
        return lambda: cdf(rv)[x]
    return build


def _sp_cdf_pspace(pmf, point):
    """``FinitePSpace.compute_cdf(X)[x]``: the table that ``cdf(X)`` returns, without the dispatch around it."""
    def build(i):
        from sympy import Rational
        rv = _sp_rv(pmf(i))
        x = Rational(*point(i))
        return lambda: rv.pspace.compute_cdf(rv)[x]
    return build


def _sp_moment(pmf, order):
    """``FinitePSpace.compute_expectation(X**n)``: what ``moment(X, n)`` computes, without the dispatch around it."""
    def build(i):
        rv = _sp_rv(pmf(i))
        n = order(i)
        return lambda: rv.pspace.compute_expectation(rv ** n)
    return build


def _sp_expectation(pmf):
    """``FinitePSpace.compute_expectation(X)``: what ``E(X)`` computes, without the dispatch around it."""
    def build(i):
        rv = _sp_rv(pmf(i))
        return lambda: rv.pspace.compute_expectation(rv)
    return build


def _sp_distribution_variance(pmf):
    def build(i):
        from sympy.stats import variance
        rv = _sp_rv(pmf(i))
        return lambda: variance(rv)
    return build


def _sp_mean(sample):
    def build(i):
        data = _sp_data(sample(i))
        n = len(data)
        return lambda: sum(data) / n
    return build


def _sp_median(sample):
    def build(i):
        data = _sp_data(sample(i))
        n = len(data)
        middle = n // 2

        def median():
            ordered = sorted(data)
            return ordered[middle] if n % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        return median
    return build


def _sp_variance(sample, ddof=0):
    def build(i):
        data = _sp_data(sample(i))
        n = len(data)

        def variance():
            total = sum(data)
            return (sum(x * x for x in data) - total * total / n) / (n - ddof)
        return variance
    return build


def _sp_stddev(sample, ddof=0):
    """The unreduced root of the variance, ``sqrt(variance, evaluate=False)``, which is what ``mathforge.stddev``
    returns: ``sympy.sqrt(variance)`` also simplifies the radical (factoring the numerator and the denominator),
    most of its time, and returns another form of the same number."""
    def build(i):
        from sympy import sqrt
        data = _sp_data(sample(i))
        n = len(data)

        def stddev():
            total = sum(data)
            return sqrt((sum(x * x for x in data) - total * total / n) / (n - ddof), evaluate=False)
        return stddev
    return build


def _sp_covariance(sample_x, sample_y, ddof=0):
    def build(i):
        x, y = _sp_data(sample_x(i)), _sp_data(sample_y(i))
        n = len(x)

        def covariance():
            return (sum(a * b for a, b in zip(x, y)) - sum(x) * sum(y) / n) / (n - ddof)
        return covariance
    return build


# ---------------------------------------------------------------------------------------------------------------------
# MathForge builders: variant i -> (public value function, args, kwargs).


def _mf_bernoulli(parameter):
    return lambda i: (mf.bernoulli, (R(*parameter(i)),), {})


def _mf_binomial(parameters):
    def build(i):
        n, p = parameters(i)
        return mf.binomial_distribution, (n, R(*p)), {}
    return build


def _mf_uniform(outcomes):
    return lambda i: (mf.discrete_uniform, (_mf_pairs(outcomes(i)),), {})


def _mf_hypergeometric(urn):
    return lambda i: (mf.hypergeometric, urn(i), {})


def _mf_on_distribution(function, pmf, extra=None, convert=None):
    """function(distribution[, extra]): the distribution is built from its PMF; extra goes through convert."""
    def build(i):
        arguments = (_mf_distribution(pmf(i)),)
        if extra is not None:
            arguments += (convert(extra(i)),)
        return function, arguments, {}
    return build


def _mf_sample(function, *samples, **kwargs):
    return lambda i: (function, tuple(_mf_pairs(sample(i)) for sample in samples), dict(kwargs))


FIXTURES = (
    # bernoulli -------------------------------------------------------------------------------------------------------
    _fixture("bernoulli", "micro", "p_over_1031", _mf_bernoulli(_p_micro), _sp_bernoulli(_p_micro), _same_pmf,
             variants=MICRO),
    _fixture("bernoulli", "realistic", "p_21_digits", _mf_bernoulli(lambda i: (10 ** 20 + 7, 3 * 10 ** 20 + 11)),
             _sp_bernoulli(lambda i: (10 ** 20 + 7, 3 * 10 ** 20 + 11)), _same_pmf),
    # binomial_distribution -------------------------------------------------------------------------------------------
    _fixture("binomial_distribution", "micro", "small_n", _mf_binomial(_binomial_small), _sp_binomial(_binomial_small),
             _same_pmf, variants=MICRO),
    _fixture("binomial_distribution", "realistic", "n40", _mf_binomial(lambda i: (40, (2, 7))),
             _sp_binomial(lambda i: (40, (2, 7))), _same_pmf, v07="binomial_pmf"),
    _fixture("binomial_distribution", "realistic", "n100", _mf_binomial(lambda i: (100, (2, 7))),
             _sp_binomial(lambda i: (100, (2, 7))), _same_pmf),
    # discrete_uniform ------------------------------------------------------------------------------------------------
    _fixture("discrete_uniform", "micro", "four_outcomes", _mf_uniform(_four_outcomes), _sp_uniform(_four_outcomes),
             _same_pmf, variants=MICRO),
    _fixture("discrete_uniform", "realistic", "fifty_outcomes", _mf_uniform(_fifty_outcomes),
             _sp_uniform(_fifty_outcomes), _same_pmf),
    _fixture("discrete_uniform", "realistic", "two_hundred_outcomes", _mf_uniform(_two_hundred_outcomes),
             _sp_uniform(_two_hundred_outcomes), _same_pmf),
    # hypergeometric --------------------------------------------------------------------------------------------------
    _fixture("hypergeometric", "micro", "small_urn", _mf_hypergeometric(_urn), _sp_hypergeometric(_urn), _same_pmf,
             variants=MICRO),
    _fixture("hypergeometric", "realistic", "urn_200", _mf_hypergeometric(lambda i: (200, 80, 40)),
             _sp_hypergeometric(lambda i: (200, 80, 40)), _same_pmf),
    _fixture("hypergeometric", "realistic", "urn_600", _mf_hypergeometric(lambda i: (600, 240, 100)),
             _sp_hypergeometric(lambda i: (600, 240, 100)), _same_pmf),
    # probability -----------------------------------------------------------------------------------------------------
    _fixture("probability", "micro", "five_outcomes",
             _mf_on_distribution(mf.probability, _micro_pmf, _micro_event, _mf_pairs),
             _sp_probability_equalities(_micro_pmf, _micro_event), _same_rational, variants=MICRO),
    _fixture("probability", "realistic", "binomial41",
             _mf_on_distribution(mf.probability, lambda i: _binomial41_pmf(), _binomial41_event, _mf_pairs),
             _sp_probability_pspace(lambda i: _binomial41_pmf(), _binomial41_event), _same_rational),
    _fixture("probability", "realistic", "outcomes_120",
             _mf_on_distribution(mf.probability, lambda i: _scaled_pmf(), _scaled_event, _mf_pairs),
             _sp_probability_pspace(lambda i: _scaled_pmf(), _scaled_event), _same_rational),
    # condition -------------------------------------------------------------------------------------------------------
    _fixture("condition", "micro", "five_outcomes",
             _mf_on_distribution(mf.condition, _micro_pmf, _micro_event_pair, _mf_pairs),
             _sp_condition_equalities(_micro_pmf, _micro_event_pair), _same_pmf, variants=MICRO),
    _fixture("condition", "realistic", "binomial41",
             _mf_on_distribution(mf.condition, lambda i: _binomial41_pmf(), _binomial41_event, _mf_pairs),
             _sp_condition(lambda i: _binomial41_pmf(), _binomial41_event), _same_pmf),
    _fixture("condition", "realistic", "outcomes_120",
             _mf_on_distribution(mf.condition, lambda i: _scaled_pmf(), _scaled_event, _mf_pairs),
             _sp_condition(lambda i: _scaled_pmf(), _scaled_event), _same_pmf),
    # cdf -------------------------------------------------------------------------------------------------------------
    _fixture("cdf", "micro", "five_outcomes", _mf_on_distribution(mf.cdf, _micro_pmf, _micro_point, _mf_point),
             _sp_cdf(_micro_pmf, _micro_point), _same_rational, variants=MICRO),
    _fixture("cdf", "realistic", "binomial41",
             _mf_on_distribution(mf.cdf, lambda i: _binomial41_pmf(), lambda i: (12, 1), _mf_point),
             _sp_cdf(lambda i: _binomial41_pmf(), lambda i: (12, 1)), _same_rational),
    _fixture("cdf", "realistic", "outcomes_120",
             _mf_on_distribution(mf.cdf, lambda i: _scaled_pmf(), lambda i: (60, 7), _mf_point),
             _sp_cdf_pspace(lambda i: _scaled_pmf(), lambda i: (60, 7)), _same_rational),
    # moment ----------------------------------------------------------------------------------------------------------
    _fixture("moment", "micro", "five_outcomes", _mf_on_distribution(mf.moment, _micro_pmf, _micro_order, int),
             _sp_moment(_micro_pmf, _micro_order), _same_rational, variants=MICRO),
    _fixture("moment", "realistic", "binomial41",
             _mf_on_distribution(mf.moment, lambda i: _binomial41_pmf(), lambda i: 3, int),
             _sp_moment(lambda i: _binomial41_pmf(), lambda i: 3), _same_rational),
    _fixture("moment", "realistic", "outcomes_120",
             _mf_on_distribution(mf.moment, lambda i: _scaled_pmf(), lambda i: 4, int),
             _sp_moment(lambda i: _scaled_pmf(), lambda i: 4), _same_rational),
    # expectation -----------------------------------------------------------------------------------------------------
    _fixture("expectation", "micro", "five_outcomes", _mf_on_distribution(mf.expectation, _micro_pmf),
             _sp_expectation(_micro_pmf), _same_rational, variants=MICRO),
    _fixture("expectation", "realistic", "binomial41",
             _mf_on_distribution(mf.expectation, lambda i: _binomial41_pmf()),
             _sp_expectation(lambda i: _binomial41_pmf()), _same_rational),
    _fixture("expectation", "realistic", "outcomes_120",
             _mf_on_distribution(mf.expectation, lambda i: _scaled_pmf()),
             _sp_expectation(lambda i: _scaled_pmf()), _same_rational),
    # distribution_variance -------------------------------------------------------------------------------------------
    _fixture("distribution_variance", "micro", "five_outcomes",
             _mf_on_distribution(mf.distribution_variance, _micro_pmf), _sp_distribution_variance(_micro_pmf),
             _same_rational, variants=MICRO),
    _fixture("distribution_variance", "realistic", "finite_variance",
             _mf_on_distribution(mf.distribution_variance, lambda i: _eight_pmf()),
             _sp_distribution_variance(lambda i: _eight_pmf()), _same_rational, v07="finite_variance"),
    _fixture("distribution_variance", "realistic", "outcomes_120",
             _mf_on_distribution(mf.distribution_variance, lambda i: _scaled_pmf()),
             _sp_distribution_variance(lambda i: _scaled_pmf()), _same_rational),
    # mean ------------------------------------------------------------------------------------------------------------
    _fixture("mean", "micro", "five_values", _mf_sample(mf.mean, _small_sample), _sp_mean(_small_sample),
             _same_rational, variants=MICRO),
    _fixture("mean", "realistic", "values_200", _mf_sample(mf.mean, _values_200), _sp_mean(_values_200),
             _same_rational),
    _fixture("mean", "realistic", "values_500", _mf_sample(mf.mean, _values_500), _sp_mean(_values_500),
             _same_rational),
    # median ----------------------------------------------------------------------------------------------------------
    _fixture("median", "micro", "five_values", _mf_sample(mf.median, _small_sample), _sp_median(_small_sample),
             _same_rational, variants=MICRO),
    _fixture("median", "realistic", "values_200", _mf_sample(mf.median, _values_200), _sp_median(_values_200),
             _same_rational),
    _fixture("median", "realistic", "values_500", _mf_sample(mf.median, _values_500), _sp_median(_values_500),
             _same_rational),
    # variance (ddof = 1, the sample variance, for the realistic and scaled samples) ----------------------------------
    _fixture("variance", "micro", "five_values", _mf_sample(mf.variance, _small_sample), _sp_variance(_small_sample),
             _same_rational, variants=MICRO),
    _fixture("variance", "realistic", "values_200", _mf_sample(mf.variance, _values_200, ddof=1),
             _sp_variance(_values_200, 1), _same_rational),
    _fixture("variance", "realistic", "values_500", _mf_sample(mf.variance, _values_500, ddof=1),
             _sp_variance(_values_500, 1), _same_rational),
    # stddev ----------------------------------------------------------------------------------------------------------
    _fixture("stddev", "micro", "five_values", _mf_sample(mf.stddev, _small_sample), _sp_stddev(_small_sample),
             _same_sqrt, variants=MICRO),
    _fixture("stddev", "realistic", "values_200", _mf_sample(mf.stddev, _values_200, ddof=1),
             _sp_stddev(_values_200, 1), _same_sqrt),
    _fixture("stddev", "realistic", "values_500", _mf_sample(mf.stddev, _values_500, ddof=1),
             _sp_stddev(_values_500, 1), _same_sqrt),
    # covariance ------------------------------------------------------------------------------------------------------
    _fixture("covariance", "micro", "five_pairs", _mf_sample(mf.covariance, _small_sample, _small_sample_y),
             _sp_covariance(_small_sample, _small_sample_y), _same_rational, variants=MICRO),
    _fixture("covariance", "realistic", "pairs_200", _mf_sample(mf.covariance, _values_200, _values_200_y, ddof=1),
             _sp_covariance(_values_200, _values_200_y, 1), _same_rational),
    _fixture("covariance", "realistic", "pairs_500", _mf_sample(mf.covariance, _values_500, _values_500_y, ddof=1),
             _sp_covariance(_values_500, _values_500_y, 1), _same_rational),
)


# ---------------------------------------------------------------------------------------------------------------------
# Edge cases: never timed. For every operation: the four type edges in the first argument, the error classes the
# operation can raise (v0.7 has invalid_input and resource_limit_exceeded; none of these operations raises
# unsupported_operation), one input far over the default limits (twice a limit or more), one explicit limit that the
# input exceeds, and the borders of each domain.


def _case(operation, label, function, *args, **kwargs):
    return EdgeCase(operation, label, lambda: (function, args, kwargs))


def _large_case(operation, label, function, *makers, **kwargs):
    """An edge case whose arguments are built when it runs, so that a 300,000-element sample is not kept in memory."""
    return EdgeCase(operation, label, lambda: (function, tuple(make() for make in makers), kwargs))


def _over_nodes():
    return tuple(range(300000))  # three times max_nodes (100,000)


def _type_edges(operation, function, *rest):
    return tuple(EdgeCase(operation, f"type_{name}", lambda v=value: (function, (v, *rest), {}))
                 for name, value in (("true", True), ("float", 1.5), ("str", "1"), ("none", None)))


def _limit(**limits):
    return {"limits": mf.ComputationLimits(**limits)}


def _dist(outcomes, weights):
    """A MathForge distribution from lists of (numerator, denominator) pairs, built when the edge case runs."""
    return mf.FiniteDistribution(_mf_pairs(outcomes), _mf_pairs(weights))


def _die():
    return _dist(((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)), ((1, 6),) * 6)


def _skewed():
    return _dist(((-1, 2), (0, 1), (1, 3), (2, 1)), ((1, 10), (2, 10), (3, 10), (4, 10)))


def _huge_outcomes():
    """Two outcomes 1/a and 1/b of about 65,000 bits each, probability 1/2 each: every input is within the default
    limits, but the mean and the variance have denominators of about twice max_integer_bits."""
    a, b = (1 << 65000) + 1, (1 << 65000) + 3
    return mf.FiniteDistribution((R(1, b), R(1, a)), (R(1, 2), R(1, 2)))


def _on(function, make, *rest, **kwargs):
    """An edge-case body that builds its distribution when it runs: (function, (distribution, *rest), kwargs).

    An argument after the distribution that is a function is called when the case runs (a large event).
    """
    return lambda: (function, (make(), *(r() if callable(r) else r for r in rest)), kwargs)


def _dist_case(operation, label, function, make, *rest, **kwargs):
    return EdgeCase(operation, label, _on(function, make, *rest, **kwargs))


EDGE_CASES = (
    # bernoulli
    *_type_edges("bernoulli", mf.bernoulli),
    _case("bernoulli", "p_above_one", mf.bernoulli, R(3, 2)),
    _case("bernoulli", "p_negative", mf.bernoulli, R(-1, 2)),
    _case("bernoulli", "p_int_two", mf.bernoulli, 2),
    _case("bernoulli", "p_zero", mf.bernoulli, R(0)),
    _case("bernoulli", "p_one", mf.bernoulli, R(1)),
    _case("bernoulli", "p_int_zero", mf.bernoulli, 0),
    _case("bernoulli", "p_int_one", mf.bernoulli, 1),
    _case("bernoulli", "p_half", mf.bernoulli, R(1, 2)),
    _case("bernoulli", "over_limit", mf.bernoulli, R(1, BIG)),
    _case("bernoulli", "explicit_limits", mf.bernoulli, R(2, 7), **_limit(**TINY)),
    # binomial_distribution
    *_type_edges("binomial_distribution", mf.binomial_distribution, R(2, 7)),
    _case("binomial_distribution", "type_p_str", mf.binomial_distribution, 4, "1"),
    _case("binomial_distribution", "type_p_float", mf.binomial_distribution, 4, 0.5),
    _case("binomial_distribution", "type_p_none", mf.binomial_distribution, 4, None),
    _case("binomial_distribution", "n_negative", mf.binomial_distribution, -1, R(1, 2)),
    _case("binomial_distribution", "p_above_one", mf.binomial_distribution, 4, R(3, 2)),
    _case("binomial_distribution", "p_negative", mf.binomial_distribution, 4, R(-1, 2)),
    _case("binomial_distribution", "n_zero", mf.binomial_distribution, 0, R(1, 2)),
    _case("binomial_distribution", "p_zero", mf.binomial_distribution, 4, R(0)),
    _case("binomial_distribution", "p_one", mf.binomial_distribution, 4, R(1)),
    _case("binomial_distribution", "p_int_one", mf.binomial_distribution, 4, 1),
    _case("binomial_distribution", "over_limit", mf.binomial_distribution, BIG, R(1, 2)),
    _case("binomial_distribution", "over_limit_nodes", mf.binomial_distribution, 300000, R(1, 2)),
    _case("binomial_distribution", "explicit_limits", mf.binomial_distribution, 4, R(2, 7), **_limit(**TINY)),
    # discrete_uniform
    *_type_edges("discrete_uniform", mf.discrete_uniform),
    _case("discrete_uniform", "empty", mf.discrete_uniform, ()),
    _case("discrete_uniform", "duplicates", mf.discrete_uniform, (1, 1, 2)),
    _case("discrete_uniform", "equal_rationals", mf.discrete_uniform, (R(1, 2), R(2, 4))),
    _case("discrete_uniform", "single", mf.discrete_uniform, (5,)),
    _case("discrete_uniform", "unsorted", mf.discrete_uniform, (3, 1, 2)),
    _case("discrete_uniform", "mixed_int_rational", mf.discrete_uniform, (1, R(1, 2), -2)),
    _case("discrete_uniform", "list_input", mf.discrete_uniform, [1, 2, 3]),
    _case("discrete_uniform", "set_input", mf.discrete_uniform, {1, 2, 3}),
    _case("discrete_uniform", "dict_input", mf.discrete_uniform, {1: 2}),
    _case("discrete_uniform", "bytes_input", mf.discrete_uniform, b"12"),
    _case("discrete_uniform", "float_item", mf.discrete_uniform, (1.5, 2)),
    _case("discrete_uniform", "bool_item", mf.discrete_uniform, (True, 2)),
    _case("discrete_uniform", "str_item", mf.discrete_uniform, ("1", 2)),
    _large_case("discrete_uniform", "over_limit", mf.discrete_uniform, _over_nodes),
    _case("discrete_uniform", "over_limit_bits", mf.discrete_uniform, (BIG, 1)),
    _case("discrete_uniform", "explicit_limits", mf.discrete_uniform, (1, 2, 3), **_limit(**TINY)),
    # hypergeometric
    *_type_edges("hypergeometric", mf.hypergeometric, 4, 3),
    _case("hypergeometric", "type_second_str", mf.hypergeometric, 10, "1", 3),
    _case("hypergeometric", "type_third_float", mf.hypergeometric, 10, 4, 1.5),
    _case("hypergeometric", "successes_above_population", mf.hypergeometric, 5, 6, 2),
    _case("hypergeometric", "draws_above_population", mf.hypergeometric, 5, 2, 6),
    _case("hypergeometric", "population_negative", mf.hypergeometric, -1, 0, 0),
    _case("hypergeometric", "empty_urn", mf.hypergeometric, 0, 0, 0),
    _case("hypergeometric", "no_draws", mf.hypergeometric, 10, 4, 0),
    _case("hypergeometric", "no_successes", mf.hypergeometric, 10, 0, 3),
    _case("hypergeometric", "all_successes", mf.hypergeometric, 10, 10, 3),
    _case("hypergeometric", "draw_everything", mf.hypergeometric, 10, 4, 10),
    _case("hypergeometric", "over_limit", mf.hypergeometric, BIG, 4, 3),
    _case("hypergeometric", "over_limit_nodes", mf.hypergeometric, 10 ** 6, 5 * 10 ** 5, 3 * 10 ** 5),
    _case("hypergeometric", "explicit_limits", mf.hypergeometric, 10, 4, 3, **_limit(**TINY)),
    # probability and condition: a distribution and an event (an iterable of outcomes)
    *_type_edges("probability", mf.probability, (1, 2)),
    _dist_case("probability", "event_str", mf.probability, _die, "1"),
    _dist_case("probability", "event_none", mf.probability, _die, None),
    _dist_case("probability", "event_float", mf.probability, _die, 1.5),
    _dist_case("probability", "event_int", mf.probability, _die, 3),
    _dist_case("probability", "event_float_item", mf.probability, _die, (1.5,)),
    _dist_case("probability", "event_bool_item", mf.probability, _die, (True,)),
    _case("probability", "distribution_dict", mf.probability, {1: 1}, (1,)),
    _case("probability", "distribution_tuple", mf.probability, ((1, 1),), (1,)),
    _dist_case("probability", "event_empty", mf.probability, _die, ()),
    _dist_case("probability", "event_outside_support", mf.probability, _die, (7, 8)),
    _dist_case("probability", "event_whole_support", mf.probability, _die, (1, 2, 3, 4, 5, 6)),
    _dist_case("probability", "event_repeated_items", mf.probability, _die, (2, 2, 2)),
    _dist_case("probability", "event_rationals", mf.probability, _skewed, (R(-1, 2), 1)),
    _dist_case("probability", "event_as_list", mf.probability, _die, [1, 2]),
    _dist_case("probability", "event_as_set", mf.probability, _die, {1, 2}),
    _dist_case("probability", "over_limit", mf.probability, _die, _over_nodes),
    _dist_case("probability", "over_limit_bits", mf.probability, _die, (BIG,)),
    _dist_case("probability", "explicit_limits", mf.probability, _die, (1, 2), **_limit(**TINY)),
    *_type_edges("condition", mf.condition, (1, 2)),
    _dist_case("condition", "event_str", mf.condition, _die, "1"),
    _dist_case("condition", "event_none", mf.condition, _die, None),
    _dist_case("condition", "event_int", mf.condition, _die, 3),
    _dist_case("condition", "event_float_item", mf.condition, _die, (1.5,)),
    _case("condition", "distribution_dict", mf.condition, {1: 1}, (1,)),
    _dist_case("condition", "event_empty", mf.condition, _die, ()),
    _dist_case("condition", "event_outside_support", mf.condition, _die, (7, 8)),
    _dist_case("condition", "event_whole_support", mf.condition, _die, (1, 2, 3, 4, 5, 6)),
    _dist_case("condition", "event_single", mf.condition, _die, (4,)),
    _dist_case("condition", "event_even", mf.condition, _die, (2, 4, 6)),
    _dist_case("condition", "event_rationals", mf.condition, _skewed, (R(-1, 2), 1, 2)),
    _dist_case("condition", "over_limit", mf.condition, _die, _over_nodes),
    _dist_case("condition", "over_limit_bits", mf.condition, _die, (BIG,)),
    _dist_case("condition", "explicit_limits", mf.condition, _die, (1, 2), **_limit(**TINY)),
    # cdf
    *_type_edges("cdf", mf.cdf, 3),
    _dist_case("cdf", "x_str", mf.cdf, _die, "1"),
    _dist_case("cdf", "x_float", mf.cdf, _die, 1.5),
    _dist_case("cdf", "x_none", mf.cdf, _die, None),
    _dist_case("cdf", "x_bool", mf.cdf, _die, True),
    _case("cdf", "distribution_dict", mf.cdf, {1: 1}, 3),
    _dist_case("cdf", "x_below_support", mf.cdf, _die, 0),
    _dist_case("cdf", "x_above_support", mf.cdf, _die, 99),
    _dist_case("cdf", "x_inside_support", mf.cdf, _die, R(7, 2)),
    _dist_case("cdf", "x_at_outcome", mf.cdf, _die, 3),
    _dist_case("cdf", "x_negative_rational", mf.cdf, _skewed, R(-1, 3)),
    _dist_case("cdf", "over_limit", mf.cdf, _die, BIG),
    _dist_case("cdf", "over_limit_negative", mf.cdf, _die, -BIG),
    _dist_case("cdf", "explicit_limits", mf.cdf, _die, 3, **_limit(**TINY)),
    # moment
    *_type_edges("moment", mf.moment, 2),
    _dist_case("moment", "order_str", mf.moment, _die, "1"),
    _dist_case("moment", "order_float", mf.moment, _die, 1.5),
    _dist_case("moment", "order_none", mf.moment, _die, None),
    _dist_case("moment", "order_bool", mf.moment, _die, True),
    _dist_case("moment", "order_rational", mf.moment, _die, R(2)),
    _case("moment", "distribution_dict", mf.moment, {1: 1}, 2),
    _dist_case("moment", "order_negative", mf.moment, _die, -1),
    _dist_case("moment", "order_zero", mf.moment, _die, 0),
    _dist_case("moment", "order_zero_with_zero_outcome", mf.moment, _skewed, 0),
    _dist_case("moment", "order_one", mf.moment, _die, 1),
    _dist_case("moment", "order_ten", mf.moment, _skewed, 10),
    _dist_case("moment", "over_limit", mf.moment, _die, BIG),
    _dist_case("moment", "over_limit_output", mf.moment, _die, 100000),
    _dist_case("moment", "explicit_limits", mf.moment, _die, 2, **_limit(**TINY)),
    # expectation and distribution_variance take only a distribution, so the default limits are exceeded through
    # the result: every input is within them, the mean and the variance are not
    *_type_edges("expectation", mf.expectation),
    _case("expectation", "distribution_dict", mf.expectation, {1: 1}),
    _dist_case("expectation", "die", mf.expectation, _die),
    _dist_case("expectation", "skewed", mf.expectation, _skewed),
    _dist_case("expectation", "single_outcome", mf.expectation, lambda: _dist(((5, 3),), ((1, 1),))),
    _dist_case("expectation", "over_limit", mf.expectation, _huge_outcomes),
    _dist_case("expectation", "explicit_limits", mf.expectation, _die, **_limit(**TINY)),
    *_type_edges("distribution_variance", mf.distribution_variance),
    _case("distribution_variance", "distribution_dict", mf.distribution_variance, {1: 1}),
    _dist_case("distribution_variance", "die", mf.distribution_variance, _die),
    _dist_case("distribution_variance", "skewed", mf.distribution_variance, _skewed),
    _dist_case("distribution_variance", "single_outcome", mf.distribution_variance,
               lambda: _dist(((5, 3),), ((1, 1),))),
    _dist_case("distribution_variance", "over_limit", mf.distribution_variance, _huge_outcomes),
    _dist_case("distribution_variance", "explicit_limits", mf.distribution_variance, _die, **_limit(**TINY)),
    # mean, median
    *_type_edges("mean", mf.mean),
    _case("mean", "empty", mf.mean, ()),
    _case("mean", "single", mf.mean, (R(7, 3),)),
    _case("mean", "ints", mf.mean, (1, 2, 4)),
    _case("mean", "mixed_int_rational", mf.mean, (1, R(1, 2), -2)),
    _case("mean", "list_input", mf.mean, [1, 2, 4]),
    _case("mean", "set_input", mf.mean, {1, 2, 4}),
    _case("mean", "dict_input", mf.mean, {1: 2}),
    _case("mean", "bytes_input", mf.mean, b"12"),
    _case("mean", "float_item", mf.mean, (1.5, 2)),
    _case("mean", "bool_item", mf.mean, (True, 2)),
    _case("mean", "str_item", mf.mean, ("1", 2)),
    _large_case("mean", "over_limit", mf.mean, _over_nodes),
    _case("mean", "over_limit_bits", mf.mean, (BIG,)),
    _case("mean", "explicit_limits", mf.mean, (1, 2, 4), **_limit(**TINY)),
    *_type_edges("median", mf.median),
    _case("median", "empty", mf.median, ()),
    _case("median", "single", mf.median, (R(7, 3),)),
    _case("median", "odd", mf.median, (5, 1, 4, 2, 3)),
    _case("median", "even", mf.median, (4, 1, 3, 2)),
    _case("median", "even_rationals", mf.median, (R(1, 2), R(1, 3), R(1, 5), R(1, 7))),
    _case("median", "duplicates", mf.median, (2, 2, 2, 9, 9)),
    _case("median", "all_equal", mf.median, (3, 3, 3, 3)),
    _case("median", "list_input", mf.median, [3, 1, 2]),
    _case("median", "float_item", mf.median, (1.5, 2)),
    _case("median", "bool_item", mf.median, (True, 2)),
    _large_case("median", "over_limit", mf.median, _over_nodes),
    _case("median", "over_limit_bits", mf.median, (BIG, 1)),
    _case("median", "explicit_limits", mf.median, (1, 2, 4), **_limit(**TINY)),
    # variance, stddev
    *_type_edges("variance", mf.variance, 0),
    _case("variance", "empty", mf.variance, ()),
    _case("variance", "ddof_str", mf.variance, (1, 2, 4), "1"),
    _case("variance", "ddof_float", mf.variance, (1, 2, 4), 1.5),
    _case("variance", "ddof_none", mf.variance, (1, 2, 4), None),
    _case("variance", "ddof_bool", mf.variance, (1, 2, 4), True),
    _case("variance", "ddof_negative", mf.variance, (1, 2, 4), -1),
    _case("variance", "ddof_equals_n", mf.variance, (1, 2, 4), 3),
    _case("variance", "ddof_above_n", mf.variance, (1, 2, 4), 4),
    _case("variance", "single_ddof_zero", mf.variance, (5,), 0),
    _case("variance", "single_ddof_one", mf.variance, (5,), 1),
    _case("variance", "population", mf.variance, (1, 2, 4), 0),
    _case("variance", "sample", mf.variance, (1, 2, 4), 1),
    _case("variance", "default_ddof", mf.variance, (1, 2, 4)),
    _case("variance", "rationals", mf.variance, (R(1, 2), R(1, 3), 4), 1),
    _case("variance", "constant", mf.variance, (3, 3, 3)),
    _large_case("variance", "over_limit", mf.variance, _over_nodes),
    _case("variance", "over_limit_bits", mf.variance, (BIG, 1)),
    _case("variance", "explicit_limits", mf.variance, (1, 2, 4), 0, **_limit(**TINY)),
    *_type_edges("stddev", mf.stddev, 0),
    _case("stddev", "empty", mf.stddev, ()),
    _case("stddev", "ddof_str", mf.stddev, (1, 2, 4), "1"),
    _case("stddev", "ddof_float", mf.stddev, (1, 2, 4), 1.5),
    _case("stddev", "ddof_bool", mf.stddev, (1, 2, 4), True),
    _case("stddev", "ddof_negative", mf.stddev, (1, 2, 4), -1),
    _case("stddev", "ddof_equals_n", mf.stddev, (1, 2, 4), 3),
    _case("stddev", "single_ddof_zero", mf.stddev, (5,), 0),
    _case("stddev", "single_ddof_one", mf.stddev, (5,), 1),
    _case("stddev", "irrational", mf.stddev, (1, 2, 4), 0),
    _case("stddev", "perfect_square", mf.stddev, (1, 3), 0),
    _case("stddev", "perfect_square_rational", mf.stddev, (R(1, 2), R(3, 2)), 0),
    _case("stddev", "constant", mf.stddev, (3, 3, 3)),
    _case("stddev", "sample", mf.stddev, (1, 2, 4), 1),
    _large_case("stddev", "over_limit", mf.stddev, _over_nodes),
    _case("stddev", "over_limit_bits", mf.stddev, (BIG, 1)),
    _case("stddev", "explicit_limits", mf.stddev, (1, 2, 4), 0, **_limit(**TINY)),
    # covariance
    *_type_edges("covariance", mf.covariance, (1, 2, 4), 0),
    _case("covariance", "y_str", mf.covariance, (1, 2, 4), "1"),
    _case("covariance", "y_none", mf.covariance, (1, 2, 4), None),
    _case("covariance", "y_float", mf.covariance, (1, 2, 4), 1.5),
    _case("covariance", "ddof_str", mf.covariance, (1, 2, 4), (2, 3, 7), "1"),
    _case("covariance", "ddof_bool", mf.covariance, (1, 2, 4), (2, 3, 7), True),
    _case("covariance", "ddof_negative", mf.covariance, (1, 2, 4), (2, 3, 7), -1),
    _case("covariance", "ddof_equals_n", mf.covariance, (1, 2, 4), (2, 3, 7), 3),
    _case("covariance", "both_empty", mf.covariance, (), ()),
    _case("covariance", "first_empty", mf.covariance, (), (1, 2)),
    _case("covariance", "length_mismatch", mf.covariance, (1, 2, 4), (2, 3)),
    _case("covariance", "single", mf.covariance, (5,), (7,), 0),
    _case("covariance", "single_ddof_one", mf.covariance, (5,), (7,), 1),
    _case("covariance", "population", mf.covariance, (1, 2, 4), (2, 3, 7), 0),
    _case("covariance", "sample", mf.covariance, (1, 2, 4), (2, 3, 7), 1),
    _case("covariance", "with_itself", mf.covariance, (1, 2, 4), (1, 2, 4), 0),
    _case("covariance", "anticorrelated", mf.covariance, (1, 2, 3), (3, 2, 1), 0),
    _case("covariance", "constant_y", mf.covariance, (1, 2, 4), (5, 5, 5), 0),
    _case("covariance", "rationals", mf.covariance, (R(1, 2), R(1, 3), 4), (R(2, 5), 1, R(-3, 7))),
    _large_case("covariance", "over_limit", mf.covariance, lambda: tuple(range(250000)), lambda: (1, 2, 3)),
    _case("covariance", "over_limit_bits", mf.covariance, (BIG, 1), (1, 2)),
    _case("covariance", "explicit_limits", mf.covariance, (1, 2, 4), (2, 3, 7), 0, **_limit(**TINY)),
    # explicit limits of the wrong type are an invalid input; a raised limit admits what the defaults refuse
    _case("bernoulli", "type_limits_int", mf.bernoulli, R(2, 7), limits=3),
    _case("binomial_distribution", "type_limits_int", mf.binomial_distribution, 4, R(2, 7), limits=3),
    _case("discrete_uniform", "type_limits_int", mf.discrete_uniform, (1, 2, 3), limits=3),
    _case("hypergeometric", "type_limits_int", mf.hypergeometric, 10, 4, 3, limits=3),
    _dist_case("probability", "type_limits_int", mf.probability, _die, (1, 2), limits=3),
    _dist_case("condition", "type_limits_int", mf.condition, _die, (1, 2), limits=3),
    _dist_case("cdf", "type_limits_int", mf.cdf, _die, 3, limits=3),
    _dist_case("moment", "type_limits_int", mf.moment, _die, 2, limits=3),
    _dist_case("expectation", "type_limits_int", mf.expectation, _die, limits=3),
    _dist_case("distribution_variance", "type_limits_int", mf.distribution_variance, _die, limits=3),
    _case("mean", "type_limits_int", mf.mean, (1, 2, 4), limits=3),
    _case("median", "type_limits_int", mf.median, (1, 2, 4), limits=3),
    _case("variance", "type_limits_int", mf.variance, (1, 2, 4), 0, limits=3),
    _case("stddev", "type_limits_int", mf.stddev, (1, 2, 4), 0, limits=3),
    _case("covariance", "type_limits_int", mf.covariance, (1, 2, 4), (2, 3, 7), 0, limits=3),
    _dist_case("cdf", "raised_limits", mf.cdf, _die, 1 << 70000, **_limit(max_integer_bits=80000)),
    _dist_case("probability", "raised_limits", mf.probability, _die, (1 << 70000,), **_limit(max_integer_bits=80000)),
)
