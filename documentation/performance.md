# Measure exact computation costs

## v0.7 resource gates

Ten new deterministic cases cover integer certificates, combinatorics, complex
square roots, probability, statistics, LU, elementary calculus and LRT primitives.
The gate includes independent verification and JSON roundtrips, and checks work,
representation nodes, evidence items and encoded bytes against explicit ceilings.
It also requires the expected deferred and resource-exhausted statuses. The
[measured counters](../benchmarks/v07-mathematics.json) are reproducible with
`python tools/check_benchmarks.py`; they are not wall-clock promises. Historical
v0.2 and v0.6 baselines remain unchanged.

## v0.6 exact algebra gates

The [v0.6 report](../benchmarks/v06-exact-algebra.json) records 21 cases spanning
factorization, resultants, algebraic arithmetic and evaluation, partial fractions,
algebraic calculus, extrema, characteristic polynomials and eigenvalues, plus
analysis and multivariate regressions. It records elapsed time and peak traced
memory separately from deterministic acceptance counters. Work, visited nodes,
evidence size and canonical output bytes have explicit ceilings; operation and
verification statuses must also match. The counters include independent checking.
The original sixteen benchmark gates remain in place.

```sh
python benchmarks/exact_algebra.py --output exact-algebra-current.json
python tools/check_benchmarks.py
```

The fixed-seed differential suite uses SymPy only as a development dependency:
`python tools/check_sympy.py`. CI runs it on pull requests. It compares exact
identities and roots without decimal tolerances. These checks supplement the
independent runtime verifiers, which do not import SymPy.

## Earlier performance measurements

v0.5 reduces arithmetic in two existing producers. Sturm root isolation carries
endpoint variation counts and zero flags into child intervals, evaluating each
new midpoint once. RREF updates only the active suffix of the pivot row and its
targets; the eliminated prefix is unchanged. Independent checking and stored
row-operation evidence remain in the timed SDK path.

## Observed before and after

The same Windows/Python 3.14.6 machine ran all sixteen existing benchmark cases
three times before and three times after the changes on 14 October 2026. These
are local working-tree observations, not cross-machine guarantees. Each figure
below is the median of three single-run measurements. The original harness has
no warmup; its first, very short case is particularly sensitive to startup noise.

| Existing case | Before v0.5 changes | After | Speed ratio | Charged work reduction |
| --- | ---: | ---: | ---: | ---: |
| Degree-20 known real roots | 2702 ms | 1139 ms | 2.37x | 55.5% |
| Degree-10 known real roots | 304 ms | 157 ms | 1.94x | 45.3% |
| 32-by-32 tridiagonal RREF | 667 ms | 444 ms | 1.50x | 31.7% |
| 16-by-16 tridiagonal RREF | 102 ms | 67 ms | 1.52x | 30.1% |

Median summed time across all cases changed from 4.334 s to 2.413 s (1.80x).
The 4-by-4 RREF median increased from 4.02 ms to 6.87 ms even though its charged
work fell by 20.5%; no timing improvement is claimed for that case. Determinant
production and independent checking were not optimized. Raw runs are retained
in [the benchmark evidence](../benchmarks/v05-performance.json).

## Reproduce current measurements

Install the checkout, then run:

```sh
python -m pip install --no-deps -e .
python benchmarks/exact_core.py --output exact-core-current.json
python tools/check_benchmarks.py
python benchmarks/multivariate.py --repeat 5 --output multivariate-current.json
```

The exact-core acceptance gate keeps the frozen v0.2 baseline and checks every
expected status plus deterministic work/evidence ceilings. It does not gate
wall-clock time. Performance tests also check unchanged exact intervals and
row traces, and explicit work/refinement exhaustion.

The multivariate harness warms each case once and reports all samples,
median, nearest-rank p95 and deterministic counters. With only five samples,
p95 is the maximum observation and should not be treated as a production latency
estimate. Measured time includes the structured operation and its independent
verification, excluding parsing, presentation and serialization. A failed
operation aborts the benchmark instead of being counted as a fast result.

Sparse expansion stores nonzero monomials. A gradient shares one source
expansion; a Hessian also reuses symmetric mixed partials. This helps polynomial
workloads but does not make dense high-degree expansions cheap. Finite budgets
remain the bound on work and result size. The new operations have no historical
performance baseline; their [sample report](../benchmarks/v05-multivariate.json)
is a starting point for subsequent comparisons.
