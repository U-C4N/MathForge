# Changelog

## 0.9.0 — 2026-11-13

- Fix the deterministic Miller–Rabin test of the trusted layer: it now uses the thirteen prime bases 2 to 41. With the twelve bases 2 to 37, its stated bound of 3,317,044,064,679,887,385,961,981 admitted the composite 318,665,857,834,031,151,167,461 = 399,165,290,221 × 798,330,580,441; no shipped result was affected, because every caller stayed below `2**61`.
- Fix `powers_by_prime` of the trusted layer, which looped forever on an empty list of primes; it now returns an empty mapping.
- Add three argument kinds for v6 operations and the MCP area tools: `polynomial`, `interval_set` (at most 32 intervals in friendly JSON) and `relation` (`"="`, `"<"`, `"<="`, `">"` or `">="`); the request budget pays for decoding their envelopes.
- Add the probe scoreboard, `tools/check_probes.py`: 167 natural mathematics requests in 12 groups, each with one status such as `ok`, `unsupported` or `resource_limit`, compared in CI with the committed baseline `benchmarks/probes-v08.json`.
- Add `tools/new_family.py`, which scaffolds a v6 operation family with its checker and tests, registers it and regenerates the API and the schemas, and lists the constants that tests and documents pin (`pins --check`).
- Add shared prime helpers for the number theory families: `mathforge.certify.ntheory.factorization` proves primes and factorizations for checkers and charges every modular power before it runs, and `mathforge.kernels.certificates` builds the matching evidence and refuses evidence whose check would not fit the work budget.
- Add the cell engine, `mathforge.certify.cells`: a checker-side certified cut of the real line at rational points and at the real roots of one polynomial. The checkers of `solve_abs`, `solve_sqrt` and the four interval set operations decide their claims on it.
- Add fifteen number theory operations to `mf.ntheory` and `mathforge.operations`, each with an independent checker: `powmod`, `multiplicative_order`, `primitive_root`, `legendre_symbol`, `jacobi_symbol`, `sqrt_mod`, `next_prime`, `prev_prime`, `linear_diophantine`, `quadratic_continued_fraction`, `convergents`, `pell_fundamental`, `moebius`, `divisor_sigma` and `integer_nth_root`, with the records `DiophantineSolution` and `PeriodicContinuedFraction`. The [v0.9 mathematics guide](documentation/v09-mathematics.md) shows each of them.
- Their checkers prove primes themselves below psi_13 = 3,317,044,064,679,887,385,961,981 and with a Pocklington certificate from psi_13 on; a result whose certificate or factorization does not fit the work budget is `resource_limit_exceeded`, never an unverified value.
- Add four algebra operations to `mf.algebra`: `count_real_roots` (distinct real roots in a closed interval with rational or infinite ends), `complex_roots` (every complex root with multiplicity, when the irreducible factors have degree at most two; otherwise deferred), `interpolate` (the polynomial of least degree through points with distinct rational abscissas) and `denest_sqrt` (`sqrt(a + b*sqrt(d))` as `sqrt(x) +- sqrt(y)`, with the record `DenestedRoot`; deferred when no such form exists).
- Add `solve_abs` and `solve_sqrt` to `mf.equations`: equations and inequalities built from polynomials and absolute values of polynomials, or linear in one square root of a polynomial, solved over the reals as `EmptySet`, `UniversalSet` or an `IntervalSet` with rational or real algebraic ends.
- Add `set_union`, `set_intersection`, `set_difference` and `set_complement` to `mf.sets`, on finite unions of real intervals. `IntervalSet` gains the unchecked methods `complement()`, `difference(other)` and `symmetric_difference(other)` and the operators `|`, `&`, `-`, `^` and `~`.
- Add `digits` to `mf.numeric`: the decimal expansion of an exact real number truncated toward zero to at most 10,000 places, as the record `TruncatedDecimal`.
- Add five operations to `mf.series`: `sum_polynomial`, `geometric_sum`, `geometric_series`, `polynomial_geometric_series` and `series_coefficients` (exact power series coefficients at a rational center up to order 4,096, as the record `PowerSeries`; deferred where a coefficient would not be rational). Its checker reuses the Taylor checker's truncated series, which now charges every coefficient operation by the length of its operands, so the budget also bounds the check's time on long coefficients. Under the default limits exp(x) and cos(x) verify to order 4,096, and functions whose coefficients grow long, such as tan(x) or sqrt(1 + x), to orders of about 400.
- Add `trig_power_antiderivative` and `trig_power_integral` to `mf.integrals`, for polynomials in `sin(a*x)` and `cos(a*x)`; the integral runs between rational multiples of pi whose angles are multiples of `pi/12` and returns the record `PiLinear`.
- Add `matrix_power` and `eigenvectors` to `mf.linalg`; `eigenvectors` gives every rational eigenvalue with its algebraic multiplicity and eigenspace basis (records `RationalEigenvectors` and `Eigenspace`). `Matrix ** k` returns `matrix_power`'s value as unchecked arithmetic, like `@`, and `Matrix * Matrix` still raises `InvalidInput`, now with a message that names `@`.
- Add `dot`, `cross`, `projection` and `triple_product` to `mf.vector`, and `distance`, `polygon_area` (signed), `line_intersection` (record `LineIntersection`) and `orientation` to `mf.geometry`.
- Add `correlation` and `linear_regression` (record `LinearFit`) to `mf.stats`, and `partitions`, `derangements`, `bell` and `stirling1` (unsigned) to `mf.discrete`.
- Each of these 34 operations has an independent checker, and the [v0.9 mathematics guide](documentation/v09-mathematics.md) shows each of them with its limits.
- Add sixteen more operations, each with an independent checker: `sturm_sequence` (the canonical Sturm sequence, with the division quotients as evidence) and `cyclotomic_polynomial` (for `phi(n) <= 1024`) to `mf.algebra`; `rational_approximation` (the closest fraction with a bounded denominator, certified by Farey neighbours) to `mf.numeric`; `set_symmetric_difference` to `mf.sets`; `residue` (of a rational function at a rational point) to `mf.series`; `discrete_log` (Pohlig–Hellman modulo a prime, record `DiscreteLogarithm`) and `two_squares` (the number of representations as a sum of two squares and one of them, record `SumOfTwoSquares`) to `mf.ntheory`; `definiteness` (record `Inertia`, certified by a congruence to a diagonal matrix), `gram_schmidt`, `hermite_normal_form` and `smith_normal_form` (with unimodular transforms as evidence) to `mf.linalg`; `convex_hull` to `mf.geometry`; and `quantile` (type 7), `mode`, `stationary_distribution` (of a Markov chain with a unique one) and `fisher_exact` (the exact two-sided p-value of a 2 x 2 table) to `mf.stats`.
- Fix the interval set operations on an end written as a rational root of a reducible polynomial, such as -1 as root 0 of `x**2 - 1`: when the separation of the ends found that root exact, it dropped the point from the line, and `set_union`, `set_intersection`, `set_difference` and `set_complement` ended in an internal error (`KeyError`) instead of answering.
- The registry now holds 126 operations besides the 42 of the legacy catalogue, 168 in all, up from 103. Three MCP area tools join: `mathforge.equations`, `mathforge.geometry` (geometry and vectors) and `mathforge.numeric`, so the MCP server offers 40 tools, up from 37; `mathforge.algebra`, `mathforge.analysis`, `mathforge.discrete`, `mathforge.linalg`, `mathforge.ntheory` and `mathforge.stats` serve the other new operations.
- Add MCP result handles: every response of a computation tool names its `result` with a `handle`, `mf:sha256:` and the SHA-256 digest of the result's canonical JSON (the digest of its workspace `ObjectRef`), and the server keeps recent results in a per-process store of at most 256 documents and 64 MiB, evicting the least recently used. An area tool argument or a `mathforge.compute` operand that takes a canonical envelope also takes a handle, decoded exactly as the envelope sent in full, within the request's limits; an unknown handle is the input error `unknown_handle`. The input schemas of the thirty v0.7 tools, `mathforge.verify` included, do not change; `mathforge.compute` now reads an operand string that matches the handle pattern as a handle (before, an `invalid_input` at stage `decoding`), although its frozen schema does not list handles, so a client that validates calls against it may refuse to send one. Without `symbol_ids`, a request with variables gives new symbol identities, and so a new handle, each time.
- Every MCP error object gains `reason`, a stable snake_case code such as `unknown_operation`, `unsupported`, `resource_limit_exceeded` or `symbol_mismatch` (a handle passed on beside text without `symbol_ids`), and `next_actions`, short suggestions such as calling `mathforge.catalog`, chosen by the tool, the field and whether a limits field could help; its six other keys keep their values, and `ErrorInfo` and serialized Results do not change. The output schemas require `handle` and both error keys.
- Add `examples/v09_workflow.py`, a tested tour of the new operations, their re-verifiable results and an MCP area tool call.

Compatibility: Python 3.11–3.14 and the dependency-free core remain. v1–v6
documents keep their bytes and verdicts. The v0.7 public API and the thirty v0.7
MCP tools are unchanged, but for the additive `handle`, `reason` and
`next_actions` keys of their responses. The new operations live in their area modules, such as
`mf.ntheory` and `mf.algebra`, and in `mathforge.operations`, not in the plain
`mathforge` namespace, and their results use JSON v6.

Known limitations: factoring bounds the operations that need it. Under the
default limits the product of the primes `2**89 - 1` and `2**107 - 1` is out of
reach, and a prime from psi_13 on needs the complete factorization of `p - 1`
for its certificate. A Pell solution is refused once `x` or `y` has more than
`max_integer_bits` bits, and a periodic continued fraction once it has more than
99,000 terms or its terms have more than `max_integer_bits` bits in all.
`solve_abs` and `solve_sqrt` factor over the integers, so a square-free problem
polynomial of degree above 64 exceeds the default limits. Rational abscissas
with many distinct long denominators limit `interpolate` to about 70 points with
16-bit denominators under the default limits. Near the work limit, some values
that a producer computes need more work to check than the budget allows (long
power series coefficients, large plane polygons, some counting and statistics
inputs, the Sturm sequences of dense polynomials of degree about 55, Fisher tables
with a total from about 52,000); they end as `resource_limit_exceeded`, never as
an unverified value. The fraction-free elimination of `definiteness`,
`gram_schmidt`, `hermite_normal_form` and `stationary_distribution` refuses some
64 x 64 inputs with long entries whose check would fit, and `discrete_log`
refuses an order with a large prime factor.

## 0.8.0 — 2026-10-21

- Add a declarative operation registry; `tools/generate_api.py` generates from it the `mathforge.operations` wrappers, the value functions and 18 lazily loaded area modules, such as `mf.ntheory`.
- Keep the v0.7 names, signatures and results unchanged; `tools/generate_api.py --check` fails when a generated file is stale.
- Add mathematical JSON v6 and a generic `Unevaluated(kind, request, reason)` record for a request without a verified answer: `NO_CONCLUSION`, no value, unknown completeness.
- Run each v6 operation through a producer and an independent checker; an unexpected producer failure is `internal_error`, and a v6 result of an unknown operation verifies as `UNKNOWN`.
- Record the format version of a v6 request in `OperationRequest.version`, so a later build's v6 requests load with their content hashes intact; v1–v5 requests never carry it.
- Add the packaged JSON v6 schema; it dispatches on each record's `type` or `operation` key and rejects a crafted document in milliseconds. The v2–v5 schemas are byte-identical.
- Add MCP area tools and `mathforge.catalog`: an area tool runs one named operation with friendly JSON or canonical envelopes; `mathforge.compute` and the thirty v0.7 tools are unchanged.
- In the area tools, JSON decimals and `true`/`false` are never exact numbers; the input schemas are advisory and the server's decoding is the authority.
- MCP clients receive v6 results as JSON v6 envelopes, and `mathforge.verify` accepts versions 1 through 6; MathForge 0.7 does not read v6 documents.
- Add `continued_fraction`, the first v6 operation: `mf.ntheory.continued_fraction` and `mf.operations.continued_fraction` give the finite simple continued fraction of an exact rational.
- Its independent checker rebuilds the rational from the convergents; `tools/check_layering.py` fails when a checker can reach any module beyond a small trusted set, even through chained imports.
- Speed up exact arithmetic: `Rational` operations on exact `Rational` and `int` operands work on the integers directly, and the computation budget charges inline.
- `Rational` instances now use slots: weak references to them work as in v0.7, but `vars()` of one raises `TypeError`, because it has no attribute dict.
- Add exact kernels: integer linear algebra with modular certificates, dense integer polynomials, an integer kernel with prime certificates, and complex and quadratic-surd arithmetic.
- Check their results with new independent `mathforge.certify` checkers, complex arithmetic by a second computation, Taylor values by truncated power series and medians by a certificate.
- Add the lean value path: a value function called without `limits` runs its producer and an independent checker directly, and falls back to the structured path for any call it does not handle.
- Near a resource limit, a value function may answer where `mathforge.operations` refuses, as the kernels need less work; it never returns a different value, and errors are otherwise the same.
- Cut the fixed cost of the structured pipeline: it inspects arguments, value and evidence once and builds requests, steps and results without repeating checks. Records and bytes are unchanged.
- Add the superseding-operation mechanism: a v6 family can take over a v0.7 operation name for new results while older records keep their format. No operation uses it yet.

Compatibility: Python 3.11–3.14 and the dependency-free core remain. v1–v5
documents keep their bytes and get the same verdicts as in v0.7, except that
some checks that ran out of budget in v0.7 now complete; some checks run new,
faster checker code. Values match v0.7, and so do errors, except that some
inputs just past a resource limit may now succeed, because exact computation got
cheaper. The v0.7 public API, the thirty v0.7 MCP tools with their input schemas
and `api_version=1` are pinned by compatibility tests. Operations added in v0.8
live in their area module and in `mathforge.operations`, not in the plain
`mathforge` namespace.

Known limitations: `integer_gcd` and `integer_lcm` on very large integers, from
about 2,000 bits (RSA size) upward, remain comparatively slow, because every
result carries an independently checked Bézout certificate. `solve_system` still
uses the v0.7 structured pipeline and will be accelerated in a later version, as
will `solve` on the smallest equations, where that pipeline's fixed cost dominates.

## 0.7.0 — 2026-10-14

- Add a separate real elementary function AST with ordered axes, three-valued domain membership and restrictions collected before simplification. Existing expression parsing and real square-root behavior remain.
- Add chain-rule derivatives, gradients, Jacobians, Hessians, finite real-algebraic-center Taylor polynomials and certified local-series, growth and oscillation limit rules.
- Add Hermite/Lazard–Rioboo–Trager rational primitives with independent quotient-algebra trace checks, elementary integration rules and whole-interval domain certificates for proper finite integrals.
- Add exact complex algebraic arithmetic, principal square roots and rational quadratic roots with multiplicities.
- Add integer GCD/LCM/Bézout, modular inverses, generalized CRT, certified primality and factorization, divisors, totients, combinatorial counts and integer sequences.
- Add finite rational distributions, conditioning, moments and descriptive statistics with explicit `ddof`; add rational matrix trace, subspaces, pivoted LU and complete least-squares solution families.
- Add explicit unevaluated integral/limit records and `Result.deferred`: unresolved questions carry no mathematical value and cannot be verified as solved results.
- Add mathematical JSON v5, exact presentation and two MCP tools for parsing real functions and invoking an allowlisted operation registry; retain historical v1–v4 bytes and hashes.
- Freeze the actual local v0.6 sources and compatibility payloads; extend independent/corruption checks, typed consumers, seeded development comparisons and deterministic resource gates.

Compatibility: Python 3.11–3.14 and the dependency-free core remain. New function
operations are explicit; legacy polynomial values retain their representations.
The new rational-integral path completes previously unsupported requests.
`RationalPrimitive` is a formal derivative identity; arbitrary root-sum point
evaluation, improper/principal-value integrals, general complex root isolation,
ODE/PDE and symbolic matrix entries are outside this release. Algorithmic
verification is independent of producers, but is not a formal proof system.

## 0.6.0 — 2026-10-14

- Add complete rational polynomial factorization with deterministic modular factorization, Hensel lifting and bounded exhaustive recombination; independently certify irreducibility as well as product reconstruction.
- Add fraction-free subresultant resultants/discriminants, domain-preserving partial fractions, Berkowitz characteristic polynomials and complete real eigenvalues with algebraic multiplicity.
- Add explicit real algebraic normalization, division, signed integer powers, minimal polynomials and algebraic evaluation with checked annihilator identities and real-root selection.
- Extend rational-function limits and polynomial definite integrals to real algebraic points; provide exact extrema coordinates through a separate `extrema_values` operation.
- Preserve historical expression construction, structural equality/hashes and evaluation; add strict mathematical JSON v4 for new records and evidence while retaining v1/v2/v3 bytes and mixed-workspace references.
- Add twelve local stdio MCP tools, including canonical scalar input for chaining algebraic outputs, bringing the total to twenty-eight.
- Separate multivariate source conversion from its checker; freeze the full v0.5 source provenance and compatibility payloads before implementation.
- Expand corruption tests, seeded development-only differential checks, deterministic resource gates, typed consumers and executable documentation.

Compatibility: Python 3.11+ and the dependency-free runtime remain. Explicit
`algebraic` operations return canonical rationals or real root descriptors;
ordinary AST operators and `evaluate`/`substitute` keep their historical behavior.
Mathematical JSON v4 is distinct from package version 0.6 and MCP envelope version 1.
Resource exhaustion remains an explicit error, never a claim of irreducibility
or absence of solutions. Verification is algorithmic, not formal proof.
Complex arithmetic, general nonlinear systems, symbolic/algebraic matrix entries
and transcendental integration remain outside scope.

## 0.5.0 — 2026-10-14

- Add sparse exact multivariate polynomial expansion, arbitrary nonnegative-order partial derivatives, gradients, Jacobians and Hessians over rational coefficients.
- Share source expansion across derivative components and reuse symmetric Hessian entries while retaining finite computation budgets.
- Independently check derivative coefficient identities, source expressions, ordered axes and matrix shapes; preserve the existing expression/tuple and JSON v2 contracts.
- Add five optional local MCP tools, bringing the curated tool count to sixteen, with bounded explicit variable lists and reusable symbol identities.
- Reduce repeated Sturm endpoint evaluations and skip known zero prefixes in RREF without changing independent checkers or row-operation evidence.
- Add multivariate SDK/transport/protocol regressions, typed examples and reproducible performance observations.

Compatibility: Python 3.11+, the dependency-free core, existing wire schemas and
historical object hashes remain. Jacobians and Hessians are tuples of expression
rows, not rational `Matrix` instances. New operation names require a v0.5 checker;
older readers can decode their generic v2 envelopes but cannot certify them.
General multivariate solving, symbolic denominators, transcendental calculus and
symbolic matrix inversion remain outside scope. Performance varies by workload.

## 0.4.0 — 2026-10-11

- Add exact limits of real rational functions at rational finite points and signed infinity, including one-sided poles and explicit nonexistence values.
- Add complete real function analysis: domain, holes/poles, zero/positive/negative/stationary sets, derivatives, maximal connected monotonic regions, and strict local extrema x-coordinates.
- Preserve excluded points in derivatives and analysis; include defined critical endpoints in monotonic intervals, merge through stationary non-turning points, and keep constant regions separate across holes.
- Add independent source-linked checks and frozen typed analysis records, with exact text and LaTeX presentation.
- Add `mathforge.limit` and `mathforge.analyze_function` to the optional stdio MCP adapter, bringing its tool count to eleven.
- Add strict mathematical JSON v3 and its packaged schema while preserving existing v1/v2 object bytes, hashes and mixed-workspace references.
- Add executable analysis examples and regression coverage for limits, disconnected domains, monotonicity and transport compatibility.

Compatibility: Python 3.11+ and the dependency-free core remain. New analysis
values and requests need a v3-aware reader; existing objects retain their minimum
v1/v2 envelope. A finite or infinite limit, or `does_not_exist`, is a completed
value. Complex roots, symbolic coefficients, transcendental calculus, approximate
output and remote HTTP serving remain outside scope. Verification is algorithmic,
not formal proof.

## 0.3.0 — 2026-10-10

- Add bounded math-text parsing with explicit symbol identity, exact decimal conversion, source spans, and domain-preserving rational expressions.
- Add separate exact text and escaped LaTeX renderers for expressions, roots, matrices, intervals and solution records; existing string methods remain unchanged.
- Add nine local stdio MCP tools through the optional `mathforge[mcp]` extra, with structured results, capped budgets and explicit symbol reuse.
- Add discoverable capability metadata and generic `Result[T]` values with typed public operation boundaries.
- Preserve mathematical v1/v2 JSON, hashes and historical workspaces, with frozen v0.2 fixtures in addition to v0.1 compatibility tests.
- Publish executable user documentation and expand base/extras packaging, protocol and deterministic benchmark release gates.

Compatibility: Python 3.11+ and the dependency-free core are retained. The MCP
adapter is optional. Presentation metadata is outside mathematical JSON; no
new JSON format version, numerical fallback or remote server is introduced.

## 0.2.0 — 2026-10-09

- Add exact polynomial arithmetic, division, gcd/Bézout identities, and square-free decomposition.
- Solve general rational-coefficient real polynomial equations using signed Sturm chains, rational reconstruction, algebraic root descriptors, and ordered multiplicities.
- Add domain-preserving rational functions, rational equations, and polynomial/rational inequalities with exact interval unions.
- Add immutable rational matrices, RREF, rank, Bareiss determinants, inverse, nullspace, and complete affine solutions of linear systems.
- Add repeated polynomial/rational derivatives, zero-constant polynomial antiderivatives, and exact definite polynomial integrals.
- Record all semantic operation arguments and check source linkage, result axes, and mathematical certificates independently of producer entrypoints.
- Bound computation and decoding resources; report exhaustion explicitly without claiming no solutions.
- Add strict v2 JSON and a packaged schema while retaining byte-identical v1 graphs and historical workspace references.
- Add multi-version CI, isolated wheel/sdist checks, a typed consumer, development-only oracle checks, and reproducible benchmarks.

Compatibility: the package remains dependency-free at runtime and supports Python 3.11+. Existing scalar quadratic forms and call patterns remain available. New request-bearing results use format v2, which v0.1 readers reject. Resource defaults can reject documents or computations that exceed their finite limits; callers can supply larger explicit limits.

## 0.1.0

Initial exact rational/expression core, univariate polynomials, linear and quadratic real equations, independent equation checks, structured SDK results, and optional v1 JSON workspaces.
