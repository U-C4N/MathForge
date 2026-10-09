# Changelog

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
