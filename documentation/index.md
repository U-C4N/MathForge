# MathForge user guide

MathForge is a bounded, dependency-free exact mathematics core for Python
applications, with an optional local MCP adapter for agents. This is an alpha
release: verification is algorithmic checking, not formal proof.

## Start with a task

- [Solve your first problem](quickstart.md)
- [Use elementary analysis, complex algebra and discrete mathematics](v07-mathematics.md)
- [Factor polynomials and compute with real algebraic numbers](exact-algebra.md)
- [Read mathematical text](parsing.md)
- [Compute limits and analyze real functions](function-analysis.md)
- [Expand and differentiate multivariate polynomials](multivariate.md)
- [Inspect measured performance and reproduce benchmarks](performance.md)
- [Show exact results as text or LaTeX](formatting.md)
- [Connect a local MCP client](mcp.md)
- [Choose a Python operation](api.md)
- [Find operations by mathematical area](areas.md)
- [Understand compatibility and limits](compatibility.md)

The library separates **representing**, **solving**, and **verifying** an object.
A representable expression is not a promise that every operation supports it.
For example, `sqrt(2)` is exact, but polynomial calculus requires rational
coefficients. The API reference and capabilities metadata describe these limits.

The mathematical path is shared: text is parsed into existing MathForge objects,
the existing operations compute and check results, and independent presentation
helpers display them. Neither LaTeX nor explanatory text is mathematical evidence.
