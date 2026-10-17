"""Dependency-free, machine-readable scope of the curated v0.7 interface, the area tools and the catalog.

The thirty v0.7 tools and the v0.7 operation entries are frozen. The area tools, the v6 operation entries
and the scope texts derive from the operation registry.
"""

from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict

from .limits import ComputationLimits

if TYPE_CHECKING:
    from .registry import OperationSpec


class ToolCapability(TypedDict):
    name: str
    sdk: str
    description: str


class OperationCapability(TypedDict):
    name: str
    sdk: str
    mcp_tool: str | None
    scope: str


class Capabilities(TypedDict):
    api_version: int
    package_version: str
    transports: list[str]
    tools: list[ToolCapability]
    operations: list[OperationCapability]
    input: dict[str, object]
    output: dict[str, object]
    limits: dict[str, int]
    excluded: list[str]


_TOOLS = (
    ("capabilities", "capabilities", "Describe supported inputs, tools, limits and exact output."),
    ("solve", "operations.solve", "Solve a real univariate polynomial/rational equation or inequality."),
    ("solve_system", "operations.solve_system", "Solve a rational linear system with an explicit variable order."),
    ("real_roots", "operations.real_roots", "Return all real polynomial roots and multiplicities."),
    ("differentiate", "operations.differentiate", "Differentiate a polynomial or rational function repeatedly."),
    ("expand_multivariate", "operations.expand_multivariate", "Expand an exact multivariate polynomial in an explicit variable order."),
    ("partial_diff", "operations.partial_diff", "Compute a repeated partial derivative of an exact multivariate polynomial."),
    ("gradient", "operations.gradient", "Compute a polynomial gradient in an explicit variable order."),
    ("jacobian", "operations.jacobian", "Compute a polynomial Jacobian with explicit function and variable order."),
    ("hessian", "operations.hessian", "Compute a polynomial Hessian in an explicit variable order."),
    ("integrate", "operations.integrate", "Integrate a rational-coefficient polynomial between exact real algebraic finite bounds."),
    ("limit", "operations.limit", "Compute a rational-function limit at a real algebraic point or signed infinity."),
    ("analyze_function", "operations.analyze_function", "Analyze a real rational function: domain, signs, derivative, monotonicity and strict extrema."),
    ("det", "operations.det", "Compute an exact rational matrix determinant."),
    ("rref", "operations.rref", "Compute rational reduced row echelon form and pivot columns."),
    ("verify", "verify", "Recheck a canonical v1/v2/v3/v4/v5 Result without trusting stored verification labels."),
    ("factor", "operations.factor", "Factor a nonzero univariate rational polynomial completely over Q."),
    ("resultant", "operations.resultant", "Compute the exact resultant of two rational polynomials in one variable."),
    ("discriminant", "operations.discriminant", "Compute a rational polynomial discriminant; constants give zero."),
    ("algebraic", "operations.algebraic", "Normalize a closed exact real algebraic expression, including independent radicals."),
    ("algebraic_div", "operations.algebraic_div", "Divide exact real algebraic scalars with a checked nonzero denominator."),
    ("algebraic_pow", "operations.algebraic_pow", "Raise an exact real algebraic scalar to a bounded signed integer power."),
    ("minimal_polynomial", "operations.minimal_polynomial", "Return the monic minimal polynomial of an exact real algebraic scalar."),
    ("evaluate_algebraic", "operations.evaluate_algebraic", "Evaluate exact expressions with complete real algebraic symbol bindings."),
    ("apart", "operations.apart", "Decompose a rational function into partial fractions over Q, preserving every excluded point."),
    ("extrema_values", "operations.extrema_values", "Return exact coordinates and values of strict local extrema of a rational function."),
    ("charpoly", "operations.charpoly", "Compute det(t I - A) for a square rational matrix."),
    ("real_eigenvalues", "operations.real_eigenvalues", "Return all real eigenvalues and algebraic multiplicities of a square rational matrix."),
    ("parse_function", "parse_function", "Parse a bounded real elementary function with explicit ordered symbols and preserved domains."),
    ("compute", "operations_v5.run", "Execute a published v0.7 operation using canonical mathematical object operands; deferred requests make no conclusion."),
)

_OPERATIONS = (
    ("expand", "rational-coefficient univariate polynomials"),
    ("expand_multivariate", "rational-coefficient multivariate polynomials with explicit variable order"),
    ("partial_diff", "rational-coefficient multivariate polynomials; nonnegative-order partial derivatives"),
    ("gradient", "rational-coefficient multivariate polynomials; ordered expression tuples"),
    ("jacobian", "rational-coefficient polynomial maps; ordered tuples of expression rows"),
    ("hessian", "rational-coefficient multivariate polynomials; ordered tuples of expression rows"),
    ("poly_divmod", "same-variable rational polynomials"),
    ("poly_gcd", "same-variable rational polynomials"),
    ("poly_xgcd", "same-variable rational polynomials"),
    ("square_free", "rational polynomials; not irreducible factorization"),
    ("real_roots", "real roots of rational-coefficient univariate polynomials"),
    ("compare_real", "exact real algebraic scalars; legacy successful representations preserved"),
    ("rational_function", "real univariate rational functions preserving excluded points"),
    ("solve", "unconstrained real univariate polynomial/rational equations and inequalities"),
    ("solve_system", "rational linear systems over unconstrained real variables"),
    ("rref", "rational matrices"), ("rank", "rational matrices"),
    ("det", "square rational matrices"), ("inverse", "nonsingular square rational matrices"),
    ("nullspace", "rational matrices"),
    ("differentiate", "polynomial/rational-function nonnegative-order derivatives"),
    ("antiderivative", "polynomials; integration constant zero"),
    ("integrate", "rational-coefficient polynomials with finite real algebraic bounds"),
    ("limit", "real univariate rational functions at real algebraic points (left/right/both) or signed infinity (both)"),
    ("analyze_function", "real univariate rational functions; exact domains, signs, stationary sets, maximal monotonic regions and strict extrema x-coordinates"),
    ("evaluate", "exact supported expressions and rational functions; complete bindings"),
    ("substitute", "exact supported expressions and restricted rational-function substitutions"),
    ("simplify", "documented domain-preserving local expression rules"),
    ("verify", "supported Result provenance and mathematical claims"),
    ("factor", "complete irreducible factorization over Q of nonzero univariate rational polynomials"),
    ("resultant", "same-variable rational polynomial resultant"),
    ("discriminant", "rational polynomial discriminant; constants give zero"),
    ("algebraic", "closed exact real algebraic expressions; canonical rationals or real root descriptors"),
    ("algebraic_div", "exact real algebraic scalar division with nonzero denominator"),
    ("algebraic_pow", "exact real algebraic scalars with signed integer exponents"),
    ("minimal_polynomial", "monic minimal polynomial over Q of a real algebraic scalar"),
    ("evaluate_algebraic", "exact expressions and rational functions with complete real algebraic symbol bindings"),
    ("apart", "partial fractions over irreducible Q factors, retaining source domain exclusions"),
    ("extrema_values", "exact real algebraic coordinates and values of rational-function strict local extrema"),
    ("charpoly", "det(t I - A) for square rational matrices, including empty 0 by 0 matrices"),
    ("real_eigenvalues", "complete real spectrum of square rational matrices with algebraic multiplicities; complex eigenvalues excluded"),
)


CATALOG_TOOL = "mathforge.catalog"
_CATALOG_DESCRIPTION = ("Search operations by name, summary or area. Each entry lists its argument kinds and defaults "
                        "and names the tool that runs it: registry operations point to their typed area tool, also "
                        "those that mathforge.compute accepts.")


def area_tools() -> list[str]:
    """The MCP area tools of registry.MCP_TOOLS that serve at least one operation, sorted by name."""
    from . import registry
    return sorted(tool for tool in registry.MCP_TOOLS if registry.tool_operations(tool))


def area_tool(spec: "OperationSpec") -> str | None:
    """The area tool that runs a registry operation; None for an operation that MCP does not expose."""
    from . import registry
    if not spec.mcp:
        return None
    return next(tool for tool, areas in registry.MCP_TOOLS.items() if spec.area in areas)


def _area_tool(tool: str) -> ToolCapability:
    from . import registry
    areas = registry.MCP_TOOLS[tool]
    named = areas[0] + " area" if len(areas) == 1 else ", ".join(areas[:-1]) + " and " + areas[-1] + " areas"
    return dict(name=tool, sdk=", ".join(areas),
                description=f"Run an operation of the {named} by name, with typed arguments: friendly JSON or "
                            "canonical object envelopes. Deferred requests make no conclusion; mathforge.catalog "
                            "lists the operations.")


def capabilities() -> Capabilities:
    """Return fresh metadata; representability is not a promise of solver support."""
    from . import __version__, registry
    exposed = {name for name, _, _ in _TOOLS}
    registered: dict[int, list[OperationCapability]] = {5: [], 6: []}
    for spec in registry.specs().values():  # The v0.7 operations first, in their published order.
        parameters = ", ".join(sorted(argument.name for argument in spec.arguments))
        # mathforge.compute keeps the v0.7 operations; a later operation runs only through its area tool.
        tool = "mathforge.compute" if spec.version == 5 else area_tool(spec)
        registered[spec.version].append(dict(name=spec.name, sdk=spec.public_name, mcp_tool=tool,
                                             scope=spec.scope + "; arguments: " + parameters))

    return {
        "api_version": 1,
        "package_version": __version__,
        "transports": ["stdio"],
        "tools": [dict(name="mathforge." + name, sdk=sdk, description=description)
                  for name, sdk, description in _TOOLS] + [_area_tool(tool) for tool in area_tools()] + [
                      dict(name=CATALOG_TOOL, sdk="registry.specs", description=_CATALOG_DESCRIPTION)],
        "operations": [dict(name=name, sdk=name if name == "verify" else "operations." + name,
                            mcp_tool="mathforge." + name if name in exposed else None, scope=scope)
                       for name, scope in _OPERATIONS] + registered[5] + [
                           dict(name="parse_function", sdk="parse_function", mcp_tool="mathforge.parse_function",
                                scope="bounded elementary text; declared ordered real axes; exact decimal literals; explicit domain predicates")
                       ] + registered[6],
        "input": {
            "expressions": "bounded MathForge text grammar; explicit declared symbols",
            "domains": ["unconstrained reals", "explicit real function domains", "real algebraic complex components", "integers", "finite rational distributions"],
            "matrix_entries_and_bounds": "matrix entries: rational strings; finite bounds: closed math text or canonical scalar object envelopes",
            "limit_points": "closed real algebraic math text or canonical scalar object envelopes, -inf or +inf",
            "algebraic_scalars": "closed math text or canonical scalar envelopes only; root display text, Result envelopes and workspaces are not operands",
            "algebraic_bindings": "explicit variables array with a complete matching name-to-scalar bindings map",
            "verification": "canonical MathForge v1/v2/v3/v4/v5 Result envelope only",
            "function_text": "parse_function requires text and an explicit ordered variables array; symbol_ids reuse identities",
            "compute_operands": "each named argument is a canonical object envelope; explicit registry operation and all arguments including defaults; Result/workspace envelopes are not operands",
            "symbol_identity": "request-local UUIDs; pass symbol_ids to reuse identities",
            "multivariate_variables": "explicit nonempty arrays of distinct declared symbols; order determines gradient and matrix coordinates",
            "state": "stateless; no filesystem or persistent workspace tools",
        },
        "output": {
            "mathematics": "canonical MathForge v1/v2/v3/v4/v5 envelopes; legacy bytes preserved",
            "presentation": ["exact text", "LaTeX"],
            "verification": "algorithmic checking, not formal proof",
            "deferred": "UnevaluatedIntegral/UnevaluatedLimit carry no value, exactness or verification claim; completed execution is not a solved problem",
        },
        "limits": asdict(ComputationLimits()),
        "excluded": ["network transport", "approximate preview", "plotting", "general CAS",
                     "arbitrary Python execution", "general command dispatcher"],
    }
