"""The v0.7 operation table: the 60 published v5 operations and the pre-registry catalogue.

Argument annotations, defaults and return types are copied verbatim from the
v0.7 signatures (``api_v7.py``); scope texts are the ones that
``mathforge.capabilities()`` publishes for each family module. Specs follow
the published order: family by family, then each family's ARGUMENTS order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import Argument, OperationSpec


@dataclass(frozen=True)
class CatalogEntry:
    """An operation that predates the registry; ``tool`` is "" when it has no dedicated MCP tool."""
    name: str
    area: str
    tool: str
    summary: str


# Published capabilities scope of each v0.7 family module.
_NUMBER_THEORY = "exact integers; deterministic bounded primality and factorization; modular and CRT certificates"
_COMBINATORICS = "nonnegative integer indices; exact bounded counts and recurrences"
_COMPLEX = ("complex scalars with real algebraic components; exact arithmetic and quadratic roots; "
            "no general complex root isolation")
_PROBABILITY = "finite rational outcome distributions; exact weights, events, conditioning and moments"
_STATISTICS = ("nonempty finite rational samples; exact descriptive statistics; "
               "nonnegative ddof smaller than sample size")
_LINEAR = ("rational matrices; complete subspace bases, singular-safe square PLU and full affine "
           "least-squares minimizer families")
_CALCULUS = ("domain-aware real elementary calculus within the documented rule catalogue; "
             "unsupported integrals/limits remain explicitly unevaluated")
_RATIONAL_INTEGRATION = ("rational functions over Q; exact Hermite reduction and logarithmic root sums; "
                         "real-domain component restrictions retained")
_FUNCTION_INTEGRATION = ("proper finite real integrals; independently checked primitives and whole-interval "
                         "domain certificates; unsupported or improper requests remain unevaluated")

# Arguments shared by several operations.
_A = Argument("a", "int", kind="int")
_B = Argument("b", "int", kind="int")
_N = Argument("n", "int", kind="int")
_K = Argument("k", "int", kind="int")
_LEFT = Argument("left", kind="complex")
_RIGHT = Argument("right", kind="complex")
_VALUE = Argument("value", kind="complex")
_P = Argument("p", kind="rational")
_DISTRIBUTION = Argument("distribution", "FiniteDistribution", kind="distribution")
_EVENT = Argument("event", kind="rational_list")
_DATA = Argument("data", kind="rational_list")
_DDOF = Argument("ddof", "int", "0", "int")
_MATRIX = Argument("A", "Matrix", kind="matrix")
_FUNCTION = Argument("expression", kind="function")
_RATIONAL_FUNCTION = Argument("expression", kind="rational_function")
_VARIABLE = Argument("variable", "Symbol", kind="symbol")
_VARIABLES = Argument("variables", kind="symbols")
_DERIVATIVE_ORDER = Argument("n", "int", "1", "int")
_LOWER = Argument("lower", kind="scalar")
_UPPER = Argument("upper", kind="scalar")

V07_SPECS: tuple[OperationSpec, ...] = (
    # number_theory
    OperationSpec("integer_gcd", "ntheory", (_A, _B), "int",
                  "Compute the nonnegative greatest common divisor of two exact integers.", _NUMBER_THEORY),
    OperationSpec("integer_lcm", "ntheory", (_A, _B), "int",
                  "Compute the nonnegative least common multiple of two exact integers.", _NUMBER_THEORY),
    OperationSpec("integer_xgcd", "ntheory", (_A, _B), "tuple[int, int, int]",
                  "Return the greatest common divisor of two integers with its Bezout coefficients.", _NUMBER_THEORY),
    OperationSpec("mod_inverse", "ntheory", (_A, Argument("modulus", "int", kind="int")), "int",
                  "Compute the inverse of an integer modulo a positive modulus, if it exists.", _NUMBER_THEORY),
    OperationSpec("crt", "ntheory", (Argument("congruences", kind="int_pairs"),), "CRTResult",
                  "Solve a system of (residue, modulus) congruences, including noncoprime moduli.", _NUMBER_THEORY),
    OperationSpec("is_prime", "ntheory", (_N,), "bool",
                  "Decide whether an integer is prime with a deterministic, checkable certificate.", _NUMBER_THEORY),
    OperationSpec("factor_integer", "ntheory", (_N,), "IntegerFactorization",
                  "Factor a nonzero integer into its sign and certified prime powers.", _NUMBER_THEORY),
    OperationSpec("divisors", "ntheory", (_N,), "tuple[int, ...]",
                  "List the positive divisors of a nonzero integer in increasing order.", _NUMBER_THEORY),
    OperationSpec("totient", "ntheory", (_N,), "int",
                  "Compute Euler's totient of a positive integer.", _NUMBER_THEORY),
    # combinatorics
    OperationSpec("factorial", "discrete", (_N,), "int",
                  "Compute the factorial of a nonnegative integer.", _COMBINATORICS),
    OperationSpec("binomial", "discrete", (_N, _K), "int",
                  "Compute the binomial coefficient C(n, k) of nonnegative integers.", _COMBINATORICS),
    OperationSpec("multinomial", "discrete", (Argument("counts", kind="int_list"),), "int",
                  "Compute the multinomial coefficient of nonnegative category counts.", _COMBINATORICS),
    OperationSpec("permutations_count", "discrete", (_N, _K), "int",
                  "Count the ordered selections of k items from n distinct items.", _COMBINATORICS),
    OperationSpec("fibonacci", "discrete", (_N,), "int",
                  "Compute the n-th Fibonacci number.", _COMBINATORICS),
    OperationSpec("catalan", "discrete", (_N,), "int",
                  "Compute the n-th Catalan number.", _COMBINATORICS),
    OperationSpec("stirling", "discrete", (_N, _K), "int",
                  "Compute the Stirling number of the second kind S(n, k).", _COMBINATORICS),
    # complex_numbers
    OperationSpec("complex_algebraic", "algebra",
                  (Argument("real", kind="scalar"), Argument("imag", default="0", kind="scalar")), "ComplexAlgebraic",
                  "Build an exact complex scalar from real algebraic real and imaginary parts.", _COMPLEX),
    OperationSpec("complex_add", "algebra", (_LEFT, _RIGHT), "ComplexAlgebraic",
                  "Add two exact complex scalars.", _COMPLEX),
    OperationSpec("complex_sub", "algebra", (_LEFT, _RIGHT), "ComplexAlgebraic",
                  "Subtract one exact complex scalar from another.", _COMPLEX),
    OperationSpec("complex_mul", "algebra", (_LEFT, _RIGHT), "ComplexAlgebraic",
                  "Multiply two exact complex scalars.", _COMPLEX),
    OperationSpec("complex_div", "algebra", (_LEFT, _RIGHT), "ComplexAlgebraic",
                  "Divide exact complex scalars with a checked nonzero denominator.", _COMPLEX),
    OperationSpec("complex_conjugate", "algebra", (_VALUE,), "ComplexAlgebraic",
                  "Return the conjugate of an exact complex scalar.", _COMPLEX),
    OperationSpec("complex_real", "algebra", (_VALUE,), "Rational | RealAlgebraicRoot",
                  "Return the real part of an exact complex scalar.", _COMPLEX),
    OperationSpec("complex_imag", "algebra", (_VALUE,), "Rational | RealAlgebraicRoot",
                  "Return the imaginary part of an exact complex scalar.", _COMPLEX),
    OperationSpec("complex_norm_squared", "algebra", (_VALUE,), "Rational | RealAlgebraicRoot",
                  "Return the squared modulus of an exact complex scalar.", _COMPLEX),
    OperationSpec("complex_sqrt", "algebra", (_VALUE,), "ComplexAlgebraic",
                  "Return the principal square root of an exact complex scalar.", _COMPLEX),
    OperationSpec("complex_pow", "algebra", (Argument("base", kind="complex"), _N), "ComplexAlgebraic",
                  "Raise an exact complex scalar to a bounded signed integer power.", _COMPLEX),
    OperationSpec("quadratic_roots", "algebra",
                  (Argument("a", kind="scalar"), Argument("b", kind="scalar"), Argument("c", kind="scalar")),
                  "tuple[ComplexRoot, ...]",
                  "Return the exact complex roots of a rational quadratic with their multiplicities.", _COMPLEX),
    # probability
    OperationSpec("bernoulli", "stats", (_P,), "FiniteDistribution",
                  "Build the Bernoulli distribution on 0 and 1 with a rational success probability.", _PROBABILITY),
    OperationSpec("binomial_distribution", "stats", (_N, _P), "FiniteDistribution",
                  "Build the binomial distribution of n independent trials with a rational success probability.",
                  _PROBABILITY),
    OperationSpec("discrete_uniform", "stats", (Argument("outcomes", kind="rational_list"),), "FiniteDistribution",
                  "Build the uniform distribution on distinct rational outcomes.", _PROBABILITY),
    OperationSpec("hypergeometric", "stats",
                  (Argument("population", "int", kind="int"), Argument("successes", "int", kind="int"),
                   Argument("draws", "int", kind="int")), "FiniteDistribution",
                  "Build the distribution of successes among draws without replacement from a finite population.",
                  _PROBABILITY),
    OperationSpec("probability", "stats", (_DISTRIBUTION, _EVENT), "Rational",
                  "Compute the exact probability of a finite event.", _PROBABILITY),
    OperationSpec("condition", "stats", (_DISTRIBUTION, _EVENT), "FiniteDistribution",
                  "Condition a finite distribution on an event of positive probability.", _PROBABILITY),
    OperationSpec("cdf", "stats", (_DISTRIBUTION, Argument("x", kind="rational")), "Rational",
                  "Compute the probability that the outcome is at most x.", _PROBABILITY),
    OperationSpec("moment", "stats", (_DISTRIBUTION, Argument("order", "int", kind="int")), "Rational",
                  "Compute a raw moment of nonnegative integer order of a finite distribution.", _PROBABILITY),
    OperationSpec("expectation", "stats", (_DISTRIBUTION,), "Rational",
                  "Compute the exact expectation of a finite distribution.", _PROBABILITY),
    OperationSpec("distribution_variance", "stats", (_DISTRIBUTION,), "Rational",
                  "Compute the exact variance of a finite distribution.", _PROBABILITY),
    # statistics
    OperationSpec("mean", "stats", (_DATA,), "Rational",
                  "Compute the exact mean of a nonempty rational sample.", _STATISTICS),
    OperationSpec("median", "stats", (_DATA,), "Rational",
                  "Compute the exact median of a nonempty rational sample.", _STATISTICS),
    OperationSpec("variance", "stats", (_DATA, _DDOF), "Rational",
                  "Compute the exact variance of a rational sample with ddof normalization.", _STATISTICS),
    OperationSpec("stddev", "stats", (_DATA, _DDOF), "Expression",
                  "Compute the exact standard deviation of a rational sample with ddof normalization; "
                  "it may be a radical.", _STATISTICS),
    OperationSpec("covariance", "stats",
                  (Argument("x", kind="rational_list"), Argument("y", kind="rational_list"), _DDOF), "Rational",
                  "Compute the exact covariance of paired rational samples with ddof normalization.", _STATISTICS),
    # linear_extensions
    OperationSpec("trace", "linalg", (_MATRIX,), "Rational",
                  "Compute the trace of a square rational matrix.", _LINEAR),
    OperationSpec("rowspace", "linalg", (_MATRIX,), "tuple[tuple[Rational, ...], ...]",
                  "Return an RREF basis of the row space of a rational matrix.", _LINEAR),
    OperationSpec("columnspace", "linalg", (_MATRIX,), "tuple[tuple[Rational, ...], ...]",
                  "Return the original pivot columns of a rational matrix as a column space basis.", _LINEAR),
    OperationSpec("lu", "linalg", (_MATRIX,), "LUDecomposition",
                  "Compute a singular-safe PLU decomposition P A = L U of a square rational matrix.", _LINEAR),
    OperationSpec("least_squares", "linalg", (_MATRIX, Argument("b", kind="rational_list")), "LeastSquaresResult",
                  "Return every least-squares minimizer of A x = b as an affine family.", _LINEAR),
    # function_calculus
    OperationSpec("function_differentiate", "analysis", (_FUNCTION, _VARIABLE, _DERIVATIVE_ORDER), "RealFunction",
                  "Differentiate a real elementary function repeatedly, preserving its domain.", _CALCULUS),
    OperationSpec("function_partial_diff", "analysis", (_FUNCTION, _VARIABLE, _DERIVATIVE_ORDER), "RealFunction",
                  "Compute a repeated partial derivative of a real elementary function.", _CALCULUS),
    OperationSpec("function_gradient", "analysis", (_FUNCTION, _VARIABLES), "tuple[RealFunction, ...]",
                  "Compute the gradient of a real elementary function in an explicit variable order.", _CALCULUS),
    OperationSpec("function_jacobian", "analysis", (Argument("expressions", kind="function_list"), _VARIABLES),
                  "tuple[tuple[RealFunction, ...], ...]",
                  "Compute the Jacobian of real elementary functions with explicit function and variable order.",
                  _CALCULUS),
    OperationSpec("function_hessian", "analysis", (_FUNCTION, _VARIABLES), "tuple[tuple[RealFunction, ...], ...]",
                  "Compute the Hessian of a real elementary function in an explicit variable order.", _CALCULUS),
    OperationSpec("function_taylor", "series",
                  (_FUNCTION, _VARIABLE, Argument("point", default="0", kind="scalar"),
                   Argument("order", "int", "5", "int")), "TaylorPolynomial",
                  "Compute a Taylor polynomial of a real elementary function at a real algebraic center.", _CALCULUS,
                  sdk="taylor"),
    OperationSpec("function_evaluate", "analysis", (_FUNCTION, Argument("bindings", kind="bindings")), "RealExpr",
                  "Evaluate a real elementary function with complete exact bindings inside its domain.", _CALCULUS),
    OperationSpec("function_antiderivative", "integrals", (_FUNCTION, _VARIABLE), "RealFunction | RationalPrimitive",
                  "Find an elementary antiderivative from the documented rule catalogue; unmatched requests are "
                  "deferred.", _CALCULUS),
    OperationSpec("function_limit", "series",
                  (_FUNCTION, _VARIABLE, Argument("point", kind="point"), Argument("side", "str", "'both'", "side")),
                  "FunctionLimitValue",
                  "Compute a directional or two-sided limit of a real elementary function at a real algebraic "
                  "point or signed infinity; unsupported limits are deferred.", _CALCULUS),
    # rational_integration
    OperationSpec("rational_antiderivative", "integrals", (_RATIONAL_FUNCTION, _VARIABLE), "RationalPrimitive",
                  "Integrate a rational function over Q by Hermite reduction and logarithmic root sums.",
                  _RATIONAL_INTEGRATION),
    OperationSpec("rational_integrate", "integrals", (_RATIONAL_FUNCTION, _VARIABLE, _LOWER, _UPPER), "RealExpr",
                  "Evaluate a proper finite integral of a rational function over Q; unsupported requests are "
                  "deferred.", _RATIONAL_INTEGRATION),
    # function_integration
    OperationSpec("function_integrate", "integrals", (_FUNCTION, _VARIABLE, _LOWER, _UPPER), "RealExpr",
                  "Evaluate a proper finite integral of a real elementary function; unsupported requests are "
                  "deferred.", _FUNCTION_INTEGRATION),
)

# The pre-registry operations published by capabilities() (its order), plus parse_function.
# Summaries of operations with a dedicated tool repeat that tool's description.
LEGACY_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("expand", "expr", "", "Expand a rational-coefficient univariate polynomial into canonical form."),
    CatalogEntry("expand_multivariate", "algebra", "mathforge.expand_multivariate",
                 "Expand an exact multivariate polynomial in an explicit variable order."),
    CatalogEntry("partial_diff", "analysis", "mathforge.partial_diff",
                 "Compute a repeated partial derivative of an exact multivariate polynomial."),
    CatalogEntry("gradient", "analysis", "mathforge.gradient",
                 "Compute a polynomial gradient in an explicit variable order."),
    CatalogEntry("jacobian", "analysis", "mathforge.jacobian",
                 "Compute a polynomial Jacobian with explicit function and variable order."),
    CatalogEntry("hessian", "analysis", "mathforge.hessian",
                 "Compute a polynomial Hessian in an explicit variable order."),
    CatalogEntry("poly_divmod", "algebra", "",
                 "Divide two same-variable rational polynomials with quotient and remainder."),
    CatalogEntry("poly_gcd", "algebra", "",
                 "Compute the greatest common divisor of two same-variable rational polynomials."),
    CatalogEntry("poly_xgcd", "algebra", "",
                 "Return the greatest common divisor of two same-variable rational polynomials with Bezout cofactors."),
    CatalogEntry("square_free", "algebra", "",
                 "Compute the square-free decomposition of a rational polynomial; not an irreducible factorization."),
    CatalogEntry("real_roots", "algebra", "mathforge.real_roots",
                 "Return all real polynomial roots and multiplicities."),
    CatalogEntry("compare_real", "algebra", "",
                 "Compare two exact real algebraic scalars, returning -1, 0 or 1."),
    CatalogEntry("rational_function", "algebra", "",
                 "Build a real univariate rational function that preserves its excluded points."),
    CatalogEntry("solve", "equations", "mathforge.solve",
                 "Solve a real univariate polynomial/rational equation or inequality."),
    CatalogEntry("solve_system", "equations", "mathforge.solve_system",
                 "Solve a rational linear system with an explicit variable order."),
    CatalogEntry("rref", "linalg", "mathforge.rref", "Compute rational reduced row echelon form and pivot columns."),
    CatalogEntry("rank", "linalg", "", "Compute the exact rank of a rational matrix."),
    CatalogEntry("det", "linalg", "mathforge.det", "Compute an exact rational matrix determinant."),
    CatalogEntry("inverse", "linalg", "", "Invert a nonsingular square rational matrix exactly."),
    CatalogEntry("nullspace", "linalg", "", "Return a rational basis of the nullspace of a rational matrix."),
    CatalogEntry("differentiate", "analysis", "mathforge.differentiate",
                 "Differentiate a polynomial or rational function repeatedly."),
    CatalogEntry("antiderivative", "integrals", "",
                 "Integrate a rational-coefficient polynomial with integration constant zero."),
    CatalogEntry("integrate", "integrals", "mathforge.integrate",
                 "Integrate a rational-coefficient polynomial between exact real algebraic finite bounds."),
    CatalogEntry("limit", "series", "mathforge.limit",
                 "Compute a rational-function limit at a real algebraic point or signed infinity."),
    CatalogEntry("analyze_function", "analysis", "mathforge.analyze_function",
                 "Analyze a real rational function: domain, signs, derivative, monotonicity and strict extrema."),
    CatalogEntry("evaluate", "expr", "",
                 "Evaluate an exact expression or rational function with complete symbol bindings."),
    CatalogEntry("substitute", "expr", "",
                 "Simultaneously replace symbols in an exact expression; unbound symbols may remain."),
    CatalogEntry("simplify", "expr", "", "Apply only the documented, domain-preserving local expression rules."),
    CatalogEntry("verify", "equations", "mathforge.verify",
                 "Recheck a canonical v1/v2/v3/v4/v5 Result without trusting stored verification labels."),
    CatalogEntry("factor", "algebra", "mathforge.factor",
                 "Factor a nonzero univariate rational polynomial completely over Q."),
    CatalogEntry("resultant", "algebra", "mathforge.resultant",
                 "Compute the exact resultant of two rational polynomials in one variable."),
    CatalogEntry("discriminant", "algebra", "mathforge.discriminant",
                 "Compute a rational polynomial discriminant; constants give zero."),
    CatalogEntry("algebraic", "algebra", "mathforge.algebraic",
                 "Normalize a closed exact real algebraic expression, including independent radicals."),
    CatalogEntry("algebraic_div", "algebra", "mathforge.algebraic_div",
                 "Divide exact real algebraic scalars with a checked nonzero denominator."),
    CatalogEntry("algebraic_pow", "algebra", "mathforge.algebraic_pow",
                 "Raise an exact real algebraic scalar to a bounded signed integer power."),
    CatalogEntry("minimal_polynomial", "algebra", "mathforge.minimal_polynomial",
                 "Return the monic minimal polynomial of an exact real algebraic scalar."),
    CatalogEntry("evaluate_algebraic", "expr", "mathforge.evaluate_algebraic",
                 "Evaluate exact expressions with complete real algebraic symbol bindings."),
    CatalogEntry("apart", "algebra", "mathforge.apart",
                 "Decompose a rational function into partial fractions over Q, preserving every excluded point."),
    CatalogEntry("extrema_values", "analysis", "mathforge.extrema_values",
                 "Return exact coordinates and values of strict local extrema of a rational function."),
    CatalogEntry("charpoly", "linalg", "mathforge.charpoly", "Compute det(t I - A) for a square rational matrix."),
    CatalogEntry("real_eigenvalues", "linalg", "mathforge.real_eigenvalues",
                 "Return all real eigenvalues and algebraic multiplicities of a square rational matrix."),
    CatalogEntry("parse_function", "expr", "mathforge.parse_function",
                 "Parse a bounded real elementary function with explicit ordered symbols and preserved domains."),
)
