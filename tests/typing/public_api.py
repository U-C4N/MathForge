"""REL-08: A consumer checks concrete public return types through py.typed."""

from typing import assert_type

import mathforge as mf


A = mf.Matrix(((1, 2), (3, 4)))
x, y = mf.symbol("x"), mf.symbol("y")

assert_type(mf.expand_multivariate((x+y)**2, (x, y)), mf.Expression)
assert_type(mf.partial_diff(x*y, x), mf.Expression)
assert_type(mf.gradient(x*y, (x, y)), tuple[mf.Expression, ...])
assert_type(mf.jacobian((x*y,), (x, y)), tuple[tuple[mf.Expression, ...], ...])
assert_type(mf.hessian(x*y, (x, y)), tuple[tuple[mf.Expression, ...], ...])
assert_type(mf.operations.hessian(x*y, (x, y)).unwrap(), tuple[tuple[mf.Expression, ...], ...])

assert_type(mf.rref(A), mf.RREFResult)
assert_type(mf.rank(A), int)
assert_type(mf.det(A), mf.Rational)
assert_type(mf.inverse(A), mf.Matrix)
assert_type(mf.nullspace(A), tuple[tuple[mf.Rational, ...], ...])
assert_type(mf.solve(mf.Eq(x**3 - 2, 0), for_=x), mf.Result[mf.operations.EquationSolutionSet])
assert_type(mf.solve_system(mf.LinearSystem(A, (5, 11), (x, y))), mf.Result[mf.operations.LinearSolutionSet])
assert_type(mf.LinearSystem(A, [5, 11], [x, y]).rhs, tuple[mf.Rational, ...])
assert_type(mf.AffineSolutionSet([x, y], [0, 0], [[1, 0], [0, 1]]).basis,
            tuple[tuple[mf.Rational, ...], ...])
assert_type(mf.integrate(x**2, x, 0, 1), mf.Rational)
assert_type(mf.verify(mf.solve(mf.Eq(x**2 - 2, 0), for_=x)), mf.VerificationReport)

# Structured results preserve their value type through unwrap(), without Any.
assert_type(mf.symbol("typed"), mf.Symbol)
assert_type(mf.operations.det(A).unwrap(), mf.Rational)
assert_type(mf.operations.det(A).value, mf.Rational | None)
assert_type(mf.operations.rref(A).unwrap(), mf.RREFResult)
assert_type(mf.operations.rank(A).unwrap(), int)
assert_type(mf.operations.inverse(A).unwrap(), mf.Matrix)
assert_type(mf.operations.nullspace(A).unwrap(), tuple[tuple[mf.Rational, ...], ...])
assert_type(mf.operations.real_roots(x**3 - 2, x).unwrap(), tuple[mf.RootRecord, ...])
assert_type(mf.operations.differentiate(x**2, x).unwrap(), mf.Expression | mf.RationalFunction)
assert_type(mf.operations.integrate(x**2, x, 0, 1).unwrap(), mf.Rational)
assert_type(mf.operations.evaluate(x, {x: 1}).unwrap(), mf.Expression)
assert_type(mf.render_text(A), str)
assert_type(mf.to_latex(A), str)
assert_type(mf.parse_expression("x^2", symbols={"x": x}), mf.Expression | mf.RationalFunction)
assert_type(mf.parse_problem("x = 0", symbols={"x": x}), mf.Eq | mf.Inequality)
assert_type(mf.capabilities(), mf.Capabilities)
assert_type(mf.limit(x, x, 0), mf.LimitValue)
assert_type(mf.operations.limit(x, x, mf.Rational(1, 2), side="left").unwrap(), mf.LimitValue)
assert_type(mf.operations.limit(x, x, "+inf").value, mf.LimitValue | None)
assert_type(mf.analyze_function(x, x), mf.FunctionAnalysis)
assert_type(mf.operations.analyze_function(x, x).unwrap().domain, mf.IntervalSet)
assert_type(mf.analyze_function(x, x).strict_extrema, tuple[mf.LocalExtremum, ...])

p = mf.Polynomial.from_expression(x**4 + 4, x)
alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
assert_type(mf.factor(p), mf.PolynomialFactorization)
assert_type(mf.operations.factor(p).unwrap(), mf.PolynomialFactorization)
assert_type(mf.resultant(p, p), mf.Rational)
assert_type(mf.discriminant(p), mf.Rational)
assert_type(mf.algebraic(alpha), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.algebraic_div(alpha, 2), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.algebraic_pow(alpha, -1), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.minimal_polynomial(alpha, x), mf.Polynomial)
assert_type(mf.evaluate_algebraic(x, {x: alpha}), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.integrate(x, x, 0, alpha), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.apart(mf.rational_function(1, x, variable=x)), mf.PartialFractionDecomposition)
assert_type(mf.extrema_values(x**2, x), tuple[mf.ExtremumValue, ...])
assert_type(mf.charpoly(A, x), mf.Polynomial)
assert_type(mf.real_eigenvalues(A), tuple[mf.RootRecord, ...])

# v0.7: exact integers, algebraic complex scalars, finite measures and new records.
assert_type(mf.integer_gcd(30, 21), int)
assert_type(mf.integer_xgcd(30, 21), tuple[int, int, int])
assert_type(mf.operations.integer_xgcd(30, 21).unwrap(), tuple[int, int, int])
assert_type(mf.crt(((2, 6), (8, 9))), mf.CRTResult)
assert_type(mf.operations.crt(((2, 6), (8, 9))).value, mf.CRTResult | None)
assert_type(mf.is_prime(65537), bool)
assert_type(mf.factor_integer(360), mf.IntegerFactorization)
assert_type(mf.divisors(12), tuple[int, ...])
assert_type(mf.factorial(12), int)
assert_type(mf.binomial(12, 5), int)
assert_type(mf.multinomial((2, 3, 1)), int)
z = mf.ComplexAlgebraic(1, 2)
assert_type(z.real, mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.complex_sqrt(z), mf.ComplexAlgebraic)
assert_type(mf.complex_real(z), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.complex_imag(z), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.complex_norm_squared(z), mf.Rational | mf.RealAlgebraicRoot)
assert_type(mf.operations.complex_pow(z, -2).unwrap(), mf.ComplexAlgebraic)
assert_type(mf.quadratic_roots(1, 0, 2), tuple[mf.ComplexRoot, ...])
distribution = mf.binomial_distribution(5, mf.Rational(1, 3))
assert_type(distribution, mf.FiniteDistribution)
assert_type(mf.expectation(distribution), mf.Rational)
assert_type(mf.condition(distribution, (1, 2)), mf.FiniteDistribution)
assert_type(mf.mean((1, 2, 3)), mf.Rational)
assert_type(mf.variance((1, 2, 3), ddof=1), mf.Rational)
assert_type(mf.stddev((1, 2, 3)), mf.Expression)
assert_type(mf.covariance((1, 2, 3), (3, 2, 1)), mf.Rational)
assert_type(mf.lu(A), mf.LUDecomposition)
assert_type(mf.rowspace(A), tuple[tuple[mf.Rational, ...], ...])
assert_type(mf.columnspace(A), tuple[tuple[mf.Rational, ...], ...])
assert_type(mf.least_squares(A, (1, 2)), mf.LeastSquaresResult)
assert_type(mf.least_squares(A, (1, 2)).basis, tuple[tuple[mf.Rational, ...], ...])
function = mf.real_function(mf.functions.sin(x), (x,))
assert_type(function, mf.RealFunction)
assert_type(mf.function_differentiate(function, x), mf.RealFunction)
assert_type(mf.operations.function_differentiate(function, x).unwrap(), mf.RealFunction)
assert_type(mf.function_gradient(function, (x,)), tuple[mf.RealFunction, ...])
assert_type(mf.function_jacobian((function,), (x,)), tuple[tuple[mf.RealFunction, ...], ...])
assert_type(mf.function_hessian(function, (x,)), tuple[tuple[mf.RealFunction, ...], ...])
assert_type(mf.taylor(function, x, 0, 4), mf.TaylorPolynomial)
assert_type(mf.function_evaluate(function, {x: 0}), mf.RealExpr)
assert_type(mf.function_antiderivative(function, x), mf.RealFunction | mf.RationalPrimitive)
assert_type(mf.function_integrate(function, x, 0, 1), mf.RealExpr)
assert_type(mf.function_limit(function, x, 0), mf.FunctionLimitValue)
assert_type(mf.operations.function_limit(function, x, 0).unwrap(), mf.FunctionLimitValue)
assert_type(mf.rational_antiderivative(mf.rational_function(1, x, variable=x), x), mf.RationalPrimitive)
assert_type(mf.rational_integrate(mf.rational_function(1, x, variable=x), x, 1, 2), mf.RealExpr)

# v0.8: the wrappers generated from the registry keep the v0.7 static types.
assert_type(mf.integer_gcd(84, 30), int)
assert_type(mf.operations.crt(((2, 6), (8, 9))).unwrap(), mf.CRTResult)

# v0.8: area modules load lazily at runtime; type checkers resolve them statically from mathforge/__init__.py.
assert_type(mf.ntheory.integer_gcd(84, 30), int)
assert_type(mf.ntheory.crt(((2, 6), (8, 9))), mf.CRTResult)
assert_type(mf.stats.mean((1, 2, 3)), mf.Rational)
assert_type(mf.linalg.lu(A), mf.LUDecomposition)
assert_type(mf.discrete.binomial(12, 5), int)

# v0.8: the pilot v6 family, reached through its area module and mathforge.operations.
assert_type(mf.ntheory.continued_fraction(mf.Rational(415, 93)), mf.ntheory.ContinuedFraction)
assert_type(mf.operations.continued_fraction(mf.Rational(415, 93)).unwrap(), mf.ntheory.ContinuedFraction)
assert_type(mf.ntheory.ContinuedFraction((4, 2, 6, 7)).terms, tuple[int, ...])

# Expected static failures; --warn-unused-ignores catches accidental Any leaks.
def reject_invalid_inputs() -> None:
    mf.det("not a matrix")  # type: ignore[arg-type]
    mf.operations.solve(mf.Eq(x, 0), for_="x")  # type: ignore[arg-type]
    mf.evaluate(x, {"x": 1})  # type: ignore[dict-item]
    wrong: str = mf.operations.det(A).unwrap()  # type: ignore[assignment]
    mf.diff(x, x, limits=10)  # type: ignore[arg-type]
    mf.parse_expression("x", symbols={"x": "x"})  # type: ignore[dict-item]
    mf.to_latex(x, limits=mf.ComputationLimits())  # type: ignore[arg-type]
    mf.limit(x, x, 0.5)  # type: ignore[arg-type]
    mf.limit(x, x, 0, side="above")  # type: ignore[arg-type]
    wrong_limit: mf.Rational = mf.operations.limit(x, x, 0).unwrap()  # type: ignore[assignment]
    mf.integer_gcd(1.5, 2)  # type: ignore[arg-type]
    mf.ComplexAlgebraic(1.5)  # type: ignore[arg-type]
    wrong_prime: str = mf.is_prime(7)  # type: ignore[assignment]
    wrong_distribution: mf.Matrix = mf.binomial_distribution(4, mf.Rational(1, 2))  # type: ignore[assignment]
    mf.ntheory.integer_gcd(1.5, 2)  # type: ignore[arg-type]
    mf.ntheory.continued_fraction(0.5)  # type: ignore[arg-type]
    mf.continued_fraction  # type: ignore[attr-defined]  # v6 operations never enter the flat namespace.
    mf.integer_gcdd  # type: ignore[attr-defined]  # The lazy-area __getattr__ must stay invisible to type checkers.
