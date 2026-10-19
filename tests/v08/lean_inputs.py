"""One valid small input per operation, for the lean value path tests (tests/v08/test_lean_paths.py).

SMALL_INPUTS maps every operation of the registry and of the legacy catalogue except ``verify`` to the keyword
arguments of its public function, by argument name. Each input is small and succeeds on the structured path.
Symbols carry fixed uids, so the values compare equal across interpreters.

The helpers name, for an operation, its public value function, its structured function and its argument names;
the concurrency probe of the tests imports them in a fresh interpreter.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import inspect
import uuid

import mathforge as mf
from mathforge import functions as f
from mathforge import registry
from mathforge.registry_v07 import LEGACY_CATALOG


def symbol(name: str, k: int) -> mf.Symbol:
    return mf.Symbol(name, uid=str(uuid.UUID(int=k)))


def _rational_function(numerator: object, denominator: object) -> mf.RationalFunction:
    """Built on the structured path, so that importing these inputs leaves the lean path and the registry cold."""
    return mf.operations.rational_function(numerator, denominator, variable=x).unwrap()


x, y = symbol("x", 1), symbol("y", 2)
_alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)  # The real cube root of 2.
_z = mf.ComplexAlgebraic(3, 4)
_distribution = mf.FiniteDistribution((0, 1, 2), (mf.Rational(1, 4), mf.Rational(1, 2), mf.Rational(1, 4)))
_tall = mf.Matrix(((1, 2), (2, 4), (1, 1)))
_square = mf.Matrix(((0, 2), (1, 3)))
_single = mf.real_function(f.sin(x), (x,))
_multi = mf.real_function(f.sin(x) * f.cos(y), (x, y))
_reciprocal = _rational_function(1, x)
_p = mf.Polynomial.from_expression(x**4 + 4, x)

SMALL_INPUTS: dict[str, dict[str, object]] = {
    # ntheory
    "integer_gcd": dict(a=84, b=30),
    "integer_lcm": dict(a=30, b=21),
    "integer_xgcd": dict(a=30, b=21),
    "mod_inverse": dict(a=-3, modulus=11),
    "crt": dict(congruences=((2, 6), (8, 9))),
    "is_prime": dict(n=65537),
    "factor_integer": dict(n=-360),
    "divisors": dict(n=24),
    "totient": dict(n=36),
    "continued_fraction": dict(value=mf.Rational(415, 93)),
    # discrete
    "factorial": dict(n=12),
    "binomial": dict(n=20, k=8),
    "multinomial": dict(counts=(2, 3, 1)),
    "permutations_count": dict(n=10, k=4),
    "fibonacci": dict(n=100),
    "catalan": dict(n=12),
    "stirling": dict(n=8, k=3),
    # algebra: complex scalars
    "complex_algebraic": dict(real=mf.sqrt(2), imag=1),
    "complex_add": dict(left=_z, right=2),
    "complex_sub": dict(left=_z, right=1),
    "complex_mul": dict(left=_z, right=_z),
    "complex_div": dict(left=_z, right=2),
    "complex_conjugate": dict(value=_z),
    "complex_real": dict(value=_z),
    "complex_imag": dict(value=_z),
    "complex_norm_squared": dict(value=_z),
    "complex_sqrt": dict(value=_z),
    "complex_pow": dict(base=_z, n=-2),
    "quadratic_roots": dict(a=1, b=0, c=2),
    # stats
    "bernoulli": dict(p=mf.Rational(1, 3)),
    "binomial_distribution": dict(n=5, p=mf.Rational(1, 3)),
    "discrete_uniform": dict(outcomes=(1, 2, 3)),
    "hypergeometric": dict(population=10, successes=4, draws=3),
    "probability": dict(distribution=_distribution, event=(1, 2)),
    "condition": dict(distribution=_distribution, event=(1, 2)),
    "cdf": dict(distribution=_distribution, x=1),
    "moment": dict(distribution=_distribution, order=3),
    "expectation": dict(distribution=_distribution),
    "distribution_variance": dict(distribution=_distribution),
    "mean": dict(data=(1, 2, 4)),
    "median": dict(data=(1, 2, 4, 8)),
    "variance": dict(data=(1, 2, 4), ddof=1),
    "stddev": dict(data=(1, 2, 4), ddof=0),
    "covariance": dict(x=(1, 2, 4), y=(2, 3, 7), ddof=1),
    # linalg
    "trace": dict(A=_square),
    "rowspace": dict(A=_tall),
    "columnspace": dict(A=_tall),
    "lu": dict(A=_square),
    "least_squares": dict(A=_tall, b=(1, 3, 2)),
    "rref": dict(A=mf.Matrix(((1, 2), (3, 4)))),
    "rank": dict(A=mf.Matrix(((1, 2), (2, 4)))),
    "det": dict(A=mf.Matrix(((1, 2), (3, 4)))),
    "inverse": dict(A=mf.Matrix(((1, 2), (3, 4)))),
    "nullspace": dict(A=mf.Matrix(((1, 2), (2, 4)))),
    "charpoly": dict(A=mf.Matrix(((0, 2), (1, 0))), variable=x),
    "real_eigenvalues": dict(A=mf.Matrix(((0, 2), (1, 0)))),
    # analysis
    "function_differentiate": dict(expression=_single, variable=x, n=1),
    "function_partial_diff": dict(expression=_multi, variable=x, n=1),
    "function_gradient": dict(expression=_multi, variables=(x, y)),
    "function_jacobian": dict(expressions=(_multi, _multi * 2), variables=(x, y)),
    "function_hessian": dict(expression=_multi, variables=(x, y)),
    "function_evaluate": dict(expression=_single, bindings={x: 0}),
    "partial_diff": dict(expression=x**2 * y, variable=x, n=1),
    "gradient": dict(expression=x**2 * y, variables=(x, y)),
    "jacobian": dict(expressions=(x * y, x + y), variables=(x, y)),
    "hessian": dict(expression=x**2 * y, variables=(x, y)),
    "differentiate": dict(expression=x**3, variable=x, n=1),
    "analyze_function": dict(expression=x**3 - 3 * x, variable=x),
    "extrema_values": dict(expression=x**3 - 6 * x, variable=x),
    # series
    "function_taylor": dict(expression=mf.real_function(f.exp(x), (x,)), variable=x, point=0, order=4),
    "function_limit": dict(expression=mf.real_function(f.sin(x) / f.as_real_expr(x), (x,)), variable=x, point=0,
                           side="both"),
    "limit": dict(expression=_rational_function(x**2 - 1, x - 1), variable=x, point=1, side="both"),
    # integrals
    "function_antiderivative": dict(expression=_single, variable=x),
    "function_integrate": dict(expression=_single, variable=x, lower=0, upper=1),
    "rational_antiderivative": dict(expression=_reciprocal, variable=x),
    "rational_integrate": dict(expression=_reciprocal, variable=x, lower=1, upper=2),
    "antiderivative": dict(expression=x**2, variable=x),
    "integrate": dict(expression=x**2, variable=x, lower=0, upper=1),
    # algebra: polynomials and real algebraic numbers
    "expand": dict(expression=(x + 1)**2, variable=x),
    "expand_multivariate": dict(expression=(x + y)**2, variables=(x, y)),
    "poly_divmod": dict(f=mf.Polynomial((-1, 0, 1), x), g=mf.Polynomial((-1, 1), x)),
    "poly_gcd": dict(f=mf.Polynomial((-1, 0, 1), x), g=mf.Polynomial((-1, 1), x)),
    "poly_xgcd": dict(f=mf.Polynomial((-1, 0, 1), x), g=mf.Polynomial((-1, 1), x)),
    "square_free": dict(f=mf.Polynomial((1, 2, 1), x)),
    "real_roots": dict(expression=x**2 - 2, variable=x),
    "compare_real": dict(left=mf.sqrt(2), right=mf.Rational(3, 2)),
    "rational_function": dict(numerator=x + 1, denominator=x - 1, variable=x),
    "factor": dict(f=_p),
    "resultant": dict(f=_p, g=mf.Polynomial((-1, 1), x)),
    "discriminant": dict(f=_p),
    "algebraic": dict(expression=mf.sqrt(2) + mf.sqrt(3)),
    "algebraic_div": dict(left=_alpha, right=_alpha),
    "algebraic_pow": dict(base=_alpha, n=-3),
    "minimal_polynomial": dict(expression=_alpha, variable=x),
    "apart": dict(expression=_rational_function(x + 1, x * x - 1)),
    # expr
    "evaluate": dict(expression=x**2 + 1, bindings={x: 2}),
    "substitute": dict(expression=x**2 + y, bindings={x: 2}),
    "simplify": dict(expression=x + x),
    "evaluate_algebraic": dict(expression=x**3, bindings={x: _alpha}),
    "parse_function": dict(text="sin(x)", symbols={"x": x}, variables=None),
    # equations: the cubic takes the v2 real-set solver, not the v0.1 rules for degree two at most
    "solve": dict(problem=mf.Eq(x**3 - x, 0), for_=x),
    "solve_system": dict(system_or_equations=(mf.Eq(x + y, 3), mf.Eq(x - y, 1)), for_=(x, y)),
}

LEGACY_NAMES = tuple(entry.name for entry in LEGACY_CATALOG)
# The public value function of a legacy operation, where its name differs from the operation's.
VALUE_NAMES = {"differentiate": "diff"}
# Legacy operations whose public function returns a Result: they have no value function.
RESULT_FUNCTIONS = frozenset({"solve", "solve_system"})


def operations() -> tuple[str, ...]:
    """Every registry and legacy operation except verify, each once."""
    return tuple(name for name in dict.fromkeys((*registry.specs(), *LEGACY_NAMES)) if name != "verify")


def public_function(operation: str) -> Callable[..., object] | None:
    """The public value function of an operation; None for solve and solve_system, which return Results."""
    if operation in RESULT_FUNCTIONS:
        return None
    if operation in LEGACY_NAMES:
        return getattr(mf, VALUE_NAMES.get(operation, operation))
    return getattr(importlib.import_module("mathforge._generated.values"), registry.spec(operation).public_name)


def structured_function(operation: str) -> Callable[..., object]:
    """The structured function: it returns a Result, except parse_function, which returns its value or raises."""
    if operation == "parse_function":
        return mf.functions.parse_function
    if operation in LEGACY_NAMES:
        return getattr(mf.operations, operation)
    return getattr(mf.operations, registry.spec(operation).public_name)


def argument_names(operation: str) -> tuple[str, ...]:
    """The operation's arguments in the order of the public function's parameters, without limits."""
    return tuple(name for name in inspect.signature(structured_function(operation)).parameters if name != "limits")
