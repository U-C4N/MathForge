"""Snapshot one structured call per v0.7 operation: the golden file tests/fixtures/v8/structured_v07.json.

    PYTHONPATH=<v0.7 worktree>/src python tools/snapshot_structured.py --out tests/fixtures/v8/structured_v07.json

writes ``{operation: mathforge.dumps(result)}``: the compact JSON text of the Result that the operation's structured
function, ``mathforge.operations.<operation>``, returns for one small input (``ARGUMENTS``). parse_function, which has
no structured function, stores the value that ``mathforge.functions.parse_function`` returns. Every registry and legacy
operation of v0.7 has an entry except ``verify``, the v6 pilot ``continued_fraction`` and the nine operations that v0.8
supersedes with v6 families (``SUPERSEDED``), whose records change by design.

The tool uses mathforge's public API only, so it runs against a v0.7 tree as well as against the current code; the
golden file comes from the v0.7 tree. Symbols are built as ``Symbol(name, uid=str(uuid.UUID(int=k)))``, so no text
depends on random state, and every call runs twice: both texts must agree. The file has sorted keys, one space of
indentation and LF line ends. tests/v08/test_pipeline_cost.py holds the current code to it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
import uuid

# Superseded by v6 families in v0.8 (the plan's ruling 2): their new records differ from v0.7's by design.
SUPERSEDED = frozenset({"rref", "rank", "nullspace", "solve_system", "rowspace", "columnspace", "least_squares",
                        "evaluate_algebraic", "function_taylor"})
EXCLUDED = SUPERSEDED | {"verify", "continued_fraction"}


def symbol(name: str, k: int):
    import mathforge as mf
    return mf.Symbol(name, uid=str(uuid.UUID(int=k)))


def arguments() -> dict[str, dict[str, object]]:
    """The keyword arguments of one structured call per operation, by operation name."""
    import mathforge as mf
    from mathforge import functions as f

    x, y = symbol("x", 1), symbol("y", 2)

    def rational_function(numerator: object, denominator: object) -> object:
        return mf.operations.rational_function(numerator, denominator, variable=x).unwrap()

    alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)  # The real cube root of 2.
    z = mf.ComplexAlgebraic(3, 4)
    distribution = mf.FiniteDistribution((0, 1, 2), (mf.Rational(1, 4), mf.Rational(1, 2), mf.Rational(1, 4)))
    square = mf.Matrix(((0, 2), (1, 3)))
    single = mf.real_function(f.sin(x), (x,))
    multi = mf.real_function(f.sin(x) * f.cos(y), (x, y))
    reciprocal = rational_function(1, x)
    p = mf.Polynomial.from_expression(x**4 + 4, x)
    linear = mf.Polynomial((-1, 1), x)
    return {
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
        "complex_add": dict(left=z, right=2),
        "complex_sub": dict(left=z, right=1),
        "complex_mul": dict(left=z, right=z),
        "complex_div": dict(left=z, right=2),
        "complex_conjugate": dict(value=z),
        "complex_real": dict(value=z),
        "complex_imag": dict(value=z),
        "complex_norm_squared": dict(value=z),
        "complex_sqrt": dict(value=z),
        "complex_pow": dict(base=z, n=-2),
        "quadratic_roots": dict(a=1, b=0, c=2),
        # stats
        "bernoulli": dict(p=mf.Rational(1, 3)),
        "binomial_distribution": dict(n=5, p=mf.Rational(1, 3)),
        "discrete_uniform": dict(outcomes=(1, 2, 3)),
        "hypergeometric": dict(population=10, successes=4, draws=3),
        "probability": dict(distribution=distribution, event=(1, 2)),
        "condition": dict(distribution=distribution, event=(1, 2)),
        "cdf": dict(distribution=distribution, x=1),
        "moment": dict(distribution=distribution, order=3),
        "expectation": dict(distribution=distribution),
        "distribution_variance": dict(distribution=distribution),
        "mean": dict(data=(1, 2, 4)),
        "median": dict(data=(1, 2, 4, 8)),
        "variance": dict(data=(1, 2, 4), ddof=1),
        "stddev": dict(data=(1, 2, 4), ddof=0),
        "covariance": dict(x=(1, 2, 4), y=(2, 3, 7), ddof=1),
        # linalg
        "trace": dict(A=square),
        "lu": dict(A=square),
        "det": dict(A=mf.Matrix(((1, 2), (3, 4)))),
        "inverse": dict(A=mf.Matrix(((1, 2), (3, 4)))),
        "charpoly": dict(A=mf.Matrix(((0, 2), (1, 0))), variable=x),
        "real_eigenvalues": dict(A=mf.Matrix(((0, 2), (1, 0)))),
        # analysis
        "function_differentiate": dict(expression=single, variable=x, n=1),
        "function_partial_diff": dict(expression=multi, variable=x, n=1),
        "function_gradient": dict(expression=multi, variables=(x, y)),
        "function_jacobian": dict(expressions=(multi, multi * 2), variables=(x, y)),
        "function_hessian": dict(expression=multi, variables=(x, y)),
        "function_evaluate": dict(expression=single, bindings={x: 0}),
        "partial_diff": dict(expression=x**2 * y, variable=x, n=1),
        "gradient": dict(expression=x**2 * y, variables=(x, y)),
        "jacobian": dict(expressions=(x * y, x + y), variables=(x, y)),
        "hessian": dict(expression=x**2 * y, variables=(x, y)),
        "differentiate": dict(expression=x**3, variable=x, n=1),
        "analyze_function": dict(expression=x**3 - 3 * x, variable=x),
        "extrema_values": dict(expression=x**3 - 6 * x, variable=x),
        # series
        "function_limit": dict(expression=mf.real_function(f.sin(x) / f.as_real_expr(x), (x,)), variable=x, point=0,
                               side="both"),
        "limit": dict(expression=rational_function(x**2 - 1, x - 1), variable=x, point=1, side="both"),
        # integrals
        "function_antiderivative": dict(expression=single, variable=x),
        "function_integrate": dict(expression=single, variable=x, lower=0, upper=1),
        "rational_antiderivative": dict(expression=reciprocal, variable=x),
        "rational_integrate": dict(expression=reciprocal, variable=x, lower=1, upper=2),
        "antiderivative": dict(expression=x**2, variable=x),
        "integrate": dict(expression=x**2, variable=x, lower=0, upper=1),
        # algebra: polynomials and real algebraic numbers
        "expand": dict(expression=(x + 1)**2, variable=x),
        "expand_multivariate": dict(expression=(x + y)**2, variables=(x, y)),
        "poly_divmod": dict(f=mf.Polynomial((-1, 0, 1), x), g=linear),
        "poly_gcd": dict(f=mf.Polynomial((-1, 0, 1), x), g=linear),
        "poly_xgcd": dict(f=mf.Polynomial((-1, 0, 1), x), g=linear),
        "square_free": dict(f=mf.Polynomial((1, 2, 1), x)),
        "real_roots": dict(expression=x**2 - 2, variable=x),
        "compare_real": dict(left=mf.sqrt(2), right=mf.Rational(3, 2)),
        "rational_function": dict(numerator=x + 1, denominator=x - 1, variable=x),
        "factor": dict(f=p),
        "resultant": dict(f=p, g=linear),
        "discriminant": dict(f=p),
        "algebraic": dict(expression=mf.sqrt(2) + mf.sqrt(3)),
        "algebraic_div": dict(left=alpha, right=alpha),
        "algebraic_pow": dict(base=alpha, n=-3),
        "minimal_polynomial": dict(expression=alpha, variable=x),
        "apart": dict(expression=rational_function(x + 1, x * x - 1)),
        # expr
        "evaluate": dict(expression=x**2 + 1, bindings={x: 2}),
        "substitute": dict(expression=x**2 + y, bindings={x: 2}),
        "simplify": dict(expression=x + x),
        "parse_function": dict(text="sin(x)", symbols={"x": x}, variables=None),
        # equations: the cubic takes the v2 real-set solver, not the v0.1 rules for degree two at most
        "solve": dict(problem=mf.Eq(x**3 - x, 0), for_=x),
    }


def structured(operation: str) -> Callable[..., object]:
    """The operation's structured function; parse_function's own function, which returns its value."""
    import mathforge as mf
    if operation == "parse_function":
        return mf.functions.parse_function
    return getattr(mf.operations, operation)


def call(operation: str, kwargs: dict[str, object]) -> str:
    """The compact JSON text of the structured call's result."""
    import mathforge as mf
    return mf.dumps(structured(operation)(**kwargs))


def snapshot() -> dict[str, str]:
    first, second = arguments(), arguments()
    texts = {}
    for operation in sorted(first):
        text = call(operation, first[operation])
        if call(operation, second[operation]) != text:
            raise SystemExit(f"snapshot_structured: {operation} gives different texts on two runs")
        texts[operation] = text
    return texts


def json_text(document: dict[str, str]) -> str:
    return json.dumps(document, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    options = parser.parse_args(argv)
    import mathforge
    print(f"snapshotting mathforge {mathforge.__version__} from {Path(mathforge.__file__).resolve().parent}",
          file=sys.stderr)
    document = snapshot()
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json_text(document), encoding="utf-8", newline="\n")
    print(f"{options.out} ({len(document)} entries)")


if __name__ == "__main__":
    main()
