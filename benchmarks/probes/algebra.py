"""Algebra probes: expansion, factorization, simplification, roots and complex arithmetic."""

from fractions import Fraction

from . import Probe, apply, atom, coefficients, exact, expression_poly, is_monomial_sum, point_set, symbols

GROUP = "algebra"
X, Y = atom("x"), atom("y")


def _parse(mf, text):
    x, y = symbols(mf, "x y")
    return mf.parse_expression(text, symbols={"x": x, "y": y})


def _polynomial(mf, build):
    x, = symbols(mf, "x")
    return mf.Polynomial.from_expression(build(x), x)


def _expand_product(mf):
    x, = symbols(mf, "x")
    return mf.expand(_parse(mf, "(x + 1)^3*(x - 2)"), x)


def _expand_multivariate(mf):
    x, y = symbols(mf, "x y")
    return mf.expand_multivariate(_parse(mf, "(x + y)^2*(x - y)"), (x, y))


def _factor_multivariate(mf):
    x, y = symbols(mf, "x y")
    return mf.factor(mf.Polynomial.from_expression(x**2 * y - y**3, x))


def _partial_fractions(mf):
    x, = symbols(mf, "x")
    return mf.apart(mf.rational_function(1, x**2 - 1, variable=x))


def _substitute_expression(mf):
    x, = symbols(mf, "x")
    return mf.substitute(_parse(mf, "x^2 + y"), {x: _parse(mf, "y + 1")})


def _solve_cubic(mf):
    x, = symbols(mf, "x")
    return mf.solve(mf.parse_problem("x^3 - 1 = 0", symbols={"x": x}), for_=x)


def _expand_binomial(mf):
    x, = symbols(mf, "x")
    return mf.expand(_parse(mf, "(x + 1)^200"), x)


def _count_real_roots(mf):
    count = mf.algebra.count_real_roots
    return count(_polynomial(mf, lambda x: x**5 - 3 * x + 1), 0, 2)  # The interval [lower, upper] is closed.


def _interpolate(mf):
    interpolate = mf.algebra.interpolate
    return interpolate((0, 1, 2), (1, 3, 7), symbols(mf, "x")[0])


def _factors(value):
    """A PolynomialFactorization as its leading coefficient and the sorted (coefficients, multiplicity) pairs."""
    pairs = sorted((coefficients(item.polynomial), item.multiplicity) for item in value.factors)
    return exact(value.coefficient), pairs


def _rational_function(value):
    """A RationalFunction as (numerator, denominator, excluded) coefficient tuples."""
    return (coefficients(value.numerator), coefficients(value.denominator),
            tuple(coefficients(item) for item in value.excluded))


def _partial_fraction_terms(value):
    return (coefficients(value.polynomial),
            sorted((coefficients(term.numerator), coefficients(term.factor), term.power) for term in value.terms))


def _complex(value):
    return exact(value.real), exact(value.imag)


PROBES = (
    Probe("algebra_expand_product", GROUP, "Expand (x + 1)^3 (x - 2).", _expand_product,
          lambda v: is_monomial_sum(v) and expression_poly(v) == (X + 1) ** 3 * (X - 2)),
    Probe("algebra_expand_multivariate", GROUP, "Expand (x + y)^2 (x - y) in x and y.", _expand_multivariate,
          lambda v: is_monomial_sum(v) and expression_poly(v) == (X + Y) ** 2 * (X - Y)),
    Probe("algebra_factor_quartic", GROUP, "Factor x^4 - 5x^2 + 4 over the rationals.",
          lambda mf: mf.factor(_polynomial(mf, lambda x: x**4 - 5 * x**2 + 4)),
          lambda v: _factors(v) == (1, [((-2, 1), 1), ((-1, 1), 1), ((1, 1), 1), ((2, 1), 1)])),
    Probe("algebra_factor_multivariate", GROUP, "Factor x^2 y - y^3.", _factor_multivariate),
    Probe("algebra_simplify_cancel", GROUP, "Simplify (x^2 - 1)/(x - 1).",
          lambda mf: mf.simplify(_parse(mf, "(x^2 - 1)/(x - 1)")),
          lambda v: _rational_function(v) == ((1, 1), (1,), ((-1, 1),))),
    Probe("algebra_simplify_sum", GROUP, "Simplify x^2/x^2 + 1/x.",
          lambda mf: mf.simplify(_parse(mf, "x^2/x^2 + 1/x")),
          lambda v: _rational_function(v) == ((1, 1), (0, 1), ((0, 1),))),
    Probe("algebra_polynomial_gcd", GROUP, "What is the gcd of x^4 - 1 and x^2 - 1?",
          lambda mf: mf.poly_gcd(_polynomial(mf, lambda x: x**4 - 1), _polynomial(mf, lambda x: x**2 - 1)),
          lambda v: coefficients(v) == (-1, 0, 1)),
    Probe("algebra_partial_fractions", GROUP, "Decompose 1/(x^2 - 1) into partial fractions.", _partial_fractions,
          lambda v: _partial_fraction_terms(v) == ((), [((Fraction(-1, 2),), (1, 1), 1),
                                                        ((Fraction(1, 2),), (-1, 1), 1)])),
    Probe("algebra_substitute_number", GROUP, "Evaluate x^2 + 1 at x = 3.",
          lambda mf: mf.substitute(_parse(mf, "x^2 + 1"), {symbols(mf, "x")[0]: 3}),
          lambda v: exact(v) == 10),
    Probe("algebra_substitute_expression", GROUP, "Substitute x = y + 1 into x^2 + y.", _substitute_expression,
          lambda v: expression_poly(v) == Y**2 + 3 * Y + 1),
    Probe("algebra_quadratic_complex_roots", GROUP, "Find the complex roots of x^2 + 1.",
          lambda mf: mf.quadratic_roots(1, 0, 1),
          lambda v: sorted(_complex(item.root) + (item.multiplicity,) for item in v) == [(0, -1, 1), (0, 1, 1)]),
    Probe("algebra_cubic_complex_roots", GROUP, "Find all complex roots of x^3 - 1.",
          lambda mf: mf.algebra.complex_roots(_polynomial(mf, lambda x: x**3 - 1))),
    Probe("algebra_cubic_real_solve", GROUP, "Solve x^3 - 1 = 0 with the real solver.", _solve_cubic,
          lambda v: point_set(v) == {1}),
    Probe("algebra_denest_with_simplify", GROUP, "Simplify sqrt(3 + 2 sqrt(2)) with simplify.",
          lambda mf: mf.simplify(_parse(mf, "sqrt(3 + 2*sqrt(2))")),
          lambda v: expression_poly(v) == 1 + apply("sqrt", 2)),
    Probe("algebra_denest_sqrt", GROUP, "Denest sqrt(3 + 2 sqrt(2)).",
          lambda mf: mf.algebra.denest_sqrt(3, 2, 2)),
    Probe("algebra_minimal_polynomial", GROUP, "Find the minimal polynomial of sqrt(2) + sqrt(3).",
          lambda mf: mf.minimal_polynomial(mf.sqrt(2) + mf.sqrt(3), symbols(mf, "x")[0]),
          lambda v: coefficients(v) == (1, 0, -10, 0, 1)),
    Probe("algebra_resultant", GROUP, "What is the resultant of x^2 - 2 and x^2 - 3?",
          lambda mf: mf.resultant(_polynomial(mf, lambda x: x**2 - 2), _polynomial(mf, lambda x: x**2 - 3)),
          lambda v: exact(v) == 1),
    Probe("algebra_complex_product", GROUP, "Multiply (2 + 3i)(1 - i).",
          lambda mf: mf.complex_mul(mf.complex_algebraic(2, 3), mf.complex_algebraic(1, -1)),
          lambda v: _complex(v) == (5, 1)),
    Probe("algebra_complex_sqrt", GROUP, "What is the principal square root of 3 + 4i?",
          lambda mf: mf.complex_sqrt(mf.complex_algebraic(3, 4)),
          lambda v: _complex(v) == (2, 1)),
    Probe("algebra_complex_quotient", GROUP, "Divide (1 + i)/(1 - i).",
          lambda mf: mf.complex_div(mf.complex_algebraic(1, 1), mf.complex_algebraic(1, -1)),
          lambda v: _complex(v) == (0, 1)),
    Probe("algebra_expand_binomial_200", GROUP, "Expand (x + 1)^200.", _expand_binomial,
          lambda v: is_monomial_sum(v) and expression_poly(v) == (X + 1) ** 200),
    Probe("algebra_count_real_roots", GROUP, "How many distinct real roots does x^5 - 3x + 1 have in [0, 2]?",
          _count_real_roots, lambda v: exact(v) == 2),
    Probe("algebra_interpolate", GROUP, "Which polynomial of degree at most 2 passes through (0, 1), (1, 3), (2, 7)?",
          _interpolate, lambda v: coefficients(v) == (1, 1, 1)),
)
