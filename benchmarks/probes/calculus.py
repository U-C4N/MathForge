"""Calculus probes: derivatives, limits, integrals, Taylor polynomials, sums and series."""

from fractions import Fraction
from math import factorial

from . import Probe, apply, atom, coefficients, exact, exact_tuple, expression_poly, function_poly, planned, symbols

GROUP = "calculus"
X, PI, E = atom("x"), atom("pi"), atom("E")
SIN, COS, EXP = apply("sin", X), apply("cos", X), apply("exp", X)
# The catalogue (section 3) names these operations but not their area.
_SUMS = ("series", "analysis")
_TRIG = ("integrals", "analysis")


def sin_times_exp_coefficients(order: int) -> tuple[Fraction, ...]:
    """The Maclaurin coefficients of sin(x) exp(x) up to x^order, from the product of the two series."""
    return tuple(sum((Fraction((-1) ** (k // 2), factorial(k) * factorial(n - k)) for k in range(1, n + 1, 2)),
                     Fraction(0)) for n in range(order + 1))


def _function(mf, text):
    x, = symbols(mf, "x")
    return mf.parse_function(text, symbols={"x": x})


def _expression(mf, text):
    x, = symbols(mf, "x")
    return mf.parse_expression(text, symbols={"x": x})


def _x(mf):
    return symbols(mf, "x")[0]


def _pi_linear(mf, value):
    """A PiLinear value algebraic + pi_coefficient*pi as a RealFunction of x, which function_poly reads."""
    return _function(mf, f"({mf.render_text(value.algebraic)}) + ({mf.render_text(value.pi_coefficient)})*pi")


def _rational(mf, numerator, denominator):
    x = _x(mf)
    return mf.rational_function(numerator(x), denominator(x), variable=x)


def _sum_of_squares(mf):
    total = planned(mf, "sum_polynomial", *_SUMS)
    k, = symbols(mf, "k")
    return total(mf.Polynomial.from_expression(k**2, k))


def _limit(kind, value):
    return lambda v: v.kind == kind and exact(v.value) == value


PROBES = (
    Probe("calculus_derivative_polynomial", GROUP, "Differentiate x^3 - 2x.",
          lambda mf: mf.diff(_expression(mf, "x^3 - 2*x"), _x(mf)),
          lambda v: expression_poly(v) == 3 * X**2 - 2),
    Probe("calculus_derivative_product", GROUP, "Differentiate sin(x) x^2.",
          lambda mf: mf.operations.function_differentiate(_function(mf, "sin(x)*x^2"), _x(mf)),
          lambda v: function_poly(v) == COS * X**2 + 2 * X * SIN),
    Probe("calculus_second_derivative_elementary", GROUP, "What is the second derivative of exp(x) sin(x)?",
          lambda mf: mf.operations.function_differentiate(_function(mf, "exp(x)*sin(x)"), _x(mf), 2),
          lambda v: function_poly(v) == 2 * EXP * COS),
    Probe("calculus_second_derivative_polynomial", GROUP, "What is the second derivative of x^4?",
          lambda mf: mf.diff(_expression(mf, "x^4"), _x(mf), 2),
          lambda v: expression_poly(v) == 12 * X**2),
    Probe("calculus_limit_sinc", GROUP, "What is the limit of sin(x)/x as x tends to 0?",
          lambda mf: mf.operations.function_limit(_function(mf, "sin(x)/x"), _x(mf), 0), _limit("finite", 1)),
    Probe("calculus_limit_compound_interest", GROUP, "What is the limit of (1 + 1/x)^x as x tends to +infinity?",
          lambda mf: mf.operations.function_limit(_function(mf, "(1 + 1/x)^x"), _x(mf), "+inf"),
          lambda v: v.kind == "finite" and function_poly(v.value) == E),
    Probe("calculus_limit_rational_plus_infinity", GROUP,
          "What is the limit of (3x^2 + 1)/(x^2 - 5) as x tends to +infinity?",
          lambda mf: mf.limit(_rational(mf, lambda x: 3 * x**2 + 1, lambda x: x**2 - 5), _x(mf), "+inf"),
          _limit("finite", 3)),
    Probe("calculus_limit_rational_minus_infinity", GROUP, "What is the limit of x/(x^2 + 1) as x tends to -infinity?",
          lambda mf: mf.limit(_rational(mf, lambda x: x, lambda x: x**2 + 1), _x(mf), "-inf"), _limit("finite", 0)),
    Probe("calculus_integral_polynomial", GROUP, "Integrate x^2 from 0 to 1.",
          lambda mf: mf.integrate(_x(mf) ** 2, _x(mf), 0, 1), lambda v: exact(v) == Fraction(1, 3)),
    Probe("calculus_antiderivative_polynomial", GROUP, "Find an antiderivative of x^2 + 1.",
          lambda mf: mf.antiderivative(_x(mf) ** 2 + 1, _x(mf)),
          lambda v: expression_poly(v) == X + X**3 / 3),
    Probe("calculus_integral_arctan", GROUP, "Integrate 1/(x^2 + 1) from 0 to 1.",
          lambda mf: mf.operations.rational_integrate(_rational(mf, lambda x: 1, lambda x: x**2 + 1), _x(mf), 0, 1),
          lambda v: function_poly(v) == PI / 4),
    Probe("calculus_antiderivative_rational_cubic", GROUP, "Find an antiderivative of 1/(x^3 + x + 1).",
          lambda mf: mf.operations.rational_antiderivative(_rational(mf, lambda x: 1, lambda x: x**3 + x + 1),
                                                           _x(mf))),
    Probe("calculus_antiderivative_sin_squared", GROUP, "Find an antiderivative of sin(x)^2.",
          lambda mf: planned(mf, "trig_power_antiderivative", *_TRIG)(_function(mf, "sin(x)^2"), _x(mf))),
    Probe("calculus_antiderivative_x_exp", GROUP, "Find an antiderivative of x exp(x).",
          lambda mf: mf.operations.function_antiderivative(_function(mf, "x*exp(x)"), _x(mf)),
          lambda v: function_poly(v) == X * EXP - EXP),
    Probe("calculus_integral_reciprocal", GROUP, "Integrate 1/x from 1 to 2.",
          lambda mf: mf.operations.function_integrate(_function(mf, "1/x"), _x(mf), 1, 2),
          lambda v: function_poly(v) == apply("log", 2)),
    # trig_power_integral takes its bounds as rational multiples of pi: 3 stands for 3*pi, 1 for pi.
    Probe("calculus_integral_sin_squared", GROUP, "Integrate sin(x)^2 from 0 to 3*pi.",
          lambda mf: planned(mf, "trig_power_integral", *_TRIG)(_function(mf, "sin(x)^2"), _x(mf), 0, 3)),
    Probe("calculus_integral_sin_squared_to_pi", GROUP, "Integrate sin(x)^2 from 0 to pi.",
          lambda mf: _pi_linear(mf, planned(mf, "trig_power_integral", *_TRIG)(_function(mf, "sin(x)^2"), _x(mf),
                                                                               0, 1)),
          lambda v: function_poly(v) == PI / 2),
    Probe("calculus_improper_divergent", GROUP, "Integrate 1/x^2 from 0 to 1.",
          lambda mf: mf.operations.function_integrate(_function(mf, "1/x^2"), _x(mf), 0, 1)),
    Probe("calculus_improper_gaussian", GROUP, "Integrate exp(-x^2) from 0 to +infinity.",
          lambda mf: mf.operations.function_integrate(_function(mf, "exp(-x^2)"), _x(mf), 0, "+inf")),
    Probe("calculus_taylor_exp", GROUP, "What is the Taylor polynomial of exp(x) at 0 of order 5?",
          lambda mf: mf.taylor(_function(mf, "exp(x)"), _x(mf), 0, 5),
          lambda v: exact_tuple(v.coefficients) == tuple(Fraction(1, factorial(n)) for n in range(6))),
    Probe("calculus_taylor_sin_coefficients", GROUP, "List the Taylor coefficients of sin(x) at 0 up to x^7.",
          lambda mf: mf.taylor(_function(mf, "sin(x)"), _x(mf), 0, 7).coefficients,
          lambda v: exact_tuple(v) == (0, 1, 0, Fraction(-1, 6), 0, Fraction(1, 120), 0, Fraction(-1, 5040))),
    Probe("calculus_integral_x_to_100", GROUP, "Integrate x^100 from 0 to 1.",
          lambda mf: mf.integrate(_x(mf) ** 100, _x(mf), 0, 1), lambda v: exact(v) == Fraction(1, 101)),
    Probe("calculus_sum_of_squares", GROUP, "What is the sum of k^2 for k = 1 to n (and so for n = 100)?",
          _sum_of_squares, lambda v: coefficients(v) == (0, Fraction(1, 6), Fraction(1, 2), Fraction(1, 3))),
    Probe("calculus_geometric_series", GROUP, "What is the sum of 1/2^k for k = 1 to infinity?",
          lambda mf: planned(mf, "geometric_series", *_SUMS)(mf.Rational(1, 2), mf.Rational(1, 2)),
          lambda v: exact(v) == 1),
    Probe("calculus_geometric_sum", GROUP, "What is 1 + 3 + 9 + ... + 3^9?",
          lambda mf: planned(mf, "geometric_sum", *_SUMS)(1, 3, 10), lambda v: exact(v) == 29524),
    Probe("calculus_series_coefficients", GROUP, "List the power series coefficients of sin(x) exp(x) at 0 to x^12.",
          lambda mf: planned(mf, "series_coefficients", *_SUMS)(_function(mf, "sin(x)*exp(x)"), _x(mf), 0, 12),
          lambda v: exact_tuple(getattr(v, "coefficients", v)) == sin_times_exp_coefficients(12)),
)
