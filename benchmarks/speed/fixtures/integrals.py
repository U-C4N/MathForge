"""Speed fixtures of the integrals area: antiderivative and integrate (polynomials), function_antiderivative,
function_integrate, rational_antiderivative and rational_integrate.

Polynomial inputs come in the two forms the SDK accepts: a ``Polynomial`` object, compared with a prebuilt SymPy
``Poly`` (the polynomial-object level, where ``Poly.integrate`` is the reference), and an ``Expression`` tree, compared
with the SymPy ``Expr`` of the same polynomial (where the Poly construction is part of the timed call, or
``integrate`` is). Elementary and rational integrands are parsed from text (``mathforge.parse_function`` and ``sympify``
of the same text) or built from the same Python arithmetic on a MathForge and a SymPy symbol.

A definite integral of a polynomial is the difference of its antiderivative at the bounds, exactly, so the SymPy side
may chain ``Poly.integrate`` and ``Poly.eval``. For elementary and rational integrands the antiderivative evaluated at
the bounds is not the integral in general (poles, branches), so the references are single calls: ``integrate`` and
its variants. Indefinite integrals are compared by derivative: two primitives agree when their difference has
derivative zero. Every ``reference`` lists the candidates with their median time in microseconds on that fixture's
inputs, with SymPy's caches cleared before each call (Python 3.14.6, SymPy 1.14.0, python ground types); there g is
the integrand as a function, f as a rational function, x its symbol and (a, b) the bounds. The chosen call is the
fastest, except that a candidate within 15% of the conventional ``integrate`` is not preferred to it.
"""

from __future__ import annotations

from fractions import Fraction
import itertools
import math

import mathforge as mf
from benchmarks.speed.fixture import EdgeCase, Fixture
from benchmarks.speed.fixtures.analysis import (
    X, Y, big_polynomial, big_raw, cubic as _cubic, decode, deep_raw, edge, equal, high_degree,
    mf_expression as _mf_expression, mf_number as _mf_number, mf_polynomial as _mf_polynomial,
    polynomial_coefficients as _coefficients, positive_symbol, same_polynomial as _same_polynomial,
    sp_expression as _sp_expression, sp_number as _sp_number, sp_poly as _sp_poly, sympy_symbols, to_sympy,
    type_edges)

_REFERENCES: dict[str, str] = {
    "antiderivative/micro/polynomial_object_cubic": (
        "prebuilt Poly.integrate() 53 us (chosen); the other calls take an Expr, not a Poly"),
    "antiderivative/micro/expression_cubic": (
        "Poly(expr, x).integrate() 717 us (chosen); integrate(expr, x) 3282 us; expr.integrate(x) "
        "3284 us; Poly.integrate() on a prebuilt Poly takes a different kind of object (54 us)"),
    "antiderivative/realistic/expression_degree12_rational": (
        "Poly(expr, x).integrate() 2304 us (chosen); integrate(expr, x) 9145 us; expr.integrate(x) "
        "9495 us; Poly.integrate() on a prebuilt Poly takes a different kind of object (51 us)"),
    "antiderivative/realistic/polynomial_object_degree40_rational": (
        "prebuilt Poly.integrate() 92 us (chosen); the other calls take an Expr, not a Poly"),
    "antiderivative/realistic/scaled_polynomial_object_degree100": (
        "prebuilt Poly.integrate() 272 us (chosen); the other calls take an Expr, not a Poly"),
    "integrate/micro/polynomial_object_cubic_integer_bounds": (
        "prebuilt Poly: integrate, eval 96 us (chosen); the other calls take an Expr, not a Poly"),
    "integrate/micro/expression_cubic_integer_bounds": (
        "Poly(expr, x): integrate, eval 713 us (chosen); integrate(expr, (x, a, b)) 8555 us; "
        "integrate and eval on a prebuilt Poly take a different kind of object (81 us)"),
    "integrate/realistic/expression_degree12_fraction_bounds": (
        "Poly(expr, x): integrate, eval 2219 us (chosen); integrate(expr, (x, a, b)) 20617 us; "
        "integrate and eval on a prebuilt Poly take a different kind of object (123 us)"),
    "integrate/realistic/polynomial_object_degree40_fraction_bounds": (
        "prebuilt Poly: integrate, eval 248 us (chosen); the other calls take an Expr, not a Poly"),
    "integrate/realistic/scaled_polynomial_object_degree100": (
        "prebuilt Poly: integrate, eval 521 us (chosen); the other calls take an Expr, not a Poly"),
    "function_antiderivative/micro/cos_kx": (
        "integrate(g, x) 2486 us (chosen); Integral(g, x).doit() 2488 us; manualintegrate(g, x) 15776 "
        "us; heurisch(g, x) 16789 us"),
    "function_antiderivative/realistic/polynomial_times_exp": (
        "integrate(g, x) 18795 us (chosen); risch_integrate(g, x) 16013 us; Integral(g, x).doit() "
        "19488 us; heurisch(g, x) 39137 us; manualintegrate(g, x) 404057 us; integrate(g, x) was the "
        "v0.7 reference; risch_integrate is 15% faster, within the margin"),
    "function_antiderivative/realistic/x_exp": (
        "integrate(g, x) 11473 us (chosen); risch_integrate(g, x) 10361 us; Integral(g, x).doit() "
        "11474 us; heurisch(g, x) 16556 us; manualintegrate(g, x) 31087 us"),
    "function_antiderivative/realistic/x_squared_sin": (
        "manualintegrate(g, x) 35731 us (chosen); heurisch(g, x) 55135 us; integrate(g, x) 57252 us; "
        "Integral(g, x).doit() 57778 us"),
    "function_antiderivative/realistic/x_log_x": (
        "integrate(g, x) 16592 us (chosen); risch_integrate(g, x) 14753 us; Integral(g, x).doit() "
        "17072 us; heurisch(g, x) 34691 us; manualintegrate(g, x) 97197 us"),
    "function_antiderivative/realistic/rational_atan": (
        "manualintegrate(g, x) 3866 us (chosen); risch_integrate(g, x) 13726 us; integrate(g, x) "
        "13770 us; Integral(g, x).doit() 13809 us; heurisch(g, x) 14315 us"),
    "function_antiderivative/realistic/scaled_degree8_times_exp": (
        "risch_integrate(g, x) 22566 us (chosen); integrate(g, x) 26820 us; Integral(g, x).doit() "
        "27274 us; heurisch(g, x) 108926 us; manualintegrate(g, x) 2596800 us"),
    "function_integrate/micro/cos_kx_unit_interval": (
        "integrate(g, (x, a, b)) 5458 us (chosen); integrate(..., meijerg=False) 5252 us; Integral(g, "
        "(x, a, b)).doit() 5356 us; integrate(..., manual=True) 19686 us"),
    "function_integrate/realistic/rational_squared": (
        "integrate(g, (x, a, b)) 29312 us (chosen); integrate(..., meijerg=False) 29430 us; "
        "Integral(g, (x, a, b)).doit() 29798 us; integrate(..., manual=True) 2084902 us; integrate(g, "
        "(x, a, b)) was the v0.7 reference"),
    "function_integrate/realistic/x_exp_unit_interval": (
        "integrate(g, (x, a, b)) 14392 us (chosen); integrate(..., meijerg=False) 14640 us; "
        "Integral(g, (x, a, b)).doit() 14654 us; integrate(..., manual=True) 35608 us"),
    "function_integrate/realistic/x_squared_sin_unit_interval": (
        "integrate(g, (x, a, b)) 64275 us (chosen); Integral(g, (x, a, b)).doit() 61245 us; "
        "integrate(..., meijerg=False) 61711 us; integrate(..., manual=True) 80783 us"),
    "function_integrate/realistic/x_log_x_one_to_two": (
        "integrate(g, (x, a, b)) 22596 us (chosen); Integral(g, (x, a, b)).doit() 22225 us; "
        "integrate(..., meijerg=False) 22394 us; integrate(..., manual=True) 99791 us"),
    "function_integrate/realistic/scaled_degree8_times_exp": (
        "integrate(g, (x, a, b)) 56790 us (chosen); Integral(g, (x, a, b)).doit() 54543 us; "
        "integrate(..., meijerg=False) 54946 us; integrate(..., manual=True) 5460342 us"),
    "rational_antiderivative/micro/constant_over_linear": (
        "integrate(f, x) 3487 us (chosen); ratint(f, x) 7009 us; manualintegrate(f, x) 25483 us"),
    "rational_antiderivative/realistic/atan_quadratic": (
        "manualintegrate(f, x) 4123 us (chosen); ratint(f, x) 12241 us; integrate(f, x) 13928 us"),
    "rational_antiderivative/realistic/linear_over_irreducible_quadratic": (
        "integrate(f, x) 23062 us (chosen); ratint(f, x) 19865 us; manualintegrate(f, x) 114588 us"),
    "rational_antiderivative/realistic/two_simple_poles": (
        "ratint(f, x) 13015 us (chosen); integrate(f, x) 17312 us; manualintegrate(f, x) 728071 us"),
    "rational_antiderivative/realistic/repeated_quadratic": (
        "ratint(f, x) 11339 us (chosen); integrate(f, x) 14802 us; manualintegrate(f, x) 2412201 us"),
    "rational_antiderivative/realistic/scaled_four_factors": (
        "integrate(f, x) 110320 us (chosen); ratint(f, x) 100471 us; manualintegrate(f, x) 2933214 us"),
    "rational_integrate/micro/constant_over_linear": (
        "integrate(f, (x, a, b)) 7366 us (chosen); integrate(..., meijerg=False) 7484 us; Integral(f, "
        "(x, a, b)).doit() 7488 us; integrate(..., manual=True) 24676 us"),
    "rational_integrate/realistic/atan_unit_interval": (
        "integrate(..., manual=True) 8454 us (chosen); Integral(f, (x, a, b)).doit() 17249 us; "
        "integrate(..., meijerg=False) 17418 us; integrate(f, (x, a, b)) 17922 us"),
    "rational_integrate/realistic/two_simple_poles_unit_interval": (
        "integrate(f, (x, a, b)) 28218 us (chosen); integrate(..., meijerg=False) 27786 us; "
        "Integral(f, (x, a, b)).doit() 27993 us; integrate(..., manual=True) 744727 us"),
    "rational_integrate/realistic/x_over_quadratic": (
        "integrate(f, (x, a, b)) 16127 us (chosen); Integral(f, (x, a, b)).doit() 16076 us; "
        "integrate(..., meijerg=False) 16201 us; integrate(..., manual=True) 20173 us"),
    "rational_integrate/realistic/irreducible_quadratic": (
        "integrate(f, (x, a, b)) 27263 us (chosen); Integral(f, (x, a, b)).doit() 27204 us; "
        "integrate(..., meijerg=False) 27560 us; integrate(..., manual=True) 134038 us"),
    "rational_integrate/realistic/scaled_double_pole": (
        "integrate(f, (x, a, b)) 38618 us (chosen); integrate(..., meijerg=False) 36636 us; "
        "Integral(f, (x, a, b)).doit() 39607 us; integrate(..., manual=True) 15439321 us"),
}


def _reference(operation: str, size: str, label: str) -> str:
    return _REFERENCES[f"{operation}/{size}/{label}"]


# --- polynomial data (the builders live in the analysis module) -------------------------------------------------------


def _bounds(i: int):
    """Integer bounds in [-5, 5]: 121 pairs in turn (4099 is coprime to 121)."""
    lower, upper = decode((i * 4099 + 5) % 121, 11, 11)
    return lower - 5, upper - 5


def _same_number(mf_value, sympy_value) -> bool:
    """A Rational or a real algebraic number against a SymPy number."""
    import sympy as sp
    if type(mf_value) is mf.Rational:
        return sp.Rational(mf_value.numerator, mf_value.denominator) == sympy_value
    return equal(to_sympy(mf_value, {}), sympy_value)


# --- antiderivative of a polynomial --------------------------------------------------------------------------------


def _antiderivative_fixture(size, label, coefficients_of, sympy_call, *, variants, form):
    """form is "object" (a Polynomial) or "expression" (an Expression)."""
    build = _mf_polynomial if form == "object" else _mf_expression

    def mathforge(i):
        return mf.antiderivative, (build(coefficients_of(i)), X), {}

    def sympy(i):
        return sympy_call(coefficients_of(i))

    return Fixture("antiderivative", size, label, mathforge, sympy, _same_polynomial,
                   _reference("antiderivative", size, label), variants)


def _integrate_poly(coefficients):
    poly = _sp_poly(coefficients)
    return lambda: poly.integrate()


def _integrate_via_poly(coefficients):
    expression, x = _sp_expression(coefficients), sympy_symbols()[X]
    import sympy as sp
    return lambda: sp.Poly(expression, x).integrate()


# --- definite integral of a polynomial -----------------------------------------------------------------------------


def _definite_fixture(size, label, coefficients_of, bounds_of, sympy_call, *, variants, form,
                      mf_bounds=None, v07=""):
    """bounds_of(i) -> (lower, upper) ints or Fractions; mf_bounds(i) gives them as MathForge values instead."""
    build = _mf_polynomial if form == "object" else _mf_expression

    def mathforge(i):
        lower, upper = (bounds_of(i) if mf_bounds is None else mf_bounds(i))
        return mf.integrate, (build(coefficients_of(i)), X, _mf_number(lower), _mf_number(upper)), {}

    def sympy(i):
        return sympy_call(coefficients_of(i), bounds_of(i))

    return Fixture("integrate", size, label, mathforge, sympy, _same_number, _reference("integrate", size, label),
                   variants, v07)


def _definite_poly(coefficients, bounds):
    poly, (lower, upper) = _sp_poly(coefficients), (_sp_number(b) for b in bounds)

    def call():
        primitive = poly.integrate()
        return primitive.eval(upper) - primitive.eval(lower)
    return call


def _definite_via_poly(coefficients, bounds):
    import sympy as sp
    expression, x, (lower, upper) = _sp_expression(coefficients), sympy_symbols()[X], (_sp_number(b) for b in bounds)

    def call():
        primitive = sp.Poly(expression, x).integrate()
        return primitive.eval(upper) - primitive.eval(lower)
    return call


# --- elementary integrands -----------------------------------------------------------------------------------------


def _function(text: str):
    return mf.parse_function(text, symbols={"x": X})


def _expression(text: str):
    import sympy as sp
    return sp.sympify(text, locals={"x": sympy_symbols()[X]})


def _without_abs_in_logs(expression):
    """log(|u|) as log(u): the same derivative u'/u, which SymPy can then simplify over the complex plane."""
    import sympy as sp
    return expression.replace(lambda e: e.func is sp.log and isinstance(e.args[0], sp.Abs),
                              lambda e: sp.log(e.args[0].args[0]))


def _same_primitive(mf_value, sympy_value) -> bool:
    """Two primitives of the same integrand differ by a constant: the derivative of the difference is zero."""
    import sympy as sp
    symbols = sympy_symbols()
    if type(mf_value) is mf.RationalPrimitive:
        mf_value = mf_value.as_function()
    ours = _without_abs_in_logs(to_sympy(mf_value, symbols))
    return equal(sp.diff(ours - _without_abs_in_logs(sympy_value), symbols[X]), 0)


def _primitive_fixture(size, label, text_of, sympy_call, *, variants, v07=""):
    def mathforge(i):
        return mf.function_antiderivative, (_function(text_of(i)), X), {}

    def sympy(i):
        return sympy_call(text_of(i))

    return Fixture("function_antiderivative", size, label, mathforge, sympy, _same_primitive,
                   _reference("function_antiderivative", size, label), variants, v07)


def _integrate_text(text):
    import sympy as sp
    function, x = _expression(text), sympy_symbols()[X]
    return lambda: sp.integrate(function, x)


def _manual_text(text):
    from sympy.integrals.manualintegrate import manualintegrate
    function, x = _expression(text), sympy_symbols()[X]
    return lambda: manualintegrate(function, x)


def _risch_text(text):
    from sympy.integrals.risch import risch_integrate
    function, x = _expression(text), sympy_symbols()[X]
    return lambda: risch_integrate(function, x)


def _definite_fixture_text(size, label, text_of, lower, upper, sympy_call, *, variants, v07=""):
    def mathforge(i):
        return mf.function_integrate, (_function(text_of(i)), X, lower, upper), {}

    def sympy(i):
        return sympy_call(text_of(i), lower, upper)

    return Fixture("function_integrate", size, label, mathforge, sympy, lambda a, b: equal(to_sympy(a, {}), b),
                   _reference("function_integrate", size, label), variants, v07)


def _definite_text(text, lower, upper):
    import sympy as sp
    function, x = _expression(text), sympy_symbols()[X]
    return lambda: sp.integrate(function, (x, lower, upper))


# --- rational integrands -------------------------------------------------------------------------------------------


_LINE_TRIPLES = tuple(t for t in itertools.product(range(1, 10), range(1, 10), range(-9, 10))
                      if math.gcd(math.gcd(t[0], t[1]), t[2]) == 1)


def _line_case(i: int) -> tuple[int, int, int]:
    """(p, q, r) of p/(q x - r) with p, q in [1, 9], r in [-9, 9] and no common factor, so that no two are one
    function: the triples in turn with a stride of 7,919, which is prime and larger than their number."""
    return _LINE_TRIPLES[i * 7919 % len(_LINE_TRIPLES)]


def _rational_primitive_fixture(size, label, parts_of, sympy_call, *, variants):
    """parts_of(i) is a function x -> (numerator, denominator), evaluated on the MathForge and the SymPy symbol."""
    def mathforge(i):
        numerator, denominator = parts_of(i)(X)
        return mf.rational_antiderivative, (mf.rational_function(numerator, denominator, variable=X), X), {}

    def sympy(i):
        x = sympy_symbols()[X]
        numerator, denominator = parts_of(i)(x)
        return sympy_call(numerator / denominator, x)

    return Fixture("rational_antiderivative", size, label, mathforge, sympy, _same_primitive,
                   _reference("rational_antiderivative", size, label), variants)


def _ratint_call(function, x):
    from sympy.integrals.rationaltools import ratint
    return lambda: ratint(function, x)


def _integrate_call(function, x):
    import sympy as sp
    return lambda: sp.integrate(function, x)


def _manual_call(function, x):
    from sympy.integrals.manualintegrate import manualintegrate
    return lambda: manualintegrate(function, x)


def _rational_definite_fixture(size, label, parts_of, bounds_of, sympy_call, *, variants):
    def mathforge(i):
        numerator, denominator = parts_of(i)(X)
        lower, upper = bounds_of(i)
        return (mf.rational_integrate, (mf.rational_function(numerator, denominator, variable=X), X, lower, upper), {})

    def sympy(i):
        x = sympy_symbols()[X]
        numerator, denominator = parts_of(i)(x)
        lower, upper = bounds_of(i)
        return sympy_call(numerator / denominator, x, lower, upper)

    return Fixture("rational_integrate", size, label, mathforge, sympy, lambda a, b: equal(to_sympy(a, {}), b),
                   _reference("rational_integrate", size, label), variants)


def _definite_integrate_call(function, x, lower, upper):
    import sympy as sp
    return lambda: sp.integrate(function, (x, lower, upper))


def _definite_manual_call(function, x, lower, upper):
    import sympy as sp
    return lambda: sp.integrate(function, (x, lower, upper), manual=True)


def _positive_line_parts(i):
    """p/(x + c) with c in [1, 9] and p in [1, 9]."""
    c, p = decode(i % 81, 9, 9)
    return lambda x: (p + 1, x + (c + 1))


def _positive_line_bounds(i):
    """Bounds a in {0, 1} and b in [2, 10] for p/(x + c): the pole -c lies left of both."""
    a, b = decode(i // 81 % 18, 2, 9)
    return a, b + 2


FIXTURES = (
    # antiderivative
    _antiderivative_fixture("micro", "polynomial_object_cubic", _cubic, _integrate_poly, variants=1000,
                            form="object"),
    _antiderivative_fixture("micro", "expression_cubic", _cubic, _integrate_via_poly, variants=1000,
                            form="expression"),
    _antiderivative_fixture("realistic", "expression_degree12_rational",
                            lambda i: _coefficients(12, 200 + i, rational=True), _integrate_via_poly, variants=8,
                            form="expression"),
    _antiderivative_fixture("realistic", "polynomial_object_degree40_rational",
                            lambda i: _coefficients(40, 300 + i, rational=True), _integrate_poly, variants=8,
                            form="object"),
    _antiderivative_fixture("realistic", "scaled_polynomial_object_degree100",
                            lambda i: _coefficients(100, 400 + i), _integrate_poly, variants=4, form="object"),
    # integrate
    _definite_fixture("micro", "polynomial_object_cubic_integer_bounds", _cubic, _bounds, _definite_poly,
                      variants=1000, form="object"),
    _definite_fixture("micro", "expression_cubic_integer_bounds", _cubic, _bounds, _definite_via_poly, variants=1000,
                      form="expression"),
    _definite_fixture("realistic", "expression_degree12_fraction_bounds",
                      lambda i: _coefficients(12, 500 + i, rational=True),
                      lambda i: (Fraction(-1, 2), Fraction(3, 2)), _definite_via_poly, variants=8,
                      form="expression"),
    _definite_fixture("realistic", "polynomial_object_degree40_fraction_bounds",
                      lambda i: _coefficients(40, 600 + i, rational=True),
                      lambda i: (Fraction(-1, 2), Fraction(3, 2)), _definite_poly, variants=8, form="object"),
    _definite_fixture("realistic", "scaled_polynomial_object_degree100",
                      lambda i: _coefficients(100, 700 + i), lambda i: (-1, 1), _definite_poly, variants=4,
                      form="object"),
    # function_antiderivative
    _primitive_fixture("micro", "cos_kx", lambda i: f"cos({i + 1}*x)", _integrate_text, variants=1000),
    _primitive_fixture("realistic", "polynomial_times_exp", lambda i: "(x**3-2*x+1)*exp(2*x)", _integrate_text,
                       variants=1, v07="elementary_integral"),
    _primitive_fixture("realistic", "x_exp", lambda i: "x*exp(x)", _integrate_text, variants=1),
    _primitive_fixture("realistic", "x_squared_sin", lambda i: "x**2*sin(x)", _manual_text, variants=1),
    _primitive_fixture("realistic", "x_log_x", lambda i: "x*log(x)", _integrate_text, variants=1),
    _primitive_fixture("realistic", "rational_atan", lambda i: "1/(1+x**2)", _manual_text, variants=1),
    _primitive_fixture("realistic", "scaled_degree8_times_exp", lambda i: "(x**8-4*x**5+2*x**3-x+7)*exp(2*x)",
                       _risch_text, variants=1),
    # function_integrate
    _definite_fixture_text("micro", "cos_kx_unit_interval", lambda i: f"cos({i + 1}*x)", 0, 1, _definite_text,
                           variants=1000),
    _definite_fixture_text("realistic", "rational_squared", lambda i: "1/(1+x*x)**2", 0, 1, _definite_text,
                           variants=1, v07="definite_integral"),
    _definite_fixture_text("realistic", "x_exp_unit_interval", lambda i: "x*exp(x)", 0, 1, _definite_text, variants=1),
    _definite_fixture_text("realistic", "x_squared_sin_unit_interval", lambda i: "x**2*sin(x)", 0, 1,
                           _definite_text, variants=1),
    _definite_fixture_text("realistic", "x_log_x_one_to_two", lambda i: "x*log(x)", 1, 2, _definite_text, variants=1),
    _definite_fixture_text("realistic", "scaled_degree8_times_exp", lambda i: "(x**8-4*x**5+2*x**3-x+7)*exp(2*x)",
                           1, 2, _definite_text, variants=1),
    # rational_antiderivative
    _rational_primitive_fixture("micro", "constant_over_linear",
                                lambda i: (lambda x, c=_line_case(i): (c[0], c[1] * x - c[2])), _integrate_call,
                                variants=1000),
    _rational_primitive_fixture("realistic", "atan_quadratic", lambda i: (lambda x: (1, x * x + 1)), _manual_call,
                                variants=1),
    _rational_primitive_fixture("realistic", "linear_over_irreducible_quadratic",
                                lambda i: (lambda x: (x + 3, x * x + 4 * x + 8)), _integrate_call, variants=1),
    _rational_primitive_fixture("realistic", "two_simple_poles", lambda i: (lambda x: (1, (x + 1) * (x + 2))),
                                _ratint_call, variants=1),
    _rational_primitive_fixture("realistic", "repeated_quadratic", lambda i: (lambda x: (x ** 3, (x * x + 1) ** 2)),
                                _ratint_call, variants=1),
    _rational_primitive_fixture(
        "realistic", "scaled_four_factors",
        lambda i: (lambda x: (3 * x ** 4 + 2 * x ** 3 + x + 5, (x * x + 1) * (x * x + x + 1) * (x - 2))),
        _integrate_call, variants=1),
    # rational_integrate
    _rational_definite_fixture("micro", "constant_over_linear", _positive_line_parts, _positive_line_bounds,
                               _definite_integrate_call, variants=1000),
    _rational_definite_fixture("realistic", "atan_unit_interval", lambda i: (lambda x: (1, 1 + x * x)),
                               lambda i: (0, 1), _definite_manual_call, variants=1),
    _rational_definite_fixture("realistic", "two_simple_poles_unit_interval",
                               lambda i: (lambda x: (1, (x + 1) * (x + 2))), lambda i: (0, 1),
                               _definite_integrate_call, variants=1),
    _rational_definite_fixture("realistic", "x_over_quadratic", lambda i: (lambda x: (x, x * x + 1)),
                               lambda i: (0, 2), _definite_integrate_call, variants=1),
    _rational_definite_fixture("realistic", "irreducible_quadratic", lambda i: (lambda x: (1, x * x + x + 1)),
                               lambda i: (0, 1), _definite_integrate_call, variants=1),
    _rational_definite_fixture("realistic", "scaled_double_pole",
                               lambda i: (lambda x: (2 * x + 3, (x + 1) ** 2 * (x + 2))), lambda i: (3, 5),
                               _definite_integrate_call, variants=1),
)

# --- edge cases ----------------------------------------------------------------------------------------------------


def _f(text: str = "sin(x)*exp(x)", names: str = "x"):
    return mf.parse_function(text, symbols={name: {"x": X, "y": Y}[name] for name in names})


def _rational(denominator=None):
    """1/(x^2 + 1) by default: a rational function without real poles."""
    return mf.rational_function(1, X * X + 1 if denominator is None else denominator, variable=X)


def _cubic_expression():
    return X ** 3 + 2 * X + 1


def _two_symbols():
    return mf.parse_function("x*y", symbols={"x": X, "y": Y})


def _polynomial_edges(operation, tail):
    """Four types, a list, and two over-limit inputs (a 140,001-bit coefficient and a degree of 2,200)."""
    return (*type_edges(operation, tail),
            edge(operation, "type_list", lambda: (([1, 2], *tail()), {})),
            edge(operation, "over_limit_integer", lambda: ((big_polynomial(), *tail()), {})),
            edge(operation, "over_limit_degree", lambda: ((high_degree(), *tail()), {})))


def _function_edges(operation, tail):
    """Four types, a list, and two over-limit inputs (a 140,001-bit constant and a depth of 400)."""
    return (*type_edges(operation, tail),
            edge(operation, "type_list", lambda: (([1, 2], *tail()), {})),
            edge(operation, "over_limit_integer", lambda: ((big_raw(), *tail()), {})),
            edge(operation, "over_limit_depth", lambda: ((deep_raw(), *tail()), {})))


EDGE_CASES: tuple[EdgeCase, ...] = (
    # antiderivative
    *_polynomial_edges("antiderivative", lambda: (X,)),
    edge("antiderivative", "variable_str", lambda: ((_cubic_expression(), "x"), {})),
    edge("antiderivative", "second_variable", lambda: ((X * Y, X), {})),
    edge("antiderivative", "irrational_coefficient", lambda: ((mf.sqrt(2) * X, X), {})),
    edge("antiderivative", "variable_with_assumption", lambda: ((_cubic_expression(), positive_symbol()), {})),
    edge("antiderivative", "constant", lambda: ((5, X), {})),
    edge("antiderivative", "zero", lambda: ((mf.Rational(0), X), {})),
    edge("antiderivative", "fraction_coefficients",
         lambda: ((mf.Rational(1, 2) * X ** 2 + mf.Rational(-3, 7) * X + 1, X), {})),
    edge("antiderivative", "polynomial_object", lambda: ((mf.Polynomial((1, 2, 3), X), X), {})),
    edge("antiderivative", "dispatch_real_function", lambda: ((_f("cos(2*x)"), X), {})),
    edge("antiderivative", "dispatch_unsupported_real_function", lambda: ((_f(), X), {})),
    edge("antiderivative", "dispatch_rational_function", lambda: ((_rational(), X), {})),
    # integrate
    *_polynomial_edges("integrate", lambda: (X, 0, 1)),
    edge("integrate", "over_limit_bound", lambda: ((_cubic_expression(), X, 0, 2 ** 140_000), {})),
    edge("integrate", "variable_str", lambda: ((_cubic_expression(), "x", 0, 1), {})),
    edge("integrate", "bound_float", lambda: ((_cubic_expression(), X, 0.5, 1), {})),
    edge("integrate", "bound_str", lambda: ((_cubic_expression(), X, "0", 1), {})),
    edge("integrate", "bound_none", lambda: ((_cubic_expression(), X, None, 1), {})),
    edge("integrate", "bound_infinite", lambda: ((_cubic_expression(), X, 0, "+inf"), {})),
    edge("integrate", "bound_symbolic", lambda: ((_cubic_expression(), X, 0, Y), {})),
    edge("integrate", "second_variable", lambda: ((X * Y, X, 0, 1), {})),
    edge("integrate", "lower_above_upper", lambda: ((_cubic_expression(), X, 1, 0), {})),
    edge("integrate", "equal_bounds", lambda: ((_cubic_expression(), X, 1, 1), {})),
    edge("integrate", "algebraic_upper_bound", lambda: ((_cubic_expression(), X, 0, mf.sqrt(2)), {})),
    edge("integrate", "algebraic_bounds", lambda: ((_cubic_expression(), X, mf.sqrt(2), mf.sqrt(3)), {})),
    edge("integrate", "fraction_bounds",
         lambda: ((_cubic_expression(), X, mf.Rational(-1, 2), mf.Rational(3, 2)), {})),
    edge("integrate", "polynomial_object", lambda: ((mf.Polynomial((1, -2, 0, 1), X), X, 0, 1), {})),
    edge("integrate", "dispatch_real_function", lambda: ((_f("cos(2*x)"), X, 0, 1), {})),
    edge("integrate", "dispatch_rational_function", lambda: ((_rational(), X, 0, 1), {})),
    # function_antiderivative
    *_function_edges("function_antiderivative", lambda: (X,)),
    edge("function_antiderivative", "variable_str", lambda: ((_f(), "x"), {})),
    edge("function_antiderivative", "variable_not_axis", lambda: ((_f(), Y), {})),
    edge("function_antiderivative", "unsupported_gaussian", lambda: ((_f("exp(x*x)"), X), {})),
    edge("function_antiderivative", "unsupported_sine_cosine_product", lambda: ((_f("sin(x)*cos(x)"), X), {})),
    edge("function_antiderivative", "unsupported_exponential_sine", lambda: ((_f("exp(x)*sin(x)"), X), {})),
    edge("function_antiderivative", "with_parameter", lambda: ((_f("x*y", "xy"), Y), {})),
    edge("function_antiderivative", "square_root", lambda: ((_f("sqrt(x)"), X), {})),
    edge("function_antiderivative", "constant", lambda: ((_f("5"), X), {})),
    edge("function_antiderivative", "rational_catalogue_entry", lambda: ((_f("1/(1+x*x)"), X), {})),
    edge("function_antiderivative", "rational_fallback_two_poles", lambda: ((_f("1/((x+1)*(x+2))"), X), {})),
    edge("function_antiderivative", "rational_fallback_irreducible_cubic", lambda: ((_f("1/(x*x*x+x+1)"), X), {})),
    edge("function_antiderivative", "rational_function_object", lambda: ((_rational(), X), {})),
    # function_integrate
    *_function_edges("function_integrate", lambda: (X, 0, 1)),
    edge("function_integrate", "over_limit_bound", lambda: ((_f("x*x"), X, 0, 2 ** 140_000), {})),
    edge("function_integrate", "variable_str", lambda: ((_f(), "x", 0, 1), {})),
    edge("function_integrate", "variable_not_axis", lambda: ((_f(), Y, 0, 1), {})),
    edge("function_integrate", "bound_infinite", lambda: ((_f("exp(-x)"), X, 0, "+inf"), {})),
    edge("function_integrate", "bound_float", lambda: ((_f(), X, 0.5, 1), {})),
    edge("function_integrate", "bound_none", lambda: ((_f(), X, None, 1), {})),
    edge("function_integrate", "bound_symbolic", lambda: ((_f(), X, 0, Y), {})),
    edge("function_integrate", "improper_pole_at_bound", lambda: ((_f("1/x"), X, 0, 1), {})),
    edge("function_integrate", "improper_pole_inside", lambda: ((_f("1/x"), X, -1, 1), {})),
    edge("function_integrate", "unsupported_gaussian", lambda: ((_f("exp(x*x)"), X, 0, 1), {})),
    edge("function_integrate", "outside_domain_logarithm", lambda: ((_f("log(x)"), X, -1, 1), {})),
    edge("function_integrate", "nonsmooth_square_root", lambda: ((_f("sqrt(x)"), X, 0, 1), {})),
    edge("function_integrate", "second_variable", lambda: ((_f("x*y", "xy"), Y, 0, 1), {})),
    edge("function_integrate", "lower_above_upper", lambda: ((_f("cos(3*x)"), X, 1, 0), {})),
    edge("function_integrate", "equal_bounds", lambda: ((_f("x*x"), X, 2, 2), {})),
    edge("function_integrate", "algebraic_bound", lambda: ((_f("x*x"), X, 0, mf.sqrt(2)), {})),
    edge("function_integrate", "logarithm_one_to_two", lambda: ((_f("log(x)"), X, 1, 2), {})),
    edge("function_integrate", "rational_integrand", lambda: ((_f("1/(x*x+1)"), X, 0, 1), {})),
    # rational_antiderivative
    *_polynomial_edges("rational_antiderivative", lambda: (X,)),
    edge("rational_antiderivative", "over_limit_depth", lambda: ((deep_raw(), X), {})),
    edge("rational_antiderivative", "variable_str", lambda: ((_rational(), "x"), {})),
    edge("rational_antiderivative", "variable_mismatch", lambda: ((_rational(), Y), {})),
    edge("rational_antiderivative", "variable_with_assumption", lambda: ((_rational(), positive_symbol()), {})),
    edge("rational_antiderivative", "polynomial_expression", lambda: ((_cubic_expression(), X), {})),
    edge("rational_antiderivative", "real_function", lambda: ((_f("1/(1+x*x)"), X), {})),
    edge("rational_antiderivative", "simple_pole", lambda: ((mf.rational_function(1, X - 3, variable=X), X), {})),
    edge("rational_antiderivative", "irreducible_quadratic", lambda: ((_rational(), X), {})),
    edge("rational_antiderivative", "repeated_irreducible_quadratic",
         lambda: ((mf.rational_function(1, (X * X + 1) ** 2, variable=X), X), {})),
    edge("rational_antiderivative", "irreducible_cubic",
         lambda: ((mf.rational_function(1, X ** 3 + X + 1, variable=X), X), {})),
    edge("rational_antiderivative", "polynomial_part",
         lambda: ((mf.rational_function(X ** 3 + 1, X * X - 1, variable=X), X), {})),
    # rational_integrate
    *_polynomial_edges("rational_integrate", lambda: (X, 0, 1)),
    edge("rational_integrate", "over_limit_depth", lambda: ((deep_raw(), X, 0, 1), {})),
    edge("rational_integrate", "over_limit_bound", lambda: ((_rational(), X, 0, 2 ** 140_000), {})),
    edge("rational_integrate", "variable_str", lambda: ((_rational(), "x", 0, 1), {})),
    edge("rational_integrate", "variable_mismatch", lambda: ((_rational(), Y, 0, 1), {})),
    edge("rational_integrate", "bound_infinite", lambda: ((_rational(), X, 0, "+inf"), {})),
    edge("rational_integrate", "bound_float", lambda: ((_rational(), X, 0.5, 1), {})),
    edge("rational_integrate", "bound_none", lambda: ((_rational(), X, None, 1), {})),
    edge("rational_integrate", "bound_symbolic", lambda: ((_rational(), X, 0, Y), {})),
    edge("rational_integrate", "pole_inside",
         lambda: ((mf.rational_function(1, X - 3, variable=X), X, 2, 4), {})),
    edge("rational_integrate", "pole_at_bound",
         lambda: ((mf.rational_function(1, X - 3, variable=X), X, 3, 4), {})),
    edge("rational_integrate", "lower_above_upper", lambda: ((_rational(), X, 1, 0), {})),
    edge("rational_integrate", "equal_bounds", lambda: ((_rational(), X, 1, 1), {})),
    edge("rational_integrate", "algebraic_bound", lambda: ((_rational(), X, 0, mf.sqrt(2)), {})),
    edge("rational_integrate", "polynomial_expression", lambda: ((X * X, X, 0, 1), {})),
    edge("rational_integrate", "second_variable", lambda: ((X * Y, X, 0, 1), {})),
    edge("rational_integrate", "logarithm_and_arctangent",
         lambda: ((mf.rational_function(X + 3, (X + 1) * (X * X + 1), variable=X), X, 0, 2), {})),
)
