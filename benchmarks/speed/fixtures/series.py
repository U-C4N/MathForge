"""Speed fixtures of the series area: function_taylor (public name ``taylor``), function_limit and limit.

Taylor and elementary-limit inputs are real elementary functions parsed from text (``mathforge.parse_function`` on
the MathForge side, ``sympify`` of the same text on the SymPy side); the rational ``limit`` takes a RationalFunction
built from the same Python arithmetic on a MathForge or a SymPy symbol. A micro fixture has 1,000 variants of one
shape; the realistic fixtures are the usual textbook expansions and limits (the v0.7 cases taylor6, sinc_limit and
double_pole carry over unchanged), and the scaled ones are a high order or a high degree.

The SymPy reference of a Taylor polynomial is ``series(...).removeO()``, the call a SymPy user makes.
``sympy.polys.ring_series.rs_series`` is several times faster, but it works on elements of a polynomial ring and
returns one, not an expression, so it is measured and excluded. The reference of a limit is ``limit`` (``gruntz`` at
infinity, where ``limit`` itself calls it and it is more than 15% faster). A candidate that is within 15% of the
conventional call is not preferred to it. Every ``reference`` lists the candidates, the excluded one among them, with
their median time in microseconds on that fixture's inputs, with SymPy's caches cleared before each call (Python
3.14.6, SymPy 1.14.0, python ground types); there g is the function, x its symbol, a the point or center, n the number
of terms and dir the direction.
"""

from __future__ import annotations

import itertools
import math

import mathforge as mf
from benchmarks.speed.fixture import EdgeCase, Fixture
from benchmarks.speed.fixtures.analysis import (
    X, Y, big_polynomial, big_raw, decode, deep_raw, edge, equal, high_degree, sympy_symbols, to_sympy, type_edges)


_REFERENCES: dict[str, str] = {
    "function_taylor/micro/exp_ax_order3": (
        "series(...).removeO() 18689 us (chosen); g.series(x, a, n).removeO() 29530 us; rs_series(g, x, n) "
        "2768 us works on polynomial-ring elements, not expressions, and is excluded"),
    "function_taylor/realistic/exp_sin_log_order6": (
        "series(...).removeO() 86666 us (chosen; g.series(x, a, n).removeO() 77134 us is within 15%); "
        "series(...).removeO() was the v0.7 reference; rs_series(g, x, n) 6234 us works on polynomial-ring "
        "elements, not expressions, and is excluded"),
    "function_taylor/realistic/sinh_exp_order6": (
        "series(...).removeO() 65122 us (chosen); g.series(x, a, n).removeO() 65986 us; rs_series(g, x, n) "
        "4769 us works on polynomial-ring elements, not expressions, and is excluded"),
    "function_taylor/realistic/x_exp_order8": (
        "series(...).removeO() 55950 us (chosen; g.series(x, a, n).removeO() 51752 us is within 15%); "
        "rs_series(g, x, n) 2868 us works on polynomial-ring elements, not expressions, and is excluded"),
    "function_taylor/realistic/exp_about_1_order6": (
        "series(...).removeO() 74087 us (chosen); g.series(x, a, n).removeO() 71851 us"),
    "function_taylor/realistic/scaled_exp_order20": (
        "series(...).removeO() 60966 us (chosen); g.series(x, a, n).removeO() 59874 us; rs_series(g, x, n) "
        "1514 us works on polynomial-ring elements, not expressions, and is excluded"),
    "function_taylor/realistic/scaled_sin_order15": (
        "series(...).removeO() 17976 us (chosen); g.series(x, a, n).removeO() 17305 us; rs_series(g, x, n) "
        "1192 us works on polynomial-ring elements, not expressions, and is excluded"),
    "function_limit/micro/sinc_scaled": (
        "limit(g, x, a, dir) 1330 us (chosen); Limit(g, x, a, dir).doit() 1351 us"),
    "function_limit/micro/decay_at_infinity": (
        "gruntz(g, x, a, dir) 12803 us (chosen); Limit(g, x, a, dir).doit() 18719 us; limit(g, x, a, "
        "dir) 18981 us"),
    "function_limit/realistic/sinc": (
        "limit(g, x, a, dir) 1064 us (chosen); Limit(g, x, a, dir).doit() 1051 us; limit(...) was the "
        "v0.7 reference"),
    "function_limit/realistic/cosine_defect": (
        "limit(g, x, a, dir) 18430 us (chosen); Limit(g, x, a, dir).doit() 18340 us"),
    "function_limit/realistic/x_log_x_from_right": (
        "limit(g, x, a, dir) 11192 us (chosen); gruntz(g, x, a, dir) 10087 us; Limit(g, x, a, "
        "dir).doit() 11385 us"),
    "function_limit/realistic/exp_defect": (
        "limit(g, x, a, dir) 61531 us (chosen); Limit(g, x, a, dir).doit() 61278 us"),
    "function_limit/realistic/log_over_x_at_infinity": (
        "gruntz(g, x, a, dir) 4855 us (chosen); Limit(g, x, a, dir).doit() 6768 us; limit(g, x, a, "
        "dir) 6778 us"),
    "function_limit/realistic/scaled_sine_defect": (
        "limit(g, x, a, dir) 38561 us (chosen); Limit(g, x, a, dir).doit() 38592 us"),
    "limit/micro/linear_pole_from_right": (
        "limit(g, x, a, dir) 2202 us (chosen); Limit(g, x, a, dir).doit() 2382 us"),
    "limit/micro/quadratics_at_infinity": (
        "limit(g, x, a, dir) 4426 us (chosen); Limit(g, x, a, dir).doit() 4641 us"),
    "limit/realistic/double_pole_algebraic_point": (
        "limit(g, x, a, dir) 18122 us (chosen); Limit(g, x, a, dir).doit() 17750 us; limit(...) was "
        "the v0.7 reference"),
    "limit/realistic/double_pole_rational_point": (
        "limit(g, x, a, dir) 4035 us (chosen); Limit(g, x, a, dir).doit() 4084 us"),
    "limit/realistic/simple_pole_from_right": (
        "limit(g, x, a, dir) 2218 us (chosen); Limit(g, x, a, dir).doit() 2296 us"),
    "limit/realistic/removable_singularity": (
        "limit(g, x, a, dir) 13159 us (chosen); Limit(g, x, a, dir).doit() 12855 us"),
    "limit/realistic/scaled_degree_8_at_infinity": (
        "limit(g, x, a, dir) 9584 us (chosen); Limit(g, x, a, dir).doit() 10019 us"),
}


def _reference(operation: str, size: str, label: str) -> str:
    return _REFERENCES[f"{operation}/{size}/{label}"]


def _function(text: str):
    """The MathForge real function of x written as text."""
    return mf.parse_function(text, symbols={"x": X})


def _expression(text: str):
    """The SymPy expression of the real symbol x written as text."""
    import sympy as sp
    return sp.sympify(text, locals={"x": sympy_symbols()[X]})


def _sqrt3_mathforge():
    return mf.sqrt(3)


def _sqrt3_sympy():
    import sympy as sp
    return sp.sqrt(3)


# --- function_taylor -----------------------------------------------------------------------------------------------


def _same_taylor(order: int):
    def agree(mf_value, sympy_value) -> bool:
        import sympy as sp
        symbols = sympy_symbols()
        x = symbols[X]
        if type(mf_value) is not mf.TaylorPolynomial or mf_value.order != order:
            return False
        point = to_sympy(mf_value.point, symbols)
        ours = sum(to_sympy(c, symbols) * (x - point) ** k for k, c in enumerate(mf_value.coefficients))
        return sp.expand(ours - sympy_value.as_expr()) == 0
    return agree


def _taylor_fixture(size, label, text_of, center, order, *, variants, v07=""):
    """text_of(i) is the function; the SymPy call is series(...).removeO()."""
    def mathforge(i):
        return mf.taylor, (_function(text_of(i)), X, center, order), {}

    def sympy(i):
        import sympy as sp
        x = sympy_symbols()[X]
        function = _expression(text_of(i))
        return lambda: sp.series(function, x, center, order + 1).removeO()

    return Fixture("function_taylor", size, label, mathforge, sympy, _same_taylor(order),
                   _reference("function_taylor", size, label), variants, v07)


# --- function_limit ------------------------------------------------------------------------------------------------


def _same_function_limit(mf_value, sympy_value) -> bool:
    import sympy as sp
    if type(mf_value) is not mf.FunctionLimitValue:
        return False
    if sympy_value == sp.oo:
        return mf_value.kind == "positive_infinity" and mf_value.value is None
    if sympy_value == -sp.oo:
        return mf_value.kind == "negative_infinity" and mf_value.value is None
    return mf_value.kind == "finite" and equal(to_sympy(mf_value.value, {}), sympy_value)


_DIRECTIONS = {"both": "+-", "right": "+", "left": "-"}


def _limit_fixture(size, label, text_of, point, side, *, variants, v07="", gruntz=False):
    """The limit of the function text_of(i) at ``point`` ("+inf", "-inf" or an int) from ``side``. ``gruntz`` selects
    sympy.series.gruntz.gruntz, which ``limit`` itself calls at infinity, where it is faster than ``limit``."""
    def mathforge(i):
        return mf.function_limit, (_function(text_of(i)), X, point, side), {}

    def sympy(i):
        import sympy as sp
        x = sympy_symbols()[X]
        function = _expression(text_of(i))
        if gruntz:
            from sympy.series.gruntz import gruntz as algorithm
            return lambda: algorithm(function, x, sp.oo)
        if point == "+inf":
            return lambda: sp.limit(function, x, sp.oo)
        if point == "-inf":
            return lambda: sp.limit(function, x, -sp.oo)
        return lambda: sp.limit(function, x, point, dir=_DIRECTIONS[side])

    return Fixture("function_limit", size, label, mathforge, sympy, _same_function_limit,
                   _reference("function_limit", size, label), variants, v07)


# --- limit of a rational function ----------------------------------------------------------------------------------


def _same_limit(mf_value, sympy_value) -> bool:
    import sympy as sp
    if type(mf_value) is not mf.LimitValue:
        return False
    if sympy_value == sp.oo:
        return mf_value.kind == "positive_infinity" and mf_value.value is None
    if sympy_value == -sp.oo:
        return mf_value.kind == "negative_infinity" and mf_value.value is None
    return mf_value.kind == "finite" and equal(to_sympy(mf_value.value, {}), sympy_value)


def _rational_fixture(size, label, parts_of, point, side, *, variants, v07="", point_of=None, sympy_point_of=None):
    """parts_of(i) -> a function x -> (numerator, denominator), evaluated on the MathForge symbol and on the SymPy
    symbol alike. The point is an int, "+inf" or "-inf"; point_of(i) and sympy_point_of(i) give it as a MathForge
    and as a SymPy value instead when it varies or is not an integer."""
    def mathforge(i):
        numerator, denominator = parts_of(i)(X)
        target = point if point_of is None else point_of(i)
        return mf.limit, (mf.rational_function(numerator, denominator, variable=X), X, target), {"side": side}

    def sympy(i):
        import sympy as sp
        x = sympy_symbols()[X]
        numerator, denominator = parts_of(i)(x)
        function = numerator / denominator
        if point == "+inf":
            return lambda: sp.limit(function, x, sp.oo)
        if point == "-inf":
            return lambda: sp.limit(function, x, -sp.oo)
        target = point if sympy_point_of is None else sympy_point_of(i)
        return lambda: sp.limit(function, x, target, dir=_DIRECTIONS[side])

    return Fixture("limit", size, label, mathforge, sympy, _same_limit, _reference("limit", size, label), variants,
                   v07)


def _pole_case(i: int) -> tuple[int, int, int]:
    """(a, b, k) of (a x + b)/(x - k): 6,156 distinct triples (2711 is coprime to that); a k + b = 0 is a hole."""
    a, b, k = decode((i * 2711 + 17) % 6156, 18, 18, 19)
    return (a - 9 if a < 9 else a - 8), (b - 9 if b < 9 else b - 8), k - 9


def _pole_parts(i):
    a, b, k = _pole_case(i)
    return lambda x: (a * x + b, x - k)


_QUADRATIC_QUADRUPLES = tuple(q for q in itertools.product(range(1, 10), repeat=4)
                              if math.gcd(math.gcd(q[0], q[1]), math.gcd(q[2], q[3])) == 1)


def _infinity_parts(i):
    """(a x^2 + b)/(c x^2 + d) with a, b, c, d in [1, 9] and no common factor, so that no two are one function: the
    quadruples in turn with a stride of 7,919, which is prime and larger than their number."""
    a, b, c, d = _QUADRATIC_QUADRUPLES[i * 7919 % len(_QUADRATIC_QUADRUPLES)]
    return lambda x: (a * x ** 2 + b, c * x ** 2 + d)


FIXTURES = (
    # function_taylor
    _taylor_fixture("micro", "exp_ax_order3", lambda i: f"exp({i + 1}*x)", 0, 3, variants=1000),
    _taylor_fixture("realistic", "exp_sin_log_order6", lambda i: "exp(x)*sin(x)+log(1+x)", 0, 6, variants=1,
                    v07="taylor6"),
    _taylor_fixture("realistic", "sinh_exp_order6", lambda i: "sinh(x)*exp(x)", 0, 6, variants=1),
    _taylor_fixture("realistic", "x_exp_order8", lambda i: "x*exp(x)", 0, 8, variants=1),
    _taylor_fixture("realistic", "exp_about_1_order6", lambda i: "exp(x)", 1, 6, variants=1),
    _taylor_fixture("realistic", "scaled_exp_order20", lambda i: "exp(x)", 0, 20, variants=1),
    _taylor_fixture("realistic", "scaled_sin_order15", lambda i: "sin(x)", 0, 15, variants=1),
    # function_limit
    _limit_fixture("micro", "sinc_scaled", lambda i: f"sin({i + 1}*x)/x", 0, "both", variants=1000),
    _limit_fixture("micro", "decay_at_infinity", lambda i: f"({i + 1}*x+1)*exp(-x)", "+inf", "both", variants=1000,
                   gruntz=True),
    _limit_fixture("realistic", "sinc", lambda i: "sin(x)/x", 0, "both", variants=1,
                   v07="sinc_limit"),
    _limit_fixture("realistic", "cosine_defect", lambda i: "(1-cos(x))/(x*x)", 0, "both", variants=1),
    _limit_fixture("realistic", "x_log_x_from_right", lambda i: "x*log(x)", 0, "right", variants=1),
    _limit_fixture("realistic", "exp_defect", lambda i: "(exp(x)-1-x)/(x*x)", 0, "both", variants=1),
    _limit_fixture("realistic", "log_over_x_at_infinity", lambda i: "log(x)/x", "+inf", "both", variants=1,
                   gruntz=True),
    _limit_fixture("realistic", "scaled_sine_defect", lambda i: "(sin(x)-x+x**3/6)/x**5", 0, "both", variants=1),
    # limit (rational function)
    _rational_fixture("micro", "linear_pole_from_right", _pole_parts, 0, "right", variants=1000,
                      point_of=lambda i: _pole_case(i)[2], sympy_point_of=lambda i: _pole_case(i)[2]),
    _rational_fixture("micro", "quadratics_at_infinity", _infinity_parts, "+inf", "both", variants=1000),
    _rational_fixture("realistic", "double_pole_algebraic_point", lambda i: (lambda x: (1, (x * x - 3) ** 2)), 0,
                      "both", variants=1, v07="double_pole",
                      point_of=lambda i: _sqrt3_mathforge(), sympy_point_of=lambda i: _sqrt3_sympy()),
    _rational_fixture("realistic", "double_pole_rational_point",
                      lambda i: (lambda x: (x ** 3 + 2 * x + 1, (x - 1) ** 2 * (x + 2))), 1, "both", variants=1),
    _rational_fixture("realistic", "simple_pole_from_right", lambda i: (lambda x: (x, (x - 2) * (x + 3))), 2,
                      "right", variants=1),
    _rational_fixture("realistic", "removable_singularity", lambda i: (lambda x: (x * x - 9, x - 3)), 3, "both",
                      variants=1),
    _rational_fixture("realistic", "scaled_degree_8_at_infinity",
                      lambda i: (lambda x: (3 * x ** 8 + x ** 5 - 2 * x + 7, 2 * x ** 8 - x ** 6 + 11)), "+inf",
                      "both", variants=1),
)

# --- edge cases ----------------------------------------------------------------------------------------------------


def _f(text: str = "sin(x)/x"):
    return _function(text)


def _pole():
    """1/(x - 2), a rational function with a pole at 2."""
    return mf.rational_function(1, X - 2, variable=X)


def _two_symbols():
    return mf.parse_function("x*y", symbols={"x": X, "y": Y})


EDGE_CASES: tuple[EdgeCase, ...] = (
    # function_taylor (public name taylor)
    *type_edges("function_taylor", lambda: (X, 0, 3), "taylor"),
    edge("function_taylor", "over_limit_integer", lambda: ((big_raw(), X, 0, 3), {}), "taylor"),
    edge("function_taylor", "over_limit_depth", lambda: ((deep_raw(), X, 0, 3), {}), "taylor"),
    edge("function_taylor", "over_limit_order", lambda: ((_f("exp(x)"), X, 0, 10 ** 7), {}), "taylor"),
    edge("function_taylor", "variable_str", lambda: ((_f("exp(x)"), "x", 0, 3), {}), "taylor"),
    edge("function_taylor", "variable_not_axis", lambda: ((_f("exp(x)"), Y, 0, 3), {}), "taylor"),
    edge("function_taylor", "order_negative", lambda: ((_f("exp(x)"), X, 0, -1), {}), "taylor"),
    edge("function_taylor", "order_float", lambda: ((_f("exp(x)"), X, 0, 1.5), {}), "taylor"),
    edge("function_taylor", "order_bool", lambda: ((_f("exp(x)"), X, 0, True), {}), "taylor"),
    edge("function_taylor", "order_zero", lambda: ((_f("exp(x)"), X, 0, 0), {}), "taylor"),
    edge("function_taylor", "center_symbolic", lambda: ((_f("exp(x)"), X, Y, 3), {}), "taylor"),
    edge("function_taylor", "center_float", lambda: ((_f("exp(x)"), X, 0.5, 3), {}), "taylor"),
    edge("function_taylor", "center_str", lambda: ((_f("exp(x)"), X, "0", 3), {}), "taylor"),
    edge("function_taylor", "center_none", lambda: ((_f("exp(x)"), X, None, 3), {}), "taylor"),
    edge("function_taylor", "center_algebraic", lambda: ((_f("exp(x)"), X, mf.sqrt(2), 2), {}), "taylor"),
    edge("function_taylor", "undefined_at_center", lambda: ((_f("log(x)"), X, 0, 2), {}), "taylor"),
    edge("function_taylor", "nonsmooth_at_center", lambda: ((_f("sqrt(x)"), X, 0, 2), {}), "taylor"),
    edge("function_taylor", "absolute_value_at_zero", lambda: ((_f("abs(x)"), X, 0, 2), {}), "taylor"),
    edge("function_taylor", "pole_at_center", lambda: ((_f("1/x"), X, 0, 2), {}), "taylor"),
    edge("function_taylor", "second_variable", lambda: ((_two_symbols(), X, 0, 2), {}), "taylor"),
    edge("function_taylor", "logarithm_about_one", lambda: ((_f("log(x)"), X, 1, 3), {}), "taylor"),
    edge("function_taylor", "default_center_and_order", lambda: ((_f("cos(x)"), X), {}), "taylor"),
    # function_limit
    *type_edges("function_limit", lambda: (X, 0, "both")),
    edge("function_limit", "over_limit_integer", lambda: ((big_raw(), X, 0, "both"), {})),
    edge("function_limit", "over_limit_depth", lambda: ((deep_raw(), X, 0, "both"), {})),
    edge("function_limit", "over_limit_point", lambda: ((_f("x*x"), X, 2 ** 140_000, "both"), {})),
    edge("function_limit", "variable_str", lambda: ((_f(), "x", 0, "both"), {})),
    edge("function_limit", "variable_not_axis", lambda: ((_f(), Y, 0, "both"), {})),
    edge("function_limit", "side_invalid", lambda: ((_f(), X, 0, "up"), {})),
    edge("function_limit", "side_none", lambda: ((_f(), X, 0, None), {})),
    edge("function_limit", "side_at_infinity", lambda: ((_f("exp(-x)"), X, "+inf", "left"), {})),
    edge("function_limit", "point_text_invalid", lambda: ((_f(), X, "nan", "both"), {})),
    edge("function_limit", "point_float", lambda: ((_f(), X, 0.5, "both"), {})),
    edge("function_limit", "point_none", lambda: ((_f(), X, None, "both"), {})),
    edge("function_limit", "point_symbolic", lambda: ((_f(), X, Y, "both"), {})),
    edge("function_limit", "point_algebraic", lambda: ((_f("x*x"), X, mf.sqrt(2), "both"), {})),
    edge("function_limit", "second_variable", lambda: ((_two_symbols(), X, 0, "both"), {})),
    edge("function_limit", "does_not_exist", lambda: ((_f("1/x"), X, 0, "both"), {})),
    edge("function_limit", "positive_infinity_from_right", lambda: ((_f("1/x"), X, 0, "right"), {})),
    edge("function_limit", "negative_infinity_from_left", lambda: ((_f("1/x"), X, 0, "left"), {})),
    edge("function_limit", "logarithm_at_zero_from_right", lambda: ((_f("log(x)"), X, 0, "right"), {})),
    edge("function_limit", "logarithm_at_zero_both_sides", lambda: ((_f("log(x)"), X, 0, "both"), {})),
    edge("function_limit", "point_outside_domain", lambda: ((_f("log(x)"), X, -1, "both"), {})),
    edge("function_limit", "unsupported_oscillation", lambda: ((_f("sin(1/x)"), X, 0, "both"), {})),
    edge("function_limit", "minus_infinity", lambda: ((_f("exp(x)"), X, "-inf", "both"), {})),
    edge("function_limit", "default_side", lambda: ((_f("sin(x)/x"), X, 0), {})),
    # limit of a rational function
    *type_edges("limit", lambda: (X, 0)),
    edge("limit", "type_list", lambda: (([1, 2], X, 0), {})),
    edge("limit", "over_limit_integer", lambda: ((big_polynomial(), X, 0), {})),
    edge("limit", "over_limit_point", lambda: ((_pole(), X, 2 ** 140_000), {})),
    edge("limit", "high_degree", lambda: ((high_degree(), X, 0), {})),
    edge("limit", "variable_str", lambda: ((_pole(), "x", 0), {})),
    edge("limit", "variable_mismatch", lambda: ((_pole(), Y, 0), {})),
    edge("limit", "side_invalid", lambda: ((_pole(), X, 2), {"side": "up"})),
    edge("limit", "side_none", lambda: ((_pole(), X, 2), {"side": None})),
    edge("limit", "side_at_infinity", lambda: ((_pole(), X, "+inf"), {"side": "left"})),
    edge("limit", "point_text_invalid", lambda: ((_pole(), X, "nan"), {})),
    edge("limit", "point_float", lambda: ((_pole(), X, 0.5), {})),
    edge("limit", "point_none", lambda: ((_pole(), X, None), {})),
    edge("limit", "point_symbolic", lambda: ((_pole(), X, Y), {})),
    edge("limit", "point_algebraic", lambda: ((_pole(), X, mf.sqrt(2)), {})),
    edge("limit", "second_variable", lambda: ((X * Y, X, 0), {})),
    edge("limit", "does_not_exist", lambda: ((_pole(), X, 2), {"side": "both"})),
    edge("limit", "positive_infinity_from_right", lambda: ((_pole(), X, 2), {"side": "right"})),
    edge("limit", "negative_infinity_from_left", lambda: ((_pole(), X, 2), {"side": "left"})),
    edge("limit", "removable_singularity", lambda: ((mf.rational_function(X * X - 4, X - 2, variable=X), X, 2), {})),
    edge("limit", "polynomial_expression", lambda: ((X ** 3 + 1, X, 2), {})),
    edge("limit", "polynomial_object", lambda: ((mf.Polynomial((1, 0, 0, 1), X), X, 2), {})),
    edge("limit", "minus_infinity", lambda: ((mf.rational_function(X ** 3 + 1, X * X - 3, variable=X), X, "-inf"), {})),
    edge("limit", "dispatch_real_function", lambda: ((_f(), X, 0), {})),
)
