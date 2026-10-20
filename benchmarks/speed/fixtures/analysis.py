"""Speed fixtures of the analysis area: function_differentiate, function_partial_diff, function_gradient,
function_jacobian, function_hessian, function_evaluate (real elementary functions), partial_diff, gradient, jacobian,
hessian (multivariate polynomial expressions), differentiate (polynomial objects and expressions, rational functions),
analyze_function and extrema_values (rational functions).

Real elementary functions are parsed from text (``mathforge.parse_function``, and ``sympify`` of the same text on the
SymPy side). Polynomials are built from the same Python arithmetic on MathForge and on SymPy symbols. A micro fixture
has 1,000 variants of one shape; the realistic fixtures are textbook sizes (the v0.7 case derivative3 carries over
unchanged), and each family that scales has one larger fixture.

``analyze_function`` and ``extrema_values`` have no single SymPy call. Their SymPy side is the chain of public calls
that SymPy offers for the same questions: ``continuous_domain``, ``solveset`` for the zeros, the signs of the function
and the signs of its derivative, ``diff``, and the second-derivative test for the strict extrema. The rational
functions of these fixtures are in lowest terms with nonzero second derivative at every stationary point, where that
chain answers exactly the question that MathForge answers.

A polynomial given as a ``Polynomial`` object is compared with a prebuilt SymPy ``Poly`` (differentiate only: the
multivariate operations take expressions only); a polynomial given as an expression is compared with the SymPy
``Expr``, where building the ``Poly`` is part of the timed call whenever that is faster than ``diff``. A rational
function given as a ``RationalFunction`` is compared with the prebuilt SymPy expression of the same fraction, and the
timed call is ``cancel(diff(...))``: ``diff`` alone leaves a fraction that need not be in lowest terms. Elements of a
fraction field (``FracElement.diff``) are ground-domain elements and never a candidate.

Every ``reference`` lists the candidates with their median time in microseconds on that fixture's inputs, with SymPy's
caches cleared before each call (Python 3.14.6, SymPy 1.14.0, python ground types); there g is the function, f the
polynomial or rational function, v its variable, n the order and p the point. The chosen call is the fastest
same-object call, whatever it is named: ``diff``, ``Matrix.jacobian`` and ``hessian`` win no tie. A reference that was
re-pointed was measured again with the suite's own sampling, together with the call it replaced: 31 to 127
interleaved samples of k calls, caches reset before each.
"""

from __future__ import annotations

from fractions import Fraction
import itertools
import uuid

import mathforge as mf
from benchmarks.speed.fixture import EdgeCase, Fixture, symbol

X = symbol("x", 1)
Y = symbol("y", 2)
Z = symbol("z", 3)
W = symbol("w", 4)
V = symbol("v", 5)

_NAMES = {"x": X, "y": Y, "z": Z, "w": W, "v": V}
_UNARY = ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log")
_MASK = (1 << 64) - 1
_MULTIPLIER = 6364136223846793005
_INCREMENT = 1442695040888963407


_REFERENCES: dict[str, str] = {
    "function_differentiate/micro/sin_kx": (
        "g.diff(v, n) 679 us (chosen); diff(g, v, n) 682 us; first pass, an earlier machine state: Derivative(g, v, "
        "n).doit() 1314 us"),
    "function_differentiate/realistic/exp_sin_cos_third_derivative": (
        "diff(g, v, n) 13143 us (chosen); g.diff(v, n) 13162 us; first pass, an earlier machine state: Derivative(g, "
        "v, n).doit() 14045 us; diff(g, v, n) was the v0.7 reference"),
    "function_differentiate/realistic/tan_exp_second_derivative": (
        "diff(g, v, n) 7908 us (chosen); Derivative(g, v, n).doit() 8207 us; g.diff(v, n) 8378 us"),
    "function_differentiate/realistic/log_sin_second_derivative": (
        "diff(g, v, n) 10008 us (chosen); g.diff(v, n) 10078 us; first pass, an earlier machine state: Derivative(g, "
        "v, n).doit() 11035 us"),
    "function_differentiate/realistic/scaled_six_terms_first_derivative": (
        "g.diff(v, n) 9988 us (chosen); diff(g, v, n) 10070 us; first pass, an earlier machine state: Derivative(g, "
        "v, n).doit() 12174 us"),
    "function_partial_diff/micro/polynomial_and_sine": (
        "g.diff(v, n) 767 us (chosen); diff(g, v, n) 768 us; first pass, an earlier machine state: Derivative(g, v, "
        "n).doit() 1914 us"),
    "function_partial_diff/realistic/two_variables_second_derivative": (
        "diff(g, v, n) 14974 us (chosen); g.diff(v, n) 15266 us; Derivative(g, v, n).doit() 15906 us"),
    "function_partial_diff/realistic/scaled_three_variables_third_derivative": (
        "diff(g, v, n) 6847 us (chosen); g.diff(v, n) 7037 us; Derivative(g, v, n).doit() 8129 us"),
    "function_gradient/micro/two_variables": (
        "[diff(g, v) for v in vars] 1751 us (chosen); Matrix([g]).jacobian(vars) (prebuilt Matrix) "
        "1777 us; Matrix([g]).jacobian(vars) (built in the call) 1779 us"),
    "function_gradient/realistic/three_variables": (
        "[diff(g, v) for v in vars] 7515 us (chosen); Matrix([g]).jacobian(vars) (built in the call) "
        "7578 us; Matrix([g]).jacobian(vars) (prebuilt Matrix) 7628 us"),
    "function_gradient/realistic/scaled_five_variables": (
        "[diff(g, v) for v in vars] 10823 us (chosen); Matrix([g]).jacobian(vars) (prebuilt Matrix) "
        "11068 us; Matrix([g]).jacobian(vars) (built in the call) 11290 us"),
    "function_jacobian/micro/two_by_two": (
        "Matrix(fs).jacobian(vars) (prebuilt Matrix) 1501 us (chosen); Matrix(fs).jacobian(vars) "
        "(built in the call) 1574 us; [[diff(g, v) ...]] 2072 us"),
    "function_jacobian/realistic/spherical_to_cartesian": (
        "Matrix(fs).jacobian(vars) (prebuilt Matrix) 6346 us (chosen); [[diff(g, v) ...]] 6445 us; "
        "Matrix(fs).jacobian(vars) (built in the call) 6662 us"),
    "function_jacobian/realistic/scaled_four_by_four": (
        "[[diff(g, v) ...]] 6729 us (chosen); Matrix(fs).jacobian(vars) (prebuilt Matrix) 6928 us; first pass, an "
        "earlier machine state: Matrix(fs).jacobian(vars) (built in the call) 11642 us"),
    "function_hessian/micro/two_variables": (
        "[[diff(g, a, b) ...]] 3222 us (chosen); hessian(g, vars) 3440 us"),
    "function_hessian/realistic/three_variables": (
        "hessian(g, vars) 18288 us (chosen); [[diff(g, a, b) ...]] 25564 us"),
    "function_hessian/realistic/scaled_four_variables": (
        "hessian(g, vars) 29262 us (chosen); [[diff(g, a, b) ...]] 40048 us"),
    "function_evaluate/micro/quadratic_at_integer": (
        "g.xreplace({x: p}) 422 us (chosen); g.subs(x, p) 523 us"),
    "function_evaluate/realistic/exp_sin_over_square_at_half": (
        "g.xreplace({x: p}) 447 us (chosen); g.subs(x, p) 593 us"),
    "function_evaluate/realistic/sine_plus_square_at_root_two": (
        "g.xreplace({x: p}) 283 us (chosen); g.subs(x, p) 376 us"),
    "function_evaluate/realistic/scaled_twenty_terms_at_fraction": (
        "g.xreplace({x: p}) 4908 us (chosen); g.subs(x, p) 6879 us"),
    "partial_diff/micro/bivariate_in_y": (
        "Poly(expr, *vars).diff((v, n)) 1016 us (chosen); diff(expr, v, n) 5021 us; Poly.diff on a "
        "prebuilt Poly takes a different kind of object (28 us)"),
    "partial_diff/realistic/dense_bivariate_degree8_second_in_x": (
        "Poly(expr, *vars).diff((v, n)) 6708 us (chosen); diff(expr, v, n) 65826 us; Poly.diff on a "
        "prebuilt Poly takes a different kind of object (34 us)"),
    "partial_diff/realistic/scaled_dense_trivariate_degree6_third_in_y": (
        "Poly(expr, *vars).diff((v, n)) 11521 us (chosen); diff(expr, v, n) 84330 us; Poly.diff on a "
        "prebuilt Poly takes a different kind of object (46 us)"),
    "gradient/micro/quadratic_form_two_variables": (
        "Poly(expr, *vars) then [P.diff(v) ...] 893 us (chosen); [diff(expr, v) ...] 5393 us; "
        "Poly.diff on a prebuilt Poly takes a different kind of object (43 us)"),
    "gradient/realistic/dense_trivariate_degree4": (
        "Poly(expr, *vars) then [P.diff(v) ...] 5075 us (chosen); [diff(expr, v) ...] 55984 us; "
        "Poly.diff on a prebuilt Poly takes a different kind of object (80 us)"),
    "gradient/realistic/scaled_dense_four_variables_degree4": (
        "Poly(expr, *vars) then [P.diff(v) ...] 9176 us (chosen); [diff(expr, v) ...] 121045 us; "
        "Poly.diff on a prebuilt Poly takes a different kind of object (146 us)"),
    "jacobian/micro/two_quadratics": (
        "Poly(g, *vars) then diff 1511 us (chosen); Matrix(fs).jacobian(vars) (prebuilt Matrix) 8997 "
        "us; Matrix(fs).jacobian(vars) (built in the call) 8997 us; [[diff(g, v) ...]] 9220 us"),
    "jacobian/realistic/three_dense_cubics_three_variables": (
        "Poly(g, *vars) then diff 7411 us (chosen); Matrix(fs).jacobian(vars) (prebuilt Matrix) 71165 "
        "us; [[diff(g, v) ...]] 71467 us; Matrix(fs).jacobian(vars) (built in the call) 72088 us"),
    "jacobian/realistic/scaled_four_dense_cubics_four_variables": (
        "Poly(g, *vars) then diff 15787 us (chosen); Matrix(fs).jacobian(vars) (built in the call) "
        "171686 us; Matrix(fs).jacobian(vars) (prebuilt Matrix) 175176 us; [[diff(g, v) ...]] 178912 "
        "us"),
    "hessian/micro/quadratic_form_two_variables": (
        "Poly(expr, *vars) then [[P.diff(a).diff(b) ...]] 993 us (chosen); [[diff(expr, a, b) ...]] "
        "5375 us; hessian(expr, vars) 6125 us"),
    "hessian/realistic/dense_trivariate_degree4": (
        "Poly(expr, *vars) then [[P.diff(a).diff(b) ...]] 6267 us (chosen); hessian(expr, vars) "
        "134814 us; [[diff(expr, a, b) ...]] 141587 us"),
    "hessian/realistic/scaled_dense_four_variables_degree4": (
        "Poly(expr, *vars) then [[P.diff(a).diff(b) ...]] 13428 us (chosen); hessian(expr, vars) "
        "259225 us; [[diff(expr, a, b) ...]] 323878 us"),
    "differentiate/micro/polynomial_object_cubic": (
        "prebuilt Poly.diff((x, n)) 18 us (chosen); the other calls take an Expr, not a Poly"),
    "differentiate/micro/expression_cubic": (
        "Poly(expr, x).diff((x, n)) 661 us (chosen); diff(expr, x, n) 3040 us; Poly.diff on a "
        "prebuilt Poly takes a different kind of object (15 us)"),
    "differentiate/realistic/expression_degree12_second_derivative": (
        "Poly(expr, x).diff((x, n)) 2174 us (chosen); diff(expr, x, n) 19776 us; Poly.diff on a "
        "prebuilt Poly takes a different kind of object (31 us)"),
    "differentiate/realistic/rational_function_second_derivative": (
        "cancel(diff(f, x, n)) 28743 us (chosen); together(diff(f, x, n)) leaves a fraction whose "
        "numerator and denominator need not be coprime, not the canonical rational function; "
        "FracElement.diff(x) on a prebuilt fraction-field element 1136 us is a ground-domain element "
        "method and is excluded"),
    "differentiate/realistic/scaled_polynomial_object_degree100_third_derivative": (
        "prebuilt Poly.diff((x, n)) 55 us (chosen); the other calls take an Expr, not a Poly"),
    "analyze_function/micro/quadratic": (
        "continuous_domain, solveset (zeros, signs of f, f' = 0, signs of f'), diff and the "
        "second-derivative test 73935 us (chosen)"),
    "analyze_function/realistic/cubic_with_two_extrema": (
        "continuous_domain, solveset (zeros, signs of f, f' = 0, signs of f'), diff and the "
        "second-derivative test 66861 us (chosen)"),
    "analyze_function/realistic/odd_rational_function": (
        "continuous_domain, solveset (zeros, signs of f, f' = 0, signs of f'), diff and the "
        "second-derivative test 141262 us (chosen)"),
    "analyze_function/realistic/two_poles": (
        "continuous_domain, solveset (zeros, signs of f, f' = 0, signs of f'), diff and the "
        "second-derivative test 189563 us (chosen)"),
    "extrema_values/micro/quadratic": (
        "solveset of f' = 0, diff and the second-derivative test, subs 3903 us (chosen)"),
    "extrema_values/realistic/cubic_with_two_extrema": (
        "solveset of f' = 0, diff and the second-derivative test, subs 5612 us (chosen)"),
    "extrema_values/realistic/odd_rational_function": (
        "solveset of f' = 0, diff and the second-derivative test, subs 22300 us (chosen)"),
    "extrema_values/realistic/rational_with_one_pole": (
        "solveset of f' = 0, diff and the second-derivative test, subs 48751 us (chosen)"),
}


def _reference(operation: str, size: str, label: str) -> str:
    return _REFERENCES[f"{operation}/{size}/{label}"]


# --- shared helpers (the series and integrals modules import these) --------------------------------------------------


def ints(seed: int, count: int, low: int, high: int) -> list[int]:
    """count integers in [low, high] from a 64-bit linear congruential stream (the same on every Python version)."""
    state = (seed * _MULTIPLIER + _INCREMENT) & _MASK
    values = []
    for _ in range(count):
        state = (state * _MULTIPLIER + _INCREMENT) & _MASK
        values.append(low + (state >> 33) % (high - low + 1))
    return values


def decode(index: int, *radices: int) -> tuple[int, ...]:
    """Mixed-radix digits of an index, the first radix varying fastest."""
    digits = []
    for radix in radices:
        index, digit = divmod(index, radix)
        digits.append(digit)
    return tuple(digits)


def sympy_symbols():
    """Real SymPy symbols for the symbols of this module, keyed by the MathForge symbol they stand for."""
    import sympy as sp
    return {symbol_: sp.Symbol(symbol_.name, real=True) for symbol_ in (X, Y, Z, W, V)}


def to_sympy(value, symbols):
    """A MathForge exact value as a SymPy expression: a RealFunction, a RealExpr, an Expression (Rational, Symbol,
    Add, Mul, Pow, Sqrt), a RealAlgebraicRoot or an int. ``symbols`` maps MathForge symbols to SymPy ones."""
    import sympy as sp
    kind = type(value)
    if kind is mf.RealFunction:
        return to_sympy(value.expression, symbols)
    if kind is mf.RealExpr:
        op = value.op
        if op == "scalar":
            return to_sympy(value.value, symbols)
        if op == "symbol":
            return symbols[value.value]
        if op == "pi":
            return sp.pi
        if op == "E":
            return sp.E
        arguments = [to_sympy(argument, symbols) for argument in value.args]
        if op == "pow":
            return arguments[0] ** value.value
        if op == "add":
            return arguments[0] + arguments[1]
        if op == "mul":
            return arguments[0] * arguments[1]
        if op == "div":
            return arguments[0] / arguments[1]
        if op == "abs":
            return sp.Abs(arguments[0])
        if op == "sqrt":
            return sp.sqrt(arguments[0])
        if op in _UNARY:
            return getattr(sp, op)(arguments[0])
        raise ValueError(op)
    if kind is mf.Rational:
        return sp.Rational(value.numerator, value.denominator)
    if kind is int:
        return sp.Integer(value)
    if kind is mf.Symbol:
        return symbols[value]
    if kind is mf.Add:
        return sp.Add(*(to_sympy(term, symbols) for term in value.terms))
    if kind is mf.Mul:
        return sp.Mul(*(to_sympy(factor, symbols) for factor in value.factors))
    if kind is mf.Pow:
        return to_sympy(value.base, symbols) ** value.exponent
    if kind is mf.Sqrt:
        return sp.sqrt(to_sympy(value.radicand, symbols))
    if kind is mf.RealAlgebraicRoot:
        t = sp.Symbol("t")
        polynomial = sp.Poly(sum(c * t ** k for k, c in enumerate(value.integer_coefficients)), t)
        return polynomial.real_roots()[value.real_index]
    raise TypeError(kind)


def equal(left, right):
    """Whether two SymPy expressions are the same function: the difference simplifies to zero."""
    import sympy as sp
    difference = left - right
    if difference == 0 or sp.simplify(difference) == 0:
        return True
    return difference.equals(0) is True


# --- polynomial data and builders (shared with the integrals module) -------------------------------------------------


def polynomial_coefficients(degree: int, seed: int, *, rational: bool = False):
    """Constant-first coefficients, the leading one nonzero: integers in [-9, 9] or fractions p/q with q <= 6."""
    values = ints(seed, degree + 1, -9, 9)
    values[-1] = values[-1] or 1
    if rational:
        return tuple(Fraction(v, d) for v, d in zip(values, ints(seed + 5, degree + 1, 1, 6)))
    return tuple(values)


def cubic(i: int):
    """The i-th of 1,000 distinct integer cubics, constant first (123,462 of them; 2711 is coprime to that)."""
    c0, c1, c2, c3 = decode((i * 2711 + 17) % 123_462, 19, 19, 19, 18)
    return c0 - 9, c1 - 9, c2 - 9, (c3 - 9 if c3 < 9 else c3 - 8)


def mf_number(value):
    return mf.Rational(value.numerator, value.denominator) if isinstance(value, Fraction) else value


def mf_polynomial(coefficients):
    """A MathForge Polynomial object in x."""
    return mf.Polynomial(tuple(mf_number(c) for c in coefficients), X)


def mf_expression(coefficients):
    """The MathForge expression tree of the polynomial in x."""
    result = 0
    for degree, c in enumerate(coefficients):
        result = result + mf_number(c) * X ** degree
    return result


def sp_number(value):
    import sympy as sp
    return sp.Rational(value.numerator, value.denominator) if isinstance(value, Fraction) else value


def sp_expression(coefficients):
    """The SymPy expression of the polynomial in x."""
    x = sympy_symbols()[X]
    return sum(sp_number(c) * x ** degree for degree, c in enumerate(coefficients))


def sp_poly(coefficients):
    """The SymPy Poly of the polynomial in x."""
    import sympy as sp
    return sp.Poly([sp_number(c) for c in reversed(coefficients)], sympy_symbols()[X])


def same_polynomial(mf_value, sympy_value) -> bool:
    """A MathForge expression against a SymPy Poly or Expr: the same polynomial."""
    import sympy as sp
    return sp.expand(to_sympy(mf_value, sympy_symbols()) - sympy_value.as_expr()) == 0


def monomials(nvars: int, degree: int, seed: int):
    """The terms (exponents, coefficient) of a dense polynomial of total degree at most ``degree``, zeros dropped."""
    exponents = [e for e in itertools.product(range(degree + 1), repeat=nvars) if sum(e) <= degree]
    return tuple((e, c) for e, c in zip(exponents, ints(seed, len(exponents), -9, 9)) if c)


def build(terms, symbols):
    """The polynomial of ``terms`` as an expression of MathForge or SymPy symbols alike (plain Python arithmetic)."""
    total = 0
    for exponents, coefficient in terms:
        term = coefficient
        for symbol_, exponent in zip(symbols, exponents):
            if exponent:
                term = term * symbol_ ** exponent
        total = total + term
    return total


# --- text functions ------------------------------------------------------------------------------------------------


def _function(text: str, names: str = "x"):
    """The MathForge real function written as text; its axes are the symbols of ``names`` in that order."""
    return mf.parse_function(text, symbols={name: _NAMES[name] for name in names})


def _expression(text: str):
    """The SymPy expression written as text."""
    import sympy as sp
    symbols = sympy_symbols()
    return sp.sympify(text, locals={name: symbols[_NAMES[name]] for name in _NAMES})


def _same_function(mf_value, sympy_value) -> bool:
    return equal(to_sympy(mf_value, sympy_symbols()), sympy_value)


def _vector(names: str):
    return tuple(_NAMES[name] for name in names)


def _sympy_vector(names: str):
    symbols = sympy_symbols()
    return [symbols[_NAMES[name]] for name in names]


# --- function_differentiate and function_partial_diff --------------------------------------------------------------


def _differentiate_fixture(operation, size, label, text_of, names, variable, order, sympy_call, *,
                           variants, v07=""):
    """operation is function_differentiate or function_partial_diff: both differentiate by one axis."""
    def mathforge(i):
        return getattr(mf, operation), (_function(text_of(i), names), _NAMES[variable], order), {}

    def sympy(i):
        return sympy_call(_expression(text_of(i)), sympy_symbols()[_NAMES[variable]], order)

    return Fixture(operation, size, label, mathforge, sympy, _same_function, _reference(operation, size, label),
                   variants, v07)


def _sympy_diff(function, variable, order):
    import sympy as sp
    return lambda: sp.diff(function, variable, order)


def _sympy_method_diff(function, variable, order):
    """``g.diff(v, n)``, the method that ``diff`` calls: the same derivative, one call less."""
    return lambda: function.diff(variable, order)


# --- function_gradient, function_jacobian and function_hessian -----------------------------------------------------


def _same_vector(mf_value, sympy_value) -> bool:
    """A tuple of RealFunctions against a list, tuple or one-row Matrix of SymPy expressions."""
    symbols = sympy_symbols()
    items = list(sympy_value)
    return (type(mf_value) is tuple and len(mf_value) == len(items)
            and all(equal(to_sympy(a, symbols), b) for a, b in zip(mf_value, items)))


def _rows(grid):
    return grid.tolist() if hasattr(grid, "tolist") else [list(row) for row in grid]


def _same_grid(mf_value, sympy_value) -> bool:
    """A tuple of tuples of RealFunctions against a SymPy Matrix or a list of lists of expressions."""
    symbols, rows = sympy_symbols(), _rows(sympy_value)
    return (type(mf_value) is tuple and len(mf_value) == len(rows)
            and all(len(row) == len(other) for row, other in zip(mf_value, rows))
            and all(equal(to_sympy(a, symbols), b) for row, other in zip(mf_value, rows)
                    for a, b in zip(row, other)))


def _gradient_fixture(size, label, text_of, names, sympy_call, *, variants):
    def mathforge(i):
        return mf.function_gradient, (_function(text_of(i), names), _vector(names)), {}

    def sympy(i):
        return sympy_call(_expression(text_of(i)), _sympy_vector(names))

    return Fixture("function_gradient", size, label, mathforge, sympy, _same_vector,
                   _reference("function_gradient", size, label), variants)


def _gradient_diffs(function, variables):
    import sympy as sp
    return lambda: [sp.diff(function, variable) for variable in variables]


def _jacobian_fixture(size, label, texts_of, names, sympy_call, *, variants):
    def mathforge(i):
        return mf.function_jacobian, (tuple(_function(t, names) for t in texts_of(i)), _vector(names)), {}

    def sympy(i):
        return sympy_call([_expression(t) for t in texts_of(i)], _sympy_vector(names))

    return Fixture("function_jacobian", size, label, mathforge, sympy, _same_grid,
                   _reference("function_jacobian", size, label), variants)


def _jacobian_matrix(functions, variables):
    import sympy as sp
    matrix = sp.Matrix(functions)
    return lambda: matrix.jacobian(variables)


def _jacobian_diffs(functions, variables):
    """The grid of first derivatives, ``[[diff(g, v) ...]]``."""
    import sympy as sp
    return lambda: [[sp.diff(function, variable) for variable in variables] for function in functions]


def _hessian_fixture(size, label, text_of, names, sympy_call, *, variants):
    def mathforge(i):
        return mf.function_hessian, (_function(text_of(i), names), _vector(names)), {}

    def sympy(i):
        return sympy_call(_expression(text_of(i)), _sympy_vector(names))

    return Fixture("function_hessian", size, label, mathforge, sympy, _same_grid,
                   _reference("function_hessian", size, label), variants)


def _hessian_call(function, variables):
    import sympy as sp
    return lambda: sp.hessian(function, variables)


def _hessian_diffs(function, variables):
    """The grid of second derivatives, ``[[diff(g, a, b) ...]]``: faster than ``hessian`` on a small function."""
    import sympy as sp
    return lambda: [[sp.diff(function, a, b) for b in variables] for a in variables]


# --- function_evaluate ---------------------------------------------------------------------------------------------


def _evaluate_fixture(size, label, text_of, point_of, sympy_point_of, sympy_call, *, variants):
    """point_of(i) and sympy_point_of(i) are the binding of x as a MathForge and as a SymPy value."""
    def mathforge(i):
        return mf.function_evaluate, (_function(text_of(i)), {X: point_of(i)}), {}

    def sympy(i):
        return sympy_call(_expression(text_of(i)), sympy_symbols()[X], sympy_point_of(i))

    return Fixture("function_evaluate", size, label, mathforge, sympy, _same_function,
                   _reference("function_evaluate", size, label), variants)


def _xreplace_call(function, variable, value):
    return lambda: function.xreplace({variable: value})


def _micro_evaluation(i: int) -> tuple[int, int]:
    """(k, p) of k x^2 + 3 x + 1 at x = p: 1,020 distinct pairs in turn (2711 is coprime to that)."""
    k, p = decode((i * 2711 + 17) % 1020, 60, 17)
    return k + 1, p - 8


def _sqrt2_mathforge(i):
    return mf.sqrt(2)


def _sqrt2_sympy(i):
    import sympy as sp
    return sp.sqrt(2)


def _fraction_mathforge(numerator, denominator):
    return lambda i: mf.Rational(numerator, denominator)


def _fraction_sympy(numerator, denominator):
    def build(i):  # SymPy is imported when the reference is built, never when the fixture list is defined.
        import sympy as sp
        return sp.Rational(numerator, denominator)

    return build


def _twenty_terms(i):
    return "+".join(f"sin({k}*x)*exp(x/{k})" for k in range(1, 21))


# --- multivariate polynomials: partial_diff, gradient, jacobian, hessian ---------------------------------------------


def _same_polynomials(mf_value, sympy_value) -> bool:
    """A tuple of expressions against a list of SymPy Polys or expressions."""
    items = list(sympy_value)
    return (type(mf_value) is tuple and len(mf_value) == len(items)
            and all(same_polynomial(a, b) for a, b in zip(mf_value, items)))


def _same_polynomial_grid(mf_value, sympy_value) -> bool:
    rows = _rows(sympy_value)
    return (type(mf_value) is tuple and len(mf_value) == len(rows)
            and all(len(row) == len(other) for row, other in zip(mf_value, rows))
            and all(same_polynomial(a, b) for row, other in zip(mf_value, rows) for a, b in zip(row, other)))


def _partial_fixture(size, label, terms_of, nvars, index, order, sympy_call, *, variants):
    names = "xyzwv"[:nvars]

    def mathforge(i):
        return mf.partial_diff, (build(terms_of(i), _vector(names)), _vector(names)[index], order), {}

    def sympy(i):
        symbols = _sympy_vector(names)
        return sympy_call(build(terms_of(i), symbols), symbols, index, order)

    return Fixture("partial_diff", size, label, mathforge, sympy, same_polynomial,
                   _reference("partial_diff", size, label), variants)


def _partial_poly_call(function, symbols, index, order):
    """The Poly of the expression, then Poly.diff: the fastest way from an Expr, conversion included."""
    import sympy as sp
    return lambda: sp.Poly(function, *symbols).diff((symbols[index], order))


def _gradient_poly_fixture(size, label, terms_of, nvars, sympy_call, *, variants):
    names = "xyzwv"[:nvars]

    def mathforge(i):
        return mf.gradient, (build(terms_of(i), _vector(names)), _vector(names)), {}

    def sympy(i):
        symbols = _sympy_vector(names)
        return sympy_call(build(terms_of(i), symbols), symbols)

    return Fixture("gradient", size, label, mathforge, sympy, _same_polynomials,
                   _reference("gradient", size, label), variants)


def _polynomial_gradient_poly(function, symbols):
    import sympy as sp
    return lambda: (lambda poly: [poly.diff(symbol_) for symbol_ in symbols])(sp.Poly(function, *symbols))


def _jacobian_poly_fixture(size, label, terms_of, nvars, nfunctions, sympy_call, *, variants):
    names = "xyzwv"[:nvars]

    def mathforge(i):
        functions = tuple(build(terms_of(i, j), _vector(names)) for j in range(nfunctions))
        return mf.jacobian, (functions, _vector(names)), {}

    def sympy(i):
        symbols = _sympy_vector(names)
        return sympy_call([build(terms_of(i, j), symbols) for j in range(nfunctions)], symbols)

    return Fixture("jacobian", size, label, mathforge, sympy, _same_polynomial_grid,
                   _reference("jacobian", size, label), variants)


def _jacobian_poly_chain(functions, symbols):
    """One Poly per function, then Poly.diff by every variable: the fastest way from Exprs, conversion included."""
    import sympy as sp
    return lambda: [(lambda poly: [poly.diff(symbol_) for symbol_ in symbols])(sp.Poly(function, *symbols))
                    for function in functions]


def _hessian_poly_fixture(size, label, terms_of, nvars, sympy_call, *, variants):
    names = "xyzwv"[:nvars]

    def mathforge(i):
        return mf.hessian, (build(terms_of(i), _vector(names)), _vector(names)), {}

    def sympy(i):
        symbols = _sympy_vector(names)
        return sympy_call(build(terms_of(i), symbols), symbols)

    return Fixture("hessian", size, label, mathforge, sympy, _same_polynomial_grid,
                   _reference("hessian", size, label), variants)


def _hessian_poly_chain(function, symbols):
    """The Poly of the expression, its first derivatives, and the upper triangle of the second (the matrix is
    symmetric): the fastest way from an Expr, conversion included."""
    import sympy as sp

    def call():
        first = [sp.Poly(function, *symbols).diff(symbol_) for symbol_ in symbols]
        grid = [[None] * len(symbols) for _ in symbols]
        for i, row in enumerate(first):
            for j in range(i, len(symbols)):
                grid[i][j] = grid[j][i] = row.diff(symbols[j])
        return grid
    return call


def _micro_bivariate(i: int):
    """a x^2 y + b y^3 + c x y with a, b, c in +-[1, 9]: 5,832 distinct triples (2711 is coprime to that)."""
    a, b, c = (v - 9 if v < 9 else v - 8 for v in decode((i * 2711 + 17) % 5832, 18, 18, 18))
    return (((2, 1), a), ((0, 3), b), ((1, 1), c))


def _micro_quadratic_form(i: int):
    """a x^2 + b x y + c y^2 with a, b, c in +-[1, 9]."""
    a, b, c = (v - 9 if v < 9 else v - 8 for v in decode((i * 2711 + 17) % 5832, 18, 18, 18))
    return (((2, 0), a), ((1, 1), b), ((0, 2), c))


def _micro_pair(i: int, j: int):
    """The j-th of two quadratic polynomials in x and y for a 2 x 2 Jacobian."""
    return _micro_quadratic_form(i + 1000 * j + j)


# --- differentiate -------------------------------------------------------------------------------------------------


def _differentiate_polynomial_fixture(size, label, coefficients_of, order, form, sympy_call, *, variants):
    """form is "object" (a Polynomial against a prebuilt Poly) or "expression" (an Expression against an Expr)."""
    build_input = mf_polynomial if form == "object" else mf_expression

    def mathforge(i):
        return mf.diff, (build_input(coefficients_of(i)), X, order), {}

    def sympy(i):
        return sympy_call(coefficients_of(i), order)

    return Fixture("differentiate", size, label, mathforge, sympy, same_polynomial,
                   _reference("differentiate", size, label), variants)


def _diff_poly_object(coefficients, order):
    poly, x = sp_poly(coefficients), sympy_symbols()[X]
    return lambda: poly.diff((x, order))


def _diff_poly_via_poly(coefficients, order):
    import sympy as sp
    expression, x = sp_expression(coefficients), sympy_symbols()[X]
    return lambda: sp.Poly(expression, x).diff((x, order))


def _same_rational_function(mf_value, sympy_value) -> bool:
    """A MathForge RationalFunction against a SymPy expression: as a fraction they agree."""
    import sympy as sp
    if type(mf_value) is not mf.RationalFunction:
        return False
    symbols = sympy_symbols()
    ours = (to_sympy(mf_value.numerator.to_expression(), symbols)
            / to_sympy(mf_value.denominator.to_expression(), symbols))
    return sp.cancel(ours - sympy_value.as_expr()) == 0


def _differentiate_rational_fixture(size, label, parts_of, order, sympy_call, *, variants):
    """parts_of(i) is a function x -> (numerator, denominator), evaluated on the MathForge and the SymPy symbol."""
    def mathforge(i):
        numerator, denominator = parts_of(i)(X)
        return mf.diff, (mf.rational_function(numerator, denominator, variable=X), X, order), {}

    def sympy(i):
        x = sympy_symbols()[X]
        numerator, denominator = parts_of(i)(x)
        return sympy_call(numerator / denominator, x, order)

    return Fixture("differentiate", size, label, mathforge, sympy, _same_rational_function,
                   _reference("differentiate", size, label), variants)


def _rational_cancel_diff(function, variable, order):
    """The derivative of the prebuilt rational expression, reduced to lowest terms."""
    import sympy as sp
    return lambda: sp.cancel(sp.diff(function, variable, order))


# --- analyze_function and extrema_values ---------------------------------------------------------------------------


def _analysis_chain(function, x):
    from sympy import S, cancel, Complement, diff, solveset
    from sympy.calculus.util import continuous_domain

    def call():
        domain = continuous_domain(function, x, S.Reals)
        excluded = Complement(S.Reals, domain)
        zeros = solveset(function, x, S.Reals)
        positive = solveset(function > 0, x, S.Reals)
        negative = solveset(function < 0, x, S.Reals)
        derivative = cancel(diff(function, x))
        stationary = solveset(derivative, x, S.Reals)
        increasing = solveset(derivative > 0, x, S.Reals)
        decreasing = solveset(derivative < 0, x, S.Reals)
        curvature = cancel(diff(derivative, x))
        extrema = [(point, "minimum" if curvature.subs(x, point) > 0 else "maximum") for point in stationary]
        return (domain, excluded, zeros, positive, negative, derivative, stationary, increasing, decreasing,
                extrema)
    return call


def _extrema_chain(function, x):
    from sympy import S, cancel, diff, solveset

    def call():
        derivative = cancel(diff(function, x))
        curvature = cancel(diff(derivative, x))
        result = []
        for point in solveset(derivative, x, S.Reals):
            result.append((point, function.subs(x, point), "minimum" if curvature.subs(x, point) > 0 else "maximum"))
        return result
    return call


def _numeric(value):
    import sympy as sp
    return sp.N(value, 40)


def _finite_points(sympy_set):
    """The elements of a finite SymPy set as sorted 40-digit numbers."""
    return sorted(_numeric(p) for p in sympy_set)


def _mf_points(interval_set):
    """The points of a MathForge IntervalSet of degenerate intervals as sorted 40-digit numbers."""
    symbols = sympy_symbols()
    return sorted(_numeric(to_sympy(i.lower, symbols)) for i in interval_set.intervals if i.lower == i.upper)


def _same_points(ours, theirs) -> bool:
    import sympy as sp
    return len(ours) == len(theirs) and all(abs(a - b) < sp.Float("1e-30") for a, b in zip(ours, theirs))


def _breakpoints(mf_sets, sympy_sets):
    """Every finite endpoint of the MathForge and SymPy sets as sorted distinct numbers (40 digits)."""
    import sympy as sp
    symbols = sympy_symbols()
    numbers = []
    for interval_set in mf_sets:
        for interval in interval_set.intervals:
            ends = (interval.lower, interval.upper)
            numbers.extend(_numeric(to_sympy(end, symbols)) for end in ends if end is not None)
    for sympy_set in sympy_sets:
        for piece in (sympy_set.args if isinstance(sympy_set, sp.Union) else (sympy_set,)):
            if isinstance(piece, sp.Interval):
                numbers.extend(_numeric(end) for end in (piece.start, piece.end) if end not in (sp.oo, -sp.oo))
            elif isinstance(piece, sp.FiniteSet):
                numbers.extend(_numeric(p) for p in piece)
    numbers.sort()
    distinct = []
    for number in numbers:
        if not distinct or abs(number - distinct[-1]) > sp.Float("1e-30"):
            distinct.append(number)
    return distinct


def _samples(points):
    """Exact rationals strictly inside every gap between breakpoints, and beyond both ends."""
    import sympy as sp
    if not points:
        return [sp.Integer(0)]
    cuts = [points[0] - 1, *((a + b) / 2 for a, b in zip(points, points[1:])), points[-1] + 1]
    return [sp.nsimplify(c, rational=True, tolerance=sp.Float("1e-20")) for c in cuts]


def _same_up_to_points(mf_set, sympy_set, samples) -> bool:
    """Two real sets agree on every sample: they are the same set up to finitely many points."""
    return all(bool(mf_set.contains(mf.Rational(int(s.p), int(s.q)))) == bool(sympy_set.contains(s)) for s in samples)


def _same_analysis(mf_value, sympy_value) -> bool:
    import sympy as sp
    if type(mf_value) is not mf.FunctionAnalysis:
        return False
    (domain, excluded, zeros, positive, negative, derivative, stationary, increasing, decreasing,
     extrema) = sympy_value
    symbols = sympy_symbols()
    regions = {behavior: mf.IntervalSet(tuple(r.interval for r in mf_value.monotonic_regions if r.behavior == behavior))
               for behavior in ("increasing", "decreasing")}
    if not (_same_points(_mf_points(mf_value.zero_set), _finite_points(zeros))
            and _same_points(_mf_points(mf_value.stationary_set), _finite_points(stationary))
            and _same_points(sorted(_numeric(to_sympy(e.point, symbols)) for e in mf_value.excluded_points),
                             _finite_points(excluded))):
        return False
    pairs = ((mf_value.domain, domain), (mf_value.positive_set, positive), (mf_value.negative_set, negative),
             (regions["increasing"], increasing), (regions["decreasing"], decreasing))
    samples = _samples(_breakpoints([a for a, _ in pairs], [b for _, b in pairs]))
    if not all(_same_up_to_points(a, b, samples) for a, b in pairs):
        return False
    ours = sorted((_numeric(to_sympy(e.point, symbols)), e.kind) for e in mf_value.strict_extrema)
    theirs = sorted((_numeric(p), kind) for p, kind in extrema)
    return (len(ours) == len(theirs) and all(a[1] == b[1] and abs(a[0] - b[0]) < sp.Float("1e-30")
                                             for a, b in zip(ours, theirs))
            and _same_rational_function(mf_value.derivative, derivative))


def _same_extrema(mf_value, sympy_value) -> bool:
    import sympy as sp
    if type(mf_value) is not tuple:
        return False
    symbols = sympy_symbols()
    ours = sorted((_numeric(to_sympy(e.point, symbols)), _numeric(to_sympy(e.value, symbols)), e.kind)
                  for e in mf_value)
    theirs = sorted((_numeric(p), _numeric(v), kind) for p, v, kind in sympy_value)
    return (len(ours) == len(theirs) and all(
        a[2] == b[2] and abs(a[0] - b[0]) < sp.Float("1e-30") and abs(a[1] - b[1]) < sp.Float("1e-30")
        for a, b in zip(ours, theirs)))


def _quadratic_coefficients(i: int):
    """(a, b, c) of a x^2 + b x + c with a in +-[1, 9], b and c in [-9, 9]: 6,498 distinct triples."""
    a, b, c = decode((i * 2711 + 17) % 6498, 18, 19, 19)
    return (a - 9 if a < 9 else a - 8), b - 9, c - 9


def _quadratic_parts(i):
    a, b, c = _quadratic_coefficients(i)
    return lambda x: (a * x ** 2 + b * x + c, 1)


def _analysis_fixture(operation, size, label, parts_of, chain, agree, *, variants, form="rational"):
    """parts_of(i) is a function x -> (numerator, denominator); the MathForge input is a RationalFunction, or the
    numerator as an Expression when form is "polynomial"."""
    public = getattr(mf, operation)

    def mathforge(i):
        numerator, denominator = parts_of(i)(X)
        function = numerator if form == "polynomial" else mf.rational_function(numerator, denominator, variable=X)
        return public, (function, X), {}

    def sympy(i):
        x = sympy_symbols()[X]
        numerator, denominator = parts_of(i)(x)
        return chain(numerator / denominator, x)

    return Fixture(operation, size, label, mathforge, sympy, agree, _reference(operation, size, label), variants)


FIXTURES = (
    # function_differentiate
    _differentiate_fixture("function_differentiate", "micro", "sin_kx", lambda i: f"sin({i + 1}*x)", "x", "x", 1,
                           _sympy_method_diff, variants=1000),
    _differentiate_fixture("function_differentiate", "realistic", "exp_sin_cos_third_derivative",
                           lambda i: "exp(x)*sin(2*x)+cos(x*x)", "x", "x", 3, _sympy_diff, variants=1,
                           v07="derivative3"),
    _differentiate_fixture("function_differentiate", "realistic", "tan_exp_second_derivative",
                           lambda i: "tan(x)*exp(-x)", "x", "x", 2, _sympy_diff, variants=1),
    _differentiate_fixture("function_differentiate", "realistic", "log_sin_second_derivative",
                           lambda i: "log(1+x*x)*sin(x)", "x", "x", 2, _sympy_diff, variants=1),
    _differentiate_fixture(
        "function_differentiate", "realistic", "scaled_six_terms_first_derivative",
        lambda i: "exp(x)*sin(2*x)+cos(x*x)+log(1+x*x)+atan(x)*x+sqrt(1+x*x)+tanh(3*x)", "x", "x", 1,
        _sympy_method_diff, variants=1),
    # function_partial_diff
    _differentiate_fixture("function_partial_diff", "micro", "polynomial_and_sine", lambda i: f"x*x*y+{i + 1}*sin(y)",
                           "xy", "y", 1, _sympy_method_diff, variants=1000),
    _differentiate_fixture("function_partial_diff", "realistic", "two_variables_second_derivative",
                           lambda i: "exp(x*y)*sin(x+y)+x*log(1+y*y)", "xy", "y", 2, _sympy_diff, variants=1),
    _differentiate_fixture("function_partial_diff", "realistic", "scaled_three_variables_third_derivative",
                           lambda i: "x*exp(y*z)+sin(x*y)+z**3*log(1+x*x)", "xyz", "z", 3, _sympy_diff, variants=1),
    # function_gradient
    _gradient_fixture("micro", "two_variables", lambda i: f"x*y+{i + 1}*sin(y)", "xy", _gradient_diffs,
                      variants=1000),
    _gradient_fixture("realistic", "three_variables", lambda i: "x*exp(y*z)+sin(x*y)+z**3*log(1+x*x)", "xyz",
                      _gradient_diffs, variants=1),
    _gradient_fixture("realistic", "scaled_five_variables",
                      lambda i: "x*exp(y*z)+sin(x*w)*cos(v*y)+z*w*v+exp(x*v)", "xyzwv", _gradient_diffs, variants=1),
    # function_jacobian
    _jacobian_fixture("micro", "two_by_two", lambda i: ("x*y", f"x+{i + 1}*y*y"), "xy", _jacobian_matrix,
                      variants=1000),
    _jacobian_fixture("realistic", "spherical_to_cartesian",
                      lambda i: ("x*sin(y)*cos(z)", "x*sin(y)*sin(z)", "x*cos(y)"), "xyz", _jacobian_matrix,
                      variants=1),
    _jacobian_fixture("realistic", "scaled_four_by_four",
                      lambda i: ("x*exp(y)+sin(z*w)", "y*cos(x)+z*w", "exp(x*z)+y*w*w", "log(1+x*x)*z+sin(y*w)"),
                      "xyzw", _jacobian_diffs, variants=1),
    # function_hessian
    _hessian_fixture("micro", "two_variables", lambda i: f"x*x*y+{i + 1}*y*y*y", "xy", _hessian_diffs, variants=1000),
    _hessian_fixture("realistic", "three_variables", lambda i: "exp(x*y)+sin(y*z)+x*z**2", "xyz", _hessian_call,
                     variants=1),
    _hessian_fixture("realistic", "scaled_four_variables", lambda i: "exp(x*y)+sin(y*z)*cos(w)+x*z*w+log(1+w*w)",
                     "xyzw", _hessian_call, variants=1),
    # function_evaluate
    _evaluate_fixture("micro", "quadratic_at_integer", lambda i: f"{_micro_evaluation(i)[0]}*x*x+3*x+1",
                      lambda i: _micro_evaluation(i)[1], lambda i: _micro_evaluation(i)[1], _xreplace_call,
                      variants=1000),
    _evaluate_fixture("realistic", "exp_sin_over_square_at_half", lambda i: "exp(x)*sin(x)/(1+x*x)",
                      _fraction_mathforge(1, 2), _fraction_sympy(1, 2), _xreplace_call, variants=1),
    _evaluate_fixture("realistic", "sine_plus_square_at_root_two", lambda i: "sin(x)+x*x", _sqrt2_mathforge,
                      _sqrt2_sympy, _xreplace_call, variants=1),
    _evaluate_fixture("realistic", "scaled_twenty_terms_at_fraction", _twenty_terms, _fraction_mathforge(3, 7),
                      _fraction_sympy(3, 7), _xreplace_call, variants=1),
    # partial_diff
    _partial_fixture("micro", "bivariate_in_y", _micro_bivariate, 2, 1, 1, _partial_poly_call, variants=1000),
    _partial_fixture("realistic", "dense_bivariate_degree8_second_in_x", lambda i: monomials(2, 8, 800 + i), 2, 0, 2,
                     _partial_poly_call, variants=8),
    _partial_fixture("realistic", "scaled_dense_trivariate_degree6_third_in_y", lambda i: monomials(3, 6, 810 + i), 3,
                     1, 3, _partial_poly_call, variants=4),
    # gradient
    _gradient_poly_fixture("micro", "quadratic_form_two_variables", _micro_quadratic_form, 2,
                           _polynomial_gradient_poly, variants=1000),
    _gradient_poly_fixture("realistic", "dense_trivariate_degree4", lambda i: monomials(3, 4, 820 + i), 3,
                           _polynomial_gradient_poly, variants=8),
    _gradient_poly_fixture("realistic", "scaled_dense_four_variables_degree4", lambda i: monomials(4, 4, 830 + i), 4,
                           _polynomial_gradient_poly, variants=4),
    # jacobian
    _jacobian_poly_fixture("micro", "two_quadratics", _micro_pair, 2, 2, _jacobian_poly_chain, variants=1000),
    _jacobian_poly_fixture("realistic", "three_dense_cubics_three_variables",
                           lambda i, j: monomials(3, 3, 840 + i * 3 + j), 3, 3, _jacobian_poly_chain, variants=8),
    _jacobian_poly_fixture("realistic", "scaled_four_dense_cubics_four_variables",
                           lambda i, j: monomials(4, 3, 850 + i * 4 + j), 4, 4, _jacobian_poly_chain, variants=4),
    # hessian
    _hessian_poly_fixture("micro", "quadratic_form_two_variables", _micro_quadratic_form, 2, _hessian_poly_chain,
                          variants=1000),
    _hessian_poly_fixture("realistic", "dense_trivariate_degree4", lambda i: monomials(3, 4, 860 + i), 3,
                          _hessian_poly_chain, variants=8),
    _hessian_poly_fixture("realistic", "scaled_dense_four_variables_degree4", lambda i: monomials(4, 4, 870 + i), 4,
                          _hessian_poly_chain, variants=4),
    # differentiate
    _differentiate_polynomial_fixture("micro", "polynomial_object_cubic", cubic, 1, "object", _diff_poly_object,
                                      variants=1000),
    _differentiate_polynomial_fixture("micro", "expression_cubic", cubic, 1, "expression", _diff_poly_via_poly,
                                      variants=1000),
    _differentiate_polynomial_fixture("realistic", "expression_degree12_second_derivative",
                                      lambda i: polynomial_coefficients(12, 880 + i, rational=True), 2, "expression",
                                      _diff_poly_via_poly, variants=8),
    _differentiate_rational_fixture("realistic", "rational_function_second_derivative",
                                    lambda i: (lambda x: (x ** 3 + 2 * x + 1, x ** 2 + x + 1)), 2,
                                    _rational_cancel_diff, variants=1),
    _differentiate_polynomial_fixture("realistic", "scaled_polynomial_object_degree100_third_derivative",
                                      lambda i: polynomial_coefficients(100, 890 + i), 3, "object", _diff_poly_object,
                                      variants=4),
    # analyze_function
    _analysis_fixture("analyze_function", "micro", "quadratic", _quadratic_parts, _analysis_chain, _same_analysis,
                      variants=1000, form="polynomial"),
    _analysis_fixture("analyze_function", "realistic", "cubic_with_two_extrema",
                      lambda i: (lambda x: (x ** 3 - 3 * x, 1)), _analysis_chain, _same_analysis, variants=1,
                      form="polynomial"),
    _analysis_fixture("analyze_function", "realistic", "odd_rational_function",
                      lambda i: (lambda x: (x, x * x + 1)), _analysis_chain, _same_analysis, variants=1),
    _analysis_fixture("analyze_function", "realistic", "two_poles",
                      lambda i: (lambda x: (1, x * x - 4)), _analysis_chain, _same_analysis, variants=1),
    # extrema_values
    _analysis_fixture("extrema_values", "micro", "quadratic", _quadratic_parts, _extrema_chain, _same_extrema,
                      variants=1000, form="polynomial"),
    _analysis_fixture("extrema_values", "realistic", "cubic_with_two_extrema",
                      lambda i: (lambda x: (x ** 3 - 3 * x, 1)), _extrema_chain, _same_extrema, variants=1,
                      form="polynomial"),
    _analysis_fixture("extrema_values", "realistic", "odd_rational_function",
                      lambda i: (lambda x: (x, x * x + 1)), _extrema_chain, _same_extrema, variants=1),
    _analysis_fixture("extrema_values", "realistic", "rational_with_one_pole",
                      lambda i: (lambda x: (x * x + 1, x - 1)), _extrema_chain, _same_extrema, variants=1),
)

# --- edge cases ----------------------------------------------------------------------------------------------------

_TYPE_EDGES = (("type_bool", True), ("type_float", 1.5), ("type_str", "1"), ("type_none", None))


def edge(operation, label, build, public=None):
    """An EdgeCase of the public function of an operation; build() returns its (args, kwargs)."""
    name = public or operation

    def mathforge():
        args, kwargs = build()
        return getattr(mf, name), args, kwargs
    return EdgeCase(operation, label, mathforge)


def type_edges(operation, tail=lambda: (), public=None):
    """The True / 1.5 / "1" / None edges in the first argument; tail() gives the other, valid arguments."""
    return tuple(edge(operation, label, lambda value=value: ((value, *tail()), {}), public)
                 for label, value in _TYPE_EDGES)


def big_raw():
    """x + 2**140000, a raw real expression whose constant has 140,001 bits against max_integer_bits = 65,536."""
    from mathforge.functions import as_real_expr
    return as_real_expr(X) + 2 ** 140_000


def deep_raw(depth: int = 400):
    """sin(sin(...(x))) nested 400 deep against max_depth = 128."""
    from mathforge import functions
    value = functions.as_real_expr(X)
    for _ in range(depth):
        value = functions.sin(value)
    return value


def big_polynomial():
    """x**3 + 2**140000 as an Expression."""
    return X ** 3 + 2 ** 140_000


def high_degree():
    """x**2200 + 1: degree 2,200 against the polynomial degree limit of 1,024."""
    return X ** 2200 + 1


def _f(text="sin(x)*exp(x)", names="x"):
    return _function(text, names)


def positive_symbol():
    """A symbol with the assumption "positive" (and a fixed uid), which no operation accepts as a variable."""
    return mf.Symbol("t", mf.Reals, (mf.Assumption("positive", mf.Truth.TRUE),),
                     uid=str(uuid.UUID(int=101)))


def _function_edges(operation, tail, public=None):
    """Edges of a function operation whose first argument is a real function: four types and two over-limit inputs."""
    return (*type_edges(operation, tail, public),
            edge(operation, "over_limit_integer", lambda: ((big_raw(), *tail()), {}), public),
            edge(operation, "over_limit_depth", lambda: ((deep_raw(), *tail()), {}), public))


def _polynomial_edges(operation, tail, public=None):
    """Edges of a polynomial operation whose first argument is an expression."""
    return (*type_edges(operation, tail, public),
            edge(operation, "type_list", lambda: (([1, 2], *tail()), {}), public),
            edge(operation, "over_limit_integer", lambda: ((big_polynomial(), *tail()), {}), public))


_TWO = lambda: _f("x*y+sin(y)", "xy")  # noqa: E731

EDGE_CASES: tuple[EdgeCase, ...] = (
    # function_differentiate
    *_function_edges("function_differentiate", lambda: (X,)),
    edge("function_differentiate", "variable_str", lambda: ((_f(), "x"), {})),
    edge("function_differentiate", "variable_not_axis", lambda: ((_f(), Y), {})),
    edge("function_differentiate", "variable_with_assumption",
         lambda: ((_f(), positive_symbol()), {})),
    edge("function_differentiate", "order_negative", lambda: ((_f(), X, -1), {})),
    edge("function_differentiate", "order_float", lambda: ((_f(), X, 1.5), {})),
    edge("function_differentiate", "order_bool", lambda: ((_f(), X, True), {})),
    edge("function_differentiate", "over_limit_order", lambda: ((_f(), X, 10 ** 7), {})),
    edge("function_differentiate", "order_zero", lambda: ((_f(), X, 0), {})),
    edge("function_differentiate", "square_root_second_derivative", lambda: ((_f("sqrt(x)"), X, 2), {})),
    edge("function_differentiate", "absolute_value", lambda: ((_f("abs(x)"), X, 1), {})),
    edge("function_differentiate", "rational_function", lambda: ((_f("1/(1+x*x)"), X, 2), {})),
    edge("function_differentiate", "raw_real_expression",
         lambda: ((mf.functions.sin(mf.functions.as_real_expr(X)), X), {})),
    # function_partial_diff
    *_function_edges("function_partial_diff", lambda: (X,)),
    edge("function_partial_diff", "variable_str", lambda: ((_TWO(), "y"), {})),
    edge("function_partial_diff", "variable_not_axis", lambda: ((_TWO(), Z), {})),
    edge("function_partial_diff", "order_negative", lambda: ((_TWO(), Y, -1), {})),
    edge("function_partial_diff", "order_float", lambda: ((_TWO(), Y, 1.5), {})),
    edge("function_partial_diff", "over_limit_order", lambda: ((_TWO(), Y, 10 ** 7), {})),
    edge("function_partial_diff", "order_zero", lambda: ((_TWO(), Y, 0), {})),
    edge("function_partial_diff", "mixed_function", lambda: ((_TWO(), Y, 2), {})),
    # function_gradient
    *_function_edges("function_gradient", lambda: ((X,),)),
    edge("function_gradient", "variables_empty", lambda: ((_f(), ()), {})),
    edge("function_gradient", "variables_str", lambda: ((_f(), "x"), {})),
    edge("function_gradient", "variables_none", lambda: ((_f(), None), {})),
    edge("function_gradient", "variable_not_symbol", lambda: ((_f(), (1,)), {})),
    edge("function_gradient", "missing_free_symbol", lambda: ((_TWO(), (X,)), {})),
    edge("function_gradient", "duplicate_variables", lambda: ((_TWO(), (X, X)), {})),
    edge("function_gradient", "two_variables", lambda: ((_TWO(), (X, Y)), {})),
    edge("function_gradient", "reordered_variables", lambda: ((_TWO(), (Y, X)), {})),
    # function_jacobian
    *tuple(edge("function_jacobian", label, lambda value=value: (((value,), (X,)), {}))
           for label, value in _TYPE_EDGES),
    edge("function_jacobian", "over_limit_integer", lambda: (((big_raw(),), (X,)), {})),
    edge("function_jacobian", "over_limit_depth", lambda: (((deep_raw(),), (X,)), {})),
    edge("function_jacobian", "functions_empty", lambda: (((), (X,)), {})),
    edge("function_jacobian", "variables_empty", lambda: (((_f(),), ()), {})),
    edge("function_jacobian", "functions_str", lambda: (("f", (X,)), {})),
    edge("function_jacobian", "functions_none", lambda: ((None, (X,)), {})),
    edge("function_jacobian", "two_functions", lambda: (((_TWO(), _f("x+y", "xy")), (X, Y)), {})),
    # function_hessian
    *_function_edges("function_hessian", lambda: ((X,),)),
    edge("function_hessian", "variables_empty", lambda: ((_f(), ()), {})),
    edge("function_hessian", "missing_free_symbol", lambda: ((_TWO(), (X,)), {})),
    edge("function_hessian", "two_variables", lambda: ((_TWO(), (X, Y)), {})),
    # function_evaluate
    *_function_edges("function_evaluate", lambda: ({X: 1},)),
    edge("function_evaluate", "bindings_none", lambda: ((_f(), None), {})),
    edge("function_evaluate", "binding_missing", lambda: ((_TWO(), {X: 1}), {})),
    edge("function_evaluate", "binding_key_str", lambda: ((_f(), {"x": 1}), {})),
    edge("function_evaluate", "binding_value_float", lambda: ((_f(), {X: 1.5}), {})),
    edge("function_evaluate", "binding_symbolic", lambda: ((_f(), {X: Y}), {})),
    edge("function_evaluate", "undefined_at_pole", lambda: ((_f("1/x"), {X: 0}), {})),
    edge("function_evaluate", "undefined_logarithm_of_negative", lambda: ((_f("log(x)"), {X: -1}), {})),
    edge("function_evaluate", "undefined_arcsine", lambda: ((_f("asin(x)"), {X: 2}), {})),
    edge("function_evaluate", "unknown_domain_membership", lambda: ((_f("sqrt(sin(x)-2)"), {X: 1}), {})),
    edge("function_evaluate", "algebraic_binding", lambda: ((_f("log(x)"), {X: mf.sqrt(2)}), {})),
    edge("function_evaluate", "two_bindings", lambda: ((_TWO(), {X: 2, Y: mf.Rational(1, 3)}), {})),
    # partial_diff
    *_polynomial_edges("partial_diff", lambda: (X,)),
    edge("partial_diff", "variable_str", lambda: ((X ** 3 + 2 * X + 1, "x"), {})),
    edge("partial_diff", "variable_not_in_expression", lambda: ((X ** 3 + 2 * X + 1, Y), {})),
    edge("partial_diff", "variable_with_assumption",
         lambda: ((X ** 3 + 2 * X + 1, positive_symbol()), {})),
    edge("partial_diff", "order_negative", lambda: ((X ** 3 + 2 * X + 1, X, -1), {})),
    edge("partial_diff", "order_float", lambda: ((X ** 3 + 2 * X + 1, X, 1.5), {})),
    edge("partial_diff", "order_beyond_degree", lambda: ((X ** 3 + 2 * X + 1, X, 10 ** 7), {})),
    edge("partial_diff", "order_zero", lambda: ((X ** 3 + 2 * X + 1, X, 0), {})),
    edge("partial_diff", "rational_function", lambda: ((mf.rational_function(1, X + 1, variable=X), X), {})),
    edge("partial_diff", "irrational_coefficient", lambda: ((mf.sqrt(2) * X, X), {})),
    edge("partial_diff", "high_degree", lambda: ((high_degree(), X), {})),
    edge("partial_diff", "mixed_polynomial", lambda: ((X ** 2 * Y + Y ** 3 + X * Y, Y, 2), {})),
    edge("partial_diff", "dispatch_real_function", lambda: ((_f(), X), {})),
    # gradient
    *_polynomial_edges("gradient", lambda: ((X,),)),
    edge("gradient", "variables_empty", lambda: ((X ** 3 + 2 * X, ()), {})),
    edge("gradient", "variables_str", lambda: ((X ** 3 + 2 * X, "x"), {})),
    edge("gradient", "variables_list", lambda: ((X * Y, [X, Y]), {})),
    edge("gradient", "missing_free_symbol", lambda: ((X * Y, (X,)), {})),
    edge("gradient", "duplicate_variables", lambda: ((X * Y, (X, X)), {})),
    edge("gradient", "two_variables", lambda: ((X ** 2 * Y + Y ** 3, (X, Y)), {})),
    edge("gradient", "dispatch_real_function", lambda: ((_f(), (X,)), {})),
    # jacobian
    *tuple(edge("jacobian", label, lambda value=value: (((value,), (X,)), {})) for label, value in _TYPE_EDGES),
    edge("jacobian", "type_list", lambda: ((([1, 2],), (X,)), {})),
    edge("jacobian", "over_limit_integer", lambda: (((big_polynomial(),), (X,)), {})),
    edge("jacobian", "functions_empty", lambda: (((), (X,)), {})),
    edge("jacobian", "functions_str", lambda: (("ab", (X,)), {})),
    edge("jacobian", "variables_empty", lambda: (((X ** 2,), ()), {})),
    edge("jacobian", "two_functions", lambda: (((X ** 2 * Y, X + Y ** 3), (X, Y)), {})),
    edge("jacobian", "dispatch_real_function", lambda: (((_f(),), (X,)), {})),
    # hessian
    *_polynomial_edges("hessian", lambda: ((X,),)),
    edge("hessian", "variables_empty", lambda: ((X ** 3, ()), {})),
    edge("hessian", "missing_free_symbol", lambda: ((X * Y, (X,)), {})),
    edge("hessian", "two_variables", lambda: ((X ** 3 * Y + Y ** 2, (X, Y)), {})),
    edge("hessian", "dispatch_real_function", lambda: ((_f(), (X,)), {})),
    # differentiate (public name diff)
    *_polynomial_edges("differentiate", lambda: (X,), "diff"),
    edge("differentiate", "variable_str", lambda: ((X ** 3 + 2 * X + 1, "x"), {}), "diff"),
    edge("differentiate", "order_negative", lambda: ((X ** 3 + 2 * X + 1, X, -1), {}), "diff"),
    edge("differentiate", "order_float", lambda: ((X ** 3 + 2 * X + 1, X, 1.5), {}), "diff"),
    edge("differentiate", "order_beyond_degree", lambda: ((X ** 3 + 2 * X + 1, X, 10 ** 7), {}), "diff"),
    edge("differentiate", "order_zero", lambda: ((X ** 3 + 2 * X + 1, X, 0), {}), "diff"),
    edge("differentiate", "variable_mismatch_rational_function",
         lambda: ((mf.rational_function(1, X + 1, variable=X), Y), {}), "diff"),
    edge("differentiate", "second_variable", lambda: ((X * Y, X), {}), "diff"),
    edge("differentiate", "rational_function", lambda: ((mf.rational_function(1, X + 1, variable=X), X), {}), "diff"),
    edge("differentiate", "over_limit_order_rational_function",
         lambda: ((mf.rational_function(1, X + 1, variable=X), X, 10 ** 7), {}), "diff"),
    edge("differentiate", "high_degree", lambda: ((high_degree(), X), {}), "diff"),
    edge("differentiate", "polynomial_object", lambda: ((mf.Polynomial((1, -2, 0, 1), X), X, 2), {}), "diff"),
    edge("differentiate", "dispatch_real_function", lambda: ((_f(), X), {}), "diff"),
    # analyze_function
    *_polynomial_edges("analyze_function", lambda: (X,)),
    edge("analyze_function", "variable_str", lambda: ((X ** 3 + 2 * X + 1, "x"), {})),
    edge("analyze_function", "second_variable", lambda: ((X * Y, X), {})),
    edge("analyze_function", "high_degree", lambda: ((high_degree(), X), {})),
    edge("analyze_function", "real_function", lambda: ((_f(), X), {})),
    edge("analyze_function", "constant", lambda: ((mf.Rational(3), X), {})),
    edge("analyze_function", "zero", lambda: ((mf.Rational(0), X), {})),
    edge("analyze_function", "removable_singularity",
         lambda: ((mf.rational_function(X * X - 1, X - 1, variable=X), X), {})),
    # extrema_values
    *_polynomial_edges("extrema_values", lambda: (X,)),
    edge("extrema_values", "variable_str", lambda: ((X ** 3 + 2 * X + 1, "x"), {})),
    edge("extrema_values", "second_variable", lambda: ((X * Y, X), {})),
    edge("extrema_values", "high_degree", lambda: ((high_degree(), X), {})),
    edge("extrema_values", "real_function", lambda: ((_f(), X), {})),
    edge("extrema_values", "constant", lambda: ((mf.Rational(3), X), {})),
    edge("extrema_values", "flat_minimum", lambda: ((X ** 4, X), {})),
    edge("extrema_values", "no_critical_points", lambda: ((X ** 3 + X, X), {})),
)
