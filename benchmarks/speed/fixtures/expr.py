"""Speed fixtures and value-corpus edge cases of the ``expr`` area: ``evaluate``, ``substitute``, ``simplify``,
``verify``, ``evaluate_algebraic`` and ``parse_function``.

The reference rule and the helpers are those of ``benchmarks.speed.fixtures.algebra``, which this module imports:

* ``evaluate`` and ``substitute`` time ``xreplace``/``subs`` on a prebuilt SymPy expression.
* ``simplify`` receives a raw, unevaluated tree on both sides (MathForge dataclass nodes; SymPy ``evaluate=False``
  nodes), and the SymPy call must re-apply its local evaluation rules to it.
* ``evaluate_algebraic`` is pinned to ``minimal_polynomial`` (the spec), applied to the SymPy value at the point.
* ``parse_function`` times SymPy's text parsers on the same text.
* ``verify`` has no SymPy counterpart: two ExemptFixtures hold it to the time of producing the same result.
"""

from __future__ import annotations

import functools

import mathforge as mf
from benchmarks.speed.fixture import ExemptFixture, Fixture, symbol
from benchmarks.speed.fixtures import algebra as shared
from benchmarks.speed.fixtures.algebra import X, Y, Z

_MICRO = shared._MICRO
_mathforge, _sympy = shared._mathforge, shared._sympy


# ---------------------------------------------------------------------------------------------------------------
# evaluate and substitute
# ---------------------------------------------------------------------------------------------------------------

def _quadratic_point(i: int):
    """Micro variant i of evaluate: a*x**2 + b*x + c at x = n; a in [1, 9], b, c, n in [-9, 9]; distinct for
    i < 29,241."""
    a, b, c, n = shared._spread(i, 9, 19, 19, 19)
    return a + 1, b - 9, c - 9, n - 9


def _evaluate_micro_mf(i):
    a, b, c, n = _quadratic_point(i)
    return a * X ** 2 + b * X + c, {X: n}


def _evaluate_micro_sp(i):
    import sympy as sp
    a, b, c, n = _quadratic_point(i)
    x = sp.Symbol("x")
    return a * x ** 2 + b * x + c, {x: n}


_EVALUATE_POINTS = ((mf.Rational(3, 7), 5), (mf.Rational(-2, 5), 3), (mf.Rational(7, 3), -4))
_EVALUATE_BIG_POINTS = (mf.Rational(3, 7), mf.Rational(-5, 3), mf.Rational(11, 4))


def _evaluate_realistic_mf(i):
    x, y = _EVALUATE_POINTS[i]
    return (X + 1) ** 8 * (X - 3) + X * Y, {X: x, Y: y}


def _evaluate_realistic_sp(i):
    import sympy as sp
    x, y = sp.symbols("x y")
    value, other = _EVALUATE_POINTS[i]
    return (x + 1) ** 8 * (x - 3) + x * y, {x: sp.Rational(value.numerator, value.denominator), y: other}


def _evaluate_scaled_mf(i):
    return (X + 1) ** 30 * (X - 3) ** 20, {X: _EVALUATE_BIG_POINTS[i]}


def _evaluate_scaled_sp(i):
    import sympy as sp
    x = sp.Symbol("x")
    value = _EVALUATE_BIG_POINTS[i]
    return (x + 1) ** 30 * (x - 3) ** 20, {x: sp.Rational(value.numerator, value.denominator)}


def _evaluate_fixtures():
    call = lambda sp, e, bindings: e.xreplace(bindings)
    agree = shared._same_expression
    return (
        Fixture("evaluate", "micro", "quadratic_at_integer", _mathforge(mf.evaluate, _evaluate_micro_mf),
                _sympy(_evaluate_micro_sp, call), agree,
                ("used: e.xreplace(bindings) 329 us; others: e.subs(bindings) 468 us; sympify(e).subs via dict "
                 "items 428 us; medians of 150 calls, SymPy 1.14.0 (python ground types), caches cleared before "
                 "each call"), variants=_MICRO),
        Fixture("evaluate", "realistic", "two_variables_at_rationals",
                _mathforge(mf.evaluate, _evaluate_realistic_mf), _sympy(_evaluate_realistic_sp, call), agree,
                ("used: e.xreplace(bindings) 318 us; others: e.subs(bindings) 622 us; sympify(e).subs via dict "
                 "items 552 us; medians of 60 calls, SymPy 1.14.0 (python ground types), caches cleared before "
                 "each call"), variants=len(_EVALUATE_POINTS)),
        Fixture("evaluate", "realistic", "scaled_large_powers_at_rationals",
                _mathforge(mf.evaluate, _evaluate_scaled_mf), _sympy(_evaluate_scaled_sp, call), agree,
                ("used: e.xreplace(bindings) 368 us; others: e.subs(bindings) 522 us; sympify(e).subs via dict "
                 "items 509 us; medians of 60 calls, SymPy 1.14.0 (python ground types), caches cleared before "
                 "each call"), variants=len(_EVALUATE_BIG_POINTS)),
    )


def _shifted_product(i: int):
    """Micro variant i of substitute: (x + a)*(x + b) with x replaced by c*y + d; a, b, d in [-5, 5], c in [1, 5];
    distinct for i < 6,655."""
    a, b, c, d = shared._spread(i, 11, 11, 5, 11)
    return a - 5, b - 5, c + 1, d - 5


def _substitute_micro_mf(i):
    a, b, c, d = _shifted_product(i)
    return (X + a) * (X + b), {X: c * Y + d}


def _substitute_micro_sp(i):
    import sympy as sp
    a, b, c, d = _shifted_product(i)
    x, y = sp.symbols("x y")
    return (x + a) * (x + b), {x: c * y + d}


_SUBSTITUTE_REALISTIC = (1, -1, 2)  # x -> y + k in (x + 1)**8 * (x - 3)
_SUBSTITUTE_SCALED = (1, 2, 3)      # x -> y + k*z in (x + 1)**20 * (x - 3)**10 + x*z


def _substitute_realistic_mf(i):
    return (X + 1) ** 8 * (X - 3), {X: Y + _SUBSTITUTE_REALISTIC[i]}


def _substitute_realistic_sp(i):
    import sympy as sp
    x, y = sp.symbols("x y")
    return (x + 1) ** 8 * (x - 3), {x: y + _SUBSTITUTE_REALISTIC[i]}


def _substitute_scaled_mf(i):
    return (X + 1) ** 20 * (X - 3) ** 10 + X * Z, {X: Y + _SUBSTITUTE_SCALED[i] * Z}


def _substitute_scaled_sp(i):
    import sympy as sp
    x, y, z = sp.symbols("x y z")
    return (x + 1) ** 20 * (x - 3) ** 10 + x * z, {x: y + _SUBSTITUTE_SCALED[i] * z}


def _substitute_fixtures():
    call = lambda sp, e, bindings: e.xreplace(bindings)
    agree = shared._same_expression
    return (
        Fixture("substitute", "micro", "product_of_shifted_x", _mathforge(mf.substitute, _substitute_micro_mf),
                _sympy(_substitute_micro_sp, call), agree,
                ("used: e.xreplace(bindings) 198 us; others: e.subs(bindings) 341 us; medians of 150 calls, SymPy "
                 "1.14.0 (python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("substitute", "realistic", "binomial_powers_for_a_shifted_y",
                _mathforge(mf.substitute, _substitute_realistic_mf), _sympy(_substitute_realistic_sp, call), agree,
                ("used: e.xreplace(bindings) 196 us; others: e.subs(bindings) 325 us; medians of 60 calls, SymPy "
                 "1.14.0 (python ground types), caches cleared before each call"), variants=len(_SUBSTITUTE_REALISTIC)),
        Fixture("substitute", "realistic", "scaled_powers_for_a_sum_of_two_symbols",
                _mathforge(mf.substitute, _substitute_scaled_mf), _sympy(_substitute_scaled_sp, call), agree,
                ("used: e.xreplace(bindings) 337 us; others: e.subs(bindings) 527 us; medians of 60 calls, SymPy "
                 "1.14.0 (python ground types), caches cleared before each call"), variants=len(_SUBSTITUTE_SCALED)),
    )


# ---------------------------------------------------------------------------------------------------------------
# simplify: raw trees. A term is (coefficient, inner exponent of x, outer exponent of x, exponent of y).
# ---------------------------------------------------------------------------------------------------------------

def _raw_terms(tag: int, i: int, count: int):
    """``count`` raw terms c * (x**e1)**e2 * y**f of variant i, with c in [-9, 9] but not 0, e1 and e2 in [1, 2] and
    f in [1, 2]; they collect into a polynomial in x and y with fewer terms."""
    terms = []
    for k in range(count):
        c = shared._draw(1, 9, tag, i, k, 0) * (-1 if shared._draw(0, 1, tag, i, k, 1) else 1)
        terms.append((c, shared._draw(1, 2, tag, i, k, 2), shared._draw(1, 2, tag, i, k, 3),
                      shared._draw(1, 2, tag, i, k, 4)))
    return tuple(terms)


def _raw_mf(terms):
    """The MathForge tree built from dataclass nodes only, so that nothing in it is collected yet."""
    nodes = tuple(mf.Mul((mf.Rational(c), mf.Pow(mf.Pow(X, e1), e2), mf.Pow(Y, f))) for c, e1, e2, f in terms)
    return mf.Add(nodes)


def _raw_sp(terms):
    import sympy as sp
    x, y = sp.symbols("x y")
    nodes = [sp.Mul(sp.Integer(c), sp.Pow(sp.Pow(x, e1, evaluate=False), e2, evaluate=False),
                    sp.Pow(y, f, evaluate=False), evaluate=False) for c, e1, e2, f in terms]
    return sp.Add(*nodes, evaluate=False)


def _like_terms(i: int):
    """Micro variant i of simplify: a*x + b*x as raw nodes; a and b in [-20, 20]; distinct for i < 1,681."""
    a, b = shared._spread(i, 41, 41)
    return a - 20, b - 20


def _same_collected(mf_value, sympy_value) -> bool:
    """The simplified MathForge expression and SymPy's value are the same collected expression: equal, and equal
    in SymPy's canonical form (a call that left like terms or nested powers alone does not match)."""
    return (shared._same_expression(mf_value, sympy_value)
            and shared._sp_expr(mf_value, shared._sp_symbols()) == sympy_value)


def _simplify_micro_mf(i):
    a, b = _like_terms(i)
    return (mf.Add((mf.Mul((mf.Rational(a), X)), mf.Mul((mf.Rational(b), X)))),)


def _simplify_micro_sp(i):
    import sympy as sp
    a, b = _like_terms(i)
    x = sp.Symbol("x")
    return (sp.Add(sp.Mul(sp.Integer(a), x, evaluate=False), sp.Mul(sp.Integer(b), x, evaluate=False),
                   evaluate=False),)


def _simplify_fixtures():
    call = lambda sp, e: e.doit()
    agree = _same_collected
    return (
        Fixture("simplify", "micro", "two_like_terms", _mathforge(mf.simplify, _simplify_micro_mf),
                _sympy(_simplify_micro_sp, call), agree,
                ("used: e.doit() 337 us; others: simplify(e) 3.72 ms; expand(e) 381 us (does not give the same "
                 "object); e.func(*e.args) [one level] 159 us (excluded: rebuilds the top node only); medians of "
                 "150 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"), variants=_MICRO),
        Fixture("simplify", "realistic", "twelve_raw_terms",
                _mathforge(mf.simplify, lambda i: (_raw_mf(_raw_terms(81, i, 12)),)),
                _sympy(lambda i: (_raw_sp(_raw_terms(81, i, 12)),), call), agree,
                ("used: e.doit() 1.19 ms; others: simplify(e) 32.27 ms (does not give the same object); expand(e) "
                 "2.23 ms; e.func(*e.args) [one level] 297 us (excluded: rebuilds the top node only); medians of "
                 "60 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=3),
        Fixture("simplify", "realistic", "scaled_sixty_raw_terms",
                _mathforge(mf.simplify, lambda i: (_raw_mf(_raw_terms(85, i, 60)),)),
                _sympy(lambda i: (_raw_sp(_raw_terms(85, i, 60)),), call), agree,
                ("used: e.doit() 1.97 ms; others: simplify(e) 32.27 ms (does not give the same object); expand(e) "
                 "5.73 ms; e.func(*e.args) [one level] 581 us (excluded: rebuilds the top node only); medians of "
                 "60 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"), variants=3),
    )


# ---------------------------------------------------------------------------------------------------------------
# evaluate_algebraic
# ---------------------------------------------------------------------------------------------------------------

def _linear_at_root(i: int):
    """Micro variant i of evaluate_algebraic: a*x + b at x = sqrt(s); a in [1, 9], b in [-9, 9], s one of 20
    non-squares; distinct for i < 3,420."""
    a, b, s = shared._spread(i, 9, 19, 20)
    return a + 1, b - 9, shared._NON_SQUARES[s]


def _evaluate_algebraic_micro_mf(i):
    a, b, s = _linear_at_root(i)
    return a * X + b, {X: mf.sqrt(s)}


def _evaluate_algebraic_micro_sp(i):
    import sympy as sp
    a, b, s = _linear_at_root(i)
    x = sp.Symbol("x")
    return a * x + b, {x: sp.sqrt(s)}, sp.Symbol("t")


def _algebraic_eval_mf(i):
    return mf.rational_function(X + 1, X * X + 1, variable=X), {X: mf.sqrt(2)}


def _algebraic_eval_sp(i):
    import sympy as sp
    x = sp.Symbol("x")
    return (x + 1) / (x * x + 1), {x: sp.sqrt(2)}, sp.Symbol("t")


_ROOT_SUM_POINTS = ((2, 3), (3, 5), (5, 7))  # x**2 - x at x = sqrt(p) + sqrt(q)


def _scaled_evaluate_mf(i):
    p, q = _ROOT_SUM_POINTS[i]
    return X ** 2 - X, {X: mf.sqrt(p) + mf.sqrt(q)}


def _scaled_evaluate_sp(i):
    import sympy as sp
    p, q = _ROOT_SUM_POINTS[i]
    x = sp.Symbol("x")
    return x ** 2 - x, {x: sp.sqrt(p) + sp.sqrt(q)}, sp.Symbol("t")


def _evaluated_minpoly(sp, expression, bindings, t):
    value = expression.xreplace(bindings)
    return sp.minimal_polynomial(value, t, compose=False), value


def _evaluate_algebraic_fixtures():
    agree = shared._same_algebraic
    return (
        Fixture("evaluate_algebraic", "micro", "linear_at_square_root",
                _mathforge(mf.evaluate_algebraic, _evaluate_algebraic_micro_mf),
                _sympy(_evaluate_algebraic_micro_sp, _evaluated_minpoly), agree,
                ("used: e.xreplace, minimal_polynomial(compose=False) 1.98 ms; others: e.subs, "
                 "minimal_polynomial(compose=False) 3.66 ms; e.xreplace, minimal_polynomial 4.54 ms; medians of "
                 "150 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"), variants=_MICRO),
        Fixture("evaluate_algebraic", "realistic", "rational_function_at_square_root_of_2",
                _mathforge(mf.evaluate_algebraic, _algebraic_eval_mf), _sympy(_algebraic_eval_sp, _evaluated_minpoly),
                agree,
                ("used: e.xreplace, minimal_polynomial(compose=False) 3.85 ms; others: e.subs, "
                 "minimal_polynomial(compose=False) 4.15 ms; e.xreplace, minimal_polynomial 4.96 ms; medians of 9 "
                 "calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                v07="algebraic_eval"),
        Fixture("evaluate_algebraic", "realistic", "scaled_polynomial_at_sum_of_square_roots",
                _mathforge(mf.evaluate_algebraic, _scaled_evaluate_mf), _sympy(_scaled_evaluate_sp, _evaluated_minpoly),
                agree,
                ("used: e.xreplace, minimal_polynomial(compose=False) 10.08 ms; others: e.subs, "
                 "minimal_polynomial(compose=False) 10.14 ms; e.xreplace, minimal_polynomial 11.80 ms; medians of "
                 "9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_ROOT_SUM_POINTS)),
    )


# ---------------------------------------------------------------------------------------------------------------
# parse_function
# ---------------------------------------------------------------------------------------------------------------

_FUNCTIONS = {"sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
              "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "exp": "exp", "log": "log", "sqrt": "sqrt",
              "abs": "Abs"}


def _sp_function(value, symbols):
    """A MathForge RealExpr as a SymPy expression; ``symbols`` maps MathForge Symbols to SymPy ones."""
    import sympy as sp
    op = value.op
    if op == "scalar":
        return shared._sp_real(value.value)
    if op == "symbol":
        return symbols[value.value]
    if op == "pi":
        return sp.pi
    if op == "E":
        return sp.E
    args = [_sp_function(argument, symbols) for argument in value.args]
    if op == "pow":
        return args[0] ** value.value
    if op == "add":
        return args[0] + args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        return args[0] / args[1]
    return getattr(sp, _FUNCTIONS[op])(*args)


def _same_function(mf_value, sympy_value) -> bool:
    """A RealFunction and a SymPy expression are the same function of x (equal, else simplifying to zero)."""
    import sympy as sp
    if type(mf_value) is not mf.RealFunction:
        return False
    expected = _sp_function(mf_value.expression, {X: sp.Symbol("x")})
    return expected == sympy_value or sp.simplify(expected - sympy_value) == 0


def _polynomial_text(i: int) -> str:
    """Micro variant i of parse_function: ``a*x**2+b*x+c`` with a in [1, 9] and b, c in [-9, 9]; distinct for
    i < 3,249."""
    a, b, c = shared._spread(i, 9, 19, 19)
    return f"{a + 1}*x**2{b - 9:+d}*x{c - 9:+d}"


_ELEMENTARY_TEXTS = ("exp(x)*sin(2*x)+cos(x*x)", "exp(x)*sin(x)+log(1+x)", "(x**3-2*x+1)*exp(2*x)", "1/(1+x*x)**2")


def _long_text(i: int, terms: int = 16) -> str:
    """Scaled variant i: a sum of ``terms`` terms c*x**k*f(m*x) with f one of sin, cos, exp."""
    parts = []
    for k in range(terms):
        c = shared._draw(1, 9, 91, i, k, 0)
        sign = "-" if shared._draw(0, 1, 91, i, k, 1) else "+"
        function = ("sin", "cos", "exp")[shared._draw(0, 2, 91, i, k, 2)]
        parts.append(f"{sign}{c}*x**{shared._draw(1, 4, 91, i, k, 3)}*{function}({shared._draw(1, 5, 91, i, k, 4)}*x)")
    return "".join(parts).lstrip("+")


def _parse_inputs_sp(text_of):
    def build(i):
        import sympy as sp
        return text_of(i), {"x": sp.Symbol("x")}
    return build


def _parse_mf(text_of):
    return lambda i: (mf.parse_function, (text_of(i),), {"symbols": {"x": X}})


def _parse_function_fixtures():
    call = lambda sp, text, names: sp.parse_expr(text, local_dict=names)
    return (
        Fixture("parse_function", "micro", "quadratic_text", _parse_mf(_polynomial_text),
                _sympy(_parse_inputs_sp(_polynomial_text), call), _same_function,
                ("used: parse_expr(text, local_dict) 628 us; others: sympify(text, locals) 635 us; sympify(text) "
                 "647 us; parse_expr(text, local_dict, transformations=()) 598 us (excluded: no number conversion: "
                 "decimals would become floats); medians of 150 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"), variants=_MICRO),
        Fixture("parse_function", "realistic", "elementary_functions",
                _parse_mf(lambda i: _ELEMENTARY_TEXTS[i]),
                _sympy(_parse_inputs_sp(lambda i: _ELEMENTARY_TEXTS[i]), call), _same_function,
                ("used: parse_expr(text, local_dict) 1.22 ms; others: sympify(text, locals) 1.29 ms; sympify(text) "
                 "1.32 ms; parse_expr(text, local_dict, transformations=()) 1.33 ms (excluded: no number "
                 "conversion: decimals would become floats); medians of 9 calls, SymPy 1.14.0 (python ground "
                 "types), caches cleared before each call"), variants=len(_ELEMENTARY_TEXTS)),
        Fixture("parse_function", "realistic", "scaled_sixteen_terms", _parse_mf(_long_text),
                _sympy(_parse_inputs_sp(_long_text), call), _same_function,
                ("used: parse_expr(text, local_dict) 4.27 ms; others: sympify(text, locals) 4.30 ms; first pass, an "
                 "earlier machine state: sympify(text) 4.71 ms; parse_expr(text, local_dict, transformations=()) 4.29"
                 " ms (excluded: no number conversion: decimals would become floats); medians of 127 samples of k = 1"
                 " calls (the suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types);"
                 " excluded candidates keep their first-pass medians"), variants=3),
    )


# ---------------------------------------------------------------------------------------------------------------
# verify: no SymPy counterpart, so its time is held to the time of producing the same result
# ---------------------------------------------------------------------------------------------------------------

@functools.cache
def _gcd_result():
    return mf.operations.integer_gcd(84, 30)


@functools.cache
def _rref16():
    """The v0.7 rref16 case: A[i, i] = 17, A[i, i +- 1] = 1, every other entry 0; its Matrix and the rref Result."""
    rows = tuple(tuple(17 if i == j else int(abs(i - j) == 1) for j in range(16)) for i in range(16))
    matrix = mf.Matrix(rows)
    return matrix, mf.operations.rref(matrix)


_VERIFY_REASON = ("verify rechecks a Result and has no SymPy counterpart; it is held to be no slower than producing "
                  "the same result through mathforge.operations")

VERIFY_FIXTURES = (
    ExemptFixture("verify", "micro", "integer_gcd_84_30_result",
                  lambda: (mf.verify, (_gcd_result(),), {}), lambda: (mf.operations.integer_gcd, (84, 30), {}),
                  _VERIFY_REASON),
    ExemptFixture("verify", "realistic", "rref16_result",
                  lambda: (mf.verify, (_rref16()[1],), {}), lambda: (mf.operations.rref, (_rref16()[0],), {}),
                  _VERIFY_REASON),
)

FIXTURES = (
    *_evaluate_fixtures(),
    *_substitute_fixtures(),
    *_simplify_fixtures(),
    *VERIFY_FIXTURES,
    *_evaluate_algebraic_fixtures(),
    *_parse_function_fixtures(),
)


# ---------------------------------------------------------------------------------------------------------------
# Edge cases (see benchmarks.speed.fixtures.algebra for the conventions)
# ---------------------------------------------------------------------------------------------------------------

_edge, _type_edges, _repeated_sum = shared._edge, shared._type_edges, shared._repeated_sum


def _integer_symbol():
    return mf.Symbol("n", mf.Integers, uid="00000000-0000-0000-0000-0000000000b1")


def _positive_symbol():
    return mf.Symbol("p", mf.Reals, (mf.Assumption("positive", mf.Truth.TRUE),),
                     uid="00000000-0000-0000-0000-0000000000b2")


def _reciprocal_pole():
    return mf.rational_function(1, X * X - 2, variable=X)


def _binding_edges(function, rest, extra=()):
    """Edges shared by the operations with an expression and a bindings mapping."""
    name = function.__name__
    return (
        *_type_edges(name, function, rest),
        _edge(name, "missing_binding", function, lambda: (X + Y, {X: 1})),
        _edge(name, "bindings_not_a_mapping", function, lambda: (X + 1, [1])),
        _edge(name, "binding_keyed_by_name", function, lambda: (X + 1, {"x": 1})),
        _edge(name, "binding_value_float", function, lambda: (X + 1, {X: 1.5})),
        _edge(name, "over_limit_integer", function, lambda: (shared._HUGE * X, {X: 1})),
        _edge(name, "over_limit_nodes", function, lambda: (_repeated_sum(), {X: 1})),
        *extra,
    )


EDGE_CASES = (
    *_binding_edges(mf.evaluate, lambda: ({X: 1},), extra=(
        _edge("evaluate", "no_bindings", mf.evaluate, lambda: (X + 1, None)),
        _edge("evaluate", "value_outside_the_symbol_domain", mf.evaluate,
              lambda: (_integer_symbol() + 1, {_integer_symbol(): mf.Rational(1, 2)})),
    )),
    *_binding_edges(mf.substitute, lambda: ({X: 1},), extra=(
        _edge("substitute", "no_bindings", mf.substitute, lambda: (X + 1, None)),
        _edge("substitute", "value_violating_an_assumption", mf.substitute,
              lambda: (_positive_symbol() + 1, {_positive_symbol(): -1})),
    )),
    *_type_edges("simplify", mf.simplify),
    _edge("simplify", "over_limit_integer", mf.simplify, lambda: (shared._HUGE * X,)),
    _edge("simplify", "over_limit_nodes", mf.simplify, lambda: (_repeated_sum(),)),
    *_type_edges("evaluate_algebraic", mf.evaluate_algebraic, lambda: ({X: mf.sqrt(2)},)),
    _edge("evaluate_algebraic", "missing_binding", mf.evaluate_algebraic, lambda: (X + Y, {X: mf.sqrt(2)})),
    _edge("evaluate_algebraic", "no_bindings", mf.evaluate_algebraic, lambda: (X + 1, None)),
    _edge("evaluate_algebraic", "bindings_not_a_mapping", mf.evaluate_algebraic, lambda: (X + 1, [1])),
    _edge("evaluate_algebraic", "rational_function_variable_unbound", mf.evaluate_algebraic,
          lambda: (mf.rational_function(X + 1, X * X + 1, variable=X), {Y: 1})),
    _edge("evaluate_algebraic", "pole_at_the_point", mf.evaluate_algebraic,
          lambda: (_reciprocal_pole(), {X: mf.sqrt(2)})),
    _edge("evaluate_algebraic", "over_limit_integer", mf.evaluate_algebraic,
          lambda: (shared._HUGE * X, {X: mf.sqrt(2)})),
    _edge("evaluate_algebraic", "over_limit_nodes", mf.evaluate_algebraic,
          lambda: (_repeated_sum(), {X: mf.sqrt(2)})),
    *_type_edges("parse_function", mf.parse_function, (), lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "blank_text", mf.parse_function, ("  ",), lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "undeclared_symbol", mf.parse_function, ("y",), lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "syntax_error", mf.parse_function, ("x+",), lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "unknown_function", mf.parse_function, ("foo(x)",), lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "reserved_name", mf.parse_function, ("x",),
          lambda: {"symbols": {"sin": symbol("sin", 9)}}),
    _edge("parse_function", "symbols_not_a_mapping", mf.parse_function, ("x",), lambda: {"symbols": [X]}),
    _edge("parse_function", "symbolic_exponent", mf.parse_function, ("x**x",), lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "fractional_exponent", mf.parse_function, ("x**0.5",), lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "numeric_literal_past_the_limit", mf.parse_function, ("1e60000",),
          lambda: {"symbols": {"x": X}}),
    _edge("parse_function", "over_limit_text", mf.parse_function, lambda: ("x+" * 100_001 + "x",),
          lambda: {"symbols": {"x": X}}),
)
