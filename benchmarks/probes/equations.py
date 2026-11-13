"""Equation probes: polynomial, radical, absolute-value and transcendental equations, systems and inequalities."""

from . import Probe, algebraic_roots, exact_tuple, intervals, planned, point_set, symbols

GROUP = "equations"


def _solve(text):
    """The attempt of the real solver on one equation or inequality in x (a and y may occur too)."""
    def run(mf):
        x, y, a = symbols(mf, "x y a")
        return mf.solve(mf.parse_problem(text, symbols={"x": x, "y": y, "a": a}), for_=x)
    return run


def _system(*texts):
    def run(mf):
        x, y, z = symbols(mf, "x y z")
        names = {"x": x, "y": y, "z": z}
        return mf.solve_system([mf.parse_problem(text, symbols=names) for text in texts], for_=(x, y, z)[:len(texts)])
    return run


def _sides(mf, lhs, rhs):
    """The two sides of an equation in x as functions, and x.

    The catalogue (section 2.3) names the arguments lhs, rhs, variable and relation but not the type of the sides:
    parse_function's RealFunction is assumed, since an Expression holds neither sqrt(x + 2) nor |x - 1|.
    """
    x, = symbols(mf, "x")
    return mf.parse_function(lhs, symbols={"x": x}), mf.parse_function(rhs, symbols={"x": x}), x


def _solve_sqrt(lhs, rhs):
    def run(mf):
        solve = mf.equations.solve_sqrt
        return solve(*_sides(mf, lhs, rhs), "=")
    return run


def _solve_abs(lhs, rhs):
    def run(mf):
        solve = mf.equations.solve_abs
        return solve(*_sides(mf, lhs, rhs), "=")
    return run


def _solve_planned(name, lhs, rhs):
    """An extended equation family whose arguments the catalogue leaves open: solve_abs's are assumed."""
    def run(mf):
        solve = planned(mf, name, "equations")
        return solve(*_sides(mf, lhs, rhs), "=")
    return run


def _cubic_roots(value):
    return algebraic_roots(value) == {((1, -3, 0, 1), index) for index in range(3)}


def _quartic_roots(value):
    return algebraic_roots(value) == {((1, 0, -10, 0, 1), index) for index in range(4)}


PROBES = (
    Probe("equations_quadratic", GROUP, "Solve x^2 - 5x + 6 = 0.", _solve("x^2 - 5*x + 6 = 0"),
          lambda v: point_set(v) == {2, 3}),
    Probe("equations_cubic_three_real_roots", GROUP, "Solve x^3 - 3x + 1 = 0.", _solve("x^3 - 3*x + 1 = 0"),
          _cubic_roots),
    Probe("equations_quartic", GROUP, "Solve x^4 - 10x^2 + 1 = 0.", _solve("x^4 - 10*x^2 + 1 = 0"), _quartic_roots),
    Probe("equations_quintic", GROUP, "Solve x^5 - x - 1 = 0, which has no rational root.", _solve("x^5 - x - 1 = 0"),
          lambda v: algebraic_roots(v) == {((-1, -1, 0, 0, 0, 1), 0)}),
    Probe("equations_sqrt_equation", GROUP, "Solve sqrt(x + 2) = x.", _solve_sqrt("sqrt(x + 2)", "x"),
          lambda v: point_set(v) == {2}),
    Probe("equations_abs_equation", GROUP, "Solve |x - 1| = 3.", _solve_abs("abs(x - 1)", "3"),
          lambda v: point_set(v) == {-2, 4}),
    Probe("equations_abs_interval_solution", GROUP, "Solve |x - 1| + |x + 2| = 3.",
          _solve_abs("abs(x - 1) + abs(x + 2)", "3"), lambda v: intervals(v) == ((-2, 1, True, True),)),
    # parse_function rewrites sqrt(x^2) as abs(x), so the parsed equation is one for solve_abs.
    Probe("equations_sqrt_identity", GROUP, "Solve sqrt(x^2) - x = 0.", _solve_abs("sqrt(x^2) - x", "0"),
          lambda v: intervals(v) == ((0, None, True, False),)),
    Probe("equations_exponential", GROUP, "Solve 2^x = 8.", _solve_planned("solve_exponential", "2^x", "8"),
          lambda v: point_set(v) == {3}),
    Probe("equations_exp_equals_three", GROUP, "Solve exp(x) = 3.", _solve("exp(x) = 3")),
    Probe("equations_logarithm", GROUP, "Solve log(x) = 1.", _solve("log(x) = 1")),
    Probe("equations_sine_zero", GROUP, "Solve sin(x) = 0.", _solve_planned("solve_trig", "sin(x)", "0")),
    Probe("equations_linear_system_3x3", GROUP, "Solve x + y + z = 6, 2x - y + z = 3, x + 2y - z = 2.",
          _system("x + y + z = 6", "2*x - y + z = 3", "x + 2*y - z = 2"),
          lambda v: exact_tuple(v.particular) == (1, 2, 3) and not v.basis),
    Probe("equations_circle_and_line", GROUP, "Solve x^2 + y^2 = 1 together with x = y.",
          _system("x^2 + y^2 = 1", "x - y = 0")),
    Probe("equations_quadratic_inequality", GROUP, "Solve x^2 - 4 < 0.", _solve("x^2 - 4 < 0"),
          lambda v: intervals(v) == ((-2, 2, False, False),)),
    Probe("equations_rational_inequality", GROUP, "Solve (x - 1)/(x + 2) >= 0.", _solve("(x - 1)/(x + 2) >= 0"),
          lambda v: intervals(v) == ((None, -2, False, False), (1, None, True, False))),
    Probe("equations_parametric_quadratic", GROUP, "Solve x^2 - a = 0 for x.", _solve("x^2 - a = 0")),
    Probe("equations_parametric_linear", GROUP, "Solve a x = 1 for x.", _solve("a*x = 1")),
    Probe("equations_no_real_solution", GROUP, "Solve x^2 + 1 = 0 over the reals.", _solve("x^2 + 1 = 0"),
          lambda v: point_set(v) == frozenset()),
)
