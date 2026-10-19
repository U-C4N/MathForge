"""Speed fixtures and value-corpus edge cases of the ``equations`` area: ``solve`` and ``solve_system``.

The reference rule and the helpers are those of ``benchmarks.speed.fixtures.algebra``, which this module imports.

``mathforge.solve`` and ``mathforge.solve_system`` return the structured ``Result`` of v0.7. The fixtures call them
through the functions ``solve`` and ``solve_system`` below, which take the solution set out of the Result
(``Result.unwrap``: an OperationError with the Result's code when the call failed). The timed work is that of the public
call, and the corpus stores the solution set rather than the Result, whose evidence a later version may change (the
plan supersedes ``solve_system``'s v2 evidence) while the set stays the same.

* ``solve`` is an equation or inequality over the reals; the SymPy call must give the real solution set, so a
  complex-domain ``roots`` is no reference. For a polynomial equation the fastest call is ``Poly(lhs - rhs, x)``
  ``.real_roots()``; for a rational inequality it is ``solve_univariate_inequality``.
* ``solve_system`` is a rational linear system; the SymPy call must give the same affine solution set (a unique point,
  or a point and directions): ``DomainMatrix.lu_solve`` for a unique solution, ``Matrix.gauss_jordan_solve`` otherwise.
"""

from __future__ import annotations

import functools

import mathforge as mf
from benchmarks.speed.fixture import Fixture, symbol
from benchmarks.speed.fixtures import algebra as shared
from benchmarks.speed.fixtures.algebra import X, Y

_MICRO = shared._MICRO
_mathforge, _sympy = shared._mathforge, shared._sympy


def solve(problem, *, for_):
    """The solution set of mathforge.solve(problem, for_=for_); OperationError with the Result's code on a failure."""
    return mf.solve(problem, for_=for_).unwrap()


def solve_system(system_or_equations, *, for_=None):
    """The solution set of mathforge.solve_system; OperationError with the Result's code on a failure."""
    return mf.solve_system(system_or_equations, for_=for_).unwrap()


# ---------------------------------------------------------------------------------------------------------------
# Comparing MathForge solution sets with SymPy values (not timed)
# ---------------------------------------------------------------------------------------------------------------

def _sympy_points(sympy_value):
    """The distinct points of a SymPy solution given as a FiniteSet, a list (roots repeated by multiplicity), a tuple
    or a dict; None for anything else."""
    import sympy as sp
    if isinstance(sympy_value, sp.FiniteSet):
        return list(sympy_value.args)
    if isinstance(sympy_value, dict):
        return list(sympy_value)
    if isinstance(sympy_value, (list, tuple)):
        distinct = []
        for point in sympy_value:
            if not any(shared._same_number(point, known) for known in distinct):
                distinct.append(point)
        return distinct
    return None


def _intervals_of(sympy_value):
    """(start, end, left_open, right_open) of every interval of a SymPy Interval or Union; None if it is not one."""
    import sympy as sp
    if sympy_value == sp.S.Reals:
        return [(-sp.oo, sp.oo, True, True)]
    if isinstance(sympy_value, sp.Interval):
        return [(sympy_value.start, sympy_value.end, sympy_value.left_open, sympy_value.right_open)]
    if isinstance(sympy_value, sp.Union) and all(isinstance(part, sp.Interval) for part in sympy_value.args):
        return sorted((part.start, part.end, part.left_open, part.right_open) for part in sympy_value.args)
    return None


def _same_end(a, b) -> bool:
    import sympy as sp
    if a in (sp.oo, -sp.oo) or b in (sp.oo, -sp.oo):
        return a == b
    return shared._same_number(a, b)


def _same_solution(mf_value, sympy_value) -> bool:
    """The solution set of solve and a SymPy solution set (or list of solutions) describe the same real set."""
    import sympy as sp
    value = mf_value
    if type(value) is mf.EmptySet:
        return sympy_value == sp.S.EmptySet or (isinstance(sympy_value, (list, tuple)) and not sympy_value)
    if type(value) is mf.UniversalSet:
        return sympy_value == sp.S.Reals
    if type(value) is mf.IntervalSet:
        found = _intervals_of(sympy_value)
        if found is None or len(found) != len(value.intervals):
            return False
        for own, (start, end, left_open, right_open) in zip(value.intervals, found):
            lower = -sp.oo if own.lower is None else shared._sp_real(own.lower)
            upper = sp.oo if own.upper is None else shared._sp_real(own.upper)
            if not (_same_end(lower, start) and _same_end(upper, end) and own.left_closed == (not left_open)
                    and own.right_closed == (not right_open)):
                return False
        return True
    if type(value) is mf.FiniteSet:
        points = _sympy_points(sympy_value)
        if points is None or len(points) != len(value.values):
            return False
        mine = sorted((shared._sp_expr(v, {}) for v in value.values), key=lambda v: sp.N(v, 30))
        theirs = sorted(points, key=lambda v: sp.N(v, 30))
        return all(shared._same_number(a, b) for a, b in zip(mine, theirs))
    return False


def _sympy_affine(sympy_value):
    """(particular point, [directions]) of a SymPy linear solution as lists of numbers; None if there is no solution.

    A solution is a column Matrix or DomainMatrix (unique), a FiniteSet holding one tuple of expressions in the free
    symbols (linsolve), a dict {symbol: expression}, or the pair (solution, parameters) that gauss_jordan_solve
    returns.
    """
    import sympy as sp
    if hasattr(sympy_value, "to_Matrix") and not isinstance(sympy_value, sp.MatrixBase):
        sympy_value = sympy_value.to_Matrix()
    parameters = ()
    if isinstance(sympy_value, tuple) and len(sympy_value) == 2 and isinstance(sympy_value[0], sp.MatrixBase):
        sympy_value, parameters = sympy_value[0], tuple(sympy_value[1])
    if isinstance(sympy_value, dict):
        keys = sorted(sympy_value, key=str)
        entries = [sympy_value[key] for key in keys]
        parameters = tuple(sorted(set().union(*(e.free_symbols for e in entries)) - set(keys), key=str))
    elif isinstance(sympy_value, sp.MatrixBase):
        entries = list(sympy_value)
    elif isinstance(sympy_value, sp.FiniteSet):
        if len(sympy_value) != 1:
            return None
        entries = list(sympy_value.args[0])
        parameters = tuple(sorted(set().union(*(e.free_symbols for e in entries)), key=str))
    else:
        return None
    origin = {p: 0 for p in parameters}
    return ([e.subs(origin) for e in entries],
            [[sp.diff(e, p) for e in entries] for p in parameters])


def _same_affine(mf_value, sympy_value) -> bool:
    """The solution set of solve_system and a SymPy solution describe the same affine set of points."""
    import sympy as sp
    value = mf_value
    found = _sympy_affine(sympy_value)
    if type(value) is mf.EmptySet:
        return found is None
    if type(value) is not mf.AffineSolutionSet or found is None:
        return False
    point, directions = found
    particular = [sp.Rational(c.numerator, c.denominator) for c in value.particular]
    basis = [[sp.Rational(c.numerator, c.denominator) for c in vector] for vector in value.basis]
    if len(point) != len(particular) or len(directions) != len(basis):
        return False
    width = len(particular)

    def rank(vectors):
        return sp.Matrix(vectors).rank() if vectors else 0

    spans_equal = rank(basis) == rank(directions) == rank(basis + directions) == len(basis)
    offset = [a - b for a, b in zip(point, particular)]
    return spans_equal and rank(basis + [offset]) == rank(basis) and width > 0


# ---------------------------------------------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------------------------------------------

def _solve_mf(problem_of):
    return lambda i: (solve, (problem_of(i),), {"for_": X})


def _solve_sp(problem_of):
    def build(i):
        import sympy as sp
        return problem_of(i, sp), sp.Symbol("x", real=True)
    return build


def _roots_quadratic(i: int):
    """Micro variant i of solve: a*(x - r)*(x - s) = 0 with a in [1, 6], r in [-9, 9], s > r; distinct for i < 1,026."""
    a, r, d = shared._spread(i, 6, 19, 9)
    r -= 9
    return a + 1, r, r + 1 + d


def _quadratic_equation_mf(i):
    a, r, s = _roots_quadratic(i)
    return mf.Eq(a * X ** 2 - a * (r + s) * X + a * r * s, 0)


def _quadratic_equation_sp(i, sp):
    a, r, s = _roots_quadratic(i)
    x = sp.Symbol("x", real=True)
    return sp.Eq(a * x ** 2 - a * (r + s) * x + a * r * s, 0)


_CUBIC_ROOTS = ((1, 2, 3), (-2, 1, 4), (-3, 0, 5))
_QUINTICS = ((-1, -1), (-3, 1), (2, -5))  # x**5 + a*x + b = 0
_DEGREE_TEN = ((3, 2), (4, 1), (2, 3))    # x**10 - a*x**5 + x - b = 0
_INEQUALITIES = ((1, 1, 2, 3), (2, -1, 1, 4), (-3, 0, 2, 5))  # (x + c)/((x - a)(x - b)(x - d)) >= 0 as (c, a, b, d)


def _cubic_mf(i):
    a, b, c = _CUBIC_ROOTS[i]
    return mf.Eq((X - a) * (X - b) * (X - c), 0)


def _cubic_sp(i, sp):
    a, b, c = _CUBIC_ROOTS[i]
    x = sp.Symbol("x", real=True)
    return sp.Eq((x - a) * (x - b) * (x - c), 0)


def _quintic_mf(i):
    a, b = _QUINTICS[i]
    return mf.Eq(X ** 5 + a * X + b, 0)


def _quintic_sp(i, sp):
    a, b = _QUINTICS[i]
    x = sp.Symbol("x", real=True)
    return sp.Eq(x ** 5 + a * x + b, 0)


def _degree_ten_mf(i):
    a, b = _DEGREE_TEN[i]
    return mf.Eq(X ** 10 - a * X ** 5 + X - b, 0)


def _degree_ten_sp(i, sp):
    a, b = _DEGREE_TEN[i]
    x = sp.Symbol("x", real=True)
    return sp.Eq(x ** 10 - a * x ** 5 + x - b, 0)


def _inequality_mf(i):
    c, a, b, d = _INEQUALITIES[i]
    function = mf.rational_function(X + c, (X - a) * (X - b) * (X - d), variable=X)
    return mf.Inequality(function, "ge", 0)


def _inequality_sp(i, sp):
    c, a, b, d = _INEQUALITIES[i]
    x = sp.Symbol("x", real=True)
    return sp.Ge((x + c) / ((x - a) * (x - b) * (x - d)), 0)


def _solve_fixtures():
    polynomial = lambda sp, problem, x: sp.Poly(problem.lhs - problem.rhs, x).real_roots()
    inequality = lambda sp, problem, x: sp.solve_univariate_inequality(problem, x, relational=False)
    return (
        Fixture("solve", "micro", "quadratic_with_integer_roots", _solve_mf(_quadratic_equation_mf),
                _sympy(_solve_sp(_quadratic_equation_sp), polynomial), _same_solution,
                ("used: Poly(lhs - rhs, x).real_roots() 1.60 ms; others: solveset(problem, x, Reals) 5.85 ms; "
                 "solve(problem, x) 7.43 ms; real_roots(lhs - rhs, multiple=False) 1.69 ms; "
                 "solve_univariate_inequality(problem, x, relational=False) 43.15 ms; roots(lhs - rhs, x) 1.77 ms "
                 "(excluded: also returns the non-real roots); medians of 150 calls, SymPy 1.14.0 (python ground "
                 "types), caches cleared before each call"), variants=_MICRO),
        Fixture("solve", "realistic", "cubic_with_integer_roots", _solve_mf(_cubic_mf),
                _sympy(_solve_sp(_cubic_sp), polynomial), _same_solution,
                ("used: Poly(lhs - rhs, x).real_roots() 2.90 ms; others: solveset(problem, x, Reals) 8.22 ms; "
                 "solve(problem, x) 9.79 ms; real_roots(lhs - rhs, multiple=False) 2.84 ms; "
                 "solve_univariate_inequality(problem, x, relational=False) 29.13 ms; roots(lhs - rhs, x) 3.20 ms "
                 "(excluded: also returns the non-real roots); medians of 9 calls, SymPy 1.14.0 (python ground "
                 "types), caches cleared before each call"),
                variants=len(_CUBIC_ROOTS)),
        Fixture("solve", "realistic", "quintic_with_irrational_roots", _solve_mf(_quintic_mf),
                _sympy(_solve_sp(_quintic_sp), polynomial), _same_solution,
                ("used: Poly(lhs - rhs, x).real_roots() 1.28 ms; others: solveset(problem, x, Reals) 61.31 ms; "
                 "solve(problem, x) 108.67 ms; real_roots(lhs - rhs, multiple=False) 1.37 ms; "
                 "solve_univariate_inequality(problem, x, relational=False) 101.17 ms; roots(lhs - rhs, x) 1.71 ms "
                 "(excluded: also returns the non-real roots); medians of 9 calls, SymPy 1.14.0 (python ground "
                 "types), caches cleared before each call"), variants=len(_QUINTICS)),
        Fixture("solve", "realistic", "rational_inequality_with_three_poles", _solve_mf(_inequality_mf),
                _sympy(_solve_sp(_inequality_sp), inequality), _same_solution,
                ("used: solve_univariate_inequality(problem, x, relational=False) 96.53 ms; others: "
                 "solveset(problem, x, Reals) 98.18 ms; solve(problem, x) 20.88 ms (does not give the same "
                 "object); Poly(lhs - rhs, x).real_roots() n/a (raised on this input); real_roots(lhs - rhs, "
                 "multiple=False) n/a (raised on this input); roots(lhs - rhs, x) n/a (excluded: also returns the "
                 "non-real roots); medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before "
                 "each call"), variants=len(_INEQUALITIES)),
        Fixture("solve", "realistic", "scaled_degree_ten_equation", _solve_mf(_degree_ten_mf),
                _sympy(_solve_sp(_degree_ten_sp), polynomial), _same_solution,
                ("used: Poly(lhs - rhs, x).real_roots() 2.80 ms; others: solveset(problem, x, Reals) 215.39 ms; "
                 "solve(problem, x) 692.34 ms; real_roots(lhs - rhs, multiple=False) 2.82 ms; "
                 "solve_univariate_inequality(problem, x, relational=False) 277.14 ms; roots(lhs - rhs, x) 3.03 ms "
                 "(excluded: also returns the non-real roots); medians of 9 calls, SymPy 1.14.0 (python ground "
                 "types), caches cleared before each call"), variants=len(_DEGREE_TEN)),
    )


# ---------------------------------------------------------------------------------------------------------------
# solve_system
# ---------------------------------------------------------------------------------------------------------------

@functools.cache
def _unknowns(count: int):
    return tuple(symbol(f"x{k + 1}", 100 + k) for k in range(count))


def _system(tag: int, i: int, rows: int, columns: int, bound: int = 9):
    """Integer rows and right-hand side of variant i: entries in [-bound, bound], the leading rows x rows block
    strictly diagonally dominant (diagonal about rows * bound + 5), so that the system is consistent and its
    solution set has dimension columns - rows."""
    diagonal = rows * bound + 5
    matrix = []
    for r in range(rows):
        row = [shared._draw(-bound, bound, tag, i, r, c) for c in range(columns)]
        row[r] = diagonal + shared._draw(0, bound, tag, i, r, r)
        matrix.append(tuple(row))
    rhs = tuple(shared._draw(-20, 20, tag, i, r, 99) for r in range(rows))
    return tuple(matrix), rhs


def _two_by_two(i: int):
    """Micro variant i: a 2 x 2 system with entries in [-9, 9] whose determinant is not zero: when it is, both
    diagonal entries grow by 1 until it is not (the determinant is a quadratic in the shift, so this stops)."""
    a, b, c, d, e, f = (v - 9 for v in shared._spread(i, 19, 19, 19, 19, 19, 19))
    shift = 0
    while (a + shift) * (d + shift) - b * c == 0:
        shift += 1
    return ((a + shift, b), (c, d + shift)), (e, f)


def _system_mf(system_of, columns):
    def build(i):
        rows, rhs = system_of(i)
        return (solve_system, (mf.LinearSystem(mf.Matrix(rows), rhs, _unknowns(columns)),), {})
    return build


def _system_sp(system_of, columns):
    def build(i):
        import sympy as sp
        from sympy.polys.matrices import DomainMatrix
        rows, rhs = system_of(i)
        a, b = sp.Matrix(rows), sp.Matrix(rhs)
        return (a, b, sp.symbols(f"x1:{columns + 1}"), DomainMatrix.from_Matrix(a).convert_to(sp.QQ),
                DomainMatrix.from_Matrix(b).convert_to(sp.QQ))
    return build


def _solve_system_fixtures():
    unique = lambda sp, a, b, symbols, da, db: da.lu_solve(db)
    family = lambda sp, a, b, symbols, da, db: a.gauss_jordan_solve(b)
    dense6 = lambda i: _system(111, i, 6, 6)
    dense12 = lambda i: _system(112, i, 12, 12)
    wide = lambda i: _system(113, i, 4, 6)
    return (
        Fixture("solve_system", "micro", "two_by_two", _system_mf(_two_by_two, 2),
                _sympy(_system_sp(_two_by_two, 2), unique), _same_affine,
                ("used: DomainMatrix.lu_solve 78 us; others: Matrix.LUsolve 939 us; Matrix.solve 712 us; "
                 "Matrix.gauss_jordan_solve 757 us; linsolve((A, b), symbols) 2.16 ms; solve_linear_system 2.28 "
                 "ms; medians of 150 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("solve_system", "realistic", "dense_6_by_6", _system_mf(dense6, 6),
                _sympy(_system_sp(dense6, 6), unique), _same_affine,
                ("used: DomainMatrix.lu_solve 236 us; others: Matrix.LUsolve 3.27 ms; Matrix.solve 1.18 ms; "
                 "Matrix.gauss_jordan_solve 1.22 ms; linsolve((A, b), symbols) 7.19 ms; solve_linear_system 7.51 "
                 "ms; medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=3),
        Fixture("solve_system", "realistic", "scaled_dense_12_by_12", _system_mf(dense12, 12),
                _sympy(_system_sp(dense12, 12), unique), _same_affine,
                ("used: DomainMatrix.lu_solve 1.27 ms; others: Matrix.LUsolve 17.04 ms; Matrix.solve 3.58 ms; "
                 "Matrix.gauss_jordan_solve 3.46 ms; linsolve((A, b), symbols) 19.95 ms; solve_linear_system 20.88 "
                 "ms (does not give the same object); medians of 9 calls, SymPy 1.14.0 (python ground types), "
                 "caches cleared before each call"), variants=3),
        Fixture("solve_system", "realistic", "underdetermined_4_by_6", _system_mf(wide, 6),
                _sympy(_system_sp(wide, 6), family), _same_affine,
                ("used: Matrix.gauss_jordan_solve 2.20 ms; others: DomainMatrix.lu_solve n/a (raised on this "
                 "input); Matrix.LUsolve n/a (raised on this input); Matrix.solve n/a (raised on this input); "
                 "linsolve((A, b), symbols) 6.04 ms; solve_linear_system 6.47 ms (does not give the same object); "
                 "medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=3),
    )


FIXTURES = (*_solve_fixtures(), *_solve_system_fixtures())


# ---------------------------------------------------------------------------------------------------------------
# Edge cases (see benchmarks.speed.fixtures.algebra for the conventions). A failed solve or solve_system raises the
# OperationError that carries the code of the failed Result.
# ---------------------------------------------------------------------------------------------------------------

_edge, _type_edges = shared._edge, shared._type_edges


def _integer_symbol():
    return mf.Symbol("n", mf.Integers, uid="00000000-0000-0000-0000-0000000000c1")


def _hundred_equations():
    """x1 = 1, ..., x100 = 1: a 100 x 100 system, 10,000 matrix entries against max_matrix_entries 4,096."""
    unknowns = _unknowns(100)
    return tuple(mf.Eq(variable, 1) for variable in unknowns), unknowns


EDGE_CASES = (
    *_type_edges("solve", solve, (), lambda: {"for_": X}),
    _edge("solve", "not_an_equation", solve, lambda: (X,), lambda: {"for_": X}),
    _edge("solve", "variable_not_a_symbol", solve, lambda: (mf.Eq(X, 1),), {"for_": 5}),
    _edge("solve", "two_variables", solve, lambda: (mf.Eq(X + Y, 0),), lambda: {"for_": X}),
    _edge("solve", "irrational_coefficient", solve, lambda: (mf.Eq(X + mf.sqrt(2), 0),), lambda: {"for_": X}),
    _edge("solve", "integer_domain_symbol", solve, lambda: (mf.Eq(_integer_symbol(), 1),),
          lambda: {"for_": _integer_symbol()}),
    _edge("solve", "identity_equation", solve, lambda: (mf.Eq(X, X),), lambda: {"for_": X}),
    _edge("solve", "equation_without_real_solution", solve, lambda: (mf.Eq(X ** 2 + 1, 0),), lambda: {"for_": X}),
    _edge("solve", "over_limit_integer", solve, lambda: (mf.Eq(shared._HUGE * X, 1),), lambda: {"for_": X}),
    *_type_edges("solve_system", solve_system),
    _edge("solve_system", "equations_without_variables", solve_system, lambda: ((mf.Eq(X, 1),),)),
    _edge("solve_system", "duplicate_variables", solve_system, lambda: ((mf.Eq(X, 1),),),
          lambda: {"for_": (X, X)}),
    _edge("solve_system", "nonlinear_equation", solve_system, lambda: ((mf.Eq(X * Y, 1),),),
          lambda: {"for_": (X, Y)}),
    _edge("solve_system", "matrix_instead_of_a_system", solve_system, lambda: (mf.Matrix(((1, 2),)),)),
    _edge("solve_system", "inconsistent_equations", solve_system, lambda: ((mf.Eq(X, 1), mf.Eq(X, 2)),),
          lambda: {"for_": (X,)}),
    _edge("solve_system", "underdetermined_equations", solve_system, lambda: ((mf.Eq(X + Y, 1),),),
          lambda: {"for_": (X, Y)}),
    _edge("solve_system", "over_limit_integer", solve_system, lambda: ((mf.Eq(shared._HUGE * X, 1),),),
          lambda: {"for_": (X,)}),
    _edge("solve_system", "over_limit_matrix_entries", solve_system, lambda: (_hundred_equations()[0],),
          lambda: {"for_": _hundred_equations()[1]}),
)
