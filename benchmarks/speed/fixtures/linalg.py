"""Speed fixtures of the linalg area: trace, rowspace, columnspace, lu, least_squares, rref, rank, det, inverse,
nullspace, charpoly and real_eigenvalues.

Inputs are exact integer or rational matrices built from a small deterministic generator (no ``random`` module, so a
variant is the same matrix on every Python version). A micro fixture has 1,000 variants of one shape (3x3, 3x4, 4x2 or
symmetric 2x2); a realistic fixture is the size a student or an engineer would use (5x5 to 12x12), and each family
that scales has one larger fixture. The v0.7 cases rref16, det8 and least_squares carry over unchanged.

The SymPy side is a prebuilt ``Matrix`` or ``DomainMatrix`` (integer matrices over ZZ, rational ones over QQ, and QQ
wherever the call needs a field) and the timed call is the call a SymPy user makes for the operation: a ``Matrix`` or
``DomainMatrix`` method, the fastest of those that return the same object. Arithmetic on the elements of the ground
domain (``DomainMatrix.diagonal()`` summed by hand) is never a candidate, nor is a hand-assembled algorithm such as
the normal equations. A call that returns a scaled form of the object (``rref_den``, ``inv_den``, ``nullspace()``
without ``divide_last``, isolating intervals instead of roots) is measured but not a candidate, and neither is
``DomainMatrix.rowspace()``, which cuts the rows of the original matrix, not of its reduced form. Where SymPy has no
single call for the whole object, the timed call is the shortest expression a SymPy user writes: the minimizer of
``solve_least_squares`` plus the residual b - A x for least_squares, and the real roots of the characteristic
polynomial, ``real_roots(Matrix.charpoly(x))``, for real_eigenvalues (``eigenvals`` returns the complex eigenvalues
too). ``Matrix.rowspace()`` returns an echelon basis of the row space, not the reduced basis that MathForge returns;
the comparison row-reduces it, because the reduced basis of a subspace is unique.

Every ``reference`` lists the candidates with their median time in microseconds on that fixture's inputs, taken with
SymPy's caches cleared before each call (Python 3.14.6, SymPy 1.14.0, python ground types), among them the ones that
were measured but excluded. The chosen call is the fastest, except that a candidate within 15% of the conventional
call (a Matrix method, then a DomainMatrix method) is not preferred to it.
"""

from __future__ import annotations

from fractions import Fraction
import uuid

import mathforge as mf
from mathforge.limits import computation
from benchmarks.speed.fixture import EdgeCase, Fixture, symbol

T = symbol("t", 1)  # the variable of charpoly

# --- deterministic data -------------------------------------------------------------------------------------------

_MASK = (1 << 64) - 1
_MULTIPLIER = 6364136223846793005
_INCREMENT = 1442695040888963407


def _ints(seed: int, count: int, low: int, high: int) -> list[int]:
    """count integers in [low, high] from a 64-bit linear congruential stream."""
    state = (seed * _MULTIPLIER + _INCREMENT) & _MASK
    values = []
    for _ in range(count):
        state = (state * _MULTIPLIER + _INCREMENT) & _MASK
        values.append(low + (state >> 33) % (high - low + 1))
    return values


def _grid(rows: int, cols: int, seed: int, low: int = -9, high: int = 9):
    values = _ints(seed, rows * cols, low, high)
    return tuple(tuple(values[r * cols + c] for c in range(cols)) for r in range(rows))


def _dominant(n: int, seed: int, low: int = -9, high: int = 9):
    """A strictly diagonally dominant integer matrix, hence nonsingular."""
    boost = n * max(abs(low), abs(high)) + 1
    return tuple(tuple(v + (boost if r == c else 0) for c, v in enumerate(row))
                 for r, row in enumerate(_grid(n, n, seed, low, high)))


def _product(rows: int, inner: int, cols: int, seed: int, low: int = -3, high: int = 3):
    """rows x cols of rank at most ``inner``: a product of two small integer matrices."""
    left, right = _grid(rows, inner, seed, low, high), _grid(inner, cols, seed + 7919, low, high)
    return tuple(tuple(sum(left[r][k] * right[k][c] for k in range(inner)) for c in range(cols)) for r in range(rows))


def _sparse(n: int, seed: int, low: int = -4, high: int = 4):
    """About a quarter of the entries are zero, so that PLU has to swap rows."""
    values, holes = _grid(n, n, seed, low, high), _grid(n, n, seed + 104729, 0, 3)
    return tuple(tuple(0 if holes[r][c] == 0 else values[r][c] for c in range(n)) for r in range(n))


def _rationals(rows: int, cols: int, seed: int):
    """Entries p/q with p in [-9, 9] and q in [1, 6]."""
    numerators, denominators = _grid(rows, cols, seed, -9, 9), _grid(rows, cols, seed + 31, 1, 6)
    return tuple(tuple(Fraction(numerators[r][c], denominators[r][c]) for c in range(cols)) for r in range(rows))


def _symmetric(n: int, seed: int, low: int = -4, high: int = 4):
    """A symmetric integer matrix: every eigenvalue is real."""
    values = iter(_ints(seed, n * (n + 1) // 2, low, high))
    rows = [[0] * n for _ in range(n)]
    for p in range(n):
        for q in range(p, n):
            rows[p][q] = rows[q][p] = next(values)
    return tuple(tuple(row) for row in rows)


def _symmetric_2x2(i: int):
    """The i-th of 1,000 distinct symmetric 2x2 matrices with entries in [-9, 9] (19**3 = 6859; 2711 is coprime)."""
    index = (i * 2711 + 17) % 6859
    a, b, c = index % 19 - 9, index // 19 % 19 - 9, index // 361 - 9
    return ((a, b), (b, c))


def _tridiagonal(n: int):
    """Diagonal 17, the adjacent diagonals 1: the v0.7 matrix rref16 for n = 16."""
    return tuple(tuple(17 if i == j else int(abs(i - j) == 1) for j in range(n)) for i in range(n))


DENSE_8 = tuple(tuple(50 if i == j else (3 * i + 5 * j) % 11 - 5 for j in range(8)) for i in range(8))  # det8
VANDERMONDE_5 = tuple((1, t, t * t) for t in range(-2, 3))  # least_squares
VANDERMONDE_5_RHS = (10, -2, 7, 2, 18)
HILBERT_6 = tuple(tuple(Fraction(1, r + c + 1) for c in range(6)) for r in range(6))

# --- builders: MathForge and SymPy objects from the same data ----------------------------------------------------


def _mf_matrix(rows):
    return mf.Matrix(tuple(tuple(mf.Rational(e.numerator, e.denominator) if isinstance(e, Fraction) else e
                                 for e in row) for row in rows))


def _sp_matrix(rows):
    import sympy as sp
    return sp.Matrix([[sp.Rational(e.numerator, e.denominator) if isinstance(e, Fraction) else e for e in row]
                      for row in rows])


def _dm(rows, field: bool = False):
    """The DomainMatrix of a matrix: over ZZ for integers, over QQ for rationals and wherever ``field`` is set."""
    from sympy import QQ
    from sympy.polys.matrices import DomainMatrix
    matrix = DomainMatrix.from_Matrix(_sp_matrix(rows))
    return matrix.convert_to(QQ) if field else matrix


# --- comparing mathematical objects --------------------------------------------------------------------------------


def _frac(value) -> Fraction:
    """A MathForge Rational, a Python int, a SymPy number or a DomainMatrix element as an exact Fraction."""
    return Fraction(int(value.numerator), int(value.denominator))


def _mf_rows(matrix) -> list[list[Fraction]]:
    return [[_frac(e) for e in row] for row in matrix.rows]


def _sp_rows(matrix) -> list[list[Fraction]]:
    """Rows of a SymPy Matrix or DomainMatrix."""
    if hasattr(matrix, "to_Matrix"):
        matrix = matrix.to_Matrix()
    return [[_frac(e) for e in row] for row in matrix.tolist()]


def _mf_vectors(vectors) -> list[tuple[Fraction, ...]]:
    return [tuple(_frac(e) for e in vector) for vector in vectors]


def _same_scalar(mf_value, sympy_value) -> bool:
    return type(mf_value) is mf.Rational and _frac(mf_value) == _frac(sympy_value)


def _same_rank(mf_value, sympy_value) -> bool:
    return type(mf_value) is int and mf_value == int(sympy_value)


def _same_matrix(mf_value, sympy_value) -> bool:
    return type(mf_value) is mf.Matrix and _mf_rows(mf_value) == _sp_rows(sympy_value)


def _same_rref(mf_value, sympy_value) -> bool:
    """sympy_value is the pair (rref matrix, pivot columns) of Matrix.rref and DomainMatrix.rref."""
    matrix, pivots = sympy_value
    return (type(mf_value) is mf.RREFResult and _mf_rows(mf_value.matrix) == _sp_rows(matrix)
            and tuple(mf_value.pivot_columns) == tuple(pivots))


def _same_nullspace(mf_value, sympy_value) -> bool:
    """A list of SymPy column vectors, or a DomainMatrix whose rows are the basis vectors."""
    if hasattr(sympy_value, "to_Matrix"):
        vectors = [tuple(row) for row in _sp_rows(sympy_value)]
    else:
        vectors = [tuple(_frac(e) for e in vector) for vector in sympy_value]
    return type(mf_value) is tuple and _mf_vectors(mf_value) == vectors


def _same_rowspace(mf_value, sympy_value) -> bool:
    """sympy_value is the list of row vectors of Matrix.rowspace(), an echelon basis of the row space. MathForge
    returns the reduced basis, which is unique for a subspace, so the echelon basis is row-reduced first."""
    import sympy as sp
    if type(mf_value) is not tuple or len(mf_value) != len(sympy_value):
        return False
    if not sympy_value:
        return True
    reduced, pivots = sp.Matrix.vstack(*sympy_value).rref()
    return (len(pivots) == len(sympy_value)
            and _mf_vectors(mf_value) == [tuple(_frac(e) for e in reduced.row(k)) for k in range(len(pivots))])


def _same_columnspace(mf_value, sympy_value) -> bool:
    """A DomainMatrix whose columns are the basis, or a list of SymPy column vectors."""
    if hasattr(sympy_value, "to_Matrix"):
        vectors = [tuple(column) for column in zip(*_sp_rows(sympy_value))]
    else:
        vectors = [tuple(_frac(e) for e in vector) for vector in sympy_value]
    return type(mf_value) is tuple and _mf_vectors(mf_value) == vectors


def _same_lu(mf_value, sympy_value) -> bool:
    """sympy_value is DomainMatrix.lu(): L, U and the row swaps that give P A = L U."""
    lower, upper, swaps = sympy_value
    size = lower.shape[0]
    permutation = [[Fraction(int(r == c)) for c in range(size)] for r in range(size)]
    for first, second in swaps:
        permutation[first], permutation[second] = permutation[second], permutation[first]
    return (type(mf_value) is mf.LUDecomposition and _mf_rows(mf_value.L) == _sp_rows(lower)
            and _mf_rows(mf_value.U) == _sp_rows(upper) and _mf_rows(mf_value.P) == permutation)


def _same_least_squares(mf_value, sympy_value) -> bool:
    """sympy_value is (minimizer, residual b - A x), each a column; the minimizer is unique (full column rank)."""
    minimizer, residual = sympy_value
    return (type(mf_value) is mf.LeastSquaresResult and mf_value.basis == ()
            and [_frac(e) for e in mf_value.particular] == [row[0] for row in _sp_rows(minimizer)]
            and [_frac(e) for e in mf_value.residual] == [row[0] for row in _sp_rows(residual)])


def _same_charpoly(mf_value, sympy_value) -> bool:
    """sympy_value lists the coefficients, highest degree first: DomainMatrix.charpoly() or PurePoly.all_coeffs()."""
    return (type(mf_value) is mf.Polynomial and [_frac(c) for c in reversed(mf_value.coefficients)]
            == [_frac(c) for c in sympy_value])


def _to_sympy(value):
    """A MathForge exact real (Rational, radical expression, RealAlgebraicRoot) as a SymPy number."""
    import sympy as sp
    if type(value) is mf.Rational:
        return sp.Rational(value.numerator, value.denominator)
    if type(value) is mf.Sqrt:
        return sp.sqrt(_to_sympy(value.radicand))
    if type(value) is mf.Add:
        return sp.Add(*(_to_sympy(term) for term in value.terms))
    if type(value) is mf.Mul:
        return sp.Mul(*(_to_sympy(factor) for factor in value.factors))
    if type(value) is mf.Pow:
        return _to_sympy(value.base) ** value.exponent
    if type(value) is mf.RealAlgebraicRoot:
        x = sp.Symbol("x")
        polynomial = sp.Poly(sum(c * x ** k for k, c in enumerate(value.integer_coefficients)), x)
        return polynomial.real_roots()[value.real_index]
    raise TypeError(type(value))


def _same_spectrum(mf_value, sympy_value) -> bool:
    """MathForge's RootRecords against SymPy's {eigenvalue: multiplicity} or [(root, multiplicity)] (all real)."""
    import sympy as sp
    pairs = list(sympy_value.items()) if isinstance(sympy_value, dict) else list(sympy_value)
    if type(mf_value) is not tuple or len(mf_value) != len(pairs):
        return False
    ours = sorted(((_to_sympy(record.root), record.multiplicity) for record in mf_value),
                  key=lambda pair: float(pair[0]))
    theirs = sorted(((value, int(multiplicity)) for value, multiplicity in pairs), key=lambda pair: float(pair[0]))
    return all(m == n and abs(sp.N(a - b, 40)) < sp.Float("1e-30") for (a, m), (b, n) in zip(ours, theirs))


# --- the timed SymPy calls -----------------------------------------------------------------------------------------


def _trace_call(rows):
    matrix = _sp_matrix(rows)
    return lambda: matrix.trace()


def _rank_zz(rows):
    matrix = _dm(rows)
    return lambda: matrix.rank()


def _rank_qq(rows):
    matrix = _dm(rows, field=True)
    return lambda: matrix.rank()


def _det_call(rows):
    matrix = _dm(rows)
    return lambda: matrix.det()


def _inverse_domain(rows):
    matrix = _dm(rows, field=True)
    return lambda: matrix.inv()


def _inverse_matrix(rows):
    matrix = _sp_matrix(rows)
    return lambda: matrix.inv()


def _rref_matrix(rows):
    matrix = _sp_matrix(rows)
    return lambda: matrix.rref()


def _rref_domain(rows):
    matrix = _dm(rows, field=True)
    return lambda: matrix.rref()


def _nullspace_matrix(rows):
    matrix = _sp_matrix(rows)
    return lambda: matrix.nullspace()


def _nullspace_domain(rows):
    matrix = _dm(rows, field=True)
    return lambda: matrix.nullspace(divide_last=True)


def _rowspace_call(rows):
    """Matrix.rowspace(). (DomainMatrix.rowspace() cuts the rows of the original matrix, not of its reduced form, so
    it is not a basis in general and no candidate.)"""
    matrix = _sp_matrix(rows)
    return lambda: matrix.rowspace()


def _columnspace_call(rows):
    matrix = _dm(rows, field=True)
    return lambda: matrix.columnspace()


def _lu_call(rows):
    matrix = _dm(rows, field=True)
    return lambda: matrix.lu()


def _charpoly_berkowitz(rows):
    matrix = _dm(rows)
    return lambda: matrix.charpoly_berk()


def _charpoly_call(rows):
    matrix = _dm(rows)
    return lambda: matrix.charpoly()


def _eigenvalues_call(rows):
    """The real roots of the characteristic polynomial with their multiplicities."""
    import sympy as sp
    matrix, x = _sp_matrix(rows), sp.Symbol("x")
    return lambda: sp.real_roots(matrix.charpoly(x), multiple=False)


def _least_squares_solve(rows, rhs):
    """The minimizer of Matrix.solve_least_squares and the residual b - A x, which is part of MathForge's result."""
    import sympy as sp
    matrix, column = _sp_matrix(rows), sp.Matrix(rhs)

    def solve():
        minimizer = matrix.solve_least_squares(column, method="LDL")
        return minimizer, column - matrix * minimizer
    return solve


# --- fixtures ------------------------------------------------------------------------------------------------------


def _fixture(operation, size, label, rows, sympy_call, agree, reference, *, variants, v07="", extra=()):
    """A fixture of an operation whose public function takes the matrix first: ``extra`` are its other arguments."""
    def mathforge(i):
        return getattr(mf, operation), (_mf_matrix(rows(i)), *extra), {}

    def sympy(i):
        return sympy_call(rows(i))

    return Fixture(operation, size, label, mathforge, sympy, agree, reference, variants, v07)


def _constant(rows):
    """A data function that returns the same matrix for every variant."""
    return lambda i: rows


def _least_squares_fixture(size, label, data, sympy_call, reference, *, variants, v07=""):
    """data(i) -> (matrix rows, right-hand side)."""
    def mathforge(i):
        rows, rhs = data(i)
        return mf.least_squares, (_mf_matrix(rows), rhs), {}

    def sympy(i):
        rows, rhs = data(i)
        return sympy_call(rows, rhs)

    return Fixture("least_squares", size, label, mathforge, sympy, _same_least_squares, reference, variants, v07)


def _micro_least_squares(i):
    return _grid(4, 2, 5000 + i, -5, 5), tuple(_ints(6000 + i, 4, -9, 9))


def _fit_data(points: int, columns: int, seed: int):
    """A polynomial fit: rows (1, t, t^2, ...) at t = -k..k and a seeded right-hand side."""
    def data(i):
        rows = tuple(tuple((t - points // 2) ** power for power in range(columns)) for t in range(points))
        return rows, tuple(_ints(seed + i, points, -20, 20))
    return data


FIXTURES = (
    # trace
    _fixture("trace", "micro", "3x3", lambda i: _grid(3, 3, 11_000 + i), _trace_call, _same_scalar,
             "Matrix.trace() 28.6 us (chosen; DomainMatrix has no trace); sum(DomainMatrix.diagonal()) 4.7 us adds "
             "ground-domain elements by hand and is excluded", variants=1000),
    _fixture("trace", "realistic", "12x12", lambda i: _grid(12, 12, 12_000 + i), _trace_call, _same_scalar,
             "Matrix.trace() 43.0 us (chosen); sum(DomainMatrix.diagonal()) 7.6 us adds ground-domain elements by "
             "hand and is excluded", variants=16),
    _fixture("trace", "realistic", "scaled_64x64", lambda i: _grid(64, 64, 13_000 + i), _trace_call, _same_scalar,
             "Matrix.trace() 111.1 us (chosen); sum(DomainMatrix.diagonal()) 17.8 us adds ground-domain elements by "
             "hand and is excluded", variants=4),
    # rank
    _fixture("rank", "micro", "3x3", lambda i: _grid(3, 3, 21_000 + i, -3, 3), _rank_zz, _same_rank,
             "DomainMatrix(ZZ).rank() 113.8 us (chosen); DomainMatrix(QQ).rank() 183.2 us; Matrix.rank() 512.3 us",
             variants=1000),
    _fixture("rank", "realistic", "8x10_rank5", lambda i: _product(8, 5, 10, 22_000 + i), _rank_zz, _same_rank,
             "DomainMatrix(ZZ).rank() 338.4 us (chosen); DomainMatrix(QQ).rank() 673.7 us; Matrix.rank() 5385.2 us",
             variants=16),
    _fixture("rank", "realistic", "scaled_16x16_rank12", lambda i: _product(16, 12, 16, 23_000 + i, -2, 2), _rank_qq,
             _same_rank,
             "DomainMatrix(QQ).rank() 1593.8 us (chosen); DomainMatrix(ZZ).rank() 2684.9 us; Matrix.rank() 18303.9 us",
             variants=4),
    # det
    _fixture("det", "micro", "3x3", lambda i: _grid(3, 3, 31_000 + i), _det_call, _same_scalar,
             "DomainMatrix(ZZ).det() 28.4 us (chosen); Matrix.det() 55.2 us; DomainMatrix(QQ).det() 58.8 us; "
             "Matrix.det(method='bareiss') 61.3 us; 'lu' 78.0 us; 'domain-ge' 82.6 us; 'berkowitz' 82.8 us",
             variants=1000),
    _fixture("det", "realistic", "dense_8x8", _constant(DENSE_8), _det_call, _same_scalar,
             "DomainMatrix(ZZ).det() 89.5 us (chosen; pinned by the spec); DomainMatrix(QQ).det() 509.2 us; "
             "Matrix.det(method='domain-ge') 842.5 us; 'bareiss' 3185.9 us (the v0.7 reference); "
             "Matrix.det() 3212.4 us; 'berkowitz' 3275.4 us; 'lu' 3638.8 us",
             variants=1, v07="det8"),
    _fixture("det", "realistic", "rational_5x5", lambda i: _rationals(5, 5, 32_000 + i), _det_call, _same_scalar,
             "DomainMatrix(QQ).det() 158.1 us (chosen); Matrix.det(method='domain-ge') 446.1 us; 'lu' 1161.0 us; "
             "Matrix.det() 1488.7 us; 'bareiss' 1498.8 us; 'berkowitz' 1708.7 us",
             variants=16),
    _fixture("det", "realistic", "scaled_24x24", lambda i: _grid(24, 24, 33_000 + i), _det_call, _same_scalar,
             "DomainMatrix(ZZ).det() 1554.8 us (chosen); DomainMatrix(QQ).det() 14418.8 us; "
             "Matrix.det(method='domain-ge') 14824.9 us; 'berkowitz' 58403.7 us; 'bareiss' 75973.8 us; "
             "Matrix.det() 76952.7 us; 'lu' 93983.5 us",
             variants=4),
    # inverse
    _fixture("inverse", "micro", "3x3", lambda i: _dominant(3, 41_000 + i, -3, 3), _inverse_domain, _same_matrix,
             "DomainMatrix(QQ).inv() 65.5 us (chosen); Matrix.inv() 196.2 us; Matrix.inv(method='GE') 292.6 us; "
             "'ADJ' 515.0 us; 'LU' 1025.1 us; 'LDL' 2496.5 us; 'CH' 4101.6 us; 'QR' 12545.9 us; "
             "DomainMatrix.inv_den() 123.1 us returns the pair (adjugate-scaled matrix, denominator), not the inverse",
             variants=1000),
    _fixture("inverse", "realistic", "dominant_8x8", lambda i: _dominant(8, 42_000 + i), _inverse_matrix,
             _same_matrix,
             "Matrix.inv() 1061.4 us (chosen); DomainMatrix(QQ).inv() 1168.5 us; Matrix.inv(method='GE') 1582.8 us; "
             "'LU' 15845.0 us; 'LDL' 21404.3 us; 'CH' 91114.1 us; 'QR' 131719.2 us; 'ADJ' 163982.8 us; "
             "DomainMatrix.inv_den() 380.2 us returns a pair, not the inverse",
             variants=16),
    _fixture("inverse", "realistic", "hilbert_6x6", _constant(HILBERT_6), _inverse_domain, _same_matrix,
             "DomainMatrix(QQ).inv() 411.6 us (chosen); Matrix.inv() 846.8 us; Matrix.inv(method='GE') 1261.1 us; "
             "'LU' 7447.9 us; 'LDL' 8478.1 us; 'CH' 28555.0 us; 'ADJ' 57909.7 us; 'QR' 58079.4 us; "
             "DomainMatrix.inv_den() 496.9 us returns a pair, not the inverse",
             variants=1),
    _fixture("inverse", "realistic", "scaled_16x16", lambda i: _dominant(16, 43_000 + i), _inverse_matrix,
             _same_matrix,
             "Matrix.inv() 12322.5 us (chosen; DomainMatrix(QQ).inv() 11782.7 us and Matrix.inv(method='GE') "
             "12221.9 us are within 5%); 'LDL' 105730.3 us; 'LU' 123240.2 us; 'CH' 447981.2 us; 'QR' 619145.1 us; "
             "'ADJ' 3867098.7 us; DomainMatrix.inv_den() 10399.0 us returns a pair, not the inverse",
             variants=2),
    # rref
    _fixture("rref", "micro", "3x4", lambda i: _grid(3, 4, 51_000 + i, -5, 5), _rref_matrix, _same_rref,
             "Matrix.rref() 98.6 us (chosen); DomainMatrix(QQ).rref() 146.9 us; DomainMatrix(ZZ).rref_den() 43.9 us "
             "returns the fraction-free pair (numerators, denominator), not the reduced matrix",
             variants=1000),
    _fixture("rref", "realistic", "tridiagonal_16x16", _constant(_tridiagonal(16)), _rref_domain, _same_rref,
             "DomainMatrix(QQ).rref() 163.1 us (chosen); Matrix.rref() 320.3 us (the v0.7 reference); "
             "DomainMatrix(ZZ).rref_den() 354.6 us returns a pair, not the reduced matrix",
             variants=1, v07="rref16"),
    _fixture("rref", "realistic", "8x12_rank6", lambda i: _product(8, 6, 12, 52_000 + i), _rref_matrix, _same_rref,
             "Matrix.rref() 319.2 us (chosen); DomainMatrix(QQ).rref() 663.0 us; DomainMatrix(ZZ).rref_den() 171.9 us "
             "returns a pair, not the reduced matrix",
             variants=16),
    _fixture("rref", "realistic", "scaled_tridiagonal_32x32", _constant(_tridiagonal(32)), _rref_domain, _same_rref,
             "DomainMatrix(QQ).rref() 394.8 us (chosen); Matrix.rref() 553.2 us; DomainMatrix(ZZ).rref_den() 630.0 us "
             "returns a pair, not the reduced matrix",
             variants=1),
    # nullspace
    _fixture("nullspace", "micro", "3x4", lambda i: _grid(3, 4, 61_000 + i, -5, 5), _nullspace_matrix,
             _same_nullspace,
             "Matrix.nullspace() 188.1 us (chosen; DomainMatrix(QQ).nullspace(divide_last=True) 178.8 us is within "
             "5%); DomainMatrix(ZZ).nullspace() 66.6 us and DomainMatrix(QQ).nullspace() 185.4 us return a scaled "
             "basis, not the canonical one with a 1 at each free variable",
             variants=1000),
    _fixture("nullspace", "realistic", "8x12_rank6", lambda i: _product(8, 6, 12, 62_000 + i), _nullspace_domain,
             _same_nullspace,
             "DomainMatrix(QQ).nullspace(divide_last=True) 732.0 us (chosen); Matrix.nullspace() 932.5 us; "
             "DomainMatrix(ZZ).nullspace() 225.0 us and DomainMatrix(QQ).nullspace() 712.2 us return a scaled basis",
             variants=16),
    _fixture("nullspace", "realistic", "scaled_14x20_rank9", lambda i: _product(14, 9, 20, 63_000 + i, -2, 2),
             _nullspace_domain, _same_nullspace,
             "DomainMatrix(QQ).nullspace(divide_last=True) 1889.2 us (chosen); Matrix.nullspace() 4038.3 us; "
             "DomainMatrix(QQ).nullspace() 1816.1 us and DomainMatrix(ZZ).nullspace() 3034.6 us return a scaled basis",
             variants=4),
    # rowspace
    _fixture("rowspace", "micro", "3x4", lambda i: _grid(3, 4, 71_000 + i, -5, 5), _rowspace_call, _same_rowspace,
             "Matrix.rowspace() 411.4 us (chosen; an echelon basis, row-reduced by agree); Matrix.rref() cut to its "
             "pivot rows 172.4 us and DomainMatrix.rref() cut likewise 217.1 us are hand-assembled and excluded; "
             "DomainMatrix.rowspace() returns the first rows of the original matrix, not a basis",
             variants=1000),
    _fixture("rowspace", "realistic", "8x12_rank6", lambda i: _product(8, 6, 12, 72_000 + i), _rowspace_call,
             _same_rowspace,
             "Matrix.rowspace() 2338.6 us (chosen); Matrix.rref() cut to its pivot rows 363.0 us and "
             "DomainMatrix.rref() cut likewise 505.0 us are hand-assembled and excluded",
             variants=16),
    _fixture("rowspace", "realistic", "scaled_14x20_rank9", lambda i: _product(14, 9, 20, 73_000 + i, -2, 2),
             _rowspace_call, _same_rowspace,
             "Matrix.rowspace() 9789.9 us (chosen); DomainMatrix.rref() cut to its pivot rows 1052.2 us and "
             "Matrix.rref() cut likewise 2306.8 us are hand-assembled and excluded",
             variants=4),
    # columnspace
    _fixture("columnspace", "micro", "3x4", lambda i: _grid(3, 4, 81_000 + i, -5, 5), _columnspace_call,
             _same_columnspace, "DomainMatrix(QQ).columnspace() 178.1 us (chosen); Matrix.columnspace() 338.4 us",
             variants=1000),
    _fixture("columnspace", "realistic", "8x12_rank6", lambda i: _product(8, 6, 12, 82_000 + i), _columnspace_call,
             _same_columnspace, "DomainMatrix(QQ).columnspace() 650.5 us (chosen); Matrix.columnspace() 3581.8 us",
             variants=16),
    _fixture("columnspace", "realistic", "scaled_14x20_rank9", lambda i: _product(14, 9, 20, 83_000 + i, -2, 2),
             _columnspace_call, _same_columnspace,
             "DomainMatrix(QQ).columnspace() 2031.8 us (chosen); Matrix.columnspace() 11819.5 us", variants=4),
    # lu
    _fixture("lu", "micro", "3x3", lambda i: _sparse(3, 91_000 + i), _lu_call, _same_lu,
             "DomainMatrix(QQ).lu() 66.2 us (chosen); Matrix.LUdecomposition() 379.4 us", variants=1000),
    _fixture("lu", "realistic", "sparse_8x8", lambda i: _sparse(8, 92_000 + i), _lu_call, _same_lu,
             "DomainMatrix(QQ).lu() 172.6 us (chosen); Matrix.LUdecomposition() 2126.0 us", variants=16),
    _fixture("lu", "realistic", "scaled_16x16", lambda i: _sparse(16, 93_000 + i), _lu_call, _same_lu,
             "DomainMatrix(QQ).lu() 1801.2 us (chosen); Matrix.LUdecomposition() 25580.4 us", variants=4),
    # least_squares
    _least_squares_fixture(
        "micro", "4x2", _micro_least_squares, _least_squares_solve,
        "Matrix.solve_least_squares(b, method='LDL') plus the residual b - A x 783.8 us (chosen); Matrix.pinv_solve "
        "plus residual 1084.0 us; Matrix.solve_least_squares(b) (method='CH', the default) plus residual 2023.5 us "
        "(the v0.7 reference); method='QR' 4238.4 us; the normal equations A^T A x = A^T b solved by hand are "
        "excluded: Matrix (A.T*A).LUsolve(A.T*b) plus residual 683.4 us, the same on DomainMatrix by lu_solve() "
        "99.8 us",
        variants=1000),
    _least_squares_fixture(
        "realistic", "vandermonde_5x3", lambda i: (VANDERMONDE_5, VANDERMONDE_5_RHS), _least_squares_solve,
        "Matrix.solve_least_squares(b, method='LDL') plus the residual b - A x 861.3 us (chosen); Matrix.pinv_solve "
        "plus residual 1340.6 us; Matrix.solve_least_squares(b) (method='CH', the default) plus residual 2604.3 us "
        "(the v0.7 reference); method='QR' 5128.2 us; the normal equations solved by hand are excluded: Matrix "
        "(A.T*A).LUsolve(A.T*b) plus residual 794.0 us, the same on DomainMatrix by lu_solve() 111.0 us",
        variants=1, v07="least_squares"),
    _least_squares_fixture(
        "realistic", "scaled_fit_30x5", _fit_data(30, 5, 101_000), _least_squares_solve,
        "Matrix.solve_least_squares(b, method='LDL') plus the residual b - A x 2815.1 us (chosen); Matrix.pinv_solve "
        "plus residual 6379.3 us; Matrix.solve_least_squares(b) (method='CH', the default) plus residual 9104.3 us; "
        "method='QR' 34058.3 us; the normal equations solved by hand are excluded: Matrix (A.T*A).LUsolve(A.T*b) "
        "plus residual 2012.3 us, the same on DomainMatrix by lu_solve() 347.2 us",
        variants=8),
    # charpoly
    _fixture("charpoly", "micro", "3x3", lambda i: _grid(3, 3, 111_000 + i), _charpoly_berkowitz, _same_charpoly,
             "DomainMatrix.charpoly_berk() 55.0 us (chosen); DomainMatrix(ZZ).charpoly() 89.3 us; "
             "Matrix.berkowitz_charpoly(x) 217.6 us; Matrix.charpoly(x) 232.2 us",
             variants=1000, extra=(T,)),
    _fixture("charpoly", "realistic", "dense_8x8", lambda i: _grid(8, 8, 112_000 + i, -5, 5), _charpoly_call,
             _same_charpoly,
             "DomainMatrix(ZZ).charpoly() 398.4 us (chosen); DomainMatrix.charpoly_berk() 424.5 us; "
             "Matrix.berkowitz_charpoly(x) 813.5 us; Matrix.charpoly(x) 838.9 us",
             variants=16, extra=(T,)),
    _fixture("charpoly", "realistic", "scaled_12x12", lambda i: _grid(12, 12, 113_000 + i, -5, 5), _charpoly_call,
             _same_charpoly,
             "DomainMatrix(ZZ).charpoly() 892.0 us (chosen); DomainMatrix.charpoly_berk() 1484.7 us; "
             "Matrix.berkowitz_charpoly(x) 1611.3 us; Matrix.charpoly(x) 1672.8 us",
             variants=4, extra=(T,)),
    # real_eigenvalues
    _fixture("real_eigenvalues", "micro", "symmetric_2x2", _symmetric_2x2, _eigenvalues_call, _same_spectrum,
             "real_roots(Matrix.charpoly(x), multiple=False) 1119.0 us (chosen); Matrix.charpoly(x).real_roots() "
             "1110.2 us (within 1%, but it repeats a root instead of pairing it with its multiplicity); "
             "Matrix.eigenvals() 1228.8 us (every eigenvalue, complex ones too); real_roots(Poly(DomainMatrix."
             "charpoly()), multiple=False) 1091.4 us is assembled from the coefficients of a DomainMatrix and "
             "excluded; Poly(...).intervals() returns isolating intervals, not the roots",
             variants=1000),
    _fixture("real_eigenvalues", "realistic", "symmetric_4x4", lambda i: _symmetric(4, 121_000 + i),
             _eigenvalues_call, _same_spectrum,
             "real_roots(Matrix.charpoly(x), multiple=False) 1348.0 us (chosen); Matrix.charpoly(x).real_roots() "
             "1313.4 us (within 3%; repeats roots); Matrix.eigenvals() 6723.7 us (every eigenvalue, complex ones "
             "too); real_roots(Poly(DomainMatrix.charpoly()), multiple=False) 1748.3 us is assembled from the "
             "coefficients of a DomainMatrix and excluded; Poly(...).intervals() returns isolating intervals, not "
             "the roots",
             variants=16),
    _fixture("real_eigenvalues", "realistic", "scaled_symmetric_6x6", lambda i: _symmetric(6, 122_000 + i),
             _eigenvalues_call, _same_spectrum,
             "real_roots(Matrix.charpoly(x), multiple=False) 4302.1 us (chosen); Matrix.charpoly(x).real_roots() "
             "3961.9 us (within 8%; repeats roots); Matrix.eigenvals() 31706.3 us (every eigenvalue, complex ones "
             "too); real_roots(Poly(DomainMatrix.charpoly()), multiple=False) 2405.8 us is assembled from the "
             "coefficients of a DomainMatrix and excluded; Poly(...).intervals() returns isolating intervals, not "
             "the roots",
             variants=4),
)

# --- edge cases ----------------------------------------------------------------------------------------------------

_TYPE_EDGES = (("type_bool", True), ("type_float", 1.5), ("type_str", "1"), ("type_none", None))


def _m(*rows):
    return mf.Matrix(rows)


def _over_limit_entries():
    """100 x 100 = 10,000 entries against max_matrix_entries = 4,096. Such a matrix cannot be built under the default
    limits, so it is built under larger ones; every operation then refuses it under the defaults."""
    with computation(mf.ComputationLimits(max_matrix_entries=40_000, max_nodes=1_000_000, max_work=10_000_000)):
        return mf.Matrix(tuple(tuple((r * 3 + c) % 7 for c in range(100)) for r in range(100)))


def _over_limit_integer():
    """An entry of 140,001 bits against max_integer_bits = 65,536."""
    with computation(mf.ComputationLimits(max_integer_bits=300_000, max_work=10_000_000)):
        return mf.Matrix(((2 ** 140_000, 1), (1, 3)))


def _edge(operation, label, build):
    """An EdgeCase of a value function; build() returns its arguments."""
    return EdgeCase(operation, label, lambda: (getattr(mf, operation), build(), {}))


def _type_edges(operation, *tail):
    """The True / 1.5 / "1" / None edges in the first argument; tail holds the other, valid arguments."""
    return tuple(EdgeCase(operation, label, lambda value=value: (getattr(mf, operation), (value, *tail), {}))
                 for label, value in _TYPE_EDGES)


def _matrix_edges(operation, *tail):
    """The edges that every matrix operation has: four type edges, a tuple instead of a Matrix, two over-limit inputs
    and the empty 0 x 0 matrix."""
    return (*_type_edges(operation, *tail),
            _edge(operation, "type_tuple", lambda: (((1, 2), (3, 4)), *tail)),
            _edge(operation, "over_limit_entries", lambda: (_over_limit_entries(), *tail)),
            _edge(operation, "over_limit_integer", lambda: (_over_limit_integer(), *tail)),
            _edge(operation, "empty", lambda: (mf.Matrix(()), *tail)))


def _halves():
    """A 2 x 2 matrix with rational entries."""
    return _m((mf.Rational(1, 2), mf.Rational(1, 3)), (mf.Rational(1, 4), 5))


def _symmetric_thirds():
    """A symmetric 2 x 2 matrix with rational entries."""
    return _m((mf.Rational(1, 2), mf.Rational(1, 3)), (mf.Rational(1, 3), 5))


def _non_square(operation, *tail):
    return _edge(operation, "non_square", lambda: (_m((1, 2, 3), (4, 5, 6)), *tail))


_SINGULAR = ((1, 2, 3), (2, 4, 6), (1, 0, 1))
_CLEAN = ((2, 1, 0), (1, 3, 1), (0, 1, 4))

EDGE_CASES: tuple[EdgeCase, ...] = (
    # trace
    *_matrix_edges("trace"),
    _non_square("trace"),
    _edge("trace", "rational_entries", lambda: (_m((mf.Rational(1, 2), 3), (4, mf.Rational(-5, 3))),)),
    # rowspace
    *_matrix_edges("rowspace"),
    _edge("rowspace", "zero_matrix", lambda: (_m((0, 0, 0), (0, 0, 0)),)),
    _edge("rowspace", "dependent_rows", lambda: (_m(*_SINGULAR),)),
    # columnspace
    *_matrix_edges("columnspace"),
    _edge("columnspace", "zero_matrix", lambda: (_m((0, 0, 0), (0, 0, 0)),)),
    _edge("columnspace", "dependent_columns", lambda: (_m((1, 2, 2), (3, 6, 6), (1, 2, 5)),)),
    # lu
    *_matrix_edges("lu"),
    _non_square("lu"),
    _edge("lu", "singular", lambda: (_m(*_SINGULAR),)),
    _edge("lu", "zero_pivot_swap", lambda: (_m((0, 2, 1), (3, 1, 4), (1, 0, 5)),)),
    # least_squares
    *_type_edges("least_squares", (1, 2, 3)),
    _edge("least_squares", "type_tuple", lambda: (((1, 2), (3, 4), (5, 6)), (1, 2, 3))),
    _edge("least_squares", "over_limit_entries", lambda: (_over_limit_entries(), tuple(range(100)))),
    _edge("least_squares", "over_limit_integer", lambda: (_over_limit_integer(), (1, 2))),
    _edge("least_squares", "empty", lambda: (mf.Matrix(()), ())),
    _edge("least_squares", "rhs_length_mismatch", lambda: (_m((1, 2), (3, 4), (5, 6)), (1, 2))),
    _edge("least_squares", "rhs_float", lambda: (_m((1, 2), (3, 4), (5, 6)), (1, 2.5, 3))),
    _edge("least_squares", "rhs_str", lambda: (_m((1, 2), (3, 4), (5, 6)), "123")),
    _edge("least_squares", "rhs_none", lambda: (_m((1, 2), (3, 4), (5, 6)), None)),
    _edge("least_squares", "rank_deficient", lambda: (_m((1, 1), (2, 2), (3, 3)), (1, 2, 4))),
    _edge("least_squares", "underdetermined", lambda: (_m((1, 2, 3),), (6,))),
    _edge("least_squares", "inconsistent_residual", lambda: (_m((1, 0), (0, 1), (1, 1)), (1, 1, 5))),
    # rref
    *_matrix_edges("rref"),
    _edge("rref", "zero_matrix", lambda: (_m((0, 0, 0), (0, 0, 0)),)),
    _edge("rref", "wide", lambda: (_m((2, 4, 6, 8), (1, 2, 3, 5)),)),
    _edge("rref", "rational_entries", lambda: (_halves(),)),
    # rank
    *_matrix_edges("rank"),
    _edge("rank", "zero_matrix", lambda: (_m((0, 0), (0, 0)),)),
    _edge("rank", "dependent_rows", lambda: (_m(*_SINGULAR),)),
    # det
    *_matrix_edges("det"),
    _non_square("det"),
    _edge("det", "singular", lambda: (_m(*_SINGULAR),)),
    _edge("det", "rational_entries", lambda: (_halves(),)),
    # inverse
    *_matrix_edges("inverse"),
    _non_square("inverse"),
    _edge("inverse", "singular", lambda: (_m((1, 2), (2, 4)),)),
    _edge("inverse", "rational_entries", lambda: (_halves(),)),
    # nullspace
    *_matrix_edges("nullspace"),
    _edge("nullspace", "full_rank", lambda: (_m(*_CLEAN),)),
    _edge("nullspace", "zero_matrix", lambda: (_m((0, 0, 0), (0, 0, 0)),)),
    _edge("nullspace", "dependent_rows", lambda: (_m(*_SINGULAR),)),
    # charpoly
    *_type_edges("charpoly", T),
    _edge("charpoly", "type_tuple", lambda: (((1, 2), (3, 4)), T)),
    _edge("charpoly", "over_limit_entries", lambda: (_over_limit_entries(), T)),
    _edge("charpoly", "over_limit_integer", lambda: (_over_limit_integer(), T)),
    _edge("charpoly", "empty", lambda: (mf.Matrix(()), T)),
    _non_square("charpoly", T),
    _edge("charpoly", "variable_str", lambda: (_m(*_CLEAN), "x")),
    _edge("charpoly", "variable_none", lambda: (_m(*_CLEAN), None)),
    _edge("charpoly", "variable_integer_domain",
          lambda: (_m(*_CLEAN), mf.Symbol("t", mf.Integers, uid=str(uuid.UUID(int=101))))),
    _edge("charpoly", "variable_with_assumption",
          lambda: (_m(*_CLEAN), mf.Symbol("t", mf.Reals, (mf.Assumption("positive", mf.Truth.TRUE),),
                                          uid=str(uuid.UUID(int=102))))),
    _edge("charpoly", "rational_entries", lambda: (_halves(), T)),
    # real_eigenvalues
    *_matrix_edges("real_eigenvalues"),
    _non_square("real_eigenvalues"),
    _edge("real_eigenvalues", "complex_only", lambda: (_m((0, 1), (-1, 0)),)),
    _edge("real_eigenvalues", "repeated", lambda: (_m((2, 0), (0, 2)),)),
    _edge("real_eigenvalues", "irrational_quadratic", lambda: (_m((-3, 1), (1, 3)),)),
    _edge("real_eigenvalues", "rational_roots", lambda: (_m((2, 1), (1, 2)),)),
    _edge("real_eigenvalues", "rational_entries", lambda: (_symmetric_thirds(),)),
)
