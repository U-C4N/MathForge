"""Linear algebra probes: determinants, inverses, kernels, systems, eigenvalues, powers and factorizations."""

from fractions import Fraction

from . import Probe, atom, coefficients, exact, exact_tuple, expression_poly, signed_sqrt, symbols

GROUP = "linalg"
A = ((2, 1, 0), (1, 3, 1), (0, 1, 4))
SINGULAR = ((1, 2), (2, 4))
# A fixed 8 x 8 integer matrix; its determinant, -149214, was checked by exact elimination over Fractions.
EIGHT = tuple(tuple((i * 7 + j * 3 + (i * j) % 5) % 11 + (1 if i == j else 0) for j in range(8)) for i in range(8))


def _rows(value):
    """A Matrix's rows (or any rows of exact numbers) as tuples of Fractions."""
    return tuple(exact_tuple(row) for row in getattr(value, "rows", value))


def _product(left, right):
    left, right = _rows(left), _rows(right)
    return tuple(tuple(sum((row[k] * right[k][j] for k in range(len(right))), Fraction(0))
                       for j in range(len(right[0]))) for row in left)


def _identity(n):
    return tuple(tuple(Fraction(int(i == j)) for j in range(n)) for i in range(n))


def _power(rows, exponent):
    result = _identity(len(rows))
    for _ in range(exponent):
        result = _product(result, rows)
    return result


def _apply(rows, vector):
    return tuple(sum((Fraction(a) * exact(b) for a, b in zip(row, vector)), Fraction(0)) for row in rows)


def _nullspace(value):
    return len(value) == 1 and any(exact(item) != 0 for item in value[0]) and _apply(SINGULAR, value[0]) == (0, 0)


def _linear_system(mf):
    x, y, z = symbols(mf, "x y z")
    return mf.solve_system(mf.LinearSystem(mf.Matrix(A), (1, 2, 3), (x, y, z)))


def _symbolic_determinant(mf):
    x, = symbols(mf, "x")
    return mf.det(mf.Matrix(((x, 1), (0, 1))))


def _lu(value):
    lower, upper, n = _rows(value.L), _rows(value.U), len(A)
    unit_lower = all(lower[i][i] == 1 and all(lower[i][j] == 0 for j in range(i + 1, n)) for i in range(n))
    triangular = all(upper[i][j] == 0 for i in range(n) for j in range(i))
    return unit_lower and triangular and _product(value.P, A) == _product(lower, upper)


PROBES = (
    Probe("linalg_det_3x3", GROUP, "What is the determinant of [[2, 1, 0], [1, 3, 1], [0, 1, 4]]?",
          lambda mf: mf.det(mf.Matrix(A)), lambda v: exact(v) == 18),
    Probe("linalg_det_8x8", GROUP, "What is the determinant of a fixed 8 x 8 integer matrix?",
          lambda mf: mf.det(mf.Matrix(EIGHT)), lambda v: exact(v) == -149214),
    Probe("linalg_inverse_3x3", GROUP, "Invert [[2, 1, 0], [1, 3, 1], [0, 1, 4]].",
          lambda mf: mf.inverse(mf.Matrix(A)), lambda v: _product(A, v) == _identity(3)),
    Probe("linalg_inverse_singular", GROUP, "Invert the singular matrix [[1, 2], [2, 4]].",
          lambda mf: mf.inverse(mf.Matrix(SINGULAR))),
    Probe("linalg_rank_singular", GROUP, "What is the rank of [[1, 2], [2, 4]]?",
          lambda mf: mf.rank(mf.Matrix(SINGULAR)), lambda v: exact(v) == 1),
    Probe("linalg_nullspace_singular", GROUP, "Find a basis of the null space of [[1, 2], [2, 4]].",
          lambda mf: mf.nullspace(mf.Matrix(SINGULAR)), _nullspace),
    Probe("linalg_linear_system", GROUP, "Solve A (x, y, z) = (1, 2, 3) for A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]].",
          _linear_system, lambda v: _apply(A, v.particular) == (1, 2, 3) and not v.basis),
    Probe("linalg_eigenvalues_rational", GROUP, "Find the real eigenvalues of [[2, 1], [1, 2]].",
          lambda mf: mf.real_eigenvalues(mf.Matrix(((2, 1), (1, 2)))),
          lambda v: sorted((exact(item.root), item.multiplicity) for item in v) == [(1, 1), (3, 1)]),
    Probe("linalg_eigenvalues_irrational", GROUP, "Find the real eigenvalues of [[0, 2], [1, 0]].",
          lambda mf: mf.real_eigenvalues(mf.Matrix(((0, 2), (1, 0)))),
          lambda v: sorted(signed_sqrt(item.root) + (item.multiplicity,) for item in v) == [(-1, 2, 1), (1, 2, 1)]),
    Probe("linalg_eigenvectors", GROUP, "Find the eigenvectors of [[2, 1, 0], [1, 3, 1], [0, 1, 4]].",
          lambda mf: mf.linalg.eigenvectors(mf.Matrix(A))),
    Probe("linalg_matrix_power_operator", GROUP, "Compute A^5 as A ** 5 for A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]].",
          lambda mf: mf.Matrix(A) ** 5, lambda v: _rows(v) == _power(A, 5)),
    Probe("linalg_matrix_power", GROUP, "Compute A^5 for A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]] with matrix_power.",
          lambda mf: mf.linalg.matrix_power(mf.Matrix(A), 5), lambda v: _rows(v) == _power(A, 5)),
    Probe("linalg_matrix_product", GROUP, "Compute A @ A for A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]].",
          lambda mf: mf.Matrix(A) @ mf.Matrix(A), lambda v: _rows(v) == ((5, 5, 1), (5, 11, 7), (1, 7, 17))),
    Probe("linalg_symbolic_determinant", GROUP, "What is the determinant of [[x, 1], [0, 1]]?", _symbolic_determinant,
          lambda v: expression_poly(v) == atom("x")),
    Probe("linalg_lu_3x3", GROUP, "Find the LU decomposition of [[2, 1, 0], [1, 3, 1], [0, 1, 4]].",
          lambda mf: mf.lu(mf.Matrix(A)), _lu),
    Probe("linalg_least_squares", GROUP, "Fit y = a + b t by least squares to (1, 1), (2, 2), (3, 4).",
          lambda mf: mf.least_squares(mf.Matrix(((1, 1), (1, 2), (1, 3))), (1, 2, 4)),
          lambda v: exact_tuple(v.particular) == (Fraction(-2, 3), Fraction(3, 2)) and not v.basis),
    Probe("linalg_charpoly_2x2", GROUP, "What is the characteristic polynomial of [[2, 1], [1, 2]]?",
          lambda mf: mf.charpoly(mf.Matrix(((2, 1), (1, 2))), symbols(mf, "x")[0]),
          lambda v: coefficients(v) == (3, -4, 1)),
)
