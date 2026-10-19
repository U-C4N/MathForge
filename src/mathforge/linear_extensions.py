"""Rational subspaces, singular-safe PLU and complete least-squares families."""

from dataclasses import dataclass

from .errors import InvalidInput
from .limits import computation, current_budget
from .matrices import Matrix, RREFResult, _vector, reduce_with_trace, verify_row_reduction
from .model import Rational
from .probability import _rational

ZERO, ONE = Rational(0), Rational(1)


def _matrix(value):
    if type(value) is not Matrix or type(value.rows) is not tuple or type(value.ncols) is not int or value.ncols < 0:
        raise InvalidInput("Expected a canonical rational Matrix")
    current_budget().matrix(len(value.rows), value.ncols)
    current_budget().nodes(len(value.rows)+value.ncols)
    for row in value.rows:
        current_budget().nodes()
        if type(row) is not tuple or len(row) != value.ncols:
            raise InvalidInput("Malformed matrix rows")
        for entry in row:
            current_budget().nodes()
            if type(entry) is not Rational:
                raise InvalidInput("Matrix entries must be canonical rationals")
            _rational(entry)
    return value


def _dot(left, right):
    value = ZERO
    for a, b in zip(left, right):
        current_budget().tick()
        value += a*b
    return value


def _tuple_basis(basis, width):
    if isinstance(basis, (str, bytes, dict)):
        raise InvalidInput("Basis must be an iterable of vectors")
    result = []
    for row in basis:
        current_budget().nodes()
        current_budget().matrix(len(result)+1, width)
        vector = _vector(row)
        if len(vector) != width:
            raise InvalidInput("Basis vector has incorrect dimension")
        result.append(vector)
    return tuple(result)


@dataclass(frozen=True)
class LUDecomposition:
    """Permutation convention P A = L U, including singular square matrices."""

    P: Matrix
    L: Matrix
    U: Matrix

    def __post_init__(self):
        with computation():
            for matrix in (self.P, self.L, self.U):
                _matrix(matrix)
            if self.P.nrows != self.P.ncols or self.P.shape != self.L.shape or self.P.shape != self.U.shape:
                raise InvalidInput("PLU factors must have matching square dimensions")


def trusted_lu(P, L, U):
    """An LUDecomposition from three canonical square Matrices of one size, with no validation: equal to
    ``LUDecomposition(P, L, U)``, for the lean value path, whose factors certify.linalg.lu has checked."""
    value = object.__new__(LUDecomposition)
    object.__setattr__(value, "P", P)
    object.__setattr__(value, "L", L)
    object.__setattr__(value, "U", U)
    return value


@dataclass(frozen=True)
class LeastSquaresResult:
    """All minimizers x=particular+sum(t_i*basis_i); residual is b-Ax."""

    particular: tuple[Rational, ...]
    basis: tuple[tuple[Rational, ...], ...]
    residual: tuple[Rational, ...]

    def __post_init__(self):
        with computation():
            particular = _vector(self.particular)
            basis = _tuple_basis(self.basis, len(particular))
            residual = _vector(self.residual)
            object.__setattr__(self, "particular", particular)
            object.__setattr__(self, "basis", basis)
            object.__setattr__(self, "residual", residual)


def trace(A):
    with computation():
        A = _matrix(A)
        if A.nrows != A.ncols:
            raise InvalidInput("Trace requires a square matrix")
        result = ZERO
        for i in range(A.nrows):
            current_budget().tick()
            result += A.rows[i][i]
        return result


def _space(A, columns):
    A = _matrix(A)
    reduced, evidence = reduce_with_trace(A)
    pivots = reduced.pivot_columns
    if columns:
        current_budget().matrix(len(pivots), A.nrows)
        current_budget().nodes(len(pivots)*A.nrows)
        result = tuple(tuple(row[j] for row in A.rows) for j in pivots)
    else:
        current_budget().nodes(len(pivots)*A.ncols)
        result = reduced.matrix.rows[:len(pivots)]
    return result, (reduced, evidence)


def rowspace(A):
    with computation():
        return _space(A, False)[0]


def columnspace(A):
    with computation():
        return _space(A, True)[0]


def lu(A):
    with computation():
        A = _matrix(A)
        if A.nrows != A.ncols:
            raise InvalidInput("LU requires a square matrix")
        n = A.nrows
        current_budget().nodes(3*n*n)
        U = [list(row) for row in A.rows]
        L = [[ONE if i == j else ZERO for j in range(n)] for i in range(n)]
        P = [[ONE if i == j else ZERO for j in range(n)] for i in range(n)]
        for k in range(n):
            current_budget().tick()
            candidate = None
            for i in range(k, n):
                current_budget().tick()
                if U[i][k] != ZERO:
                    candidate = i
                    break
            if candidate is None:
                continue
            if candidate != k:
                U[k], U[candidate] = U[candidate], U[k]
                P[k], P[candidate] = P[candidate], P[k]
                for j in range(k):
                    current_budget().tick()
                    L[k][j], L[candidate][j] = L[candidate][j], L[k][j]
            for i in range(k+1, n):
                current_budget().tick()
                multiplier = U[i][k] / U[k][k]
                L[i][k] = multiplier
                U[i][k] = ZERO
                for j in range(k+1, n):
                    current_budget().tick()
                    U[i][j] -= multiplier * U[k][j]
        return LUDecomposition(Matrix(P, ncols=n), Matrix(L, ncols=n), Matrix(U, ncols=n))


def _normal_equations(A, b):
    n = A.ncols
    current_budget().matrix(n, n)
    current_budget().nodes(A.nrows*n+n*n+n)
    columns = tuple(tuple(row[j] for row in A.rows) for j in range(n))
    normal = Matrix((tuple(_dot(a, c) for c in columns) for a in columns), ncols=n)
    rhs = tuple(_dot(column, b) for column in columns)
    return normal, rhs


def _least_squares(A, b):
    A, b = _matrix(A), _vector(b)
    if len(b) != A.nrows:
        raise InvalidInput("Right hand side must match matrix row count")
    n = A.ncols
    normal, rhs = _normal_equations(A, b)
    augmented = Matrix((row+(scalar,) for row, scalar in zip(normal.rows, rhs)), ncols=n+1)
    reduced, steps = reduce_with_trace(augmented)
    if n in reduced.pivot_columns:
        raise ArithmeticError("Exact normal equations unexpectedly inconsistent")
    current_budget().nodes(n)
    particular = [ZERO]*n
    for i, pivot in enumerate(reduced.pivot_columns):
        current_budget().tick()
        particular[pivot] = reduced.matrix.rows[i][-1]
    pivots = set(reduced.pivot_columns)
    current_budget().matrix(n-len(pivots), n)
    current_budget().nodes((n-len(pivots))*n)
    basis = []
    for free in range(n):
        current_budget().tick()
        if free in pivots:
            continue
        vector = [ZERO]*n
        vector[free] = ONE
        for i, pivot in enumerate(reduced.pivot_columns):
            current_budget().tick()
            vector[pivot] = -reduced.matrix.rows[i][free]
        basis.append(tuple(vector))
    current_budget().nodes(A.nrows)
    residual = tuple(target-_dot(row, particular) for row, target in zip(A.rows, b))
    return LeastSquaresResult(tuple(particular), tuple(basis), residual), (normal, rhs, reduced, steps)


def least_squares(A, b):
    with computation():
        return _least_squares(A, b)[0]


ARGUMENTS = {
    "trace": {"A"}, "rowspace": {"A"}, "columnspace": {"A"},
    "lu": {"A"}, "least_squares": {"A", "b"},
}
TYPES = {"LUDecomposition": LUDecomposition, "LeastSquaresResult": LeastSquaresResult}


def produce(op, args):
    with computation():
        if op not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[op]:
            raise InvalidInput("Unknown linear operation or incorrect arguments")
        if op in ("rowspace", "columnspace"):
            return _space(args["A"], op == "columnspace")
        if op == "least_squares":
            return _least_squares(args["A"], args["b"])
        return globals()[op](**args), ()


def _rank(rows, width):
    """Independent forward elimination, used only by the checker."""
    current_budget().matrix(len(rows), width)
    current_budget().nodes(len(rows)*width)
    rows = [list(row) for row in rows]
    rank = 0
    for column in range(width):
        current_budget().tick()
        candidate = None
        for i in range(rank, len(rows)):
            current_budget().tick()
            if rows[i][column] != ZERO:
                candidate = i
                break
        if candidate is None:
            continue
        rows[rank], rows[candidate] = rows[candidate], rows[rank]
        for i in range(rank+1, len(rows)):
            current_budget().tick()
            scale = rows[i][column] / rows[rank][column]
            for j in range(column, width):
                current_budget().tick()
                rows[i][j] -= scale * rows[rank][j]
        rank += 1
    return rank


def _canonical_vector(value, width):
    if type(value) is not tuple or len(value) != width:
        return False
    for entry in value:
        current_budget().nodes()
        if type(entry) is not Rational:
            return False
        _rational(entry)
    return True


def check_operation(op, args, value, evidence):
    """Verify identities and completeness without calling production algorithms."""
    with computation():
        try:
            if (op not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[op]
                    or type(evidence) is not tuple):
                return False
            A = _matrix(args["A"])
            if op == "trace":
                if A.nrows != A.ncols or evidence != () or type(value) is not Rational:
                    return False
                answer = ZERO
                for i, row in enumerate(A.rows):
                    current_budget().tick()
                    answer += row[i]
                return answer == value
            if op in ("rowspace", "columnspace"):
                if len(evidence) != 2 or type(value) is not tuple:
                    return False
                reduced, steps = evidence
                if type(reduced) is not RREFResult:
                    return False
                _matrix(reduced.matrix)
                if not verify_row_reduction(A, reduced, steps):
                    return False
                width = A.nrows if op == "columnspace" else A.ncols
                if len(value) != len(reduced.pivot_columns):
                    return False
                for vector in value:
                    current_budget().tick()
                    if not _canonical_vector(vector, width):
                        return False
                if op == "rowspace":
                    return value == reduced.matrix.rows[:len(reduced.pivot_columns)]
                for vector, pivot in zip(value, reduced.pivot_columns):
                    for i, row in enumerate(A.rows):
                        current_budget().tick()
                        if vector[i] != row[pivot]:
                            return False
                return True
            if op == "lu":
                if type(value) is not LUDecomposition or evidence != () or A.nrows != A.ncols:
                    return False
                n = A.nrows
                P, L, U = value.P, value.L, value.U
                for matrix in (P, L, U):
                    _matrix(matrix)
                    if matrix.shape != A.shape:
                        return False
                seen = set()
                for i in range(n):
                    position = None
                    for j in range(n):
                        current_budget().tick()
                        if P.rows[i][j] == ONE:
                            if position is not None:
                                return False
                            position = j
                        elif P.rows[i][j] != ZERO:
                            return False
                        if i < j and L.rows[i][j] != ZERO:
                            return False
                        if i == j and L.rows[i][j] != ONE:
                            return False
                        if i > j and U.rows[i][j] != ZERO:
                            return False
                    if position is None or position in seen:
                        return False
                    seen.add(position)
                    for j in range(n):
                        product = ZERO
                        for k in range(n):
                            current_budget().tick()
                            product += L.rows[i][k] * U.rows[k][j]
                        if product != A.rows[position][j]:
                            return False
                return True
            if type(value) is not LeastSquaresResult or len(evidence) != 4:
                return False
            b = _vector(args["b"])
            if len(b) != A.nrows or not _canonical_vector(value.particular, A.ncols):
                return False
            if not _canonical_vector(value.residual, A.nrows) or type(value.basis) is not tuple:
                return False
            current_budget().matrix(len(value.basis), A.ncols)
            for vector in value.basis:
                current_budget().tick()
                if not _canonical_vector(vector, A.ncols):
                    return False
            normal, rhs, reduced, steps = evidence
            _matrix(normal)
            if type(reduced) is not RREFResult:
                return False
            _matrix(reduced.matrix)
            if normal.shape != (A.ncols, A.ncols) or not _canonical_vector(rhs, A.ncols):
                return False
            # Independently check every Gram matrix/right-hand-side entry.
            for i in range(A.ncols):
                target = ZERO
                for row, scalar in zip(A.rows, b):
                    current_budget().tick()
                    target += row[i] * scalar
                if target != rhs[i]:
                    return False
                for j in range(A.ncols):
                    target = ZERO
                    for row in A.rows:
                        current_budget().tick()
                        target += row[i] * row[j]
                    if target != normal.rows[i][j]:
                        return False
            augmented = Matrix((row+(scalar,) for row, scalar in zip(normal.rows, rhs)), ncols=A.ncols+1)
            if not verify_row_reduction(augmented, reduced, steps):
                return False
            for row, scalar, residual in zip(A.rows, b, value.residual):
                current_budget().tick()
                if scalar - _dot(row, value.particular) != residual:
                    return False
                for vector in value.basis:
                    if _dot(row, vector) != ZERO:
                        return False
            for j in range(A.ncols):
                total = ZERO
                for row, residual in zip(A.rows, value.residual):
                    current_budget().tick()
                    total += row[j]*residual
                if total != ZERO:
                    return False
            return (len(value.basis) == A.ncols - _rank(A.rows, A.ncols)
                    and _rank(value.basis, A.ncols) == len(value.basis))
        except (InvalidInput, TypeError, ValueError, KeyError, AttributeError, IndexError, ZeroDivisionError):
            return False
