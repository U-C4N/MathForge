"""Exact rational matrices, affine systems, and independently replayable evidence.

Matrices retain empty dimensions.  The production eliminators and the checking
helpers share exact scalar values, but checking never calls a production solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable

from .errors import InvalidInput, UnsupportedOperation
from .limits import ResourceLimitError, computation, current_budget
from .model import Add, Mul, Pow, Rational, Reals, Symbol


_ZERO = Rational(0)
_ONE = Rational(1)
Vector = tuple[Rational, ...]
RowOperation = tuple


def _rational(value) -> Rational:
    if type(value) is int:
        value = Rational(value)
    if type(value) is not Rational:
        raise InvalidInput("Matrix and vector entries must be exact integers or rationals")
    budget = current_budget()
    if budget is not None:
        budget.check_int(value.numerator)
        budget.check_int(value.denominator)
    return value


def _collection(values, name):
    if isinstance(values, (str, bytes, dict)):
        raise InvalidInput(f"{name} must be an iterable")
    try:
        return iter(values)
    except TypeError as exc:
        raise InvalidInput(f"{name} must be an iterable") from exc


def _vector(values, name="Vector") -> Vector:
    result = []
    for value in _collection(values, name):
        current_budget().nodes()
        result.append(_rational(value))
    return tuple(result)


def _variables(values) -> tuple[Symbol, ...]:
    result = []
    identities = set()
    for variable in _collection(values, "Variables"):
        current_budget().nodes()
        if type(variable) is not Symbol:
            raise InvalidInput("System variables must be Symbols")
        if variable.uid in identities:
            raise InvalidInput("System variables must have distinct identities")
        if variable.domain != Reals or variable.assumptions:
            raise UnsupportedOperation("Systems require real variables without additional assumptions")
        identities.add(variable.uid)
        result.append(variable)
    return tuple(result)


@dataclass(frozen=True, init=False)
class Matrix:
    """Immutable rational rows with an explicit column count for empty matrices."""

    rows: tuple[Vector, ...]
    ncols: int

    def __init__(self, rows: Iterable[Iterable[Rational | int]], *, ncols: int | None = None):
        with computation():
            budget = current_budget()
            if ncols is not None and (type(ncols) is not int or ncols < 0):
                raise InvalidInput("ncols must be a nonnegative integer")
            result = []
            width = ncols
            if width is not None:
                budget.nodes(width)
            for row in _collection(rows, "Matrix rows"):
                budget.nodes()
                if width is not None:
                    budget.matrix(len(result) + 1, width)
                entries = []
                for value in _collection(row, "Matrix row"):
                    budget.nodes()
                    budget.matrix(len(result) + 1, len(entries) + 1)
                    entries.append(_rational(value))
                if width is None:
                    width = len(entries)
                if len(entries) != width:
                    raise InvalidInput("Matrix rows must have the same declared number of columns")
                budget.matrix(len(result) + 1, width)
                result.append(tuple(entries))
            object.__setattr__(self, "rows", tuple(result))
            object.__setattr__(self, "ncols", 0 if width is None else width)

    @property
    def nrows(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    def __len__(self) -> int:
        return self.nrows

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if type(key) is tuple and len(key) == 2:
            row, column = key
            if type(row) is not int or type(column) is not int:
                raise TypeError("Matrix indices must be integers")
            return self.rows[row][column]
        if type(key) is int:
            return self.rows[key]
        raise TypeError("Use an integer row index or a pair of integer indices")

    @property
    def T(self) -> Matrix:
        with computation():
            current_budget().matrix(self.ncols, self.nrows)
            current_budget().nodes(self.ncols)
            return Matrix((tuple(self.rows[i][j] for i in range(self.nrows))
                           for j in range(self.ncols)), ncols=self.nrows)

    def __add__(self, other) -> Matrix:
        with computation():
            _matrix(other)
            if self.shape != other.shape:
                raise InvalidInput("Matrix addition requires equal shapes")
            return Matrix((tuple(a + b for a, b in zip(left, right))
                           for left, right in zip(self.rows, other.rows)), ncols=self.ncols)

    def __sub__(self, other) -> Matrix:
        with computation():
            _matrix(other)
            if self.shape != other.shape:
                raise InvalidInput("Matrix subtraction requires equal shapes")
            return Matrix((tuple(a - b for a, b in zip(left, right))
                           for left, right in zip(self.rows, other.rows)), ncols=self.ncols)

    def __neg__(self) -> Matrix:
        return self * -1

    def __mul__(self, scalar) -> Matrix:
        with computation():
            scalar = _rational(scalar)
            return Matrix((tuple(value * scalar for value in row) for row in self.rows),
                          ncols=self.ncols)

    def __rmul__(self, scalar) -> Matrix:
        return self * scalar

    def __matmul__(self, other) -> Matrix:
        with computation():
            _matrix(other)
            if self.ncols != other.nrows:
                raise InvalidInput("Matrix multiplication requires matching inner dimensions")
            current_budget().matrix(self.nrows, other.ncols)
            return Matrix((tuple(_dot(row, tuple(other.rows[k][j]
                                                for k in range(other.nrows)))
                                 for j in range(other.ncols)) for row in self.rows),
                          ncols=other.ncols)

    @classmethod
    def identity(cls, size: int) -> Matrix:
        with computation():
            if type(size) is not int or size < 0:
                raise InvalidInput("Identity size must be a nonnegative integer")
            current_budget().matrix(size, size)
            return cls((tuple(_ONE if i == j else _ZERO for j in range(size))
                        for i in range(size)), ncols=size)


def _matrix(value) -> Matrix:
    if type(value) is not Matrix:
        raise InvalidInput("Expected a Matrix")
    current_budget().matrix(value.nrows, value.ncols)
    for row in value.rows:
        for entry in row:
            current_budget().nodes()
            _rational(entry)
    return value


def _dot(left, right) -> Rational:
    result = _ZERO
    for a, b in zip(left, right):
        current_budget().tick()
        result = result + a * b
    return result


@dataclass(frozen=True)
class RREFResult:
    matrix: Matrix
    pivot_columns: tuple[int, ...]

    def __post_init__(self):
        with computation():
            _matrix(self.matrix)
            pivots = []
            for column in _collection(self.pivot_columns, "Pivot columns"):
                current_budget().nodes()
                if (type(column) is not int or column < 0 or column >= self.matrix.ncols
                        or (pivots and column <= pivots[-1])):
                    raise InvalidInput("Pivot columns must be strictly increasing valid indices")
                pivots.append(column)
            if len(pivots) > self.matrix.nrows:
                raise InvalidInput("There cannot be more pivots than rows")
            object.__setattr__(self, "pivot_columns", tuple(pivots))


@dataclass(frozen=True, init=False)
class LinearSystem:
    coefficients: Matrix
    rhs: Vector
    variables: tuple[Symbol, ...]

    def __init__(self, coefficients: Matrix, rhs: Iterable[Rational | int],
                 variables: Iterable[Symbol]) -> None:
        with computation():
            coefficients = _matrix(coefficients)
            normalized_rhs = _vector(rhs, "Right hand side")
            normalized_variables = _variables(variables)
            if (len(normalized_rhs) != coefficients.nrows
                    or len(normalized_variables) != coefficients.ncols):
                raise InvalidInput("System matrix, right hand side, and variables have incompatible shapes")
            object.__setattr__(self, "coefficients", coefficients)
            object.__setattr__(self, "rhs", normalized_rhs)
            object.__setattr__(self, "variables", normalized_variables)


@dataclass(frozen=True, init=False)
class AffineSolutionSet:
    variables: tuple[Symbol, ...]
    particular: Vector
    basis: tuple[Vector, ...]

    def __init__(self, variables: Iterable[Symbol], particular: Iterable[Rational | int],
                 basis: Iterable[Iterable[Rational | int]]) -> None:
        with computation():
            normalized_variables = _variables(variables)
            normalized_particular = _vector(particular, "Particular solution")
            if len(normalized_particular) != len(normalized_variables):
                raise InvalidInput("Particular solution must have one coordinate per variable")
            normalized_basis = []
            for vector in _collection(basis, "Affine basis"):
                current_budget().nodes()
                vector = _vector(vector, "Basis vector")
                if len(vector) != len(normalized_variables):
                    raise InvalidInput("Basis vectors must have one coordinate per variable")
                current_budget().matrix(len(normalized_basis) + 1, len(normalized_variables))
                normalized_basis.append(vector)
            object.__setattr__(self, "variables", normalized_variables)
            object.__setattr__(self, "particular", normalized_particular)
            object.__setattr__(self, "basis", tuple(normalized_basis))

    @property
    def domain(self):
        return Reals


def reduce_with_trace(A: Matrix, *, limits=None) -> tuple[RREFResult, tuple[RowOperation, ...]]:
    """Gauss--Jordan reduction with a compact elementary-row-operation trace."""
    with computation(limits):
        A = _matrix(A)
        rows = [list(row) for row in A.rows]
        pivots = []
        trace = []
        pivot_row = 0

        def record(operation):
            current_budget().evidence(len(operation))
            trace.append(operation)

        for column in range(A.ncols):
            current_budget().tick()
            candidate = next((i for i in range(pivot_row, A.nrows)
                              if rows[i][column].numerator), None)
            if candidate is None:
                continue
            if candidate != pivot_row:
                rows[candidate], rows[pivot_row] = rows[pivot_row], rows[candidate]
                record(("swap", pivot_row, candidate))
            pivot = rows[pivot_row][column]
            if pivot != _ONE:
                multiplier = _ONE / pivot
                rows[pivot_row] = [value * multiplier for value in rows[pivot_row]]
                record(("scale", pivot_row, multiplier))
            for i in range(A.nrows):
                if i == pivot_row or not rows[i][column].numerator:
                    continue
                multiplier = -rows[i][column]
                rows[i] = [value + multiplier * pivot_value
                           for value, pivot_value in zip(rows[i], rows[pivot_row])]
                record(("add", i, pivot_row, multiplier))
            pivots.append(column)
            pivot_row += 1
            if pivot_row == A.nrows:
                break
        reduced = RREFResult(Matrix(rows, ncols=A.ncols), tuple(pivots))
        current_budget().evidence(A.nrows * A.ncols + len(pivots))
        return reduced, tuple(trace)


def rref(A: Matrix, *, limits=None) -> RREFResult:
    return reduce_with_trace(A, limits=limits)[0]


def rank(A: Matrix, *, limits=None) -> int:
    return len(reduce_with_trace(A, limits=limits)[0].pivot_columns)


def _nullspace_from_reduced(reduced: Matrix, pivots: tuple[int, ...], width: int) -> tuple[Vector, ...]:
    pivot_set = set(pivots)
    current_budget().matrix(width - len(pivots), width)
    basis = []
    for free in range(width):
        if free in pivot_set:
            continue
        vector = [_ZERO] * width
        vector[free] = _ONE
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced.rows[row][free]
        basis.append(tuple(vector))
    return tuple(basis)


def nullspace(A: Matrix, *, limits=None) -> tuple[Vector, ...]:
    with computation(limits):
        reduction, _ = reduce_with_trace(A)
        return _nullspace_from_reduced(reduction.matrix, reduction.pivot_columns, A.ncols)


def _int_product(a: int, b: int) -> int:
    budget = current_budget()
    budget.product(a, b)
    result = a * b
    budget.check_int(result)
    return result


def det(A: Matrix, *, limits=None) -> Rational:
    """Determinant by integer Bareiss elimination after row denominator clearing."""
    with computation(limits):
        A = _matrix(A)
        if A.nrows != A.ncols:
            raise InvalidInput("Determinant requires a square matrix")
        n = A.nrows
        if n == 0:
            return _ONE
        integers = []
        scale_product = 1
        for row in A.rows:
            scale = 1
            for value in row:
                scale = _int_product(scale // gcd(scale, value.denominator), value.denominator)
            scale_product = _int_product(scale_product, scale)
            integers.append([_int_product(value.numerator, scale // value.denominator) for value in row])
        sign = 1
        previous = 1
        for k in range(n - 1):
            candidate = next((i for i in range(k, n) if integers[i][k]), None)
            if candidate is None:
                return _ZERO
            if candidate != k:
                integers[candidate], integers[k] = integers[k], integers[candidate]
                sign = -sign
            pivot = integers[k][k]
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    numerator = (_int_product(pivot, integers[i][j])
                                 - _int_product(integers[i][k], integers[k][j]))
                    current_budget().check_int(numerator)
                    value, remainder = divmod(numerator, previous)
                    if remainder:
                        raise ArithmeticError("Bareiss elimination encountered a nonexact division")
                    integers[i][j] = value
                integers[i][k] = 0
            previous = pivot
        return Rational(sign * integers[-1][-1], scale_product)


def inverse(A: Matrix, *, limits=None) -> Matrix:
    with computation(limits):
        A = _matrix(A)
        if A.nrows != A.ncols:
            raise InvalidInput("Inverse requires a square matrix")
        n = A.nrows
        current_budget().matrix(n, 2 * n)
        augmented = Matrix((row + tuple(_ONE if i == j else _ZERO for j in range(n))
                            for i, row in enumerate(A.rows)), ncols=2 * n)
        reduction, _ = reduce_with_trace(augmented)
        if reduction.pivot_columns[:n] != tuple(range(n)):
            raise InvalidInput("Matrix is singular", code="singular_matrix")
        return Matrix((row[n:] for row in reduction.matrix.rows), ncols=n)


def _augmented(system: LinearSystem) -> Matrix:
    A = system.coefficients
    current_budget().matrix(A.nrows, A.ncols + 1)
    return Matrix((row + (value,) for row, value in zip(A.rows, system.rhs)), ncols=A.ncols + 1)


def solve_linear_system(system: LinearSystem, *, limits=None):
    """Return (solution, (augmented, reduced, pivots, row_operations))."""
    from .contracts import EmptySet

    with computation(limits):
        if type(system) is not LinearSystem:
            raise InvalidInput("Expected a LinearSystem")
        augmented = _augmented(system)
        reduction, trace = reduce_with_trace(augmented)
        width = system.coefficients.ncols
        evidence = (augmented, reduction.matrix, reduction.pivot_columns, trace)
        if width in reduction.pivot_columns:
            return EmptySet(Reals), evidence
        particular = [_ZERO] * width
        for i, pivot in enumerate(reduction.pivot_columns):
            particular[pivot] = reduction.matrix.rows[i][-1]
        basis = _nullspace_from_reduced(reduction.matrix, reduction.pivot_columns, width)
        return AffineSolutionSet(system.variables, tuple(particular), basis), evidence


def linear_system_from_equations(equations, variables, *, limits=None) -> LinearSystem:
    """Compile bounded sparse polynomial expressions, then require affine degree.

    Cancellation occurs before the degree check. Undeclared symbols are never
    interpreted as symbolic parameters, and source equations remain with callers.
    """
    from .contracts import Eq

    with computation(limits):
        variables = _variables(variables)
        indices = {variable.uid: (i, variable) for i, variable in enumerate(variables)}
        width = len(variables)
        constant = (0,) * width

        def plus(left, right):
            result = dict(left)
            current_budget().nodes(len(left))
            for monomial, coefficient in right.items():
                current_budget().tick()
                value = result.get(monomial, _ZERO) + coefficient
                if value.numerator:
                    result[monomial] = value
                    current_budget().nodes()
                else:
                    result.pop(monomial, None)
            return result

        def times(left, right):
            result = {}
            for first, a in left.items():
                for second, b in right.items():
                    current_budget().nodes(width + 1)
                    monomial = tuple(i + j for i, j in zip(first, second))
                    value = result.get(monomial, _ZERO) + a * b
                    if value.numerator:
                        result[monomial] = value
                    else:
                        result.pop(monomial, None)
            return result

        def convert(expression, depth=0):
            budget = current_budget()
            budget.nodes()
            if depth > budget.limits.max_depth:
                raise ResourceLimitError("Affine expression exceeds the depth limit")
            if type(expression) is Rational:
                return {constant: _rational(expression)} if expression.numerator else {}
            if type(expression) is Symbol:
                entry = indices.get(expression.uid)
                if entry is None:
                    raise UnsupportedOperation("Equation contains an undeclared free symbol")
                index, declared = entry
                if declared != expression:
                    raise InvalidInput("The same symbol identity has conflicting metadata")
                monomial = [0] * width
                monomial[index] = 1
                return {tuple(monomial): _ONE}
            if type(expression) is Add:
                result = {}
                for term in expression.terms:
                    result = plus(result, convert(term, depth + 1))
                return result
            if type(expression) is Mul:
                result = {constant: _ONE}
                for factor in expression.factors:
                    result = times(result, convert(factor, depth + 1))
                return result
            if type(expression) is Pow:
                exponent = expression.exponent
                budget.check_int(exponent)
                if exponent == 0:
                    return {constant: _ONE}
                factor = convert(expression.base, depth + 1)
                result = {constant: _ONE}
                while exponent:
                    budget.tick()
                    if exponent & 1:
                        result = times(result, factor)
                    exponent >>= 1
                    if exponent:
                        factor = times(factor, factor)
                return result
            raise UnsupportedOperation("Linear equations require rational polynomial coefficients")

        coefficients = []
        rhs = []
        for equation in _collection(equations, "Equations"):
            current_budget().nodes()
            if type(equation) is not Eq:
                raise InvalidInput("System input must contain Eq values")
            negative_rhs = {monomial: -value for monomial, value in convert(equation.rhs).items()}
            polynomial = plus(convert(equation.lhs), negative_rhs)
            row = [_ZERO] * width
            value = _ZERO
            for monomial, coefficient in polynomial.items():
                degree = sum(monomial)
                if degree > 1:
                    raise UnsupportedOperation("Equation is not affine after exact cancellation")
                if degree == 0:
                    value = -coefficient
                else:
                    row[monomial.index(1)] = coefficient
            current_budget().matrix(len(coefficients) + 1, width)
            coefficients.append(tuple(row))
            rhs.append(value)
        return LinearSystem(Matrix(coefficients, ncols=width), tuple(rhs), variables)


def replay_row_operations(A: Matrix, row_operations, *, limits=None) -> Matrix:
    """Check and replay only invertible elementary row operations."""
    with computation(limits):
        A = _matrix(A)
        rows = [list(row) for row in A.rows]
        for operation in _collection(row_operations, "Row operations"):
            if type(operation) is not tuple or not operation or type(operation[0]) is not str:
                raise InvalidInput("Malformed row operation")
            current_budget().evidence(len(operation))
            name = operation[0]
            if name not in ("swap", "scale", "add"):
                raise InvalidInput("Unknown row operation")
            expected = 4 if name == "add" else 3
            if len(operation) != expected:
                raise InvalidInput("Incorrect row operation arity")
            indices = operation[1:3] if name != "scale" else operation[1:2]
            if any(type(i) is not int or i < 0 or i >= A.nrows for i in indices):
                raise InvalidInput("Invalid row index")
            if name == "swap":
                _, first, second = operation
                rows[first], rows[second] = rows[second], rows[first]
            elif name == "scale":
                _, row, factor = operation
                if type(factor) is not Rational or not factor.numerator:
                    raise InvalidInput("Row scaling requires a nonzero Rational")
                rows[row] = [factor * value for value in rows[row]]
            else:
                _, target, source, factor = operation
                if target == source or type(factor) is not Rational:
                    raise InvalidInput("Row addition requires distinct rows and a Rational factor")
                rows[target] = [value + factor * source_value
                                for value, source_value in zip(rows[target], rows[source])]
        return Matrix(rows, ncols=A.ncols)


def is_rref(A: Matrix, pivot_columns, *, limits=None) -> bool:
    with computation(limits):
        if type(A) is not Matrix or type(pivot_columns) is not tuple:
            return False
        _matrix(A)
        actual = []
        found_zero = False
        for i, row in enumerate(A.rows):
            current_budget().tick()
            pivot = next((j for j, value in enumerate(row) if value.numerator), None)
            if pivot is None:
                found_zero = True
                continue
            if found_zero or (actual and pivot <= actual[-1]) or row[pivot] != _ONE:
                return False
            if any(A.rows[k][pivot].numerator for k in range(A.nrows) if k != i):
                return False
            actual.append(pivot)
        return (all(type(column) is int for column in pivot_columns)
                and tuple(actual) == pivot_columns)


def verify_row_reduction(A: Matrix, reduction: RREFResult, row_operations, *, limits=None) -> bool:
    with computation(limits):
        try:
            return (type(reduction) is RREFResult
                    and replay_row_operations(A, row_operations) == reduction.matrix
                    and is_rref(reduction.matrix, reduction.pivot_columns))
        except (InvalidInput, TypeError, ValueError, IndexError, ZeroDivisionError):
            return False


def _checking_rank(rows, ncols: int) -> int:
    """Independent forward elimination, not the production Gauss--Jordan path."""
    rows = [list(row) for row in rows]
    count = 0
    for column in range(ncols):
        current_budget().tick()
        pivot_index = None
        for i in range(count, len(rows)):
            if rows[i][column].numerator:
                pivot_index = i
                break
        if pivot_index is None:
            continue
        rows[count], rows[pivot_index] = rows[pivot_index], rows[count]
        pivot = rows[count][column]
        for i in range(count + 1, len(rows)):
            if not rows[i][column].numerator:
                continue
            factor = rows[i][column] / pivot
            for j in range(column, ncols):
                rows[i][j] = rows[i][j] - factor * rows[count][j]
        count += 1
        if count == len(rows):
            break
    return count


def check_rank(A: Matrix, value, *, limits=None) -> bool:
    with computation(limits):
        A = _matrix(A)
        return type(value) is int and value == _checking_rank(A.rows, A.ncols)


def check_nullspace(A: Matrix, basis, *, limits=None) -> bool:
    with computation(limits):
        A = _matrix(A)
        if type(basis) is not tuple:
            return False
        if any(type(vector) is not tuple or len(vector) != A.ncols
               or any(type(value) is not Rational for value in vector) for vector in basis):
            return False
        current_budget().matrix(len(basis), A.ncols)
        for vector in basis:
            for value in vector:
                current_budget().nodes()
                _rational(value)
        if any(_dot(row, vector).numerator for row in A.rows for vector in basis):
            return False
        return (len(basis) == A.ncols - _checking_rank(A.rows, A.ncols)
                and _checking_rank(basis, A.ncols) == len(basis))


def check_determinant(A: Matrix, value, *, limits=None) -> bool:
    """Check a determinant with rational Gaussian elimination, never Bareiss."""
    with computation(limits):
        A = _matrix(A)
        if type(value) is not Rational or A.nrows != A.ncols:
            return False
        rows = [list(row) for row in A.rows]
        product = _ONE
        for column in range(A.ncols):
            current_budget().tick()
            pivot_index = next((i for i in range(column, A.nrows)
                                if rows[i][column].numerator), None)
            if pivot_index is None:
                return value == _ZERO
            if pivot_index != column:
                rows[column], rows[pivot_index] = rows[pivot_index], rows[column]
                product = -product
            pivot = rows[column][column]
            product = product * pivot
            for i in range(column + 1, A.nrows):
                factor = rows[i][column] / pivot
                for j in range(column + 1, A.ncols):
                    rows[i][j] = rows[i][j] - factor * rows[column][j]
        return product == value


def verify_linear_solution(system: LinearSystem, solution, evidence, *, limits=None) -> bool:
    """Check source linkage, invertible row replay, membership and completeness."""
    from .contracts import EmptySet

    with computation(limits):
        try:
            if type(system) is not LinearSystem or type(evidence) is not tuple or len(evidence) != 4:
                return False
            augmented, reduced, pivots, row_operations = evidence
            if type(reduced) is not Matrix or augmented != _augmented(system):
                return False
            if not verify_row_reduction(augmented, RREFResult(reduced, pivots), row_operations):
                return False
            width = system.coefficients.ncols
            inconsistent = width in pivots
            if type(solution) is EmptySet:
                return solution.domain == Reals and inconsistent
            if (type(solution) is not AffineSolutionSet or inconsistent
                    or solution.variables != system.variables):
                return False
            if any(_dot(row, solution.particular) != rhs
                   for row, rhs in zip(system.coefficients.rows, system.rhs)):
                return False
            if any(_dot(row, vector).numerator
                   for row in system.coefficients.rows for vector in solution.basis):
                return False
            return (len(solution.basis) == width - len(pivots)
                    and _checking_rank(solution.basis, width) == len(solution.basis))
        except (InvalidInput, TypeError, ValueError, IndexError, ZeroDivisionError):
            return False
