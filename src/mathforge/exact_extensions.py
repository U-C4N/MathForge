"""Exact partial fractions and rational spectra with source-linked checks.

Partial fractions split over Q, retain every original domain exclusion, and
never silently split irreducible real or complex factors. Characteristic
polynomials use division-free Berkowitz; their checker instead evaluates
det(t I - A) by rational Gaussian elimination at distinct exact points.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from .algebraic import RootRecord
from .errors import InvalidInput, UnsupportedOperation
from .limits import computation, current_budget
from .matrices import Matrix
from .model import Rational, Reals, Symbol
from .polynomial import Polynomial
from .rational_function import RationalFunction, normalize_exclusions


def _factor_key(polynomial):
    return (polynomial.degree, tuple((c.numerator, c.denominator) for c in polynomial.coefficients))


def _variable(variable):
    if type(variable) is not Symbol:
        raise InvalidInput("A characteristic polynomial requires an explicit Symbol")
    if variable.domain is not Reals or variable.assumptions:
        raise UnsupportedOperation("Only unconstrained real symbols are supported")


@dataclass(frozen=True)
class PartialFractionTerm:
    numerator: Polynomial
    factor: Polynomial
    power: int

    def __post_init__(self):
        with computation() as budget:
            budget.inspect((self.numerator, self.factor, self.power))
            if type(self.numerator) is not Polynomial or type(self.factor) is not Polynomial:
                raise InvalidInput("Partial-fraction terms require polynomial numerator and factor")
            if self.numerator.variable != self.factor.variable:
                raise InvalidInput("Partial-fraction numerator and factor variables must agree")
            if (self.factor.degree <= 0 or self.factor.leading_coefficient != Rational(1)
                    or self.numerator.is_zero or self.numerator.degree >= self.factor.degree):
                raise InvalidInput("A partial-fraction term requires a nonzero proper numerator and monic factor")
            if type(self.power) is not int or self.power <= 0:
                raise InvalidInput("A denominator power must be a positive integer")


@dataclass(frozen=True)
class PartialFractionDecomposition:
    polynomial: Polynomial
    terms: tuple[PartialFractionTerm, ...]
    excluded: tuple[Polynomial, ...]

    def __post_init__(self):
        with computation() as budget:
            if type(self.polynomial) is not Polynomial:
                raise InvalidInput("The polynomial part must be a Polynomial")
            if type(self.terms) not in (tuple, list) or type(self.excluded) not in (tuple, list):
                raise InvalidInput("Terms and domain exclusions must be finite sequences")
            terms, excluded = tuple(self.terms), tuple(self.excluded)
            budget.inspect((self.polynomial, terms, excluded))
            if any(type(term) is not PartialFractionTerm for term in terms):
                raise InvalidInput("Expected PartialFractionTerm records")
            if any(term.factor.variable != self.polynomial.variable for term in terms):
                raise InvalidInput("Every term must use the polynomial part's variable")
            keys = tuple((_factor_key(term.factor), term.power) for term in terms)
            if any(left >= right for left, right in zip(keys, keys[1:])):
                raise InvalidInput("Partial-fraction terms must be distinct and canonically ordered")
            if any(type(p) is not Polynomial or p.variable != self.polynomial.variable for p in excluded):
                raise InvalidInput("Domain exclusions require polynomials in the same variable")
            if normalize_exclusions(excluded) != excluded:
                raise InvalidInput("Domain exclusions must be canonical")
            object.__setattr__(self, "terms", terms)
            object.__setattr__(self, "excluded", excluded)

    @property
    def variable(self):
        return self.polynomial.variable

    def to_rational_function(self, *, limits=None):
        """Recombine the value while retaining all original excluded points."""
        with computation(limits):
            powers = {}
            for term in self.terms:
                powers[term.factor] = max(powers.get(term.factor, 0), term.power)
            denominator = Polynomial((1,), self.variable)
            for factor, power in powers.items():
                denominator = denominator * factor ** power
            numerator = self.polynomial * denominator
            for term in self.terms:
                numerator = numerator + term.numerator * denominator.exact_div(term.factor ** term.power)
            return RationalFunction._normalize(numerator, denominator, self.excluded)


def _apart(expression):
    from .polynomial_exact import factor_with_evidence

    if type(expression) is not RationalFunction:
        raise InvalidInput("apart requires a domain-preserving RationalFunction")
    _validate_rational_function(expression)
    _variable(expression.variable)
    factorization, factor_evidence = factor_with_evidence(expression.denominator)
    quotient, remainder = expression.numerator.divmod(expression.denominator)
    terms = []
    for entry in factorization.factors:
        factor, multiplicity = entry.polynomial, entry.multiplicity
        denominator_power = factor ** multiplicity
        complement = expression.denominator.exact_div(denominator_power)
        common, inverse, _ = complement.xgcd(denominator_power)
        if common != Polynomial((1,), expression.variable):
            raise ArithmeticError("Factorization did not provide coprime denominator powers")
        residual = (remainder * inverse) % denominator_power
        for offset in range(multiplicity):
            current_budget().tick()
            residual, coefficient = residual.divmod(factor)
            if not coefficient.is_zero:
                terms.append(PartialFractionTerm(coefficient, factor, multiplicity - offset))
        if not residual.is_zero:
            raise ArithmeticError("Partial-fraction numerator exceeded its denominator degree")
    terms.sort(key=lambda term: (_factor_key(term.factor), term.power))
    value = PartialFractionDecomposition(quotient, tuple(terms), expression.excluded)
    evidence = (factorization, factor_evidence)
    current_budget().inspect(evidence, evidence=True)
    return value, evidence


def _square_matrix(A):
    if type(A) is not Matrix or A.nrows != A.ncols:
        raise InvalidInput("A square rational Matrix is required")
    budget = current_budget()
    budget.inspect(A)
    budget.matrix(A.nrows, A.ncols)
    if type(A.rows) is not tuple or type(A.ncols) is not int or A.ncols < 0:
        raise InvalidInput("Malformed rational matrix")
    for row in A.rows:
        if type(row) is not tuple or len(row) != A.ncols:
            raise InvalidInput("Malformed rational matrix row")
        for value in row:
            if (type(value) is not Rational or type(value.numerator) is not int
                    or type(value.denominator) is not int or value.denominator <= 0
                    or gcd(value.numerator, value.denominator) != 1):
                raise InvalidInput("Matrix entries must be canonical rational values")
    return A


def _dot(left, right):
    total = Rational(0)
    for a, b in zip(left, right):
        current_budget().tick()
        total = total + a * b
    return total


def _charpoly(A, variable):
    """Berkowitz Toeplitz products for successive trailing principal blocks."""
    _square_matrix(A)
    _variable(variable)
    budget = current_budget()
    coefficients = (Rational(1),)  # Descending powers, initially det of 0 x 0.
    for start in range(A.nrows - 1, -1, -1):
        size = A.nrows - start
        sequence = [Rational(1), -A.rows[start][start]]
        row = A.rows[start][start + 1:]
        vector = tuple(A.rows[i][start] for i in range(start + 1, A.nrows))
        budget.nodes(size + len(vector))
        for exponent in range(size - 1):
            sequence.append(-_dot(row, vector))
            if exponent < size - 2:
                vector = tuple(_dot(A.rows[i][start + 1:], vector)
                               for i in range(start + 1, A.nrows))
                budget.nodes(len(vector))
        next_coefficients = []
        for index in range(size + 1):
            next_coefficients.append(sum((sequence[index - j] * coefficients[j]
                                          for j in range(min(index + 1, len(coefficients)))), Rational(0)))
        budget.nodes(len(next_coefficients))
        coefficients = tuple(next_coefficients)
    return Polynomial(tuple(reversed(coefficients)), variable)


def _det_at(A, point):
    """Checking-only rational Gaussian determinant, independent of Berkowitz."""
    n = A.nrows
    budget = current_budget()
    budget.matrix(n, n)
    budget.nodes(n * n)
    rows = [[(point if i == j else Rational(0)) - A.rows[i][j]
             for j in range(n)] for i in range(n)]
    result = Rational(1)
    for column in range(n):
        budget.tick()
        pivot_row = next((i for i in range(column, n) if rows[i][column].numerator), None)
        if pivot_row is None:
            return Rational(0)
        if pivot_row != column:
            rows[column], rows[pivot_row] = rows[pivot_row], rows[column]
            result = -result
        pivot = rows[column][column]
        result = result * pivot
        for i in range(column + 1, n):
            if not rows[i][column].numerator:
                continue
            multiplier = rows[i][column] / pivot
            for j in range(column + 1, n):
                rows[i][j] = rows[i][j] - multiplier * rows[column][j]
            rows[i][column] = Rational(0)
    return result


def _check_charpoly(A, variable, value):
    _square_matrix(A)
    _variable(variable)
    if (type(value) is not Polynomial or value.variable != variable
            or value.degree != A.nrows or value.leading_coefficient != Rational(1)):
        return False
    from .polynomial_exact import _validate
    _validate(value)
    # Two polynomials of degree <= n agreeing at n+1 points are identical.
    for integer in range(A.nrows + 1):
        point, evaluated = Rational(integer), Rational(0)
        for coefficient in reversed(value.coefficients):
            evaluated = evaluated * point + coefficient
        if evaluated != _det_at(A, point):
            return False
    return True


def _real_eigenvalues(A):
    from .root_isolation import ROOT_VARIABLE, real_roots
    from .root_certificates import root_certificate

    polynomial = _charpoly(A, ROOT_VARIABLE)
    records = real_roots(polynomial)
    evidence = (polynomial, root_certificate(polynomial, records))
    current_budget().inspect(evidence, evidence=True)
    return records, evidence


def _check_apart(expression, value, evidence):
    from .polynomial_exact import _validate, check_factor

    if (type(expression) is not RationalFunction or type(value) is not PartialFractionDecomposition
            or type(evidence) is not tuple or len(evidence) != 2):
        return False
    _variable(expression.variable)
    _validate_rational_function(expression)
    factorization, certificate = evidence
    if not check_factor(expression.denominator, factorization, certificate):
        return False
    if value.polynomial.variable != expression.variable or value.excluded != expression.excluded:
        return False
    _validate(value.polynomial)
    # Recheck shape even for records forged by bypassing their frozen constructors.
    if PartialFractionDecomposition(value.polynomial, value.terms, value.excluded) != value:
        return False
    factors = {entry.polynomial: entry.multiplicity for entry in factorization.factors}
    numerator = value.polynomial * expression.denominator
    for term in value.terms:
        _validate(term.numerator, term.factor)
        if PartialFractionTerm(term.numerator, term.factor, term.power) != term:
            return False
        if term.factor not in factors or term.power > factors[term.factor]:
            return False
        power = term.factor ** term.power
        cofactor, remainder = expression.denominator.divmod(power)
        if not remainder.is_zero or cofactor * power != expression.denominator:
            return False
        numerator = numerator + term.numerator * cofactor
    # A proper-fraction sum plus a polynomial has a unique polynomial part.
    # Cross multiplication verifies the identity, independently of CRT/xgcd.
    return numerator == expression.numerator


def _validate_rational_function(expression):
    """Do not trust frozen source records with constructor-bypassed domains."""
    from .polynomial_exact import _validate
    _validate(expression.numerator, expression.denominator)
    if type(expression.excluded) is not tuple:
        raise InvalidInput("Rational-function exclusions must be immutable")
    for exclusion in expression.excluded:
        _validate(exclusion)
    if RationalFunction(expression.numerator, expression.denominator, expression.excluded) != expression:
        raise InvalidInput("Rational-function source must be canonical")


_ARGUMENTS = {"apart": {"expression"}, "charpoly": {"A", "variable"}, "real_eigenvalues": {"A"}}


def produce(operation, args):
    """Produce a typed value and immutable evidence under the active budget."""
    with computation() as budget:
        budget.inspect(args)
        if operation not in _ARGUMENTS or type(args) is not dict or set(args) != _ARGUMENTS[operation]:
            raise InvalidInput("Unknown exact extension or invalid operation arguments")
        if operation == "apart":
            return _apart(args["expression"])
        if operation == "charpoly":
            return _charpoly(args["A"], args["variable"]), ()
        return _real_eigenvalues(args["A"])


def check_operation(operation, args, value, evidence):
    """Check source identities and completeness; resource exhaustion propagates."""
    with computation() as budget:
        budget.inspect((args, value, evidence), evidence=True)
        try:
            if (type(operation) is not str or operation not in _ARGUMENTS or type(args) is not dict
                    or set(args) != _ARGUMENTS[operation] or type(evidence) is not tuple):
                return False
            if operation == "apart":
                return _check_apart(args["expression"], value, evidence)
            if operation == "charpoly":
                return evidence == () and _check_charpoly(args["A"], args["variable"], value)
            from .root_isolation import ROOT_VARIABLE
            from .root_certificates import check_root_certificate
            if (type(value) is not tuple or any(type(record) is not RootRecord for record in value)
                    or len(evidence) != 2):
                return False
            if any(RootRecord(record.root, record.multiplicity) != record for record in value):
                return False
            polynomial, root_evidence = evidence
            return (_check_charpoly(args["A"], ROOT_VARIABLE, polynomial)
                    and check_root_certificate(polynomial, value, root_evidence))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError,
                InvalidInput, UnsupportedOperation):
            return False
