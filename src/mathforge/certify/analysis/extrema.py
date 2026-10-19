"""Independent checker of extrema_values when every real stationary point is rational (mathforge.lean.analysis_extrema).

The claim: the strict local extrema of f = N/D, on the domain that the source's excluded polynomials leave, are the
given points with the given kinds and values, in increasing order of the points.

The evidence, ``("rational-stationary-v1", c, roots, quadratics)``, factors the raw numerator of the derivative,
P = N'D - ND', as c * prod (x - r)^m * prod q: the distinct rational roots r in increasing order with their
multiplicities m, and quadratics q of negative discriminant, which have no real root.

The checker reads N, D and the exclusions from the source itself (it runs no producer conversion), recomputes P and
multiplies the factorization back, so the r are all the real zeros of f' = P/D^2. Where D(r) != 0, D^2 is positive
near r and f' changes sign at r exactly when m is odd. The sign of f' just right of r is the sign of c * prod (r -
r')^m' * prod q(r) over the other factors: f has a minimum at r when it is positive and a maximum when it is
negative. A point where D or an excluded polynomial vanishes lies outside the domain and is no extremum. Each value is
N(r)/D(r). Record classes are identified by name: a checker never imports producer code.
"""

from mathforge.limits import computation, current_budget
from mathforge.model import Add, Mul, Pow, Rational, Symbol

FORMAT = "rational-stationary-v1"
_RATIONAL_FUNCTION = ("mathforge.rational_function", "RationalFunction")
_POLYNOMIAL = ("mathforge.polynomial", "Polynomial")
_EXTREMUM = ("mathforge.analysis_types", "ExtremumValue")
_MAX_DEGREE = 1024  # The dense degree bound of the polynomial records.
_ZERO, _ONE = Rational(0), Rational(1)


class _Reject(Exception):
    """The claim is not accepted."""


def _named(value):
    kind = type(value)
    return kind.__module__, kind.__qualname__


# --- Dense coefficient tuples over Q: constant first, no trailing zeros, () for zero -----------------------------

def _trim(values):
    end = len(values)
    while end and not values[end - 1].numerator:
        end -= 1
    return tuple(values[:end])


def _add(a, b):
    if len(a) < len(b):
        a, b = b, a
    out = list(a)
    for i, y in enumerate(b):
        out[i] = out[i] + y
    return _trim(out)


def _sub(a, b):
    return _add(a, tuple(-y for y in b))


def _mul(a, b):
    if not a or not b:
        return ()
    if len(a) + len(b) - 2 > _MAX_DEGREE:
        raise _Reject
    current_budget().tick(len(a) * len(b))
    out = [_ZERO] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        if u.numerator:
            for j, v in enumerate(b):
                out[i + j] = out[i + j] + u * v
    return _trim(out)


def _derivative(a):
    return _trim([a[i] * i for i in range(1, len(a))])


def _at(a, point):
    result = _ZERO
    for coefficient in reversed(a):
        result = result * point + coefficient
    return result


# --- The source -----------------------------------------------------------------------------------------------------

def _coefficients(polynomial, variable):
    if _named(polynomial) != _POLYNOMIAL or polynomial.variable != variable:
        raise _Reject
    values = polynomial.coefficients
    if type(values) is not tuple or not values or any(type(value) is not Rational for value in values):
        raise _Reject
    return _trim(values)


def _expression(node, variable):
    """A polynomial expression of the model in the variable, read as a coefficient tuple."""
    current_budget().tick()
    kind = type(node)
    if kind is Rational:
        return _trim((node,))
    if kind is Symbol:
        if node != variable:
            raise _Reject
        return (_ZERO, _ONE)
    if kind is Add:
        total = ()
        for term in node.terms:
            total = _add(total, _expression(term, variable))
        return total
    if kind is Mul:
        total = (_ONE,)
        for factor in node.factors:
            total = _mul(total, _expression(factor, variable))
        return total
    if kind is Pow:
        base, exponent = _expression(node.base, variable), node.exponent
        if type(exponent) is not int or exponent < 0 or (len(base) - 1) * exponent > _MAX_DEGREE:
            raise _Reject
        result = (_ONE,)
        for _ in range(exponent):
            result = _mul(result, base)
        return result
    raise _Reject


def _source(expression, variable):
    """(N, D, exclusions) of the source: a RationalFunction, a Polynomial or a polynomial expression."""
    if type(variable) is not Symbol:
        raise _Reject
    named = _named(expression)
    if named == _RATIONAL_FUNCTION:
        excluded = expression.excluded
        if type(excluded) is not tuple:
            raise _Reject
        return (_coefficients(expression.numerator, variable), _coefficients(expression.denominator, variable),
                tuple(_coefficients(polynomial, variable) for polynomial in excluded))
    if named == _POLYNOMIAL:
        return _coefficients(expression, variable), (_ONE,), ()
    return _expression(expression, variable), (_ONE,), ()


# --- The check ------------------------------------------------------------------------------------------------------

def _expected(args, evidence):
    """The extrema that the factorization proves: (point, value, kind) in increasing order of the points."""
    numerator, denominator, excluded = _source(args["expression"], args["variable"])
    stationary = _sub(_mul(_derivative(numerator), denominator), _mul(numerator, _derivative(denominator)))
    if not denominator or not stationary:
        raise _Reject
    if type(evidence) is not tuple or len(evidence) != 4 or evidence[0] != FORMAT:
        raise _Reject
    _, lead, roots, quadratics = evidence
    if (type(lead) is not Rational or not lead.numerator or type(roots) is not tuple
            or type(quadratics) is not tuple):
        raise _Reject
    degree, previous = 0, None
    for root in roots:
        if type(root) is not tuple or len(root) != 2:
            raise _Reject
        point, multiplicity = root
        if (type(point) is not Rational or type(multiplicity) is not int or multiplicity < 1
                or (previous is not None and not previous < point)):
            raise _Reject
        previous, degree = point, degree + multiplicity
    for quadratic in quadratics:
        if (type(quadratic) is not tuple or len(quadratic) != 3
                or any(type(value) is not Rational for value in quadratic) or not quadratic[2].numerator
                or not quadratic[1] * quadratic[1] - 4 * quadratic[0] * quadratic[2] < _ZERO):
            raise _Reject
        degree += 2
    if degree != len(stationary) - 1:
        raise _Reject
    product = (lead,)
    for point, multiplicity in roots:
        for _ in range(multiplicity):
            product = _mul(product, (-point, _ONE))
    for quadratic in quadratics:
        product = _mul(product, quadratic)
    if product != stationary:
        raise _Reject
    expected = []
    for index, (point, multiplicity) in enumerate(roots):
        if multiplicity % 2 == 0:
            continue
        at_denominator = _at(denominator, point)
        if not at_denominator.numerator or any(not _at(polynomial, point).numerator for polynomial in excluded):
            continue
        sign = 1 if lead.numerator > 0 else -1
        for other, (root, power) in enumerate(roots):
            if other != index and power % 2 and point < root:
                sign = -sign
        for quadratic in quadratics:
            if quadratic[2].numerator < 0:
                sign = -sign
        expected.append((point, _at(numerator, point) / at_denominator, "minimum" if sign > 0 else "maximum"))
    return expected


def check_operation(args, value, evidence):
    """Whether the claimed extrema are the strict local extrema of the source, as the evidence proves."""
    try:
        with computation():
            expected = _expected(args, evidence)
            if type(value) is not tuple or len(value) != len(expected):
                return False
            for record, (point, at, kind) in zip(value, expected):
                if (_named(record) != _EXTREMUM or type(record.point) is not Rational or record.point != point
                        or type(record.value) is not Rational or record.value != at or record.kind != kind):
                    return False
            return True
    except (_Reject, AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
