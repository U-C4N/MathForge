"""Independent checker of function_taylor values (mathforge.lean.series_taylor), by truncated power series.

The producer differentiates the source symbolically n times and evaluates the derivatives at the center. This
checker never differentiates an expression: it recomputes the Taylor coefficients at the center from the source
tree, node by node, as truncated power series in exact rational arithmetic (mathforge.trusted.series), and compares.

A node is computed only where its series at the center a is rational and the node is analytic there; anything else
makes the checker decline (False), and the caller then uses another check. With U, V the children's series, F the
node's own and U0 = U(a):

    scalar c (a Rational), the variable    (c, 0, ...), (a, 1, 0, ...)
    add, mul, div (V0 != 0)                 U + V, U V, U / V
    pow k                                   U^k for k >= 0; (1/U)^-k for k < 0 and U0 != 0
    exp (U0 = 0)                            F' = U' F, F0 = 1
    log (U0 = 1)                            F' = U' / U, F0 = 0
    sqrt (U0 = q^2, q > 0 rational)         F^2 = U, F0 = q
    abs (U0 != 0)                           sign(U0) U
    sin, cos (U0 = 0)                       S' = U' C, C' = -U' S, S0 = 0, C0 = 1
    sinh, cosh (U0 = 0)                     S' = U' C, C' = U' S, S0 = 0, C0 = 1
    tan, tanh (U0 = 0)                      F' = U' (1 + F^2), F' = U' (1 - F^2), F0 = 0
    atan (U0 = 0)                           F' = U' / (1 + U^2), F0 = 0
    asin (U0 = 0)                           F' = U' / W with W^2 = 1 - U^2, W0 = 1, F0 = 0
    acos, pi, E, any other symbol           declined

The value is accepted when its variable is the argument, its point is the scalar a, its n + 1 coefficients are the
scalars of the series, and its polynomial is a function of the variable alone, with the domain TRUE, whose
expression, read as a polynomial, is sum c_k (x - a)^k. Record classes are identified by name: a checker never
imports producer code.
"""

from math import isqrt

from mathforge.limits import computation, current_budget
from mathforge.model import Rational, Symbol
from mathforge.trusted import series

_REAL_EXPR = ("mathforge.functions", "RealExpr")
_REAL_FUNCTION = ("mathforge.functions", "RealFunction")
_DOMAIN = ("mathforge.functions", "DomainPredicate")
_TAYLOR = ("mathforge.function_calculus", "TaylorPolynomial")
_MAX_DEGREE = 4096  # A polynomial read from the value larger than this is declined.
_ZERO, _ONE = series.ZERO, series.ONE


class _Decline(Exception):
    """The series at the center is not computed here."""


def _named(value):
    kind = type(value)
    return kind.__module__, kind.__qualname__


def _flow(u, seed, step):
    """The series F with F[0] = seed and F' = U' G, where step(F[:k]) returns G[k - 1] from F[0..k-1]."""
    out = [seed]
    for k in range(1, len(u)):
        total = _ZERO
        for j in range(1, k + 1):
            if u[j].numerator:
                total = total + j * u[j] * step(out, k - j)
        out.append(total / k)
    return tuple(out)


def _sine_pair(u, sign):
    """(S, C) with S' = U' C, C' = sign U' S, S0 = 0, C0 = 1."""
    s, c = [_ZERO], [_ONE]
    for k in range(1, len(u)):
        ts = tc = _ZERO
        for j in range(1, k + 1):
            if u[j].numerator:
                ts = ts + j * u[j] * c[k - j]
                tc = tc + j * u[j] * s[k - j]
        s.append(ts / k)
        c.append(tc / k if sign > 0 else -tc / k)
    return tuple(s), tuple(c)


def _square_root(u, root):
    """F with F^2 = U and F0 = root != 0."""
    out = [root]
    twice = 2 * root
    for k in range(1, len(u)):
        total = u[k]
        for j in range(1, k):
            total = total - out[j] * out[k - j]
        out.append(total / twice)
    return tuple(out)


def _quotient_integral(u, w):
    """The primitive of U' / W with constant term 0, truncated to the length of U."""
    if len(u) == 1:
        return (_ZERO,)
    return series.integral(series.mul(series.derivative(u), series.reciprocal(w)[:-1]))


def _rational_root(value):
    """The positive rational square root of a positive Rational, or None."""
    numerator, denominator = value.numerator, value.denominator
    if numerator <= 0:
        return None
    a, b = isqrt(numerator), isqrt(denominator)
    return Rational(a, b) if a * a == numerator and b * b == denominator else None


def _unary(op, u):
    head = u[0]
    if op == "abs":
        if not head.numerator:
            raise _Decline
        return u if head.numerator > 0 else series.neg(u)
    if op == "log":
        if head != _ONE:
            raise _Decline
        return _quotient_integral(u, u)
    if op == "sqrt":
        root = _rational_root(head)
        if root is None:
            raise _Decline
        return _square_root(u, root)
    if head.numerator:
        raise _Decline  # Every other function is expanded about an argument that vanishes at the center.
    if op == "exp":
        return _flow(u, _ONE, lambda f, m: f[m])
    if op in ("sin", "cos"):
        s, c = _sine_pair(u, -1)
        return s if op == "sin" else c
    if op in ("sinh", "cosh"):
        s, c = _sine_pair(u, 1)
        return s if op == "sinh" else c
    if op in ("tan", "tanh"):
        sign = 1 if op == "tan" else -1

        def step(f, m):  # (1 +- F^2)[m] from F[0..m]
            square = _ZERO
            for i in range(m + 1):
                square = square + f[i] * f[m - i]
            return (_ONE if m == 0 else _ZERO) + (square if sign > 0 else -square)
        return _flow(u, _ZERO, step)
    if op == "atan":
        return _quotient_integral(u, series.add(series.constant(_ONE, len(u)), series.mul(u, u)))
    if op == "asin":
        w = _square_root(series.sub(series.constant(_ONE, len(u)), series.mul(u, u)), _ONE)
        return _quotient_integral(u, w)
    raise _Decline


def _series(node, variable, center, length, memo):
    """The truncated series of a RealExpr node at the center; memo maps id(node) to (node, series)."""
    found = memo.get(id(node))
    if found is not None:
        return found[1]
    if _named(node) != _REAL_EXPR:
        raise _Decline
    current_budget().tick(length)
    op, args = node.op, node.args
    if op == "scalar":
        if type(node.value) is not Rational or args:
            raise _Decline
        result = series.constant(node.value, length)
    elif op == "symbol":
        if node.value != variable or args:
            raise _Decline
        result = (center,) if length == 1 else (center, _ONE) + (_ZERO,) * (length - 2)
    elif op in ("add", "mul", "div"):
        if type(args) is not tuple or len(args) != 2:
            raise _Decline
        u, v = (_series(arg, variable, center, length, memo) for arg in args)
        if op == "add":
            result = series.add(u, v)
        elif op == "mul":
            result = series.mul(u, v)
        else:
            if not v[0].numerator:
                raise _Decline
            result = series.mul(u, series.reciprocal(v))
    elif op == "pow":
        if type(args) is not tuple or len(args) != 1 or type(node.value) is not int:
            raise _Decline
        u, exponent = _series(args[0], variable, center, length, memo), node.value
        if exponent >= 0:
            result = series.power(u, exponent)
        elif u[0].numerator:
            result = series.power(series.reciprocal(u), -exponent)
        else:
            raise _Decline
    elif type(args) is tuple and len(args) == 1:
        result = _unary(op, _series(args[0], variable, center, length, memo))
    else:
        raise _Decline
    memo[id(node)] = (node, result)
    return result


# --- Reading the value's polynomial ---------------------------------------------------------------------------------

def _trim(values):
    end = len(values)
    while end and not values[end - 1].numerator:
        end -= 1
    return tuple(values[:end])


def _padd(a, b):
    if len(a) < len(b):
        a, b = b, a
    out = list(a)
    for i, v in enumerate(b):
        out[i] = out[i] + v
    return _trim(out)


def _pmul(a, b):
    if not a or not b:
        return ()
    if len(a) + len(b) - 2 > _MAX_DEGREE:
        raise _Decline
    current_budget().tick(len(a) * len(b))
    out = [_ZERO] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        if u.numerator:
            for j, v in enumerate(b):
                out[i + j] = out[i + j] + u * v
    return _trim(out)


def _polynomial(node, variable):
    """A RealExpr read as a polynomial in the variable with rational coefficients (a trimmed tuple)."""
    if _named(node) != _REAL_EXPR:
        raise _Decline
    current_budget().tick()
    op, args = node.op, node.args
    if op == "scalar" and type(node.value) is Rational:
        return _trim((node.value,))
    if op == "symbol" and node.value == variable:
        return (_ZERO, _ONE)
    if op in ("add", "mul") and type(args) is tuple and len(args) == 2:
        a, b = (_polynomial(arg, variable) for arg in args)
        return _padd(a, b) if op == "add" else _pmul(a, b)
    if op == "div" and type(args) is tuple and len(args) == 2:
        divisor = args[1]
        if _named(divisor) != _REAL_EXPR or divisor.op != "scalar" or type(divisor.value) is not Rational:
            raise _Decline
        if not divisor.value.numerator:
            raise _Decline
        inverse = _ONE / divisor.value
        return tuple(c * inverse for c in _polynomial(args[0], variable))
    if op == "pow" and type(args) is tuple and len(args) == 1 and type(node.value) is int and node.value >= 0:
        base, exponent, result = _polynomial(args[0], variable), node.value, (_ONE,)
        if len(base) > 1 and (len(base) - 1) * exponent > _MAX_DEGREE:
            raise _Decline
        while exponent:
            if exponent & 1:
                result = _pmul(result, base)
            exponent >>= 1
            if exponent:
                base = _pmul(base, base)
        return result
    raise _Decline


def _scalar(node, value):
    return (_named(node) == _REAL_EXPR and node.op == "scalar" and node.args == ()
            and type(node.value) is Rational and node.value == value)


# --- The check ------------------------------------------------------------------------------------------------------

def _check(args, value):
    variable, order, point = args["variable"], args["order"], args["point"]
    if type(variable) is not Symbol or type(order) is not int or order < 0:
        return False
    if type(point) is int:
        center = Rational(point)
    elif type(point) is Rational:
        center = point
    else:
        return False
    source = args["expression"]
    named = _named(source)
    if named == _REAL_FUNCTION:
        if source.variables != (variable,):
            return False
        expression = source.expression
    elif named == _REAL_EXPR:
        expression = source
    else:
        return False
    length = order + 1
    coefficients = _series(expression, variable, center, length, {})
    if _named(value) != _TAYLOR or value.variable != variable or not _scalar(value.point, center):
        return False
    claimed = value.coefficients
    if type(claimed) is not tuple or len(claimed) != length:
        return False
    if not all(_scalar(c, expected) for c, expected in zip(claimed, coefficients)):
        return False
    polynomial = value.polynomial
    if (_named(polynomial) != _REAL_FUNCTION or polynomial.variables != (variable,)
            or _named(polynomial.domain) != _DOMAIN or polynomial.domain.kind != "true"):
        return False
    expected, shift = (), (_ONE,)
    for coefficient in coefficients:
        expected = _padd(expected, tuple(c * coefficient for c in shift))
        shift = _pmul(shift, (-center, _ONE))
    return _polynomial(polynomial.expression, variable) == expected


def check_operation(operation, args, value, evidence):
    """Whether the TaylorPolynomial is the source's Taylor expansion; False also when the series is declined."""
    try:
        with computation():
            return operation == "function_taylor" and _check(args, value)
    except (_Decline, AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
