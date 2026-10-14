"""Exact algebraic complex scalars, kept separate from the total-real AST.

Scalar arithmetic carries the existing real-algebraic elimination certificates.
The checker reconstructs the complex identities and checks these certificates;
it never calls this module's producers or the real arithmetic producer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

from . import algebraic_arithmetic as aa
from .algebraic import RealAlgebraicRoot, compare_real
from .errors import InvalidInput
from .limits import computation, current_budget
from .model import Expression, Rational, as_expression, sqrt


RealAlgebraic = Rational | RealAlgebraicRoot
ZERO, ONE = Rational(0), Rational(1)
ARGUMENTS = {
    "complex_algebraic": {"real", "imag"},
    **{name: {"left", "right"} for name in ("complex_add", "complex_sub", "complex_mul", "complex_div")},
    **{name: {"value"} for name in ("complex_conjugate", "complex_real", "complex_imag", "complex_norm_squared", "complex_sqrt")},
    "complex_pow": {"base", "n"}, "quadratic_roots": {"a", "b", "c"},
}


def _normal(expression):
    return aa.produce("algebraic", {"expression": as_expression(expression)})


@dataclass(frozen=True, init=False)
class ComplexAlgebraic:
    real: RealAlgebraic
    imag: RealAlgebraic = ZERO

    def __init__(self, real: Expression | int, imag: Expression | int = 0):
        with computation():
            real, _ = _normal(real)
            imag, _ = _normal(imag)
            object.__setattr__(self, "real", real)
            object.__setattr__(self, "imag", imag)

    def __add__(self, other):
        return produce("complex_add", {"left": self, "right": other})[0]

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return produce("complex_sub", {"left": self, "right": other})[0]

    def __rsub__(self, other):
        return produce("complex_sub", {"left": other, "right": self})[0]

    def __mul__(self, other):
        return produce("complex_mul", {"left": self, "right": other})[0]

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return produce("complex_div", {"left": self, "right": other})[0]

    def __rtruediv__(self, other):
        return produce("complex_div", {"left": other, "right": self})[0]

    def __neg__(self):
        return produce("complex_sub", {"left": 0, "right": self})[0]

    def __pow__(self, n):
        return produce("complex_pow", {"base": self, "n": n})[0]

    def conjugate(self):
        return produce("complex_conjugate", {"value": self})[0]


@dataclass(frozen=True)
class ComplexRoot:
    root: ComplexAlgebraic
    multiplicity: int

    def __post_init__(self):
        with computation() as budget:
            if type(self.root) is not ComplexAlgebraic or type(self.multiplicity) is not int or self.multiplicity <= 0:
                raise InvalidInput("A complex root requires a scalar and positive integer multiplicity")
            budget.inspect((self.root, self.multiplicity))


TYPES = {"complex_algebraic": ComplexAlgebraic, "complex_root": ComplexRoot}


def _record(real, imag):
    # Both components have just been normalized with certificates by a producer.
    value = object.__new__(ComplexAlgebraic)
    object.__setattr__(value, "real", real)
    object.__setattr__(value, "imag", imag)
    return value


def _parts(value):
    if type(value) is ComplexAlgebraic:
        if type(value.real) not in (Rational, RealAlgebraicRoot) or type(value.imag) not in (Rational, RealAlgebraicRoot):
            raise InvalidInput("Complex components must be normalized real algebraic scalars")
        aa.validate_exact_scalars(value)
        return value.real, value.imag
    return as_expression(value), ZERO


def _pair(real, imag):
    a, ac = _normal(real)
    b, bc = _normal(imag)
    return _record(a, b), (ac, bc)


def _check_real(expression, value, certificate):
    return type(value) in (Rational, RealAlgebraicRoot) and aa.check_operation(
        "algebraic", {"expression": as_expression(expression)}, value, certificate)


def _check_pair(real, imag, value, evidence):
    return (type(value) is ComplexAlgebraic and type(evidence) is tuple and len(evidence) == 2
            and _check_real(real, value.real, evidence[0]) and _check_real(imag, value.imag, evidence[1]))


def _binary_formula(operation, left, right):
    a, b = _parts(left)
    c, d = _parts(right)
    if operation == "complex_add":
        return a+c, b+d, None
    if operation == "complex_sub":
        return a-c, b-d, None
    if operation == "complex_mul":
        return a*c-b*d, a*d+b*c, None
    return a*c+b*d, b*c-a*d, c*c+d*d


def _binary(operation, left, right):
    real, imag, denominator = _binary_formula(operation, left, right)
    if denominator is None:
        return _pair(real, imag)
    a, ac = aa.produce("algebraic_div", {"left": real, "right": denominator})
    b, bc = aa.produce("algebraic_div", {"left": imag, "right": denominator})
    return _record(a, b), (ac, bc)


def _check_binary(operation, left, right, value, evidence):
    if type(value) is not ComplexAlgebraic or type(evidence) is not tuple or len(evidence) != 2:
        return False
    # Reconstruct Cartesian identities independently of the producer's formula helper.
    a, b = _parts(left)
    c, d = _parts(right)
    if operation == "complex_add":
        expected = (a+c, b+d)
    elif operation == "complex_sub":
        expected = (a-c, b-d)
    elif operation == "complex_mul":
        expected = (a*c + (-b)*d, b*c + a*d)
    else:
        denominator = c*c+d*d
        return (aa.check_operation("algebraic_div", {"left": a*c+b*d, "right": denominator}, value.real, evidence[0])
                and aa.check_operation("algebraic_div", {"left": b*c-a*d, "right": denominator}, value.imag, evidence[1]))
    return _check_pair(*expected, value, evidence)


def _sqrt_real(value):
    """Select the nonnegative root of t**2-value, including nonrational value."""
    if compare_real(value, ZERO) < 0:
        raise InvalidInput("A real square root requires a nonnegative argument")
    if type(value) is Rational:
        result, _ = _normal(sqrt(value))
    else:
        from .polynomial import Polynomial
        from .root_isolation import ROOT_VARIABLE, real_roots
        coefficients = [0] * (2 * len(value.integer_coefficients) - 1)
        current_budget().nodes(len(coefficients))
        for index, coefficient in enumerate(value.integer_coefficients):
            coefficients[2*index] = coefficient
        result = None
        for record in real_roots(Polynomial(tuple(coefficients), ROOT_VARIABLE)):
            current_budget().tick()
            if compare_real(record.root, ZERO) < 0:
                continue
            candidate, _ = _normal(record.root)
            squared, _ = _normal(candidate*candidate)
            if compare_real(squared, value) == 0:
                result = candidate
                break
        if result is None:
            raise ArithmeticError("No nonnegative algebraic square root was isolated")
    _, normalized = _normal(result)
    squared, certificate = _normal(result*result)
    if squared != value:
        raise ArithmeticError("Algebraic square-root identity failed")
    return result, (normalized, certificate)


def _check_sqrt_real(source, value, evidence):
    return (type(value) in (Rational, RealAlgebraicRoot) and type(evidence) is tuple and len(evidence) == 2
            and compare_real(value, ZERO) >= 0
            and _check_real(value, value, evidence[0])
            and _check_real(value*value, source, evidence[1]))


def _sqrt_complex(source):
    base, base_proof = _pair(*_parts(source))
    a, b = base.real, base.imag
    norm, norm_proof = _normal(a*a+b*b)
    radius, radius_proof = _sqrt_real(norm)
    u2, u2_proof = _normal((radius+a)/2)
    v2, v2_proof = _normal((radius-a)/2)
    u, u_proof = _sqrt_real(u2)
    v, v_proof = _sqrt_real(v2)
    sign = -1 if compare_real(b, ZERO) < 0 else 1
    value, final_proof = _pair(u, sign*v)
    return value, (base, base_proof, norm, norm_proof, radius, radius_proof,
                   u2, u2_proof, v2, v2_proof, u, u_proof, v, v_proof, final_proof)


def _check_sqrt_complex(source, value, evidence):
    if type(evidence) is not tuple or len(evidence) != 15:
        return False
    base, bp, norm, np, radius, rp, u2, up2, v2, vp2, u, up, v, vp, final = evidence
    if not _check_pair(*_parts(source), base, bp):
        return False
    a, b = base.real, base.imag
    if not (_check_real(a*a+b*b, norm, np) and _check_sqrt_real(norm, radius, rp)
            and _check_real((radius+a)/2, u2, up2) and _check_real((radius-a)/2, v2, vp2)
            and _check_sqrt_real(u2, u, up) and _check_sqrt_real(v2, v, vp)):
        return False
    return _check_pair(u, (-1 if compare_real(b, ZERO) < 0 else 1)*v, value, final)


def _power(base, n):
    normalized, base_proof = _pair(*_parts(base))
    nodes = [normalized, _record(ONE, ZERO)]
    trace, power_index, result_index, remaining = [], 0, 1, abs(n)
    while remaining:
        current_budget().tick()
        if remaining & 1:
            value, proof = _binary("complex_mul", nodes[result_index], nodes[power_index])
            trace.append((result_index, power_index, value, proof))
            nodes.append(value)
            result_index = len(nodes)-1
        remaining //= 2
        if remaining:
            value, proof = _binary("complex_mul", nodes[power_index], nodes[power_index])
            trace.append((power_index, power_index, value, proof))
            nodes.append(value)
            power_index = len(nodes)-1
    value, inverse_proof = nodes[result_index], ()
    if n < 0:
        value, inverse_proof = _binary("complex_div", 1, value)
    return value, (normalized, base_proof, tuple(trace), result_index, inverse_proof)


def _check_power(base, n, value, evidence):
    if type(evidence) is not tuple or len(evidence) != 5:
        return False
    normalized, base_proof, trace, result_index, inverse_proof = evidence
    if not _check_pair(*_parts(base), normalized, base_proof) or type(trace) is not tuple:
        return False
    nodes, exponents = [normalized, _record(ONE, ZERO)], [1, 0]
    parents = [(), ()]
    for entry in trace:
        current_budget().tick()
        if type(entry) is not tuple or len(entry) != 4:
            return False
        left, right, answer, proof = entry
        if any(type(i) is not int or not 0 <= i < len(nodes) for i in (left, right)):
            return False
        if not _check_binary("complex_mul", nodes[left], nodes[right], answer, proof):
            return False
        exponent = exponents[left]+exponents[right]
        current_budget().check_int(exponent)
        nodes.append(answer)
        exponents.append(exponent)
        parents.append((left, right))
    if type(result_index) is not int or not 0 <= result_index < len(nodes) or exponents[result_index] != abs(n):
        return False
    reachable, pending = set(), [result_index]
    while pending:
        index = pending.pop()
        if index not in reachable:
            reachable.add(index)
            pending.extend(parents[index])
    if any(index not in reachable for index in range(2, len(nodes))):
        return False
    if n < 0:
        return _check_binary("complex_div", 1, nodes[result_index], value, inverse_proof)
    return inverse_proof == () and type(value) is ComplexAlgebraic and value == nodes[result_index]


def _rational(value):
    value = as_expression(value)
    if type(value) is not Rational:
        raise InvalidInput("Quadratic coefficients must be exact rational numbers")
    return value


def _compare_complex(left, right):
    real = compare_real(left.real, right.real)
    return real if real else compare_real(left.imag, right.imag)


def _quadratic(a, b, c):
    discriminant = b*b-4*a*c
    radical, radical_proof = _sqrt_complex(discriminant)
    signs = (1,) if discriminant == ZERO else (-1, 1)
    values = []
    for sign in signs:
        # Division is by a rational scalar, so these remain total real expressions.
        answer, proof = _pair((-b+sign*radical.real)/(2*a), sign*radical.imag/(2*a))
        values.append((answer, sign, proof))
    values.sort(key=cmp_to_key(lambda left, right: _compare_complex(left[0], right[0])))
    roots = tuple(ComplexRoot(value, 2 if len(values) == 1 else 1) for value, _, _ in values)
    return roots, (radical, radical_proof, tuple((sign, proof) for _, sign, proof in values))


def _validate(operation, args):
    if operation not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[operation]:
        raise InvalidInput("Invalid complex operation arguments")
    if operation == "quadratic_roots":
        a, b, c = (_rational(args[key]) for key in ("a", "b", "c"))
        if a == ZERO:
            raise InvalidInput("A quadratic equation requires a nonzero leading coefficient")
        return a, b, c
    if operation == "complex_algebraic":
        as_expression(args["real"])
        as_expression(args["imag"])
    elif operation == "complex_pow":
        _parts(args["base"])
        if type(args["n"]) is not int:
            raise InvalidInput("A complex exponent must be an exact integer")
        current_budget().check_int(args["n"])
    else:
        for operand in args.values():
            _parts(operand)


def produce(operation, args, *, limits=None):
    with computation(limits) as budget:
        budget.inspect(args)
        coefficients = _validate(operation, args)
        if operation == "complex_algebraic":
            answer = _pair(args["real"], args["imag"])
        elif operation in ("complex_add", "complex_sub", "complex_mul", "complex_div"):
            answer = _binary(operation, args["left"], args["right"])
        elif operation == "complex_pow":
            answer = _power(args["base"], args["n"])
        elif operation == "complex_sqrt":
            answer = _sqrt_complex(args["value"])
        elif operation == "quadratic_roots":
            answer = _quadratic(*coefficients)
        else:
            a, b = _parts(args["value"])
            if operation == "complex_conjugate":
                answer = _pair(a, -b)
            else:
                expression = a if operation == "complex_real" else b if operation == "complex_imag" else a*a+b*b
                value, proof = _normal(expression)
                answer = value, (proof,)
        budget.inspect(answer[1], evidence=True)
        return answer


def check_operation(operation, args, value, evidence, *, limits=None):
    with computation(limits) as budget:
        try:
            budget.inspect(args)
            budget.inspect((value, evidence), evidence=True)
            coefficients = _validate(operation, args)
            if type(evidence) is not tuple:
                return False
            if operation == "complex_algebraic":
                return _check_pair(args["real"], args["imag"], value, evidence)
            if operation in ("complex_add", "complex_sub", "complex_mul", "complex_div"):
                return _check_binary(operation, args["left"], args["right"], value, evidence)
            if operation == "complex_pow":
                return _check_power(args["base"], args["n"], value, evidence)
            if operation == "complex_sqrt":
                return _check_sqrt_complex(args["value"], value, evidence)
            if operation == "quadratic_roots":
                if type(value) is not tuple or len(evidence) != 3:
                    return False
                a, b, c = coefficients
                radical, radical_proof, records = evidence
                discriminant = b*b-4*a*c
                if not _check_sqrt_complex(discriminant, radical, radical_proof):
                    return False
                expected_count = 1 if discriminant == ZERO else 2
                if len(value) != expected_count or type(records) is not tuple or len(records) != expected_count:
                    return False
                seen, previous = set(), None
                for root, record in zip(value, records):
                    if type(root) is not ComplexRoot or type(root.multiplicity) is not int or root.multiplicity != 3-expected_count:
                        return False
                    if type(record) is not tuple or len(record) != 2 or type(record[0]) is not int:
                        return False
                    sign, proof = record
                    if sign not in (-1, 1) or sign in seen:
                        return False
                    seen.add(sign)
                    if not _check_pair((-b+sign*radical.real)/(2*a), sign*radical.imag/(2*a), root.root, proof):
                        return False
                    if previous is not None and _compare_complex(previous, root.root) >= 0:
                        return False
                    previous = root.root
                return seen == ({1} if expected_count == 1 else {-1, 1})
            a, b = _parts(args["value"])
            if operation == "complex_conjugate":
                return _check_pair(a, -b, value, evidence)
            expression = a if operation == "complex_real" else b if operation == "complex_imag" else a*a+b*b
            return len(evidence) == 1 and _check_real(expression, value, evidence[0])
        except (InvalidInput, TypeError, ValueError, AttributeError, KeyError, IndexError, ZeroDivisionError):
            return False
