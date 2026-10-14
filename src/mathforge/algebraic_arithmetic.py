"""Certified arithmetic of closed real algebraic expressions.

The public expression constructors deliberately retain their structural semantics.
This module performs explicit normalization and carries a source-linked arithmetic
DAG.  The checker checks rational determinant identities and isolating intervals;
it never invokes the arithmetic or factorization producers.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from math import comb, gcd, isqrt

from .algebraic import RealAlgebraicRoot, as_endpoint, compare_real, interval_for
from .errors import InvalidInput, UnsupportedOperation
from .limits import ResourceLimitError, current_budget
from .model import (Add, Mul, Pow, Rational, Sqrt, Symbol, as_expression,
                    substitute_expression)
from .polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from .rational_function import RationalFunction, validate_variable
from .root_isolation import (ROOT_VARIABLE, count_real_roots, evaluate_rational,
                             primitive_integer_coefficients, refine_interval,
                             root_work)


ZERO, ONE = Rational(0), Rational(1)
OPERATIONS = frozenset(("algebraic", "algebraic_div", "algebraic_pow",
                        "minimal_polynomial", "evaluate_algebraic", "compare_real"))


def validate_exact_scalars(value):
    """Reject scalar records forged by bypassing their frozen constructors."""
    pending, visited, checked_roots = [value], set(), set()
    while pending:
        item = pending.pop()
        current_budget().tick()
        if type(item) is Rational:
            if (type(item.numerator) is not int or type(item.denominator) is not int or
                    item.denominator <= 0 or gcd(item.numerator, item.denominator) != 1):
                raise InvalidInput("Algebraic arithmetic requires canonical rational values")
            continue
        if type(item) is RealAlgebraicRoot:
            coefficients = item.integer_coefficients
            if (type(coefficients) is not tuple or len(coefficients) < 2 or
                    any(type(c) is not int for c in coefficients) or
                    type(item.real_index) is not int or item.real_index < 0):
                raise InvalidInput("Malformed algebraic root descriptor")
            key = coefficients, item.real_index
            if key not in checked_roots:
                if RealAlgebraicRoot(coefficients, item.real_index) != item:
                    raise InvalidInput("Algebraic root descriptor is not canonical")
                checked_roots.add(key)
            continue
        if type(item) in (tuple, list, dict) or (is_dataclass(item) and not isinstance(item, type)):
            if id(item) in visited:
                continue
            visited.add(id(item))
            if type(item) is dict:
                pending.extend(child for pair in item.items() for child in pair)
            elif type(item) in (tuple, list):
                pending.extend(item)
            else:
                pending.extend(getattr(item, field.name) for field in fields(item))


def _poly(coefficients):
    return Polynomial(tuple(coefficients), ROOT_VARIABLE)


def _defining(value):
    if type(value) is Rational:
        return _poly((-value, ONE))
    if type(value) is RealAlgebraicRoot:
        return _poly(value.integer_coefficients).monic()
    raise InvalidInput("Expected a normalized real algebraic value")


def _order(value):
    if type(value) is not int:
        raise InvalidInput("An algebraic exponent must be an integer")
    current_budget().check_int(value)
    return value


def _combine(kind, children):
    children = list(children)
    if not children:
        return ("rational", ZERO if kind == "add" else ONE)
    while len(children) > 1:
        children = [(kind, children[i], children[i+1]) if i+1 < len(children) else children[i]
                    for i in range(0, len(children), 2)]
    return children[0]


def _recipe(expression):
    expression = as_expression(expression)
    current_budget().nodes()
    if type(expression) is Rational:
        return ("rational", expression)
    if type(expression) is RealAlgebraicRoot:
        return ("root", expression)
    if type(expression) is Sqrt:
        return ("sqrt", expression.radicand)
    if type(expression) in (Add, Mul):
        children = expression.terms if type(expression) is Add else expression.factors
        kind = "add" if type(expression) is Add else "mul"
        return _combine(kind, (_recipe(child) for child in children))
    if type(expression) is Pow:
        return ("pow", _recipe(expression.base), _order(expression.exponent))
    raise UnsupportedOperation("Algebraic arithmetic requires a closed real algebraic expression")


def _bindings(raw):
    if type(raw) is not tuple:
        raise InvalidInput("Recorded bindings must be an immutable sequence")
    result = {}
    for pair in raw:
        if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not Symbol:
            raise InvalidInput("Bindings must map Symbols to exact values")
        if pair[0] in result:
            raise InvalidInput("Bindings cannot repeat a symbol identity")
        result[pair[0]] = as_expression(pair[1])
    return result


def _polynomial_recipe(polynomial, at):
    # Balanced sums avoid a deep synthetic Horner AST for large degree inputs.
    return _combine("add", (("mul", ("rational", coefficient), ("pow", at, degree))
                            for degree, coefficient in enumerate(polynomial.coefficients)
                            if coefficient != ZERO))


def _evaluation_recipe(expression, raw_bindings):
    bindings = _bindings(raw_bindings)
    if type(expression) is Polynomial:
        expression = expression.to_expression()
    if type(expression) is not RationalFunction:
        return _recipe(substitute_expression(as_expression(expression), bindings))
    # Validate every binding, including unused ones, as the historical API does.
    from .model import _check_replacement
    for variable, replacement in bindings.items():
        _check_replacement(variable, replacement)
        _recipe(replacement)
    if expression.variable not in bindings:
        raise InvalidInput("Evaluation requires a binding for the rational function variable")
    return ("rf", expression, _recipe(bindings[expression.variable]))


def _operation_recipe(operation, args):
    if operation in ("algebraic", "minimal_polynomial"):
        if operation == "minimal_polynomial":
            validate_variable(args["variable"])
        return _recipe(args["expression"])
    if operation == "algebraic_div":
        return ("div", _recipe(args["left"]), _recipe(args["right"]))
    if operation == "algebraic_pow":
        return ("pow", _recipe(args["base"]), _order(args["n"]))
    if operation == "evaluate_algebraic":
        return _evaluation_recipe(args["expression"], args["bindings"])
    raise UnsupportedOperation("Unknown algebraic operation")


def _checker_fold(kind, parts):
    """Reconstruct the source grouping independently of producer _combine."""
    parts = tuple(parts)
    if not parts:
        return ("rational", Rational(0 if kind == "add" else 1))
    if len(parts) == 1:
        return parts[0]
    split = 1 << ((len(parts)-1).bit_length()-1)
    return kind, _checker_fold(kind, parts[:split]), _checker_fold(kind, parts[split:])


def _checker_source(source):
    current_budget().tick()
    if type(source) is int:
        source = Rational(source)
    if type(source) in (Rational, RealAlgebraicRoot):
        return ("rational" if type(source) is Rational else "root", source)
    if type(source) is Sqrt:
        return "sqrt", source.radicand
    if type(source) is Add:
        return _checker_fold("add", tuple(_checker_source(term) for term in source.terms))
    if type(source) is Mul:
        return _checker_fold("mul", tuple(_checker_source(factor) for factor in source.factors))
    if type(source) is Pow:
        if type(source.exponent) is not int or source.exponent < 0:
            raise InvalidInput("Malformed source power")
        return "pow", _checker_source(source.base), source.exponent
    raise UnsupportedOperation("The source is not a closed algebraic expression")


def _checker_substitution(expression, raw_bindings):
    """Independent simultaneous source substitution; no producer helper calls."""
    from .model import _check_replacement, add, mul, power, sqrt
    if type(raw_bindings) is not tuple:
        raise InvalidInput("Malformed recorded bindings")
    substitutions = {}
    for pair in raw_bindings:
        if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not Symbol:
            raise InvalidInput("Malformed recorded binding")
        variable, replacement = pair
        if variable in substitutions:
            raise InvalidInput("Repeated recorded binding")
        replacement = as_expression(replacement)
        # Match the established replacement validation, including local
        # simplification, without invoking substitute_expression.
        from .model import simplify_expression
        if type(expression) is not RationalFunction:
            replacement = simplify_expression(replacement)
        _check_replacement(variable, replacement)
        substitutions[variable] = replacement

    def visit(node):
        current_budget().tick()
        if type(node) is Symbol:
            return substitutions.get(node, node)
        if type(node) in (Rational, RealAlgebraicRoot):
            return node
        if type(node) is Sqrt:
            return sqrt(node.radicand)
        if type(node) is Add:
            return add(*(visit(child) for child in node.terms))
        if type(node) is Mul:
            return mul(*(visit(child) for child in node.factors))
        if type(node) is Pow:
            return power(visit(node.base), node.exponent)
        raise InvalidInput("Malformed source expression")

    if type(expression) is RationalFunction:
        for value in substitutions.values():
            _checker_source(value)
        if expression.variable not in substitutions:
            raise InvalidInput("Missing rational-function binding")
        return "rf", expression, _checker_source(substitutions[expression.variable])
    if type(expression) is Polynomial:
        # Independent conversion from coefficients, not Polynomial.to_expression.
        expression = add(*(mul(coefficient, power(expression.variable, degree))
                           for degree, coefficient in enumerate(expression.coefficients)
                           if coefficient != ZERO))
    return _checker_source(visit(as_expression(expression)))


def _checker_request(operation, args):
    if operation == "algebraic":
        return _checker_source(args["expression"])
    if operation == "minimal_polynomial":
        validate_variable(args["variable"])
        return _checker_source(args["expression"])
    if operation == "algebraic_div":
        return "div", _checker_source(args["left"]), _checker_source(args["right"])
    if operation == "algebraic_pow":
        if type(args["n"]) is not int:
            raise InvalidInput("An algebraic exponent must be an integer")
        return "pow", _checker_source(args["base"]), args["n"]
    if operation == "evaluate_algebraic":
        return _checker_substitution(args["expression"], args["bindings"])
    raise UnsupportedOperation("Unknown algebraic claim")


def _checker_polynomial_recipe(polynomial, at):
    parts = []
    for degree in range(len(polynomial.coefficients)):
        coefficient = polynomial.coefficients[degree]
        if coefficient.numerator:
            parts.append(("mul", ("rational", coefficient), ("pow", at, degree)))
    return _checker_fold("add", parts)


def _checker_defining(value):
    if type(value) is Rational:
        return Polynomial((-value, Rational(1)), ROOT_VARIABLE)
    coefficients = tuple(Rational(c, value.integer_coefficients[-1]) for c in value.integer_coefficients)
    return Polynomial(coefficients, ROOT_VARIABLE)


def _checker_sqrt(radicand):
    numerator, denominator = isqrt(radicand.numerator), isqrt(radicand.denominator)
    if numerator*numerator == radicand.numerator and denominator*denominator == radicand.denominator:
        return Rational(numerator, denominator)
    polynomial = Polynomial((-radicand, Rational(0), Rational(1)), ROOT_VARIABLE)
    return RealAlgebraicRoot(primitive_integer_coefficients(polynomial), 1)


def _modular_power(polynomial, exponent, modulus):
    result = _poly((ONE,))
    while exponent:
        current_budget().tick()
        if exponent & 1:
            result = (result * polynomial).divmod(modulus)[1]
        exponent >>= 1
        if exponent:
            polynomial = (polynomial * polynomial).divmod(modulus)[1]
    return result


def _sample_polynomials(kind, values, point, exponent=None):
    """Specialize the elimination identity; the first polynomial is monic."""
    first = _defining(values[0])
    if kind == "pow":
        residue = _modular_power(_poly((ZERO, ONE)), abs(exponent), first)
        second = (_poly((point,)) - residue if exponent >= 0
                  else residue * point - ONE)
        return first, second
    second = _defining(values[1])
    if kind == "add":
        # Q(z-t), evaluated at a rational z.
        coefficients = [ZERO] * (second.degree + 1)
        for k, coefficient in enumerate(second.coefficients):
            for j in range(k + 1):
                current_budget().tick()
                coefficients[j] += coefficient * comb(k, j) * point**(k-j) * (-1 if j % 2 else 1)
        return first, _poly(coefficients)
    if kind == "mul":
        # t^n Q(z/t); zero operands are handled before elimination.
        return first, _poly(tuple(second.coefficients[second.degree-j] *
                                 point**(second.degree-j) for j in range(second.degree + 1)))
    if kind == "div":
        # Eliminate the denominator beta from Q(beta), P(z*beta).
        return second, _poly(tuple(c * point**i for i, c in enumerate(first.coefficients)))
    raise InvalidInput("Unknown algebraic elimination identity")


def _degree_bound(kind, values):
    degree = _defining(values[0]).degree
    if kind != "pow":
        degree *= _defining(values[1]).degree
    if degree > MAX_POLYNOMIAL_DEGREE:
        raise ResourceLimitError("Algebraic elimination exceeds the polynomial degree budget")
    current_budget().nodes(degree + 1)
    return degree


def _elimination_preflight(kind, values, degree, exponent):
    """Bound specialized coefficient and interpolation growth before sampling.

    Rational arithmetic still checks every intermediate product. This early
    check prevents constructing plainly oversized sample powers, binomial
    coefficients or interpolation factorials only to exhaust the bit budget.
    It does not assume that a small final minimal polynomial implies cheap work.
    """
    budget = current_budget()
    first = _defining(values[0])
    second_degree = min(first.degree-1, abs(exponent)) if kind == "pow" else _defining(values[1]).degree
    budget.matrix(first.degree + second_degree, first.degree + second_degree)
    bits = max(max(c.numerator.bit_length(), c.denominator.bit_length())
               for value in values for c in _defining(value).coefficients)
    sample_degree = max(first.degree, second_degree)
    sample_bound = bits + sample_degree * (max(1, degree.bit_length()) + 1) + sample_degree.bit_length()
    interpolation_bound = degree * max(1, degree.bit_length()) + sample_bound
    if kind == "pow" and exponent:
        # Polynomial reduction of t^n can grow linearly in |n| in coefficient
        # bits. Compare before multiplying a potentially enormous exponent.
        per_power = bits + first.degree.bit_length()
        if abs(exponent) > budget.limits.max_integer_bits // max(1, per_power):
            raise ResourceLimitError("Algebraic power exceeds the coefficient-growth budget")
        sample_bound += abs(exponent) * per_power
    if max(sample_bound, interpolation_bound) > budget.limits.max_integer_bits:
        raise ResourceLimitError("Algebraic elimination exceeds the coefficient-growth budget")


def _interpolate(values):
    """Newton interpolation at 0, 1, ..., using finite differences."""
    differences = list(values)
    result, basis, factorial = _poly((ZERO,)), _poly((ONE,)), 1
    for index in range(len(values)):
        current_budget().tick()
        result += basis * (differences[0] / factorial)
        differences = [right-left for left, right in zip(differences, differences[1:])]
        if index + 1 < len(values):
            basis *= _poly((-index, ONE))
            current_budget().product(factorial, index + 1)
            factorial *= index + 1
    return result


def _annihilator(kind, values, exponent=None):
    from .polynomial_exact import resultant
    degree = _degree_bound(kind, values)
    _elimination_preflight(kind, values, degree, exponent)
    samples = []
    for index in range(degree + 1):
        current_budget().tick()
        first, second = _sample_polynomials(kind, values, Rational(index), exponent)
        samples.append(resultant(first, second))
    return _interpolate(samples)


def _determinant(rows):
    """Checker-only rational Gaussian determinant, not the matrix producer."""
    size = len(rows)
    current_budget().matrix(size, size)
    rows = [list(row) for row in rows]
    result = ONE
    for column in range(size):
        current_budget().tick()
        pivot = next((i for i in range(column, size) if rows[i][column] != ZERO), None)
        if pivot is None:
            return ZERO
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            result = -result
        value = rows[column][column]
        result *= value
        for i in range(column + 1, size):
            if rows[i][column] == ZERO:
                continue
            ratio = rows[i][column] / value
            for j in range(column + 1, size):
                rows[i][j] -= ratio * rows[column][j]
            rows[i][column] = ZERO
    return result


def _resultant_determinant(first, second):
    m, n = first.degree, second.degree
    if second.is_zero:
        return ZERO
    if n == 0:
        return second.coefficients[0]**m
    if m == 0:
        return first.coefficients[0]**n
    current_budget().matrix(m+n, m+n)
    rows = []
    a, b = tuple(reversed(first.coefficients)), tuple(reversed(second.coefficients))
    for offset in range(n):
        rows.append((ZERO,)*offset + a + (ZERO,)*(n-1-offset))
    for offset in range(m):
        rows.append((ZERO,)*offset + b + (ZERO,)*(m-1-offset))
    return _determinant(rows)


def _check_annihilator(kind, values, polynomial, exponent=None):
    if type(polynomial) is not Polynomial or polynomial.variable != ROOT_VARIABLE or polynomial.is_zero:
        return False
    degree = _checker_defining(values[0]).degree
    if kind != "pow":
        degree *= _checker_defining(values[1]).degree
    if degree > MAX_POLYNOMIAL_DEGREE:
        raise ResourceLimitError("Algebraic elimination exceeds the polynomial degree budget")
    if polynomial.degree > degree:
        return False
    for index in range(degree + 1):
        current_budget().tick()
        first, second = _checker_specialization(kind, values, Rational(index), exponent)
        if evaluate_rational(polynomial, Rational(index)) != _resultant_determinant(first, second):
            return False
    return True


def _checker_specialization(kind, values, at, exponent):
    """Independent elimination specialization by polynomial composition.

    Producer addition uses binomial coefficients; this checker composes by
    Horner. Powers use left-to-right modular exponentiation, not the producer's
    repeated squaring implementation.
    """
    first = _checker_defining(values[0])
    t = Polynomial((ZERO, ONE), ROOT_VARIABLE)
    if kind == "pow":
        residue = Polynomial((ONE,), ROOT_VARIABLE)
        for bit in bin(abs(exponent))[2:]:
            current_budget().tick()
            residue = (residue*residue) % first
            if bit == "1":
                residue = (residue*t) % first
        if exponent < 0:
            return first, residue*at - 1
        return first, Polynomial((at,), ROOT_VARIABLE) - residue
    second = _checker_defining(values[1])
    composed = Polynomial((ZERO,), ROOT_VARIABLE)
    if kind == "add":
        linear = Polynomial((at, -ONE), ROOT_VARIABLE)
        for coefficient in reversed(second.coefficients):
            composed = composed*linear + coefficient
        return first, composed
    if kind == "mul":
        composed = Polynomial((second.coefficients[0],), ROOT_VARIABLE)
        power = ONE
        for coefficient in second.coefficients[1:]:
            power *= at
            composed = composed*t + coefficient*power
        return first, composed
    if kind == "div":
        linear = Polynomial((ZERO, at), ROOT_VARIABLE)
        for coefficient in reversed(first.coefficients):
            composed = composed*linear + coefficient
        return second, composed
    raise InvalidInput("Unknown elimination claim")


def _checker_enclosure(kind, intervals, exponent=None):
    low, high = intervals[0]
    if kind in ("root", "sqrt"):
        return low, high
    if kind == "pow":
        if exponent < 0:
            if low <= ZERO and high >= ZERO:
                return None
            low, high = ONE/high, ONE/low
            exponent = -exponent
        if exponent == 0:
            return ONE, ONE
        if exponent % 2:
            return low**exponent, high**exponent
        if low >= ZERO:
            return low**exponent, high**exponent
        if high <= ZERO:
            return high**exponent, low**exponent
        return ZERO, max((-low)**exponent, high**exponent)
    a, b = intervals[1]
    if kind == "add":
        return low+a, high+b
    if kind == "div":
        if a <= ZERO and b >= ZERO:
            return None
        a, b = ONE/b, ONE/a
    products = [low*a, low*b, high*a, high*b]
    products.sort()
    return products[0], products[-1]


def _checker_direct(kind, values, claimed, exponent=None):
    """Check elementary identities without asking the producer for a result."""
    if kind == "pow":
        base = values[0]
        if exponent == 0:
            return claimed == ONE
        if exponent == 1:
            return claimed == base
        if type(base) is not Rational:
            return False
        if exponent < 0:
            return base != ZERO and claimed * (base**(-exponent)) == ONE
        return claimed == base**exponent
    left, right = values
    if kind == "add":
        if left == ZERO:
            return claimed == right
        if right == ZERO:
            return claimed == left
        return type(left) is Rational and type(right) is Rational and claimed == left+right
    if kind == "mul":
        if left == ZERO or right == ZERO:
            return claimed == ZERO
        if left == ONE:
            return claimed == right
        if right == ONE:
            return claimed == left
        return type(left) is Rational and type(right) is Rational and claimed == left*right
    if right == ZERO:
        return False
    if left == ZERO:
        return claimed == ZERO
    if left == right:
        return claimed == ONE
    if right == ONE:
        return claimed == left
    return type(left) is Rational and type(right) is Rational and claimed*right == left


def _interval_product(a, b):
    values = tuple(x*y for x in a for y in b)
    return min(values), max(values)


def _enclosure(kind, intervals, exponent=None):
    left = intervals[0]
    if kind in ("root", "sqrt"):
        return left
    if kind == "pow":
        n = abs(exponent)
        if not n:
            return ONE, ONE
        ends = (left[0]**n, left[1]**n)
        result = (min(ends), max(ends))
        if n % 2 == 0 and left[0] <= ZERO <= left[1]:
            result = ZERO, result[1]
        if exponent < 0:
            if result[0] <= ZERO <= result[1]:
                return None
            result = ONE/result[1], ONE/result[0]
        return result
    right = intervals[1]
    if kind == "add":
        return left[0]+right[0], left[1]+right[1]
    if kind == "mul":
        return _interval_product(left, right)
    if right[0] <= ZERO <= right[1]:
        return None
    return _interval_product(left, (ONE/right[1], ONE/right[0]))


def _contains(polynomial, interval):
    return count_real_roots(polynomial, *interval, include_lower=True, include_upper=True)


def _valid_interval(value, interval):
    if (type(interval) is not tuple or len(interval) != 2 or
            any(type(x) is not Rational for x in interval) or interval[0] > interval[1]):
        return False
    if type(value) is Rational:
        return interval == (value, value)
    polynomial = _checker_defining(value)
    return (_contains(polynomial, interval) == 1 and
            count_real_roots(polynomial, None, interval[0]) == value.real_index)


def _select(kind, values, annihilator, exponent=None):
    from .polynomial_exact import factor_with_evidence
    if annihilator.is_zero or annihilator.degree <= 0:
        raise ArithmeticError("Elimination did not produce a nonconstant annihilator")
    factorization, factor_evidence = factor_with_evidence(annihilator)
    intervals = tuple(interval_for(value) for value in values)
    while True:
        current_budget().tick()
        enclosure = _enclosure(kind, intervals, exponent)
        if enclosure is not None and _contains(annihilator, enclosure) == 1:
            break
        refined = []
        for value, interval in zip(values, intervals):
            refined.append(interval if interval[0] == interval[1] else
                           refine_interval(_defining(value), interval))
        intervals = tuple(refined)
    selected = [item.polynomial for item in factorization.factors if _contains(item.polynomial, enclosure)]
    if len(selected) != 1:
        raise ArithmeticError("Factor selection did not identify exactly one real root")
    polynomial = selected[0]
    if polynomial.degree == 1:
        value = -polynomial.coefficients[0] / polynomial.coefficients[1]
    else:
        index = count_real_roots(polynomial, None, enclosure[0])
        value = RealAlgebraicRoot(primitive_integer_coefficients(polynomial), index)
    certificate = (annihilator, factorization, factor_evidence, intervals)
    return value, certificate


def _check_selection(kind, values, value, certificate, exponent=None, expected=None):
    from .polynomial_exact import check_factor
    if type(certificate) is not tuple or len(certificate) != 4:
        return False
    polynomial, factorization, factor_evidence, intervals = certificate
    if (type(polynomial) is not Polynomial or polynomial.variable != ROOT_VARIABLE or
            polynomial.degree <= 0 or type(intervals) is not tuple or len(intervals) != len(values)):
        return False
    if expected is not None:
        if polynomial != expected:
            return False
    elif not _check_annihilator(kind, values, polynomial, exponent):
        return False
    if not check_factor(polynomial, factorization, factor_evidence):
        return False
    if not all(_valid_interval(v, interval) for v, interval in zip(values, intervals)):
        return False
    enclosure = _checker_enclosure(kind, intervals, exponent)
    if enclosure is None or _contains(polynomial, enclosure) != 1:
        return False
    selected = [item.polynomial for item in factorization.factors if _contains(item.polynomial, enclosure)]
    if len(selected) != 1:
        return False
    chosen = selected[0]
    if chosen.degree == 1:
        return type(value) is Rational and value == -chosen.coefficients[0] / chosen.coefficients[1]
    return (type(value) is RealAlgebraicRoot and
            value.integer_coefficients == primitive_integer_coefficients(chosen) and
            value.real_index == count_real_roots(chosen, None, enclosure[0]))


def _rational_result(kind, values, exponent=None):
    """Cheap identities; returning None means elimination is necessary."""
    if kind == "pow":
        if exponent == 0:
            return ONE
        if exponent == 1:
            return values[0]
        if exponent < 0 and values[0] == ZERO:
            raise InvalidInput("An algebraic denominator is zero", code="division_by_zero")
        if type(values[0]) is Rational:
            return values[0]**exponent if exponent >= 0 else ONE/(values[0]**(-exponent))
        return None
    left, right = values
    if kind == "div" and right == ZERO:
        raise InvalidInput("An algebraic denominator is zero", code="division_by_zero")
    if kind == "mul" and (left == ZERO or right == ZERO):
        return ZERO
    if kind == "div" and left == ZERO:
        return ZERO
    if kind == "add" and left == ZERO:
        return right
    if kind == "add" and right == ZERO:
        return left
    if kind == "mul" and left == ONE:
        return right
    if kind in ("mul", "div") and right == ONE:
        return left
    if kind == "div" and left == right:
        return ONE
    if type(left) is Rational and type(right) is Rational:
        return {"add": lambda: left+right, "mul": lambda: left*right,
                "div": lambda: left/right}[kind]()
    return None


class _Engine:
    def __init__(self):
        self.nodes = []
        self.memo = {}

    def read(self, recipe, depth=0):
        if depth > current_budget().limits.max_depth:
            raise ResourceLimitError("Algebraic expression exceeds the depth budget")
        if recipe in self.memo:
            return self.memo[recipe]
        current_budget().tick()
        kind = recipe[0]
        if kind == "rational":
            operands, value, certificate = (recipe[1],), recipe[1], ()
        elif kind in ("root", "sqrt"):
            source = recipe[1] if kind == "root" else as_endpoint(Sqrt(recipe[1]))
            operands = (recipe[1],)
            if type(source) is Rational:
                value, certificate = source, ("rational",)
            else:
                annihilator = (_defining(source) if kind == "root" else _poly((-recipe[1], ZERO, ONE)))
                value, certificate = _select(kind, (source,), annihilator)
        elif kind == "rf":
            function, argument_recipe = recipe[1:]
            argument = self.read(argument_recipe, depth+1)
            if not function.is_defined_at(self.nodes[argument][2]):
                raise InvalidInput("The rational function is undefined at this point", code="undefined_at_point")
            quotient = ("div", _polynomial_recipe(function.numerator, argument_recipe),
                        _polynomial_recipe(function.denominator, argument_recipe))
            evaluated = self.read(quotient, depth+1)
            operands, value, certificate = (argument, evaluated), self.nodes[evaluated][2], ()
        else:
            indices = ((self.read(recipe[1], depth+1),) if kind == "pow" else
                       (self.read(recipe[1], depth+1), self.read(recipe[2], depth+1)))
            values = tuple(self.nodes[index][2] for index in indices)
            exponent = recipe[2] if kind == "pow" else None
            operands = (*indices, exponent) if kind == "pow" else indices
            value = _rational_result(kind, values, exponent)
            if value is not None:
                certificate = ("rational",)
            else:
                value, certificate = _select(kind, values, _annihilator(kind, values, exponent), exponent)
        index = len(self.nodes)
        self.nodes.append((kind, operands, value, certificate))
        current_budget().evidence()
        self.memo[recipe] = index
        return index


def _check_trace(recipe, nodes, root):
    if type(nodes) is not tuple or type(root) is not int or not 0 <= root < len(nodes):
        return False
    visited, expected_at = set(), {}

    def check(expected, index, depth=0):
        current_budget().tick()
        if depth > current_budget().limits.max_depth:
            raise ResourceLimitError("Algebraic certificate exceeds the depth budget")
        if type(index) is not int or not 0 <= index < len(nodes):
            return False
        if index in expected_at:
            return expected_at[index] == expected
        node = nodes[index]
        if type(node) is not tuple or len(node) != 4:
            return False
        kind, operands, value, certificate = node
        if kind != expected[0] or type(operands) is not tuple or type(value) not in (Rational, RealAlgebraicRoot):
            return False
        expected_at[index] = expected
        if kind == "rational":
            valid = operands == (expected[1],) and value == expected[1] and certificate == ()
        elif kind in ("root", "sqrt"):
            if operands != (expected[1],):
                return False
            source = expected[1] if kind == "root" else _checker_sqrt(expected[1])
            if type(source) is Rational:
                valid = value == source and certificate == ("rational",)
            else:
                polynomial = _checker_defining(source) if kind == "root" else _poly((-expected[1], ZERO, ONE))
                valid = _check_selection(kind, (source,), value, certificate, expected=polynomial)
        elif kind == "rf":
            if len(operands) != 2 or any(type(i) is not int or not 0 <= i < index for i in operands):
                return False
            function, argument_recipe = expected[1:]
            quotient = ("div", _checker_polynomial_recipe(function.numerator, argument_recipe),
                        _checker_polynomial_recipe(function.denominator, argument_recipe))
            valid = (check(argument_recipe, operands[0], depth+1) and check(quotient, operands[1], depth+1) and
                     function.is_defined_at(nodes[operands[0]][2]) and
                     value == nodes[operands[1]][2] and certificate == ())
        elif kind in ("add", "mul", "div", "pow"):
            indices = operands[:1] if kind == "pow" else operands
            if len(operands) != 2 or any(type(i) is not int or not 0 <= i < index for i in indices):
                return False
            exponent = expected[2] if kind == "pow" else None
            if kind == "pow" and (type(operands[1]) is not int or operands[1] != exponent):
                return False
            if not check(expected[1], indices[0], depth+1):
                return False
            if kind != "pow" and not check(expected[2], indices[1], depth+1):
                return False
            values = tuple(nodes[i][2] for i in indices)
            if ((kind == "div" and values[1] == ZERO) or
                    (kind == "pow" and exponent < 0 and values[0] == ZERO)):
                return False
            valid = (_checker_direct(kind, values, value, exponent) if certificate == ("rational",) else
                     _check_selection(kind, values, value, certificate, exponent))
        else:
            return False
        if valid:
            visited.add(index)
        return valid

    return check(recipe, root) and visited == set(range(len(nodes)))


def _evidence(engine, roots):
    evidence = ("algebraic_trace_v1", tuple(engine.nodes), tuple(roots))
    current_budget().inspect(evidence, evidence=True)
    return evidence


def produce(operation, args, *, limits=None):
    with root_work(limits) as budget:
        budget.inspect(args)
        validate_exact_scalars(args)
        engine = _Engine()
        if operation == "compare_real":
            left = engine.read(_recipe(args["left"]))
            right = engine.read(_recipe(args["right"]))
            value = compare_real(engine.nodes[left][2], engine.nodes[right][2])
            return value, _evidence(engine, (left, right))
        recipe = _operation_recipe(operation, args)
        index = engine.read(recipe)
        value = engine.nodes[index][2]
        if operation == "minimal_polynomial":
            polynomial = _defining(value)
            value = Polynomial(polynomial.coefficients, args["variable"])
        return value, _evidence(engine, (index,))


def check_operation(operation, args, value, evidence, *, limits=None):
    with root_work(limits) as budget:
        budget.inspect(args)
        budget.inspect((value, evidence), evidence=True)
        try:
            validate_exact_scalars((args, value, evidence))
            if type(evidence) is not tuple or len(evidence) != 3 or evidence[0] != "algebraic_trace_v1":
                return False
            nodes, roots = evidence[1:]
            if type(nodes) is not tuple or type(roots) is not tuple:
                return False
            if operation == "compare_real":
                if len(roots) != 2 or type(value) is not int or value not in (-1, 0, 1):
                    return False
                # Roots may share nodes; check their reachable source DAGs.
                if not _check_multi_trace((_checker_source(args["left"]), _checker_source(args["right"])), nodes, roots):
                    return False
                return value == compare_real(nodes[roots[0]][2], nodes[roots[1]][2])
            recipe = _checker_request(operation, args)
            if len(roots) != 1 or not _check_trace(recipe, nodes, roots[0]):
                return False
            normalized = nodes[roots[0]][2]
            if operation == "minimal_polynomial":
                return (type(value) is Polynomial and value.variable == args["variable"] and
                        value.coefficients == _checker_defining(normalized).coefficients)
            return type(value) is type(normalized) and value == normalized
        except (InvalidInput, UnsupportedOperation, AttributeError, IndexError, KeyError,
                TypeError, ValueError, ZeroDivisionError):
            return False


def _check_multi_trace(recipes, nodes, roots):
    """Project each reachable DAG without invoking the arithmetic engine."""
    all_reachable = set()
    for recipe, root in zip(recipes, roots):
        reachable = set()

        def visit(index, depth=0):
            if depth > current_budget().limits.max_depth:
                raise ResourceLimitError("Algebraic certificate exceeds the depth budget")
            if type(index) is not int or not 0 <= index < len(nodes):
                raise InvalidInput("Invalid algebraic trace reference")
            if index in reachable:
                return
            node = nodes[index]
            if type(node) is not tuple or len(node) != 4:
                raise InvalidInput("Invalid algebraic trace node")
            kind, operands = node[:2]
            references = operands[:1] if kind == "pow" else operands if kind in ("add", "mul", "div", "rf") else ()
            for child in references:
                if type(child) is not int or child >= index:
                    raise InvalidInput("Noncausal algebraic trace reference")
                visit(child, depth+1)
            reachable.add(index)

        visit(root)
        order = sorted(reachable)
        indices = {original: new for new, original in enumerate(order)}
        projected = []
        for original in order:
            kind, operands, value, certificate = nodes[original]
            if kind == "pow":
                operands = (indices[operands[0]], operands[1])
            elif kind in ("add", "mul", "div", "rf"):
                operands = tuple(indices[i] for i in operands)
            projected.append((kind, operands, value, certificate))
        if not _check_trace(recipe, tuple(projected), indices[root]):
            return False
        all_reachable.update(reachable)
    return all_reachable == set(range(len(nodes)))
