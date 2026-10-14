"""Bounded sparse polynomials over the rationals, using the existing exact AST.

Axes are a nonempty ordered tuple of distinct, unconstrained real Symbols. All
symbols in an explicit-axis expression must occur in that tuple; names do not
identify symbols. ``partial_diff`` infers its axes (including the differentiation
variable) in UUID order. Inputs are existing expressions or exact Python ints;
polynomial/rational-function wrappers and irrational constants are unsupported.

Every result is a canonical expanded expression. Gradients preserve axis order;
Jacobians have one row per function and one column per axis; Hessians use axis
order on both dimensions. Derivative order zero returns the expanded identity.
Sparse coefficient maps are private, operation-local implementation details.

``check_operation`` checks canonical result shapes and coefficient identities
without calling producers or their differentiation/assembly helpers. Its empty
evidence tuple is intentional: exact source coefficients suffice for checking.
"""

from __future__ import annotations

from math import gcd

from .errors import InvalidInput, UnsupportedOperation
from .limits import computation, current_budget
from .model import Add, Expression, Mul, Pow, Rational, Reals, Symbol, _radd, _rmul, _sort_key


def _variable(variable):
    if type(variable) is not Symbol:
        raise InvalidInput("Multivariate axes must be Symbols")
    if variable.domain is not Reals or variable.assumptions:
        raise UnsupportedOperation("Only unconstrained real variables are supported")


def _axes(variables):
    if type(variables) is not tuple or not variables:
        raise InvalidInput("variables must be a nonempty ordered tuple of Symbols")
    seen = {}
    for variable in variables:
        _variable(variable)
        if variable.uid in seen:
            raise InvalidInput("Multivariate axes require distinct Symbol UUIDs")
        seen[variable.uid] = variable
    return seen


def _input(expression):
    if type(expression) is int:
        return Rational(expression)
    if type(expression) in (Rational, Symbol, Add, Mul, Pow):
        return expression
    if isinstance(expression, Expression) or type(expression).__name__ in ("Polynomial", "RationalFunction"):
        raise UnsupportedOperation("Multivariate operations require rational polynomial expressions")
    raise InvalidInput("Expected an exact expression or Python int; float and bool are unsupported")


def _scan(expressions, known, *, infer=False):
    """Validate the entire raw tree, including factors and bases that vanish."""
    budget = current_budget()
    pending = list(expressions)
    visited = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        budget.tick()
        if type(node) is Symbol:
            _variable(node)
            previous = known.get(node.uid)
            if previous is not None and previous != node:
                raise InvalidInput("A Symbol UUID has inconsistent metadata")
            if previous is None:
                if not infer:
                    raise UnsupportedOperation("Every free Symbol must be included in variables")
                known[node.uid] = node
        elif type(node) is Rational:
            budget.check_int(node.numerator)
            budget.check_int(node.denominator)
            if node.denominator <= 0 or gcd(node.numerator, node.denominator) != 1:
                raise InvalidInput("Rational coefficients must be normalized")
        elif type(node) is Add:
            pending.extend(node.terms)
        elif type(node) is Mul:
            pending.extend(node.factors)
        elif type(node) is Pow:
            if type(node.exponent) is not int or node.exponent < 0:
                raise UnsupportedOperation("Polynomial powers require nonnegative integer exponents")
            budget.check_int(node.exponent)
            pending.append(node.base)
        else:
            raise UnsupportedOperation("Only rational polynomial expressions are supported")


def _prepare(expressions, variables, *, infer=False):
    budget = current_budget()
    budget.inspect((expressions, variables))
    known = _axes(variables)
    expressions = tuple(_input(expression) for expression in expressions)
    _scan(expressions, known, infer=infer)
    if infer:
        variables = tuple(known[key] for key in sorted(known))
    return expressions, variables


def _store(polynomial, monomial, coefficient):
    """Every newly allocated sparse entry consumes the shared node budget."""
    if coefficient.numerator:
        if monomial not in polynomial:
            current_budget().nodes()
        polynomial[monomial] = coefficient
    else:
        polynomial.pop(monomial, None)


def _accumulate(result, terms):
    budget = current_budget()
    for monomial, coefficient in terms.items():
        budget.tick()
        previous = result.get(monomial)
        _store(result, monomial, coefficient if previous is None else _radd(previous, coefficient))


def _product(left, right):
    budget = current_budget()
    result = {}
    for first, a in left.items():
        for second, b in right.items():
            budget.tick(len(first) + 1)
            monomial = tuple(x + y for x, y in zip(first, second))
            for exponent in monomial:
                budget.check_int(exponent)
            coefficient = _rmul(a, b)
            if monomial in result:
                coefficient = _radd(result[monomial], coefficient)
            _store(result, monomial, coefficient)
    return result


def _convert_many(expressions, variables):
    """Convert each shared AST node once; exponentiation uses repeated squaring."""
    budget = current_budget()
    positions = {variable.uid: index for index, variable in enumerate(variables)}
    zero = (0,) * len(variables)
    one = Rational(1)
    memo = {}

    def convert(node):
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        budget.tick()
        if type(node) is Rational:
            result = {zero: node} if node.numerator else {}
            budget.nodes(len(result))
        elif type(node) is Symbol:
            powers = list(zero)
            powers[positions[node.uid]] = 1
            result = {tuple(powers): one}
            budget.nodes()
        elif type(node) is Add:
            result = {}
            for child in node.terms:
                _accumulate(result, convert(child))
        elif type(node) is Mul:
            result = {zero: one}
            budget.nodes()
            for child in node.factors:
                result = _product(result, convert(child))
        else:  # The complete input tree has already been validated.
            exponent = node.exponent
            result = {zero: one}
            budget.nodes()
            base = convert(node.base) if exponent else {}
            while exponent:
                budget.tick()
                if exponent & 1:
                    result = _product(result, base)
                exponent >>= 1
                if exponent:
                    base = _product(base, base)
        memo[id(node)] = result
        return result

    return tuple(convert(expression) for expression in expressions)


def _to_expression(polynomial, variables):
    """Assemble normalized flat nodes directly, avoiding repeated AST rewrites."""
    budget = current_budget()
    terms = []
    for monomial, coefficient in polynomial.items():
        budget.tick(len(variables) + 1)
        factors = [variable if exponent == 1 else Pow(variable, exponent)
                   for variable, exponent in zip(variables, monomial) if exponent]
        factors.sort(key=_sort_key)
        if not factors or coefficient != Rational(1):
            factors.insert(0, coefficient)
        terms.append(factors[0] if len(factors) == 1 else Mul(tuple(factors)))
        budget.nodes(len(factors) + 1)
    terms.sort(key=_sort_key)
    return Rational(0) if not terms else terms[0] if len(terms) == 1 else Add(tuple(terms))


def _differentiate(polynomial, axis, order=1):
    budget = current_budget()
    if order == 0:
        return polynomial
    result = {}
    for monomial, coefficient in polynomial.items():
        budget.tick()
        degree = monomial[axis]
        if degree < order:
            continue
        multiplier = 1
        for factor in range(degree - order + 1, degree + 1):
            budget.product(multiplier, factor)
            multiplier *= factor
            budget.check_int(multiplier)
        powers = list(monomial)
        powers[axis] -= order
        _store(result, tuple(powers), _rmul(coefficient, Rational(multiplier)))
    return result


def _order(n):
    if type(n) is not int or n < 0:
        raise InvalidInput("Derivative order must be a nonnegative integer")
    current_budget().check_int(n)


def expand_multivariate(expression, variables, *, limits=None) -> Expression:
    """Collect all rational polynomial terms on explicit, ordered real axes."""
    with computation(limits):
        expressions, variables = _prepare((expression,), variables)
        return _to_expression(_convert_many(expressions, variables)[0], variables)


def partial_diff(expression, variable, n=1, *, limits=None) -> Expression:
    """Return the n-th partial derivative, inferring every free real Symbol."""
    with computation(limits):
        _order(n)
        expressions, variables = _prepare((expression,), (variable,), infer=True)
        source = _convert_many(expressions, variables)[0]
        return _to_expression(_differentiate(source, variables.index(variable), n), variables)


def gradient(expression, variables, *, limits=None) -> tuple[Expression, ...]:
    """Return first partials in axis order, expanding the source only once."""
    with computation(limits) as budget:
        expressions, variables = _prepare((expression,), variables)
        budget.matrix(1, len(variables))
        source = _convert_many(expressions, variables)[0]
        return tuple(_to_expression(_differentiate(source, axis), variables) for axis in range(len(variables)))


def jacobian(expressions, variables, *, limits=None) -> tuple[tuple[Expression, ...], ...]:
    """Return function-by-axis partials, sharing conversion of common AST nodes."""
    with computation(limits) as budget:
        if type(expressions) is not tuple or not expressions:
            raise InvalidInput("Jacobian expressions must be a nonempty ordered tuple")
        expressions, variables = _prepare(expressions, variables)
        budget.matrix(len(expressions), len(variables))
        return tuple(tuple(_to_expression(_differentiate(source, axis), variables)
                           for axis in range(len(variables)))
                     for source in _convert_many(expressions, variables))


def hessian(expression, variables, *, limits=None) -> tuple[tuple[Expression, ...], ...]:
    """Return symmetric polynomial second partials, computing each pair once."""
    with computation(limits) as budget:
        expressions, variables = _prepare((expression,), variables)
        size = len(variables)
        budget.matrix(size, size)
        source = _convert_many(expressions, variables)[0]
        rows = [[None] * size for _ in variables]
        for first in range(size):
            derivative = _differentiate(source, first)
            for second in range(first, size):
                value = _to_expression(_differentiate(derivative, second), variables)
                rows[first][second] = rows[second][first] = value
        return tuple(tuple(row) for row in rows)


def _read_claim(value, variables):
    """Independently read a canonical expanded AST, rejecting malformed shapes."""
    if type(value) not in (Rational, Symbol, Pow, Mul, Add):
        raise InvalidInput("Claimed values must be exact expanded expressions")
    known = _axes(variables)
    _scan((value,), known)
    positions = {variable.uid: index for index, variable in enumerate(variables)}
    terms = value.terms if type(value) is Add else (value,)
    if tuple(sorted(terms, key=_sort_key)) != terms:
        raise InvalidInput("Expanded terms must use canonical order")
    result = {}
    budget = current_budget()
    for term in terms:
        budget.tick()
        coefficient = Rational(1)
        factors = term.factors if type(term) is Mul else (term,)
        if type(factors[0]) is Rational:
            coefficient, factors = factors[0], factors[1:]
            if factors and coefficient == Rational(1):
                raise InvalidInput("Redundant unit coefficient")
        if tuple(sorted(factors, key=_sort_key)) != factors:
            raise InvalidInput("Expanded factors must use canonical order")
        powers = [0] * len(variables)
        for factor in factors:
            budget.tick()
            if type(factor) is Symbol:
                variable, exponent = factor, 1
            elif type(factor) is Pow and type(factor.base) is Symbol and factor.exponent >= 2:
                variable, exponent = factor.base, factor.exponent
            else:
                raise InvalidInput("A claimed term is not an expanded monomial")
            axis = positions[variable.uid]
            if powers[axis]:
                raise InvalidInput("Repeated variable in a claimed monomial")
            powers[axis] = exponent
        monomial = tuple(powers)
        if monomial in result or not coefficient.numerator and (len(terms) != 1 or factors):
            raise InvalidInput("Claimed terms must be nonzero and collected")
        _store(result, monomial, coefficient)
    return result


def _check_derivative(source, claimed, orders):
    """Coefficient-wise falling-factor identity, separate from producer calculus."""
    budget = current_budget()
    matched = 0
    for powers, coefficient in source.items():
        budget.tick(len(orders) + 1)
        if any(degree < order for degree, order in zip(powers, orders)):
            continue
        target = tuple(degree - order for degree, order in zip(powers, orders))
        numerator = coefficient.numerator
        denominator = coefficient.denominator
        for degree, order in zip(powers, orders):
            for offset in range(order):
                factor = degree - offset
                # Cancel before multiplication; this checker uses integer
                # arithmetic independently of the producer's Rational factors.
                divisor = gcd(factor, denominator)
                factor //= divisor
                denominator //= divisor
                budget.product(numerator, factor)
                numerator *= factor
                budget.check_int(numerator)
        reported = claimed.get(target)
        if reported is None or reported.numerator != numerator or reported.denominator != denominator:
            return False
        matched += 1
    return len(claimed) == matched


def _checked_sources(expressions, variables):
    """Independently expand source trees using normalized integer-pair maps.

    No producer polynomial conversion, collection, convolution, coefficient
    arithmetic or expression assembly helper is used. Left-to-right powering
    and an explicit postorder traversal also differ from producer conversion.
    Shared code is limited to validated AST/scalar representations and budgets.
    """
    budget = current_budget()
    width = len(variables)
    positions = {variable.uid: index for index, variable in enumerate(variables)}
    zero = (0,) * width

    def product_integer(a, b):
        budget.product(a, b)
        result = a * b
        budget.check_int(result)
        return result

    def plus(a, b):
        common = gcd(a[1], b[1])
        left, right = b[1] // common, a[1] // common
        numerator = product_integer(a[0], left) + product_integer(b[0], right)
        budget.check_int(numerator)
        denominator = product_integer(a[1], left)
        divisor = gcd(numerator, denominator)
        return numerator // divisor, denominator // divisor

    def times(a, b):
        first, second = gcd(a[0], b[1]), gcd(b[0], a[1])
        return (product_integer(a[0] // first, b[0] // second),
                product_integer(a[1] // second, b[1] // first))

    def insert(target, powers, coefficient):
        budget.tick()
        if powers in target:
            coefficient = plus(target[powers], coefficient)
        if coefficient[0]:
            if powers not in target:
                budget.nodes()
            target[powers] = coefficient
        else:
            target.pop(powers, None)

    def convolution(left, right):
        result = {}
        for a, x in left.items():
            for b, y in right.items():
                budget.tick(width + 1)
                powers = tuple(a[index] + b[index] for index in range(width))
                for power in powers:
                    budget.check_int(power)
                insert(result, powers, times(x, y))
        return result

    memo = {}
    for expression in expressions:
        pending = [(expression, False)]
        while pending:
            node, ready = pending.pop()
            if id(node) in memo:
                continue
            budget.tick()
            if not ready and type(node) in (Add, Mul, Pow):
                pending.append((node, True))
                children = node.terms if type(node) is Add else node.factors if type(node) is Mul else (node.base,)
                pending.extend((child, False) for child in reversed(children))
                continue
            if type(node) is Rational:
                result = {zero: (node.numerator, node.denominator)} if node.numerator else {}
                budget.nodes(len(result))
            elif type(node) is Symbol:
                powers = tuple(1 if index == positions[node.uid] else 0 for index in range(width))
                result = {powers: (1, 1)}
                budget.nodes()
            elif type(node) is Add:
                result = {}
                for child in node.terms:
                    for powers, coefficient in memo[id(child)].items():
                        insert(result, powers, coefficient)
            elif type(node) is Mul:
                result = {zero: (1, 1)}
                budget.nodes()
                for child in reversed(node.factors):
                    result = convolution(memo[id(child)], result)
            else:
                result = {zero: (1, 1)}
                budget.nodes()
                base = memo[id(node.base)]
                for bit in range(node.exponent.bit_length() - 1, -1, -1):
                    budget.tick()
                    result = convolution(result, result)
                    if (node.exponent >> bit) & 1:
                        result = convolution(base, result)
            memo[id(node)] = result
    return tuple({powers: Rational(*coefficient) for powers, coefficient in memo[id(expression)].items()}
                 for expression in expressions)


_ARGUMENTS = {
    "expand_multivariate": {"expression", "variables"},
    "partial_diff": {"expression", "variable", "n"},
    "gradient": {"expression", "variables"},
    "jacobian": {"expressions", "variables"},
    "hessian": {"expression", "variables"},
}


def check_operation(operation, args, value, evidence):
    """Check exact coefficients, axes, shape and order; limits propagate.

    ``args`` is the exact recorded argument dict, including ``n`` for partial
    derivatives. No producer entry point, derivative helper or renderer is used.
    """
    with computation() as budget:
        budget.inspect((args, value, evidence), evidence=True)
        try:
            if (type(operation) is not str or operation not in _ARGUMENTS or type(args) is not dict
                    or set(args) != _ARGUMENTS[operation] or type(evidence) is not tuple or evidence != ()):
                return False
            if operation == "partial_diff":
                _order(args["n"])
                expressions, variables = _prepare((args["expression"],), (args["variable"],), infer=True)
            else:
                expressions = args["expressions"] if operation == "jacobian" else (args["expression"],)
                if type(expressions) is not tuple or not expressions:
                    return False
                expressions, variables = _prepare(expressions, args["variables"])
            sources = _checked_sources(expressions, variables)
            size = len(variables)
            zero = (0,) * size
            if operation == "expand_multivariate":
                return _read_claim(value, variables) == sources[0]
            if operation == "partial_diff":
                orders = list(zero)
                orders[variables.index(args["variable"])] = args["n"]
                return _check_derivative(sources[0], _read_claim(value, variables), orders)
            if type(value) is not tuple:
                return False
            if operation == "gradient":
                if len(value) != size:
                    return False
                rows = (value,)
            else:
                count = size if operation == "hessian" else len(expressions)
                if len(value) != count or any(type(row) is not tuple or len(row) != size for row in value):
                    return False
                rows = value
            budget.matrix(len(rows), size)
            for row_index, row in enumerate(rows):
                for axis, claimed in enumerate(row):
                    orders = list(zero)
                    orders[axis] += 1
                    if operation == "hessian":
                        orders[row_index] += 1
                    source = sources[row_index] if operation == "jacobian" else sources[0]
                    if not _check_derivative(source, _read_claim(claimed, variables), orders):
                        return False
            return True
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, UnsupportedOperation):
            return False
