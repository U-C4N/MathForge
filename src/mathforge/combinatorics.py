"""Exact combinatorial counts with bounded producers and separate identities."""
from __future__ import annotations

from .errors import InvalidInput
from .limits import computation, current_budget, ResourceLimitError


ARGUMENTS = {
    "factorial": {"n"}, "binomial": {"n", "k"}, "multinomial": {"counts"},
    "permutations_count": {"n", "k"}, "fibonacci": {"n"},
    "catalan": {"n"}, "stirling": {"n", "k"},
}
TYPES = {}


def _natural(value):
    if type(value) is not int or value < 0:
        raise InvalidInput("Combinatorial indices must be nonnegative exact integers")
    current_budget().check_int(value)
    return value


def _mul(a, b):
    current_budget().product(a, b)
    value = a * b
    current_budget().check_int(value)
    return value


def _add(a, b):
    current_budget().tick()
    value = a + b
    current_budget().check_int(value)
    return value


def _power(base, exponent):
    value = 1
    while exponent:
        current_budget().tick()
        if exponent & 1:
            value = _mul(value, base)
        exponent //= 2
        if exponent:
            base = _mul(base, base)
    return value


def _factorial(n):
    value = 1
    for item in range(2, n + 1):
        value = _mul(value, item)
    return value


def _choose(n, k):
    if k > n:
        return 0
    k = min(k, n - k)
    value = 1
    for index in range(1, k + 1):
        value = _mul(value, n - k + index) // index
    return value


def _fib(n):
    if n > 2 * current_budget().limits.max_integer_bits + 2:
        raise ResourceLimitError("Fibonacci output exceeds the integer bit budget")
    a, b = 0, 1
    for bit in bin(n)[2:]:
        current_budget().tick()
        c = _mul(a, _add(_mul(2, b), -a))
        d = _add(_mul(a, a), _mul(b, b))
        a, b = (c, d) if bit == "0" else (d, _add(c, d))
    return a


def _stirling(n, k):
    if k > n:
        return 0
    if k == 0:
        return int(n == 0)
    if k in (1, n):
        return 1
    current_budget().nodes(k + 1)
    row = [1] + [0] * k
    for degree in range(1, n + 1):
        for count in range(min(degree, k), 0, -1):
            row[count] = _add(row[count - 1], _mul(count, row[count]))
        row[0] = 0
    return row[k]


def _validate(operation, args):
    if operation not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[operation]:
        raise InvalidInput("Invalid combinatorics operation arguments")
    if operation == "multinomial":
        if type(args["counts"]) is not tuple:
            raise InvalidInput("Multinomial counts must be an immutable tuple")
        for value in args["counts"]:
            _natural(value)
    else:
        for value in args.values():
            _natural(value)


def _produce(operation, args, *, limits=None):
    with computation(limits) as budget:
        budget.inspect(args)
        _validate(operation, args)
        if operation == "multinomial":
            total, value, coefficients = 0, 1, []
            for count in args["counts"]:
                total = _add(total, count)
                coefficient = _choose(total, count)
                value = _mul(value, coefficient)
                coefficients.append(coefficient)
            budget.inspect(coefficients, evidence=True)
            return value, tuple(coefficients)
        n = args["n"]
        if operation == "factorial":
            return _factorial(n), ()
        if operation == "binomial":
            return _choose(n, args["k"]), ()
        if operation == "permutations_count":
            k = args["k"]
            value = 0 if k > n else 1
            if k <= n:
                for item in range(n - k + 1, n + 1):
                    value = _mul(value, item)
            return value, ()
        if operation == "fibonacci":
            return _fib(n), ()
        if operation == "catalan":
            return _choose(_mul(2, n), n) // (n + 1), ()
        return _stirling(n, args["k"]), ()


def produce(operation, args, *, limits=None):
    with computation(limits) as budget:
        value, evidence = _produce(operation, args)
        budget.inspect(value)
        budget.inspect(evidence, evidence=True)
        return value, evidence


def _check_choose(n, k, value):
    """Reverse the binomial recurrence from the claim back to C(n-k,0)=1."""
    if type(value) is not int or value < 0:
        return False
    if k > n:
        return value == 0
    k = min(k, n - k)
    for index in range(k, 0, -1):
        dividend = _mul(value, index)
        value, remainder = divmod(dividend, n - k + index)
        if remainder:
            return False
    return value == 1


def _balanced_product(low, high, depth=0):
    current_budget().tick()
    if depth > current_budget().limits.max_depth:
        raise ResourceLimitError("Combinatorial identity exceeds the depth budget")
    if low > high:
        return 1
    if low == high:
        return low
    middle = (low + high) // 2
    return _mul(_balanced_product(low, middle, depth+1), _balanced_product(middle + 1, high, depth+1))


def _matmul(a, b):
    return tuple(_add(_mul(a[2*i], b[j]), _mul(a[2*i+1], b[j+2]))
                 for i in range(2) for j in range(2))


def _check_fibonacci(n, value):
    if n > 2 * current_budget().limits.max_integer_bits + 2:
        raise ResourceLimitError("Fibonacci output exceeds the integer bit budget")
    # The off-diagonal entry of [[1,1],[1,0]]**n is F_n.
    matrix, power = (1, 0, 0, 1), (1, 1, 1, 0)
    while n:
        current_budget().tick()
        if n & 1:
            matrix = _matmul(matrix, power)
        n //= 2
        if n:
            power = _matmul(power, power)
    return value == matrix[1]


def _check_stirling(n, k, value):
    if k > n:
        return value == 0
    if k == 0:
        return value == int(n == 0)
    if k in (1, n):
        return value == 1
    # Inclusion/exclusion for onto maps: k! S(n,k) = sum (-1)**(k-j) C(k,j) j**n.
    total, coefficient = 0, 1
    for index in range(k + 1):
        term = _mul(coefficient, _power(index, n))
        total = _add(total, -term if (k - index) & 1 else term)
        if index < k:
            coefficient = _mul(coefficient, k - index) // (index + 1)
    return _mul(value, _balanced_product(1, k)) == total


def check_operation(operation, args, value, evidence, *, limits=None):
    with computation(limits) as budget:
        try:
            budget.inspect(args)
            budget.inspect((value, evidence), evidence=True)
            _validate(operation, args)
            if type(value) is not int or value < 0 or type(evidence) is not tuple:
                return False
            if operation == "multinomial":
                if len(evidence) != len(args["counts"]):
                    return False
                total, product = 0, 1
                for count, coefficient in zip(args["counts"], evidence):
                    total = _add(total, count)
                    if not _check_choose(total, count, coefficient):
                        return False
                    product = _mul(product, coefficient)
                return product == value
            if evidence:
                return False
            n = args["n"]
            if operation == "factorial":
                return value == _balanced_product(1, n)
            if operation == "binomial":
                return _check_choose(n, args["k"], value)
            if operation == "permutations_count":
                k = args["k"]
                if k > n:
                    return value == 0
                # Successively cancel each choice; the empty arrangement has count one.
                for divisor in range(n, n-k, -1):
                    budget.tick()
                    value, remainder = divmod(value, divisor)
                    if remainder:
                        return False
                return value == 1
            if operation == "fibonacci":
                return _check_fibonacci(n, value)
            if operation == "catalan":
                expected = 1
                for index in range(n):
                    expected = _mul(expected, 2 * (2 * index + 1)) // (index + 2)
                return value == expected
            return _check_stirling(n, args["k"], value)
        except (InvalidInput, TypeError, ValueError, AttributeError, KeyError, IndexError, ZeroDivisionError):
            return False
