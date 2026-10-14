"""Explicit v4 operation registry; persisted names never select arbitrary code."""

from .errors import InvalidInput, UnsupportedOperation


ARGUMENTS = {
    "factor": {"f"}, "resultant": {"f", "g"}, "discriminant": {"f"},
    "algebraic": {"expression"}, "algebraic_div": {"left", "right"},
    "algebraic_pow": {"base", "n"}, "minimal_polynomial": {"expression", "variable"},
    "evaluate_algebraic": {"expression", "bindings"}, "apart": {"expression"},
    "extrema_values": {"expression", "variable"}, "charpoly": {"A", "variable"},
    "real_eigenvalues": {"A"}, "compare_real": {"left", "right"},
    "limit": {"expression", "variable", "point", "side"},
    "integrate": {"expression", "variable", "lower", "upper"},
}
NEW_OPERATIONS = frozenset(ARGUMENTS) - {"compare_real", "limit", "integrate"}


def validate_arguments(operation, args):
    if operation not in ARGUMENTS:
        raise UnsupportedOperation("Unknown v4 operation")
    if type(args) is not dict or set(args) != ARGUMENTS[operation]:
        raise InvalidInput("Recorded arguments do not match the v4 operation contract")
    from .model import Expression, Symbol
    from .polynomial import Polynomial
    from .matrices import Matrix
    from .rational_function import RationalFunction
    for key in ("f", "g"):
        if key in args and type(args[key]) is not Polynomial:
            raise InvalidInput("Polynomial operands are required")
    if "A" in args and type(args["A"]) is not Matrix:
        raise InvalidInput("A rational Matrix is required")
    if "variable" in args and type(args["variable"]) is not Symbol:
        raise InvalidInput("An explicit Symbol is required")
    if "expression" in args:
        allowed = (Expression, Polynomial, RationalFunction)
        if not isinstance(args["expression"], allowed):
            raise InvalidInput("An exact expression, polynomial or rational function is required")
    if operation == "apart" and type(args["expression"]) is not RationalFunction:
        raise InvalidInput("apart requires a RationalFunction")
    for key in ("left", "right", "base", "lower", "upper"):
        if key in args and not isinstance(args[key], Expression):
            raise InvalidInput("Exact scalar operands are required")
    if "n" in args and type(args["n"]) is not int:
        raise InvalidInput("The exponent must be a Python integer")
    if "bindings" in args:
        values = args["bindings"]
        if type(values) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                or type(pair[0]) is not Symbol or not isinstance(pair[1], Expression) for pair in values):
            raise InvalidInput("Bindings must contain Symbol/exact-expression pairs")
        if len({pair[0].uid for pair in values}) != len(values):
            raise InvalidInput("Duplicate binding symbol identity")


def _module(operation):
    if operation in ("factor", "resultant", "discriminant"):
        from . import polynomial_exact
        return polynomial_exact
    if operation in ("apart", "charpoly", "real_eigenvalues"):
        from . import exact_extensions
        return exact_extensions
    if operation in ("limit", "integrate", "extrema_values"):
        from . import algebraic_analysis
        return algebraic_analysis
    if operation in ARGUMENTS:
        from . import algebraic_arithmetic
        return algebraic_arithmetic
    raise UnsupportedOperation("Unknown v4 operation")


def produce(operation, args):
    validate_arguments(operation, args)
    return _module(operation).produce(operation, args)


def check_operation(operation, args, value, evidence):
    validate_arguments(operation, args)
    return _module(operation).check_operation(operation, args, value, evidence)
