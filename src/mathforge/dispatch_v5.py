"""Fixed v0.7 operation/type registry; no document selects executable code."""

from importlib import import_module

from .errors import InvalidInput, UnsupportedOperation

_MODULES = ("number_theory", "combinatorics", "complex_numbers", "probability",
            "statistics", "linear_extensions", "functions", "function_calculus",
            "rational_integration", "function_integration")


def modules():
    return tuple(import_module("." + name, __package__) for name in _MODULES)


def operation_arguments():
    result = {}
    for module in modules():
        for name, keys in getattr(module, "ARGUMENTS", {}).items():
            if name in result:
                raise RuntimeError("Duplicate v5 operation: " + name)
            result[name] = frozenset(keys)
    return result


def record_types():
    result = {}
    for module in modules():
        for tag, cls in getattr(module, "TYPES", {}).items():
            if tag in result:
                raise RuntimeError("Duplicate v5 type: " + tag)
            result[tag] = cls
    return result


def module_for(operation):
    for module in modules():
        if operation in getattr(module, "ARGUMENTS", {}):
            return module
    raise UnsupportedOperation("Unknown v5 operation")


def validate_arguments(operation, args):
    module = module_for(operation)
    if type(args) is not dict or set(args) != set(module.ARGUMENTS[operation]):
        raise InvalidInput("Arguments do not match the v5 operation contract")
    return module


def produce(operation, args):
    return validate_arguments(operation, args).produce(operation, args)


def check_operation(operation, args, value, evidence):
    return validate_arguments(operation, args).check_operation(operation, args, value, evidence)
