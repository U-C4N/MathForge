"""Fixed v0.7 operation/type registry; no document selects executable code.

The tables come from :mod:`mathforge.registry`, built once and cached. These
views stay v5-only: ``mathforge.compute`` and its frozen MCP schema accept
exactly the v0.7 operations, whatever v6 families the registry also holds.
"""

from . import registry
from .errors import InvalidInput, UnsupportedOperation


def modules():
    return registry._family_modules(5)


def operation_arguments():
    return registry.operation_arguments(5)


def record_types():
    return registry.record_types(5)


def module_for(operation):
    if operation not in registry.operation_arguments(5):
        raise UnsupportedOperation("Unknown v5 operation")
    return registry.family(operation)


def validate_arguments(operation, args):
    module = module_for(operation)
    if type(args) is not dict or set(args) != registry.operation_arguments(5)[operation]:
        raise InvalidInput("Arguments do not match the v5 operation contract")
    return module


def produce(operation, args):
    return validate_arguments(operation, args).produce(operation, args)


def check_operation(operation, args, value, evidence):
    return validate_arguments(operation, args).check_operation(operation, args, value, evidence)
