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


# operation -> (the v5 argument table it was looked up in, its family module, its argument names). One registry
# build answers every entry: an entry of an older table (the registry rebuilt) is looked up again.
_LOOKUPS: dict = {}


def _lookup(operation):
    table = registry.operation_arguments(5)
    entry = _LOOKUPS.get(operation)
    if entry is None or entry[0] is not table:
        names = table.get(operation)
        if names is None:
            raise UnsupportedOperation("Unknown v5 operation")
        entry = _LOOKUPS[operation] = (table, registry.family(operation), names)
    return entry


def module_for(operation):
    return _lookup(operation)[1]


def validate_arguments(operation, args):
    _, module, names = _lookup(operation)
    if type(args) is not dict or args.keys() != names:
        raise InvalidInput("Arguments do not match the v5 operation contract")
    return module


def produce(operation, args):
    return validate_arguments(operation, args).produce(operation, args)


def check_operation(operation, args, value, evidence):
    return validate_arguments(operation, args).check_operation(operation, args, value, evidence)
