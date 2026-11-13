"""Shared helpers of the v0.9 family tests (tests/v09/<area>/test_<family>.py).

A family test subclasses FamilyTestCase and names its family module in FAMILY. The family then stays registered
while the tests of the class run: through its area package once FAMILIES lists it, and through
registry._testing_families before that, so that the same tests pass before and after
`python tools/new_family.py register`. An unregistered family has no generated functions yet: run() and call()
reach its operations as those functions and the MCP boundary do, and value() answers what its value function will.

The helpers that take an operation need its family registered, so they are called from the tests of a
FamilyTestCase, or inside registered(family).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping
import contextlib
import dataclasses
import importlib
import json
from types import ModuleType
from typing import Any, ClassVar
import unittest

import mathforge as mf
from mathforge import lean, operations_v6, registry
from mathforge.capabilities import area_tool
from mathforge.contracts import CheckStatus, ExecutionStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.mcp.boundary import ToolResponse, call_tool
from mathforge.operations_v5 import _freeze

HUGE = 2 ** 65536  # One bit more than the default max_integer_bits admits.
CLAIMS = ("result.source", "result.contract", "result.trace", "result.mathematics")  # Of a verified v6 value.


@contextlib.contextmanager
def registered(family: ModuleType) -> Iterator[None]:
    """Keep a family registered for a with block; there is nothing to do once its area package lists it."""
    if family.__name__ in registry.v6_families():
        yield
    else:
        with registry._testing_families(family):
            yield


def run(operation: str, /, *, limits: ComputationLimits | None = None, **arguments: object) -> mf.Result[Any]:
    """The Result of operations_v6.run, called as the generated function mathforge.operations.<name> calls it: the
    arguments by name, with the default of each one that the call leaves out. TypeError for an unknown or a
    missing argument, as that function raises; a test of another argument set calls operations_v6.run itself."""
    return operations_v6.run(operation, _complete(registry.spec(operation), arguments), limits=limits)


def _complete(spec: registry.OperationSpec, arguments: Mapping[str, object]) -> dict[str, object]:
    """Every argument of an operation in the order of its spec, the defaults filled in as a generated function
    fills them; TypeError for an unknown or a missing argument, as a call of that function raises."""
    unknown = sorted(set(arguments) - {argument.name for argument in spec.arguments})
    if unknown:
        raise TypeError(f"{spec.name} takes no argument {', '.join(unknown)}")
    values = {}
    for argument in spec.arguments:
        if argument.name in arguments:
            values[argument.name] = arguments[argument.name]
        elif argument.default is None:
            raise TypeError(f"{spec.name} needs its argument {argument.name}")
        else:
            values[argument.name] = ast.literal_eval(argument.default)
    return values


def lean_value(operation: str, /, **arguments: object) -> object:
    """What the lean path answers: the verified value, or mathforge.lean.FALLBACK when it leaves the call to the
    structured path. Defaults are filled in."""
    values = _complete(registry.spec(operation), arguments)
    return getattr(lean.functions, operation)(*values.values())


def value(operation: str, /, **arguments: object) -> object:
    """What the value function mathforge.<area>.<name> returns, or raises. tools/generate_api.py writes that
    function when the family is registered; before that, this does what the function will: it takes the lean
    path's value, and the structured path's when the lean path falls back."""
    spec = registry.spec(operation)
    function = getattr(importlib.import_module("mathforge." + spec.area), spec.public_name, None)
    if function is not None:
        return function(**arguments)
    found = lean_value(operation, **arguments)
    if found is not lean.FALLBACK:
        return found
    return run(operation, **arguments).unwrap()


def call(operation: str, arguments: object, /, **fields: object) -> ToolResponse:
    """The response of the MCP area tool that serves an operation. arguments is the JSON object of the request;
    fields are its optional symbol_ids and limits."""
    tool = area_tool(registry.spec(operation))
    if tool is None:
        raise ValueError(f"{operation} is not exposed over MCP")
    return call_tool(tool, dict(fields, operation=operation, arguments=arguments))


def _symbol_ids(values: Iterable[object]) -> dict[str, str]:
    """The UUID of every symbol in some values by its name, read from their canonical JSON."""
    ids, pending = {}, [mf.to_data(_freeze(tuple(values)))["data"]]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            if node.get("type") == "symbol":
                ids[node["name"]] = node["uid"]
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)
    return ids


def call_values(operation: str, /, **arguments: object) -> ToolResponse:
    """call() for Python values: each is encoded through the kind of its argument, and symbol_ids carries the
    identities of their symbols, so that the Result equals the one that run() gives for the same values."""
    from mathforge.mcp.kinds import KINDS
    kinds = {argument.name: argument.kind for argument in registry.spec(operation).arguments}
    encoded = {name: KINDS[kinds[name]].encode(item) for name, item in arguments.items()}
    return call(operation, encoded, symbol_ids=_symbol_ids(arguments.values()))


def checker(family: ModuleType) -> ModuleType:
    """The checker module of a family: its CHECKER, imported (a testing family may hold the module itself)."""
    source = family.CHECKER
    return importlib.import_module(source) if type(source) is str else source


def verdict(family: ModuleType, operation: str, args: object, value: object, evidence: object = (), *,
            limits: ComputationLimits | None = None) -> object:
    """What a family's checker answers to a claim, called as the verifier calls it: inside a fresh budget. args
    are the arguments as a request holds them, where run() has frozen every list into a tuple."""
    with computation(limits, fresh=True):
        return checker(family).check_operation(operation, args, value, evidence)


def forged(cls: type, /, **fields: object) -> Any:
    """A record made with object.__new__, without __init__ and __post_init__, so that its fields may break the
    canonical form, which no decoded document can do. A field that is left out stays unset."""
    unknown = sorted(set(fields) - {field.name for field in dataclasses.fields(cls)})
    if unknown:
        raise TypeError(f"{cls.__name__} has no field {', '.join(unknown)}")
    record = object.__new__(cls)
    for name, item in fields.items():
        object.__setattr__(record, name, item)
    return record


def impostor(record: Any) -> Any:
    """An object of another class that has the record's class name and field values; only its module differs."""
    names = [field.name for field in dataclasses.fields(record)]
    cls = dataclasses.make_dataclass(type(record).__name__, names, frozen=True)
    return cls(*(getattr(record, name) for name in names))


def tampered(result: mf.Result[Any], value: object, evidence: object = None) -> mf.Result[Any]:
    """A verified Result with another value in it and in its step, with other evidence when some is given, and
    without the stored checks: what mf.verify has to reject."""
    step = result.steps[0]
    outputs = (value, step.outputs[1] if evidence is None else evidence)
    return dataclasses.replace(result, value=value, verification=(),
                               steps=(dataclasses.replace(step, outputs=outputs),))


def statuses(report: mf.VerificationReport) -> dict[str, CheckStatus]:
    """The status of each check of a report by its claim."""
    return {check.claim: check.status for check in report.checks}


def oversized_matrix(size: int = 65) -> mf.Matrix:
    """A square matrix with more entries than the default max_matrix_entries admits (65 x 65 by default), built
    inside a budget wide enough for it."""
    with computation(ComputationLimits(max_matrix_entries=size * size), fresh=True):
        return mf.Matrix(tuple(tuple((i + 2 * j) % 7 for j in range(size)) for i in range(size)))


class FamilyTestCase(unittest.TestCase):
    """The tests of one v6 family, FAMILY, which stays registered while the tests of the class run."""

    FAMILY: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.enterClassContext(registered(cls.FAMILY))

    def roundtrip(self, result: mf.Result[Any]) -> mf.Result[Any]:
        """The Result decoded from its JSON text. The text has format version 6, the decoded Result equals the
        original and encodes to the same text, and mf.verify accepts both."""
        text = mf.dumps(result)
        self.assertEqual(json.loads(text)["version"], 6)
        decoded = mf.loads(text)
        self.assertEqual(decoded, result)
        self.assertEqual(mf.dumps(decoded), text)
        for label, candidate in (("in memory", result), ("decoded", decoded)):
            report = mf.verify(candidate)
            self.assertTrue(report.verified, (label, statuses(report)))
        return decoded

    def answer(self, response: ToolResponse) -> mf.Result[Any]:
        """The Result that a successful response of an MCP tool carries."""
        self.assertFalse(response.is_error, response.payload)
        self.assertIsNone(response.payload["error"])
        return mf.from_data(response.payload["result"])

    def assertToolError(self, response: ToolResponse, code: str) -> dict[str, Any]:
        """The response of an MCP tool reports an error of that code; returns the error object (its stage, code,
        message and field)."""
        self.assertTrue(response.is_error, response.payload)
        error = response.payload["error"]
        self.assertEqual(error["code"], code, error)
        return error

    def assertInvalidInput(self, result: mf.Result[Any]) -> None:
        """The Result reports rejected input, never the producer's internal_error."""
        self.assertIs(result.execution_status, ExecutionStatus.INVALID_INPUT, result.error)
        self.assertEqual(result.error.code, "invalid_input")

    def assertResourceLimit(self, result: mf.Result[Any]) -> None:
        """The Result reports an exceeded computation budget."""
        self.assertIs(result.execution_status, ExecutionStatus.ERROR, result.error)
        self.assertEqual(result.error.code, "resource_limit_exceeded")

    def assertAccepts(self, operation: str, args: object, value: object, evidence: object = (), *,
                      limits: ComputationLimits | None = None) -> None:
        """The checker of FAMILY answers True to the claim."""
        self.assertIs(verdict(self.FAMILY, operation, args, value, evidence, limits=limits), True)

    def assertRejects(self, operation: str, args: object, value: object, evidence: object = (), *,
                      limits: ComputationLimits | None = None, may_exhaust: bool = False) -> None:
        """The checker of FAMILY answers False to the claim: it neither accepts it nor raises, which would leave
        the claim UNKNOWN. With may_exhaust, running out of budget counts as well, as for oversized evidence: the
        verifier then reports the resources, and never a verified claim."""
        try:
            answer = verdict(self.FAMILY, operation, args, value, evidence, limits=limits)
        except ResourceLimitError:
            if not may_exhaust:
                raise
            return
        self.assertIs(answer, False)
