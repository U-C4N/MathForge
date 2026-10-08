"""Versioned, allowlisted JSON transport for the common mathematical model.

All mathematical integers use canonical decimal strings. The decoder executes
no source code and never imports a type named by the document.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
import json
import math
import re
from typing import Any
from uuid import UUID

from . import contracts as c
from . import model as m
from .errors import MathForgeError, SerializationError
from .polynomial import Polynomial


FORMAT = "mathforge"
VERSION = 1


def _decimal(number: int) -> str:
    """Convert arbitrarily large integers without changing Python's global limit."""
    if type(number) is not int:
        raise SerializationError("An exact integer is required")
    sign = "-" if number < 0 else ""
    number = abs(number)
    chunks = []
    while number >= 1_000_000_000:
        number, chunk = divmod(number, 1_000_000_000)
        chunks.append(f"{chunk:09d}")
    return sign + str(number) + "".join(reversed(chunks))


def _integer(value: Any) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value):
        raise SerializationError("An integer must be a canonical decimal string")
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    total = 0
    for index in range(0, len(digits), 9):
        chunk = digits[index:index + 9]
        total = total * 10 ** len(chunk) + int(chunk)
    return -total if negative else total


def _fields(value: Any, fields: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields.split()):
        raise SerializationError(f"Expected exactly these fields: {fields}")
    return value


def _string(value: Any) -> str:
    if type(value) is not str:
        raise SerializationError("A string is required")
    return value


def _list(value: Any) -> list[Any]:
    if type(value) is not list:
        raise SerializationError("A JSON array is required")
    return value


def _enum(cls: type[Enum], value: Any) -> Enum:
    try:
        return cls(_string(value))
    except ValueError as exc:
        raise SerializationError(f"Unknown {cls.__name__} value") from exc


def _typed(value: Any, cls: type | tuple[type, ...]) -> Any:
    if not isinstance(value, cls):
        raise SerializationError("Unexpected mathematical object type")
    return value


_ENUMS = {
    "domain": m.Domain,
    "truth": m.Truth,
    "execution_status": c.ExecutionStatus,
    "outcome": c.Outcome,
    "exactness": c.Exactness,
    "completeness": c.Completeness,
    "check_status": c.CheckStatus,
    "verification_level": c.VerificationLevel,
    "step_relation": c.StepRelation,
}


class _Encoder:
    def __init__(self, *, allow_workspace: bool = False) -> None:
        self.symbols: dict[str, Any] = {}
        self.allow_workspace = allow_workspace

    def encode(self, value: Any) -> dict[str, Any]:
        from .workspace import FrozenOptions, ObjectRef, OperationRecord, Workspace

        t = type(value)
        if value is None:
            return {"type": "none"}
        if t in (bool, str):
            return {"type": "boolean" if t is bool else "text", "value": value}
        if t is tuple:
            return {"type": "tuple", "items": [self.encode(item) for item in value]}
        if t is int:
            return {"type": "integer", "value": _decimal(value)}
        for tag, cls in _ENUMS.items():
            if t is cls:
                return {"type": tag, "value": value.value}
        if t is m.Rational:
            return {"type": "rational", "numerator": _decimal(value.numerator),
                    "denominator": _decimal(value.denominator)}
        if t is m.Assumption:
            return {"type": "assumption", "name": value.name, "truth": value.truth.value}
        if t is m.Symbol:
            encoded = {"type": "symbol", "name": value.name, "uid": str(value.uid),
                       "domain": value.domain.value,
                       "assumptions": [self.encode(a) for a in value.assumptions]}
            self._symbol(str(value.uid), encoded)
            return encoded
        if t is m.Add:
            return {"type": "add", "terms": [self.encode(v) for v in value.terms]}
        if t is m.Mul:
            return {"type": "mul", "factors": [self.encode(v) for v in value.factors]}
        if t is m.Pow:
            return {"type": "pow", "base": self.encode(value.base),
                    "exponent": _decimal(value.exponent)}
        if t is m.Sqrt:
            return {"type": "sqrt", "radicand": self.encode(value.radicand)}
        if t is Polynomial:
            return {"type": "polynomial", "coefficients": [self.encode(v) for v in value.coefficients],
                    "variable": self.encode(value.variable)}
        if t is c.Eq:
            return {"type": "equation", "lhs": self.encode(value.lhs), "rhs": self.encode(value.rhs)}
        if t is c.FiniteSet:
            return {"type": "finite_set", "values": [self.encode(v) for v in value.values],
                    "domain": value.domain.value}
        if t in (c.EmptySet, c.UniversalSet):
            return {"type": "empty_set" if t is c.EmptySet else "universal_set", "domain": value.domain.value}
        if t is c.Condition:
            return {"type": "condition", "lhs": self.encode(value.lhs), "relation": value.relation,
                    "rhs": self.encode(value.rhs), "truth": value.truth.value}
        if t is c.Check:
            return {"type": "check", "claim": value.claim, "status": value.status.value,
                    "level": value.level.value, "method": value.method,
                    "subject": self.optional(value.subject), "details": value.details}
        if t is c.VerificationReport:
            return {"type": "verification_report", "checks": [self.encode(v) for v in value.checks]}
        if t is c.Step:
            return {"type": "step", "rule": value.rule,
                    "inputs": [self.encode(v) for v in value.inputs],
                    "outputs": [self.encode(v) for v in value.outputs],
                    "preconditions": [self.encode(v) for v in value.preconditions],
                    "relation": value.relation.value, "audit": [self.encode(v) for v in value.audit],
                    "explanation": value.explanation}
        if t is c.ErrorInfo:
            return {"type": "error_info", "code": value.code, "message": value.message}
        if t is c.Result:
            return {"type": "result", "execution_status": value.execution_status.value,
                    "outcome": value.outcome.value, "value": self.optional(value.value),
                    "exactness": value.exactness.value, "completeness": value.completeness.value,
                    "verification": [self.encode(v) for v in value.verification],
                    "method": value.method, "domain": None if value.domain is None else value.domain.value,
                    "assumptions": [self.encode(v) for v in value.assumptions],
                    "steps": [self.encode(v) for v in value.steps], "problem": self.optional(value.problem),
                    "expression": self.optional(value.expression), "for_": self.optional(value.for_),
                    "precision": None if value.precision is None else _decimal(value.precision),
                    "tolerance": self.optional(value.tolerance), "error_bound": self.optional(value.error_bound),
                    "error": self.optional(value.error)}
        if t is ObjectRef:
            return {"type": "object_ref", "identifier": value.identifier}
        if t is OperationRecord:
            return {"type": "operation_record", "operation": value.operation,
                    "inputs": [{"name": name, "ref": self.encode(ref)} for name, ref in value.inputs],
                    "options": self.option(value.options), "result_ref": self.encode(value.result_ref)}
        if t is FrozenOptions:
            return self.option(value)
        if t is Workspace:
            if not self.allow_workspace:
                raise SerializationError("A Workspace must be the top-level document")
            self.allow_workspace = False
            return {"type": "workspace", "objects": [
                {"ref": key, "object": self.encode(obj)} for key, obj in sorted(value._objects.items())
            ], "history": [self.encode(record) for record in value.history]}
        raise SerializationError(f"Unsupported object type: {t.__name__}")

    def optional(self, value: Any) -> Any:
        return None if value is None else self.encode(value)

    def _symbol(self, identifier: str, data: Any) -> None:
        previous = self.symbols.setdefault(identifier, data)
        if previous != data:
            raise SerializationError("Conflicting metadata for the same symbol identity")

    def option(self, value: Any) -> Any:
        from .workspace import FrozenOptions

        if value is None or type(value) in (str, bool):
            return value
        if type(value) is int:
            return {"type": "option_integer", "value": _decimal(value)}
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) is FrozenOptions:
            return {"type": "options", "entries": [
                {"name": key, "value": self.option(item)} for key, item in value.items()
            ]}
        if type(value) is tuple:
            return {"type": "option_array", "items": [self.option(item) for item in value]}
        raise SerializationError("Options must be deeply immutable JSON data")


class _Decoder:
    def __init__(self, *, allow_workspace: bool = False) -> None:
        self.symbols: dict[str, tuple[Any, m.Symbol]] = {}
        self.allow_workspace = allow_workspace

    def decode(self, data: Any) -> Any:
        from .workspace import FrozenOptions, ObjectRef, OperationRecord, Workspace

        if type(data) is not dict or type(data.get("type")) is not str:
            raise SerializationError("A tagged mathematical object is required")
        tag = data["type"]
        if tag == "none":
            _fields(data, "type")
            return None
        if tag in ("boolean", "text"):
            _fields(data, "type value")
            expected = bool if tag == "boolean" else str
            if type(data["value"]) is not expected:
                raise SerializationError("Invalid primitive value")
            return data["value"]
        if tag == "tuple":
            _fields(data, "type items")
            return self.sequence(data["items"])
        if tag in _ENUMS:
            _fields(data, "type value")
            return _enum(_ENUMS[tag], data["value"])
        if tag == "integer":
            _fields(data, "type value")
            return _integer(data["value"])
        if tag == "rational":
            _fields(data, "type numerator denominator")
            n, d = _integer(data["numerator"]), _integer(data["denominator"])
            result = m.Rational(n, d)
            if result.numerator != n or result.denominator != d:
                raise SerializationError("A rational must be normalized with a positive denominator")
            return result
        if tag == "assumption":
            _fields(data, "type name truth")
            return m.Assumption(_string(data["name"]), _enum(m.Truth, data["truth"]))
        if tag == "symbol":
            _fields(data, "type name uid domain assumptions")
            identifier = _string(data["uid"])
            uid = UUID(identifier)
            if str(uid) != identifier:
                raise SerializationError("Symbol identities must be canonical UUID strings")
            name = _string(data["name"])
            domain = _enum(m.Domain, data["domain"])
            assumptions = self.sequence(data["assumptions"], m.Assumption)
            metadata = (name, domain, assumptions)
            if identifier in self.symbols:
                previous, symbol = self.symbols[identifier]
                if previous != metadata:
                    raise SerializationError("Conflicting metadata for the same symbol identity")
                return symbol
            symbol = m.Symbol(name=name, domain=domain, assumptions=assumptions, uid=identifier)
            if symbol.assumptions != assumptions:
                raise SerializationError("Symbol assumptions must use canonical ordering")
            self.symbols[identifier] = (metadata, symbol)
            return symbol
        if tag in ("add", "mul"):
            field = "terms" if tag == "add" else "factors"
            _fields(data, f"type {field}")
            return (m.Add if tag == "add" else m.Mul)(self.sequence(data[field], m.Expression))
        if tag == "pow":
            _fields(data, "type base exponent")
            return m.Pow(self.required(data["base"], m.Expression), _integer(data["exponent"]))
        if tag == "sqrt":
            _fields(data, "type radicand")
            return m.Sqrt(self.required(data["radicand"], m.Rational))
        if tag == "polynomial":
            _fields(data, "type coefficients variable")
            coeffs = self.sequence(data["coefficients"], m.Rational)
            result = Polynomial(coeffs, self.required(data["variable"], m.Symbol))
            if result.coefficients != coeffs:
                raise SerializationError("Polynomial coefficients must be normalized")
            return result
        if tag == "equation":
            _fields(data, "type lhs rhs")
            return c.Eq(self.required(data["lhs"], m.Expression), self.required(data["rhs"], m.Expression))
        if tag == "finite_set":
            _fields(data, "type values domain")
            values = self.sequence(data["values"], m.Expression)
            result = c.FiniteSet(values, _enum(m.Domain, data["domain"]))
            if result.values != values:
                raise SerializationError("Finite sets cannot contain structural duplicates")
            return result
        if tag in ("empty_set", "universal_set"):
            _fields(data, "type domain")
            return (c.EmptySet if tag == "empty_set" else c.UniversalSet)(_enum(m.Domain, data["domain"]))
        if tag == "condition":
            _fields(data, "type lhs relation rhs truth")
            return c.Condition(self.required(data["lhs"], m.Expression), _string(data["relation"]),
                               self.required(data["rhs"], m.Expression), _enum(m.Truth, data["truth"]))
        if tag == "check":
            _fields(data, "type claim status level method subject details")
            return c.Check(_string(data["claim"]), _enum(c.CheckStatus, data["status"]),
                           _enum(c.VerificationLevel, data["level"]), _string(data["method"]),
                           self.optional(data["subject"], m.Expression), _string(data["details"]))
        if tag == "verification_report":
            _fields(data, "type checks")
            return c.VerificationReport(self.sequence(data["checks"], c.Check))
        if tag == "step":
            _fields(data, "type rule inputs outputs preconditions relation audit explanation")
            explanation = None if data["explanation"] is None else _string(data["explanation"])
            return c.Step(rule=_string(data["rule"]), inputs=self.sequence(data["inputs"]),
                          outputs=self.sequence(data["outputs"]),
                          preconditions=self.sequence(data["preconditions"], c.Condition),
                          relation=_enum(c.StepRelation, data["relation"]),
                          audit=self.sequence(data["audit"], c.Check), explanation=explanation)
        if tag == "error_info":
            _fields(data, "type code message")
            return c.ErrorInfo(_string(data["code"]), _string(data["message"]))
        if tag == "result":
            _fields(data, "type execution_status outcome value exactness completeness verification method domain assumptions steps problem expression for_ precision tolerance error_bound error")
            return c.Result(
                execution_status=_enum(c.ExecutionStatus, data["execution_status"]),
                outcome=_enum(c.Outcome, data["outcome"]), value=self.optional(data["value"]),
                exactness=_enum(c.Exactness, data["exactness"]),
                completeness=_enum(c.Completeness, data["completeness"]),
                verification=self.sequence(data["verification"], c.Check), method=_string(data["method"]),
                domain=None if data["domain"] is None else _enum(m.Domain, data["domain"]),
                assumptions=self.sequence(data["assumptions"], m.Assumption),
                steps=self.sequence(data["steps"], c.Step), problem=self.optional(data["problem"], c.Eq),
                expression=self.optional(data["expression"], m.Expression), for_=self.optional(data["for_"], m.Symbol),
                precision=None if data["precision"] is None else _integer(data["precision"]),
                tolerance=self.optional(data["tolerance"], m.Rational),
                error_bound=self.optional(data["error_bound"], m.Rational),
                error=self.optional(data["error"], c.ErrorInfo),
            )
        if tag == "object_ref":
            _fields(data, "type identifier")
            return ObjectRef(_string(data["identifier"]))
        if tag == "operation_record":
            _fields(data, "type operation inputs options result_ref")
            pairs = []
            for entry in _list(data["inputs"]):
                _fields(entry, "name ref")
                pairs.append((_string(entry["name"]), self.required(entry["ref"], ObjectRef)))
            options = _typed(self.option(data["options"]), FrozenOptions)
            return OperationRecord(_string(data["operation"]), tuple(pairs), options,
                                   self.required(data["result_ref"], ObjectRef))
        if tag == "options":
            return self.option(data)
        if tag == "workspace":
            if not self.allow_workspace:
                raise SerializationError("A Workspace must be the top-level document")
            self.allow_workspace = False
            _fields(data, "type objects history")
            objects = {}
            for entry in _list(data["objects"]):
                _fields(entry, "ref object")
                identifier = ObjectRef(_string(entry["ref"])).identifier
                if identifier in objects:
                    raise SerializationError("Duplicate stored object reference")
                objects[identifier] = self.decode(entry["object"])
            history = self.sequence(data["history"], OperationRecord)
            return Workspace._restore(objects, history)
        raise SerializationError(f"Unknown mathematical object tag: {tag}")

    def required(self, data: Any, cls: type | tuple[type, ...]) -> Any:
        return _typed(self.decode(data), cls)

    def optional(self, data: Any, cls: type | tuple[type, ...] | None = None) -> Any:
        if data is None:
            return None
        result = self.decode(data)
        return result if cls is None else _typed(result, cls)

    def sequence(self, data: Any, cls: type | tuple[type, ...] | None = None) -> tuple[Any, ...]:
        values = tuple(self.decode(item) for item in _list(data))
        if cls is not None:
            for value in values:
                _typed(value, cls)
        return values

    def option(self, data: Any) -> Any:
        from .workspace import FrozenOptions

        if data is None or type(data) in (str, bool):
            return data
        if type(data) is float and math.isfinite(data):
            return data
        if type(data) is not dict:
            raise SerializationError("Invalid operation option")
        if data.get("type") == "option_integer":
            _fields(data, "type value")
            return _integer(data["value"])
        if data.get("type") == "option_array":
            _fields(data, "type items")
            return tuple(self.option(value) for value in _list(data["items"]))
        if data.get("type") == "options":
            _fields(data, "type entries")
            pairs = []
            for entry in _list(data["entries"]):
                _fields(entry, "name value")
                pairs.append((_string(entry["name"]), self.option(entry["value"])))
            return FrozenOptions(tuple(pairs))
        raise SerializationError("Unknown operation option tag")


def _check_symbol_consistency(objects: Iterable[Any]) -> None:
    encoder = _Encoder()
    for obj in objects:
        encoder.encode(obj)


def to_data(value: Any) -> dict[str, Any]:
    """Return a versioned JSON-compatible document for a supported object."""
    from .workspace import Workspace

    try:
        return {"format": FORMAT, "version": VERSION,
                "kind": "workspace" if type(value) is Workspace else "object",
                "data": _Encoder(allow_workspace=type(value) is Workspace).encode(value)}
    except SerializationError:
        raise
    except (MathForgeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise SerializationError("Unable to serialize the mathematical object") from exc


def from_data(document: Any) -> Any:
    """Validate and reconstruct a document using a fixed type allowlist."""
    from .workspace import Workspace

    try:
        _fields(document, "format version kind data")
        if document["format"] != FORMAT or type(document["format"]) is not str:
            raise SerializationError("Unknown document format")
        if type(document["version"]) is not int or document["version"] != VERSION:
            raise SerializationError("Unsupported MathForge format version")
        if document["kind"] not in ("object", "workspace"):
            raise SerializationError("Unknown document kind")
        value = _Decoder(allow_workspace=document["kind"] == "workspace").decode(document["data"])
        if (type(value) is Workspace) != (document["kind"] == "workspace"):
            raise SerializationError("Document kind does not match its content")
        return value
    except SerializationError:
        raise
    except (MathForgeError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise SerializationError("Invalid mathematical document") from exc


def dumps(value: Any, *, indent: int | None = None) -> str:
    """Serialize deterministically; compact output is the content-hash input."""
    return json.dumps(to_data(value), sort_keys=True, ensure_ascii=True, allow_nan=False,
                      indent=indent, separators=(",", ":") if indent is None else None)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise SerializationError(f"Non-finite JSON value: {value}")


def loads(text: str) -> Any:
    """Load JSON without duplicate keys, non-finite numbers or executable data."""
    if type(text) is not str:
        raise SerializationError("loads() requires JSON text")
    try:
        document = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except SerializationError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise SerializationError("Invalid JSON document") from exc
    return from_data(document)
