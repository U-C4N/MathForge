"""Versioned, allowlisted JSON transport for the common mathematical model.

All mathematical integers use canonical decimal strings. The decoder executes
no source code and never imports a type named by the document.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextvars import ContextVar
from enum import Enum
import json
import math
import re
from dataclasses import fields as dataclass_fields
from typing import Any
from uuid import UUID

from . import contracts as c
from . import model as m
from .errors import MathForgeError, SerializationError
from .polynomial import Polynomial


FORMAT = "mathforge"
VERSION = 6
_decode_limits = ContextVar("mathforge_decode_limits", default=None)


def _resource_limit(message: str):
    raise SerializationError(message, code="resource_limit_exceeded")


def _limits(value):
    from .limits import DecodeLimits

    if value is None:
        return DecodeLimits()
    if type(value) is not DecodeLimits:
        raise SerializationError("decode_limits must be a DecodeLimits value")
    return value


def _v2_types():
    """Fixed imports, never names supplied by an input document."""
    from .algebraic import RealAlgebraicRoot, RootRecord
    from .intervals import Interval, IntervalSet
    from .matrices import AffineSolutionSet, LinearSystem, Matrix, RREFResult
    from .polynomial import SquareFreeDecomposition, SquareFreeFactor
    from .rational_function import RationalFunction

    return {
        "square_free_factor": SquareFreeFactor,
        "square_free_decomposition": SquareFreeDecomposition,
        "real_algebraic_root": RealAlgebraicRoot,
        "root_record": RootRecord,
        "rational_function": RationalFunction,
        "matrix": Matrix,
        "rref_result": RREFResult,
        "linear_system": LinearSystem,
        "affine_solution_set": AffineSolutionSet,
        "inequality": c.Inequality,
        "interval": Interval,
        "interval_set": IntervalSet,
        "operation_request": c.OperationRequest,
    }


_V2_TAGS = frozenset((
    "square_free_factor", "square_free_decomposition", "real_algebraic_root",
    "root_record", "rational_function", "matrix", "rref_result", "linear_system",
    "affine_solution_set", "inequality", "interval", "interval_set", "operation_request",
))


def _v3_types():
    """The v3 analysis records are a fixed, non-executable type allowlist."""
    from .analysis_types import (
        ExcludedPoint, FunctionAnalysis, LimitValue, LocalExtremum, MonotonicRegion,
    )

    return {
        "limit_value": LimitValue,
        "excluded_point": ExcludedPoint,
        "monotonic_region": MonotonicRegion,
        "local_extremum": LocalExtremum,
        "function_analysis": FunctionAnalysis,
    }


_V3_TAGS = frozenset((
    "limit_value", "excluded_point", "monotonic_region", "local_extremum", "function_analysis",
))
_V3_OPERATIONS = frozenset(("limit", "analyze_function"))


def _v4_types():
    from .polynomial_exact import PolynomialFactor, PolynomialFactorization
    from .exact_extensions import PartialFractionTerm, PartialFractionDecomposition
    from .analysis_types import ExtremumValue
    return {"polynomial_factor": PolynomialFactor,
            "polynomial_factorization": PolynomialFactorization,
            "partial_fraction_term": PartialFractionTerm,
            "partial_fraction_decomposition": PartialFractionDecomposition,
            "extremum_value": ExtremumValue}


_V4_TAGS = frozenset(("polynomial_factor", "polynomial_factorization", "partial_fraction_term",
                      "partial_fraction_decomposition", "extremum_value"))


_RECORD_PREFIXES = {"v5_": 5, "v6_": 6}


def _record_types(version):
    """The registry record classes of format v5 (the frozen v0.7 view) or v6, by tag."""
    if version == 5:
        from .dispatch_v5 import record_types
        return record_types()
    from . import registry
    return registry.record_types(6)


def _request_version(request):
    from .dispatch_v5 import operation_arguments, validate_arguments as validate_new
    if request.operation in operation_arguments():
        validate_new(request.operation, dict(request.arguments))
        return 5
    # Names this build does not register fall through: a later build's v6 request still decodes.
    from . import registry
    if request.operation in registry.operation_arguments(6):
        from .operations_v6 import validate_arguments as validate_v6
        validate_v6(request.operation, dict(request.arguments))
        return 6
    from .exact_dispatch import NEW_OPERATIONS, validate_arguments
    if request.operation in NEW_OPERATIONS:
        validate_arguments(request.operation, dict(request.arguments))
        return 4
    args = dict(request.arguments)
    if request.operation == "limit":
        return 3 if type(args.get("point")) in (m.Rational, str) else 4
    return 3 if request.operation == "analyze_function" else 2


def _validate_v3_request(request, *, version=3):
    """Validate the semantic input shape without executing an operation."""
    if request.operation not in _V3_OPERATIONS:
        return
    arguments = dict(request.arguments)
    expected = {"expression", "variable"}
    if request.operation == "limit":
        expected.update(("point", "side"))
    if set(arguments) != expected:
        raise SerializationError("Invalid v3 operation arguments")
    _typed(arguments["expression"], (m.Expression, Polynomial, _v2_types()["rational_function"]))
    _typed(arguments["variable"], m.Symbol)
    if request.operation == "limit":
        point = arguments["point"]
        if not (isinstance(point, m.Expression) if version >= 4 else type(point) is m.Rational) and not (type(point) is str and point in ("+inf", "-inf")):
            raise SerializationError("A limit point must be rational or signed infinity")
        if type(arguments["side"]) is not str or arguments["side"] not in ("left", "right", "both"):
            raise SerializationError("A limit side must be left, right or both")


def _boolean(value):
    if type(value) is not bool:
        raise SerializationError("A boolean is required")
    return value


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
    if type(value) is not str:
        raise SerializationError("An integer must be a canonical decimal string")
    limits = _decode_limits.get()
    if limits is not None and len(value) - value.startswith("-") > limits.max_integer_digits:
        _resource_limit("Integer exceeds the configured decimal digit limit")
    if not re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value):
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
        self.minimum_version = 1

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
            if value.rule.startswith("v6."):
                self.minimum_version = max(self.minimum_version, 6)
            if value.rule.startswith("v5."):
                self.minimum_version = max(self.minimum_version, 5)
            if value.rule.startswith("v4."):
                self.minimum_version = max(self.minimum_version, 4)
            return {"type": "step", "rule": value.rule,
                    "inputs": [self.encode(v) for v in value.inputs],
                    "outputs": [self.encode(v) for v in value.outputs],
                    "preconditions": [self.encode(v) for v in value.preconditions],
                    "relation": value.relation.value, "audit": [self.encode(v) for v in value.audit],
                    "explanation": value.explanation}
        if t is c.ErrorInfo:
            return {"type": "error_info", "code": value.code, "message": value.message}
        if t is c.Result:
            if type(value.expression) is Polynomial:
                self.minimum_version = max(self.minimum_version, 2)
            result = {"type": "result", "execution_status": value.execution_status.value,
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
            if value.request is not None:
                self.minimum_version = max(self.minimum_version, 2)
                result["request"] = self.encode(value.request)
            if value.deferred is not None:
                self.minimum_version = max(self.minimum_version, 5)
                result["deferred"] = self.encode(value.deferred)
            return result
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
        classes = _v2_types()
        tag = next((tag for tag, cls in classes.items() if t is cls), None)
        if tag is not None:
            self.minimum_version = max(self.minimum_version, 2)
            if tag == "operation_request":
                required_version = _request_version(value)
                _validate_v3_request(value, version=required_version)
                self.minimum_version = max(self.minimum_version, required_version)
            return self.v2(tag, value)
        classes = _v3_types()
        tag = next((tag for tag, cls in classes.items() if t is cls), None)
        if tag is not None:
            self.minimum_version = max(self.minimum_version, 3)
            if tag == "limit_value" and value.value is not None and type(value.value) is not m.Rational:
                self.minimum_version = max(self.minimum_version, 4)
            return self.v3(tag, value)
        classes = _v4_types()
        tag = next((tag for tag, cls in classes.items() if t is cls), None)
        if tag is not None:
            self.minimum_version = max(self.minimum_version, 4)
            return self.v4(tag, value)
        for prefix, version in _RECORD_PREFIXES.items():
            for tag, cls in _record_types(version).items():
                if t is cls:
                    self.minimum_version = max(self.minimum_version, version)
                    return {"type": prefix + tag, **{
                        field.name: self.encode(getattr(value, field.name))
                        for field in dataclass_fields(cls)}}
        raise SerializationError(f"Unsupported object type: {t.__name__}")

    def v4(self, tag, value):
        data = {"type": tag}
        if tag == "polynomial_factor":
            data.update(polynomial=self.encode(value.polynomial), multiplicity=_decimal(value.multiplicity))
        elif tag == "polynomial_factorization":
            data.update(coefficient=self.encode(value.coefficient), factors=[self.encode(v) for v in value.factors])
        elif tag == "partial_fraction_term":
            data.update(numerator=self.encode(value.numerator), factor=self.encode(value.factor), power=_decimal(value.power))
        elif tag == "partial_fraction_decomposition":
            data.update(polynomial=self.encode(value.polynomial), terms=[self.encode(v) for v in value.terms],
                        excluded=[self.encode(v) for v in value.excluded])
        elif tag == "extremum_value":
            data.update(point=self.encode(value.point), value=self.encode(value.value), kind=value.kind)
        return data

    def v3(self, tag, value):
        data = {"type": tag}
        if tag == "limit_value":
            data.update(kind=value.kind, value=self.optional(value.value))
        elif tag in ("excluded_point", "local_extremum"):
            data.update(point=self.encode(value.point), kind=value.kind)
        elif tag == "monotonic_region":
            data.update(interval=self.encode(value.interval), behavior=value.behavior)
        elif tag == "function_analysis":
            for name in ("function", "domain", "zero_set", "positive_set", "negative_set",
                         "derivative", "stationary_set"):
                data[name] = self.encode(getattr(value, name))
            for name in ("excluded_points", "monotonic_regions", "strict_extrema"):
                data[name] = [self.encode(item) for item in getattr(value, name)]
        return data

    def v2(self, tag, value):
        data = {"type": tag}
        if tag == "square_free_factor":
            data.update(polynomial=self.encode(value.polynomial), multiplicity=_decimal(value.multiplicity))
        elif tag == "square_free_decomposition":
            data.update(coefficient=self.encode(value.coefficient),
                        factors=[self.encode(v) for v in value.factors])
        elif tag == "real_algebraic_root":
            data.update(integer_coefficients=[_decimal(v) for v in value.integer_coefficients],
                        real_index=_decimal(value.real_index))
        elif tag == "root_record":
            data.update(root=self.encode(value.root), multiplicity=_decimal(value.multiplicity))
        elif tag == "rational_function":
            data.update(numerator=self.encode(value.numerator), denominator=self.encode(value.denominator),
                        excluded=[self.encode(v) for v in value.excluded])
        elif tag == "matrix":
            data.update(rows=[[self.encode(v) for v in row] for row in value.rows],
                        ncols=_decimal(value.ncols))
        elif tag == "rref_result":
            data.update(matrix=self.encode(value.matrix),
                        pivot_columns=[_decimal(v) for v in value.pivot_columns])
        elif tag == "linear_system":
            data.update(coefficients=self.encode(value.coefficients),
                        rhs=[self.encode(v) for v in value.rhs],
                        variables=[self.encode(v) for v in value.variables])
        elif tag == "affine_solution_set":
            data.update(variables=[self.encode(v) for v in value.variables],
                        particular=[self.encode(v) for v in value.particular],
                        basis=[[self.encode(v) for v in row] for row in value.basis])
        elif tag == "inequality":
            data.update(lhs=self.encode(value.lhs), relation=value.relation, rhs=self.encode(value.rhs))
        elif tag == "interval":
            data.update(lower=self.optional(value.lower), upper=self.optional(value.upper),
                        left_closed=value.left_closed, right_closed=value.right_closed)
        elif tag == "interval_set":
            data.update(intervals=[self.encode(v) for v in value.intervals])
        elif tag == "operation_request":
            data.update(operation=value.operation,
                        arguments=[{"name": name, "value": self.encode(v)} for name, v in value.arguments])
        return data

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
    def __init__(self, *, allow_workspace: bool = False, version: int = 1) -> None:
        self.symbols: dict[str, tuple[Any, m.Symbol]] = {}
        self.allow_workspace = allow_workspace
        self.version = version

    def decode(self, data: Any) -> Any:
        from .workspace import FrozenOptions, ObjectRef, OperationRecord, Workspace

        if type(data) is not dict or type(data.get("type")) is not str:
            raise SerializationError("A tagged mathematical object is required")
        tag = data["type"]
        if tag[:3] in _RECORD_PREFIXES:
            return self.record(_RECORD_PREFIXES[tag[:3]], tag[3:], data)
        if tag in _V4_TAGS:
            if self.version < 4:
                raise SerializationError("This mathematical type requires format v4")
            return self.v4(tag, data)
        if tag in _V3_TAGS:
            if self.version < 3:
                raise SerializationError("This mathematical type requires format v3")
            return self.v3(tag, data)
        if tag in _V2_TAGS:
            if self.version < 2:
                raise SerializationError("This mathematical type requires format v2")
            return self.v2(tag, data)
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
            scalar = (m.Expression, _v2_types()["rational_function"]) if self.version >= 2 else m.Expression
            return c.Eq(self.required(data["lhs"], scalar), self.required(data["rhs"], scalar))
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
            if _string(data["rule"]).startswith("v4.") and self.version < 4:
                raise SerializationError("This evidence rule requires format v4")
            if _string(data["rule"]).startswith("v5.") and self.version < 5:
                raise SerializationError("This evidence rule requires format v5")
            if _string(data["rule"]).startswith("v6.") and self.version < 6:
                raise SerializationError("This evidence rule requires format v6")
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
            fields = "type execution_status outcome value exactness completeness verification method domain assumptions steps problem expression for_ precision tolerance error_bound error"
            has_deferred = "deferred" in data
            if has_deferred and self.version < 5:
                raise SerializationError("Unevaluated results require format v5")
            _fields(data, fields + (" request" if self.version >= 2 else "")
                    + (" deferred" if has_deferred else ""))
            classes = _v2_types() if self.version >= 2 else {}
            problem_types = (c.Eq, c.Inequality, classes["linear_system"]) if self.version >= 2 else c.Eq
            expression_types = (m.Expression, Polynomial, classes["rational_function"], classes["matrix"]) if self.version >= 2 else m.Expression
            if self.version >= 5:
                from .functions import RealExpr, RealFunction
                from .rational_integration import RationalPrimitive
                expression_types += (RealExpr, RealFunction, RationalPrimitive)
            return c.Result(
                execution_status=_enum(c.ExecutionStatus, data["execution_status"]),
                outcome=_enum(c.Outcome, data["outcome"]), value=self.optional(data["value"]),
                exactness=_enum(c.Exactness, data["exactness"]),
                completeness=_enum(c.Completeness, data["completeness"]),
                verification=self.sequence(data["verification"], c.Check), method=_string(data["method"]),
                domain=None if data["domain"] is None else _enum(m.Domain, data["domain"]),
                assumptions=self.sequence(data["assumptions"], m.Assumption),
                steps=self.sequence(data["steps"], c.Step), problem=self.optional(data["problem"], problem_types),
                expression=self.optional(data["expression"], expression_types), for_=self.optional(data["for_"], m.Symbol),
                precision=None if data["precision"] is None else _integer(data["precision"]),
                tolerance=self.optional(data["tolerance"], m.Rational),
                error_bound=self.optional(data["error_bound"], m.Rational),
                error=self.optional(data["error"], c.ErrorInfo),
                request=self.optional(data["request"], c.OperationRequest) if self.version >= 2 else None,
                deferred=self.decode(data["deferred"]) if has_deferred else None,
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

    def record(self, version, tag, data):
        """A registry record of format v5 or v6: its dataclass fields, which must re-encode canonically."""
        if self.version < version:
            raise SerializationError(f"This mathematical type requires format v{version}")
        cls = _record_types(version).get(tag)
        if cls is None:
            raise SerializationError(f"Unknown v{version} mathematical type")
        names = tuple(field.name for field in dataclass_fields(cls))
        _fields(data, "type " + " ".join(names))
        result = cls(**{name: self.decode(data[name]) for name in names})
        if _Encoder().encode(result) != data:
            raise SerializationError(f"A v{version} record must have canonical fields")
        return result

    def v2(self, tag, data):
        types = _v2_types()
        cls = types[tag]
        scalar = (m.Expression, types["rational_function"])
        if tag == "square_free_factor":
            _fields(data, "type polynomial multiplicity")
            result = cls(self.required(data["polynomial"], Polynomial), _integer(data["multiplicity"]))
        elif tag == "square_free_decomposition":
            _fields(data, "type coefficient factors")
            result = cls(self.required(data["coefficient"], m.Rational),
                         self.sequence(data["factors"], types["square_free_factor"]))
        elif tag == "real_algebraic_root":
            _fields(data, "type integer_coefficients real_index")
            result = cls(tuple(_integer(v) for v in _list(data["integer_coefficients"])),
                         _integer(data["real_index"]))
        elif tag == "root_record":
            _fields(data, "type root multiplicity")
            result = cls(self.required(data["root"], m.Expression), _integer(data["multiplicity"]))
        elif tag == "rational_function":
            _fields(data, "type numerator denominator excluded")
            result = cls._from_normal_form(self.required(data["numerator"], Polynomial),
                                           self.required(data["denominator"], Polynomial),
                                           self.sequence(data["excluded"], Polynomial))
        elif tag == "matrix":
            _fields(data, "type rows ncols")
            result = cls(tuple(self.sequence(row, m.Rational) for row in _list(data["rows"])),
                         ncols=_integer(data["ncols"]))
        elif tag == "rref_result":
            _fields(data, "type matrix pivot_columns")
            result = cls(self.required(data["matrix"], types["matrix"]),
                         tuple(_integer(v) for v in _list(data["pivot_columns"])))
        elif tag == "linear_system":
            _fields(data, "type coefficients rhs variables")
            result = cls(self.required(data["coefficients"], types["matrix"]),
                         self.sequence(data["rhs"], m.Rational), self.sequence(data["variables"], m.Symbol))
        elif tag == "affine_solution_set":
            _fields(data, "type variables particular basis")
            result = cls(self.sequence(data["variables"], m.Symbol),
                         self.sequence(data["particular"], m.Rational),
                         tuple(self.sequence(row, m.Rational) for row in _list(data["basis"])))
        elif tag == "inequality":
            _fields(data, "type lhs relation rhs")
            result = cls(self.required(data["lhs"], scalar), _string(data["relation"]),
                         self.required(data["rhs"], scalar))
        elif tag == "interval":
            _fields(data, "type lower upper left_closed right_closed")
            endpoint = (m.Rational, types["real_algebraic_root"])
            result = cls(self.optional(data["lower"], endpoint), self.optional(data["upper"], endpoint),
                         _boolean(data["left_closed"]), _boolean(data["right_closed"]))
        elif tag == "interval_set":
            _fields(data, "type intervals")
            result = cls(self.sequence(data["intervals"], types["interval"]))
        elif tag == "operation_request":
            _fields(data, "type operation arguments")
            if data["operation"] in _V3_OPERATIONS and self.version < 3:
                raise SerializationError("This operation request requires format v3")
            arguments = []
            for item in _list(data["arguments"]):
                _fields(item, "name value")
                arguments.append((_string(item["name"]), self.decode(item["value"])))
            result = cls(_string(data["operation"]), tuple(arguments))
            required_version = _request_version(result)
            if self.version < required_version:
                raise SerializationError(f"This operation request requires format v{required_version}")
            _validate_v3_request(result, version=self.version)
        encoded = _Encoder().encode(result)
        _upgrade_v2_results(encoded)
        if encoded != data:
            raise SerializationError(f"Noncanonical {tag} encoding")
        return result

    def v3(self, tag, data):
        types = _v3_types()
        older = _v2_types()
        cls = types[tag]
        endpoint = (m.Rational, older["real_algebraic_root"])
        if tag == "limit_value":
            _fields(data, "type kind value")
            result = cls(_string(data["kind"]), self.optional(data["value"], endpoint if self.version >= 4 else m.Rational))
        elif tag in ("excluded_point", "local_extremum"):
            _fields(data, "type point kind")
            result = cls(self.required(data["point"], endpoint), _string(data["kind"]))
        elif tag == "monotonic_region":
            _fields(data, "type interval behavior")
            result = cls(self.required(data["interval"], older["interval"]), _string(data["behavior"]))
        elif tag == "function_analysis":
            _fields(data, "type function domain excluded_points zero_set positive_set negative_set derivative stationary_set monotonic_regions strict_extrema")
            result = cls(
                function=self.required(data["function"], older["rational_function"]),
                domain=self.required(data["domain"], older["interval_set"]),
                excluded_points=self.sequence(data["excluded_points"], types["excluded_point"]),
                zero_set=self.required(data["zero_set"], older["interval_set"]),
                positive_set=self.required(data["positive_set"], older["interval_set"]),
                negative_set=self.required(data["negative_set"], older["interval_set"]),
                derivative=self.required(data["derivative"], older["rational_function"]),
                stationary_set=self.required(data["stationary_set"], older["interval_set"]),
                monotonic_regions=self.sequence(data["monotonic_regions"], types["monotonic_region"]),
                strict_extrema=self.sequence(data["strict_extrema"], types["local_extremum"]),
            )
        encoded = _Encoder().encode(result)
        _upgrade_v2_results(encoded)
        if encoded != data:
            raise SerializationError(f"Noncanonical {tag} encoding")
        return result

    def v4(self, tag, data):
        types = _v4_types()
        cls = types[tag]
        if tag == "polynomial_factor":
            _fields(data, "type polynomial multiplicity")
            result = cls(self.required(data["polynomial"], Polynomial), _integer(data["multiplicity"]))
        elif tag == "polynomial_factorization":
            _fields(data, "type coefficient factors")
            result = cls(self.required(data["coefficient"], m.Rational),
                         self.sequence(data["factors"], types["polynomial_factor"]))
        elif tag == "partial_fraction_term":
            _fields(data, "type numerator factor power")
            result = cls(self.required(data["numerator"], Polynomial),
                         self.required(data["factor"], Polynomial), _integer(data["power"]))
        elif tag == "partial_fraction_decomposition":
            _fields(data, "type polynomial terms excluded")
            result = cls(self.required(data["polynomial"], Polynomial),
                         self.sequence(data["terms"], types["partial_fraction_term"]),
                         self.sequence(data["excluded"], Polynomial))
        else:
            _fields(data, "type point value kind")
            endpoint = (m.Rational, _v2_types()["real_algebraic_root"])
            result = cls(self.required(data["point"], endpoint), self.required(data["value"], endpoint),
                         _string(data["kind"]))
        encoded = _Encoder().encode(result)
        _upgrade_v2_results(encoded)
        if encoded != data:
            raise SerializationError(f"Noncanonical {tag} encoding")
        return result

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


def _upgrade_v2_results(data):
    """A v2+ envelope uses the extended Result shape, including legacy children."""
    pending = [data]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            if node.get("type") == "result":
                node.setdefault("request", None)
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)


def _guard_data(document, limits):
    """Bound traversal before constructors, including cyclic from_data inputs."""
    pending = [(document, 0, False)]
    active = set()
    nodes = visits = byte_estimate = 0
    while pending:
        node, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(node))
            continue
        if depth > limits.max_depth:
            _resource_limit("JSON nesting exceeds the configured depth limit")
        visits += 1
        if visits > limits.max_nodes * 32:
            _resource_limit("JSON structure exceeds the configured traversal limit")
        if type(node) in (dict, list):
            if len(node) > limits.max_nodes * 32:
                _resource_limit("JSON container exceeds the configured traversal limit")
            if id(node) in active:
                raise SerializationError("Cyclic JSON container")
            active.add(id(node))
            pending.append((node, depth, True))
            if type(node) is dict:
                if any(type(key) is not str for key in node):
                    raise SerializationError("JSON object keys must be strings")
                if "type" in node:
                    nodes += 1
                    if nodes > limits.max_nodes:
                        _resource_limit("Document exceeds the configured tagged-node limit")
                if node.get("type") == "workspace" and type(node.get("objects")) is list:
                    if len(node["objects"]) > limits.max_workspace_objects:
                        _resource_limit("Workspace exceeds the configured object limit")
                byte_estimate += sum(len(key.encode("utf-8", errors="surrogatepass")) + 3 for key in node) + 2
                pending.extend((value, depth + 1, False) for value in node.values())
            else:
                byte_estimate += len(node) + 2
                pending.extend((value, depth + 1, False) for value in node)
        elif type(node) is str:
            # Historical ensure_ascii JSON can safely preserve lone escaped
            # surrogates in Python strings; budget inspection must not reject
            # a legacy string that the transport itself still roundtrips.
            byte_estimate += len(node.encode("utf-8", errors="surrogatepass")) + 2
        elif node is None or type(node) in (bool, int, float):
            byte_estimate += 4
        else:
            raise SerializationError("from_data requires ordinary JSON-compatible values")
        if byte_estimate > limits.max_bytes:
            _resource_limit("Document exceeds the configured byte limit")


def _guard_text(text, limits):
    if len(text.encode("utf-8")) > limits.max_bytes:
        _resource_limit("JSON text exceeds the configured byte limit")
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > limits.max_depth:
                _resource_limit("JSON text exceeds the configured depth limit")
        elif char in "]}":
            depth -= 1


def to_data(value: Any) -> dict[str, Any]:
    """Return a versioned JSON-compatible document for a supported object."""
    from .workspace import Workspace

    try:
        encoder = _Encoder(allow_workspace=type(value) is Workspace)
        data = encoder.encode(value)
        if encoder.minimum_version >= 2:
            _upgrade_v2_results(data)
        return {"format": FORMAT, "version": encoder.minimum_version,
                "kind": "workspace" if type(value) is Workspace else "object",
                "data": data}
    except SerializationError:
        raise
    except (MathForgeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise SerializationError("Unable to serialize the mathematical object") from exc


def from_data(document: Any, *, decode_limits=None, computation_limits=None) -> Any:
    """Validate and reconstruct a document using a fixed type allowlist."""
    from .workspace import Workspace
    from .limits import computation

    limits = _limits(decode_limits)
    token = _decode_limits.set(limits)
    try:
        _guard_data(document, limits)
        _fields(document, "format version kind data")
        if document["format"] != FORMAT or type(document["format"]) is not str:
            raise SerializationError("Unknown document format")
        if type(document["version"]) is not int or document["version"] not in (1, 2, 3, 4, 5, 6):
            raise SerializationError("Unsupported MathForge format version")
        if document["kind"] not in ("object", "workspace"):
            raise SerializationError("Unknown document kind")
        with computation(computation_limits, fresh=True):
            value = _Decoder(allow_workspace=document["kind"] == "workspace",
                             version=document["version"]).decode(document["data"])
        if (type(value) is Workspace) != (document["kind"] == "workspace"):
            raise SerializationError("Document kind does not match its content")
        return value
    except SerializationError:
        raise
    except (MathForgeError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        if getattr(exc, "code", None) == "resource_limit_exceeded":
            raise SerializationError(str(exc), code="resource_limit_exceeded") from exc
        raise SerializationError("Invalid mathematical document") from exc
    finally:
        _decode_limits.reset(token)


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


def loads(text: str, *, decode_limits=None, computation_limits=None) -> Any:
    """Load JSON without duplicate keys, non-finite numbers or executable data."""
    if type(text) is not str:
        raise SerializationError("loads() requires JSON text")
    limits = _limits(decode_limits)
    token = _decode_limits.set(limits)
    try:
        _guard_text(text, limits)
        document = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant,
                              parse_int=_integer)
    except SerializationError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise SerializationError("Invalid JSON document") from exc
    finally:
        _decode_limits.reset(token)
    return from_data(document, decode_limits=limits, computation_limits=computation_limits)
