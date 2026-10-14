"""Small explicit JSON schemas for the curated tools (no optional dependency)."""

from dataclasses import asdict
from ..limits import ComputationLimits
from .boundary import MAX_EQUATIONS, MAX_SCALAR_CHARACTERS, MAX_VARIABLES, operand_tags


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def input_schemas() -> dict[str, dict]:
    from ..dispatch_v5 import operation_arguments
    text = {"type": "string", "maxLength": 16384}
    name = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]*$", "maxLength": 128, "not": {"const": "sqrt"}}
    rational = {"type": "string", "pattern": "^[+-]?[0-9]+(?:/[+-]?[0-9]+)?$",
                "maxLength": MAX_SCALAR_CHARACTERS}
    scalar_text = {"type": "string", "maxLength": MAX_SCALAR_CHARACTERS,
                   "description": "Closed exact mathematical text; use scalar object envelopes for root reuse."}
    scalar_envelope = _object({"format": {"const": "mathforge"}, "version": {"type": "integer", "enum": [1, 2, 3, 4]},
        "kind": {"const": "object"}, "data": {"type": "object", "required": ["type"],
        "properties": {"type": {"enum": ["rational", "sqrt", "real_algebraic_root", "add", "mul", "pow"]}}}},
        ["format", "version", "kind", "data"])
    scalar = {"anyOf": [scalar_text, scalar_envelope]}
    limits = _object({key: {"type": "integer", "minimum": 1, "maximum": limit}
                      for key, limit in asdict(ComputationLimits()).items()})
    optional_limits = {"anyOf": [limits, {"type": "null"}], "default": None}
    ids = {"anyOf": [{"type": "object", "additionalProperties": {"type": "string", "format": "uuid"},
                      "maxProperties": MAX_VARIABLES}, {"type": "null"}], "default": None}
    common = {"variable": dict(name, default="x"), "symbol_ids": ids, "limits": optional_limits}
    expression = dict(common, expression=text)
    variables = {"type": "array", "items": name, "minItems": 1,
                 "maxItems": MAX_VARIABLES, "uniqueItems": True}
    multivariate = {"expression": text, "variables": variables, "symbol_ids": ids, "limits": optional_limits}
    rows = {"type": "array", "maxItems": 4096,
            "items": {"type": "array", "maxItems": 4096, "items": rational}}
    operand = _object({"format": {"const": "mathforge"},
        "version": {"type": "integer", "enum": [1, 2, 3, 4, 5]}, "kind": {"const": "object"},
        "data": {"type": "object", "required": ["type"],
                 "properties": {"type": {"enum": sorted(operand_tags())}}}},
        ["format", "version", "kind", "data"])
    return {
        "mathforge.capabilities": _object({}),
        "mathforge.solve": _object(dict(common, problem=text), ["problem"]),
        "mathforge.solve_system": _object({
            "equations": {"type": "array", "items": text, "maxItems": MAX_EQUATIONS},
            "variables": {"type": "array", "items": name, "maxItems": MAX_VARIABLES, "uniqueItems": True},
            "symbol_ids": ids, "limits": optional_limits}, ["equations", "variables"]),
        "mathforge.real_roots": _object(expression, ["expression"]),
        "mathforge.differentiate": _object(dict(expression, order={"type": "integer", "minimum": 0,
                                                                   "maximum": 1024, "default": 1}), ["expression"]),
        "mathforge.expand_multivariate": _object(multivariate, ["expression", "variables"]),
        "mathforge.partial_diff": _object(dict(multivariate, variable=name,
            order={"type": "integer", "minimum": 0, "maximum": 1024, "default": 1}),
            ["expression", "variables", "variable"]),
        "mathforge.gradient": _object(multivariate, ["expression", "variables"]),
        "mathforge.jacobian": _object({
            "expressions": {"type": "array", "items": text, "minItems": 1, "maxItems": MAX_EQUATIONS},
            "variables": variables, "symbol_ids": ids, "limits": optional_limits}, ["expressions", "variables"]),
        "mathforge.hessian": _object(multivariate, ["expression", "variables"]),
        "mathforge.integrate": _object(dict(expression, lower=scalar, upper=scalar), ["expression", "lower", "upper"]),
        "mathforge.limit": _object(dict(expression,
            point={"anyOf": [scalar_text, {"type": "string", "enum": ["-inf", "+inf"]}, scalar_envelope]},
            side={"type": "string", "enum": ["both", "left", "right"], "default": "both"}),
            ["expression", "point"]),
        "mathforge.analyze_function": _object(expression, ["expression"]),
        "mathforge.det": _object({"rows": rows, "limits": optional_limits}, ["rows"]),
        "mathforge.rref": _object({"rows": rows, "limits": optional_limits,
            "ncols": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 4096}, {"type": "null"}], "default": None}}, ["rows"]),
        "mathforge.verify": _object({"result": {"type": "object"}, "limits": optional_limits}, ["result"]),
        "mathforge.factor": _object(expression, ["expression"]),
        "mathforge.resultant": _object(dict(common, f=text, g=text), ["f", "g"]),
        "mathforge.discriminant": _object(expression, ["expression"]),
        "mathforge.algebraic": _object({"expression": scalar, "limits": optional_limits}, ["expression"]),
        "mathforge.algebraic_div": _object({"left": scalar, "right": scalar, "limits": optional_limits}, ["left", "right"]),
        "mathforge.algebraic_pow": _object({"base": scalar,
            "n": {"type": "integer", "minimum": -1024, "maximum": 1024}, "limits": optional_limits}, ["base", "n"]),
        "mathforge.minimal_polynomial": _object(dict(common, expression=scalar), ["expression"]),
        "mathforge.evaluate_algebraic": _object({"expression": text, "variables": variables,
            "bindings": {"type": "object", "propertyNames": name, "additionalProperties": scalar,
                         "maxProperties": MAX_VARIABLES}, "symbol_ids": ids, "limits": optional_limits},
            ["expression", "variables", "bindings"]),
        "mathforge.apart": _object(expression, ["expression"]),
        "mathforge.extrema_values": _object(expression, ["expression"]),
        "mathforge.charpoly": _object(dict(common, rows=rows), ["rows"]),
        "mathforge.real_eigenvalues": _object({"rows": rows, "limits": optional_limits}, ["rows"]),
        "mathforge.parse_function": _object({"text": text,
            "variables": {"type": "array", "items": name, "maxItems": MAX_VARIABLES, "uniqueItems": True},
            "symbol_ids": ids, "limits": optional_limits}, ["text", "variables"]),
        "mathforge.compute": _object({"operation": {"type": "string", "enum": sorted(operation_arguments())},
            "arguments": {"type": "object", "maxProperties": MAX_VARIABLES,
                          "additionalProperties": operand,
                          "description": "Exactly the selected operation's named arguments, each a canonical object envelope. Include explicit defaults."},
            "limits": optional_limits}, ["operation", "arguments"]),
    }


def output_schema() -> dict:
    nullable_text = {"type": ["string", "null"]}
    error = _object({"stage": {"type": "string"}, "code": {"type": "string"}, "message": {"type": "string"},
                     "field": nullable_text, "start": {"type": ["integer", "null"]},
                     "end": {"type": ["integer", "null"]}}, ["stage", "code", "message", "field", "start", "end"])
    return _object({"api_version": {"const": 1}, "result": {"type": ["object", "null"]},
                    "text": {"type": "string"}, "latex": nullable_text,
                    "symbols": {"type": "array", "items": _object({"name": {"type": "string"}, "uid": {"type": "string"}}, ["name", "uid"])},
                    "error": {"anyOf": [error, {"type": "null"}]}},
                   ["api_version", "result", "text", "latex", "symbols", "error"])


def capabilities_schema() -> dict:
    return _object({"api_version": {"const": 1}, "package_version": {"type": "string"},
                    "transports": {"type": "array", "items": {"type": "string"}},
                    "tools": {"type": "array", "items": _object({
                        "name": {"type": "string"}, "sdk": {"type": "string"}, "description": {"type": "string"}},
                        ["name", "sdk", "description"])},
                    "operations": {"type": "array", "items": _object({
                        "name": {"type": "string"}, "sdk": {"type": "string"},
                        "mcp_tool": {"type": ["string", "null"]}, "scope": {"type": "string"}},
                        ["name", "sdk", "mcp_tool", "scope"])},
                    "input": {"type": "object"}, "output": {"type": "object"}, "limits": {"type": "object"},
                    "excluded": {"type": "array", "items": {"type": "string"}}},
                   ["api_version", "package_version", "transports", "tools", "operations", "input", "output", "limits", "excluded"])
