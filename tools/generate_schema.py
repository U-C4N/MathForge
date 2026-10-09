"""Generate the structural v2 JSON Schema, or check that the saved copy is current.

The codec remains responsible for mathematical canonicalization, content hashes,
UUID consistency, shape dependencies, root validity, and resource limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DESTINATION = Path(__file__).resolve().parents[1] / "src/mathforge/schemas/mathforge-v2.schema.json"


def ref(name):
    return {"$ref": f"#/$defs/{name}"}


def array(item, **constraints):
    return {"type": "array", "items": item, **constraints}


def nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


def choice(*names):
    return {"oneOf": [ref(name) for name in names]}


def fields(**properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def tagged(tag, **properties):
    return fields(type={"const": tag}, **properties)


def build_schema():
    text = {"type": "string"}
    boolean = {"type": "boolean"}
    definitions = {
        "decimal": {"type": "string", "pattern": r"^(0|[1-9][0-9]*|-[1-9][0-9]*)$"},
        "nonnegative_decimal": {"type": "string", "pattern": r"^(0|[1-9][0-9]*)$"},
        "positive_decimal": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
        "sha256": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
        "uuid": {"type": "string", "pattern": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"},
    }
    enumerations = {
        "domain": ("reals", "rationals", "integers"),
        "truth": ("true", "false", "unknown"),
        "execution_status": ("completed", "unsupported", "invalid_input", "error"),
        "outcome": ("value", "solutions", "candidates", "no_solution", "no_conclusion", "partial", "not_applicable"),
        "exactness": ("exact", "approximate", "mixed", "not_applicable"),
        "completeness": ("complete", "partial", "unknown", "not_applicable"),
        "check_status": ("verified", "failed", "unknown", "unchecked"),
        "verification_level": ("none", "exact_substitution", "algorithmic_check"),
        "step_relation": ("equivalence", "implication", "approximation", "derivation", "evaluation"),
    }
    nodes = []

    def node(definition_key, **properties):
        definitions[definition_key] = tagged(definition_key, **properties)
        nodes.append(definition_key)

    for name, values in enumerations.items():
        definitions[f"{name}_value"] = {"enum": list(values)}
        node(name, value=ref(f"{name}_value"))
    node("none")
    node("boolean", value=boolean)
    node("text", value=text)
    node("integer", value=ref("decimal"))
    node("rational", numerator=ref("decimal"), denominator=ref("positive_decimal"))
    node("assumption", name={"enum": ["positive", "nonnegative", "nonzero"]}, truth=ref("truth_value"))
    node("symbol", name={"type": "string", "minLength": 1}, uid=ref("uuid"),
         domain=ref("domain_value"), assumptions=array(ref("assumption")))
    node("add", terms=array(ref("expression"), minItems=1))
    node("mul", factors=array(ref("expression"), minItems=1))
    node("pow", base=ref("expression"), exponent=ref("nonnegative_decimal"))
    node("sqrt", radicand=ref("rational"))
    node("polynomial", coefficients=array(ref("rational"), minItems=1), variable=ref("symbol"))
    node("equation", lhs=ref("scalar"), rhs=ref("scalar"))
    node("finite_set", values=array(ref("expression")), domain=ref("domain_value"))
    node("empty_set", domain=ref("domain_value"))
    node("universal_set", domain=ref("domain_value"))
    node("condition", lhs=ref("expression"), relation={"enum": ["eq", "ne", "lt", "le", "gt", "ge"]},
         rhs=ref("expression"), truth=ref("truth_value"))
    node("check", claim={"type": "string", "minLength": 1}, status=ref("check_status_value"),
         level=ref("verification_level_value"), method=text, subject=nullable(ref("expression")), details=text)
    node("verification_report", checks=array(ref("check")))
    node("step", rule={"type": "string", "minLength": 1}, inputs=array(ref("node")),
         outputs=array(ref("node")), preconditions=array(ref("condition")),
         relation=ref("step_relation_value"), audit=array(ref("check")), explanation=nullable(text))
    node("error_info", code={"type": "string", "minLength": 1}, message=text)
    node("tuple", items=array(ref("node")))
    node("object_ref", identifier=ref("sha256"))
    node("square_free_factor", polynomial=ref("polynomial"), multiplicity=ref("positive_decimal"))
    node("square_free_decomposition", coefficient=ref("rational"), factors=array(ref("square_free_factor")))
    node("real_algebraic_root", integer_coefficients=array(ref("decimal"), minItems=2),
         real_index=ref("nonnegative_decimal"))
    node("root_record", root=ref("expression"), multiplicity=ref("positive_decimal"))
    node("rational_function", numerator=ref("polynomial"), denominator=ref("polynomial"),
         excluded=array(ref("polynomial")))
    node("matrix", rows=array(array(ref("rational"))), ncols=ref("nonnegative_decimal"))
    node("rref_result", matrix=ref("matrix"), pivot_columns=array(ref("nonnegative_decimal")))
    node("linear_system", coefficients=ref("matrix"), rhs=array(ref("rational")), variables=array(ref("symbol")))
    node("affine_solution_set", variables=array(ref("symbol")), particular=array(ref("rational")),
         basis=array(array(ref("rational"))))
    node("inequality", lhs=ref("scalar"), relation={"enum": ["lt", "le", "gt", "ge"]}, rhs=ref("scalar"))
    node("interval", lower=nullable(choice("rational", "real_algebraic_root")),
         upper=nullable(choice("rational", "real_algebraic_root")),
         left_closed=boolean, right_closed=boolean)
    node("interval_set", intervals=array(ref("interval")))
    node("operation_request", operation={"type": "string", "minLength": 1},
         arguments=array(fields(name={"type": "string", "minLength": 1}, value=ref("node"))))
    definitions["options"] = tagged("options", entries=array(fields(name=text, value=ref("option"))))
    definitions["option_integer"] = tagged("option_integer", value=ref("decimal"))
    definitions["option_array"] = tagged("option_array", items=array(ref("option")))
    definitions["option"] = {"anyOf": [
        {"type": ["null", "string", "boolean", "number"]},
        ref("options"), ref("option_integer"), ref("option_array"),
    ]}
    node("operation_record", operation={"type": "string", "minLength": 1},
         inputs=array(fields(name=text, ref=ref("object_ref"))), options=ref("options"), result_ref=ref("object_ref"))
    node("result", execution_status=ref("execution_status_value"), outcome=ref("outcome_value"),
         value=nullable(ref("node")), exactness=ref("exactness_value"), completeness=ref("completeness_value"),
         verification=array(ref("check")), method=text, domain=nullable(ref("domain_value")),
         assumptions=array(ref("assumption")), steps=array(ref("step")),
         problem=nullable(choice("equation", "inequality", "linear_system")),
         expression=nullable({"anyOf": [ref("expression"), ref("polynomial"), ref("rational_function"), ref("matrix")]}),
         for_=nullable(ref("symbol")), precision=nullable(ref("positive_decimal")),
         tolerance=nullable(ref("rational")), error_bound=nullable(ref("rational")),
         error=nullable(ref("error_info")), request=nullable(ref("operation_request")))
    definitions["workspace"] = tagged("workspace", objects=array(fields(ref=ref("sha256"), object=ref("node"))),
                                       history=array(ref("operation_record")))
    definitions["expression"] = choice("rational", "symbol", "add", "mul", "pow", "sqrt", "real_algebraic_root")
    definitions["scalar"] = {"anyOf": [ref("expression"), ref("rational_function")]}
    definitions["node"] = choice(*nodes)
    schema = fields(format={"const": "mathforge"}, version={"const": 2},
                    kind={"enum": ["object", "workspace"]},
                    data={"anyOf": [ref("node"), ref("workspace")]})
    schema.update({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:mathforge:serialization:v2",
        "title": "MathForge format v2 structural schema",
        "description": ("Structural transport constraints only. Canonical arithmetic, symbol consistency, "
                        "shape relations, mathematical claims, content hashes and resource limits require "
                        "the MathForge decoder and verifier."),
        "$defs": definitions,
        "allOf": [{"if": {"properties": {"kind": {"const": "workspace"}}},
                   "then": {"properties": {"data": ref("workspace")}},
                   "else": {"properties": {"data": ref("node")}}}],
    })
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail instead of rewriting a stale schema")
    arguments = parser.parse_args()
    text = json.dumps(build_schema(), indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    if arguments.check:
        if not DESTINATION.is_file() or DESTINATION.read_text(encoding="utf-8") != text:
            raise SystemExit("Schema is missing or stale; run python tools/generate_schema.py")
        print("Saved v2 schema matches its generator.")
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        DESTINATION.write_text(text, encoding="utf-8")
        print(DESTINATION)


if __name__ == "__main__":
    main()
