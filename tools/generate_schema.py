"""Generate structural v2–v5 JSON Schemas, or check their saved copies.

The codec remains responsible for mathematical canonicalization, content hashes,
UUID consistency, shape dependencies, root validity, and resource limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path


DESTINATION = Path(__file__).resolve().parents[1] / "src/mathforge/schemas/mathforge-v2.schema.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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


def build_schema(version=2):
    if version not in (2, 3, 4, 5):
        raise ValueError("Schema version must be 2, 3, 4 or 5")
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
    if version >= 3:
        finite_value = choice("rational", "real_algebraic_root") if version >= 4 else ref("rational")
        node("limit_value", kind={"enum": ["finite", "positive_infinity", "negative_infinity", "does_not_exist"]},
             value=nullable(finite_value))
        definitions["limit_value"]["allOf"] = [{
            "if": {"properties": {"kind": {"const": "finite"}}},
            "then": {"properties": {"value": finite_value}},
            "else": {"properties": {"value": {"type": "null"}}},
        }]
        node("excluded_point", point=choice("rational", "real_algebraic_root"),
             kind={"enum": ["hole", "pole"]})
        node("monotonic_region", interval=ref("interval"),
             behavior={"enum": ["increasing", "decreasing", "constant"]})
        node("local_extremum", point=choice("rational", "real_algebraic_root"),
             kind={"enum": ["minimum", "maximum"]})
        node("function_analysis", function=ref("rational_function"), domain=ref("interval_set"),
             excluded_points=array(ref("excluded_point")), zero_set=ref("interval_set"),
             positive_set=ref("interval_set"), negative_set=ref("interval_set"),
             derivative=ref("rational_function"), stationary_set=ref("interval_set"),
             monotonic_regions=array(ref("monotonic_region")), strict_extrema=array(ref("local_extremum")))

        def request(operation, arguments):
            items = [fields(name={"const": name}, value=value) for name, value in sorted(arguments.items())]
            return tagged("operation_request", operation={"const": operation},
                          arguments={"type": "array", "prefixItems": items,
                                     "minItems": len(items), "maxItems": len(items), "items": False})

        legacy_request = definitions["operation_request"]
        legacy_request["properties"]["operation"]["not"] = {"enum": ["limit", "analyze_function"]}
        expression = {"anyOf": [ref("expression"), ref("polynomial"), ref("rational_function")]}
        point = {"oneOf": [ref("rational"), tagged("text", value={"enum": ["+inf", "-inf"]})]}
        if version >= 4:
            point = {"oneOf": [ref("expression"), tagged("text", value={"enum": ["+inf", "-inf"]})]}
        definitions["operation_request"] = {"oneOf": [
            legacy_request,
            request("limit", {"expression": expression, "variable": ref("symbol"),
                              "point": point, "side": tagged("text", value={"enum": ["left", "right", "both"]})}),
            request("analyze_function", {"expression": expression, "variable": ref("symbol")}),
        ]}
        if version >= 4:
            node("polynomial_factor", polynomial=ref("polynomial"), multiplicity=ref("positive_decimal"))
            node("polynomial_factorization", coefficient=ref("rational"), factors=array(ref("polynomial_factor")))
            node("partial_fraction_term", numerator=ref("polynomial"), factor=ref("polynomial"), power=ref("positive_decimal"))
            node("partial_fraction_decomposition", polynomial=ref("polynomial"),
                 terms=array(ref("partial_fraction_term")), excluded=array(ref("polynomial")))
            node("extremum_value", point=choice("rational", "real_algebraic_root"),
                 value=choice("rational", "real_algebraic_root"), kind={"enum": ["minimum", "maximum"]})
            scalar = ref("expression")
            pair = tagged("tuple", items={"type": "array", "prefixItems": [ref("symbol"), scalar],
                                           "minItems": 2, "maxItems": 2, "items": False})
            bindings = tagged("tuple", items=array(pair))
            parameters = {
                "factor": {"f": ref("polynomial")},
                "resultant": {"f": ref("polynomial"), "g": ref("polynomial")},
                "discriminant": {"f": ref("polynomial")},
                "algebraic": {"expression": expression},
                "algebraic_div": {"left": scalar, "right": scalar},
                "algebraic_pow": {"base": scalar, "n": ref("integer")},
                "minimal_polynomial": {"expression": expression, "variable": ref("symbol")},
                "evaluate_algebraic": {"expression": expression, "bindings": bindings},
                "apart": {"expression": ref("rational_function")},
                "extrema_values": {"expression": expression, "variable": ref("symbol")},
                "charpoly": {"A": ref("matrix"), "variable": ref("symbol")},
                "real_eigenvalues": {"A": ref("matrix")},
            }
            legacy_request["properties"]["operation"]["not"]["enum"].extend(parameters)
            definitions["operation_request"]["oneOf"].extend(request(name, values) for name, values in parameters.items())
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
    if version >= 5:
        from mathforge.dispatch_v5 import record_types, operation_arguments
        for tag, cls in record_types().items():
            node("v5_" + tag, **{field.name: ref("node") for field in dataclass_fields(cls)})
        definitions["result"]["properties"]["deferred"] = choice("v5_unevaluated_integral", "v5_unevaluated_limit")
        definitions["result"]["properties"]["expression"]["anyOf"].append(
            choice("v5_real_expr", "v5_real_function", "v5_rational_primitive"))
        new_parameters = operation_arguments()
        legacy_request["properties"]["operation"]["not"]["enum"].extend(sorted(new_parameters))
        definitions["operation_request"]["oneOf"].extend(
            request(name, {key: ref("node") for key in sorted(keys)})
            for name, keys in sorted(new_parameters.items()))
    definitions["workspace"] = tagged("workspace", objects=array(fields(ref=ref("sha256"), object=ref("node"))),
                                       history=array(ref("operation_record")))
    definitions["expression"] = choice("rational", "symbol", "add", "mul", "pow", "sqrt", "real_algebraic_root")
    definitions["scalar"] = {"anyOf": [ref("expression"), ref("rational_function")]}
    definitions["node"] = choice(*nodes)
    if version >= 5:
        # Gate recursion on discriminators. Generic proof fields may share names
        # across many record types; exploring all mismatching records first can
        # otherwise multiply recursive validation exponentially.
        for name, definition in tuple(definitions.items()):
            properties = definition.get("properties", {})
            if "const" in properties.get("type", {}):
                definitions[name] = {
                    "if": {"type": "object", "required": ["type"],
                           "properties": {"type": properties["type"]}},
                    "then": definition, "else": False}
        definitions["operation_request"]["oneOf"] = [
            {"if": {"type": "object", "required": ["operation"],
                    "properties": {"operation": entry["properties"]["operation"]}},
             "then": entry, "else": False}
            for entry in definitions["operation_request"]["oneOf"]]
    schema = fields(format={"const": "mathforge"}, version={"const": version},
                    kind={"enum": ["object", "workspace"]},
                    data={"anyOf": [ref("node"), ref("workspace")]})
    schema.update({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:mathforge:serialization:v{version}",
        "title": f"MathForge format v{version} structural schema",
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
    parser.add_argument("--version", type=int, choices=(2, 3, 4, 5), help="Process only one schema version")
    arguments = parser.parse_args()
    for version in ((arguments.version,) if arguments.version else (2, 3, 4, 5)):
        destination = DESTINATION.with_name(f"mathforge-v{version}.schema.json")
        text = json.dumps(build_schema(version), indent=2, ensure_ascii=True, sort_keys=True) + "\n"
        current = destination.is_file() and destination.read_text(encoding="utf-8") == text
        if arguments.check:
            if not current:
                raise SystemExit(f"V{version} schema is missing or stale; run python tools/generate_schema.py")
            print(f"Saved v{version} schema matches its generator.")
        else:
            # In particular, leave the historical v2 file byte-for-byte intact.
            if not current:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8", newline="\n")
            print(destination)


if __name__ == "__main__":
    main()
