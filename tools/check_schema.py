"""SER-03/07/08 REL-07: Validate structural schema against real v2 codec output.

This is a development-only check requiring jsonschema==4.26.0. The schema is not
used to load mathematical objects at runtime, and is not a mathematical verifier.
"""

import copy
from importlib import resources
import json

import mathforge as mf


def main() -> None:
    try:
        from jsonschema import Draft202012Validator, ValidationError
    except ImportError as exc:
        raise SystemExit("Install jsonschema==4.26.0 to run the development schema check") from exc
    schema_path = resources.files("mathforge").joinpath("schemas/mathforge-v2.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    x, y = mf.symbol("x"), mf.symbol("y")
    A = mf.Matrix(((1, 2), (3, 4)))
    system = mf.LinearSystem(A, (5, 11), (x, y))
    rf = mf.rational_function(x**2 - 1, x - 1, variable=x)
    values = [A, mf.rref(A), system, mf.solve_system(system), rf,
              mf.real_roots(x**3 - 2, x), mf.square_free(mf.Polynomial.from_expression((x - 1)**3, x)),
              mf.solve(mf.Eq(x**3 - 2, 0), for_=x),
              mf.solve(mf.Inequality(rf, "ge", 0), for_=x),
              mf.operations.integrate(x**2, x, 0, 1)]
    workspace = mf.Workspace()
    workspace.put(mf.Rational(1, 3))
    for value in values:
        workspace.put(value)
    values.append(workspace)
    documents = [mf.to_data(value) for value in values]
    for document in documents:
        assert document["version"] == 2, document
        validator.validate(document)
    malformed = []
    bad = copy.deepcopy(documents[0])
    bad["data"]["ncols"] = True
    malformed.append(bad)
    bad = copy.deepcopy(documents[0])
    bad["data"]["rows"][0][0]["numerator"] = "01"
    malformed.append(bad)
    bad = copy.deepcopy(documents[3])
    del bad["data"]["request"]
    malformed.append(bad)
    bad = copy.deepcopy(documents[0])
    bad["data"]["unknown"] = True
    malformed.append(bad)
    for document in malformed:
        try:
            validator.validate(document)
        except ValidationError:
            continue
        raise AssertionError("Malformed transport shape passed the schema")
    print(f"V2 structural schema accepts {len(documents)} real documents and rejects {len(malformed)} corrupt shapes.")


if __name__ == "__main__":
    main()
