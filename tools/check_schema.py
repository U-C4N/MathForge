"""SER-03/07/08 REL-07: Validate structural schemas against v2–v6 codec output.

This is a development-only check requiring jsonschema==4.26.0. The schema is not
used to load mathematical objects at runtime, and is not a mathematical verifier.
"""

import copy
from importlib import resources
import json

import mathforge as mf
from mathforge.analysis_types import ExcludedPoint, FunctionAnalysis, LimitValue, LocalExtremum, MonotonicRegion


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

    v3_path = resources.files("mathforge").joinpath("schemas/mathforge-v3.schema.json")
    v3_schema = json.loads(v3_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(v3_schema)
    v3_validator = Draft202012Validator(v3_schema)
    zero, one = mf.Rational(0), mf.Rational(1)
    all_reals, empty = mf.IntervalSet((mf.Interval(None, None),)), mf.IntervalSet(())
    root = mf.RealAlgebraicRoot((-2, 0, 1), 1)
    analysis = FunctionAnalysis(
        function=mf.rational_function(x**2, 1, variable=x), domain=all_reals,
        excluded_points=(), zero_set=mf.IntervalSet((mf.Interval(zero, zero, True, True),)),
        positive_set=mf.IntervalSet((mf.Interval(None, zero), mf.Interval(zero, None))),
        negative_set=empty, derivative=mf.rational_function(2*x, 1, variable=x),
        stationary_set=mf.IntervalSet((mf.Interval(zero, zero, True, True),)),
        monotonic_regions=(MonotonicRegion(mf.Interval(None, zero, False, True), "decreasing"),
                           MonotonicRegion(mf.Interval(zero, None, True, False), "increasing")),
        strict_extrema=(LocalExtremum(zero, "minimum"),),
    )
    v3_values = [LimitValue("finite", one), LimitValue("positive_infinity"),
                 LimitValue("negative_infinity"), LimitValue("does_not_exist"),
                 ExcludedPoint(root, "hole"), ExcludedPoint(zero, "pole"),
                 analysis.monotonic_regions[0], analysis.strict_extrema[0], analysis,
                 mf.OperationRequest("limit", (("expression", rf), ("variable", x),
                                              ("point", "+inf"), ("side", "both"))),
                 mf.OperationRequest("analyze_function", (("expression", rf), ("variable", x)))]
    # These exercise real public result envelopes, including their certificates.
    v3_values.extend((mf.operations.limit(rf, x, 1), mf.operations.analyze_function(rf, x)))
    for value in v3_values:
        workspace.put(value)
    v3_values.append(workspace)
    v3_documents = [mf.to_data(value) for value in v3_values]
    for document in v3_documents:
        assert document["version"] == 3, document
        v3_validator.validate(document)
        try:
            validator.validate(document)
        except ValidationError:
            pass
        else:
            raise AssertionError("A v3 document passed the historical v2 schema")
    v3_malformed = []
    for index, changes in ((0, {"value": None}), (1, {"value": mf.to_data(one)["data"]}),
                           (4, {"kind": "unknown"}), (6, {"extra": True}),
                           (8, {"zero_set": {"type": "text", "value": "all"}})):
        bad = copy.deepcopy(v3_documents[index])
        bad["data"].update(changes)
        v3_malformed.append(bad)
    bad = copy.deepcopy(v3_documents[9])
    bad["data"]["arguments"].pop()
    v3_malformed.append(bad)
    for document in v3_malformed:
        try:
            v3_validator.validate(document)
        except ValidationError:
            continue
        raise AssertionError("Malformed v3 transport shape passed the schema")
    print(f"V3 structural schema accepts {len(v3_documents)} real documents and rejects {len(v3_malformed)} corrupt shapes.")

    v4_path = resources.files("mathforge").joinpath("schemas/mathforge-v4.schema.json")
    v4_schema = json.loads(v4_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(v4_schema)
    v4_validator = Draft202012Validator(v4_schema)
    p = mf.Polynomial((4, 0, 0, 0, 1), x)
    alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
    v4_values = [mf.operations.factor(p), mf.operations.resultant(p, p), mf.operations.discriminant(p),
                 mf.operations.algebraic(mf.sqrt(2)+mf.sqrt(3)), mf.operations.algebraic_div(alpha, 2),
                 mf.operations.algebraic_pow(alpha, -1), mf.operations.minimal_polynomial(alpha, x),
                 mf.operations.evaluate_algebraic(x**3, {x: alpha}), mf.operations.apart(rf),
                 mf.operations.extrema_values(x**3-6*x, x), mf.operations.charpoly(A, x),
                 mf.operations.real_eigenvalues(A), mf.operations.limit(x, x, alpha),
                 mf.operations.integrate(x, x, 0, mf.sqrt(2))]
    for value in v4_values:
        assert value.execution_status is mf.ExecutionStatus.COMPLETED, value.error
        document = mf.to_data(value)
        assert document["version"] == 4
        v4_validator.validate(document)
        assert mf.loads(mf.dumps(value)) == value
        workspace.put(value)
    v4_validator.validate(mf.to_data(workspace))
    malformed = mf.to_data(v4_values[0].value)
    malformed["data"]["factors"][0]["multiplicity"] = "0"
    try:
        v4_validator.validate(malformed)
    except ValidationError:
        pass
    else:
        raise AssertionError("Invalid v4 factor multiplicity passed schema")
    print(f"V4 structural schema accepts {len(v4_values)} operation results and a mixed workspace; corrupt shapes rejected.")

    v5_schema = json.loads(resources.files("mathforge").joinpath("schemas/mathforge-v5.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(v5_schema)
    v5_validator = Draft202012Validator(v5_schema)
    from mathforge import functions as f
    real_function = mf.real_function(f.sin(x), (x,))
    v5_values = [mf.operations.integer_xgcd(-30, 21), mf.operations.crt(((0, 2), (1, 2))),
        mf.operations.is_prime(65537), mf.operations.factor_integer(-360),
        mf.operations.multinomial((2, 3, 1)), mf.operations.complex_sqrt(mf.ComplexAlgebraic(3, 4)),
        mf.operations.quadratic_roots(1, 0, 2), mf.operations.binomial_distribution(5, mf.Rational(1, 3)),
        mf.operations.stddev((1, 2, 4)), mf.operations.lu(A),
        mf.operations.least_squares(mf.Matrix(((1, 1), (2, 2))), (1, 3)),
        mf.operations.function_differentiate(real_function, x), mf.operations.taylor(real_function, x, 0, 4),
        mf.operations.function_integrate(real_function, x, 0, 1),
        mf.operations.function_limit(mf.real_function(f.sin(x)/f.as_real_expr(x), (x,)), x, 0),
        mf.operations.rational_antiderivative(mf.rational_function(1, x**5-x+1, variable=x), x),
        mf.operations.rational_integrate(mf.rational_function(1, x, variable=x), x, 1, 2)]
    for value in v5_values:
        assert value.execution_status is mf.ExecutionStatus.COMPLETED and value.value is not None, value.error
        assert mf.verify(value).verified
    deferred = mf.operations.function_antiderivative(mf.real_function(f.sin(f.as_real_expr(x)**2), (x,)), x)
    assert deferred.outcome is mf.Outcome.NO_CONCLUSION and not mf.verify(deferred).verified
    v5_values.append(deferred)
    for value in v5_values:
        document = mf.to_data(value)
        assert document["version"] == 5
        v5_validator.validate(document)
        assert mf.loads(mf.dumps(value)) == value
        try:
            v4_validator.validate(document)
        except ValidationError:
            pass
        else:
            raise AssertionError("A v5 document passed the historical v4 schema")
        workspace.put(value)
    v5_validator.validate(mf.to_data(workspace))
    corrupt = mf.to_data(v5_values[3])
    corrupt["data"]["extra"] = True
    try:
        v5_validator.validate(corrupt)
    except ValidationError:
        pass
    else:
        raise AssertionError("An unexpected v5 field passed the schema")
    print(f"V5 structural schema accepts {len(v5_values)} results, including an explicitly deferred request and mixed workspace.")

    v6_schema = json.loads(resources.files("mathforge").joinpath("schemas/mathforge-v6.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(v6_schema)
    v6_validator = Draft202012Validator(v6_schema)
    from mathforge.deferred import Reason, Unevaluated
    rational = mf.Rational(415, 93)
    pilot = mf.operations.continued_fraction(rational)
    assert pilot.execution_status is mf.ExecutionStatus.COMPLETED and mf.verify(pilot).verified, pilot.error
    request = mf.OperationRequest("continued_fraction", (("value", rational),))
    unevaluated = Unevaluated("evaluate", request, Reason("no_applicable_rule"))
    deferred = mf.Result(execution_status=mf.ExecutionStatus.COMPLETED, outcome=mf.Outcome.NO_CONCLUSION,
                         completeness=mf.Completeness.UNKNOWN, method="continued_fraction", request=request,
                         deferred=unevaluated)
    v6_values = [pilot, pilot.value, request, unevaluated, deferred]
    for value in v6_values:
        document = mf.to_data(value)
        assert document["version"] == 6
        v6_validator.validate(document)
        assert mf.loads(mf.dumps(value)) == value
        try:
            v5_validator.validate(document)
        except ValidationError:
            pass
        else:
            raise AssertionError("A v6 document passed the historical v5 schema")
        workspace.put(value)
    v6_validator.validate(mf.to_data(workspace))
    v6_malformed = []
    for change in ({"extra": True}, {"value": {"type": "v6_nope"}}):
        bad = mf.to_data(pilot)
        bad["data"].update(change)
        v6_malformed.append(bad)
    bad = mf.to_data(pilot)
    bad["data"]["request"]["arguments"][0]["name"] = "number"
    v6_malformed.append(bad)
    bad = mf.to_data(deferred)
    bad["data"]["deferred"] = bad["data"]["deferred"]["reason"]
    v6_malformed.append(bad)
    for document in v6_malformed:
        try:
            v6_validator.validate(document)
        except ValidationError:
            continue
        raise AssertionError("Malformed v6 transport shape passed the schema")
    print(f"V6 structural schema accepts {len(v6_values)} values, including a deferred request, and a mixed workspace; "
          f"{len(v6_malformed)} corrupt shapes rejected.")


if __name__ == "__main__":
    main()
