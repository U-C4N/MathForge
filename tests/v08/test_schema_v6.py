"""The packaged JSON v6 structural schema: v6 documents validate, and type dispatch keeps validation fast."""

import importlib.util
from importlib import resources
import json
from pathlib import Path
import subprocess
import sys
import unittest

import mathforge as m
from mathforge.deferred import Reason, Unevaluated

SPEED_BOUND = 10.0  # Seconds. The same 60 documents take about 65 s against the v5 schema.

# Validation is timed in a fresh interpreter, as test_lazy_areas times imports: the operating system may slow
# down a test process that has been busy for minutes, and the bound is about the schema, not the process.
TIMING = """
import json, sys, time
from jsonschema import Draft202012Validator
work = json.load(sys.stdin)
validator = Draft202012Validator(work["schema"])
start = time.perf_counter()
for document in work["documents"]:
    validator.validate(document)
print(json.dumps(time.perf_counter() - start))
"""


def frontend_results():
    """The 60 v0.7 operation results built from test_frontend_integration_v7.samples(), loaded by path."""
    path = Path(__file__).resolve().parents[1] / "test_frontend_integration_v7.py"
    spec = importlib.util.spec_from_file_location("frontend_integration_v7_samples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.results()


class SchemaV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            raise unittest.SkipTest("Development-only jsonschema is unavailable") from None
        path = resources.files("mathforge").joinpath("schemas/mathforge-v6.schema.json")
        cls.schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)
        value = m.Rational(415, 93)
        cls.pilot = m.operations.continued_fraction(value)
        request = m.OperationRequest("continued_fraction", (("value", value),))
        cls.deferred = m.Result(
            execution_status=m.ExecutionStatus.COMPLETED, outcome=m.Outcome.NO_CONCLUSION,
            completeness=m.Completeness.UNKNOWN, method="continued_fraction", request=request,
            deferred=Unevaluated("evaluate", request, Reason("no_applicable_rule")))

    def canonical(self, value):
        """The document of a value that has crossed dumps and loads unchanged."""
        restored = m.loads(m.dumps(value))
        self.assertEqual(restored, value)
        document = m.to_data(restored)
        self.assertEqual(document["version"], 6)
        return document

    def test_v6_schema_accepts_v6_documents(self):
        self.assertEqual(self.pilot.value.terms, (4, 2, 6, 7))
        for label, value in (("pilot result", self.pilot), ("deferred result", self.deferred)):
            with self.subTest(document=label):
                self.validator.validate(self.canonical(value))

    def test_v6_schema_rejects_unknown_tags(self):
        top = {"format": "mathforge", "version": 6, "kind": "object", "data": {"type": "v6_nope"}}
        nested = self.canonical(self.pilot)
        nested["data"]["value"] = {"type": "v6_nope"}
        for label, document in (("top level", top), ("result value", nested)):
            with self.subTest(document=label):
                self.assertFalse(self.validator.is_valid(document))

    def test_v6_schema_checks_v6_request_arguments(self):
        """A registered v6 operation names exactly its arguments; an operation of a later build stays open."""
        document = self.canonical(self.pilot)
        document["data"]["request"]["arguments"][0]["name"] = "number"
        self.assertFalse(self.validator.is_valid(document))
        request = m.OperationRequest("later_operation", (("anything", 1),))
        later = m.to_data(request)
        later["version"] = 6
        self.assertEqual(m.from_data(later), request)  # The decoder accepts it, so the schema must too.
        self.validator.validate(later)

    # No docstring: unittest would print it on its own line, between the test's name and the measurement.
    def test_v6_schema_is_fast(self):
        documents = []
        for result in frontend_results().values():
            document = m.to_data(result)
            document["version"] = 6
            documents.append(document)
        self.assertEqual(len(documents), 60)
        work = json.dumps({"schema": self.schema, "documents": documents})
        output = subprocess.run([sys.executable, "-I", "-c", TIMING], input=work,
                                capture_output=True, text=True, timeout=300)
        if output.returncode:  # An invalid document lands here too: validate() raised in the child.
            raise AssertionError(output.stderr)
        elapsed = json.loads(output.stdout)
        print(f"v6 schema: {len(documents)} documents validated in {elapsed:.2f} s in a fresh interpreter "
              f"(bound {SPEED_BOUND:.0f} s)", file=sys.stderr)
        self.assertLess(elapsed, SPEED_BOUND)


if __name__ == "__main__":
    unittest.main()
