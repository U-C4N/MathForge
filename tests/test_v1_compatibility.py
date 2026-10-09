"""Frozen v0.1 wire bytes and content identities (SER-01, SER-02, SER-04)."""

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest


FIXTURES = Path(__file__).parent / "fixtures" / "v1"
BASELINE = "d001141"


def _write_baseline():
    """Called only in a subprocess importing the isolated, historical package."""
    import mathforge as m
    import mathforge.serialization as legacy_codec
    from mathforge.workspace import FrozenOptions

    if m.__version__ != "0.1.0":
        raise RuntimeError("Golden files must come from the v0.1.0 implementation")
    x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000001")
    y = m.Symbol("y", m.Rationals, (m.Assumption("nonzero", m.Truth.TRUE),),
                 uid="00000000-0000-4000-8000-000000000002")
    expr = m.Add((m.Rational(2, 3), m.Mul((m.Rational(-2), x)), m.Pow(x, 2), m.Sqrt(m.Rational(2))))
    poly = m.Polynomial((1, -2, 3), x)
    eq = m.Eq(x**2, 2)
    check = m.Check("example", m.CheckStatus.VERIFIED,
                    m.VerificationLevel.ALGORITHMIC_CHECK, "fixture", x, "exact record")
    condition = m.Condition(m.Rational(1), "gt", m.Rational(0), m.Truth.TRUE)
    step = m.Step("fixture", (poly, eq, (1, "text", True)), (expr,),
                  (condition,), m.StepRelation.DERIVATION, (check,), "legacy trace")
    error = m.ErrorInfo("fixture_error", "an explicit error")
    solved = m.solve(eq, for_=x)
    failed = m.Result(m.ExecutionStatus.INVALID_INPUT, m.Outcome.NOT_APPLICABLE,
                      method="fixture", problem=eq, for_=x, error=error)
    approximate = m.Result(m.ExecutionStatus.COMPLETED, m.Outcome.VALUE, m.Rational(7, 5),
                           exactness=m.Exactness.APPROXIMATE,
                           completeness=m.Completeness.UNKNOWN, expression=x,
                           precision=20, tolerance=m.Rational(1, 100),
                           error_bound=m.Rational(1, 1000))
    enums = tuple(value for enum in (m.Domain, m.Truth, m.ExecutionStatus, m.Outcome,
                                    m.Exactness, m.Completeness, m.CheckStatus,
                                    m.VerificationLevel, m.StepRelation) for value in enum)
    options = FrozenOptions((
        ("nested", {"integer": 2**80, "array": [None, True, "text", 1.25]}),
    ))
    ws = m.Workspace()
    xref, pref, rref = ws.put(x), ws.put(eq), ws.put(solved)
    record = ws.record("solve", {"problem": pref, "for_": xref},
                       {"exact": True, "nested": {"count": 2**80, "ratio": 1.25}}, rref)
    values = (None, False, True, "MathForge π", -(2**128), enums, x, y, expr, poly,
              eq, m.FiniteSet((1, m.sqrt(2))), m.EmptySet(), m.UniversalSet(),
              condition, check, m.VerificationReport((check,)), step, error,
              solved, failed, approximate, xref, record, options)
    fixtures = {"legacy_objects.json": m.dumps(values), "legacy_workspace.json": m.dumps(ws)}
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest = {"baseline": BASELINE, "package_version": m.__version__, "files": {}}
    for name, payload in fixtures.items():
        (FIXTURES / name).write_text(payload, encoding="utf-8", newline="")
        manifest["files"][name] = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    reader = Path(legacy_codec.__file__).read_bytes()
    (FIXTURES / "serialization_v1.py").write_bytes(reader)
    manifest["reader"] = {"file": "serialization_v1.py",
                          "sha256": hashlib.sha256(reader).hexdigest()}
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _generate_from_git():
    """Deliberate maintenance command; never run as part of the test suite."""
    repo = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE, "src/mathforge"], cwd=repo, text=True
    ).splitlines()
    with tempfile.TemporaryDirectory(prefix="mathforge-v1-baseline-") as temp:
        root = Path(temp)
        for name in paths:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=repo))
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--write-baseline"],
                       cwd=root, env=env, check=True)


class V1CompatibilityTests(unittest.TestCase):
    def test_every_frozen_v1_byte_and_hash(self):
        """SER-01: the frozen historical codec is the oracle, not current code."""
        import mathforge as m
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["baseline"], BASELINE)
        for filename, digest in manifest["files"].items():
            with self.subTest(filename=filename):
                payload = (FIXTURES / filename).read_text(encoding="utf-8")
                self.assertEqual("sha256:" + hashlib.sha256(payload.encode()).hexdigest(), digest)
                restored = m.loads(payload)
                self.assertEqual(m.dumps(restored), payload)
                self.assertEqual(m.to_data(restored)["version"], 1)

    def test_legacy_results_have_no_new_wire_field(self):
        """SER-02/VER-13: old requests stay absent and valid legacy solutions reverify."""
        import mathforge as m
        values = m.loads((FIXTURES / "legacy_objects.json").read_text(encoding="utf-8"))
        results = [value for value in values if type(value) is m.Result]
        self.assertTrue(results)
        for result in results:
            self.assertIsNone(getattr(result, "request", None))
            data = m.to_data(result)
            self.assertEqual(data["version"], 1)
            self.assertNotIn("request", data["data"])
            if result.outcome is m.Outcome.SOLUTIONS:
                self.assertTrue(m.verify(result).verified)

    def test_historical_v1_reader_explicitly_rejects_v2(self):
        """SER-20: exercise the original decoder, not an imitation of its rule."""
        import mathforge as m
        from mathforge.matrices import Matrix
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        source = FIXTURES / manifest["reader"]["file"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest["reader"]["sha256"])
        spec = importlib.util.spec_from_file_location("mathforge._historical_v1_transport", source)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        self.assertEqual(reader.VERSION, 1)
        self.assertEqual(reader.loads(reader.dumps(m.Rational(2, 3))), m.Rational(2, 3))
        with self.assertRaises(m.SerializationError):
            reader.loads(m.dumps(Matrix(((1,),))))

    def test_legacy_escaped_python_strings_remain_roundtrippable(self):
        """SER-01: byte-budget inspection preserves old ensure_ascii string support."""
        import mathforge as m
        for value in ("\ud800", "\udfff", "normal \u03c0 \U0001f600"):
            with self.subTest(value=ascii(value)):
                payload = m.dumps(value)
                self.assertEqual(m.to_data(value)["version"], 1)
                self.assertEqual(m.loads(payload), value)


if __name__ == "__main__":
    if "--generate-baseline" in sys.argv:
        _generate_from_git()
    elif "--write-baseline" in sys.argv:
        _write_baseline()
    else:
        unittest.main()
