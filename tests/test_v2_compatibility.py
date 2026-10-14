"""Frozen v0.2 wire bytes and identities, generated only from the release tag."""

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest


FIXTURES = Path(__file__).parent / "fixtures" / "v2"
BASELINE = "0b9a42b7b92309b659120ae93094f18c60391b6a"


def _write_baseline():
    import mathforge as m
    import mathforge.serialization as codec

    if m.__version__ != "0.2.0":
        raise RuntimeError("Golden files must come from the v0.2.0 implementation")
    x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000031")
    y = m.Symbol("y", uid="00000000-0000-4000-8000-000000000032")
    p, q = m.Polynomial((-1, 0, 1), x), m.Polynomial((-1, 1), x)
    A = m.Matrix(((1, 2), (3, 4)))
    system = m.LinearSystem(m.Matrix(((1, 1),)), (1,), (x, y))
    rf = m.rational_function(x**2 - 1, x - 1, variable=x)
    root = m.RealAlgebraicRoot((-2, 0, 0, 1), 0)
    values = (
        x, y, A, m.Matrix((), ncols=3), m.rref(A), system,
        m.AffineSolutionSet((x, y), (1, 0), ((-1, 1),)),
        rf, root, m.RootRecord(root, 2), m.square_free(p * q),
        m.Inequality(rf, "ge", 0), m.Interval(None, root),
        m.IntervalSet((m.Interval(None, m.Rational(0)), m.Interval(m.Rational(1), None))),
        m.operations.expand((x + 1)**3, x),
        m.operations.poly_divmod(p, q), m.operations.poly_gcd(p, q),
        m.operations.poly_xgcd(p, q), m.operations.square_free(p * q),
        m.operations.real_roots(x**3 - 2, x), m.operations.compare_real(root, 1),
        m.operations.rational_function(x**2 - 1, x - 1, variable=x),
        m.operations.differentiate(rf, x), m.operations.antiderivative(x**2, x),
        m.operations.integrate(x**2, x, 0, 1), m.operations.evaluate(x + 1, {x: 2}),
        m.operations.substitute(x + 1, {x: y}), m.operations.simplify(x + x),
        m.operations.rref(A), m.operations.rank(A), m.operations.det(A),
        m.operations.inverse(A), m.operations.nullspace(A), m.solve_system(system),
        m.solve(m.Eq(x**3, 2), for_=x), m.solve(m.Inequality(rf, "ge", 0), for_=x),
        m.operations.differentiate(x**2, x, limits=m.ComputationLimits(max_work=1)),
    )
    ws = m.Workspace()
    refs = [ws.put(value) for value in values]
    ws.record("differentiate", {"source": refs[0]}, {"n": 1}, refs[22])
    payloads = {"objects.json": m.dumps(values), "workspace.json": m.dumps(ws)}
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest = {"baseline": BASELINE, "tag": "v0.2.0", "package_version": m.__version__,
                "generation": "git show of tagged src/mathforge in an isolated temporary source tree",
                "files": {}, "object_refs": [ref.identifier for ref in refs]}
    for name, payload in payloads.items():
        encoded = payload.encode("utf-8")
        (FIXTURES / name).write_bytes(encoded)
        manifest["files"][name] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    reader = Path(codec.__file__).read_bytes()
    (FIXTURES / "serialization_v2.py").write_bytes(reader)
    manifest["reader"] = {"file": "serialization_v2.py", "sha256": hashlib.sha256(reader).hexdigest()}
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def _generate_from_git():
    repo = Path(__file__).resolve().parents[1]
    actual = subprocess.check_output(["git", "rev-parse", "v0.2.0^{commit}"], cwd=repo, text=True).strip()
    if actual != BASELINE:
        raise RuntimeError("v0.2.0 tag does not match the recorded release commit")
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE, "src/mathforge"], cwd=repo, text=True
    ).splitlines()
    with tempfile.TemporaryDirectory(prefix="mathforge-v2-baseline-") as temporary:
        root = Path(temporary)
        for name in paths:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=repo))
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        env.pop("PYTHONHOME", None)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--write-baseline"],
                       cwd=root, env=env, check=True)


class V2CompatibilityTests(unittest.TestCase):
    def test_frozen_bytes_hashes_and_version(self):
        import mathforge as m
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["baseline"], BASELINE)
        self.assertEqual(manifest["package_version"], "0.2.0")
        for name, digest in manifest["files"].items():
            with self.subTest(name=name):
                payload = (FIXTURES / name).read_text(encoding="utf-8")
                self.assertEqual("sha256:" + hashlib.sha256(payload.encode()).hexdigest(), digest)
                restored = m.loads(payload)
                self.assertEqual(m.dumps(restored), payload)
                self.assertEqual(m.to_data(restored)["version"], 2)

    def test_workspace_references_and_results(self):
        import mathforge as m
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        values = m.loads((FIXTURES / "objects.json").read_text(encoding="utf-8"))
        ws = m.loads((FIXTURES / "workspace.json").read_text(encoding="utf-8"))
        for value, identity in zip(values, manifest["object_refs"], strict=True):
            self.assertEqual(ws.put(value).identifier, identity)
            if type(value) is m.Result and value.execution_status is m.ExecutionStatus.COMPLETED:
                self.assertTrue(m.verify(value).verified)

    def test_historical_v2_reader_and_current_writer(self):
        import mathforge as m
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        source = FIXTURES / manifest["reader"]["file"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest["reader"]["sha256"])
        spec = importlib.util.spec_from_file_location("mathforge._historical_v2_transport", source)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        for name in manifest["files"]:
            payload = (FIXTURES / name).read_text(encoding="utf-8")
            self.assertEqual(reader.dumps(reader.loads(m.dumps(m.loads(payload)))), payload)


if __name__ == "__main__":
    if "--generate-baseline" in sys.argv:
        _generate_from_git()
    elif "--write-baseline" in sys.argv:
        _write_baseline()
    else:
        unittest.main()
