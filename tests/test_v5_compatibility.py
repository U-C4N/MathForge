"""Frozen v0.5 working-tree bytes and identities, captured before v0.6 edits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import mathforge as m

FIXTURES = Path(__file__).parent / "fixtures" / "v5"


class V5CompatibilityTests(unittest.TestCase):
    def test_frozen_payloads_and_reverification(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["package_version"], "0.5.0")
        for name, digest in manifest["files"].items():
            data = (FIXTURES / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
            self.assertEqual(m.dumps(m.loads(data.decode())), data.decode())
        values = m.loads((FIXTURES / "objects.json").read_text(encoding="utf-8"))
        workspace = m.loads((FIXTURES / "workspace.json").read_text(encoding="utf-8"))
        for value, ref in zip(values, manifest["object_refs"], strict=True):
            self.assertEqual(workspace.put(value).identifier, ref)
            if type(value) is m.Result:
                self.assertTrue(m.verify(value).verified)

    def test_historical_reader(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        source = FIXTURES / "serialization_v3.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest["reader_sha256"])
        spec = importlib.util.spec_from_file_location("mathforge._v05_codec", source)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        for name in manifest["files"]:
            text = (FIXTURES / name).read_text(encoding="utf-8")
            self.assertEqual(reader.dumps(reader.loads(m.dumps(m.loads(text)))), text)


if __name__ == "__main__":
    unittest.main()
