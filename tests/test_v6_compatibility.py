"""Historical working-tree payloads captured before any v0.7 source changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import mathforge as m

FIXTURES = Path(__file__).parent / "fixtures" / "v6"


class V6CompatibilityTests(unittest.TestCase):
    def test_bytes_references_and_checks(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(manifest["package_version"], "0.6.0")
        for name, digest in manifest["files"].items():
            data = (FIXTURES / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
            self.assertEqual(m.dumps(m.loads(data.decode())), data.decode())
        values = m.loads((FIXTURES / "objects.json").read_text())
        workspace = m.loads((FIXTURES / "workspace.json").read_text())
        for value, ref in zip(values, manifest["object_refs"], strict=True):
            self.assertEqual(workspace.put(value).identifier, ref)
            if type(value) is m.Result:
                self.assertTrue(m.verify(value).verified)

    def test_historical_reader(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        source = FIXTURES / "serialization_v4.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest["reader_sha256"])
        spec = importlib.util.spec_from_file_location("mathforge._v06_codec", source)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        for name in manifest["files"]:
            payload = (FIXTURES / name).read_text()
            self.assertEqual(reader.dumps(reader.loads(m.dumps(m.loads(payload)))), payload)
