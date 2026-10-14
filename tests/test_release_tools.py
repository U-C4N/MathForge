"""Release checks fail closed without adding dependencies to the exact core."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


def load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseToolTests(unittest.TestCase):
    def test_rc_version_is_release_independent(self):
        tool = load_tool("check_release_candidate")
        self.assertEqual(tool.candidate_version("0.3.0", 1), "0.3.0rc1")
        self.assertEqual(tool.candidate_version("12.4.2", 3), "12.4.2rc3")
        for target in ("0.3", "0.3.0rc1", "0.3.0.dev1", "01.3.0"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                tool.candidate_version(target, 1)
        for number in (0, -1, True, 1.5):
            with self.subTest(number=number), self.assertRaises(ValueError):
                tool.candidate_version("0.3.0", number)

    def test_fingerprint_normalizes_only_package_version(self):
        tool = load_tool("check_release_candidate")
        first = [("mathforge/__init__.py", b'__version__ = "0.3.0"\r\n'),
                 ("mathforge/mcp/__init__.py", b"def main(): pass\n")]
        candidate = [("mathforge/__init__.py", b'__version__ = "0.3.0rc2"\n'), first[1]]
        self.assertEqual(tool.fingerprint(first), tool.fingerprint(candidate))
        self.assertNotEqual(tool.fingerprint(first), tool.fingerprint(candidate[:1]))
        self.assertTrue(tool.core_member("mathforge/mcp/__init__.py"))

    def test_benchmark_rejects_silent_errors_missing_cases_and_growth(self):
        tool = load_tool("check_benchmarks")
        self.assertEqual(len(tool.THRESHOLDS), 16)
        for item in tool.BASELINE["measurements"]:
            self.assertEqual(tool.THRESHOLDS[item["case"]],
                             (item["work"] * 110 // 100, item["evidence_items"] * 110 // 100))
        report = {"measurements": [
            {"case": name, "execution_status": "error" if name.startswith("intentional_") else "completed",
             "error_code": "resource_limit_exceeded" if name.startswith("intentional_") else None,
             "work": ceiling[0], "evidence_items": ceiling[1]}
            for name, ceiling in tool.THRESHOLDS.items()]}
        tool.check_report(report)
        for mutation in ("error", "work", "evidence", "missing", "duplicate"):
            bad = deepcopy(report)
            if mutation == "error":
                bad["measurements"][0]["execution_status"] = "error"
            elif mutation == "work":
                bad["measurements"][0]["work"] += 1
            elif mutation == "evidence":
                bad["measurements"][0]["evidence_items"] += 1
            elif mutation == "missing":
                bad["measurements"].pop()
            else:
                bad["measurements"][-1] = bad["measurements"][0]
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                tool.check_report(bad)

    def test_candidate_metadata_normalization_preserves_dependency_checks(self):
        tool = load_tool("check_release_candidate")
        with tempfile.TemporaryDirectory() as temporary:
            hashes = []
            for index, (version, dependency) in enumerate((("0.3.0rc1", "mcp>=2.3"),
                                                           ("0.3.0", "mcp>=2.3"),
                                                           ("0.3.0", "mcp>=2.4"))):
                wheel = Path(temporary) / f"sample-{index}.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr(f"mathforge-{version}.dist-info/METADATA",
                                     f"Name: mathforge\nVersion: {version}\nRequires-Dist: {dependency}\n")
                    archive.writestr(f"mathforge-{version}.dist-info/entry_points.txt",
                                     "[console_scripts]\nmathforge-mcp = mathforge.mcp:main\n")
                hashes.append(tool.wheel_metadata_fingerprint(wheel))
            self.assertEqual(hashes[0], hashes[1])
            self.assertNotEqual(hashes[1], hashes[2])

    def test_document_fences_preserve_source_and_reject_unclosed(self):
        tool = load_tool("check_docs")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "guide.md"
            path.write_text("# Guide\n\n```python\nassert 1 == 1\n```\n", encoding="utf-8")
            self.assertEqual(list(tool.python_fences(path)), [(4, "assert 1 == 1\n")])
            path.write_text("```python\nassert False\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(tool.python_fences(path))

    @unittest.skipUnless(importlib.util.find_spec("packaging"), "development-only packaging checker")
    def test_extra_markers_do_not_create_default_dependencies(self):
        tool = load_tool("check_distribution")
        extra = ['mcp<3,>=2.3; extra == "mcp"']
        self.assertEqual(tool.active_dependencies(extra), [])
        self.assertEqual(tool.active_dependencies(extra, extra="mcp"), extra)
        self.assertEqual(tool.active_dependencies(["unexpected>=1"]), ["unexpected>=1"])
        self.assertEqual(tool.active_dependencies(['unexpected; python_version >= "3.11"']),
                         ['unexpected; python_version >= "3.11"'])


if __name__ == "__main__":
    unittest.main()
