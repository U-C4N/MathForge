"""Independent checkers cannot reach producer code, directly or through imports of trusted code."""

import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import mathforge

REPOSITORY = Path(__file__).resolve().parents[2]
LAYER_PACKAGES = ("trusted", "kernels", "certify", "areas")


def load_tool(name):
    path = REPOSITORY / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repository_package():
    """The source tree in a checkout; the installed package in the distribution harness (no src/)."""
    source = REPOSITORY / "src" / "mathforge"
    return source if source.is_dir() else Path(mathforge.__file__).resolve().parent


class LayeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool("check_layering")

    def tree(self, files):
        """Write {path below mathforge/: source} into a temporary tree and return the package directory."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        package = Path(directory.name) / "mathforge"
        for relative, source in files.items():
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return package

    def test_repository_has_no_layering_violations(self):
        self.assertEqual(self.tool.violations(repository_package()), [])

    def test_layer_packages_exist_and_are_documented(self):
        # A renamed or deleted layer package would make the check above pass vacuously.
        for name in LAYER_PACKAGES:
            with self.subTest(name):
                path = repository_package() / name / "__init__.py"
                self.assertTrue(path.is_file(), path)
                self.assertTrue(ast.get_docstring(ast.parse(path.read_bytes())), f"{name} states no layer rule")

    def test_detects_transitive_violation(self):
        package = self.tree({
            "certify/a.py": "from mathforge.trusted import b\n",
            "trusted/b.py": "from mathforge.kernels import c\n",
            "kernels/c.py": "",
        })
        self.assertEqual(self.tool.violations(package), [("mathforge.certify.a", "mathforge.kernels.c")])

    def test_relative_imports_are_resolved(self):
        package = self.tree({"certify/x.py": "from ..areas import y\n"})
        self.assertEqual(self.tool.violations(package), [("mathforge.certify.x", "mathforge.areas.y")])

    def test_relative_imports_in_a_package_initializer_start_at_that_package(self):
        package = self.tree({
            "certify/__init__.py": "from . import sibling\nfrom .. import areas\n",
            "certify/sibling.py": "",
        })
        self.assertEqual(self.tool.violations(package), [("mathforge.certify", "mathforge.areas")])

    def test_every_import_form_resolves_to_its_target(self):
        forms = {
            "import mathforge": "mathforge",
            "import mathforge.kernels.c": "mathforge.kernels.c",
            "import mathforge.kernels.c as k": "mathforge.kernels.c",
            "from mathforge import Rational": "mathforge.Rational",
            "from mathforge import kernels": "mathforge.kernels",
            "from mathforge.kernels import c": "mathforge.kernels.c",
            "from mathforge.kernels.c import value": "mathforge.kernels.c.value",
            "from mathforge.areas import *": "mathforge.areas",
            "from .. import areas": "mathforge.areas",
            "from ..kernels.c import value": "mathforge.kernels.c.value",
        }
        for statement, target in forms.items():
            with self.subTest(statement):
                package = self.tree({"certify/s.py": statement + "\n"})
                self.assertEqual(self.tool.violations(package), [("mathforge.certify.s", target)])

    def test_imports_nested_in_other_statements_are_scanned(self):
        nestings = {
            "function": "def late():\n    from mathforge.kernels import c\n",
            "try": "try:\n    from mathforge.kernels import c\nexcept ImportError:\n    c = None\n",
            "type checking": "if TYPE_CHECKING:\n    from mathforge.kernels import c\n",
        }
        for label, source in nestings.items():
            with self.subTest(label):
                package = self.tree({"certify/n.py": source})
                self.assertEqual(self.tool.violations(package), [("mathforge.certify.n", "mathforge.kernels.c")])

    def test_permitted_imports_are_not_violations(self):
        package = self.tree({
            "certify/ok.py": ("import math\n"
                              "from mathforge import errors\n"
                              "from mathforge.model import Rational\n"
                              "from mathforge.limits import computation\n"
                              "from mathforge.trusted import arithmetic\n"
                              "from . import sibling\n"
                              "from .sibling import helper\n"),
            "certify/sibling.py": "from mathforge.errors import InvalidInput\n",
            "trusted/arithmetic.py": "from mathforge.model import Rational\n",
            # The allowed leaves import only each other, as in the real tree.
            "errors.py": "",
            "limits.py": "from mathforge.errors import MathForgeError\n",
            "model.py": "from mathforge.errors import InvalidInput\nfrom mathforge.limits import current_budget\n",
        })
        self.assertEqual(self.tool.violations(package), [])

    def test_prefix_match_respects_package_boundaries(self):
        package = self.tree({"certify/p.py": "from mathforge import modelling\nimport mathforge.limits_extra\n"})
        self.assertEqual(self.tool.violations(package), [
            ("mathforge.certify.p", "mathforge.limits_extra"),
            ("mathforge.certify.p", "mathforge.modelling"),
        ])

    def test_allowed_leaf_modules_are_searched_too(self):
        leaves = {"errors": "InvalidInput", "limits": "computation", "model": "Rational"}
        for leaf, name in leaves.items():
            with self.subTest(leaf):
                package = self.tree({
                    "certify/a.py": f"from mathforge.{leaf} import {name}\n",
                    f"{leaf}.py": "from mathforge.kernels import c\n",
                    "kernels/c.py": "",
                })
                self.assertEqual(self.tool.violations(package), [("mathforge.certify.a", "mathforge.kernels.c")])

    def test_chains_between_allowed_modules_are_followed(self):
        package = self.tree({
            "certify/a.py": "from mathforge.model import Rational\n",
            "model.py": "from mathforge.limits import computation\n",
            "limits.py": "from mathforge.errors import InvalidInput\n",
            "errors.py": "from mathforge.kernels import c\n",
            "kernels/c.py": "",
        })
        pair = ("mathforge.certify.a", "mathforge.kernels.c")
        self.assertEqual(self.tool.violations(package), [pair])
        self.assertEqual(self.tool.find_violations(package)[pair], [
            "mathforge.certify.a", "mathforge.model", "mathforge.limits", "mathforge.errors", "mathforge.kernels.c"])

    def test_the_top_level_facade_is_not_searched(self):
        # Every import of mathforge.* runs mathforge/__init__.py, which imports everything; that implicit
        # path is not what the rule forbids, so only explicit imports of the facade's names are violations.
        package = self.tree({
            "__init__.py": "from mathforge.kernels import c\n",
            "certify/a.py": "from mathforge.model import Rational\n",
            "model.py": "from mathforge.errors import InvalidInput\n",
            "errors.py": "",
            "kernels/c.py": "",
        })
        self.assertEqual(self.tool.violations(package), [])

    def test_package_initializers_on_the_import_path_are_searched(self):
        package = self.tree({
            "certify/a.py": "from mathforge.trusted.b import value\n",
            "trusted/__init__.py": "from mathforge.kernels import c\n",
            "trusted/b.py": "value = 1\n",
            "kernels/c.py": "",
        })
        self.assertEqual(self.tool.violations(package), [("mathforge.certify.a", "mathforge.kernels.c")])

    def test_import_cycles_terminate_and_every_reaching_checker_is_reported(self):
        package = self.tree({
            "certify/a.py": "from mathforge.certify import b\n",
            "certify/b.py": "from mathforge.certify import a\nfrom mathforge.trusted import t\n",
            "trusted/t.py": "from mathforge.certify import a\nfrom mathforge.kernels import c\n",
            "kernels/c.py": "",
        })
        self.assertEqual(self.tool.violations(package), [
            ("mathforge.certify.a", "mathforge.kernels.c"),
            ("mathforge.certify.b", "mathforge.kernels.c"),
        ])

    def test_a_missing_package_directory_is_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            self.tool.violations(Path(directory) / "mathforge")

    def test_main_confirms_a_clean_tree(self):
        package = self.tree({"certify/a.py": "from mathforge.errors import InvalidInput\n"})
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.tool.main(package)
        self.assertEqual(output.getvalue().strip(), "No layering violations")

    def test_main_exits_with_one_and_shows_the_import_path(self):
        package = self.tree({
            "certify/a.py": "from mathforge.trusted import b\n",
            "trusted/b.py": "from mathforge.kernels import c\n",
        })
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
            self.tool.main(package)
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("mathforge.certify.a -> mathforge.trusted.b -> mathforge.kernels.c", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
