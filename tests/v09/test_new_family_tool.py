"""tools/new_family.py, which starts, registers and pins a v6 family, and the helpers of tests/v09/support.py.

The end-to-end test copies the package, the tools, the files that pins reads and the shared helpers into a
temporary directory. There it scaffolds the doubling family, checks that the scaffolded tests fail with "TODO",
fills the family in, runs its tests before and after registering it, runs its operation, compares pins before and
after, and lets the registration of a family whose checker imports producer code fail. Every program runs in a
fresh interpreter that imports the copy; the repository itself is never written, and its generated files, schemas
and area packages hash the same before and after.
"""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import mathforge
from mathforge import registry
from mathforge.registry_v07 import LEGACY_CATALOG

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL = REPOSITORY / "tools" / "new_family.py"
# The tested package's root: src/ in a checkout, site-packages in the distribution harness (no src/).
SRC = Path(mathforge.__file__).resolve().parents[1]
AREA, FAMILY, OPERATION = "ntheory", "doubling", "double_integer"
MODULE = f"mathforge.areas.{AREA}.{FAMILY}"
# A family whose checker imports producer code, so that tools/check_layering.py fails its registration. Its area
# has had no family so far, so scaffold writes the three package initializers as well.
BROKEN_AREA, BROKEN_FAMILY, BROKEN_OPERATION = "special", "breach", "breach_layering"
BROKEN_MODULE = f"mathforge.areas.{BROKEN_AREA}.{BROKEN_FAMILY}"

# Runs the tests of a module in a fresh interpreter that imports the copy below argv[1], and prints the outcome as
# JSON, with whether FAMILIES of the area package argv[3] lists the family argv[4].
RUNNER = """\
import importlib, json, sys, unittest
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "src")]
import mathforge
assert Path(mathforge.__file__).resolve().is_relative_to(root.resolve()), mathforge.__file__
outcome = unittest.TestResult()
unittest.defaultTestLoader.loadTestsFromName(sys.argv[2]).run(outcome)
print(json.dumps({"run": outcome.testsRun, "skipped": len(outcome.skipped),
                  "failures": [[test.id(), text] for test, text in outcome.failures],
                  "errors": [[test.id(), text] for test, text in outcome.errors],
                  "declared": sys.argv[4] in importlib.import_module(sys.argv[3]).FAMILIES}))
"""

# Runs the registered operation through the generated functions and the MCP tools of the copy below argv[1].
PROBE = """\
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import mathforge as mf
from mathforge.mcp.boundary import call_tool
result = mf.operations.double_integer(21, times=3)
reply = call_tool("mathforge.ntheory", {"operation": "double_integer", "arguments": {"n": 21, "times": 3}})
listing = call_tool("mathforge.catalog", {"query": "double_integer"}).payload["operations"]
print(json.dumps({
    "package": str(Path(mf.__file__).resolve().parent),
    "values": [mf.ntheory.double_integer(21, 3).value, mf.ntheory.double_integer(21).value],
    "result": [result.value.value, result.steps[0].rule, result.request.version],
    "verified": [mf.verify(result).verified, mf.verify(mf.loads(mf.dumps(result))).verified],
    "mcp": [reply.is_error, reply.payload["text"]],
    "catalog": [[entry["name"], entry["tool"]] for entry in listing],
}))
"""

# The doubling family, written into the templates that scaffold produced: each pattern matches one block there.
PRODUCER_BLOCKS = (
    (r"^from \.\.\.errors import UnsupportedOperation\n", """\
from dataclasses import dataclass

from ...errors import InvalidInput
from ...limits import computation
"""),
    (r"^SPECS = \(\n.*?^\)\n", """\
SPECS = (
    OperationSpec("double_integer", "ntheory",
                  (Argument("n", "int", None, "int"), Argument("times", "int", "1", "int")),
                  "DoubledInteger", "Double an exact integer a number of times.",
                  "exact integers; doubled a nonnegative number of times", version=6),
)
"""),
    (r"^TYPES: dict\[str, type\] = \{\}\n", '''\
@dataclass(frozen=True)
class DoubledInteger:
    """n * 2**times, the value of double_integer(n, times)."""
    value: int

    def __post_init__(self):
        with computation() as budget:
            if type(self.value) is not int:  # Neither bool nor float is an exact integer.
                raise InvalidInput("A doubled integer is an exact integer")
            budget.check_int(self.value)


TYPES = {"doubled_integer": DoubledInteger}
'''),
    (r"^def produce\(operation, args\):\n.*\Z", '''\
def produce(operation, args):
    """Doubles n once for each unit of times; the checker needs no evidence."""
    n, times = args["n"], args["times"]  # run() admits exact integers only, for both.
    if times < 0:
        raise InvalidInput("double_integer doubles a nonnegative number of times")
    with computation() as budget:
        for _ in range(times):
            budget.tick()
            n += n
            budget.check_int(n)
    return DoubledInteger(n), ()
'''),
)
CHECKER_BLOCKS = (
    (r"^def check_operation\(operation, args, value, evidence\):\n.*\Z", '''\
from mathforge.limits import computation

RECORD = ("mathforge.areas.ntheory.doubling", "DoubledInteger")  # Module and name of the value's class.


def check_operation(operation, args, value, evidence):
    """Halves the claimed value times times where the producer doubled n: every halving must be exact and the
    quotient must be n, which holds exactly when the value is n * 2**times."""
    n, times = args.get("n"), args.get("times")
    if (operation != "double_integer" or type(n) is not int or type(times) is not int or times < 0
            or type(evidence) is not tuple or evidence
            or (type(value).__module__, type(value).__qualname__) != RECORD):
        return False
    claimed = getattr(value, "value", None)  # A record forged with object.__new__ may lack the field.
    if type(claimed) is not int:
        return False
    if claimed == 0:
        return n == 0
    with computation() as budget:
        budget.check_int(claimed)  # Bounds the cost of each halving.
        for _ in range(times):
            budget.tick()
            if claimed % 2:
                return False
            claimed //= 2
    return claimed == n
'''),
)
BROKEN_PRODUCER_BLOCKS = (
    (r"^SPECS = \(\n.*?^\)\n", """\
SPECS = (
    OperationSpec("breach_layering", "special", (Argument("n", "int", None, "int"),), "int",
                  "Refuse every input.", "tests only", version=6),
)
"""),
)
BROKEN_CHECKER_BLOCKS = (
    (r"^def check_operation\(operation, args, value, evidence\):\n.*\Z", """\
def check_operation(operation, args, value, evidence):
    from mathforge.operations import run  # Producer code, which tools/check_layering.py refuses.
    return False
"""),
)

# The tests of the doubling family, written with tests/v09/support.py. They replace the scaffolded ones and run
# both before and after the registration.
DEMO_TESTS = r'''"""The doubling family of tests/v09/test_new_family_tool.py, tested through tests/v09/support.py."""

import importlib
import random
from unittest.mock import patch

import mathforge as mf
from mathforge import registry
from mathforge.areas.ntheory import doubling as family
from mathforge.contracts import CheckStatus

from .. import support

Doubled = family.DoubledInteger


class DoublingTests(support.FamilyTestCase):
    FAMILY = family

    def test_generated_functions_come_with_the_registration(self):
        declared = family.__name__ in importlib.import_module("mathforge.areas.ntheory").FAMILIES
        self.assertIn(family.__name__, registry.v6_families())
        self.assertEqual(hasattr(mf.ntheory, "double_integer"), declared)
        self.assertEqual(hasattr(mf.operations, "double_integer"), declared)

    def test_known_values(self):
        for arguments, doubled in (({"n": 3}, 6), ({"n": -7, "times": 0}, -7), ({"n": 5, "times": 4}, 80),
                                   ({"n": 0, "times": 9}, 0), ({"n": 2 ** 70, "times": 2}, 2 ** 72)):
            with self.subTest(arguments=arguments):
                result = support.run("double_integer", **arguments)
                self.assertEqual(result.value, Doubled(doubled))
                self.assertEqual([check.claim for check in result.verification], list(support.CLAIMS))
                self.assertTrue(mf.verify(result).verified)

    def test_invalid_inputs(self):
        for bad in (0.5, True, "1", None, (1,), [1]):
            with self.subTest(bad=bad):
                self.assertInvalidInput(support.run("double_integer", n=bad))
                self.assertInvalidInput(support.run("double_integer", n=1, times=bad))
        self.assertInvalidInput(support.run("double_integer", n=1, times=-1))
        error = self.assertToolError(support.call("double_integer", {"n": 0.5}), "invalid_input")
        self.assertEqual(error["field"], "arguments.n")

    def test_resource_limits(self):
        self.assertResourceLimit(support.run("double_integer", n=support.HUGE))
        self.assertResourceLimit(support.run("double_integer", n=1, times=65536))
        self.assertResourceLimit(support.run("double_integer", n=1, times=3, limits=mf.ComputationLimits(max_work=1)))
        self.assertEqual(support.oversized_matrix().shape, (65, 65))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        for n, times in ((5, 3), (-5, 1), (0, 1000), (10 ** 30, 0)):
            with self.subTest(n=n, times=times):
                self.assertAccepts("double_integer", {"n": n, "times": times}, Doubled(n * 2 ** times))

    def test_checker_rejects_corrupted_claims(self):
        args, good = {"n": 3, "times": 1}, Doubled(6)
        self.assertAccepts("double_integer", args, good)
        cases = (
            ("wrong value", args, Doubled(8), ()),
            ("value that halves unevenly", {"n": 3, "times": 2}, Doubled(13), ()),
            ("impostor class", args, support.impostor(good), ()),
            ("forged float field", args, support.forged(Doubled, value=6.0), ()),
            ("forged without its field", args, support.forged(Doubled), ()),
            ("plain int", args, 6, ()),
            ("bool argument", {"n": True, "times": 1}, Doubled(2), ()),
            ("float argument", {"n": 3.0, "times": 1}, good, ()),
            ("str argument", {"n": "3", "times": 1}, good, ()),
            ("missing argument", {"n": 3}, good, ()),
            ("negative times", {"n": 3, "times": -1}, good, ()),
            ("extra evidence", args, good, (1,)),
            ("list evidence", args, good, []),
            ("oversized evidence", args, good, (0,) * 200_000),
        )
        for label, claimed, value, evidence in cases:
            with self.subTest(label):
                self.assertRejects("double_integer", claimed, value, evidence)
        self.assertRejects("triple_integer", args, good)

    def test_verification_never_runs_the_producer(self):
        result = support.run("double_integer", n=21, times=2)
        decoded = mf.loads(mf.dumps(result))
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            self.assertTrue(mf.verify(result).verified)
            self.assertTrue(mf.verify(decoded).verified)
            self.assertEqual(support.run("double_integer", n=21).error.code, "internal_error")
        statuses = support.statuses(mf.verify(support.tampered(result, Doubled(85))))
        self.assertIs(statuses["result.mathematics"], CheckStatus.FAILED)

    def test_json_and_rendering(self):
        result = support.run("double_integer", n=21)
        self.assertEqual(self.roundtrip(result).value, Doubled(42))
        self.assertEqual(mf.render_text(result.value), "DoubledInteger(value=42)")
        self.assertEqual(mf.to_latex(result.value), r"\text{DoubledInteger}\left(\text{value}=42\right)")

    def test_value_function_equals_the_structured_path(self):
        rng = random.Random(20261112)
        for _ in range(25):
            n, times = rng.randrange(-10 ** 30, 10 ** 30), rng.randrange(80)
            with self.subTest(n=n, times=times):
                expected = support.run("double_integer", n=n, times=times).value
                self.assertEqual(support.lean_value("double_integer", n=n, times=times), expected)
                self.assertEqual(support.value("double_integer", n=n, times=times), expected)
        self.assertEqual(support.value("double_integer", n=4), Doubled(8))  # times defaults to 1.

    def test_mcp_area_tool(self):
        expected = support.run("double_integer", n=21, times=3)
        self.assertEqual(self.answer(support.call("double_integer", {"n": 21, "times": 3})), expected)
        self.assertEqual(self.answer(support.call_values("double_integer", n=21, times=3)), expected)
        self.assertEqual(self.answer(support.call("double_integer", {"n": 21})), support.run("double_integer", n=21))
'''

ITEM = re.compile(r"^  (?P<item>\S.*?)  \[(?P<status>[^\]]+)\]$")  # "  <test>: <label>  [<status>]"


def load_tool():
    """tools/new_family.py as a module (it changes nothing when imported)."""
    spec = importlib.util.spec_from_file_location("new_family_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves the string annotations of Pin through it.
    spec.loader.exec_module(module)
    return module


def command(*arguments):
    """python with arguments, without PYTHONPATH: the tool finds the package below its root itself, and register
    gives the generators the root's src. The output is decoded as UTF-8, which -X utf8 makes the child write."""
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-X", "utf8", *map(str, arguments)], env=environment,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)


def output(process):
    return process.stdout + process.stderr


def require(process, label):
    """Stop the set-up with the output of a step that failed, rather than with a symptom further on."""
    if process.returncode:
        raise AssertionError(f"{label} failed with status {process.returncode}:\n{output(process)}")


def run_tests(root, module):
    """The outcome of the tests of a module in a fresh interpreter that imports the copy below root."""
    process = command("-I", "-c", RUNNER, root, module, f"mathforge.areas.{AREA}", MODULE)
    require(process, f"Running {module}")
    return json.loads(process.stdout.splitlines()[-1])


def replace_blocks(path, blocks):
    """Replace one block of a scaffolded file per (pattern, replacement) pair."""
    text = path.read_text(encoding="utf-8")
    for pattern, replacement in blocks:
        text, count = re.subn(pattern, lambda match: replacement, text, flags=re.MULTILINE | re.DOTALL)
        if count != 1:
            raise AssertionError(f"{path} holds {count} blocks that match {pattern!r}, not one")
    path.write_text(text, encoding="utf-8", newline="\n")


def digests(paths, base):
    """The SHA-256 of the bytes of each file by its path relative to base."""
    return {path.relative_to(base).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def generated_files(package):
    """The files below a mathforge package that the generators write: the generated API and the JSON schemas."""
    return [*sorted((package / "_generated").glob("*.py")), *(package / f"{area}.py" for area in registry.AREAS),
            *sorted((package / "schemas").glob("*.json"))]


def generated_texts(package):
    """The text of each file that the generators write below a mathforge package, by its path there. Text rather
    than bytes: a checkout may hold the files with CRLF line ends, which the generators count as current, and a
    file that they rewrite gets LF line ends."""
    return {path.relative_to(package).as_posix(): path.read_text(encoding="utf-8")
            for path in generated_files(package)}


def differing(before, after):
    """The paths whose texts differ between two snapshots of generated_texts."""
    return {path for path in {*before, *after} if before.get(path) != after.get(path)}


def real_path(relative):
    """A path that scaffold names below a root, in the tested checkout (src/ maps to the package's root)."""
    return SRC / relative[len("src/"):] if relative.startswith("src/") else REPOSITORY / relative


def value_of(text):
    """The value of a literal that pins prints, a name such as PILOT standing for the string "<PILOT>"."""
    class Names(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.Constant("<" + node.id + ">")

    return ast.literal_eval(Names().visit(ast.parse(text, mode="eval")))


def elements(text):
    """The items of a pinned tuple or list, as value_of gives them; for a passage that is an assertEqual call, the
    source texts of the items of the tuple that it compares with."""
    statement = ast.parse(text).body[0]
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return [ast.unparse(item) for item in statement.value.args[1].elts]
    return list(value_of(text))


def listing(text):
    """A pins listing as ({(file, item): (status, value text)}, {(section, label): rest}): the items are the pinned
    constants, the other entries the lines "  label: rest" of the schema sizes and of what operations lack."""
    items, entries, section, key, entry = {}, {}, None, None, None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            section, key, entry = line, None, None
        elif line.startswith("      "):  # A line of a value, or the continuation of an entry.
            if key is not None:
                items[key][1].append(line[6:])
            else:
                entries[entry] += " " + line.strip()
        elif ITEM.match(line):
            match = ITEM.match(line)
            key, entry = (section, match["item"]), None
            items[key] = (match["status"], [])
        else:
            label, _, rest = line.strip().partition(": ")
            key, entry = None, (section, label)
            entries[entry] = rest
    return {key: (status, "\n".join(lines)) for key, (status, lines) in items.items()}, entries


class NewFamilyToolTests(unittest.TestCase):
    """scaffold, register and pins on a copy of the tree, in the order in which a family's author runs them."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tool = tool = load_tool()
        directory = tempfile.TemporaryDirectory(prefix="mathforge-new-family-", ignore_cleanup_errors=True)
        cls.addClassCleanup(directory.cleanup)
        cls.root = root = Path(directory.name).resolve()
        cls.files = files = tool.templates(AREA, FAMILY, [OPERATION])
        cls.broken_files = broken_files = tool.templates(BROKEN_AREA, BROKEN_FAMILY, [BROKEN_OPERATION])
        cls.real_before = cls.real_state()

        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(SRC / "mathforge", root / "src" / "mathforge", ignore=ignore)
        shutil.copytree(REPOSITORY / "tools", root / "tools", ignore=ignore)
        for relative in (*tool.PINNED_FILES, "tests/v09/__init__.py", tool.SUPPORT):
            if not (root / relative).exists():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPOSITORY / relative, root / relative)
        # A regular package: found before a tests package that some other distribution may have installed.
        (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
        package, copied = root / "src" / "mathforge", root / "tools" / "new_family.py"
        initializer = root / f"src/mathforge/areas/{AREA}/__init__.py"
        own = [path for path, (_, written) in files.items() if written]
        cls.existed = {path: (root / path).exists() for path in files}
        cls.generated_before = generated_texts(package)
        cls.initializer_before = initializer.read_text(encoding="utf-8")

        cls.pins_before = command(TOOL, "--root", root, "pins")
        cls.scaffolded = command(copied, "scaffold", AREA, FAMILY, "--ops", OPERATION, "--root", root)
        require(cls.scaffolded, "scaffold")
        cls.scaffolded_texts = {path: (root / path).read_text(encoding="utf-8") for path in files}
        cls.skeleton = run_tests(root, f"tests.v09.{AREA}.test_{FAMILY}")
        cls.unfinished = command(copied, "register", AREA, FAMILY, "--root", root)
        cls.initializer_unfinished = initializer.read_text(encoding="utf-8")

        replace_blocks(root / own[0], PRODUCER_BLOCKS)
        replace_blocks(root / own[1], CHECKER_BLOCKS)
        (root / own[2]).write_text(DEMO_TESTS, encoding="utf-8", newline="\n")
        cls.filled_texts = {path: (root / path).read_text(encoding="utf-8") for path in own}
        cls.unregistered = run_tests(root, f"tests.v09.{AREA}.test_{FAMILY}")

        cls.registered = command(copied, "register", AREA, FAMILY, "--root", root)
        require(cls.registered, "register")
        cls.initializer_registered = initializer.read_text(encoding="utf-8")
        cls.generated_registered = generated_texts(package)
        cls.registered_tests = run_tests(root, f"tests.v09.{AREA}.test_{FAMILY}")
        cls.probe = command("-I", "-c", PROBE, root / "src")
        cls.pins_after = command(TOOL, "pins", "--root", root)
        cls.pins_default = command(copied, "pins")
        cls.pins_check = command(copied, "pins", "--check")
        cls.again = command(copied, "register", AREA, FAMILY)
        cls.initializer_again = initializer.read_text(encoding="utf-8")
        cls.generated_again = generated_texts(package)
        cls.collision = command(TOOL, "scaffold", AREA, "colliding", "--ops", "integer_gcd,sum,_hidden", "--root", root)
        cls.overwrite = command(TOOL, "scaffold", AREA, FAMILY, "--ops", "triple_integer", "--root", root)
        cls.after_overwrite = {path: (root / path).read_text(encoding="utf-8") for path in own}

        cls.broken_existed = {path: (root / path).exists() for path in broken_files}
        cls.broken_scaffolded = command(copied, "scaffold", BROKEN_AREA, BROKEN_FAMILY, "--ops", BROKEN_OPERATION)
        require(cls.broken_scaffolded, "scaffold of the broken family")
        broken_own = [path for path, (_, written) in broken_files.items() if written]
        replace_blocks(root / broken_own[0], BROKEN_PRODUCER_BLOCKS)
        replace_blocks(root / broken_own[1], BROKEN_CHECKER_BLOCKS)
        broken_initializer = root / f"src/mathforge/areas/{BROKEN_AREA}/__init__.py"
        cls.broken_initializer_before = broken_initializer.read_text(encoding="utf-8")
        cls.broken = command(copied, "register", BROKEN_AREA, BROKEN_FAMILY)
        cls.broken_initializer_after = broken_initializer.read_text(encoding="utf-8")
        cls.generated_broken = generated_texts(package)
        # The undo that register prints: the family leaves FAMILIES, and the two generators run again.
        broken_initializer.write_text(cls.broken_initializer_before, encoding="utf-8", newline="\n")
        cls.undo = [command(root / "tools" / "generate_api.py"), command(root / "tools" / "generate_schema.py")]
        cls.generated_undone = generated_texts(package)
        cls.real_after = cls.real_state()

    @classmethod
    def real_state(cls):
        """What the tool would have changed in the tested checkout: the digests of the generated files, schemas
        and area packages, and whether the own files of the two test families exist there."""
        package = SRC / "mathforge"
        state = digests([*generated_files(package), *sorted((package / "areas").rglob("*.py"))], package)
        for path, (_, written) in (*cls.files.items(), *cls.broken_files.items()):
            if written:
                state[path] = real_path(path).exists()
        return state

    def test_scaffold_writes_the_templates_and_keeps_existing_packages(self):
        self.assertEqual(self.scaffolded.returncode, 0, output(self.scaffolded))
        for path, (text, written) in self.files.items():
            with self.subTest(path=path):
                verb = "kept    " if self.existed[path] else "created "
                self.assertIn(f"{verb} {path}\n", self.scaffolded.stdout)
                if written or not self.existed[path]:
                    self.assertFalse(self.existed[path])
                    self.assertEqual(self.scaffolded_texts[path], text)
        self.assertEqual(self.scaffolded_texts[f"src/mathforge/areas/{AREA}/__init__.py"], self.initializer_before)
        self.assertIn("The family is not registered.", self.scaffolded.stdout)
        self.assertEqual(self.broken_scaffolded.returncode, 0, output(self.broken_scaffolded))
        for path, (_, written) in self.broken_files.items():  # An area without packages so far gets all three.
            with self.subTest(path=path):
                self.assertFalse(written and self.broken_existed[path])
                self.assertIn(("kept     " if self.broken_existed[path] else "created  ") + path + "\n",
                              self.broken_scaffolded.stdout)

    def test_the_scaffolded_tests_fail_with_todo(self):
        tree = ast.parse(self.scaffolded_texts[f"tests/v09/{AREA}/test_{FAMILY}.py"])
        methods = [node.name for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
        self.assertEqual(len(methods), 9)
        self.assertEqual((self.skeleton["run"], self.skeleton["errors"]), (len(methods), []))
        self.assertEqual(len(self.skeleton["failures"]), len(methods))
        for test, text in self.skeleton["failures"]:
            with self.subTest(test=test):
                self.assertTrue(text.strip().splitlines()[-1].startswith("AssertionError: TODO"), text)
        self.assertFalse(self.skeleton["declared"])

    def test_register_refuses_a_family_whose_specs_hold_todo(self):
        self.assertEqual(self.unfinished.returncode, 1, output(self.unfinished))
        self.assertIn("Nothing was registered", self.unfinished.stderr)
        self.assertIn("TODO", self.unfinished.stderr)
        self.assertEqual(self.initializer_unfinished, self.initializer_before)

    def test_the_family_tests_pass_before_and_after_the_registration(self):
        for label, outcome, declared in (("before", self.unregistered, False),
                                         ("after", self.registered_tests, True)):
            with self.subTest(label):
                self.assertEqual((outcome["failures"], outcome["errors"], outcome["skipped"]), ([], [], 0))
                self.assertEqual(outcome["run"], DEMO_TESTS.count("\n    def test_"))
                self.assertIs(outcome["declared"], declared)

    def test_register_appends_the_family_and_regenerates_the_copy(self):
        self.assertEqual(self.registered.returncode, 0, output(self.registered))
        self.assertEqual(self.initializer_registered, self.tool.with_family(self.initializer_before, MODULE))
        families = ast.literal_eval(self.tool._assigned(ast.parse(self.initializer_registered), "FAMILIES"))
        before = ast.literal_eval(self.tool._assigned(ast.parse(self.initializer_before), "FAMILIES"))
        self.assertEqual(families, (*before, MODULE))
        echoed = self.registered.stdout
        for line in (f"Appended {MODULE} to FAMILIES of src/mathforge/areas/{AREA}/__init__.py.",
                     "$ python tools/generate_api.py", "mathforge/_generated/structured.py",
                     "$ python tools/generate_schema.py", "mathforge-v6.schema.json",
                     "$ python tools/check_layering.py", "No layering violations", f"Registered {MODULE}."):
            self.assertIn(line, echoed)
        changed = differing(self.generated_before, self.generated_registered)
        self.assertEqual(changed, {"_generated/structured.py", "_generated/values.py", f"{AREA}.py",
                                   "schemas/mathforge-v6.schema.json"})
        for path in changed:
            self.assertIn(OPERATION, self.generated_registered[path], path)
        self.assertIn("DoubledInteger", self.generated_registered[f"{AREA}.py"])

    def test_the_operation_runs_and_verifies_from_the_copy(self):
        self.assertEqual(self.probe.returncode, 0, output(self.probe))
        found = json.loads(self.probe.stdout.splitlines()[-1])
        self.assertTrue(Path(found["package"]).is_relative_to(self.root), found["package"])
        self.assertEqual(found["values"], [168, 42])
        self.assertEqual(found["result"], [168, "v6.double_integer", 6])
        self.assertEqual(found["verified"], [True, True])
        self.assertEqual(found["mcp"], [False, "completed / value [exact; complete]: DoubledInteger(value=168)"])
        self.assertEqual(found["catalog"], [[OPERATION, "mathforge.ntheory"]])

    def test_pins_report_the_values_that_the_family_changes(self):
        tool = self.tool
        for process in (self.pins_before, self.pins_after, self.pins_default):
            self.assertEqual(process.returncode, 0, output(process))
        self.assertEqual(self.pins_check.returncode, 1, output(self.pins_check))
        self.assertIn(str(self.root), self.pins_after.stdout.splitlines()[0])
        self.assertEqual(self.pins_default.stdout, self.pins_after.stdout)  # The default root holds the tool.
        before, before_entries = listing(self.pins_before.stdout)
        after, after_entries = listing(self.pins_after.stdout)
        self.assertEqual(set(after), set(before))
        self.assertEqual(set(after_entries), set(before_entries))
        values = {
            (tool.REGISTRY_TESTS, 'test_area_assignment: names("ntheory")'): lambda names: names | {OPERATION},
            (tool.REGISTRY_TESTS, "test_area_assignment: {area: size for area, size in sizes.items() if size}"):
                lambda sizes: dict(sizes, ntheory=sizes["ntheory"] + 1),
            (tool.DUAL_TESTS, "test_families_come_from_area_packages: "
                              'importlib.import_module("mathforge.areas.ntheory").FAMILIES'):
                lambda names: (*names, MODULE),
            (tool.MCP_TESTS, "test_catalog_filters_sorts_and_limits: defaults"):
                lambda defaults: {**defaults, (OPERATION, "times"): 1},
        }
        passages = {
            (tool.DUAL_TESTS, "test_capabilities_list_each_name_once: the entries after the frozen ones, with the "
                              "demo's"):
                lambda text: re.sub(r"\+ (\d+)\)$", lambda match: f"+ {int(match[1]) + 1})", text),
        }
        # Where the registry's order puts the family: after the families of the area packages before its own, so
        # the module and the operation join these lists once each, and the demo's entries stay last.
        joined = {
            (tool.PILOT_TESTS, "test_registry_entry: registry.v6_families()"): (MODULE, None),
            (tool.DUAL_TESTS, "test_families_come_from_area_packages: registry.v6_families()"): (MODULE, None),
            (tool.DUAL_TESTS, "test_families_come_from_area_packages: the families with the demo's"):
                (repr(MODULE), "demo.__name__"),
            (tool.DUAL_TESTS, 'test_capabilities_list_each_name_once: names[len(frozen["operations"]):]'):
                (OPERATION, "demo_variable"),
        }
        self.assertLessEqual({*values, *passages, *joined}, set(before))
        for key, (status, text) in before.items():
            with self.subTest(pin=key):
                if key in values:
                    self.assertEqual(after[key][0], "STALE")
                    self.assertEqual(value_of(after[key][1]), values[key](value_of(text)))
                elif key in passages:
                    self.assertEqual(after[key], ("STALE", passages[key](text)))
                elif key in joined:
                    item, last = joined[key]
                    self.assertEqual(after[key][0], "STALE")
                    grown, kept = elements(after[key][1]), elements(text)
                    self.assertEqual(grown.count(item), 1)
                    self.assertEqual([element for element in grown if element != item], kept)
                    if last is not None:
                        self.assertEqual(grown[-1], last)
                else:
                    self.assertEqual(after[key], (status, text))
        for (section, label), rest in after_entries.items():
            with self.subTest(entry=label):
                if section == "Operations that still lack":
                    self.assertNotIn(OPERATION, before_entries[section, label].split(", "))
                    self.assertIn(OPERATION, rest.split(", "))
                elif label == "mathforge.ntheory" or label.startswith("all "):  # The schema sizes that grow.
                    self.assertGreater(int(rest.split()[0]), int(before_entries[section, label].split()[0]))
                else:
                    self.assertEqual(rest, before_entries[section, label])
        operations = [int(re.search(r"(\d+) registry operations", process.stdout)[1])
                      for process in (self.pins_before, self.pins_after)]
        self.assertEqual(operations[1], operations[0] + 1)
        self.assertTrue(self.pins_after.stdout.rstrip().splitlines()[-1].startswith("Not current: "))

    def test_register_again_leaves_families_alone(self):
        self.assertEqual(self.again.returncode, 0, output(self.again))
        self.assertIn(f"already lists {MODULE}", self.again.stdout)
        self.assertEqual(self.initializer_again, self.initializer_registered)
        self.assertEqual(differing(self.generated_registered, self.generated_again), set())

    def test_scaffold_refuses_names_in_use_and_never_overwrites(self):
        self.assertEqual(self.collision.returncode, 1, output(self.collision))
        refusal = self.collision.stderr
        self.assertIn("Nothing was written:", refusal)
        for line in ("'integer_gcd' is a registered operation", "'sum' is a Python builtin",
                     "'_hidden' starts with an underscore"):
            self.assertIn(line, refusal)
        for path, (_, written) in self.tool.templates(AREA, "colliding", ["triple_integer"]).items():
            if written:
                self.assertFalse((self.root / path).exists(), path)
        self.assertEqual(self.overwrite.returncode, 1, output(self.overwrite))
        self.assertIn("scaffold never overwrites a file", self.overwrite.stderr)
        for path in self.filled_texts:
            self.assertIn(f"  {path}", self.overwrite.stderr)
        self.assertEqual(self.after_overwrite, self.filled_texts)

    def test_a_failed_registration_says_how_to_undo_it(self):
        self.assertEqual(self.broken.returncode, 1, output(self.broken))
        self.assertIn("Layering violations", self.broken.stderr)
        self.assertIn(f"mathforge.certify.{BROKEN_AREA}.{BROKEN_FAMILY} -> mathforge.operations", self.broken.stderr)
        self.assertIn(f"To undo, remove {json.dumps(BROKEN_MODULE)} from FAMILIES in "
                      f"src/mathforge/areas/{BROKEN_AREA}/__init__.py", self.broken.stderr)
        self.assertEqual(self.broken_initializer_after,
                         self.tool.with_family(self.broken_initializer_before, BROKEN_MODULE))
        # The generators ran before the check failed; following the undo restores the files that they wrote.
        self.assertEqual(differing(self.generated_registered, self.generated_broken),
                         {"_generated/structured.py", "_generated/values.py", f"{BROKEN_AREA}.py",
                          "schemas/mathforge-v6.schema.json"})
        self.assertIn(BROKEN_OPERATION, self.generated_broken[f"{BROKEN_AREA}.py"])
        for process in self.undo:
            self.assertEqual(process.returncode, 0, output(process))
        self.assertEqual(differing(self.generated_registered, self.generated_undone), set())

    def test_the_repository_is_untouched(self):
        self.assertEqual(self.real_after, self.real_before)
        self.assertFalse(any(value is True for value in self.real_after.values()))


class NameValidationTests(unittest.TestCase):
    """The names that scaffold refuses before it writes anything."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tool = load_tool()

    def problems(self, *operations, area=AREA, family=FAMILY):
        return self.tool.name_problems(mathforge, area, family, list(operations))

    def refused(self, name, reason):
        problems = self.problems(name)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith(f"{name!r} "), problems)
        self.assertIn(reason, problems[0])

    def test_new_names_pass(self):
        self.assertEqual(self.problems(OPERATION, "halve_integer"), [])
        self.assertEqual(self.problems(OPERATION, area="logic", family="truth_tables"), [])

    def test_operation_names_in_use_are_refused(self):
        cases = [
            *((spec.name, "is a registered operation") for spec in registry.specs().values()),
            *((spec.public_name, f"is the public name of the registered operation {spec.name}")
              for spec in registry.specs().values() if spec.public_name != spec.name),
            *((entry.name, "is an operation of the legacy catalogue") for entry in LEGACY_CATALOG),
            *((kind, "is an MCP argument kind") for kind in registry.KIND_NAMES
              if not registry.is_older_operation(kind)),  # rational_function is a legacy operation as well.
            ("envelope", "names a definition of the MCP area-tool schemas"),
            *((name, "registry.WRAPPER_NAMES") for name in registry.WRAPPER_NAMES if not name.startswith("_")),
            ("replace", "is defined or imported in mathforge/operations.py"),
            ("ContinuedFraction", "is a name of mathforge.ntheory"),
            ("kernels", "is a module of the mathforge package"),
            ("dumps", "is an attribute of the mathforge package"),
            ("sum", "is a Python builtin"),
            ("_double", "starts with an underscore"),
            ("class", "is not an ASCII identifier"),
            ("double-integer", "is not an ASCII identifier"),
            ("zähle", "is not an ASCII identifier"),
        ]
        for name, reason in cases:
            with self.subTest(name=name):
                self.refused(name, reason)

    def test_areas_families_and_lists_are_checked(self):
        self.assertEqual(self.problems(OPERATION, area="calculus"),
                         ["'calculus' is not an area: the areas are " + ", ".join(registry.AREAS)])
        for family in ("class", "__init__", "2x", "bad-name", "FAMILIES"):
            with self.subTest(family=family):
                problems = self.problems(OPERATION, family=family)
                self.assertEqual(len(problems), 1, problems)
                self.assertTrue(problems[0].startswith(f"{family!r} cannot name a family module"), problems)
        self.assertEqual(self.problems(), ["--ops names no operation"])
        self.assertEqual(self.problems(OPERATION, OPERATION), [f"{OPERATION!r} is given 2 times"])


class FamiliesEditTests(unittest.TestCase):
    """register edits FAMILIES of an area package as a person would, and leaves the rest of the file as it is."""

    PILOT = textwrap.dedent('''\
        """Number theory producer families (mathforge.ntheory); their checkers live in mathforge.certify.ntheory."""

        # The full module names of this area's v6 families, which mathforge.registry.v6_families() joins.
        FAMILIES = ("mathforge.areas.ntheory.continued_fraction",)
        ''')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tool = load_tool()

    def edit(self, source):
        return self.tool.with_family(source, MODULE)

    def test_a_tuple_on_one_line_stays_there_while_it_fits(self):
        expected = self.PILOT.replace('continued_fraction",)', f'continued_fraction", "{MODULE}")')
        self.assertEqual(self.edit(self.PILOT), expected)
        self.assertEqual(self.edit('FAMILIES = ()\n'), f'FAMILIES = ("{MODULE}",)\n')
        self.assertEqual(self.edit('FAMILIES: tuple[str, ...] = ()  # None yet.\n'),
                         f'FAMILIES: tuple[str, ...] = ("{MODULE}",)  # None yet.\n')

    def test_a_line_that_grows_too_long_becomes_one_name_per_line(self):
        source = ('FAMILIES = ("mathforge.areas.ntheory.continued_fraction", '
                  '"mathforge.areas.ntheory.modular_powers")\n')
        self.assertEqual(self.edit(source), textwrap.dedent(f'''\
            FAMILIES = (
                "mathforge.areas.ntheory.continued_fraction",
                "mathforge.areas.ntheory.modular_powers",
                "{MODULE}",
            )
            '''))

    def test_a_tuple_on_several_lines_gains_a_line(self):
        trailing = 'FAMILIES = (\n    "mathforge.areas.ntheory.continued_fraction",\n)  # Pilot first.\n'
        self.assertEqual(self.edit(trailing), 'FAMILIES = (\n    "mathforge.areas.ntheory.continued_fraction",\n'
                                              f'    "{MODULE}",\n)  # Pilot first.\n')
        aligned = 'FAMILIES = ("mathforge.areas.ntheory.continued_fraction",\n            "mathforge.areas.x.y")\n'
        self.assertEqual(self.edit(aligned), aligned.replace('.y")', f'.y",\n            "{MODULE}")'))

    def test_an_initializer_without_families_gains_the_assignment(self):
        self.assertEqual(self.edit('"""Special functions."""\n'),
                         f'"""Special functions."""\n\n{self.tool.FAMILIES_COMMENT}\nFAMILIES = ("{MODULE}",)\n')

    def test_line_ends_and_byte_order_mark_are_kept(self):
        crlf = self.PILOT.replace("\n", "\r\n")
        edited = self.edit(crlf)
        self.assertEqual(edited, self.edit(self.PILOT).replace("\n", "\r\n"))
        self.assertEqual(self.edit("﻿" + self.PILOT), "﻿" + self.edit(self.PILOT))

    def test_a_listed_family_is_left_alone(self):
        self.assertIsNone(self.edit(self.edit(self.PILOT)))

    def test_families_that_are_no_plain_tuple_of_strings_are_refused(self):
        for source in ('PILOT = "x"\nFAMILIES = (PILOT,)\n', 'FAMILIES = ["mathforge.areas.ntheory.x"]\n',
                       "FAMILIES = (\n", 'FAMILIES = tuple(["mathforge.areas.ntheory.x"])\n'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.edit(source)


class TemplateTests(unittest.TestCase):
    """The templates that scaffold writes are Python of the supported versions, in the project's line length."""

    def test_templates(self):
        tool = load_tool()
        files = tool.templates(AREA, FAMILY, [OPERATION, "halve_integer"])
        source, tests = "src/mathforge", f"tests/v09/{AREA}"
        self.assertEqual({path: written for path, (_, written) in files.items()}, {
            f"{source}/areas/{AREA}/__init__.py": False, f"{source}/areas/{AREA}/{FAMILY}.py": True,
            f"{source}/certify/{AREA}/__init__.py": False, f"{source}/certify/{AREA}/{FAMILY}.py": True,
            f"{tests}/__init__.py": False, f"{tests}/test_{FAMILY}.py": True})
        long_names = tool.templates("equations", "a_family_name_of_forty_characters_length",
                                    ["an_operation_name_of_forty_characters_xx", "and_another_operation_name_after_it"])
        for path, (text, _) in [*files.items(), *long_names.items()]:
            with self.subTest(path=path):
                ast.parse(text, feature_version=(3, 11))
                self.assertTrue(text.endswith("\n") and not text.endswith("\n\n"))
                self.assertLessEqual(max(len(line) for line in text.splitlines()), tool.LINE_LENGTH)
        producer = ast.parse(files[f"{source}/areas/{AREA}/{FAMILY}.py"][0])
        specs = [node.args[0].value for node in ast.walk(producer)
                 if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "OperationSpec"]
        self.assertEqual(specs, [OPERATION, "halve_integer"])
        self.assertEqual(ast.literal_eval(tool._assigned(producer, "CHECKER")), f"mathforge.certify.{AREA}.{FAMILY}")
        self.assertEqual(len(tool.placeholders(files[f"{source}/areas/{AREA}/{FAMILY}.py"][0])), 10)
        package = ast.parse(files[f"{source}/areas/{AREA}/__init__.py"][0])
        self.assertEqual(ast.literal_eval(tool._assigned(package, "FAMILIES")), ())
        checker = ast.parse(files[f"{source}/certify/{AREA}/{FAMILY}.py"][0])
        self.assertEqual([node.name for node in checker.body if isinstance(node, ast.FunctionDef)], ["check_operation"])
        self.assertFalse([node for node in ast.walk(checker) if isinstance(node, (ast.Import, ast.ImportFrom))])
        skeleton = ast.parse(files[f"{tests}/test_{FAMILY}.py"][0])
        (case,) = [node for node in skeleton.body if isinstance(node, ast.ClassDef)]
        self.assertEqual(case.name, "DoublingTests")
        for method in (node for node in case.body if isinstance(node, ast.FunctionDef)):
            with self.subTest(method=method.name):
                call = method.body[-1].value
                self.assertEqual(ast.unparse(call.func), "self.fail")
                self.assertTrue(ast.literal_eval(call.args[0]).startswith("TODO: "))


class RepositoryPinsTests(unittest.TestCase):
    def test_every_pinned_constant_is_current(self):
        """What `python tools/new_family.py pins` reports for this tree: every location holds its value."""
        statuses = load_tool().pin_statuses(mathforge, REPOSITORY)
        self.assertEqual({f"{pin.path} {pin.where}: {pin.label}": status for pin, status in statuses
                          if status != "current"}, {})
        self.assertGreater(len(statuses), 15)


if __name__ == "__main__":
    unittest.main()
