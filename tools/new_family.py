"""Start a v6 operation family, register it, and print the constants that tests and documents pin.

    python tools/new_family.py scaffold <area> <family> --ops name1,name2
    python tools/new_family.py register <area> <family>
    python tools/new_family.py pins [--check]

Every subcommand takes --root DIR, the repository to work on (default: the one that holds this tool), so that a
test can run it on a copy of the tree. documentation/families.md is the checklist that the commands serve.

scaffold checks the names and writes templates, never over a file that exists:

- src/mathforge/areas/<area>/<family>.py, the producer: one OperationSpec per operation, TYPES, CHECKER and a
  produce() that implements nothing yet, with TODO marks and the producer rules;
- src/mathforge/certify/<area>/<family>.py, the independent checker: check_operation() answers False;
- tests/v09/<area>/test_<family>.py, the definition of done: every test fails with "TODO" until it is written;
- the initializers of the three packages, where they are missing.

It does not register the family: the tree keeps working while the family is written, and tests/v09/support.py
runs an unregistered family. An operation name must be new and public: no attribute or module of the mathforge
package, no name of mathforge/operations.py or of the area module, no registered or legacy operation, no argument
kind, not "envelope" or "handle", no name that the generated wrappers use, no Python builtin and no leading underscore.

register appends the family's module name to FAMILIES of its area package (where FAMILIES lists it already, it
says so and leaves the file alone), then runs tools/generate_api.py, tools/generate_schema.py and
tools/check_layering.py of the same repository, with PYTHONPATH set to its src, and echoes their output. When one
of them fails, it stops with a nonzero status and says how to undo the registration. It refuses a family whose
SPECS still hold a TODO mark, before it changes anything.

pins prints, ready to paste, the values of the constants that change when operations are added, each below the
file and the test or the passage that holds it, in the shape that the location uses (a string that the test
module binds to a name, such as PILOT, is written as that name), and says whether that location holds the value
now ("current") or not ("STALE", "MISSING"). It also lists the sizes of the MCP input schemas against their
budgets, and the operations that still lack an input in tests/v08/lean_inputs.py, a typing assertion in
tests/typing/public_api.py or a differential test in tools/check_sympy.py. With --check it exits with status 1
unless everything is current.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import json
import keyword
import os
from pathlib import Path
import pkgutil
from string import Template
import subprocess
import sys
import textwrap
from types import ModuleType


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = "mathforge"
LINE_LENGTH = 120
PLACEHOLDER = "TODO"  # scaffold marks with it what the author writes; register refuses SPECS that still hold it.
# What register runs once FAMILIES names the family, in this order, each from the tools directory of the root.
REGISTRATION_TOOLS = ("generate_api.py", "generate_schema.py", "check_layering.py")
FAMILIES_COMMENT = "# The full module names of this area's v6 families, which mathforge.registry.v6_families() joins."
SUPPORT = "tests/v09/support.py"  # The shared helpers that the scaffolded tests import.

# The files that pins reads below the root, besides the package.
REGISTRY_TESTS = "tests/v08/test_registry.py"
PILOT_TESTS = "tests/v08/test_continued_fraction.py"
DUAL_TESTS = "tests/v08/test_dual_versions.py"
GENERATED_TESTS = "tests/v08/test_generated_api.py"
MCP_TESTS = "tests/v08/test_mcp_areas.py"
MCP_GUIDE = "documentation/mcp.md"
README = "README.md"
LEAN_INPUTS = "tests/v08/lean_inputs.py"
TYPING = "tests/typing/public_api.py"
DIFFERENTIAL = "tools/check_sympy.py"
PINNED_FILES = (REGISTRY_TESTS, PILOT_TESTS, DUAL_TESTS, GENERATED_TESTS, MCP_TESTS, MCP_GUIDE, README, LEAN_INPUTS,
                TYPING, DIFFERENTIAL)
# The budgets of tests/v08/test_mcp_areas.py (AREA_SCHEMA_BUDGET, SCHEMA_BUDGET), in bytes of compact JSON.
AREA_SCHEMA_BUDGET, SCHEMA_BUDGET = 16 * 1024, 128 * 1024
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")

AREA_PACKAGE = Template('''\
"""Producer families of the $area area (mathforge.$area); their checkers live in mathforge.certify.$area."""

$comment
FAMILIES = ()
''')

CHECKER_PACKAGE = Template('''\
"""Independent checkers of the $area families in mathforge.areas.$area."""
''')

TESTS_PACKAGE = Template('''\
"""Tests of the v0.9 families of mathforge.$area; a package so that unittest discovery descends into it."""
''')

PRODUCER = Template('''\
"""$todo: what the $family family computes, in one line (mathforge.$area).

$todo: for each operation, its domain, the canonical form of its value and the evidence that its checker needs.
mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/$area/test_$family.py is the definition of done.
"""

from __future__ import annotations

from ...errors import UnsupportedOperation
from ...registry import Argument, OperationSpec

# The producer rules:
# - Imports at the top come only from modules that `import mathforge` already loads; kernels are imported inside
#   produce(). Never import mathforge.certify, mathforge.lean, mathforge.mcp, mathforge._generated or a generated
#   area module (mathforge.ntheory, ...).
# - produce() validates its inputs first, with tests like `type(x) is int` (a bool, a float or a str is never an
#   exact number), and raises InvalidInput. It builds records only from validated data. For a Matrix argument it
#   calls current_budget().matrix(rows, cols) before it validates the shape.
# - The value is never None. The evidence is a tuple of ints, Rationals, tuples, str and records: no list, dict or
#   float. Under the verifier, 23 + 2 * request + 2 * value + evidence nodes stay within max_evidence_items.
# - Every operation takes at least one argument. Only an Expression or a Matrix that Result.expression should hold
#   is named `expression` or `A`.
# - Operation names are new, and nothing is superseded (no SUPERSEDES): tools/new_family.py checked these names.
# - No speed-suite fixture and no hand-written lean module: the generic lean runner serves v6 operations.

# $todo: for each operation, its arguments (name, annotation text, default text or None, MCP argument kind), its
# return annotation, its one-sentence summary and its scope text.
SPECS = (
$specs
)

# $todo: the record classes of the values, by tag: frozen dataclasses that this module defines, whose __post_init__
# checks the canonical form and which have no computing methods. A record serializes as "v6_" + tag.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.$area.$family"


def produce(operation, args):
    """$todo: the algorithm of each operation, and the evidence that it hands to the checker."""
    raise UnsupportedOperation(f"{operation} is not implemented yet")
''')

SPEC = Template('''\
    OperationSpec("$operation", "$area",
                  (Argument("$todo", "$todo", None, "object"),),
                  "$todo", "$todo: one sentence for the catalog.", "$todo: the scope text.", version=6),''')

CHECKER = Template('''\
"""Independent checker of the family mathforge.areas.$area.$family.

$todo: for each operation, the theorem that makes True imply the claim: the identity or the certificate that is
verified, and why no other value passes it. The checker never reruns the producer's algorithm.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""


def check_operation(operation, args, value, evidence):
    # $todo: test every shape with `type(x) is int` (a bool, a float or a str is no exact number, a list no tuple),
    # charge the budget in every loop and before every builtin pow, which cannot be interrupted (about
    # bits(exponent) * size(modulus)), and answer True only when the theorem holds. Prime proofs go through
    # certify.ntheory.factorization.prime_proof(n, certificate), and a certified factorization is
    # ((p, e, certificate), ...) with strictly increasing proved primes. Any doubt answers False; a checker that
    # raises leaves the claim UNKNOWN instead of rejecting it.
    return False
''')

TESTS = Template('''\
"""$summary

These tests are the definition of done: each one fails with "$todo" until it is written. tests/v09/support.py holds
the shared helpers, and tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY
registered while its tests run, before and after

    python tools/new_family.py register $area $family
"""

from unittest.mock import patch

import mathforge as mf
from mathforge.areas.$area import $family as family

from .. import support


class ${name}Tests(support.FamilyTestCase):
    FAMILY = family

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        self.fail("$todo: support.run(operation, **arguments) for known values, each with mf.verify(result).verified")

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error."""
        self.fail("$todo: self.assertInvalidInput(support.run(...)) for each of these inputs, in every argument")

    def test_resource_limits(self):
        """An integer of 2**65536, a 65 x 65 matrix where one is taken, and max_work=1 exceed the budget."""
        self.fail("$todo: self.assertResourceLimit(...) for support.HUGE, support.oversized_matrix() and max_work=1")

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen."""
        self.fail("$todo: self.assertAccepts(operation, args, value, evidence) for true claims built by hand")

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        self.fail("$todo: a table of corruptions, each through self.assertRejects(operation, args, value, evidence)")

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        self.fail("$todo: patch.object(family, 'produce', side_effect=AssertionError), then mf.verify on a Result "
                  "and on mf.loads(mf.dumps(result))")

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX."""
        self.fail("$todo: self.roundtrip(result), mf.render_text(result.value) and mf.to_latex(result.value)")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        self.fail("$todo: support.lean_value(...) and support.value(...) against support.run(...).value, seeded")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        self.fail("$todo: self.answer(support.call(operation, {...})) against support.run(operation, ...)")
''')


def _mathforge(root: Path) -> ModuleType:
    """mathforge imported from root/src, never from another checkout or from an installed copy."""
    source = root / "src"
    if not (source / PACKAGE / "__init__.py").is_file():
        raise SystemExit(f"There is no {PACKAGE} package below {source}")
    sys.path.insert(0, str(source))
    try:
        import mathforge
    except Exception as error:  # A SyntaxError in a generated module, as much as an ImportError.
        raise SystemExit(f"Importing {PACKAGE} from {source} failed: {type(error).__name__}: {error}") from None
    found = Path(mathforge.__file__).resolve()
    if not found.is_relative_to(source.resolve()):
        raise SystemExit(f"{PACKAGE} is already imported from {found.parent}, not from {source}")
    return mathforge


def _identifier(name: str) -> bool:
    return name.isascii() and name.isidentifier() and not keyword.iskeyword(name)


def _assigned(tree: ast.Module, name: str) -> ast.expr | None:
    """The value that a module assigns to a name at its top level, in its last such assignment; None without one."""
    found = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                found = node.value
    return found


def _texts(node: ast.AST | None) -> list[str]:
    """Every string literal inside a syntax tree."""
    return [] if node is None else [inner.value for inner in ast.walk(node)
                                    if isinstance(inner, ast.Constant) and type(inner.value) is str]


# ---------------------------------------------------------------------------------------------------------------
# scaffold

def name_problems(mathforge: ModuleType, area: str, family: str, operations: Sequence[str]) -> list[str]:
    """Why scaffold refuses an area, a family name or operation names; an empty list when it takes them all."""
    registry = mathforge.registry
    public = {spec.public_name: spec.name for spec in registry.specs().values()}
    modules = {info.name for info in pkgutil.iter_modules(mathforge.__path__)}
    # The generated area module that the value functions join, with the record classes of its families.
    area_names = vars(getattr(mathforge, area)) if area in registry.AREAS else {}

    def taken(name: str) -> str | None:
        """How an operation name collides with a name that is in use."""
        if name.startswith("_"):  # Every generated function is public API, which a leading underscore denies.
            return "starts with an underscore, which no public function name does"
        if name in registry.specs():
            return "is a registered operation"
        if name in public:
            return f"is the public name of the registered operation {public[name]}"
        if registry.is_older_operation(name):
            return "is an operation of the legacy catalogue"
        if name in registry.KIND_NAMES:
            return "is an MCP argument kind"
        if name in registry.DEFINITION_NAMES:
            return "names a definition of the MCP area-tool schemas"
        if name in registry.WRAPPER_NAMES:
            return "is a name that the generated wrappers use (registry.WRAPPER_NAMES)"
        if name in vars(mathforge.operations):
            return f"is defined or imported in {PACKAGE}/operations.py"
        if name in area_names:
            return f"is a name of {PACKAGE}.{area}, the area module that its value function joins"
        if name in modules:  # Importing the module makes it an attribute of the package.
            return f"is a module of the {PACKAGE} package"
        if hasattr(mathforge, name):
            return f"is an attribute of the {PACKAGE} package"
        if hasattr(builtins, name):  # mathforge.operations star-imports every wrapper, over the builtin.
            return f"is a Python builtin, which its wrapper would hide in {PACKAGE}.operations"
        return None

    problems = []
    if area not in registry.AREAS:
        problems.append(f"{area!r} is not an area: the areas are {', '.join(registry.AREAS)}")
    if not _identifier(family) or family.startswith("__"):
        problems.append(f"{family!r} cannot name a family module: it must be an ASCII identifier, no keyword and "
                        "no dunder name")
    elif family == "FAMILIES":  # Importing the module would bind the name in the package, over the tuple.
        problems.append(f"{family!r} cannot name a family module: it is the tuple of the area package")
    if not operations:
        problems.append("--ops names no operation")
    for index, name in enumerate(operations):
        if name in operations[:index]:
            continue
        if operations.count(name) > 1:
            problems.append(f"{name!r} is given {operations.count(name)} times")
        reason = taken(name) if _identifier(name) else "is not an ASCII identifier, or is a keyword"
        if reason is not None:
            problems.append(f"{name!r} {reason}")
    return problems


def templates(area: str, family: str, operations: Sequence[str]) -> dict[str, tuple[str, bool]]:
    """The files of a new family by path below the root: their text, and whether scaffold must create them. The
    others are package initializers, which it writes only where they are missing."""
    name = "".join(part.capitalize() for part in family.split("_"))
    values = dict(area=area, family=family, todo=PLACEHOLDER, comment=FAMILIES_COMMENT,
                  name=name if name.isidentifier() else "Family" + name)
    values["specs"] = "\n".join(SPEC.substitute(values, operation=operation) for operation in operations)
    summary = f"The {family} family of {PACKAGE}.{area}: {', '.join(operations)}."
    values["summary"] = textwrap.fill(summary, LINE_LENGTH - 3, break_long_words=False, break_on_hyphens=False)
    source, tests = f"src/{PACKAGE}", f"tests/v09/{area}"
    return {
        f"{source}/areas/{area}/__init__.py": (AREA_PACKAGE.substitute(values), False),
        f"{source}/areas/{area}/{family}.py": (PRODUCER.substitute(values), True),
        f"{source}/certify/{area}/__init__.py": (CHECKER_PACKAGE.substitute(values), False),
        f"{source}/certify/{area}/{family}.py": (CHECKER.substitute(values), True),
        f"{tests}/__init__.py": (TESTS_PACKAGE.substitute(values), False),
        f"{tests}/test_{family}.py": (TESTS.substitute(values), True),
    }


def scaffold(root: Path, area: str, family: str, operations: Sequence[str]) -> None:
    """Write the templates of a new family below root. SystemExit, with nothing written, when a name is refused
    or when one of the family's own files exists."""
    problems = name_problems(_mathforge(root), area, family, operations)
    if problems:
        raise SystemExit("\n".join(["Nothing was written:", *("  " + problem for problem in problems)]))
    files = templates(area, family, operations)
    existing = [path for path, (_, own) in files.items() if own and (root / path).exists()]
    if existing:
        raise SystemExit("\n".join(["Nothing was written: scaffold never overwrites a file, and these exist:",
                                    *("  " + path for path in existing)]))
    for path, (text, _) in files.items():
        target = root / path
        if target.exists():
            print("kept     " + path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        print("created  " + path)
    if not (root / SUPPORT).is_file():
        print(f"Warning: the tests of the family import {SUPPORT}, which is missing below {root}.")
    print(f"The family is not registered. Replace every {PLACEHOLDER}, make\n"
          f"  python -m unittest tests.v09.{area}.test_{family}\n"
          f"pass, then run\n  python tools/new_family.py register {area} {family}")


# ---------------------------------------------------------------------------------------------------------------
# register

def with_family(source: str, module: str) -> str | None:
    """The text of an area package initializer with a module name appended to FAMILIES; None when FAMILIES lists it.

    Everything else stays as it is, line ends included. A tuple on one line stays there while the line fits
    LINE_LENGTH and becomes one name per line beyond that; a tuple that spans lines gains a line; an initializer
    without FAMILIES gains the assignment. ValueError when FAMILIES is no plain tuple of string literals, or is
    written in a way that the edit would not extend correctly.
    """
    mark = "\ufeff" if source.startswith("\ufeff") else ""
    body = source[len(mark):]
    newline = "\r\n" if "\r\n" in body else "\n"
    entry = json.dumps(module)
    try:
        value = _assigned(ast.parse(body), "FAMILIES")
    except SyntaxError as error:
        raise ValueError(f"the initializer does not parse: {error}") from None
    if value is None:
        names: list[str] = []
        gap = "" if not body or body.endswith("\n") else newline
        edited = body + gap + (newline if body.strip() else "") + FAMILIES_COMMENT + newline \
            + f"FAMILIES = ({entry},)" + newline
    else:
        if not isinstance(value, ast.Tuple) or len(_texts(value)) != len(value.elts):
            raise ValueError("FAMILIES is not a plain tuple of string literals")
        names = _texts(value)
        if module in names:
            return None
        lines = body.split("\n")

        def position(line: int, column: int) -> int:
            """The index in body of a location of the syntax tree: lines count from 1, columns in UTF-8 bytes."""
            before = sum(len(text) + 1 for text in lines[:line - 1])
            return before + len(lines[line - 1].encode("utf-8")[:column].decode("utf-8"))

        start, end = position(value.lineno, value.col_offset), position(value.end_lineno, value.end_col_offset)
        segments = [body[position(item.lineno, item.col_offset):position(item.end_lineno, item.end_col_offset)]
                    for item in value.elts]
        if not segments:
            edited = body[:start] + f"({entry},)" + body[end:]
        elif value.lineno == value.end_lineno:
            line = lines[value.lineno - 1].rstrip("\r")
            flat = "(" + ", ".join([*segments, entry]) + ")"
            if len(line) - (end - start) + len(flat) > LINE_LENGTH:
                indent = line[:len(line) - len(line.lstrip())]
                flat = "(" + "".join(f"{newline}{indent}    {segment}," for segment in [*segments, entry]) \
                    + newline + indent + ")"
            edited = body[:start] + flat + body[end:]
        else:
            if body[start] != "(" or body[end - 1] != ")":
                raise ValueError("FAMILIES spans lines without parentheses")
            last = value.elts[-1]
            finish = position(last.end_lineno, last.end_col_offset)
            prefix = lines[last.lineno - 1][:position(last.lineno, last.col_offset) - position(last.lineno, 0)]
            indent = prefix if not prefix.strip() else " " * len(prefix)  # Below the last name, where it starts.
            after = body[finish:end - 1]  # Between the last name and the closing parenthesis.
            if "\n" not in after:  # The parenthesis closes the last name's line: the new name takes it over.
                edited = body[:finish] + "," + newline + indent + entry + body[end - 1:]
            else:  # The new name gets a line of its own, with a trailing comma when the last name has one.
                comma = after.lstrip(" \t").startswith(",")
                stop = body.index("\n", finish)
                if body[stop - 1] == "\r":
                    stop -= 1
                edited = body[:finish] + ("" if comma else ",") + body[finish:stop] + newline + indent + entry \
                    + ("," if comma else "") + body[stop:]
    try:
        written = _assigned(ast.parse(edited), "FAMILIES")
    except SyntaxError:
        written = None
    if not isinstance(written, ast.Tuple) or len(written.elts) != len(names) + 1 or _texts(written) != [*names, module]:
        raise ValueError("FAMILIES is written in a way that this tool cannot extend")
    return mark + edited


def placeholders(source: str) -> list[str]:
    """The texts in SPECS of a family module that still hold the mark of scaffold."""
    return [text for text in _texts(_assigned(ast.parse(source), "SPECS")) if PLACEHOLDER in text]


def register(root: Path, area: str, family: str) -> None:
    """Append a family to FAMILIES of its area package below root and regenerate what derives from the registry.

    SystemExit when the family cannot be registered, before anything changes, and when a tool fails, after the
    change, with the way to undo it.
    """
    if not _identifier(area) or not _identifier(family):
        raise SystemExit("Nothing was registered: an area and a family are named by ASCII identifiers")
    module = f"{PACKAGE}.areas.{area}.{family}"
    package = f"src/{PACKAGE}"
    initializer, producer = f"{package}/areas/{area}/__init__.py", f"{package}/areas/{area}/{family}.py"
    needed = [initializer, producer, f"{package}/certify/{area}/{family}.py",
              *(f"tools/{tool}" for tool in REGISTRATION_TOOLS)]
    missing = [path for path in needed if not (root / path).is_file()]
    if missing:
        raise SystemExit("\n".join(["Nothing was registered: these files are missing below " + str(root),
                                    *("  " + path for path in missing)]))
    try:
        unfinished = placeholders((root / producer).read_text(encoding="utf-8"))
    except (ValueError, SyntaxError) as error:  # UnicodeDecodeError is a ValueError.
        raise SystemExit(f"Nothing was registered: {producer} does not parse: {error}") from None
    if unfinished:
        raise SystemExit(f"Nothing was registered: SPECS of {producer} still holds {len(unfinished)} "
                         f"{PLACEHOLDER} marks, the first in {unfinished[0]!r}")
    try:
        edited = with_family((root / initializer).read_bytes().decode("utf-8"), module)
    except ValueError as error:
        raise SystemExit(f"Nothing was registered: {initializer}: {error}. Add {json.dumps(module)} to FAMILIES "
                         "there by hand, then run register again.") from None
    if edited is None:
        print(f"FAMILIES of {initializer} already lists {module}: only the generated files are refreshed.")
    else:
        (root / initializer).write_bytes(edited.encode("utf-8"))
        print(f"Appended {module} to FAMILIES of {initializer}.")
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    for tool in REGISTRATION_TOOLS:
        print(f"$ python tools/{tool}", flush=True)  # Flushed: the tool writes to the same stream.
        status = subprocess.run([sys.executable, str(root / "tools" / tool)], cwd=root, env=environment).returncode
        if status:
            origin = "listed there before this run" if edited is None else "which this run appended"
            raise SystemExit(
                f"tools/{tool} failed with status {status}: {module} is not registered cleanly.\n"
                f"To undo, remove {json.dumps(module)} from FAMILIES in {initializer} ({origin}) and run\n"
                "  python tools/generate_api.py\n  python tools/generate_schema.py\n"
                "so that the generated files match the registry again. To go on instead, repair the family and "
                "run register again.")
    print(f"Registered {module}. Run `python tools/new_family.py pins` for the constants that tests and documents "
          "pin.")


# ---------------------------------------------------------------------------------------------------------------
# pins

_STATED = object()  # The value of a pin that is found by its text.


@dataclass(frozen=True)
class Pin:
    """One pinned constant: where it stands, and its current value in the shape that the location uses.

    A pin with a value is the second argument of the call self.assertEqual(<label>, ...) in a test file, and is
    compared as a Python value. Any other pin is a passage, searched as text with any white space between words.
    """
    path: str    # The file, relative to the root
    where: str   # The test method, or the part of the document
    label: str   # What the value is compared with there, or what the passage states
    text: str    # The value or the passage, ready to paste
    value: object = _STATED


def _source(value: object, names: Mapping[str, str] | None = None) -> str:
    """The source text of a pinned value on one line, with the double quotes that the tests use. A string that
    names maps to a name, a constant of the location's module, is written as that name."""
    if type(value) is str:
        return names[value] if names and value in names else json.dumps(value)
    if type(value) is tuple:
        return "(" + ", ".join(_source(item, names) for item in value) + (",)" if len(value) == 1 else ")")
    if type(value) is list:
        return "[" + ", ".join(_source(item, names) for item in value) + "]"
    if type(value) is dict:
        return "{" + ", ".join(f"{_source(key, names)}: {_source(item, names)}" for key, item in value.items()) + "}"
    return repr(value)


def _display(opening: str, items: Sequence[str], closing: str, width: int = 100) -> str:
    """A bracketed list of items on lines of about width characters, the later lines aligned inside the bracket."""
    lines, line, fresh = [], opening, True
    for index, item in enumerate(items):
        piece = item + ("," if index + 1 < len(items) else "")
        if not fresh and len(line) + 1 + len(piece) > width:
            lines.append(line)
            line, fresh = " " * len(opening), True
        line += piece if fresh else " " + piece
        fresh = False
    return "\n".join([*lines, line + closing])


def _wrapped(value: object, names: Mapping[str, str] | None = None) -> str:
    """The source text of a pinned value, wrapped between the items of its outer bracket."""
    if type(value) is dict:
        return _display("{", [f"{_source(key, names)}: {_source(item, names)}" for key, item in value.items()], "}")
    if type(value) is list:
        return _display("[", [_source(item, names) for item in value], "]")
    if type(value) is tuple:
        return _display("(", [_source(item, names) for item in value], ",)" if len(value) == 1 else ")")
    return _source(value, names)


def _number(count: int) -> str:
    return NUMBERS[count] if count < len(NUMBERS) else str(count)


def pins_of(mathforge: ModuleType, tree: _Tree) -> list[Pin]:
    """The constants that tests and documents pin and that change with the registered operations, written as the
    files below the root of tree write them."""
    import importlib
    from mathforge.capabilities import area_tools
    from mathforge.mcp.boundary import MAX_CATALOG_LIMIT, call_tool
    registry = mathforge.registry
    specs = list(registry.specs().values())
    names = {area: [spec.name for spec in specs if spec.area == area] for area in registry.AREAS}
    later = [spec for spec in specs if spec.version == 6 and not registry.is_older_operation(spec.name)]
    families = registry.v6_families()
    # As test_catalog_filters_sorts_and_limits reads them: from the catalog tool, which lists at most its limit.
    catalog = {entry["name"]: entry
               for entry in call_tool("mathforge.catalog", {"limit": MAX_CATALOG_LIMIT}).payload["operations"]}
    defaults = {(spec.name, argument["name"]): argument["default"] for spec in specs if spec.name in catalog
                for argument in catalog[spec.name]["arguments"] if not argument["required"]}
    tools = area_tools()
    total = len(mathforge.capabilities()["tools"])
    quoted = [f"`{tool}`" for tool in tools]
    rows = ["| `{}` | {} |".format(tool, ", ".join(f"`{area}`" for area in registry.MCP_TOOLS[tool])) for tool in tools]

    def compared(path: str, where: str, label: str, value: object, text: str | None = None) -> Pin:
        return Pin(path, where, label, _wrapped(value, tree.names(path)) if text is None else text, value)

    def named_set(path: str, items: Sequence[str]) -> str:
        """A set display of strings in the given order, written as the location writes them."""
        return _display("{", [_source(item, tree.names(path)) for item in items], "}") if items else "set()"

    # Inside _testing_families(demo), the demo family follows the declared ones.
    with_demo = ", ".join([*(_source(name, tree.names(DUAL_TESTS)) for name in families), "demo.__name__"])
    pins = [
        compared(REGISTRY_TESTS, "test_every_spec_has_known_area_and_kind", "len(registry.KIND_NAMES)",
                 len(registry.KIND_NAMES)),
        compared(REGISTRY_TESTS, "test_every_spec_has_known_area_and_kind", "len(registry.MCP_TOOLS)",
                 len(registry.MCP_TOOLS)),
        *(compared(REGISTRY_TESTS, "test_area_assignment", f'names("{area}")', set(names[area]),
                   named_set(REGISTRY_TESTS, names[area])) for area in ("ntheory", "series", "integrals")),
        compared(REGISTRY_TESTS, "test_area_assignment", "{area: size for area, size in sizes.items() if size}",
                 {area: len(found) for area, found in names.items() if found}),
        compared(REGISTRY_TESTS, "test_area_assignment",
                 '[spec.public_name for spec in registry.area_operations("series")]',
                 [spec.public_name for spec in registry.area_operations("series")]),
        compared(REGISTRY_TESTS, "test_area_assignment", "len(analysis)",
                 len(registry.tool_operations("mathforge.analysis"))),
        compared(PILOT_TESTS, "test_registry_entry", "registry.v6_families()", families),
        compared(DUAL_TESTS, "test_families_come_from_area_packages", "registry.v6_families()", families),
        compared(DUAL_TESTS, "test_families_come_from_area_packages",
                 'importlib.import_module("mathforge.areas.ntheory").FAMILIES',
                 importlib.import_module(PACKAGE + ".areas.ntheory").FAMILIES),
        Pin(DUAL_TESTS, "test_families_come_from_area_packages", "the families with the demo's",
            f"self.assertEqual(registry.v6_families(), ({with_demo}))"),
        Pin(DUAL_TESTS, "test_capabilities_list_each_name_once", "the entries after the frozen ones, with the demo's",
            f'self.assertEqual(len(operations), len(frozen["operations"]) + {len(later) + 1})'),
        compared(DUAL_TESTS, "test_capabilities_list_each_name_once", 'names[len(frozen["operations"]):]',
                 [*(spec.name for spec in later), "demo_variable"]),
    ]
    if later:
        pins.append(Pin(GENERATED_TESTS, "test_a_wrapper_name_outside_the_reserved_names_is_refused",
                        "the first v6 wrapper of structured.py",
                        f'"^Generated mathforge/_generated/structured.py: {later[0].public_name} "'))
    listed = ", ".join(quoted[:-1]) + " and " + quoted[-1] if len(quoted) > 1 else "".join(quoted)
    pins += [
        compared(MCP_TESTS, "test_catalog_filters_sorts_and_limits", "defaults", defaults),
        Pin(MCP_GUIDE, "the tool list", "the area tools", f"{_number(len(tools)).capitalize()} exist today: {listed}."),
        Pin(MCP_GUIDE, "the tool list", "the number of tools", f"That makes {total} tools today."),
        Pin(MCP_GUIDE, "the first example", "the area tools",
            _display(f"assert names[{total - len(tools) - 1}:-1] == [", [json.dumps(tool) for tool in tools], "]")),
        Pin(MCP_GUIDE, "the first example", "the number of tools",
            f'assert names[-1] == "mathforge.catalog" and len(names) == {total}'),
        Pin(MCP_GUIDE, "Run an operation through an area tool", "the table of the area tools",
            "\n".join([f"The {_number(len(tools))} tools serve:", "| Tool | Areas |", "| --- | --- |", *rows])),
        Pin(README, "Connect an AI agent", "the number of tools", f"offers {total} tools:"),
    ]
    return pins


class _Tree:
    """The files below a root that pins reads, each read once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._texts: dict[str, str | None] = {}
        self._constants: dict[str, dict[str, ast.expr]] = {}
        self._compared: dict[str, dict[str, list[object]]] = {}

    def text(self, path: str) -> str | None:
        """The text of a file; None when it cannot be read."""
        if path not in self._texts:
            try:
                self._texts[path] = (self.root / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                self._texts[path] = None
        return self._texts[path]

    def tree(self, path: str) -> ast.Module | None:
        """The syntax tree of a Python file; None when it cannot be read or parsed."""
        text = self.text(path)
        try:
            return None if text is None else ast.parse(text)
        except SyntaxError:
            return None

    def constants(self, path: str) -> dict[str, ast.expr]:
        """The literals that a Python file binds to plain names at its top level (PILOT = "..."), by name."""
        if path not in self._constants:
            tree = self.tree(path)
            self._constants[path] = {
                node.targets[0].id: node.value for node in (tree.body if tree is not None else ())
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and _literal(node.value) is not _STATED}
        return self._constants[path]

    def names(self, path: str) -> dict[str, str]:
        """The strings among those constants, each mapped to its name (to the first, for a string bound twice)."""
        found: dict[str, str] = {}
        for name, node in self.constants(path).items():
            value = _literal(node)
            if type(value) is str:
                found.setdefault(value, name)
        return found

    def compared(self, path: str) -> dict[str, list[object]]:
        """What the assertEqual calls of a test file compare with a literal: the literals by the source text of
        the call's first argument. A second argument counts when it is a Python literal once the module's constants
        stand for their names; a call that compares with anything else is left out."""
        if path not in self._compared:
            tree = self.tree(path)
            found: dict[str, list[object]] = {}
            constants = self.constants(path)

            class Named(ast.NodeTransformer):
                def visit_Name(self, node: ast.Name) -> ast.AST:
                    return constants.get(node.id, node)

            for node in ast.walk(tree) if tree is not None else ():
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "assertEqual" and len(node.args) >= 2):
                    literal = _literal(Named().visit(copy.deepcopy(node.args[1])))
                    if literal is not _STATED:
                        found.setdefault(ast.unparse(node.args[0]), []).append(literal)
            self._compared[path] = found
        return self._compared[path]


def _literal(node: ast.AST) -> object:
    """The value of a syntax tree that is a Python literal; _STATED for any other tree."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _STATED


def _calls(node: ast.AST, function: str) -> bool:
    """Whether a syntax tree is a call of the function of that plain name."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function


def _words(text: str) -> str:
    return " ".join(text.split())


def status(pin: Pin, tree: _Tree) -> str:
    """"current" when the pin's location holds its value; otherwise what is wrong, starting in capitals."""
    text = tree.text(pin.path)
    if text is None:
        return "MISSING: the file cannot be read"
    if pin.value is _STATED:
        return "current" if _words(pin.text) in _words(text) else "STALE"
    pinned = tree.compared(pin.path).get(ast.unparse(ast.parse(pin.label, mode="eval").body), [])
    if not pinned:
        return "MISSING: the file has no such assertEqual call with a literal"
    if all(type(found) is type(pin.value) and found == pin.value for found in pinned):
        return "current"
    return f"STALE: the file pins {pinned[0]!r}" if type(pinned[0]) is int else "STALE"


def pin_statuses(mathforge: ModuleType, root: Path) -> list[tuple[Pin, str]]:
    """Every pin of the tree below root with its status: "current", or what is wrong, starting in capitals."""
    tree = _Tree(root)
    return [(pin, status(pin, tree)) for pin in pins_of(mathforge, tree)]


def lacking(mathforge: ModuleType, tree: _Tree) -> list[tuple[str, list[str] | None]]:
    """What the operations still lack: (where an entry belongs, the operations without one). The list is None when
    the file cannot be read or does not hold the table."""
    from mathforge.registry_v07 import LEGACY_CATALOG
    registry = mathforge.registry
    later = [spec for spec in registry.specs().values() if spec.version == 6]
    # As tests/v08/lean_inputs.py counts them: every registry and legacy operation except verify.
    every = [name for name in dict.fromkeys((*registry.specs(), *(entry.name for entry in LEGACY_CATALOG)))
             if name != "verify"]
    inputs, typing, differential = tree.tree(LEAN_INPUTS), tree.tree(TYPING), tree.tree(DIFFERENTIAL)
    small = _assigned(inputs, "SMALL_INPUTS") if inputs is not None else None
    keys = None if not isinstance(small, ast.Dict) else {
        key.value for key in small.keys if isinstance(key, ast.Constant) and type(key.value) is str}
    # The functions that the first argument of an assert_type(...) calls, as in mf.<area>.<name>(...).
    typed = None if typing is None else {
        call.func.attr for node in ast.walk(typing) if _calls(node, "assert_type") and node.args
        for call in ast.walk(node.args[0]) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)}
    table = _assigned(differential, "COVERED_OPERATIONS") if differential is not None else None
    covered = None if table is None else set(_texts(table))
    return [
        (f"an input in SMALL_INPUTS of {LEAN_INPUTS}",
         None if keys is None else [name for name in every if name not in keys]),
        (f"an assert_type in {TYPING} (v6 operations)",
         None if typed is None else [spec.name for spec in later if spec.public_name not in typed]),
        (f"a name in COVERED_OPERATIONS of {DIFFERENTIAL} (v6 operations)",
         None if covered is None else [spec.name for spec in later if spec.name not in covered]),
    ]


def listing(mathforge: ModuleType, root: Path) -> tuple[list[str], int]:
    """The lines that pins prints for the tree below root, and how many of its items are not current."""
    from mathforge.capabilities import area_tools
    from mathforge.mcp.boundary import MAX_CATALOG_LIMIT
    from mathforge.mcp.schemas import input_schemas
    from mathforge.registry_v07 import LEGACY_CATALOG
    registry = mathforge.registry
    tree = _Tree(root)
    count = len(registry.specs()) + len([entry for entry in LEGACY_CATALOG if entry.name not in registry.specs()])
    lines = [f"Pinned constants of {root}: {PACKAGE} {mathforge.__version__}, {len(registry.specs())} registry "
             f"operations and {len(LEGACY_CATALOG)} legacy ones.",
             "Below each location stands its current value, ready to paste; [current] says that the location holds "
             "it."]
    if count > MAX_CATALOG_LIMIT:
        lines.append(f"The catalog holds {count} operations and lists at most {MAX_CATALOG_LIMIT}: {MCP_TESTS} "
                     "reads the defaults from a listing that is no longer complete.")
    open_items = 0
    path = None
    for pin, state in pin_statuses(mathforge, root):
        if pin.path != path:
            path = pin.path
            lines += ["", path]
        open_items += state != "current"
        lines.append(f"  {pin.where}: {pin.label}  [{state}]")
        lines += ["      " + line if line else "" for line in pin.text.splitlines()]
    sizes = {name: len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
             for name, schema in input_schemas().items()}
    lines += ["", f"MCP input schemas, in bytes of compact JSON ({MCP_TESTS} allows {AREA_SCHEMA_BUDGET} for an area "
                  f"tool and {SCHEMA_BUDGET} in all)"]
    for label, size, budget in (*((tool, sizes[tool], AREA_SCHEMA_BUDGET) for tool in area_tools()),
                                (f"all {len(sizes)} tools", sum(sizes.values()), SCHEMA_BUDGET)):
        open_items += size > budget
        lines.append(f"  {label}: {size} ({100 * size // budget}%)" + ("  [OVER BUDGET]" if size > budget else ""))
    lines += ["", "Operations that still lack"]
    for label, names in lacking(mathforge, tree):
        open_items += 1 if names is None else len(names)
        found = "MISSING: the file or its table cannot be read" if names is None else ", ".join(names) or "none"
        lines += textwrap.wrap(f"{label}: {found}", LINE_LENGTH, initial_indent="  ", subsequent_indent="      ",
                               break_long_words=False, break_on_hyphens=False)
    lines += ["", "Everything is current." if not open_items else f"Not current: {open_items} of the items above."]
    return lines, open_items


def pins(root: Path, check: bool) -> None:
    """Print the pinned constants of the tree below root; with check, SystemExit(1) unless all are current."""
    lines, open_items = listing(_mathforge(root), root)
    print("\n".join(lines))
    if check and open_items:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # A path may hold characters that the console's code page lacks.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPOSITORY, metavar="DIR",
                        help="The repository to work on (default: the one that holds this tool)")
    # A subcommand takes --root too; without it there, the value given before the subcommand, or the default, stays.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=argparse.SUPPRESS, metavar="DIR", help="The repository to work on")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("scaffold", "Write the templates of a new family"),
                       ("register", "Append a family to FAMILIES and regenerate the API and the schemas")):
        command = commands.add_parser(name, parents=[common], help=text)
        command.add_argument("area", help="One of registry.AREAS")
        command.add_argument("family", help="The module name of the family")
        if name == "scaffold":
            command.add_argument("--ops", required=True, metavar="NAME[,NAME...]", help="The operation names")
    command = commands.add_parser("pins", parents=[common], help="Print the constants that tests and documents pin")
    command.add_argument("--check", action="store_true", help="Exit with status 1 unless everything is current")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")
    if arguments.command == "register":  # It reads no registry itself: the generators that it runs report.
        register(root, arguments.area, arguments.family)
        return
    try:
        if arguments.command == "scaffold":
            scaffold(root, arguments.area, arguments.family,
                     [name.strip() for name in arguments.ops.split(",") if name.strip()])
        else:
            pins(root, arguments.check)
    except RuntimeError as error:  # The registry build refuses a family, at its first lookup.
        raise SystemExit(f"The registry of {PACKAGE} below {root} does not build: {error}") from None


if __name__ == "__main__":
    main()
