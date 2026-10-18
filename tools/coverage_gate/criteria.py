"""The eight coverage criteria of the v0.8 gate, and the adapter protocol that the eighth one runs.

An inventory row counts as covered ("yes") only when all eight criteria hold for the MathForge operations it lists
(Row.mathforge_operations):

1. capabilities: the row lists at least one operation, and mathforge.capabilities() lists every one of them;
2. checker: every operation has a registered independent checker. A registry operation needs
   mathforge.registry.checker() to resolve; an operation of the legacy catalogue (mathforge.registry_v07.LEGACY_CATALOG)
   counts as checked, because mathforge.verify's v2-v4 verifiers check every v0.7 Result kind;
3. json: every Result that the row's adapter returns survives a JSON round trip, loads(dumps(r)) == r, and verifies,
   verify(...).verified; at least one Result must have been returned;
4. mcp: every operation is reachable over MCP: a registry operation through the area tool that serves it (the 60 v0.7
   operations also through mathforge.compute), a legacy operation through its own tool (CatalogEntry.tool);
5. docs: the Python examples of documentation/*.md call every operation by its public name, through mf., mathforge.,
   operations. or an area module, as in mf.ntheory.continued_fraction(...). tools/check_docs.py executes those
   examples in the gate, so this criterion only finds the calls;
6. differential: tools/check_sympy.py compares every operation with SymPy (its COVERED_OPERATIONS);
7. speed: every operation's worst speed ratio against SymPy is at most SPEED_LIMIT;
8. depth: the row has an adapter, and at least ceil(DEPTH_THRESHOLD x n) of its n reference examples MATCH.

No criterion holds vacuously: criteria 2 and 4-7 need the row to list an operation, like criterion 1. The status of a
row is "yes" when all eight hold, "partial" when criterion 1 holds but not all, and "no" otherwise. An operation name
that capabilities() does not list is reported as "unknown operation: <name>", never just counted as unmet.

The speed file of criterion 7 is JSON that the Phase 1 speed suite writes:

    {"operations": {"<operation name>": <ratio>, ...}}

where a ratio is the operation's worst MathForge-to-SymPy time ratio over its fixtures, a positive finite number. Other
top-level keys are ignored. A missing file, or an operation that the file does not list, leaves criterion 7 unmet; a
malformed file raises ValueError.

The adapter protocol. An adapter (ADAPTERS maps a row id to it) poses one SymPy reference example to the row's MathForge
operation and judges the answer: adapter(example) -> (outcome, result), where

- MATCH: MathForge computed an exact, verified value that equals SymPy's recorded one (same_value); result is the Result;
- MISMATCH: MathForge computed a verified value that differs from SymPy's; result is the Result;
- UNVERIFIED: MathForge computed a value whose verification does not pass; result is the Result;
- UNEVALUATED: the example cannot be posed to the operation (an input outside its domain, an API it does not compute),
  or MathForge returned no value (invalid input, unsupported, no conclusion); result is None;
- ERROR: the adapter failed; result is None. evaluate_row also records an exception the adapter raises, or a reply
  outside the protocol, as ERROR, and goes on with the next example.

Only a MATCH whose Result passes criterion 3's checks counts towards depth. The modules of coverage_gate.adapters
register their adapters in ADAPTERS; this module imports that package last, so importing it makes every adapter
available. SymPy is imported only when same_value or an adapter runs.
"""

import ast
import contextlib
import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

import mathforge
from mathforge import Result, registry
from mathforge.capabilities import area_tool
from mathforge.errors import UnsupportedOperation
from mathforge.registry_v07 import LEGACY_CATALOG

from .examples import Example
from .inventory import Row

__all__ = ["CRITERIA", "DEPTH_THRESHOLD", "SPEED_LIMIT", "Facts", "Outcome", "Adapter", "ADAPTERS", "RowVerdict",
           "gather_facts", "evaluate_row", "same_value", "public_name"]

CRITERIA = ("capabilities", "checker", "json", "mcp", "docs", "differential", "speed", "depth")
DEPTH_THRESHOLD = 0.8
SPEED_LIMIT = 3.0

# The names before ".<public name>(" that make a call in a documentation example a call of an operation (criterion 5).
_QUALIFIERS = frozenset({"mf", "mathforge", "operations", *registry.AREAS})


@dataclass(frozen=True)
class Facts:
    """What criteria 1, 2 and 4-7 compare a row's operations with; gather_facts collects them from a checkout."""
    operations: frozenset[str]    # The operation names of mathforge.capabilities()["operations"]
    checkers: frozenset[str]      # The operations with a registered independent checker
    mcp: frozenset[str]           # The operations that an MCP tool serves
    documented: frozenset[str]    # The public names that the Python examples of documentation/*.md call
    differential: frozenset[str]  # tools/check_sympy.COVERED_OPERATIONS
    speed: Mapping[str, float]    # Operation -> worst speed ratio; empty without a speed file


class Outcome(Enum):
    """An adapter's judgment of one reference example (the module docstring gives the protocol)."""
    MATCH = "match"
    MISMATCH = "mismatch"
    UNEVALUATED = "unevaluated"
    UNVERIFIED = "unverified"
    ERROR = "error"


Adapter = Callable[[Example], tuple[Outcome, Result[Any] | None]]
ADAPTERS: dict[str, Adapter] = {}  # Row id -> adapter; the modules of coverage_gate.adapters fill it.


@dataclass(frozen=True)
class RowVerdict:
    """What evaluate_row found for one row. criteria holds every name of CRITERIA, in that order; matched counts the
    examples whose MATCH counts towards depth, examples all the examples evaluated. details explains, in order, the
    unknown operations, each unmet criterion, then every example that did not count as a match."""
    row: str
    criteria: Mapping[str, bool]
    status: str  # yes | partial | no
    matched: int
    examples: int
    details: tuple[str, ...]


def public_name(operation: str) -> str:
    """The name that documentation calls an operation by: a registry operation's public name (its function in an area
    module and in mathforge.operations, such as taylor for function_taylor), else the name itself, as for the legacy
    catalogue's functions in mathforge.operations."""
    try:
        return registry.spec(operation).public_name
    except UnsupportedOperation:
        return operation


def gather_facts(root: Path, *, speed_file: Path | None) -> Facts:
    """The facts of the installed mathforge, of root/documentation (root is the checkout) and of tools/check_sympy.py,
    with the speed ratios of speed_file (none when it is None or does not exist). The module docstring gives the rule
    of each criterion and the speed file's format."""
    import check_sympy  # A script beside coverage_gate: tools/ is on sys.path whenever coverage_gate imports.

    operations = frozenset(entry["name"] for entry in mathforge.capabilities()["operations"])
    checkers = {entry.name for entry in LEGACY_CATALOG}
    mcp = {entry.name for entry in LEGACY_CATALOG if entry.tool}
    for name, spec in registry.specs().items():
        with contextlib.suppress(UnsupportedOperation):
            registry.checker(name)
            checkers.add(name)
        if spec.version == 5 or area_tool(spec) is not None:  # mathforge.compute serves the v0.7 operations.
            mcp.add(name)
    called = _called_names(Path(root) / "documentation")
    return Facts(operations=operations, checkers=frozenset(checkers), mcp=frozenset(mcp),
                 documented=frozenset(name for name in map(public_name, operations) if name in called),
                 differential=frozenset(check_sympy.COVERED_OPERATIONS), speed=_speed_ratios(speed_file))


def _called_names(documentation: Path) -> set[str]:
    """The names called as <qualifier>.<name>(...) in the Python fences of documentation/*.md: the fences that
    tools/check_docs.py executes, parsed, so that comments and strings never count."""
    from check_docs import python_fences

    called = set()
    for path in sorted(documentation.glob("*.md")):
        for line, source in python_fences(path):
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                raise ValueError(f"{path}:{line}: the Python example does not parse: {error}") from error
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    owner = node.func.value
                    qualifier = (owner.id if isinstance(owner, ast.Name)
                                 else owner.attr if isinstance(owner, ast.Attribute) else None)
                    if qualifier in _QUALIFIERS:
                        called.add(node.func.attr)
    return called


def _speed_ratios(path: Path | None) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return {}
    try:
        document = json.loads(Path(path).read_bytes().decode("utf-8-sig"))
        ratios = document.get("operations") if isinstance(document, dict) else None
        if not isinstance(ratios, dict):
            raise ValueError('a speed file is an object with an "operations" object')
        for name, ratio in ratios.items():
            if (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio)
                    or ratio <= 0):
                raise ValueError(f"the ratio of {name} must be a positive finite number, not {ratio!r}")
        return {name: float(ratio) for name, ratio in ratios.items()}
    except ValueError as error:  # json.JSONDecodeError and UnicodeDecodeError are ValueErrors as well.
        raise ValueError(f"{path}: {error}") from error


def evaluate_row(row: Row, examples: Sequence[Example], facts: Facts) -> RowVerdict:
    """The eight criteria for one row, its status and the explanation of what fails. examples are the row's reference
    examples; ValueError when one belongs to another row."""
    strangers = [example.id for example in examples if example.row != row.id]
    if strangers:
        raise ValueError(f"examples of other rows than {row.id}: {', '.join(strangers)}")
    operations = row.mathforge_operations
    details = [f"unknown operation: {name}" for name in operations if name not in facts.operations]
    met: dict[str, bool] = {}

    def per_operation(criterion: str, problem: str, failing: Sequence[str]) -> None:
        if operations and failing:
            details.append(f"{criterion}: {problem}: {', '.join(failing)}")
        met[criterion] = bool(operations) and not failing

    if not operations:
        details.append("capabilities: the row lists no MathForge operations")
    met["capabilities"] = bool(operations) and all(name in facts.operations for name in operations)
    per_operation("checker", "no registered checker", [name for name in operations if name not in facts.checkers])

    adapter = ADAPTERS.get(row.id)
    judged = [_judge(adapter, example) for example in examples] if adapter is not None else []
    problems = [(example, problem) for example, (_, result, problem) in zip(examples, judged)
                if result is not None and problem]
    details.extend(f"json: {example.id}: {problem}" for example, problem in problems)
    if not any(result is not None for _, result, _ in judged):
        details.append("json: no Result to check")
    met["json"] = any(result is not None for _, result, _ in judged) and not problems

    per_operation("mcp", "no MCP tool", [name for name in operations if name not in facts.mcp])
    undocumented = []
    for name in operations:
        public = public_name(name)
        if public not in facts.documented:
            undocumented.append(name if public == name else f"{name} (public name {public})")
    per_operation("docs", "no documentation example", undocumented)
    per_operation("differential", "no SymPy differential test",
                  [name for name in operations if name not in facts.differential])
    unmeasured = [name for name in operations if name not in facts.speed]
    slow = [f"{name} ({facts.speed[name]})" for name in operations
            if name in facts.speed and facts.speed[name] > SPEED_LIMIT]
    if operations and unmeasured:
        details.append(f"speed: no speed ratio: {', '.join(unmeasured)}")
    if slow:
        details.append(f"speed: ratio above {SPEED_LIMIT}: {', '.join(slow)}")
    met["speed"] = bool(operations) and not unmeasured and not slow

    matched = sum(outcome is Outcome.MATCH and result is not None and not problem
                  for outcome, result, problem in judged)
    needed = math.ceil(Fraction(str(DEPTH_THRESHOLD)) * len(examples))  # Exact: 0.8 x 10 must need 8, not 9.
    if adapter is None:
        details.append(f"depth: no adapter for row {row.id}")
    elif not examples:
        details.append("depth: the row has no reference examples")
    elif matched < needed:
        details.append(f"depth: {matched} of {len(examples)} examples match; {needed} needed")
    met["depth"] = adapter is not None and bool(examples) and matched >= needed

    for example, (outcome, result, problem) in zip(examples, judged):
        if outcome is Outcome.MATCH and result is not None and not problem:
            continue
        if outcome is Outcome.MATCH:
            details.append(f"{example.id}: match, not counted: {problem or 'no Result'}")
        elif outcome is Outcome.ERROR and problem:
            details.append(f"{example.id}: error: {problem}")
        else:
            details.append(f"{example.id}: {outcome.value}")

    criteria = {name: met[name] for name in CRITERIA}
    status = "yes" if all(criteria.values()) else "partial" if criteria["capabilities"] else "no"
    return RowVerdict(row.id, criteria, status, matched, len(examples), tuple(details))


def _judge(adapter: Adapter, example: Example) -> tuple[Outcome, Result[Any] | None, str]:
    """The adapter's (outcome, result) for an example, and what is wrong: for ERROR, the failure; for a Result, why it
    fails criterion 3 ("" when it passes)."""
    try:
        reply: object = adapter(example)
    except Exception as error:  # One failing example must not stop the gate's report.
        return Outcome.ERROR, None, f"{type(error).__name__}: {error}"
    if not (isinstance(reply, tuple) and len(reply) == 2 and isinstance(reply[0], Outcome)
            and (reply[1] is None or isinstance(reply[1], Result))):
        return Outcome.ERROR, None, f"the adapter returned {reply!r:.200}, not (Outcome, Result | None)"
    outcome, result = reply
    return outcome, result, "" if result is None else _result_problem(result)


def _result_problem(result: Result[Any]) -> str:
    """Why a Result fails criterion 3: it must survive a JSON round trip unchanged and verify once read back."""
    try:
        decoded = mathforge.loads(mathforge.dumps(result))
    except Exception as error:
        return f"does not survive a JSON round trip ({type(error).__name__}: {error})"
    if decoded != result:
        return "changes in a JSON round trip"
    try:
        verified = mathforge.verify(decoded).verified
    except Exception as error:
        return f"cannot be verified ({type(error).__name__}: {error})"
    return "" if verified else "does not verify"


# ----------------------------------------------------------------------------------------------------------------------
# Comparison with SymPy

def same_value(sympy_srepr: str, candidate: object) -> bool:
    """Whether candidate equals the value that SymPy recorded as sympy_srepr, judged on the SymPy side: structural
    equality first, then, for two expressions, simplify(a - b) == 0. Lists and tuples (and SymPy Tuples) compare
    element by element, dicts key by key (keys structurally). A candidate is a SymPy object, a Python number, bool or
    Fraction, or a container of them: adapters convert MathForge values first, and anything SymPy does not convert
    strictly is unequal.

    The text is SymPy's srepr from the frozen examples file; it is evaluated with SymPy's names and no builtins, the
    inverse that srepr documents (eval(srepr(expr)) == expr). ValueError when it does not evaluate."""
    import sympy
    try:
        expected = eval(sympy_srepr, {"__builtins__": {}, **_sympy_names()})
    except Exception as error:
        raise ValueError(f"cannot rebuild a SymPy value from {sympy_srepr!r}: {error}") from error
    return _same(sympy, expected, candidate)


@functools.cache
def _sympy_names() -> dict[str, Any]:
    import sympy
    return {name: value for name, value in vars(sympy).items() if not name.startswith("_")}


def _same(sympy: Any, expected: object, candidate: object) -> bool:
    sequences = (list, tuple, sympy.Tuple)
    if isinstance(expected, sequences):
        return (isinstance(candidate, sequences) and len(candidate) == len(expected)
                and all(_same(sympy, item, other) for item, other in zip(expected, candidate)))
    if isinstance(expected, dict):
        if not isinstance(candidate, dict) or len(candidate) != len(expected):
            return False
        keyed = {}
        for key, value in candidate.items():
            converted = _sympified(sympy, key)
            if converted is None:
                return False
            keyed[converted] = value
        for key, value in expected.items():
            converted = _sympified(sympy, key)
            if converted is None or converted not in keyed or not _same(sympy, value, keyed[converted]):
                return False
        return True
    if expected is None or isinstance(expected, str):
        return type(candidate) is type(expected) and candidate == expected
    value, reference = _sympified(sympy, candidate), _sympified(sympy, expected)
    if value is None or reference is None:
        return False
    if value == reference:
        return True
    if isinstance(value, sympy.Expr) and isinstance(reference, sympy.Expr):
        return bool(sympy.simplify(reference - value) == 0)
    return False


def _sympified(sympy: Any, value: object) -> Any:
    """value converted strictly by SymPy, or None when SymPy does not convert it (a string, a foreign object)."""
    try:
        return sympy.sympify(value, strict=True)
    except (sympy.SympifyError, TypeError, ValueError):
        return None


# Last: the adapter modules register themselves in ADAPTERS, and they import the names above.
from . import adapters as _adapters  # noqa: E402,F401
