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
7. speed: every operation's worst speed ratio against SymPy is at most SPEED_LIMIT, or, for an operation that the
   spec exempts from the ratio gate, its absolute time budget is met;
8. depth: the row has an adapter, and at least ceil(DEPTH_THRESHOLD x n) of its n reference examples MATCH.

No criterion holds vacuously: criteria 2 and 4-7 need the row to list an operation, like criterion 1. The status of a
row is "yes" when all eight hold, "partial" when criterion 1 holds but not all, and "no" otherwise. An operation name
that capabilities() does not list is reported as "unknown operation: <name>", never just counted as unmet.

The speed file of criterion 7 is JSON that the Phase 1 speed suite writes:

    {"operations": {"<operation name>": <ratio> | {"exempt": "<reason>", "budget_met": <true | false>}, ...}}

An operation is listed with one of two values. A ratio is its worst MathForge-to-SymPy time ratio over its fixtures, a
positive finite number; the criterion holds for the operation when the ratio is at most SPEED_LIMIT. The exemption
object is for an operation that the spec exempts from the ratio gate because SymPy has no counterpart to time it
against (verify, provenance and proof records, the proven is_prime: spec H2), which gets an absolute time budget
instead. It has exactly two keys: "exempt", a non-empty string that says why, and "budget_met", a JSON boolean that
says whether the operation kept its budget; the criterion holds for the operation exactly when budget_met is true.
Other top-level keys are ignored. A missing file, or an operation that the file does not list, leaves criterion 7
unmet; a malformed file (another value, an exemption object with a key missing, another key, an empty reason or a
budget_met that is not a boolean) raises ValueError.

The adapter protocol. An adapter (ADAPTERS maps a row id to it) poses one SymPy reference example to the row's MathForge
operation and judges the answer: adapter(example) -> (outcome, result), where

- MATCH: MathForge computed an exact, verified value that equals SymPy's recorded one (same_value); result is the Result;
- MISMATCH: MathForge computed a verified value that differs from SymPy's; result is the Result;
- UNVERIFIED: MathForge computed a value whose verification does not pass; result is the Result;
- UNEVALUATED: the example cannot be posed to the operation (an input outside its domain, an API it does not compute),
  or MathForge returned no value (invalid input, unsupported, no conclusion); result is None;
- ERROR: the adapter failed; result is None. evaluate_row also records an exception the adapter raises, a reply
  outside the protocol, or an adapter call that exceeds the time budget, as ERROR, and goes on with the next example.

evaluate_row runs each adapter call in a worker thread of its own and waits for it at most EXAMPLE_BUDGET seconds,
measured with the monotonic clock: the adapter executes the example's setup in this process and compares with SymPy,
and neither has a limit of its own. An example over budget is ERROR with the detail "time budget exceeded" (and the
budget in seconds). Its thread cannot be stopped: it runs on in the background, as a daemon thread that does not keep
the process alive, and shares the interpreter with the examples after it. Whether an example that takes about
EXAMPLE_BUDGET seconds finishes in time depends on the machine's load.

Only a MATCH whose Result passes criterion 3's checks counts towards depth. The modules of coverage_gate.adapters
register their adapters in ADAPTERS; this module imports that package last, so importing it makes every adapter
available. SymPy is imported only when same_value or an adapter runs.
"""

import ast
import contextlib
import functools
import itertools
import json
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
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

__all__ = ["CRITERIA", "DEPTH_THRESHOLD", "SPEED_LIMIT", "EXAMPLE_BUDGET", "SpeedExemption", "Facts", "Outcome",
           "Adapter", "ADAPTERS", "RowVerdict", "gather_facts", "evaluate_row", "same_value", "public_name"]

CRITERIA = ("capabilities", "checker", "json", "mcp", "docs", "differential", "speed", "depth")
DEPTH_THRESHOLD = 0.8
SPEED_LIMIT = 3.0
EXAMPLE_BUDGET = 60.0  # Seconds of wall time, on the monotonic clock, that one adapter call may take
_POLL = 0.5  # Seconds between two checks of a running adapter call, so that Ctrl+C reaches the gate on every platform
_RENAMINGS = 720  # same_value tries every renaming of Dummies when there are at most this many (six of one kind)

# The names before ".<public name>(" that make a call in a documentation example a call of an operation (criterion 5).
_QUALIFIERS = frozenset({"mf", "mathforge", "operations", *registry.AREAS})


@dataclass(frozen=True)
class SpeedExemption:
    """The speed-file entry of an operation that the spec exempts from the ratio gate, SymPy having no counterpart to
    time it against (verify, provenance and proof records, the proven is_prime: spec H2): why it is exempt, and whether
    it kept the absolute time budget that it gets instead."""
    reason: str
    budget_met: bool


@dataclass(frozen=True)
class Facts:
    """What criteria 1, 2 and 4-7 compare a row's operations with; gather_facts collects them from a checkout."""
    operations: frozenset[str]    # The operation names of mathforge.capabilities()["operations"]
    checkers: frozenset[str]      # The operations with a registered independent checker
    mcp: frozenset[str]           # The operations that an MCP tool serves
    documented: frozenset[str]    # The public names that the Python examples of documentation/*.md call
    differential: frozenset[str]  # tools/check_sympy.COVERED_OPERATIONS
    speed: Mapping[str, float | SpeedExemption]  # Operation -> worst speed ratio or exemption; empty without a file


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
    with the speed ratios and exemptions of speed_file (none when it is None or does not exist). The module docstring
    gives the rule of each criterion and the speed file's format."""
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
                 differential=frozenset(check_sympy.COVERED_OPERATIONS), speed=_speed_entries(speed_file))


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


def _speed_entries(path: Path | None) -> dict[str, float | SpeedExemption]:
    if path is None or not Path(path).exists():
        return {}
    try:
        document = json.loads(Path(path).read_bytes().decode("utf-8-sig"))
        entries = document.get("operations") if isinstance(document, dict) else None
        if not isinstance(entries, dict):
            raise ValueError('a speed file is an object with an "operations" object')
        return {name: _speed_entry(name, entry) for name, entry in entries.items()}
    except ValueError as error:  # json.JSONDecodeError and UnicodeDecodeError are ValueErrors as well.
        raise ValueError(f"{path}: {error}") from error


def _speed_entry(name: str, entry: object) -> float | SpeedExemption:
    """One operation's value in the speed file, checked strictly: a ratio or an exemption object."""
    if isinstance(entry, dict):
        if set(entry) != {"exempt", "budget_met"}:
            raise ValueError(f'the exemption of {name} must have exactly the keys "exempt" and "budget_met", not '
                             f"{sorted(entry)}")
        reason, met = entry["exempt"], entry["budget_met"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f'the exemption of {name} must say why in a non-empty "exempt" string, not {reason!r}')
        if not isinstance(met, bool):
            raise ValueError(f'"budget_met" of {name} must be true or false, not {met!r}')
        return SpeedExemption(reason, met)
    if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry) or entry <= 0:
        raise ValueError(f"the value of {name} must be a positive finite ratio or an exemption object "
                         f'{{"exempt": ..., "budget_met": ...}}, not {entry!r}')
    return float(entry)


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
    slow: list[str] = []
    over_budget: list[str] = []
    for name in operations:
        entry = facts.speed.get(name)
        if isinstance(entry, SpeedExemption):
            if not entry.budget_met:
                over_budget.append(f"{name} (exempt: {entry.reason})")
        elif entry is not None and entry > SPEED_LIMIT:
            slow.append(f"{name} ({entry})")
    if operations and unmeasured:
        details.append(f"speed: no speed ratio: {', '.join(unmeasured)}")
    if slow:
        details.append(f"speed: ratio above {SPEED_LIMIT}: {', '.join(slow)}")
    if over_budget:
        details.append(f"speed: absolute time budget not met: {', '.join(over_budget)}")
    met["speed"] = bool(operations) and not unmeasured and not slow and not over_budget

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
    fails criterion 3 ("" when it passes). The adapter call has EXAMPLE_BUDGET seconds (see _call_in_budget)."""
    try:
        finished, reply = _call_in_budget(adapter, example)
    except Exception as error:  # One failing example must not stop the gate's report.
        return Outcome.ERROR, None, f"{type(error).__name__}: {error}"
    if not finished:
        return Outcome.ERROR, None, f"time budget exceeded ({EXAMPLE_BUDGET:g} s)"
    if not (isinstance(reply, tuple) and len(reply) == 2 and isinstance(reply[0], Outcome)
            and (reply[1] is None or isinstance(reply[1], Result))):
        return Outcome.ERROR, None, f"the adapter returned {reply!r:.200}, not (Outcome, Result | None)"
    outcome, result = reply
    return outcome, result, "" if result is None else _result_problem(result)


def _call_in_budget(adapter: Adapter, example: Example) -> tuple[bool, object]:
    """(True, the adapter's reply) when the call ends within EXAMPLE_BUDGET seconds of the monotonic clock, else (False,
    None). The call runs in a daemon worker thread of its own, so that an example over budget does not hold up the gate;
    that thread cannot be stopped and finishes in the background. An exception that the adapter raises is raised here,
    in the caller's thread."""
    ended: list[tuple[bool, object]] = []  # (raised, the reply or the exception), once the call is over

    def call() -> None:
        try:
            ended.append((False, adapter(example)))
        except BaseException as error:  # Handed to the caller, which decides; nothing escapes the thread.
            ended.append((True, error))

    worker = threading.Thread(target=call, name=f"coverage-example-{example.id}", daemon=True)
    deadline = time.monotonic() + EXAMPLE_BUDGET
    worker.start()
    while worker.is_alive() and (left := deadline - time.monotonic()) > 0:
        worker.join(min(left, _POLL))
    if not ended:
        return False, None
    raised, value = ended[0]
    if raised and isinstance(value, BaseException):
        raise value
    return True, value


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
    equality first, then structural equality up to a consistent renaming of Dummies, then, for two expressions,
    simplify(a - b) == 0. Lists and tuples (and SymPy Tuples) compare element by element, dicts key by key (keys
    structurally). A candidate is a SymPy object, a Python number, bool or Fraction, or a container of them: adapters
    convert MathForge values first, and anything SymPy does not convert strictly is unequal.

    A Dummy (the variable of an ImageSet's Lambda, of a RootSum, of an Integral) is equal only to itself, and the frozen
    srepr renumbers SymPy's Dummies from 0, so the Dummies of the value rebuilt from it never equal an adapter's. The
    candidate's Dummies are therefore renamed to the reference's, each to a distinct one of the same class and
    assumptions (the name may differ): first in order of first appearance in a preorder walk of the two values, then,
    when that does not make them equal and there are at most _RENAMINGS renamings, in every other way, because the order
    of a sorted container (an Add, a FiniteSet) can follow the Dummies' indices. The simplify fallback compares with the
    candidate renamed in order of first appearance. Nothing is renamed when the two values hold different numbers or
    kinds of Dummies: no renaming is consistent then.

    The text is SymPy's srepr from the frozen examples file; it is evaluated with SymPy's names and no builtins, the
    inverse that srepr documents (eval(srepr(expr)) == expr). ValueError when it does not evaluate."""
    import sympy
    try:
        expected = eval(sympy_srepr, {"__builtins__": {}, **_sympy_names()})
    except Exception as error:
        raise ValueError(f"cannot rebuild a SymPy value from {sympy_srepr!r}: {error}") from error
    if _same(sympy, expected, candidate, simplify=False):
        return True
    first = candidate
    for number, renamed in enumerate(_renamings(sympy, expected, candidate)):
        if _same(sympy, expected, renamed, simplify=False):
            return True
        if number == 0:
            first = renamed
    return _same(sympy, expected, first)


@functools.cache
def _sympy_names() -> dict[str, Any]:
    import sympy
    return {name: value for name, value in vars(sympy).items() if not name.startswith("_")}


def _same(sympy: Any, expected: object, candidate: object, *, simplify: bool = True) -> bool:
    sequences = (list, tuple, sympy.Tuple)
    if isinstance(expected, sequences):
        return (isinstance(candidate, sequences) and len(candidate) == len(expected)
                and all(_same(sympy, item, other, simplify=simplify) for item, other in zip(expected, candidate)))
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
            if (converted is None or converted not in keyed
                    or not _same(sympy, value, keyed[converted], simplify=simplify)):
                return False
        return True
    if expected is None or isinstance(expected, str):
        return type(candidate) is type(expected) and candidate == expected
    value, reference = _sympified(sympy, candidate), _sympified(sympy, expected)
    if value is None or reference is None:
        return False
    if value == reference:
        return True
    if simplify and isinstance(value, sympy.Expr) and isinstance(reference, sympy.Expr):
        return bool(sympy.simplify(reference - value) == 0)
    return False


def _renamings(sympy: Any, expected: object, candidate: object) -> Iterator[object]:
    """candidate with its Dummies consistently renamed to those of expected: each to a distinct Dummy of expected with
    the same class and assumptions. The first renaming pairs the Dummies of a kind in their order of first appearance;
    every other renaming follows when there are at most _RENAMINGS in all. Nothing when no renaming is consistent."""
    targets: dict[tuple[Any, ...], list[Any]] = {}
    sources: dict[tuple[Any, ...], list[Any]] = {}
    for groups, value in ((targets, expected), (sources, candidate)):
        for dummy in _dummies(sympy, value):
            groups.setdefault((type(dummy), tuple(sorted(dummy.assumptions0.items()))), []).append(dummy)
    if not sources or {kind: len(group) for kind, group in targets.items()} != {
            kind: len(group) for kind, group in sources.items()}:
        return
    kinds = list(sources)
    if math.prod(math.factorial(len(sources[kind])) for kind in kinds) <= _RENAMINGS:
        choices: Iterator[tuple[tuple[Any, ...], ...]] = itertools.product(
            *(itertools.permutations(targets[kind]) for kind in kinds))  # The identities come first.
    else:
        choices = iter([tuple(tuple(targets[kind]) for kind in kinds)])
    for choice in choices:
        yield _renamed(sympy, candidate, {old: new for kind, chosen in zip(kinds, choice)
                                          for old, new in zip(sources[kind], chosen)})


def _dummies(sympy: Any, value: object) -> list[Any]:
    """The Dummies of value, each once, in order of first appearance in a preorder walk: a SymPy object through its
    args, a list or tuple item by item, a dict key and value in its order, another object as SymPy converts it."""
    found: dict[Any, None] = {}
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, sympy.Dummy):
            found.setdefault(item, None)
            continue
        if isinstance(item, dict):
            children = [part for pair in item.items() for part in pair]
        elif isinstance(item, (list, tuple)):
            children = list(item)
        elif isinstance(item, sympy.Basic):
            children = list(item.args)
        else:
            converted = _sympified(sympy, item)
            children = [converted] if isinstance(converted, sympy.Basic) else []
        pending.extend(reversed(children))
    return list(found)


def _renamed(sympy: Any, value: object, renaming: Mapping[Any, Any]) -> object:
    """value with the Dummies of renaming replaced throughout (lists, tuples and dicts rebuilt); a value that holds
    none of them unchanged."""
    if isinstance(value, dict):
        return {_renamed(sympy, key, renaming): _renamed(sympy, item, renaming) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_renamed(sympy, item, renaming) for item in value]
        return items if isinstance(value, list) else tuple(items)
    if isinstance(value, sympy.Basic):
        return value.xreplace(renaming)
    converted = _sympified(sympy, value)
    if isinstance(converted, sympy.Basic) and converted.has(*renaming):
        return converted.xreplace(renaming)
    return value


def _sympified(sympy: Any, value: object) -> Any:
    """value converted strictly by SymPy, or None when SymPy does not convert it (a string, a foreign object)."""
    try:
        return sympy.sympify(value, strict=True)
    except (sympy.SympifyError, TypeError, ValueError):
        return None


# Last: the adapter modules register themselves in ADAPTERS, and they import the names above.
from . import adapters as _adapters  # noqa: E402,F401
