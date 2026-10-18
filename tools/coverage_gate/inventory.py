"""The coverage-gate inventory: its rows, the rules they must satisfy, and the digest that freezes them.

The inventory is one local JSON file (docs/v08/sympy_1_14_subtopics.json, gitignored and never committed) with a row for
every SymPy 1.14 subtopic. Curators fill it in during Phase 0 and then freeze it; tools/check_coverage.py evaluates it.
This module is the model layer: it reads and writes the file, checks the rules and computes the digest of the frozen
content. It imports neither sympy nor mathforge, and nothing it writes or returns depends on PYTHONHASHSEED.
"""

import contextlib
import dataclasses
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # For the annotation only: the digest reads an example's fields and never imports the class.
    from .examples import Example

__all__ = ["PLAN_GROUPS", "WAVES", "FROZEN_FIELDS", "PLANNED", "Row", "Inventory", "load_inventory", "write_inventory",
           "validate", "group_counts", "frozen_digest"]

PLAN_GROUPS = ("v07", "partial", "t1", "t2_selected", "t2_flexible", "buffer", "t3_unplanned", "out_of_scope")
WAVES = ("A", "B", "C", "D", "E")
# What the freeze digest covers. note, status and mathforge_operations stay editable after the freeze: they record the
# coverage that later phases reach, not the claim that was frozen.
FROZEN_FIELDS = ("id", "area", "subtopic", "sympy_apis", "sympy_api_paths", "v07_status", "tier",
                 "plan_group", "wave", "family", "missing", "sympy_probe")
PLANNED = {"t1": 85, "partial": 55, "t2_selected": 75}  # buffer (wave E): at most 43


@dataclass(frozen=True)
class Row:
    """One SymPy subtopic. The field order is the key order of a written row."""
    id: str                                  # "NN.NN": the area number, a dot, the subtopic number
    area: str
    subtopic: str
    sympy_apis: str                          # The curator's raw text ("Permutation (cycles, parity)"), never parsed
    sympy_api_paths: tuple[str, ...]         # Full names that resolve in SymPy 1.14
    v07_status: str                          # yes | partial | no
    tier: str | None                         # T1 | T2 | T3, exactly when v07_status is "no"
    note: str
    plan_group: str | None                   # One of PLAN_GROUPS
    wave: str | None                         # One of WAVES
    family: str | None                       # A1..E8, or F-<NAME> for a foundation family
    missing: str                             # What a partial or selected row still lacks
    sympy_probe: str | None                  # The smallest call that works in SymPy 1.14, one expression
    mathforge_operations: tuple[str, ...]    # Registered operation names that cover the row
    status: str                              # The computed coverage status (yes | partial | no)


@dataclass(frozen=True)
class Inventory:
    """meta holds every top-level key but rows, in file order: inventory, created, source, frozen, notes, removed and
    whatever else the file carries. removed lists the rows that validation took out ({id, subtopic, reason}). Treat
    meta as read-only and build a changed inventory with dataclasses.replace."""
    meta: Mapping[str, object]
    rows: tuple[Row, ...]


_ROW_FIELDS = tuple(field.name for field in dataclasses.fields(Row))
_OPTIONAL_TEXT_FIELDS = ("tier", "plan_group", "wave", "family", "sympy_probe")
_TEXT_LIST_FIELDS = ("sympy_api_paths", "mathforge_operations")
# What a row may leave out, and how it reads then: the fields that curation fills in later, which the first inventory
# did not have (sympy_api_paths, wave, missing) or left empty (the rest).
_DEFAULTS = {"sympy_api_paths": (), "mathforge_operations": (), "missing": "", "wave": None, "family": None,
             "plan_group": None, "sympy_probe": None}
_LEGACY_FIELD = "reference_examples"  # Written empty by the first inventory; examples live in their own file now.

_ID_PATTERN = re.compile(r"\d\d\.\d\d", re.ASCII)
_FAMILY_PATTERN = re.compile(r"F-[A-Z]+|[A-E]\d{1,2}", re.ASCII)
_V07_STATUSES = ("yes", "partial", "no")
_TIERS = ("T1", "T2", "T3")
_GROUPS_BY_TIER = {"T1": ("t1",),
                   "T2": ("t2_selected", "t2_flexible", "out_of_scope"),
                   "T3": ("buffer", "t3_unplanned", "out_of_scope")}
_PLANNED_WAVES, _BUFFER_WAVES = WAVES[:-1], WAVES[-1:]  # A-D carry the planned work; E is the buffer.
_WAVES_BY_GROUP = {"partial": _PLANNED_WAVES, "t1": _PLANNED_WAVES, "t2_selected": _PLANNED_WAVES,
                   "buffer": _BUFFER_WAVES}  # Every other group takes no wave.
_GROUPS_NEEDING_MISSING = ("partial", "t2_selected")
_UNASSIGNED = "unassigned"  # The group_counts key of the rows that have no plan_group yet.


def load_inventory(path: Path) -> Inventory:
    """Read an inventory. A row may leave out the fields that curation fills in later (sympy_api_paths,
    mathforge_operations, missing, wave, family, plan_group, sympy_probe): they read as empty, and so does an absent
    meta.removed. The legacy reference_examples field is accepted when empty and dropped. Any other unknown field, any
    other missing field and a value of the wrong type raise ValueError that names the file and the row, so that writing
    the inventory back can never lose data silently."""
    path = Path(path)
    try:
        document = json.loads(path.read_bytes().decode("utf-8-sig"))
        if not isinstance(document, dict) or not isinstance(document.get("rows"), list):
            raise ValueError("an inventory is an object with a 'rows' list")
        meta = {key: value for key, value in document.items() if key != "rows"}
        if not isinstance(meta.setdefault("removed", []), list):
            raise ValueError("meta 'removed' must be a list")
        return Inventory(meta, tuple(_read_row(item, index) for index, item in enumerate(document["rows"])))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _read_row(item: object, index: int) -> Row:
    if not isinstance(item, dict):
        raise ValueError(f"row #{index}: expected an object, not {type(item).__name__}")
    name = item["id"] if isinstance(item.get("id"), str) else f"#{index}"  # A row without a usable id has its position.
    unknown = sorted(set(item) - set(_ROW_FIELDS) - {_LEGACY_FIELD})
    if unknown:
        raise ValueError(f"row {name}: unknown field{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}")
    if item.get(_LEGACY_FIELD, []) != []:
        raise ValueError(f"row {name}: {_LEGACY_FIELD} is not empty; the examples belong in reference_examples.json")
    values: dict[str, Any] = {}
    for field in _ROW_FIELDS:
        if field in item:
            values[field] = _typed(field, item[field], name)
        elif field in _DEFAULTS:
            values[field] = _DEFAULTS[field]
        else:
            raise ValueError(f"row {name}: missing field {field}")
    return Row(**values)


def _typed(field: str, value: object, name: str) -> object:
    if field in _TEXT_LIST_FIELDS:
        if isinstance(value, list) and all(isinstance(entry, str) for entry in value):
            return tuple(value)
        expected = "a list of strings"
    elif field in _OPTIONAL_TEXT_FIELDS:
        if value is None or isinstance(value, str):
            return value
        expected = "a string or null"
    else:
        if isinstance(value, str):
            return value
        expected = "a string"
    raise ValueError(f"row {name}: {field} must be {expected}")


def write_inventory(path: Path, inventory: Inventory) -> None:
    """Write the canonical file: meta keys first in their order (removed last when it was absent), then the rows with
    their keys in Row field order, indent 1, UTF-8 without a BOM, "\\n" line ends and a closing newline. The legacy
    reference_examples field is not written. The text goes to a temporary file beside the target, which then replaces
    it, so a crash leaves either the old inventory or the new one, never half of one."""
    path = Path(path)
    meta = dict(inventory.meta)
    meta.setdefault("removed", [])
    rows = [{name: getattr(row, name) for name in _ROW_FIELDS} for row in inventory.rows]
    text = json.dumps({**meta, "rows": rows}, ensure_ascii=False, indent=1) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:  # Bytes, so Windows does not turn "\n" into "\r\n".
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def validate(inventory: Inventory, *, complete: bool) -> list[str]:
    """The rule violations of the inventory, as messages that start with the row id, in row order (the shape rules of a
    row before its completeness rules). An empty list means the inventory passes.

    The shape rules always apply: ids are unique and look like NN.NN; the area name and the id prefix pair off one to
    one; v07_status is yes, partial or no; a tier (T1, T2 or T3) is present exactly when v07_status is no.
    complete=True adds what a finished inventory needs: a plan group that fits the status and tier, the wave of that
    group, a family for every wave, a missing text where work is planned on a partial or selected row, and the SymPy
    api paths and probe on every row."""
    problems: list[str] = []
    occurrences = Counter(row.id for row in inventory.rows)
    duplicates_reported: set[str] = set()
    first_with_prefix: dict[str, Row] = {}
    first_with_area: dict[str, Row] = {}
    for row in inventory.rows:
        well_formed = _ID_PATTERN.fullmatch(row.id) is not None
        if not well_formed:
            problems.append(f"{row.id}: id must look like NN.NN")
        if occurrences[row.id] > 1 and row.id not in duplicates_reported:
            duplicates_reported.add(row.id)
            problems.append(f"{row.id}: duplicate id ({occurrences[row.id]} rows)")
        if well_formed:
            earlier = first_with_prefix.setdefault(row.id[:2], row)
            if earlier.area != row.area:
                problems.append(f"{row.id}: area {row.area!r} differs from {earlier.area!r} of {earlier.id}, "
                                "which has the same id prefix")
            earlier = first_with_area.setdefault(row.area, row)
            if earlier.id[:2] != row.id[:2]:
                problems.append(f"{row.id}: area {row.area!r} is also the area of {earlier.id}, "
                                "which has another id prefix")
        problems.extend(_status_problems(row))
        if complete:
            problems.extend(_completeness_problems(row))
    return problems


def _status_problems(row: Row) -> list[str]:
    found = []
    if row.v07_status not in _V07_STATUSES:
        found.append(f"{row.id}: v07_status must be yes, partial or no, not {row.v07_status!r}")
    if row.v07_status == "no":
        if row.tier not in _TIERS:
            found.append(f"{row.id}: v07_status 'no' needs a tier of T1, T2 or T3, not {row.tier!r}")
    elif row.tier is not None:
        found.append(f"{row.id}: tier {row.tier!r} is only for v07_status 'no', not {row.v07_status!r}")
    return found


def _completeness_problems(row: Row) -> list[str]:
    found = []
    allowed = _allowed_groups(row)
    if allowed is not None and row.plan_group not in allowed:  # Without a valid status and tier nothing can be said.
        basis = f"v07_status {row.v07_status!r}" + (f", tier {row.tier!r}" if row.v07_status == "no" else "")
        found.append(f"{row.id}: plan_group {row.plan_group!r} does not fit {basis}; expected {_either(allowed)}")
    if row.plan_group in PLAN_GROUPS:  # A missing or unknown group has been reported already.
        waves = _WAVES_BY_GROUP.get(row.plan_group, ())
        if not (row.wave in waves if waves else row.wave is None):
            expected = _either(waves) if waves else "no wave"
            found.append(f"{row.id}: wave {row.wave!r} does not fit plan_group {row.plan_group!r}; expected {expected}")
    if row.wave is not None and (row.family is None or _FAMILY_PATTERN.fullmatch(row.family) is None):
        found.append(f"{row.id}: a row with wave {row.wave!r} needs a family like A3, B13 or F-INT, not {row.family!r}")
    if row.plan_group in _GROUPS_NEEDING_MISSING and not row.missing.strip():
        found.append(f"{row.id}: plan_group {row.plan_group!r} needs a non-empty missing")
    if not row.sympy_api_paths:
        found.append(f"{row.id}: sympy_api_paths is empty")
    if row.sympy_probe is None or not row.sympy_probe.strip():
        found.append(f"{row.id}: sympy_probe is empty")
    return found


def _allowed_groups(row: Row) -> tuple[str, ...] | None:
    """The plan groups that the v0.7 status and the tier allow; None when they are not valid themselves."""
    if row.v07_status == "yes":
        return ("v07",)
    if row.v07_status == "partial":
        return ("partial",)
    if row.v07_status == "no" and row.tier is not None:
        return _GROUPS_BY_TIER.get(row.tier)  # None for an unknown tier
    return None


def _either(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def group_counts(inventory: Inventory) -> dict[str, int]:
    """Rows per plan group: every group of PLAN_GROUPS in its order (0 when empty), then "unassigned" for the rows with
    no plan_group yet, then any group name outside PLAN_GROUPS in name order, so a mistyped group stays visible and the
    counts always add up to the number of rows."""
    seen = Counter(_UNASSIGNED if row.plan_group is None else row.plan_group for row in inventory.rows)
    counts = {name: seen.pop(name, 0) for name in (*PLAN_GROUPS, _UNASSIGNED)}
    counts.update(sorted(seen.items()))
    return counts


def frozen_digest(inventory: Inventory, examples: Mapping[str, Sequence["Example"]]) -> str:
    """The SHA-256 (hex) that freezes the inventory and its example sets.

    It is taken over the UTF-8 bytes of the canonical JSON (sort_keys, ensure_ascii=False, separators "," and ":") of
    {"rows": [...], "examples": {...}}: the rows in id order, each reduced to FROZEN_FIELDS, and the examples by row id
    (a row with no examples is left out), each row's examples in id order with every field they have. Row order, example
    order, meta, note, status and mathforge_operations do not matter. An example is a dataclass instance or a mapping.
    """
    ordered = sorted(inventory.rows, key=lambda row: row.id)
    rows = [{name: getattr(row, name) for name in FROZEN_FIELDS} for row in ordered]
    sets = {row_id: sorted((_example_data(example) for example in group), key=lambda data: str(data["id"]))
            for row_id, group in examples.items() if group}
    text = json.dumps({"rows": rows, "examples": sets}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _example_data(example: object) -> dict[str, object]:
    if dataclasses.is_dataclass(example) and not isinstance(example, type):
        return dataclasses.asdict(example)
    if isinstance(example, Mapping):
        return dict(example)
    raise TypeError(f"an example is a dataclass instance or a mapping, not {type(example).__name__}")
