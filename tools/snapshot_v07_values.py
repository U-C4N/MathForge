"""Snapshot what v0.7 returns for every speed fixture and edge case: the value corpus of tests/fixtures/v8/values_v07/.

    PYTHONPATH=<v0.7 worktree>/src python tools/snapshot_v07_values.py --areas ntheory,discrete \\
        --out-dir tests/fixtures/v8/values_v07

writes one file per area, ``<out-dir>/<area>.json``, from the module benchmarks/speed/fixtures/<area>.py:

    {entry id: {"value": to_data(value) | {"repr": repr(value)} | null, "error": [class name, code] | null}}

* An entry is a Fixture with variants 0, 1 and 2 (entry id ``<fixture id>#<i>``; a fixture that declares fewer
  variants has fewer entries) or an EdgeCase (entry id ``<operation>/edge/<label>``). ExemptFixtures have no entry.
* A call that returns stores ``to_data(value)``. ``{"repr": repr(value)}`` is stored instead only for a plain builtin
  (an int, float, str, bytes, bool, None, complex, or a tuple, list, set, frozenset or dict of them) that ``to_data``
  rejects, such as a list or a float. The repr of a set or dict follows its iteration order.
* A call that raises stores ``[class name, code]``, where code is the exception's ``code`` attribute or null.
* The key order and layout of a file are fixed (sorted keys, one space of indentation, LF line ends), so a rerun
  against the same sources changes nothing.

mathforge is imported from PYTHONPATH: run the tool against a v0.7 tree, never against the code the corpus is to
judge. The tool refuses when mathforge resolves under this repository's src or when its version is not 0.7.x, so
that a v0.8 tree cannot be snapshotted as if it were v0.7 (``--allow-current-tree`` skips that check, for the tool's
own tests on throwaway fixture packages). The fixture modules come from this repository (``--package`` names another
importable package, for tests). Every entry runs twice and the two outcomes must agree, which catches inputs that
carry random state (build symbols with benchmarks.speed.fixture.symbol, not mathforge.symbol). A file that already
exists with different content is the frozen v0.7 reference, so it is replaced only with ``--force``; nothing is written
when any file is refused.

TODO(v0.9): record the provenance of the snapshot (the version and the source hash of the v0.7 tree) in the corpus
files. v0.8.0 does not: a corpus file is one object that maps entry ids to outcomes, and its readers
(tests/v08/test_values_v07.py checks every key as an entry id) would have to learn a second kind of key in each of the ten
files, which means a --force snapshot of the frozen v0.7 reference. The refusal above is what keeps the files honest
meanwhile.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import functools
import importlib
import json
from pathlib import Path
import re
import sys
import types
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
SRC = REPOSITORY / "src"
DEFAULT_PACKAGE = "benchmarks.speed.fixtures"
V07_VERSION = re.compile(r"0\.7\.\d+")  # 0.7.<patch>, with any suffix: the version of the tree that a snapshot is of.
VARIANTS = 3  # A Fixture contributes its variants 0, 1 and 2.
PLAIN_SCALARS = (bool, int, float, complex, str, bytes, type(None))
PLAIN_SEQUENCES = (tuple, list, set, frozenset)

Call = tuple[Callable[..., object], tuple, dict]  # (public value function, args, kwargs)


class SnapshotError(Exception):
    """A fixture module or one of its entries cannot be snapshotted; the message says which and why."""


def repository_importable() -> None:
    """Put the repository root on sys.path, so that ``benchmarks.speed`` imports. Its src is never added."""
    root = str(REPOSITORY)
    if root not in sys.path:
        sys.path.insert(0, root)


def check_tree(module: types.ModuleType) -> None:
    """Refuse a mathforge that is not a v0.7.x tree outside this repository's src: the corpus is what v0.7 returns, so
    a tree that is this repository's or any other version must not be snapshotted into it."""
    location = Path(module.__file__).resolve()
    if SRC.resolve() in location.parents:
        raise SnapshotError(f"mathforge resolves to {location}, under this repository's src: the corpus is a snapshot "
                            "of v0.7, not of the code it judges; run the tool with PYTHONPATH=<v0.7 worktree>/src")
    version = getattr(module, "__version__", None)
    if not (isinstance(version, str) and V07_VERSION.match(version)):
        raise SnapshotError(f"mathforge {version!r} ({location}) is not a 0.7.x release: the corpus is a snapshot of "
                            "v0.7; run the tool with PYTHONPATH=<v0.7 worktree>/src")


def fixture_classes() -> tuple[Any, Any, Any]:
    repository_importable()
    from benchmarks.speed.fixture import EdgeCase, ExemptFixture, Fixture
    return Fixture, ExemptFixture, EdgeCase


def import_area(package: str, area: str) -> types.ModuleType:
    name = f"{package}.{area}"
    repository_importable()
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and (name == error.name or name.startswith(error.name + ".")):
            # The module itself or one of its parent packages is missing; an import inside the module is not.
            raise SnapshotError(f"no fixture module {name}: {error}") from error
        raise


def module_entries(module: types.ModuleType) -> dict[str, Callable[[], Call]]:
    """Entry id -> a zero-argument builder of (function, args, kwargs), for every entry of a fixture module."""
    fixture_type, exempt_type, edge_type = fixture_classes()
    entries: dict[str, Callable[[], Call]] = {}

    def add(entry_id: str, build: Callable[[], Call]) -> None:
        if entry_id in entries:
            raise SnapshotError(f"{module.__name__}: the entry id {entry_id} occurs twice")
        entries[entry_id] = build

    for name in ("FIXTURES", "EDGE_CASES"):
        if not hasattr(module, name):
            raise SnapshotError(f"{module.__name__} defines no {name}")
    for fixture in module.FIXTURES:
        if isinstance(fixture, fixture_type):
            if type(fixture.variants) is not int or fixture.variants < 1:
                raise SnapshotError(f"{fixture.id}: variants must be a positive integer, not {fixture.variants!r}")
            for i in range(min(VARIANTS, fixture.variants)):
                add(f"{fixture.id}#{i}", functools.partial(fixture.mathforge, i))
        elif not isinstance(fixture, exempt_type):
            raise SnapshotError(f"{module.__name__}.FIXTURES holds a {type(fixture).__name__}, "
                                "not a Fixture or an ExemptFixture")
    for case in module.EDGE_CASES:
        if not isinstance(case, edge_type):
            raise SnapshotError(f"{module.__name__}.EDGE_CASES holds a {type(case).__name__}, not an EdgeCase")
        add(f"{case.operation}/edge/{case.label}", case.mathforge)
    return entries


def is_plain(value: object) -> bool:
    """Whether a value is built from exact builtin scalars and containers only, so that its repr says all of it."""
    pending: list[Any] = [value]
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind in PLAIN_SCALARS:
            continue
        if kind in PLAIN_SEQUENCES:
            pending.extend(item)
        elif kind is dict:
            pending.extend(item)
            pending.extend(item.values())
        else:
            return False
    return True


def encode_value(value: object) -> dict:
    """The stored form of a returned value: its to_data document, or {"repr": ...} for a plain builtin to_data rejects."""
    from mathforge.errors import SerializationError
    from mathforge.serialization import to_data

    try:
        document = to_data(value)
    except SerializationError as error:
        if not is_plain(value):
            raise SnapshotError(f"to_data rejects a {type(value).__name__} ({error}), which is not a plain builtin, "
                                "so the corpus cannot store it") from error
        return {"repr": repr(value)}
    # Through JSON, so that a fresh document compares equal to the stored one whatever containers it was built from.
    return json.loads(json.dumps(document, allow_nan=False))


def outcome(call: Call) -> dict:
    """Run a call and return its stored form: a value, or the class name and code of the exception it raised."""
    function, args, kwargs = call
    try:
        value = function(*args, **kwargs)
    except Exception as error:
        return {"value": None, "error": [type(error).__name__, getattr(error, "code", None)]}
    return {"value": encode_value(value), "error": None}


def run_entry(entry_id: str, build: Callable[[], Call]) -> dict:
    """Build an entry's call afresh and return its outcome; a fixture that cannot be run raises SnapshotError."""
    try:
        call = build()
    except Exception as error:
        raise SnapshotError(f"{entry_id}: the fixture could not build its call: "
                            f"{type(error).__name__}: {error}") from error
    try:
        return outcome(call)
    except SnapshotError as error:
        raise SnapshotError(f"{entry_id}: {error}") from error


def snapshot_area(package: str, area: str) -> dict[str, dict]:
    """The corpus document of one area: entry id -> outcome, each outcome checked to be the same on a second run."""
    document = {}
    for entry_id, build in module_entries(import_area(package, area)).items():
        first = run_entry(entry_id, build)
        if run_entry(entry_id, build) != first:
            raise SnapshotError(f"{entry_id}: two runs gave different results, so the input carries random state "
                                "(build symbols with benchmarks.speed.fixture.symbol)")
        document[entry_id] = first
    return document


def json_text(document: dict) -> str:
    return json.dumps(document, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


def parse_areas(text: str) -> list[str]:
    areas = [area.strip() for area in text.split(",")]
    for area in areas:
        if not area.isidentifier():
            raise SnapshotError(f"--areas needs comma-separated area names, not {area!r}")
    if len(set(areas)) != len(areas):
        raise SnapshotError("--areas names an area twice")
    return areas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--areas", required=True, help="Comma-separated areas, for example ntheory,discrete,stats")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory that receives <area>.json for each area")
    parser.add_argument("--package", default=DEFAULT_PACKAGE,
                        help=f"Importable package that holds the area modules (default {DEFAULT_PACKAGE})")
    parser.add_argument("--force", action="store_true",
                        help="Replace files whose content differs; they are the frozen v0.7 reference")
    parser.add_argument("--allow-current-tree", action="store_true",
                        help="Snapshot whatever mathforge is importable, without the check that it is a 0.7.x tree "
                             "outside this repository's src: for the tool's own tests on throwaway fixture packages")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    try:
        areas = parse_areas(arguments.areas)
        import mathforge
        if not arguments.allow_current_tree:
            check_tree(mathforge)
        print(f"snapshotting mathforge {mathforge.__version__} from {Path(mathforge.__file__).resolve().parent}",
              file=sys.stderr)
        documents = {area: snapshot_area(arguments.package, area) for area in areas}
    except SnapshotError as error:
        raise SystemExit(f"snapshot_v07_values: {error}")
    texts = {area: json_text(document) for area, document in documents.items()}
    paths = {area: arguments.out_dir / f"{area}.json" for area in areas}
    differing = [paths[area] for area in areas
                 if paths[area].is_file() and paths[area].read_text(encoding="utf-8") != texts[area]]
    if differing and not arguments.force:
        # Nothing is written, so a refused run never leaves a half-updated reference.
        raise SystemExit("\n".join([
            "Refusing to replace corpus files whose content differs from this snapshot; nothing was written:",
            *("  " + str(path) for path in differing),
            "They are the frozen v0.7 reference that the corpus test holds later versions to. A difference means that "
            "a fixture changed or that mathforge is not the v0.7 tree it should be; pass --force only for the first."]))
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    for area in areas:
        if not paths[area].is_file() or paths[area].read_text(encoding="utf-8") != texts[area]:
            paths[area].write_text(texts[area], encoding="utf-8", newline="\n")
        print(f"{paths[area]} ({len(documents[area])} entries)")


if __name__ == "__main__":
    main()
