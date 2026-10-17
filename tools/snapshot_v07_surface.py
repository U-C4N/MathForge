"""Capture, or check, the v0.7 public surface before v0.8 changes anything.

Three reference snapshots are written to tests/fixtures/v8/. api_v7_signatures.json is
{"api_v7": {name: signature}, "values_v7": {name: signature}, "operations": [name, ...]}:
each signature is str(inspect.signature(f)) for a name in mathforge.api_v7.__all__ or
mathforge.values_v7.__all__, and the operations list holds the sorted callable public
names of mathforge.operations (every name without a leading underscore for which
callable() is true, so re-exported classes and helpers are listed with the operation
functions). mcp_v7_input_schemas.json holds mathforge.mcp.schemas.input_schemas() and
capabilities_v07.json holds mathforge.capabilities().

Run it once, on the unmodified v0.7 sources. Afterwards the files are the reference that
v0.8 tests compare against, and they are not regenerated. --check recomputes the values
and fails, showing what differs, when they do not match the files on disk.
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
DESTINATION = REPOSITORY / "tests" / "fixtures" / "v8"
MAXIMUM_DIFF_LINES = 40
sys.path.insert(0, str(REPOSITORY / "src"))


def signatures(module) -> dict[str, str]:
    return {name: str(inspect.signature(getattr(module, name))) for name in module.__all__}


def public_callables(module) -> list[str]:
    return sorted(name for name in dir(module)
                  if not name.startswith("_") and callable(getattr(module, name)))


def surface() -> dict[str, object]:
    """Compute every snapshot document from the importable sources, keyed by file name."""
    import mathforge
    from mathforge import api_v7, operations, values_v7
    from mathforge.mcp.schemas import input_schemas
    return {
        "api_v7_signatures.json": {"api_v7": signatures(api_v7), "values_v7": signatures(values_v7),
                                   "operations": public_callables(operations)},
        "mcp_v7_input_schemas.json": input_schemas(),
        "capabilities_v07.json": mathforge.capabilities(),
    }


def saved_text(name: str) -> str | None:
    path = DESTINATION / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def difference(name: str, saved: str | None, current: str) -> str:
    if saved is None:
        return f"{name}: the saved snapshot is missing"
    lines = list(difflib.unified_diff(saved.splitlines(), current.splitlines(),
                                      f"{name} (saved)", f"{name} (current)", lineterm="", n=0))
    if len(lines) > MAXIMUM_DIFF_LINES:
        lines[MAXIMUM_DIFF_LINES:] = [f"... {len(lines) - MAXIMUM_DIFF_LINES} more diff lines"]
    return "\n".join([f"{name}: the current surface differs from the saved snapshot", *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Write nothing; fail if the recomputed surface differs from the saved snapshots")
    arguments = parser.parse_args()
    texts = {name: json.dumps(document, sort_keys=True, indent=1) + "\n"
             for name, document in surface().items()}
    saved = {name: saved_text(name) for name in texts}
    if arguments.check:
        problems = [difference(name, saved[name], text) for name, text in texts.items() if saved[name] != text]
        if problems:
            raise SystemExit("\n".join(problems))
        print("v0.7 surface snapshot matches")
        return
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        if saved[name] != text:
            (DESTINATION / name).write_text(text, encoding="utf-8", newline="\n")
        print(DESTINATION / name)


if __name__ == "__main__":
    main()
