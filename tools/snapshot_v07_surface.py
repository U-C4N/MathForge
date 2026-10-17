"""Capture the v0.7 public surface before v0.8 changes anything.

Three reference snapshots are written to tests/fixtures/v8/. api_v7_signatures.json is
{"api_v7": {name: signature}, "values_v7": {name: signature}, "operations": {name: signature}},
where each signature is str(inspect.signature(f)). The api_v7 and values_v7 maps cover the
names in mathforge.api_v7.__all__ and mathforge.values_v7.__all__. The operations map covers
exactly the operation functions of mathforge.operations: its public (no leading underscore)
functions that are defined in mathforge.operations or in mathforge.api_v7, so imported
classes, helpers and type aliases are not listed. mcp_v7_input_schemas.json holds
mathforge.mcp.schemas.input_schemas() and capabilities_v07.json holds mathforge.capabilities().

It ran once, on the unmodified v0.7 sources. The files are the reference that the v0.8 tests
compare against, and they are not regenerated: writing refuses to replace a saved snapshot whose
content differs unless --force is given. There is no check mode: the current surface keeps the
v0.7 one and adds to it (area tools, the catalog, v6 operations), so it differs from the files by
design. The pinned-fixture tests are the live guard of the v0.7 part (test_generated_api,
test_registry and test_mcp_areas in tests/v08).
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
DESTINATION = REPOSITORY / "tests" / "fixtures" / "v8"
OPERATION_MODULES = {"mathforge.operations", "mathforge.api_v7"}
sys.path.insert(0, str(REPOSITORY / "src"))


def signatures(module) -> dict[str, str]:
    return {name: str(inspect.signature(getattr(module, name))) for name in module.__all__}


def operation_signatures(module) -> dict[str, str]:
    result = {}
    for name in dir(module):
        function = getattr(module, name)
        if (not name.startswith("_") and inspect.isfunction(function)
                and function.__module__ in OPERATION_MODULES):
            result[name] = str(inspect.signature(function))
    return result


def surface() -> dict[str, object]:
    """Compute every snapshot document from the importable sources, keyed by file name."""
    import mathforge
    from mathforge import api_v7, operations, values_v7
    from mathforge.mcp.schemas import input_schemas
    return {
        "api_v7_signatures.json": {"api_v7": signatures(api_v7), "values_v7": signatures(values_v7),
                                   "operations": operation_signatures(operations)},
        "mcp_v7_input_schemas.json": input_schemas(),
        "capabilities_v07.json": mathforge.capabilities(),
    }


def saved_text(name: str) -> str | None:
    path = DESTINATION / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite saved snapshots whose content differs (the reference is normally kept)")
    arguments = parser.parse_args()
    texts = {name: json.dumps(document, sort_keys=True, indent=1) + "\n"
             for name, document in surface().items()}
    saved = {name: saved_text(name) for name in texts}
    differing = [name for name, text in texts.items() if saved[name] is not None and saved[name] != text]
    if differing and not arguments.force:
        # Nothing is written, so a refused run never leaves a half-updated reference.
        paths = [(DESTINATION / name).relative_to(REPOSITORY).as_posix() for name in differing]
        raise SystemExit("\n".join([
            "Refusing to overwrite saved snapshots that differ from the current surface; nothing was written:",
            *("  " + path for path in paths),
            "They are the frozen v0.7 reference that the pinned-fixture tests compare against; "
            "pass --force only to replace the reference on purpose."]))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        if saved[name] != text:
            (DESTINATION / name).write_text(text, encoding="utf-8", newline="\n")
        print(DESTINATION / name)


if __name__ == "__main__":
    main()
