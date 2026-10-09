"""Report acceptance IDs explicitly mapped by test or developer-check docstrings.

This checks traceability, not whether a test passes or proves the promised claim.
Run --strict when every required row has a maintained automated/manual mapping.
The acceptance matrix is local planning material, excluded from source releases.
Use --matrix PATH to select a local copy. A missing default matrix skips this
optional report, while --strict or an explicit missing path fails clearly.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
import re
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
IDENTIFIER = re.compile(r"\b(?:POL|ROOT|RF|SET|LIN|CAL|VER|SER|LIM|REL)-\d{2}\b")
GROUP = re.compile(r"\b(POL|ROOT|RF|SET|LIN|CAL|VER|SER|LIM|REL)-(\d{2}(?:\s*[–/-]\s*(?:\1-)?\d{2})*)")


def identifiers(text):
    result = set(IDENTIFIER.findall(text))
    for prefix, expression in GROUP.findall(text):
        expression = expression.replace(f"{prefix}-", "")
        for part in expression.split("/"):
            boundaries = re.split(r"\s*[–-]\s*", part.strip())
            first, last = int(boundaries[0]), int(boundaries[-1])
            result.update(f"{prefix}-{index:02d}" for index in range(first, last + 1))
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--matrix", type=Path, help="Path to the local acceptance matrix")
    arguments = parser.parse_args()
    matrix = (arguments.matrix.resolve() if arguments.matrix is not None else
              REPOSITORY / "docs/v0.2.0-acceptance.md")
    if not matrix.is_file():
        message = (f"Local acceptance matrix not found: {matrix}. "
                   "Planning docs are not distributed; provide --matrix PATH to check coverage.")
        if arguments.strict or arguments.matrix is not None:
            parser.error(message)
        print(f"SKIPPED: {message}")
        return
    acceptance = matrix.read_text(encoding="utf-8")
    cases = {}
    for line in acceptance.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) > 3 and IDENTIFIER.fullmatch(fields[1]):
            if fields[1] in cases:
                raise SystemExit(f"Duplicate acceptance row: {fields[1]}")
            cases[fields[1]] = fields[2]
    if not cases:
        parser.error(f"No acceptance rows found in {matrix}")
    mapped = defaultdict(list)
    for directory in ("tests", "tools", "benchmarks"):
        for path in (REPOSITORY / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                docstring = ast.get_docstring(node) or ""
                for identifier in identifiers(docstring):
                    relative = path.relative_to(REPOSITORY).as_posix()
                    mapped[identifier].append(f"{relative}:{getattr(node, 'lineno', 1)}")
    for path in (REPOSITORY / ".github" / "workflows").glob("*.yml"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                for identifier in identifiers(line):
                    mapped[identifier].append(f"{path.relative_to(REPOSITORY).as_posix()}:{number}")
    missing = sorted(set(cases) - set(mapped))
    unknown = sorted(set(mapped) - set(cases))
    print(f"Acceptance rows: {len(cases)}; explicitly mapped: {len(set(cases) & set(mapped))}")
    for identifier in missing:
        print(f"MISSING {identifier}: {cases[identifier]}")
    for identifier in unknown:
        print(f"UNKNOWN {identifier}: {', '.join(mapped[identifier])}")
    if arguments.strict and (missing or unknown):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
