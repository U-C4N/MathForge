"""Evaluate the v0.8 coverage gate on the frozen SymPy 1.14 inventory (internal developer tooling).

    .venv-mcp/Scripts/python.exe tools/check_coverage.py INVENTORY [--examples PATH] [--speed PATH] [--report DIR]
        [--gate] [--write-status] [--freeze [--reason TEXT]]

The prechecks come first: the inventory must pass validate(complete=True), every row must have 1 to 20 reference
examples in the examples file (default: reference_examples.json beside the inventory), and the examples file must list
no row that the inventory lacks. Then the frozen-inventory guard: when the freeze manifest
(tools/coverage_gate/freeze.json) exists, the digest of the inventory's frozen fields and of the examples
(coverage_gate.inventory.frozen_digest) must equal the manifest's, or the run stops with
"Frozen inventory changed: <expected> != <found>". Any of these failures exits with status 2 and lists the problems;
nothing is written.

--freeze writes the manifest, {"digest", "rows", "examples", "frozen", "history": [{"digest", "rows", "examples",
"date", "reason"}]}, and nothing else. Once a manifest exists, only --freeze --reason TEXT rewrites it, for a changed
digest: the replaced record (its digest, its row and example counts and the date it was frozen) joins the history with
the reason. A re-freeze exists to correct a recorded error, never to drop a hard example or to shrink a set: when the
new freeze has fewer rows or fewer examples than the one it replaces, a warning on stderr says so.

Otherwise every row is evaluated against the eight criteria of coverage_gate.criteria, and the report goes to --report
(default: coverage/ beside the inventory): report.json holds every row's verdict; report.md the totals, the "yes
without speed" count, the yes share of the frozen denominator, the rows that 50% and 55% need, the criteria, plan-group
and area tables and every row's status with its unmet criteria. --speed names criterion 7's speed file (default:
speed/latest.json beside the inventory; a missing file means no speed facts). --write-status stores each row's computed
status in the inventory's status field, which the digest does not cover. --gate exits with status 1 when less than 50%
of the rows are yes; it counts the frozen inventory only, so it needs the manifest.

Exit status: 0 when the run completes (and, with --gate, the gate passes); 1 when the gate fails; 2 for a usage or
input error, failed prechecks or a changed frozen inventory. The same inputs give the same report bytes, unless an
example's adapter call takes about criteria.EXAMPLE_BUDGET seconds: whether it ends within the budget depends on the
machine's load. Run it with the interpreter that has SymPy 1.14 (.venv-mcp): the adapters compare with SymPy. This is
internal tooling: mathforge never imports it, and the inventory, the examples and the reports stay under the gitignored
docs/.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
REPOSITORY = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))  # Also under `python -I`, which leaves the script's directory off sys.path.

from coverage_gate import criteria  # noqa: E402
from coverage_gate.examples import HIGH, LOW, Example, load_examples  # noqa: E402
from coverage_gate.inventory import (PLAN_GROUPS, Inventory, frozen_digest, load_inventory, validate,  # noqa: E402
                                     write_inventory)

MANIFEST = TOOLS / "coverage_gate" / "freeze.json"
MANIFEST_KEYS = ("digest", "rows", "examples", "frozen", "history")
HISTORY_KEYS = ("digest", "rows", "examples", "date", "reason")  # A replaced record, in the manifest's own key order
COUNTS = ("rows", "examples")
GATE, TARGET = Fraction(1, 2), Fraction(11, 20)  # The release fails below GATE; TARGET leaves a margin.
STATUSES = ("yes", "partial", "no")


def main(argv: Sequence[str] | None = None, *, manifest: Path = MANIFEST, facts: criteria.Facts | None = None,
         today: str | None = None) -> int:
    """The command line. manifest, facts (instead of gather_facts on this checkout) and today (an ISO date for the
    freeze) let tests run the tool on synthetic data."""
    for stream in (sys.stdout, sys.stderr):  # Subtopics are not ASCII; a narrow console must not stop the run.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="check_coverage.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("inventory", type=Path, help="the inventory JSON file")
    parser.add_argument("--examples", type=Path, help="the examples file (default: reference_examples.json beside it)")
    parser.add_argument("--speed", type=Path, help="the speed file (default: speed/latest.json beside it, if present)")
    parser.add_argument("--report", type=Path, help="the report directory (default: coverage/ beside it)")
    parser.add_argument("--gate", action="store_true", help="exit with status 1 when less than 50%% of rows are yes")
    parser.add_argument("--write-status", action="store_true", help="store each row's computed status")
    parser.add_argument("--freeze", action="store_true", help="write the freeze manifest (and nothing else)")
    parser.add_argument("--reason", help="why a frozen inventory is frozen again (with --freeze)")
    arguments = parser.parse_args(argv)
    if arguments.reason is not None and not arguments.freeze:
        parser.error("--reason goes with --freeze")
    if arguments.freeze and (arguments.gate or arguments.write_status):
        parser.error("--freeze writes the manifest only; run without it for the report, --gate and --write-status")
    directory = arguments.inventory.parent
    examples_path = arguments.examples or directory / "reference_examples.json"

    try:
        for path in (arguments.inventory, examples_path):
            if not path.is_file():
                raise ValueError(f"{path}: no such file")
        inventory = load_inventory(arguments.inventory)
        examples = load_examples(examples_path)
        recorded = read_manifest(manifest)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    problems = precheck(inventory, examples)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(f"error: {len(problems)} precheck problem{'s' if len(problems) > 1 else ''}; nothing was written",
              file=sys.stderr)
        return 2
    digest = frozen_digest(inventory, examples)
    if arguments.freeze:
        return freeze(manifest, recorded, digest, inventory, examples, arguments.reason,
                      today or date.today().isoformat())
    if recorded is not None and recorded["digest"] != digest:
        print(f"Frozen inventory changed: {recorded['digest']} != {digest}", file=sys.stderr)
        print(f"error: frozen with {recorded['rows']} rows and {recorded['examples']} examples, now "
              f"{len(inventory.rows)} rows and {_count(examples)} examples; restore the frozen files, or correct a "
              "recorded error with --freeze --reason TEXT", file=sys.stderr)
        return 2
    if arguments.gate and recorded is None:
        print(f"error: --gate counts the frozen inventory only, and {manifest} does not exist; run --freeze first",
              file=sys.stderr)
        return 2

    try:
        if facts is None:
            facts = criteria.gather_facts(REPOSITORY, speed_file=arguments.speed or directory / "speed" / "latest.json")
    except ValueError as error:  # A malformed speed file.
        print(f"error: {error}", file=sys.stderr)
        return 2
    verdicts = [criteria.evaluate_row(row, examples[row.id], facts) for row in inventory.rows]
    report = build_report(arguments.inventory.name, inventory, examples, verdicts, digest,
                          None if recorded is None else str(recorded["frozen"]))
    reports = arguments.report or directory / "coverage"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "report.json").write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    (reports / "report.md").write_bytes(render_markdown(report).encode("utf-8"))
    if arguments.write_status:
        rows = tuple(dataclasses.replace(row, status=verdict.status) for row, verdict in zip(inventory.rows, verdicts))
        if rows != inventory.rows:
            write_inventory(arguments.inventory, dataclasses.replace(inventory, rows=rows))
    totals, denominator = report["totals"], report["denominator"]
    print(f"Coverage: {totals['yes']} of {denominator} rows yes ({_percent(totals['yes'], denominator)}), "
          f"{totals['partial']} partial, {totals['no']} no; {totals['yes_without_speed']} yes without speed. "
          f"Wrote {reports / 'report.md'} and report.json.")
    if arguments.gate and Fraction(totals["yes"], denominator) < GATE:
        print(f"Gate failed: {totals['yes']} of {denominator} rows yes ({_percent(totals['yes'], denominator)}), "
              f"below the 50% gate ({report['thresholds']['0.50']} needed).", file=sys.stderr)
        return 1
    return 0


def precheck(inventory: Inventory, examples: Mapping[str, Sequence[Example]]) -> list[str]:
    """What stops a run before anything is evaluated: the validation problems of a complete inventory, then the rows
    without 1 to HIGH examples, then the example rows that the inventory does not have."""
    problems = list(validate(inventory, complete=True))
    if not inventory.rows:
        problems.append("the inventory has no rows")
    for row in inventory.rows:
        count = len(examples.get(row.id, ()))
        if not 1 <= count <= HIGH:
            problems.append(f"{row.id}: {count} reference examples; every row needs 1 to {HIGH}")
    known = {row.id for row in inventory.rows}
    problems.extend(f"{row_id}: the examples file lists a row that is not in the inventory"
                    for row_id in sorted(set(examples) - known))
    return problems


def read_manifest(path: Path) -> dict[str, Any] | None:
    """The freeze manifest, or None when there is none; ValueError for a malformed one."""
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_bytes().decode("utf-8"))
        if not isinstance(document, dict) or set(document) != set(MANIFEST_KEYS):
            raise ValueError(f"a freeze manifest is an object with exactly {', '.join(MANIFEST_KEYS)}")
        digest, history = document["digest"], document["history"]
        if not (isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")):
            raise ValueError("digest is not a SHA-256 hex digest")
        if not all(_is_count(document[key]) for key in COUNTS):
            raise ValueError("rows and examples are counts")
        if not isinstance(document["frozen"], str):
            raise ValueError("frozen is not a date")
        if not isinstance(history, list) or not all(
                isinstance(entry, dict) and set(entry) == set(HISTORY_KEYS)
                and all(isinstance(entry[key], str) for key in ("digest", "date", "reason"))
                and all(_is_count(entry[key]) for key in COUNTS) for entry in history):
            raise ValueError(f"history is not a list of {{{', '.join(HISTORY_KEYS)}}} records (rows and examples "
                             "counts, the others strings)")
        return document
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def freeze(path: Path, recorded: Mapping[str, Any] | None, digest: str, inventory: Inventory,
           examples: Mapping[str, Sequence[Example]], reason: str | None, today: str) -> int:
    """Write the manifest for the digest: the first freeze takes no reason, a re-freeze needs one and a changed
    digest. A re-freeze keeps the replaced record, counts included, in the history, and warns on stderr when it has
    fewer rows or examples than that record."""
    reason = None if reason is None else reason.strip()
    if recorded is None and reason is not None:
        print("error: --reason records why a frozen inventory is frozen again; the first freeze takes none",
              file=sys.stderr)
        return 2
    history: list[dict[str, str]] = []
    if recorded is not None:
        if not reason:
            print(f"error: the inventory is frozen already ({recorded['digest']}, {recorded['frozen']}); only "
                  "--freeze --reason TEXT freezes it again, to correct a recorded error", file=sys.stderr)
            return 2
        if recorded["digest"] == digest:
            print("error: the frozen digest is unchanged; there is nothing to freeze again", file=sys.stderr)
            return 2
        history = [*recorded["history"], {"digest": recorded["digest"], "rows": recorded["rows"],
                                          "examples": recorded["examples"], "date": recorded["frozen"],
                                          "reason": reason}]
    counts = {"rows": len(inventory.rows), "examples": _count(examples)}
    record = {"digest": digest, **counts, "frozen": today, "history": history}
    _write_atomically(path, json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    fewer = [f"fewer {name} ({recorded[name]} -> {counts[name]})" for name in COUNTS
             if recorded is not None and counts[name] < recorded[name]]
    if fewer:
        print(f"warning: the new freeze has {' and '.join(fewer)} than the one it replaces; a re-freeze corrects a "
              "recorded error and never drops a hard example or shrinks a set", file=sys.stderr)
    again = f"; {len(history)} earlier record{'s' if len(history) > 1 else ''} in its history" if history else ""
    print(f"Froze {record['rows']} rows and {record['examples']} examples: {digest} in {path}{again}.")
    return 0


def build_report(name: str, inventory: Inventory, examples: Mapping[str, Sequence[Example]],
                 verdicts: Sequence[criteria.RowVerdict], digest: str, frozen: str | None) -> dict[str, Any]:
    """report.json: the totals over the denominator (every row of the inventory), the thresholds as row counts, the
    criteria counts and every row's verdict, in inventory order."""
    denominator = len(verdicts)
    totals = {status: sum(verdict.status == status for verdict in verdicts) for status in STATUSES}
    totals["yes_without_speed"] = sum(_yes_without_speed(verdict) for verdict in verdicts)
    return {
        "inventory": name,
        "digest": digest,
        "frozen": frozen,
        "denominator": denominator,
        "examples": _count(examples),
        "short": sum(len(examples[row.id]) < LOW for row in inventory.rows),
        "totals": totals,
        "ratio": totals["yes"] / denominator,
        "thresholds": {"0.50": math.ceil(GATE * denominator), "0.55": math.ceil(TARGET * denominator)},
        "criteria": {name: sum(verdict.criteria[name] for verdict in verdicts) for name in criteria.CRITERIA},
        "rows": [{"id": row.id, "area": row.area, "subtopic": row.subtopic, "plan_group": row.plan_group,
                  "v07_status": row.v07_status, "status": verdict.status, "criteria": dict(verdict.criteria),
                  "matched": verdict.matched, "examples": verdict.examples, "details": list(verdict.details)}
                 for row, verdict in zip(inventory.rows, verdicts)],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    totals, denominator, rows = report["totals"], report["denominator"], report["rows"]
    gate, target = report["thresholds"]["0.50"], report["thresholds"]["0.55"]
    yes = totals["yes"]
    standing = ("below the 50% gate" if yes < gate else "the 50% gate passes, below the 55% target" if yes < target
                else "the 55% target is met")
    frozen = f"frozen {report['frozen']}" if report["frozen"] else "not frozen"
    lines = ["# Coverage gate report", "",
             f"Inventory `{report['inventory']}`, digest `{report['digest']}` ({frozen}).",
             f"Denominator: {denominator} rows ({report['examples']} reference examples, {report['short']} short sets).",
             f"50% needs {gate} yes rows and 55% needs {target}; {yes} {'row is' if yes == 1 else 'rows are'} yes: "
             f"{standing}.", "",
             "| Status | Rows | Share |", "|---|---:|---:|"]
    for label, key in (("yes", "yes"), ("partial", "partial"), ("no", "no"), ("yes without speed", "yes_without_speed")):
        lines.append(f"| {label} | {totals[key]} | {_percent(totals[key], denominator)} |")
    lines += ["", "## Criteria", "", "| Criterion | Rows meeting it |", "|---|---:|"]
    lines += [f"| {name} | {count} |" for name, count in report["criteria"].items()]
    lines += ["", "## Plan groups", "", "| Group | Rows | yes | partial | no |", "|---|---:|---:|---:|---:|"]
    for group in PLAN_GROUPS:  # A complete inventory has no other group.
        members = [row for row in rows if row["plan_group"] == group]
        lines.append(f"| {group} | {len(members)} | " + " | ".join(
            str(sum(row["status"] == status for row in members)) for status in STATUSES) + " |")
    lines += ["", "## Areas", "", "| Area | Rows | yes | partial | no | yes without speed |",
              "|---|---:|---:|---:|---:|---:|"]
    for area in dict.fromkeys(row["area"] for row in rows):  # In the order of first appearance.
        members = [row for row in rows if row["area"] == area]
        counts = [sum(row["status"] == status for row in members) for status in STATUSES]
        without_speed = sum(all(met for name, met in row["criteria"].items() if name != "speed") for row in members)
        lines.append(f"| {_cell(area)} | {len(members)} | " + " | ".join(map(str, counts)) + f" | {without_speed} |")
    lines += ["", "## Rows", "", "| Row | Subtopic | Status | Unmet criteria |", "|---|---|---|---|"]
    for row in rows:
        unmet = ", ".join(name for name, met in row["criteria"].items() if not met)
        lines.append(f"| {row['id']} | {_cell(row['subtopic'])} | {row['status']} | {unmet} |")
    return "\n".join(lines) + "\n"


def _yes_without_speed(verdict: criteria.RowVerdict) -> bool:
    return all(met for name, met in verdict.criteria.items() if name != "speed")


def _count(examples: Mapping[str, Sequence[Example]]) -> int:
    return sum(len(items) for items in examples.values())


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _write_atomically(path: Path, text: str) -> None:
    """UTF-8 without a BOM and with "\\n" line ends, through a temporary file that then replaces the target."""
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


if __name__ == "__main__":
    sys.exit(main())
