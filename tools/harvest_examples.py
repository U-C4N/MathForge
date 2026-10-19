"""Harvest SymPy 1.14 reference examples for the coverage-gate inventory (internal developer tooling).

    .venv-mcp/Scripts/python.exe tools/harvest_examples.py INVENTORY --out EXAMPLES [--rows 27.14,27.15]
        [--doc-src DIR] [--workers N] [--curation FILE] [--report PATH]

For each chosen row (default: every row) the harvest collects candidates from the docstrings of the row's APIs, then
from the SymPy 1.14 guide sources under --doc-src (optional), then, only for a set that stays below ten, from the
equality asserts of SymPy's installed test modules. Every candidate runs in a sandboxed worker at PYTHONHASHSEED=0 with
a 60 s timeout, and what cannot serve as a reference is excluded with its reason. At most twenty examples are picked,
in turn across the row's APIs, and each of them runs again at PYTHONHASHSEED=1 and at PYTHONHASHSEED=2, each time in a
worker process started for it alone: an example whose record differs is excluded as unstable, so that no record
depends on the hash seed or on the examples that ran before it. The curation file (default
docs/v08/harvest/curation.json; absent means none) then drops and adds examples per row (an addition is verified the
same way; a drop for the reason sympy-incorrect, because the output SymPy 1.14 recorded is mathematically wrong, must
carry an evidence string); the final ids are <row>/01, <row>/02, ... in that order.

EXAMPLES keeps every row this run did not harvest and gets the harvested ones replaced. The report (default
docs/v08/harvest/report.md) gives per row the candidate, selected and excluded counts, every selected and excluded
example, the curation applied (each drop with its reason, a sympy-incorrect drop with its evidence too), the short rows
(fewer than ten examples) and the API paths that do not resolve. Neither file holds a time or a thread or process
detail, and no record depends on the number of workers; which examples are selected is decided by the runs at
PYTHONHASHSEED=0 in long-lived workers (coverage_gate.examples states what is and is not guaranteed). The exclusion
counts can vary by a few between runs, because whether an example near the timeout finishes depends on the machine's
load; the report's header says so.

Exit status 0 when both files are written; 2 for a usage, input or curation error, or when the harvester's SymPy is not
1.14; 1 when the harvest fails (a traceback and a message go to stderr); 130 when interrupted. Only status 0 means the
files were written. Run it with the interpreter that has SymPy 1.14 (.venv-mcp); the workers run without
any PYTHON*, SYMPY* or MPMATH* environment variable but PYTHONHASHSEED. This is internal tooling: mathforge never imports
it.
"""

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPOSITORY = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))  # Also under `python -I`, which leaves the script's directory off sys.path.

from coverage_gate import examples as harvesting  # noqa: E402
from coverage_gate.inventory import Row, load_inventory  # noqa: E402

DEFAULT_CURATION = Path("docs/v08/harvest/curation.json")  # Relative to the repository
DEFAULT_REPORT = Path("docs/v08/harvest/report.md")
TIMEOUT = 60.0
OUTPUT_WIDTH = 120  # Recorded outputs longer than this are shortened in the report (never in the examples file).


@dataclass(frozen=True)
class RowResult:
    row: Row
    unresolved: tuple[tuple[str, str], ...]
    candidates: int                                  # Every candidate the harvest examined
    examples: tuple[harvesting.Example, ...]         # Final, numbered
    excluded: tuple[harvesting.Excluded, ...]
    notes: tuple[tuple[str, str, str, str, str], ...]  # Curation: (action, source, code, reason or output, evidence)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Subtopics are not ASCII; a narrow console must not stop the run.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="harvest_examples.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("inventory", type=Path, help="the inventory JSON file")
    parser.add_argument("--out", type=Path, required=True, help="the examples file to write (rows outside the run stay)")
    parser.add_argument("--rows", help="comma-separated row ids, e.g. 27.14,27.15 (default: every row)")
    parser.add_argument("--doc-src", type=Path, help="SymPy 1.14 doc/src directory with the guide sources")
    parser.add_argument("--workers", type=_positive, default=1, help="rows harvested in parallel (default 1)")
    parser.add_argument("--curation", type=Path, help=f"curation file (default {DEFAULT_CURATION.as_posix()} if present)")
    parser.add_argument("--report", type=Path, help=f"report path (default {DEFAULT_REPORT.as_posix()})")
    arguments = parser.parse_args(argv)
    report_path = arguments.report if arguments.report is not None else REPOSITORY / DEFAULT_REPORT
    started = time.monotonic()
    try:
        harvesting.require_sympy()  # The workers check their own SymPy as they start.
        inventory = load_inventory(arguments.inventory)
        rows = _chosen_rows(inventory.rows, arguments.rows)
        curation, curation_label = _curation(arguments.curation, inventory.rows)
        existing = harvesting.load_examples(arguments.out) if arguments.out.exists() else {}
        if arguments.doc_src is not None and not arguments.doc_src.is_dir():
            raise ValueError(f"{arguments.doc_src}: --doc-src is not a directory")
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        return _harvest_and_write(arguments, rows, curation, curation_label, existing, report_path, started)
    except KeyboardInterrupt:
        print("error: interrupted; the examples file and the report were not written", file=sys.stderr)
        return 130
    except Exception as error:  # Whatever fails, the run never ends silently, and never with status 0.
        traceback.print_exception(error, file=sys.stderr)
        print(f"error: the harvest failed ({type(error).__name__}: {error})", file=sys.stderr)
        return 1


def _harvest_and_write(arguments: argparse.Namespace, rows: Sequence[Row], curation: Mapping[str, Mapping],
                       curation_label: str, existing: Mapping[str, Sequence[harvesting.Example]], report_path: Path,
                       started: float) -> int:
    results, versions, failures = _harvest(rows, doc_src=arguments.doc_src, curation=curation,
                                           workers=arguments.workers)
    if failures:
        for failure in failures:
            traceback.print_exception(failure, file=sys.stderr)
        print(f"error: a harvest thread failed ({type(failures[0]).__name__}: {failures[0]}); nothing was written",
              file=sys.stderr)
        return 1
    missing = [row.id for row in rows if row.id not in results]
    if missing:
        print(f"error: no result for rows {', '.join(missing)}; nothing was written", file=sys.stderr)
        return 1
    errors = [results[row.id] for row in rows if isinstance(results[row.id], harvesting.CurationError)]
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print("error: curation could not be applied; nothing was written", file=sys.stderr)
        return 2

    finished = [results[row.id] for row in rows]
    merged = {**existing, **{row.id: list(result.examples) for row, result in zip(rows, finished)}}
    text = render_report(inventory=arguments.inventory, out=arguments.out, doc_src=arguments.doc_src,
                         curation=curation_label, sympy=", ".join(sorted(versions)) or "unknown", results=finished)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    harvesting.write_examples(arguments.out, merged)  # Atomic: the old file stays until the new one is complete.
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(text.encode("utf-8"))
    except OSError as error:
        print(f"error: wrote {arguments.out} but not the report: {error}", file=sys.stderr)
        return 1
    selected = sum(len(result.examples) for result in finished)
    short = sum(len(result.examples) < harvesting.LOW for result in finished)
    _say(f"Harvested {len(rows)} rows in {time.monotonic() - started:.1f} s: {selected} examples, {short} short rows, "
         f"{sum(len(result.excluded) for result in finished)} exclusions. Wrote {arguments.out} and {report_path}.")
    return 0


def _say(text: str) -> None:
    """Progress for the person at the console; a closed or broken stdout must not stop the harvest."""
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _chosen_rows(rows: Sequence[Row], text: str | None) -> list[Row]:
    if text is None:
        return list(rows)
    wanted = [name.strip() for name in text.split(",") if name.strip()]
    if not wanted:
        raise ValueError("--rows names no row")
    known = {row.id for row in rows}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(f"--rows names rows that are not in the inventory: {', '.join(unknown)}")
    chosen = set(wanted)
    return [row for row in rows if row.id in chosen]  # Inventory order


def _curation(path: Path | None, rows: Sequence[Row]) -> tuple[dict, str]:
    if path is None:
        label = DEFAULT_CURATION.as_posix()
        path = REPOSITORY / DEFAULT_CURATION
        if not path.exists():
            return {}, f"none ({label} not found)"
    else:
        label = path.as_posix()
    entries = harvesting.load_curation(path)
    unknown = sorted(set(entries) - {row.id for row in rows})
    if unknown:
        raise harvesting.CurationError(f"{path}: rows that are not in the inventory: {', '.join(unknown)}")
    return entries, f"`{label}` ({len(entries)} rows)"


def _harvest(rows: Sequence[Row], *, doc_src: Path | None, curation: Mapping[str, Mapping],
             workers: int) -> tuple[dict[str, "RowResult | harvesting.CurationError"], set[str], list[BaseException]]:
    """Harvest the rows on `workers` threads, each with a long-lived runner at hash seed 0 and FreshRunners at hash
    seeds 1 and 2 that verify every selected example, each in a worker process of its own. Returns the results by row,
    the SymPy versions the workers reported and the exceptions that stopped threads (then the other threads stop after
    their current row, and some rows have no result)."""
    pending: queue.Queue = queue.Queue()
    for row in rows:
        pending.put(row)
    results: dict[str, RowResult | harvesting.CurationError] = {}
    versions: set[str] = set()
    failures: list[BaseException] = []
    lock = threading.Lock()

    def work() -> None:
        try:
            with harvesting.ExampleRunner(hash_seed=0, timeout=TIMEOUT) as first, \
                    harvesting.FreshRunner(hash_seed=1, timeout=TIMEOUT) as second, \
                    harvesting.FreshRunner(hash_seed=2, timeout=TIMEOUT) as third:
                check = (second, third)
                with lock:
                    versions.add(first.sympy_version or "unknown")
                while not failures:
                    try:
                        row = pending.get_nowait()
                    except queue.Empty:
                        return
                    began = time.monotonic()
                    result = _harvest_row(row, first, check, doc_src, curation.get(row.id))
                    with lock:
                        results[row.id] = result
                        progress = _progress(len(results), len(rows), row, result, time.monotonic() - began)
                    _say(progress)
        except BaseException as error:  # Reported by the main thread, with its traceback.
            with lock:
                failures.append(error)

    threads = [threading.Thread(target=work, name=f"harvest-{number}", daemon=True)
               for number in range(min(workers, len(rows)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, versions, failures


def _harvest_row(row: Row, first: harvesting.ExampleRunner, check: tuple[harvesting.FreshRunner, ...],
                 doc_src: Path | None, entry: Mapping | None) -> "RowResult | harvesting.CurationError":
    candidates, excluded = harvesting.harvest(row, first, doc_src=doc_src, check_runner=check)
    selected = harvesting.select(candidates, apis=harvesting.resolvable_api_paths(row))
    try:
        examples, notes = harvesting.curate(row, selected, entry, first, check)
    except harvesting.CurationError as error:
        return error
    return RowResult(row, tuple(harvesting.unresolved_apis(row)), len(candidates) + len(excluded), tuple(examples),
                     tuple(excluded), tuple(notes))


def _progress(done: int, total: int, row: Row, result: "RowResult | harvesting.CurationError", seconds: float) -> str:
    prefix = f"[{done}/{total}] {row.id} {row.subtopic}"
    if isinstance(result, harvesting.CurationError):
        return f"{prefix}: curation error ({seconds:.1f} s)"
    short = " (short)" if len(result.examples) < harvesting.LOW else ""
    return f"{prefix}: {len(result.examples)} selected{short}, {len(result.excluded)} excluded ({seconds:.1f} s)"


# ----------------------------------------------------------------------------------------------------------------------
# The report

def render_report(*, inventory: Path, out: Path, doc_src: Path | None, curation: str, sympy: str,
                  results: Sequence[RowResult]) -> str:
    """The harvest report as Markdown. It prints no time and no thread or process detail, so runs that make the same
    decisions give the same report. Codes, outputs and evidence appear as JSON strings, ready to be copied into a
    curation file."""
    lines = ["# SymPy reference example harvest", "",
             f"- Inventory: `{inventory.as_posix()}`",
             f"- Examples file: `{out.as_posix()}` (the rows of this report were replaced; other rows were kept)",
             f"- SymPy: {sympy}",
             f"- Guide sources: `{doc_src.as_posix()}`" if doc_src is not None
             else "- Guide sources: not used (--doc-src not given)",
             f"- Curation: {curation}",
             f"- Rows harvested: {len(results)}",
             f"- Every candidate ran at PYTHONHASHSEED=0 with a {TIMEOUT:g} s timeout; each selected one ran again at "
             "PYTHONHASHSEED=1 and 2, each time in a worker process started for it alone, and had to give the same "
             "record.",
             f"- Exclusion counts can vary by a few between runs: whether an example near the {TIMEOUT:g} s timeout "
             "finishes depends on the machine's load. Each selected record, by contrast, was verified in fresh "
             "processes.",
             ""]
    reasons = Counter(item.reason for result in results for item in result.excluded)
    short = [result for result in results if len(result.examples) < harvesting.LOW]
    lines += ["## Summary", "", "| | Count |", "|---|---:|",
              f"| Candidates | {sum(result.candidates for result in results)} |",
              f"| Selected | {sum(len(result.examples) for result in results)} |",
              f"| Excluded | {sum(len(result.excluded) for result in results)} |",
              f"| Short rows (fewer than {harvesting.LOW} examples) | {len(short)} |", "",
              "| Exclusion reason | Excluded |", "|---|---:|"]
    lines += [f"| {reason} | {reasons.get(reason, 0)} |" for reason in harvesting.EXCLUSION_REASONS]
    lines += ["", f"## Short rows (fewer than {harvesting.LOW} examples)", ""]
    lines += [f"- {result.row.id} {result.row.subtopic}: {len(result.examples)}" for result in short] or ["None."]
    lines += ["", "## API paths that do not resolve", ""]
    unresolved = [f"- {result.row.id} `{path}`: {reason}" if path else f"- {result.row.id}: {reason}"
                  for result in results for path, reason in result.unresolved]
    lines += unresolved or ["None."]
    lines += ["", "## Rows", "",
              "| Row | Subtopic | Candidates | Selected | Excluded | Docstring | Guide | Tests | Short |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for result in results:
        sources = Counter(item.source.partition(":")[0] for item in result.examples)
        lines.append(f"| {result.row.id} | {_cell(result.row.subtopic)} | {result.candidates} | "
                     f"{len(result.examples)} | {len(result.excluded)} | {sources['docstring']} | {sources['guide']} | "
                     f"{sources['tests']} | {'short' if len(result.examples) < harvesting.LOW else ''} |")
    lines += ["", "## Row details"]
    for result in results:
        lines += ["", f"### {result.row.id} {result.row.subtopic}", "",
                  f"Candidates {result.candidates}, selected {len(result.examples)}, excluded {len(result.excluded)}.",
                  ""]
        lines += ["Selected:", ""]
        lines += [f"1. `{item.id}` `{item.source}` {_code(item.code)} → {_code(_shortened(item.output))}"
                  + (" (numeric)" if item.kind == "numeric" else "") for item in result.examples] or ["None."]
        if result.excluded:
            lines += ["", "Excluded:", ""]
            lines += [f"- {item.reason}: `{item.source}` {_code(item.code)}" for item in result.excluded]
        if result.notes:
            lines += ["", "Curation:", ""]
            lines += [_curation_line(*note) for note in result.notes]
        if result.unresolved:
            lines += ["", "Unresolved API paths:", ""]
            lines += [f"- `{path}`: {reason}" if path else f"- {reason}" for path, reason in result.unresolved]
    return "\n".join(lines) + "\n"


def _curation_line(action: str, source: str, code: str, detail: str, evidence: str) -> str:
    """One applied curation: a drop with its reason, and with its evidence when it has one (shown whole: it is what
    justifies the drop), an addition with the output SymPy recorded for it."""
    if action != "drop":
        return f"- added: `{source}` {_code(code)} → {_code(_shortened(detail))}"
    line = f"- dropped ({detail}): `{source}` {_code(code)}"
    return f"{line}; evidence: {_code(evidence)}" if evidence else line


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _shortened(text: str) -> str:
    return text if len(text) <= OUTPUT_WIDTH else text[:OUTPUT_WIDTH - 3] + "..."


def _code(text: str) -> str:
    """A JSON string literal in a Markdown code span: unambiguous, one line, ready to paste into a curation file."""
    literal = json.dumps(text, ensure_ascii=False)
    fence = "``" if "`" in literal else "`"
    padding = " " if fence == "``" else ""
    return f"{fence}{padding}{literal}{padding}{fence}"


if __name__ == "__main__":
    sys.exit(main())
