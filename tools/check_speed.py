"""The v0.8 speed gate: every operation within the limit of its fastest public SymPy call (internal developer tooling).

    .venv-mcp/Scripts/python.exe tools/check_speed.py [--details docs/v08/speed/latest-details.json] [--limit 3.0]
        [--target 2.0]

Reads the details file that benchmarks/speed_suite.py wrote and exits with status 1 when
* an operation of the 102 in tests/fixtures/v8/capabilities_v07.json lacks a micro or a realistic fixture; verify, which
  has no SymPy counterpart, needs ExemptFixtures of both sizes instead, and so may any operation that is listed in
  criteria.SPEED_EXEMPTIONS;
* the details' source hash (source_sha256) differs from the working tree's: they were measured on other code;
* a fixture's ratio, MathForge's median over the median of SymPy's reset-cache calls, is above --limit;
* an exemption's budget is unmet: its timed call took longer than the production of the same result;
* an exemption names an operation outside criteria.SPEED_EXEMPTIONS (tools/coverage_gate/criteria.py).

It prints the 15 worst ratios, every operation above --target (a goal, not a gate), and the exemptions. The exit status
is 0 when nothing fails, 1 when the gate fails, and 2 for a usage error or details that cannot be read. The coverage
gate (tools/check_coverage.py) reads the speed file that the same run wrote beside the details. Details that come from
a partial run (--areas or --ops) cannot pass, since they leave operations without fixtures. This is internal tooling:
mathforge never imports it, and the results stay under the gitignored docs/v08/speed/.
"""

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
REPOSITORY = TOOLS.parent
for _entry in (TOOLS, REPOSITORY):  # Also under `python -I`, which leaves the script's directory off sys.path.
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from benchmarks import speed_suite  # noqa: E402
from coverage_gate import criteria  # noqa: E402

CAPABILITIES = REPOSITORY / "tests" / "fixtures" / "v8" / "capabilities_v07.json"
DEFAULT_DETAILS = REPOSITORY / "docs" / "v08" / "speed" / "latest-details.json"
DEFAULT_TARGET = 2.0
WORST = 15


def load_operations(path: Path) -> list[str]:
    """The operation names of the capabilities file: the 102 that the gate covers."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return [entry["name"] for entry in document["operations"]]


def load_details(path: Path) -> dict[str, Any]:
    """The details document, checked for the fields that the gate reads; ValueError says what is wrong."""
    document = json.loads(path.read_bytes().decode("utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("the details are not a JSON object")
    meta, fixtures, exempt = document.get("meta"), document.get("fixtures"), document.get("exempt")
    if not isinstance(meta, dict) or not isinstance(meta.get("source_sha256"), str):
        raise ValueError('the details have no "meta" object with a "source_sha256" string')
    if not isinstance(fixtures, dict) or not isinstance(exempt, dict):
        raise ValueError('the details have no "fixtures" and "exempt" objects')
    for table, entries in (("fixtures", fixtures), ("exempt", exempt)):
        for identifier, entry in entries.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("operation"), str) \
                    or entry.get("size") not in speed_suite.SIZES:
                raise ValueError(f'{table}["{identifier}"] needs an "operation" string and a "size" of '
                                 f"{' or '.join(speed_suite.SIZES)}")
    for identifier, entry in fixtures.items():
        ratio = entry.get("ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f'fixtures["{identifier}"] needs a positive finite "ratio", not {ratio!r}')
    for identifier, entry in exempt.items():
        if not isinstance(entry.get("budget_met"), bool):
            raise ValueError(f'exempt["{identifier}"] needs a "budget_met" of true or false')
    return document


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def gate(details: Mapping[str, Any], operations: Sequence[str], limit: float, working_hash: str) -> list[str]:
    """What fails the gate, one line each; empty when the details pass."""
    meta, fixtures, exempt = details["meta"], details["fixtures"], details["exempt"]
    failures: list[str] = []
    if meta["source_sha256"] != working_hash:
        failures.append(f"stale details: measured on source {meta['source_sha256'][:12]}, but the working tree is "
                        f"{working_hash[:12]}; run benchmarks/speed_suite.py again")
    measured: dict[str, set[str]] = {}
    for entry in fixtures.values():
        measured.setdefault(entry["operation"], set()).add(entry["size"])
    for entry in exempt.values():  # An exemption is a fixture of an operation that may have one.
        if entry["operation"] in criteria.SPEED_EXEMPTIONS:
            measured.setdefault(entry["operation"], set()).add(entry["size"])
    for size in speed_suite.SIZES:
        lacking = [name if name != "verify" else "verify (needs ExemptFixtures)" for name in operations
                   if size not in measured.get(name, ())]
        if lacking:
            failures.append(f"no {size} fixture for {plural(len(lacking), 'operation')}: {', '.join(lacking)}")
    for identifier, entry in sorted(fixtures.items()):
        if entry["ratio"] > limit:
            failures.append(f"ratio above the limit {limit}: {identifier} = {entry['ratio']:.2f}")
    for identifier, entry in sorted(exempt.items()):
        if entry["operation"] not in criteria.SPEED_EXEMPTIONS:
            failures.append(f"exemption outside criteria.SPEED_EXEMPTIONS: {identifier} exempts {entry['operation']}, "
                            f"which is not one of {', '.join(sorted(criteria.SPEED_EXEMPTIONS))}")
        if not entry["budget_met"]:
            failures.append(f"exemption budget not met: {identifier} (timed {_microseconds(entry.get('timed'))}, "
                            f"budget {_microseconds(entry.get('budget'))})")
    if failures and meta.get("partial"):
        scope = ", ".join(f"--{name} {','.join(meta[name])}" for name in ("areas", "ops") if meta.get(name))
        failures.insert(0, f"these details come from a partial run ({scope or 'a filtered run'}): the operations "
                           "it left out have no fixtures here")
    return failures


def _microseconds(statistics: object) -> str:
    median = statistics.get("median_us") if isinstance(statistics, dict) else None
    return "unknown" if not isinstance(median, (int, float)) else f"{median:.4g} us"


def render(details: Mapping[str, Any], operations: Sequence[str], limit: float, target: float) -> str:
    """The report: the worst ratios, the operations above the target and the exemptions."""
    fixtures, exempt = details["fixtures"], details["exempt"]
    lines = [f"speed gate: {plural(len(fixtures), 'fixture')} and {plural(len(exempt), 'exemption')} for "
             f"{plural(len(operations), 'operation')}; limit {limit}, target {target}"]
    worst = sorted(fixtures.items(), key=lambda item: (-item[1]["ratio"], item[0]))[:WORST]
    lines.append(f"{len(worst)} worst ratios:")
    for identifier, entry in worst:
        times = ""
        if all(isinstance(entry.get(mode), dict) for mode in ("mathforge", "sympy_reset")):
            times = (f"  (mathforge {_microseconds(entry['mathforge'])}, "
                     f"sympy_reset {_microseconds(entry['sympy_reset'])})")
        lines.append(f"  {entry['ratio']:8.2f}  {identifier}{times}")
    per_operation: dict[str, tuple[float, str]] = {}
    for identifier, entry in sorted(fixtures.items()):
        if entry["operation"] not in per_operation or entry["ratio"] > per_operation[entry["operation"]][0]:
            per_operation[entry["operation"]] = (entry["ratio"], identifier)
    above = sorted(((ratio, name, identifier) for name, (ratio, identifier) in per_operation.items()
                    if ratio > target), key=lambda item: (-item[0], item[1]))
    lines.append(f"operations above the target {target}: {len(above)}")
    for ratio, name, identifier in above:
        lines.append(f"  {name:<28} {ratio:8.2f}  {identifier}" + ("  (above the limit)" if ratio > limit else ""))
    for identifier, entry in sorted(exempt.items()):
        lines.append(f"exempt {identifier}: budget {'met' if entry['budget_met'] else 'NOT met'} "
                     f"(timed {_microseconds(entry.get('timed'))}, budget {_microseconds(entry.get('budget'))})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, working_hash: str | None = None,
         capabilities: Path = CAPABILITIES) -> int:
    """The command line. working_hash (instead of the hash of this checkout's src) and capabilities let tests run the
    gate on synthetic data."""
    for stream in (sys.stdout, sys.stderr):  # A narrow console must not stop the report.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="check_speed.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS,
                        help="the details file of benchmarks/speed_suite.py "
                             "(default: docs/v08/speed/latest-details.json)")
    parser.add_argument("--limit", type=float, default=criteria.SPEED_LIMIT,
                        help="the largest ratio any fixture may have (default 3.0)")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET,
                        help="the ratio that operations are listed above, without failing (default 2.0)")
    arguments = parser.parse_args(argv)
    for name in ("limit", "target"):
        value = getattr(arguments, name)
        if not (math.isfinite(value) and value > 0):
            parser.error(f"--{name} must be a positive number")
    if arguments.target > arguments.limit:
        parser.error("--target must not be above --limit")
    try:
        operations = load_operations(capabilities)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {capabilities}: {error!r}", file=sys.stderr)
        return 2
    try:
        details = load_details(arguments.details)
    except (OSError, ValueError) as error:
        print(f"error: {arguments.details}: {error}", file=sys.stderr)
        return 2
    try:
        actual_hash = working_hash if working_hash is not None else speed_suite.source_hash(REPOSITORY)
    except speed_suite.SuiteError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    failures = gate(details, operations, arguments.limit, actual_hash)
    report = render(details, operations, arguments.limit, arguments.target)
    print(report, flush=True)  # Flushed ahead of the causes, which go to stderr.
    if failures:
        print(f"check_speed: the speed gate fails, {plural(len(failures), 'cause')}:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("check_speed: the speed gate passes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
