"""The probe scoreboard: which natural mathematics requests MathForge answers (internal developer tooling).

    python tools/check_probes.py [--baseline PATH] [--list | --update [--accept-regression ID ...]] [--timings]

benchmarks/probes defines the probes: natural requests with a stable id, their attempt and, wherever the answer is
known, its check. Each probe gets one status from the outcome of its attempt alone, never from timing: ok, wrong,
deferred, unsupported, invalid_input, resource_limit or error (benchmarks/probes/__init__.py defines them).

The default mode compares the fresh statuses with the committed baseline, benchmarks/probes-v08.json, and exits with
status 1 on any difference (a probe whose status changed, a probe added, a probe removed) and on any wrong answer. A
status that improves is a difference too, so that the baseline records every step of growth: --update rewrites it,
and refuses when a probe that was ok is no longer ok (or is gone), unless --accept-regression names that probe, and
whenever a probe is wrong. --list prints each probe's id, status and request. Every mode prints the table of groups
by statuses with the totals; the default mode and --update also print the differences. --timings adds the slowest
probes (timings are printed, never compared).

The baseline records the mathforge version, each probe's status and the totals, with sorted keys, LF line ends and no
timestamp. A different mathforge version in it is reported but is no difference. The tool runs against the mathforge
that Python imports, an installed distribution included, and adds the checkout's src/ to sys.path only when mathforge
is not importable. The exit status is 0 when the scoreboard matches the baseline (or --update or --list succeeds), 1
on a difference, a wrong answer or a refused update, and 2 for a usage error or a baseline that cannot be read.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import time

TOOLS = Path(__file__).resolve().parent
REPOSITORY = TOOLS.parent
if str(REPOSITORY) not in sys.path:  # benchmarks/ is a namespace package at the repository root; also under -I.
    sys.path.insert(0, str(REPOSITORY))
try:
    import mathforge
except ImportError:  # A checkout without an installed distribution.
    sys.path.insert(0, str(REPOSITORY / "src"))
    import mathforge

from benchmarks import probes  # noqa: E402

BASELINE = REPOSITORY / "benchmarks" / "probes-v08.json"
FORMAT = "mathforge-probes/1"
SLOWEST = 10  # Probes that --timings names.


class BaselineError(Exception):
    """A baseline file that is missing, malformed or inconsistent."""


def totals(statuses: Mapping[str, str]) -> dict[str, int]:
    """The number of probes in each status (zeros included), and of all probes."""
    counts = {status: 0 for status in probes.STATUSES}
    for status in statuses.values():
        counts[status] += 1
    return {"all": len(statuses), **counts}


def document(version: str, statuses: Mapping[str, str]) -> dict[str, object]:
    """The baseline document of the statuses, measured with mathforge of that version."""
    return {"format": FORMAT, "mathforge": version, "probes": dict(sorted(statuses.items())),
            "totals": totals(statuses)}


def dumps(baseline: Mapping[str, object]) -> str:
    """A baseline's canonical text: sorted keys, two-space indents, ASCII and a final newline."""
    return json.dumps(baseline, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def load(path: Path) -> dict[str, object]:
    """A validated baseline document; BaselineError when it cannot be read or does not hold together."""
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BaselineError(f"cannot read the baseline {path}: {error}") from error
    if type(baseline) is not dict or set(baseline) != {"format", "mathforge", "probes", "totals"}:
        raise BaselineError(f"{path} is no probe baseline: it needs exactly format, mathforge, probes and totals")
    if baseline["format"] != FORMAT or type(baseline["mathforge"]) is not str:
        raise BaselineError(f"{path} has the format {baseline['format']!r}, not {FORMAT!r}, or no mathforge version")
    statuses = baseline["probes"]
    if type(statuses) is not dict or not all(type(key) is str and value in probes.STATUSES
                                             for key, value in statuses.items()):
        raise BaselineError(f"{path} maps a probe to something other than one of {', '.join(probes.STATUSES)}")
    if baseline["totals"] != totals(statuses):
        raise BaselineError(f"the totals of {path} do not match its probes")
    return baseline


def differences(old: Mapping[str, str], new: Mapping[str, str]) -> list[tuple[str, str | None, str | None]]:
    """(id, old status, new status) for every probe whose status changed, appeared (old None) or vanished (new None)."""
    return [(key, old.get(key), new.get(key)) for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)]


def regressions(old: Mapping[str, str], new: Mapping[str, str]) -> list[str]:
    """The probes that were ok and are no longer ok, or no longer exist."""
    return sorted(key for key, status in old.items() if status == "ok" and new.get(key) != "ok")


def table(probe_list: Sequence[probes.Probe], statuses: Mapping[str, str]) -> list[str]:
    """The lines of the table of groups by statuses, with a total row and column."""
    groups = [group for group in probes.GROUPS if any(probe.group == group for probe in probe_list)]
    counts = {group: totals({probe.id: statuses[probe.id] for probe in probe_list if probe.group == group})
              for group in groups}
    counts["total"] = totals({probe.id: statuses[probe.id] for probe in probe_list})
    columns = (*probes.STATUSES, "all")
    width = max(len("group"), *(len(name) for name in counts))
    header = "group".ljust(width) + "".join(f"  {column}" for column in columns)
    lines = [header]
    for name, row in counts.items():
        lines.append(name.ljust(width) + "".join(f"  {row[column]:>{len(column)}}" for column in columns))
    return lines


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:  # A baseline outside the repository.
        return str(path)


def _run(probe_list: Sequence[probes.Probe], timings: bool) -> dict[str, probes.Outcome]:
    outcomes, durations = {}, []
    for probe in probe_list:
        start = time.perf_counter()
        outcomes[probe.id] = probes.run_probe(probe, mathforge)
        durations.append((time.perf_counter() - start, probe.id))
    if timings:
        print(f"Slowest probes (all {len(durations)} took {sum(seconds for seconds, _ in durations):.2f} s):")
        for seconds, key in sorted(durations, reverse=True)[:SLOWEST]:
            print(f"  {seconds:7.3f} s  {key}")
        print()
    return outcomes


def _print_differences(changes: list[tuple[str, str | None, str | None]], outcomes: Mapping[str, probes.Outcome],
                       baseline_name: str) -> None:
    if not changes:
        print(f"Differences from {baseline_name}: none")
        return
    print(f"Differences from {baseline_name}: {len(changes)}")
    for key, old, new in changes:
        detail = f"  ({outcomes[key].detail})" if new is not None and new != "ok" else ""
        print(f"  {key}: {old or 'absent'} -> {new or 'absent'}{detail}")


def main(argv: Sequence[str] | None = None, *, outcomes: Mapping[str, probes.Outcome] | None = None) -> int:
    """Run the tool; outcomes, a test's stand-in for running the probes, maps every probe's id to its Outcome."""
    parser = argparse.ArgumentParser(description="Compare the probe scoreboard with its committed baseline.")
    parser.add_argument("--baseline", type=Path, default=BASELINE,
                        help="the baseline file (default: benchmarks/probes-v08.json)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print every probe's id, status and request")
    mode.add_argument("--update", action="store_true", help="rewrite the baseline with the fresh statuses")
    parser.add_argument("--accept-regression", action="append", default=[], metavar="ID",
                        help="with --update: accept that this probe, ok in the baseline, is no longer ok")
    parser.add_argument("--timings", action="store_true", help="also print the slowest probes")
    arguments = parser.parse_args(argv)
    if arguments.accept_regression and not arguments.update:
        parser.error("--accept-regression only goes with --update")
    try:
        probe_list = probes.all_probes()
    except ValueError as error:
        print(f"check_probes: {error}", file=sys.stderr)
        return 2
    if outcomes is None:
        outcomes = _run(probe_list, arguments.timings)
    fresh = {probe.id: outcomes[probe.id].status for probe in probe_list}
    version = mathforge.__version__
    path, name = arguments.baseline, _display(arguments.baseline)

    if arguments.list:
        width = max(len(probe.id) for probe in probe_list)
        status_width = max(len(status) for status in probes.STATUSES)
        for probe in probe_list:
            print(f"{probe.id.ljust(width)}  {fresh[probe.id].ljust(status_width)}  {probe.request}")
        print()
        print("\n".join(table(probe_list, fresh)))
        return 0

    baseline = None
    if path.exists() or not arguments.update:
        try:
            baseline = load(path)
        except BaselineError as error:
            print(f"check_probes: {error}", file=sys.stderr)
            return 2
    old = baseline["probes"] if baseline is not None else {}
    print(f"Probe scoreboard of mathforge {version}: {len(probe_list)} probes; baseline {name}"
          + (f" (mathforge {baseline['mathforge']})" if baseline is not None else " (new)"))
    print()
    print("\n".join(table(probe_list, fresh)))
    print()
    changes = differences(old, fresh)
    _print_differences(changes, outcomes, name)
    wrong = [key for key, status in fresh.items() if status == "wrong"]
    if wrong:
        print(f"Wrong answers: {len(wrong)}")
        for key in wrong:
            print(f"  {key}: {outcomes[key].detail}")

    if not arguments.update:
        if baseline is not None and baseline["mathforge"] != version:
            print(f"Note: the baseline was measured with mathforge {baseline['mathforge']}, this is {version}.")
        if changes and not wrong:
            print("Record the new statuses with --update (a probe that is no longer ok also needs "
                  "--accept-regression ID).")
        return 1 if changes or wrong else 0
    regressed = regressions(old, fresh)
    unknown = sorted(set(arguments.accept_regression) - set(regressed))
    if unknown:
        print(f"check_probes: --accept-regression names no regression: {', '.join(unknown)}", file=sys.stderr)
        return 2
    refused = [key for key in regressed if key not in arguments.accept_regression]
    if wrong:
        print("Not updated: a wrong answer never enters the baseline; fix the operation or the probe's check.")
        return 1
    if refused:
        print(f"Not updated: no longer ok: {', '.join(refused)}; pass --accept-regression ID for each to accept it.")
        return 1
    text = dumps(document(version, fresh))
    if baseline is not None and dumps(baseline) == text:
        print(f"{name} is up to date.")
        return 0
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Updated {name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
