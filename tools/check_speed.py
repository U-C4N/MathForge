"""The v0.8 speed gate: every operation within the limit of its fastest public SymPy call (internal developer tooling).

    .venv-mcp/Scripts/python.exe tools/check_speed.py [--details docs/v08/speed/latest-details.json] [--limit 3.0]
        [--target 2.0] [--allow-quick]

Reads the details file that benchmarks/speed_suite.py wrote and exits with status 1 when
* an operation of the 102 in tests/fixtures/v8/capabilities_v07.json lacks a micro or a realistic fixture; verify, which
  has no SymPy counterpart, needs ExemptFixtures of both sizes instead, and so may any operation that is listed in
  criteria.SPEED_EXEMPTIONS;
* the details are stale: their source hash (source_sha256) differs from the working tree's, or their instrument hash
  (instrument_sha256, over benchmarks/speed_suite.py and benchmarks/speed/**: the fixtures, their references and the
  suite itself) differs from the working tree's, so they were measured on other code or with another instrument;
* the fixtures that were measured are not the fixtures that the area modules define now (neither more nor fewer ids);
* SymPy was not 1.14.0, the reference the limit is defined against;
* the measurement protocol was not met: at least 3 rounds and 9 repeats, a pinned cpu, a complete (not partial) run and
  a recorded sample_ms. --allow-quick waives this minimum and prints that the result is not a gate run;
* a fixture is above --limit in any round: the median ratio (MathForge's median over the median of SymPy's reset-cache
  calls) stays the number that is reported, but the largest ratio of a single round is the one that is gated, because
  one SymPy call has measured 3.3 times slower in one run than in another. A fixture in WAIVED does not fail on this:
  it is printed as "waived: <reason>", or as within the limit when it has come back under it;
* an exemption's budget is unmet: its timed call took longer than the production of the same result;
* an exemption names an operation outside criteria.SPEED_EXEMPTIONS (tools/coverage_gate/criteria.py).

It prints the 15 worst ratios, every operation above --target (a goal, not a gate), the exemptions and the waivers. The
exit status is 0 when nothing fails, 1 when the gate fails, and 2 for a usage error or details that cannot be read. The
coverage gate (tools/check_coverage.py) reads the speed file that the same run wrote beside the details. Details that
come from a partial run (--areas or --ops) cannot pass, since they leave operations without fixtures. Comparing the
measured ids with the current fixtures imports the fixture modules, so mathforge must be importable (PYTHONPATH=src in
a worktree). This is internal tooling: mathforge never imports it, and the results stay under the gitignored
docs/v08/speed/.
"""

import argparse
import json
import math
import sys
from collections.abc import Collection, Mapping, Sequence
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
SYMPY_VERSION = "1.14.0"  # The reference the limit is defined against; the details' meta["sympy"] must say so.
LISTED = 5  # How many ids a failure line names before it says "...".

_BEZOUT = "the independently checked Bezout certificate dominates at this size (v0.8.0 known limitation)"
_STRUCTURED = "solve_system still runs the v0.7 structured pipeline; accelerated in v0.9"
# The documented v0.8.0 exceptions (a user decision): fixtures that may be above the limit without failing the gate. Each
# is printed as "waived: <reason>", and reported as such when it has come back within the limit. The ids must exist: a
# test holds WAIVED to the fixture modules, so a renamed fixture cannot leave a waiver behind.
WAIVED: dict[str, str] = {
    "integer_gcd/realistic/bits30000": _BEZOUT,
    "integer_gcd/realistic/rsa_moduli": _BEZOUT,
    "integer_lcm/realistic/bits20000": _BEZOUT,
    "integer_lcm/realistic/rsa_moduli": _BEZOUT,
    "solve_system/micro/two_by_two": _STRUCTURED,
    "solve_system/realistic/dense_6_by_6": _STRUCTURED,
    "solve_system/realistic/scaled_dense_12_by_12": _STRUCTURED,
    "solve_system/realistic/underdetermined_4_by_6": _STRUCTURED,
}


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
    rounds = meta.get("rounds")
    for identifier, entry in fixtures.items():
        ratio = entry.get("ratio")
        if not _positive(ratio):
            raise ValueError(f'fixtures["{identifier}"] needs a positive finite "ratio", not {ratio!r}')
        round_ratios = entry.get("round_ratios")
        if not (isinstance(round_ratios, list) and round_ratios and all(_positive(value) for value in round_ratios)):
            raise ValueError(f'fixtures["{identifier}"] needs "round_ratios", a list of positive finite ratios, one '
                             f"per round, not {round_ratios!r}")
        if _is_int(rounds) and len(round_ratios) != rounds:
            raise ValueError(f'fixtures["{identifier}"] has {len(round_ratios)} round_ratios for {rounds} rounds')
    for identifier, entry in exempt.items():
        if not isinstance(entry.get("budget_met"), bool):
            raise ValueError(f'exempt["{identifier}"] needs a "budget_met" of true or false')
    return document


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _named(identifiers: Collection[str]) -> str:
    ordered = sorted(identifiers)
    return ", ".join(ordered[:LISTED]) + (", ..." if len(ordered) > LISTED else "")


def worst_ratio(entry: Mapping[str, Any]) -> float:
    """The number that the limit is held to: the largest ratio of a single round, or the median ratio when that is
    larger (the pooled median is not bounded by the rounds')."""
    return max(entry["ratio"], *entry["round_ratios"])


def _rounds(entry: Mapping[str, Any]) -> str:
    return ", ".join(f"{value:.2f}" for value in entry["round_ratios"])


def protocol_failures(meta: Mapping[str, Any]) -> list[str]:
    """How the details fall short of the measurement protocol: rounds, repeats, a pinned cpu, a complete run and a
    recorded sample_ms. (The partial-run failure is added by gate, with the scope of the run.)"""
    failures = []
    rounds, repeat, cpu, sample_ms = meta.get("rounds"), meta.get("repeat"), meta.get("cpu"), meta.get("sample_ms")
    if not (_is_int(rounds) and rounds >= speed_suite.MIN_ROUNDS):
        failures.append(f"too few rounds: {rounds!r}; the gate needs at least {speed_suite.MIN_ROUNDS}")
    if not (_is_int(repeat) and repeat >= speed_suite.MIN_REPEAT):
        failures.append(f"too few repeats: {repeat!r}; the gate needs at least {speed_suite.MIN_REPEAT}")
    if not (_is_int(cpu) and cpu >= 0):
        failures.append(f"the run was not pinned to a cpu (cpu is {cpu!r}); run the suite with --cpu N")
    if not _positive(sample_ms):
        failures.append(f"no sample_ms was recorded (it is {sample_ms!r}), so the sample length is unknown")
    return failures


def gate(details: Mapping[str, Any], operations: Sequence[str], limit: float, working_hash: str, *,
         instrument_hash: str, current_ids: tuple[Collection[str], Collection[str]],
         allow_quick: bool = False) -> list[str]:
    """What fails the gate, one line each; empty when the details pass.

    working_hash and instrument_hash are the working tree's source and instrument hashes; current_ids are the ids of
    the Fixtures and of the ExemptFixtures that the area modules define now. allow_quick waives the measurement
    protocol (not the staleness, id, SymPy, ratio or exemption checks)."""
    meta, fixtures, exempt = details["meta"], details["fixtures"], details["exempt"]
    failures: list[str] = []
    if meta["source_sha256"] != working_hash:
        failures.append(f"stale details: measured on source {meta['source_sha256'][:12]}, but the working tree is "
                        f"{working_hash[:12]}; run benchmarks/speed_suite.py again")
    measured_instrument = meta.get("instrument_sha256")
    if not isinstance(measured_instrument, str):
        failures.append("stale details: they carry no instrument_sha256, so nothing says which fixtures, references "
                        "and suite measured them; run benchmarks/speed_suite.py again")
    elif measured_instrument != instrument_hash:
        failures.append(f"stale details: measured with instrument {measured_instrument[:12]}, but benchmarks/speed_suite"
                        f".py and benchmarks/speed are {instrument_hash[:12]} now; run benchmarks/speed_suite.py again")
    if meta.get("sympy") != SYMPY_VERSION:
        failures.append(f"SymPy {SYMPY_VERSION} is the reference of the limit, but the details were measured with "
                        f"{'no SymPy' if meta.get('sympy') is None else 'SymPy ' + str(meta.get('sympy'))}")
    if not allow_quick:
        failures.extend(protocol_failures(meta))
    current_fixtures, current_exempt = set(current_ids[0]), set(current_ids[1])
    for kind, measured, current in (("fixture", set(fixtures), current_fixtures),
                                    ("exempt fixture", set(exempt), current_exempt)):
        unknown = measured - current
        if unknown:
            failures.append(f"the details measured {plural(len(unknown), kind)} that no fixture module defines now: "
                            f"{_named(unknown)}; the fixtures changed after the run")
        lacking = current - measured
        if lacking and not meta.get("partial"):
            failures.append(f"the details lack {plural(len(lacking), kind)} that the fixture modules define now: "
                            f"{_named(lacking)}; the fixtures changed after the run")
    measured_sizes: dict[str, set[str]] = {}
    for entry in fixtures.values():
        measured_sizes.setdefault(entry["operation"], set()).add(entry["size"])
    for entry in exempt.values():  # An exemption is a fixture of an operation that may have one.
        if entry["operation"] in criteria.SPEED_EXEMPTIONS:
            measured_sizes.setdefault(entry["operation"], set()).add(entry["size"])
    for size in speed_suite.SIZES:
        lacking_operations = [name if name != "verify" else "verify (needs ExemptFixtures)" for name in operations
                              if size not in measured_sizes.get(name, ())]
        if lacking_operations:
            failures.append(f"no {size} fixture for {plural(len(lacking_operations), 'operation')}: "
                            f"{', '.join(lacking_operations)}")
    for identifier, entry in sorted(fixtures.items()):
        worst = worst_ratio(entry)
        if worst > limit and identifier not in WAIVED:
            if entry["ratio"] > limit:
                failures.append(f"ratio above the limit {limit}: {identifier} = {entry['ratio']:.2f} "
                                f"(rounds {_rounds(entry)})")
            else:
                failures.append(f"ratio above the limit {limit} in a round: {identifier} = {worst:.2f} in its worst "
                                f"round (median {entry['ratio']:.2f}; rounds {_rounds(entry)})")
    for identifier, entry in sorted(exempt.items()):
        if entry["operation"] not in criteria.SPEED_EXEMPTIONS:
            failures.append(f"exemption outside criteria.SPEED_EXEMPTIONS: {identifier} exempts {entry['operation']}, "
                            f"which is not one of {', '.join(sorted(criteria.SPEED_EXEMPTIONS))}")
        if not entry["budget_met"]:
            failures.append(f"exemption budget not met: {identifier} (timed {_microseconds(entry.get('timed'))}, "
                            f"budget {_microseconds(entry.get('budget'))})")
    if meta.get("partial"):
        # A partial run is never a gate run. With --allow-quick it is named only when something else failed, since
        # the operations that it left out are then the likely reason.
        if failures or not allow_quick:
            scope = ", ".join(f"--{name} {','.join(meta[name])}" for name in ("areas", "ops") if meta.get(name))
            failures.insert(0, f"these details come from a partial run ({scope or 'a filtered run'}): the "
                               "operations it left out have no fixtures here")
    elif not allow_quick and meta.get("partial") is not False:
        failures.insert(0, f"these details are not marked as a complete run (partial is {meta.get('partial')!r})")
    return failures


def _microseconds(statistics: object) -> str:
    median = statistics.get("median_us") if isinstance(statistics, dict) else None
    return "unknown" if not isinstance(median, (int, float)) else f"{median:.4g} us"


def waiver_lines(details: Mapping[str, Any], limit: float) -> list[str]:
    """The waived fixtures of the details: "waived: <reason>" for each that is above the limit in some round, and a
    note that the waiver is not needed for each that has come back within it."""
    lines = []
    for identifier, entry in sorted(details["fixtures"].items()):
        reason = WAIVED.get(identifier)
        if reason is None:
            continue
        worst = worst_ratio(entry)
        if worst > limit:
            lines.append(f"waived: {reason} [{identifier}: ratio {entry['ratio']:.2f}, worst {worst:.2f}, "
                         f"limit {limit}]")
        else:
            lines.append(f"waived fixture within the limit: {identifier}: ratio {entry['ratio']:.2f}, worst "
                         f"{worst:.2f}, limit {limit}; its waiver ({reason}) is no longer needed")
    return lines


def render(details: Mapping[str, Any], operations: Sequence[str], limit: float, target: float) -> str:
    """The report: the worst ratios, the operations above the target, the exemptions and the waivers."""
    fixtures, exempt = details["fixtures"], details["exempt"]
    lines = [f"speed gate: {plural(len(fixtures), 'fixture')} and {plural(len(exempt), 'exemption')} for "
             f"{plural(len(operations), 'operation')}; limit {limit}, target {target}"]
    worst = sorted(fixtures.items(), key=lambda item: (-item[1]["ratio"], item[0]))[:WORST]
    lines.append(f"{len(worst)} worst ratios:")
    for identifier, entry in worst:
        times = ""
        if all(isinstance(entry.get(mode), dict) for mode in ("mathforge", "sympy_reset")):
            times = (f"mathforge {_microseconds(entry['mathforge'])}, "
                     f"sympy_reset {_microseconds(entry['sympy_reset'])}; ")
        lines.append(f"  {entry['ratio']:8.2f}  {identifier}  ({times}rounds {_rounds(entry)})")
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
    lines.extend(waiver_lines(details, limit))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, working_hash: str | None = None, instrument_hash: str | None = None,
         current_ids: tuple[Collection[str], Collection[str]] | None = None, capabilities: Path = CAPABILITIES) -> int:
    """The command line. working_hash and instrument_hash (instead of the hashes of this checkout), current_ids
    (instead of the ids of its fixture modules) and capabilities let tests run the gate on synthetic data."""
    for stream in (sys.stdout, sys.stderr):  # A narrow console must not stop the report.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="check_speed.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS,
                        help="the details file of benchmarks/speed_suite.py "
                             "(default: docs/v08/speed/latest-details.json)")
    parser.add_argument("--limit", type=float, default=criteria.SPEED_LIMIT,
                        help="the largest ratio any fixture may have in any round (default 3.0)")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET,
                        help="the ratio that operations are listed above, without failing (default 2.0)")
    parser.add_argument("--allow-quick", action="store_true",
                        help="do not require the measurement protocol (3 rounds, 9 repeats, a pinned cpu, a complete "
                             "run); the result is then not a gate run")
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
        actual_instrument = (instrument_hash if instrument_hash is not None
                             else speed_suite.instrument_hash(REPOSITORY))
        actual_ids = current_ids if current_ids is not None else speed_suite.current_ids()
    except (speed_suite.SuiteError, ImportError) as error:
        print(f"error: cannot describe the working tree: {error}", file=sys.stderr)
        return 2
    failures = gate(details, operations, arguments.limit, actual_hash, instrument_hash=actual_instrument,
                    current_ids=actual_ids, allow_quick=arguments.allow_quick)
    quick = ("not a gate run (--allow-quick): the measurement protocol of 3 rounds, 9 repeats, a pinned cpu and a "
             "complete run is not required")
    if arguments.allow_quick:
        print(quick)
    report = render(details, operations, arguments.limit, arguments.target)
    print(report, flush=True)  # Flushed ahead of the causes, which go to stderr.
    if failures:
        print(f"check_speed: the speed gate fails, {plural(len(failures), 'cause')}:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    if arguments.allow_quick:
        print("check_speed: not a gate run: no check that --allow-quick leaves on has failed, but this is not the "
              "speed gate passing")
        return 0
    print("check_speed: the speed gate passes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
