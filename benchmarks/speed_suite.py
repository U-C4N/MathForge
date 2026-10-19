"""The v0.8 speed suite, the measuring instrument of the Phase 1 speed gate.

    python benchmarks/speed_suite.py --output docs/v08/speed/latest.json [--details PATH] [--rounds 3] [--repeat 9]
        [--warmups 2] [--cpu N] [--areas a,b] [--ops x,y] [--sample-ms 2]

Every Fixture of benchmarks/speed/fixtures/<area>.py (one module per registry area; --areas and --ops select) times a
public MathForge value function, which verifies its own value, against the fastest public SymPy call that returns the
same mathematical object. Run it with the interpreter that has SymPy 1.14 (.venv-mcp) and gate its details with
tools/check_speed.py.

Method
* One fresh worker process per round: sys.executable, with this repository's src first on sys.path and
  SYMPY_GROUND_TYPES=python and MPMATH_NOGMPY=1 in its environment. The worker asserts that mathforge was imported from
  that src, and that SymPy and mpmath run their pure Python types. Every round must report the same SHA-256 over
  src/mathforge/**/*.py, sorted by path (source_hash), or the run stops and writes nothing. --cpu N pins the worker to
  one logical CPU (os.sched_setaffinity on Linux, SetProcessAffinityMask on Windows).
* In a round the fixtures run in an order shuffled by random.Random(20261019 + round). For each Fixture:
    first_call  the first MathForge call alone, on variant 0, after a reset;
    warmups     --warmups passes, each running both sides on variants 0 to min(variants, 5) - 1 and checking `agree`;
                a False aborts the run and names the fixture;
    k           the calls per sample, calibrated once: a SymPy sample lasts at least --sample-ms, with at most
                `variants` calls;
    samples     --repeat samples per mode, the modes in shuffled order on every repeat:
        mathforge    k calls of the public value function on consecutive variants 0 to k - 1;
        sympy_reset  k calls of the SymPy reference on the same variants;
        sympy_warm   variant 0 repeated k times with SymPy's caches warm (primed, no reset): the memoization effect;
        structured   mathforge.operations.<public name> on variant 0 repeated min(k, 100) times, paired in the same
                     sample with the value function on variant 0 repeated as often. Reported (overhead_us), never
                     gated.
  A sample is its k calls divided by k. Before every sample except sympy_warm the inputs are built (untimed), the
  collector runs, and the reset clears SymPy's caches as benchmarks/fresh_comparison.py does (clear_cache(),
  CRootOf.clear_cache(), the factor cache) and then every BoundedCache (mathforge.trusted.caches.clear_all()). The
  collector is off while the calls run. Both sides time prebuilt inputs and a zero-argument call.
* ratio = median(mathforge) / median(sympy_reset) over the samples of all rounds. An ExemptFixture (verify, which has
  no SymPy counterpart) times one call per sample of `timed`, the call that is held to a budget, and of `budget`, the
  production of the same result: budget_met = median(timed) <= median(budget).

Files
* --details (default: --output with "-details" added to its stem): {"meta": ..., "fixtures": {id: ...}, "exempt":
  {id: ...}}. meta holds the source hash (source_sha256), the Python, SymPy and mpmath versions, the ground types, cpu,
  rounds, repeat, warmups, sample_ms, the date, and areas, ops and partial. A fixture entry holds its operation, size,
  label, variants, k, v07 and reference, the modes mathforge, sympy_reset and sympy_warm (median_us, min_us, p95_us,
  round_medians_us and the raw samples_us), ratio, first_call, and structured (median_us, value_median_us,
  overhead_us, or the error that stopped it). An exempt entry holds timed and budget (as a mode), budget_met and reason.
* --output, the coverage gate's speed file: {"operations": {operation: its worst ratio | {"exempt": reason,
  "budget_met": bool}}} (tools/coverage_gate/criteria.py). An operation with an exempt fixture whose budget is not met
  is written as the exemption object with budget_met false, so that the file fails it however good its ratios are.
* A partial run (--areas or --ops) writes the details only.

Exit status: 0 when the run completes, whatever the ratios; the gate is tools/check_speed.py. A bad option, a fixture
that disagrees with SymPy, a failed round or a changed source tree stops the run with status 2 and writes nothing.
--modules (comma-separated dotted module names, which replace the area modules and cannot go with --areas) is for the
suite's own tests. Dev tooling: mathforge never imports it. Results stay under the gitignored docs/v08/speed/.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pkgutil
import platform
import random
import re
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter_ns
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_PACKAGE = "benchmarks.speed.fixtures"
SEED = 20261019  # A round shuffles with random.Random(SEED + round index).
SIZES = ("micro", "realistic")
MODES = ("mathforge", "sympy_reset", "sympy_warm", "structured")
AGREE_VARIANTS = 5  # A warmup compares the two sides on variants 0 to min(variants, 5) - 1.
STRUCTURED_CALLS = 100  # The most calls of one path in a structured sample: it is reported only, so it stays cheap.
REFERENCE_ENVIRONMENT = {"SYMPY_GROUND_TYPES": "python", "MPMATH_NOGMPY": "1"}
DETAILS_FORMAT = 1


class SuiteError(Exception):
    """An option or a fixture cannot be measured: the message names it and says why. The run stops, writing nothing."""


@dataclass(frozen=True)
class Config:
    repeat: int
    warmups: int
    sample_ms: float


def source_hash(root: Path = ROOT) -> str:
    """SHA-256 over src/mathforge/**/*.py under root, sorted by path, each file as its root-relative path, a NUL and
    its bytes: what a speed measurement was taken against, and what tools/check_speed.py compares the tree with."""
    root = Path(root)
    package = root / "src" / "mathforge"
    paths = sorted(package.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise SuiteError(f"{package} holds no Python sources")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + path.read_bytes())
    return digest.hexdigest()


def repository_importable() -> None:
    """Put this repository's src, then its root, first on sys.path: mathforge and benchmarks.speed import from here."""
    for path in (str(ROOT), str(SRC)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def root_importable() -> None:
    """Make the repository root importable, so that benchmarks.speed does, without moving anything else."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def fixture_types() -> tuple[type, type]:
    """The Fixture and ExemptFixture classes of benchmarks.speed.fixture."""
    root_importable()
    from benchmarks.speed.fixture import ExemptFixture, Fixture
    return Fixture, ExemptFixture


def entry_id(entry: Any) -> str:
    """The id of a Fixture or an ExemptFixture: <operation>/<size>/<label>."""
    return f"{entry.operation}/{entry.size}/{entry.label}"


def discover(package: str, areas: Sequence[str] | None) -> list[str]:
    """The area modules of a fixture package: every module whose name does not start with an underscore, or the named
    ones, which must exist."""
    root_importable()
    try:
        parent = importlib.import_module(package)
    except ImportError as error:
        raise SuiteError(f"cannot import the fixture package {package}: {error}") from error
    available = sorted(name for _, name, _ in pkgutil.iter_modules(parent.__path__) if not name.startswith("_"))
    if areas is None:
        return available
    unknown = [area for area in areas if area not in available]
    if unknown:
        raise SuiteError(f"no fixture module {', '.join(unknown)} in {package}; "
                         f"it holds: {', '.join(available) or 'no modules'}")
    return list(areas)


def module_names(areas: Sequence[str] | None, modules: Sequence[str] | None) -> list[str]:
    """The fixture modules to load: the named ones (--modules, for the suite's own tests), else the area modules of
    benchmarks/speed/fixtures that --areas selects, all of them without it."""
    if modules is not None:
        return list(modules)
    return [f"{DEFAULT_PACKAGE}.{area}" for area in discover(DEFAULT_PACKAGE, areas)]


def load_entries(names: Sequence[str], ops: Sequence[str] | None) -> list[Any]:
    """Every Fixture and ExemptFixture of the named modules, in module and declaration order, filtered by operation.
    Each id must be unique across the modules; a named operation must have a fixture."""
    fixture_type, exempt_type = fixture_types()
    entries: list[Any] = []
    owners: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            raise SuiteError(f"cannot import {name}: {type(error).__name__}: {error}") from error
        declared = getattr(module, "FIXTURES", None)
        if not isinstance(declared, (tuple, list)):
            raise SuiteError(f"{name} defines no FIXTURES tuple")
        for entry in declared:
            if not isinstance(entry, (fixture_type, exempt_type)):
                raise SuiteError(f"{name}.FIXTURES holds a {type(entry).__name__}, not a Fixture or an ExemptFixture")
            identifier = entry_id(entry)
            if entry.size not in SIZES:
                raise SuiteError(f"{identifier}: the size must be one of {', '.join(SIZES)}")
            if identifier in owners:
                raise SuiteError(f"{identifier} is defined in {owners[identifier]} and in {name}")
            if isinstance(entry, fixture_type) and (type(entry.variants) is not int or entry.variants < 1):
                raise SuiteError(f"{identifier}: variants must be a positive integer, not {entry.variants!r}")
            if isinstance(entry, exempt_type) and not (isinstance(entry.reason, str) and entry.reason.strip()):
                raise SuiteError(f"{identifier}: an ExemptFixture must say why in a non-empty reason")
            owners[identifier] = name
            entries.append(entry)
    if ops is not None:
        known = sorted({entry.operation for entry in entries})
        missing = [op for op in ops if op not in known]
        if missing:
            raise SuiteError(f"no fixture for operation {', '.join(missing)} in the selected modules"
                             + (f"; they have fixtures for: {', '.join(known)}" if len(known) <= 40 else ""))
        entries = [entry for entry in entries if entry.operation in ops]
    if not entries:
        raise SuiteError("no fixtures selected")
    return entries


# --- Process setup ----------------------------------------------------------------------------------------------

def pin_cpu(cpu: int) -> None:
    """Pin this process to one logical CPU, and check that the pin holds."""
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError as error:
            raise SuiteError(f"cannot pin the process to CPU {cpu}: {error}") from error
        if os.sched_getaffinity(0) != {cpu}:
            raise SuiteError(f"the process is not pinned to CPU {cpu}")
    elif sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        if not 0 <= cpu < 64:
            raise SuiteError(f"cannot pin the process to CPU {cpu}: Windows takes 0 to 63")
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # The pseudo handle is -1: a C int would not carry it.
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                    ctypes.POINTER(ctypes.c_size_t)]
        kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(process, 1 << cpu):
            raise SuiteError(f"cannot pin the process to CPU {cpu}: {ctypes.WinError()}")
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask)) \
                or process_mask.value != 1 << cpu:
            raise SuiteError(f"the process is not pinned to CPU {cpu}")
    else:
        raise SuiteError(f"--cpu is not supported on {sys.platform}: it needs os.sched_setaffinity or Windows")


def reference_versions() -> dict[str, Any]:
    """Import SymPy and mpmath when installed, require their pure Python types, and report what runs."""
    info: dict[str, Any] = {"sympy": None, "mpmath": None, "sympy_ground_types": None, "mpmath_backend": None}
    if importlib.util.find_spec("sympy") is not None:
        import sympy
        info["sympy"] = sympy.__version__
        try:
            from sympy.external.gmpy import GROUND_TYPES
        except ImportError:  # A SymPy without the attribute: the environment variable is all there is to go by.
            GROUND_TYPES = None
        info["sympy_ground_types"] = GROUND_TYPES
        if GROUND_TYPES not in (None, "python"):
            raise SuiteError(f"SymPy runs with the {GROUND_TYPES!r} ground types, not 'python'")
    if importlib.util.find_spec("mpmath") is not None:
        import mpmath
        import mpmath.libmp
        info["mpmath"] = mpmath.__version__
        info["mpmath_backend"] = getattr(mpmath.libmp, "BACKEND", None)
        if info["mpmath_backend"] not in (None, "python"):
            raise SuiteError(f"mpmath runs with the {info['mpmath_backend']!r} backend, not 'python'")
    return info


class Reset:
    """The reset before a sample: SymPy's caches as benchmarks/fresh_comparison.py clears them, then every
    BoundedCache of MathForge. SymPy is cleared once it has been imported; before that it has nothing to clear."""

    def __init__(self) -> None:
        from mathforge.trusted.caches import clear_all
        self._clear_all = clear_all
        self._sympy: tuple[Callable[[], object], ...] | None = None

    def __call__(self) -> None:
        if self._sympy is None and "sympy" in sys.modules:
            import sympy
            from sympy.core.cache import clear_cache
            from sympy.ntheory.factor_ import factor_cache
            self._sympy = (clear_cache, sympy.CRootOf.clear_cache, factor_cache.cache_clear)
        if self._sympy is not None:
            for clear in self._sympy:
                clear()
        self._clear_all()


# --- Timing -----------------------------------------------------------------------------------------------------

def run_calls(calls: Sequence[Callable[[], object]]) -> int:
    """Nanoseconds to call every zero-argument callable once, in order, with the collector off."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = perf_counter_ns()
        for call in calls:
            call()
        return perf_counter_ns() - start
    finally:
        if enabled:
            gc.enable()


def attempt(name: str, what: str, function: Callable[..., Any], *arguments: Any) -> Any:
    """function(*arguments), with a failure turned into a SuiteError that names the fixture and what it was doing."""
    try:
        return function(*arguments)
    except SuiteError:
        raise
    except Exception as error:
        raise SuiteError(f"{name}: {what} raised {type(error).__name__}: {error}") from error


def short(value: object, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def mathforge_call(fixture: Any, i: int) -> Callable[[], object]:
    """A zero-argument call of the public value function on variant i, its inputs prebuilt."""
    function, args, kwargs = fixture.mathforge(i)
    return partial(function, *args, **kwargs)


def check_agreement(fixture: Any, i: int, reset: Callable[[], None]) -> None:
    """Run both sides on variant i, each after a reset, and abort when they are not the same mathematical object."""
    name = entry_id(fixture)
    function, args, kwargs = attempt(name, f"building the MathForge call of variant {i}", fixture.mathforge, i)
    reference = attempt(name, f"building the SymPy call of variant {i}", fixture.sympy, i)
    reset()
    mathforge_value = attempt(name, f"the MathForge call of variant {i}", lambda: function(*args, **kwargs))
    reset()
    sympy_value = attempt(name, f"the SymPy call of variant {i}", reference)
    agreed = attempt(name, f"agree on variant {i}", fixture.agree, mathforge_value, sympy_value)
    if not agreed:
        raise SuiteError(f"{name}: MathForge and SymPy disagree on variant {i}: MathForge gave "
                         f"{short(mathforge_value)}, SymPy gave {short(sympy_value)}")


def calibrate(fixture: Any, config: Config, reset: Callable[[], None]) -> int:
    """The calls per sample: the fewest, from 1 to `variants`, whose SymPy sample lasts at least config.sample_ms."""
    name = entry_id(fixture)
    target = config.sample_ms * 1_000_000
    k = 1
    while True:
        calls = attempt(name, "building SymPy calls", lambda: [fixture.sympy(i) for i in range(k)])
        gc.collect()
        reset()
        elapsed = attempt(name, f"calibrating k={k}", run_calls, calls)
        if elapsed >= target or k >= fixture.variants:
            return k
        k = min(fixture.variants, max(k * 2, math.ceil(k * target / max(elapsed, 1) * 1.1)))


def find_structured(fixture: Any, reset: Callable[[], None]) -> tuple[Callable[..., object] | None, str]:
    """The structured counterpart of the fixture's value function, mathforge.operations.<public name>, and a status:
    "ok", "same" (the value function is the structured one, so there is nothing to compare) or a message saying
    that there is no such function or that it raised on variant 0."""
    import mathforge
    function, args, kwargs = fixture.mathforge(0)
    try:
        name = mathforge.registry.spec(fixture.operation).public_name
    except mathforge.UnsupportedOperation:  # A legacy operation: its name is its public name.
        name = fixture.operation
    structured = getattr(mathforge.operations, name, None)
    if structured is None:
        return None, f"mathforge.operations has no {name}"
    if structured is function:
        return None, "same"
    try:
        reset()
        structured(*args, **kwargs)
    except Exception as error:
        return None, short(f"mathforge.operations.{name} raised {type(error).__name__}: {error}", 240)
    return structured, "ok"


def measure_fixture(fixture: Any, config: Config, rng: random.Random, reset: Callable[[], None]) -> dict[str, Any]:
    """One round of one Fixture: its raw per-call samples in microseconds, k and first_call."""
    name = entry_id(fixture)
    call = attempt(name, "building the MathForge call of variant 0", mathforge_call, fixture, 0)
    gc.collect()
    reset()
    first_call = attempt(name, "the first MathForge call", run_calls, [call]) / 1000
    for _ in range(config.warmups):
        for i in range(min(fixture.variants, AGREE_VARIANTS)):
            check_agreement(fixture, i, reset)
    k = calibrate(fixture, config, reset)
    structured, status = attempt(name, "the structured probe", find_structured, fixture, reset)

    def sample_mathforge() -> float:
        calls = [mathforge_call(fixture, i) for i in range(k)]
        gc.collect()
        reset()
        return run_calls(calls) / k / 1000

    def sample_sympy_reset() -> float:
        calls = [fixture.sympy(i) for i in range(k)]
        gc.collect()
        reset()
        return run_calls(calls) / k / 1000

    def sample_sympy_warm() -> float:
        reference = fixture.sympy(0)
        reference()  # Primed: the caches are warm for every timed call.
        calls = [reference] * k
        gc.collect()
        return run_calls(calls) / k / 1000

    def sample_structured() -> tuple[float, float]:
        # Variant 0 repeated through each path, one after the other in a random order, each with inputs of its own:
        # the difference of the two is what the structured pipeline costs on top of the value path.
        function, args, kwargs = fixture.mathforge(0)
        _, structured_args, structured_kwargs = fixture.mathforge(0)
        repeats = min(k, STRUCTURED_CALLS)
        calls = {"value": [partial(function, *args, **kwargs)] * repeats,
                 "structured": [partial(structured, *structured_args, **structured_kwargs)] * repeats}
        order = list(calls)
        rng.shuffle(order)
        times = {}
        for which in order:
            gc.collect()
            reset()
            times[which] = run_calls(calls[which]) / repeats / 1000
        return times["structured"], times["value"]

    samplers = {"mathforge": sample_mathforge, "sympy_reset": sample_sympy_reset, "sympy_warm": sample_sympy_warm}
    samples: dict[str, list[float]] = {mode: [] for mode in samplers}
    structured_samples: list[tuple[float, float]] = []
    modes = [mode for mode in MODES if mode != "structured" or status == "ok"]
    for _ in range(config.repeat):
        order = list(modes)
        rng.shuffle(order)
        for mode in order:
            if mode == "structured":
                structured_samples.append(attempt(name, f"timing structured (k={k})", sample_structured))
            else:
                samples[mode].append(attempt(name, f"timing {mode} (k={k})", samplers[mode]))
    if status == "ok":
        outcome: dict[str, Any] = {"status": "ok", "structured_us": [pair[0] for pair in structured_samples],
                                   "value_us": [pair[1] for pair in structured_samples]}
    elif status == "same":
        outcome = {"status": "same"}
    else:
        outcome = {"status": "error", "message": status}
    return {"operation": fixture.operation, "size": fixture.size, "label": fixture.label, "variants": fixture.variants,
            "reference": fixture.reference, "v07": fixture.v07, "k": k, "first_call_us": first_call,
            "samples_us": samples, "structured": outcome}


def measure_exempt(fixture: Any, config: Config, rng: random.Random, reset: Callable[[], None]) -> dict[str, Any]:
    """One round of one ExemptFixture: one call per sample of `timed` and of `budget`, in microseconds."""
    name = entry_id(fixture)

    def build(role: str) -> Callable[[], object]:
        function, args, kwargs = getattr(fixture, role)()
        return partial(function, *args, **kwargs)

    for _ in range(config.warmups):
        for role in ("timed", "budget"):
            call = attempt(name, f"building the {role} call", build, role)
            reset()
            value = attempt(name, f"the {role} call", call)
            if role == "timed" and getattr(value, "verified", True) is False:
                raise SuiteError(f"{name}: the timed call did not verify its result, so its time says nothing")
    samples: dict[str, list[float]] = {"timed": [], "budget": []}
    for _ in range(config.repeat):
        order = list(samples)
        rng.shuffle(order)
        for role in order:
            call = attempt(name, f"building the {role} call", build, role)
            gc.collect()
            reset()
            samples[role].append(attempt(name, f"timing the {role} call", run_calls, [call]) / 1000)
    return {"operation": fixture.operation, "size": fixture.size, "label": fixture.label, "reason": fixture.reason,
            "samples_us": samples}


# --- The worker: one round --------------------------------------------------------------------------------------

def run_worker(arguments: argparse.Namespace) -> int:
    repository_importable()
    for variable, value in REFERENCE_ENVIRONMENT.items():
        os.environ.setdefault(variable, value)
    if arguments.cpu is not None:
        pin_cpu(arguments.cpu)
    import mathforge
    source = SRC.resolve()
    if source not in Path(mathforge.__file__).resolve().parents:
        raise SuiteError(f"mathforge was imported from {mathforge.__file__}, not from {source}: the suite would "
                         "measure another source tree")
    environment = reference_versions()
    names = module_names(parse_list(arguments.areas), parse_list(arguments.modules))
    entries = load_entries(names, parse_list(arguments.ops))
    fixture_type, _ = fixture_types()
    config = Config(arguments.repeat, arguments.warmups, arguments.sample_ms)
    rng = random.Random(SEED + arguments.round_index)
    rng.shuffle(entries)
    reset = Reset()
    record: dict[str, Any] = {"round": arguments.round_index, "source_sha256": source_hash(), "modules": names,
                              "mathforge_version": mathforge.__version__, "python": platform.python_version(),
                              "platform": platform.platform(), "environment": environment, "fixtures": {},
                              "exempt": {}}
    for index, entry in enumerate(entries, 1):
        identifier = entry_id(entry)
        if isinstance(entry, fixture_type):
            measured = measure_fixture(entry, config, rng, reset)
            record["fixtures"][identifier] = measured
            mathforge_us = median(measured["samples_us"]["mathforge"])
            sympy_us = median(measured["samples_us"]["sympy_reset"])
            summary = f"mathforge {mathforge_us:.3f} us, sympy {sympy_us:.3f} us, k={measured['k']}"
            if sympy_us > 0:
                summary += f", ratio {mathforge_us / sympy_us:.2f}"
        else:
            measured = measure_exempt(entry, config, rng, reset)
            record["exempt"][identifier] = measured
            summary = (f"timed {median(measured['samples_us']['timed']):.3f} us, "
                       f"budget {median(measured['samples_us']['budget']):.3f} us")
        print(f"round {arguments.round_index + 1} [{index}/{len(entries)}] {identifier}: {summary}", flush=True)
    arguments.worker_result.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    return 0


# --- The parent: rounds, aggregation, files ---------------------------------------------------------------------

def significant(value: float | None) -> float | None:
    """Six significant digits: enough for a timing, and a short number in the files."""
    return None if value is None else float(f"{value:.6g}")


def describe(rounds: Sequence[Sequence[float]]) -> dict[str, Any]:
    """The statistics of one mode over the samples of all rounds."""
    pooled = [value for samples in rounds for value in samples]
    ordered = sorted(pooled)
    return {"median_us": significant(median(pooled)), "min_us": significant(ordered[0]),
            "p95_us": significant(ordered[math.ceil(0.95 * len(ordered)) - 1]),
            "round_medians_us": [significant(median(samples)) for samples in rounds],
            "samples_us": [significant(value) for value in pooled]}


def positive_ratio(name: str, numerator: float, denominator: float) -> float:
    """numerator / denominator for two medians that a gate ratio is made of: both must be real times."""
    if not (numerator > 0 and denominator > 0):
        raise SuiteError(f"{name}: a median time of {min(numerator, denominator)} us is not a time the timer can "
                         "measure, so a ratio of it says nothing; give the fixture more variants")
    return numerator / denominator


def summarize_structured(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [record["structured"] for record in records]
    status = outcomes[0]["status"]
    if status == "same":
        return {"same_function": True, "overhead_us": 0.0}
    if status == "error":
        return {"error": outcomes[0]["message"], "overhead_us": None}
    structured = [outcome["structured_us"] for outcome in outcomes]
    value = [outcome["value_us"] for outcome in outcomes]
    differences = [[s - v for s, v in zip(first, second)] for first, second in zip(structured, value)]
    pooled = [difference for round_differences in differences for difference in round_differences]
    summary = describe(structured)
    return {"median_us": summary["median_us"], "value_median_us": significant(median(
                [v for samples in value for v in samples])),
            "overhead_us": significant(median(pooled)),
            "round_overhead_us": [significant(median(d)) for d in differences],
            "samples_us": summary["samples_us"]}


def summarize_fixture(identifier: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    modes = {mode: [record["samples_us"][mode] for record in records]
             for mode in ("mathforge", "sympy_reset", "sympy_warm")}
    pooled = {mode: median([value for samples in rounds for value in samples]) for mode, rounds in modes.items()}
    ratio = positive_ratio(identifier, pooled["mathforge"], pooled["sympy_reset"])
    round_ratios = [positive_ratio(identifier, median(mathforge), median(sympy))
                    for mathforge, sympy in zip(modes["mathforge"], modes["sympy_reset"])]
    ks = sorted(record["k"] for record in records)
    first_calls = [record["first_call_us"] for record in records]
    return {"operation": first["operation"], "size": first["size"], "label": first["label"],
            "variants": first["variants"], "k": ks[(len(ks) - 1) // 2], "k_by_round": [r["k"] for r in records],
            "v07": first["v07"], "reference": first["reference"],
            **{mode: describe(rounds) for mode, rounds in modes.items()},
            "ratio": significant(ratio), "round_ratios": [significant(value) for value in round_ratios],
            "first_call": {"median_us": significant(median(first_calls)),
                           "rounds_us": [significant(value) for value in first_calls]},
            "structured": summarize_structured(records)}


def summarize_exempt(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    timed = describe([record["samples_us"]["timed"] for record in records])
    budget = describe([record["samples_us"]["budget"] for record in records])
    timed_median = median([v for record in records for v in record["samples_us"]["timed"]])
    budget_median = median([v for record in records for v in record["samples_us"]["budget"]])
    # The ratio is for the reader only: the budget is met when timed is no slower than budget, whatever the ratio.
    return {"operation": first["operation"], "size": first["size"], "label": first["label"], "reason": first["reason"],
            "timed": timed, "budget": budget, "budget_met": bool(timed_median <= budget_median),
            "ratio": significant(timed_median / budget_median) if budget_median > 0 else None}


def build_details(rounds: Sequence[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    for table in ("fixtures", "exempt"):
        if any(set(record[table]) != set(rounds[0][table]) for record in rounds):
            raise SuiteError(f"the rounds did not measure the same {table}")
    return {"meta": meta,
            "fixtures": {identifier: summarize_fixture(identifier, [record["fixtures"][identifier]
                                                                    for record in rounds])
                         for identifier in sorted(rounds[0]["fixtures"])},
            "exempt": {identifier: summarize_exempt([record["exempt"][identifier] for record in rounds])
                       for identifier in sorted(rounds[0]["exempt"])}}


def build_criteria(details: dict[str, Any]) -> dict[str, Any]:
    """The coverage gate's speed file: the worst ratio of an operation's fixtures, or an exemption object."""
    ratios: dict[str, float] = {}
    exempt: dict[str, list[dict[str, Any]]] = {}
    for entry in details["fixtures"].values():
        ratios[entry["operation"]] = max(ratios.get(entry["operation"], 0.0), entry["ratio"])
    for entry in details["exempt"].values():
        exempt.setdefault(entry["operation"], []).append(entry)
    operations: dict[str, Any] = {}
    for operation in sorted(set(ratios) | set(exempt)):
        entries = exempt.get(operation, [])
        reason = "; ".join(dict.fromkeys(entry["reason"] for entry in entries))
        if entries and not all(entry["budget_met"] for entry in entries):
            operations[operation] = {"exempt": reason, "budget_met": False}
        elif operation in ratios:
            operations[operation] = ratios[operation]
        else:
            operations[operation] = {"exempt": reason, "budget_met": True}
    return {"operations": operations}


def render_table(details: dict[str, Any]) -> str:
    """The fixtures of a run, one line each, with the worst ratio last."""
    lines = [f"{'fixture':<48} {'k':>5} {'mathforge us':>13} {'sympy us':>10} {'warm us':>9} {'ratio':>8} "
             f"{'struct +us':>11} {'first us':>10}"]
    worst = None
    for identifier, entry in details["fixtures"].items():
        overhead = entry["structured"].get("overhead_us")
        lines.append(f"{identifier:<48} {entry['k']:>5} {entry['mathforge']['median_us']:>13.4g} "
                     f"{entry['sympy_reset']['median_us']:>10.4g} {entry['sympy_warm']['median_us']:>9.4g} "
                     f"{entry['ratio']:>8.2f} {'-' if overhead is None else format(overhead, '.4g'):>11} "
                     f"{entry['first_call']['median_us']:>10.4g}")
        if worst is None or entry["ratio"] > details["fixtures"][worst]["ratio"]:
            worst = identifier
    for identifier, entry in details["exempt"].items():
        lines.append(f"{identifier:<48} exempt: timed {entry['timed']['median_us']:.4g} us, budget "
                     f"{entry['budget']['median_us']:.4g} us, budget {'met' if entry['budget_met'] else 'NOT met'}")
    if worst is not None:
        lines.append(f"worst ratio {details['fixtures'][worst]['ratio']:.2f}: {worst}")
    return "\n".join(lines)


_NUMBERS = re.compile(r"\[\s*(-?\d[\d.eE+-]*(?:,\s*-?\d[\d.eE+-]*)*)\s*\]")  # An array of numbers and nothing else.


def json_text(document: dict[str, Any]) -> str:
    """The document, indented, with every array of numbers (the raw samples) on one line."""
    text = json.dumps(document, indent=1, allow_nan=False)
    return _NUMBERS.sub(lambda match: "[" + re.sub(r",\s*", ", ", match.group(1)) + "]", text) + "\n"


def require_writable(path: Path) -> None:
    """Refuse an output that cannot be written, before a long run rather than after it."""
    nearest = path.resolve().parent
    while not nearest.exists():
        nearest = nearest.parent
    if not nearest.is_dir() or not os.access(nearest, os.W_OK) or path.is_dir() \
            or (path.exists() and not os.access(path, os.W_OK)):
        raise SuiteError(f"cannot write {path}")


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write a document in one step: a run that stops half way never leaves half a file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json_text(document), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    names = list(dict.fromkeys(name.strip() for name in text.split(",")))
    if not all(names):
        raise SuiteError(f"expected comma-separated names, not {text!r}")
    return names


def run_parent(arguments: argparse.Namespace) -> int:
    output = arguments.output
    details_path = arguments.details or output.with_name(f"{output.stem}-details{output.suffix}")
    if details_path.resolve() == output.resolve():
        raise SuiteError("--output and --details name the same file")
    areas, ops = parse_list(arguments.areas), parse_list(arguments.ops)
    partial_run = areas is not None or ops is not None
    module_names(areas, parse_list(arguments.modules))  # A misspelled area is refused here, before a worker starts.
    require_writable(details_path)
    if not partial_run:
        require_writable(output)
    expected = source_hash()
    environment = dict(os.environ, **REFERENCE_ENVIRONMENT)
    rounds: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mathforge-speed-") as scratch:
        for index in range(arguments.rounds):
            result = Path(scratch) / f"round-{index}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--round-index", str(index),
                       "--worker-result", str(result), "--repeat", str(arguments.repeat),
                       "--warmups", str(arguments.warmups), "--sample-ms", repr(arguments.sample_ms)]
            for flag, value in (("--areas", arguments.areas), ("--modules", arguments.modules),
                                ("--ops", arguments.ops), ("--cpu", arguments.cpu)):
                if value is not None:
                    command += [flag, str(value)]
            completed = subprocess.run(command, cwd=ROOT, env=environment)
            if completed.returncode != 0:
                raise SuiteError(f"round {index + 1} of {arguments.rounds} failed with exit status "
                                 f"{completed.returncode}; nothing was written")
            rounds.append(json.loads(result.read_text(encoding="utf-8")))
    hashes = {record["source_sha256"] for record in rounds} | {expected}
    if len(hashes) != 1:
        raise SuiteError("the source changed during the run, so the rounds measured different code "
                         f"({', '.join(sorted(h[:12] for h in hashes))}); nothing was written")
    first = rounds[0]
    meta = {"format": DETAILS_FORMAT, "source_sha256": expected, "mathforge_version": first["mathforge_version"],
            "python": first["python"], "platform": first["platform"], **first["environment"],
            "cpu": arguments.cpu, "rounds": arguments.rounds, "repeat": arguments.repeat,
            "warmups": arguments.warmups, "sample_ms": arguments.sample_ms,
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "modules": first["modules"],
            "areas": areas, "ops": ops, "partial": partial_run}
    details = build_details(rounds, meta)
    write_json(details_path, details)
    print(render_table(details))
    print(f"details: {details_path}")
    if partial_run:
        print("a partial run (--areas or --ops) writes the details only; run all areas for the speed file")
    else:
        write_json(output, build_criteria(details))
        print(f"speed file: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="the coverage gate's speed file, for example "
                                                    "docs/v08/speed/latest.json (not written by a partial run)")
    parser.add_argument("--details", type=Path, help="the details file (default: --output with -details added to "
                                                     "its stem)")
    parser.add_argument("--rounds", type=int, default=3, help="worker processes, one per round (default 3)")
    parser.add_argument("--repeat", type=int, default=9, help="samples per mode in a round (default 9)")
    parser.add_argument("--warmups", type=int, default=2, help="warmup passes per fixture and round (default 2)")
    parser.add_argument("--cpu", type=int, help="pin each worker to this logical CPU")
    parser.add_argument("--areas", help="comma-separated area modules to run, for example ntheory,discrete")
    parser.add_argument("--ops", help="comma-separated operations to run, for example det,rref")
    parser.add_argument("--sample-ms", type=float, default=2.0,
                        help="the least duration of a SymPy sample, in milliseconds (default 2)")
    parser.add_argument("--modules", help="comma-separated importable modules that define FIXTURES, in place of the "
                                          "area modules of benchmarks/speed/fixtures: for the suite's own tests")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--round-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # A value in a message may not be ASCII; a narrow console must not stop it.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if min(arguments.rounds, arguments.repeat, arguments.warmups) < 1:
        parser.error("--rounds, --repeat and --warmups must be positive")
    if not (math.isfinite(arguments.sample_ms) and arguments.sample_ms > 0):
        parser.error("--sample-ms must be a positive number")
    if arguments.cpu is not None and arguments.cpu < 0:
        parser.error("--cpu must not be negative")
    if arguments.worker:
        if arguments.worker_result is None:
            parser.error("--worker needs --worker-result")
    elif arguments.output is None:
        parser.error("--output is required")
    if arguments.modules is not None and arguments.areas is not None:
        parser.error("--modules replaces the area modules, so it cannot go with --areas")
    try:
        return run_worker(arguments) if arguments.worker else run_parent(arguments)
    except SuiteError as error:
        print(f"speed_suite: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
