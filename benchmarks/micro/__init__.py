"""Micro benchmarks with explicit targets for MathForge's hot paths.

Each module here is a script, ``python benchmarks/micro/<name>.py [--cpu N] [--rounds 9]``, that measures the
checkout's own ``src``. ``--cpu`` pins the process to one CPU (Linux and Windows). Every round times each case once,
interleaved with the others, over enough calls to last about 50 ms; the reported figure is the median over the rounds
in nanoseconds per call. Targets are checked after the table, and a missed one makes the script exit with status 1.
Times are observations of one machine, not CI gates.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
from statistics import median
import sys
import timeit
from typing import Callable, ContextManager


@dataclass(frozen=True)
class Case:
    """One timed statement. ``context`` makes the context manager entered around each sample (a budget, say)."""

    label: str
    statement: str
    namespace: dict = field(default_factory=dict)
    context: Callable[[], ContextManager] = nullcontext


@dataclass(frozen=True)
class Target:
    """``value`` must be at most ``limit``; ``unit`` is printed after both."""

    label: str
    value: float
    limit: float
    unit: str


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cpu", type=int, help="pin the process to this CPU before measuring")
    parser.add_argument("--rounds", type=int, default=9, help="interleaved rounds per case (default 9)")
    parsed = parser.parse_args()
    if parsed.rounds < 1:
        parser.error("--rounds must be positive")
    if parsed.cpu is not None and parsed.cpu < 0:
        parser.error("--cpu must be a CPU number")
    return parsed


def pin(cpu: int) -> None:
    """Restrict this process to one CPU: sched_setaffinity on Linux, SetProcessAffinityMask on Windows."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    elif sys.platform == "win32":
        import ctypes
        if cpu >= 8 * ctypes.sizeof(ctypes.c_size_t):
            raise SystemExit(f"--cpu {cpu} is outside this process's processor group")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1 << cpu):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        raise SystemExit(f"--cpu is supported on Linux and Windows, not on {sys.platform}")


def measure(cases: list[Case], rounds: int, sample_seconds: float = 0.05) -> dict[str, float]:
    """Median nanoseconds per call of each case over interleaved rounds."""
    timers = {case.label: timeit.Timer(case.statement, globals=case.namespace) for case in cases}
    numbers = {}
    for case in cases:  # Calibrate (and warm up) once, inside the case's context.
        with case.context():
            loops, seconds = timers[case.label].autorange()
        numbers[case.label] = max(1, round(loops * sample_seconds / seconds))
    samples: dict[str, list[float]] = {case.label: [] for case in cases}
    for _ in range(rounds):
        for case in cases:
            number = numbers[case.label]
            with case.context():
                seconds = timers[case.label].timeit(number)
            samples[case.label].append(seconds / number * 1e9)
    return {label: median(values) for label, values in samples.items()}


def report(title: str, medians: dict[str, float], targets: list[Target]) -> bool:
    """Print the table and the targets; True when every target is met."""
    print(title)
    width = max(len(label) for label in medians)
    print(f"{'case':<{width}}  median ns/call")
    for label, nanoseconds in medians.items():
        print(f"{label:<{width}}  {nanoseconds:14.1f}")
    met = True
    if targets:
        print("targets")
        width = max(len(target.label) for target in targets)
        for target in targets:
            ok = target.value <= target.limit
            met &= ok
            print(f"  {target.label:<{width}}  {target.value:8.2f}{target.unit} <= {target.limit:.2f}{target.unit}"
                  f"  {'met' if ok else 'MISSED'}")
    return met
