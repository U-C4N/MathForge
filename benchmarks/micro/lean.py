"""Micro benchmark of the lean value path: the generic runner on integer_gcd(84, 30) and on the legacy det of a 2x2.

    python benchmarks/micro/lean.py [--cpu N] [--rounds 9] [--check]

It prints the median microseconds per call of ``mathforge.lean.run_value`` on both inputs, against their targets
(15 us and 20 us), and, for reference, the value functions (which take the lean path) and the structured path.
``--cpu`` pins the process to one CPU (sched_setaffinity on Linux, SetProcessAffinityMask on Windows). ``--check``
exits with status 1 when a target is missed. The repository's src comes first on sys.path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import time
import timeit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mathforge  # noqa: E402
from mathforge.lean import FALLBACK, run_value  # noqa: E402

TARGETS_US = {"run_value integer_gcd(84, 30)": 15.0, "run_value det(2x2)": 20.0}


def pin(cpu: int) -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    elif sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1 << cpu):
            raise OSError(f"SetProcessAffinityMask failed for CPU {cpu}")
    else:
        raise OSError("Pinning to a CPU is not supported on this platform")


def median_us(call, rounds: int) -> float:
    """The median over rounds of the mean time per call, each round lasting about 0.2 s."""
    timer = timeit.Timer(call)
    number = 1
    while True:  # Calibrate the calls per round.
        start = time.perf_counter()
        timer.timeit(number)
        if time.perf_counter() - start >= 0.2 or number >= 1 << 20:
            break
        number *= 2
    return statistics.median(timer.repeat(rounds, number)) / number * 1e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, help="Pin the process to this CPU")
    parser.add_argument("--rounds", type=int, default=9, help="Timed rounds per case (default 9)")
    parser.add_argument("--check", action="store_true", help="Exit with status 1 when a target is missed")
    options = parser.parse_args(argv)
    if options.cpu is not None:
        pin(options.cpu)
    gcd_arguments = {"a": 84, "b": 30}
    matrix = mathforge.Matrix(((1, 2), (3, 4)))
    det_arguments = {"A": matrix}
    # The runner must answer, not fall back: a fallback would time a no-op.
    for operation, arguments, expected in (("integer_gcd", gcd_arguments, 6),
                                           ("det", det_arguments, mathforge.Rational(-2))):
        value = run_value(operation, arguments)
        if value is FALLBACK or value != expected:
            raise SystemExit(f"run_value({operation!r}) gave {value!r}, not {expected!r}")
    cases = {
        "run_value integer_gcd(84, 30)": lambda: run_value("integer_gcd", gcd_arguments),
        "run_value det(2x2)": lambda: run_value("det", det_arguments),
        "mathforge.integer_gcd(84, 30)": lambda: mathforge.integer_gcd(84, 30),
        "mathforge.det(2x2)": lambda: mathforge.det(matrix),
        "operations.integer_gcd(84, 30)": lambda: mathforge.operations.integer_gcd(84, 30),
        "operations.det(2x2)": lambda: mathforge.operations.det(matrix),
    }
    print(f"Python {sys.version.split()[0]}, mathforge {mathforge.__file__}, cpu {options.cpu}")
    missed = []
    for label, call in cases.items():
        median = median_us(call, options.rounds)
        target = TARGETS_US.get(label)
        verdict = "" if target is None else (f"target {target:.0f} us: " + ("met" if median <= target else "MISSED"))
        if target is not None and median > target:
            missed.append(label)
        print(f"{label:34s} {median:10.2f} us  {verdict}")
    return 1 if options.check and missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
