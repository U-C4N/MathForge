"""Inputs shared by the statistics tests: seeded random samples, outcomes and the corpus fixtures of the area."""

from __future__ import annotations

import importlib
from pathlib import Path
import random
import sys

import mathforge as mf
from mathforge.model import trusted_rational

R = mf.Rational
REPOSITORY = Path(__file__).resolve().parents[3]
DESCRIPTIVE = ("mean", "median", "variance", "stddev", "covariance")
FLAVORS = ("ints", "fractions", "decimals", "mixed", "ties", "wide")


def outcome(call):
    """("value", value) or ("error", exception class, code); a failed Result counts as its unwrap's error."""
    try:
        value = call()
        if isinstance(value, mf.Result):
            value = value.unwrap()
        return ("value", value)
    except Exception as error:
        return ("error", type(error), getattr(error, "code", None))


def structured(operation, *arguments):
    """The structured path's outcome of an operation of the area, for its positional arguments."""
    return outcome(lambda: getattr(mf.operations, operation)(*arguments))


def sample(rng: random.Random, size: int, flavor: str) -> tuple:
    """A tuple of `size` ints and Rationals of one flavor."""
    if flavor == "ints":
        return tuple(rng.randint(-50, 50) for _ in range(size))
    if flavor == "fractions":
        return tuple(R(rng.randint(-40, 40), rng.randint(1, 12)) for _ in range(size))
    if flavor == "decimals":
        return tuple(R(rng.randint(1000, 9999), 100) for _ in range(size))
    if flavor == "mixed":
        return tuple(rng.randint(-9, 9) if rng.random() < 0.4 else R(rng.randint(-30, 30), rng.randint(1, 9))
                     for _ in range(size))
    if flavor == "ties":
        pool = [R(rng.randint(-6, 6), rng.randint(1, 4)) for _ in range(max(1, size // 3))] + [0, 1]
        return tuple(rng.choice(pool) for _ in range(size))
    if flavor == "wide":
        return tuple(R(rng.randint(-2 ** 200, 2 ** 200), rng.randint(1, 2 ** 100)) for _ in range(size))
    raise ValueError(flavor)


def random_cases(seed: int, count: int):
    """(size, flavor, sample, other sample of the same size and flavor, ddof) for `count` seeded random draws."""
    rng = random.Random(seed)
    for index in range(count):
        size = rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 40, 40, 64, 300 if index % 25 == 0 else 12))
        flavor = FLAVORS[index % len(FLAVORS)]
        yield size, flavor, sample(rng, size, flavor), sample(rng, size, flavor), rng.randrange(size)


def distinct_outcomes(rng: random.Random, size: int) -> tuple:
    """`size` distinct ints and Rationals in a random order."""
    seen, result = set(), []
    while len(result) < size:
        value = rng.randint(-60, 60) if rng.random() < 0.3 else R(rng.randint(-200, 200), rng.randint(1, 14))
        key = R(value) if type(value) is int else value
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def exact(item):
    """The Rational of an int or a Rational entry."""
    return R(item) if type(item) is int else item


def forged(numerator, denominator):
    """A Rational that its constructor would never make: no validation, no reduction."""
    return trusted_rational(numerator, denominator)


def fixtures():
    """The speed fixtures and edge cases of the stats area (benchmarks/speed/fixtures/stats.py)."""
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    return importlib.import_module("benchmarks.speed.fixtures.stats")


def corpus_calls(operations):
    """(label, operation, positional arguments) of the fixture variants 0-2 and the edge cases of the operations that
    pass no explicit limits, rebuilt from the fixture module. Keyword arguments are made positional."""
    module = fixtures()
    calls = []
    for fixture in module.FIXTURES:
        if fixture.operation in operations:
            for variant in range(min(3, fixture.variants)):
                function, args, kwargs = fixture.mathforge(variant)
                calls.append((f"{fixture.id}#{variant}", fixture.operation, (*args, *kwargs.values())))
    for edge in module.EDGE_CASES:
        if edge.operation in operations:
            function, args, kwargs = edge.mathforge()
            if "limits" not in kwargs:
                calls.append((f"{edge.operation}/edge/{edge.label}", edge.operation, (*args, *kwargs.values())))
    return calls
