"""Seeded rational integrands shared by the rational-integration tests and the run that pins their v0.7 values.

``sources(mathforge)`` takes the mathforge module to build with, so that the same integrands can be built by a v0.7
tree and by the current one. Only public, version-stable API is used: Symbol with a fixed uid, Symbol arithmetic and
rational_function. Run with a v0.7 tree's src first on PYTHONPATH, this file writes the pinned digests:

    PYTHONPATH=<v0.7 tree>/src python tests/v08/integrals/seeded.py tests/fixtures/v8/rational_integration_v07.json
"""

from __future__ import annotations

import hashlib
import json
import random
import sys

SEED = 20261019
COUNT = 200
UID = "00000000-0000-4000-8000-000000000071"


def sources(mathforge, count: int = COUNT, seed: int = SEED):
    """``count`` rational functions of x: numerators of degree <= 6, denominators of degree 1 to 6 that are products
    of linear factors, irreducible quadratics and cubics, some of them squared (repeated and complex poles)."""
    x = mathforge.Symbol("x", uid=UID)
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        factors, degree = [], 0
        while degree == 0 or (degree < 6 and rng.random() < 0.6):
            kind = rng.random()
            if kind < 0.45:
                factor, size = x - rng.randint(-5, 5), 1
            elif kind < 0.8:
                a, b = rng.randint(-4, 4), rng.randint(1, 6)
                factor, size = x * x + a * x + (a * a // 4 + b), 2  # Negative discriminant: complex poles.
            else:
                factor, size = x ** 3 - rng.randint(1, 3) * x + rng.randint(1, 3), 3
            power = 1 if rng.random() < 0.7 else 2
            if degree + size * power > 6:
                continue
            factors.append(factor ** power)
            degree += size * power
        denominator = factors[0]
        for factor in factors[1:]:
            denominator = denominator * factor
        coefficients = [rng.randint(-5, 5) for _ in range(rng.randint(0, 6) + 1)]
        if not any(coefficients):
            coefficients[0] = 1
        numerator = sum((c * x ** k for k, c in enumerate(coefficients) if c), mathforge.Rational(0))
        out.append(mathforge.rational_function(numerator, denominator, variable=x))
    return x, out


def digest(document) -> str:
    """The SHA-256 of a to_data document in canonical JSON."""
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def pinned(mathforge) -> dict:
    """The digests of the seeded integrands' antiderivatives under the given mathforge (the v0.7 tree's)."""
    from mathforge.serialization import to_data
    x, functions = sources(mathforge)
    values = [digest(to_data(mathforge.rational_antiderivative(f, x))) for f in functions]
    return {"seed": SEED, "count": COUNT, "mathforge": mathforge.__version__, "antiderivatives": values}


if __name__ == "__main__":
    import mathforge as _mathforge
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(pinned(_mathforge), handle, indent=1, sort_keys=True)
        handle.write("\n")
