"""Internal developer tooling for the v0.8 coverage gate.

The gate makes the release's coverage claim measurable: a local inventory of SymPy subtopics (never committed), the
digest that freezes it, and the checks that count which rows MathForge covers. This package is not part of mathforge:
nothing under src/ imports it, and README.md and documentation/ never mention it. Import it with tools/ on sys.path,
as the tests and the scripts in tools/ do.
"""
