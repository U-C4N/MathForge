"""The v0.8 speed suite's fixtures: the interface in fixture.py and one module per registry area in fixtures/.

benchmarks/speed_suite.py times every fixture against the fastest public SymPy call that returns the same mathematical
object; tools/snapshot_v07_values.py stores what v0.7 returns for the same inputs, and tests/v08/test_values_v07.py
holds every later version to those values. benchmarks/ is a namespace package: import this one from the repository
root, as ``benchmarks.speed.fixture``.
"""
