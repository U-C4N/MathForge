"""One module per registry area that holds v0.7 operations: ntheory, discrete, stats, algebra, expr, equations, analysis,
integrals, series and linalg.

Each ``<area>.py`` defines two tuples:

* ``FIXTURES: tuple[Fixture | ExemptFixture, ...]``, the timed comparisons of the area's operations. Every operation
  has at least one ``micro`` and one ``realistic`` Fixture; ``verify`` has ExemptFixtures instead.
* ``EDGE_CASES: tuple[EdgeCase, ...]``, inputs that are never timed and exist for the v0.7 value corpus: the errors an
  operation can raise, one input past a default limit and the ``True``, ``1.5``, ``"1"`` and ``None`` type edges.

A module imports SymPy only inside its ``sympy`` builders, never at module level, so that the unit suite, the
snapshot tool and the corpus test run without SymPy. Symbols come from ``benchmarks.speed.fixture.symbol(name, k)``,
which has a fixed uid, so that every value, and the corpus entry that stores it, is the same in every process.

This package itself imports nothing; consumers import the area modules by name.
"""
