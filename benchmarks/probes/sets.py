"""Set probes: unions of real intervals and the operations on them."""

from . import Probe, intervals, symbols

GROUP = "sets"


def _set(mf, *items):
    """An IntervalSet from (lower, upper, left_closed, right_closed) tuples of ints."""
    return mf.IntervalSet(tuple(mf.Interval(mf.Rational(lower), mf.Rational(upper), left, right)
                                for lower, upper, left, right in items))


def _closed_0_2(mf):
    return _set(mf, (0, 2, True, True))


def _half_open_1_3(mf):
    return _set(mf, (1, 3, True, False))


def _solve_as_set(mf):
    x, = symbols(mf, "x")
    return mf.solve(mf.parse_problem("x^2 < 4", symbols={"x": x}), for_=x)


PROBES = (
    Probe("sets_union", GROUP, "What is [0, 2] united with [1, 3)?",
          lambda mf: _closed_0_2(mf).union(_half_open_1_3(mf)), lambda v: intervals(v) == ((0, 3, True, False),)),
    Probe("sets_intersection", GROUP, "What is [0, 2] intersected with [1, 3)?",
          lambda mf: _closed_0_2(mf).intersection(_half_open_1_3(mf)),
          lambda v: intervals(v) == ((1, 2, True, True),)),
    Probe("sets_complement", GROUP, "What is the complement of [0, 2] in the real line?",
          lambda mf: mf.sets.set_complement(_closed_0_2(mf)),
          lambda v: intervals(v) == ((None, 0, False, False), (2, None, False, False))),
    Probe("sets_difference", GROUP, "What is [0, 3] minus (1, 2)?",
          lambda mf: mf.sets.set_difference(_set(mf, (0, 3, True, True)), _set(mf, (1, 2, False, False))),
          lambda v: intervals(v) == ((0, 1, True, True), (2, 3, True, True))),
    Probe("sets_union_operator", GROUP, "Unite [0, 2] and [1, 3) with the | operator.",
          lambda mf: _closed_0_2(mf) | _half_open_1_3(mf), lambda v: intervals(v) == ((0, 3, True, False),)),
    Probe("sets_contains", GROUP, "Does [0, 2] contain 2?", lambda mf: _closed_0_2(mf).contains(2),
          lambda v: v is True),
    Probe("sets_solution_interval", GROUP, "Give the solutions of x^2 < 4 as a set of intervals.", _solve_as_set,
          lambda v: intervals(v) == ((-2, 2, False, False),)),
)
