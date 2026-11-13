"""Statistics probes: finite distributions, conditional probability and descriptive statistics of data."""

from fractions import Fraction

from . import Probe, exact, signed_sqrt

GROUP = "stats"
DIE = (1, 2, 3, 4, 5, 6)
DATA = (2, 4, 4, 4, 5, 5, 7, 9)


def _is(value):
    return lambda v: exact(v) == value


PROBES = (
    Probe("stats_die_expectation", GROUP, "What is the expected value of a fair die?",
          lambda mf: mf.expectation(mf.discrete_uniform(DIE)), _is(Fraction(7, 2))),
    Probe("stats_die_variance", GROUP, "What is the variance of a fair die?",
          lambda mf: mf.distribution_variance(mf.discrete_uniform(DIE)), _is(Fraction(35, 12))),
    Probe("stats_binomial_probability", GROUP, "For X ~ Binomial(10, 1/4), what is P(X = 3)?",
          lambda mf: mf.probability(mf.binomial_distribution(10, mf.Rational(1, 4)), (3,)),
          _is(Fraction(32805, 131072))),
    Probe("stats_conditional_probability", GROUP, "For X ~ Binomial(4, 1/2), what is P(X = 3 | X >= 2)?",
          lambda mf: mf.probability(mf.condition(mf.binomial_distribution(4, mf.Rational(1, 2)), (2, 3, 4)), (3,)),
          _is(Fraction(4, 11))),
    Probe("stats_mean", GROUP, "What is the mean of 1, 2, 3, 4?", lambda mf: mf.mean((1, 2, 3, 4)),
          _is(Fraction(5, 2))),
    Probe("stats_median", GROUP, "What is the median of 5, 1, 3?", lambda mf: mf.median((5, 1, 3)), _is(3)),
    Probe("stats_sample_variance", GROUP, "What is the sample variance of 2, 4, 4, 4, 5, 5, 7, 9?",
          lambda mf: mf.variance(DATA, ddof=1), _is(Fraction(32, 7))),
    Probe("stats_population_stddev", GROUP, "What is the population standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?",
          lambda mf: mf.stddev(DATA), _is(2)),
    Probe("stats_stddev_two_values", GROUP, "What is the population standard deviation of 1 and 2?",
          lambda mf: mf.stddev((1, 2)), _is(Fraction(1, 2))),
    Probe("stats_covariance", GROUP, "What is the population covariance of (1, 2, 3) and (2, 4, 7)?",
          lambda mf: mf.covariance((1, 2, 3), (2, 4, 7)), _is(Fraction(5, 3))),
    Probe("stats_linear_regression", GROUP, "Fit a least-squares line to the points (1, 2), (2, 4), (3, 7).",
          lambda mf: mf.stats.linear_regression((1, 2, 3), (2, 4, 7))),
    Probe("stats_correlation", GROUP, "What is the correlation coefficient of (1, 2, 3) and (2, 4, 7)?",
          lambda mf: mf.stats.correlation((1, 2, 3), (2, 4, 7)), lambda v: signed_sqrt(v) == (1, Fraction(75, 76))),
    Probe("stats_geometric_distribution", GROUP, "Build the geometric distribution with success probability 1/3.",
          lambda mf: mf.stats.geometric_distribution(mf.Rational(1, 3))),
    Probe("stats_poisson_distribution", GROUP, "Build the Poisson distribution with mean 2.",
          lambda mf: mf.stats.poisson_distribution(2)),
)
