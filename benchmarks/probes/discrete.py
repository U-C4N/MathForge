"""Discrete mathematics probes: counting functions and integer sequences."""

from math import comb, factorial

from . import Probe

GROUP = "discrete"


def fibonacci(n: int) -> int:
    """F(n) by fast doubling: F(2k) = F(k) (2 F(k+1) - F(k)), F(2k+1) = F(k)^2 + F(k+1)^2."""
    a, b = 0, 1  # F(k), F(k + 1) for k = the leading bits of n read so far.
    for bit in bin(n)[2:]:
        a, b = a * (2 * b - a), a * a + b * b
        if bit == "1":
            a, b = b, a + b
    return a


PROBES = (
    Probe("discrete_binomial", GROUP, "In how many ways can 3 of 10 items be chosen?",
          lambda mf: mf.binomial(10, 3), lambda v: v == comb(10, 3)),
    Probe("discrete_permutations", GROUP, "How many ordered arrangements of 2 out of 5 items are there?",
          lambda mf: mf.permutations_count(5, 2), lambda v: v == 20),
    Probe("discrete_multinomial", GROUP, "What is the multinomial coefficient 6!/(2! 3! 1!)?",
          lambda mf: mf.multinomial((2, 3, 1)), lambda v: v == 60),
    Probe("discrete_partitions", GROUP, "In how many ways can 10 be written as a sum of positive integers?",
          lambda mf: mf.discrete.partitions(10), lambda v: v == 42),
    Probe("discrete_catalan", GROUP, "What is the 10th Catalan number?", lambda mf: mf.catalan(10),
          lambda v: v == comb(20, 10) // 11),
    Probe("discrete_stirling", GROUP, "In how many ways can 5 items be split into 2 non-empty groups?",
          lambda mf: mf.stirling(5, 2), lambda v: v == 15),
    Probe("discrete_fibonacci_200", GROUP, "What is the 200th Fibonacci number?", lambda mf: mf.fibonacci(200),
          lambda v: v == fibonacci(200)),
    Probe("discrete_fibonacci_5000", GROUP, "What is the 5000th Fibonacci number (1045 digits)?",
          lambda mf: mf.fibonacci(5000), lambda v: v == fibonacci(5000)),
    Probe("discrete_derangements", GROUP, "How many permutations of 4 items leave no item in place?",
          lambda mf: mf.discrete.derangements(4), lambda v: v == 9),
    Probe("discrete_factorial_30", GROUP, "What is 30!?", lambda mf: mf.factorial(30),
          lambda v: v == factorial(30)),
)
