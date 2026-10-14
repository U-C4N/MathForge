"""Finite, exact rational probability without implicit sampling or independence."""

from dataclasses import dataclass

from .errors import InvalidInput
from .limits import computation, current_budget
from .model import Rational

ZERO, ONE = Rational(0), Rational(1)


def _rational(value):
    if type(value) is int:
        value = Rational(value)
    if type(value) is not Rational:
        raise InvalidInput("Expected an exact integer or rational")
    current_budget().check_int(value.numerator)
    current_budget().check_int(value.denominator)
    if value.denominator <= 0 or Rational(value.numerator, value.denominator) != value:
        raise InvalidInput("Noncanonical rational")
    return value


def _values(values):
    if isinstance(values, (str, bytes, dict)):
        raise InvalidInput("Expected a finite iterable of rational values")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise InvalidInput("Expected a finite iterable of rational values") from exc
    result = []
    for value in iterator:
        current_budget().nodes()
        result.append(_rational(value))
    return tuple(result)


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise InvalidInput(f"{name} must be a nonnegative integer")
    current_budget().check_int(value)
    return value


def _parameter(value):
    value = _rational(value)
    if not ZERO <= value <= ONE:
        raise InvalidInput("Probability must lie between zero and one")
    return value


def _total(values):
    result = ZERO
    for value in values:
        current_budget().tick()
        result += value
    return result


def _choose(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        current_budget().tick()
        current_budget().product(result, n - k + i)
        result = result * (n - k + i) // i
        current_budget().check_int(result)
    return result


@dataclass(frozen=True, init=False)
class FiniteDistribution:
    """Sorted positive probability mass; duplicate outcomes are aggregated."""

    outcomes: tuple[Rational, ...]
    probabilities: tuple[Rational, ...]

    def __init__(self, outcomes, probabilities):
        with computation():
            outcomes, probabilities = _values(outcomes), _values(probabilities)
            if len(outcomes) != len(probabilities) or not outcomes:
                raise InvalidInput("Outcomes and probabilities must have equal nonzero length")
            mass = {}
            for outcome, probability in zip(outcomes, probabilities):
                current_budget().nodes()
                probability = _parameter(probability)
                mass[outcome] = mass.get(outcome, ZERO) + probability
            if _total(mass.values()) != ONE:
                raise InvalidInput("Probabilities must sum exactly to one")
            support = sorted(outcome for outcome, weight in mass.items() if weight != ZERO)
            current_budget().nodes(2 * len(support))
            object.__setattr__(self, "outcomes", tuple(support))
            object.__setattr__(self, "probabilities", tuple(mass[outcome] for outcome in support))


def _distribution(value):
    if type(value) is not FiniteDistribution:
        raise InvalidInput("Expected a FiniteDistribution")
    if (type(value.outcomes) is not tuple or type(value.probabilities) is not tuple
            or len(value.outcomes) != len(value.probabilities) or not value.outcomes):
        raise InvalidInput("Malformed distribution")
    previous = None
    total = ZERO
    for outcome, weight in zip(value.outcomes, value.probabilities):
        current_budget().nodes(2)
        if type(outcome) is not Rational or type(weight) is not Rational:
            raise InvalidInput("Noncanonical distribution entries")
        _rational(outcome)
        _rational(weight)
        if weight <= ZERO or (previous is not None and outcome <= previous):
            raise InvalidInput("Distribution requires positive weights and sorted unique support")
        previous = outcome
        total += weight
    if total != ONE:
        raise InvalidInput("Distribution weights must sum to one")
    return value


def bernoulli(p):
    with computation():
        p = _parameter(p)
        return FiniteDistribution((0, 1), (ONE - p, p))


def binomial_distribution(n, p):
    with computation():
        n, p = _integer(n, "n"), _parameter(p)
        current_budget().nodes(n + 1)
        if p == ZERO or p == ONE:
            return FiniteDistribution((0 if p == ZERO else n,), (ONE,))
        weights = []
        weight = (ONE - p) ** n
        for k in range(n + 1):
            current_budget().tick()
            weights.append(weight)
            if k < n:
                weight = weight * Rational(n - k, k + 1) * p / (ONE - p)
        return FiniteDistribution(range(n + 1), weights)


def discrete_uniform(outcomes):
    with computation():
        outcomes = _values(outcomes)
        if not outcomes or len(set(outcomes)) != len(outcomes):
            raise InvalidInput("Uniform outcomes must be nonempty and distinct")
        current_budget().nodes(len(outcomes))
        return FiniteDistribution(outcomes, (Rational(1, len(outcomes)),) * len(outcomes))


def _hyperparameters(population, successes, draws):
    population = _integer(population, "population")
    successes, draws = _integer(successes, "successes"), _integer(draws, "draws")
    if successes > population or draws > population:
        raise InvalidInput("Successes and draws cannot exceed the population")
    return population, successes, draws


def hypergeometric(population, successes, draws):
    with computation():
        population, successes, draws = _hyperparameters(population, successes, draws)
        lower, upper = max(0, draws - population + successes), min(draws, successes)
        current_budget().nodes(upper - lower + 1)
        denominator = _choose(population, draws)
        weights = []
        for k in range(lower, upper + 1):
            current_budget().tick()
            first, second = _choose(successes, k), _choose(population - successes, draws - k)
            current_budget().product(first, second)
            weights.append(Rational(first * second, denominator))
        return FiniteDistribution(range(lower, upper + 1), weights)


def probability(distribution, event):
    with computation():
        distribution, event = _distribution(distribution), set(_values(event))
        return _total(weight for outcome, weight in zip(distribution.outcomes, distribution.probabilities)
                      if outcome in event)


def condition(distribution, event):
    with computation():
        distribution, event = _distribution(distribution), set(_values(event))
        normalizer = _total(weight for outcome, weight in zip(distribution.outcomes, distribution.probabilities)
                            if outcome in event)
        if normalizer == ZERO:
            raise InvalidInput("Cannot condition on an event of probability zero")
        outcomes, weights = [], []
        for outcome, weight in zip(distribution.outcomes, distribution.probabilities):
            current_budget().nodes(2)
            if outcome in event:
                outcomes.append(outcome)
                weights.append(weight / normalizer)
        return FiniteDistribution(outcomes, weights)


def cdf(distribution, x):
    with computation():
        distribution, x = _distribution(distribution), _rational(x)
        return _total(weight for outcome, weight in zip(distribution.outcomes, distribution.probabilities)
                      if outcome <= x)


def moment(distribution, order):
    with computation():
        distribution, order = _distribution(distribution), _integer(order, "order")
        return _total(weight * outcome ** order
                      for outcome, weight in zip(distribution.outcomes, distribution.probabilities))


def expectation(distribution):
    return moment(distribution, 1)


def distribution_variance(distribution):
    with computation():
        distribution = _distribution(distribution)
        center = moment(distribution, 1)
        return _total(weight * (outcome - center) ** 2
                      for outcome, weight in zip(distribution.outcomes, distribution.probabilities))


ARGUMENTS = {
    "bernoulli": {"p"}, "binomial_distribution": {"n", "p"},
    "discrete_uniform": {"outcomes"}, "hypergeometric": {"population", "successes", "draws"},
    "probability": {"distribution", "event"}, "condition": {"distribution", "event"},
    "cdf": {"distribution", "x"}, "moment": {"distribution", "order"},
    "expectation": {"distribution"}, "distribution_variance": {"distribution"},
}
TYPES = {"FiniteDistribution": FiniteDistribution}


def produce(op, args):
    with computation():
        if op not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[op]:
            raise InvalidInput("Unknown probability operation or incorrect arguments")
        return globals()[op](**args), ()


def check_operation(op, args, value, evidence):
    """Check definitions/PMF identities, never call a probability producer."""
    with computation():
        try:
            if (op not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[op]
                    or type(evidence) is not tuple or evidence != ()):
                return False
            if op in ("bernoulli", "binomial_distribution", "discrete_uniform", "hypergeometric", "condition"):
                _distribution(value)
                actual = dict(zip(value.outcomes, value.probabilities))
                expected = {}
                if op in ("bernoulli", "binomial_distribution"):
                    n = 1 if op == "bernoulli" else _integer(args["n"], "n")
                    p = _parameter(args["p"])
                    current_budget().nodes(n + 1)
                    for k in range(n + 1):
                        current_budget().tick()
                        weight = Rational(_choose(n, k)) * p**k * (ONE-p)**(n-k)
                        if weight != ZERO:
                            expected[Rational(k)] = weight
                elif op == "discrete_uniform":
                    outcomes = _values(args["outcomes"])
                    if not outcomes or len(set(outcomes)) != len(outcomes):
                        return False
                    current_budget().nodes(len(outcomes))
                    expected = {outcome: Rational(1, len(outcomes)) for outcome in outcomes}
                elif op == "hypergeometric":
                    n, k, draws = _hyperparameters(args["population"], args["successes"], args["draws"])
                    lower, upper = max(0, draws - n + k), min(draws, k)
                    current_budget().nodes(upper - lower + 1)
                    # Verify adjacent-mass balance plus exact normalization.
                    if tuple(value.outcomes) != tuple(Rational(i) for i in range(lower, upper + 1)):
                        return False
                    for i in range(lower, upper):
                        current_budget().tick()
                        current_budget().product(i + 1, n-k-draws+i+1)
                        current_budget().product(k-i, draws-i)
                        left = actual[Rational(i + 1)] * Rational((i + 1) * (n-k-draws+i+1))
                        right = actual[Rational(i)] * Rational((k-i) * (draws-i))
                        if left != right:
                            return False
                    return True
                else:
                    source = _distribution(args["distribution"])
                    event = set(_values(args["event"]))
                    selected = []
                    total = ZERO
                    for outcome, weight in zip(source.outcomes, source.probabilities):
                        current_budget().nodes()
                        if outcome in event:
                            selected.append((outcome, weight))
                            total += weight
                    if total == ZERO:
                        return False
                    expected = {outcome: weight / total for outcome, weight in selected}
                return actual == expected
            if type(value) is not Rational:
                return False
            _rational(value)
            source = _distribution(args["distribution"])
            event = set(_values(args["event"])) if op == "probability" else None
            point = _rational(args["x"]) if op == "cdf" else None
            order = _integer(args["order"], "order") if op == "moment" else 1
            answer, second = ZERO, ZERO
            for outcome, weight in zip(source.outcomes, source.probabilities):
                current_budget().tick()
                if op == "probability":
                    if outcome in event:
                        answer += weight
                elif op == "cdf":
                    if outcome <= point:
                        answer += weight
                else:
                    answer += weight * outcome**order
                    if op == "distribution_variance":
                        second += weight * outcome * outcome
            if op == "distribution_variance":
                answer = second - answer * answer
            return value == answer
        except (InvalidInput, TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError):
            return False
