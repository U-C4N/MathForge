"""Independent checker of the family mathforge.areas.discrete.counting: partitions, derangements, bell and stirling1.

The checker never reruns the producer's algorithms: it verifies each claim by a recurrence run from the claim itself,
a second identity or an evaluation of a polynomial. Every argument must be an exact int >= 0 (type(x) is int: a bool,
a float, a str or a Rational is no integer), the value an exact int, and the evidence exactly the layout below (a
list is no tuple); anything else answers False, and so does a claim on an input outside the operation's domain
(a negative argument), which has no value at all. Work past the budget raises ResourceLimitError, which leaves a claim
unverified, never verified; the Work paragraph at the end states the charges.

The theorem that makes True imply the claim, for each operation:

partitions(n) = v, evidence (table,).
    table = (t_0, ..., t_n) holds exact ints, t_0 = 1, v = t_n, and for every m = 1, ..., n
        t_m = sum over k >= 1 of (-1)**(k + 1) * (t_(m - k(3k - 1)/2) + t_(m - k(3k + 1)/2)),
    the terms with a negative index left out. The checker generates the generalized pentagonal numbers 1, 2, 5, 7, 12,
    15, ... and their signs + + - - + + ... itself, by adding to 0 in turn the odd numbers 1, 3, 5, ... and the natural
    numbers 1, 2, 3, ...: k(3k - 1)/2 - (k - 1)(3k - 2)/2 = 2k - 1 and k(3k + 1)/2 - k(3k - 1)/2 = k. Euler's pentagonal
    number theorem, prod_(k >= 1) (1 - x**k) = sum_(j in Z) (-1)**j x**(j(3j - 1)/2), times the generating function
    sum_m p(m) x**m = prod_(k >= 1) 1/(1 - x**k) gives 1, and the coefficient of x**m (m >= 1) of that product is the
    recurrence above for p. It determines t_m from t_0, ..., t_(m - 1), so t_0 = 1 = p(0) and induction give t_m = p(m)
    for every m <= n, and v = t_n = p(n). The checker verifies the entries in increasing order and adds only entries
    that it has verified, so a forged giant entry is refused before any arithmetic touches it. A second, independent
    identity is checked at m = n: n p(n) = sum_(k = 1..n) sigma(k) p(n - k), the logarithmic derivative of the
    generating function (x P'(x)/P(x) = sum_k sigma(k) x**k), with the divisor sums sigma(k) from a sieve of the
    checker's own. It holds for the partition numbers whatever the pentagonal numbers are, so it guards the checker's
    own generation of them: a table that satisfied a wrongly generated recurrence would fail it.

derangements(n) = v, evidence ().
    v >= 0, and the inverse recurrence x_n = v, x_(m - 1) = (x_m - (-1)**m)/m is exact (the division leaves no
    remainder) for m = n, n - 1, ..., 1 and ends at x_0 = 1. The derangement numbers satisfy D(0) = 1 and
    D(m) = m D(m - 1) + (-1)**m for m >= 1 (D(m) = m! sum_(i = 0..m) (-1)**i/i!, by inclusion and exclusion over the
    fixed points). Upward from x_0 = 1 = D(0), x_m = m x_(m - 1) + (-1)**m = m D(m - 1) + (-1)**m = D(m), so
    v = x_n = D(n). A wrong v differs from D(n) by some e != 0, and every exact step divides that difference by m: the
    chain meets a remainder, or it ends at 1 + e/n! != 1. v < 2**max(1, n * bit_length(n)) is tested first
    (D(n) <= n! <= n**n, and D(0) = 1), so a forged giant v costs nothing.

bell(n) = v, evidence ().
    n! v = sum_(j = 0..n) C(n, j) j**n D(n - j), with 0**0 = 1, where the checker computes the binomial coefficients,
    the powers (square-and-multiply of its own, every product charged) and the derangement numbers D(i) (D(0) = 1,
    D(i) = i D(i - 1) + (-1)**i) itself. The identity: B(n) = sum_(k = 0..n) S(n, k) and
    S(n, k) = (1/k!) sum_(j = 0..k) (-1)**(k - j) C(k, j) j**n (inclusion and exclusion over the surjections onto k
    blocks), and exchanging the two sums, sum_(k = j..n) (-1)**(k - j)/(k - j)! = D(n - j)/(n - j)!, so
    B(n) = sum_j j**n D(n - j)/(j! (n - j)!). It is a formula of its own, no recurrence of the Bell numbers (Bell's
    triangle, B(n + 1) = sum_k C(n, k) B(k)): an N-version check, for which no certificate is needed. The size test
    runs first: 2**(n - 1) <= B(n) <= n! for n >= 1 (the 2**(n - 1) set partitions into at most two blocks, and the
    permutation whose cycles are the blocks tells set partitions apart), and B(0) = 1, so the bit length of v lies
    between max(1, n) and max(1, n * bit_length(n)); a forged giant v is refused before any product, also for n = 0.
    The work depends on n alone, apart from the product n! v.

stirling1(n, k) = v, evidence (row,).
    row = (r_0, ..., r_n) holds n + 1 exact ints >= 0, v = r_k for k <= n and v = 0 for k > n, and the polynomial
    P(x) = r_0 + r_1 x + ... + r_n x**n takes the values of R(x) = x(x + 1)...(x + n - 1) at the n + 1 integer points
    1, 0, -1, ..., -(n - 1): P(1) = n! = R(1), and P(-j) = 0 = R(-j) for j = 0, ..., n - 1. The checker divides P by
    x + 0, x + 1, ..., x + n - 1 in turn (synthetic division, the quotient's coefficients from the top); each remainder,
    the current quotient's value at -j, must vanish, which it does exactly when P(-j) does, since the factors x + i
    divided out before are not 0 at -j. Two polynomials of degree <= n that agree at n + 1 points are equal, so P = R,
    and the unsigned Stirling numbers of the first kind are the coefficients of the rising factorial:
    x(x + 1)...(x + n - 1) = sum_k c(n, k) x**k (the permutations of n items counted by their cycles). So
    r_i = c(n, i) for every i, and v = c(n, k), which is 0 for k > n. The full row is the evidence for every k.
    The order of the tests bounds the work by n. Cheap necessary conditions come first: every entry has at most
    n * bit_length(n) bits (c(n, i) <= n! <= n**n), and r_n = c(n, n) = 1. Then P(1) = n! (the product 2 * 3 * ...
    stops as soon as it passes the sum of the row), after which no entry exceeds n!, since all are >= 0. Then the
    divisions. After the division by x + j the quotient of R is (x + j + 1)...(x + n - 1), whose coefficients are
    >= 0 and sum to its value at 1, n!/(j + 1)!: the checker refuses at once a coefficient of P's quotient outside
    [0, n!/(j + 1)!], which the true row never has (so the refusal costs no completeness), and then no integer of the
    division by x + j is longer than n!/j!. Each division is charged before it runs by that length and n - j alone,
    so a forged row never costs the divisions more than the true row of the same n; only the pass over the entries
    and their sum are charged by the row's own longest entry.

Work. The checker runs in the active computation budget (the verifier opens a fresh one) or in one of the default
limits, and charges every step before it runs, on the scale of the units of trusted.integers, where a unit takes at
most about 0.3 microseconds: a pass over an integer of b bits (an addition, a product by a short integer, a division
by one) costs a unit and one more per 2,048 bits; a loop that charges every step pays one unit more per step for the
charge itself, and a division of stirling1, charged as a whole, a pass of its longest integer per step; a long
division costs division_work; a product of two long integers costs a unit and one more per 64 pairs of their 64-bit
words, four times trusted.integers.product_work, since products of 1,000 to 30,000 bits took 0.3 to 0.8
microseconds per unit of product_work (measured with CPython 3.15 on one machine; 0.06 to 0.19 per unit of this
charge). Every such product passes Budget.product, so that none exceeds max_integer_bits; the other steps form
integers that the claim bounds: the checker's own n!, the entries and the value of the claim, and in the divisions
of stirling1 integers no longer than n!. The charges depend on the claim alone. Under the default max_work of
2,000,000 the true claims at the card's caps verify with partitions(5000) at about 443,000 units (0.013 s on that
machine, the best of three runs; about n = 14,000 is the largest that fits) and derangements(5900) at about 273,000
(0.031 s). The card sets no cap for bell and stirling1: the largest n that fits is 1,854 for bell (n + 1 terms of up
to about n * log2(n) bits each; bell(1500) takes about 1,017,000 units, 0.10 s) and 1,042 for stirling1 (n(n + 1)/2
steps of the divisions; stirling1(1000, k) takes about 1,754,000 units, 0.11 s, whatever k is). The producer refuses
larger n with ResourceLimitError, so that producer and checker fail at the same point. Up to these caps a forged
claim of partitions, bell or stirling1 costs at most about what the true claim of the same arguments costs, and one
of derangements(5900) at most about 371,000 units (a value of the largest size that the size test admits, exact at
every step), so the checker refuses every forged claim within the default budget, with False.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer. The family defines no record class (every value is an int), so the checker names none and
tests each value's exact type instead.
"""

from __future__ import annotations

from collections.abc import Callable

from mathforge.limits import Budget, computation
from mathforge.trusted.integers import division_work

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "partitions": ("n",),
    "derangements": ("n",),
    "bell": ("n",),
    "stirling1": ("n", "k"),
}
_PASS_BITS = 11  # A pass over an integer costs a unit and one more per 2**11 bits of it.
_STEP = 1  # What a loop step that charges itself pays for the charge, beyond the work of the step.


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for arguments other than the operation's exact ints >= 0 and for any malformed
    value or evidence; ResourceLimitError past the budget."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = _ARGUMENTS.get(operation)
    if names is None or len(args) != len(names) or any(type(args.get(name)) is not int for name in names):
        return False
    numbers = [args[name] for name in names]
    if any(number < 0 for number in numbers) or type(value) is not int or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        return _CHECKS[operation](budget, *numbers, value, evidence)


def _pass(bits: int) -> int:
    """The units of one pass over an integer of the given bit length: an addition, a product by a short integer or a
    division by one."""
    return 1 + (bits >> _PASS_BITS)


def _times(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs: Budget.product's unit, which refuses with ResourceLimitError a product
    that could exceed max_integer_bits, then _STEP and one unit per 64 pairs of the 64-bit words of the factors."""
    budget.product(left, right)
    budget.tick(_STEP + ((left.bit_length() >> 6) * (right.bit_length() >> 6) >> 6))
    return left * right


def _power(budget: Budget, base: int, exponent: int) -> int:
    """base**exponent for exponent >= 0, by left-to-right square-and-multiply, every product charged by _times."""
    if not exponent:
        return 1
    result = base
    for bit in bin(exponent)[3:]:  # The bits after the leading 1, from the top.
        result = _times(budget, result, result)
        if bit == "1":
            result = _times(budget, result, base)
    return result


def _factorial(budget: Budget, n: int, ceiling: int | None = None) -> int | None:
    """n!, the product 2 * 3 * ... * n, each step charged before it runs (ResourceLimitError past max_integer_bits).
    With a ceiling, None as soon as a partial product exceeds it: then n! does too, and it is never formed."""
    value = 1
    for factor in range(2, n + 1):
        budget.product(value, factor)
        budget.tick(_pass(value.bit_length()))
        value *= factor
        if ceiling is not None and value > ceiling:
            return None
    return value


def _pentagonal(budget: Budget, n: int) -> tuple[list[int], list[int]]:
    """The generalized pentagonal numbers 1, 2, 5, 7, 12, 15, ... up to n, increasing, and their signs 1, 1, -1, -1,
    ...: from 0 the gaps are in turn the odd numbers 1, 3, 5, ... and the natural numbers 1, 2, 3, ..., which reach
    k(3k - 1)/2 and k(3k + 1)/2 for k = 1, 2, ..., both with the sign (-1)**(k + 1)."""
    offsets: list[int] = []
    signs: list[int] = []
    number, odd, natural, sign = 0, 1, 1, 1
    while True:
        budget.tick()
        number += odd  # k(3k - 1)/2
        odd += 2
        if number > n:
            return offsets, signs
        offsets.append(number)
        signs.append(sign)
        number += natural  # k(3k + 1)/2
        natural += 1
        if number > n:
            return offsets, signs
        offsets.append(number)
        signs.append(sign)
        sign = -sign


def _partitions(budget: Budget, n: int, value: int, evidence: tuple) -> bool:
    if len(evidence) != 1:
        return False
    table = evidence[0]
    if type(table) is not tuple or len(table) != n + 1:
        return False
    first, last = table[0], table[n]
    if type(first) is not int or first != 1 or type(last) is not int or last != value:
        return False
    offsets, signs = _pentagonal(budget, n)
    count = 0  # How many generalized pentagonal numbers are at most m.
    for m in range(1, n + 1):
        if count < len(offsets) and offsets[count] == m:
            count += 1
        # The terms of the recurrence, all verified entries no longer than table[m - 1] (p is nondecreasing).
        budget.tick(1 + count * _pass(table[m - 1].bit_length()))
        total = 0
        for index in range(count):
            if signs[index] > 0:
                total += table[m - offsets[index]]
            else:
                total -= table[m - offsets[index]]
        entry = table[m]
        if type(entry) is not int or entry != total:  # Verified in increasing order: never added before this.
            return False
    return _divisor_identity(budget, n, table)


def _divisor_identity(budget: Budget, n: int, table: tuple) -> bool:
    """n * table[n] == sum of sigma(k) * table[n - k] over k = 1, ..., n, for a table that the pentagonal recurrence
    has verified (no entry exceeds table[n]); sigma(k), the sum of the divisors of k, from a sieve."""
    budget.tick(1 + n)
    sigma = [0] * (n + 1)
    for divisor in range(1, n + 1):
        budget.tick(1 + n // divisor)
        for multiple in range(divisor, n + 1, divisor):
            sigma[multiple] += divisor
    # A short product and an addition per k: sigma(k) <= k * k <= n * n, every entry is at most table[n], and the
    # sum of n such products is at most table[n] * n**3.
    budget.tick(1 + n * _pass(table[n].bit_length() + 3 * n.bit_length()))
    total = 0
    for k in range(1, n + 1):
        total += sigma[k] * table[n - k]
    return total == n * table[n]


def _derangements(budget: Budget, n: int, value: int, evidence: tuple) -> bool:
    if evidence or value < 0 or value.bit_length() > max(1, n * n.bit_length()):  # D(n) <= n! <= n**n.
        return False
    current = value
    for m in range(n, 0, -1):
        # The step x_m -/+ 1, then the division by m: two passes and a long division.
        bits = current.bit_length()
        budget.tick(_STEP + 2 * _pass(bits) + division_work(bits, m.bit_length()))
        current, remainder = divmod(current - 1 if m % 2 == 0 else current + 1, m)  # (x_m - (-1)**m) / m.
        if remainder:
            return False
    return current == 1


def _bell(budget: Budget, n: int, value: int, evidence: tuple) -> bool:
    # 2**(n - 1) <= B(n) <= n! <= n**n for n >= 1, and B(0) = 1: a bit length from max(1, n) to
    # max(1, n * bit_length(n)), tested before any product, so that a forged giant value costs nothing (also for
    # n = 0, where the product n! v below would otherwise meet it).
    if evidence or value < 1 or not max(1, n) <= value.bit_length() <= max(1, n * n.bit_length()):
        return False
    factorial = _factorial(budget, n)
    target = _times(budget, value, factorial)  # n! v, which the sum must equal.
    total, binomial, derangement = 0, 1, 1  # binomial = C(n, i) and derangement = D(i), for i = 0 first.
    for i in range(n + 1):
        j = n - i  # The term C(n, j) j**n D(n - j), with C(n, j) = C(n, i) and D(n - j) = D(i).
        if i:
            # D(i) = i D(i - 1) + (-1)**i, and C(n, i) = C(n, i - 1) (j + 1) / i: two passes over each.
            budget.tick(_STEP + 2 * _pass(derangement.bit_length() + i.bit_length())
                        + 2 * _pass(binomial.bit_length() + j.bit_length()))
            derangement = i * derangement + (1 if i % 2 == 0 else -1)
            binomial = binomial * (j + 1) // i
        if not derangement or (not j and n):  # D(1) = 0, and 0**n = 0 for n >= 1.
            continue
        term = _times(budget, _times(budget, binomial, derangement), _power(budget, j, n))
        budget.tick(_STEP + _pass(max(term.bit_length(), total.bit_length())))
        total += term
    return total == target


def _stirling1(budget: Budget, n: int, k: int, value: int, evidence: tuple) -> bool:
    if len(evidence) != 1:
        return False
    row = evidence[0]
    if type(row) is not tuple or len(row) != n + 1:
        return False
    bound = max(1, n * n.bit_length())  # c(n, i) <= n! <= n**n < 2**(n * bit_length(n)).
    budget.tick(1 + n)
    longest = 0  # The bit length of the longest entry, at most bound.
    for entry in row:
        if type(entry) is not int or entry < 0:
            return False
        bits = entry.bit_length()
        if bits > bound:
            return False
        if bits > longest:
            longest = bits
    # The top entry c(n, n) = 1 first: a necessary condition that costs nothing.
    if value != (row[k] if k <= n else 0) or row[n] != 1:
        return False
    # The point 1: P(1) = r_0 + ... + r_n = n!. The sum of the n + 1 entries stays below 2**(longest + bit_length(n)
    # + 1), charged by the row as it is (a long row of short entries for a giant n costs little); the product
    # 2 * 3 * ... stops as soon as it passes the sum. Once P(1) = n! holds, no entry exceeds n!, as all are >= 0.
    budget.tick(1 + (n + 1) * _pass(longest + n.bit_length() + 1))
    limit = sum(row)
    if _factorial(budget, n, limit) != limit:
        return False
    # The points 0, -1, ..., -(n - 1): divide by x + j for j = 0, 1, ... in turn. Before the division by x + j,
    # poly[j:] holds the current quotient c_0, ..., c_(n - j), lowest first, and limit = n!/j!. Synthetic division
    # from the top, b_(t - 1) = c_t - j b_t, writes the coefficient b_(t - 1) of the next quotient where c_t was, so
    # that it lies in poly[j + 1:]; the remainder c_0 - j b_0 is the value of the current quotient at -j. Every b
    # must lie in [0, n!/(j + 1)!], as the coefficients of R's quotient (x + j + 1)...(x + n - 1) do: they are >= 0
    # and sum to its value at 1. Then c_t <= n!/j! and 0 <= j b_t < n!/j!, so no integer of the pass is longer than
    # n!/j!, and the charge of the pass, which depends on n and j alone, is paid before it runs: the divisions of a
    # forged row never cost more than those of the true row of the same n.
    poly = list(row)
    budget.tick(1 + n)
    for j in range(n):
        bits = limit.bit_length()
        # The division of limit by j + 1, then n - j steps of a product by the short j, a subtraction and two
        # comparisons: a pass each, at most about 0.1 microseconds per unit measured on true and forged rows.
        budget.tick((n - j + 1) * _pass(bits))
        limit //= j + 1
        carry = poly[n]  # b_(n - j - 1) = c_(n - j) = r_n = 1: the top coefficient of every quotient.
        for index in range(n - 1, j, -1):
            carry = poly[index] - j * carry
            if carry < 0 or carry > limit:
                return False
            poly[index] = carry
        if poly[j] != j * carry:  # The remainder c_0 - j b_0, the value of the current quotient at -j.
            return False
    return True


_CHECKS: dict[str, Callable[..., bool]] = {
    "partitions": _partitions,
    "derangements": _derangements,
    "bell": _bell,
    "stirling1": _stirling1,
}
