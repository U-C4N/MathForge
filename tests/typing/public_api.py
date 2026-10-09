"""REL-08: A consumer checks concrete public return types through py.typed."""

from typing import assert_type

import mathforge as mf


A = mf.Matrix(((1, 2), (3, 4)))
x, y = mf.symbol("x"), mf.symbol("y")

assert_type(mf.rref(A), mf.RREFResult)
assert_type(mf.rank(A), int)
assert_type(mf.det(A), mf.Rational)
assert_type(mf.inverse(A), mf.Matrix)
assert_type(mf.nullspace(A), tuple[tuple[mf.Rational, ...], ...])
assert_type(mf.solve(mf.Eq(x**3 - 2, 0), for_=x), mf.Result)
assert_type(mf.solve_system(mf.LinearSystem(A, (5, 11), (x, y))), mf.Result)
assert_type(mf.LinearSystem(A, [5, 11], [x, y]).rhs, tuple[mf.Rational, ...])
assert_type(mf.AffineSolutionSet([x, y], [0, 0], [[1, 0], [0, 1]]).basis,
            tuple[tuple[mf.Rational, ...], ...])
assert_type(mf.integrate(x**2, x, 0, 1), mf.Rational)
assert_type(mf.verify(mf.solve(mf.Eq(x**2 - 2, 0), for_=x)), mf.VerificationReport)
