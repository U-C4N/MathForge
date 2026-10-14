"""Exact multivariate derivatives, with source-linked checks and transport."""

import mathforge as mf

x, y = mf.symbol("x"), mf.symbol("y")
f = mf.parse_expression("x^2*y + y^3", symbols={"x": x, "y": y})
assert mf.partial_diff(f, x) == 2*x*y
assert mf.gradient(f, (x, y)) == (2*x*y, x**2+3*y**2)
assert mf.jacobian((f, x*y), (x, y)) == ((2*x*y, x**2+3*y**2), (y, x))
assert mf.hessian(f, (x, y)) == ((2*y, 2*x), (2*x, 6*y))

expanded = mf.expand_multivariate((x+y)**3, (x, y))
assert mf.evaluate(expanded, {x: 2, y: 3}) == mf.Rational(125)
result = mf.operations.hessian(f, (x, y))
restored = mf.loads(mf.dumps(result))
assert mf.verify(restored).verified
assert mf.to_data(restored)["version"] == 2
print("Gradient:", mf.render_text(mf.gradient(f, (x, y))))
print("Hessian:", mf.to_latex(restored.unwrap()))
