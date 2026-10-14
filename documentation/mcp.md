# Connect a local MCP client

The adapter exposes thirty tools over **stdio only**. It does not open a port,
require an API key, retain a server Workspace, or offer file-management tools.

## Install and start

Install in the Python environment your MCP client will launch:

```sh
python -m pip install -e ".[mcp]"
mathforge-mcp
```

The command waits for protocol messages on stdin; it is not a calculator prompt.
The equivalent module entrypoint is `python -m mathforge.mcp`. The base package
does not import the SDK. Starting the server without its extra gives an explicit
installation error on stderr and a nonzero exit status.

The adapter uses the official MCP Python SDK v2 (`mcp>=2.3,<3`); integration tests
pin `mcp==2.3.0`. Logs use stderr and stdout contains only protocol messages.

## Configure a client

For a client that uses an `mcpServers` configuration:

```json
{
  "mcpServers": {
    "mathforge": {
      "command": "mathforge-mcp",
      "args": []
    }
  }
}
```

If the client does not inherit your environment's PATH, set `command` to the
absolute path of that environment's `mathforge-mcp` executable. Alternatively,
set it to the absolute Python executable and use `args` equal to
`["-m", "mathforge.mcp"]`. Host-specific configuration locations vary.

## Call the curated tools

| Tool | Required arguments | Other arguments |
| --- | --- | --- |
| `mathforge.capabilities` | None | None |
| `mathforge.solve` | `problem` | `variable`, `symbol_ids`, `limits` |
| `mathforge.solve_system` | `equations`, `variables` | `symbol_ids`, `limits` |
| `mathforge.real_roots` | `expression` | `variable`, `symbol_ids`, `limits` |
| `mathforge.differentiate` | `expression` | `variable`, `order`, `symbol_ids`, `limits` |
| `mathforge.integrate` | `expression`, `lower`, `upper` | `variable`, `symbol_ids`, `limits` |
| `mathforge.limit` | `expression`, `point` | `variable`, `side`, `symbol_ids`, `limits` |
| `mathforge.analyze_function` | `expression` | `variable`, `symbol_ids`, `limits` |
| `mathforge.expand_multivariate` | `expression`, `variables` | `symbol_ids`, `limits` |
| `mathforge.partial_diff` | `expression`, `variables`, `variable` | `order`, `symbol_ids`, `limits` |
| `mathforge.gradient` | `expression`, `variables` | `symbol_ids`, `limits` |
| `mathforge.jacobian` | `expressions`, `variables` | `symbol_ids`, `limits` |
| `mathforge.hessian` | `expression`, `variables` | `symbol_ids`, `limits` |
| `mathforge.det` | `rows` | `limits` |
| `mathforge.rref` | `rows` | `ncols`, `limits` |
| `mathforge.verify` | `result` | `limits` |
| `mathforge.factor` | `expression` | `variable`, `symbol_ids`, `limits` |
| `mathforge.resultant` | `f`, `g` | `variable`, `symbol_ids`, `limits` |
| `mathforge.discriminant` | `expression` | `variable`, `symbol_ids`, `limits` |
| `mathforge.algebraic` | `expression` | `limits` |
| `mathforge.algebraic_div` | `left`, `right` | `limits` |
| `mathforge.algebraic_pow` | `base`, `n` | `limits` |
| `mathforge.minimal_polynomial` | `expression` | `variable`, `symbol_ids`, `limits` |
| `mathforge.evaluate_algebraic` | `expression`, `variables`, `bindings` | `symbol_ids`, `limits` |
| `mathforge.apart` | `expression` | `variable`, `symbol_ids`, `limits` |
| `mathforge.extrema_values` | `expression` | `variable`, `symbol_ids`, `limits` |
| `mathforge.charpoly` | `rows` | `variable`, `symbol_ids`, `limits` |
| `mathforge.real_eigenvalues` | `rows` | `limits` |
| `mathforge.parse_function` | `text`, `variables` | `symbol_ids`, `limits` |
| `mathforge.compute` | `operation`, `arguments` | `limits` |

The existing expression/problem tools use the [shared text grammar](parsing.md). The default
variable is `x`, the default derivative order is `1`, and orders range from
0 through 1024. Systems require an explicit ordered variable list and equations,
not inequalities. Matrix rows are arrays of arrays; entries are integer or
fraction **strings**, such as `"2"` and `"-3/4"`, never JSON
floats. `rref(rows=[], ncols=3)` preserves the 0-by-3 matrix shape.

Finite limit points and integral bounds accept closed exact text such as
`"-3/4"` or `"sqrt(2)"`, or canonical scalar envelopes described below. A limit
point also accepts exactly `"-inf"`/`"+inf"`. `side` defaults to `"both"`; finite points also
allow `"left"` or `"right"`. Infinity requires `"both"`. The dedicated
`mathforge.limit` and `mathforge.analyze_function` tools support real univariate
rational functions with rational coefficients. Elementary-function limits and
integrals use `mathforge.compute` with the v0.7 operation names below.

Multivariate tools require a nonempty ordered `variables` list; every symbol in
their polynomial text must be declared. `partial_diff.variable` must be in that
list. `jacobian.expressions` is a nonempty list of polynomial strings; rows follow
the functions and columns follow the variables. For example:

```json
{"expression": "x^2*y + y^3", "variables": ["x", "y"]}
```

These arguments work with `mathforge.gradient` and `mathforge.hessian`. See the
[multivariate guide](multivariate.md) for exact output shapes and scope.

Example `mathforge.limit` arguments for a removable hole:

```json
{
  "expression": "((x-1)*(x^2+2))/(x*(x-1))",
  "point": "1",
  "side": "both",
  "variable": "x"
}
```

Example `mathforge.solve` arguments:

```json
{
  "problem": "x^3 - 2 = 0",
  "variable": "x",
  "symbol_ids": {"x": "00000000-0000-4000-8000-000000000001"}
}
```

## Reuse an algebraic result

Scalar operands accept bounded closed expression text or a canonical mathematical
object envelope containing a `Rational`, `RealAlgebraicRoot` or supported scalar
expression. A complete Result or Workspace is not a scalar operand. The display
text `RootOf(...)` is deliberately not parser syntax: extract the value from the
returned result and send its canonical envelope to the next call.

```python
import mathforge as mf
from mathforge.mcp.boundary import call_tool

reply = call_tool("mathforge.real_roots", {"expression": "x^3-2"})
root_result = mf.from_data(reply.payload["result"])
alpha = root_result.unwrap()[0].root
powered = call_tool("mathforge.algebraic_pow", {"base": mf.to_data(alpha), "n": 3})
assert mf.from_data(powered.payload["result"]).unwrap() == mf.Rational(2)
assert powered.payload["api_version"] == 1
```

`algebraic_pow.n` is an integer from -1024 through 1024. For
`evaluate_algebraic`, `variables` is a nonempty declared variable list and
`bindings` must contain exactly those names. For example:

```json
{"expression": "x^2+y^2", "variables": ["x", "y"], "bindings": {"x": "sqrt(2)", "y": "sqrt(3)"}}
```

## Parse real functions and compute v0.7 operations

`mathforge.parse_function` requires an explicit ordered `variables` list and
returns a canonical `RealFunction`. Its grammar supports exact literals and the
documented elementary functions, including `sin`, `cos`, `exp` and `log`.
Domains are retained before simplification. An empty variable list represents a
closed function. Reuse the returned symbol UUIDs in subsequent operands.

`mathforge.compute.operation` is an enum of published v0.7 operations, available
through `mathforge.capabilities` and the tool's input schema. Every value in
`arguments` must be a canonical `to_data(...)` object envelope, including integers,
text and tuples. Include every required operation argument; Python convenience
defaults are not inserted by this tool. Result, Workspace and execution-metadata
objects are rejected as operands, even inside a tuple. All operands share one
symbol-identity decoding context.

```python
parsed = call_tool("mathforge.parse_function", {
    "text": "log(x)+sin(x)", "variables": ["x"],
    "symbol_ids": {"x": "00000000-0000-4000-8000-000000000007"},
})
function = mf.from_data(parsed.payload["result"])
axis, = function.variables
differentiated = call_tool("mathforge.compute", {
    "operation": "function_differentiate",
    "arguments": {"expression": mf.to_data(function),
                  "variable": mf.to_data(axis), "n": mf.to_data(1)},
})
result = mf.from_data(differentiated.payload["result"])
assert mf.verify(result).verified
assert differentiated.payload["result"]["version"] == 5
```

The fixed registry covers integer arithmetic, counting, finite probability,
statistics, complex algebraic values, linear algebra extensions and domain-aware
calculus. See the [v0.7 mathematics guide](v07-mathematics.md) for operation names
and supported mathematical scopes.

## Understand the result

Computation tools return a structured wrapper containing:

- `api_version`: adapter response contract version, independent of mathematical JSON.
- `result`: the unchanged `to_data(Result)` envelope, `to_data(RealFunction)` for `parse_function`, or `to_data(VerificationReport)` for `verify`.
- `text`, `latex`: exact human-readable presentation, not replacement evidence.
- `symbols`: explicit name/UUID records for reuse through `symbol_ids`.
- `error`: stage, code, message and available field/source-span information, or null.

`capabilities` instead returns its own descriptive metadata object. Tools declare
input/output schemas. Structured results are also supplied as JSON text content
for compatible clients. No expression or certificate is silently truncated.

Symbol identities are request-local. Supply the returned UUIDs on later calls
when identity must be preserved; fresh calls need not have equal content hashes.
Only declared variables may appear in `symbol_ids` or input text.

`NO_SOLUTION` is a successful mathematical outcome. A finite/infinite limit or
`does_not_exist` is likewise a completed value, without the MCP error flag.
An unsupported elementary integral or limit may instead complete with
`outcome=NO_CONCLUSION`, `value=null` and an explicit `deferred` request. This is
not an MCP execution error and makes no claim of divergence or nonexistence.
The deferred source, bounds or limit direction remain linked to the request.
Unsupported operations,
invalid input and computation failures set the MCP error flag while retaining
any core `Result`. A successfully obtained `FAILED` or `UNKNOWN` verification
report is not a tool execution error. A presentation failure retains the canonical
result and reports its presentation stage separately.

`verify` accepts only a complete v1/v2/v3/v4/v5 MathForge **Result envelope**, not a string,
Workspace or arbitrary mathematical object. It rechecks mathematical claims;
stored verification labels are not trusted.

## Resource boundaries

Optional `limits` fields may lower, but never exceed, `ComputationLimits`
defaults. The adapter also bounds request/response size, matrix entries, numbers,
equation/variable counts and parser/renderer work. Input failures before a core
result exists have `result=null`. Oversize results return `response_too_large`
instead of a truncated certificate. Larger computations remain available through
the Python API with explicitly supplied finite limits.

## Test the protocol

```sh
python -m pip install -r requirements-mcp.txt
python tools/check_mcp.py
```

The protocol gate uses a real subprocess, checks the advertised tools and actual
calls, and refuses a run where optional integration tests were silently skipped.
