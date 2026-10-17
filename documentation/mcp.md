# Connect a local MCP client

The adapter serves its tools over **stdio only**. It does not open a port,
require an API key, retain a server Workspace, or offer file-management tools.

The tool list has three parts, always in this order:

1. The thirty **curated tools** of v0.7. Their names, order and input schemas are unchanged.
2. The **area tools**. Each serves a group of mathematical areas, and the list contains only the tools whose
   areas hold at least one operation. Six exist today: `mathforge.algebra`, `mathforge.analysis`,
   `mathforge.discrete`, `mathforge.linalg`, `mathforge.ntheory` and `mathforge.stats`.
3. `mathforge.catalog`, which finds operations by name, summary or area.

That makes 37 tools today. The list grows as areas gain operations; the protocol's tool listing and
`mathforge.capabilities` always show the current one. `api_version` stays 1.

```python
import mathforge as mf

names = [tool["name"] for tool in mf.capabilities()["tools"]]
assert names[0] == "mathforge.capabilities" and names[29] == "mathforge.compute"
assert names[30:-1] == ["mathforge.algebra", "mathforge.analysis", "mathforge.discrete", "mathforge.linalg",
                        "mathforge.ntheory", "mathforge.stats"]
assert names[-1] == "mathforge.catalog" and len(names) == 37
```

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

The thirty curated tools of v0.7 remain as they were. Their inputs are:

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

`mathforge.compute` still accepts only these v0.7 operations; its schema is frozen
with the rest of the curated tools. Operations added later run through the area
tools below, and so do the v0.7 operations, with typed arguments instead of
envelopes. An area tool runs a v0.7 operation exactly as `mathforge.compute` does
and returns a version 5 result.

## Run an operation through an area tool

An area tool runs one operation of its areas, named in the request. The areas are those of the
[area modules](areas.md). The six tools serve:

| Tool | Areas |
| --- | --- |
| `mathforge.algebra` | `expr`, `algebra`, `sets` |
| `mathforge.analysis` | `analysis`, `series`, `integrals` |
| `mathforge.discrete` | `discrete` |
| `mathforge.linalg` | `linalg` |
| `mathforge.ntheory` | `ntheory`, `crypto` |
| `mathforge.stats` | `stats` |

The input is an object:

```json
{
  "operation": "continued_fraction",
  "arguments": {"value": "415/93"},
  "symbol_ids": {},
  "limits": {}
}
```

- `operation` is required: one of the tool's operations. The tool's input schema enumerates them, and
  `mathforge.catalog` lists them with their arguments.
- `arguments` is required: the operation's named arguments, exactly those that the catalog lists. An argument
  with a default may be omitted, and the default applies, as in the Python API.
- `symbol_ids` is optional: variable names mapped to UUID strings, to keep symbol identities across calls. A name
  that the arguments use resolves through this table, and any other name gets a fresh identity. The v0.7 tools
  accept only the names a request declares; an area tool also accepts entries that a request does not use, so a
  client can keep one table for a whole session. The response's `symbols` lists the names that the request used.
- `limits` is optional and lowers the computation limits, as for every other tool.

```python
import mathforge as mf
from mathforge.mcp.boundary import call_tool

reply = call_tool("mathforge.ntheory", {
    "operation": "continued_fraction",
    "arguments": {"value": "415/93"},
})
assert not reply.is_error and reply.payload["api_version"] == 1
result = mf.from_data(reply.payload["result"])
assert result.value.terms == (4, 2, 6, 7)
assert "[4; 2, 6, 7]" in reply.payload["text"]
assert reply.payload["result"]["version"] == 6
assert mf.verify(result).verified
```

The response is the usual wrapper described under [Understand the result](#understand-the-result). Its `result`
is the unchanged `to_data(Result)` envelope: format version 6 for an operation added in v0.8, such as
`continued_fraction`, and version 5 for a v0.7 operation.

### Write the arguments

Each argument has a **kind**, which the catalog reports. An argument takes the friendly JSON form of its kind
below, or a canonical object envelope.

| Kind | Friendly JSON form | Example |
| --- | --- | --- |
| `int` | An integer, or a string of decimal digits with an optional sign | `12`, `"-7"` |
| `rational` | An integer, or a string `"p"` or `"p/q"` | `"-3/4"` |
| `scalar` | An integer, or closed exact mathematical text | `"sqrt(2)/3"` |
| `point` | A `scalar`, or the text `"-inf"` or `"+inf"` | `"+inf"` |
| `side` | `"both"`, `"left"` or `"right"` | `"left"` |
| `int_list` | An array of `int` | `[2, 3, "5"]` |
| `rational_list` | An array of `rational` | `[1, "3/2", 2]` |
| `int_pairs` | An array of pairs of `int` | `[[2, 3], [3, 5]]` |
| `complex` | `[real, imaginary]`: two `scalar` | `[1, "sqrt(2)"]` |
| `matrix` | An array of rows, each an array of `rational` | `[[1, 2], [3, 4]]` |
| `distribution` | `{"outcomes": [...], "probabilities": [...]}`: two arrays of `rational` | `{"outcomes": [0, 1], "probabilities": ["1/2", "1/2"]}` |
| `function` | `{"text": ..., "variables": [...]}`: a real elementary function and its ordered variables | `{"text": "log(x)+sin(x)", "variables": ["x"]}` |
| `function_list` | An array of `function` | `[{"text": "x*y", "variables": ["x", "y"]}]` |
| `rational_function` | `{"text": ..., "variable": ...}` | `{"text": "1/(x^2-1)", "variable": "x"}` |
| `symbol` | A variable name | `"x"` |
| `symbols` | An array of variable names | `["x", "y"]` |
| `bindings` | An object from variable names to `scalar` | `{"x": "sqrt(2)"}` |
| `object` | None: a canonical envelope only | |

JSON decimal numbers and `true`/`false` are never exact numbers: send `"1/2"`, not `0.5`. A string of digits has at
most 1024 characters, and the request's `limits`, such as `max_integer_bits`, apply to every number. `scalar` text
must be closed, without free variables, and follows the [shared text grammar](parsing.md), as does the text of a
`rational_function`; `function` text follows the grammar of `mathforge.parse_function`.

A **canonical object envelope** is exactly what `mf.to_data` returns for a value, in format version 1 through 6.
Use it to pass a value from an earlier result, such as a real algebraic root, that has no friendly form. It may
stand for a whole argument or for an element of a list, a pair, a complex number, a matrix or `bindings`. The
server applies the checks of `mathforge.compute` to each envelope: it has exactly the members `format`,
`version`, `kind` and `data`; `kind` is `object`; `data` contains operand types only, so a Result, a Workspace,
execution metadata and the v6 `Unevaluated` and `Reason` records are not operands; and the envelope is in its
canonical minimum format.

```python
def run(tool, operation, **arguments):
    reply = call_tool(tool, {"operation": operation, "arguments": arguments})
    assert not reply.is_error, reply.payload["error"]
    return mf.from_data(reply.payload["result"]).value

assert run("mathforge.discrete", "binomial", n=10, k="3") == 120
assert run("mathforge.stats", "mean", data=[1, "3/2", 2]) == mf.Rational(3, 2)
assert run("mathforge.ntheory", "crt", congruences=[[2, 3], [3, 5]]).modulus == 15
assert run("mathforge.linalg", "trace", A=[[1, 2], [3, 4]]) == mf.Rational(5)
assert run("mathforge.discrete", "binomial", n=mf.to_data(10), k=3) == 120

root_reply = call_tool("mathforge.real_roots", {"expression": "x^3-2"})
cube_root_of_two = mf.from_data(root_reply.payload["result"]).unwrap()[0].root
cube = run("mathforge.algebra", "complex_pow", base=[mf.to_data(cube_root_of_two), 0], n=3)
assert cube == mf.ComplexAlgebraic(2, 0)

x_id = "00000000-0000-4000-8000-000000000001"
reply = call_tool("mathforge.analysis", {
    "operation": "function_differentiate",
    "arguments": {"expression": {"text": "log(x)+sin(x)", "variables": ["x"]}, "variable": "x"},
    "symbol_ids": {"x": x_id, "y": "00000000-0000-4000-8000-000000000002"},  # y is unused
})
assert not reply.is_error and reply.payload["symbols"] == [{"name": "x", "uid": x_id}]
```

A call that the server cannot decode is an error response with `result` null. Its `error` names the stage, a code
and the field that failed, such as `arguments.n` below.

```python
bad = call_tool("mathforge.discrete", {"operation": "binomial", "arguments": {"n": 10.0, "k": 3}})
assert bad.is_error and bad.payload["result"] is None
assert bad.payload["error"]["stage"] == "decoding" and bad.payload["error"]["field"] == "arguments.n"
```

### Treat the schemas as hints

An area tool's input schema documents the shape of a call. `operation` lists the tool's operations. The tool's
`$defs` define each operation's arguments object, named like the operation, each argument kind that the tool
uses, and the canonical `envelope`; `arguments` refers to those definitions. The schema stays within a portable
subset of JSON Schema: a plain object at the root, local `$ref` references, and no `if`/`then`/`else` or
combinator at the root.

The schemas are advisory. Neither the SDK nor this server rejects a call for violating `inputSchema`; a client
may. A schema cannot tie `arguments` to the chosen `operation` either: `arguments` may match any operation of
the tool. The server therefore decodes every call itself, against the named operation, and its decoding is the
authority. A call that satisfies the schema can still fail, for example with the arguments of another operation
or with numbers beyond the limits, and a client must not rely on the schema to reject a wrong call.

## Find operations with the catalog

`mathforge.catalog` lists operations, the tool that runs each and the arguments it takes. All its fields are
optional:

- `query`: text of at most 128 characters, matched case-insensitively as a substring of an operation's name,
  summary or area.
- `area`: one of the 18 areas of the [area modules](areas.md).
- `limit`: the largest number of entries to return, from 1 through 200 (default 50). It is not the `limits` of
  the computation tools, which the catalog does not take.

The reply is `{"api_version": 1, "operations": [...], "total": ..., "error": null}`. `operations` is sorted by
area name and then by operation name, and `total` counts every match, also those beyond `limit`. There is no
paging: when `total` exceeds the entries returned, narrow the search with `query` or `area`, or raise `limit`.
Each entry looks like this:

```json
{
  "name": "continued_fraction",
  "area": "ntheory",
  "tool": "mathforge.ntheory",
  "summary": "Return the finite simple continued fraction of an exact rational.",
  "arguments": [{"name": "value", "kind": "rational", "required": true, "default": null}],
  "returns": "ContinuedFraction"
}
```

`tool` names the tool that runs the operation: the area tool for every operation of the registry, including those
that `mathforge.compute` also accepts. An argument is `required` unless it has a default, which `default` gives
as JSON. `returns` is the return type as the Python API names it. Older operations that the registry does not
hold, such as `solve`, `expand` and `rank`, are listed too, without `arguments` and with an empty `returns`; their
`tool` is the curated tool that runs them, or an empty string when none does. A failure returns the same shape
with an empty `operations` list, a `total` of 0 and an `error` object, and sets the MCP error flag.

```python
reply = call_tool("mathforge.catalog", {"query": "continued"})
assert not reply.is_error and reply.payload["error"] is None
entry = next(entry for entry in reply.payload["operations"] if entry["name"] == "continued_fraction")
assert (entry["area"], entry["tool"]) == ("ntheory", "mathforge.ntheory")
assert entry["arguments"] == [{"name": "value", "kind": "rational", "required": True, "default": None}]

reply = call_tool("mathforge.catalog", {"area": "discrete", "limit": 3})
entries = reply.payload["operations"]
assert len(entries) == 3 and reply.payload["total"] > 3
assert {entry["area"] for entry in entries} == {"discrete"}
assert [entry["name"] for entry in entries] == sorted(entry["name"] for entry in entries)

reply = call_tool("mathforge.catalog", {"query": "rank"})
entries = reply.payload["operations"]
assert entries == sorted(entries, key=lambda entry: (entry["area"], entry["name"]))
older = next(entry for entry in entries if entry["name"] == "rank")
assert older["arguments"] == [] and older["returns"] == ""
```

## Understand the result

Computation tools return a structured wrapper containing:

- `api_version`: adapter response contract version, independent of mathematical JSON.
- `result`: the unchanged `to_data(Result)` envelope, `to_data(RealFunction)` for `parse_function`, or `to_data(VerificationReport)` for `verify`.
- `text`, `latex`: exact human-readable presentation, not replacement evidence.
- `symbols`: explicit name/UUID records for reuse through `symbol_ids`.
- `error`: stage, code, message and available field/source-span information, or null.

`capabilities` instead returns its own descriptive metadata object, which lists every
tool, the area tools and the catalog included, and each operation with its `mcp_tool`
(null when no tool runs it). `catalog` returns its own listing, described above. For the
operations of v0.7, `capabilities` names `mathforge.compute`, as its frozen v0.7 entries do,
while the catalog names the typed area tool; both tools run them. Tools declare
input/output schemas. Structured results are also supplied as JSON text content
for compatible clients. No expression or certificate is silently truncated.

Symbol identities are request-local. Supply the returned UUIDs on later calls
when identity must be preserved; fresh calls need not have equal content hashes.
For the v0.7 tools, only declared variables may appear in `symbol_ids` or input
text. An area tool also accepts `symbol_ids` entries that its request does not use.

`NO_SOLUTION` is a successful mathematical outcome. A finite/infinite limit or
`does_not_exist` is likewise a completed value, without the MCP error flag.
An unsupported elementary integral or limit may instead complete with
`outcome=NO_CONCLUSION`, `value=null` and an explicit `deferred` request. This is
not an MCP execution error and makes no claim of divergence or nonexistence.
The deferred source, bounds or limit direction remain linked to the request.
An operation added in v0.8 that reaches no verified answer reports a generic
`Unevaluated` record in the same field, with its kind, the request and a reason
code, under the same `NO_CONCLUSION` outcome.
Unsupported operations,
invalid input and computation failures set the MCP error flag while retaining
any core `Result`. A successfully obtained `FAILED` or `UNKNOWN` verification
report is not a tool execution error. A presentation failure retains the canonical
result and reports its presentation stage separately.

`verify` accepts only a complete v1/v2/v3/v4/v5/v6 MathForge **Result envelope**, not a string,
Workspace or arbitrary mathematical object. It rechecks mathematical claims;
stored verification labels are not trusted.

## Migration note for MCP clients

A v0.7 client keeps working. The changes are additive, and `api_version` stays 1.

- The tool listing and `mathforge.capabilities` show the area tools and the catalog after the thirty curated
  tools, which keep their names, order, input schemas and results. A client that expects exactly thirty tools
  must accept more.
- The result of an operation added in v0.8 is a JSON **v6** envelope (`"version": 6`); `continued_fraction` is the
  first. Results of operations that existed in v0.7 keep the format version they had (1 through 5), also when an
  area tool runs them, so results of format versions 1 through 5 are unchanged. MathForge 0.7 does not read v6
  documents.
- A v6 request names its format version as well, `"version": 6` beside `operation` and `arguments`, so a build
  that does not know the operation still keeps the request's format; requests of the v0.7 operations never do.
- `mathforge.verify` accepts Result envelopes of versions 1 through 6, although its tool description, frozen with
  the rest of the v0.7 surface, still says v1–v5. The `input` and `output` texts of `mathforge.capabilities` are frozen
  too, and name envelope versions 1–5.
- A v6 result whose operation this build does not know, for example one made by a later build, does not verify:
  the report is not verified and holds a single `UNKNOWN` check. The reply is an ordinary response, never an error.

```python
from mathforge.deferred import Reason, Unevaluated

request = mf.OperationRequest("operation_of_a_later_build", (("value", 1),), version=6)
later = mf.Result(execution_status=mf.ExecutionStatus.COMPLETED, outcome=mf.Outcome.NO_CONCLUSION,
                  completeness=mf.Completeness.UNKNOWN, method=request.operation, request=request,
                  deferred=Unevaluated("evaluate", request, Reason("no_applicable_rule")))
document = mf.to_data(later)
assert document["version"] == 6 and document["data"]["deferred"]["type"] == "v6_unevaluated"
assert document["data"]["request"]["version"] == 6
reply = call_tool("mathforge.verify", {"result": document})
assert not reply.is_error
report = mf.from_data(reply.payload["result"])
assert not report.verified
assert [check.status for check in report.checks] == [mf.CheckStatus.UNKNOWN]
```

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
