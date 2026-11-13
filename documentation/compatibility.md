# Compatibility and release checks

## What v0.9 preserves

v0.9 is additive. Every v0.8 operation keeps its name, signature, values and errors,
the thirty v0.7 MCP tools keep their names, order and input schemas, the v0.8 area
tools accept the new operations of their areas besides their old ones, three area
tools join them (40 tools in all), and v1–v6 documents keep their bytes and verdicts. The 65 new operations
live in their area modules and in `mathforge.operations` and write JSON v6. Two
behaviours of existing types are new: `IntervalSet` gains set operators (`|`, `&`,
`-`, `^`, `~`) and `Matrix ** k` raises a square matrix to a power; `Matrix * Matrix`
still raises `InvalidInput`, now naming `@` in its message. The check of power
series charges each coefficient operation by the length of its operands, so a
long `taylor` or `series_coefficients` check stops at its work limit sooner and
never runs long within it.

MCP responses gain additive keys, and `api_version` stays 1: each computation
tool's response names its result with a `handle` (`mf:sha256:` and the digest of
the result's canonical JSON), which an area tool argument or a `mathforge.compute`
operand may send in place of the envelope, and each error object gains `reason`
and `next_actions` beside its six keys, whose values do not change. A client that
rejects unknown keys must accept these. `ErrorInfo` and serialized Results do not
change. The input schemas of the thirty v0.7 tools do not change either, so the
frozen schema of `mathforge.compute` does not list handles; an operand string that
matches the handle pattern, an `invalid_input` before, now resolves as a handle or
fails with `unknown_handle`.

## What v0.8 preserves

The dependency-free Python 3.11–3.14 core and the v0.7 public API remain. Every
v0.7 operation keeps its name, signature and values, and its errors apart from
resource limits; the thirty v0.7 MCP tools keep their names, order and input
schemas, and `api_version` stays 1. Operations added since v0.8 live in their area
module and in `mathforge.operations`, not in the plain `mathforge` namespace.

Mathematical JSON v1–v5 documents keep their bytes and content hashes, also in
mixed workspaces. The results of the v0.7 operations keep their format version and
serialize to the same bytes. Only the operations added since v0.8 write JSON v6,
which MathForge 0.7 does not read. A v6 request records its format version, so a
later build's v6 requests keep their content hashes when this build loads them, and
a v6 result whose operation a build does not know verifies as `UNKNOWN`, never as
failed.

Stored v1–v5 documents get the same verdicts as in v0.7, except that some checks
that ran out of budget in v0.7, and so stayed `UNKNOWN`, now complete. Some checks
run new, faster checker code: those of determinants, inverses, real roots, real
comparisons, factor irreducibility, medians and rational primitives.

Exact arithmetic is cheaper in v0.8, so computations charge less work. Computation
limits remain ceilings, not cost promises: an input just past a limit may now
succeed where v0.7 reported a resource-limit error. A value function called without
`limits` may answer through the lean value path described under
[performance](performance.md); a call that path does not handle takes the structured
path of `mathforge.operations`. The value path never returns a value other than
the structured path's, and its errors are the same. Only near a resource limit may
it answer where the structured path refuses, because the new exact kernels need
less work.

## What v0.7 preserves

The dependency-free Python 3.11–3.14 core, legacy expression parser and real
`sqrt` semantics remain. The new `RealFunction` layer has explicit domains;
it does not add partial transcendental nodes to historical expression trees.
New rational-integral support completes previously unsupported requests.

Mathematical JSON v5 is additive. Old v1–v4 payload bytes, content hashes and
workspace references are frozen in compatibility tests. New types, requests,
proofs and deferred records require v5; old schemas remain unchanged. The v0.6
working-tree source manifest and 19 compatibility objects are recorded in
`tests/fixtures/v6/manifest.json`, including uncommitted source provenance.
`Result.deferred` is omitted when absent, preserving historical result bytes.
MCP retains `api_version=1` and adds two tools (thirty total).

An unevaluated result has `NO_CONCLUSION`, no value and unknown completeness.
It is serializable and renderable, but cannot pass verification as a solved
mathematical result. Failed proofs, exhausted budgets and invalid arguments
are not converted into successful or deferred calculations.

## What v0.6 preserves

The dependency-free Python 3.11+ core, rational matrices, expression constructors,
and existing root constructor/equality/hash behavior are unchanged. Use
`compare_real` for mathematical equality between different exact representations.
Explicit `algebraic` normalization returns a rational or a root of a primitive
irreducible polynomial; it does not change legacy `evaluate` or `substitute`.

New factorization, partial-fraction and extrema-value records and exact-algebra
certificates use mathematical JSON v4. An algebraic finite `LimitValue` also
requires v4. Old objects retain their minimum v1/v2/v3 encoding and content
hashes, including in mixed workspaces. The required document version only rises;
an older object encountered later cannot downgrade a v4 envelope. The packaged
v1/v2/v3 schemas remain unchanged. MCP keeps its outer `api_version=1` contract.

The v0.5 compatibility fixtures freeze the actual working tree, including sources
that were not yet committed, along with canonical payloads and a historical
reader. Source hashes and snapshot provenance are recorded in
`tests/fixtures/v5/manifest.json`.

## What v0.5 preserves

- Python 3.11+ and a dependency-free base install.
- Existing mathematical operations, `str()` methods and runtime `Result` identity.
- Mathematical JSON formats v1/v2, their integer-string representation and content hashes.
- Existing v1/v2 workspaces and explicit symbol UUIDs.
- Independent algorithmic verification and finite resource limits.

v0.5 multivariate values reuse existing expressions and tuples. Their new
operation names use the generic v2 request envelope and need a v0.5 verifier;
older implementations can decode these envelopes but report unknown operations.
The rational-only `Matrix` API is unchanged. Sturm/RREF optimizations reduce
charged work; computation limits remain ceilings, not historical cost promises.

New limit/function-analysis records and their operation requests use mathematical
JSON v3. Older values still use their minimum v1/v2 envelope, with the same bytes
and hashes. A mixed workspace uses v3 while each historical object retains its
original content reference. v1/v2 readers reject a v3 document explicitly.

The v2 schema remains unchanged; a separate packaged v3 schema adds allowlisted
record tags and request shapes. Decoding rejects unknown fields/types,
noncanonical data and documents exceeding configured resource limits.
Parser/renderer settings and MCP presentation metadata are outside mathematical
JSON. `Result[T]` retains its runtime identity. The optional MCP SDK and its
transitive dependencies are installed only when the `mcp` extra is requested.

## Recheck imported results

```python
import mathforge as mf

x = mf.symbol("x")
result = mf.solve(mf.Eq(x**2, 2), for_=x)
loaded = mf.loads(mf.dumps(result))
assert mf.verify(loaded).verified

workspace = mf.Workspace()
old_ref = workspace.put(result)
old_payload = mf.dumps(result)
analysis = mf.operations.analyze_function(x**2, x)
new_ref = workspace.put(analysis)
assert mf.to_data(workspace)["version"] == 3
restored = mf.loads(mf.dumps(workspace))
assert mf.dumps(restored.get(old_ref)) == old_payload
assert restored.put(restored.get(old_ref)) == old_ref
assert mf.verify(restored.get(new_ref)).verified
```

Decoding checks structure and consistency, not mathematical truth. A stored
verification label is not trusted evidence. Unsupported historical claims may
produce an `UNKNOWN` report instead of being silently certified.

## Development and release validation

```sh
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python tools/check_docs.py
python tools/generate_schema.py --check
python tools/check_schema.py
python tools/check_benchmarks.py
python tools/check_sympy.py
python -m mypy --follow-imports=silent --ignore-missing-imports --warn-unused-ignores tests/typing/public_api.py
python -m build
python tools/check_distribution.py
```

Distribution checks use isolated environments outside the source checkout, for
both wheel and sdist. Base tests do not require an MCP install; extras checks
must run real stdio tests and must not silently skip them. The optional SDK
version used by CI is pinned in `requirements-mcp.txt`.

The release-candidate tool reads the final version from package metadata and
accepts `--rc-number` (default `1`). It checks candidate/final normalized package
fingerprints and does not tag, push or publish. Historical fixture provenance
identifies the original release used to create each golden record.

Benchmark acceptance uses deterministic work/evidence counters and expected
outcomes; wall-clock timings are informative only. A scenario that previously
completed must not silently become an error. The intentional resource-limit
case must still report that limit explicitly.
