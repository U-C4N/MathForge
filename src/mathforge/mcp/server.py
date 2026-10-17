"""Official MCP SDK v2 transport glue; all imports are lazy and stdio-only."""

import json
import sys
from importlib import metadata


def _sdk():
    try:
        version = metadata.version("mcp")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError('The MCP adapter requires the optional extra: pip install "mathforge[mcp]"') from exc
    parts = version.split(".")
    if len(parts) < 2 or parts[0] != "2" or not parts[1].isdigit() or int(parts[1]) < 3:
        raise RuntimeError('The MCP adapter requires mcp>=2.3,<3; install "mathforge[mcp]"')
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.shared.exceptions import MCPError
    return types, Server, MCPError


def create_server():
    """Build the curated tool server. No network transport is exposed by this API."""
    types, Server, MCPError = _sdk()
    from .. import __version__
    from ..capabilities import CATALOG_TOOL, capabilities
    from .boundary import call_tool
    from .schemas import input_schemas, output_schema, capabilities_schema, catalog_schema
    import anyio

    schemas = input_schemas()
    descriptors = capabilities()["tools"]
    outputs = {"mathforge.capabilities": capabilities_schema(), CATALOG_TOOL: catalog_schema()}
    # Serialize CPU-bound calls per server; the worker keeps the protocol loop responsive.
    limiter = anyio.CapacityLimiter(1)

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(
            name=item["name"], description=item["description"], inputSchema=schemas[item["name"]],
            outputSchema=outputs.get(item["name"]) or output_schema(),
            annotations=types.ToolAnnotations(readOnlyHint=True, openWorldHint=False),
        ) for item in descriptors])

    async def execute(ctx, params):
        if params.name not in schemas:
            raise MCPError(-32602, "Unknown MathForge tool")
        response = await anyio.to_thread.run_sync(call_tool, params.name, params.arguments or {}, limiter=limiter)
        encoded = json.dumps(response.payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        return types.CallToolResult(content=[types.TextContent(type="text", text=encoded)],
                                    structuredContent=response.payload, isError=response.is_error)

    return Server("MathForge", version=__version__,
                  instructions="Exact, bounded mathematics. Find an operation with mathforge.catalog, then call the area tool it names (for example mathforge.ntheory) with the operation name and its typed arguments: friendly JSON such as integers, 'p/q' fractions, mathematical text and variable names, or canonical object envelopes from earlier results. Pass symbol_ids to keep symbol identities across calls. Use parse_function for domain-aware real functions; compute accepts only the published v0.7 operations, with canonical object operands. A deferred result makes no mathematical conclusion. Text/LaTeX presentation never replaces canonical results. No filesystem tools.",
                  get_tool_input_schema=schemas.get, on_list_tools=list_tools, on_call_tool=execute)


def main() -> None:
    """Run only stdio. Diagnostic messages never enter the protocol stdout."""
    if len(sys.argv) != 1:
        print("mathforge-mcp accepts no arguments and serves local stdio only.", file=sys.stderr)
        raise SystemExit(2)
    try:
        server = create_server()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    import anyio
    from mcp.server.stdio import stdio_server

    async def run():
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    anyio.run(run)
