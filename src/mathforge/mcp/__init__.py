"""Optional local stdio adapter. Importing this module never imports the MCP SDK."""


def main() -> None:
    """Run the optional MCP server; install ``mathforge[mcp]`` first."""
    from .server import main as run
    run()


__all__ = ["main"]
