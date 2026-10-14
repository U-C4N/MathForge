"""Require the optional MCP SDK and run real adapter tests without skipped tests."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import unittest


def main() -> None:
    try:
        installed = version("mcp")
    except PackageNotFoundError as exc:
        raise SystemExit("MCP checks require: python -m pip install -r requirements-mcp.txt") from exc
    if installed.split(".", 1)[0] != "2":
        raise SystemExit(f"MCP checks require SDK 2.x, found {installed}")
    directory = Path(__file__).resolve().parents[1] / "tests" / "mcp"
    suite = unittest.defaultTestLoader.discover(str(directory), pattern="test_*.py")
    if suite.countTestCases() == 0:
        raise SystemExit("No MCP integration tests were discovered")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit("MCP integration checks failed or skipped tests")
    print(f"MCP SDK {installed}: {result.testsRun} integration tests passed, none skipped.")


if __name__ == "__main__":
    main()
