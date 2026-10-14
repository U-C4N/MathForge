"""REL-11: Execute shipped README/public Python fences with real assertions.

Fences share one namespace per document, as sequential tutorial snippets do.
Examples run in a temporary working directory so save demonstrations do not
overwrite a checkout file. No snippets are rewritten or assertions removed.
Pass explicit paths to also check local documentation that is not distributed.
Default discovery ignores public prose-only pages, never internal docs/.
Use --mcp in an extras environment to include documentation/mcp.md.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
OPENING = re.compile(r"^\s*```(?:python|py)\s*$")
CLOSING = re.compile(r"^\s*```\s*$")


def python_fences(path: Path):
    block = None
    beginning = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if block is None:
            if OPENING.fullmatch(line):
                block, beginning = [], number + 1
        elif CLOSING.fullmatch(line):
            yield beginning, "\n".join(block) + "\n"
            block = None
        else:
            block.append(line)
    if block is not None:
        raise ValueError(f"Unclosed Python fence in {path}:{beginning}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path,
                        help="Markdown files to check (default: README and public executable pages)")
    parser.add_argument("--mcp", action="store_true", help="Include optional MCP documentation")
    arguments = parser.parse_args()
    if arguments.files:
        documents = [path.resolve() for path in arguments.files]
    else:
        documents = [REPOSITORY / "README.md"]
        for path in sorted((REPOSITORY / "documentation").rglob("*.md")):
            if path.name == "mcp.md" and not arguments.mcp:
                continue
            if list(python_fences(path)):
                documents.append(path)
    for document in documents:
        if not document.is_file():
            parser.error(f"Documentation file does not exist: {document}")
    count = 0
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="mathforge-docs-") as temporary:
        try:
            os.chdir(temporary)
            for document in documents:
                namespace = {"__name__": "__mathforge_document__", "__file__": str(document)}
                document_count = 0
                for beginning, source in python_fences(document):
                    # Prefixing blank lines keeps traceback numbers tied to Markdown.
                    code = compile("\n" * (beginning - 1) + source, str(document), "exec")
                    exec(code, namespace)
                    document_count += 1
                if document_count == 0:
                    raise AssertionError(f"No executable Python fences found in {document}")
                count += document_count
                print(f"{document.name}: {document_count} Python examples passed.")
        finally:
            os.chdir(previous)
    print(f"Executed {count} documented Python examples across {len(documents)} documents.")


if __name__ == "__main__":
    main()
