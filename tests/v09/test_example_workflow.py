"""examples/v09_workflow.py runs as written: the release example of v0.9 stays true."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "v09_workflow.py"


class ExampleWorkflowTests(unittest.TestCase):
    def test_the_v09_workflow_runs(self):
        spec = importlib.util.spec_from_file_location("v09_workflow", EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module.main()
        self.assertEqual(output.getvalue(), "MathForge v0.9 workflow verified\n")


if __name__ == "__main__":
    unittest.main()
