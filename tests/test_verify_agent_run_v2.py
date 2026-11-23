"""Runs the six synthetic-run cases of the official verdict script (verify_agent_run_v2)."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_v2_cases as cases  # noqa: E402


class VerifyAgentRunV2(unittest.TestCase):
    def test_synthetic_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            cases.HERE = Path(folder)
            cases.FIXTURES = Path(folder) / 'fixtures'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cases.main()
        self.assertEqual(code, 0, output.getvalue())
        self.assertIn('6/6 cases passed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
