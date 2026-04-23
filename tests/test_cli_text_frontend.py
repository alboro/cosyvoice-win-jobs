from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosyvoice_win.cli import maybe_disable_text_frontend_imports


class TestTextFrontendImports(unittest.TestCase):
    def test_disabled_frontend_blocks_wetext_probe(self):
        with maybe_disable_text_frontend_imports(False):
            with self.assertRaisesRegex(ImportError, "wetext disabled"):
                __import__("wetext")

    def test_disabled_frontend_does_not_block_normal_imports(self):
        with maybe_disable_text_frontend_imports(False):
            module = __import__("json")

        self.assertEqual(module.__name__, "json")


if __name__ == "__main__":
    unittest.main()
