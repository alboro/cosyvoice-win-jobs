from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosyvoice_win.cli import (
    CosyVoiceModelOptions,
    build_model_kwargs,
    ensure_cosyvoice3_prompt_text,
    maybe_disable_text_frontend_imports,
)


class TestTextFrontendImports(unittest.TestCase):
    def test_disabled_frontend_blocks_wetext_probe(self):
        with maybe_disable_text_frontend_imports(False):
            with self.assertRaisesRegex(ImportError, "wetext disabled"):
                __import__("wetext")

    def test_disabled_frontend_does_not_block_normal_imports(self):
        with maybe_disable_text_frontend_imports(False):
            module = __import__("json")

        self.assertEqual(module.__name__, "json")

    def test_cosyvoice3_model_kwargs_do_not_include_load_jit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "cosyvoice3.yaml").write_text("", encoding="utf-8")

            kwargs = build_model_kwargs(CosyVoiceModelOptions(model_dir=model_dir, load_jit=True))

        self.assertNotIn("load_jit", kwargs)
        self.assertIn("load_vllm", kwargs)

    def test_cosyvoice3_prompt_text_gets_required_marker(self):
        model = type("CosyVoice3", (), {})()

        prompt = ensure_cosyvoice3_prompt_text(model, "Здравствуйте.")

        self.assertEqual(prompt, "You are a helpful assistant.<|endofprompt|>Здравствуйте.")


if __name__ == "__main__":
    unittest.main()
