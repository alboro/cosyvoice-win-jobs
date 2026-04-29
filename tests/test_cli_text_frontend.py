from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosyvoice_win.cli import (
    CosyVoiceModelOptions,
    QUESTION_INTONATION_INSTRUCTION,
    build_model_kwargs,
    build_runtime_instruction_text,
    ensure_cosyvoice3_prompt_text,
    maybe_disable_text_frontend_imports,
    normalize_instruction_text,
    resolve_effective_mode,
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


class TestInstructionPromotion(unittest.TestCase):
    def test_normalize_instruction_text_prefers_explicit_instruct_text(self):
        self.assertEqual(
            normalize_instruction_text("use russian diction", "ignored instructions"),
            "use russian diction",
        )

    def test_zero_shot_with_instructions_is_promoted_to_instruct2(self):
        self.assertEqual(
            resolve_effective_mode(
                "zero_shot",
                text="Привет.",
                instructions="Read the target text in Russian.",
            ),
            "instruct2",
        )

    def test_question_mark_is_ignored_by_default(self):
        self.assertEqual(
            resolve_effective_mode("zero_shot", text='Это точно?"'),
            "zero_shot",
        )

    def test_trailing_question_mark_promotes_when_explicitly_enabled(self):
        self.assertEqual(
            resolve_effective_mode("zero_shot", text='Это точно?"', fix_question_intonation=True),
            "instruct2",
        )

    def test_question_mark_before_quote_and_period_still_promotes_when_enabled(self):
        self.assertEqual(
            resolve_effective_mode("zero_shot", text='Это точно?".', fix_question_intonation=True),
            "instruct2",
        )

    def test_question_prompt_is_appended_once_when_enabled(self):
        instruction = build_runtime_instruction_text(text='Это точно?"', fix_question_intonation=True)
        self.assertEqual(instruction, QUESTION_INTONATION_INSTRUCTION)

        combined = build_runtime_instruction_text(
            text='Это точно?"',
            instructions="Speak clearly.",
            fix_question_intonation=True,
        )
        self.assertEqual(
            combined,
            "Speak clearly.\n" + QUESTION_INTONATION_INSTRUCTION,
        )

    def test_trailing_question_mark_does_not_promote_when_disabled(self):
        self.assertEqual(
            resolve_effective_mode(
                "zero_shot",
                text='Это точно?"',
                fix_question_intonation=False,
            ),
            "zero_shot",
        )

    def test_cross_lingual_with_question_mark_is_not_auto_promoted(self):
        self.assertEqual(
            resolve_effective_mode("cross_lingual", text="Это точно?", fix_question_intonation=True),
            "cross_lingual",
        )


if __name__ == "__main__":
    unittest.main()
