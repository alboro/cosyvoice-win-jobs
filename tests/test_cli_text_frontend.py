from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosyvoice_win.cli import (
    CosyVoiceModelOptions,
    CosyVoiceSynthesisOptions,
    ResolvedReference,
    build_model_kwargs,
    build_runtime_instruction_text,
    ensure_cosyvoice3_prompt_text,
    generation_controls,
    iter_synthesis,
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

    def test_generation_controls_apply_seed_and_sampling_then_restore(self):
        original_sampling = object()
        llm = SimpleNamespace(sampling=original_sampling)
        model = type("CosyVoice3", (), {"model": SimpleNamespace(llm=llm)})()
        options = CosyVoiceSynthesisOptions(seed=123, temperature=0.5, top_p=0.7, top_k=12)
        calls = []

        def fake_ras(scores, decoded_tokens, sampling, **kwargs):
            calls.append((scores, decoded_tokens, sampling, kwargs))
            return 42

        with (
            patch("cosyvoice_win.cli.load_ras_sampling", return_value=fake_ras),
            patch("cosyvoice_win.cli.set_all_random_seed") as set_seed,
        ):
            with generation_controls(model, options):
                self.assertIsNot(llm.sampling, original_sampling)
                self.assertEqual(llm.sampling(4.0, [1, 2], 25), 42)
                set_seed.assert_called_once_with(123)

        self.assertIs(llm.sampling, original_sampling)
        self.assertEqual(calls[0][0], 8.0)
        self.assertEqual(calls[0][3], {"top_p": 0.7, "top_k": 12})

    def test_generation_controls_reject_vllm_specific_overrides(self):
        llm = SimpleNamespace(sampling=object(), vllm=object())
        model = type("CosyVoice3", (), {"model": SimpleNamespace(llm=llm)})()

        with self.assertRaisesRegex(RuntimeError, "load_vllm=true"):
            with generation_controls(model, CosyVoiceSynthesisOptions(seed=123)):
                pass


class TestInstructionPromotion(unittest.TestCase):
    def test_normalize_instruction_text_prefers_explicit_instruct_text(self):
        self.assertEqual(
            normalize_instruction_text("use russian diction", "ignored instructions"),
            "use russian diction",
        )

    def test_zero_shot_with_instructions_stays_zero_shot(self):
        self.assertEqual(
            resolve_effective_mode(
                "zero_shot",
                text="Привет.",
                instructions="Read the target text in Russian.",
            ),
            "zero_shot",
        )

    def test_explicit_instruct2_with_instructions_stays_instruct2(self):
        self.assertEqual(
            resolve_effective_mode(
                "instruct2",
                text="hello.",
                instructions="Read the target text in Russian.",
            ),
            "instruct2",
        )

    def test_instruct2_does_not_reuse_cached_prompt_text(self):
        class FakeCosyVoice:
            def inference_instruct2(self, text, instruct_text, prompt_audio, **kwargs):
                yield {
                    "text": text,
                    "instruct_text": instruct_text,
                    "prompt_audio": prompt_audio,
                    "kwargs": kwargs,
                }

        reference = ResolvedReference(
            audio_path=Path("reference.wav"),
            prompt_text="Reference transcript must not be used as instruct prompt.",
            reference_source_label="test",
            prompt_source_label="test",
        )
        options = CosyVoiceSynthesisOptions(
            mode="instruct2",
            instruct_text="Read the target text with question intonation.",
        )

        with patch("cosyvoice_win.cli.load_prompt_audio_16k", return_value="prompt-audio"):
            result = list(
                iter_synthesis(
                    FakeCosyVoice(),
                    text="Target text?",
                    voice_id="cached_voice",
                    reference=reference,
                    options=options,
                )
            )[0]

        self.assertEqual(result["instruct_text"], "Read the target text with question intonation.")
        self.assertEqual(result["prompt_audio"], "prompt-audio")
        self.assertEqual(result["kwargs"]["zero_shot_spk_id"], "")


if __name__ == "__main__":
    unittest.main()
