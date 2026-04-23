from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAKE_PROJECT_ROOT = Path(tempfile.gettempdir()) / "cosyvoice_win_jobs_test_root"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if "uvicorn" not in sys.modules:
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)

if "fastapi" not in sys.modules:
    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def get(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

        def delete(self, *args, **kwargs):
            return lambda func: func

    sys.modules["fastapi"] = types.SimpleNamespace(
        FastAPI=_DummyFastAPI,
        HTTPException=Exception,
        status=types.SimpleNamespace(
            HTTP_201_CREATED=201,
            HTTP_202_ACCEPTED=202,
            HTTP_400_BAD_REQUEST=400,
            HTTP_404_NOT_FOUND=404,
            HTTP_409_CONFLICT=409,
            HTTP_500_INTERNAL_SERVER_ERROR=500,
        ),
    )

if "fastapi.responses" not in sys.modules:
    sys.modules["fastapi.responses"] = types.SimpleNamespace(FileResponse=object, Response=object)

if "pydantic" not in sys.modules:
    class _DummyBaseModel:
        pass

    def _dummy_field(default=None, **kwargs):
        return default

    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_DummyBaseModel, Field=_dummy_field)

if "cosyvoice_win.cli" not in sys.modules:
    fake_cli = types.SimpleNamespace(
        DEFAULT_FP16=True,
        DEFAULT_MODEL_DIR=FAKE_PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B",
        DEFAULT_MODEL_ID="CosyVoice2-0.5B",
        DEFAULT_MODE="zero_shot",
        DEFAULT_SHARED_DIR=FAKE_PROJECT_ROOT / "shared",
        DEFAULT_SPEED=1.0,
        DEFAULT_TEXT_FRONTEND=False,
        PROJECT_ROOT=FAKE_PROJECT_ROOT,
        REFERENCE_AUDIO_EXTENSIONS={".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"},
        CosyVoiceModelOptions=lambda **kwargs: kwargs,
        CosyVoiceSynthesisOptions=lambda **kwargs: kwargs,
        ResolvedReference=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ensure_zero_shot_speaker=lambda *args, **kwargs: None,
        estimate_audio_duration_seconds=lambda text, speed=1.0: float(len(text.split())),
        find_reference_audio_in_shared=lambda *args, **kwargs: FAKE_PROJECT_ROOT / "shared" / "reference.wav",
        find_reference_text_for_audio=lambda *args, **kwargs: ("prompt", str(FAKE_PROJECT_ROOT / "shared" / "reference.txt")),
        format_duration=lambda seconds: f"{seconds:.1f}s",
        load_model=lambda *args, **kwargs: object(),
        load_prompt_audio_16k=lambda *args, **kwargs: object(),
        parse_on_off=lambda value: bool(value) if isinstance(value, bool) else str(value).lower() == "on",
        resolve_dir=lambda value: Path(value),
        resolve_model_dir=lambda value: Path(value),
        synthesize_to_file=lambda *args, **kwargs: 1,
    )
    sys.modules["cosyvoice_win.cli"] = fake_cli

from cosyvoice_win.server import JobStore, VoiceStore


class DummyRequest:
    def __init__(
        self,
        *,
        input: str,
        model: str = "CosyVoice2-0.5B",
        voice: str = "reference",
        response_format: str = "wav",
        mode: str | None = None,
        text_frontend: bool | None = None,
        speed: float | None = None,
        instruct_text: str | None = None,
        reference_audio_base64: str | None = None,
        reference_audio_filename: str | None = None,
        reference_text: str | None = None,
        force_rebuild_voice: bool | None = None,
        metadata: dict | None = None,
    ):
        self.input = input
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.mode = mode
        self.text_frontend = text_frontend
        self.speed = speed
        self.instruct_text = instruct_text
        self.reference_audio_base64 = reference_audio_base64
        self.reference_audio_filename = reference_audio_filename
        self.reference_text = reference_text
        self.force_rebuild_voice = force_rebuild_voice
        self.metadata = metadata


def iso_utc(hours_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


class TestJobStoreCleanup(unittest.TestCase):
    def test_mark_downloaded_updates_job_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(DummyRequest(input="hello"))
            store.update_job(job["id"], status="completed", completed_at=iso_utc(1), audio_ready=True)
            store.audio_path(job["id"]).write_bytes(b"RIFF")

            updated = store.mark_downloaded(job["id"])

            self.assertEqual(updated["download_count"], 1)
            self.assertIsNotNone(updated["first_downloaded_at"])
            self.assertIsNotNone(updated["last_downloaded_at"])

    def test_cleanup_removes_old_completed_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(DummyRequest(input="hello"))
            store.update_job(job["id"], status="completed", completed_at=iso_utc(30), audio_ready=True)
            store.audio_path(job["id"]).write_bytes(b"RIFF")

            removed = store.cleanup_expired(
                job_retention=timedelta(hours=24),
                downloaded_job_retention=timedelta(hours=6),
            )

            self.assertEqual(removed, [job["id"]])
            self.assertFalse(store.job_dir(job["id"]).exists())
            self.assertIsNone(store.get_job(job["id"]))

    def test_request_payload_keeps_cosyvoice_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(
                DummyRequest(
                    input="hello",
                    voice="reference_long",
                    mode="zero_shot",
                    text_frontend=False,
                    speed=1.0,
                    reference_text="exact transcript",
                )
            )

            payload_path = store.job_dir(job["id"]) / "request.json"
            payload = payload_path.read_text(encoding="utf-8")

            self.assertIn('"voice": "reference_long"', payload)
            self.assertIn('"mode": "zero_shot"', payload)
            self.assertIn('"reference_text": "exact transcript"', payload)


class TestVoiceStore(unittest.TestCase):
    def test_put_and_get_voice_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VoiceStore(Path(temp_dir))
            path = store.put(
                "reference_long",
                {
                    "voice": "reference_long",
                    "mode": "zero_shot",
                    "reference_text_present": True,
                },
            )

            self.assertTrue(path.exists())
            loaded = store.get("reference_long")
            self.assertEqual(loaded["voice"], "reference_long")
            self.assertTrue(loaded["reference_text_present"])

    def test_delete_removes_profile_and_reference_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VoiceStore(Path(temp_dir))
            store.put("narrator", {"voice": "narrator"})
            audio = store.reference_path("narrator", ".wav")
            audio.write_bytes(b"RIFF")

            self.assertTrue(store.profile_path("narrator").exists())
            self.assertTrue(audio.exists())

            deleted = store.delete("narrator")

            self.assertTrue(deleted)
            self.assertFalse(store.profile_path("narrator").exists())
            self.assertFalse(audio.exists())
            self.assertFalse(store.delete("narrator"))

    def test_clear_reference_audio_keeps_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VoiceStore(Path(temp_dir))
            store.put("narrator", {"voice": "narrator"})
            store.reference_path("narrator", ".m4a").write_bytes(b"x")

            store.clear_reference_audio("narrator")

            self.assertTrue(store.profile_path("narrator").exists())
            self.assertFalse(store.reference_path("narrator", ".m4a").exists())


class TestVoiceRegistrar(unittest.TestCase):
    def test_marks_ready_on_success(self):
        from cosyvoice_win.server import VoiceRegistrar

        with tempfile.TemporaryDirectory() as temp_dir:
            voices = VoiceStore(Path(temp_dir))
            voices.put("anna", {"voice": "anna", "status": "registering", "error": None})

            class FakeWorker:
                def register_voice(self, *, voice_id, reference_audio, reference_text, mode):
                    # mimic _prepare_voice rewriting the profile on success
                    voices.put(voice_id, {"voice": voice_id, "reference_audio": str(reference_audio)})
                    return voices.profile_path(voice_id)

            registrar = VoiceRegistrar(FakeWorker(), voices)
            registrar._register("anna", Path(temp_dir) / "anna.wav", "transcript", "zero_shot")

            profile = voices.get("anna")
            self.assertEqual(profile["status"], "ready")
            self.assertIsNone(profile["error"])

    def test_marks_failed_on_error(self):
        from cosyvoice_win.server import VoiceRegistrar

        with tempfile.TemporaryDirectory() as temp_dir:
            voices = VoiceStore(Path(temp_dir))
            voices.put("anna", {"voice": "anna", "status": "registering", "error": None})

            class FakeWorker:
                def register_voice(self, **kwargs):
                    raise RuntimeError("boom")

            registrar = VoiceRegistrar(FakeWorker(), voices)
            registrar._register("anna", Path(temp_dir) / "anna.wav", "transcript", "zero_shot")

            profile = voices.get("anna")
            self.assertEqual(profile["status"], "failed")
            self.assertEqual(profile["error"], "boom")


class TestResponseFormat(unittest.TestCase):
    def test_encode_output_wav_passthrough(self):
        from cosyvoice_win.server import encode_output

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "a.wav"
            wav.write_bytes(b"RIFFwavbytes")
            content, media_type, filename = encode_output(wav, "wav")
            self.assertEqual(content, b"RIFFwavbytes")
            self.assertEqual(media_type, "audio/wav")
            self.assertEqual(filename, "speech.wav")

    def test_direct_response_formats_include_transcodes(self):
        from cosyvoice_win.server import DIRECT_RESPONSE_FORMATS

        self.assertEqual(DIRECT_RESPONSE_FORMATS, {"wav", "mp3", "opus", "ogg"})

    def test_encode_output_opus_transcodes(self):
        import shutil
        import wave

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")
        from cosyvoice_win.server import encode_output

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "a.wav"
            with wave.open(str(wav), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(b"\x00\x01" * 4800)  # 0.2s of audio
            content, media_type, filename = encode_output(wav, "opus")
            self.assertEqual(media_type, "audio/ogg")
            self.assertEqual(filename, "speech.opus")
            self.assertTrue(content.startswith(b"OggS"), "expected an Ogg stream")


class TestBuildRequestPayload(unittest.TestCase):
    def test_payload_matches_prepare_voice_keys(self):
        from cosyvoice_win.server import build_request_payload

        payload = build_request_payload(
            DummyRequest(input="hello", voice="narrator", mode="zero_shot", reference_text="exact")
        )
        self.assertEqual(payload["input"], "hello")
        self.assertEqual(payload["voice"], "narrator")
        self.assertEqual(payload["mode"], "zero_shot")
        self.assertEqual(payload["reference_text"], "exact")
        self.assertEqual(payload["metadata"], {})


if __name__ == "__main__":
    unittest.main()
