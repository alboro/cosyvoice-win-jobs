from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import os
import queue
import shutil
import struct
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from cosyvoice_win.cli import (
    DEFAULT_FP16,
    DEFAULT_FIX_QUESTION_INTONATION,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_MODE,
    DEFAULT_SHARED_DIR,
    DEFAULT_SPEED,
    DEFAULT_TEXT_FRONTEND,
    PROJECT_ROOT,
    REFERENCE_AUDIO_EXTENSIONS,
    CosyVoiceModelOptions,
    CosyVoiceSynthesisOptions,
    ResolvedReference,
    ensure_zero_shot_speaker,
    estimate_audio_duration_seconds,
    find_reference_audio_in_shared,
    find_reference_text_for_audio,
    format_duration,
    build_runtime_instruction_text,
    load_model,
    load_prompt_audio_16k,
    parse_on_off,
    resolve_effective_mode,
    resolve_dir,
    resolve_model_dir,
    iter_synthesis,
    synthesize_to_file,
)

DEFAULT_JOBS_DIR = PROJECT_ROOT / ".data" / "jobs"
DEFAULT_VOICES_DIR = PROJECT_ROOT / ".data" / "voices"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8040
DEFAULT_JOB_OUTPUT_NAME = "audio.wav"
DEFAULT_JOB_RETENTION_HOURS = 24
DEFAULT_DOWNLOADED_JOB_RETENTION_HOURS = 6
DEFAULT_CLEANUP_INTERVAL_SECONDS = 900
SUPPORTED_JOB_RESPONSE_FORMATS = {"wav"}
SUPPORTED_DIRECT_RESPONSE_FORMATS = {"wav", "pcm"}

AUDIO_MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
}

logger = logging.getLogger(__name__)


class CreateTTSJobRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str = Field(default=DEFAULT_MODEL_ID)
    voice: str = Field(default="reference")
    response_format: str = Field(default="wav")
    mode: str | None = Field(default=None, pattern="^(zero_shot|cross_lingual|instruct2)$")
    text_frontend: bool | None = None
    speed: float | None = Field(default=None, gt=0)
    stream: bool | None = None
    fix_question_intonation: bool | None = None
    instructions: str | None = None
    instruct_text: str | None = None
    reference_audio_base64: str | None = None
    reference_audio_filename: str | None = None
    reference_text: str | None = None
    force_rebuild_voice: bool | None = None
    metadata: dict[str, Any] | None = None


class RegisterVoiceRequest(BaseModel):
    """One-time registration of a named voice. Upload a reference clip once
    (base64 or data-URI) plus its transcript; the server builds the zero-shot
    speaker, stores the reference under ``.data/voices/``, and writes a profile.
    Synthesis requests then refer to it by ``voice`` id alone."""

    name: str = Field(..., min_length=1)
    reference_audio_base64: str = Field(..., min_length=1)
    reference_text: str | None = None
    filename: str | None = None
    mode: str | None = Field(default=None, pattern="^(zero_shot|cross_lingual)$")
    overwrite: bool = False


@dataclass(slots=True)
class ServerSettings:
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_SERVER_PORT
    shared_dir: Path = DEFAULT_SHARED_DIR
    jobs_dir: Path = DEFAULT_JOBS_DIR
    voices_dir: Path = DEFAULT_VOICES_DIR
    model_id: str = DEFAULT_MODEL_ID
    model_dir: Path = DEFAULT_MODEL_DIR
    mode: str = DEFAULT_MODE
    text_frontend: bool = DEFAULT_TEXT_FRONTEND
    fix_question_intonation: bool = DEFAULT_FIX_QUESTION_INTONATION
    speed: float = DEFAULT_SPEED
    fp16: bool = DEFAULT_FP16
    load_jit: bool = False
    load_trt: bool = False
    load_vllm: bool = False
    job_retention_hours: int = DEFAULT_JOB_RETENTION_HOURS
    downloaded_job_retention_hours: int = DEFAULT_DOWNLOADED_JOB_RETENTION_HOURS
    cleanup_interval_seconds: int = DEFAULT_CLEANUP_INTERVAL_SECONDS
    api_key: str = ""


@dataclass(slots=True)
class LoadedModel:
    tts: Any
    load_seconds: float


class CreateSpeechRequest(CreateTTSJobRequest):
    """OpenAI-compatible /v1/audio/speech request.

    The async jobs API accepts the same superset for compatibility, but only
    this endpoint uses ``stream`` to return audio directly.
    """


def audio_media_type(response_format: str) -> str:
    normalized = response_format.lower()
    if normalized == "pcm":
        return "audio/pcm; codecs=pcm_s16le"
    return "audio/wav"


def wav_header(sample_rate: int, data_size: int | None = None, *, channels: int = 1) -> bytes:
    """Build a PCM WAV header.

    ``data_size=None`` is used for chunked transfer where the final length is
    not known before streaming starts. Most clients tolerate the max-size
    placeholder and play the stream incrementally.
    """
    bits_per_sample = 16
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    if data_size is None:
        riff_size = 0xFFFFFFFF
        data_size = 0xFFFFFFFF
    else:
        riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def output_to_pcm16_bytes(output: dict[str, Any]) -> bytes:
    speech = output["tts_speech"].detach().cpu().float()
    if getattr(speech, "dim", lambda: 0)() == 2:
        if speech.shape[0] == 1:
            speech = speech.squeeze(0)
        else:
            speech = speech[0]
    speech = speech.clamp(-1.0, 1.0).mul(32767.0).short().contiguous()
    return speech.numpy().tobytes()


def render_audio_bytes(outputs, sample_rate: int, response_format: str) -> bytes:
    pcm = b"".join(output_to_pcm16_bytes(output) for output in outputs)
    if response_format.lower() == "pcm":
        return pcm
    return wav_header(sample_rate, len(pcm)) + pcm


class JobStore:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        for job_file in self.jobs_dir.glob("*/job.json"):
            job = json.loads(job_file.read_text(encoding="utf-8"))
            if job["status"] in {"queued", "in_progress"}:
                now = utcnow_iso()
                job["status"] = "failed"
                job["error"] = "Server restarted before the job completed."
                job["failed_at"] = now
                job["updated_at"] = now
                job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
            self._jobs[job["id"]] = job

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def audio_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / DEFAULT_JOB_OUTPUT_NAME

    def uploaded_reference_path(self, job_id: str) -> Path | None:
        job = self.get_job(job_id)
        if not job or not job.get("reference_filename"):
            return None
        return self.job_dir(job_id) / str(job["reference_filename"])

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    def create_job(self, request: CreateTTSJobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = utcnow_iso()
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        audio_path = job_dir / DEFAULT_JOB_OUTPUT_NAME

        job = {
            "id": job_id,
            "object": "tts.job",
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "failed_at": None,
            "model": request.model,
            "voice": request.voice,
            "response_format": request.response_format,
            "mode": request.mode or DEFAULT_MODE,
            "input_characters": len(request.input),
            "input_words": len(request.input.split()),
            "estimated_speech_seconds": estimate_audio_duration_seconds(request.input, request.speed or DEFAULT_SPEED),
            "status_url": f"/v1/tts/jobs/{job_id}",
            "audio_url": f"/v1/tts/jobs/{job_id}/audio",
            "audio_ready": False,
            "audio_path": str(audio_path),
            "error": None,
            "metadata": request.metadata or {},
            "download_count": 0,
            "first_downloaded_at": None,
            "last_downloaded_at": None,
            "segments_returned": None,
            "runtime_seconds": None,
            "model_load_seconds": None,
            "synthesis_seconds": None,
            "voice_cached": None,
            "voice_profile_path": None,
            "reference_filename": None,
        }

        request_payload = {
            "input": request.input,
            "model": request.model,
            "voice": request.voice,
            "response_format": request.response_format,
            "mode": request.mode,
            "text_frontend": request.text_frontend,
            "speed": request.speed,
            "stream": request.stream,
            "fix_question_intonation": request.fix_question_intonation,
            "instructions": request.instructions,
            "instruct_text": request.instruct_text,
            "reference_text": request.reference_text,
            "force_rebuild_voice": request.force_rebuild_voice,
            "metadata": request.metadata or {},
            "reference_audio_uploaded": bool(request.reference_audio_base64),
            "reference_audio_filename": request.reference_audio_filename,
        }

        if request.reference_audio_base64:
            reference_path = decode_reference_audio_to_file(
                encoded=request.reference_audio_base64,
                filename=request.reference_audio_filename,
                job_dir=job_dir,
            )
            request_payload["reference_file"] = reference_path.name
            job["reference_filename"] = reference_path.name

        (job_dir / "request.json").write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self._lock:
            self._jobs[job_id] = job
            self._write_job(job)
        return self.public_view(job)

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.update(updates)
            job["updated_at"] = utcnow_iso()
            self._write_job(job)
            return dict(job)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def mark_downloaded(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            now = utcnow_iso()
            job["download_count"] = int(job.get("download_count") or 0) + 1
            job["first_downloaded_at"] = job.get("first_downloaded_at") or now
            job["last_downloaded_at"] = now
            job["updated_at"] = now
            self._write_job(job)
            return dict(job)

    def public_view(self, job: dict[str, Any]) -> dict[str, Any]:
        public_job = dict(job)
        public_job.pop("audio_path", None)
        return public_job

    def cleanup_expired(self, *, job_retention: timedelta, downloaded_job_retention: timedelta) -> list[str]:
        expired: list[tuple[str, Path]] = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if not self._is_terminal(job):
                    continue
                if not self._is_expired(
                    job,
                    now=now,
                    job_retention=job_retention,
                    downloaded_job_retention=downloaded_job_retention,
                ):
                    continue
                expired.append((job_id, self.job_dir(job_id)))
                self._jobs.pop(job_id, None)

        removed_ids: list[str] = []
        for job_id, job_dir in expired:
            shutil.rmtree(job_dir, ignore_errors=True)
            removed_ids.append(job_id)
        return removed_ids

    def _write_job(self, job: dict[str, Any]) -> None:
        (self.job_dir(job["id"]) / "job.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _is_terminal(job: dict[str, Any]) -> bool:
        return str(job.get("status") or "") in {"completed", "failed"}

    def _is_expired(
        self,
        job: dict[str, Any],
        *,
        now: datetime,
        job_retention: timedelta,
        downloaded_job_retention: timedelta,
    ) -> bool:
        downloaded_at = parse_iso_datetime(job.get("last_downloaded_at"))
        if downloaded_at is not None:
            return now >= downloaded_at + downloaded_job_retention

        terminal_at = (
            parse_iso_datetime(job.get("completed_at"))
            or parse_iso_datetime(job.get("failed_at"))
            or parse_iso_datetime(job.get("updated_at"))
        )
        if terminal_at is None:
            return False
        return now >= terminal_at + job_retention


class VoiceStore:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(voice_id: str) -> str:
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in (voice_id or "")).strip("_")
        return safe_name or "reference"

    def profile_path(self, voice_id: str) -> Path:
        return self.voices_dir / f"{self._safe(voice_id)}.json"

    def reference_path(self, voice_id: str, suffix: str) -> Path:
        """Stable location for a registered voice's reference audio, kept next to
        its profile so the speaker cache can be rebuilt after a restart."""
        return self.voices_dir / f"{self._safe(voice_id)}{suffix}"

    def get(self, voice_id: str) -> dict[str, Any] | None:
        path = self.profile_path(voice_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, voice_id: str, payload: dict[str, Any]) -> Path:
        path = self.profile_path(voice_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def clear_reference_audio(self, voice_id: str) -> None:
        safe = self._safe(voice_id)
        for path in self.voices_dir.glob(f"{safe}.*"):
            if path.suffix.lower() != ".json":
                path.unlink()

    def delete(self, voice_id: str) -> bool:
        """Remove a voice profile and its stored reference audio. Returns whether
        a profile existed."""
        profile = self.profile_path(voice_id)
        existed = profile.is_file()
        if existed:
            profile.unlink()
        self.clear_reference_audio(voice_id)
        return existed

    def count(self) -> int:
        return sum(1 for _ in self.voices_dir.glob("*.json"))


class JobGarbageCollector:
    def __init__(self, settings: ServerSettings, store: JobStore):
        self.settings = settings
        self.store = store
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cosyvoice-job-cleaner", daemon=True)
        self._startup_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._startup_lock:
            if self._started:
                return
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        self._stop_event.set()

    def sweep_now(self) -> list[str]:
        removed = self.store.cleanup_expired(
            job_retention=timedelta(hours=self.settings.job_retention_hours),
            downloaded_job_retention=timedelta(hours=self.settings.downloaded_job_retention_hours),
        )
        if removed:
            logger.info("Cleaned up %s expired TTS job(s): %s", len(removed), ", ".join(removed))
        return removed

    def _run(self) -> None:
        self.sweep_now()
        while not self._stop_event.wait(self.settings.cleanup_interval_seconds):
            self.sweep_now()


class SynthesisWorker:
    def __init__(self, settings: ServerSettings, store: JobStore, voices: VoiceStore):
        self.settings = settings
        self.store = store
        self.voices = voices
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="cosyvoice-job-worker", daemon=True)
        self._loaded_model: LoadedModel | None = None
        self._seeded_voice_ids: set[str] = set()
        self._startup_lock = threading.Lock()
        self._started = False
        # CosyVoice is one resident, non-re-entrant torch model. The polling worker
        # thread and the synchronous /v1/audio/speech endpoint (FastAPI threadpool)
        # can both reach it, so all inference/unload is serialized here.
        self._infer_lock = threading.Lock()

    def start(self) -> None:
        with self._startup_lock:
            if self._started:
                return
            self._thread.start()
            self._started = True

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _ensure_model(self) -> LoadedModel:
        if self._loaded_model is not None:
            return self._loaded_model

        started = time.perf_counter()
        tts = load_model(
            CosyVoiceModelOptions(
                model_dir=self.settings.model_dir,
                fp16=self.settings.fp16,
                load_jit=self.settings.load_jit,
                load_trt=self.settings.load_trt,
                load_vllm=self.settings.load_vllm,
                text_frontend=self.settings.text_frontend,
            )
        )
        self._loaded_model = LoadedModel(tts=tts, load_seconds=time.perf_counter() - started)
        return self._loaded_model

    def synthesize_request(
        self,
        request: "CreateTTSJobRequest",
        *,
        output_path: Path,
        uploaded_reference: Path | None = None,
    ) -> int:
        """Synchronously render one request to ``output_path`` using the shared
        resident model. Powers ``POST /v1/audio/speech`` (no polling). Serialized
        with the polling worker via ``_infer_lock``."""
        payload = build_request_payload(request)
        voice_id = str(payload["voice"]).strip() or "reference"
        options = self._build_job_synthesis_options(payload)
        with self._infer_lock:
            loaded_model = self._ensure_model()
            reference, _ = self._prepare_voice(
                loaded_model.tts,
                voice_id=voice_id,
                uploaded_reference=uploaded_reference,
                payload=payload,
            )
            return synthesize_to_file(
                loaded_model.tts,
                text=str(payload["input"]).strip(),
                voice_id=voice_id,
                reference=reference,
                output_path=output_path,
                options=options,
            )

    def unload(self) -> list[str]:
        """Drop the resident CosyVoice model so RSS/VRAM falls back down. Clients
        call this when a batch finishes; the next request reloads it lazily."""
        import gc as _gc

        with self._infer_lock:
            freed: list[str] = []
            if self._loaded_model is not None:
                self._loaded_model = None
                self._seeded_voice_ids.clear()
                freed.append("cosyvoice")
            _gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return freed

    def register_voice(
        self,
        *,
        voice_id: str,
        reference_audio: Path,
        reference_text: str | None,
        mode: str,
    ) -> Path:
        """Build (or rebuild) the zero-shot speaker for ``voice_id`` from a stored
        reference and persist its profile, so later requests can synthesize by id
        alone. Powers ``POST /v1/voices``."""
        payload = {
            "input": "",
            "voice": voice_id,
            "mode": mode,
            "text_frontend": None,
            "speed": None,
            "reference_text": reference_text,
            "force_rebuild_voice": True,
        }
        with self._infer_lock:
            loaded_model = self._ensure_model()
            _, profile_path = self._prepare_voice(
                loaded_model.tts,
                voice_id=voice_id,
                uploaded_reference=reference_audio,
                payload=payload,
            )
        if profile_path is None:
            raise RuntimeError(f"Failed to register voice '{voice_id}': speaker cache was not created.")
        return profile_path

    def forget_voice(self, voice_id: str) -> None:
        self._seeded_voice_ids.discard(voice_id)

    def _process_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return

        log_path = self.store.log_path(job_id)
        with log_path.open("a", encoding="utf-8") as log_file:
            route_python_logging_to_stream(log_file)
            try:
                started_at = utcnow_iso()
                self.store.update_job(job_id, status="in_progress", started_at=started_at)
                log_line(log_file, f"Job {job_id} started at {started_at}")

                payload = self._load_job_request_payload(job_id)
                voice_id = str(payload["voice"]).strip() or "reference"
                options = self._build_job_synthesis_options(payload)
                output_path = self.store.audio_path(job_id)

                runtime_started = time.perf_counter()
                synthesis_started = time.perf_counter()

                # Serialize against the synchronous /v1/audio/speech endpoint and
                # /v1/unload: the resident CosyVoice model is not re-entrant.
                with self._infer_lock:
                    with redirect_stdout(log_file), redirect_stderr(log_file):
                        loaded_model = self._ensure_model()
                    log_line(log_file, f"Model ready in {loaded_model.load_seconds:.2f}s")

                    with redirect_stdout(log_file), redirect_stderr(log_file):
                        reference, voice_profile_path = self._prepare_voice(
                            loaded_model.tts,
                            voice_id=voice_id,
                            uploaded_reference=self.store.uploaded_reference_path(job_id),
                            payload=payload,
                        )
                        segments_returned = synthesize_to_file(
                            loaded_model.tts,
                            text=str(payload["input"]).strip(),
                            voice_id=voice_id,
                            reference=reference,
                            output_path=output_path,
                            options=options,
                        )

                synthesis_seconds = time.perf_counter() - synthesis_started
                wall_seconds = time.perf_counter() - runtime_started
                completed_at = utcnow_iso()
                self.store.update_job(
                    job_id,
                    status="completed",
                    completed_at=completed_at,
                    audio_ready=output_path.is_file(),
                    runtime_seconds=wall_seconds,
                    model_load_seconds=loaded_model.load_seconds,
                    synthesis_seconds=synthesis_seconds,
                    segments_returned=segments_returned,
                    voice_cached=True,
                    voice_profile_path=str(voice_profile_path) if voice_profile_path else None,
                )
                log_line(log_file, f"Segments returned: {segments_returned}")
                log_line(log_file, f"Runtime: {format_duration(wall_seconds)}")
                log_line(log_file, f"Synthesis: {format_duration(synthesis_seconds)}")
                log_line(log_file, f"Job {job_id} completed at {completed_at}")
            except Exception as exc:
                failed_at = utcnow_iso()
                self.store.update_job(
                    job_id,
                    status="failed",
                    failed_at=failed_at,
                    error=str(exc),
                    audio_ready=False,
                )
                log_line(log_file, f"Job {job_id} failed at {failed_at}: {exc}")
            finally:
                route_python_logging_to_stream(sys.stderr)

    def _load_job_request_payload(self, job_id: str) -> dict[str, Any]:
        request_path = self.store.job_dir(job_id) / "request.json"
        return json.loads(request_path.read_text(encoding="utf-8"))

    def _build_job_synthesis_options(self, payload: dict[str, Any]) -> CosyVoiceSynthesisOptions:
        fix_question_intonation = _pick_override(
            payload.get("fix_question_intonation"),
            self.settings.fix_question_intonation,
        )
        instruct_text = build_runtime_instruction_text(
            text=str(payload.get("input") or ""),
            instruct_text=payload.get("instruct_text"),
            instructions=payload.get("instructions"),
            fix_question_intonation=bool(fix_question_intonation),
        )
        requested_mode = str(payload.get("mode") or self.settings.mode)
        return CosyVoiceSynthesisOptions(
            mode=resolve_effective_mode(
                requested_mode,
                text=str(payload.get("input") or ""),
                instruct_text=payload.get("instruct_text"),
                instructions=payload.get("instructions"),
                fix_question_intonation=bool(fix_question_intonation),
            ),
            requested_mode=requested_mode,
            text_frontend=_pick_override(payload.get("text_frontend"), self.settings.text_frontend),
            speed=float(_pick_override(payload.get("speed"), self.settings.speed)),
            fix_question_intonation=bool(fix_question_intonation),
            instruct_text=instruct_text,
            stream=bool(payload.get("stream")),
        )

    def iter_direct_synthesis(
        self,
        payload: dict[str, Any],
        *,
        uploaded_reference: Path | None = None,
    ):
        voice_id = str(payload["voice"]).strip() or "reference"
        options = self._build_job_synthesis_options(payload)
        with self._infer_lock:
            loaded_model = self._ensure_model()
            reference, _voice_profile_path = self._prepare_voice(
                loaded_model.tts,
                voice_id=voice_id,
                payload=payload,
                uploaded_reference=uploaded_reference,
            )
            yield from iter_synthesis(
                loaded_model.tts,
                text=str(payload["input"]).strip(),
                voice_id=voice_id,
                reference=reference,
                options=options,
            )

    def synthesize_direct_bytes(
        self,
        payload: dict[str, Any],
        *,
        response_format: str,
        uploaded_reference: Path | None = None,
    ) -> bytes:
        outputs = list(self.iter_direct_synthesis(payload, uploaded_reference=uploaded_reference))
        loaded_model = self._loaded_model
        if loaded_model is None:
            raise RuntimeError("CosyVoice model was not loaded.")
        return render_audio_bytes(outputs, loaded_model.tts.sample_rate, response_format)

    def stream_direct_audio(
        self,
        payload: dict[str, Any],
        *,
        response_format: str,
        uploaded_reference: Path | None = None,
    ):
        voice_id = str(payload["voice"]).strip() or "reference"
        options = self._build_job_synthesis_options(payload)
        with self._infer_lock:
            loaded_model = self._ensure_model()
            reference, _voice_profile_path = self._prepare_voice(
                loaded_model.tts,
                voice_id=voice_id,
                payload=payload,
                uploaded_reference=uploaded_reference,
            )
            if response_format.lower() == "wav":
                yield wav_header(loaded_model.tts.sample_rate)
            for output in iter_synthesis(
                loaded_model.tts,
                text=str(payload["input"]).strip(),
                voice_id=voice_id,
                reference=reference,
                options=options,
            ):
                chunk = output_to_pcm16_bytes(output)
                if chunk:
                    yield chunk

    def _prepare_voice(
        self,
        cosyvoice_model,
        *,
        voice_id: str,
        payload: dict[str, Any],
        uploaded_reference: Path | None = None,
    ) -> tuple[ResolvedReference, Path | None]:
        force_rebuild = bool(payload.get("force_rebuild_voice"))
        profile = self.voices.get(voice_id)

        shared_audio: Path | None = None
        shared_text: str | None = None
        shared_text_source = "<none>"

        if uploaded_reference is None:
            try:
                shared_audio = find_reference_audio_in_shared(self.settings.shared_dir, voice_id)
                shared_text, shared_text_source = find_reference_text_for_audio(shared_audio, voice_id)
            except FileNotFoundError:
                shared_audio = None

        profile_audio: Path | None = None
        profile_text = ""
        profile_text_source = "<none>"
        if profile:
            audio_raw = str(profile.get("reference_audio") or "").strip()
            if audio_raw:
                candidate = Path(audio_raw)
                if candidate.is_file():
                    profile_audio = candidate
            profile_text = str(profile.get("reference_text") or "").strip()
            if profile_text:
                profile_text_source = "<voice profile>"

        prompt_audio_path = uploaded_reference or shared_audio or profile_audio
        prompt_text = (payload.get("reference_text") or "").strip() or shared_text or profile_text or ""
        if (payload.get("reference_text") or "").strip():
            prompt_source = "uploaded reference text"
        elif shared_text:
            prompt_source = shared_text_source
        else:
            prompt_source = profile_text_source

        mode = str(payload.get("mode") or self.settings.mode)
        if prompt_audio_path is not None and mode in {"cross_lingual", "instruct2"}:
            return (
                ResolvedReference(
                    audio_path=prompt_audio_path,
                    prompt_text=prompt_text or None,
                    reference_source_label=f"direct {mode} reference",
                    prompt_source_label=prompt_source,
                ),
                None,
            )

        should_seed_cache = force_rebuild or voice_id not in self._seeded_voice_ids
        if prompt_audio_path is not None and should_seed_cache:
            prompt_audio_16k = load_prompt_audio_16k(prompt_audio_path)
            if mode == "zero_shot" and not prompt_text:
                raise ValueError("zero_shot mode requires reference_text or a sidecar transcript for the shared voice.")
            ensure_zero_shot_speaker(
                cosyvoice_model,
                voice_id=voice_id,
                prompt_text=prompt_text,
                prompt_audio_16k=prompt_audio_16k,
                persist=True,
            )
            self._seeded_voice_ids.add(voice_id)
            profile_path = self.voices.put(
                voice_id,
                {
                    "voice": voice_id,
                    "created_at": utcnow_iso(),
                    "mode": payload.get("mode") or self.settings.mode,
                    "reference_audio": str(prompt_audio_path),
                    "reference_text": prompt_text,
                    "reference_text_present": bool(prompt_text),
                    "prompt_source": prompt_source,
                },
            )
            return (
                ResolvedReference(
                    audio_path=None,
                    prompt_text=None,
                    reference_source_label="cached speaker",
                    prompt_source_label="<cached>",
                ),
                profile_path,
            )

        if profile is not None and voice_id in self._seeded_voice_ids:
            return (
                ResolvedReference(
                    audio_path=None,
                    prompt_text=None,
                    reference_source_label="cached speaker",
                    prompt_source_label="<cached>",
                ),
                self.voices.profile_path(voice_id),
            )

        if profile is not None:
            raise ValueError(
                f"Voice '{voice_id}' has stored metadata, but its reference audio is unavailable for cache restore."
            )

        raise ValueError(
            f"Voice '{voice_id}' is not cached and no usable reference bundle was found. "
            "Seed the cache with shared voice files or upload reference_audio_base64 + reference_text."
        )


class VoiceRegistrar:
    """Runs voice registration off the request thread. ``POST /v1/voices`` writes
    a ``status: "registering"`` profile and returns immediately; this single
    background worker then builds the zero-shot speaker (a heavy, model-loading
    step) and flips the profile to ``ready`` or ``failed``. Clients poll
    ``GET /v1/voices/{name}`` for the outcome."""

    def __init__(self, worker: SynthesisWorker, voices: VoiceStore):
        self.worker = worker
        self.voices = voices
        self._queue: queue.Queue[tuple[str, Path, str | None, str]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="cosyvoice-voice-registrar", daemon=True)
        self._started = False
        self._startup_lock = threading.Lock()

    def start(self) -> None:
        with self._startup_lock:
            if self._started:
                return
            self._thread.start()
            self._started = True

    def submit(self, *, voice_id: str, reference_audio: Path, reference_text: str | None, mode: str) -> None:
        self._queue.put((voice_id, reference_audio, reference_text, mode))

    def _run(self) -> None:
        while True:
            voice_id, reference_audio, reference_text, mode = self._queue.get()
            try:
                self._register(voice_id, reference_audio, reference_text, mode)
            finally:
                self._queue.task_done()

    def _register(self, voice_id: str, reference_audio: Path, reference_text: str | None, mode: str) -> None:
        try:
            self.worker.register_voice(
                voice_id=voice_id,
                reference_audio=reference_audio,
                reference_text=reference_text,
                mode=mode,
            )
            # register_voice rewrote the profile via _prepare_voice; stamp it ready.
            self._mark(voice_id, status="ready", error=None)
            logger.info("Registered voice '%s'.", voice_id)
        except Exception as exc:  # keep a failed profile so the client can see why
            self._mark(voice_id, status="failed", error=str(exc))
            logger.warning("Voice registration failed for '%s': %s", voice_id, exc)

    def _mark(self, voice_id: str, *, status: str, error: str | None) -> None:
        profile = self.voices.get(voice_id) or {"voice": voice_id}
        profile["status"] = status
        profile["error"] = error
        profile["updated_at"] = utcnow_iso()
        self.voices.put(voice_id, profile)


def parse_args(argv: list[str] | None = None) -> ServerSettings:
    parser = argparse.ArgumentParser(
        prog="cosyvoice-win-server",
        description="Async polling jobs server for CosyVoice.",
    )
    parser.add_argument("--host", default=os.environ.get("COSYVOICE_HOST", DEFAULT_SERVER_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("COSYVOICE_PORT", str(DEFAULT_SERVER_PORT))),
    )
    parser.add_argument("--shared-dir", default=os.environ.get("COSYVOICE_SHARED_DIR", str(DEFAULT_SHARED_DIR)))
    parser.add_argument("--jobs-dir", default=os.environ.get("COSYVOICE_JOBS_DIR", str(DEFAULT_JOBS_DIR)))
    parser.add_argument("--voices-dir", default=os.environ.get("COSYVOICE_VOICES_DIR", str(DEFAULT_VOICES_DIR)))
    parser.add_argument("--model-id", default=os.environ.get("COSYVOICE_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    parser.add_argument("--mode", choices=("zero_shot", "cross_lingual", "instruct2"), default=DEFAULT_MODE)
    parser.add_argument(
        "--text-frontend",
        choices=("on", "off"),
        default="on" if DEFAULT_TEXT_FRONTEND else "off",
    )
    parser.add_argument(
        "--fix-question-intonation",
        choices=("on", "off"),
        default="on" if DEFAULT_FIX_QUESTION_INTONATION else "off",
    )
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--fp16", choices=("on", "off"), default="on" if DEFAULT_FP16 else "off")
    parser.add_argument("--load-jit", choices=("on", "off"), default="off")
    parser.add_argument("--load-trt", choices=("on", "off"), default="off")
    parser.add_argument("--load-vllm", choices=("on", "off"), default="off")
    parser.add_argument("--job-retention-hours", type=int, default=DEFAULT_JOB_RETENTION_HOURS)
    parser.add_argument(
        "--downloaded-job-retention-hours",
        type=int,
        default=DEFAULT_DOWNLOADED_JOB_RETENTION_HOURS,
    )
    parser.add_argument("--cleanup-interval-seconds", type=int, default=DEFAULT_CLEANUP_INTERVAL_SECONDS)
    parser.add_argument("--api-key", default=os.environ.get("COSYVOICE_API_KEY", ""))
    args = parser.parse_args(argv)
    return ServerSettings(
        host=args.host,
        port=args.port,
        shared_dir=resolve_dir(args.shared_dir),
        jobs_dir=Path(args.jobs_dir).expanduser().resolve(),
        voices_dir=Path(args.voices_dir).expanduser().resolve(),
        model_id=args.model_id,
        model_dir=resolve_model_dir(args.model_dir),
        mode=args.mode,
        text_frontend=parse_on_off(args.text_frontend),
        fix_question_intonation=parse_on_off(args.fix_question_intonation),
        speed=args.speed,
        fp16=parse_on_off(args.fp16),
        load_jit=parse_on_off(args.load_jit),
        load_trt=parse_on_off(args.load_trt),
        load_vllm=parse_on_off(args.load_vllm),
        job_retention_hours=max(args.job_retention_hours, 0),
        downloaded_job_retention_hours=max(args.downloaded_job_retention_hours, 0),
        cleanup_interval_seconds=max(args.cleanup_interval_seconds, 1),
        api_key=args.api_key,
    )


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    server_settings = settings or ServerSettings(
        shared_dir=resolve_dir(str(DEFAULT_SHARED_DIR)),
        jobs_dir=DEFAULT_JOBS_DIR,
        voices_dir=DEFAULT_VOICES_DIR,
        model_dir=DEFAULT_MODEL_DIR,
    )
    store = JobStore(server_settings.jobs_dir)
    voices = VoiceStore(server_settings.voices_dir)
    worker = SynthesisWorker(server_settings, store, voices)
    registrar = VoiceRegistrar(worker, voices)
    gc = JobGarbageCollector(server_settings, store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        registrar.start()
        gc.start()
        yield
        gc.stop()

    app = FastAPI(title="cosyvoice-win-jobs", version="0.1.0", lifespan=lifespan)
    app.state.settings = server_settings
    app.state.store = store
    app.state.voices = voices
    app.state.worker = worker
    app.state.registrar = registrar
    app.state.gc = gc

    # Auth: if COSYVOICE_API_KEY is set, require "Authorization: Bearer <key>" (or
    # "X-Api-Key: <key>") on all /v1/* routes. /health is always open (used by deploy).
    _api_key = server_settings.api_key or None
    if _api_key:
        @app.middleware("http")
        async def _bearer_auth(request: Request, call_next):
            if request.url.path.startswith("/v1"):
                auth = request.headers.get("Authorization", "")
                x_key = request.headers.get("X-Api-Key", "")
                if auth == f"Bearer {_api_key}" or x_key == _api_key:
                    return await call_next(request)
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        gc.sweep_now()
        return {
            "status": "ok",
            "model": server_settings.model_id,
            "model_dir": str(server_settings.model_dir),
            "shared_dir": str(server_settings.shared_dir),
            "jobs_dir": str(server_settings.jobs_dir),
            "voices_dir": str(server_settings.voices_dir),
            "cached_voices": voices.count(),
            "mode": server_settings.mode,
            "text_frontend": server_settings.text_frontend,
            "fix_question_intonation": server_settings.fix_question_intonation,
            "speed": server_settings.speed,
            "fp16": server_settings.fp16,
            "job_retention_hours": server_settings.job_retention_hours,
            "downloaded_job_retention_hours": server_settings.downloaded_job_retention_hours,
        }

    @app.get("/v1/voices")
    def list_voices() -> dict[str, Any]:
        cached = (
            sorted(path.stem for path in server_settings.voices_dir.glob("*.json"))
            if server_settings.voices_dir.is_dir()
            else []
        )
        shared = sorted(
            {
                path.stem
                for path in server_settings.shared_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS
            }
        )
        return {
            "model": server_settings.model_id,
            "mode": server_settings.mode,
            "cached_voices": cached,
            "shared_references": shared,
        }

    @app.post("/v1/voices", status_code=status.HTTP_202_ACCEPTED)
    def register_voice(request: RegisterVoiceRequest) -> dict[str, Any]:
        """Register a named voice from an uploaded reference. Building the zero-shot
        speaker loads the full model, so this returns immediately with
        ``status: "registering"`` and queues the work; poll ``status_url`` for the
        outcome (``ready`` / ``failed``). Synthesis then refers to the voice by
        ``voice`` id (no per-request upload needed)."""
        name = safe_voice_name(request.name)
        mode = request.mode or server_settings.mode
        reference_text = (request.reference_text or "").strip()
        if mode == "zero_shot" and not reference_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="zero_shot voices require reference_text matching the reference audio.",
            )

        existing = voices.get(name)
        if existing is not None and not request.overwrite:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Voice '{name}' already exists. Set overwrite=true to replace it.",
            )
        if existing is not None and existing.get("status") == "registering":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Voice '{name}' is already being registered.",
            )

        try:
            data = decode_audio_base64(request.reference_audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        suffix = infer_reference_suffix(request.filename, None)
        voices.clear_reference_audio(name)
        reference_path = voices.reference_path(name, suffix)
        reference_path.write_bytes(data)

        now = utcnow_iso()
        voices.put(
            name,
            {
                "voice": name,
                "status": "registering",
                "error": None,
                "mode": mode,
                "reference_audio": str(reference_path),
                "reference_text": reference_text,
                "reference_text_present": bool(reference_text),
                "created_at": now,
                "updated_at": now,
            },
        )
        registrar.submit(
            voice_id=name,
            reference_audio=reference_path,
            reference_text=reference_text or None,
            mode=mode,
        )

        return {
            "name": name,
            "status": "registering",
            "mode": mode,
            "status_url": f"/v1/voices/{name}",
            "created": existing is None,
        }

    @app.get("/v1/voices/{name}")
    def get_voice(name: str) -> dict[str, Any]:
        profile = voices.get(safe_voice_name(name))
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")
        return profile

    @app.delete("/v1/voices/{name}")
    def delete_voice(name: str) -> dict[str, Any]:
        safe = safe_voice_name(name)
        deleted = voices.delete(safe)
        worker.forget_voice(safe)
        return {"name": safe, "deleted": deleted}

    @app.post("/v1/unload")
    def unload() -> dict[str, Any]:
        """Free the resident CosyVoice model so memory drops back down. Clients
        call this at the end of a batch; the next request reloads it lazily."""
        return {"unloaded": worker.unload()}

    @app.post("/v1/tts/jobs", status_code=status.HTTP_202_ACCEPTED)
    def create_tts_job(request: CreateTTSJobRequest) -> dict[str, Any]:
        gc.sweep_now()
        if request.model != server_settings.model_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only model '{server_settings.model_id}' is currently supported.",
            )

        if request.response_format.lower() not in SUPPORTED_JOB_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only response_format='wav' is currently supported.",
            )

        normalized_request = CreateTTSJobRequest(
            input=request.input.strip(),
            model=request.model,
            voice=(request.voice or "reference").strip() or "reference",
            response_format=request.response_format.lower(),
            mode=resolve_effective_mode(
                request.mode or server_settings.mode,
                text=request.input.strip(),
                instruct_text=request.instruct_text,
                instructions=request.instructions,
                fix_question_intonation=_pick_override(
                    request.fix_question_intonation,
                    server_settings.fix_question_intonation,
                ),
            ),
            text_frontend=_pick_override(request.text_frontend, server_settings.text_frontend),
            speed=_pick_override(request.speed, server_settings.speed),
            stream=bool(request.stream),
            fix_question_intonation=_pick_override(
                request.fix_question_intonation,
                server_settings.fix_question_intonation,
            ),
            instructions=(request.instructions or "").strip() or None,
            instruct_text=build_runtime_instruction_text(
                text=request.input.strip(),
                instruct_text=request.instruct_text,
                instructions=request.instructions,
                fix_question_intonation=_pick_override(
                    request.fix_question_intonation,
                    server_settings.fix_question_intonation,
                ),
            ),
            reference_audio_base64=request.reference_audio_base64,
            reference_audio_filename=request.reference_audio_filename,
            reference_text=(request.reference_text or "").strip() or None,
            force_rebuild_voice=bool(request.force_rebuild_voice),
            metadata=request.metadata,
        )

        if not normalized_request.input:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="input must not be empty.")

        try:
            job = store.create_job(normalized_request)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        worker.submit(job["id"])
        return job

    @app.post("/v1/audio/speech")
    def create_audio_speech(request: CreateSpeechRequest):
        gc.sweep_now()
        if request.model != server_settings.model_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only model '{server_settings.model_id}' is currently supported.",
            )

        response_format = request.response_format.lower()
        supported_formats = SUPPORTED_DIRECT_RESPONSE_FORMATS | DIRECT_RESPONSE_FORMATS
        if response_format not in supported_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supported response_format values: " + ", ".join(sorted(supported_formats)) + ".",
            )
        if request.stream and response_format not in SUPPORTED_DIRECT_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Streaming supports only response_format='wav' or response_format='pcm'.",
            )

        payload = {
            "input": request.input.strip(),
            "model": request.model,
            "voice": (request.voice or "reference").strip() or "reference",
            "response_format": response_format,
            "mode": resolve_effective_mode(
                request.mode or server_settings.mode,
                text=request.input.strip(),
                instruct_text=request.instruct_text,
                instructions=request.instructions,
                fix_question_intonation=_pick_override(
                    request.fix_question_intonation,
                    server_settings.fix_question_intonation,
                ),
            ),
            "text_frontend": _pick_override(request.text_frontend, server_settings.text_frontend),
            "fix_question_intonation": _pick_override(
                request.fix_question_intonation,
                server_settings.fix_question_intonation,
            ),
            "speed": _pick_override(request.speed, server_settings.speed),
            "stream": bool(request.stream),
            "instructions": (request.instructions or "").strip() or None,
            "instruct_text": build_runtime_instruction_text(
                text=request.input.strip(),
                instruct_text=request.instruct_text,
                instructions=request.instructions,
                fix_question_intonation=_pick_override(
                    request.fix_question_intonation,
                    server_settings.fix_question_intonation,
                ),
            ),
            "reference_text": (request.reference_text or "").strip() or None,
            "force_rebuild_voice": bool(request.force_rebuild_voice),
            "metadata": request.metadata or {},
            "reference_audio_uploaded": bool(request.reference_audio_base64),
            "reference_audio_filename": request.reference_audio_filename,
        }
        if not payload["input"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="input must not be empty.")

        def decode_uploaded_reference(temp_dir: Path) -> Path | None:
            if not request.reference_audio_base64:
                return None
            return decode_reference_audio_to_file(
                encoded=request.reference_audio_base64,
                filename=request.reference_audio_filename,
                job_dir=temp_dir,
            )

        if request.stream:
            def audio_stream():
                with tempfile.TemporaryDirectory(prefix="cosyvoice-direct-") as temp_dir_raw:
                    uploaded_reference = decode_uploaded_reference(Path(temp_dir_raw))
                    try:
                        yield from worker.stream_direct_audio(
                            payload,
                            response_format=response_format,
                            uploaded_reference=uploaded_reference,
                        )
                    except Exception:
                        logger.exception("Direct streaming speech synthesis failed.")
                        raise

            return StreamingResponse(
                audio_stream(),
                media_type=audio_media_type(response_format),
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        with tempfile.TemporaryDirectory(prefix="cosyvoice-direct-") as temp_dir_raw:
            uploaded_reference = decode_uploaded_reference(Path(temp_dir_raw))
            try:
                if response_format in SUPPORTED_DIRECT_RESPONSE_FORMATS:
                    content = worker.synthesize_direct_bytes(
                        payload,
                        response_format=response_format,
                        uploaded_reference=uploaded_reference,
                    )
                    media_type = audio_media_type(response_format)
                    filename = f"speech.{response_format}"
                else:
                    output_path = Path(temp_dir_raw) / DEFAULT_JOB_OUTPUT_NAME
                    worker.synthesize_request(
                        request,
                        output_path=output_path,
                        uploaded_reference=uploaded_reference,
                    )
                    content, media_type, filename = encode_output(output_path, response_format)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Voice": str(payload["voice"]),
            },
        )

    @app.get("/v1/tts/jobs/{job_id}")
    def get_tts_job(job_id: str) -> dict[str, Any]:
        gc.sweep_now()
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        return store.public_view(job)

    @app.get("/v1/tts/jobs/{job_id}/audio")
    def get_tts_job_audio(job_id: str) -> FileResponse:
        gc.sweep_now()
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        audio_path = store.audio_path(job_id)
        if audio_path.is_file():
            store.mark_downloaded(job_id)
            return FileResponse(path=audio_path, media_type="audio/wav", filename=f"{job_id}.wav")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Audio is not ready yet.",
                "status": job["status"],
                "error": job.get("error"),
            },
        )

    return app


def decode_reference_audio_to_file(encoded: str, filename: str | None, job_dir: Path) -> Path:
    mime_type, payload = split_data_uri(encoded)
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("reference_audio_base64 is not valid base64.") from exc
    suffix = infer_reference_suffix(filename, mime_type)
    reference_path = job_dir / f"reference{suffix}"
    reference_path.write_bytes(decoded)
    return reference_path


def decode_audio_base64(value: str) -> bytes:
    """Decode a base64 / data-URI audio payload to raw bytes."""
    _, payload = split_data_uri(value)
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("reference_audio_base64 is not valid base64.") from exc


def safe_voice_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in (name or "")).strip("_")
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid voice name.")
    return cleaned


def split_data_uri(value: str) -> tuple[str | None, str]:
    normalized = value.strip()
    if normalized.startswith("data:"):
        if "," not in normalized:
            raise ValueError("Invalid data URI for reference audio.")
        header, payload = normalized.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or None
        return mime_type, payload.strip()
    return None, normalized


def infer_reference_suffix(filename: str | None, mime_type: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    if mime_type and mime_type in AUDIO_MIME_TO_EXT:
        return AUDIO_MIME_TO_EXT[mime_type]
    return ".wav"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def log_line(handle, message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    handle.write(f"[{timestamp}] {message}\n")
    handle.flush()


def route_python_logging_to_stream(stream: Any) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = stream


def _pick_override(value: Any, default: Any) -> Any:
    return default if value is None else value


def build_request_payload(request: "CreateTTSJobRequest") -> dict[str, Any]:
    """Flatten a request model into the dict shape ``_prepare_voice`` and
    ``_build_job_synthesis_options`` consume (the same keys ``create_job`` writes
    to request.json), so the polling worker and the sync endpoint share one path."""
    return {
        "input": request.input,
        "model": request.model,
        "voice": request.voice,
        "response_format": request.response_format,
        "mode": request.mode,
        "text_frontend": request.text_frontend,
        "speed": request.speed,
        "stream": request.stream,
        "fix_question_intonation": request.fix_question_intonation,
        "instructions": request.instructions,
        "instruct_text": request.instruct_text,
        "reference_text": request.reference_text,
        "force_rebuild_voice": request.force_rebuild_voice,
        "metadata": request.metadata or {},
    }


# CosyVoice renders WAV; everything else is an ffmpeg transcode. Keys are the
# response_format values the synchronous /v1/audio/speech endpoint accepts.
# Value: (ffmpeg args, mime, download filename). wav is handled separately.
_FFMPEG_OUTPUTS: dict[str, tuple[list[str], str, str]] = {
    "mp3": (["-codec:a", "libmp3lame", "-q:a", "4", "-f", "mp3"], "audio/mpeg", "speech.mp3"),
    "opus": (["-codec:a", "libopus", "-b:a", "48k", "-f", "ogg"], "audio/ogg", "speech.opus"),
    "ogg": (["-codec:a", "libvorbis", "-q:a", "4", "-f", "ogg"], "audio/ogg", "speech.ogg"),
}
DIRECT_RESPONSE_FORMATS = {"wav", *_FFMPEG_OUTPUTS}


def encode_output(wav_path: Path, response_format: str | None) -> tuple[bytes, str, str]:
    """Return the rendered WAV as-is, or transcode it (mp3/opus/ogg) via ffmpeg.
    Returns (bytes, media_type, download_filename)."""
    fmt = (response_format or "wav").lower()
    spec = _FFMPEG_OUTPUTS.get(fmt)
    if spec is None:
        return wav_path.read_bytes(), "audio/wav", "speech.wav"
    encoder_args, media_type, filename = spec
    return ffmpeg_transcode(wav_path, encoder_args, fmt), media_type, filename


def ffmpeg_transcode(wav_path: Path, encoder_args: list[str], fmt: str) -> bytes:
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path), *encoder_args, "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:500]
        raise RuntimeError(f"ffmpeg wav->{fmt} transcode failed: {detail or 'no output'}")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False, log_level="warning")
    return 0


app = create_app(parse_args([]))


if __name__ == "__main__":
    raise SystemExit(main())
