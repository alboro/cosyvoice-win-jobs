from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import warnings
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_DIR = PROJECT_ROOT / "shared"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_MODEL_ID = "Fun-CosyVoice3-0.5B"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / DEFAULT_MODEL_ID
DEFAULT_REFERENCE_PREFIX = "reference"
DEFAULT_MODE = "zero_shot"
DEFAULT_SPEED = 1.0
DEFAULT_TEXT_FRONTEND = False
DEFAULT_FP16 = True
DEFAULT_STREAM = False
DEFAULT_FIX_QUESTION_INTONATION = False
COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
COSYVOICE3_PROMPT_MARKER = "<|endofprompt|>"
QUESTION_INTONATION_INSTRUCTION = (
    "Read only the target text with clear question intonation. Do not read the instruction aloud."
)
QUESTION_TRAILING_CLOSERS = "\"'`»«“”„‟’‘)]}>】』」"
QUESTION_ENDING_PATTERN = re.compile(
    r"\?+(?:[" + re.escape(QUESTION_TRAILING_CLOSERS) + r"]+)?(?:[.,;:!…]+)?\s*$"
)

REFERENCE_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
}
REFERENCE_TEXT_EXTENSIONS = (".txt", ".lab")

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(slots=True)
class ResolvedReference:
    audio_path: Path | None
    prompt_text: str | None
    reference_source_label: str
    prompt_source_label: str


@dataclass(slots=True)
class ResolvedRun:
    text: str
    output_path: Path
    voice: str
    reference: ResolvedReference
    text_source_label: str


@dataclass(slots=True)
class CosyVoiceModelOptions:
    model_dir: Path = DEFAULT_MODEL_DIR
    fp16: bool = DEFAULT_FP16
    load_jit: bool = False
    load_trt: bool = False
    load_vllm: bool = False
    text_frontend: bool = DEFAULT_TEXT_FRONTEND


@dataclass(slots=True)
class CosyVoiceSynthesisOptions:
    mode: str = DEFAULT_MODE
    requested_mode: str | None = None
    mode_reason: str | None = None
    text_frontend: bool = DEFAULT_TEXT_FRONTEND
    speed: float = DEFAULT_SPEED
    stream: bool = DEFAULT_STREAM
    fix_question_intonation: bool = DEFAULT_FIX_QUESTION_INTONATION
    instruct_text: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosyvoice-win",
        description="Windows-hosted CosyVoice2 CLI for stable zero-shot cloning.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Default mode: INPUT_FILE OUTPUT [VOICE]. "
            "With --text: TEXT OUTPUT [VOICE]. "
            "With --file: OUTPUT [VOICE]."
        ),
    )
    parser.add_argument("--file", dest="input_file", help="Read text from a UTF-8 file.")
    parser.add_argument("--text", action="store_true", help="Treat the first positional input as literal text.")
    parser.add_argument("--shared-dir", default=str(DEFAULT_SHARED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--voice", default=DEFAULT_REFERENCE_PREFIX, help="Voice id / shared reference prefix.")
    parser.add_argument(
        "--reference-prefix",
        default=DEFAULT_REFERENCE_PREFIX,
        help="Default shared reference prefix when no voice is provided.",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Explicit audio reference file. If omitted, shared voice lookup is used.",
    )
    parser.add_argument("--reference-text", default=None, help="Inline transcript for the reference audio.")
    parser.add_argument("--reference-text-file", default=None, help="File containing the reference transcript.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--mode", choices=("zero_shot", "cross_lingual", "instruct2"), default=DEFAULT_MODE)
    parser.add_argument(
        "--instructions",
        default=None,
        help="Non-spoken instruction text. In zero_shot mode this auto-promotes the request to instruct2.",
    )
    parser.add_argument("--instruct-text", default=None, help="Instruction text for CosyVoice2 instruct2 mode.")
    parser.add_argument("--text-frontend", choices=("on", "off"), default="off")
    parser.add_argument(
        "--fix-question-intonation",
        choices=("on", "off"),
        default="on" if DEFAULT_FIX_QUESTION_INTONATION else "off",
        help="Auto-promote trailing questions to instruct2 and add a hidden question-intonation instruction.",
    )
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--fp16", choices=("on", "off"), default="on" if DEFAULT_FP16 else "off")
    parser.add_argument("--load-jit", choices=("on", "off"), default="off")
    parser.add_argument("--load-trt", choices=("on", "off"), default="off")
    parser.add_argument("--load-vllm", choices=("on", "off"), default="off")
    parser.add_argument("--stream", choices=("on", "off"), default="off")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    return parser


def parse_on_off(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    raise ValueError(f"Expected 'on' or 'off', got: {value}")


def resolve_dir(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_path(path_value: str | Path, output_dir: Path, overwrite: bool) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (output_dir / path).resolve()
    else:
        path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    return path


def resolve_text_file(path_value: str, shared_dir: Path) -> Path:
    raw_path = Path(path_value).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else [Path.cwd() / raw_path, shared_dir / raw_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Input text file not found: {path_value}")


def read_text_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input text file is empty: {path}")
    return text


def normalize_instruction_text(
    instruct_text: str | None = None,
    instructions: str | None = None,
) -> str | None:
    value = (instruct_text or instructions or "").strip()
    return value or None


def ends_with_question_intonation_trigger(text: str) -> bool:
    return bool(QUESTION_ENDING_PATTERN.search(text.rstrip()))


def build_runtime_instruction_text(
    *,
    text: str,
    instruct_text: str | None = None,
    instructions: str | None = None,
    fix_question_intonation: bool = DEFAULT_FIX_QUESTION_INTONATION,
) -> str | None:
    parts: list[str] = []
    base_instruction = normalize_instruction_text(instruct_text, instructions)
    if base_instruction:
        parts.append(base_instruction)
    if fix_question_intonation and ends_with_question_intonation_trigger(text):
        if not base_instruction or QUESTION_INTONATION_INSTRUCTION not in base_instruction:
            parts.append(QUESTION_INTONATION_INSTRUCTION)
    if not parts:
        return None
    return "\n".join(parts)


def resolve_mode_reason(
    *,
    text: str,
    instruct_text: str | None = None,
    instructions: str | None = None,
    fix_question_intonation: bool = DEFAULT_FIX_QUESTION_INTONATION,
) -> str | None:
    has_explicit_instruction = bool(normalize_instruction_text(instruct_text, instructions))
    has_question_trigger = fix_question_intonation and ends_with_question_intonation_trigger(text)
    if has_explicit_instruction and has_question_trigger:
        return "instructions were provided and the input ends with a question mark"
    if has_explicit_instruction:
        return "instructions were provided"
    if has_question_trigger:
        return "the input ends with a question mark"
    return None


def resolve_effective_mode(
    mode: str,
    *,
    text: str = "",
    instruct_text: str | None = None,
    instructions: str | None = None,
    fix_question_intonation: bool = DEFAULT_FIX_QUESTION_INTONATION,
) -> str:
    runtime_instruction = build_runtime_instruction_text(
        text=text,
        instruct_text=instruct_text,
        instructions=instructions,
        fix_question_intonation=fix_question_intonation,
    )
    if runtime_instruction and mode == "zero_shot":
        return "instruct2"
    return mode


def is_supported_reference_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS


def find_reference_audio_in_shared(shared_dir: Path, prefix: str) -> Path:
    prefix_normalized = prefix.lower()
    candidates = [
        candidate.resolve()
        for candidate in shared_dir.rglob("*")
        if is_supported_reference_audio(candidate) and candidate.stem.lower().startswith(prefix_normalized)
    ]
    if not candidates:
        raise FileNotFoundError(f"No reference audio found in {shared_dir} for prefix '{prefix}'.")
    return max(candidates, key=lambda item: (item.stat().st_mtime, item.name.lower()))


def find_reference_text_for_audio(audio_path: Path, prefix: str) -> tuple[str | None, str]:
    direct_candidates = [audio_path.with_suffix(ext) for ext in REFERENCE_TEXT_EXTENSIONS]
    fallback_candidates = [audio_path.parent / f"{prefix}{ext}" for ext in REFERENCE_TEXT_EXTENSIONS]
    for candidate in [*direct_candidates, *fallback_candidates]:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text, str(candidate)
    return None, "<none>"


def resolve_reference(
    *,
    shared_dir: Path,
    voice: str,
    reference_prefix: str,
    reference_file: str | None,
    reference_text: str | None,
    reference_text_file: str | None,
) -> ResolvedReference:
    prefix = voice or reference_prefix
    prompt_text = reference_text.strip() if reference_text else None
    prompt_source_label = "<inline>"

    if reference_text_file:
        text_path = resolve_text_file(reference_text_file, shared_dir)
        prompt_text = read_text_file(text_path)
        prompt_source_label = str(text_path)

    if reference_file:
        audio_path = resolve_text_file(reference_file, shared_dir)
        if not is_supported_reference_audio(audio_path):
            raise ValueError(f"Unsupported reference audio file: {audio_path}")
        if prompt_text is None:
            prompt_text, prompt_source_label = find_reference_text_for_audio(audio_path, audio_path.stem)
        return ResolvedReference(
            audio_path=audio_path,
            prompt_text=prompt_text,
            reference_source_label="explicit file",
            prompt_source_label=prompt_source_label,
        )

    try:
        audio_path = find_reference_audio_in_shared(shared_dir, prefix)
    except FileNotFoundError:
        return ResolvedReference(
            audio_path=None,
            prompt_text=prompt_text,
            reference_source_label="cache only",
            prompt_source_label=prompt_source_label if prompt_text else "<none>",
        )

    if prompt_text is None:
        prompt_text, prompt_source_label = find_reference_text_for_audio(audio_path, prefix)

    return ResolvedReference(
        audio_path=audio_path,
        prompt_text=prompt_text,
        reference_source_label=f"newest shared match for prefix '{prefix}'",
        prompt_source_label=prompt_source_label,
    )


def resolve_cli_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    shared_dir: Path,
    output_dir: Path,
) -> ResolvedRun:
    if args.input_file and args.text:
        parser.error("Use either --file or --text, not both.")

    if args.input_file:
        if len(args.inputs) not in (1, 2):
            parser.error("With --file you must pass: OUTPUT [VOICE]")
        input_path = resolve_text_file(args.input_file, shared_dir)
        voice = args.inputs[1] if len(args.inputs) == 2 else args.voice
        return ResolvedRun(
            text=read_text_file(input_path),
            output_path=resolve_output_path(args.inputs[0], output_dir, args.overwrite),
            voice=voice,
            reference=resolve_reference(
                shared_dir=shared_dir,
                voice=voice,
                reference_prefix=args.reference_prefix,
                reference_file=args.reference_file,
                reference_text=args.reference_text,
                reference_text_file=args.reference_text_file,
            ),
            text_source_label=str(input_path),
        )

    if args.text:
        if len(args.inputs) not in (2, 3):
            parser.error("With --text you must pass: TEXT OUTPUT [VOICE]")
        text = args.inputs[0].strip()
        if not text:
            parser.error("Text must not be empty.")
        voice = args.inputs[2] if len(args.inputs) == 3 else args.voice
        return ResolvedRun(
            text=text,
            output_path=resolve_output_path(args.inputs[1], output_dir, args.overwrite),
            voice=voice,
            reference=resolve_reference(
                shared_dir=shared_dir,
                voice=voice,
                reference_prefix=args.reference_prefix,
                reference_file=args.reference_file,
                reference_text=args.reference_text,
                reference_text_file=args.reference_text_file,
            ),
            text_source_label="<inline text>",
        )

    if len(args.inputs) not in (2, 3):
        parser.error("Usage: INPUT_FILE OUTPUT [VOICE]")

    input_path = resolve_text_file(args.inputs[0], shared_dir)
    voice = args.inputs[2] if len(args.inputs) == 3 else args.voice
    return ResolvedRun(
        text=read_text_file(input_path),
        output_path=resolve_output_path(args.inputs[1], output_dir, args.overwrite),
        voice=voice,
        reference=resolve_reference(
            shared_dir=shared_dir,
            voice=voice,
            reference_prefix=args.reference_prefix,
            reference_file=args.reference_file,
            reference_text=args.reference_text,
            reference_text_file=args.reference_text_file,
        ),
        text_source_label=str(input_path),
    )


def resolve_model_dir(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def ensure_vendor_paths(project_root: Path = PROJECT_ROOT) -> None:
    vendor_root = project_root / "vendor" / "CosyVoice"
    matcha_root = vendor_root / "third_party" / "Matcha-TTS"
    for candidate in (vendor_root, matcha_root):
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def load_cosyvoice_runtime():
    ensure_vendor_paths()
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.file_utils import load_wav

    return AutoModel, load_wav


@contextmanager
def maybe_disable_text_frontend_imports(text_frontend: bool):
    if text_frontend:
        yield
        return

    real_import = builtins.__import__
    blocked_roots = {"ttsfrd", "wetext"}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_name = name.split(".", 1)[0]
        if level == 0 and root_name in blocked_roots:
            raise ImportError(f"{root_name} disabled because CosyVoice text_frontend is off")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def load_model(model_options: CosyVoiceModelOptions):
    AutoModel, _ = load_cosyvoice_runtime()
    model_kwargs = build_model_kwargs(model_options)
    with maybe_disable_text_frontend_imports(model_options.text_frontend):
        return AutoModel(**model_kwargs)


def build_model_kwargs(model_options: CosyVoiceModelOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_dir": str(model_options.model_dir),
        "fp16": model_options.fp16,
    }
    if (model_options.model_dir / "cosyvoice3.yaml").is_file():
        kwargs["load_trt"] = model_options.load_trt
        kwargs["load_vllm"] = model_options.load_vllm
    elif (model_options.model_dir / "cosyvoice.yaml").is_file():
        kwargs["load_jit"] = model_options.load_jit
        kwargs["load_trt"] = model_options.load_trt
    else:
        kwargs["load_jit"] = model_options.load_jit
        kwargs["load_trt"] = model_options.load_trt
        kwargs["load_vllm"] = model_options.load_vllm
    return kwargs


def load_prompt_audio_16k(audio_path: Path):
    # Current upstream CosyVoice APIs accept a path/file-like object and call
    # load_wav internally. Passing a preloaded tensor breaks newer checkouts.
    return str(audio_path)


def ensure_zero_shot_speaker(
    cosyvoice_model,
    *,
    voice_id: str,
    prompt_text: str,
    prompt_audio_16k,
    persist: bool = True,
) -> None:
    if not hasattr(cosyvoice_model, "add_zero_shot_spk"):
        raise RuntimeError("This CosyVoice runtime does not expose add_zero_shot_spk().")
    prompt_text = ensure_cosyvoice3_prompt_text(cosyvoice_model, prompt_text)
    ok = cosyvoice_model.add_zero_shot_spk(prompt_text, prompt_audio_16k, voice_id)
    if ok is not True:
        raise RuntimeError(f"Failed to add zero-shot speaker '{voice_id}'.")
    if persist and hasattr(cosyvoice_model, "save_spkinfo"):
        cosyvoice_model.save_spkinfo()


def iter_synthesis(
    cosyvoice_model,
    *,
    text: str,
    voice_id: str,
    reference: ResolvedReference,
    options: CosyVoiceSynthesisOptions,
):
    prompt_audio = load_prompt_audio_16k(reference.audio_path) if reference.audio_path else ""
    prompt_text = reference.prompt_text or ""

    if options.mode == "zero_shot":
        prompt_text = ensure_cosyvoice3_prompt_text(cosyvoice_model, prompt_text)
        return cosyvoice_model.inference_zero_shot(
            text,
            prompt_text,
            prompt_audio,
            zero_shot_spk_id=voice_id or "",
            stream=options.stream,
            speed=options.speed,
            text_frontend=options.text_frontend,
        )

    if options.mode == "cross_lingual":
        text = ensure_cosyvoice3_tts_text(cosyvoice_model, text)
        return cosyvoice_model.inference_cross_lingual(
            text,
            prompt_audio,
            zero_shot_spk_id=voice_id or "",
            stream=options.stream,
            speed=options.speed,
            text_frontend=options.text_frontend,
        )

    if options.mode == "instruct2":
        if not hasattr(cosyvoice_model, "inference_instruct2"):
            raise RuntimeError("This CosyVoice runtime does not expose inference_instruct2().")
        instruct_text = (options.instruct_text or "").strip()
        if not instruct_text:
            raise ValueError("instruct2 mode requires instruct_text.")
        instruct_text = ensure_cosyvoice3_instruct_text(cosyvoice_model, instruct_text)
        if not prompt_audio:
            raise ValueError("instruct2 mode requires a reference audio file.")
        return cosyvoice_model.inference_instruct2(
            text,
            instruct_text,
            prompt_audio,
            zero_shot_spk_id=voice_id or "",
            stream=options.stream,
            speed=options.speed,
            text_frontend=options.text_frontend,
        )

    raise ValueError(f"Unsupported mode: {options.mode}")


def is_cosyvoice3_model(cosyvoice_model) -> bool:
    return cosyvoice_model.__class__.__name__ == "CosyVoice3"


def ensure_cosyvoice3_prompt_text(cosyvoice_model, prompt_text: str) -> str:
    if not is_cosyvoice3_model(cosyvoice_model) or COSYVOICE3_PROMPT_MARKER in prompt_text:
        return prompt_text
    return f"{COSYVOICE3_PROMPT_PREFIX}{prompt_text}"


def ensure_cosyvoice3_tts_text(cosyvoice_model, text: str) -> str:
    if not is_cosyvoice3_model(cosyvoice_model) or COSYVOICE3_PROMPT_MARKER in text:
        return text
    return f"{COSYVOICE3_PROMPT_PREFIX}{text}"


def ensure_cosyvoice3_instruct_text(cosyvoice_model, instruct_text: str) -> str:
    if not is_cosyvoice3_model(cosyvoice_model) or COSYVOICE3_PROMPT_MARKER in instruct_text:
        return instruct_text
    return f"You are a helpful assistant. {instruct_text}{COSYVOICE3_PROMPT_MARKER}"


def save_outputs_to_wav(outputs, sample_rate: int, output_path: Path) -> None:
    import torch
    import torchaudio

    waveforms = []
    for output in outputs:
        speech = output["tts_speech"]
        if getattr(speech, "dim", lambda: 0)() == 1:
            speech = speech.unsqueeze(0)
        waveforms.append(speech.detach().cpu())

    if not waveforms:
        raise RuntimeError("CosyVoice returned no audio segments.")

    merged = waveforms[0] if len(waveforms) == 1 else torch.cat(waveforms, dim=1)
    torchaudio.save(str(output_path), merged, sample_rate)


def synthesize_to_file(
    cosyvoice_model,
    *,
    text: str,
    voice_id: str,
    reference: ResolvedReference,
    output_path: Path,
    options: CosyVoiceSynthesisOptions,
) -> int:
    if options.mode == "zero_shot" and not reference.audio_path and not voice_id:
        raise ValueError("zero_shot mode requires either a cached voice id or a reference bundle.")
    if options.mode == "zero_shot" and reference.audio_path and not (reference.prompt_text or "").strip():
        raise ValueError("zero_shot mode requires reference_text / sidecar transcript for the reference audio.")

    outputs = list(
        iter_synthesis(
            cosyvoice_model,
            text=text,
            voice_id=voice_id,
            reference=reference,
            options=options,
        )
    )
    save_outputs_to_wav(outputs, cosyvoice_model.sample_rate, output_path)
    return len(outputs)


def estimate_audio_duration_seconds(text: str, speed: float = DEFAULT_SPEED) -> float:
    words = len(text.split())
    wpm = 145.0 * max(speed, 0.1)
    return (words / wpm) * 60.0


def format_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours >= 1:
        parts.append(f"{int(hours)}h")
    if minutes >= 1 or hours >= 1:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{secs:.1f}s")
    return " ".join(parts)


def print_run_summary(
    *,
    started_at: datetime,
    run: ResolvedRun,
    model_options: CosyVoiceModelOptions,
    options: CosyVoiceSynthesisOptions,
) -> None:
    word_count = len(run.text.split())
    print(f"Start: {started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Input: {run.text_source_label}")
    print(f"Output: {run.output_path}")
    print(f"Voice: {run.voice}")
    print(f"Reference source: {run.reference.reference_source_label}")
    if run.reference.audio_path:
        print(f"Reference audio: {run.reference.audio_path}")
    print(f"Reference text source: {run.reference.prompt_source_label}")
    print(f"Model dir: {model_options.model_dir}")
    if options.requested_mode and options.requested_mode != options.mode:
        reason = options.mode_reason or "automatic runtime rule"
        print(f"Mode: {options.mode} (requested {options.requested_mode}; promoted because {reason})")
    else:
        print(f"Mode: {options.mode}")
    print(f"Text frontend: {'enabled' if options.text_frontend else 'disabled'}")
    print(f"Fix question intonation: {'enabled' if options.fix_question_intonation else 'disabled'}")
    print(f"Speed: {options.speed}")
    print(f"fp16: {'enabled' if model_options.fp16 else 'disabled'}")
    print(f"Text size: {len(run.text)} chars, {word_count} words")
    print(f"Estimated speech: {format_duration(estimate_audio_duration_seconds(run.text, options.speed))}")


def run_doctor(shared_dir: Path, model_dir: Path) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Shared dir: {shared_dir}")
    print(f"Model dir: {model_dir}")
    print(f"CosyVoice vendor present: {(PROJECT_ROOT / 'vendor' / 'CosyVoice').is_dir()}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch: ERROR ({exc})")

    try:
        AutoModel, _ = load_cosyvoice_runtime()
        print("cosyvoice_runtime: OK")
        del AutoModel
    except Exception as exc:
        print(f"cosyvoice_runtime: ERROR ({exc})")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    shared_dir = resolve_dir(args.shared_dir)
    output_dir = resolve_dir(args.output_dir)
    model_dir = resolve_model_dir(args.model_dir)

    if args.doctor:
        return run_doctor(shared_dir, model_dir)

    try:
        run = resolve_cli_inputs(args, parser, shared_dir, output_dir)
        text_frontend = parse_on_off(args.text_frontend)
        fix_question_intonation = parse_on_off(args.fix_question_intonation)
        instruct_text = build_runtime_instruction_text(
            text=run.text,
            instruct_text=args.instruct_text,
            instructions=args.instructions,
            fix_question_intonation=fix_question_intonation,
        )
        requested_mode = args.mode
        mode_reason = resolve_mode_reason(
            text=run.text,
            instruct_text=args.instruct_text,
            instructions=args.instructions,
            fix_question_intonation=fix_question_intonation,
        )
        effective_mode = resolve_effective_mode(
            requested_mode,
            text=run.text,
            instruct_text=args.instruct_text,
            instructions=args.instructions,
            fix_question_intonation=fix_question_intonation,
        )
        model_options = CosyVoiceModelOptions(
            model_dir=model_dir,
            fp16=parse_on_off(args.fp16),
            load_jit=parse_on_off(args.load_jit),
            load_trt=parse_on_off(args.load_trt),
            load_vllm=parse_on_off(args.load_vllm),
            text_frontend=text_frontend,
        )
        options = CosyVoiceSynthesisOptions(
            mode=effective_mode,
            requested_mode=requested_mode,
            mode_reason=mode_reason,
            text_frontend=text_frontend,
            speed=args.speed,
            stream=parse_on_off(args.stream),
            fix_question_intonation=fix_question_intonation,
            instruct_text=instruct_text,
        )

        if options.mode == "zero_shot" and run.reference.audio_path and not (run.reference.prompt_text or "").strip():
            raise ValueError("zero_shot mode requires a transcript for the reference audio.")

        started_at = datetime.now().astimezone()
        run_started = time.perf_counter()
        print_run_summary(started_at=started_at, run=run, model_options=model_options, options=options)

        model_load_started = time.perf_counter()
        cosyvoice_model = load_model(model_options)
        model_load_seconds = time.perf_counter() - model_load_started
        print(f"Model ready in {format_duration(model_load_seconds)}")

        if options.mode == "zero_shot" and run.reference.audio_path:
            prompt_audio_16k = load_prompt_audio_16k(run.reference.audio_path)
            ensure_zero_shot_speaker(
                cosyvoice_model,
                voice_id=run.voice,
                prompt_text=run.reference.prompt_text or "",
                prompt_audio_16k=prompt_audio_16k,
                persist=True,
            )
            cached_reference = ResolvedReference(
                audio_path=None,
                prompt_text=None,
                reference_source_label="cached speaker",
                prompt_source_label="<cached>",
            )
        else:
            cached_reference = run.reference

        synthesis_started = time.perf_counter()
        segment_count = synthesize_to_file(
            cosyvoice_model,
            text=run.text,
            voice_id=run.voice,
            reference=cached_reference,
            output_path=run.output_path,
            options=options,
        )
        synthesis_seconds = time.perf_counter() - synthesis_started

        wall_seconds = time.perf_counter() - run_started
        finished_at = datetime.now().astimezone()
        print(f"Saved: {run.output_path}")
        print(f"Finish: {finished_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Wall time: {format_duration(wall_seconds)}")
        print(f"Model load: {format_duration(model_load_seconds)}")
        print(f"Synthesis: {format_duration(synthesis_seconds)}")
        print(f"Segments returned: {segment_count}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
