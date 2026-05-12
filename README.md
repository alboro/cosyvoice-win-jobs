# cosyvoice-win-jobs

Minimal polling jobs server for **CosyVoice2/CosyVoice3** on a Windows host.

This repo is deliberately narrow. It is meant for one job:

- stable long-form voice cloning for audiobook-style pipelines
- Windows-hosted deployment
- shared local voice references
- async jobs API with polling
- OpenAI-style direct speech endpoint with optional streaming

The key design choice is different from the Qwen experiment:

- do **not** recompute the speaker embedding for every request when possible
- build a cached `zero_shot_spk_id` once
- reuse that cached speaker across requests for better timbre consistency

That mirrors the official CosyVoice2 examples, where a zero-shot speaker can be added once and reused later.

## Why this repo exists

For audiobook work, per-request quality is not enough. We also need:

- consistent timbre from chunk to chunk
- predictable async jobs
- an API that a separate orchestrator can poll

Official CosyVoice already provides the core inference logic and an official FastAPI runtime. This wrapper keeps the official inference path, but adds a simpler Windows-hosted jobs interface inspired by `xtts-win-jobs`.

## Official CosyVoice notes that shaped this wrapper

According to the official CosyVoice repo:

- CosyVoice 2 supports Russian
- CosyVoice 2 supports zero-shot and cross-lingual voice cloning
- CosyVoice provides FastAPI and gRPC deployment examples
- the official examples show saving a zero-shot speaker and reusing it later with `zero_shot_spk_id`
- the official examples also note that for CosyVoice2 demo-style output you may want `text_frontend=False`

For this reason this wrapper defaults to:

- `mode=zero_shot`
- `text_frontend=false`
- `speed=1.0`
- voice cache reuse by `voice`

## Recommendation for your hardware

For a Windows machine with **RTX 3070 Ti 8GB**, the most realistic first deployment is:

- Windows host
- WSL2 Ubuntu recommended for the actual model runtime
- `Fun-CosyVoice3-0.5B` for Russian audiobook work
- `CosyVoice2-0.5B` as the older fallback
- `fp16=true`
- `load_vllm=false`
- `load_trt=false`
- `load_jit=false`

Native Windows may work, but I would treat it as the experimental path. WSL2 is the safer operational target.

Current native Windows smoke status:

- tested with local Miniforge under `.conda/miniforge`
- tested with `FunAudioLLM/CosyVoice2-0.5B`
- tested with `FunAudioLLM/Fun-CosyVoice3-0.5B`
- tested with CUDA on RTX 3070 Ti
- CLI zero-shot synthesis works with UTF-8 Russian text files
- polling jobs API works on `127.0.0.1:8040`

## Features

- polling jobs API: `POST /v1/tts/jobs`, `GET /v1/tts/jobs/{id}`, `GET /v1/tts/jobs/{id}/audio`
- direct OpenAI-style synthesis: `POST /v1/audio/speech` (regular, PCM, or streaming response)
- voice management: `GET /v1/voices` (list), `POST /v1/voices` (register a named voice from an uploaded reference — async, returns `202` and builds the speaker in the background), `GET /v1/voices/{name}` (poll status), `DELETE /v1/voices/{name}` (remove)
- model unload: `POST /v1/unload` (free the resident model between batches)
- `response_format` transcoding to `mp3` / `opus` / `ogg` on `/v1/audio/speech` (via ffmpeg)
- shared voice discovery from `shared/`
- sidecar prompt text discovery for zero-shot cloning
- zero-shot speaker caching by `voice` id
- persisted job metadata under `.data/jobs/`
- persisted voice metadata under `.data/voices/`
- simple local CLI for direct synthesis

## Shared voice layout

Put reference bundles into `shared/` like this:

- `reference_long.wav`
- `reference_long.txt`

Or:

- `anna_take2.flac`
- `anna_take2.txt`

When you submit a job with `"voice": "reference_long"`, the server will:

1. find the newest matching audio file
2. find the matching sidecar text
3. create or refresh the zero-shot speaker cache
4. synthesize future chunks using the cached speaker id

That cache-first behavior is the main quality lever for chunk-to-chunk consistency.

## API

### Create job

```json
POST /v1/tts/jobs
{
  "input": "Это тестовая фраза.",
  "model": "Fun-CosyVoice3-0.5B",
  "voice": "reference_long",
  "response_format": "wav",
  "mode": "zero_shot",
  "text_frontend": false,
  "fix_question_intonation": true,
  "speed": 1.0
}
```

### Create job with uploaded reference

```json
POST /v1/tts/jobs
{
  "input": "Это тестовая фраза.",
  "model": "Fun-CosyVoice3-0.5B",
  "voice": "reference_long",
  "response_format": "wav",
  "mode": "zero_shot",
  "text_frontend": false,
  "fix_question_intonation": true,
  "speed": 1.0,
  "reference_audio_base64": "<base64-or-data-uri>",
  "reference_audio_filename": "reference.wav",
  "reference_text": "Точный текст референса."
}
```

### Synchronous synthesis (no polling)

Renders and returns the audio in a single request. Accepts the same body as a
job, plus `response_format` of `wav` (default), `mp3`, `opus`, or `ogg`
(non-wav formats are transcoded with ffmpeg). Shares the resident model and
zero-shot speaker cache with the jobs worker.

```cmd
curl -X POST http://127.0.0.1:8040/v1/audio/speech ^
  -H "Content-Type: application/json" ^
  -d "{\"input\":\"Это тестовая фраза.\",\"voice\":\"reference_long\",\"response_format\":\"mp3\"}" ^
  --output result.mp3
```

### List voices

```cmd
curl http://127.0.0.1:8040/v1/voices
```

Returns the cached voice ids (`.data/voices/`) and the reference prefixes
discovered under `shared/`.

### Register a new voice

Upload a reference clip once (base64 or data-URI) plus its transcript. Building
the zero-shot speaker loads the full model, so the call returns **immediately**
with `status: "registering"` and does the heavy work in the background — the
client does not wait. After it finishes, synthesis can refer to the voice by
`voice` id alone (no per-request upload). `reference_text` is required for
`zero_shot`; set `overwrite: true` to replace an existing voice.

```json
POST /v1/voices            -> 202 Accepted
{
  "name": "anna",
  "reference_audio_base64": "<base64-or-data-uri>",
  "filename": "anna.wav",
  "reference_text": "Точный текст референса.",
  "mode": "zero_shot",
  "overwrite": false
}
```

Response:

```json
{ "name": "anna", "status": "registering", "mode": "zero_shot", "status_url": "/v1/voices/anna", "created": true }
```

### Poll voice status

Poll until `status` is `ready` (usable) or `failed` (see `error`).

```cmd
curl http://127.0.0.1:8040/v1/voices/anna
```

```json
{ "voice": "anna", "status": "ready", "reference_audio": "...", "reference_text_present": true }
```

### Delete a voice

```cmd
curl -X DELETE http://127.0.0.1:8040/v1/voices/anna
```

### Unload the model

Frees the resident CosyVoice model so memory drops back down between batches;
the next request reloads it lazily.

```cmd
curl -X POST http://127.0.0.1:8040/v1/unload
```
### Direct OpenAI-style speech

```json
POST /v1/audio/speech
{
  "input": "Это тестовая фраза.",
  "model": "Fun-CosyVoice3-0.5B",
  "voice": "reference_long",
  "response_format": "wav",
  "mode": "zero_shot",
  "text_frontend": false,
  "fix_question_intonation": true,
  "speed": 1.0
}
```

This endpoint returns audio bytes directly. It supports:

- `response_format="wav"` for regular clients
- `response_format="pcm"` for raw signed 16-bit PCM
- `stream=true` for chunked transfer as CosyVoice yields audio segments

The polling jobs API still only stores and returns WAV files.

`instruct2` is available only when requested explicitly with `"mode":"instruct2"`.
It is not enabled automatically for `zero_shot` because CosyVoice can leak the
instruction text into generated speech.

`fix_question_intonation=true` only appends the question-intonation helper to
explicit `instruct2` requests. Plain `zero_shot` requests ignore instructions.

### Poll status

```cmd
curl http://127.0.0.1:8040/v1/tts/jobs/<job_id>
```

### Download audio

```cmd
curl http://127.0.0.1:8040/v1/tts/jobs/<job_id>/audio --output result.wav
```

## epub_to_audiobook integration

This project is intentionally compatible with the polling style already used in `epub_to_audiobook`.

Typical settings on that side would look like:

```ini
[tts]
tts = openai
voice_name = reference_long
output_format = wav
model_name = Fun-CosyVoice3-0.5B
speed = 1.0

[tts.openai]
openai_base_url = http://desktop-6l3hnor.local:8040
openai_enable_polling = true
openai_submit_url = /v1/tts/jobs
openai_status_url_template = /v1/tts/jobs/{job_id}
openai_download_url_template = /v1/tts/jobs/{job_id}/audio
openai_job_id_path = id
openai_job_status_path = status
openai_poll_interval = 5
openai_poll_timeout = 14400
openai_submit_omit_fields = instructions
openai_submit_extra_fields = {"mode":"zero_shot","text_frontend":false}
```

If the voice cache already exists on the server, `voice_name` alone is enough. If not, you first seed the cache with a shared `voice.wav + voice.txt` pair or a request that includes uploaded reference audio and `reference_text`.

For CosyVoice3 the wrapper automatically adds the required `<|endofprompt|>` marker to prompt/instruction text before calling the official runtime.

If you send `instructions` while `mode=zero_shot`, they are kept in job metadata
but are not passed to the model. Use `mode=instruct2` only for short experiments,
not for production audiobook generation.

## Installation idea

This wrapper expects the official CosyVoice codebase to be available under `vendor/CosyVoice/`.

The included bootstrap script is designed to:

1. clone the official repo
2. create a dedicated conda env with Python 3.10
3. install official CosyVoice requirements
4. install this wrapper in editable mode

On Windows the wrapper scripts first look for a repo-local conda executable at `.conda/miniforge/Scripts/conda.exe`, then fall back to `conda` from `PATH`.

If ModelScope downloads fail because of local SSL certificates, downloading from Hugging Face is a valid fallback:

- `FunAudioLLM/Fun-CosyVoice3-0.5B` into `pretrained_models/Fun-CosyVoice3-0.5B`
- `FunAudioLLM/CosyVoice2-0.5B` into `pretrained_models/CosyVoice2-0.5B`

Model weights are not committed. Put them under:

- `pretrained_models/Fun-CosyVoice3-0.5B`
- `pretrained_models/CosyVoice2-0.5B`

## Quick Start

```cmd
scripts\bootstrap_windows.cmd
cosyvoice-win-server.cmd --host 127.0.0.1 --port 8040 --model-id Fun-CosyVoice3-0.5B --model-dir pretrained_models\Fun-CosyVoice3-0.5B
```

Then place a reference bundle into `shared/` and submit a job.

For Russian CLI smoke tests, prefer `--file` with a UTF-8 text file instead of inline Cyrillic through `cmd.exe`:

```cmd
cosyvoice-win.cmd --file input\ru_smoke.txt ru_smoke.wav reference_dictor_short --overwrite
```

## Caveats

- This wrapper does not try to be a universal OpenAI TTS clone.
- `response_format` is currently `wav` only.
- `mode=zero_shot` is the primary path for audiobook use.
- `cross_lingual` is supported as an alternative mode.
- `instruct2` is intentionally not wired in this first version because cached-speaker reuse around instruct mode is less predictable than zero-shot reuse.

## Sources

- Official CosyVoice repo: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- Official example usage showing `add_zero_shot_spk` and `save_spkinfo`
- Official deployment docs describing FastAPI/grpc runtime

## License

MIT for the wrapper code in this repository.

CosyVoice itself and its models keep their own licenses.
