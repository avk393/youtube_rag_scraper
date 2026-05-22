#!/usr/bin/env python3
"""
caption_frames.py - Step 4 of the YouTube -> RAG pipeline.

Sends each unique keyframe from step 3 to a vision model and records, per
frame, a verbatim transcription of any on-screen text plus a short
description of what the frame depicts. These visual descriptions are what
step 5 fuses into the spoken transcript by timestamp.

Backends (choose with --backend):

  anthropic   Claude via the Anthropic API (default; needs ANTHROPIC_API_KEY).
              Default model: Haiku 4.5 (the cheapest current Claude model).

  gemini      Google Gemini via the google-genai SDK. Reads GEMINI_API_KEY or
              GOOGLE_API_KEY from the environment. Default model: gemini-2.5-flash.

  ollama      Local Ollama server, using its native /api/chat endpoint with
              format:"json" for structured output. Default --base-url is
              http://localhost:11434. No API key needed. Pull a vision model
              first, e.g. `ollama pull qwen2.5vl`.

  openai      Any OpenAI-compatible /chat/completions endpoint. Works with
              OpenAI itself, OpenRouter, Groq, vLLM, LM Studio, llama.cpp
              server, etc. Requires --base-url and --model. API key comes
              from --api-key, --api-key-env, or $OPENAI_API_KEY.

For each data/<video_id>/ folder it reads keyframes.json + frames/*.jpg and
writes:

  data/<video_id>/visuals.json
      {
        "video_id": "...",
        "backend":  "anthropic" | "gemini" | "ollama" | "openai",
        "model":    "...",
        "frames": [
          { "id": 1, "timestamp": 12.25,
            "ocr": "verbatim on-screen text ...",
            "description": "what the frame shows ...",
            "has_text": true },
          ...
        ]
      }

Re-running is safe and resumes: frames already captioned are kept, so a run
interrupted by an API error can simply be restarted.

Usage:
  # Claude (default)
  export ANTHROPIC_API_KEY=sk-ant-...
  python caption_frames.py

  # Gemini
  export GEMINI_API_KEY=...
  python caption_frames.py --backend gemini

  # Local Ollama with a vision model
  ollama pull qwen2.5vl
  python caption_frames.py --backend ollama --model qwen2.5vl

  # OpenAI proper
  export OPENAI_API_KEY=sk-...
  python caption_frames.py --backend openai \\
      --base-url https://api.openai.com/v1 --model gpt-4o-mini

  # OpenRouter (Mistral, Qwen, anything they host)
  export OPENROUTER_API_KEY=sk-or-...
  python caption_frames.py --backend openai \\
      --base-url https://openrouter.ai/api/v1 \\
      --api-key-env OPENROUTER_API_KEY \\
      --model mistralai/mistral-small-3.1-24b-instruct

Requirements:
  pip install anthropic      (only for --backend anthropic)
  pip install google-genai   (only for --backend gemini)
  pip install requests       (only for --backend openai or ollama)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("caption")

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
MAX_TOKENS = 1500
HTTP_TIMEOUT = 120

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}

PROMPT_TEMPLATE = """\
This image is a single frame from a video{title_clause}. It will be indexed \
in a knowledge base alongside the video's spoken transcript.

Respond with ONLY a JSON object (no markdown fences, no commentary) with \
exactly these two keys:

  "text"   - a verbatim transcription of all readable on-screen text: slide
             content, code, captions, diagram labels, UI text. Use an empty
             string if the frame has no meaningful text.
  "visual" - 2 to 4 sentences describing what the frame depicts: diagrams,
             charts, UI state, physical demonstrations, scenes. Focus on
             information a reader could NOT get from the spoken words alone.
             Use an empty string if the frame is purely decorative.
"""


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def encode_image(path: Path) -> tuple[str, str]:
    """Return (base64 data, media type) for an image file."""
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    return base64.standard_b64encode(path.read_bytes()).decode("ascii"), media_type


def image_media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _title_clause(title: str) -> str:
    return f' titled "{title}"' if title else ""


def _build_prompt(title: str) -> str:
    return PROMPT_TEMPLATE.format(title_clause=_title_clause(title))


def _parse_caption(raw: str) -> dict:
    """Turn the model's raw text into {'ocr', 'description'}."""
    parsed = extract_json(raw)
    if parsed is None:
        # Couldn't parse JSON - keep the raw text rather than losing the work.
        return {"ocr": "", "description": raw.strip()}
    return {
        "ocr": str(parsed.get("text") or "").strip(),
        "description": str(parsed.get("visual") or "").strip(),
    }


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class _HTTPBackend:
    """Shared retry/HTTP plumbing for backends that talk plain JSON over HTTP."""

    _RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def _init_http(self, timeout: float) -> None:
        try:
            import requests
        except ImportError:
            sys.exit("requests is not installed. Run: pip install requests")
        self._requests = requests
        self.session = requests.Session()
        self.timeout = timeout

    def _post_with_retry(self, url: str, payload: dict, headers: dict) -> dict:
        """POST JSON with exponential backoff on transient HTTP failures."""
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.session.post(url, json=payload, headers=headers,
                                          timeout=self.timeout)
            except self._requests.RequestException as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in self._RETRY_STATUS and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"all retries exhausted: {last_error}")


class AnthropicBackend:
    """Claude via the Anthropic SDK."""

    name = "anthropic"

    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError:
            sys.exit("the anthropic SDK is not installed. Run: pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set - export it before running")
        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model

    def caption(self, image_path: Path, title: str) -> dict:
        data, media_type = encode_image(image_path)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type, "data": data}},
                    {"type": "text", "text": _build_prompt(title)},
                ],
            }],
        )
        raw = "".join(block.text for block in message.content if block.type == "text")
        return _parse_caption(raw)


class GeminiBackend:
    """Google Gemini via the google-genai SDK."""

    name = "gemini"

    def __init__(self, model: str, api_key: str | None):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            sys.exit("the google-genai SDK is not installed. Run: pip install google-genai")

        # The SDK reads GEMINI_API_KEY / GOOGLE_API_KEY automatically; only
        # pass an explicit key if the caller provided one.
        if api_key:
            self.client = genai.Client(api_key=api_key)
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            self.client = genai.Client()
        else:
            sys.exit("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set - export it before running")
        self._types = types
        self.model = model

    def caption(self, image_path: Path, title: str) -> dict:
        image_part = self._types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type=image_media_type(image_path),
        )
        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=MAX_TOKENS,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image_part, _build_prompt(title)],
            config=config,
        )
        return _parse_caption(response.text or "")


class OllamaBackend(_HTTPBackend):
    """Local Ollama server via its native /api/chat endpoint."""

    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float):
        self._init_http(timeout)
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model

    def caption(self, image_path: Path, title: str) -> dict:
        data, _ = encode_image(image_path)
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",          # native Ollama JSON-mode
            "options": {"num_predict": MAX_TOKENS},
            "messages": [{
                "role": "user",
                "content": _build_prompt(title),
                "images": [data],      # base64 strings live on the message itself
            }],
        }
        body = self._post_with_retry(self.url, payload, {"Content-Type": "application/json"})
        raw = (body.get("message") or {}).get("content") or ""
        return _parse_caption(raw)


class OpenAIBackend(_HTTPBackend):
    """Any OpenAI-compatible /chat/completions endpoint.

    Used for OpenAI itself, OpenRouter, Groq, vLLM, LM Studio, llama.cpp
    server, and Ollama's OpenAI-compatibility layer at /v1.
    """

    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float):
        self._init_http(timeout)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model = model

    def caption(self, image_path: Path, title: str) -> dict:
        data, media_type = encode_image(image_path)
        payload = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            # Honoured by OpenAI, Ollama's compat layer, and most others;
            # servers that don't recognise it generally just ignore it.
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{data}"}},
                    {"type": "text", "text": _build_prompt(title)},
                ],
            }],
        }
        body = self._post_with_retry(self.url, payload, self.headers)
        raw = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return _parse_caption(raw)


def make_backend(args: argparse.Namespace):
    """Build the chosen backend from CLI args."""
    if args.backend == "anthropic":
        return AnthropicBackend(model=args.model or DEFAULT_ANTHROPIC_MODEL)

    if args.backend == "gemini":
        return GeminiBackend(model=args.model or DEFAULT_GEMINI_MODEL,
                              api_key=args.api_key)

    if args.backend == "ollama":
        if not args.model:
            sys.exit("--model is required for --backend ollama "
                     "(e.g. --model qwen2.5vl)")
        return OllamaBackend(
            model=args.model,
            base_url=args.base_url or DEFAULT_OLLAMA_BASE_URL,
            timeout=args.timeout,
        )

    if args.backend == "openai":
        if not args.model:
            sys.exit("--model is required for --backend openai")
        if not args.base_url:
            sys.exit("--base-url is required for --backend openai "
                     "(e.g. https://api.openai.com/v1)")
        api_key = args.api_key
        if not api_key and args.api_key_env:
            api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAIBackend(model=args.model, base_url=args.base_url,
                              api_key=api_key, timeout=args.timeout)

    sys.exit(f"unknown backend: {args.backend}")


# --------------------------------------------------------------------------
# Per-video processing
# --------------------------------------------------------------------------

def video_title(video_dir: Path) -> str:
    """Best-effort lookup of a video's title from its metadata.json."""
    meta_path = video_dir / "metadata.json"
    if meta_path.exists():
        try:
            return (json.loads(meta_path.read_text()).get("title") or "")[:120]
        except (json.JSONDecodeError, OSError):
            pass
    return ""


def process_video(video_dir: Path, backend, concurrency: int, force: bool) -> str:
    """Caption every keyframe of one video. Returns ok/skipped/failed."""
    video_id = video_dir.name
    kf_path = video_dir / "keyframes.json"
    if not kf_path.exists():
        log.error("no keyframes.json in %s - run step 3 (keyframes.py) first", video_id)
        return "failed"

    keyframes = json.loads(kf_path.read_text()).get("keyframes", [])
    if not keyframes:
        log.warning("%s has no keyframes", video_id)
        return "skipped"

    out_path = video_dir / "visuals.json"
    title = video_title(video_dir)

    # Resume: keep frames captioned by a previous run unless --force.
    done: dict[int, dict] = {}
    if out_path.exists() and not force:
        for frame in json.loads(out_path.read_text()).get("frames", []):
            done[frame["id"]] = frame
    todo = [kf for kf in keyframes if kf["id"] not in done]
    if not todo:
        log.info("skip %s - all %d frames already captioned", video_id, len(keyframes))
        return "skipped"
    log.info("  captioning %d frame(s) (%d already done)", len(todo), len(done))

    results = dict(done)
    completed = 0

    def work(kf: dict) -> dict | None:
        try:
            caption = backend.caption(video_dir / kf["file"], title)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the batch
            log.warning("  frame %d failed: %s", kf["id"], exc)
            return None
        return {
            "id": kf["id"],
            "timestamp": kf["timestamp"],
            "ocr": caption["ocr"],
            "description": caption["description"],
            "has_text": bool(caption["ocr"]),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for frame in pool.map(work, todo):
            completed += 1
            if frame is not None:
                results[frame["id"]] = frame
            if completed % 10 == 0:
                log.info("  ... %d/%d", completed, len(todo))

    frames = [results[i] for i in sorted(results)]
    out_path.write_text(json.dumps(
        {"video_id": video_id, "backend": backend.name, "model": backend.model,
         "frames": frames},
        indent=2, ensure_ascii=False))

    failed = len(keyframes) - len(frames)
    suffix = f" ({failed} failed - re-run to retry)" if failed else ""
    log.info("done  %s - %d/%d frames captioned%s",
             video_id, len(frames), len(keyframes), suffix)
    return "ok"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def discover_video_dirs(data_dir: Path, ids: list[str]) -> list[Path]:
    """Return the video directories to process (those that have keyframes)."""
    if ids:
        return [data_dir / vid for vid in ids]
    if not data_dir.is_dir():
        return []
    return sorted(d for d in data_dir.iterdir()
                  if d.is_dir() and (d / "keyframes.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Caption video keyframes (OCR + description) with a chosen vision-model backend.",
    )
    parser.add_argument("video_ids", nargs="*",
                        help="specific video ids to process (default: all in --data-dir)")
    parser.add_argument("--data-dir", default="data", help="root data directory (default: data)")

    parser.add_argument("--backend",
                        choices=("anthropic", "gemini", "ollama", "openai"),
                        default="anthropic", help="vision-model backend (default: anthropic)")
    parser.add_argument("--model", default=None,
                        help=("model name passed to the backend "
                              f"(defaults: anthropic={DEFAULT_ANTHROPIC_MODEL}, "
                              f"gemini={DEFAULT_GEMINI_MODEL}; required for ollama/openai)"))
    parser.add_argument("--base-url", default=None,
                        help="endpoint base URL "
                             f"(defaults: ollama={DEFAULT_OLLAMA_BASE_URL}; required for openai)")
    parser.add_argument("--api-key", default=None,
                        help="API key for openai/gemini backends (overrides --api-key-env / env vars)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="env var to read the OpenAI API key from (default: OPENAI_API_KEY)")
    parser.add_argument("--timeout", type=float, default=HTTP_TIMEOUT,
                        help=f"HTTP timeout in seconds for openai/ollama backends (default: {HTTP_TIMEOUT})")

    parser.add_argument("--concurrency", type=int, default=4,
                        help="number of frames to caption in parallel (default: 4)")
    parser.add_argument("--force", action="store_true", help="re-caption already-captioned frames")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("anthropic", "httpx", "httpcore", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data_dir = Path(args.data_dir)
    targets = discover_video_dirs(data_dir, args.video_ids)
    if not targets:
        log.error("no videos with keyframes found in %s/ - run step 3 first", data_dir)
        return 1

    backend = make_backend(args)
    log.info("%d video(s) to process - backend=%s model=%s",
             len(targets), backend.name, backend.model)

    tally = {"ok": 0, "skipped": 0, "failed": 0}
    for i, video_dir in enumerate(targets, 1):
        log.info("[%d/%d] %s", i, len(targets), video_dir.name)
        tally[process_video(video_dir, backend, args.concurrency, args.force)] += 1

    log.info("finished: %(ok)d processed, %(skipped)d skipped, %(failed)d failed", tally)
    return 0 if tally["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())