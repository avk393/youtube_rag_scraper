#!/usr/bin/env python3
"""
transcribe.py - Step 2 of the YouTube -> RAG pipeline.

Turns each ingested video's audio into a clean, timestamped transcript.

Primary path : faster-whisper on audio.m4a -> punctuated text with
               segment- AND word-level timestamps (best for chunking and
               for aligning visual keyframes later).
Fallback path: parse the YouTube captions.*.vtt that step 1 saved. Used when
               audio is missing, faster-whisper is not installed, or
               --captions-only is passed. Caption timestamps are cue-level
               only (no word timestamps) and auto-caption text is approximate.

For each data/<video_id>/ folder it writes:

  data/<video_id>/transcript.json
      {
        "video_id": "...",
        "source":   "whisper" | "captions",
        "language": "en",
        "model":    "small",              # whisper path only
        "duration": 613.4,
        "text":     "full transcript ...",
        "segments": [ {id, start, end, text}, ... ],
        "words":    [ {start, end, word}, ... ]    # whisper path only
      }

Usage:
  python transcribe.py                     # every un-transcribed video in data/
  python transcribe.py <video_id> [...]    # only the named videos
  python transcribe.py --model large-v3    # higher accuracy, slower
  python transcribe.py --device cuda --compute-type float16   # GPU
  python transcribe.py --captions-only     # skip Whisper, parse VTT only

Requirements:
  pip install faster-whisper        (only needed for the Whisper path)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from functools import reduce
from pathlib import Path

log = logging.getLogger("transcribe")


# --------------------------------------------------------------------------
# WebVTT caption parsing (fallback path)
# --------------------------------------------------------------------------

# A cue timing line, e.g. "00:01:02.500 --> 00:01:05.000 align:start position:0%"
_TS = re.compile(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})")
# Inline tags YouTube embeds in auto-caption text: <00:00:01.234>, <c>, </c>
_TAG = re.compile(r"<[^>]+>")


def _vtt_time(value: str) -> float:
    """Convert an HH:MM:SS.mmm timestamp into seconds."""
    h, m, s = value.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _dedup_segments(segments: list[dict]) -> list[dict]:
    """Remove the rolling overlap YouTube auto-captions repeat between cues.

    For each cue, any leading words that duplicate the tail of the previous
    cue are trimmed. Cues that are fully contained in the previous one are
    dropped (their time range is merged into it). Manual captions, which have
    no overlap, pass through unchanged.
    """
    out: list[dict] = []
    for seg in segments:
        if not out:
            out.append(dict(seg))
            continue
        prev_words = out[-1]["text"].split()
        cur_words = seg["text"].split()
        overlap = 0
        for k in range(min(len(prev_words), len(cur_words)), 0, -1):
            if prev_words[-k:] == cur_words[:k]:
                overlap = k
                break
        remainder = cur_words[overlap:]
        if remainder:
            new = dict(seg)
            new["text"] = " ".join(remainder)
            out.append(new)
        else:  # fully overlapping cue - just extend the previous time range
            out[-1]["end"] = max(out[-1]["end"], seg["end"])
    for idx, seg in enumerate(out):
        seg["id"] = idx
    return out


def parse_vtt(path: Path) -> tuple[list[dict], str | None]:
    """Parse a WebVTT file into deduplicated (id, start, end, text) segments."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    language: str | None = None
    raw: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if language is None and line.lower().startswith("language:"):
            language = line.split(":", 1)[1].strip() or None
        match = _TS.search(line)
        if not match:
            i += 1
            continue
        start, end = _vtt_time(match.group(1)), _vtt_time(match.group(2))
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(_TAG.sub("", lines[i]))
            i += 1
        text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
        if text:
            raw.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return _dedup_segments(raw), language


# --------------------------------------------------------------------------
# Whisper transcription (primary path)
# --------------------------------------------------------------------------

def load_model(model_size: str, device: str, compute_type: str):
    """Load a faster-whisper model. Raises ImportError if it is not installed."""
    from faster_whisper import WhisperModel  # imported lazily - it is optional

    log.info("loading Whisper model '%s' (device=%s, compute=%s) ...",
             model_size, device, compute_type)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_whisper(model, audio_path: Path, language: str | None) -> dict:
    """Transcribe an audio file, returning segments, words, text and metadata."""
    # `segments` is a generator - the actual work happens while iterating it.
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,        # None -> auto-detect
        word_timestamps=True,
        vad_filter=True,          # skip silence; avoids repetition hallucinations
    )

    segments: list[dict] = []
    words: list[dict] = []
    for idx, seg in enumerate(segments_gen):
        segments.append({
            "id": idx,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
        })
        for word in (seg.words or []):
            words.append({
                "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
                "word": word.word.strip(),
            })

    return {
        "language": info.language,
        "duration": round(float(info.duration), 3),
        "text": " ".join(s["text"] for s in segments).strip(),
        "segments": segments,
        "words": words,
    }


# --------------------------------------------------------------------------
# Per-video processing
# --------------------------------------------------------------------------

def video_title(video_dir: Path) -> str:
    """Best-effort lookup of a video's title from its metadata.json."""
    meta_path = video_dir / "metadata.json"
    if meta_path.exists():
        try:
            return (json.loads(meta_path.read_text()).get("title") or "")[:50]
        except (json.JSONDecodeError, OSError):
            pass
    return ""


def process_video(
    video_dir: Path,
    model,
    model_size: str,
    language: str | None,
    captions_only: bool,
    force: bool,
) -> str:
    """Transcribe one video directory. Returns 'ok', 'skipped' or 'failed'."""
    video_id = video_dir.name
    out_path = video_dir / "transcript.json"

    if not video_dir.is_dir():
        log.error("no such directory: %s", video_dir)
        return "failed"
    if out_path.exists() and not force:
        log.info("skip %s - already transcribed", video_id)
        return "skipped"

    audio_path = video_dir / "audio.m4a"
    vtt_files = sorted(video_dir.glob("captions.*.vtt"))
    result: dict | None = None

    # --- primary: Whisper -------------------------------------------------
    if model is not None and not captions_only and audio_path.exists():
        try:
            started = time.time()
            data = transcribe_whisper(model, audio_path, language)
            result = {"video_id": video_id, "source": "whisper",
                      "model": model_size, **data}
            log.info("  whisper: %d segments, %d words in %.1fs",
                     len(data["segments"]), len(data["words"]),
                     time.time() - started)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            log.warning("Whisper failed for %s (%s) - trying captions", video_id, exc)

    # --- fallback: YouTube captions --------------------------------------
    if result is None and vtt_files:
        log.info("  using captions: %s", vtt_files[0].name)
        segments, lang = parse_vtt(vtt_files[0])
        result = {
            "video_id": video_id,
            "source": "captions",
            "language": language or lang,
            "duration": segments[-1]["end"] if segments else 0.0,
            "text": " ".join(s["text"] for s in segments).strip(),
            "segments": segments,
            "words": [],
        }

    if result is None:
        log.error("no transcript for %s - no audio and no captions", video_id)
        return "failed"

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("done  %s - %s (source=%s, %d segments)",
             video_id, video_title(video_dir), result["source"],
             len(result["segments"]))
    return "ok"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def discover_video_dirs(data_dir: Path, ids: list[str]) -> list[Path]:
    """Return the video directories to process."""
    if ids:
        return [data_dir / vid for vid in ids]
    if not data_dir.is_dir():
        return []
    return sorted(d for d in data_dir.iterdir()
                  if d.is_dir() and (d / "metadata.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe ingested YouTube videos into timestamped transcript.json files.",
    )
    parser.add_argument("video_ids", nargs="*",
                        help="specific video ids to transcribe (default: all in --data-dir)")
    parser.add_argument("--data-dir", default="data", help="root data directory (default: data)")
    parser.add_argument("--model", default="small",
                        help="faster-whisper model: tiny/base/small/medium/large-v3 (default: small)")
    parser.add_argument("--device", default="auto",
                        help="auto / cpu / cuda (default: auto)")
    parser.add_argument("--compute-type", default="int8",
                        help="int8 for CPU; use float16 with --device cuda (default: int8)")
    parser.add_argument("--language", default=None,
                        help="force a language code (default: auto-detect)")
    parser.add_argument("--captions-only", action="store_true",
                        help="skip Whisper and parse the saved .vtt captions instead")
    parser.add_argument("--force", action="store_true", help="re-transcribe existing transcripts")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep third-party chatter (HF downloads, HTTP requests) out of the output.
    for noisy in ("faster_whisper", "huggingface_hub", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data_dir = Path(args.data_dir)
    targets = discover_video_dirs(data_dir, args.video_ids)
    if not targets:
        log.error("no video directories found in %s/ - run step 1 (ingest.py) first", data_dir)
        return 1

    pending = [d for d in targets if args.force or not (d / "transcript.json").exists()]
    log.info("%d video(s) found, %d to transcribe", len(targets), len(pending))

    # Load the Whisper model once and reuse it for every video.
    model = None
    if not args.captions_only and pending:
        try:
            model = load_model(args.model, args.device, args.compute_type)
        except ImportError:
            log.warning("faster-whisper not installed - falling back to captions only")
            log.warning("install it with:  pip install faster-whisper")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load Whisper model (%s) - falling back to captions", exc)

    tally = {"ok": 0, "skipped": 0, "failed": 0}
    for i, video_dir in enumerate(targets, 1):
        log.info("[%d/%d] %s", i, len(targets), video_dir.name)
        tally[process_video(video_dir, model, args.model,
                            args.language, args.captions_only, args.force)] += 1

    log.info("finished: %(ok)d transcribed, %(skipped)d skipped, %(failed)d failed", tally)
    return 0 if tally["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())