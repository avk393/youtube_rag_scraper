#!/usr/bin/env python3
"""
chunk.py - Step 5 of the YouTube -> RAG pipeline.

Fuses each video's spoken transcript (step 2), keyframe metadata (step 3),
and frame captions (step 4) into a single timestamped "visual transcript",
then breaks that into the embeddable chunks the RAG will actually index.

Fusion. Walks the video on one timeline, interleaving Whisper segments with
visual events placed at each keyframe occurrence. Consecutive re-appearances
of the same keyframe (e.g. periodic re-samples of a static slide) collapse
to a single marker; a slide that genuinely recurs later still gets its own
marker each time.

Chunking. Hybrid strategy:
  - Chapter boundaries (from the YouTube metadata's `chapters` field) are
    hard breaks - chapters never share a chunk.
  - When a chunk would exceed --max-tokens, it prefers to close at the most
    recent visual change within the last 25% of its atoms, falling back to
    the current event boundary.
  - --overlap-tokens worth of the previous chunk's tail is prepended to each
    subsequent chunk for retrieval recall across boundaries.

For each data/<video_id>/ folder it reads transcript.json, keyframes.json,
visuals.json, and metadata.json, and writes:

  data/<video_id>/visual_transcript.json
      The merged event timeline plus a rendered text version, for inspection.

  data/<video_id>/chunks.jsonl
      One chunk per line, ready to feed into an embedding pipeline:
      {
        "chunk_id":      "VIDEOID_0001",
        "video_id":      "...",
        "title":         "...",
        "channel":       "...",
        "url":           "https://youtube.com/watch?v=...&t=261s",
        "chapter":       "Introduction" | null,
        "start":         261.4,
        "end":           332.5,
        "text":          "(overlap prefix) ...chunk content...",
        "token_count":   547,
        "keyframes":     [{ "id": 3, "file": "frames/frame_0003.jpg",
                            "timestamp": 265.2, "has_text": true }, ...],
        "has_ocr_text":  true
      }

Usage:
  python chunk.py                              # all videos with visuals.json
  python chunk.py <video_id> [...]
  python chunk.py --max-tokens 800 --overlap-tokens 100

Requirements:
  pip install tiktoken
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    sys.exit("tiktoken is not installed. Run: pip install tiktoken")

log = logging.getLogger("chunk")

# cl100k_base is GPT-4 / text-embedding-3's encoding; a fine proxy for
# token-budgeting across model families.
_ENC = tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def format_timestamp(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def deep_link(webpage_url: str | None, seconds: float) -> str | None:
    """YouTube deep-link URL with a &t=Ns timestamp."""
    if not webpage_url:
        return None
    t = max(int(round(seconds)), 0)
    sep = "&" if "?" in webpage_url else "?"
    return f"{webpage_url}{sep}t={t}s"


def chapter_at(t: float, chapters: list[dict]) -> str | None:
    """Title of the chapter that contains time `t`, if any."""
    for c in chapters or []:
        start = c.get("start_time", 0.0)
        end = c.get("end_time", float("inf"))
        if start <= t < end:
            return c.get("title")
    return None


# --------------------------------------------------------------------------
# Fusion: build the merged event timeline
# --------------------------------------------------------------------------

def render_visual(event: dict) -> str:
    """Render a visual event as an inline marker for the chunk text."""
    bits = []
    if event["description"]:
        bits.append(event["description"])
    if event["ocr"]:
        bits.append(f'on screen: "{event["ocr"]}"')
    body = " ".join(bits) if bits else "(no caption)"
    return f"[VISUAL @ {format_timestamp(event['start'])} — {body}]"


def render_event(event: dict) -> str:
    if event["type"] == "speech":
        return event["text"].strip()
    return render_visual(event)


def build_events(transcript: dict, keyframes_data: dict, visuals: dict) -> list[dict]:
    """Merge speech segments with per-occurrence visual events on one timeline."""
    visuals_by_id = {f["id"]: f for f in visuals.get("frames", [])}
    events: list[dict] = []

    for seg in transcript.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        events.append({
            "type": "speech",
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text,
        })

    missing_visuals = []
    for kf in keyframes_data.get("keyframes", []):
        visual = visuals_by_id.get(kf["id"])
        if visual is None:
            missing_visuals.append(kf["id"])
            continue
        ocr = (visual.get("ocr") or "").strip()
        description = (visual.get("description") or "").strip()
        has_text = bool(visual.get("has_text", bool(ocr)))
        for ts in kf.get("occurrences") or [kf.get("timestamp", 0.0)]:
            events.append({
                "type": "visual",
                "start": float(ts),
                "end": float(ts),
                "frame_id": kf["id"],
                "file": kf["file"],
                "ocr": ocr,
                "description": description,
                "has_text": has_text,
            })
    if missing_visuals:
        log.warning("  %d keyframes had no caption and were skipped: %s",
                    len(missing_visuals), missing_visuals[:5])

    # Sort by start; on ties, place visual markers before speech that begins
    # at the same instant so they read as "[VISUAL] ...speech..." in the text.
    events.sort(key=lambda e: (e["start"], 0 if e["type"] == "visual" else 1))

    # Collapse consecutive duplicate visual occurrences (the periodic re-
    # samples of a static slide); a slide that genuinely recurs later still
    # gets its own marker because the dedup is only against the previous
    # visual event, not all of history.
    deduped: list[dict] = []
    last_visual_id: int | None = None
    for e in events:
        if e["type"] == "visual":
            if e["frame_id"] == last_visual_id:
                continue
            last_visual_id = e["frame_id"]
        deduped.append(e)
    return deduped


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def _recent_visual_break(atoms: list[dict], lookback_frac: float = 0.25) -> int | None:
    """Index of the most recent visual atom within the last `lookback_frac`
    of the chunk, or None if there isn't one."""
    if not atoms:
        return None
    n = len(atoms)
    cutoff = max(0, n - max(1, int(n * lookback_frac) + 1))
    for i in range(n - 1, cutoff - 1, -1):
        if atoms[i]["event"]["type"] == "visual":
            return i
    return None


def chunk_events(events: list[dict], max_tokens: int, overlap_tokens: int,
                  chapters: list[dict]) -> list[dict]:
    """Greedy chunker with chapter-hard / visual-soft / token-cap breaks."""
    if not events:
        return []

    # Pre-render and pre-tokenize each event so we can budget cheaply.
    atoms = [{"event": e, "text": render_event(e), "tokens": count_tokens(render_event(e))}
             for e in events]

    raw: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    chunk_chapter: str | None = None

    for atom in atoms:
        e = atom["event"]
        event_chapter = chapter_at(e["start"], chapters)

        # Chapter break: hard boundary BEFORE adding this atom. We close the
        # current chunk whenever the incoming atom belongs to a different
        # chapter than the chunk we've been building.
        if current and event_chapter != chunk_chapter:
            raw.append(current)
            current, current_tokens = [], 0

        # Token cap reached: close the chunk, preferring a recent visual break.
        if current and current_tokens + atom["tokens"] > max_tokens:
            split = _recent_visual_break(current)
            if split is not None and split > 0:
                raw.append(current[:split])
                current = current[split:]
                current_tokens = sum(a["tokens"] for a in current)
            else:
                raw.append(current)
                current, current_tokens = [], 0

        if not current:
            chunk_chapter = event_chapter
        current.append(atom)
        current_tokens += atom["tokens"]

    if current:
        raw.append(current)

    # Prepend overlap from the previous chunk's tail to every chunk but the first.
    chunks = []
    for i, main_atoms in enumerate(raw):
        overlap: list[dict] = []
        if i > 0 and overlap_tokens > 0:
            running = 0
            for atom in reversed(raw[i - 1]):
                if running + atom["tokens"] > overlap_tokens:
                    break
                overlap.insert(0, atom)
                running += atom["tokens"]
        chunks.append({"overlap": overlap, "main": main_atoms})

    return chunks


def build_chunk_record(chunk: dict, video_id: str, metadata: dict,
                        chapters: list[dict], index: int) -> dict:
    """Assemble the final dict that lands in chunks.jsonl."""
    overlap, main = chunk["overlap"], chunk["main"]
    all_atoms = overlap + main
    text = " ".join(a["text"] for a in all_atoms).strip()

    # Start/end reflect the NEW content of this chunk, not the overlap prefix,
    # so a deep-link lands the viewer on the chunk's real content.
    main_events = [a["event"] for a in main]
    start = main_events[0]["start"]
    end = max(e["end"] for e in main_events)

    # Unique keyframes referenced anywhere in the chunk (including overlap).
    seen: dict[int, dict] = {}
    for a in all_atoms:
        e = a["event"]
        if e["type"] == "visual" and e["frame_id"] not in seen:
            seen[e["frame_id"]] = {
                "id": e["frame_id"],
                "file": e["file"],
                "timestamp": e["start"],
                "has_text": e["has_text"],
            }

    return {
        "chunk_id": f"{video_id}_{index:04d}",
        "video_id": video_id,
        "title": metadata.get("title"),
        "channel": metadata.get("channel") or metadata.get("uploader"),
        "url": deep_link(metadata.get("webpage_url"), start),
        "chapter": chapter_at(start, chapters),
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text,
        "token_count": sum(a["tokens"] for a in all_atoms),
        "keyframes": list(seen.values()),
        "has_ocr_text": any(kf["has_text"] for kf in seen.values()),
    }


# --------------------------------------------------------------------------
# Per-video processing
# --------------------------------------------------------------------------

def process_video(video_dir: Path, max_tokens: int, overlap_tokens: int,
                   force: bool) -> str:
    video_id = video_dir.name
    transcript_path = video_dir / "transcript.json"
    keyframes_path = video_dir / "keyframes.json"
    visuals_path = video_dir / "visuals.json"

    for path, step in ((transcript_path, "step 2"),
                       (keyframes_path, "step 3"),
                       (visuals_path, "step 4")):
        if not path.exists():
            log.error("%s missing - run %s first", path.name, step)
            return "failed"

    vt_path = video_dir / "visual_transcript.json"
    chunks_path = video_dir / "chunks.jsonl"
    if vt_path.exists() and chunks_path.exists() and not force:
        log.info("skip %s - already chunked", video_id)
        return "skipped"

    metadata: dict = {}
    meta_path = video_dir / "metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("  could not read metadata.json")

    transcript = json.loads(transcript_path.read_text())
    keyframes_data = json.loads(keyframes_path.read_text())
    visuals = json.loads(visuals_path.read_text())
    chapters = metadata.get("chapters") or []

    # Extend the last chapter so events sitting on the trailing edge (a frame
    # whose occurrence lands at exactly the video duration, say) resolve to
    # that chapter rather than spawning a tiny "no chapter" orphan chunk.
    if chapters:
        chapters = [dict(c) for c in sorted(chapters, key=lambda c: c.get("start_time", 0.0))]
        chapters[-1]["end_time"] = float("inf")

    # 1. Fuse
    events = build_events(transcript, keyframes_data, visuals)
    if not events:
        log.warning("%s has no content to chunk", video_id)
        return "skipped"

    rendered = " ".join(render_event(e) for e in events)
    duration = float(metadata.get("duration") or keyframes_data.get("duration") or 0.0)

    vt_path.write_text(json.dumps({
        "video_id": video_id,
        "title": metadata.get("title"),
        "duration": round(duration, 3),
        "events": events,
        "rendered": rendered,
    }, indent=2, ensure_ascii=False))

    # 2. Chunk
    chunks = chunk_events(events, max_tokens, overlap_tokens, chapters)

    with chunks_path.open("w", encoding="utf-8") as fh:
        for i, ch in enumerate(chunks, 1):
            record = build_chunk_record(ch, video_id, metadata, chapters, i)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("done  %s - %d events -> %d chunks", video_id, len(events), len(chunks))
    return "ok"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def discover_video_dirs(data_dir: Path, ids: list[str]) -> list[Path]:
    if ids:
        return [data_dir / vid for vid in ids]
    if not data_dir.is_dir():
        return []
    return sorted(d for d in data_dir.iterdir()
                  if d.is_dir() and (d / "visuals.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse transcripts and frame captions into chunks ready for embedding.",
    )
    parser.add_argument("video_ids", nargs="*",
                        help="specific video ids to process (default: all in --data-dir)")
    parser.add_argument("--data-dir", default="data", help="root data directory (default: data)")
    parser.add_argument("--max-tokens", type=int, default=600,
                        help="approximate token cap per chunk (default: 600)")
    parser.add_argument("--overlap-tokens", type=int, default=80,
                        help="overlap added to each chunk from the previous one (default: 80)")
    parser.add_argument("--force", action="store_true", help="re-chunk already-processed videos")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    targets = discover_video_dirs(data_dir, args.video_ids)
    if not targets:
        log.error("no videos with visuals.json found in %s/ - run steps 1-4 first", data_dir)
        return 1

    log.info("%d video(s) to process (max_tokens=%d, overlap=%d)",
             len(targets), args.max_tokens, args.overlap_tokens)
    tally = {"ok": 0, "skipped": 0, "failed": 0}
    for i, video_dir in enumerate(targets, 1):
        log.info("[%d/%d] %s", i, len(targets), video_dir.name)
        tally[process_video(video_dir, args.max_tokens, args.overlap_tokens, args.force)] += 1

    log.info("finished: %(ok)d processed, %(skipped)d skipped, %(failed)d failed", tally)
    return 0 if tally["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())