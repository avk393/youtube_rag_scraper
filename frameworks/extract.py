#!/usr/bin/env python3
"""extract.py - Stage 1 of the framework layer.

Turns one video's raw pull (transcript + metadata, plus optional chart
descriptions from the vision step) into a structured extraction JSON that
separates durable methodology from quarantined perishable opinion.

  data/<id>/transcript.json  (+ metadata.json, + optional visuals.json)
      -> extractions/<id>.json   (validates against frameworks.schema.Extraction)

Usage:
  python frameworks/extract.py                 # all videos in data/ with a transcript
  python frameworks/extract.py VIDEO_ID [...]  # specific videos
  python frameworks/extract.py --force         # re-extract already-done videos
  python frameworks/extract.py --model claude-sonnet-5   # cheaper model

Requires ANTHROPIC_API_KEY (in .env or the environment).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

try:  # works both as `python frameworks/extract.py` and `python -m frameworks.extract`
    from . import llm
    from .schema import Extraction
except ImportError:
    import llm
    from schema import Extraction

log = logging.getLogger("extract")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "prompts" / "extract.v1.md"


def load_chart_descriptions(visuals_path: Path) -> str:
    """Format the chart/table frames from visuals.json for the prompt.

    Keeps only frames that actually carry analytical content (text/OCR or a
    description); returns "(none)" when there is no visuals file or nothing
    qualifies. Talking-head frames with empty ocr/description are dropped.
    """
    if not visuals_path.exists():
        return "(none)"
    try:
        data = json.loads(visuals_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read %s: %s", visuals_path.name, exc)
        return "(none)"

    blocks: list[str] = []
    for frame in data.get("frames") or []:
        ocr = (frame.get("ocr") or "").strip()
        desc = (frame.get("description") or "").strip()
        if not frame.get("has_text") and not ocr and not desc:
            continue
        ts = frame.get("timestamp")
        parts = [f"[t={ts:.1f}s]" if isinstance(ts, (int, float)) else "[frame]"]
        if desc:
            parts.append(f"Description: {desc}")
        if ocr:
            parts.append(f"OCR: {ocr}")
        blocks.append(" ".join(parts))

    return "\n\n".join(blocks) if blocks else "(none)"


def extract_one(
    client: "llm.anthropic.Anthropic",
    video_dir: Path,
    out_dir: Path,
    model: str,
    force: bool,
) -> bool:
    """Extract a single video. Returns True on success (or clean skip)."""
    video_id = video_dir.name
    transcript_path = video_dir / "transcript.json"
    metadata_path = video_dir / "metadata.json"
    out_path = out_dir / f"{video_id}.json"

    if not transcript_path.exists():
        log.warning("skip %s - no transcript.json (run transcribe.py first)", video_id)
        return False
    if out_path.exists() and not force:
        log.info("skip %s - already extracted", video_id)
        return True

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    text = html.unescape(transcript.get("text") or "").strip()
    if not text:
        log.warning("skip %s - empty transcript", video_id)
        return False

    meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    charts = load_chart_descriptions(video_dir / "visuals.json")

    rendered = llm.render_template(
        PROMPT_PATH,
        VIDEO_ID=video_id,
        TITLE=meta.get("title") or "",
        CHANNEL=meta.get("channel") or "",
        PUBLISH_DATE=meta.get("upload_date") or "",
        TRANSCRIPT=text,
        CHART_DESCRIPTIONS=charts,
    )
    system, user = llm.split_system_user(rendered)

    try:
        result: Extraction = llm.parse_structured(
            client,
            model=model,
            system=system,
            user=user or rendered,
            output_format=Extraction,
        )
    except RuntimeError as exc:
        log.error("extraction failed for %s: %s", video_id, exc)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    n_fw = len(result.frameworks)
    tag = "LOW-VALUE" if result.low_value else f"{n_fw} framework(s)"
    log.info("done  %s - %s [%s]", video_id, ", ".join(result.analytical_domains) or "-", tag)
    return True


def discover_videos(data_dir: Path, video_ids: list[str]) -> list[Path]:
    if video_ids:
        return [data_dir / vid for vid in video_ids]
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and (d / "transcript.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_ids", nargs="*", help="video ids to extract (default: all in data/)")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"), help="raw pull root (default: data/)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "extractions"), help="output dir (default: extractions/)")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL, help=f"Claude model (default: {llm.DEFAULT_MODEL})")
    parser.add_argument("--force", action="store_true", help="re-extract already-done videos")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        parser.error(f"data dir not found: {data_dir}")
    out_dir = Path(args.out_dir)

    videos = discover_videos(data_dir, args.video_ids)
    if not videos:
        log.info("no videos to extract in %s", data_dir)
        return 0

    client = llm.get_client()
    log.info("%d video(s) to extract with %s", len(videos), args.model)

    ok, failed = 0, 0
    for i, video_dir in enumerate(videos, 1):
        log.info("[%d/%d] %s", i, len(videos), video_dir.name)
        if extract_one(client, video_dir, out_dir, args.model, args.force):
            ok += 1
        else:
            failed += 1

    log.info("finished: %d ok, %d failed - output in %s/", ok, failed, out_dir)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
