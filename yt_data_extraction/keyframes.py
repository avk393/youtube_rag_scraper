#!/usr/bin/env python3
"""
keyframes.py - Step 3 of the YouTube -> RAG pipeline.

Extracts a deduplicated set of keyframes from each ingested video, so step 4
can OCR / caption the meaningful visuals (slides, diagrams, demo screens)
without paying to process thousands of near-identical frames.

Selection strategy:
  1. Scene detection (PySceneDetect ContentDetector) - one frame shortly into
     every detected scene. Catches slide changes and hard cuts precisely.
  2. Periodic safety-net sampling - one frame every --interval seconds.
     Catches slowly-evolving visuals (whiteboards, screen recordings) that
     never trigger a hard cut.
  3. Deduplication by perceptual hash (structure) plus mean colour - near-
     identical frames collapse to a single entry. A visual that recurs (e.g.
     a recap slide) is stored once, with every timestamp it appeared at
     recorded under "occurrences".

For each data/<video_id>/ folder it writes:

  data/<video_id>/frames/frame_0001.jpg, frame_0002.jpg, ...
  data/<video_id>/keyframes.json
      {
        "video_id": "...", "fps": 30.0, "duration": 612.4,
        "keyframes": [
          { "id": 1, "file": "frames/frame_0001.jpg",
            "timestamp": 12.25, "occurrences": [12.25, 305.2],
            "phash": "c3e1...", "width": 1280, "height": 720 },
          ...
        ]
      }

Usage:
  python keyframes.py                      # all videos in data/ without keyframes
  python keyframes.py <video_id> [...]     # only the named videos
  python keyframes.py --interval 30 --threshold 27

Requirements:
  pip install scenedetect imagehash
  ffmpeg must be installed and on PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image
    from scenedetect import detect as scene_detect, ContentDetector
except ImportError as exc:
    sys.exit(
        f"missing dependency '{exc.name}'.\n"
        "install the step-3 requirements with:\n"
        "  pip install scenedetect imagehash"
    )

log = logging.getLogger("keyframes")

JPEG_QUALITY = 90
# Pillow renamed the resampling constants; this works on old and new versions.
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


# --------------------------------------------------------------------------
# Frame helpers
# --------------------------------------------------------------------------

def grab_frame(cap, fps: float, timestamp: float, max_width: int):
    """Seek to `timestamp` seconds and return a PIL.Image (or None on failure)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max_width and img.width > max_width:
        new_height = round(img.height * max_width / img.width)
        img = img.resize((max_width, new_height), RESAMPLE)
    return img


def mean_color(img):
    """Average RGB colour of an image.

    Perceptual hashing captures structure, not colour, so two differently
    coloured but equally flat frames (e.g. a black fade and a solid title
    card) hash alike. Comparing mean colour as well keeps them distinct.
    """
    return np.asarray(img.convert("RGB"), dtype="float32").reshape(-1, 3).mean(axis=0)


def meta_duration(video_dir: Path) -> float | None:
    """Authoritative duration from step 1's metadata.json, if available."""
    meta_path = video_dir / "metadata.json"
    if meta_path.exists():
        try:
            value = json.loads(meta_path.read_text()).get("duration")
            return float(value) if value else None
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            return None
    return None


def find_video_file(video_dir: Path) -> Path | None:
    """Locate the downloaded video file inside a data/<id>/ directory."""
    primary = video_dir / "video.mp4"
    if primary.exists():
        return primary
    others = [p for p in video_dir.glob("video.*")
              if p.suffix.lower() not in (".vtt", ".json")]
    return others[0] if others else None


# --------------------------------------------------------------------------
# Candidate timestamp selection
# --------------------------------------------------------------------------

def _tc_seconds(timecode) -> float:
    """Seconds from a PySceneDetect FrameTimecode, across library versions
    (0.7 deprecated .get_seconds() in favour of the .seconds property)."""
    return timecode.seconds if hasattr(timecode, "seconds") else timecode.get_seconds()


def scene_sample_times(video_path: Path, threshold: float, min_scene_len: int,
                       duration: float) -> list[float]:
    """Return one sample time shortly into each detected scene.

    Sampling 0.25s past the cut avoids the blurred/transition frame that
    sometimes sits right on a scene boundary.
    """
    try:
        scenes = scene_detect(
            str(video_path),
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len),
        )
    except Exception as exc:  # noqa: BLE001 - detection is best-effort
        log.warning("  scene detection failed (%s) - using periodic samples only", exc)
        scenes = []

    if not scenes:                       # no cuts -> the whole video is one scene
        return [0.0]
    times = []
    for start, end in scenes:
        s, e = _tc_seconds(start), _tc_seconds(end)
        times.append(min(s + 0.25, max(s, e - 0.05)))
    return times


def periodic_times(duration: float, interval: float) -> list[float]:
    """Evenly spaced safety-net sample times across the video."""
    if interval <= 0:
        return []
    times, t = [], 0.0
    while t < duration:
        times.append(t)
        t += interval
    return times


# --------------------------------------------------------------------------
# Per-video processing
# --------------------------------------------------------------------------

def process_video(video_dir: Path, threshold: float, min_scene_len: int,
                   interval: float, hash_threshold: int, color_threshold: float,
                   max_width: int, force: bool) -> str:
    """Extract deduplicated keyframes for one video. Returns ok/skipped/failed."""
    video_id = video_dir.name
    out_path = video_dir / "keyframes.json"

    if not video_dir.is_dir():
        log.error("no such directory: %s", video_dir)
        return "failed"
    if out_path.exists() and not force:
        log.info("skip %s - keyframes already extracted", video_id)
        return "skipped"

    video_path = find_video_file(video_dir)
    if video_path is None:
        log.error("no video file in %s", video_dir)
        return "failed"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("could not open %s", video_path)
        return "failed"

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = meta_duration(video_dir) or (frame_count / fps if fps else 0.0)
    if fps <= 0 or duration <= 0:
        log.error("invalid video metadata for %s (fps=%s, duration=%s)",
                  video_id, fps, duration)
        cap.release()
        return "failed"

    # Merge scene-change and periodic sample times into one candidate set.
    candidates = sorted({
        round(t, 2)
        for t in scene_sample_times(video_path, threshold, min_scene_len, duration)
        + periodic_times(duration, interval)
        if 0 <= t < duration
    })
    log.info("  %d candidate frame(s) to inspect", len(candidates))

    frames_dir = video_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    if force:                            # clear stale frames on a re-run
        for old in frames_dir.glob("frame_*.jpg"):
            old.unlink()

    # Extract each candidate, then keep it only if it is perceptually distinct
    # from every keyframe kept so far.
    keyframes: list[dict] = []
    for timestamp in candidates:
        img = grab_frame(cap, fps, timestamp, max_width)
        if img is None:
            continue
        phash = imagehash.phash(img)
        color = mean_color(img)

        # A frame is a duplicate only if it matches an existing keyframe in
        # BOTH structure (perceptual hash) and colour.
        duplicate_of = next(
            (kf for kf in keyframes
             if phash - kf["_hash"] <= hash_threshold
             and float(np.linalg.norm(color - kf["_color"])) <= color_threshold),
            None,
        )
        if duplicate_of is not None:
            duplicate_of["occurrences"].append(timestamp)
            continue

        kf_id = len(keyframes) + 1
        rel_path = f"frames/frame_{kf_id:04d}.jpg"
        img.save(video_dir / rel_path, quality=JPEG_QUALITY)
        keyframes.append({
            "id": kf_id,
            "file": rel_path,
            "timestamp": timestamp,
            "occurrences": [timestamp],
            "phash": str(phash),
            "width": img.width,
            "height": img.height,
            "_hash": phash,              # internal only - stripped before write
            "_color": color,             # internal only - stripped before write
        })

    cap.release()

    result = {
        "video_id": video_id,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration": round(duration, 3),
        "scene_threshold": threshold,
        "sample_interval": interval,
        "hash_threshold": hash_threshold,
        "color_threshold": color_threshold,
        "keyframes": [{k: v for k, v in kf.items() if not k.startswith("_")}
                      for kf in keyframes],
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("done  %s - %d keyframes kept from %d candidates",
             video_id, len(keyframes), len(candidates))
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
        description="Extract deduplicated keyframes from ingested YouTube videos.",
    )
    parser.add_argument("video_ids", nargs="*",
                        help="specific video ids to process (default: all in --data-dir)")
    parser.add_argument("--data-dir", default="data", help="root data directory (default: data)")
    parser.add_argument("--threshold", type=float, default=27.0,
                        help="scene-detection sensitivity; lower = more scenes (default: 27)")
    parser.add_argument("--min-scene-len", type=int, default=15,
                        help="minimum scene length in frames (default: 15)")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="periodic safety-net sample interval in seconds, 0 to disable (default: 60)")
    parser.add_argument("--hash-threshold", type=int, default=5,
                        help="perceptual-hash distance below which frames are treated as duplicates (default: 5)")
    parser.add_argument("--color-threshold", type=float, default=16.0,
                        help="mean-colour distance below which frames are treated as duplicates (default: 16)")
    parser.add_argument("--max-width", type=int, default=0,
                        help="downscale frames wider than this many pixels (default: 0, no resize)")
    parser.add_argument("--force", action="store_true", help="re-extract existing keyframes")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("PIL", "scenedetect", "pyscenedetect"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data_dir = Path(args.data_dir)
    targets = discover_video_dirs(data_dir, args.video_ids)
    if not targets:
        log.error("no video directories found in %s/ - run step 1 (ingest.py) first", data_dir)
        return 1

    log.info("%d video(s) to process", len(targets))
    tally = {"ok": 0, "skipped": 0, "failed": 0}
    for i, video_dir in enumerate(targets, 1):
        log.info("[%d/%d] %s", i, len(targets), video_dir.name)
        tally[process_video(video_dir, args.threshold, args.min_scene_len,
                            args.interval, args.hash_threshold, args.color_threshold,
                            args.max_width, args.force)] += 1

    log.info("finished: %(ok)d processed, %(skipped)d skipped, %(failed)d failed", tally)
    return 0 if tally["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())