#!/usr/bin/env python3
"""
ingest.py - Step 1 of the YouTube -> RAG pipeline.

Pulls raw source material for one or more YouTube videos so that later stages
(transcription, keyframe extraction, captioning) have everything they need:

  * video file      - capped resolution, used later for keyframe extraction
  * audio track     - mono 16 kHz, the ideal input for Whisper transcription
  * captions        - YouTube's subtitles / auto-captions, a free transcript baseline
  * metadata.json   - slim, RAG-relevant fields (title, channel, chapters, ...)

Output layout (one directory per video, keyed by the YouTube video id):

  <output-dir>/<video_id>/
      video.mp4
      audio.m4a
      captions.en.vtt        (only if subtitles exist)
      metadata.json

Usage:
  python ingest.py <url> [<url> ...]
  python ingest.py --urls-file urls.txt
  python ingest.py <playlist-or-channel-url>          # expanded automatically

Requirements:
  pip install yt-dlp
  ffmpeg must be installed and on PATH
  a JS runtime (deno or node) is recommended - recent yt-dlp needs one for
  full YouTube format extraction
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
except ImportError:
    sys.exit("yt-dlp is not installed. Run: pip install yt-dlp")

log = logging.getLogger("ingest")

# Tell yt-dlp to use node (already on PATH) for JS extraction; avoids the
# "no supported JavaScript runtime" warning that causes some formats to be missing.
_NODE_PATH = shutil.which("node")
_BASE_YDL_OPTS: dict = {"js_runtimes": {"nodejs": {"path": _NODE_PATH}}} if _NODE_PATH else {}


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

# Fields worth keeping from yt-dlp's (very large) info dict. Everything here is
# useful downstream - `chapters` in particular helps with semantic chunking.
_META_FIELDS = (
    "id", "title", "description", "webpage_url",
    "channel", "channel_id", "uploader", "uploader_id",
    "duration", "upload_date", "language",
    "view_count", "like_count", "categories", "tags",
    "width", "height", "fps", "thumbnail", "chapters",
)


def slim_metadata(info: dict, captions: list[str]) -> dict:
    """Reduce yt-dlp's info dict to the fields the RAG pipeline cares about."""
    meta = {key: info.get(key) for key in _META_FIELDS}
    meta["caption_files"] = captions
    meta["ingested_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return meta


# --------------------------------------------------------------------------
# Audio extraction
# --------------------------------------------------------------------------

def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Pull a mono 16 kHz audio track out of the video file via ffmpeg.

    Mono / 16 kHz is what speech models (Whisper) expect and keeps files tiny.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",                       # drop the video stream
        "-ac", "1", "-ar", "16000",  # mono, 16 kHz
        "-c:a", "aac", "-b:a", "64k",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("audio extraction failed for %s: %s", video_path.name, exc)
        return False


# --------------------------------------------------------------------------
# URL expansion
# --------------------------------------------------------------------------

def expand_to_videos(url: str, insecure: bool = False) -> list[str]:
    """Expand a playlist / channel URL into individual video URLs.

    A plain video URL is returned unchanged.
    """
    opts = {**_BASE_YDL_OPTS, "quiet": True, "skip_download": True, "extract_flat": "in_playlist"}
    if insecure:
        opts["nocheckcertificate"] = True
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        log.error("could not read %s: %s", url, exc)
        return []

    if info.get("_type") in ("playlist", "multi_video") or info.get("entries"):
        urls = []
        for entry in info.get("entries") or []:
            if not entry:
                continue
            urls.append(entry.get("webpage_url") or entry.get("url") or entry["id"])
        return urls
    return [url]


# --------------------------------------------------------------------------
# Download a single video
# --------------------------------------------------------------------------

def download_video(
    url: str,
    output_dir: Path,
    resolution: int,
    langs: list[str],
    cookies: Path | None,
    force: bool,
    insecure: bool = False,
) -> dict | None:
    """Download one video plus its audio, captions and metadata.

    Returns the slim metadata dict, or None if the video was skipped / failed.
    """
    # We need the video id before downloading so the skip-existing check works.
    probe_opts = {**_BASE_YDL_OPTS, "quiet": True, "skip_download": True, "noplaylist": True}
    if cookies:
        probe_opts["cookiefile"] = str(cookies)
    if insecure:
        probe_opts["nocheckcertificate"] = True
    try:
        with YoutubeDL(probe_opts) as ydl:
            probe = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        log.error("skipping %s (unavailable): %s", url, exc)
        return None

    video_id = probe["id"]
    video_dir = output_dir / video_id
    meta_path = video_dir / "metadata.json"

    if meta_path.exists() and not force:
        log.info("skip %s - already ingested", video_id)
        return json.loads(meta_path.read_text())

    video_dir.mkdir(parents=True, exist_ok=True)

    has_ffmpeg = bool(shutil.which("ffmpeg"))
    if has_ffmpeg:
        fmt = (
            f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={resolution}]+bestaudio/"
            f"best[height<={resolution}]/best"
        )
    else:
        # Without ffmpeg we can only use pre-muxed formats.
        fmt = f"best[height<={resolution}][ext=mp4]/best[height<={resolution}]/best"

    ydl_opts = {
        **_BASE_YDL_OPTS,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(video_dir / "video.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "subtitlesformat": "vtt",
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
    }
    if cookies:
        ydl_opts["cookiefile"] = str(cookies)
    if insecure:
        ydl_opts["nocheckcertificate"] = True

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        log.error("download failed for %s: %s", video_id, exc)
        return None

    # Locate the video file that was actually written.
    video_path = None
    for dl in info.get("requested_downloads") or []:
        if dl.get("filepath"):
            video_path = Path(dl["filepath"])
            break
    if video_path is None or not video_path.exists():
        candidates = [p for p in video_dir.glob("video.*") if p.suffix != ".vtt"]
        video_path = candidates[0] if candidates else None
    if video_path is None:
        log.error("no video file produced for %s", video_id)
        return None

    # Extract the audio track used by the transcription step.
    extract_audio(video_path, video_dir / "audio.m4a")

    # yt-dlp names subtitle files after the video stem (video.en.vtt);
    # normalise them to captions.<lang>.vtt for a predictable layout.
    for vtt in video_dir.glob("video.*.vtt"):
        new_name = "captions." + vtt.name[len("video."):]
        vtt.rename(video_dir / new_name)

    # Record which caption files actually landed on disk.
    captions = sorted(p.name for p in video_dir.glob("captions.*.vtt"))

    meta = slim_metadata(info, captions)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info("done  %s - %s", video_id, (meta.get("title") or "")[:60])
    return meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def collect_input_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls)
    if args.urls_file:
        text = Path(args.urls_file).read_text()
        urls += [ln.strip() for ln in text.splitlines()
                 if ln.strip() and not ln.startswith("#")]
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube video, audio, captions and metadata for a RAG pipeline.",
    )
    parser.add_argument("urls", nargs="*", help="video / playlist / channel URLs")
    parser.add_argument("--urls-file", help="text file with one URL per line (# for comments)")
    parser.add_argument("--output-dir", default="data", help="root output directory (default: data)")
    parser.add_argument("--resolution", type=int, default=720,
                        help="max video height in pixels (default: 720)")
    parser.add_argument("--langs", default="en",
                        help="comma-separated caption languages (default: en)")
    parser.add_argument("--cookies", help="path to a cookies.txt file (for age-restricted videos)")
    parser.add_argument("--force", action="store_true", help="re-download already-ingested videos")
    parser.add_argument("--insecure", action="store_true",
                        help="skip TLS certificate verification (e.g. behind a corporate proxy)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found on PATH - audio extraction and merging will fail")

    raw_urls = collect_input_urls(args)
    if not raw_urls:
        parser.error("no URLs given - pass them as arguments or via --urls-file")

    output_dir = Path(args.output_dir)
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    cookies = Path(args.cookies) if args.cookies else None

    # Expand any playlist / channel URLs into individual video URLs.
    video_urls: list[str] = []
    for url in raw_urls:
        video_urls.extend(expand_to_videos(url, args.insecure))
    log.info("%d video(s) to process", len(video_urls))

    ok, failed = 0, 0
    for i, url in enumerate(video_urls, 1):
        log.info("[%d/%d] %s", i, len(video_urls), url)
        meta = download_video(url, output_dir, args.resolution, langs,
                              cookies, args.force, args.insecure)
        if meta:
            ok += 1
        else:
            failed += 1

    log.info("finished: %d ok, %d failed - output in %s/", ok, failed, output_dir)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())