#!/usr/bin/env python3
"""
embed.py - Step 6 of the YouTube -> RAG pipeline.

Reads each video's chunks.jsonl from step 5, embeds every chunk with a chosen
embedding model, and writes the vectors plus chunk metadata into a LanceDB
table. The resulting table is what your OpenClaw agent will query.

Embedding backends (choose with --backend):

  voyage      Voyage AI - Anthropic's recommended embedding provider for
              Claude-paired retrieval (https://docs.voyageai.com). Reads
              VOYAGE_API_KEY. Default model: voyage-3.5-lite.

  gemini      Google Gemini embeddings via the google-genai SDK. Reads
              GEMINI_API_KEY or GOOGLE_API_KEY. Default model:
              gemini-embedding-001 (3072-d; reducible via --dimensions).

  ollama      Local Ollama server, using its native /api/embed batch
              endpoint. Default --base-url is http://localhost:11434. No
              API key. Pull an embedding model first, e.g.
              `ollama pull nomic-embed-text` or `ollama pull bge-m3`.

  openai      Any OpenAI-compatible /v1/embeddings endpoint - OpenAI itself,
              OpenRouter, vLLM, llama.cpp server, LM Studio, etc. Requires
              --base-url and --model. API key from --api-key,
              --api-key-env, or $OPENAI_API_KEY.

Vector store: LanceDB (embedded; no server required). Each video's chunks
land in one shared table; the agent queries against that table for retrieval.

Re-running is safe and resumes: chunks already present in the table for a
video are skipped. Use --force to drop a video's existing rows and re-embed
from scratch (useful when you switch embedding model).

Usage:
  # Voyage (default)
  export VOYAGE_API_KEY=pa-...
  python embed.py

  # Local Ollama
  ollama pull nomic-embed-text
  python embed.py --backend ollama --model nomic-embed-text

  # OpenAI
  export OPENAI_API_KEY=sk-...
  python embed.py --backend openai \\
      --base-url https://api.openai.com/v1 --model text-embedding-3-small

  # Gemini, reduced to 1024 dims to save storage
  export GEMINI_API_KEY=...
  python embed.py --backend gemini --dimensions 1024

Requirements:
  pip install lancedb
  pip install voyageai      (only for --backend voyage)
  pip install google-genai  (only for --backend gemini)
  pip install requests      (only for --backend openai or ollama)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("embed")

DEFAULT_VOYAGE_MODEL = "voyage-3.5-lite"
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_DB_PATH = "lancedb"
DEFAULT_TABLE_NAME = "chunks"
DEFAULT_BATCH_SIZE = 64
HTTP_TIMEOUT = 120


# --------------------------------------------------------------------------
# Embedding backends
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


class VoyageBackend:
    """Voyage AI embeddings (Anthropic's recommended provider for Claude)."""

    name = "voyage"

    def __init__(self, model: str, dimensions: int | None):
        try:
            import voyageai
        except ImportError:
            sys.exit("the voyageai SDK is not installed. Run: pip install voyageai")
        if not os.environ.get("VOYAGE_API_KEY"):
            sys.exit("VOYAGE_API_KEY is not set - export it before running")
        self.client = voyageai.Client()
        self.model = model
        self.dimensions = dimensions      # may be None until first call

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        kwargs = {"model": self.model, "input_type": input_type}
        if self.dimensions:
            kwargs["output_dimension"] = self.dimensions
        result = self.client.embed(texts, **kwargs)
        embeddings = result.embeddings
        if self.dimensions is None and embeddings:
            self.dimensions = len(embeddings[0])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class GeminiBackend:
    """Google Gemini embeddings via the google-genai SDK."""

    name = "gemini"

    def __init__(self, model: str, api_key: str | None, dimensions: int | None):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            sys.exit("the google-genai SDK is not installed. Run: pip install google-genai")

        if api_key:
            self.client = genai.Client(api_key=api_key)
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            self.client = genai.Client()
        else:
            sys.exit("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self._types = types
        self.model = model
        self.dimensions = dimensions

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config_kwargs: dict = {"task_type": task_type}
        if self.dimensions:
            config_kwargs["output_dimensionality"] = self.dimensions
        config = self._types.EmbedContentConfig(**config_kwargs)
        result = self.client.models.embed_content(
            model=self.model, contents=texts, config=config,
        )
        embeddings = [e.values for e in result.embeddings]
        if self.dimensions is None and embeddings:
            self.dimensions = len(embeddings[0])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class OllamaBackend(_HTTPBackend):
    """Local Ollama embeddings via the native /api/embed batch endpoint."""

    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float,
                 dimensions: int | None):
        self._init_http(timeout)
        self.url = base_url.rstrip("/") + "/api/embed"
        self.model = model
        self.dimensions = dimensions

    def _embed(self, texts: list[str]) -> list[list[float]]:
        body = self._post_with_retry(
            self.url, {"model": self.model, "input": texts},
            {"Content-Type": "application/json"},
        )
        embeddings = body.get("embeddings") or []
        if self.dimensions is None and embeddings:
            self.dimensions = len(embeddings[0])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class OpenAIBackend(_HTTPBackend):
    """Any OpenAI-compatible /v1/embeddings endpoint."""

    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str,
                 timeout: float, dimensions: int | None):
        self._init_http(timeout)
        self.url = base_url.rstrip("/") + "/embeddings"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model = model
        self.dimensions = dimensions

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        body = self._post_with_retry(self.url, payload, self.headers)
        embeddings = [item["embedding"] for item in body.get("data") or []]
        if self.dimensions is None and embeddings:
            self.dimensions = len(embeddings[0])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def make_backend(args: argparse.Namespace):
    """Build the chosen backend from CLI args."""
    if args.backend == "voyage":
        return VoyageBackend(model=args.model or DEFAULT_VOYAGE_MODEL,
                              dimensions=args.dimensions)

    if args.backend == "gemini":
        return GeminiBackend(model=args.model or DEFAULT_GEMINI_MODEL,
                              api_key=args.api_key, dimensions=args.dimensions)

    if args.backend == "ollama":
        if not args.model:
            sys.exit("--model is required for --backend ollama "
                     "(e.g. --model nomic-embed-text)")
        return OllamaBackend(model=args.model,
                              base_url=args.base_url or DEFAULT_OLLAMA_BASE_URL,
                              timeout=args.timeout,
                              dimensions=args.dimensions)

    if args.backend == "openai":
        if not args.model:
            sys.exit("--model is required for --backend openai")
        if not args.base_url:
            sys.exit("--base-url is required for --backend openai")
        api_key = args.api_key
        if not api_key and args.api_key_env:
            api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAIBackend(model=args.model, base_url=args.base_url,
                              api_key=api_key, timeout=args.timeout,
                              dimensions=args.dimensions)

    sys.exit(f"unknown backend: {args.backend}")


# --------------------------------------------------------------------------
# Vector store (LanceDB)
# --------------------------------------------------------------------------

class VectorStore:
    """Thin LanceDB wrapper for the chunks table."""

    def __init__(self, db_path: str, table_name: str):
        try:
            import lancedb
        except ImportError:
            sys.exit("lancedb is not installed. Run: pip install lancedb")
        self._lancedb = lancedb
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = (self.db.open_table(table_name)
                      if table_name in self.db.table_names() else None)

    def table_dimensions(self) -> int | None:
        if self.table is None:
            return None
        try:
            vec_type = self.table.schema.field("vector").type
            return getattr(vec_type, "list_size", None)
        except (KeyError, ValueError):
            return None

    def existing_chunk_ids(self, video_id: str | None = None) -> set[str]:
        """Chunk IDs already in the table; optionally filter to one video."""
        if self.table is None:
            return set()
        try:
            df = self.table.to_pandas()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read existing chunk_ids (%s); treating as empty", exc)
            return set()
        if "chunk_id" not in df.columns:
            return set()
        if video_id is not None and "video_id" in df.columns:
            df = df[df["video_id"] == video_id]
        return set(df["chunk_id"].dropna().tolist())

    def delete_video(self, video_id: str) -> int:
        """Remove all rows for a given video. Returns approximate count removed."""
        if self.table is None:
            return 0
        before = self.table.count_rows()
        # video_id from YouTube is alphanumeric + - / _, so this is safe.
        self.table.delete(f"video_id = '{video_id}'")
        return before - self.table.count_rows()

    def add(self, records: list[dict]) -> None:
        if not records:
            return
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=records)
            return
        expected = self.table_dimensions()
        actual = len(records[0]["vector"])
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"vector dimension mismatch: table '{self.table_name}' was created with "
                f"{expected}-d vectors but the backend produces {actual}-d. Use --force "
                f"to drop existing rows for the affected video, or --table-name to use "
                f"a fresh table for this model.")
        self.table.add(records)


# --------------------------------------------------------------------------
# Per-video processing
# --------------------------------------------------------------------------

def build_record(chunk: dict, vector: list[float], backend) -> dict:
    """Turn one chunks.jsonl entry plus its embedding into a table row."""
    return {
        "chunk_id": chunk["chunk_id"],
        "video_id": chunk["video_id"],
        "title": chunk.get("title") or "",
        "channel": chunk.get("channel") or "",
        "url": chunk.get("url") or "",
        "chapter": chunk.get("chapter") or "",
        "start": float(chunk["start"]),
        "end": float(chunk["end"]),
        "text": chunk["text"],
        "token_count": int(chunk.get("token_count", 0)),
        "has_ocr_text": bool(chunk.get("has_ocr_text", False)),
        # Lists of dicts are awkward in vector-DB schemas; stash as JSON and let
        # the retrieval layer parse it back when it wants the frame references.
        "keyframes_json": json.dumps(chunk.get("keyframes") or []),
        "model": backend.model,
        "backend": backend.name,
        "vector": [float(x) for x in vector],
    }


def process_video(video_dir: Path, backend, store: VectorStore,
                   batch_size: int, force: bool) -> str:
    """Embed and index one video's chunks. Returns ok/skipped/failed."""
    video_id = video_dir.name
    chunks_path = video_dir / "chunks.jsonl"
    if not chunks_path.exists():
        log.error("no chunks.jsonl in %s - run step 5 (chunk.py) first", video_id)
        return "failed"

    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    if not chunks:
        log.warning("%s has an empty chunks.jsonl", video_id)
        return "skipped"

    if force:
        removed = store.delete_video(video_id)
        if removed:
            log.info("  --force: removed %d existing row(s) for this video", removed)

    existing = store.existing_chunk_ids(video_id)
    todo = [c for c in chunks if c["chunk_id"] not in existing]
    if not todo:
        log.info("skip %s - all %d chunks already embedded", video_id, len(chunks))
        return "skipped"
    log.info("  embedding %d chunk(s) (%d already done)", len(todo), len(existing))

    embedded = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        try:
            vectors = backend.embed_documents([c["text"] for c in batch])
        except Exception as exc:  # noqa: BLE001
            log.error("  embedding batch failed at offset %d: %s", start, exc)
            return "failed" if embedded == 0 else "ok"
        if len(vectors) != len(batch):
            log.error("  backend returned %d vectors for %d chunks", len(vectors), len(batch))
            return "failed" if embedded == 0 else "ok"
        store.add([build_record(c, v, backend) for c, v in zip(batch, vectors)])
        embedded += len(batch)
        if embedded % (batch_size * 2) == 0 or embedded == len(todo):
            log.info("  ... %d/%d", embedded, len(todo))

    log.info("done  %s - embedded %d new chunk(s) (dim=%s)",
             video_id, embedded, backend.dimensions)
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
                  if d.is_dir() and (d / "chunks.jsonl").exists())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed chunks.jsonl into a LanceDB vector table.",
    )
    parser.add_argument("video_ids", nargs="*",
                        help="specific video ids to embed (default: all in --data-dir)")
    parser.add_argument("--data-dir", default="data",
                        help="root data directory (default: data)")

    parser.add_argument("--backend", choices=("voyage", "gemini", "ollama", "openai"),
                        default="voyage", help="embedding backend (default: voyage)")
    parser.add_argument("--model", default=None,
                        help=("model name passed to the backend (defaults: "
                              f"voyage={DEFAULT_VOYAGE_MODEL}, "
                              f"gemini={DEFAULT_GEMINI_MODEL}; required for ollama/openai)"))
    parser.add_argument("--base-url", default=None,
                        help=("endpoint base URL "
                              f"(defaults: ollama={DEFAULT_OLLAMA_BASE_URL}; "
                              "required for openai)"))
    parser.add_argument("--api-key", default=None,
                        help="API key override for openai/gemini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="env var to read the OpenAI key from (default: OPENAI_API_KEY)")
    parser.add_argument("--dimensions", type=int, default=None,
                        help="reduce output to this many dims (Matryoshka-capable models only)")
    parser.add_argument("--timeout", type=float, default=HTTP_TIMEOUT,
                        help=f"HTTP timeout for openai/ollama (default: {HTTP_TIMEOUT})")

    parser.add_argument("--db-path", default=DEFAULT_DB_PATH,
                        help=f"LanceDB directory (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME,
                        help=f"LanceDB table name (default: {DEFAULT_TABLE_NAME})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"chunks per embedding API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--force", action="store_true",
                        help="drop existing rows for each video and re-embed from scratch")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "lance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data_dir = Path(args.data_dir)
    targets = discover_video_dirs(data_dir, args.video_ids)
    if not targets:
        log.error("no videos with chunks.jsonl found in %s/ - run step 5 first", data_dir)
        return 1

    backend = make_backend(args)
    store = VectorStore(args.db_path, args.table_name)
    log.info("%d video(s) - backend=%s model=%s db=%s/%s",
             len(targets), backend.name, backend.model, args.db_path, args.table_name)

    tally = {"ok": 0, "skipped": 0, "failed": 0}
    for i, video_dir in enumerate(targets, 1):
        log.info("[%d/%d] %s", i, len(targets), video_dir.name)
        tally[process_video(video_dir, backend, store,
                            args.batch_size, args.force)] += 1

    log.info("finished: %(ok)d processed, %(skipped)d skipped, %(failed)d failed", tally)
    return 0 if tally["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())