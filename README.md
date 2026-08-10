# youtube_rag_scraper

1. Extract youtube video data. Utilize script pipeline in /yt_data_extraction. A 6-step pipeline that turns YouTube videos into a searchable RAG vector database, combining spoken transcripts with visual frame captions.
2. 
---

## Pipeline for Youtube Data Extraction ./yt_data_extraction

### Step 1 — `ingestion.py`: Download video assets
Downloads a YouTube video (or playlist/channel) and extracts its video file, mono 16 kHz audio track, subtitle captions, and a slim metadata JSON. Each video lands in `data/<video_id>/` with predictable filenames for later steps.
```bash
pip install yt-dlp
# ffmpeg must be on PATH
python ingestion.py https://www.youtube.com/watch?v=VIDEO_ID
python ingestion.py --urls-file urls.txt          # batch from a file
python ingestion.py https://youtube.com/playlist?list=PL...  # whole playlist
```

### Step 2 — `transcribe.py`: Generate timestamped transcript
Transcribes the audio via `faster-whisper` (primary) or parses the YouTube `.vtt` captions saved in step 1 (fallback). Writes `transcript.json` with segment- and word-level timestamps used by the chunker in step 5.
```bash
pip install faster-whisper
python transcribe.py                          # all videos in data/
python transcribe.py VIDEO_ID                 # one video
python transcribe.py --model large-v3         # higher accuracy
python transcribe.py --captions-only          # skip Whisper, use VTT only
python transcribe.py --device cuda --compute-type float16  # GPU
```

### Step 3 — `keyframes.py`: Extract deduplicated keyframes
Detects scene changes (PySceneDetect) and adds periodic safety-net samples, then deduplicates near-identical frames using perceptual hashing + mean color. Saves unique frames as JPEGs and writes `keyframes.json` with timestamps and occurrence lists.
```bash
pip install scenedetect imagehash
python keyframes.py                           # all videos in data/
python keyframes.py VIDEO_ID                  # one video
python keyframes.py --interval 30 --threshold 27
```

### Step 4 — `caption_frames.py`: Caption keyframes with a vision model
Sends each unique keyframe JPEG to a vision model and records verbatim OCR text plus a short visual description. Supports Anthropic (default), Gemini, Ollama, and any OpenAI-compatible endpoint. Writes `visuals.json` and resumes safely if interrupted.
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python caption_frames.py                      # Claude Haiku (default)

# Gemini
pip install google-genai
export GEMINI_API_KEY=...
python caption_frames.py --backend gemini

# Local Ollama
ollama pull qwen2.5vl
python caption_frames.py --backend ollama --model qwen2.5vl
```

### Step 5 — `chunk.py`: Fuse and chunk into RAG-ready segments
Merges the transcript segments from step 2 with the visual events from steps 3–4 onto a single timeline, then splits the result into token-capped chunks. Chapter boundaries from YouTube metadata are hard breaks; visual change points are preferred soft break points. Writes `visual_transcript.json` and `chunks.jsonl`.
```bash
pip install tiktoken
python chunk.py                               # all videos with visuals.json
python chunk.py VIDEO_ID
python chunk.py --max-tokens 800 --overlap-tokens 100
```

### Step 6 — `embed.py`: Embed chunks into LanceDB
Reads each video's `chunks.jsonl`, embeds every chunk with a chosen embedding model in batches, and upserts the vectors into a local LanceDB table. Supports Voyage AI (default), Gemini, Ollama, and any OpenAI-compatible endpoint. Re-running is safe and resumes from where it left off.
```bash
pip install lancedb voyageai
export VOYAGE_API_KEY=pa-...
python embed.py                               # Voyage (default)

# Local Ollama
ollama pull nomic-embed-text
python embed.py --backend ollama --model nomic-embed-text

# OpenAI
export OPENAI_API_KEY=sk-...
python embed.py --backend openai \
    --base-url https://api.openai.com/v1 --model text-embedding-3-small
```
---

## Full pipeline (quick reference)

```bash
python ingestion.py <youtube-url>
python transcribe.py
python keyframes.py
python caption_frames.py
python chunk.py
python embed.py
```

Output ends up in `lancedb/` — a local vector store ready to query from your RAG agent.

---

## Framework layer (`frameworks/`)

A second, **separate** track that reuses stages 1–2 (transcript) and optionally
stage 4 (chart captions) (then diverges from the generic RAG architected in the 
original pipeline for YouTube Data Extraction above). 
Instead of embedding raw chunks, it distills each transcript into a structured **analytical
framework** and merges frameworks per domain into a **playbook**. The governing
rule: durable *methodology* is extracted and kept; perishable *opinion* (price
targets, timing calls) is quarantined and never enters a playbook. YouTube is the
source of method only — current facts come from real data APIs (see
[`analysis/data_sources.md`](analysis/data_sources.md)).

Requires `ANTHROPIC_API_KEY` (in `.env` or the environment).

### Stage 1 — `frameworks/extract.py`: transcript → structured extraction

Mines reusable method (signals, reasoning chains, decision rules, and the live
`data_needed` to apply each), quarantines perishable opinion, and honestly flags
low-value/hype videos. Output is schema-validated JSON (via Anthropic structured
outputs) per video.

```bash
python frameworks/extract.py                 # all videos in data/ with a transcript
python frameworks/extract.py VIDEO_ID        # one video
python frameworks/extract.py --force         # re-extract
python frameworks/extract.py --model claude-sonnet-5   # cheaper model
```

Writes `extractions/<video_id>.json`.

### Stage 2 — `frameworks/synthesize.py`: extractions → domain playbook

Merges every extraction tagged with a domain into a markdown playbook: consensus
methods, contested approaches (surfaced, never arbitrarily resolved), a
consolidated data checklist, and a source-spread note. Perishable content is
stripped before anything reaches the model.

```bash
python frameworks/synthesize.py macro        # one domain
python frameworks/synthesize.py --all        # every domain with >=1 source
```

Writes `playbooks/<domain>.md`. For a domain too large for one call, synthesize
in batches then merge with `frameworks/cross_synthesize.py`.

### Layout

```
prompts/        versioned prompt templates (extract.v1.md, synthesize.v1.md, ...)
extractions/    per-video structured JSON (durable knowledge — git-tracked)
playbooks/      one markdown playbook per domain
analysis/       stage-3 (retrieve + live data + reason) and data adapters — deferred
```

Domains: `macro, valuation, technical, sector, company_specific, risk_management,
trading_strategy, portfolio_construction, behavioral, other`.


# TODO:
- Improve video chart interpretation. Feed correlating transcript into LLM with images to interpret chart context better