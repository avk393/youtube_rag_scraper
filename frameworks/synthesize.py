#!/usr/bin/env python3
"""synthesize.py - Stage 2 of the framework layer.

Merges many per-video extractions for ONE domain into a durable markdown
playbook: consensus methods, contested approaches (surfaced, never resolved
arbitrarily), a consolidated data checklist, and a source-spread note.

  extractions/*.json  (filtered to a domain)  ->  playbooks/<domain>.md

Perishable content is stripped before anything reaches the model — the
playbook is methodology only and must stay valid a year from now.

Usage:
  python frameworks/synthesize.py macro
  python frameworks/synthesize.py --all           # every domain with >=1 source
  python frameworks/synthesize.py valuation --model claude-sonnet-5

Requires ANTHROPIC_API_KEY (in .env or the environment).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:  # works both as a script and as `python -m frameworks.synthesize`
    from . import llm
    from .schema import DOMAINS
except ImportError:
    import llm
    from schema import DOMAINS

log = logging.getLogger("synthesize")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "prompts" / "synthesize.v1.md"


def load_extractions(extractions_dir: Path) -> list[dict]:
    """Load every extraction JSON, skipping unreadable files."""
    out: list[dict] = []
    for path in sorted(extractions_dir.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read %s: %s", path.name, exc)
    return out


def select_for_domain(extractions: list[dict], domain: str) -> list[dict]:
    """Keep non-low-value extractions tagged with `domain`, perishable stripped."""
    selected = []
    for ext in extractions:
        if ext.get("low_value"):
            continue
        if domain not in (ext.get("analytical_domains") or []):
            continue
        # Quarantine boundary: perishable opinion must never enter a playbook.
        selected.append({k: v for k, v in ext.items() if k != "perishable_content"})
    return selected


def synthesize_domain(
    client: "llm.anthropic.Anthropic",
    domain: str,
    extractions: list[dict],
    out_dir: Path,
    model: str,
) -> bool:
    sources = select_for_domain(extractions, domain)
    if not sources:
        log.warning("skip %s - no non-low-value extractions tagged with this domain", domain)
        return False

    prompt = llm.render_template(
        PROMPT_PATH,
        DOMAIN=domain,
        EXTRACTIONS_JSON=json.dumps(sources, indent=2, ensure_ascii=False),
    )
    log.info("synthesizing %s from %d source(s)...", domain, len(sources))
    try:
        markdown = llm.stream_text(client, model=model, prompt=prompt)
    except RuntimeError as exc:
        log.error("synthesis failed for %s: %s", domain, exc)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain}.md"
    out_path.write_text(markdown.strip() + "\n", encoding="utf-8")
    log.info("done  %s -> %s", domain, out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain", nargs="?", help=f"domain to synthesize (one of: {', '.join(DOMAINS)})")
    parser.add_argument("--all", action="store_true", help="synthesize every domain with at least one source")
    parser.add_argument("--extractions-dir", default=str(REPO_ROOT / "extractions"), help="input dir (default: extractions/)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "playbooks"), help="output dir (default: playbooks/)")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL, help=f"Claude model (default: {llm.DEFAULT_MODEL})")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.all and not args.domain:
        parser.error("give a domain, or pass --all")
    if args.domain and args.domain not in DOMAINS:
        parser.error(f"unknown domain '{args.domain}'. Valid: {', '.join(DOMAINS)}")

    extractions_dir = Path(args.extractions_dir)
    if not extractions_dir.exists():
        parser.error(f"extractions dir not found: {extractions_dir} (run extract.py first)")

    extractions = load_extractions(extractions_dir)
    if not extractions:
        log.info("no extractions found in %s", extractions_dir)
        return 0

    if args.all:
        present = sorted({d for ext in extractions if not ext.get("low_value") for d in (ext.get("analytical_domains") or [])})
        targets = [d for d in present if d in DOMAINS]
    else:
        targets = [args.domain]

    if not targets:
        log.info("no domains to synthesize")
        return 0

    client = llm.get_client()
    ok, failed = 0, 0
    for domain in targets:
        if synthesize_domain(client, domain, extractions, Path(args.out_dir), args.model):
            ok += 1
        else:
            failed += 1

    log.info("finished: %d ok, %d skipped/failed - output in %s/", ok, failed, args.out_dir)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
