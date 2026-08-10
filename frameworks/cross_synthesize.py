#!/usr/bin/env python3
"""cross_synthesize.py - Stage 2b: merge batch playbooks (two-stage synthesis).

For a domain with too many sources to synthesize in one call, run synthesize.py
over disjoint batches to get several partial playbooks, then merge them here.
This keeps each model call within a comfortable context budget instead of one
oversized call.

  playbooks/<domain>.batch*.md  ->  playbooks/<domain>.md

Usage:
  python frameworks/cross_synthesize.py macro playbooks/macro.batch1.md playbooks/macro.batch2.md
  python frameworks/cross_synthesize.py macro --inputs-glob 'playbooks/macro.batch*.md'

This is the deferred half of the pipeline: single-call synthesis (synthesize.py)
covers the current dozens-of-videos scale. Wire this in only once a single
domain's extractions stop fitting one call.

Requires ANTHROPIC_API_KEY (in .env or the environment).
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

try:
    from . import llm
    from .schema import DOMAINS
except ImportError:
    import llm
    from schema import DOMAINS

log = logging.getLogger("cross_synthesize")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "prompts" / "cross_synthesize.v1.md"


def merge_playbooks(client, domain: str, paths: list[Path], out_path: Path, model: str) -> bool:
    docs = []
    for i, path in enumerate(paths, 1):
        if not path.exists():
            log.error("input not found: %s", path)
            return False
        docs.append(f"=== INPUT PLAYBOOK {i} ({path.name}) ===\n\n{path.read_text(encoding='utf-8')}")

    prompt = llm.render_template(PROMPT_PATH, DOMAIN=domain, PLAYBOOKS="\n\n".join(docs))
    log.info("merging %d playbook(s) for %s...", len(paths), domain)
    try:
        markdown = llm.stream_text(client, model=model, prompt=prompt)
    except RuntimeError as exc:
        log.error("merge failed for %s: %s", domain, exc)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown.strip() + "\n", encoding="utf-8")
    log.info("done  %s -> %s", domain, out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain", help=f"domain being merged (one of: {', '.join(DOMAINS)})")
    parser.add_argument("inputs", nargs="*", help="batch playbook markdown files to merge")
    parser.add_argument("--inputs-glob", help="glob for batch playbook files (alternative to listing them)")
    parser.add_argument("--out", help="output path (default: playbooks/<domain>.md)")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL, help=f"Claude model (default: {llm.DEFAULT_MODEL})")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.domain not in DOMAINS:
        parser.error(f"unknown domain '{args.domain}'. Valid: {', '.join(DOMAINS)}")

    paths = [Path(p) for p in args.inputs]
    if args.inputs_glob:
        paths += [Path(p) for p in sorted(glob.glob(args.inputs_glob))]
    if len(paths) < 2:
        parser.error("need at least two batch playbooks to merge")

    out_path = Path(args.out) if args.out else REPO_ROOT / "playbooks" / f"{args.domain}.md"
    client = llm.get_client()
    return 0 if merge_playbooks(client, args.domain, paths, out_path, args.model) else 1


if __name__ == "__main__":
    sys.exit(main())
