"""Thin Anthropic client + prompt-template plumbing shared by the framework stages.

Conventions mirror the existing pipeline scripts (see `caption_frames.py`):
an env-guarded client, a model constant with a `--model` override, and no
hidden magic. Prompts live as versioned `.md` files under `prompts/` and use
the project's `{{PLACEHOLDER}}` convention.

Two call paths:
  * `parse_structured` — extraction. Uses the Messages API structured-output
    mode so the reply is guaranteed to validate against a pydantic model; the
    stable instruction+schema block is sent as a cached `system` prompt and the
    per-video input as the user turn.
  * `stream_text` — synthesis. Streams a (potentially long) markdown playbook
    and returns its text.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Type, TypeVar

try:
    import anthropic
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("the anthropic SDK is not installed. Run: pip install anthropic")

try:  # optional: load ANTHROPIC_API_KEY etc. from a local .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass

from pydantic import BaseModel

# Default to the strongest current Claude model for the adversarial methodology
# reasoning these prompts require. Override per run with the scripts' --model
# flag (e.g. claude-sonnet-5) — don't hard-code a downgrade.
DEFAULT_MODEL = "claude-opus-5"

# Prompt templates that want the stable instructions cached separately from the
# volatile per-item input put this line between the two halves. Everything
# before it becomes the (cacheable) system prompt; everything after, the user
# turn. Templates without the marker are treated as a single user message.
SYSTEM_DELIMITER = "===PER_VIDEO_INPUT==="

T = TypeVar("T", bound=BaseModel)


def get_client() -> "anthropic.Anthropic":
    """Construct an Anthropic client, failing loudly if no key is configured."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set - add it to .env or export it before "
            "running (get one at https://console.anthropic.com/)."
        )
    return anthropic.Anthropic(max_retries=5)


def render_template(path: str | Path, /, **variables: object) -> str:
    """Read a prompt template and substitute `{{KEY}}` tokens."""
    text = Path(path).read_text(encoding="utf-8")
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def split_system_user(rendered: str) -> tuple[str, str | None]:
    """Split a rendered template into (system, user) on `SYSTEM_DELIMITER`.

    If the marker is absent, the whole text is the user turn and there is no
    dedicated system prompt.
    """
    if SYSTEM_DELIMITER in rendered:
        system, user = rendered.split(SYSTEM_DELIMITER, 1)
        return system.strip(), user.strip()
    return "", rendered.strip()


def _text_of(message: "anthropic.types.Message") -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def parse_structured(
    client: "anthropic.Anthropic",
    *,
    model: str,
    system: str,
    user: str,
    output_format: Type[T],
    max_tokens: int = 16000,
) -> T:
    """Run a structured-output extraction and return the validated model.

    The `system` block is cached (it's stable across items), so the instruction
    prefix is billed at read rates after the first call. Raises on refusal or a
    truncated / unparseable response.
    """
    system_param = (
        [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if system
        else anthropic.NOT_GIVEN
    )
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system_param,
        messages=[{"role": "user", "content": user}],
        output_format=output_format,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(
            f"model refused the request (category="
            f"{getattr(response.stop_details, 'category', None)})"
        )
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "response hit max_tokens before completing - raise max_tokens or "
            "shorten the input"
        )
    if response.parsed_output is None:
        raise RuntimeError(
            f"model returned no parseable structured output "
            f"(stop_reason={response.stop_reason})"
        )
    return response.parsed_output


def stream_text(
    client: "anthropic.Anthropic",
    *,
    model: str,
    prompt: str,
    max_tokens: int = 64000,
) -> str:
    """Stream a long free-text (markdown) response and return the text."""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError(
            f"model refused the request (category="
            f"{getattr(message.stop_details, 'category', None)})"
        )
    text = _text_of(message)
    if not text.strip():
        raise RuntimeError(
            f"model returned empty text (stop_reason={message.stop_reason})"
        )
    return text
