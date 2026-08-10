"""Pydantic models for the per-video extraction JSON.

These mirror the schema documented in `prompts/extract.v1.md` field-for-field.
They double as the structured-output schema handed to the Anthropic API (via
`client.messages.parse(output_format=Extraction)`), so a successful call is
guaranteed to validate — there is no separate JSON-parsing/repair step.

Field descriptions here become JSON-schema `description`s, which also nudge the
model toward the right semantics for each field.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# The domains used to tag/route frameworks. Kept as a module-level constant so
# extract/synthesize can share the canonical list.
Domain = Literal[
    "macro",
    "valuation",
    "technical",
    "sector",
    "company_specific",
    "risk_management",
    "trading_strategy",
    "portfolio_construction",
    "behavioral",
    "other",
]

DOMAINS: tuple[str, ...] = (
    "macro",
    "valuation",
    "technical",
    "sector",
    "company_specific",
    "risk_management",
    "trading_strategy",
    "portfolio_construction",
    "behavioral",
    "other",
)


class Framework(BaseModel):
    """One reusable analytical method mined from the transcript."""

    name: str = Field(description="Short label for the method.")
    summary: str = Field(description="1-2 sentences: what it helps you decide.")
    signals_and_metrics: list[str] = Field(
        description="Specific things to look at."
    )
    reasoning_chain: str = Field(
        description="How signals connect to conclusions — the causal/logical "
        "steps, in your own words."
    )
    decision_rules: list[str] = Field(
        description="Explicit if-then rules stated or implied."
    )
    conditions_of_applicability: str = Field(
        description="Regime/context dependence: when does this hold and when "
        "does it break?"
    )
    data_needed: list[str] = Field(
        description="What live data you'd fetch to APPLY this (e.g. "
        '"10Y-2Y treasury spread", "gross margin trend over 8 quarters").'
    )
    speaker_caveats: list[str] = Field(
        description="Hedges/limitations the speaker acknowledged."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well-reasoned vs merely asserted this was."
    )


class EngagesWith(BaseModel):
    """Where the speaker agrees/disagrees with a common position."""

    claim: str
    stance: Literal["supports", "contradicts", "nuances"]
    their_reasoning: str


class Perishable(BaseModel):
    """QUARANTINED content — captured but never used as framework."""

    type: Literal["prediction", "price_target", "stock_pick", "timing_call", "other"]
    content: str
    as_of_date: str = Field(
        description="Usually the publish_date; this is timestamped."
    )


class Extraction(BaseModel):
    """Full structured extraction for a single video."""

    video_id: str
    title: str
    channel: str
    publish_date: str

    low_value: bool = Field(
        description="True if no reusable method is present."
    )
    low_value_reason: Optional[str] = None

    analytical_domains: list[Domain] = Field(
        description="Subset of the canonical domains this video addresses."
    )
    core_questions: list[str] = Field(
        description="The analytical questions this video addresses, phrased as "
        "reusable questions."
    )

    frameworks: list[Framework]
    engages_with_other_views: list[EngagesWith]
    perishable_content: list[Perishable]

    quality_notes: str = Field(
        description="Honest read: does this speaker show their work, or just "
        "assert with confidence?"
    )
