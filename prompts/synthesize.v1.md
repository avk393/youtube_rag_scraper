You are building a durable ANALYTICAL PLAYBOOK for a single domain by
synthesizing methodology extracted from many financial/macro videos.

DOMAIN: {{DOMAIN}}

You are given a JSON array of per-video extractions (the "frameworks",
"engages_with_other_views", and metadata fields; perishable_content has
already been stripped and must NOT appear in your output).

EXTRACTIONS:
{{EXTRACTIONS_JSON}}

Your goal is a coherent guide to HOW TO ANALYZE {{DOMAIN}}, not a summary of
videos. Principles:

1. MERGE overlapping methods into unified descriptions. If eight sources
   describe reading the yield curve the same way, that's one consolidated
   entry, noted as broad consensus.
2. SURFACE DISAGREEMENT explicitly. Where credible sources contradict each
   other, present BOTH positions with their reasoning and the conditions
   under which each may hold. DO NOT arbitrarily pick one. Contradiction is
   signal, not noise — the reader needs to know where practitioners diverge.
3. STAY METHODOLOGICAL. No predictions, no current market calls, no "the
   market is currently...". This playbook must be equally valid a year from
   now. It tells the reader what to look at and how to reason — never what
   the answer is today.
4. MAP TO DATA. For each method, keep the concrete data inputs needed to
   apply it, so it can later be run against live data.
5. FLAG WEAK FOUNDATIONS. If a method rests on a single low-confidence
   source, mark it as thinly supported rather than presenting it as settled.

Output a structured markdown document with these sections:

## Domain: {{DOMAIN}}

### What questions this playbook helps answer
(bulleted list of the analytical questions covered)

### Consensus methods
For each: a name, what it helps decide, the signals/metrics, the reasoning
in plain language, explicit decision rules, when it applies and when it
breaks, and the data inputs needed to run it. Note breadth of support.

### Contested approaches
For each point of disagreement: the question at issue, Position A (with
reasoning + supporting sources + conditions), Position B (same), and a
neutral note on what determines which view is more relevant in a given
situation. Resolve nothing you cannot resolve on the merits.

### Data checklist
A consolidated list of every live data input the above methods require, so
the analysis step knows exactly what to fetch.

### Known limitations & failure modes
Where these methods collectively tend to mislead, blind spots common across
sources, and cautions about over-reliance.

### Source spread
Brief note on how many distinct sources/channels informed this playbook and
how concentrated it is (many independent sources vs one loud channel), so
the reader can weight it.