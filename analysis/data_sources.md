# Live data sources for `data_needed`

Every framework in an extraction lists the concrete `data_needed` inputs required
to *apply* it. This file maps those inputs to real data sources — the bridge from
YouTube-derived **method** to real-world **facts**.

**Hard rule:** facts about the current world come only from these sources, never
from the playbooks. Playbooks say *how* to analyze; they must never leak a stale
"the 10Y is at X%" into a live analysis. Keep the two streams separate.

Budget for now: **free tiers only.**

## Coverage by difficulty

### TODO:
- 10-K report tool query

### Easy / free — macro & rates
- **FRED** (Federal Reserve Bank of St. Louis). Free API key, deep history,
  clean series. Covers most macro `data_needed` verbatim:
  - "10Y-2Y treasury spread" → series `T10Y2Y`
  - "2Y yield" → `DGS2`, "10Y yield" → `DGS10`, "Fed funds" → `FEDFUNDS`/`DFF`
  - CPI → `CPIAUCSL`, unemployment → `UNRATE`, real GDP → `GDPC1`, etc.
  - Libraries: `fredapi`, or plain `requests` against `api.stlouisfed.org`.

### Easy / free — prices & technicals
- **Alpaca** — already connected as an MCP server in this environment; free
  market-data tier (quotes, bars/OHLCV). Good for prices, moving averages,
  volatility, and anything technical-domain frameworks ask for.
- **`yfinance`** — unofficial Yahoo Finance wrapper; convenient fallback for
  OHLCV and basic ratios. No key. Treat as best-effort (unofficial endpoint).

### Hard / expensive — company fundamentals  ⚠️ the bottleneck
Frameworks like "gross margin trend over 8 quarters" or "FCF yield vs peers"
need normalized quarterly fundamentals across many tickers. On free tiers this
is the constraint:
- **SEC EDGAR `companyfacts` API** — free, authoritative, but raw XBRL keyed by
  US-GAAP tags. Requires a normalization layer (map tags → metrics, align
  fiscal periods, compute derived ratios). Endpoint:
  `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` (send a descriptive
  User-Agent).
- **Paid alternatives** (deferred — budget is free-tier for now): Financial
  Modeling Prep, Tiingo. These return clean, pre-normalized quarterly
  fundamentals and remove the EDGAR wrangling.

**Decision:** until budget changes, build a small EDGAR normalization helper
(or accept coarser fundamentals) rather than pay for a fundamentals API.

## Stage 3 (deferred)

The retrieve-+-reason step is not designed yet (per the project roadmap). When
built, it will: classify what kind of analysis a question needs → pull the
relevant playbook(s) → fetch the `data_needed` from the sources above → reason
over live data. The trap to design against: the playbook leaking stale facts
into the analysis. Adapters for FRED / Alpaca / EDGAR will live in this
`analysis/` package.
