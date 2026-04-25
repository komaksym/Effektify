# Counterfactual Coach — ROADMAP

> Long-horizon, global plan. Source of truth for **where this is going** and **what "done" looks like**. Slice-level execution belongs in `PLANS.md`. Detailed product spec lives in the PRD.

## Context

Effektify's existing AI agent answers retrospective questions ("what happened?"). The Counterfactual Coach answers the prospective one ("what would have happened with a different allocation?"). This roadmap sequences a ~48h build of a working demo on the Robyn open-source MMM benchmark — packaged so a reviewer (Max / Martin / Fredrik) can see in 3 minutes how it drops into Effektify as an App. The deliverable is a Loom + repo + live Streamlit app, optimised for "interview within 48h," not for production hardening.

## TLDR

1. **Ship**: working Streamlit recommendation app on the Robyn dataset + ~3-min Loom + public repo + live deploy.
2. **Stack**: deterministic Python (`scipy.curve_fit`, bootstrap, `scipy.optimize.minimize`) does the math; a 4-node LangGraph agent only translates structured JSON into grounded English.
3. **In scope**: per-channel saturation fits + 90% bootstrap CIs + constrained optimizer + diagnosis labels + grounded agent + comparison-table UI + untested-channel surfacing.
4. **Out of scope**: adstock, Bayesian hierarchies, incrementality, MTA, auth, persistence, week selector, slider override, untested-channel recommendations.
5. **Deferred (called out, not built)**: v2 = same pipeline against Effektify's connected customer data; v3 = cross-customer hierarchical pooling for cold-start channels.
6. **Done = ** comparison table on screen with `total_delta ≥ $2,000`, every quoted number regex-grounded against tool output, Loom under 3:10, email sent.

## High-Level Flow

```
Robyn CSV (data/robyn.csv)
   │
   ▼
ingest.py ──────────────► clean DataFrame {week, channel, spend, revenue}
   │
   ▼
curve_fit.py ───────────► joint fit  rev = base + Σ α_c · log(1 + s_c/β_c)
   │                       via scipy.optimize.least_squares  (one fit, K channels)
   ▼
bootstrap.py ───────────► 500 week-resamples, joint refit each → param 90% CIs
   │                       (cached as cache/channel_curves.json at startup)
   ▼
optimizer.py ───────────► constrained reallocation (5–60%, ≤1.5× hist max)
   │                       + diagnosis labels
   │                       (saturated / room-to-grow / near-optimum / flat / untested)
   ▼
agent_graph.py ─────────► LangGraph: diagnose → recommend → validate → format
   │                       tools = {fit_curves, simulate, find_optimum}
   ▼
grounding.py ───────────► regex check on agent prose vs structured JSON (1% tol, 1 retry)
   │
   ▼
ui_app.py (Streamlit) ──► comparison table + per-row curves + total Δ + CI band
```

The LLM's only jobs: (1) tool dispatch, (2) JSON → English. Numbers in prose must already exist in tool output — enforced.

## Hard Constraints

- **Deps ceiling**: `pandas`, `numpy`, `scipy`, `langgraph`, `langchain-anthropic` (or equivalent), `pydantic`, `streamlit`, `plotly` (or `matplotlib`). No `torch`, no PyMC, no Robyn library beyond its CSV.
- **Numerical purity**: all numbers come from deterministic Python. LLM output is regex-validated against tool output within 1% tolerance, 1 retry, then fail-loud (visible to the user).
- **Realism guards**: `0.05 × budget ≤ spend_i ≤ 0.60 × budget`; `spend_i ≤ 1.5 × max(historical_spend_i)`.
- **Quality gate**: channels with `R² < 0.3` excluded from optimization, surfaced as "needs more data".
- **Untested channels**: listed transparently, never given a recommendation.
- **Demo surface**: single hand-picked week, hardcoded. No selector. No upload UI. No slider.
- **Layering**: modeling layer never imports `langgraph` or `streamlit`; agent layer never imports `streamlit`. UI / domain / data are separated per project conventions.

## Milestones

Each milestone closes only after `ruff` + tests + a manual smoke check pass. If the milestone touches numerical code, also commit a sanity plot to `reports/`.

### M0 — Scaffolding & data ingestion
- **Goal**: repo skeleton + Robyn CSV loaded into a typed DataFrame.
- **Deliverable**: `data/robyn.csv` committed; `src/ingest.py` returns `DataFrame[week, channel, spend, revenue]`.
- **Acceptance**: `uv run python -m src.ingest` prints `(n_weeks, n_channels, total_spend, total_revenue)` matching Robyn's published numbers; ruff clean.

### M1 — Modeling core (curves + CIs)
- **Goal**: per-channel saturation curves jointly estimated with 90% bootstrap CIs, cached at startup.
- **Deliverable**: `src/curve_fit.py`, `src/bootstrap.py`, `src/simulator.py`. Cache file `cache/channel_curves.json` carries per-channel α/β + α_ci/β_ci + weeks_of_data + historical_spend_max, plus joint-model `base`/`base_ci`/`r_squared` and an `untested` list.
- **Note on the PRD pivot**: PRD §6/§8 said "fit per channel via `curve_fit`" — that's mathematically broken on aggregate revenue (univariate R² capped at corr², ≤ 0.20 on Robyn). M1 fits jointly via `scipy.optimize.least_squares`: one nonlinear regression `rev_t = base + Σ α_c · log(1+s_c,t/β_c)`. Per-channel (α, β) shape is preserved; the `R² < 0.3` quality gate moves from per-channel univariate R² to joint R² + per-channel CI tightness (M2 picks the exact rule).
- **Acceptance**:
  - Joint model R² ≥ 0.45 on Robyn. (Multistart-confirmed ceiling is ~0.484. The remaining variance correlates 0.65 with `competitor_sales_B`, an external regressor Robyn includes but real Effektify customers wouldn't have — so this is a realistic production ceiling, not a fitting bug. The R² < 0.3 quality gate in PRD §8 referred to per-channel univariate fit; with joint fit we use joint R² ≥ 0.45 as the trust signal and per-channel CI tightness as the per-channel quality flag, finalised in M2.)
  - Bootstrap on a synthetic 2-channel joint toy dataset recovers all (base, α_c, β_c) inside 90% CIs.
  - Untested channels (zero historical spend) surface in `JointFit.untested`, never fed into the optimizer.
  - Per-channel "attributed-revenue" curve plots saved to `reports/curves/`.

### M2 — Optimizer & diagnosis
- **Goal**: constrained reallocation emitting the PRD §7 optimizer output dict.
- **Deliverable**: `src/optimizer.py` produces full structured dict (channels with current/recommended/delta/diagnosis/marginal_roas/in_observed_range, untested_channels, low_confidence_channels, totals + CI).
- **Acceptance**:
  - On the picked demo week: `total_delta ≥ $2,000`.
  - No channel breaches the 5–60% / 1.5×-historical bounds.
  - Diagnosis labels match the PRD §8 rule table exactly.
  - Untested channels never assigned a recommendation.
  - Hand-built 3-channel test where the right answer is known recovers it.

### M3 — Agent layer (grounded translation only)
- **Goal**: 4-node LangGraph (`diagnose → recommend → validate → format`) producing Pydantic-validated `ChannelRecommendation`s with 100% grounded numbers.
- **Deliverable**: `src/agent_graph.py`, `src/grounding.py`. Pydantic schema enforced on output; grounding regex (`\$?[\d,]+(?:\.\d+)?`) checks every number against structured input within 1% tolerance; one retry on failure, then visible warning.
- **Acceptance**:
  - End-to-end run on cached optimizer output: every number in agent prose passes grounding.
  - Forced-failure test (mock agent that hallucinates a value) is caught and surfaced.

### M4 — UI (the demo)
- **Goal**: Streamlit page that loads with the demo week's recommendation already on screen, no clicks required.
- **Deliverable**: `src/ui_app.py`, runnable via `streamlit run src/ui_app.py`.
- **Acceptance**:
  - Sticky header: total spend + dollar delta visible above the fold.
  - One row per active channel with columns: Channel | Current | Recommended | Δ | Diagnosis | R² | Why.
  - Row click expands the saturation curve with current + recommended markers and shaded CI band.
  - Untested-channels section visible below the table.
  - Total delta with CI bar at the bottom.
  - Footer trust signals (data source, bootstrap count, 1.5× bound, v3 pooling note).
  - Screenshot saved to `reports/ui.png`.

### M5 — Demo prep & deploy
- **Goal**: ready-to-record state on a public URL.
- **Deliverable**:
  - Demo week hardcoded in `src/config.py`.
  - All curves + optimizer output + agent output pre-computed at startup (cached on disk so first paint is instant).
  - README with one-command setup: `uv sync && streamlit run src/ui_app.py`.
  - Streamlit Community Cloud deploy, URL captured.
- **Acceptance**: cold-clone → `uv sync` → app live in < 60s on a fresh machine; live URL renders identically.

### M6 — Loom + submission
- **Goal**: ship.
- **Deliverable**: ~3-min Loom following PRD §13 script (captions on); one-paragraph email to Fredrik with repo + live + Loom links and the dollar-delta hook in sentence #1.
- **Acceptance**:
  - Loom under 3:10.
  - Architecture diagram visible at the 1:15 beat.
  - "$X spent — same budget, $Y left on the table" said in the first 15s.
  - Email sent.

## Out of Scope (v1)
- Adstock / lagged effects / cross-channel interactions.
- Geo-experiments, incrementality, MTA, per-campaign granularity.
- Authentication, accounts, persistence, real customer data ingestion.
- Hierarchical pooling across customers (v3).
- Recommendations for untested channels (structural limit, surfaced not solved).
- Slider override, week selector, file-upload UI.
- Production-grade error handling beyond the happy path.

## Deferred — v2 / v3
- **v2**: wire the same pipeline into Effektify's connected data layer per customer; remove CSV affordance; add scheduled runs and persistence of recommendations.
- **v3**: cross-customer hierarchical pooling so cold-start channels inherit priors from similar brands. **Mentioned in Loom only — not built.**

## Risk → Mitigation (rolled up from PRD §12)
- Curves fit poorly → R² gate + transparent "needs more data" surfacing.
- Optimizer extremes → 5–60% / 1.5× hard caps.
- Agent hallucinates numbers → Pydantic + grounding regex + retry + fail-loud.
- Reviewer reads it as "just a chart" → Loom opens with the dollar number on screen at 0:05.
- Reviewer suspects causal overclaim → footer + Loom: "descriptive, not causal".
- Reviewer asks about new channels → explicit untested-channels block + v3 pooling beat.
- Reviewer assumes the LLM is the brain → architecture diagram on screen at 1:15 with explicit narration.

## Cut Order (if time slips)
1. Streamlit Cloud deploy (fall back to local + Loom).
2. CI band on the per-row curves (keep the curves themselves).
3. Per-channel curve hover/expansion (keep the table).
4. Anything that isn't: the table + the dollar number + the grounded prose.

## Verification (end-to-end)
- `uv run pytest` — unit tests (curve_fit toy recovery, optimizer hand-built scenario, grounding forced-failure).
- `uv run ruff check .` — lint clean.
- `uv run python -m src.precompute` — regenerates `cache/channel_curves.json` and `cache/optimizer_output.json` from scratch.
- `streamlit run src/ui_app.py` — open localhost, confirm comparison table renders, total delta ≥ $2,000, click a row → curve + CI band, untested section visible.
- Cold-clone walkthrough on a clean directory before recording the Loom.

## Critical Files (to be created)
- [data/robyn.csv](data/robyn.csv) — input dataset.
- [src/ingest.py](src/ingest.py) — CSV → DataFrame.
- [src/curve_fit.py](src/curve_fit.py) — `scipy.curve_fit` per channel.
- [src/bootstrap.py](src/bootstrap.py) — 500-resample CIs.
- [src/simulator.py](src/simulator.py) — evaluate fitted curves at any allocation.
- [src/optimizer.py](src/optimizer.py) — SLSQP constrained reallocation + diagnosis labels.
- [src/agent_graph.py](src/agent_graph.py) — 4-node LangGraph + Pydantic schemas.
- [src/grounding.py](src/grounding.py) — regex grounding check.
- [src/ui_app.py](src/ui_app.py) — Streamlit page.
- [src/config.py](src/config.py) — demo week + constants.
- [src/precompute.py](src/precompute.py) — startup cache builder.
- [PLANS.md](PLANS.md) — current slice's short-term plan, refreshed at each milestone.
