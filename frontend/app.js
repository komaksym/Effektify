const DATA_URL = "./data/brief.json"

const state = {
  brief: null,
  openChannel: null,
  curveCache: new Map(),
}

const elements = {}

document.addEventListener("DOMContentLoaded", () => {
  elements.healthStatus = document.getElementById("healthStatus")
  elements.hero = document.getElementById("hero")
  elements.warnings = document.getElementById("warnings")
  elements.ledgerSection = document.getElementById("ledgerSection")
  elements.asides = document.getElementById("asides")
  elements.footer = document.getElementById("footer")
  elements.refreshBtn = document.getElementById("refreshBtn")
  elements.shareBtn = document.getElementById("shareBtn")

  elements.refreshBtn.addEventListener("click", () => refreshBrief(true))
  elements.shareBtn.addEventListener("click", shareCurrentUrl)
  elements.hero.addEventListener("click", onHeroClick)
  elements.ledgerSection.addEventListener("click", onLedgerClick)

  void refreshBrief(false)
})

async function refreshBrief(forceRefresh) {
  const requestUrl = forceRefresh
    ? `${DATA_URL}?ts=${Date.now()}`
    : DATA_URL

  setButtonState(elements.refreshBtn, "Refreshing...")

  try {
    const response = await fetch(requestUrl, { cache: "no-store" })
    if (!response.ok) {
      throw new Error(`Unable to load brief.json (${response.status})`)
    }

    state.brief = await response.json()
    state.curveCache.clear()

    const channels = state.brief.channels ?? []
    if (
      !state.openChannel ||
      !channels.some((row) => row.channel === state.openChannel)
    ) {
      state.openChannel = channels[0]?.channel ?? null
    }

    render()
    setButtonState(elements.refreshBtn, "Refreshed", 1200)
  } catch (error) {
    renderLoadError(error)
    setButtonState(elements.refreshBtn, "Refresh", 0)
  }
}

async function shareCurrentUrl() {
  try {
    await navigator.clipboard.writeText(window.location.href)
    setButtonState(elements.shareBtn, "Copied", 1200)
  } catch {
    setButtonState(elements.shareBtn, "Copy unavailable", 1400)
  }
}

function onHeroClick(event) {
  const target = event.target.closest("[data-scroll-target]")
  if (!target) {
    return
  }

  const { scrollTarget } = target.dataset
  if (scrollTarget === "ledger") {
    elements.ledgerSection.scrollIntoView({ behavior: "smooth" })
  }
}

function onLedgerClick(event) {
  const button = event.target.closest("[data-channel]")
  if (!button) {
    return
  }

  const { channel } = button.dataset
  state.openChannel = state.openChannel === channel ? null : channel
  renderLedgerSection()
}

function render() {
  renderHealthStatus()
  renderWarnings()
  renderHero()
  renderLedgerSection()
  renderAsides()
  renderFooter()
}

function renderLoadError(error) {
  const message = error instanceof Error ? error.message : "Unknown UI error"
  elements.healthStatus.textContent = "model check failed"
  elements.hero.innerHTML = `
    <div class="hero-surface">
      <p class="mono-label">Frontend data error</p>
      <h1 class="hero-title">We could not load the brief.</h1>
      <p class="hero-intro">${escapeHtml(message)}</p>
    </div>
  `
  elements.warnings.classList.remove("hidden")
  elements.warnings.innerHTML = `
    <strong>Unable to render the brief.</strong>
    <ul><li>${escapeHtml(message)}</li></ul>
  `
  elements.ledgerSection.innerHTML = ""
  elements.asides.innerHTML = ""
  elements.footer.innerHTML = ""
}

function renderHealthStatus() {
  const { meta } = state.brief
  elements.healthStatus.textContent = `model healthy · last refresh ${formatRelativeTime(
    meta.generated_at
  )}`
}

function renderWarnings() {
  const warnings = state.brief.meta.warnings ?? []
  if (!warnings.length) {
    elements.warnings.classList.add("hidden")
    elements.warnings.innerHTML = ""
    return
  }

  elements.warnings.classList.remove("hidden")
  elements.warnings.innerHTML = `
    <strong>Grounding warnings</strong>
    <ul>${warnings.map((warning) => `<li>${warning}</li>`).join("")}</ul>
  `
}

function renderHero() {
  const { meta, hero, channels, asides } = state.brief
  const readyCount = channels.length
  const needsDataCount = asides.low_confidence_channels.length
  const untriedCount = asides.untested_channels.length
  const upliftPercent =
    ((hero.total_recommended_revenue - hero.total_current_revenue) /
      hero.total_current_revenue) *
    100

  elements.hero.innerHTML = `
    <div class="hero-surface">
      <div class="hero-upper">
        <div class="hero-copy">
          <div class="hero-kicker mono-label">
            <span>Weekly brief № ${String(meta.brief_number).padStart(3, "0")}</span>
            <span>Week of ${formatHeroEyebrowWeek(meta.week_iso)}</span>
            <span>All marketing channels</span>
          </div>

          <h1 class="hero-title">What if you'd <span>spent differently?</span></h1>
          <p class="hero-intro">
            Same budget, smarter split. We compared your actual spend last week
            to every plausible alternative and found where the money would have
            worked harder. Every number below is calculated, not guessed.
          </p>
        </div>

        <div class="hero-side">
          <article class="hero-compare-card">
            <div class="hero-compare-grid">
              <div>
                <div class="mono-label">Demo week</div>
                <div class="hero-compare-value">${escapeHtml(meta.week_iso)}</div>
              </div>
              <div>
                <div class="mono-label">Compared against</div>
                <div class="hero-compare-value">Your actual spend</div>
              </div>
              <button
                class="hero-view-button"
                type="button"
                data-scroll-target="ledger"
              >
                View
              </button>
            </div>
          </article>

          <div class="hero-stamps">
            <div class="meta-chip">
              <span class="chip-label">Total budget</span>
              <span class="chip-value">${formatCurrency(hero.budget)}</span>
            </div>
            <div class="meta-chip">
              <span class="chip-label">Channels</span>
              <span class="chip-value">${readyCount} ready · ${needsDataCount} needs more data · ${untriedCount} untried</span>
            </div>
            <div class="meta-chip">
              <span class="chip-label">Numbers checked</span>
              <span class="chip-value">${meta.warnings.length ? `${meta.warnings.length} warning${meta.warnings.length > 1 ? "s" : ""}` : "All reconciled"}</span>
            </div>
            <div class="meta-chip">
              <span class="chip-label">Guardrails</span>
              <span class="chip-value">Active channels stay inside 5%-60% · historical cap 1.5×</span>
            </div>
          </div>
        </div>
      </div>

      <div class="hero-divider"></div>

      <div class="impact-grid">
        <div>
          <div class="impact-label-row">
            <span class="mono-label">Extra revenue you could have earned</span>
            <span class="impact-label-sep" aria-hidden="true">·</span>
            <span class="mono-label">Same ${formatCurrency(hero.budget)} budget, redistributed</span>
          </div>

          <div class="impact-number-row">
            <div class="impact-number ${deltaToneClass(hero.delta_point)}">
              ${renderImpactFigure(hero.delta_point)}
            </div>
            <div class="impact-copy">${formatImpactCopy(hero.delta_point)}</div>
          </div>

          <p class="impact-summary">
            With the same total spend, a smarter split between your channels
            would have generated about <strong>${formatSignedPercent(
              upliftPercent
            )}</strong> more modeled revenue last week.
            ${renderConfidenceSentence(hero.delta_low, hero.delta_high)}
          </p>

          <div class="impact-range">
            <div class="range-header">
              <span class="mono-label">Likely range (9 times out of 10)</span>
              <span class="mono-label">Across ${meta.bootstrap_count} simulations</span>
            </div>
            <div class="range-chart">
              ${renderRangeChart(hero.delta_low, hero.delta_point, hero.delta_high)}
            </div>
            <div class="range-footer">
              <span><span class="range-value">${formatSignedCurrency(hero.delta_low)}</span> cautious</span>
              <span><span class="range-value">${formatSignedCurrency(hero.delta_point)}</span> best estimate</span>
              <span><span class="range-value">${formatSignedCurrency(hero.delta_high)}</span> optimistic</span>
            </div>
          </div>
        </div>

        <article class="equation-card">
          <p class="equation-title mono-label">How we got the number</p>
          <div class="equation-strip">
            <div class="equation-value">
              <div class="equation-number">${formatCurrency(
                hero.total_recommended_revenue
              )}</div>
              <span>With smarter split</span>
            </div>
            <div class="equation-operator" aria-hidden="true">-</div>
            <div class="equation-value">
              <div class="equation-number">${formatCurrency(
                hero.total_current_revenue
              )}</div>
              <span>What you actually earned</span>
            </div>
            <div class="equation-operator" aria-hidden="true">=</div>
            <div class="equation-value">
              <div class="equation-number accent">${formatSignedCurrency(
                hero.delta_point
              )}</div>
              <span>Missed upside</span>
            </div>
          </div>
          <div class="equation-divider"></div>
          <p class="equation-note">
            Model fit on <strong>~${Math.max(
              1,
              Math.round(meta.history_weeks / 52)
            )} years</strong> of your weekly history.<br />
            Explains <strong>${fitNarrative(meta.joint_r_squared)}</strong> of the
            swings in revenue.
          </p>
        </article>
      </div>
    </div>
  `
}

function renderLedgerSection() {
  const rows = state.brief.channels ?? []

  elements.ledgerSection.innerHTML = `
    <div class="section-head">
      <div>
        <p class="section-overline mono-label">Per-channel reallocation</p>
        <h2 class="section-title">Per-channel reallocation</h2>
      </div>
      <p class="section-copy">
        Click any row to expand the saturation curve and read the agent's
        grounded reasoning.
      </p>
    </div>

    <div class="section-divider"></div>

    <div class="ledger-board">
      <div class="table-header">
        <span>#</span>
        <span>Channel</span>
        <span>Current</span>
        <span>Recommended</span>
        <span>Impact</span>
        <span>Spend → revenue</span>
        <span>Read</span>
        <span>Confidence</span>
        <span> </span>
      </div>
      ${rows
        .map((row, index) =>
          renderLedgerRow(row, index + 1, row.channel === state.openChannel)
        )
        .join("")}
    </div>
  `
}

function renderLedgerRow(row, index, isOpen) {
  const read = deriveRead(row)
  return `
    <div class="ledger-entry ${isOpen ? "is-open" : ""}">
      <button
        class="ledger-toggle"
        type="button"
        data-channel="${escapeAttribute(row.channel)}"
        aria-expanded="${String(isOpen)}"
      >
        <div class="ledger-grid">
          <div class="row-index">${String(index).padStart(2, "0")}</div>
          <div>
            <div class="channel-cell">
              <span class="channel-swatch ${escapeAttribute(row.channel)}"></span>
              <div>
                <span class="channel-name">${escapeHtml(
                  prettyChannelName(row.channel)
                )}</span>
                <div class="row-diagnosis">
                  <span class="tag-pill ${diagnosisToneClass(row)}">
                    ${escapeHtml(row.diagnosis_label)}
                  </span>
                </div>
              </div>
            </div>
          </div>
          <div class="ledger-metric">
            <div class="ledger-metric-label">Current</div>
            <div class="ledger-metric-value">${formatCurrency(
              row.current_spend
            )}</div>
          </div>
          <div class="ledger-metric">
            <div class="ledger-metric-label">Recommended</div>
            <div class="ledger-metric-value">${formatCurrency(
              row.recommended_spend
            )}</div>
          </div>
          <div>
            <div class="impact-pill ${impactPillTone(row.delta_revenue)}">
              ${formatCompactSignedCurrency(row.delta_revenue)}
            </div>
          </div>
          <div class="sparkline">
            ${renderSparkline(row)}
          </div>
          <div class="ledger-read ${read.tone}">
            <span class="read-dot"></span>
            <span>${read.label}</span>
          </div>
          <div class="confidence-wrap">
            ${renderConfidenceCell(row.confidence)}
          </div>
          <div class="chevron-wrap">
            <span class="chevron" aria-hidden="true">
              <svg viewBox="0 0 16 16">
                <path
                  d="M5 3.5 10 8l-5 4.5"
                  fill="none"
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="1.8"
                />
              </svg>
            </span>
          </div>
        </div>
      </button>
      ${
        isOpen
          ? `
            <div class="ledger-detail">
              <div class="ledger-detail-grid">
                <article class="detail-curve-card">
                  <p class="detail-term">Response curve</p>
                  <div class="detail-curve">
                    ${renderCurve(row)}
                  </div>
                  <div class="curve-legend">
                    <span><i class="legend-band"></i> 90% bootstrap band</span>
                    <span><i class="legend-line"></i> fitted curve</span>
                    <span><i class="legend-current"></i> current spend</span>
                    <span><i class="legend-rec"></i> recommended spend</span>
                  </div>
                </article>
                <article class="detail-metrics-card">
                  <div class="detail-metrics-grid">
                    <div>
                      <p class="detail-term">Current revenue</p>
                      <div class="detail-value">${formatCurrency(
                        row.current_revenue
                      )}</div>
                    </div>
                    <div>
                      <p class="detail-term">Recommended revenue</p>
                      <div class="detail-value">${formatCurrency(
                        row.recommended_revenue
                      )}</div>
                    </div>
                    <div>
                      <p class="detail-term">mROAS at current</p>
                      <div class="detail-value">${formatNumber(
                        row.marginal_roas_at_current,
                        1
                      )}</div>
                    </div>
                    <div>
                      <p class="detail-term">mROAS at recommended</p>
                      <div class="detail-value">${formatNumber(
                        row.marginal_roas_at_recommended,
                        1
                      )}</div>
                    </div>
                    <div>
                      <p class="detail-term">Historical max</p>
                      <div class="detail-value">${formatCurrency(
                        row.historical_max
                      )}</div>
                    </div>
                    <div>
                      <p class="detail-term">Observed range</p>
                      <div class="detail-value">${
                        row.in_observed_range
                          ? "Inside history"
                          : "Above history"
                      }</div>
                    </div>
                  </div>
                  <p class="detail-note">${row.grounded_reasoning}</p>
                </article>
              </div>
            </div>
          `
          : ""
      }
    </div>
  `
}

function renderAsides() {
  const { asides } = state.brief
  const untested = asides.untested_channels ?? []
  const lowConfidence = asides.low_confidence_channels ?? []

  elements.asides.innerHTML = `
    <div class="section-head">
      <div>
        <p class="section-overline mono-label">Channels we're leaving alone</p>
        <h2 class="section-title">Channels we're leaving alone</h2>
      </div>
      <p class="section-copy">
        We only move money where the data is solid. These channels stay at
        current spend.
      </p>
    </div>

    <div class="section-divider"></div>

    <div class="aside-grid">
      <article class="aside-card">
        <h3>Channels you haven't tried yet</h3>
        <p class="aside-subtitle mono-label">No spending history to learn from</p>
        ${
          untested.length
            ? `
              <div class="tag-list">
                ${untested
                  .map(
                    (channel) => `
                      <span class="tag-pill">${escapeHtml(
                        prettyChannelName(channel)
                      )}</span>
                    `
                  )
                  .join("")}
              </div>
              <p class="aside-body">
                We can't recommend a budget for a channel we've never seen
                perform. Want to know what these could do? Run a small
                <strong>pilot test</strong> and we'll learn from it.
              </p>
            `
            : `
              <p class="empty-state">
                No structurally untried channels in this demo week.
              </p>
            `
        }
      </article>

      <article class="aside-card">
        <h3>Channels with weak signal</h3>
        <p class="aside-subtitle mono-label">Not enough clear pattern to act on</p>
        ${
          lowConfidence.length
            ? `
              <div class="low-signal-list">
                ${lowConfidence
                  .map(
                    (row) => `
                      <div class="low-signal-item">
                        <div class="low-signal-head">
                          <div class="low-signal-title">
                            <span class="channel-swatch ${escapeAttribute(
                              row.channel
                            )}"></span>
                            <span>${escapeHtml(
                              prettyChannelName(row.channel)
                            )}</span>
                          </div>
                          ${renderConfidenceCell(row.confidence)}
                        </div>
                        <p class="aside-body">
                          Spend stays at <strong>${formatCurrency(
                            row.current_spend
                          )}</strong>. ${row.grounded_reasoning}
                        </p>
                      </div>
                    `
                  )
                  .join("")}
              </div>
            `
            : `
              <p class="empty-state">
                No low-confidence channels were frozen this week.
              </p>
            `
        }
      </article>
    </div>
  `
}

function renderFooter() {
  const { meta } = state.brief
  elements.footer.innerHTML = `
    <div class="footer-surface">
      <div class="footer-grid">
        <div>
          <p class="footer-term">Data source</p>
          <p class="footer-value">
            Robyn benchmark spend and revenue, flattened to one locked demo
            week.
          </p>
        </div>
        <div>
          <p class="footer-term">Bootstrap support</p>
          <p class="footer-value">
            ${meta.bootstrap_count} resamples back the confidence band shown in
            each active row.
          </p>
        </div>
        <div>
          <p class="footer-term">Guardrails</p>
          <p class="footer-value">
            Reallocation stays on the same weekly budget and respects the 1.5×
            historical spend ceiling.
          </p>
        </div>
        <div>
          <p class="footer-term">Scope note</p>
          <p class="footer-value">
            Untested channels stay out of the recommendation. Cross-customer
            pooling remains a later v3 step.
          </p>
        </div>
      </div>
    </div>
  `
}

function renderConfidenceCell(confidence) {
  return `
    <span class="confidence-cell">
      <span class="confidence-bars is-${escapeAttribute(confidence)}" aria-hidden="true">
        <span></span>
        <span></span>
        <span></span>
      </span>
      <span class="confidence-text">${escapeHtml(confidence)}</span>
    </span>
  `
}

function deriveRead(row) {
  if (
    row.diagnosis_label === "flat / needs more data" ||
    row.confidence === "low"
  ) {
    return { label: "Needs more data", tone: "hold" }
  }
  if (row.action === "increase") {
    return { label: "Spend more", tone: "more" }
  }
  if (row.action === "decrease") {
    return { label: "Spend less", tone: "less" }
  }
  return { label: "Right where it should be", tone: "hold" }
}

function diagnosisToneClass(row) {
  if (row.action === "increase") {
    return "grow"
  }
  if (row.action === "decrease") {
    return "cut"
  }
  return "hold"
}

function impactPillTone(value) {
  if (value > 0) {
    return "positive"
  }
  if (value < 0) {
    return "negative"
  }
  return "flat"
}

function renderSparkline(row) {
  const curve = getCurveData(row)
  const width = 142
  const height = 42
  const paddingX = 6
  const paddingY = 6
  const xScale = scaleLinear(0, curve.maxSpend, paddingX, width - paddingX)
  const yScale = scaleLinear(0, curve.maxRevenue, height - paddingY, paddingY)

  const line = toLinePath(
    curve.xs.map((x, index) => ({ x, y: curve.point[index] })),
    xScale,
    yScale
  )

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeAttribute(
      `${row.channel} sparkline`
    )}">
      <path
        d="${line}"
        fill="none"
        stroke="#d66b12"
        stroke-width="2"
        stroke-linecap="round"
      />
      <circle
        cx="${xScale(row.current_spend)}"
        cy="${yScale(row.current_revenue)}"
        r="3.7"
        fill="#21180f"
      />
      <circle
        cx="${xScale(row.recommended_spend)}"
        cy="${yScale(row.recommended_revenue)}"
        r="3.7"
        fill="#3d9347"
      />
    </svg>
  `
}

function renderCurve(row) {
  const curve = getCurveData(row)
  const width = 760
  const height = 356
  const padding = { top: 16, right: 28, bottom: 32, left: 18 }
  const xScale = scaleLinear(
    0,
    curve.maxSpend,
    padding.left,
    width - padding.right
  )
  const yScale = scaleLinear(
    0,
    curve.maxRevenue,
    height - padding.bottom,
    padding.top
  )

  const band = toBandPath(curve.xs, curve.low, curve.high, xScale, yScale)
  const line = toLinePath(
    curve.xs.map((x, index) => ({ x, y: curve.point[index] })),
    xScale,
    yScale
  )
  const historyX = xScale(row.historical_max)

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeAttribute(
      `${row.channel} response curve`
    )}">
      <line
        x1="${padding.left}"
        y1="${height - padding.bottom}"
        x2="${width - padding.right}"
        y2="${height - padding.bottom}"
        stroke="#d8ccb9"
        stroke-width="1"
      />
      <path d="${band}" fill="rgba(214, 107, 18, 0.14)" />
      <path
        d="${line}"
        fill="none"
        stroke="#d66b12"
        stroke-width="4"
        stroke-linecap="round"
      />
      <line
        x1="${historyX}"
        y1="${padding.top}"
        x2="${historyX}"
        y2="${height - padding.bottom}"
        stroke="#948777"
        stroke-width="1.8"
        stroke-dasharray="6 7"
      />
      <circle
        cx="${xScale(row.recommended_spend)}"
        cy="${yScale(row.recommended_revenue)}"
        r="9"
        fill="#3d9347"
      />
      <circle
        cx="${xScale(row.current_spend)}"
        cy="${yScale(row.current_revenue)}"
        r="9"
        fill="#21180f"
      />
      <text
        x="${historyX + 8}"
        y="${padding.top + 20}"
        fill="#6f6558"
        font-family="JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
        font-size="12"
      >
        history max
      </text>
      <text
        x="${padding.left}"
        y="${height - 6}"
        fill="#6f6558"
        font-family="JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
        font-size="12"
      >
        0
      </text>
      <text
        x="${width - padding.right}"
        y="${height - 6}"
        fill="#6f6558"
        font-family="JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
        font-size="12"
        text-anchor="end"
      >
        spend
      </text>
    </svg>
  `
}

function renderRangeChart(low, point, high) {
  const maxAbs = Math.max(Math.abs(low), Math.abs(high), 1)
  const span = maxAbs * 1.4
  const xmin = -span
  const xmax = span
  const toPercent = (value) => ((value - xmin) / (xmax - xmin)) * 100
  const lowPct = toPercent(low)
  const highPct = toPercent(high)
  const pointPct = toPercent(point)
  const zeroPct = toPercent(0)

  return `
    <div class="ci-track" role="img" aria-label="Likely revenue delta range">
      <div class="ci-axis"></div>
      <div class="ci-zero" style="left:${zeroPct}%"></div>
      <div class="ci-band" style="left:${lowPct}%; width:${Math.max(
        highPct - lowPct,
        0
      )}%"></div>
      <div class="ci-point" style="left:${pointPct}%"></div>
    </div>
  `
}

function getCurveData(row) {
  const cached = state.curveCache.get(row.channel)
  if (cached) {
    return cached
  }

  const maxSpend = Math.max(
    row.historical_max * 1.15,
    row.current_spend,
    row.recommended_spend,
    1
  )
  const xs = Array.from({ length: 60 }, (_, index) => (maxSpend * index) / 59)
  const point = xs.map((x) => saturation(x, row.alpha, row.beta))
  const low = []
  const high = []

  for (const x of xs) {
    const samples = row.alpha_samples.map((alpha, index) =>
      saturation(x, alpha, row.beta_samples[index])
    )
    samples.sort((left, right) => left - right)
    low.push(percentileSorted(samples, 0.05))
    high.push(percentileSorted(samples, 0.95))
  }

  const maxRevenue = Math.max(
    ...high,
    ...point,
    row.current_revenue,
    row.recommended_revenue,
    1
  )

  const result = { xs, point, low, high, maxSpend, maxRevenue }
  state.curveCache.set(row.channel, result)
  return result
}

function saturation(spend, alpha, beta) {
  return alpha * Math.log1p(spend / beta)
}

function percentileSorted(sortedValues, quantile) {
  if (!sortedValues.length) {
    return 0
  }
  if (sortedValues.length === 1) {
    return sortedValues[0]
  }

  const position = (sortedValues.length - 1) * quantile
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) {
    return sortedValues[lower]
  }

  const weight = position - lower
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight
}

function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  const domainSpan = domainMax - domainMin || 1
  const rangeSpan = rangeMax - rangeMin
  return (value) => rangeMin + ((value - domainMin) / domainSpan) * rangeSpan
}

function toLinePath(points, xScale, yScale) {
  return points
    .map((point, index) => {
      const prefix = index === 0 ? "M" : "L"
      return `${prefix}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(
        2
      )}`
    })
    .join(" ")
}

function toBandPath(xs, low, high, xScale, yScale) {
  const upper = xs.map((x, index) => ({ x, y: high[index] }))
  const lower = xs
    .map((x, index) => ({ x, y: low[index] }))
    .reverse()

  return `${toLinePath(upper, xScale, yScale)} ${lower
    .map(
      (point) =>
        `L${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)}`
    )
    .join(" ")} Z`
}

function prettyChannelName(value) {
  if (value === "tv") {
    return "TV"
  }
  if (value === "ooh") {
    return "Ooh"
  }
  return value
    .split(/[_-]/g)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ")
}

function fitNarrative(rSquared) {
  if (rSquared >= 0.6) {
    return "most of the swings"
  }
  if (rSquared >= 0.45) {
    return "roughly half"
  }
  if (rSquared >= 0.3) {
    return "some meaningful share"
  }
  return "a limited slice"
}

function renderConfidenceSentence(low, high) {
  if (low > 0) {
    return "Even the cautious end of the range stays positive."
  }
  if (high < 0) {
    return "Even the optimistic end of the range stays below zero, so the current split still looks safer."
  }
  return "The midpoint is positive, but the 90% range still crosses zero, so treat the upside as directional rather than guaranteed."
}

function formatImpactCopy(delta) {
  if (delta > 0) {
    return "left on<br />the table"
  }
  if (delta < 0) {
    return "at risk<br />with this mix"
  }
  return "with no<br />clear upside"
}

function formatRelativeTime(isoString) {
  const diffMs = Date.now() - new Date(isoString).getTime()
  const minutes = Math.max(1, Math.round(diffMs / 60000))
  if (minutes < 60) {
    return `${minutes} min ago`
  }
  const hours = Math.round(minutes / 60)
  if (hours < 24) {
    return `${hours} hr ago`
  }
  const days = Math.round(hours / 24)
  return `${days} day${days === 1 ? "" : "s"} ago`
}

function formatHeroWeek(isoString) {
  const date = new Date(`${isoString}T00:00:00`)
  const parts = new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  }).formatToParts(date)

  const day = parts.find((part) => part.type === "day")?.value ?? ""
  const month = parts.find((part) => part.type === "month")?.value ?? ""
  const year = parts.find((part) => part.type === "year")?.value ?? ""

  return `${day} ${month.toUpperCase()}<br />${year}`
}

function formatHeroEyebrowWeek(isoString) {
  const date = new Date(`${isoString}T00:00:00`)
  const parts = new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  }).formatToParts(date)

  const day = parts.find((part) => part.type === "day")?.value ?? ""
  const month = parts.find((part) => part.type === "month")?.value ?? ""
  const year = parts.find((part) => part.type === "year")?.value ?? ""

  return `${day} ${month.toUpperCase()} ${year}`
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value)
}

function formatSignedCurrency(value) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : ""
  return `${sign}${formatCurrency(Math.abs(value))}`
}

function renderImpactFigure(value) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : ""
  const signMarkup = sign
    ? `<span class="impact-sign">${sign}</span>`
    : ""

  return `${signMarkup}<span class="impact-amount">${formatCurrency(
    Math.abs(value)
  )}</span>`
}

function formatCompactSignedCurrency(value) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : ""
  const abs = Math.abs(value)
  if (abs < 1000) {
    return `${sign}${formatCurrency(abs)}`
  }

  return `${sign}${new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(abs)}`
}

function formatSignedPercent(value) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : ""
  return `${sign}${Math.abs(value).toFixed(2)}%`
}

function formatNumber(value, digits) {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)
}

function deltaToneClass(value) {
  if (value > 0) {
    return "delta-positive"
  }
  if (value < 0) {
    return "delta-negative"
  }
  return "delta-flat"
}

function setButtonState(button, text, resetAfterMs = 0) {
  const original = button.dataset.originalLabel || button.textContent
  button.dataset.originalLabel = original
  button.textContent = text

  if (resetAfterMs > 0) {
    window.setTimeout(() => {
      button.textContent = original
    }, resetAfterMs)
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function escapeAttribute(text) {
  return escapeHtml(text)
}
