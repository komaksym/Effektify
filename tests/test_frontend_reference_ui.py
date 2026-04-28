"""Static checks for reference-critical frontend UI details."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _read_frontend_file(name: str) -> str:
    return (FRONTEND / name).read_text()


def _css_block(css: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n\}}", css, re.S)
    assert match is not None, f"missing CSS selector {selector}"
    return match.group("body")


def test_static_shell_keeps_reference_rail_and_crumbs() -> None:
    html = _read_frontend_file("index.html")

    assert 'class="sidebar-separator"' in html
    assert 'aria-label="Agent"' in html
    assert 'aria-label="Knowledge"' in html
    assert "<strong>Effektify</strong>" in html
    assert "<span>Apps</span>" in html


def test_styles_preserve_editorial_paper_and_primary_week_card() -> None:
    css = _read_frontend_file("styles.css")

    body_block = _css_block(css, "body")
    assert "radial-gradient" not in body_block
    assert "linear-gradient" not in body_block

    view_button_block = _css_block(css, ".hero-view-button")
    assert "background: var(--text)" in view_button_block
    assert "color: var(--surface)" in view_button_block


def test_ledger_detail_uses_reference_reasoning_layout() -> None:
    app_js = _read_frontend_file("app.js")

    assert "Spend vs. revenue" in app_js
    assert "class=\"detail-note reasoning\"" in app_js
    assert "class=\"meta-grid\"" in app_js


def test_removed_reference_footer_stays_removed() -> None:
    html = _read_frontend_file("index.html")
    app_js = _read_frontend_file("app.js")
    css = _read_frontend_file("styles.css")

    assert 'id="footer"' not in html
    assert "renderFooter" not in app_js
    assert ".page-footer" not in css
    assert ".footer-surface" not in css


def test_channel_swatches_align_to_channel_name_line() -> None:
    css = _read_frontend_file("styles.css")

    channel_cell_block = _css_block(css, ".channel-cell")
    swatch_block = _css_block(css, ".channel-swatch")

    assert "align-items: flex-start" in channel_cell_block
    assert "margin-top: 9px" in swatch_block


def test_hero_equation_card_keeps_reference_scale() -> None:
    css = _read_frontend_file("styles.css")

    impact_grid_block = _css_block(css, ".impact-grid")
    equation_card_block = _css_block(css, ".equation-card")
    equation_number_block = _css_block(css, ".equation-number")

    assert "minmax(540px, 0.95fr)" in impact_grid_block
    assert "max-width: 760px" in equation_card_block
    assert "padding: 28px 32px 24px" in equation_card_block
    assert "font-size: calc(22px * var(--font-scale))" in equation_number_block


def test_workspace_content_column_stays_centered() -> None:
    css = _read_frontend_file("styles.css")

    workspace_block = _css_block(css, ".workspace")

    assert "max-width: 1380px" in workspace_block
    assert "margin: 0 auto" in workspace_block


def test_expanded_curve_shows_only_current_and_recommended_guides() -> None:
    app_js = _read_frontend_file("app.js")
    render_curve = app_js.split("function renderCurve(row) {", 1)[1].split(
        "function renderCurveGrid",
        1,
    )[0]

    assert "historyX" not in render_curve
    assert 'x1="${currentX}"' in render_curve
    assert 'x1="${recommendedX}"' in render_curve


def test_channel_summary_distinguishes_moves_from_holds() -> None:
    app_js = _read_frontend_file("app.js")

    assert "movedCount" in app_js
    assert "heldCount" in app_js
    assert "ready ·" not in app_js
    assert "more modeled revenue last week" not in app_js
