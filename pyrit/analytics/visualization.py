# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Visualization utilities for analytics results.

This module provides interactive HTML report generation for ``AnalysisResult``
using Plotly. Plotly is lazy-imported so the core analytics module has no
hard dependency on it. If plotly is not installed, a clear error is raised
at call time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from pyrit.analytics.result_analysis import AnalysisResult, AttackStats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ASR_COLORSCALE: list[list[object]] = [
    [0.0, "#e74c3c"],
    [0.5, "#f39c12"],
    [1.0, "#2ecc71"],
]

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         max-width:1200px;margin:0 auto;padding:24px;background:#f0f2f5;color:#333}}
    h1{{margin-bottom:4px;font-size:1.8em}}
    .subtitle{{color:#999;font-size:.9em;margin-bottom:24px}}
    .card{{background:white;border-radius:10px;padding:24px;margin-bottom:24px;
           box-shadow:0 1px 6px rgba(0,0,0,.08)}}
    .kpi-row{{display:flex;gap:16px;flex-wrap:wrap}}
    .kpi{{flex:1;min-width:110px;text-align:center;padding:16px;
          border-radius:8px;background:#f8f9fa}}
    .kpi-value{{font-size:2.2em;font-weight:700;line-height:1}}
    .kpi-label{{color:#666;font-size:.85em;margin-top:6px}}
    .green{{color:#2ecc71}}.yellow{{color:#f39c12}}.red{{color:#e74c3c}}
    h2{{margin:0 0 20px;font-size:.85em;text-transform:uppercase;
        letter-spacing:.06em;color:#999;border-bottom:1px solid #f0f0f0;
        padding-bottom:10px}}
  </style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>
"""

_DEFAULT_COMPOSITE_DIMS: list[tuple[str, str]] = [
    ("harm_category", "attack_type"),
    ("harm_category", "converter_type"),
    ("attack_type", "converter_type"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_plotly() -> tuple[Any, Any]:
    """
    Lazy-import plotly and return (graph_objects, io) modules.

    Returns:
        tuple: The ``plotly.graph_objects`` and ``plotly.io`` modules.

    Raises:
        ImportError: If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        return go, pio
    except ImportError as err:
        raise ImportError("plotly is required for HTML report features. Install it with:  pip install plotly") from err


def _asr_css_class(rate: Optional[float]) -> str:
    """
    Return a CSS class name for colour-coding a success rate.

    Returns:
        str: ``"green"``, ``"yellow"``, ``"red"``, or ``""`` when unknown.
    """
    if rate is None:
        return ""
    if rate >= 0.6:
        return "green"
    if rate >= 0.3:
        return "yellow"
    return "red"


def _asr_bar_color(rate: Optional[float]) -> str:
    """
    Return an RGB colour string for a bar based on success rate.

    Returns:
        str: CSS colour string.
    """
    if rate is None:
        return "rgba(180,180,180,0.5)"
    if rate >= 0.6:
        return "#2ecc71"
    if rate >= 0.3:
        return "#f39c12"
    return "#e74c3c"


def _is_sparse(
    dim_data: dict[Any, AttackStats],
    *,
    threshold: float = 0.5,
) -> bool:
    """
    Return True when more than *threshold* fraction of cells lack decided data.

    Returns:
        bool: True if the dimension data is too sparse to display.
    """
    if not dim_data:
        return True
    none_count = sum(1 for s in dim_data.values() if s.success_rate is None)
    return none_count / len(dim_data) > threshold


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_summary_html(*, result: AnalysisResult, title: str) -> str:
    """
    Build the top-level KPI summary card as an HTML string.

    Returns:
        str: HTML string for the summary card.
    """
    o = result.overall
    rate_str = f"{o.success_rate:.0%}" if o.success_rate is not None else "N/A"
    cls = _asr_css_class(o.success_rate)
    total = o.successes + o.failures + o.undetermined
    kpis = [
        (rate_str, "Overall ASR", cls),
        (str(total), "Total Attacks", ""),
        (str(o.successes), "Successes", "green"),
        (str(o.failures), "Failures", "red"),
        (str(o.undetermined), "Undetermined", ""),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-value {c}">{v}</div><div class="kpi-label">{lbl}</div></div>'
        for v, lbl, c in kpis
    )
    return f'<div class="card"><div class="kpi-row">{kpi_html}</div></div>'


def _build_bar_figure(
    go: Any,
    *,
    dim_name: str,
    dim_data: dict[Any, AttackStats],
) -> Any:
    """
    Build a sorted horizontal bar chart of success rates for one dimension.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    items = sorted(
        dim_data.items(),
        key=lambda kv: kv[1].success_rate if kv[1].success_rate is not None else -1,
        reverse=True,
    )
    labels = [str(k) for k, _ in items]
    rates = [s.success_rate if s.success_rate is not None else 0.0 for _, s in items]
    colors = [_asr_bar_color(s.success_rate) for _, s in items]
    hover = [
        (
            f"{k}<br>ASR: {s.success_rate:.1%}<br>✓ {s.successes} &nbsp; ✗ {s.failures} &nbsp; ? {s.undetermined}"
            if s.success_rate is not None
            else f"{k}<br>No decided outcomes &nbsp; ? {s.undetermined}"
        )
        for k, s in items
    ]
    text = [f"{r:.0%}" if s.success_rate is not None else "—" for (_, s), r in zip(items, rates, strict=True)]
    fig = go.Figure(
        go.Bar(
            x=rates,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            text=text,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Success Rate by {dim_name}",
        xaxis={"title": "Success Rate", "range": [0, 1.25], "tickformat": ".0%"},
        yaxis={"autorange": "reversed"},
        height=max(300, len(labels) * 44 + 100),
        margin={"l": 10, "r": 80, "t": 48, "b": 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _build_z_matrix(
    *,
    row_keys: list[str],
    col_keys: list[str],
    lookup: dict[tuple[str, str], AttackStats],
) -> tuple[list[list[Optional[float]]], list[list[str]]]:
    """
    Build the z-value and annotation matrices for a heatmap.

    Returns:
        tuple: (z matrix, text annotation matrix).
    """
    z: list[list[Optional[float]]] = []
    text: list[list[str]] = []
    for row in row_keys:
        z_row: list[Optional[float]] = []
        t_row: list[str] = []
        for col in col_keys:
            stats = lookup.get((row, col))
            if stats and stats.success_rate is not None:
                z_row.append(stats.success_rate)
                t_row.append(f"{stats.success_rate:.0%}<br>{stats.successes}/{stats.total_decided}")
            else:
                z_row.append(None)
                t_row.append("—" if not stats else f"?{stats.undetermined}")
        z.append(z_row)
        text.append(t_row)
    return z, text


def _build_heatmap_figure(
    go: Any,
    *,
    dim_name: tuple[str, str],
    dim_data: dict[Any, AttackStats],
) -> Any:
    """
    Build a 2D success-rate heatmap for a composite dimension.

    Returns:
        plotly.graph_objects.Figure: The heatmap figure.
    """
    row_dim, col_dim = dim_name
    row_keys = sorted({str(k[0]) for k in dim_data})
    col_keys = sorted({str(k[1]) for k in dim_data})
    lookup = {(str(k[0]), str(k[1])): v for k, v in dim_data.items()}
    z, text = _build_z_matrix(row_keys=row_keys, col_keys=col_keys, lookup=lookup)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_keys,
            y=row_keys,
            text=text,
            texttemplate="%{text}",
            colorscale=_ASR_COLORSCALE,
            zmin=0,
            zmax=1,
            colorbar={"title": "ASR", "tickformat": ".0%"},
            hovertemplate=f"{row_dim}: %{{y}}<br>{col_dim}: %{{x}}<br>%{{text}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Success Rate: {row_dim} \u00d7 {col_dim}",
        xaxis_title=col_dim,
        yaxis_title=row_dim,
        height=max(350, len(row_keys) * 54 + 130),
        margin={"l": 10, "r": 10, "t": 54, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _build_faceted_heatmap_figure(
    go: Any,
    *,
    dim_name: tuple[str, str, str],
    dim_data: dict[Any, AttackStats],
) -> Any:
    """
    Build a 3D heatmap with a dropdown to filter by the third dimension.

    Rows = dim_name[0], Columns = dim_name[1], Dropdown = dim_name[2].

    Returns:
        plotly.graph_objects.Figure: The faceted heatmap figure.
    """
    row_dim, col_dim, facet_dim = dim_name
    row_keys = sorted({str(k[0]) for k in dim_data})
    col_keys = sorted({str(k[1]) for k in dim_data})
    facet_vals = sorted({str(k[2]) for k in dim_data})

    traces: list[Any] = []
    for i, fval in enumerate(facet_vals):
        lookup = {(str(k[0]), str(k[1])): v for k, v in dim_data.items() if str(k[2]) == fval}
        z, text = _build_z_matrix(row_keys=row_keys, col_keys=col_keys, lookup=lookup)
        traces.append(
            go.Heatmap(
                z=z,
                x=col_keys,
                y=row_keys,
                text=text,
                texttemplate="%{text}",
                colorscale=_ASR_COLORSCALE,
                zmin=0,
                zmax=1,
                visible=(i == 0),
                name=fval,
                colorbar={"title": "ASR", "tickformat": ".0%"},
                hovertemplate=f"{row_dim}: %{{y}}<br>{col_dim}: %{{x}}<br>%{{text}}<extra></extra>",
            )
        )

    buttons = [
        {
            "label": fval,
            "method": "update",
            "args": [
                {"visible": [j == i for j in range(len(facet_vals))]},
                {"title": f"{row_dim} \u00d7 {col_dim}  |  {facet_dim}: {fval}"},
            ],
        }
        for i, fval in enumerate(facet_vals)
    ]
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{row_dim} \u00d7 {col_dim}  |  {facet_dim}: {facet_vals[0]}",
        xaxis_title=col_dim,
        yaxis_title=row_dim,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.01,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        height=max(400, len(row_keys) * 54 + 160),
        margin={"l": 10, "r": 10, "t": 90, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _build_coverage_table_figure(
    go: Any,
    *,
    result: AnalysisResult,
) -> Any:
    """
    Build a data coverage table showing sample sizes per dimension key.

    Cells with fewer than 5 decided outcomes are highlighted in pink.

    Returns:
        plotly.graph_objects.Figure: The table figure.
    """
    rows: list[tuple[str, str, AttackStats]] = [("overall", "all", result.overall)]
    for dim_name, dim_data in result.dimensions.items():
        if isinstance(dim_name, str):
            for key, stats in dim_data.items():
                rows.append((dim_name, str(key), stats))

    dims = [r[0] for r in rows]
    keys = [r[1] for r in rows]
    decided = [r[2].total_decided for r in rows]
    undetermined = [r[2].undetermined for r in rows]
    asr = [f"{r[2].success_rate:.0%}" if r[2].success_rate is not None else "—" for r in rows]
    decided_colors = ["#fff5f5" if n < 5 else "white" for n in decided]

    fig = go.Figure(
        go.Table(
            header={
                "values": [
                    "<b>Dimension</b>",
                    "<b>Key</b>",
                    "<b>Decided</b>",
                    "<b>Undetermined</b>",
                    "<b>ASR</b>",
                ],
                "fill_color": "#f0f2f5",
                "align": "left",
                "font": {"size": 12, "color": "#444"},
                "height": 36,
            },
            cells={
                "values": [dims, keys, decided, undetermined, asr],
                "fill_color": ["white", "white", decided_colors, "white", "white"],
                "align": "left",
                "font": {"size": 11},
                "height": 28,
            },
        )
    )
    fig.update_layout(
        title="Data Coverage  (pink = fewer than 5 decided outcomes)",
        height=min(600, max(300, len(rows) * 30 + 120)),
        margin={"l": 0, "r": 0, "t": 44, "b": 10},
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_html(
    result: AnalysisResult,
    path: Union[str, Path],
    *,
    title: str = "Attack Analysis Report",
    composite_dims: Optional[list[tuple[str, str]]] = None,
    sparsity_threshold: float = 0.5,
) -> Path:
    """
    Save a fully interactive HTML attack analysis report using Plotly.

    The report contains:

    * A KPI summary card (overall ASR, total attacks, outcome counts).
    * One horizontal bar chart per single dimension in *result*.
    * Heatmaps for each 2-tuple composite dimension (default: all three
      combinations of ``attack_type``, ``converter_type``, and
      ``harm_category``).
    * A dropdown-faceted heatmap for any 3-tuple composite dimension found
      in *result* (e.g. ``("harm_category", "converter_type", "attack_type")``).
    * A data-coverage table flagging low sample sizes.

    The output is a single self-contained ``.html`` file — no server needed.

    Args:
        result (AnalysisResult): The analysis result to report on.
        path (str | Path): Output file path (e.g. ``"report.html"``).
        title (str): Report title shown in the header. Defaults to
            ``"Attack Analysis Report"``.
        composite_dims (list[tuple[str, str]] | None): 2D heatmap pairs to
            include. Defaults to all combinations of ``attack_type``,
            ``converter_type``, and ``harm_category``.
        sparsity_threshold (float): Skip a heatmap when the fraction of
            empty cells exceeds this value. Defaults to ``0.5``.

    Returns:
        Path: The path to the saved HTML file.

    Raises:
        ImportError: If plotly is not installed.
    """
    go, pio = _import_plotly()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if composite_dims is None:
        composite_dims = _DEFAULT_COMPOSITE_DIMS

    def _div(fig: Any) -> str:
        html: str = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        return html

    sections: list[str] = [_build_summary_html(result=result, title=title)]

    # Single-dimension bar charts
    single_dims = [d for d in result.dimensions if isinstance(d, str)]
    if single_dims:
        bar_html = "".join(_div(_build_bar_figure(go, dim_name=d, dim_data=result.dimensions[d])) for d in single_dims)
        sections.append(f'<div class="card"><h2>By Dimension</h2>{bar_html}</div>')

    # 2D heatmaps
    heatmap_parts: list[str] = []
    for dims in composite_dims:
        if dims not in result.dimensions:
            continue
        if _is_sparse(result.dimensions[dims], threshold=sparsity_threshold):
            continue
        heatmap_parts.append(_div(_build_heatmap_figure(go, dim_name=dims, dim_data=result.dimensions[dims])))

    # 3-tuple faceted heatmaps
    three_d_dims = [d for d in result.dimensions if isinstance(d, tuple) and len(d) == 3]
    for dim_name in three_d_dims:
        if _is_sparse(result.dimensions[dim_name], threshold=sparsity_threshold):
            continue
        heatmap_parts.append(
            _div(_build_faceted_heatmap_figure(go, dim_name=dim_name, dim_data=result.dimensions[dim_name]))
        )

    if heatmap_parts:
        sections.append(f'<div class="card"><h2>Cross-Dimensional Analysis</h2>{"".join(heatmap_parts)}</div>')

    # Coverage table
    sections.append(f'<div class="card">{_div(_build_coverage_table_figure(go, result=result))}</div>')

    body = "\n".join(sections)
    path.write_text(_HTML_TEMPLATE.format(title=title, body=body), encoding="utf-8")
    return path
