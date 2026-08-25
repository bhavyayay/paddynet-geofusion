"""
build_dashboard.py

Developer 3's dashboard: turns the pipeline's output files into a
single self-contained HTML page a non-programmer can open and read.

Shows:
  - Predicted vs actual yield per region/year (from the evaluation run)
  - Historical Haryana yield trend
  - Model accuracy metrics (MAE / RMSE / R2 / MAPE)
  - Parcel classification summary from Developer 1's output, when
    data/processed/parcel_predictions.parquet exists (it requires
    Copernicus credentials to regenerate, so the dashboard degrades
    gracefully when it's absent)

No JavaScript libraries or internet access needed -- charts are plain
inline SVG, so the page works anywhere, including air-gapped review.

Usage:
    python -m src.dashboard.build_dashboard   # writes reports/dashboard.html
or served live by the API at GET /dashboard.
"""

import html
import json
import os

import pandas as pd

EVALUATION_REPORT_PATH = "data/processed/evaluation_report.json"
EVALUATION_PREDICTIONS_PATH = "data/processed/evaluation_predictions.parquet"
PARCEL_PREDICTIONS_PATH = "data/processed/parcel_predictions.parquet"
GROUND_TRUTH_PATH = "data/ground_truth/haryana_rice_yield_combined.csv"
OUTPUT_PATH = "reports/dashboard.html"

# Simple color scheme, chosen for contrast (WCAG-friendly on white)
C_ACTUAL = "#1a5fb4"      # blue
C_PREDICTED = "#e66100"   # orange
C_GRID = "#d0d0d0"
C_TEXT = "#222222"


# ----------------------------------------------------------------------
# SVG chart helpers (dependency-free)
# ----------------------------------------------------------------------

def _scale(value, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) * (out_max - out_min) / (vmax - vmin)


def grouped_bar_chart(labels, actual, predicted, title,
                      width=960, height=380, y_label="Yield (t/ha)"):
    """Two-series grouped bar chart (actual vs predicted) as inline SVG."""
    margin_l, margin_r, margin_t, margin_b = 60, 20, 40, 110
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    vmax = max(max(actual), max(predicted)) * 1.15
    n = len(labels)
    group_w = plot_w / max(n, 1)
    bar_w = min(group_w * 0.35, 28)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(title)}" '
             f'style="max-width:100%;height:auto;font-family:sans-serif">']
    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" '
                 f'font-size="15" font-weight="bold" fill="{C_TEXT}">'
                 f'{html.escape(title)}</text>')

    # Y axis: gridlines + tick labels
    for i in range(5):
        yv = vmax * i / 4
        y = margin_t + plot_h - _scale(yv, 0, vmax, 0, plot_h)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" '
                     f'x2="{width - margin_r}" y2="{y:.1f}" '
                     f'stroke="{C_GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 6}" y="{y + 4:.1f}" '
                     f'text-anchor="end" font-size="11" fill="{C_TEXT}">'
                     f'{yv:.1f}</text>')
    parts.append(f'<text x="14" y="{margin_t + plot_h / 2}" font-size="12" '
                 f'fill="{C_TEXT}" transform="rotate(-90 14 {margin_t + plot_h / 2})" '
                 f'text-anchor="middle">{html.escape(y_label)}</text>')

    # Bars + x labels
    for i, (label, a, p) in enumerate(zip(labels, actual, predicted)):
        cx = margin_l + group_w * (i + 0.5)
        for offset, val, color in ((-bar_w, a, C_ACTUAL), (0, p, C_PREDICTED)):
            bh = _scale(val, 0, vmax, 0, plot_h)
            parts.append(f'<rect x="{cx + offset:.1f}" '
                         f'y="{margin_t + plot_h - bh:.1f}" '
                         f'width="{bar_w:.1f}" height="{bh:.1f}" '
                         f'fill="{color}"><title>{html.escape(label)}: '
                         f'{val:.2f} t/ha</title></rect>')
        parts.append(f'<text x="{cx:.1f}" y="{margin_t + plot_h + 12}" '
                     f'font-size="10" fill="{C_TEXT}" text-anchor="end" '
                     f'transform="rotate(-45 {cx:.1f} {margin_t + plot_h + 12})">'
                     f'{html.escape(label)}</text>')

    # Legend
    lx = margin_l
    ly = height - 18
    for name, color in (("Actual", C_ACTUAL), ("Predicted (RF)", C_PREDICTED)):
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 16}" y="{ly}" font-size="12" '
                     f'fill="{C_TEXT}">{name}</text>')
        lx += 130

    parts.append("</svg>")
    return "".join(parts)


def line_chart(x_labels, series, title, width=960, height=340,
               y_label="Yield (t/ha)"):
    """Multi-series line chart as inline SVG.
    series: list of (name, values, color)."""
    margin_l, margin_r, margin_t, margin_b = 60, 20, 40, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_vals = [v for _, vals, _ in series for v in vals if v is not None]
    vmax = max(all_vals) * 1.15
    n = len(x_labels)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(title)}" '
             f'style="max-width:100%;height:auto;font-family:sans-serif">']
    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" '
                 f'font-size="15" font-weight="bold" fill="{C_TEXT}">'
                 f'{html.escape(title)}</text>')

    for i in range(5):
        yv = vmax * i / 4
        y = margin_t + plot_h - _scale(yv, 0, vmax, 0, plot_h)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" '
                     f'x2="{width - margin_r}" y2="{y:.1f}" '
                     f'stroke="{C_GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 6}" y="{y + 4:.1f}" '
                     f'text-anchor="end" font-size="11" fill="{C_TEXT}">'
                     f'{yv:.1f}</text>')
    parts.append(f'<text x="14" y="{margin_t + plot_h / 2}" font-size="12" '
                 f'fill="{C_TEXT}" transform="rotate(-90 14 {margin_t + plot_h / 2})" '
                 f'text-anchor="middle">{html.escape(y_label)}</text>')

    def px(i):
        return margin_l + _scale(i, 0, max(n - 1, 1), 0, plot_w)

    for name, vals, color in series:
        pts = [(px(i), margin_t + plot_h - _scale(v, 0, vmax, 0, plot_h))
               for i, v in enumerate(vals) if v is not None]
        if len(pts) > 1:
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{d}" fill="none" '
                         f'stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')

    step = max(1, n // 12)
    for i in range(0, n, step):
        parts.append(f'<text x="{px(i):.1f}" y="{margin_t + plot_h + 16}" '
                     f'font-size="10" fill="{C_TEXT}" text-anchor="middle">'
                     f'{html.escape(str(x_labels[i]))}</text>')

    lx = margin_l
    ly = height - 12
    for name, _, color in series:
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 16}" y="{ly}" font-size="12" '
                     f'fill="{C_TEXT}">{html.escape(name)}</text>')
        lx += 160

    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# Page sections
# ----------------------------------------------------------------------

def metrics_section(report: dict) -> str:
    rows = []
    for model_name, scopes in report["models"].items():
        for scope, m in scopes.items():
            rows.append(
                f"<tr><td>{html.escape(model_name)}</td>"
                f"<td>{html.escape(scope.replace('_', ' '))}</td>"
                f"<td>{m['mae_t_ha']:.3f}</td><td>{m['rmse_t_ha']:.3f}</td>"
                f"<td>{m['r2']:.3f}</td><td>{m['mape_pct']:.1f}%</td>"
                f"<td>{m['n_samples']}</td></tr>"
            )
    return f"""
<section aria-labelledby="metrics-h">
  <h2 id="metrics-h">Model Accuracy vs Real Historical Yields</h2>
  <p>Recommended model: <strong>{html.escape(report['recommended_model'])}</strong>
     (lowest error on data it never saw during training).</p>
  <table>
    <caption class="sr-only">Model accuracy metrics</caption>
    <thead><tr><th scope="col">Model</th><th scope="col">Scope</th>
      <th scope="col">MAE (t/ha)</th><th scope="col">RMSE (t/ha)</th>
      <th scope="col">R&sup2;</th><th scope="col">MAPE</th>
      <th scope="col">Rows</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p class="caveat">{html.escape(report['caveat'])}</p>
</section>"""


def predictions_section(preds: pd.DataFrame) -> str:
    # District-level rows (2021-22) make the most intuitive comparison
    districts = preds[preds["level"] == "district"].sort_values("region")
    chart1 = ""
    if len(districts) > 0:
        chart1 = grouped_bar_chart(
            labels=list(districts["region"]),
            actual=list(districts["yield_t_ha"]),
            predicted=list(districts["pred_random_forest_t_ha"]),
            title="Predicted vs Actual Rice Yield by District (Haryana, 2021-22)",
        )

    # State-level trend over time
    state = preds[preds["level"] == "state"].sort_values("year_clean")
    chart2 = ""
    if len(state) > 0:
        chart2 = line_chart(
            x_labels=list(state["year_clean"]),
            series=[
                ("Actual", list(state["yield_t_ha"]), C_ACTUAL),
                ("Predicted (RF)", list(state["pred_random_forest_t_ha"]), C_PREDICTED),
            ],
            title="Haryana State Rice Yield Over Time: Actual vs Predicted",
        )

    return f"""
<section aria-labelledby="preds-h">
  <h2 id="preds-h">Predictions vs Reality</h2>
  {chart1}
  {chart2}
</section>"""


def parcel_section() -> str:
    """Developer 1's classification output, if present."""
    if not os.path.exists(PARCEL_PREDICTIONS_PATH):
        return """
<section aria-labelledby="parcel-h">
  <h2 id="parcel-h">Satellite Parcel Classification (Developer 1)</h2>
  <p class="caveat">No parcel classification file found at
  <code>data/processed/parcel_predictions.parquet</code>. Run Developer 1's
  pipeline (<code>./run_pipeline.sh</code>, requires Copernicus credentials)
  to populate this section with classified parcels and vegetation indices.</p>
</section>"""

    df = pd.read_parquet(PARCEL_PREDICTIONS_PATH)
    counts = df["class_label"].value_counts().to_dict()
    count_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in counts.items()
    )
    idx_means = {c: df[c].mean() for c in ("ndvi", "ndwi", "msavi") if c in df}
    idx_rows = "".join(
        f"<tr><td>{k.upper()}</td><td>{v:.3f}</td></tr>"
        for k, v in idx_means.items()
    )
    return f"""
<section aria-labelledby="parcel-h">
  <h2 id="parcel-h">Satellite Parcel Classification (Developer 1)</h2>
  <div class="cols">
    <table>
      <caption>Parcels by class</caption>
      <thead><tr><th scope="col">Class</th><th scope="col">Count</th></tr></thead>
      <tbody>{count_rows}</tbody>
    </table>
    <table>
      <caption>Mean vegetation indices</caption>
      <thead><tr><th scope="col">Index</th><th scope="col">Mean</th></tr></thead>
      <tbody>{idx_rows}</tbody>
    </table>
  </div>
</section>"""


def build_dashboard_html() -> str:
    """Assemble the full dashboard page. Raises FileNotFoundError if the
    evaluation artifacts haven't been generated yet."""
    if not (os.path.exists(EVALUATION_REPORT_PATH)
            and os.path.exists(EVALUATION_PREDICTIONS_PATH)):
        raise FileNotFoundError(
            "Evaluation artifacts missing. Run "
            "'python -m src.evaluation.evaluate_models' first."
        )
    with open(EVALUATION_REPORT_PATH) as f:
        report = json.load(f)
    preds = pd.read_parquet(EVALUATION_PREDICTIONS_PATH)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PaddyNet-GeoFusion Dashboard</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; color: {C_TEXT};
         max-width: 1020px; margin: 0 auto; padding: 1rem 1.5rem; }}
  h1 {{ border-bottom: 3px solid {C_ACTUAL}; padding-bottom: .4rem; }}
  section {{ margin: 2rem 0; }}
  table {{ border-collapse: collapse; margin: .8rem 0; }}
  th, td {{ border: 1px solid #bbb; padding: .35rem .7rem; text-align: left; }}
  th {{ background: #eef3fa; }}
  .caveat {{ background: #fff4e5; border-left: 4px solid {C_PREDICTED};
             padding: .6rem .8rem; font-size: .92rem; }}
  .cols {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
  .sr-only {{ position: absolute; width: 1px; height: 1px; overflow: hidden;
              clip: rect(0 0 0 0); }}
  code {{ background: #f2f2f2; padding: .1rem .3rem; }}
  footer {{ margin-top: 3rem; font-size: .85rem; color: #555;
            border-top: 1px solid #ccc; padding-top: .6rem; }}
</style>
</head>
<body>
<h1>PaddyNet-GeoFusion &mdash; Rice Yield Dashboard</h1>
<p>Rice yield prediction for Haryana, India, from satellite imagery,
weather data, and government crop statistics.
Predictions come from the trained Random Forest model
(see <a href="/docs">API docs</a> to request one programmatically).</p>
{metrics_section(report)}
{predictions_section(preds)}
{parcel_section()}
<footer>Generated by <code>src/dashboard/build_dashboard.py</code> &middot;
evaluation source: <code>{html.escape(EVALUATION_REPORT_PATH)}</code></footer>
</body>
</html>"""


def main():
    html_out = build_dashboard_html()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html_out)
    print(f"Dashboard written to {OUTPUT_PATH}")
    print("Open it in a browser, or run the API and visit /dashboard.")


if __name__ == "__main__":
    main()
