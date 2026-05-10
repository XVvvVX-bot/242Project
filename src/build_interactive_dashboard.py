from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def build_payload() -> dict:
    preds = pd.read_csv(OUT / "predictions.csv")
    metrics = pd.read_csv(OUT / "metrics.csv")

    model_order = [
        "lag_mlp",
        "lag_mlp_revin",
        "temporal_transformer",
        "temporal_transformer_revin",
        "seasonal_naive_7",
        "moving_average_28",
    ]
    metrics["model"] = pd.Categorical(metrics["model"], categories=model_order, ordered=True)
    metrics = metrics.sort_values("model")

    series_meta = (
        preds[["series_id", "state_id", "store_id", "dept_id"]]
        .drop_duplicates()
        .sort_values(["store_id", "dept_id"])
    )

    series_summary = []
    for series_id, group in preds.groupby("series_id"):
        actual = group[group["model"] == "lag_mlp"]["actual"]
        best = group[group["model"] == "lag_mlp"]
        series_summary.append(
            {
                "series_id": series_id,
                "mean_actual": round(float(actual.mean()), 3),
                "total_actual": round(float(actual.sum()), 3),
                "lag_mlp_mae": round(float((best["prediction"] - best["actual"]).abs().mean()), 3),
            }
        )

    records = []
    for row in preds.itertuples(index=False):
        records.append(
            {
                "model": row.model,
                "series_id": row.series_id,
                "date": row.date,
                "actual": round(float(row.actual), 3),
                "prediction": round(float(row.prediction), 3),
                "abs_error": round(abs(float(row.prediction) - float(row.actual)), 3),
            }
        )

    return {
        "metrics": metrics.to_dict(orient="records"),
        "series": series_meta.to_dict(orient="records"),
        "series_summary": series_summary,
        "records": records,
        "model_labels": {
            "lag_mlp": "Lag MLP",
            "lag_mlp_revin": "Lag MLP + RevIN",
            "temporal_transformer": "Temporal Transformer",
            "temporal_transformer_revin": "Transformer + RevIN",
            "seasonal_naive_7": "Seasonal Naive",
            "moving_average_28": "Moving Average",
        },
    }


def write_dashboard(payload: dict) -> None:
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Retail Demand Forecasting Dashboard</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d9dee7;
      --panel: #ffffff;
      --bg: #f7f8fb;
      --accent: #1f6feb;
      --green: #207567;
      --purple: #7c3aed;
      --teal: #0f766e;
      --orange: #b35c00;
      --red: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 12px 28px;
      backdrop-filter: blur(10px);
    }}
    .topbar {{
      max-width: 1280px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }}
    .brand {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 230px;
    }}
    .brand strong {{
      font-size: 15px;
      line-height: 1.2;
    }}
    .brand span {{
      color: var(--muted);
      font-size: 12px;
    }}
    nav {{
      display: flex;
      align-items: center;
      gap: 4px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    nav a, .presentation-button {{
      border: 1px solid transparent;
      border-radius: 6px;
      color: #344054;
      background: transparent;
      padding: 7px 9px;
      font-size: 13px;
      text-decoration: none;
      cursor: pointer;
    }}
    nav a:hover, .presentation-button:hover {{
      border-color: var(--line);
      background: #f6f8fb;
    }}
    h1 {{
      margin: 0;
      font-size: 46px;
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 16px 0 0;
      color: #4b5565;
      font-size: 18px;
      line-height: 1.5;
      max-width: 780px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 24px 40px;
    }}
    .presentation-section {{
      scroll-margin-top: 82px;
    }}
    .hero {{
      background: transparent;
      border: 0;
      border-radius: 0;
      overflow: visible;
      min-height: calc(100vh - 76px);
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 30px;
      align-items: center;
      padding: 38px 0 32px;
    }}
    .hero-copy {{
      padding: 8px 0;
    }}
    .hero-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 24px;
    }}
    .primary-link, .secondary-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 14px;
      text-decoration: none;
      border: 1px solid var(--accent);
    }}
    .primary-link {{
      background: var(--accent);
      color: #ffffff;
    }}
    .secondary-link {{
      color: var(--accent);
      background: #ffffff;
    }}
    .hero-panel {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 18px 38px rgba(23, 32, 42, 0.08);
    }}
    .hero-panel h2 {{
      font-size: 18px;
      margin-bottom: 14px;
    }}
    .hero-metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .hero-metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      min-height: 106px;
      background: #fbfcff;
    }}
    .hero-metric .value {{
      font-size: 30px;
      font-weight: 700;
      line-height: 1;
    }}
    .hero-metric .name {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .talk-track {{
      margin-top: 14px;
      color: #475467;
      font-size: 13px;
      line-height: 1.45;
    }}
    .section-intro {{
      padding: 22px 24px 0;
    }}
    .section-intro h2 {{
      font-size: 25px;
      margin-bottom: 8px;
    }}
    .section-intro p {{
      max-width: 860px;
      color: var(--muted);
      margin: 0 0 18px;
      line-height: 1.5;
      font-size: 15px;
    }}
    .story-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      padding: 0 24px 24px;
    }}
    .story-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
      padding: 16px;
      min-height: 150px;
    }}
    .story-card h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .story-card p {{
      margin: 0;
      color: #596579;
      font-size: 14px;
      line-height: 1.45;
    }}
    .story-number {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      margin-bottom: 12px;
      border-radius: 50%;
      background: #e8f0ff;
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }}
    .result-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      padding: 0 24px 24px;
    }}
    .result-card {{
      border-left: 4px solid var(--accent);
      background: #fbfcff;
      border-radius: 8px;
      padding: 14px;
      min-height: 108px;
    }}
    .result-card .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .result-card .value {{
      font-size: 24px;
      font-weight: 700;
      line-height: 1;
    }}
    .result-card .note {{
      color: #596579;
      font-size: 12px;
      margin-top: 8px;
      line-height: 1.35;
    }}
    .explorer-lead {{
      padding: 20px 24px;
      background: #17202a;
      color: #ffffff;
      border-radius: 8px;
      margin-bottom: 18px;
    }}
    .explorer-lead h2 {{
      font-size: 24px;
      margin-bottom: 6px;
    }}
    .explorer-lead p {{
      color: #d9dee7;
      margin: 0;
      max-width: 900px;
      line-height: 1.5;
    }}
    body.present-mode {{
      background: #ffffff;
    }}
    body.present-mode header {{
      position: static;
    }}
    body.present-mode .presentation-section {{
      min-height: 92vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}
    body.present-mode .hero {{
      display: grid;
    }}
    body.present-mode table {{
      font-size: 15px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(260px, 380px) 1fr;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
      padding: 8px 10px;
      font-size: 14px;
    }}
    .toggle-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .toggle {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      padding: 7px 10px;
      font-size: 13px;
      cursor: pointer;
      user-select: none;
    }}
    .toggle input {{ margin: 0; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 84px;
    }}
    .metric .name {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 26px;
      line-height: 1;
      font-weight: 700;
    }}
    .metric .detail {{
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 18px;
      overflow: hidden;
    }}
    .section-head {{
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    h2 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }}
    .section-note {{
      color: var(--muted);
      font-size: 13px;
      margin: 0;
    }}
    .chart-wrap {{
      padding: 12px 14px 16px;
      overflow-x: auto;
    }}
    svg {{
      width: 100%;
      min-width: 760px;
      height: 390px;
      display: block;
    }}
    .axis text {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .axis line, .axis path, .grid line {{
      stroke: #dfe4ec;
      stroke-width: 1;
    }}
    .grid line {{ stroke-dasharray: 3 5; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      padding: 0 16px 14px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 18px;
      height: 3px;
      border-radius: 2px;
      background: currentColor;
    }}
    .table-wrap {{
      overflow-x: auto;
      padding: 0 0 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{
      background: #f3f5f9;
      color: #344054;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    .bar-cell {{
      width: 100%;
      min-width: 180px;
      display: grid;
      grid-template-columns: 1fr 54px;
      align-items: center;
      gap: 10px;
    }}
    .bar-track {{
      height: 9px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: var(--accent);
    }}
    .empty {{
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 820px) {{
      main {{ padding: 16px; }}
      .topbar {{ align-items: flex-start; }}
      nav {{ display: none; }}
      h1 {{ font-size: 34px; }}
      .hero {{ grid-template-columns: 1fr; min-height: auto; padding-top: 18px; }}
      .hero-metrics, .story-grid, .result-strip, .controls, .two-col {{ grid-template-columns: 1fr; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }}
      header {{ padding: 16px; }}
      .subtitle {{ font-size: 16px; }}
    }}
    @media (max-width: 540px) {{
      .metric-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 30px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <strong>Retail Demand Forecasting</strong>
        <span>Walmart M5 | trained models | final dashboard</span>
      </div>
      <nav aria-label="Presentation sections">
        <a href="#overview">Overview</a>
        <a href="#data">Data</a>
        <a href="#models">Models</a>
        <a href="#results">Results</a>
        <a href="#explorer">Explorer</a>
        <button class="presentation-button" id="presentButton" type="button">Presentation Mode</button>
      </nav>
    </div>
  </header>
  <main>
    <section class="presentation-section hero" id="overview">
      <div class="hero-copy">
        <h1>Forecasting store-department retail demand from real sales data</h1>
        <p class="subtitle">This project trains our own neural forecasting models on Walmart M5 sales data, compares them against practical retail baselines, and uses this dashboard to explain the full pipeline from data to decisions.</p>
        <div class="hero-actions">
          <a class="primary-link" href="#results">Show final result</a>
          <a class="secondary-link" href="#explorer">Open forecast explorer</a>
        </div>
      </div>
      <div class="hero-panel">
        <h2>Presentation Summary</h2>
        <div class="hero-metrics">
          <div class="hero-metric">
            <div class="value">70</div>
            <div class="name">store-department demand series</div>
          </div>
          <div class="hero-metric">
            <div class="value">56</div>
            <div class="name">holdout days for validation</div>
          </div>
          <div class="hero-metric">
            <div class="value">52.9</div>
            <div class="name">best MAE from Lag MLP + RevIN</div>
          </div>
          <div class="hero-metric">
            <div class="value">8.79%</div>
            <div class="name">best WAPE on final validation</div>
          </div>
        </div>
        <p class="talk-track">Talk track: the goal is practical short-term demand planning. We aggregate SKU-level M5 sales into department-level operational series, train models from scratch, and show whether neural forecasting and RevIN beat simple rules.</p>
      </div>
    </section>

    <section class="presentation-section" id="data">
      <div class="section-intro">
        <h2>Data and Forecasting Task</h2>
        <p>The dataset is the public M5 Walmart sales data. We aggregate item-level sales into store-department demand series and ask each model to predict tomorrow's demand using a recent rolling history plus calendar signals.</p>
      </div>
      <div class="story-grid">
        <div class="story-card">
          <span class="story-number">1</span>
          <h3>Operational target</h3>
          <p>Daily unit demand for each store and department pair, useful for replenishment, staffing, and exception monitoring.</p>
        </div>
        <div class="story-card">
          <span class="story-number">2</span>
          <h3>Input signals</h3>
          <p>Recent sales history, day-of-week and month cycles, event indicator, SNAP indicator, and learned series identity.</p>
        </div>
        <div class="story-card">
          <span class="story-number">3</span>
          <h3>Validation design</h3>
          <p>The final 56 days are held out chronologically, so each forecast is evaluated like a real future prediction.</p>
        </div>
      </div>
    </section>

    <section class="presentation-section" id="models">
      <div class="section-intro">
        <h2>Modeling Approach</h2>
        <p>We compare simple retail heuristics against two neural architectures trained on our prepared data. Reversible instance normalization is added as a practical experiment for handling local demand shifts.</p>
      </div>
      <div class="story-grid">
        <div class="story-card">
          <span class="story-number">A</span>
          <h3>Baselines</h3>
          <p>Seasonal naive and 28-day moving average provide practical reference points that a planner could understand immediately.</p>
        </div>
        <div class="story-card">
          <span class="story-number">B</span>
          <h3>Lag MLP</h3>
          <p>A feed-forward neural model learns from lagged sales, calendar features, and store-department embeddings.</p>
        </div>
        <div class="story-card">
          <span class="story-number">C</span>
          <h3>Temporal Transformer</h3>
          <p>A Transformer encoder reads the lag window as a sequence and learns which prior days matter through self-attention.</p>
        </div>
      </div>
      <div class="story-grid">
        <div class="story-card">
          <span class="story-number">D</span>
          <h3>RevIN experiment</h3>
          <p>Each window is normalized by its own mean and standard deviation, then predictions are transformed back to unit-sales scale.</p>
        </div>
        <div class="story-card">
          <span class="story-number">E</span>
          <h3>Tuning scope</h3>
          <p>The sweep varies lookback length, learning rate, model width/depth, dropout, and embedding size for both neural families.</p>
        </div>
        <div class="story-card">
          <span class="story-number">F</span>
          <h3>Selected settings</h3>
          <p>Final MLP uses 84 days and 384/192 hidden layers. Final Transformer uses 56 days, d_model 96, and two encoder layers.</p>
        </div>
      </div>
    </section>

    <section class="presentation-section" id="results">
      <div class="section-intro">
        <h2>Final Result</h2>
        <p>The tuned RevIN MLP is the best model overall. RevIN also improves the Transformer, confirming that local window normalization is useful for this nonstationary retail demand task.</p>
      </div>
      <div class="result-strip">
        <div class="result-card" style="border-left-color: var(--purple)">
          <div class="label">Best model</div>
          <div class="value">Lag MLP + RevIN</div>
          <div class="note">Lowest global validation MAE.</div>
        </div>
        <div class="result-card" style="border-left-color: var(--purple)">
          <div class="label">Best MAE</div>
          <div class="value">52.9</div>
          <div class="note">Average absolute units per store-department day.</div>
        </div>
        <div class="result-card" style="border-left-color: var(--teal)">
          <div class="label">Transformer gain from RevIN</div>
          <div class="value">-2.6 MAE</div>
          <div class="note">59.2 down to 56.6 after RevIN.</div>
        </div>
        <div class="result-card" style="border-left-color: var(--green)">
          <div class="label">Takeaway</div>
          <div class="value">Tuned simplicity wins</div>
          <div class="note">The compact MLP beats the larger attention model here.</div>
        </div>
      </div>
      <div class="table-wrap" style="padding: 0 24px 24px">
        <table>
          <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>sMAPE</th><th>WAPE</th></tr></thead>
          <tbody id="presentationMetricBody"></tbody>
        </table>
      </div>
    </section>

    <div class="explorer-lead presentation-section" id="explorer">
      <h2>Interactive Forecast Explorer</h2>
      <p>Use this section during Q&A: choose any store-department series, toggle true demand and model forecasts, and compare both global and selected-series errors.</p>
    </div>

    <div class="controls">
      <div>
        <label for="seriesSelect">Store-department series</label>
        <select id="seriesSelect"></select>
      </div>
      <div>
        <label>Visible models</label>
        <div class="toggle-row" id="modelToggles"></div>
      </div>
    </div>

    <div class="metric-grid" id="metricGrid"></div>

    <section>
      <div class="section-head">
        <h2>Holdout Forecast</h2>
        <p class="section-note" id="chartNote"></p>
      </div>
      <div class="chart-wrap">
        <svg id="forecastChart" viewBox="0 0 980 390" role="img" aria-label="True and predicted demand chart"></svg>
      </div>
      <div class="legend" id="legend"></div>
    </section>

    <div class="two-col">
      <section>
        <div class="section-head">
          <h2>Selected Series Errors</h2>
          <p class="section-note">Lower is better.</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>WAPE</th></tr></thead>
            <tbody id="seriesErrorBody"></tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>Global Metrics</h2>
          <p class="section-note">Final 56-day validation set.</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>sMAPE</th><th>WAPE</th></tr></thead>
            <tbody id="globalMetricBody"></tbody>
          </table>
        </div>
      </section>
    </div>

    <section>
      <div class="section-head">
        <h2>Highest Volume Series</h2>
        <p class="section-note">Useful starting points for forecast inspection.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Series</th><th>Total true demand</th><th>Mean daily true demand</th><th>Lag MLP MAE</th></tr></thead>
          <tbody id="volumeBody"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const DATA = {payload_json};
    const colors = {{
      actual: "#111827",
      lag_mlp: "#1f6feb",
      lag_mlp_revin: "#7c3aed",
      temporal_transformer: "#207567",
      temporal_transformer_revin: "#0f766e",
      seasonal_naive_7: "#b35c00",
      moving_average_28: "#b42318"
    }};
    const modelOrder = ["lag_mlp", "lag_mlp_revin", "temporal_transformer", "temporal_transformer_revin", "seasonal_naive_7", "moving_average_28"];
    const state = {{
      series: "CA_1_FOODS_3",
      visible: new Set(["actual", "lag_mlp", "lag_mlp_revin", "temporal_transformer_revin", "seasonal_naive_7"])
    }};
    const presentationTargets = ["overview", "data", "models", "results", "explorer"];

    function formatNumber(value, digits = 2) {{
      return Number(value).toLocaleString(undefined, {{ maximumFractionDigits: digits, minimumFractionDigits: digits }});
    }}

    function initPresentationControls() {{
      const button = document.getElementById("presentButton");
      button.addEventListener("click", () => {{
        document.body.classList.toggle("present-mode");
        button.textContent = document.body.classList.contains("present-mode") ? "Exit Presentation" : "Presentation Mode";
        document.getElementById("overview").scrollIntoView({{ behavior: "smooth", block: "start" }});
      }});
      document.addEventListener("keydown", event => {{
        if (!document.body.classList.contains("present-mode")) return;
        if (!["ArrowRight", "ArrowDown", "PageDown", "ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) return;
        const currentTop = window.scrollY;
        const positions = presentationTargets.map(id => {{
          const el = document.getElementById(id);
          return {{ id, top: el.getBoundingClientRect().top + window.scrollY }};
        }});
        const currentIndex = positions.reduce((best, item, idx) => {{
          return Math.abs(item.top - currentTop) < Math.abs(positions[best].top - currentTop) ? idx : best;
        }}, 0);
        const direction = ["ArrowRight", "ArrowDown", "PageDown"].includes(event.key) ? 1 : -1;
        const next = Math.max(0, Math.min(positions.length - 1, currentIndex + direction));
        document.getElementById(positions[next].id).scrollIntoView({{ behavior: "smooth", block: "start" }});
        event.preventDefault();
      }});
    }}

    function initControls() {{
      const select = document.getElementById("seriesSelect");
      DATA.series.forEach(row => {{
        const option = document.createElement("option");
        option.value = row.series_id;
        option.textContent = `${{row.series_id}}`;
        select.appendChild(option);
      }});
      select.value = state.series;
      select.addEventListener("change", () => {{
        state.series = select.value;
        render();
      }});

      const toggles = document.getElementById("modelToggles");
      [["actual", "True Demand"], ...modelOrder.map(model => [model, DATA.model_labels[model]])].forEach(([model, labelText]) => {{
        const label = document.createElement("label");
        label.className = "toggle";
        label.style.color = colors[model];
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = state.visible.has(model);
        input.addEventListener("change", () => {{
          if (input.checked) state.visible.add(model);
          else state.visible.delete(model);
          render();
        }});
        const span = document.createElement("span");
        span.textContent = labelText;
        label.appendChild(input);
        label.appendChild(span);
        toggles.appendChild(label);
      }});
    }}

    function renderPresentationMetrics() {{
      const body = document.getElementById("presentationMetricBody");
      body.innerHTML = "";
      DATA.metrics
        .slice()
        .sort((a, b) => a.MAE - b.MAE)
        .forEach(row => {{
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${{DATA.model_labels[row.model]}}</td><td>${{formatNumber(row.MAE, 1)}}</td><td>${{formatNumber(row.RMSE, 1)}}</td><td>${{formatNumber(row.sMAPE_pct, 1)}}%</td><td>${{formatNumber(row.WAPE_pct, 1)}}%</td>`;
          body.appendChild(tr);
        }});
    }}

    function recordsForSeries() {{
      return DATA.records.filter(row => row.series_id === state.series);
    }}

    function groupedByModel(records) {{
      const groups = {{}};
      records.forEach(row => {{
        if (!groups[row.model]) groups[row.model] = [];
        groups[row.model].push(row);
      }});
      Object.values(groups).forEach(rows => rows.sort((a, b) => a.date.localeCompare(b.date)));
      return groups;
    }}

    function computeSeriesErrors(records) {{
      const groups = groupedByModel(records);
      return modelOrder.map(model => {{
        const rows = groups[model] || [];
        const n = rows.length || 1;
        const abs = rows.map(row => Math.abs(row.prediction - row.actual));
        const sq = rows.map(row => Math.pow(row.prediction - row.actual, 2));
        const actualSum = rows.reduce((acc, row) => acc + Math.abs(row.actual), 0) || 1;
        return {{
          model,
          mae: abs.reduce((a, b) => a + b, 0) / n,
          rmse: Math.sqrt(sq.reduce((a, b) => a + b, 0) / n),
          wape: 100 * abs.reduce((a, b) => a + b, 0) / actualSum
        }};
      }});
    }}

    function renderMetrics(records) {{
      const summary = DATA.series_summary.find(row => row.series_id === state.series);
      const errors = computeSeriesErrors(records).find(row => row.model === "lag_mlp");
      const bestGlobal = DATA.metrics.slice().sort((a, b) => a.MAE - b.MAE)[0];
      const transformer = DATA.metrics.find(row => row.model === "temporal_transformer");
      const grid = document.getElementById("metricGrid");
      grid.innerHTML = "";
      [
        ["Selected Series", state.series, "Store-department pair"],
        ["Mean Daily True Demand", formatNumber(summary.mean_actual, 1), "Validation holdout demand"],
        ["Lag MLP Series MAE", formatNumber(errors.mae, 1), "Units per day"],
        ["Best Global Model", DATA.model_labels[bestGlobal.model], `MAE ${{formatNumber(bestGlobal.MAE, 1)}}`]
      ].forEach(([name, value, detail]) => {{
        const el = document.createElement("div");
        el.className = "metric";
        el.innerHTML = `<div class="name">${{name}}</div><div class="value">${{value}}</div><div class="detail">${{detail}}</div>`;
        grid.appendChild(el);
      }});
    }}

    function linePath(points, xScale, yScale) {{
      return points.map((p, i) => `${{i === 0 ? "M" : "L"}} ${{xScale(p.x).toFixed(2)}} ${{yScale(p.y).toFixed(2)}}`).join(" ");
    }}

    function renderChart(records) {{
      const svg = document.getElementById("forecastChart");
      svg.innerHTML = "";
      const groups = groupedByModel(records);
      const actualRows = (groups["lag_mlp"] || []).map((row, i) => ({{ x: i, y: row.actual, date: row.date }}));
      const visibleModels = modelOrder.filter(model => state.visible.has(model));
      const allValues = actualRows.map(row => row.y);
      visibleModels.forEach(model => (groups[model] || []).forEach(row => allValues.push(row.prediction)));
      if (!actualRows.length) {{
        svg.innerHTML = `<text x="490" y="195" text-anchor="middle" fill="#5f6b7a">No data for this series.</text>`;
        return;
      }}

      const margin = {{ top: 22, right: 24, bottom: 48, left: 64 }};
      const width = 980 - margin.left - margin.right;
      const height = 390 - margin.top - margin.bottom;
      const yMax = Math.max(...allValues) * 1.08;
      const yMin = 0;
      const xMax = actualRows.length - 1;
      const xScale = x => margin.left + (x / Math.max(xMax, 1)) * width;
      const yScale = y => margin.top + height - ((y - yMin) / Math.max(yMax - yMin, 1)) * height;

      for (let i = 0; i <= 5; i++) {{
        const y = yMin + (i / 5) * (yMax - yMin);
        const gy = yScale(y);
        svg.insertAdjacentHTML("beforeend", `<g class="grid"><line x1="${{margin.left}}" y1="${{gy}}" x2="${{margin.left + width}}" y2="${{gy}}"></line></g>`);
        svg.insertAdjacentHTML("beforeend", `<g class="axis"><text x="${{margin.left - 10}}" y="${{gy + 4}}" text-anchor="end">${{Math.round(y).toLocaleString()}}</text></g>`);
      }}
      const tickIdx = [0, 13, 27, 41, 55].filter(i => i < actualRows.length);
      tickIdx.forEach(i => {{
        const x = xScale(i);
        const label = actualRows[i].date.slice(5);
        svg.insertAdjacentHTML("beforeend", `<g class="axis"><line x1="${{x}}" y1="${{margin.top + height}}" x2="${{x}}" y2="${{margin.top + height + 5}}"></line><text x="${{x}}" y="${{margin.top + height + 22}}" text-anchor="middle">${{label}}</text></g>`);
      }});
      svg.insertAdjacentHTML("beforeend", `<g class="axis"><line x1="${{margin.left}}" y1="${{margin.top + height}}" x2="${{margin.left + width}}" y2="${{margin.top + height}}"></line><line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{margin.top + height}}"></line></g>`);

      if (state.visible.has("actual")) {{
        const actualPath = linePath(actualRows, xScale, yScale);
        svg.insertAdjacentHTML("beforeend", `<path d="${{actualPath}}" fill="none" stroke="${{colors.actual}}" stroke-width="3"></path>`);
      }}

      visibleModels.forEach(model => {{
        const pts = (groups[model] || []).map((row, i) => ({{ x: i, y: row.prediction }}));
        if (pts.length) {{
          svg.insertAdjacentHTML("beforeend", `<path d="${{linePath(pts, xScale, yScale)}}" fill="none" stroke="${{colors[model]}}" stroke-width="2.4" opacity="0.92"></path>`);
        }}
      }});
      document.getElementById("chartNote").textContent = `${{actualRows[0].date}} to ${{actualRows[actualRows.length - 1].date}}`;
    }}

    function renderLegend() {{
      const legend = document.getElementById("legend");
      const items = [];
      if (state.visible.has("actual")) items.push(["actual", "True Demand"]);
      modelOrder.filter(model => state.visible.has(model)).forEach(model => items.push([model, DATA.model_labels[model]]));
      legend.innerHTML = "";
      items.forEach(([key, label]) => {{
        const div = document.createElement("div");
        div.className = "legend-item";
        div.style.color = colors[key];
        div.innerHTML = `<span class="swatch"></span><span>${{label}}</span>`;
        legend.appendChild(div);
      }});
    }}

    function renderTables(records) {{
      const errors = computeSeriesErrors(records);
      const body = document.getElementById("seriesErrorBody");
      body.innerHTML = "";
      const maxMae = Math.max(...errors.map(row => row.mae));
      errors.forEach(row => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${{DATA.model_labels[row.model]}}</td>
          <td><div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${{100 * row.mae / maxMae}}%; background:${{colors[row.model]}}"></div></div><span>${{formatNumber(row.mae, 1)}}</span></div></td>
          <td>${{formatNumber(row.rmse, 1)}}</td>
          <td>${{formatNumber(row.wape, 1)}}%</td>`;
        body.appendChild(tr);
      }});

      const globalBody = document.getElementById("globalMetricBody");
      globalBody.innerHTML = "";
      DATA.metrics.forEach(row => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{DATA.model_labels[row.model]}}</td><td>${{formatNumber(row.MAE, 1)}}</td><td>${{formatNumber(row.RMSE, 1)}}</td><td>${{formatNumber(row.sMAPE_pct, 1)}}%</td><td>${{formatNumber(row.WAPE_pct, 1)}}%</td>`;
        globalBody.appendChild(tr);
      }});

      const volumeBody = document.getElementById("volumeBody");
      volumeBody.innerHTML = "";
      DATA.series_summary
        .slice()
        .sort((a, b) => b.total_actual - a.total_actual)
        .slice(0, 10)
        .forEach(row => {{
          const tr = document.createElement("tr");
          tr.innerHTML = `<td><button type="button" style="border:0;background:transparent;color:var(--accent);cursor:pointer;padding:0;font:inherit" data-series="${{row.series_id}}">${{row.series_id}}</button></td><td>${{formatNumber(row.total_actual, 0)}}</td><td>${{formatNumber(row.mean_actual, 1)}}</td><td>${{formatNumber(row.lag_mlp_mae, 1)}}</td>`;
          volumeBody.appendChild(tr);
        }});
      volumeBody.querySelectorAll("button").forEach(button => {{
        button.addEventListener("click", () => {{
          state.series = button.dataset.series;
          document.getElementById("seriesSelect").value = state.series;
          render();
          window.scrollTo({{ top: 0, behavior: "smooth" }});
        }});
      }});
    }}

    function render() {{
      const records = recordsForSeries();
      renderMetrics(records);
      renderChart(records);
      renderLegend();
      renderTables(records);
    }}

    initPresentationControls();
    initControls();
    renderPresentationMetrics();
    render();
  </script>
</body>
</html>
"""
    (OUT / "interactive_dashboard.html").write_text(html, encoding="utf-8")
    (OUT / "dashboard.html").write_text(html, encoding="utf-8")


def main() -> None:
    payload = build_payload()
    write_dashboard(payload)
    print(OUT / "interactive_dashboard.html")
    print(OUT / "dashboard.html")


if __name__ == "__main__":
    main()
