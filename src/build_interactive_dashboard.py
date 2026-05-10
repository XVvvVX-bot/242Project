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


def write_presentation(payload_json: str) -> None:
    html = PRESENTATION_HTML.replace("__PAYLOAD__", payload_json)
    (OUT / "dashboard.html").write_text(html, encoding="utf-8")


def write_interactive(payload_json: str) -> None:
    html = INTERACTIVE_HTML.replace("__PAYLOAD__", payload_json)
    (OUT / "interactive_dashboard.html").write_text(html, encoding="utf-8")


def write_dashboard(payload: dict) -> None:
    payload_json = json.dumps(payload, separators=(",", ":"))
    write_presentation(payload_json)
    write_interactive(payload_json)


PRESENTATION_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Retail Demand Forecasting Presentation</title>
  <style>
    :root {
      --ink: #17202a;
      --muted: #647084;
      --line: #d9dee7;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --accent: #1f6feb;
      --purple: #7c3aed;
      --teal: #0f766e;
      --green: #207567;
      --orange: #b35c00;
      --red: #b42318;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
      overflow: hidden;
    }
    .deck-shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: 62px 1fr 56px;
    }
    header, footer {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 0 28px;
    }
    footer {
      border-top: 1px solid var(--line);
      border-bottom: 0;
    }
    .brand {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 240px;
    }
    .brand strong { font-size: 15px; }
    .brand span, .counter { color: var(--muted); font-size: 12px; }
    nav { display: flex; align-items: center; gap: 8px; }
    button, .link-button {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: #344054;
      padding: 8px 11px;
      font-size: 13px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    button:hover, .link-button:hover { background: #f5f7fb; }
    .primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    main {
      position: relative;
      overflow: hidden;
    }
    .slide {
      display: none;
      height: calc(100vh - 118px);
      padding: 46px 64px;
      align-items: center;
      gap: 32px;
    }
    .slide.active { display: grid; }
    .title-slide { grid-template-columns: minmax(0, 1.1fr) minmax(340px, 0.9fr); }
    .content-slide { grid-template-columns: minmax(0, 0.95fr) minmax(420px, 1.05fr); }
    .single-slide { grid-template-columns: 1fr; align-content: center; }
    h1, h2, h3, p { margin: 0; }
    h1 {
      font-size: 58px;
      line-height: 1.02;
      letter-spacing: 0;
      max-width: 880px;
    }
    h2 {
      font-size: 44px;
      line-height: 1.08;
      letter-spacing: 0;
      max-width: 760px;
    }
    h3 { font-size: 19px; line-height: 1.2; }
    .lead {
      margin-top: 20px;
      color: #4b5565;
      font-size: 21px;
      line-height: 1.46;
      max-width: 800px;
    }
    .speaker-note {
      margin-top: 22px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
      max-width: 760px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 18px 38px rgba(23, 32, 42, 0.08);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
      padding: 16px;
      min-height: 118px;
    }
    .metric .value {
      font-size: 36px;
      line-height: 1;
      font-weight: 700;
    }
    .metric .label {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .card-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .card {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-height: 156px;
    }
    .card p {
      margin-top: 9px;
      color: #596579;
      font-size: 15px;
      line-height: 1.45;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 30px;
      height: 30px;
      border-radius: 50%;
      background: #e8f0ff;
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 13px;
    }
    .process {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }
    .step {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 16px;
      min-height: 150px;
    }
    .step strong { display: block; font-size: 16px; margin-bottom: 8px; }
    .step span { color: var(--muted); font-size: 14px; line-height: 1.4; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 16px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      background: #f2f5fa;
      color: #344054;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    tr:last-child td { border-bottom: 0; }
    .takeaway-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 22px;
    }
    .takeaway {
      border-left: 5px solid var(--accent);
      background: #ffffff;
      border-radius: 8px;
      padding: 18px;
      min-height: 138px;
    }
    .takeaway strong { display: block; font-size: 24px; line-height: 1.1; }
    .takeaway span { display: block; margin-top: 10px; color: var(--muted); line-height: 1.4; }
    .progress {
      height: 6px;
      border-radius: 999px;
      background: #edf1f7;
      overflow: hidden;
      flex: 1;
      max-width: 360px;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: var(--accent);
      transition: width 160ms ease;
    }
    @media (max-width: 900px) {
      body { overflow: auto; }
      .deck-shell { display: block; }
      header, footer { position: sticky; z-index: 5; padding: 12px 16px; }
      header { top: 0; }
      footer { bottom: 0; }
      .slide { height: auto; min-height: calc(100vh - 118px); padding: 28px 18px; }
      .title-slide, .content-slide, .single-slide { grid-template-columns: 1fr; }
      h1 { font-size: 38px; }
      h2 { font-size: 32px; }
      .lead { font-size: 17px; }
      .metric-grid, .card-grid, .process, .takeaway-grid { grid-template-columns: 1fr; }
      nav .link-button { display: none; }
    }
  </style>
</head>
<body>
  <div class="deck-shell">
    <header>
      <div class="brand">
        <strong>Retail Demand Forecasting</strong>
        <span>Presentation deck</span>
      </div>
      <nav>
        <a class="link-button" href="interactive_dashboard.html">Open Interactive Dashboard</a>
        <button id="prevBtn" type="button">Prev</button>
        <button id="nextBtn" type="button" class="primary">Next</button>
      </nav>
    </header>

    <main>
      <section class="slide title-slide active">
        <div>
          <h1>Forecasting Walmart store-department demand</h1>
          <p class="lead">A complete data-to-model project: prepare public M5 sales data, train our own forecasting models, tune RevIN variants, and explain results through a separate live dashboard.</p>
          <p class="speaker-note">Use the arrow keys or the Prev/Next buttons to move through this deck. The final slide links to the interactive forecast explorer.</p>
        </div>
        <div class="panel">
          <div class="metric-grid">
            <div class="metric"><div class="value">70</div><div class="label">store-department series</div></div>
            <div class="metric"><div class="value">56</div><div class="label">validation days</div></div>
            <div class="metric"><div class="value">52.9</div><div class="label">best MAE from Lag MLP + RevIN</div></div>
            <div class="metric"><div class="value">8.79%</div><div class="label">best WAPE on holdout</div></div>
          </div>
        </div>
      </section>

      <section class="slide content-slide">
        <div>
          <h2>Why this problem matters</h2>
          <p class="lead">Retail forecasts affect replenishment, staffing, stockout prevention, and waste reduction.</p>
          <p class="speaker-note">We keep the project practical: the output is not an academic benchmark only; it is the type of short-term forecast a retail planning team could inspect.</p>
        </div>
        <div class="card-grid">
          <div class="card"><span class="tag">1</span><h3>Low forecasts</h3><p>Increase stockout risk and reduce service levels for customers.</p></div>
          <div class="card"><span class="tag">2</span><h3>High forecasts</h3><p>Create excess inventory, labor waste, and holding costs.</p></div>
          <div class="card"><span class="tag">3</span><h3>Operational level</h3><p>We aggregate to store-department demand so the task is interpretable and feasible.</p></div>
          <div class="card"><span class="tag">4</span><h3>Decision support</h3><p>The model should help planners investigate risk, not blindly automate ordering.</p></div>
        </div>
      </section>

      <section class="slide content-slide">
        <div>
          <h2>Data and learning task</h2>
          <p class="lead">We use Walmart M5 sales and calendar data, then create rolling-window supervised examples.</p>
        </div>
        <div class="panel">
          <div class="process">
            <div class="step"><strong>Raw data</strong><span>Item-store daily sales plus calendar variables.</span></div>
            <div class="step"><strong>Aggregate</strong><span>10 stores times 7 departments gives 70 time series.</span></div>
            <div class="step"><strong>Features</strong><span>Lagged demand, weekday/month cycles, events, SNAP, series ID.</span></div>
            <div class="step"><strong>Target</strong><span>Next-day unit demand for each store-department pair.</span></div>
            <div class="step"><strong>Validation</strong><span>Final 56 days held out chronologically.</span></div>
          </div>
        </div>
      </section>

      <section class="slide content-slide">
        <div>
          <h2>Models we trained</h2>
          <p class="lead">The comparison includes simple operational baselines and two neural architectures trained from scratch.</p>
        </div>
        <div class="card-grid">
          <div class="card"><span class="tag">A</span><h3>Seasonal naive</h3><p>Predicts demand from the same weekday one week earlier.</p></div>
          <div class="card"><span class="tag">B</span><h3>Moving average</h3><p>Predicts the previous 28-day average.</p></div>
          <div class="card"><span class="tag">C</span><h3>Lag MLP</h3><p>Uses lagged demand, calendar features, and learned series embeddings.</p></div>
          <div class="card"><span class="tag">D</span><h3>Temporal Transformer</h3><p>Reads the lag window as a sequence and learns attention over prior days.</p></div>
        </div>
      </section>

      <section class="slide content-slide">
        <div>
          <h2>RevIN and tuning</h2>
          <p class="lead">Reversible instance normalization helps each model adapt to local demand-level shifts.</p>
          <p class="speaker-note">For each example, RevIN normalizes the input window using that window's mean and standard deviation, then transforms predictions back to unit-sales scale.</p>
        </div>
        <div class="card-grid">
          <div class="card"><span class="tag">R</span><h3>RevIN experiment</h3><p>Normalizes each rolling window locally without leaking the target day.</p></div>
          <div class="card"><span class="tag">T</span><h3>Tuning scope</h3><p>Lookback length, learning rate, MLP width, Transformer width/depth, dropout, and embeddings.</p></div>
          <div class="card"><span class="tag">M</span><h3>Final MLP</h3><p>84-day lookback, hidden layers 384 and 192, dropout 0.15, lr 0.0007.</p></div>
          <div class="card"><span class="tag">F</span><h3>Final Transformer</h3><p>56-day lookback, d_model 96, 4 heads, 2 layers, dropout 0.15, lr 0.0007.</p></div>
        </div>
      </section>

      <section class="slide single-slide">
        <div>
          <h2>Final validation results</h2>
          <p class="lead">The tuned Lag MLP + RevIN model is best overall. RevIN also improves the Transformer.</p>
        </div>
        <table aria-label="Final validation metrics">
          <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>sMAPE</th><th>WAPE</th></tr></thead>
          <tbody id="metricsBody"></tbody>
        </table>
        <div class="takeaway-grid">
          <div class="takeaway" style="border-left-color: var(--purple)"><strong>52.9 MAE</strong><span>Best result from Lag MLP + RevIN.</span></div>
          <div class="takeaway" style="border-left-color: var(--teal)"><strong>-2.6 MAE</strong><span>Transformer improvement from adding RevIN.</span></div>
          <div class="takeaway" style="border-left-color: var(--green)"><strong>39% lower</strong><span>MAE reduction versus seasonal naive.</span></div>
        </div>
      </section>

      <section class="slide content-slide">
        <div>
          <h2>What we learned</h2>
          <p class="lead">Neural forecasting helps, but model complexity is not automatically better. For this aggregated retail task, tuned normalization and strong lag features matter most.</p>
          <p class="speaker-note">A good next step would add price and promotion features, probabilistic forecasts, and inventory-cost evaluation.</p>
        </div>
        <div class="panel">
          <div class="card-grid">
            <div class="card"><span class="tag">1</span><h3>Practical baselines matter</h3><p>Seasonal naive is simple but still meaningful for retail weekly patterns.</p></div>
            <div class="card"><span class="tag">2</span><h3>RevIN is useful</h3><p>Local window normalization improved both neural model families.</p></div>
            <div class="card"><span class="tag">3</span><h3>MLP wins here</h3><p>The compact lag model beat the larger attention model after tuning.</p></div>
            <div class="card"><span class="tag">4</span><h3>Inspect live</h3><p><a class="link-button primary" href="interactive_dashboard.html">Open Interactive Dashboard</a></p></div>
          </div>
        </div>
      </section>
    </main>

    <footer>
      <div class="counter" id="slideCounter"></div>
      <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
      <div class="counter">Use arrow keys, PageUp/PageDown, or buttons</div>
    </footer>
  </div>

  <script>
    const DATA = __PAYLOAD__;
    const labels = DATA.model_labels;
    const slides = Array.from(document.querySelectorAll(".slide"));
    let current = 0;

    function fmt(value, digits = 1) {
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }

    function renderMetrics() {
      const body = document.getElementById("metricsBody");
      body.innerHTML = "";
      DATA.metrics.slice().sort((a, b) => a.MAE - b.MAE).forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${labels[row.model]}</td><td>${fmt(row.MAE)}</td><td>${fmt(row.RMSE)}</td><td>${fmt(row.sMAPE_pct)}%</td><td>${fmt(row.WAPE_pct)}%</td>`;
        body.appendChild(tr);
      });
    }

    function showSlide(index) {
      current = Math.max(0, Math.min(slides.length - 1, index));
      slides.forEach((slide, i) => slide.classList.toggle("active", i === current));
      document.getElementById("slideCounter").textContent = `Slide ${current + 1} of ${slides.length}`;
      document.getElementById("progressBar").style.width = `${100 * (current + 1) / slides.length}%`;
      document.getElementById("prevBtn").disabled = current === 0;
      document.getElementById("nextBtn").disabled = current === slides.length - 1;
    }

    document.getElementById("prevBtn").addEventListener("click", () => showSlide(current - 1));
    document.getElementById("nextBtn").addEventListener("click", () => showSlide(current + 1));
    document.addEventListener("keydown", event => {
      if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(event.key)) {
        showSlide(current + 1);
        event.preventDefault();
      }
      if (["ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) {
        showSlide(current - 1);
        event.preventDefault();
      }
    });

    renderMetrics();
    showSlide(0);
  </script>
</body>
</html>
"""


INTERACTIVE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Retail Demand Forecast Explorer</title>
  <style>
    :root {
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
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 14px 28px;
    }
    .topbar {
      max-width: 1280px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .brand strong { display: block; font-size: 18px; }
    .brand span { color: var(--muted); font-size: 13px; }
    .deck-link {
      min-height: 36px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      color: var(--accent);
      background: #ffffff;
      padding: 8px 11px;
      font-size: 13px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
    }
    main {
      max-width: 1280px;
      margin: 0 auto;
      padding: 22px 24px 32px;
    }
    .intro {
      margin-bottom: 18px;
      background: #17202a;
      color: #ffffff;
      border-radius: 8px;
      padding: 20px 22px;
    }
    .intro h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.15;
    }
    .intro p {
      margin: 8px 0 0;
      color: #d9dee7;
      max-width: 860px;
      line-height: 1.5;
      font-size: 14px;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(260px, 380px) 1fr;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
      padding: 8px 10px;
      font-size: 14px;
    }
    .toggle-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .toggle {
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
    }
    .toggle input { margin: 0; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 84px;
    }
    .metric .name { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .metric .value { font-size: 26px; line-height: 1; font-weight: 700; }
    .metric .detail { margin-top: 7px; color: var(--muted); font-size: 12px; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 18px;
      overflow: hidden;
    }
    .section-head {
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    h2 { margin: 0; font-size: 16px; line-height: 1.2; }
    .section-note { color: var(--muted); font-size: 13px; margin: 0; }
    .chart-wrap { padding: 12px 14px 16px; overflow-x: auto; }
    svg { width: 100%; min-width: 760px; height: 390px; display: block; }
    .axis text { fill: var(--muted); font-size: 11px; }
    .axis line, .axis path, .grid line { stroke: #dfe4ec; stroke-width: 1; }
    .grid line { stroke-dasharray: 3 5; }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      padding: 0 16px 14px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 18px; height: 3px; border-radius: 2px; background: currentColor; }
    .table-wrap { overflow-x: auto; padding: 0 0 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      background: #f3f5f9;
      color: #344054;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    tr:last-child td { border-bottom: 0; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .bar-cell {
      width: 100%;
      min-width: 180px;
      display: grid;
      grid-template-columns: 1fr 54px;
      align-items: center;
      gap: 10px;
    }
    .bar-track { height: 9px; background: #edf1f7; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--accent); }
    @media (max-width: 820px) {
      main { padding: 16px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .controls, .two-col { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      header { padding: 16px; }
      .intro h1 { font-size: 22px; }
    }
    @media (max-width: 540px) {
      .metric-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <strong>Retail Demand Forecast Explorer</strong>
        <span>Interactive validation dashboard</span>
      </div>
      <a class="deck-link" href="dashboard.html">Back to Presentation Deck</a>
    </div>
  </header>
  <main>
    <div class="intro">
      <h1>Explore true demand and model forecasts by store-department series</h1>
      <p>This page is separate from the presentation deck. Use it for live Q&A: select a series, toggle True Demand and forecast lines, and compare global and selected-series metrics.</p>
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
    const DATA = __PAYLOAD__;
    const colors = {
      actual: "#111827",
      lag_mlp: "#1f6feb",
      lag_mlp_revin: "#7c3aed",
      temporal_transformer: "#207567",
      temporal_transformer_revin: "#0f766e",
      seasonal_naive_7: "#b35c00",
      moving_average_28: "#b42318"
    };
    const modelOrder = ["lag_mlp", "lag_mlp_revin", "temporal_transformer", "temporal_transformer_revin", "seasonal_naive_7", "moving_average_28"];
    const state = {
      series: "CA_1_FOODS_3",
      visible: new Set(["actual", "lag_mlp", "lag_mlp_revin", "temporal_transformer_revin", "seasonal_naive_7"])
    };

    function formatNumber(value, digits = 2) {
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }

    function initControls() {
      const select = document.getElementById("seriesSelect");
      DATA.series.forEach(row => {
        const option = document.createElement("option");
        option.value = row.series_id;
        option.textContent = row.series_id;
        select.appendChild(option);
      });
      select.value = state.series;
      select.addEventListener("change", () => {
        state.series = select.value;
        render();
      });

      const toggles = document.getElementById("modelToggles");
      [["actual", "True Demand"], ...modelOrder.map(model => [model, DATA.model_labels[model]])].forEach(([model, labelText]) => {
        const label = document.createElement("label");
        label.className = "toggle";
        label.style.color = colors[model];
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = state.visible.has(model);
        input.addEventListener("change", () => {
          if (input.checked) state.visible.add(model);
          else state.visible.delete(model);
          render();
        });
        const span = document.createElement("span");
        span.textContent = labelText;
        label.appendChild(input);
        label.appendChild(span);
        toggles.appendChild(label);
      });
    }

    function recordsForSeries() {
      return DATA.records.filter(row => row.series_id === state.series);
    }

    function groupedByModel(records) {
      const groups = {};
      records.forEach(row => {
        if (!groups[row.model]) groups[row.model] = [];
        groups[row.model].push(row);
      });
      Object.values(groups).forEach(rows => rows.sort((a, b) => a.date.localeCompare(b.date)));
      return groups;
    }

    function computeSeriesErrors(records) {
      const groups = groupedByModel(records);
      return modelOrder.map(model => {
        const rows = groups[model] || [];
        const n = rows.length || 1;
        const abs = rows.map(row => Math.abs(row.prediction - row.actual));
        const sq = rows.map(row => Math.pow(row.prediction - row.actual, 2));
        const actualSum = rows.reduce((acc, row) => acc + Math.abs(row.actual), 0) || 1;
        return {
          model,
          mae: abs.reduce((a, b) => a + b, 0) / n,
          rmse: Math.sqrt(sq.reduce((a, b) => a + b, 0) / n),
          wape: 100 * abs.reduce((a, b) => a + b, 0) / actualSum
        };
      });
    }

    function renderMetrics(records) {
      const summary = DATA.series_summary.find(row => row.series_id === state.series);
      const errors = computeSeriesErrors(records).find(row => row.model === "lag_mlp");
      const bestGlobal = DATA.metrics.slice().sort((a, b) => a.MAE - b.MAE)[0];
      const grid = document.getElementById("metricGrid");
      grid.innerHTML = "";
      [
        ["Selected Series", state.series, "Store-department pair"],
        ["Mean Daily True Demand", formatNumber(summary.mean_actual, 1), "Validation holdout demand"],
        ["Lag MLP Series MAE", formatNumber(errors.mae, 1), "Units per day"],
        ["Best Global Model", DATA.model_labels[bestGlobal.model], `MAE ${formatNumber(bestGlobal.MAE, 1)}`]
      ].forEach(([name, value, detail]) => {
        const el = document.createElement("div");
        el.className = "metric";
        el.innerHTML = `<div class="name">${name}</div><div class="value">${value}</div><div class="detail">${detail}</div>`;
        grid.appendChild(el);
      });
    }

    function linePath(points, xScale, yScale) {
      return points.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x).toFixed(2)} ${yScale(p.y).toFixed(2)}`).join(" ");
    }

    function renderChart(records) {
      const svg = document.getElementById("forecastChart");
      svg.innerHTML = "";
      const groups = groupedByModel(records);
      const actualRows = (groups["lag_mlp"] || []).map((row, i) => ({ x: i, y: row.actual, date: row.date }));
      const visibleModels = modelOrder.filter(model => state.visible.has(model));
      const allValues = actualRows.map(row => row.y);
      visibleModels.forEach(model => (groups[model] || []).forEach(row => allValues.push(row.prediction)));
      if (!actualRows.length) {
        svg.innerHTML = `<text x="490" y="195" text-anchor="middle" fill="#5f6b7a">No data for this series.</text>`;
        return;
      }

      const margin = { top: 22, right: 24, bottom: 48, left: 64 };
      const width = 980 - margin.left - margin.right;
      const height = 390 - margin.top - margin.bottom;
      const yMax = Math.max(...allValues) * 1.08;
      const yMin = 0;
      const xMax = actualRows.length - 1;
      const xScale = x => margin.left + (x / Math.max(xMax, 1)) * width;
      const yScale = y => margin.top + height - ((y - yMin) / Math.max(yMax - yMin, 1)) * height;

      for (let i = 0; i <= 5; i++) {
        const y = yMin + (i / 5) * (yMax - yMin);
        const gy = yScale(y);
        svg.insertAdjacentHTML("beforeend", `<g class="grid"><line x1="${margin.left}" y1="${gy}" x2="${margin.left + width}" y2="${gy}"></line></g>`);
        svg.insertAdjacentHTML("beforeend", `<g class="axis"><text x="${margin.left - 10}" y="${gy + 4}" text-anchor="end">${Math.round(y).toLocaleString()}</text></g>`);
      }
      [0, 13, 27, 41, 55].filter(i => i < actualRows.length).forEach(i => {
        const x = xScale(i);
        const label = actualRows[i].date.slice(5);
        svg.insertAdjacentHTML("beforeend", `<g class="axis"><line x1="${x}" y1="${margin.top + height}" x2="${x}" y2="${margin.top + height + 5}"></line><text x="${x}" y="${margin.top + height + 22}" text-anchor="middle">${label}</text></g>`);
      });
      svg.insertAdjacentHTML("beforeend", `<g class="axis"><line x1="${margin.left}" y1="${margin.top + height}" x2="${margin.left + width}" y2="${margin.top + height}"></line><line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + height}"></line></g>`);

      if (state.visible.has("actual")) {
        svg.insertAdjacentHTML("beforeend", `<path d="${linePath(actualRows, xScale, yScale)}" fill="none" stroke="${colors.actual}" stroke-width="3"></path>`);
      }

      visibleModels.forEach(model => {
        const pts = (groups[model] || []).map((row, i) => ({ x: i, y: row.prediction }));
        if (pts.length) {
          svg.insertAdjacentHTML("beforeend", `<path d="${linePath(pts, xScale, yScale)}" fill="none" stroke="${colors[model]}" stroke-width="2.4" opacity="0.92"></path>`);
        }
      });
      document.getElementById("chartNote").textContent = `${actualRows[0].date} to ${actualRows[actualRows.length - 1].date}`;
    }

    function renderLegend() {
      const legend = document.getElementById("legend");
      const items = [];
      if (state.visible.has("actual")) items.push(["actual", "True Demand"]);
      modelOrder.filter(model => state.visible.has(model)).forEach(model => items.push([model, DATA.model_labels[model]]));
      legend.innerHTML = "";
      items.forEach(([key, label]) => {
        const div = document.createElement("div");
        div.className = "legend-item";
        div.style.color = colors[key];
        div.innerHTML = `<span class="swatch"></span><span>${label}</span>`;
        legend.appendChild(div);
      });
    }

    function renderTables(records) {
      const errors = computeSeriesErrors(records);
      const body = document.getElementById("seriesErrorBody");
      body.innerHTML = "";
      const maxMae = Math.max(...errors.map(row => row.mae));
      errors.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${DATA.model_labels[row.model]}</td>
          <td><div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${100 * row.mae / maxMae}%; background:${colors[row.model]}"></div></div><span>${formatNumber(row.mae, 1)}</span></div></td>
          <td>${formatNumber(row.rmse, 1)}</td>
          <td>${formatNumber(row.wape, 1)}%</td>`;
        body.appendChild(tr);
      });

      const globalBody = document.getElementById("globalMetricBody");
      globalBody.innerHTML = "";
      DATA.metrics.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${DATA.model_labels[row.model]}</td><td>${formatNumber(row.MAE, 1)}</td><td>${formatNumber(row.RMSE, 1)}</td><td>${formatNumber(row.sMAPE_pct, 1)}%</td><td>${formatNumber(row.WAPE_pct, 1)}%</td>`;
        globalBody.appendChild(tr);
      });

      const volumeBody = document.getElementById("volumeBody");
      volumeBody.innerHTML = "";
      DATA.series_summary
        .slice()
        .sort((a, b) => b.total_actual - a.total_actual)
        .slice(0, 10)
        .forEach(row => {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td><button type="button" style="border:0;background:transparent;color:var(--accent);cursor:pointer;padding:0;font:inherit" data-series="${row.series_id}">${row.series_id}</button></td><td>${formatNumber(row.total_actual, 0)}</td><td>${formatNumber(row.mean_actual, 1)}</td><td>${formatNumber(row.lag_mlp_mae, 1)}</td>`;
          volumeBody.appendChild(tr);
        });
      volumeBody.querySelectorAll("button").forEach(button => {
        button.addEventListener("click", () => {
          state.series = button.dataset.series;
          document.getElementById("seriesSelect").value = state.series;
          render();
          window.scrollTo({ top: 0, behavior: "smooth" });
        });
      });
    }

    function render() {
      const records = recordsForSeries();
      renderMetrics(records);
      renderChart(records);
      renderLegend();
      renderTables(records);
    }

    initControls();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    write_dashboard(payload)
    print(OUT / "dashboard.html")
    print(OUT / "interactive_dashboard.html")


if __name__ == "__main__":
    main()
