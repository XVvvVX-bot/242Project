from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass
class Config:
    data_dir: Path = Path("data/m5")
    out_dir: Path = Path("outputs")
    val_days: int = 56
    batch_size: int = 512
    epochs: int = 5
    mlp_seq_len: int = 84
    mlp_lr: float = 7e-4
    transformer_seq_len: int = 56
    transformer_lr: float = 7e-4
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15
    mlp_hidden: tuple[int, ...] = (384, 192)
    mlp_dropout: float = 0.15
    mlp_emb_dim: int = 24
    seed: int = 242
    max_train_windows_per_series: int = 900


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs(cfg: Config) -> None:
    (cfg.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "models").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "processed").mkdir(parents=True, exist_ok=True)


def load_panel(cfg: Config) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    sales_path = cfg.data_dir / "sales_train_validation.csv"
    calendar_path = cfg.data_dir / "calendar.csv"
    if not sales_path.exists() or not calendar_path.exists():
        raise FileNotFoundError(
            "Missing M5 CSVs. Download and extract m5-forecasting-accuracy.zip into data/m5."
        )

    calendar = pd.read_csv(calendar_path)
    day_cols = [c for c in pd.read_csv(sales_path, nrows=0).columns if c.startswith("d_")]
    meta_cols = ["state_id", "store_id", "dept_id", "cat_id"]
    sales = pd.read_csv(sales_path, usecols=meta_cols + day_cols)

    grouped = sales.groupby(meta_cols, as_index=False)[day_cols].sum()
    grouped["series_id"] = grouped["store_id"] + "_" + grouped["dept_id"]
    grouped = grouped.sort_values(["store_id", "dept_id"]).reset_index(drop=True)

    y = grouped[day_cols].to_numpy(dtype=np.float32)

    cal = calendar[calendar["d"].isin(day_cols)].copy()
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal = cal.sort_values("d_num").reset_index(drop=True)
    cal["event_flag"] = cal["event_name_1"].notna().astype(np.float32)
    cal["wday_sin"] = np.sin(2 * np.pi * cal["wday"] / 7.0)
    cal["wday_cos"] = np.cos(2 * np.pi * cal["wday"] / 7.0)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12.0)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12.0)

    state_to_snap = {
        "CA": cal["snap_CA"].to_numpy(dtype=np.float32),
        "TX": cal["snap_TX"].to_numpy(dtype=np.float32),
        "WI": cal["snap_WI"].to_numpy(dtype=np.float32),
    }
    base_cov = cal[["wday_sin", "wday_cos", "month_sin", "month_cos", "event_flag"]].to_numpy(
        dtype=np.float32
    )

    covariates = []
    for state_id in grouped["state_id"]:
        snap = state_to_snap[state_id][:, None]
        covariates.append(np.concatenate([base_cov, snap], axis=1))
    x_cov = np.stack(covariates).astype(np.float32)

    grouped[["series_id", "state_id", "store_id", "dept_id", "cat_id"]].to_csv(
        cfg.out_dir / "processed" / "series_metadata.csv", index=False
    )

    panel = grouped[["series_id", "state_id", "store_id", "dept_id", "cat_id"] + day_cols]
    panel.to_csv(cfg.out_dir / "processed" / "department_store_panel.csv", index=False)

    return y, x_cov, grouped, cal


class WindowDataset(Dataset):
    def __init__(
        self,
        y_norm: np.ndarray,
        y_raw: np.ndarray,
        x_cov: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        indices: list[tuple[int, int]],
        seq_len: int,
        use_revin: bool = False,
    ) -> None:
        self.y_norm = y_norm
        self.y_raw = y_raw
        self.x_cov = x_cov
        self.means = means
        self.stds = stds
        self.indices = indices
        self.seq_len = seq_len
        self.use_revin = use_revin

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s, t = self.indices[idx]
        start = t - self.seq_len
        cov_seq = self.x_cov[s, start:t, :]
        current_cov = self.x_cov[s, t, :]
        if self.use_revin:
            raw_window = self.y_raw[s, start:t].astype(np.float32)
            local_mean = np.float32(raw_window.mean())
            local_std = np.float32(raw_window.std())
            if local_std < 1.0:
                local_std = np.float32(1.0)
            lag = ((raw_window - local_mean) / local_std)[:, None]
            target = np.float32((self.y_raw[s, t] - local_mean) / local_std)
            mean = local_mean
            std = local_std

            mean_feature = np.float32((local_mean - self.means[s]) / self.stds[s])
            std_feature = np.float32(local_std / self.stds[s])
            stat_seq = np.tile(np.array([[mean_feature, std_feature]], dtype=np.float32), (self.seq_len, 1))
            cov_seq = np.concatenate([cov_seq, stat_seq], axis=1)
            current_cov = np.concatenate(
                [current_cov, np.array([mean_feature, std_feature], dtype=np.float32)]
            )
        else:
            lag = self.y_norm[s, start:t, None]
            target = np.float32(self.y_norm[s, t])
            mean = np.float32(self.means[s])
            std = np.float32(self.stds[s])
        x_seq = np.concatenate([lag, cov_seq], axis=1)
        return {
            "x_seq": torch.tensor(x_seq, dtype=torch.float32),
            "current_cov": torch.tensor(current_cov, dtype=torch.float32),
            "series": torch.tensor(s, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_raw": torch.tensor(self.y_raw[s, t], dtype=torch.float32),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "time": torch.tensor(t, dtype=torch.long),
        }


class LagMLP(nn.Module):
    def __init__(
        self,
        seq_len: int,
        cov_dim: int,
        n_series: int,
        hidden_sizes: Sequence[int] = (128, 64),
        dropout: float = 0.10,
        emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.series_emb = nn.Embedding(n_series, emb_dim)
        in_dim = seq_len + cov_dim + emb_dim
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_seq: torch.Tensor, current_cov: torch.Tensor, series: torch.Tensor) -> torch.Tensor:
        lag = x_seq[:, :, 0]
        emb = self.series_emb(series)
        x = torch.cat([lag, current_cov, emb], dim=1)
        return self.net(x).squeeze(1)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_series: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.series_emb = nn.Embedding(n_series, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_seq: torch.Tensor, current_cov: torch.Tensor, series: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_seq)
        x = x + self.pos_emb[:, : x.shape[1], :] + self.series_emb(series)[:, None, :]
        h = self.encoder(x)
        return self.head(h[:, -1, :]).squeeze(1)


def build_indices(
    n_series: int,
    n_days: int,
    seq_len: int,
    train_end: int,
    max_train_windows_per_series: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    train_indices: list[tuple[int, int]] = []
    val_indices: list[tuple[int, int]] = []
    rng = np.random.default_rng(242)

    for s in range(n_series):
        possible = np.arange(seq_len, train_end)
        if len(possible) > max_train_windows_per_series:
            possible = np.sort(rng.choice(possible, size=max_train_windows_per_series, replace=False))
        train_indices.extend((s, int(t)) for t in possible)
        val_indices.extend((s, t) for t in range(train_end, n_days))
    return train_indices, val_indices


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    lr: float | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model.to(device)
    if lr is None:
        raise ValueError("train_model now requires an explicit learning rate per model family.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x_seq = batch["x_seq"].to(device)
            current_cov = batch["current_cov"].to(device)
            series = batch["series"].to(device)
            target = batch["target"].to(device)
            pred = model(x_seq, current_cov, series)
            loss = loss_fn(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_mae_norm = evaluate_normalized(model, val_loader, device)
        row = {
            "model": name,
            "epoch": epoch,
            "train_mae_norm": float(np.mean(train_losses)),
            "val_mae_norm": val_mae_norm,
        }
        history.append(row)
        print(json.dumps(row))
        if val_mae_norm < best_val:
            best_val = val_mae_norm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), cfg.out_dir / "models" / f"{name}.pt")
    return model, history


@torch.no_grad()
def evaluate_normalized(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    vals = []
    for batch in loader:
        pred = model(
            batch["x_seq"].to(device),
            batch["current_cov"].to(device),
            batch["series"].to(device),
        )
        vals.append(torch.mean(torch.abs(pred - batch["target"].to(device))).item())
    return float(np.mean(vals))


@torch.no_grad()
def predict_model(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    meta: pd.DataFrame,
    cal: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        pred_norm = model(
            batch["x_seq"].to(device),
            batch["current_cov"].to(device),
            batch["series"].to(device),
        ).cpu()
        pred_raw = pred_norm * batch["std"] + batch["mean"]
        pred_raw = torch.clamp(pred_raw, min=0.0)
        for sid, t, y_true, y_pred in zip(
            batch["series"].numpy(),
            batch["time"].numpy(),
            batch["target_raw"].numpy(),
            pred_raw.numpy(),
        ):
            m = meta.iloc[int(sid)]
            date = cal.iloc[int(t)]["date"]
            rows.append(
                {
                    "model": name,
                    "series_id": m["series_id"],
                    "state_id": m["state_id"],
                    "store_id": m["store_id"],
                    "dept_id": m["dept_id"],
                    "date": date,
                    "t": int(t),
                    "actual": float(y_true),
                    "prediction": float(y_pred),
                }
            )
    return pd.DataFrame(rows)


def baseline_predictions(
    y: np.ndarray,
    meta: pd.DataFrame,
    cal: pd.DataFrame,
    train_end: int,
    seq_len: int,
) -> pd.DataFrame:
    rows = []
    for s in range(y.shape[0]):
        m = meta.iloc[s]
        for t in range(train_end, y.shape[1]):
            baselines = {
                "seasonal_naive_7": y[s, t - 7],
                "moving_average_28": y[s, max(0, t - 28) : t].mean(),
            }
            for model, pred in baselines.items():
                rows.append(
                    {
                        "model": model,
                        "series_id": m["series_id"],
                        "state_id": m["state_id"],
                        "store_id": m["store_id"],
                        "dept_id": m["dept_id"],
                        "date": cal.iloc[t]["date"],
                        "t": int(t),
                        "actual": float(y[s, t]),
                        "prediction": max(0.0, float(pred)),
                    }
                )
    return pd.DataFrame(rows)


def compute_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, df in preds.groupby("model"):
        err = df["prediction"].to_numpy() - df["actual"].to_numpy()
        actual = df["actual"].to_numpy()
        pred = df["prediction"].to_numpy()
        mae = np.mean(np.abs(err))
        rmse = math.sqrt(np.mean(err**2))
        smape = np.mean(2 * np.abs(err) / np.maximum(np.abs(actual) + np.abs(pred), 1e-6)) * 100
        wape = np.sum(np.abs(err)) / max(np.sum(np.abs(actual)), 1e-6) * 100
        rows.append({"model": model, "MAE": mae, "RMSE": rmse, "sMAPE_pct": smape, "WAPE_pct": wape})
    return pd.DataFrame(rows).sort_values("MAE")


def plot_outputs(preds: pd.DataFrame, metrics: pd.DataFrame, history: pd.DataFrame, cfg: Config) -> None:
    fig_dir = cfg.out_dir / "figures"

    plt.figure(figsize=(8, 4.5))
    order = metrics.sort_values("MAE")
    plt.bar(order["model"], order["MAE"], color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"][: len(order)])
    plt.ylabel("Validation MAE (units/day)")
    plt.title("Forecast accuracy by model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_mae_comparison.png", dpi=180)
    plt.close()

    if not history.empty:
        plt.figure(figsize=(7.5, 4.5))
        for model, df in history.groupby("model"):
            plt.plot(df["epoch"], df["train_mae_norm"], marker="o", label=f"{model} train")
            plt.plot(df["epoch"], df["val_mae_norm"], marker="s", label=f"{model} val")
        plt.xlabel("Epoch")
        plt.ylabel("MAE on normalized demand")
        plt.title("Training and validation loss curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=180)
        plt.close()

    best_model = metrics.iloc[0]["model"]
    model_preds = preds[preds["model"] == best_model]
    example_series = (
        model_preds.groupby("series_id")["actual"].mean().sort_values(ascending=False).index[:3].tolist()
    )
    for series_id in example_series:
        df = model_preds[model_preds["series_id"] == series_id].sort_values("date")
        plt.figure(figsize=(9, 4.2))
        plt.plot(pd.to_datetime(df["date"]), df["actual"], label="actual", linewidth=2)
        plt.plot(pd.to_datetime(df["date"]), df["prediction"], label=best_model, linewidth=2)
        plt.ylabel("Daily unit sales")
        plt.title(f"Holdout forecast: {series_id}")
        plt.legend()
        plt.tight_layout()
        safe_name = series_id.replace("/", "_")
        plt.savefig(fig_dir / f"forecast_{safe_name}.png", dpi=180)
        plt.close()

    transformer = preds[preds["model"] == "temporal_transformer_revin"].copy()
    if transformer.empty:
        transformer = preds[preds["model"] == "temporal_transformer"].copy()
    if not transformer.empty:
        transformer["abs_error"] = (transformer["prediction"] - transformer["actual"]).abs()
        dept = (
            transformer.groupby("dept_id")["abs_error"]
            .mean()
            .sort_values(ascending=False)
            .reset_index(name="MAE")
        )
        plt.figure(figsize=(8, 4.2))
        plt.bar(dept["dept_id"], dept["MAE"], color="#E45756")
        plt.ylabel("MAE")
        plt.title("Transformer errors by department")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "transformer_mae_by_department.png", dpi=180)
        plt.close()


def write_report_notes(cfg: Config, metrics: pd.DataFrame) -> None:
    best = metrics.iloc[0]
    notes = f"""# Report Notes

## Project description and motivation

Retailers need daily demand forecasts to decide how much inventory to replenish, where to allocate labor, and when to investigate likely stockout risk. We model demand for Walmart store-department combinations using the public M5 sales dataset.

## Data and learning problem

The original M5 data contains daily item-level unit sales across Walmart stores from 2011-01-29 through 2016-04-24. To keep training computationally feasible and operationally interpretable, this project aggregates item sales into 70 store-department series: 10 stores times 7 departments. The supervised learning task is:

Given a tuned rolling history of sales for a store-department series, plus calendar variables, predict the next day's unit demand.

The final 56 days are held out for validation.

## Methodology

Baselines:

- Seasonal naive: next demand equals demand from the same weekday one week earlier.
- Moving average: next demand equals the previous 28-day average.
- Lag MLP: feed-forward neural network over lagged sales, current calendar features, and a learned series embedding.
- RevIN variants: the same neural architectures trained with reversible instance normalization, where each input window is normalized by its own mean and standard deviation and predictions are transformed back with those same statistics.

Main model:

- Temporal Transformer encoder trained from scratch.
- Inputs are normalized lagged demand and calendar features.
- The model uses positional embeddings and a learned series embedding so one global network can share statistical strength across departments and stores.
- RevIN is evaluated as an additional advanced normalization procedure for nonstationary demand.
- Hyperparameter tuning explored lookback length, learning rate, neural width/depth, dropout, and embedding size.
- Final MLP settings: {cfg.mlp_seq_len}-day lookback, hidden layers {cfg.mlp_hidden}, dropout {cfg.mlp_dropout}, embedding size {cfg.mlp_emb_dim}, learning rate {cfg.mlp_lr}.
- Final Transformer settings: {cfg.transformer_seq_len}-day lookback, d_model {cfg.d_model}, {cfg.n_heads} attention heads, {cfg.n_layers} encoder layers, dropout {cfg.dropout}, learning rate {cfg.transformer_lr}.
- Objective: minimize mean absolute error on normalized demand using AdamW with early model selection by validation MAE.

## Current validation result

Best model by MAE: {best['model']}

| Model | MAE | RMSE | sMAPE (%) | WAPE (%) |
|---|---:|---:|---:|---:|
"""
    for _, row in metrics.iterrows():
        notes += (
            f"| {row['model']} | {row['MAE']:.3f} | {row['RMSE']:.3f} | "
            f"{row['sMAPE_pct']:.2f} | {row['WAPE_pct']:.2f} |\n"
        )

    notes += """
## Result discussion

Use `outputs/figures/training_curves.png` to discuss underfitting/overfitting. Use `outputs/figures/model_mae_comparison.png` to compare the neural models against simple operational baselines. Forecast plots for high-volume series are saved under `outputs/figures/forecast_*.png`.

## Safety, security, and ethics

The model should not directly automate replenishment without guardrails. Forecast errors can create stockouts for essential goods or overstock waste for perishable goods. Historical demand reflects past prices, promotions, local demographics, and possible stockout censoring; these biases can be amplified by a deployed system. A practical deployment should include human override, service-level constraints, uncertainty estimates, and monitoring for drift after holidays, economic shocks, or assortment changes.
"""
    (cfg.out_dir / "report_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-train-windows-per-series", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mlp-seq-len", type=int, default=84)
    parser.add_argument("--mlp-lr", type=float, default=7e-4)
    parser.add_argument("--transformer-seq-len", type=int, default=56)
    parser.add_argument("--transformer-lr", type=float, default=7e-4)
    parser.add_argument("--mlp-hidden", type=str, default="384,192")
    parser.add_argument("--mlp-dropout", type=float, default=0.15)
    parser.add_argument("--mlp-emb-dim", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        max_train_windows_per_series=args.max_train_windows_per_series,
        batch_size=args.batch_size,
        mlp_seq_len=args.mlp_seq_len,
        mlp_lr=args.mlp_lr,
        transformer_seq_len=args.transformer_seq_len,
        transformer_lr=args.transformer_lr,
        mlp_hidden=tuple(int(x.strip()) for x in args.mlp_hidden.split(",") if x.strip()),
        mlp_dropout=args.mlp_dropout,
        mlp_emb_dim=args.mlp_emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    y, x_cov, meta, cal = load_panel(cfg)
    n_series, n_days = y.shape
    train_end = n_days - cfg.val_days

    means = y[:, :train_end].mean(axis=1)
    stds = y[:, :train_end].std(axis=1)
    stds = np.where(stds < 1.0, 1.0, stds)
    y_norm = (y - means[:, None]) / stds[:, None]

    def make_loaders(seq_len: int, use_revin: bool) -> tuple[DataLoader, DataLoader]:
        train_indices, val_indices = build_indices(
            n_series=n_series,
            n_days=n_days,
            seq_len=seq_len,
            train_end=train_end,
            max_train_windows_per_series=cfg.max_train_windows_per_series,
        )
        train_ds = WindowDataset(y_norm, y, x_cov, means, stds, train_indices, seq_len, use_revin=use_revin)
        val_ds = WindowDataset(y_norm, y, x_cov, means, stds, val_indices, seq_len, use_revin=use_revin)
        return (
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False),
        )

    mlp_train_loader, mlp_val_loader = make_loaders(cfg.mlp_seq_len, use_revin=False)
    mlp_revin_train_loader, mlp_revin_val_loader = make_loaders(cfg.mlp_seq_len, use_revin=True)
    transformer_train_loader, transformer_val_loader = make_loaders(cfg.transformer_seq_len, use_revin=False)
    transformer_revin_train_loader, transformer_revin_val_loader = make_loaders(
        cfg.transformer_seq_len, use_revin=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Series: {n_series}; days: {n_days}; "
        f"MLP seq_len: {cfg.mlp_seq_len}; Transformer seq_len: {cfg.transformer_seq_len}"
    )

    cov_dim = x_cov.shape[-1]
    input_dim = 1 + cov_dim
    all_history: list[dict[str, float]] = []
    pred_frames = [baseline_predictions(y, meta, cal, train_end, max(cfg.mlp_seq_len, cfg.transformer_seq_len))]

    set_seed(cfg.seed + 1)
    mlp = LagMLP(
        seq_len=cfg.mlp_seq_len,
        cov_dim=cov_dim,
        n_series=n_series,
        hidden_sizes=cfg.mlp_hidden,
        dropout=cfg.mlp_dropout,
        emb_dim=cfg.mlp_emb_dim,
    )
    mlp, hist = train_model("lag_mlp", mlp, mlp_train_loader, mlp_val_loader, cfg, device, lr=cfg.mlp_lr)
    all_history.extend(hist)
    pred_frames.append(predict_model("lag_mlp", mlp, mlp_val_loader, meta, cal, device))

    set_seed(cfg.seed + 2)
    transformer = TemporalTransformer(
        input_dim=input_dim,
        n_series=n_series,
        seq_len=cfg.transformer_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    transformer, hist = train_model(
        "temporal_transformer",
        transformer,
        transformer_train_loader,
        transformer_val_loader,
        cfg,
        device,
        lr=cfg.transformer_lr,
    )
    all_history.extend(hist)
    pred_frames.append(predict_model("temporal_transformer", transformer, transformer_val_loader, meta, cal, device))

    revin_cov_dim = cov_dim + 2
    revin_input_dim = 1 + revin_cov_dim

    set_seed(cfg.seed + 3)
    revin_mlp = LagMLP(
        seq_len=cfg.mlp_seq_len,
        cov_dim=revin_cov_dim,
        n_series=n_series,
        hidden_sizes=cfg.mlp_hidden,
        dropout=cfg.mlp_dropout,
        emb_dim=cfg.mlp_emb_dim,
    )
    revin_mlp, hist = train_model(
        "lag_mlp_revin",
        revin_mlp,
        mlp_revin_train_loader,
        mlp_revin_val_loader,
        cfg,
        device,
        lr=cfg.mlp_lr,
    )
    all_history.extend(hist)
    pred_frames.append(predict_model("lag_mlp_revin", revin_mlp, mlp_revin_val_loader, meta, cal, device))

    set_seed(cfg.seed + 4)
    revin_transformer = TemporalTransformer(
        input_dim=revin_input_dim,
        n_series=n_series,
        seq_len=cfg.transformer_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    revin_transformer, hist = train_model(
        "temporal_transformer_revin",
        revin_transformer,
        transformer_revin_train_loader,
        transformer_revin_val_loader,
        cfg,
        device,
        lr=cfg.transformer_lr,
    )
    all_history.extend(hist)
    pred_frames.append(
        predict_model(
            "temporal_transformer_revin",
            revin_transformer,
            transformer_revin_val_loader,
            meta,
            cal,
            device,
        )
    )

    preds = pd.concat(pred_frames, ignore_index=True)
    metrics = compute_metrics(preds)
    history = pd.DataFrame(all_history)

    preds.to_csv(cfg.out_dir / "predictions.csv", index=False)
    metrics.to_csv(cfg.out_dir / "metrics.csv", index=False)
    history.to_csv(cfg.out_dir / "train_history.csv", index=False)

    plot_outputs(preds, metrics, history, cfg)
    write_report_notes(cfg, metrics)

    print("\nValidation metrics")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
