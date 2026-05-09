from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_forecaster import (
    Config,
    LagMLP,
    TemporalTransformer,
    WindowDataset,
    build_indices,
    compute_metrics,
    ensure_dirs,
    load_panel,
    predict_model,
    set_seed,
    train_model,
)


ROOT = Path(__file__).resolve().parents[1]
TUNE_OUT = ROOT / "outputs" / "tuning"


def loaders_for_config(cfg: Config, use_revin: bool):
    y, x_cov, meta, cal = load_panel(cfg)
    n_series, n_days = y.shape
    train_end = n_days - cfg.val_days
    means = y[:, :train_end].mean(axis=1)
    stds = y[:, :train_end].std(axis=1)
    stds = np.where(stds < 1.0, 1.0, stds)
    y_norm = (y - means[:, None]) / stds[:, None]
    train_indices, val_indices = build_indices(
        n_series=n_series,
        n_days=n_days,
        seq_len=cfg.mlp_seq_len,
        train_end=train_end,
        max_train_windows_per_series=cfg.max_train_windows_per_series,
    )
    train_ds = WindowDataset(y_norm, y, x_cov, means, stds, train_indices, cfg.mlp_seq_len, use_revin=use_revin)
    val_ds = WindowDataset(y_norm, y, x_cov, means, stds, val_indices, cfg.mlp_seq_len, use_revin=use_revin)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return y, x_cov, meta, cal, train_loader, val_loader


def run_candidate(kind: str, params: dict, device: torch.device) -> dict:
    cfg = Config(
        out_dir=TUNE_OUT / f"{kind}_{params['name']}",
        epochs=params.get("epochs", 5),
        mlp_seq_len=params["seq_len"],
        mlp_lr=params["lr"],
        transformer_seq_len=params["seq_len"],
        transformer_lr=params["lr"],
        batch_size=512,
        max_train_windows_per_series=700,
        mlp_hidden=tuple(params.get("mlp_hidden", (128, 64))),
        mlp_dropout=params.get("mlp_dropout", 0.10),
        mlp_emb_dim=params.get("mlp_emb_dim", 8),
        d_model=params.get("d_model", 48),
        n_heads=params.get("n_heads", 4),
        n_layers=params.get("n_layers", 2),
        dropout=params.get("dropout", 0.10),
    )
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    y, x_cov, meta, cal, train_loader, val_loader = loaders_for_config(cfg, use_revin=True)
    n_series = y.shape[0]
    cov_dim = x_cov.shape[-1] + 2
    start = time.time()

    if kind == "mlp_revin":
        model = LagMLP(
            seq_len=cfg.mlp_seq_len,
            cov_dim=cov_dim,
            n_series=n_series,
            hidden_sizes=cfg.mlp_hidden,
            dropout=cfg.mlp_dropout,
            emb_dim=cfg.mlp_emb_dim,
        )
    elif kind == "transformer_revin":
        model = TemporalTransformer(
            input_dim=1 + cov_dim,
            n_series=n_series,
            seq_len=cfg.transformer_seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(kind)

    lr = cfg.mlp_lr if kind == "mlp_revin" else cfg.transformer_lr
    model, history = train_model(kind, model, train_loader, val_loader, cfg, device, lr=lr)
    preds = predict_model(kind, model, val_loader, meta, cal, device)
    metrics = compute_metrics(preds).iloc[0].to_dict()
    best_val_norm = min(row["val_mae_norm"] for row in history)
    row = {
        "kind": kind,
        "name": params["name"],
        "seq_len": params["seq_len"],
        "lr": lr,
        "epochs": cfg.epochs,
        "best_val_mae_norm": best_val_norm,
        "seconds": round(time.time() - start, 2),
        **{k: v for k, v in params.items() if k != "name"},
        **metrics,
    }
    print(json.dumps(row))
    return row


def main() -> None:
    TUNE_OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_candidates = [
        {"name": "mlp_a", "seq_len": 56, "lr": 1e-3, "mlp_hidden": (128, 64), "mlp_dropout": 0.10, "mlp_emb_dim": 8},
        {"name": "mlp_b", "seq_len": 56, "lr": 1e-3, "mlp_hidden": (256, 128), "mlp_dropout": 0.10, "mlp_emb_dim": 16},
        {"name": "mlp_c", "seq_len": 56, "lr": 5e-4, "mlp_hidden": (256, 128), "mlp_dropout": 0.05, "mlp_emb_dim": 16},
        {"name": "mlp_d", "seq_len": 84, "lr": 1e-3, "mlp_hidden": (256, 128), "mlp_dropout": 0.10, "mlp_emb_dim": 16},
        {"name": "mlp_e", "seq_len": 84, "lr": 5e-4, "mlp_hidden": (256, 128, 64), "mlp_dropout": 0.10, "mlp_emb_dim": 16},
        {"name": "mlp_f", "seq_len": 112, "lr": 5e-4, "mlp_hidden": (256, 128), "mlp_dropout": 0.10, "mlp_emb_dim": 16},
        {"name": "mlp_g", "seq_len": 84, "lr": 7e-4, "mlp_hidden": (384, 192), "mlp_dropout": 0.15, "mlp_emb_dim": 24},
        {"name": "mlp_h", "seq_len": 56, "lr": 7e-4, "mlp_hidden": (256, 128, 64), "mlp_dropout": 0.20, "mlp_emb_dim": 16},
    ]
    transformer_candidates = [
        {"name": "tf_a", "seq_len": 56, "lr": 1e-3, "d_model": 48, "n_heads": 4, "n_layers": 2, "dropout": 0.10},
        {"name": "tf_b", "seq_len": 56, "lr": 5e-4, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.10},
        {"name": "tf_c", "seq_len": 84, "lr": 5e-4, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.10},
        {"name": "tf_d", "seq_len": 84, "lr": 5e-4, "d_model": 64, "n_heads": 4, "n_layers": 3, "dropout": 0.15},
        {"name": "tf_e", "seq_len": 56, "lr": 7e-4, "d_model": 96, "n_heads": 4, "n_layers": 2, "dropout": 0.15},
        {"name": "tf_f", "seq_len": 112, "lr": 5e-4, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.15},
    ]

    rows = []
    for params in mlp_candidates:
        rows.append(run_candidate("mlp_revin", params, device))
    for params in transformer_candidates:
        rows.append(run_candidate("transformer_revin", params, device))

    results = pd.DataFrame(rows).sort_values(["kind", "MAE"])
    results.to_csv(ROOT / "outputs" / "tuning_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
