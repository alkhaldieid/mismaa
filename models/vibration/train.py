"""Train + evaluate both Phase 4 vibration models and write results/phase4.md.

Usage (repo root, venv)::

    python -m models.vibration.train                 # both models, full config
    python -m models.vibration.train --models gb     # feature model only (fast)

Both models share one across-load split (train loads 0/1/2, test load 3) and are
scored by accuracy + a 4x4 confusion matrix. Runs are logged to ./mlruns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from models.common import REPO_ROOT, load_yaml, mlflow_setup, select_device, set_seed

from .cnn import build_cnn
from .dataset import CLASS_NAMES, build_segmented_dataset
from .faults import geometry_from_config
from .features import build_feature_matrix, feature_names


@dataclass
class ModelResult:
    """Evaluation outcome for one Phase 4 model."""

    name: str
    accuracy: float
    confusion: np.ndarray
    n_train: int
    n_test: int
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gradient boosting (feature model)
# ---------------------------------------------------------------------------


def run_gradient_boosting(data, cfg: dict[str, Any]) -> tuple[ModelResult, list[str]]:
    """Fit the gradient-boosting classifier on engineered features and evaluate."""
    geometry = geometry_from_config(cfg)
    sr = data.sample_rate
    feat_cfg = cfg["features"]

    x_train = build_feature_matrix(
        data.train_segments, data.train_rpm, geometry, sr, feat_cfg
    )
    x_test = build_feature_matrix(
        data.test_segments, data.test_rpm, geometry, sr, feat_cfg
    )

    gb_cfg = cfg["gradient_boosting"]
    clf = GradientBoostingClassifier(
        n_estimators=int(gb_cfg["n_estimators"]),
        learning_rate=float(gb_cfg["learning_rate"]),
        max_depth=int(gb_cfg["max_depth"]),
        subsample=float(gb_cfg["subsample"]),
        random_state=int(gb_cfg["seed"]),
    )
    clf.fit(x_train, data.train_labels)
    pred = clf.predict(x_test)

    acc = float(accuracy_score(data.test_labels, pred))
    cm = confusion_matrix(data.test_labels, pred, labels=list(range(len(CLASS_NAMES))))
    names = feature_names(feat_cfg)
    importances = dict(
        sorted(
            zip(names, clf.feature_importances_.tolist(), strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    result = ModelResult(
        name="gradient_boosting",
        accuracy=acc,
        confusion=cm,
        n_train=x_train.shape[0],
        n_test=x_test.shape[0],
        extra={"feature_importances": importances},
    )
    return result, names


# ---------------------------------------------------------------------------
# 1D-CNN (raw signal model)
# ---------------------------------------------------------------------------


def _normalize_segments(x: np.ndarray) -> np.ndarray:
    """Per-segment z-score. Removes absolute amplitude (which shifts with load),
    forcing the CNN to key on waveform *shape* -> better across-load transfer."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def run_cnn(data, cfg: dict[str, Any], *, device: str) -> ModelResult:
    """Train the 1D-CNN on raw segments and evaluate on the held-out load."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cnn_cfg = cfg["cnn"]
    set_seed(int(cnn_cfg["seed"]))

    x_train = _normalize_segments(data.train_segments)
    x_test = _normalize_segments(data.test_segments)
    y_train = data.train_labels

    xt = torch.from_numpy(x_train).unsqueeze(1)  # (N,1,L)
    yt = torch.from_numpy(y_train)

    n = xt.shape[0]
    n_val = max(1, int(round(float(cnn_cfg["val_fraction"]) * n)))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    loader = DataLoader(
        TensorDataset(xt[tr_idx], yt[tr_idx]),
        batch_size=int(cnn_cfg["batch_size"]),
        shuffle=True,
    )

    model = build_cnn(cnn_cfg, n_classes=len(CLASS_NAMES)).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cnn_cfg["lr"]),
        weight_decay=float(cnn_cfg["weight_decay"]),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    xv = xt[val_idx].to(device)
    yv = yt[val_idx].to(device)
    for epoch in range(int(cnn_cfg["epochs"])):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
        model.eval()
        with torch.no_grad():
            val_acc = (model(xv).argmax(1) == yv).float().mean().item()
        print(
            f"    epoch {epoch + 1:>3}/{cnn_cfg['epochs']}  "
            f"train_ce={running / len(tr_idx):.4f}  val_acc={val_acc:.4f}"
        )

    model.eval()
    with torch.no_grad():
        xtest = torch.from_numpy(x_test).unsqueeze(1).to(device)
        pred = model(xtest).argmax(1).cpu().numpy()

    acc = float(accuracy_score(data.test_labels, pred))
    cm = confusion_matrix(data.test_labels, pred, labels=list(range(len(CLASS_NAMES))))
    return ModelResult(
        name="cnn_1d",
        accuracy=acc,
        confusion=cm,
        n_train=x_train.shape[0],
        n_test=x_test.shape[0],
        extra={"device": device},
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def confusion_markdown(cm: np.ndarray, class_names: tuple[str, ...]) -> list[str]:
    """Render a confusion matrix (rows = true, cols = predicted) as markdown."""
    header = "| true \\ pred | " + " | ".join(class_names) + " |"
    sep = "|" + "---|" * (len(class_names) + 1)
    lines = [header, sep]
    for i, name in enumerate(class_names):
        row = " | ".join(str(int(v)) for v in cm[i])
        lines.append(f"| **{name}** | {row} |")
    return lines


def write_results_markdown(
    results: list[ModelResult], cfg: dict[str, Any], path: str | Path
) -> None:
    """Write accuracy + confusion matrices for every model to markdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tl = ", ".join(str(x) for x in cfg["split"]["train_loads"])
    lines = [
        "# Phase 4 — Vibration fault diagnosis (CWRU)",
        "",
        "4-class bearing-fault classification: "
        + ", ".join(CLASS_NAMES).replace("_", " ")
        + ".",
        "",
        f"**Split:** generalisation across load — train on motor loads {tl} (HP), "
        f"test on the unseen load {cfg['split']['test_load']}. Holding out a whole "
        "operating condition is harder and more honest than a random segment split, "
        "which would leak near-duplicate neighbouring windows from the same file "
        "into the test set and inflate accuracy.",
        "",
        "| Model | Test accuracy | Train segments | Test segments |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.accuracy:.4f} | {r.n_train} | {r.n_test} |")
    lines.append("")

    for r in results:
        lines.append(f"## {r.name} — confusion matrix")
        lines.append("")
        lines += confusion_markdown(r.confusion, CLASS_NAMES)
        lines.append("")
        if "feature_importances" in r.extra:
            top = list(r.extra["feature_importances"].items())[:6]
            lines.append("Top features by gradient-boosting importance:")
            lines.append("")
            for name, imp in top:
                lines.append(f"- `{name}`: {imp:.3f}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[results] wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Phase 4 vibration models.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "vibration.yaml"))
    parser.add_argument("--models", default="both", choices=["both", "gb", "cnn"])
    parser.add_argument("--epochs", type=int, default=None, help="override CNN epochs")
    parser.add_argument("--device", default=None, help="override CNN device")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--experiment", default="mismaa-phase4")
    parser.add_argument("--results-out", default=str(REPO_ROOT / "results" / "phase4.md"))
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg["cnn"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["cnn"]["device"] = args.device

    set_seed(int(cfg["gradient_boosting"]["seed"]))
    data = build_segmented_dataset(cfg)
    print(
        f"[data] train segments={data.train_segments.shape[0]} "
        f"test segments={data.test_segments.shape[0]} "
        f"(loads {cfg['split']['train_loads']} -> {cfg['split']['test_load']})"
    )

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow_setup(args.experiment, enabled=True)

    results: list[ModelResult] = []

    if args.models in ("both", "gb"):
        print("[gb] gradient boosting on engineered features ...")
        gb_result, _ = run_gradient_boosting(data, cfg)
        print(f"  -> gradient_boosting accuracy = {gb_result.accuracy:.4f}")
        results.append(gb_result)
        if use_mlflow:
            _log_mlflow(gb_result, cfg, params=cfg["gradient_boosting"])

    if args.models in ("both", "cnn"):
        device = select_device(cfg["cnn"]["device"])
        print(f"[cnn] 1D-CNN on raw segments (device={device}) ...")
        cnn_result = run_cnn(data, cfg, device=device)
        print(f"  -> cnn_1d accuracy = {cnn_result.accuracy:.4f}")
        results.append(cnn_result)
        if use_mlflow:
            _log_mlflow(cnn_result, cfg, params={**cfg["cnn"], "device": device})

    write_results_markdown(results, cfg, args.results_out)

    print("\n=== summary ===")
    for r in results:
        print(f"  {r.name}: accuracy={r.accuracy:.4f}")


def _log_mlflow(result: ModelResult, cfg: dict[str, Any], *, params: dict[str, Any]) -> None:
    """Log one model's params + metrics to MLflow."""
    import mlflow

    with mlflow.start_run(run_name=result.name):
        mlflow.set_tags({"phase": "4", "model": result.name})
        mlflow.log_params(
            {
                f"{result.name}.{k}": v
                for k, v in params.items()
                if isinstance(v, (int, float, str, bool))
            }
        )
        mlflow.log_params(
            {
                "train_loads": str(cfg["split"]["train_loads"]),
                "test_load": cfg["split"]["test_load"],
                "segment_length": cfg["segment"]["length"],
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": result.accuracy,
                "n_train": float(result.n_train),
                "n_test": float(result.n_test),
            }
        )


if __name__ == "__main__":
    main()
