"""Train and evaluate the Phase 2 convolutional autoencoder, per machine id.

Usage (from the repo root, in the venv)::

    python -m models.train --machine-type pump --machine-id id_00 --epochs 2   # smoke
    python -m models.train --machine-type all --machine-id all                  # full run

Each (machine-type, machine-id) is trained independently on its normal clips and
scored on its held-out normal + abnormal test clips. Every run is logged to the
repo-local MLflow store (``./mlruns``); a summary table is written to
``results/phase2.md``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dsp import load_config
from models.autoencoder import build_autoencoder
from models.common import REPO_ROOT, load_yaml, mlflow_setup, select_device, set_seed
from models.dataset import (
    MACHINE_IDS,
    MACHINE_TYPES,
    prepare_machine,
    standardize_apply,
    standardize_fit,
)
from models.evaluate import (
    clip_scores,
    compute_auc,
    compute_pauc,
    reconstruction_error_per_window,
)

# Static description of the pipeline, prepended to results/phase2.md so the table
# is self-documenting and reproducible from the config alone.
_METHOD_NOTE = (
    "**Method:** DCASE-2020-Task-2-style baseline. Each 10 s MIMII clip (single mic, "
    "channel 0) becomes a 64-band log-Mel spectrogram (`configs/dsp.yaml`), sliced into "
    "5-frame context windows. A convolutional autoencoder is trained on **normal clips "
    "only** (per-mel-band z-score fit on train); a clip's anomaly score is the mean "
    "reconstruction error over its windows. AUC and partial-AUC (max FPR = 0.1, "
    "McClish-standardised) are computed with scikit-learn. Higher is better; 0.5 is chance. "
    "Test sets are balanced (held-out normal count matched to abnormal), per machine id."
)


@dataclass
class MachineResult:
    """Evaluation outcome for one machine id."""

    machine_type: str
    machine_id: str
    auc: float
    pauc: float
    n_train_windows: int
    n_test_clips: int
    n_test_normal: int
    n_test_abnormal: int
    epochs: int
    final_val_loss: float
    seconds: float


def _train_model(model, train_x: np.ndarray, cfg: dict[str, Any], device: str):
    """Fit the autoencoder on normal windows; return (model, final_val_loss)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tcfg = cfg["train"]
    x = torch.from_numpy(np.ascontiguousarray(train_x)).float().unsqueeze(1)  # (N,1,M,F)

    n = x.shape[0]
    n_val = max(1, int(round(float(tcfg["val_fraction"]) * n))) if n > 1 else 0
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    x_train, x_val = x[train_idx], x[val_idx]

    loader = DataLoader(
        TensorDataset(x_train),
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        drop_last=False,
    )

    model = model.to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )
    loss_fn = torch.nn.MSELoss()

    final_val = float("nan")
    for epoch in range(int(tcfg["epochs"])):
        model.train()
        running = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.shape[0]
            seen += batch.shape[0]
        train_loss = running / max(seen, 1)

        if n_val > 0:
            model.eval()
            with torch.no_grad():
                xv = x_val.to(device)
                final_val = loss_fn(model(xv), xv).item()
        print(
            f"    epoch {epoch + 1:>3}/{tcfg['epochs']}  "
            f"train_mse={train_loss:.4f}  val_mse={final_val:.4f}"
        )
    return model, final_val


def train_one_machine(
    machine_type: str,
    machine_id: str,
    cfg: dict[str, Any],
    dsp_cfg,
    *,
    device: str,
    use_cache: bool,
    use_mlflow: bool,
    experiment: str,
) -> MachineResult:
    """Prepare features, train, score, and (optionally) log one machine id."""
    t0 = time.perf_counter()
    set_seed(int(cfg["train"]["seed"]))

    feats = prepare_machine(
        machine_type, machine_id, dsp_cfg, cfg, use_cache=use_cache, verbose=True
    )
    n_mels = dsp_cfg.mel.n_mels
    n_frames = int(cfg["features"]["context_frames"])

    # Per-mel-band z-score, fit on normal-only train windows and applied to test.
    train_x, test_x = feats.train_x, feats.test_x
    if bool(cfg["features"].get("standardize", False)):
        mean, std = standardize_fit(train_x)
        train_x = standardize_apply(train_x, mean, std)
        test_x = standardize_apply(test_x, mean, std)

    model = build_autoencoder(cfg["model"], n_mels=n_mels, n_frames=n_frames)
    model, final_val = _train_model(model, train_x, cfg, device)

    errors = reconstruction_error_per_window(
        model, test_x, device=device, batch_size=int(cfg["train"]["batch_size"])
    )
    scores = clip_scores(errors, feats.test_window_clip, feats.n_test_clips)
    y_true = feats.test_clip_label
    auc = compute_auc(y_true, scores)
    pauc = compute_pauc(y_true, scores, p=float(cfg["eval"]["pauc_p"]))

    n_abn = int((y_true == 1).sum())
    n_nrm = int((y_true == 0).sum())
    result = MachineResult(
        machine_type=machine_type,
        machine_id=machine_id,
        auc=auc,
        pauc=pauc,
        n_train_windows=int(feats.train_x.shape[0]),
        n_test_clips=feats.n_test_clips,
        n_test_normal=n_nrm,
        n_test_abnormal=n_abn,
        epochs=int(cfg["train"]["epochs"]),
        final_val_loss=float(final_val),
        seconds=time.perf_counter() - t0,
    )
    print(
        f"  -> {machine_type}/{machine_id}: AUC={auc:.4f} pAUC={pauc:.4f} "
        f"({result.seconds:.0f}s, {result.n_train_windows} train windows)"
    )

    if use_mlflow:
        import mlflow

        with mlflow.start_run(run_name=f"{machine_type}_{machine_id}"):
            mlflow.set_tags(
                {"phase": "2", "machine_type": machine_type, "machine_id": machine_id}
            )
            mlflow.log_params(
                {
                    "lr": cfg["train"]["lr"],
                    "batch_size": cfg["train"]["batch_size"],
                    "epochs": cfg["train"]["epochs"],
                    "bottleneck": cfg["model"]["bottleneck"],
                    "base_channels": cfg["model"]["base_channels"],
                    "context_frames": n_frames,
                    "context_hop": cfg["features"]["context_hop"],
                    "n_mels": n_mels,
                    "device": device,
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc,
                    "pauc": pauc,
                    "final_val_loss": float(final_val),
                    "n_train_windows": float(result.n_train_windows),
                }
            )
    return result


def _resolve_machines(machine_type: str, machine_id: str) -> list[tuple[str, str]]:
    """Expand ``all`` selectors into concrete (type, id) pairs."""
    types = MACHINE_TYPES if machine_type == "all" else (machine_type,)
    ids = MACHINE_IDS if machine_id == "all" else (machine_id,)
    return [(t, i) for t in types for i in ids]


def write_results_markdown(results: list[MachineResult], path: str | Path, *, title: str) -> None:
    """Write a per-id AUC/pAUC table plus per-machine-type means to markdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", _METHOD_NOTE, ""]
    header = (
        "| Machine | id | AUC | pAUC (p=0.1) | Train windows | "
        "Test (norm/abn) | Epochs | Time (s) |"
    )
    lines += [header, "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.machine_type} | {r.machine_id} | {r.auc:.4f} | {r.pauc:.4f} | "
            f"{r.n_train_windows} | {r.n_test_normal}/{r.n_test_abnormal} | "
            f"{r.epochs} | {r.seconds:.0f} |"
        )
    lines.append("")
    lines.append("## Mean per machine type")
    lines.append("")
    lines += ["| Machine | mean AUC | mean pAUC |", "|---|---|---|"]
    for mt in MACHINE_TYPES:
        subset = [r for r in results if r.machine_type == mt]
        if not subset:
            continue
        mean_auc = float(np.mean([r.auc for r in subset]))
        mean_pauc = float(np.mean([r.pauc for r in subset]))
        lines.append(f"| {mt} | {mean_auc:.4f} | {mean_pauc:.4f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[results] wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 acoustic anomaly AE.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "train.yaml"))
    parser.add_argument("--machine-type", default="all", choices=[*MACHINE_TYPES, "all"])
    parser.add_argument("--machine-id", default="all", choices=[*MACHINE_IDS, "all"])
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--device", default=None, help="override config device")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--experiment", default="mismaa-phase2")
    parser.add_argument(
        "--results-out",
        default=None,
        help="markdown path to write; omit to skip (e.g. for smoke tests)",
    )
    parser.add_argument("--results-title", default="Phase 2 — Acoustic anomaly detection")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["train"]["device"] = args.device

    dsp_cfg = load_config()
    set_seed(int(cfg["train"]["seed"]))
    device = select_device(cfg["train"]["device"])
    print(f"[device] {device}")

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow_setup(args.experiment, enabled=True)

    results: list[MachineResult] = []
    for mt, mid in _resolve_machines(args.machine_type, args.machine_id):
        print(f"[train] {mt}/{mid}")
        results.append(
            train_one_machine(
                mt,
                mid,
                cfg,
                dsp_cfg,
                device=device,
                use_cache=not args.no_cache,
                use_mlflow=use_mlflow,
                experiment=args.experiment,
            )
        )

    print("\n=== summary ===")
    for r in results:
        print(f"  {r.machine_type}/{r.machine_id}: AUC={r.auc:.4f} pAUC={r.pauc:.4f}")
    for mt in MACHINE_TYPES:
        subset = [r for r in results if r.machine_type == mt]
        if subset:
            print(
                f"  {mt} mean: AUC={np.mean([r.auc for r in subset]):.4f} "
                f"pAUC={np.mean([r.pauc for r in subset]):.4f}"
            )

    if args.results_out:
        write_results_markdown(results, args.results_out, title=args.results_title)
    # Surface asdict for potential downstream tooling / debugging.
    _ = [asdict(r) for r in results]


if __name__ == "__main__":
    main()
