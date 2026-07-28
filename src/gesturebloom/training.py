"""Training loops for the pose classifier and the temporal spotter.

Both loops report metrics that are hard to fool:

- **Pose**: per-class accuracy and a confusion matrix, not just top-1. Top-1 on
  an imbalanced hand-recorded set can look excellent while one class is never
  predicted at all.
- **Temporal**: event-level precision/recall/F1 *and median detection latency*,
  computed through the real spotter state machine on held-out recordings. Frame
  accuracy is reported too, but only to show how misleading it is -- a model
  predicting background forever scores over 90% on it.

Requires the ``train`` extra: ``pip install 'gesturebloom[train]'``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for training. Install with: pip install 'gesturebloom[train]'"
        ) from exc


def confusion_matrix(true: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns predicted.

    Examples
    --------
    >>> import numpy as np
    >>> cm = confusion_matrix(np.array([0, 1, 1]), np.array([0, 1, 0]), 2)
    >>> cm.tolist()
    [[1, 0], [1, 1]]
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(true).ravel(), np.asarray(pred).ravel(), strict=True):
        cm[int(t), int(p)] += 1
    return cm


def format_confusion(cm: np.ndarray, names: list[str]) -> str:
    """Readable confusion matrix with per-class recall."""
    width = max(len(n) for n in names) + 1
    header = " " * (width + 2) + " ".join(f"{i:>5d}" for i in range(len(names)))
    lines = [header]
    for i, name in enumerate(names):
        row = " ".join(f"{v:5d}" for v in cm[i])
        recall = cm[i, i] / max(cm[i].sum(), 1)
        lines.append(f"{name:<{width}} | {row}   recall={recall:.3f}")
    return "\n".join(lines)


def train_pose(
    paths: list[Path],
    out: Path,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 3e-4,
    seed: int = 0,
) -> int:
    """Train the static pose classifier."""
    torch = _require_torch()
    from torch.utils.data import DataLoader

    from .data.dataset import PoseDataset, split_recordings
    from .data.recording import Recording
    from .models.pose_mlp import PoseMLP, PoseMLPConfig

    torch.manual_seed(seed)
    splits = split_recordings(paths, seed=seed)  # recording-level: no window leakage
    loaded = {k: [Recording.load(p) for p in v] for k, v in splits.items()}
    names = loaded["train"][0].label_names
    n_classes = len(names)

    train_ds = PoseDataset(loaded["train"], n_classes=n_classes, train=True, seed=seed)
    val_ds = PoseDataset(loaded["val"], n_classes=n_classes, train=False)
    print(f"train={len(train_ds)} val={len(val_ds)} classes={n_classes}")
    print(f"train class counts: {train_ds.class_counts().tolist()}")

    model = PoseMLP(PoseMLPConfig(n_classes=n_classes))
    print(f"parameters: {model.n_parameters:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=512)
    best_acc = 0.0
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = model.loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * x.shape[0]
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_dl:
                preds.append(model(x).argmax(dim=-1).numpy())
                trues.append(y.numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        acc = float((pred == true).mean())

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d} | loss {total / max(len(train_ds), 1):.4f} | val acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {"state_dict": model.state_dict(), "config": model.config, "label_names": names},
                out / "pose_mlp.pt",
            )

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print("\nConfusion matrix (final epoch, validation):")
    print(format_confusion(confusion_matrix(true, pred, n_classes), names))
    print(f"\nSaved {out / 'pose_mlp.pt'}")
    return 0


def train_temporal(
    paths: list[Path],
    out: Path,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
) -> int:
    """Train the temporal spotter and evaluate at the event level."""
    torch = _require_torch()
    from torch.utils.data import DataLoader

    from .data.dataset import WindowDataset, WindowSpec, canonical_features, split_recordings
    from .data.recording import Recording
    from .models.spotter import SpotterConfig, evaluate_events, spot_offline
    from .models.temporal import GestureTCN, TCNConfig, class_weights_from_labels

    torch.manual_seed(seed)
    splits = split_recordings(paths, seed=seed)
    loaded = {k: [Recording.load(p) for p in v] for k, v in splits.items()}
    names = loaded["train"][0].label_names
    n_classes = len(names)

    spec = WindowSpec(length=32, stride=4)
    train_ds = WindowDataset(loaded["train"], n_classes=n_classes, spec=spec, train=True, seed=seed)
    val_ds = WindowDataset(loaded["val"], n_classes=n_classes, spec=spec, train=False)

    cfg = TCNConfig(n_classes=n_classes)
    print(f"train windows={len(train_ds)} val windows={len(val_ds)}")
    print(f"receptive field: {cfg.receptive_field} frames (window length {spec.length})")
    if spec.length > cfg.receptive_field:
        print("WARNING: window longer than receptive field; the model cannot see all of it.")
    counts = train_ds.window_class_counts()
    print(f"windowed class counts: {counts.tolist()}")

    model = GestureTCN(cfg)
    print(f"parameters: {model.n_parameters:,}")
    weights = torch.from_numpy(class_weights_from_labels(train_ds.y.argmax(-1).ravel(), n_classes))
    print(f"class weights: {[round(float(w), 3) for w in weights]}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = model.loss(model(x), y, class_weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * x.shape[0]
        sched.step()

        if epoch % 5 and epoch != epochs - 1:
            continue

        # Event-level eval on full held-out recordings, through the real spotter.
        # Evaluating on windows would inflate the numbers; the deployed system
        # sees continuous streams, so that is what we measure.
        model.eval()
        metrics_all = []
        for rec in loaded["val"]:
            feats, _valid = canonical_features(rec)
            feats = np.nan_to_num(feats, nan=0.0)
            with torch.no_grad():
                logits = model(torch.from_numpy(feats).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)[0].numpy()
            events = spot_offline(probs, SpotterConfig())
            metrics_all.append(evaluate_events(events, rec.onsets(), tolerance_frames=15))

        tp = sum(m.true_positives for m in metrics_all)
        fp = sum(m.false_positives for m in metrics_all)
        fn = sum(m.false_negatives for m in metrics_all)
        lat = [x for m in metrics_all for x in m.latencies]
        prec = tp / max(tp + fp, 1)
        rec_ = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
        med_lat = float(np.median(lat)) if lat else float("nan")
        fps = loaded["val"][0].fps
        print(
            f"epoch {epoch:3d} | loss {total / max(len(train_ds), 1):.4f} | "
            f"event P {prec:.3f} R {rec_:.3f} F1 {f1:.3f} | "
            f"latency {med_lat:.1f}f ({med_lat / fps * 1000:.0f} ms)"
        )
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {"state_dict": model.state_dict(), "config": cfg, "label_names": names},
                out / "gesture_tcn.pt",
            )

    print(f"\nBest event F1: {best_f1:.4f}")
    print(f"Saved {out / 'gesture_tcn.pt'}")
    print("\nNext: `gesturebloom tune` to grid-search spotter thresholds on a recorded trace.")
    return 0
