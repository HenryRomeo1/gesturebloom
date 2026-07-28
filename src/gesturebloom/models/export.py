"""ONNX export.

Worth doing even though the Python pipeline runs fine, for two reasons that are
about the repo as a portfolio artifact rather than about performance:

1. It proves the model is genuinely deployable -- no Python-only ops, no dynamic
   control flow that only works in eager mode.
2. onnxruntime CPU inference is typically 2-4x faster than PyTorch eager for
   models this small, because the overhead *is* the cost at 40k parameters.

The verification step at the end is the part people skip. An export that produces
different numbers than the source model is worse than no export, and it happens
easily -- a training-mode BatchNorm, a squeezed dynamic axis. Always assert
agreement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_onnx(
    checkpoint: str | Path,
    out: str | Path,
    task: str = "pose",
    opset: int = 17,
    verify_tolerance: float = 1e-4,
) -> Path:
    """Export a checkpoint to ONNX and verify numerical agreement.

    Parameters
    ----------
    task:
        ``"pose"`` exports a fixed ``(1, D)`` input. ``"temporal"`` exports with a
        dynamic time axis so a single graph serves any window length -- necessary
        because the deployed spotter streams variable-length buffers.
    verify_tolerance:
        Max allowed absolute difference between torch and onnxruntime outputs.

    Raises
    ------
    AssertionError
        If the exported graph disagrees with the source model.
    """
    import torch

    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if task == "pose":
        from .pose_mlp import PoseMLP

        model = PoseMLP(ckpt["config"])
        dummy = torch.randn(1, ckpt["config"].input_dim)
        dynamic_axes = {"landmarks": {0: "batch"}, "logits": {0: "batch"}}
    else:
        from .temporal import GestureTCN

        model = GestureTCN(ckpt["config"])
        dummy = torch.randn(1, 32, ckpt["config"].input_dim)
        dynamic_axes = {
            "landmarks": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "time"},
        }

    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # critical: BatchNorm/Dropout must be in inference mode

    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=opset,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping verification (install to verify)")
        return out

    with torch.no_grad():
        expected = model(dummy).numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"landmarks": dummy.numpy()})[0]

    diff = float(np.abs(expected - actual).max())
    assert diff < verify_tolerance, (
        f"ONNX export disagrees with the source model (max abs diff {diff:.2e} > "
        f"{verify_tolerance:.0e}). Check that the model is in eval() mode and that "
        f"no dynamic axis was collapsed."
    )
    print(f"verified: max abs diff {diff:.2e}")

    if task == "temporal":
        # Prove the dynamic time axis actually works at a length we did not trace.
        alt = np.random.randn(1, 97, ckpt["config"].input_dim).astype(np.float32)
        session.run(None, {"landmarks": alt})
        print("verified: dynamic time axis accepts T=97")

    return out
