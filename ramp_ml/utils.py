# ramp_ml/utils.py
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds for python/random, numpy, torch for reproducibility.

    Note:
      - Exact bitwise reproducibility is easiest on CPU.
      - On GPU, deterministic=True forces deterministic algorithms where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # May raise if an op has no deterministic implementation.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _jsonable(obj: Any) -> Any:
    """Best-effort JSON serialization for argparse Namespace / dict-like objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    # numpy scalars
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def save_predictions_json(path: str, all_pred: Dict[str, np.ndarray]) -> None:
    out = {k: np.asarray(v, dtype=int).tolist() for k, v in all_pred.items()}
    save_json(path, out)


def save_scores_npy(out_dir: str, scores: Dict[str, np.ndarray]) -> None:
    """
    Save per-series probability score arrays as .npy:
      out_dir/T1_score.npy, out_dir/T2_score.npy, ...
    """
    ensure_dir(out_dir)
    for tname, arr in scores.items():
        np.save(os.path.join(out_dir, f"{tname}_score.npy"), np.asarray(arr, dtype=np.float32))


def save_checkpoint(
    path: str,
    model: nn.Module,
    args_dict: Dict[str, Any],
    train_list: list[str],
    test_list: list[str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "args": args_dict,
        "train_list": train_list,
        "test_list": test_list,
        "torch_version": torch.__version__,
        "created_unix": time.time(),
    }
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str) -> Dict[str, Any]:
    # PyTorch 2.6+ defaults weights_only=True; our checkpoint contains metadata objects.
    # This is safe here because the checkpoint is produced by this repo (trusted).
    return torch.load(path, map_location=device, weights_only=False)


def apply_checkpoint(model: nn.Module, ckpt: Dict[str, Any]) -> None:
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()