import argparse
from typing import Set, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ramp_ml.io import read_T_map, read_VITA_map, sanity_check
from ramp_ml.dataset import MultiResetWindows
from ramp_ml.model import TCNVITA
from ramp_ml.inference import infer_scores_one_series, pick_events_one_per_window
from ramp_ml.gating import filter_by_ramp_gate
from ramp_ml.eval import match_events_with_tolerance
from ramp_ml.plotting import plot_series_with_events
from ramp_ml.utils import (
    set_seed,
    ensure_dir,
    save_json,
    save_predictions_json,
    save_scores_npy,
    save_checkpoint,
    load_checkpoint,
    apply_checkpoint,
)

def _parse_series_list(arg: Optional[str], available: Set[str], what: str) -> Optional[List[str]]:
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        return None
    names = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in names if x not in available]
    if bad:
        raise ValueError(f"{what}: unknown series {bad}. Available: {sorted(list(available))}")
    return names


def _subset_map(full: Dict[str, np.ndarray], keep: List[str]) -> Dict[str, np.ndarray]:
    return {k: full[k] for k in keep if k in full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_path", required=True)
    ap.add_argument("--VITA_path", required=True)
    ap.add_argument("--t_sheet", type=int, default=0)
    ap.add_argument("--VITA_sheet", type=int, default=0)
    ap.add_argument("--mapping", default="VITA_to_T", choices=["VITA_to_T", "same"])

    ap.add_argument("--fs", type=float, default=1.0)
    ap.add_argument("--win_sec", type=float, default=180.0)
    ap.add_argument("--stride_sec", type=float, default=25.0)
    ap.add_argument("--pos_radius_sec", type=float, default=2.0)
    ap.add_argument("--series_balance", default="equal", choices=["equal", "proportional"])

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)

    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--min_sep_sec", type=float, default=25.0)
    ap.add_argument("--drop_pre_sec", type=float, default=2.0)
    ap.add_argument("--drop_post_sec", type=float, default=2.0)

    ap.add_argument("--use_ramp_gate", action="store_true")
    ap.add_argument("--ramp_sec", type=float, default=10.0)
    ap.add_argument("--ramp_min_rise", type=float, default=0.6)
    ap.add_argument("--ramp_slope_min", type=float, default=0.05)

    ap.add_argument("--event_mode", default="drop", choices=["drop", "rise", "both"])

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_dir", default="plots")
    ap.add_argument("--max_points", type=int, default=None)

    ap.add_argument("--train_T", default=None)
    ap.add_argument("--test_T", default=None)
    ap.add_argument("--match_tol_sec", type=float, default=2.0)

    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--deterministic", action="store_true",
                    help="Force deterministic algorithms (recommended for reproduction)")

    ap.add_argument("--save_dir", type=str, default=None, help="Directory to save run artifacts")
    ap.add_argument("--save_model", action="store_true", help="Save checkpoint/config/predictions into save_dir")
    ap.add_argument("--save_scores", action="store_true", help="Also save per-series score arrays (.npy) into save_dir")

    ap.add_argument("--load_model", type=str, default=None,
                    help="Path to checkpoint.pt to reproduce predictions (skips training)")
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                    help="Force device (cpu is most reproducible)")

    args = ap.parse_args()

    # device selection (CPU is easiest for exact reproducibility)
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed, deterministic=args.deterministic)

    fs = float(args.fs)
    win = int(round(args.win_sec * fs))
    stride = int(round(args.stride_sec * fs))
    pos_radius = int(round(args.pos_radius_sec * fs))
    min_sep = int(round(args.min_sep_sec * fs))
    pre_n = max(1, int(round(args.drop_pre_sec * fs)))
    post_n = max(1, int(round(args.drop_post_sec * fs)))
    tol_n = max(0, int(round(args.match_tol_sec * fs)))

    # load
    T_all = read_T_map(args.t_path, sheet=args.t_sheet)
    VITA_all = read_VITA_map(args.VITA_path, sheet=args.VITA_sheet, mapping=args.mapping)
    sanity_check(T_all, VITA_all)

    available = set(T_all.keys())
    train_list = _parse_series_list(args.train_T, available, "--train_T")
    test_list = _parse_series_list(args.test_T, available, "--test_T")

    if train_list is None:
        train_list = sorted(list(available), key=lambda s: int(s[1:]))
    if test_list is None:
        test_list = sorted(list(available), key=lambda s: int(s[1:]))

    T_train = _subset_map(T_all, train_list)
    VITA_train = _subset_map(VITA_all, train_list)
    T_test = _subset_map(T_all, test_list)
    VITA_test = _subset_map(VITA_all, test_list)

    print("\nTRAIN series:", train_list)
    print("TEST  series:", test_list)
    print("event_mode:", args.event_mode)

    save_dir = None
    if args.save_model or args.save_scores:
        if args.save_dir is None:
            raise ValueError("--save_dir must be provided when using --save_model or --save_scores")
        save_dir = ensure_dir(args.save_dir)

    # dataset + loader
    ds = MultiResetWindows(T_train, VITA_train, win=win, stride=stride, pos_radius=pos_radius, series_balance=args.series_balance)
    if len(ds) == 0:
        raise ValueError("Training dataset is empty. Check train_T and win/stride.")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCNVITA().to(device)

    # If loading a checkpoint, skip training and reproduce predictions
    if args.load_model is not None:
        ckpt = load_checkpoint(args.load_model, device=device)
        apply_checkpoint(model, ckpt)
        print(f"[loaded] {args.load_model}")

    pos_frac = float(ds.labels.mean()) if ds.labels.size else 0.0
    pos_weight = torch.tensor([(1.0 - pos_frac) / (pos_frac + 1e-8)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.load_model is None:
        # train
        model.train()
        for epoch in range(args.epochs):
            losses = []
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            print(f"epoch {epoch+1:02d} loss={float(np.mean(losses)):.4f}")

    # test
    print("\nInference / Compare-to-VITA (TEST set):")
    total_tp = total_fp = total_fn = 0
    all_pred = {}
    all_scores = {}

    for tname in sorted(T_test.keys(), key=lambda s: int(s[1:])):
        T = T_test[tname]
        score = infer_scores_one_series(model, T, win=win, stride=stride, device=device)

        if args.save_scores:
            all_scores[tname] = score

        pred_idx = pick_events_one_per_window(
            score=score,
            T=T,
            thr=args.thr,
            min_sep=min_sep,
            pre_n=pre_n,
            post_n=post_n,
            event_mode=args.event_mode,
        )

        pred_idx = filter_by_ramp_gate(
            pred_idx=pred_idx,
            T=T,
            fs=fs,
            use_ramp_gate=args.use_ramp_gate,
            ramp_sec=args.ramp_sec,
            ramp_min_rise=args.ramp_min_rise,
            ramp_slope_min=args.ramp_slope_min,
            event_mode=args.event_mode,
        )

        all_pred[tname] = pred_idx

        gt = VITA_test.get(tname, np.array([], dtype=int))
        tp, fp, fn = match_events_with_tolerance(pred_idx, gt, tol=tol_n)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"  {tname}: pred={len(pred_idx)} gt={len(gt)} | TP={tp} FP={fp} FN={fn} (tol=±{tol_n}s)")

        if args.plot:
            import os
            os.makedirs(args.plot_dir, exist_ok=True)
            out_png = os.path.join(args.plot_dir, f"{tname}_events.png")
            plot_series_with_events(
                T=T,
                pred_idx=pred_idx,
                gt_idx=gt,
                title=f"{tname} | pred={len(pred_idx)} gt={len(gt)}",
                out_png=out_png,
                max_points=args.max_points,
            )

    prec = total_tp / (total_tp + total_fp + 1e-12)
    rec = total_tp / (total_tp + total_fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    print(f"\nTEST summary: TP={total_tp} FP={total_fp} FN={total_fn} | Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

    if save_dir is not None:
        # Save config used for this run (including seed)
        save_json(os.path.join(save_dir, "config.json"), vars(args))
        save_json(os.path.join(save_dir, "train_test.json"), {"train_list": train_list, "test_list": test_list})

        # Save predictions
        save_predictions_json(os.path.join(save_dir, "pred_idx.json"), all_pred)

        # Save scores (optional)
        if args.save_scores:
            save_scores_npy(save_dir, all_scores)

        # Save checkpoint (optional)
        if args.save_model:
            save_checkpoint(
                os.path.join(save_dir, "checkpoint.pt"),
                model=model,
                args_dict=vars(args),
                train_list=train_list,
                test_list=test_list,
            )
            print(f"[saved] {os.path.join(save_dir, 'checkpoint.pt')}")