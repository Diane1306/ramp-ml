from typing import List
import numpy as np


def passes_ramp_gate(
    T: np.ndarray,
    i: int,
    fs: float,
    ramp_sec: float,
    ramp_min_rise: float,
    ramp_slope_min: float,
    event_mode: str,  # drop | rise | both
) -> bool:
    n = len(T)
    wr = int(round(ramp_sec * fs))
    if wr < 2:
        return True
    if i < 0 or i >= n:
        return False

    end_b = i
    start_b = end_b - wr
    if start_b < 0 or end_b > n:
        return False

    seg_b = T[start_b:end_b].astype(float)
    if len(seg_b) < 2:
        return False

    # slope before
    tt_b = np.arange(len(seg_b), dtype=float)
    A_b = np.vstack([tt_b, np.ones_like(tt_b)]).T
    slope_before, _ = np.linalg.lstsq(A_b, seg_b, rcond=None)[0]

    # slope after (~ramp_sec seconds)
    wa = max(int(ramp_sec), int(round(ramp_sec * fs)))
    start_a = i
    end_a = min(n, i + wa)
    if end_a - start_a < 2:
        slope_after = 0.0
    else:
        seg_a = T[start_a:end_a].astype(float)
        tt_a = np.arange(len(seg_a), dtype=float)
        A_a = np.vstack([tt_a, np.ones_like(tt_a)]).T
        slope_after, _ = np.linalg.lstsq(A_a, seg_a, rcond=None)[0]

    # your rejection condition: decreasing before AND increasing after
    if float(slope_before) <= 0.0 and float(slope_after) > 0.0:
        return False

    sb = float(slope_before)
    if event_mode == "drop":
        if sb < ramp_slope_min:
            return False
    elif event_mode == "rise":
        if sb > -ramp_slope_min:
            return False
    else:  # both
        if abs(sb) < ramp_slope_min:
            return False

    low_ramp = float(np.min(seg_b))
    high_ramp = float(np.max(seg_b))

    near_span = max(1, int(round(1.0 * fs)))
    near0 = max(0, i - near_span)
    near1 = min(n, i + near_span + 1)
    near_med = float(np.median(T[near0:near1].astype(float)))

    if event_mode == "drop":
        rise = near_med - low_ramp
        if rise < ramp_min_rise:
            return False
    elif event_mode == "rise":
        fall = high_ramp - near_med
        if fall < ramp_min_rise:
            return False
    else:
        rise = near_med - low_ramp
        fall = high_ramp - near_med
        if max(rise, fall) < ramp_min_rise:
            return False

    return True


def filter_by_ramp_gate(
    pred_idx: np.ndarray,
    T: np.ndarray,
    fs: float,
    use_ramp_gate: bool,
    ramp_sec: float,
    ramp_min_rise: float,
    ramp_slope_min: float,
    event_mode: str,
) -> np.ndarray:
    if (not use_ramp_gate) or pred_idx is None or len(pred_idx) == 0:
        return pred_idx

    keep: List[int] = []
    for i in pred_idx.tolist():
        if passes_ramp_gate(T, i, fs, ramp_sec, ramp_min_rise, ramp_slope_min, event_mode):
            keep.append(i)
    return np.array(keep, dtype=int)