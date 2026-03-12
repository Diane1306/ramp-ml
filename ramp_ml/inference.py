from typing import List
import numpy as np
import torch
import torch.nn as nn


def infer_scores_one_series(model: nn.Module, T: np.ndarray, win: int, stride: int, device: str) -> np.ndarray:
    model.eval()
    n = len(T)
    score = np.full(n, -np.inf, dtype=np.float32)

    def robust_norm(seg: np.ndarray) -> np.ndarray:
        m = np.median(seg)
        mad = np.median(np.abs(seg - m)) + 1e-6
        return (seg - m) / (1.4826 * mad)

    with torch.no_grad():
        for s in range(0, n - win + 1, stride):
            seg = robust_norm(T[s:s + win].astype(np.float32))
            xb = torch.from_numpy(seg[None, None, :]).to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy().squeeze(0)
            score[s:s + win] = np.maximum(score[s:s + win], probs.astype(np.float32))
    return score


def local_max_candidates(score: np.ndarray, thr: float) -> List[int]:
    n = len(score)
    cand: List[int] = []
    for i in range(1, n - 1):
        if score[i] >= thr and score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            cand.append(i)
    return cand


def event_magnitude_mean(T: np.ndarray, i: int, pre_n: int, post_n: int, mode: str) -> float:
    n = len(T)
    if i <= 0 or i >= n:
        return -np.inf

    a0 = max(0, i - pre_n)
    a1 = i
    b0 = i
    b1 = min(n, i + post_n)
    if a1 - a0 < 1 or b1 - b0 < 1:
        return -np.inf

    before = float(np.mean(T[a0:a1]))
    after = float(np.mean(T[b0:b1]))

    drop_mag = before - after
    rise_mag = after - before

    if mode == "drop":
        return drop_mag
    if mode == "rise":
        return rise_mag
    return max(drop_mag, rise_mag)  # both


def pick_events_one_per_window(
    score: np.ndarray,
    T: np.ndarray,
    thr: float,
    min_sep: int,
    pre_n: int,
    post_n: int,
    event_mode: str,
) -> np.ndarray:
    cand = local_max_candidates(score, thr=thr)
    if not cand:
        return np.array([], dtype=int)

    cand = sorted(cand)
    picks: List[int] = []
    cluster: List[int] = [cand[0]]

    def best_in_cluster(idxs: List[int]) -> int:
        best_i = idxs[0]
        best_mag = event_magnitude_mean(T, best_i, pre_n, post_n, event_mode)
        best_score = float(score[best_i])
        for j in idxs[1:]:
            m = event_magnitude_mean(T, j, pre_n, post_n, event_mode)
            s = float(score[j])
            if (m > best_mag) or (m == best_mag and s > best_score):
                best_i, best_mag, best_score = j, m, s
        return best_i

    for idx in cand[1:]:
        if idx - cluster[-1] <= min_sep:
            cluster.append(idx)
        else:
            picks.append(best_in_cluster(cluster))
            cluster = [idx]

    picks.append(best_in_cluster(cluster))
    return np.array(picks, dtype=int)