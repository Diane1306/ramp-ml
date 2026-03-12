from typing import Tuple
import numpy as np


def match_events_with_tolerance(pred: np.ndarray, gt: np.ndarray, tol: int) -> Tuple[int, int, int]:
    pred = np.asarray(pred, dtype=int)
    gt = np.asarray(gt, dtype=int)

    if pred.size == 0 and gt.size == 0:
        return (0, 0, 0)
    if pred.size == 0:
        return (0, 0, int(gt.size))
    if gt.size == 0:
        return (0, int(pred.size), 0)

    pred = np.sort(pred)
    gt = np.sort(gt)

    used = np.zeros(gt.shape[0], dtype=bool)
    tp = 0

    for p in pred:
        jbest = -1
        dbest = None
        j0 = np.searchsorted(gt, p)
        for j in [j0 - 2, j0 - 1, j0, j0 + 1, j0 + 2]:
            if j < 0 or j >= gt.size or used[j]:
                continue
            d = abs(int(gt[j]) - int(p))
            if d <= tol and (dbest is None or d < dbest):
                dbest = d
                jbest = j
        if jbest >= 0:
            used[jbest] = True
            tp += 1

    fp = int(pred.size) - tp
    fn = int(gt.size) - int(np.sum(used))
    return tp, fp, fn