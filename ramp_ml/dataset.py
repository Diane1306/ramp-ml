from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleRef:
    tname: str
    start: int


class MultiResetWindows(Dataset):
    """
    window -> per-sample label vector (1 near reset indices)
    """
    def __init__(
        self,
        T_map: Dict[str, np.ndarray],
        reset_map: Dict[str, np.ndarray],
        win: int,
        stride: int,
        pos_radius: int,
        series_balance: str = "equal",  # equal or proportional
    ):
        self.T_map = T_map
        self.reset_map = reset_map
        self.win = int(win)
        self.stride = int(stride)
        self.pos_radius = int(pos_radius)
        self.series_balance = series_balance

        self.samples: List[SampleRef] = []
        self._labels: List[np.ndarray] = []
        per_series: Dict[str, List[Tuple[SampleRef, np.ndarray]]] = {}

        for tname, x in self.T_map.items():
            n = len(x)
            if n < self.win:
                continue

            resets = self.reset_map.get(tname, np.array([], dtype=int))
            reset_set = set(int(i) for i in resets)

            refs: List[Tuple[SampleRef, np.ndarray]] = []
            for s in range(0, n - self.win + 1, self.stride):
                e = s + self.win
                y = np.zeros(self.win, dtype=np.float32)
                for ridx in reset_set:
                    if s <= ridx < e:
                        center = ridx - s
                        lo = max(0, center - self.pos_radius)
                        hi = min(self.win, center + self.pos_radius + 1)
                        y[lo:hi] = 1.0
                refs.append((SampleRef(tname=tname, start=s), y))
            per_series[tname] = refs

        if series_balance == "equal":
            counts = [len(v) for v in per_series.values() if len(v) > 0]
            if not counts:
                raise ValueError("No usable windows. Check win/stride and data length.")
            k = min(counts)
            for _, refs in per_series.items():
                for r, y in refs[:k]:
                    self.samples.append(r)
                    self._labels.append(y)
        elif series_balance == "proportional":
            for refs in per_series.values():
                for r, y in refs:
                    self.samples.append(r)
                    self._labels.append(y)
        else:
            raise ValueError("series_balance must be 'equal' or 'proportional'")

        self.labels = np.stack(self._labels, axis=0) if self._labels else np.zeros((0, self.win), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def robust_norm(seg: np.ndarray) -> np.ndarray:
        m = np.median(seg)
        mad = np.median(np.abs(seg - m)) + 1e-6
        return (seg - m) / (1.4826 * mad)

    def __getitem__(self, i: int):
        ref = self.samples[i]
        x = self.T_map[ref.tname]
        seg = x[ref.start:ref.start + self.win].astype(np.float32).copy()
        seg = self.robust_norm(seg)
        y = self.labels[i]
        return torch.from_numpy(seg[None, :]), torch.from_numpy(y)