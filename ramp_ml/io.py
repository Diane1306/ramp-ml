import re
from typing import Dict
import numpy as np
import pandas as pd


def _read_table(path: str, sheet: int = 0) -> pd.DataFrame:
    pl = path.lower()
    if pl.endswith(".xlsx") or pl.endswith(".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    if pl.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}. Use .xlsx/.xls/.csv")


def read_T_map(t_path: str, sheet: int = 0) -> Dict[str, np.ndarray]:
    df = _read_table(t_path, sheet=sheet)
    out: Dict[str, np.ndarray] = {}
    for c in df.columns:
        name = str(c).strip()
        if re.fullmatch(r"T\d+", name):
            arr = df[name].astype("float32").dropna().to_numpy()
            if len(arr) > 0:
                out[name] = arr
    if not out:
        raise ValueError("No T# columns found (expect columns named like T1, T2, ...).")
    return out


def read_VITA_map(VITA_path: str, sheet: int = 0, mapping: str = "VITA_to_T") -> Dict[str, np.ndarray]:
    """
    mapping:
      - VITA_to_T: VITA1 -> T1, VITA2 -> T2, ...
      - same: reset columns already named T1, T2, ...
    """
    df = _read_table(VITA_path, sheet=sheet)
    out: Dict[str, np.ndarray] = {}

    if mapping == "same":
        for c in df.columns:
            name = str(c).strip()
            if re.fullmatch(r"T\d+", name):
                out[name] = np.sort(df[c].dropna().astype(int).to_numpy())
        return out

    if mapping == "VITA_to_T":
        for c in df.columns:
            col = str(c).strip()
            m = re.fullmatch(r"VITA(\d+)", col, flags=re.IGNORECASE)
            if not m:
                continue
            j = int(m.group(1))
            tname = f"T{j}"
            out[tname] = np.sort(df[c].dropna().astype(int).to_numpy())
        return out

    raise ValueError(f"Unknown mapping={mapping}")


def sanity_check(T_map: Dict[str, np.ndarray], VITA_map: Dict[str, np.ndarray]) -> None:
    for tname, idx in VITA_map.items():
        if tname not in T_map:
            print(f"[WARN] {tname} exists in reset file but not in T file.")
            continue
        n = len(T_map[tname])
        bad = idx[(idx < 0) | (idx >= n)]
        if len(bad) > 0:
            print(f"[WARN] {tname} has out-of-range indices (0..{n-1}), e.g. {bad[:10]}")
    for tname in T_map.keys():
        if tname not in VITA_map:
            print(f"[WARN] {tname} exists in T file but has no reset indices in reset file.")