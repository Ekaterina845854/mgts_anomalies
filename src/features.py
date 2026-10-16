"""Признаки окон «кадр × бин» в единицах шума, только по текущему и прошлым кадрам.

z, abs_z — отклонение от нормы; rate_z — изменение за кадр без общего сдвига;
contrast_z — z минус медиана соседей ±3; intra_std_ratio — разброс внутри бина;
temporal_std_z, persistence — std z и число |z|>3 за 5 кадров; width_bins — ширина участка |z|>3.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .preprocess import Prepared, _quiet

FEATURE_NAMES = ["z", "abs_z", "rate_z", "contrast_z", "intra_std_ratio",
                 "temporal_std_z", "persistence", "width_bins"]
ACTIVE_Z = 3.0
HISTORY = 5


def _rolling_past(a: np.ndarray, segment: np.ndarray, k: int, fn) -> np.ndarray:
    """fn по последним k кадрам (включая текущий) внутри сегмента."""
    T = a.shape[0]
    out = np.full_like(a, np.nan, dtype=np.float32)
    seg_start = np.zeros(T, dtype=int)
    for t in range(1, T):
        seg_start[t] = seg_start[t - 1] if segment[t] == segment[t - 1] else t
    with _quiet():
        for t in range(T):
            a0 = max(seg_start[t], t - k + 1)
            if t - a0 + 1 >= 2:
                out[t] = fn(a[a0:t + 1], axis=0)
    return out


def run_width(active: np.ndarray) -> np.ndarray:
    """Длина непрерывного активного участка по оси бинов."""
    T, B = active.shape
    out = np.zeros((T, B), dtype=np.float32)
    for t in np.nonzero(active.any(axis=1))[0]:
        row = active[t]
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        for s, e in zip(starts, ends):
            out[t, s:e] = e - s
    return out


def compute_features(prep: Prepared) -> dict[str, np.ndarray]:
    z = prep.z
    valid = np.isfinite(z)
    # изменение за кадр минус медианное изменение кадра
    dv = np.full_like(prep.vb, np.nan)
    dv[1:] = prep.vb[1:] - prep.vb[:-1]
    dv[np.isnan(prep.dt_s)] = np.nan  # не через разрыв
    with _quiet():
        dv = dv - np.nanmedian(dv, axis=1, keepdims=True)
    rate_z = dv / prep.sigma[None, :]
    z0 = np.where(valid, z, 0.0)
    neigh = median_filter(z0, size=(1, 7), mode="nearest")
    contrast = z0 - neigh
    active = valid & (np.abs(z0) > ACTIVE_Z)
    persistence = _rolling_past(active.astype(np.float32), prep.segment, HISTORY, np.sum)
    temporal_std = _rolling_past(z, prep.segment, HISTORY, np.nanstd)
    feats = {
        "z": z,
        "abs_z": np.abs(z),
        "rate_z": rate_z.astype(np.float32),
        "contrast_z": contrast.astype(np.float32),
        "intra_std_ratio": prep.intra_std,
        "temporal_std_z": temporal_std,
        "persistence": persistence,
        "width_bins": run_width(active),
    }
    # окно вычислимо только при наличии нормы
    for k in feats:
        feats[k] = np.where(valid, feats[k], np.nan).astype(np.float32)
    return feats


def feature_matrix(feats: dict[str, np.ndarray], mask: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Матрица (N, F) по окнам с конечными признаками и их flat_index."""
    stack = np.stack([feats[n] for n in FEATURE_NAMES], axis=-1)  # (T, B, F)
    T, B, F = stack.shape
    flat = stack.reshape(-1, F)
    ok = np.isfinite(flat).all(axis=1)
    if mask is not None:
        ok &= np.broadcast_to(mask, (T, B)).reshape(-1)
    idx = np.nonzero(ok)[0]
    return flat[idx], idx
