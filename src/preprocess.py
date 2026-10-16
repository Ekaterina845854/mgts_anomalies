"""SourceData (кадры × точки) -> Prepared (кадры × бины).

Бин ~1 единица x; разрыв по времени начинает сегмент; норма — медиана прошлых кадров;
общий сдвиг кадра вычитается; шум — 1.4826·MAD на train с нижней границей.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import SourceData, to_epoch_ns

SPLITS = ("train", "validation", "test")


@dataclass
class Prepared:
    source_id: str
    dts_id: str
    channel_id: int
    times: pd.DatetimeIndex
    t_sec: np.ndarray  # секунды от начала источника
    dt_s: np.ndarray  # шаг до предыдущего кадра (NaN в начале сегмента)
    segment: np.ndarray  # номер непрерывного сегмента
    split: np.ndarray  # 'train' | 'validation' | 'test'
    bin_x_start: np.ndarray
    bin_x_end: np.ndarray
    bin_points: int
    vb: np.ndarray  # (T, B) среднее значение бина
    base: np.ndarray  # (T, B) локальная норма (NaN, если истории мало)
    cm: np.ndarray  # (T,) общий сдвиг кадра
    dev: np.ndarray  # (T, B) отклонение от нормы без общего сдвига
    sigma: np.ndarray  # (B,) робастный шум бина
    z: np.ndarray  # (T, B) dev / sigma
    intra_std: np.ndarray  # (T, B) разброс точек внутри бина
    x_unit: str
    tz_known: bool
    tz_offset_h: float | None
    dt_median_s: float
    params: dict

    @property
    def bin_x_center(self) -> np.ndarray:
        return (self.bin_x_start + self.bin_x_end) / 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.vb.shape

    def split_mask(self, name: str) -> np.ndarray:
        return self.split == name


def choose_bin_points(src: SourceData, cfg: dict) -> int:
    p = cfg["preprocess"]
    if src.x_unit.startswith("м"):
        return max(1, int(round(p["bin_target_x"] / src.dx)))
    return int(p["bin_points_fallback"])


def make_segments(t_sec: np.ndarray, gap_factor: float, gap_min_s: float = 0.0
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Разрыв = шаг > max(gap_factor × медианный шаг, gap_min_s)."""
    dt = np.diff(t_sec)
    med = float(np.median(dt)) if len(dt) else float("nan")
    new_seg = np.r_[True, dt > max(gap_factor * med, gap_min_s)]
    seg = np.cumsum(new_seg) - 1
    dt_s = np.r_[np.nan, dt].astype(np.float64)
    dt_s[new_seg] = np.nan
    return seg, dt_s, med


def make_split(n: int, fractions: dict) -> np.ndarray:
    """Хронологическое разбиение train/validation/test."""
    tr = fractions["train"]
    va = fractions["validation"]
    i1, i2 = int(round(n * tr)), int(round(n * (tr + va)))
    out = np.empty(n, dtype=object)
    out[:i1], out[i1:i2], out[i2:] = "train", "validation", "test"
    return out


def bin_values(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, X) -> среднее и std по бинам из k точек, индексы начала бинов."""
    T, X = values.shape
    B = int(np.ceil(X / k))
    pad = B * k - X
    v = values if pad == 0 else np.concatenate(
        [values, np.full((T, pad), np.nan, dtype=values.dtype)], axis=1)
    v = v.reshape(T, B, k)
    with np.errstate(all="ignore"), _quiet():
        mean = np.nanmean(v, axis=2)
        std = np.nanstd(v, axis=2) if k > 1 else np.zeros((T, B), dtype=np.float32)
    return mean.astype(np.float32), std.astype(np.float32), np.arange(B) * k


class _quiet:
    """Подавление 'Mean of empty slice' для пустых бинов."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def trailing_median(vb: np.ndarray, t_sec: np.ndarray, segment: np.ndarray,
                    window_s: float, min_frames: int, max_frames: int = 60) -> np.ndarray:
    """Медиана по прошлым кадрам сегмента в окне [t - window_s, t), не более max_frames кадров."""
    T, B = vb.shape
    out = np.full((T, B), np.nan, dtype=np.float32)
    seg_start = np.zeros(T, dtype=int)
    for t in range(1, T):
        seg_start[t] = seg_start[t - 1] if segment[t] == segment[t - 1] else t
    lo = np.searchsorted(t_sec, t_sec - window_s, side="left")
    has_nan = np.isnan(vb).any()
    with _quiet():
        for t in range(T):
            a = max(lo[t], seg_start[t])
            if t - a < min_frames:
                continue
            idx = np.arange(a, t)
            if len(idx) > max_frames:
                idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]
            w = vb[idx]
            out[t] = np.nanmedian(w, axis=0) if has_nan else np.median(w, axis=0)
    return out


def robust_sigma(a: np.ndarray, floor_frac: float, floor_abs: float) -> np.ndarray:
    """1.4826·MAD по оси 0 с нижней границей; результат всегда конечен и > 0."""
    with _quiet():
        med = np.nanmedian(a, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(a - med), axis=0)
    finite = mad[np.isfinite(mad) & (mad > 0)]
    typical = float(np.median(finite)) if finite.size else 1.0
    floor = max(floor_frac * typical, floor_abs)
    mad = np.where(np.isfinite(mad), mad, typical)
    return np.maximum(mad, floor).astype(np.float32)


def prepare(src: SourceData, cfg: dict) -> Prepared:
    p = cfg["preprocess"]
    ns = to_epoch_ns(src.times)
    t_sec = (ns - ns[0]) / 1e9
    seg, dt_s, dt_med = make_segments(t_sec, p["gap_factor"], p.get("gap_min_s", 0.0))
    split = make_split(src.n_frames, p["split"])
    k = choose_bin_points(src, cfg)
    vb, intra, starts = bin_values(src.values, k)
    ends = np.minimum(starts + k, src.n_points) - 1
    x = src.x
    bin_x_start = x[starts]
    bin_x_end = x[ends] + src.dx  # правая граница последней точки бина
    base = trailing_median(vb, t_sec, seg, p["baseline_window_min"] * 60, p["baseline_min_frames"])
    raw_dev = vb - base
    with _quiet():
        cm = np.nanmedian(raw_dev, axis=1)
    cm = np.where(np.isfinite(cm), cm, np.nan).astype(np.float32)
    dev = (raw_dev - cm[:, None]).astype(np.float32)
    train = split == "train"
    ref = dev[train] if np.isfinite(dev[train]).sum() > 0 else dev
    sigma = robust_sigma(ref, p["noise_floor_frac"], p["noise_floor_abs"])
    z = (dev / sigma[None, :]).astype(np.float32)
    # разброс внутри бина относительно типичного на train
    with _quiet():
        intra_ref = np.nanmedian(intra[train] if train.any() else intra, axis=0)
    intra_ref = np.where(np.isfinite(intra_ref) & (intra_ref > 0), intra_ref,
                         np.nanmedian(intra_ref[intra_ref > 0]) if (intra_ref > 0).any() else 1.0)
    intra_ratio = (intra / intra_ref[None, :]).astype(np.float32)
    return Prepared(
        source_id=src.source_id, dts_id=src.dts_id, channel_id=src.channel_id,
        times=src.times, t_sec=t_sec, dt_s=dt_s, segment=seg, split=split,
        bin_x_start=bin_x_start, bin_x_end=bin_x_end, bin_points=k,
        vb=vb, base=base, cm=cm, dev=dev, sigma=sigma, z=z, intra_std=intra_ratio,
        x_unit=src.x_unit, tz_known=src.tz_known, tz_offset_h=src.tz_offset_h,
        dt_median_s=dt_med,
        params=dict(bin_points=k, n_segments=int(seg.max() + 1), dt_median_s=dt_med,
                    baseline_window_min=p["baseline_window_min"],
                    split_counts={s: int((split == s).sum()) for s in SPLITS}),
    )
