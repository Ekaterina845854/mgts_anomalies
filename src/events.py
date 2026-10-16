"""Срабатывания окон -> события единого формата.

Объединение внутри источника и сегмента с допусками max_gap_s / max_gap_bins;
событие — от min_cells окон, одиночное окно — при score >= single_cell_score_mult × порог.
Категории (длительность, протяжённость, форма, последействие) описательные, не причины.
Уровень участка — контраст с соседними бинами, чтобы общий дрейф не влиял.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import ndimage

from .detectors import DETECTOR_CODES, DETECTOR_LABELS, DetectorResult
from .preprocess import Prepared, _quiet

CABLE_NOTE = "кабель не сопоставлен"

EVENT_COLUMNS = [
    "event_id", "source_id", "dts_id", "channel_id", "cable_id", "cable_note",
    "detector", "detector_label", "event_date", "start_time", "end_time", "peak_time",
    "duration_s", "x_start", "x_end", "x_peak", "width_x", "x_unit",
    "anomaly_score", "anomaly_score_mean", "anomaly_score_norm", "threshold",
    "n_cells", "n_frames", "n_bins", "amplitude", "amplitude_z", "direction",
    "pre_level", "post_level", "step_shift", "rise_time_s", "abrupt", "persistent",
    "shape", "duration_class", "extent_class", "event_type", "type_confidence",
    "split", "tz_known", "tz_offset_h", "t0", "t1", "b0", "b1",
    "features", "labels", "pattern_tags",
]

EVENT_TYPES = ["Резкий импульс", "Ступенчатое изменение", "Длительное отклонение",
               "Кратковременное плавное отклонение"]


def _seg_starts(segment: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(np.diff(segment)) + 1, len(segment)]


def label_flags(flags: np.ndarray, segment: np.ndarray, gap_frames: int, gap_bins: int
                ) -> np.ndarray:
    """Разметка связных областей срабатываний с допусками; 0 = не событие."""
    T, B = flags.shape
    labels = np.zeros((T, B), dtype=np.int32)
    rt, rb = math.ceil(gap_frames / 2), math.ceil(gap_bins / 2)
    struct = np.ones((2 * rt + 1, 2 * rb + 1), dtype=bool)
    offset = 0
    bounds = _seg_starts(segment)
    for a, b in zip(bounds[:-1], bounds[1:]):
        f = flags[a:b]
        if not f.any():
            continue
        grown = ndimage.binary_dilation(f, structure=struct) if (rt or rb) else f
        lab, n = ndimage.label(grown, structure=np.ones((3, 3), dtype=bool))
        lab = np.where(f, lab, 0)
        lab[lab > 0] += offset
        labels[a:b] = lab
        offset += n
    return labels


def _context_series(prep: Prepared, b0: int, b1: int, pad: int = 3, ring: int = 8) -> np.ndarray:
    B = prep.shape[1]
    with _quiet():
        inner = np.nanmean(prep.vb[:, b0:b1 + 1], axis=1)
        left = prep.vb[:, max(0, b0 - pad - ring):max(0, b0 - pad)]
        right = prep.vb[:, min(B, b1 + 1 + pad):min(B, b1 + 1 + pad + ring)]
        neigh = np.concatenate([left, right], axis=1)
        ref = np.nanmean(neigh, axis=1) if neigh.shape[1] else np.zeros(prep.shape[0])
    return (inner - ref).astype(np.float64)


def characterize(prep: Prepared, t0: int, t1: int, b0: int, b1: int, tpeak: int, amp: float,
                 cfg: dict) -> dict:
    ec = cfg["events"]
    s = _context_series(prep, b0, b1)
    seg = prep.segment
    W = ec["context_window_min"] * 60
    ts = prep.t_sec
    same = seg == seg[t0]
    pre_m = same & (ts >= ts[t0] - W) & (np.arange(len(ts)) < t0)
    post_m = same & (ts <= ts[t1] + W) & (np.arange(len(ts)) > t1)
    d = np.diff(s)
    d = d[np.isfinite(d)]
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if d.size > 5 else np.nan
    with _quiet():
        pre = float(np.nanmedian(s[pre_m])) if pre_m.sum() >= 3 else None
        post = float(np.nanmedian(s[post_m])) if post_m.sum() >= 3 else None
    abrupt = persistent = None
    if pre is not None and np.isfinite(s[tpeak]):
        path = abs(s[tpeak] - pre)
        lo = max(t0 - 1, 0)
        jumps = np.abs(np.diff(s[lo:tpeak + 1])) if tpeak > lo else np.array([path])
        jumps = jumps[np.isfinite(jumps)]
        # форма — только если путь до пика заметно больше шума
        if jumps.size and np.isfinite(noise) and path >= ec["step_persist_z"] * noise:
            abrupt = bool(jumps.max() >= ec["abrupt_rise_frac"] * path)
    if pre is not None and post is not None and np.isfinite(noise) and noise > 0:
        shift = post - pre
        persistent = bool(abs(shift) >= ec["step_persist_z"] * noise and np.sign(shift) == np.sign(amp))
    return dict(pre_level=pre, post_level=post,
                step_shift=(post - pre) if pre is not None and post is not None else None,
                abrupt=abrupt, persistent=persistent, context_noise=float(noise) if np.isfinite(noise) else None)


def classify(duration_s: float, width_x: float, abrupt, persistent, cfg: dict) -> dict:
    ec = cfg["events"]
    duration_class = "кратковременное" if duration_s <= ec["short_max_s"] else "длительное"
    extent_class = "локальное" if width_x <= ec["local_max_x"] else "распределённое"
    shape = {True: "резкое", False: "плавное", None: "не определено"}[abrupt]
    if abrupt and persistent:
        etype = "Ступенчатое изменение"
    elif abrupt and duration_class == "кратковременное":
        etype = "Резкий импульс"
    elif duration_class == "длительное":
        etype = "Длительное отклонение"
    elif abrupt:
        etype = "Резкий импульс"
    else:
        etype = "Кратковременное плавное отклонение"
    conf = "обычная" if abrupt is not None and persistent is not None else "низкая (мало контекста)"
    return dict(duration_class=duration_class, extent_class=extent_class, shape=shape,
                event_type=etype, type_confidence=conf)


def _fmt_date(ts: pd.Timestamp, offset_h) -> str:
    return (ts + pd.Timedelta(hours=offset_h or 0)).strftime("%Y-%m-%d")


def aggregate_events(prep: Prepared, res: DetectorResult, feats: dict, cfg: dict) -> pd.DataFrame:
    ec = cfg["events"]
    flags = res.flags
    dtm = prep.dt_median_s if np.isfinite(prep.dt_median_s) and prep.dt_median_s > 0 else 60.0
    gap_frames = max(0, int(round(ec["max_gap_s"] / dtm)) - 1)
    labels = label_flags(flags, prep.segment, gap_frames, int(ec["max_gap_bins"]))
    objs = ndimage.find_objects(labels)
    rows = []
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        cells = labels[sl] == k
        sc = res.scores[sl]
        n_cells = int(cells.sum())
        peak = float(np.nanmax(np.where(cells, sc, -np.inf)))
        if n_cells < ec["min_cells"] and peak < ec["single_cell_score_mult"] * res.threshold:
            continue
        ts_, bs_ = sl[0].start, sl[1].start
        tt, bb = np.nonzero(cells)
        tt, bb = tt + ts_, bb + bs_
        dev = prep.dev[tt, bb]
        j = int(np.nanargmax(np.abs(dev))) if np.isfinite(dev).any() else 0
        amp = float(dev[j]) if np.isfinite(dev[j]) else 0.0
        pi = int(np.nanargmax(res.scores[tt, bb]))
        tpeak, bpeak = int(tt[pi]), int(bb[pi])
        t0, t1, b0, b1 = int(tt.min()), int(tt.max()), int(bb.min()), int(bb.max())
        duration = float(prep.t_sec[t1] - prep.t_sec[t0])
        x0, x1 = float(prep.bin_x_start[b0]), float(prep.bin_x_end[b1])
        ctx = characterize(prep, t0, t1, b0, b1, tpeak, amp, cfg)
        cls = classify(duration, x1 - x0, ctx["abrupt"], ctx["persistent"], cfg)
        feat_summary = {n: float(np.nanmax(np.abs(feats[n][tt, bb]))) for n in feats
                        if np.isfinite(feats[n][tt, bb]).any()}
        feat_summary["context_noise"] = ctx["context_noise"]
        rows.append(dict(
            source_id=prep.source_id, dts_id=prep.dts_id, channel_id=int(prep.channel_id),
            cable_id=None, cable_note=CABLE_NOTE, detector=res.detector,
            detector_label=DETECTOR_LABELS[res.detector],
            event_date=_fmt_date(prep.times[t0], prep.tz_offset_h),
            start_time=prep.times[t0], end_time=prep.times[t1], peak_time=prep.times[tpeak],
            duration_s=duration, x_start=x0, x_end=x1,
            x_peak=float((prep.bin_x_start[bpeak] + prep.bin_x_end[bpeak]) / 2),
            width_x=x1 - x0, x_unit=prep.x_unit,
            anomaly_score=peak, anomaly_score_mean=float(np.nanmean(res.scores[tt, bb])),
            anomaly_score_norm=peak / res.threshold if res.threshold > 0 else None,
            threshold=float(res.threshold), n_cells=n_cells, n_frames=int(len(np.unique(tt))),
            n_bins=b1 - b0 + 1, amplitude=amp, amplitude_z=float(prep.z[tt[j], bb[j]]),
            direction="рост значения" if amp >= 0 else "снижение значения",
            pre_level=ctx["pre_level"], post_level=ctx["post_level"], step_shift=ctx["step_shift"],
            rise_time_s=float(prep.t_sec[tpeak] - prep.t_sec[t0]),
            abrupt=ctx["abrupt"], persistent=ctx["persistent"], **cls,
            split=str(prep.split[t0]), tz_known=prep.tz_known, tz_offset_h=prep.tz_offset_h,
            t0=t0, t1=t1, b0=b0, b1=b1,
            features=json.dumps(feat_summary, ensure_ascii=False), labels="[]", pattern_tags="[]",
        ))
    df = pd.DataFrame(rows, columns=[c for c in EVENT_COLUMNS if c != "event_id"])
    df = df.sort_values(["start_time", "x_start"], kind="stable").reset_index(drop=True)
    code = DETECTOR_CODES[res.detector]
    df.insert(0, "event_id", [f"{code}-{prep.source_id}-{i + 1:04d}" for i in range(len(df))])
    return df


def validate_event_frame(df: pd.DataFrame) -> list[str]:
    """Нарушения контракта события (пусто — всё верно)."""
    errs = []
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        errs.append(f"нет столбцов: {missing}")
        return errs
    for c in ["event_id", "start_time", "dts_id", "channel_id", "anomaly_score", "detector", "event_date"]:
        if df[c].isna().any():
            errs.append(f"пустые значения в обязательном поле {c}")
    if df["event_id"].duplicated().any():
        errs.append("повторяющиеся event_id")
    if (df["end_time"] < df["start_time"]).any():
        errs.append("end_time < start_time")
    if (df["x_end"] < df["x_start"]).any():
        errs.append("x_end < x_start")
    if df["cable_id"].notna().any():
        errs.append("cable_id заполнен без подтверждённого соответствия")
    if not np.isfinite(df["anomaly_score"].astype(float)).all():
        errs.append("anomaly_score не конечен")
    return errs
