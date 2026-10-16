"""Детекторы аномалий на сетке «кадр × бин».

scores (T, B): больше — необычнее, NaN — не оценивалось; threshold в шкале метода.
threshold_mode: fixed — порог в σ (A, C, D); quantile — квантиль validation (IF без шкалы в σ).
score_norm = score / threshold, >= 1 — срабатывание.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .features import FEATURE_NAMES, feature_matrix
from .preprocess import Prepared

DETECTOR_LABELS = {
    "statistical": "Статистический (median/MAD)",
    "isolation_forest": "Isolation Forest",
    "regression": "Регрессия + остатки",
    "autoencoder": "Автоэнкодер (P2)",
}
DETECTOR_CODES = {"statistical": "A", "isolation_forest": "B", "regression": "C", "autoencoder": "D"}


@dataclass
class DetectorResult:
    detector: str
    source_id: str
    scores: np.ndarray
    threshold: float
    config: dict
    score_transform: str
    fit_time_s: float = 0.0
    score_time_s: float = 0.0
    peak_mem_mb: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def flags(self) -> np.ndarray:
        return np.isfinite(self.scores) & (self.scores >= self.threshold)


def calibrate_threshold(scores: np.ndarray, val_mask: np.ndarray, target_rate: float,
                        min_threshold: float, mode: str = "quantile") -> tuple[float, dict]:
    """Порог (quantile или fixed) и доля срабатываний на validation."""
    v = scores[val_mask].ravel()  # val_mask: (T,)
    v = v[np.isfinite(v)]
    source = "validation"
    if v.size < 100:
        v = scores[np.isfinite(scores)]
        source = "all (validation слишком мала)"
    if v.size == 0:
        return float("inf"), dict(threshold_source="нет оценок", quantile_value=None)
    q = float(np.quantile(v, 1 - target_rate))
    thr = float(min_threshold) if mode == "fixed" else max(q, min_threshold)
    return thr, dict(threshold_mode=mode, threshold_source=source, quantile_value=q,
                     validation_alarm_rate=float(np.mean(v >= thr)),
                     n_calibration_windows=int(v.size))


class _Measure:
    """Время и пик памяти."""

    def __enter__(self):
        tracemalloc.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / 2**20


def detect_statistical(prep: Prepared, feats: dict, cfg: dict) -> DetectorResult:
    """A. Робастное отклонение от локальной нормы: |dev| / (1.4826·MAD на train)."""
    dcfg = cfg["detectors"]
    with _Measure() as m:
        scores = np.abs(prep.z).astype(np.float32)
    thr, info = calibrate_threshold(scores, prep.split_mask("validation"), dcfg["target_rate"],
                                    dcfg["statistical"]["min_threshold"],
                                    dcfg["statistical"].get("threshold_mode", "fixed"))
    return DetectorResult(
        "statistical", prep.source_id, scores, thr,
        config=dict(dcfg["statistical"], target_rate=dcfg["target_rate"],
                    baseline_window_min=cfg["preprocess"]["baseline_window_min"], **info),
        score_transform="|robust z|, без преобразования",
        fit_time_s=0.0, score_time_s=m.elapsed, peak_mem_mb=m.peak_mb)


def detect_isolation_forest(prep: Prepared, feats: dict, cfg: dict) -> DetectorResult:
    """B. Isolation Forest на признаках окон; score = -score_samples ∈ [0, 1]."""
    dcfg = cfg["detectors"]
    p = dcfg["isolation_forest"]
    rng = np.random.default_rng(p["random_state"])
    T, B = prep.shape
    train_mask = np.broadcast_to(prep.split_mask("train")[:, None], (T, B))
    with _Measure() as mf:
        Xtr, _ = feature_matrix(feats, train_mask)
        if len(Xtr) > p["train_sample"]:
            Xtr = Xtr[rng.choice(len(Xtr), p["train_sample"], replace=False)]
        scaler = RobustScaler(quantile_range=(5, 95)).fit(Xtr)
        model = IsolationForest(n_estimators=p["n_estimators"], max_samples=p["max_samples"],
                                contamination="auto", random_state=p["random_state"], n_jobs=-1)
        model.fit(scaler.transform(Xtr))
    with _Measure() as ms:
        Xall, idx = feature_matrix(feats)
        s = np.empty(len(Xall), dtype=np.float32)
        step = 200_000
        for i in range(0, len(Xall), step):
            s[i:i + step] = -model.score_samples(scaler.transform(Xall[i:i + step]))
        scores = np.full(T * B, np.nan, dtype=np.float32)
        scores[idx] = s
        scores = scores.reshape(T, B)
    thr, info = calibrate_threshold(scores, prep.split_mask("validation"), dcfg["target_rate"],
                                    p["min_threshold"], p.get("threshold_mode", "quantile"))
    return DetectorResult(
        "isolation_forest", prep.source_id, scores, thr,
        config=dict(p, features=FEATURE_NAMES, n_train_windows=int(len(Xtr)),
                    target_rate=dcfg["target_rate"], **info),
        score_transform="-IsolationForest.score_samples (больше = аномальнее)",
        fit_time_s=mf.elapsed, score_time_s=ms.elapsed, peak_mem_mb=max(mf.peak_mb, ms.peak_mb))
