"""D. Автоэнкодер (P2): ошибка восстановления патча z.

Патч — текущий и PATCH_T-1 прошлых кадров × бины b±PATCH_HALF_B; через разрыв сегмента — нули.
Сеть 35→24→6→24→35 (MLPRegressor) обучается на train без 1 % патчей с наибольшим |z|.
score = |ошибка центра| / (1.4826·MAD ошибки на train), порог фиксированный в σ.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from .detectors import DetectorResult, _Measure, calibrate_threshold
from .preprocess import Prepared

PATCH_T = 5
PATCH_HALF_B = 3
CLIP_Z = 20.0
SCALE = 5.0  # вход сети ~[-1, 1]


def _lagged(prep: Prepared) -> list[np.ndarray]:
    """z со сдвигом 0..PATCH_T-1 кадров назад, нули через разрыв сегмента, паддинг по x."""
    z = np.nan_to_num(np.clip(prep.z, -CLIP_Z, CLIP_Z), nan=0.0) / SCALE
    T, B = z.shape
    out = []
    for k in range(PATCH_T):
        lag = np.zeros_like(z)
        if k == 0:
            lag[:] = z
        else:
            same = np.zeros(T, dtype=bool)
            same[k:] = prep.segment[k:] == prep.segment[:-k]
            lag[k:] = np.where(same[k:, None], z[:-k], 0.0)
        out.append(np.pad(lag, ((0, 0), (PATCH_HALF_B, PATCH_HALF_B)), mode="edge").astype(np.float32))
    return out


def _patches(lagged: list[np.ndarray], t0: int, t1: int, B: int) -> np.ndarray:
    """Матрица патчей для кадров [t0, t1): ((t1-t0)·B, PATCH_T·(2·PATCH_HALF_B+1))."""
    cols = [lagged[k][t0:t1, d:d + B] for k in range(PATCH_T) for d in range(2 * PATCH_HALF_B + 1)]
    return np.stack(cols, axis=-1).reshape(-1, len(cols))


CENTER = PATCH_HALF_B  # центр текущего кадра


def detect_autoencoder(prep: Prepared, feats: dict, cfg: dict) -> DetectorResult:
    dcfg = cfg["detectors"]
    p = dcfg["autoencoder"]
    rng = np.random.default_rng(p["random_state"])
    T, B = prep.shape
    valid = np.isfinite(prep.z)
    with _Measure() as mf:
        lagged = _lagged(prep)
        tr_frames = np.flatnonzero(prep.split_mask("train"))
        cand = np.flatnonzero(valid[tr_frames].ravel())
        if cand.size < 500:
            raise ValueError(f"{prep.source_id}: недостаточно train-окон для автоэнкодера ({cand.size})")
        pick = rng.choice(cand, min(p["train_sample"], cand.size), replace=False)
        rows = tr_frames[pick // B]
        cols = pick % B
        X = np.stack([lagged[k][rows, cols + d] for k in range(PATCH_T)
                      for d in range(2 * PATCH_HALF_B + 1)], axis=-1)
        keep = np.abs(X).max(1) <= np.quantile(np.abs(X).max(1), 1 - p["trim_quantile"])
        X = X[keep]
        model = MLPRegressor(hidden_layer_sizes=tuple(p["hidden"]), activation="tanh",
                             alpha=p["alpha"], max_iter=p["max_iter"], early_stopping=True,
                             validation_fraction=0.1, random_state=p["random_state"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, X)
        err_tr = np.abs(X[:, CENTER] - model.predict(X)[:, CENTER])
        med = np.median(err_tr)
        mad = 1.4826 * np.median(np.abs(err_tr - med))
        scale = max(mad, 0.05 * med, 1e-6)
    with _Measure() as ms:
        err = np.empty(T * B, dtype=np.float32)
        step = max(1, 150_000 // B)
        for t0 in range(0, T, step):
            t1 = min(T, t0 + step)
            Xc = _patches(lagged, t0, t1, B)
            err[t0 * B:t1 * B] = np.abs(Xc[:, CENTER] - model.predict(Xc)[:, CENTER])
        scores = ((err.reshape(T, B) - med) / scale).clip(min=0).astype(np.float32)
        scores[~valid] = np.nan
    thr, info = calibrate_threshold(scores, prep.split_mask("validation"), dcfg["target_rate"],
                                    p["min_threshold"], p.get("threshold_mode", "fixed"))
    return DetectorResult(
        "autoencoder", prep.source_id, scores, thr,
        config=dict(p, architecture=f"{PATCH_T * (2 * PATCH_HALF_B + 1)}-"
                    + "-".join(map(str, p["hidden"])) + f"-{PATCH_T * (2 * PATCH_HALF_B + 1)}",
                    patch=f"{PATCH_T} кадров × {2 * PATCH_HALF_B + 1} бинов", n_train_windows=int(len(X)),
                    n_iter=int(model.n_iter_), train_loss=float(model.loss_),
                    target_rate=dcfg["target_rate"], **info),
        score_transform="(|ошибка восстановления центра| − медиана) / (1.4826·MAD) на train, отрицательные → 0",
        fit_time_s=mf.elapsed, score_time_s=ms.elapsed, peak_mem_mb=max(mf.peak_mb, ms.peak_mb),
        extra=dict(train_error_median=float(med), train_error_scale=float(scale)))
