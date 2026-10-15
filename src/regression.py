"""C. Регрессионный детектор: прогноз нормального значения бина и оценка по остатку.

Модель (Ridge, отдельно для каждого источника) прогнозирует изменение значения бина
за кадр y = v[t] - v[t-1] по признакам, известным до момента t или не содержащим
целевого значения:
  * d1 = v[t-1] - v[t-2], d2 = v[t-2] - v[t-3]      - инерция (прошлое);
  * pull = base[t] - v[t-1]                         - возврат к локальной норме (прошлое);
  * cm_other = медиана изменений бинов ПРОТИВОПОЛОЖНОЙ чётности в кадре t
               - общий сдвиг кадра без участия самого бина (нет утечки цели);
  * nb = среднее соседних бинов в t-1 минус v[t-1]   - пространственная диффузия;
  * dt_min = шаг по времени, мин.
Лаги не переходят через разрыв сегмента. Обучение - только на train-части.
score = |остаток| / (1.4826·MAD остатков бина на train).
Регрессия не определяет тип или причину события - только отклонение от ожидаемого.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .detectors import DetectorResult, _Measure, calibrate_threshold
from .preprocess import Prepared, _quiet, robust_sigma

REG_FEATURES = ["d1", "d2", "pull", "cm_other", "nb", "dt_min"]


def build_regression_design(prep: Prepared, n_lags: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (X (T,B,F), y (T,B), valid (T,B))."""
    vb, T = prep.vb, prep.shape[0]
    B = prep.shape[1]
    lag_ok = np.zeros(T, dtype=bool)
    for t in range(n_lags, T):
        lag_ok[t] = (prep.segment[t - n_lags] == prep.segment[t])
    nanT = np.full((1, B), np.nan, dtype=np.float32)

    def lag(a, k):
        return np.concatenate([np.repeat(nanT, k, 0), a[:-k]], 0) if k else a

    v1, v2, v3 = lag(vb, 1), lag(vb, 2), lag(vb, 3)
    y = vb - v1
    d1, d2 = v1 - v2, v2 - v3
    pull = prep.base - v1
    with _quiet():
        cm_even = np.nanmedian(y[:, 0::2], axis=1)
        cm_odd = np.nanmedian(y[:, 1::2], axis=1)
    cm_other = np.empty_like(y)
    cm_other[:, 0::2] = cm_odd[:, None]
    cm_other[:, 1::2] = cm_even[:, None]
    left = np.concatenate([v1[:, :1], v1[:, :-1]], 1)
    right = np.concatenate([v1[:, 1:], v1[:, -1:]], 1)
    nb = (left + right) / 2 - v1
    dt_min = np.repeat((prep.dt_s / 60.0)[:, None], B, 1).astype(np.float32)
    X = np.stack([d1, d2, pull, cm_other, nb, dt_min], axis=-1).astype(np.float32)
    valid = lag_ok[:, None] & np.isfinite(X).all(-1) & np.isfinite(y)
    return X, y.astype(np.float32), valid


def detect_regression(prep: Prepared, feats: dict, cfg: dict) -> DetectorResult:
    dcfg = cfg["detectors"]
    p = dcfg["regression"]
    rng = np.random.default_rng(p["random_state"])
    T, B = prep.shape
    with _Measure() as mf:
        X, y, valid = build_regression_design(prep, p["n_lags"])
        train = valid & prep.split_mask("train")[:, None]
        ti = np.flatnonzero(train)
        if ti.size < 50:
            raise ValueError(f"{prep.source_id}: недостаточно train-окон для регрессии ({ti.size})")
        if ti.size > p["train_sample"]:
            ti = rng.choice(ti, p["train_sample"], replace=False)
        Xf, yf = X.reshape(-1, X.shape[-1]), y.ravel()
        model = Ridge(alpha=p["alpha"]).fit(Xf[ti], yf[ti])
    with _Measure() as ms:
        vi = np.flatnonzero(valid)
        pred = np.full(T * B, np.nan, dtype=np.float32)
        pred[vi] = model.predict(Xf[vi])
        pred = pred.reshape(T, B)
        resid = y - pred
        sigma_r = robust_sigma(np.where(train, resid, np.nan), cfg["preprocess"]["noise_floor_frac"],
                               cfg["preprocess"]["noise_floor_abs"])
        scores = (np.abs(resid) / sigma_r[None, :]).astype(np.float32)
        scores[~valid] = np.nan
    thr, info = calibrate_threshold(scores, prep.split_mask("validation"), dcfg["target_rate"],
                                    p["min_threshold"], p.get("threshold_mode", "fixed"))
    metrics = regression_metrics(prep, y, pred, valid)
    return DetectorResult(
        "regression", prep.source_id, scores, thr,
        config=dict(p, model="Ridge", features=REG_FEATURES, target="v[t]-v[t-1]",
                    n_train_windows=int(ti.size), target_rate=dcfg["target_rate"], **info),
        score_transform="|остаток| / (1.4826·MAD остатков на train)",
        fit_time_s=mf.elapsed, score_time_s=ms.elapsed, peak_mem_mb=max(mf.peak_mb, ms.peak_mb),
        extra=dict(coef=dict(zip(REG_FEATURES, map(float, model.coef_))),
                   intercept=float(model.intercept_), metrics=metrics))


def regression_metrics(prep: Prepared, y: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict:
    """MAE/RMSE прогноза значения (в единицах сигнала) на test.

    Ошибка прогноза v[t] равна ошибке прогноза изменения y. Для контекста приводится
    наивный прогноз «как в прошлом кадре» (y_hat = 0). "Квази-нормальные" окна -
    test-окна с |z| < 3 (без явных отклонений по детектору A; это не разметка).
    """
    test = valid & prep.split_mask("test")[:, None]
    quiet = test & np.isfinite(prep.z) & (np.abs(prep.z) < 3)
    out = {}
    for name, m in (("test_all", test), ("test_quasi_normal", quiet)):
        if m.sum() == 0:
            out[name] = None
            continue
        e = (y - pred)[m]
        e0 = y[m]
        out[name] = dict(n=int(m.sum()), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e**2))),
                         mae_persistence=float(np.mean(np.abs(e0))),
                         rmse_persistence=float(np.sqrt(np.mean(e0**2))))
    return out
