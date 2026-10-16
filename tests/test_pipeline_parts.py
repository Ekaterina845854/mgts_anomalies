"""Части пайплайна на синтетических данных с известной аномалией."""
import json

import numpy as np
import pandas as pd
import pytest

from src.detectors import calibrate_threshold, detect_isolation_forest, detect_statistical
from src.evaluation import compare_detectors, labeled_metrics, load_labels, append_label, match_events
from src.events import EVENT_COLUMNS, aggregate_events, label_flags, validate_event_frame
from src.features import FEATURE_NAMES, compute_features, feature_matrix
from src.patterns import apply_patterns, find_sync_within_source
from src.preprocess import make_segments, prepare, robust_sigma
from src.regression import detect_regression
from src.autoencoder import detect_autoencoder, _lagged, _patches, CENTER


@pytest.fixture
def prepared(synthetic_source, cfg):
    src, truth = synthetic_source
    prep = prepare(src, cfg)
    return prep, compute_features(prep), truth


def test_segments_split_on_gaps():
    t = np.array([0, 60, 120, 3000, 3060, 3120], dtype=float)
    seg, dt_s, med = make_segments(t, gap_factor=5, gap_min_s=900)
    assert list(seg) == [0, 0, 0, 1, 1, 1] and np.isnan(dt_s[3]) and med == 60


def test_robust_sigma_handles_constant_and_nan():
    a = np.zeros((50, 3), dtype=np.float32)
    a[:, 1] = np.random.default_rng(0).normal(0, 1, 50)
    a[:, 2] = np.nan
    s = robust_sigma(a, 0.25, 1e-3)
    assert np.isfinite(s).all() and (s > 0).all()


def test_prepare_shapes_and_normalization(prepared):
    prep, feats, _ = prepared
    T, B = prep.shape
    assert prep.bin_points == 4 and B == 100
    assert set(prep.split) == {"train", "validation", "test"}
    assert (prep.split[: int(0.6 * T)] == "train").all()  # хронологическое разбиение
    z = prep.z[np.isfinite(prep.z) & (prep.split == "train")[:, None]]
    assert 0.6 < np.median(np.abs(z)) / 0.6745 < 1.6  # робастный шум ~1σ на нормальных данных
    assert np.isnan(prep.base[:5]).all()  # нет истории - нет нормы


def test_features_finite_where_norm_exists(prepared):
    prep, feats, _ = prepared
    assert set(FEATURE_NAMES) <= set(feats)
    X, idx = feature_matrix(feats)
    assert np.isfinite(X).all() and len(X) > 0.9 * prep.vb.size


def _hit(res, prep, truth):
    fl = res.flags
    b = (prep.bin_x_start >= truth["x0"] - 1) & (prep.bin_x_end <= truth["x1"] + 1)
    return fl[truth["t0"]:truth["t1"] + 1][:, b].any()


@pytest.mark.parametrize("fn", [detect_statistical, detect_isolation_forest, detect_regression,
                                detect_autoencoder])
def test_detectors_find_injected_anomaly(prepared, cfg, fn):
    prep, feats, truth = prepared
    res = fn(prep, feats, cfg)
    assert res.scores.shape == prep.shape
    s = res.scores[np.isfinite(res.scores)]
    assert np.isfinite(res.threshold) and (s >= 0).all()
    assert _hit(res, prep, truth), f"{res.detector} не нашёл синтетическую аномалию"
    quiet = res.flags[: truth["t0"] - 5]
    assert quiet.mean() < 0.01  # мало срабатываний на нормальном участке


def test_threshold_modes():
    rng = np.random.default_rng(1)
    s = np.abs(rng.normal(size=(1000, 10))).astype(np.float32)
    val = np.zeros(1000, dtype=bool)
    val[600:800] = True
    thr_q, info_q = calibrate_threshold(s, val, 0.01, 0.0, "quantile")
    thr_f, info_f = calibrate_threshold(s, val, 0.01, 6.0, "fixed")
    assert abs(info_q["validation_alarm_rate"] - 0.01) < 0.005 and thr_f == 6.0
    assert info_f["validation_alarm_rate"] == 0.0


def test_label_flags_merges_within_gaps():
    f = np.zeros((20, 20), dtype=bool)
    f[5, 5] = f[7, 5] = True     # разрыв 1 кадр -> одно событие при gap_frames=1
    f[5, 15] = True              # далеко по x -> отдельное событие
    seg = np.zeros(20, dtype=int)
    lab = label_flags(f, seg, gap_frames=1, gap_bins=2)
    assert lab[5, 5] == lab[7, 5] != lab[5, 15]
    seg[6:] = 1                  # разрыв сегмента разделяет
    lab = label_flags(f, seg, gap_frames=1, gap_bins=2)
    assert lab[5, 5] != lab[7, 5]


def test_events_contract_and_location(prepared, cfg):
    prep, feats, truth = prepared
    res = detect_statistical(prep, feats, cfg)
    ev = aggregate_events(prep, res, feats, cfg)
    assert list(ev.columns) == EVENT_COLUMNS
    assert validate_event_frame(ev) == []
    assert ev["cable_id"].isna().all() and (ev["cable_note"] == "кабель не сопоставлен").all()
    hit = ev[(ev["t0"] <= truth["t1"]) & (ev["t1"] >= truth["t0"])
             & (ev["x_start"] <= truth["x1"]) & (ev["x_end"] >= truth["x0"])]
    assert len(hit) == 1
    e = hit.iloc[0]
    assert e["amplitude"] > 3 and e["direction"] == "рост значения"
    assert e["event_type"] in ("Резкий импульс", "Кратковременное плавное отклонение")
    assert e["extent_class"] == "локальное" and e["duration_class"] == "кратковременное"


def test_single_points_are_not_events(prepared, cfg):
    prep, feats, _ = prepared
    res = detect_statistical(prep, feats, cfg)
    res.scores = np.zeros_like(res.scores)
    res.scores[50, 50] = res.threshold * 1.1  # одиночное слабое окно
    ev = aggregate_events(prep, res, feats, cfg)
    assert len(ev) == 0


def test_contract_validator_catches_violations():
    df = pd.DataFrame([{c: None for c in EVENT_COLUMNS}])
    assert validate_event_frame(df)
    assert validate_event_frame(pd.DataFrame({"event_id": ["x"]}))


def _ev(rows):
    base = dict(source_id="s", dts_id="d", channel_id=1, detector="statistical", direction="рост значения",
                tz_known=True, duration_s=60.0, width_x=2.0, amplitude=1.0, anomaly_score_norm=2.0,
                abrupt=True, persistent=False)
    out = []
    for i, r in enumerate(rows):
        d = dict(base, event_id=f"E{i}", **r)
        d["start_time"] = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(seconds=d.pop("t"))
        d["end_time"] = d["start_time"] + pd.Timedelta(seconds=60)
        d["x_peak"] = (d["x_start"] + d["x_end"]) / 2
        out.append(d)
    return pd.DataFrame(out)


def test_repeats_periodicity_and_sync(cfg):
    rows = [dict(t=3600 * k, x_start=10.0, x_end=12.0) for k in range(5)]          # повтор раз в час
    rows += [dict(t=30, x_start=100.0, x_end=101.0)]                                 # синхронно с первым
    rows += [dict(t=99999, x_start=500.0, x_end=501.0)]                              # единичное
    ev = _ev(rows)
    out, pat = apply_patterns(ev, cfg, {"s": 60.0}, {"s": (ev["start_time"].min().timestamp(),
                                                            ev["start_time"].max().timestamp())})
    rep = out[out["x_start"] == 10.0]
    assert rep["repeat_group"].nunique() == 1 and (rep["repeat_count"] == 5).all()
    g = pat["repeat_groups"]
    assert bool(g.loc[g["n_events"] == 5, "periodic"].iloc[0])
    assert abs(g.loc[g["n_events"] == 5, "interval_mean_s"].iloc[0] - 3600) < 1
    assert out.loc[out["event_id"] == "E6", "repeat_count"].iloc[0] == 1
    assert "единичное" in json.loads(out.loc[out["event_id"] == "E6", "pattern_tags"].iloc[0])[0]
    assert out.loc[out["event_id"] == "E0", "sync_size"].iloc[0] == 2
    assert out.loc[out["event_id"] == "E5", "sync_group"].iloc[0] == out.loc[out["event_id"] == "E0", "sync_group"].iloc[0]


def test_sync_ignores_same_place():
    ev = _ev([dict(t=0, x_start=10.0, x_end=12.0), dict(t=10, x_start=11.0, x_end=13.0)])
    assert (find_sync_within_source(ev, 120, {}) == -1).all()


def test_metrics_absent_without_labels(cfg):
    ev = _ev([dict(t=0, x_start=10.0, x_end=12.0)])
    res = labeled_metrics(ev, pd.DataFrame(columns=["label"]), cfg)
    assert res["available"] is False and res["per_detector"] is None


def test_metrics_with_labels(tmp_path, cfg):
    ev = _ev([dict(t=0, x_start=10.0, x_end=12.0), dict(t=5000, x_start=50.0, x_end=52.0),
              dict(t=20000, x_start=80.0, x_end=81.0)])
    p = tmp_path / "labels.csv"
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    append_label(p, dict(source_id="s", start_time=t0.isoformat(), end_time=(t0 + pd.Timedelta(seconds=60)).isoformat(),
                         x_start=10, x_end=12, label="аномалия"))
    append_label(p, dict(source_id="s", start_time=(t0 + pd.Timedelta(seconds=5000)).isoformat(),
                         end_time=(t0 + pd.Timedelta(seconds=5060)).isoformat(), x_start=50, x_end=52, label="норма"))
    append_label(p, dict(source_id="s", start_time=(t0 + pd.Timedelta(seconds=40000)).isoformat(),
                         end_time=(t0 + pd.Timedelta(seconds=40060)).isoformat(), x_start=0, x_end=5, label="аномалия"))
    labels = load_labels(p)
    res = labeled_metrics(ev, labels, cfg)
    r = res["per_detector"][0]
    assert res["available"] and r["tp"] == 1 and r["fp"] == 1
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and abs(r["f1"] - 0.5) < 1e-9


def test_invalid_label_value_rejected(tmp_path):
    p = tmp_path / "l.csv"
    pd.DataFrame([dict(label_id="1", source_id="s", start_time="2026-01-01", end_time="2026-01-01",
                       x_start=0, x_end=1, label="broken")]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_labels(p)


def test_match_and_compare(cfg):
    a = _ev([dict(t=0, x_start=10.0, x_end=12.0), dict(t=9000, x_start=60.0, x_end=61.0)])
    b = _ev([dict(t=100, x_start=11.0, x_end=13.0)]).assign(detector="regression", event_id="R0")
    m = match_events(a, b, 300, 2)
    assert m.tolist() == [[True], [False]]
    ev = pd.concat([a, b]).assign(split="test")
    meta = pd.DataFrame([dict(source_id="s", detector=d, flagged_windows_eval=1, windows_eval=10, fit_time_s=0,
                              score_time_s=0, peak_mem_mb=1, validation_alarm_rate=0.0)
                         for d in ("statistical", "regression")])
    c = compare_detectors(ev, meta, {"s": 3.0}, cfg)
    t = {r["detector"]: r for r in c["table"]}
    assert t["statistical"]["events"] == 2 and t["statistical"]["unique"] == 1
    assert t["regression"]["matched_with_other"] == 1 and abs(t["statistical"]["events_per_hour"] - 2 / 3) < 1e-9


def test_export_roundtrip(tmp_path, prepared, cfg):
    prep, feats, _ = prepared
    ev = aggregate_events(prep, detect_statistical(prep, feats, cfg), feats, cfg)
    p = tmp_path / "ev.csv"
    ev.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert list(back.columns) == EVENT_COLUMNS and len(back) == len(ev)
    pq = tmp_path / "ev.parquet"
    ev.to_parquet(pq)
    assert pd.read_parquet(pq)["start_time"].dt.tz is not None


def test_autoencoder_patch_is_causal_and_centered(prepared):
    """Центр патча - текущее окно; в патч не попадают будущие кадры."""
    prep, _, _ = prepared
    lag = _lagged(prep)
    T, B = prep.shape
    X = _patches(lag, 0, T, B).reshape(T, B, -1)
    z = np.nan_to_num(np.clip(prep.z, -20, 20), nan=0.0) / 5
    assert np.allclose(X[..., CENTER], z)
    zf = prep.z.copy()
    zf[100:] = 99.0                       # меняем будущее
    prep2 = type(prep)(**{**prep.__dict__, "z": zf})
    X2 = _patches(_lagged(prep2), 0, T, B).reshape(T, B, -1)
    assert np.allclose(X2[:100], X[:100])


def test_autoencoder_can_be_disabled(synthetic_source, cfg):
    from src.pipeline import run_source
    src, _ = synthetic_source
    cfg["detectors"]["autoencoder"]["enabled"] = False
    out = run_source(src, cfg)
    assert "autoencoder" not in out["results"]
