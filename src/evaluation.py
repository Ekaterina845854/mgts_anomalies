"""Оценка и сравнение детекторов.

Без разметки — диагностика на test; P/R/F1 — только по data/labels/labels.csv.
Синтетический тест проверяет работоспособность, а не качество на реальных аномалиях.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import epoch_s

LABEL_COLUMNS = ["label_id", "source_id", "start_time", "end_time", "x_start", "x_end",
                 "label", "annotator", "comment", "event_id"]
LABEL_VALUES = ("аномалия", "норма", "неопределённо")


def _overlap(a0, a1, b0, b1, tol):
    return (a0 <= b1 + tol) & (b0 <= a1 + tol)


def match_events(a: pd.DataFrame, b: pd.DataFrame, time_tol_s: float, x_tol: float
                 ) -> np.ndarray:
    """Матрица совпадений (len(a), len(b)) по времени и x в одном источнике."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=bool)
    ta0 = epoch_s(a["start_time"])[:, None]
    ta1 = epoch_s(a["end_time"])[:, None]
    tb0 = epoch_s(b["start_time"])[None, :]
    tb1 = epoch_s(b["end_time"])[None, :]
    same = a["source_id"].to_numpy()[:, None] == b["source_id"].to_numpy()[None, :]
    t_ok = _overlap(ta0, ta1, tb0, tb1, time_tol_s)
    x_ok = _overlap(a["x_start"].to_numpy()[:, None], a["x_end"].to_numpy()[:, None],
                    b["x_start"].to_numpy()[None, :], b["x_end"].to_numpy()[None, :], x_tol)
    return same & t_ok & x_ok


def _match_by_source(a: pd.DataFrame, b: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Есть ли у каждого события a/b совпадение в другой таблице."""
    ec = cfg["evaluation"]
    ma = np.zeros(len(a), dtype=bool)
    mb = np.zeros(len(b), dtype=bool)
    ia = {s: np.flatnonzero(a["source_id"].to_numpy() == s) for s in a["source_id"].unique()}
    ib = {s: np.flatnonzero(b["source_id"].to_numpy() == s) for s in b["source_id"].unique()}
    for s, idx_a in ia.items():
        idx_b = ib.get(s)
        if idx_b is None:
            continue
        m = match_events(a.iloc[idx_a], b.iloc[idx_b], ec["match_time_tol_s"], ec["match_x_tol"])
        ma[idx_a] |= m.any(1)
        mb[idx_b] |= m.any(0)
    return ma, mb


def compare_detectors(events: pd.DataFrame, results_meta: pd.DataFrame, test_hours: dict,
                      cfg: dict, split: str = "test", sources: list | None = None) -> dict:
    """Сравнение детекторов на test-части всех источников."""
    ev = events if split is None else events[events["split"] == split]
    rm = results_meta
    if sources is not None:
        ev = ev[ev["source_id"].isin(sources)]
        rm = rm[rm["source_id"].isin(sources)]
    hours = sum(test_hours.get(s, 0.0) for s in (sources or test_hours.keys()))
    dets = list(rm["detector"].unique())
    by = {d: ev[ev["detector"] == d].reset_index(drop=True) for d in dets}
    table, overlap = [], []
    matched_any = {d: np.zeros(len(by[d]), dtype=bool) for d in dets}
    matched_all = {d: np.ones(len(by[d]), dtype=bool) for d in dets}
    for d1 in dets:
        for d2 in dets:
            if d1 >= d2:
                continue
            m1, m2 = _match_by_source(by[d1], by[d2], cfg)
            matched_any[d1] |= m1
            matched_any[d2] |= m2
            matched_all[d1] &= m1
            matched_all[d2] &= m2
            overlap.append(dict(detector_a=d1, detector_b=d2, a_matched=int(m1.sum()),
                                a_total=len(m1), b_matched=int(m2.sum()), b_total=len(m2)))
    for d in dets:
        r = rm[rm["detector"] == d]
        n = len(by[d])
        table.append(dict(
            detector=d, events=n, flagged_windows=int(r["flagged_windows_eval"].sum()),
            windows=int(r["windows_eval"].sum()),
            events_per_hour=n / hours if hours > 0 else None,
            matched_with_other=int(matched_any[d].sum()),
            unique=int((~matched_any[d]).sum()),
            consensus_all=int(matched_all[d].sum()) if len(dets) > 1 else None,
            fit_time_s=float(r["fit_time_s"].sum()), score_time_s=float(r["score_time_s"].sum()),
            peak_mem_mb=float(r["peak_mem_mb"].max()),
            validation_alarm_rate=float(r["validation_alarm_rate"].mean()),
        ))
    unique_ids = {d: by[d].loc[~matched_any[d], "event_id"].tolist() for d in dets}
    return dict(table=table, overlap=overlap, hours=hours, unique_ids=unique_ids, split=split)


def window_agreement(flags_by_det: dict[str, np.ndarray]) -> list[dict]:
    """Коэффициент Жаккара флагов окон между парами детекторов."""
    out = []
    ks = sorted(flags_by_det)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            fa, fb = flags_by_det[a], flags_by_det[b]
            inter = int((fa & fb).sum())
            union = int((fa | fb).sum())
            out.append(dict(detector_a=a, detector_b=b, intersection=inter, union=union,
                            jaccard=inter / union if union else None))
    return out


def label_template_path(path: Path) -> Path:
    """Пустой шаблон разметки, если файла нет."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=LABEL_COLUMNS).to_csv(path, index=False, encoding="utf-8")
    return path


def load_labels(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(path, dtype={"source_id": str, "label_id": str, "event_id": str})
    if df.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    bad = ~df["label"].isin(LABEL_VALUES)
    if bad.any():
        raise ValueError(f"Недопустимые значения label: {sorted(df.loc[bad, 'label'].unique())}; "
                         f"допустимы {LABEL_VALUES}")
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    return df


def append_label(path: Path, row: dict) -> None:
    path = label_template_path(path)
    df = pd.read_csv(path, dtype=str)
    row = {c: row.get(c, "") for c in LABEL_COLUMNS}
    if not row["label_id"]:
        row["label_id"] = f"L{len(df) + 1:05d}"
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")


def labeled_metrics(events: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> dict:
    """Event-level P/R/F1: TP — пересечение с «аномалия», FP — только с «норма».

    Без участков «аномалия» возвращает None.
    """
    ec = cfg["evaluation"]
    lab = labels[labels["label"].isin(["аномалия", "норма"])].copy()
    n_anom = int((lab["label"] == "аномалия").sum())
    status = dict(n_labels=int(len(labels)), n_anomaly=n_anom,
                  n_normal=int((lab["label"] == "норма").sum()),
                  n_uncertain=int((labels["label"] == "неопределённо").sum()))
    if n_anom == 0:
        return dict(status, available=False, per_detector=None,
                    reason="нет участков с меткой «аномалия» - P/R/F1 не вычисляются")
    lab["x_start"] = lab["x_start"].astype(float)
    lab["x_end"] = lab["x_end"].astype(float)
    per = []
    for det, ev in events.groupby("detector"):
        ev = ev[ev["source_id"].isin(lab["source_id"].unique())]
        m = match_events(ev, lab, ec["match_time_tol_s"], ec["match_x_tol"])
        is_anom = (lab["label"] == "аномалия").to_numpy()
        hit_anom = m[:, is_anom].any(1)
        hit_norm = m[:, ~is_anom].any(1)
        tp = int(hit_anom.sum())
        fp = int((hit_norm & ~hit_anom).sum())
        found = int(m[:, is_anom].any(0).sum())
        p = tp / (tp + fp) if tp + fp else None
        r = found / n_anom
        f1 = 2 * p * r / (p + r) if p is not None and (p + r) > 0 else None
        per.append(dict(detector=det, tp=tp, fp=fp, anomalies_found=found, anomalies_total=n_anom,
                        precision=p, recall=r, f1=f1, events_in_labeled_zone=int((hit_anom | hit_norm).sum())))
    return dict(status, available=True, per_detector=per, reason=None)


def inject_synthetic(values: np.ndarray, x: np.ndarray, t_sec: np.ndarray, split: np.ndarray,
                     noise: float, cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Гауссовы по x и прямоугольные по времени отклонения в test-части; возвращает (значения, истина)."""
    sc = cfg["evaluation"]["synthetic"]
    rng = np.random.default_rng(sc["random_state"])
    out = values.copy()
    test_idx = np.flatnonzero(split == "test")
    rows = []
    T, X = values.shape
    lo, hi = sc["amplitude_noise_mult"]
    used = np.zeros(T, dtype=bool)
    for k in range(sc["n_events"]):
        for _ in range(50):
            dur = int(rng.integers(1, 8))
            t0 = int(rng.choice(test_idx[:-dur])) if len(test_idx) > dur else int(test_idx[0])
            if not used[max(0, t0 - 10):t0 + dur + 10].any():
                break
        used[t0:t0 + dur] = True
        width = float(rng.uniform(1.0, 6.0))  # в единицах x
        xc = float(rng.uniform(x[0] + 10 * width, x[-1] - 10 * width))
        amp = float(rng.uniform(lo, hi) * noise * rng.choice([-1, 1]))
        shape = np.exp(-0.5 * ((x - xc) / (width / 2.355)) ** 2)
        out[t0:t0 + dur] += (amp * shape)[None, :].astype(out.dtype)
        rows.append(dict(label_id=f"S{k + 1:03d}", t0=t0, t1=t0 + dur - 1,
                         x_start=xc - width, x_end=xc + width, amplitude=amp,
                         amp_noise_mult=abs(amp) / noise, label="аномалия"))
    return out, pd.DataFrame(rows)


def synthetic_benchmark(src, cfg: dict) -> dict:
    """Полный путь детекции на источнике с инъекциями."""
    from .data_loader import SourceData
    from .pipeline import run_source  # избегаем циклического импорта

    base = run_source(src, cfg)
    prep0 = base["prep"]
    noise = float(np.median(prep0.sigma))
    vals, truth = inject_synthetic(src.values, src.x, prep0.t_sec, prep0.split, noise, cfg)
    syn = SourceData(src.source_id, src.file_name, src.dts_id, src.channel_id, src.times, src.x,
                     vals, src.tz_known, src.tz_offset_h, src.x_unit, src.report, src.issues)
    res = run_source(syn, cfg)
    truth["source_id"] = src.source_id
    truth["start_time"] = prep0.times[truth["t0"].to_numpy()]
    truth["end_time"] = prep0.times[truth["t1"].to_numpy()]
    rows = []
    ev_all = res["events"]
    for det in res["results"]:
        ev = ev_all[ev_all["detector"] == det] if len(ev_all) else ev_all
        ev = ev[ev["split"] == "test"] if len(ev) else ev
        m = match_events(ev, truth, cfg["evaluation"]["match_time_tol_s"], cfg["evaluation"]["match_x_tol"])
        found = m.any(0)
        am = truth["amp_noise_mult"].to_numpy()
        buckets = {}
        for name, lo_, hi_ in (("<6σ", 0, 6), ("6–10σ", 6, 10), ("≥10σ", 10, np.inf)):
            sel = (am >= lo_) & (am < hi_)
            buckets[name] = dict(n=int(sel.sum()), recall=float(found[sel].mean()) if sel.any() else None)
        rows.append(dict(detector=det, injected=len(truth), found=int(found.sum()),
                         recall=float(found.mean()), recall_by_amplitude=buckets,
                         test_events=len(ev), events_matching_injection=int(m.any(1).sum())))
    return dict(source_id=src.source_id, noise=noise, n_injected=len(truth), per_detector=rows,
                truth=truth.assign(start_time=truth["start_time"].astype(str),
                                   end_time=truth["end_time"].astype(str)).to_dict("records"),
                note="Синтетический тест: искусственные отклонения в копии данных; "
                     "не является оценкой качества на реальных аномалиях")
