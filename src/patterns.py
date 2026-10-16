"""Закономерности по событиям: повторы, периодичность, синхронность, совпадения, кластеры.

Совпадения между приборами — только кандидаты: синхронизация часов не подтверждена.
Кластерам присваиваются номера, без физических названий.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data_loader import epoch_s
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["log_duration", "log_width", "log_abs_amplitude", "signed_dir",
                    "log_score_norm", "abrupt_f", "persistent_f"]


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _groups(dsu: _DSU, n: int) -> np.ndarray:
    return np.array([dsu.find(i) for i in range(n)])


def find_repeats(ev: pd.DataFrame, x_tol: float) -> pd.Series:
    """Метка группы повторений для каждого события (индекс = индекс ev)."""
    out = pd.Series(-1, index=ev.index, dtype=int)
    gid = 0
    for _, g in ev.groupby(["source_id", "detector", "direction"], sort=False):
        # группа — непрерывный блок по x_peak с размахом <= 2·x_tol
        g = g.sort_values("x_peak")
        xp = g["x_peak"].to_numpy()
        block = np.empty(len(g), dtype=int)
        first = -np.inf
        cur = gid - 1
        for i in range(len(g)):
            if xp[i] - first > 2 * x_tol:
                cur += 1
                first = xp[i]
            block[i] = cur
        out.loc[g.index] = block
        gid = cur + 1
    return out


def repeat_stats(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Статистика групп повторений: размер, интервалы, CV, флаг периодичности."""
    pc = cfg["patterns"]
    rows = []
    for gid, g in ev.groupby("repeat_group"):
        g = g.sort_values("start_time")
        n = len(g)
        iv = np.diff(epoch_s(g["start_time"])) if n > 1 else np.array([])
        mean = float(iv.mean()) if iv.size else None
        std = float(iv.std(ddof=1)) if iv.size > 1 else None
        cv = std / mean if mean and std is not None and mean > 0 else None
        periodic = bool(n >= pc["min_repeats_periodic"] and cv is not None and cv <= pc["periodic_cv_max"])
        rows.append(dict(repeat_group=int(gid), source_id=g["source_id"].iloc[0],
                         detector=g["detector"].iloc[0], n_events=n,
                         x_min=float(g["x_start"].min()), x_max=float(g["x_end"].max()),
                         first=g["start_time"].iloc[0], last=g["start_time"].iloc[-1],
                         interval_mean_s=mean, interval_std_s=std, interval_cv=cv,
                         periodic=periodic, event_ids=",".join(g["event_id"])))
    return pd.DataFrame(rows)


def find_sync_within_source(ev: pd.DataFrame, window_s: float, dt_by_source: dict) -> pd.Series:
    """Блоки событий источника, начавшихся в пределах max(window_s, шаг); синхронны при непересекающихся участках."""
    out = pd.Series(-1, index=ev.index, dtype=int)
    gid = 0
    for (src, _det), g in ev.groupby(["source_id", "detector"], sort=False):
        g = g.sort_values("start_time")
        w = max(window_s, dt_by_source.get(src, 0.0))
        t = epoch_s(g["start_time"])
        xs, xe = g["x_start"].to_numpy(), g["x_end"].to_numpy()
        i = 0
        while i < len(g):
            j = i
            while j + 1 < len(g) and t[j + 1] - t[i] <= w:
                j += 1
            if j > i:
                a, b = xs[i:j + 1], xe[i:j + 1]
                disjoint = (b[:, None] < a[None, :]) | (b[None, :] < a[:, None])
                if disjoint.any():
                    out.loc[g.index[i:j + 1]] = gid
                    gid += 1
            i = j + 1
    return out


def cross_source_coincidences(ev: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Кандидаты совпадений между разными приборами (только с известным поясом)."""
    e = ev[ev["tz_known"].astype(bool)].sort_values("start_time")
    t = epoch_s(e["start_time"])
    src = e["source_id"].to_numpy()
    dev = (e["dts_id"].astype(str) + "/" + e["channel_id"].astype(str)).to_numpy()
    det = e["detector"].to_numpy()
    ids = e["event_id"].to_numpy()
    rows, j0 = [], 0
    for i in range(len(e)):
        while t[i] - t[j0] > window_s:
            j0 += 1
        for j in range(j0, i):
            if dev[i] != dev[j] and det[i] == det[j]:
                rows.append(dict(detector=det[i], event_a=ids[j], source_a=src[j], event_b=ids[i],
                                 source_b=src[i], dt_s=float(t[i] - t[j]),
                                 note="синхронизация часов приборов не подтверждена"))
    return pd.DataFrame(rows, columns=["detector", "event_a", "source_a", "event_b", "source_b",
                                       "dt_s", "note"])


def event_rate_autocorr(ev: pd.DataFrame, source_times: dict, min_events: int = 10,
                        bin_s: float = 600.0, max_lag_bins: int = 36, min_lag_bins: int = 3
                        ) -> pd.DataFrame:
    """Автокорреляция числа событий в 10-минутных интервалах.

    Пик ищется на лагах >= 30 мин и значим при r > 2/sqrt(N).
    """
    rows = []
    for (src, det), g in ev.groupby(["source_id", "detector"]):
        if src not in source_times:
            continue
        t0, t1 = source_times[src]
        nb = int((t1 - t0) // bin_s) + 1
        if len(g) < min_events or nb < 3 * 6:
            rows.append(dict(source_id=src, detector=det, n_events=len(g), n_bins=nb,
                             best_lag_s=None, acf=None, significant=False,
                             note="ряд слишком короткий для автокорреляции"))
            continue
        tt = (epoch_s(g["start_time"]) - t0) // bin_s
        counts = np.bincount(tt.astype(int).clip(0, nb - 1), minlength=nb).astype(float)
        c = counts - counts.mean()
        denom = (c ** 2).sum()
        L = min(max_lag_bins, nb // 3)
        acf = np.array([(c[:-k] * c[k:]).sum() / denom for k in range(1, L + 1)]) if denom > 0 else np.zeros(L)
        bound = 2 / np.sqrt(nb)
        lags = np.arange(1, len(acf) + 1)
        peak = np.zeros(len(acf), dtype=bool)
        if len(acf) >= 3:
            peak[1:-1] = (acf[1:-1] > acf[:-2]) & (acf[1:-1] >= acf[2:])
        cand = np.flatnonzero(peak & (lags >= min_lag_bins))
        if cand.size == 0:
            rows.append(dict(source_id=src, detector=det, n_events=len(g), n_bins=nb,
                             best_lag_s=None, acf=None, significant=False,
                             note="нет локального пика автокорреляции на лагах >= 30 мин"))
            continue
        k = int(cand[np.argmax(acf[cand])])
        rows.append(dict(source_id=src, detector=det, n_events=len(g), n_bins=nb,
                         best_lag_s=float(lags[k] * bin_s), acf=float(acf[k]),
                         significant=bool(acf[k] > bound),
                         note=f"граница значимости {bound:.2f}"))
    return pd.DataFrame(rows)


def cluster_events(ev: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    """DBSCAN на интерпретируемых признаках; -1 = шум (не отнесено ни к одной группе)."""
    pc = cfg["patterns"]
    if len(ev) < max(10, pc["dbscan_min_samples"] * 2):
        return pd.Series(-1, index=ev.index), pd.DataFrame()
    f = pd.DataFrame(index=ev.index)
    f["log_duration"] = np.log1p(ev["duration_s"].astype(float))
    f["log_width"] = np.log1p(ev["width_x"].astype(float))
    f["log_abs_amplitude"] = np.log1p(ev["amplitude"].abs().astype(float))
    f["signed_dir"] = np.sign(ev["amplitude"].astype(float))
    f["log_score_norm"] = np.log(ev["anomaly_score_norm"].astype(float).clip(lower=1e-6))
    f["abrupt_f"] = ev["abrupt"].map({True: 1.0, False: 0.0}).fillna(0.5).astype(float)
    f["persistent_f"] = ev["persistent"].map({True: 1.0, False: 0.0}).fillna(0.5).astype(float)
    X = StandardScaler().fit_transform(f[CLUSTER_FEATURES].to_numpy())
    lab = DBSCAN(eps=pc["dbscan_eps"], min_samples=pc["dbscan_min_samples"]).fit_predict(X)
    labels = pd.Series(lab, index=ev.index)
    prof = f.assign(cluster=lab, duration_s=ev["duration_s"], width_x=ev["width_x"],
                    amplitude=ev["amplitude"]).groupby("cluster").agg(
        n=("duration_s", "size"), duration_s_median=("duration_s", "median"),
        width_x_median=("width_x", "median"), amplitude_median=("amplitude", "median"),
        abrupt_share=("abrupt_f", "mean"), persistent_share=("persistent_f", "mean")).reset_index()
    return labels, prof


def apply_patterns(ev: pd.DataFrame, cfg: dict, dt_by_source: dict, source_times: dict
                   ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Добавляет в события repeat_group, sync_group, cluster и pattern_tags."""
    pc = cfg["patterns"]
    ev = ev.copy()
    if ev.empty:
        for c in ("repeat_group", "repeat_count", "sync_group", "sync_size", "cluster"):
            ev[c] = pd.Series(dtype=int)
        return ev, {}
    ev["repeat_group"] = find_repeats(ev, pc["repeat_x_tol"])
    rs = repeat_stats(ev, cfg)
    size = rs.set_index("repeat_group")["n_events"]
    periodic = rs.set_index("repeat_group")["periodic"]
    cv = rs.set_index("repeat_group")["interval_cv"]
    ev["repeat_count"] = ev["repeat_group"].map(size).astype(int)
    ev["sync_group"] = find_sync_within_source(ev, pc["sync_window_s"], dt_by_source)
    sync_size = ev[ev["sync_group"] >= 0].groupby("sync_group").size()
    ev["sync_size"] = ev["sync_group"].map(sync_size).fillna(0).astype(int)
    ev["cluster"], cluster_profile = cluster_events(ev, cfg)
    cross = cross_source_coincidences(ev, pc["sync_window_s"])
    cross_ids = set(cross["event_a"]) | set(cross["event_b"]) if len(cross) else set()

    tags_all = []
    for i, r in ev.iterrows():
        tags = []
        if r["repeat_count"] >= 2:
            tags.append(f"повторяющееся ({r['repeat_count']} раз на участке)")
            if periodic.get(r["repeat_group"], False):
                tags.append(f"возможно периодическое (CV интервалов {cv[r['repeat_group']]:.2f})")
        else:
            tags.append("единичное: не встречалось в наблюдаемом периоде")
        if r["sync_size"] >= 2:
            tags.append(f"синхронное ({r['sync_size']} участков одного канала)")
        if r["event_id"] in cross_ids:
            tags.append("совпадение по времени с другим прибором (часы не сверены)")
        tags_all.append(json.dumps(tags, ensure_ascii=False))
    ev["pattern_tags"] = tags_all
    ev["repetition_class"] = np.where(ev["repeat_count"] >= 2, "повторяющееся", "единичное")
    acf = event_rate_autocorr(ev, source_times)
    return ev, dict(repeat_groups=rs, cross_source=cross, autocorr=acf,
                    cluster_profile=cluster_profile)
