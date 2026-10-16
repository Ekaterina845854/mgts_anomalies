"""Представление готовых событий для интерфейса; в детекции не участвует."""
from __future__ import annotations

import numpy as np
import pandas as pd

ATTENTION_COLORS = {"Высокий": "#b42318", "Средний": "#b45309", "Низкий": "#626b75"}
HIGH_SHARE = 0.2
MEDIUM_SHARE = 0.3
PRIORITY_WEIGHTS = {"score": 0.4, "amp": 0.3, "agree": 0.3}


def _get(e, name, fallback=None):
    v = e.get(name, fallback)
    if v is None:
        return fallback
    try:
        if pd.isna(v):
            return fallback
    except (TypeError, ValueError):
        pass
    return v


def _true(v):
    return v is True or isinstance(v, (bool, np.bool_)) and bool(v)


def x_label(e) -> str:
    # метры только если единица X указана в данных
    suffix = "м" if str(_get(e, "x_unit", "")).startswith("м") else "ед. X"
    a, b = float(_get(e, "x_start", 0)), float(_get(e, "x_end", 0))
    def n(v):
        return f"{v:.0f}" if abs(v - round(v)) < 0.06 else f"{v:.1f}"
    return f"{n(a)}–{n(b)} {suffix}"


def event_title(e) -> str:
    direction = "повышение" if float(_get(e, "amplitude", 0)) >= 0 else "снижение"
    abrupt = _get(e, "abrupt")
    persistent = _true(_get(e, "persistent"))
    length = str(_get(e, "duration_class", ""))
    if _true(abrupt) and persistent:
        prefix = "Резкое устойчивое"
    elif _true(abrupt):
        prefix = "Резкое кратковременное" if length == "кратковременное" else "Резкое"
    elif str(_get(e, "event_type", "")) == "Ступенчатое изменение":
        prefix = "Ступенчатое"
    elif abrupt is False or (isinstance(abrupt, (bool, np.bool_)) and not abrupt):
        prefix = "Плавное устойчивое" if length == "длительное" else "Постепенное"
    elif length == "длительное" or persistent:
        prefix = "Устойчивое"
    else:
        prefix = "Кратковременное"
    return f"{prefix} {direction} сигнала на участке {x_label(e)}"


def method_counts(events: pd.DataFrame, tol_s: float = 300, tol_x: float = 2) -> pd.Series:
    """Число разных методов, нашедших пересекающееся событие в том же источнике (±tol_s, ±tol_x)."""
    if events.empty:
        return pd.Series(dtype=int, index=events.index)
    result = pd.Series(1, index=events.index, dtype=int)
    tol = int(tol_s * 1e9)
    for _, g in events.groupby("source_id", sort=False):
        start = pd.to_datetime(g["start_time"], utc=True).astype("int64").to_numpy()
        end = pd.to_datetime(g["end_time"], utc=True).astype("int64").to_numpy()
        xmin = g["x_start"].to_numpy(dtype=float)
        xmax = g["x_end"].to_numpy(dtype=float)
        methods = g["detector"].astype(str).to_numpy()
        order = np.argsort(start, kind="stable")
        starts_sorted = start[order]
        counts = np.ones(len(g), dtype=int)
        for i in range(len(g)):
            cand = order[:np.searchsorted(starts_sorted, end[i] + tol, side="right")]
            matches = cand[(end[cand] >= start[i] - tol) &
                           (xmin[cand] <= xmax[i] + tol_x) &
                           (xmax[cand] >= xmin[i] - tol_x)]
            counts[i] = len(set(methods[matches]))
        result.loc[g.index] = counts
    return result


def _pct_in_detector(values: pd.Series, detector: pd.Series) -> pd.Series:
    # середина ранга: у единственного события метода 0.5, а не 1.0
    g = values.groupby(detector)
    return (g.rank(method="average") - 0.5) / g.transform("size")


def attention_levels(events: pd.DataFrame) -> pd.Series:
    """Приоритет просмотра, а не порог детектора.

    Шкалы методов несопоставимы, поэтому оценка и |z| переводятся в ранги внутри метода
    и смешиваются с согласием методов. Высокий — верхние HIGH_SHARE событий,
    средний — следующие MEDIUM_SHARE, остальные — низкий.
    """
    if events.empty:
        return pd.Series(dtype=str, index=events.index)
    det = events["detector"].astype(str)
    score = _pct_in_detector(events["anomaly_score_norm"].astype(float).fillna(0), det)
    amp = _pct_in_detector(events["amplitude_z"].astype(float).abs().fillna(0), det)
    n_methods = max(det.nunique(), 2)
    agree = ((events["method_count"].astype(float) - 1) / (n_methods - 1)).clip(0, 1)
    w = PRIORITY_WEIGHTS
    rank = (w["score"] * score + w["amp"] * amp + w["agree"] * agree).rank(pct=True)
    return pd.Series(np.select([rank > 1 - HIGH_SHARE, rank > 1 - HIGH_SHARE - MEDIUM_SHARE],
                               ["Высокий", "Средний"], "Низкий"), index=events.index)


def add_presentation_fields(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.assign(title=pd.Series(dtype=str), method_count=pd.Series(dtype=int),
                             attention=pd.Series(dtype=str))
    df = events.copy()
    df["title"] = [event_title(row) for _, row in df.iterrows()]
    df["method_count"] = method_counts(df)
    df["attention"] = attention_levels(df)
    return df


def hypothesis(e) -> str:
    """Предположение, не диагноз."""
    desc = str(_get(e, "direction", ""))
    if _true(_get(e, "persistent")):
        return ("Возможно устойчивое изменение локальных условий или режима измерений. "
                "Для проверки необходимы данные о нагрузке, среде и состоянии прибора.")
    if _true(_get(e, "abrupt")):
        return ("Возможны кратковременное локальное воздействие либо артефакт измерения. "
                "Нужно сопоставить с соседними координатами и независимыми наблюдениями.")
    if "снижен" in desc:
        return "Возможно временное охлаждение или изменение условий регистрации; причина не установлена."
    return ("Возможно изменение температуры, нагрузки или режима измерения. "
            "Источник изменения по данным датчика не установлен.")


def representative_events(events: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Показательные события без дублей одной находки разными методами."""
    if events.empty:
        return events.head(0)
    score = events["anomaly_score_norm"].fillna(0).astype(float)
    ranked = events.assign(_rank=(events["attention"].map({"Высокий": 3, "Средний": 2, "Низкий": 1})
                                   .fillna(0) * 1000 + events["method_count"] * 100 +
                                   score.clip(0, 100))).sort_values("_rank", ascending=False)
    chosen = []
    for _, e in ranked.iterrows():
        if any(e["source_id"] == old["source_id"] and
               e["start_time"] <= old["end_time"] + pd.Timedelta(minutes=5) and
               e["end_time"] >= old["start_time"] - pd.Timedelta(minutes=5) and
               e["x_start"] <= old["x_end"] + 2 and e["x_end"] >= old["x_start"] - 2
               for old in chosen):
            continue
        chosen.append(e)
        if len(chosen) >= limit:
            break
    return pd.DataFrame(chosen).drop(columns=["_rank"], errors="ignore") if chosen else events.head(0)
