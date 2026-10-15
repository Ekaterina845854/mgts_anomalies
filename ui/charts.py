"""Plotly-графики. Все данные - из сохранённых результатов пайплайна."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .common import (DET_COLORS, DET_ORDER, DET_SYMBOLS, DET_SHORT, DIVERGING, INK, MUTED, SEQ_BLUE,
                     TYPE_COLORS, TYPE_ORDER, fmt_num, plot_layout, to_local, x_unit_short)

MAX_T, MAX_B = 700, 450


def _block_reduce(a: np.ndarray, ft: int, fb: int) -> np.ndarray:
    """Прореживание матрицы блочным средним (только для отображения)."""
    T, B = a.shape
    Tn, Bn = T // ft * ft, B // fb * fb
    if Tn == 0 or Bn == 0:
        return a
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a[:Tn, :Bn].reshape(Tn // ft, ft, Bn // fb, fb), axis=(1, 3))


def heatmap(prep: dict, tmask: np.ndarray, offset, mode: str, x_unit: str,
            events: pd.DataFrame | None = None, selected: str | None = None, height: int = 430):
    """Тепловая карта «время × координата» (бины). mode: 'value' | 'z'."""
    data = prep["vb"] if mode == "value" else prep["z"]
    idx = np.flatnonzero(tmask)
    a = data[idx]
    ft = max(1, int(np.ceil(len(idx) / MAX_T)))
    fb = max(1, int(np.ceil(a.shape[1] / MAX_B)))
    red = _block_reduce(a, ft, fb)
    tt = to_local(prep["times"][idx], offset)[::ft][:red.shape[0]]
    xc = ((prep["bin_x_start"] + prep["bin_x_end"]) / 2)[::fb][:red.shape[1]]
    if mode == "value":
        lo, hi = np.nanpercentile(red, [1, 99.5])
        hm = go.Heatmap(z=red.T, x=tt, y=xc, colorscale=SEQ_BLUE, zmin=lo, zmax=hi,
                        colorbar=dict(title=dict(text="значение", side="right"), thickness=12),
                        hovertemplate="%{x}<br>x=%{y:.1f}<br>значение=%{z:.2f}<extra></extra>")
    else:
        hm = go.Heatmap(z=red.T, x=tt, y=xc, colorscale=DIVERGING, zmin=-8, zmax=8,
                        colorbar=dict(title=dict(text="z (σ)", side="right"), thickness=12),
                        hovertemplate="%{x}<br>x=%{y:.1f}<br>z=%{z:.1f}<extra></extra>")
    fig = go.Figure(hm)
    if events is not None and len(events):
        add_event_boxes(fig, events, offset, selected)
    plot_layout(fig, height=height, legend=False)
    fig.update_yaxes(title=f"координата вдоль волокна, {x_unit_short(x_unit)}")
    fig.update_xaxes(title=None)
    return fig


def add_event_boxes(fig, ev: pd.DataFrame, offset, selected: str | None = None, max_boxes: int = 40,
                    max_labels: int = 15):
    """Рамки (у сильнейших) + кликабельные маркеры всех событий в точке пика (customdata = event_id)."""
    ev = ev.sort_values("anomaly_score_norm", ascending=False)
    top = set(ev["event_id"].head(max_labels)) | {selected}
    for r in ev.head(max_boxes).itertuples():
        sel = r.event_id == selected
        x0 = to_local(r.start_time, offset)
        x1 = to_local(r.end_time, offset)
        if x1 <= x0:
            x1 = x0 + pd.Timedelta(seconds=60)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=r.x_start, y1=r.x_end,
                      line=dict(color=INK if not sel else "#d03b3b", width=2.5 if sel else 1.2,
                                dash="solid" if sel else "dot"), fillcolor="rgba(0,0,0,0)")
    fig.add_trace(go.Scatter(
        x=to_local(ev["peak_time"], offset), y=ev["x_peak"], mode="markers+text",
        marker=dict(size=[12 if e == selected else 8 for e in ev["event_id"]], color="#ffffff",
                    line=dict(color=["#d03b3b" if e == selected else INK for e in ev["event_id"]], width=2)),
        text=[(f"<b>{e}</b>" if e == selected else "#" + e.split("-")[-1].lstrip("0")) if e in top else ""
              for e in ev["event_id"]],
        textposition="top center", textfont=dict(size=10, color=INK),
        customdata=ev[["event_id"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>%{x}<br>x=%{y:.1f}<extra>нажмите для деталей</extra>",
        showlegend=False))


def fragment_chart(prep: dict, ev: pd.Series, offset, height=210):
    """Среднее значение участка события до/во время/после события."""
    b0, b1, t0, t1 = int(ev["b0"]), int(ev["b1"]), int(ev["t0"]), int(ev["t1"])
    tsec = prep["t_sec"]
    span = max(tsec[t1] - tsec[t0], 600.0)
    m = (tsec >= tsec[t0] - 2 * span) & (tsec <= tsec[t1] + 2 * span)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.nanmean(prep["vb"][m][:, b0:b1 + 1], axis=1)
    tt = to_local(prep["times"][m], offset)
    fig = go.Figure(go.Scatter(x=tt, y=s, mode="lines", line=dict(color="#2a78d6", width=2),
                               name="среднее по участку",
                               hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    x0 = to_local(prep["times"][t0], offset)
    x1 = to_local(prep["times"][t1], offset)
    if x1 <= x0:
        x1 = x0 + pd.Timedelta(seconds=30)
    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(208,59,59,0.14)", line_width=0)
    plot_layout(fig, height=height, legend=False)
    fig.update_yaxes(title="значение")
    return fig


def profile_chart(prep: dict, t_idx: int, offset, x_unit: str, ev: pd.Series | None = None,
                  height=300):
    """Профиль вдоль волокна в момент t_idx и медианный профиль источника для сравнения."""
    xc = (prep["bin_x_start"] + prep["bin_x_end"]) / 2
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(prep["vb"], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xc, y=med, mode="lines", name="медиана за весь период",
                             line=dict(color=MUTED, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=xc, y=prep["vb"][t_idx], mode="lines",
                             name=f"профиль {to_local(prep['times'][t_idx], offset):%d.%m %H:%M:%S}",
                             line=dict(color="#2a78d6", width=2)))
    if ev is not None:
        fig.add_vrect(x0=ev["x_start"], x1=ev["x_end"], fillcolor="rgba(208,59,59,0.14)", line_width=0,
                      annotation_text=ev["event_id"], annotation_position="top left")
    plot_layout(fig, height=height)
    fig.update_xaxes(title=f"координата, {x_unit_short(x_unit)}")
    fig.update_yaxes(title="значение")
    return fig


def timeline(ev: pd.DataFrame, offset, height=260):
    """Наложение событий всех алгоритмов на одну временную шкалу."""
    fig = go.Figure()
    for d in DET_ORDER:
        e = ev[ev["detector"] == d]
        if e.empty:
            continue
        st = to_local(e["start_time"], offset)
        en = to_local(e["end_time"], offset)
        dur = (en - st).dt.total_seconds().clip(lower=60) * 1000
        fig.add_trace(go.Bar(base=st, x=dur, y=[DET_SHORT[d]] * len(e), orientation="h",
                             marker=dict(color=DET_COLORS[d], line=dict(width=0)), name=DET_SHORT[d],
                             customdata=np.c_[e["event_id"], e["x_start"].round(1), e["x_end"].round(1)],
                             hovertemplate="<b>%{customdata[0]}</b><br>x %{customdata[1]}–%{customdata[2]}"
                                           "<extra></extra>"))
    plot_layout(fig, height=height)
    fig.update_xaxes(type="date", title=None)
    fig.update_layout(barmode="overlay", bargap=0.45)
    return fig


def space_time_overlay(ev: pd.DataFrame, offset, x_unit: str, height=380):
    """События всех алгоритмов в координатах «время × расстояние» (прямоугольники)."""
    fig = go.Figure()
    for d in DET_ORDER:
        e = ev[ev["detector"] == d]
        if e.empty:
            continue
        xs, ys = [], []
        for r in e.itertuples():
            a = to_local(r.start_time, offset)
            b = to_local(r.end_time, offset)
            if b <= a:
                b = a + pd.Timedelta(seconds=60)
            xs += [a, b, b, a, a, None]
            ys += [r.x_start, r.x_start, r.x_end, r.x_end, r.x_start, None]
        dash = {"statistical": "solid", "isolation_forest": "dash", "regression": "dot",
                "autoencoder": "dashdot"}[d]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=DET_COLORS[d], width=2, dash=dash),
                                 name=DET_SHORT[d], hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=to_local(e["peak_time"], offset), y=e["x_peak"], mode="markers",
                                 marker=dict(size=9, color=DET_COLORS[d], symbol=DET_SYMBOLS[d],
                                             line=dict(color="#fff", width=2)),
                                 customdata=e[["event_id"]].to_numpy(), showlegend=False,
                                 hovertemplate="<b>%{customdata[0]}</b><br>%{x}<br>x=%{y:.1f}<extra></extra>"))
    plot_layout(fig, height=height)
    fig.update_yaxes(title=f"координата, {x_unit_short(x_unit)}")
    return fig


def grouped_bars(df: pd.DataFrame, cat: str, series: str, value: str, colors: dict, order_cat=None,
                 order_series=None, labels: dict | None = None, height=300, text=True):
    """Сгруппированные столбцы; цвет закреплён за сущностью (series)."""
    fig = go.Figure()
    for s in order_series or df[series].unique():
        d = df[df[series] == s]
        if order_cat is not None:
            d = d.set_index(cat).reindex(order_cat).reset_index()
        fig.add_trace(go.Bar(x=d[cat], y=d[value], name=(labels or {}).get(s, s),
                             marker=dict(color=colors.get(s, MUTED), line=dict(color="#fff", width=2)),
                             text=d[value] if text else None, textposition="outside",
                             hovertemplate="%{x}: %{y}<extra>" + str((labels or {}).get(s, s)) + "</extra>"))
    plot_layout(fig, height=height)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return fig


def type_bars(ev: pd.DataFrame, height=300):
    c = ev.groupby(["detector", "event_type"]).size().rename("n").reset_index()
    c["detector"] = c["detector"].map(DET_SHORT)
    return grouped_bars(c, "event_type", "detector", "n",
                        {DET_SHORT[k]: v for k, v in DET_COLORS.items()}, order_cat=TYPE_ORDER,
                        order_series=[DET_SHORT[d] for d in DET_ORDER], height=height)


def score_hist(ev: pd.DataFrame, height=260):
    fig = go.Figure()
    for d in DET_ORDER:
        e = ev[ev["detector"] == d]
        fig.add_trace(go.Histogram(x=e["anomaly_score_norm"], name=DET_SHORT[d], opacity=0.75,
                                   marker=dict(color=DET_COLORS[d]), nbinsx=40))
    plot_layout(fig, height=height)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="оценка / порог (≥1 = срабатывание)")
    return fig


__all__ = ["heatmap", "fragment_chart", "profile_chart", "timeline", "space_time_overlay",
           "grouped_bars", "type_bars", "score_hist", "TYPE_COLORS", "fmt_num"]
