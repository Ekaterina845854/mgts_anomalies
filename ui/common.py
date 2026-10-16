"""Общие элементы интерфейса: стили, цвета, загрузка результатов, форматирование, фильтры."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src.repository import ResultStore

# палитра методов; 4-й цвет дополнительно различается формой маркера
DET_COLORS = {"statistical": "#2a78d6", "isolation_forest": "#eb6834", "regression": "#1baf7a",
              "autoencoder": "#6250d6"}
DET_SYMBOLS = {"statistical": "circle", "isolation_forest": "square", "regression": "diamond",
               "autoencoder": "triangle-up"}
DET_SHORT = {"statistical": "Статистический", "isolation_forest": "Isolation Forest",
             "regression": "Регрессия", "autoencoder": "Автоэнкодер"}
DET_ORDER = ["statistical", "isolation_forest", "regression", "autoencoder"]
TYPE_COLORS = {"Резкий импульс": "#eda100", "Ступенчатое изменение": "#e87ba4",
               "Длительное отклонение": "#008300", "Кратковременное плавное отклонение": "#e34948"}
TYPE_ORDER = list(TYPE_COLORS)
SEQ_BLUE = [[0.0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"], [0.5, "#3987e5"],
            [0.67, "#256abf"], [0.83, "#184f95"], [1.0, "#0d366b"]]
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec7f7f"],
             [1.0, "#a52626"]]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
h1 {font-size: 1.85rem !important; font-weight: 650 !important; margin-bottom: 0 !important;}
.fa-card {background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 12px;
          padding: 14px 16px; margin-bottom: 12px;}
.fa-card h4 {margin: 0 0 8px 0; font-size: 1.02rem; font-weight: 640;}
.fa-kv {display: grid; grid-template-columns: 46% 54%; row-gap: 5px; font-size: 0.88rem;}
.fa-kv .k {color: #52514e;} .fa-kv .v {color: #0b0b0b; font-weight: 520; word-break: break-word;}
.fa-badge {display: inline-block; padding: 2px 9px; border-radius: 10px; font-size: 0.78rem;
           background: #f3e9e4; color: #8a3b1c; font-weight: 560;}
.fa-badge.gray {background: #eceeea; color: #3d4440;}
.fa-badge.green {background: #e3efe6; color: #1f5a3b;}
.fa-stat {background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 12px;
          padding: 12px 14px;}
.fa-stat .lbl {color: #52514e; font-size: 0.82rem;} .fa-stat .val {font-size: 1.45rem; font-weight: 640;}
.fa-stat .hint {color: #898781; font-size: 0.76rem;}
.fa-det {border-radius: 12px; border: 1px solid rgba(11,11,11,0.08); background:#fff; padding: 12px 14px;}
.fa-det .fa-kv {grid-template-columns: 58% 42%; font-size: 0.84rem;}
.fa-det .name {font-weight: 640; display:flex; align-items:center; gap:8px; margin-bottom:6px;}
.fa-dot {width: 10px; height: 10px; border-radius: 50%; display:inline-block;}
.fa-note {color: #52514e; font-size: 0.84rem;}
[data-testid="stSidebar"] {background: #EEF1EC;}
.fa-brand {font-size: 1.25rem; font-weight: 700; color: #1B1F1D; margin-bottom: 0;}
.fa-brand-sub {color: #52514e; font-size: 0.8rem; margin-top: -4px;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str) -> None:
    st.title(title)


def card_open(title: str, extra: str = "") -> str:
    return f"<div class='fa-card'><h4>{title} {extra}</h4>"


def kv_html(pairs: list[tuple[str, str]]) -> str:
    rows = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in pairs)
    return f"<div class='fa-kv'>{rows}</div>"


def stat_tile(label: str, value: str, hint: str = "") -> str:
    h = f"<div class='hint'>{hint}</div>" if hint else ""
    return f"<div class='fa-stat'><div class='lbl'>{label}</div><div class='val'>{value}</div>{h}</div>"


def store() -> ResultStore:
    return ResultStore()


@st.cache_data(show_spinner=False)
def _events(mtime: float) -> pd.DataFrame:
    ev = store().load_events()
    if ev.empty:
        return ev
    for c in ("start_time", "end_time", "peak_time"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    ev["pattern_tags_list"] = ev["pattern_tags"].map(json.loads)
    ev["device_channel"] = ev["dts_id"].str[:8] + " / к" + ev["channel_id"].astype(str)
    return ev


@st.cache_data(show_spinner=False)
def _json(name: str, mtime: float):
    return store().load_json(name)


@st.cache_data(show_spinner=False, max_entries=8)
def _prepared(source_id: str, mtime: float) -> dict:
    return store().load_prepared(source_id)


@st.cache_data(show_spinner=False, max_entries=24)
def _scores(source_id: str, detector: str, mtime: float):
    return store().load_scores(source_id, detector)


@st.cache_data(show_spinner=False)
def _table(name: str, mtime: float) -> pd.DataFrame:
    return store().load_table(name)


def mtime() -> float:
    return store().mtime()


def events() -> pd.DataFrame:
    return _events(mtime())


def metrics() -> dict:
    return _json("metrics.json", mtime()) or {}


def data_report() -> dict:
    return _json("data_report.json", mtime()) or {}


def manifest() -> dict | None:
    return _json("run_manifest.json", mtime())


def prepared(source_id: str) -> dict:
    return _prepared(source_id, mtime())


def scores(source_id: str, detector: str):
    return _scores(source_id, detector, mtime())


def table(name: str) -> pd.DataFrame:
    return _table(name, mtime())


def source_info() -> pd.DataFrame:
    return pd.DataFrame(data_report().get("source_info", []))


def require_results() -> bool:
    if not store().exists():
        st.warning("Результаты пайплайна не найдены. Запустите расчёт: `python -m src.pipeline` "
                   "или на странице «Данные и ограничения» → «Параметры и пересчёт».")
        return False
    return True


def tz_label(info: dict | pd.Series) -> str:
    if not info.get("tz_known", True):
        return "пояс не указан"
    off = info.get("tz_offset_h")
    if off is None or (isinstance(off, float) and np.isnan(off)):
        return "UTC"
    return f"UTC{'+' if off >= 0 else '−'}{abs(off):02.0f}"


def to_local(ts, offset) -> pd.Timestamp | pd.Series | pd.DatetimeIndex:
    """UTC -> наивное локальное время источника (для подписей и осей)."""
    off = 0.0 if offset is None or (isinstance(offset, float) and np.isnan(offset)) else float(offset)
    if isinstance(ts, pd.Series):
        return (ts + pd.Timedelta(hours=off)).dt.tz_localize(None)
    return (ts + pd.Timedelta(hours=off)).tz_localize(None)


def fmt_dt(ts, offset, fmt="%d.%m.%Y %H:%M:%S") -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return to_local(pd.Timestamp(ts), offset).strftime(fmt)


def fmt_dur(sec) -> str:
    if sec is None or pd.isna(sec):
        return "—"
    sec = float(sec)
    if sec < 60:
        return f"{sec:.0f} с"
    if sec < 3600:
        return f"{sec / 60:.0f} мин"
    return f"{sec // 3600:.0f} ч {(sec % 3600) / 60:.0f} мин"


def fmt_num(v, nd=2, unit="") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "—"
    return f"{v:,.{nd}f}".replace(",", " ") + (f" {unit}" if unit else "")


def x_unit_short(x_unit: str) -> str:
    return "м" if str(x_unit).startswith("м") else "инд."


def source_label(row) -> str:
    return f"Файл {row['file']} · прибор {row['dts_id'][:8]} / канал {row['channel_id']}"


def source_selector(key: str = "src", label: str = "Источник (файл · прибор / канал)") -> dict:
    si = source_info()
    opts = si["source_id"].tolist()
    labels = {r.source_id: source_label(r._asdict()) for r in si.itertuples()}
    cur = st.session_state.get("g_source", opts[0])
    sel = st.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0,
                       format_func=lambda s: labels[s], key=key)
    st.session_state["g_source"] = sel
    return si[si["source_id"] == sel].iloc[0].to_dict()


def detector_selector(key: str = "det", allow_all: bool = False) -> str:
    opts = (["all"] if allow_all else []) + DET_ORDER
    cur = st.session_state.get("g_detector", "statistical")
    idx = opts.index(cur) if cur in opts else 0
    sel = st.selectbox("Алгоритм", opts, index=idx, key=key,
                       format_func=lambda d: "Все алгоритмы" if d == "all" else DET_SHORT[d])
    if sel != "all":
        st.session_state["g_detector"] = sel
    return sel


def period_selector(times: pd.DatetimeIndex, offset, key: str = "period") -> tuple[datetime, datetime]:
    loc = to_local(times, offset)
    t0, t1 = loc[0].to_pydatetime(), loc[-1].to_pydatetime()
    if t0 == t1:
        return t0, t1
    return st.slider("Период (локальное время источника)", min_value=t0, max_value=t1,
                     value=(t0, t1), format="DD.MM HH:mm", key=key)


def plot_layout(fig, height=360, legend=True, margin=None):
    fig.update_layout(
        height=height, margin=margin or dict(l=10, r=10, t=10, b=10), paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff", font=dict(family="system-ui, Segoe UI, sans-serif", size=12, color=INK2),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK))
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False)
    return fig
