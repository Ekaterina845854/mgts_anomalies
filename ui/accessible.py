"""Экраны интерфейса поверх готовых результатов, без пересчёта детекторов."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ARTIFACTS_DIR, LABELS_PATH
from src.evaluation import LABEL_VALUES, append_label
from . import charts, views
from .common import (DET_SHORT, events, fmt_dt, fmt_dur, fmt_num, header, manifest, mtime,
                     plot_layout, prepared, require_results, source_info, to_local)
from .presentation import (ATTENTION_COLORS, HIGH_SHARE, MEDIUM_SHARE, add_presentation_fields,
                           event_title, hypothesis, representative_events, x_label)

NOTICE = "Событие — необычное изменение сигнала, которое стоит проверить. Это ещё не авария."
METHOD_FORMS = ("метод", "метода", "методов")
EVENT_FORMS = ("событие", "события", "событий")


@st.cache_data(show_spinner=False)
def annotated_events(_mtime: float) -> pd.DataFrame:
    return add_presentation_fields(events())


def _all() -> pd.DataFrame:
    return annotated_events(mtime())


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _src_lookup():
    si = source_info()
    return {r["source_id"]: r for _, r in si.iterrows()} if not si.empty else {}


def _file(e, lookup):
    row = lookup.get(e["source_id"])
    return str(row["file"]) if row is not None else str(e["source_id"])


def _dt(e):
    known = bool(e.get("tz_known", False))
    text = fmt_dt(e["start_time"], e.get("tz_offset_h") if known else None)
    return text if known else text + " (пояс не указан)"


def _goto_detail(event_id: str):
    st.session_state["ui_selected_event"] = event_id
    st.session_state["ui_show_detail_here"] = True


def _summary_timeline(ev: pd.DataFrame, lookup):
    sources = list(ev["source_id"].drop_duplicates())
    if not sources:
        st.caption("Событий нет.")
        return
    sid = st.selectbox("Источник", sources, format_func=lambda s: _file({"source_id": s}, lookup),
                       key="summary_timeline_source")
    subset = ev[ev["source_id"] == sid].copy()
    if subset.empty:
        st.caption("У этого источника событий нет.")
        return
    off = subset["tz_offset_h"].iloc[0] if bool(subset["tz_known"].iloc[0]) else None
    fig = go.Figure()
    for att in ("Высокий", "Средний", "Низкий"):
        part = subset[subset["attention"] == att]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=to_local(part["peak_time"], off), y=part["x_peak"], name=att,
            mode="markers", marker=dict(color=ATTENTION_COLORS[att], size=8, opacity=0.8),
            customdata=np.array([[str(r["title"]), str(r["event_id"])] for _, r in part.iterrows()]),
            hovertemplate="%{customdata[0]}<br>%{x}<br>X=%{y:.1f}<extra></extra>"))
    plot_layout(fig, height=290)
    fig.update_xaxes(title="Время (пояс источника)")
    fig.update_yaxes(title="Координата X")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("Источники показаны по отдельности: у части файлов не указан часовой пояс. "
               "Близкие точки часто означают одно событие, найденное разными методами.")


def _example(ev: pd.DataFrame) -> None:
    if ev.empty:
        st.info("Пример появится, когда в результатах будут события.")
        return
    eid = str(representative_events(ev, limit=5).iloc[0]["event_id"])
    if st.button("▶ Показать пример", type="primary", key="show_example",
                 help="Открыть карточку одного из самых заметных событий"):
        _goto_detail(eid)


def page_summary():
    header("Итоги")
    st.info(NOTICE)
    man = manifest()
    if not man:
        st.warning("Результатов пока нет. Запустите `python -m src.pipeline` из корня проекта.")
        return
    ev = _all()
    test_n = int((ev["split"] == "test").sum()) if not ev.empty else 0
    matched = int((ev["method_count"] >= 2).sum()) if not ev.empty else 0
    cols = st.columns(5)
    statuses = {"ok": "Успешно", "success": "Успешно"}
    failed = ARTIFACTS_DIR / "last_failed_run.json"
    complete = ARTIFACTS_DIR / "results" / "run_manifest.json"
    has_new_failure = failed.exists() and (not complete.exists() or failed.stat().st_mtime > complete.stat().st_mtime)
    if has_new_failure:
        st.error("Последний запуск расчёта упал. Показаны результаты предыдущего успешного запуска.")
    values = [
        ("Последний расчёт", "Ошибка" if has_new_failure else
         statuses.get(str(man.get("status", "")), str(man.get("status", "неизвестно")))),
        ("Источников", str(man.get("n_sources", "—"))),
        ("Событий", str(len(ev))),
        ("Событий на test", str(test_n)),
        ("Совпадений методов", str(matched)),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, value)
    st.caption(f"Расчёт завершён {str(man.get('finished_utc', '—'))[:19].replace('T', ' ')} UTC. "
               "Совпадения методов — события, которые нашёл ещё хотя бы один метод (±300 с, ±2 ед. X).")
    if ev.empty:
        st.info("Событий не найдено.")
        return
    lookup = _src_lookup()
    a, b = st.columns([1, 2.2])
    with a:
        _example(ev)
        st.markdown("#### Уровни внимания")
        for level in ("Высокий", "Средний", "Низкий"):
            n = int((ev["attention"] == level).sum())
            st.markdown(f"<span style='color:{ATTENTION_COLORS[level]};font-weight:650'>{level}</span>"
                        f" — {n} {_plural(n, EVENT_FORMS)}", unsafe_allow_html=True)
        st.caption("Уровень подсказывает, что смотреть первым, а не насколько опасно событие.")
    with b:
        st.markdown("#### Самые заметные события")
        for _, e in representative_events(ev, 5).iterrows():
            n = int(e["method_count"])
            with st.container(border=True):
                st.markdown(f"**{event_title(e)}**")
                st.caption(f"{_file(e, lookup)} · {_dt(e)} · {n} {_plural(n, METHOD_FORMS)} · "
                           f"уровень: {e['attention'].lower()}")
                if int(e.get("repeat_count", 0) or 0) >= 2:
                    st.markdown("<span style='color:#2264ac'>↻ Повторяется</span>", unsafe_allow_html=True)
                if st.button("Подробнее", key=f"summary_detail_{e['event_id']}"):
                    _goto_detail(str(e["event_id"]))
    chosen = st.session_state.get("ui_selected_event")
    if st.session_state.get("ui_show_detail_here") and chosen in set(ev["event_id"]):
        st.divider()
        detail_card(ev.loc[ev["event_id"] == chosen].iloc[0])
    st.markdown("#### События во времени")
    _summary_timeline(ev, lookup)


def detail_card(e: pd.Series):
    lookup = _src_lookup()
    si = lookup.get(e["source_id"])
    off = si.get("tz_offset_h") if si is not None and bool(si.get("tz_known", False)) else None
    st.markdown(f"### {event_title(e)}")
    st.caption(f"Событие {e['event_id']} · {_file(e, lookup)} · "
               f"{DET_SHORT.get(e['detector'], e['detector'])}")
    if int(e.get("repeat_count", 0) or 0) >= 2:
        st.markdown("<span style='color:#2264ac;font-weight:600'>↻ Событие повторяется на этом участке</span>",
                    unsafe_allow_html=True)
    st.markdown("#### Что произошло")
    st.write(f"{'Повышение' if float(e['amplitude']) >= 0 else 'Снижение'} сигнала на участке {x_label(e)}, "
             f"длительность {fmt_dur(e['duration_s'])}.")
    st.markdown("#### Где и когда")
    x1, x2 = st.columns(2)
    x1.write(f"**Источник:** {_file(e, lookup)}, канал {e['channel_id']}")
    x1.write(f"**Участок:** {x_label(e)}, пик X={fmt_num(e['x_peak'], 1)}")
    x2.write(f"**Начало:** {fmt_dt(e['start_time'], off)}")
    x2.write(f"**Конец:** {fmt_dt(e['end_time'], off)}")
    if not bool(e.get("tz_known", False)):
        st.caption("Часовой пояс источника не указан.")
    st.markdown("#### Насколько необычно")
    lev = e["attention"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Уровень внимания:** <span style='color:{ATTENTION_COLORS[lev]}'>{lev}</span>",
                unsafe_allow_html=True)
    c2.metric("Оценка / порог", fmt_num(e.get("anomaly_score_norm"), 2, "×"))
    c3.metric("Амплитуда", f"{float(e['amplitude']):+.2f}")
    st.caption("Амплитуда в единицах прибора: без паспорта DTS в °C не пересчитываем.")
    st.markdown("#### Другие методы")
    n = int(e["method_count"])
    st.write("Другие методы это событие не нашли." if n < 2 else
             f"Событие нашли {n} {_plural(n, METHOD_FORMS)}, включая этот.")
    st.markdown("#### Сигнал до, во время и после")
    try:
        prep = prepared(e["source_id"])
        st.plotly_chart(charts.fragment_chart(prep, e, off, height=270), width="stretch",
                        config={"displaylogo": False}, key=f"fragment_{e['event_id']}")
        st.caption("Выделен интервал события, линия — среднее по участку.")
    except (FileNotFoundError, KeyError, OSError) as err:
        st.warning(f"Не удалось построить график: {err}. Перезапустите пайплайн.")
    st.markdown("#### Возможная причина")
    st.info(hypothesis(e))
    with st.expander("Технические детали"):
        st.markdown(views.event_details_html(e, si.to_dict() if si is not None else
                                             {"tz_known": False, "tz_offset_h": None}), unsafe_allow_html=True)
        try:
            feats = json.loads(e.get("features", "{}"))
            if isinstance(feats, dict) and feats:
                st.dataframe(pd.DataFrame({"Признак": list(feats), "Значение": list(feats.values())}),
                             hide_index=True, width="stretch")
        except (TypeError, ValueError):
            st.caption("Признаки не сохранены.")
        st.caption("Методы совпадают, если события пересекаются в одном источнике с допуском ±300 с и ±2 ед. X. "
                   "Уровень внимания — ранг внутри метода по оценке/порогу и |z| плюс число совпавших методов: "
                   f"высокий — верхние {HIGH_SHARE:.0%}, средний — следующие {MEDIUM_SHARE:.0%}.")
    with st.expander("Разметка эксперта"):
        lab = st.radio("Метка", LABEL_VALUES, horizontal=True, key=f"new_lab_{e['event_id']}")
        who = st.text_input("Эксперт", key=f"new_who_{e['event_id']}")
        comment = st.text_input("Комментарий", key=f"new_cmt_{e['event_id']}")
        if st.button("Сохранить метку", key=f"new_save_{e['event_id']}"):
            append_label(LABELS_PATH, dict(
                source_id=e["source_id"], start_time=e["start_time"].isoformat(),
                end_time=e["end_time"].isoformat(), x_start=e["x_start"], x_end=e["x_end"],
                label=lab, annotator=who, comment=comment, event_id=e["event_id"]))
            st.success("Метка сохранена, метрики обновятся после следующего запуска пайплайна.")


def _main_table(f: pd.DataFrame):
    header_cols = ["Дата и время", "Источник", "Участок", "Описание", "Внимание",
                   "Методов", "Длительность", ""]
    widths = [1.55, 0.85, 1, 3.2, 0.85, 0.65, 0.9, 1.0]
    for col, title in zip(st.columns(widths), header_cols):
        col.markdown(f"**{title}**")
    lookup = _src_lookup()
    for _, e in f.iterrows():
        cols = st.columns(widths, vertical_alignment="center")
        values = [_dt(e), _file(e, lookup), x_label(e), e["title"], e["attention"],
                  str(e["method_count"]), fmt_dur(e["duration_s"])]
        for i, (col, val) in enumerate(zip(cols, values)):
            if i == 4:
                col.markdown(f"<span style='color:{ATTENTION_COLORS[e['attention']]};font-weight:600'>"
                             f"{html.escape(val)}</span>", unsafe_allow_html=True)
            else:
                col.write(val)
        if cols[-1].button("Подробнее", key=f"open_{e['event_id']}"):
            _goto_detail(str(e["event_id"]))


def page_events():
    header("События")
    if not require_results():
        return
    st.info(NOTICE)
    mode = st.segmented_control("Режим просмотра", ["Каталог и карточка", "Карта сигналов", "Технический каталог"],
                                default="Каталог и карточка", key="ui_event_mode")
    if mode == "Карта сигналов":
        views.page_detection()
        return
    if mode != "Каталог и карточка":
        views.page_catalog()
        return
    ev = _all()
    if ev.empty:
        st.info("Событий не найдено.")
        return
    lookup = _src_lookup()
    a, b, c, d = st.columns(4)
    with a:
        sel = st.selectbox("Источник", ["all"] + list(ev["source_id"].unique()), key="ui_src",
                           format_func=lambda s: "Все источники" if s == "all" else _file({"source_id": s}, lookup))
    with b:
        lev = st.selectbox("Уровень внимания", ["Все", "Высокий", "Средний", "Низкий"], key="ui_attention")
    with c:
        split = st.selectbox("Часть выборки", ["all", "train", "validation", "test"], key="ui_split",
                             format_func=lambda s: "Все" if s == "all" else s)
    with d:
        search = st.text_input("Поиск", placeholder="участок, описание, ID", key="ui_search")
    f = ev.copy()
    if sel != "all":
        f = f[f["source_id"] == sel]
    if lev != "Все":
        f = f[f["attention"] == lev]
    if split != "all":
        f = f[f["split"] == split]
    if search.strip():
        f = f[f["title"].str.contains(search, case=False, regex=False) |
              f["event_id"].str.contains(search, case=False, regex=False)]
    f = f.sort_values(["method_count", "anomaly_score_norm"], ascending=False)
    st.caption(f"Показано {len(f)} из {len(ev)}. Одно место на волокне может встречаться "
               "несколько раз — по записи от каждого метода.")
    per_page = 12
    pages = max(1, int(np.ceil(len(f) / per_page)))
    page = st.number_input("Страница", 1, pages, value=1, key="ui_page")
    _main_table(f.iloc[(page - 1) * per_page:page * per_page])
    chosen = st.session_state.get("ui_selected_event")
    if chosen in set(ev["event_id"]):
        st.divider()
        detail_card(ev[ev["event_id"] == chosen].iloc[0])
    st.download_button("Скачать таблицу событий (CSV)",
                       f.drop(columns=["pattern_tags_list", "device_channel"], errors="ignore")
                        .to_csv(index=False).encode("utf-8-sig"),
                       file_name="events_full.csv", mime="text/csv")


def page_patterns():
    header("Закономерности")
    if not require_results():
        return
    mode = st.segmented_control("Режим", ["Повторяемость и классификация", "Пространственный обзор"],
                                default="Повторяемость и классификация", key="ui_pattern_mode")
    if mode == "Повторяемость и классификация":
        views.page_classification()
    else:
        views.page_spatial()


def page_methods():
    views.page_comparison()


def page_data_limits():
    views.page_about()
