"""Экраны приложения Fiber Analytics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import LABELS_PATH, USER_CONFIG_PATH, load_config, save_user_config
from src.evaluation import LABEL_VALUES, append_label, compare_detectors, load_labels

from . import charts
from .common import (DET_COLORS, DET_ORDER, DET_SHORT, MUTED, TYPE_COLORS, TYPE_ORDER, data_report,
                     detector_selector, events, fmt_dt, fmt_dur, fmt_num, header, kv_html, manifest,
                     metrics, period_selector, plot_layout, prepared, require_results, source_info,
                     source_label, source_selector, stat_tile, table, to_local, tz_label, x_unit_short)

PAGE_SIZE = 25


def _offset(si: dict):
    return si["tz_offset_h"] if si.get("tz_known") else None


def _local_start(ev: pd.DataFrame) -> pd.Series:
    off = ev["tz_offset_h"].astype(float).fillna(0.0)
    return (ev["start_time"] + pd.to_timedelta(off, unit="h")).dt.tz_localize(None)


def display_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Удобочитаемое представление событий для таблиц и экспорта."""
    loc = _local_start(ev)
    return pd.DataFrame({
        "ID": ev["event_id"],
        "Дата и время": loc.dt.strftime("%d.%m.%Y %H:%M:%S"),
        "Пояс": np.where(ev["tz_known"], ev["tz_offset_h"].map(
            lambda o: f"UTC{'+' if (o or 0) >= 0 else '−'}{abs(o or 0):02.0f}"), "не указан"),
        "Прибор / канал": ev["dts_id"].str[:8] + " / " + ev["channel_id"].astype(str),
        "Кабель": ev["cable_id"].fillna("не сопоставлен"),
        "Участок x": ev["x_start"].round(1).astype(str) + "–" + ev["x_end"].round(1).astype(str),
        "Тип события": ev["event_type"],
        "Алгоритм": ev["detector"].map(DET_SHORT),
        "Оценка/порог": ev["anomaly_score_norm"].round(2),
        "Длительность": ev["duration_s"].map(fmt_dur),
        "Амплитуда": ev["amplitude"].round(2),
        "Повторы": ev["repeat_count"],
        "Часть": ev["split"],
    })


def export_frame(ev: pd.DataFrame) -> bytes:
    out = ev.drop(columns=[c for c in ("pattern_tags_list", "device_channel") if c in ev.columns]).copy()
    for c in ("start_time", "end_time", "peak_time"):
        out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.insert(1, "start_time_local", _local_start(ev).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy())
    return out.to_csv(index=False).encode("utf-8-sig")


def event_details_html(e: pd.Series, si: dict) -> str:
    off = _offset(si)
    tags = e["pattern_tags_list"] if isinstance(e.get("pattern_tags_list"), list) else json.loads(e["pattern_tags"])
    unit = x_unit_short(e["x_unit"])
    pairs = [
        ("ID события", f"<b>{e['event_id']}</b>"),
        ("Дата", fmt_dt(e["start_time"], off, "%d.%m.%Y")),
        ("Начало / конец", f"{fmt_dt(e['start_time'], off, '%H:%M:%S')} – {fmt_dt(e['end_time'], off, '%H:%M:%S')}"
                           f" <span style='color:{MUTED}'>({tz_label(si)})</span>"),
        ("Прибор / канал", f"{e['dts_id'][:8]}… / {e['channel_id']}"),
        ("Кабель", "<span class='fa-badge gray'>кабель не сопоставлен</span>"),
        ("Участок", f"{e['x_start']:.1f} – {e['x_end']:.1f} {unit} (пик {e['x_peak']:.1f})"),
        ("Оценка аномальности", f"{e['anomaly_score']:.3f} · {e['anomaly_score_norm']:.2f}× порога"),
        ("Тип события", f"<span class='fa-badge'>{e['event_type']}</span>"),
        ("Признаки", f"{e['duration_class']}, {e['extent_class']}, {e['shape']}, {e['repetition_class']}"),
        ("Длительность", fmt_dur(e["duration_s"])),
        ("Амплитуда", f"{e['amplitude']:+.2f} ({e['amplitude_z']:+.1f}σ), {e['direction']}"),
        ("Последействие", {True: "уровень сохраняется", False: "возврат к исходному",
                           None: "не определено"}.get(e["persistent"] if pd.notna(e["persistent"]) else None)),
        ("Алгоритм", e["detector_label"]),
        ("Достоверность типа", e["type_confidence"]),
        ("Закономерности", "<br>".join(tags) if tags else "—"),
    ]
    return kv_html(pairs)


def _pick_event(ev: pd.DataFrame, key: str) -> str | None:
    sel = st.session_state.get(key)
    if sel in set(ev["event_id"]):
        return sel
    if ev.empty:
        return None
    return ev.sort_values("anomaly_score_norm", ascending=False)["event_id"].iloc[0]


def _on_chart_select(chart_key: str, target: str):
    def cb():
        state = st.session_state.get(chart_key)
        pts = (state or {}).get("selection", {}).get("points", []) if isinstance(state, dict) else \
            getattr(getattr(state, "selection", None), "points", []) or []
        for p in pts:
            cd = p.get("customdata")
            if cd:
                st.session_state[target] = cd[0] if isinstance(cd, list) else cd
                return
    return cb


def _on_table_select(table_key: str, ids: list, target: str):
    def cb():
        state = st.session_state.get(table_key)
        rows = state["selection"]["rows"] if state else []
        if rows:
            st.session_state[target] = ids[rows[0]]
    return cb


def page_detection():
    header("Обнаружение событий", "Сигнал вдоль волокна во времени и найденные аномалии по выбранному источнику")
    if not require_results():
        return
    c1, c2, c3, c4 = st.columns([2.3, 2.7, 1.5, 0.9], vertical_alignment="bottom")
    with c1:
        si = source_selector("det_src")
    sid, off = si["source_id"], _offset(si)
    prep = prepared(sid)
    with c2:
        p0, p1 = period_selector(prep["times"], off, key=f"det_period_{sid}")
    with c3:
        det = detector_selector("det_det")
    with c4:
        if st.button("Обновить", type="primary", width="stretch",
                     help="Перечитать результаты последнего расчёта"):
            st.cache_data.clear()
            st.rerun()
    tloc = to_local(prep["times"], off)
    tmask = np.asarray((tloc >= pd.Timestamp(p0)) & (tloc <= pd.Timestamp(p1)))
    ev = events()
    ev = ev[(ev["source_id"] == sid) & (ev["detector"] == det)]
    ls = _local_start(ev)
    ev = ev[(ls >= pd.Timestamp(p0)) & (ls <= pd.Timestamp(p1))]

    o1, o2, o3 = st.columns([2.4, 1.4, 4], vertical_alignment="center")
    with o1:
        mode = st.segmented_control("Слой", ["value", "z"], default="value", key="det_mode",
                                    format_func=lambda m: "Исходный сигнал" if m == "value"
                                    else "Отклонение от нормы (σ)", label_visibility="collapsed") or "value"
    with o2:
        show_ev = st.toggle("Показать аномалии", value=True, key="det_show")
    with o3:
        st.markdown(f"<div class='fa-note'>Событий в периоде: <b>{len(ev)}</b> · шаг опроса "
                    f"{fmt_dur(si['dt_median_s'])} · бин {si['bin_points']} точ. · время {tz_label(si)}"
                    f"{' · x — индекс точки, единицы не подтверждены' if not str(si['x_unit']).startswith('м') else ''}"
                    "</div>", unsafe_allow_html=True)

    sel = _pick_event(ev, "sel_event")
    left, right = st.columns([2.35, 1])
    with left, st.container(border=True):
        st.markdown("#### Сигнал вдоль волокна во времени")
        if tmask.sum() < 2:
            st.info("В выбранном периоде меньше двух измерений — тепловая карта не строится.")
        else:
            fig = charts.heatmap(prep, tmask, off, mode, si["x_unit"], ev if show_ev else None, sel)
            key = f"hm_{sid}_{det}"
            st.plotly_chart(fig, width="stretch", key=key, on_select=_on_chart_select(key, "sel_event"),
                            selection_mode="points", config={"displaylogo": False})
            st.caption("Нажмите на маркер события, чтобы открыть детали. Подписаны 15 событий с наибольшей "
                       "оценкой. Отображение прореживается блочным средним до ≤700×450 ячеек; расчёты "
                       "выполнены на полном разрешении бинов.")
    with right, st.container(border=True):
        st.markdown("#### Детали события")
        if sel is None:
            st.info("В выбранном периоде этот алгоритм событий не нашёл.")
        else:
            e = ev[ev["event_id"] == sel].iloc[0]
            st.markdown(event_details_html(e, si), unsafe_allow_html=True)
            st.markdown("##### Фрагмент сигнала")
            st.plotly_chart(charts.fragment_chart(prep, e, off), width="stretch",
                            config={"displayModeBar": False})
            st.caption("Среднее значение по участку события; красная полоса — интервал события.")

    if sel is not None:
        e = ev[ev["event_id"] == sel].iloc[0]
        with st.container(border=True):
            st.markdown("#### Профиль вдоль волокна в момент пика")
            tpk = int(np.argmin(np.abs(prep["times"] - e["peak_time"])))
            st.plotly_chart(charts.profile_chart(prep, tpk, off, si["x_unit"], e), width="stretch",
                            config={"displaylogo": False})

    with st.container(border=True):
        st.markdown("#### События в выбранном периоде")
        if len(ev):
            evs = ev.sort_values("start_time")
            ids = evs["event_id"].tolist()
            tkey = f"det_tbl_{sid}_{det}"
            st.dataframe(display_table(evs), hide_index=True, width="stretch", height=280,
                         on_select=_on_table_select(tkey, ids, "sel_event"), selection_mode="single-row",
                         key=tkey)
        else:
            st.caption("Нет событий.")


def page_catalog():
    header("Каталог событий", "Обнаруженные события всех алгоритмов: фильтры, карточка события, повторы и синхронность")
    if not require_results():
        return
    ev = events()
    si_all = source_info()
    loc = _local_start(ev)
    f1, f2, f3, f4, f5 = st.columns([2, 1.6, 1.6, 1.8, 1.4])
    with f1:
        d0, d1 = loc.min().date(), loc.max().date()
        rng = st.date_input("Период (локальные даты)", value=(d0, d1), min_value=d0, max_value=d1,
                            format="DD.MM.YYYY", key="cat_period")
    with f2:
        srcs = st.multiselect("Источник", si_all["source_id"].tolist(), key="cat_src",
                              format_func=lambda s: f"Файл {s}.csv", placeholder="Все")
    with f3:
        dets = st.multiselect("Алгоритм", DET_ORDER, format_func=DET_SHORT.get, key="cat_det",
                              placeholder="Все")
    with f4:
        types = st.multiselect("Тип события", TYPE_ORDER, key="cat_type", placeholder="Все")
    with f5:
        q = st.text_input("Поиск по ID", key="cat_q", placeholder="например A-4-0006")
    g1, g2, g3, g4 = st.columns([1.6, 1.6, 1.6, 3])
    with g1:
        split = st.selectbox("Часть данных", ["все", "train", "validation", "test"], key="cat_split")
    with g2:
        sort = st.selectbox("Сортировка", ["Оценка ↓", "Время ↑", "Время ↓", "Длительность ↓", "Амплитуда |↓|"],
                            key="cat_sort")
    with g3:
        rep = st.selectbox("Повторяемость", ["все", "повторяющиеся", "единичные", "синхронные"], key="cat_rep")

    m = pd.Series(True, index=ev.index)
    if isinstance(rng, tuple) and len(rng) == 2:
        m &= (loc.dt.date >= rng[0]) & (loc.dt.date <= rng[1])
    if srcs:
        m &= ev["source_id"].isin(srcs)
    if dets:
        m &= ev["detector"].isin(dets)
    if types:
        m &= ev["event_type"].isin(types)
    if q:
        m &= ev["event_id"].str.contains(q.strip(), case=False, regex=False)
    if split != "все":
        m &= ev["split"] == split
    if rep == "повторяющиеся":
        m &= ev["repeat_count"] >= 2
    elif rep == "единичные":
        m &= ev["repeat_count"] < 2
    elif rep == "синхронные":
        m &= ev["sync_size"] >= 2
    f = ev[m]
    key_sort = {"Оценка ↓": ("anomaly_score_norm", False), "Время ↑": ("start_time", True),
                "Время ↓": ("start_time", False), "Длительность ↓": ("duration_s", False)}
    if sort == "Амплитуда |↓|":
        f = f.reindex(f["amplitude"].abs().sort_values(ascending=False).index)
    else:
        c, asc = key_sort[sort]
        f = f.sort_values(c, ascending=asc, kind="stable")

    k = st.columns(5)
    k[0].markdown(stat_tile("Событий", f"{len(f):,}".replace(",", " "), f"из {len(ev)} всего"), unsafe_allow_html=True)
    for i, d in enumerate(DET_ORDER, start=1):
        k[i].markdown(stat_tile(DET_SHORT[d], str(int((f["detector"] == d).sum())), "событий"), unsafe_allow_html=True)
    k[4].markdown(stat_tile("Повторяющихся", str(int((f["repeat_count"] >= 2).sum())),
                            f"синхронных: {int((f['sync_size'] >= 2).sum())}"), unsafe_allow_html=True)

    st.write("")
    n_pages = max(1, int(np.ceil(len(f) / PAGE_SIZE)))
    pc1, pc2, pc3 = st.columns([1.2, 4, 1.6], vertical_alignment="bottom")
    with pc1:
        page = st.number_input("Страница", 1, n_pages, 1, key="cat_page")
    with pc2:
        st.caption(f"Показано {min(len(f), (page - 1) * PAGE_SIZE + 1)}–{min(len(f), page * PAGE_SIZE)} "
                   f"из {len(f)} событий · страниц: {n_pages}")
    with pc3:
        st.download_button("⬇ Экспорт CSV (фильтр)", export_frame(f), file_name="events_filtered.csv",
                           mime="text/csv", width="stretch", disabled=f.empty)
    pg = f.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    ids = pg["event_id"].tolist()
    tkey = f"cat_tbl_{page}_{hash((tuple(srcs), tuple(dets), tuple(types), q, split, sort, rep)) % 10**8}"
    st.dataframe(display_table(pg), hide_index=True, width="stretch", height=min(940, 38 + 35 * len(pg)),
                 on_select=_on_table_select(tkey, ids, "cat_sel"), selection_mode="single-row", key=tkey)
    st.caption("Выберите строку, чтобы открыть карточку события.")

    sel = st.session_state.get("cat_sel")
    if sel not in set(ev["event_id"]):
        return
    event_card(ev, ev[ev["event_id"] == sel].iloc[0])


def event_card(ev_all: pd.DataFrame, e: pd.Series):
    si = source_info().set_index("source_id").loc[e["source_id"]].to_dict()
    si["source_id"] = e["source_id"]
    off = _offset(si)
    prep = prepared(e["source_id"])
    st.markdown(f"### Карточка события {e['event_id']}")
    a, b = st.columns([1, 1.6])
    with a:
        with st.container(border=True):
            st.markdown(event_details_html(e, si), unsafe_allow_html=True)
        feats = json.loads(e["features"])
        with st.expander("Признаки окна (максимум |значения| по событию)"):
            st.dataframe(pd.DataFrame({"признак": list(feats), "значение": [fmt_num(v, 3) for v in feats.values()]}),
                         hide_index=True, width="stretch")
        with st.expander("Разметка эксперта для этого события"):
            lab = st.radio("Метка", LABEL_VALUES, horizontal=True, key=f"lab_{e['event_id']}")
            who = st.text_input("Эксперт", key=f"who_{e['event_id']}")
            cmt = st.text_input("Комментарий", key=f"cmt_{e['event_id']}")
            if st.button("Сохранить метку", key=f"save_{e['event_id']}"):
                append_label(LABELS_PATH, dict(
                    source_id=e["source_id"], start_time=e["start_time"].isoformat(),
                    end_time=e["end_time"].isoformat(), x_start=e["x_start"], x_end=e["x_end"],
                    label=lab, annotator=who, comment=cmt, event_id=e["event_id"]))
                st.success(f"Метка сохранена в {LABELS_PATH.name}. P/R/F1 пересчитываются при следующем "
                           "запуске пайплайна. Метки, поставленные по найденным событиям, оценивают точность; "
                           "для полноты нужны и независимо размеченные участки.")
    with b:
        st.markdown("#### Контекст сигнала")
        t0, t1 = int(e["t0"]), int(e["t1"])
        tsec = prep["t_sec"]
        span = max(tsec[t1] - tsec[t0], 1800.0)
        tm = (tsec >= tsec[t0] - span) & (tsec <= tsec[t1] + span)
        neigh = ev_all[(ev_all["source_id"] == e["source_id"]) & (ev_all["detector"] == e["detector"])]
        fig = charts.heatmap(prep, tm, off, "z", si["x_unit"], neigh, e["event_id"], height=330)
        pad = max(20.0, 3 * (e["x_end"] - e["x_start"]))
        fig.update_yaxes(range=[e["x_start"] - pad, e["x_end"] + pad])
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.plotly_chart(charts.fragment_chart(prep, e, off, height=200), width="stretch",
                        config={"displayModeBar": False})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Похожие события на том же участке**")
        rep = ev_all[(ev_all["repeat_group"] == e["repeat_group"]) & (ev_all["event_id"] != e["event_id"])]
        if rep.empty:
            st.caption("Не встречалось в наблюдаемом периоде (у этого алгоритма).")
        else:
            st.dataframe(display_table(rep)[["ID", "Дата и время", "Участок x", "Тип события"]],
                         hide_index=True, width="stretch", height=210)
            g = table("repeat_groups")
            if not g.empty:
                r = g[g["repeat_group"] == e["repeat_group"]]
                if len(r) and pd.notna(r["interval_mean_s"].iloc[0]):
                    cv = r["interval_cv"].iloc[0]
                    st.caption(f"Средний интервал {fmt_dur(r['interval_mean_s'].iloc[0])}"
                               + (f", CV {cv:.2f}" if pd.notna(cv) else "")
                               + (" — возможно периодическое" if bool(r["periodic"].iloc[0]) else ""))
    with c2:
        st.markdown("**Синхронные события того же канала**")
        if e["sync_group"] < 0:
            st.caption("Синхронных событий на других участках не найдено.")
        else:
            sy = ev_all[(ev_all["sync_group"] == e["sync_group"]) & (ev_all["event_id"] != e["event_id"])]
            st.dataframe(display_table(sy)[["ID", "Дата и время", "Участок x", "Тип события"]],
                         hide_index=True, width="stretch", height=210)
    with c3:
        st.markdown("**Тот же участок у других алгоритмов**")
        o = ev_all[(ev_all["source_id"] == e["source_id"]) & (ev_all["detector"] != e["detector"])]
        tol = pd.Timedelta(seconds=300)
        o = o[(o["start_time"] <= e["end_time"] + tol) & (o["end_time"] >= e["start_time"] - tol)
              & (o["x_start"] <= e["x_end"] + 2) & (o["x_end"] >= e["x_start"] - 2)]
        if o.empty:
            st.caption("Другие алгоритмы это событие не нашли (уникальная находка).")
        else:
            st.dataframe(display_table(o)[["ID", "Алгоритм", "Участок x", "Оценка/порог"]],
                         hide_index=True, width="stretch", height=210)


def page_classification():
    header("Классификация", "Описательные категории событий, кластеры и закономерности — без физических причин")
    if not require_results():
        return
    ev = events()
    cfg = (manifest() or {}).get("config", load_config())
    ec = cfg["events"]
    dsel = st.multiselect("Алгоритмы", DET_ORDER, default=DET_ORDER, format_func=DET_SHORT.get, key="cls_det")
    ev = ev[ev["detector"].isin(dsel)]
    t1, t2, t3 = st.tabs(["Категории", "Кластеры (DBSCAN)", "Повторяемость и периодичность"])
    with t1:
        a, b = st.columns([1.3, 1])
        with a:
            st.markdown("**Типы событий по алгоритмам**")
            st.plotly_chart(charts.type_bars(ev), width="stretch")
        with b:
            st.markdown("**Правила категорий (прозрачные, настраиваемые)**")
            st.dataframe(pd.DataFrame([
                ("кратковременное / длительное", f"длительность ≤ {ec['short_max_s']:.0f} с"),
                ("локальное / распределённое", f"ширина ≤ {ec['local_max_x']} ед. x"),
                ("резкое / плавное", f"за один кадр пройдено ≥ {ec['abrupt_rise_frac']:.0%} пути от фона до пика "
                                     f"(если путь ≥ {ec['step_persist_z']} шумов, иначе «не определено»)"),
                ("сохраняется / возврат", f"через {ec['context_window_min']:.0f} мин уровень отличается от исходного "
                                          f"на ≥ {ec['step_persist_z']} шумов"),
                ("единичное / повторяющееся", "есть ли события того же алгоритма с пиком в пределах ±2 ед. x"),
                ("Резкий импульс", "резкое и кратковременное, без последействия"),
                ("Ступенчатое изменение", "резкое и уровень сохраняется"),
                ("Длительное отклонение", "длительное, не ступенька"),
                ("Кратковременное плавное", "кратковременное, не резкое"),
            ], columns=["категория", "правило"]), hide_index=True, width="stretch", height=360)
            st.caption("Категории описывают форму сигнала, а не причину. Уровень участка — среднее бинов события "
                       "минус среднее соседних бинов.")
        cats = []
        for col, title in (("duration_class", "Длительность"), ("extent_class", "Протяжённость"),
                           ("shape", "Форма"), ("repetition_class", "Повторяемость"), ("direction", "Направление")):
            c = ev.groupby([col, "detector"]).size().rename("n").reset_index()
            c["группа"] = title
            c = c.rename(columns={col: "категория"})
            cats.append(c)
        cat = pd.concat(cats)
        st.markdown("**Независимые признаки событий**")
        cols = st.columns(5)
        for i, (title, g) in enumerate(cat.groupby("группа", sort=False)):
            with cols[i]:
                st.caption(title)
                g = g.assign(detector=g["detector"].map(DET_SHORT))
                fig = charts.grouped_bars(g, "категория", "detector", "n",
                                          {DET_SHORT[k]: v for k, v in DET_COLORS.items()},
                                          order_series=[DET_SHORT[d] for d in dsel], height=260, text=False)
                fig.update_layout(showlegend=(i == 0))
                st.plotly_chart(fig, width="stretch", key=f"cat_{i}")
    with t2:
        prof = table("cluster_profile")
        st.markdown("Кластеры строятся DBSCAN по признакам: log длительности, log ширины, log |амплитуды|, "
                    "знак, log(оценка/порог), резкость, последействие (стандартизованы). "
                    "Номер кластера — только идентификатор группы похожих по форме событий; "
                    "**физический смысл кластерам не присваивается**. −1 — не отнесены ни к одной группе.")
        a, b = st.columns([1.5, 1])
        with a:
            fig = go.Figure()
            top = ev["cluster"].value_counts().index[:8].tolist()
            for cl in sorted(set(ev["cluster"])):
                e = ev[ev["cluster"] == cl]
                name = "не в кластере (−1)" if cl == -1 else (f"кластер {cl}" if cl in top else "прочие кластеры")
                fig.add_trace(go.Scattergl(
                    x=e["duration_s"].clip(lower=1), y=e["amplitude"].abs().clip(lower=0.01), mode="markers",
                    name=name, legendgroup=name, showlegend=name not in [t.name for t in fig.data],
                    marker=dict(size=7, color=MUTED if cl == -1 else (
                        list(TYPE_COLORS.values())[top.index(cl) % 4] if cl in top[:4] else
                        ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"][top.index(cl) % 4] if cl in top else "#c3c2b7"),
                        line=dict(color="#fff", width=1)),
                    customdata=e[["event_id", "event_type"]].to_numpy(),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>длит. %{x:.0f} с, |A| %{y:.2f}"
                                  "<extra>" + name + "</extra>"))
            plot_layout(fig, height=430)
            fig.update_xaxes(type="log", title="длительность, с (лог.)")
            fig.update_yaxes(type="log", title="|амплитуда| (лог.)")
            st.plotly_chart(fig, width="stretch")
        with b:
            if prof.empty:
                st.info("Событий недостаточно для кластеризации.")
            else:
                st.dataframe(prof.rename(columns={"cluster": "кластер", "n": "событий",
                                                  "duration_s_median": "длит. медиана, с",
                                                  "width_x_median": "ширина медиана", "amplitude_median": "амплитуда медиана",
                                                  "abrupt_share": "доля резких", "persistent_share": "доля с последействием"}
                                         ).round(2), hide_index=True, width="stretch", height=430)
    with t3:
        rg = table("repeat_groups")
        ac = table("autocorr")
        cs = table("cross_source")
        if not rg.empty:
            rg = rg[rg["detector"].isin(dsel)]
            a, b, c = st.columns(3)
            a.markdown(stat_tile("Групп повторений (≥2 событий)", str(int((rg["n_events"] >= 2).sum()))),
                       unsafe_allow_html=True)
            b.markdown(stat_tile("Возможно периодических", str(int(rg["periodic"].sum())),
                                 "≥4 повторов и CV интервалов ≤ 0.3"), unsafe_allow_html=True)
            c.markdown(stat_tile("Совпадений между приборами", str(len(cs)),
                                 "часы приборов не сверены"), unsafe_allow_html=True)
            st.markdown("**Группы повторений** (одинаковый источник, алгоритм, направление, пик в пределах ±2 ед. x)")
            show = rg[rg["n_events"] >= 2].sort_values("n_events", ascending=False).copy()
            show["detector"] = show["detector"].map(DET_SHORT)
            show["interval_mean"] = show["interval_mean_s"].map(fmt_dur)
            st.dataframe(show[["repeat_group", "source_id", "detector", "n_events", "x_min", "x_max",
                               "interval_mean", "interval_cv", "periodic", "event_ids"]].round(2),
                         hide_index=True, width="stretch", height=320)
        st.markdown("**Автокорреляция числа событий (интервалы 10 мин, лаги ≥ 30 мин)**")
        if not ac.empty:
            ac = ac[ac["detector"].isin(dsel)].copy()
            ac["detector"] = ac["detector"].map(DET_SHORT)
            ac["best_lag"] = ac["best_lag_s"].map(fmt_dur)
            st.dataframe(ac[["source_id", "detector", "n_events", "best_lag", "acf", "significant", "note"]].round(3),
                         hide_index=True, width="stretch")
            st.caption("Значимый пик автокорреляции — повод проверить периодичность вручную, а не доказательство: "
                       "ряды короткие (9–48 ч), а события кучкуются во времени.")
        st.markdown("**Совпадения по времени между разными приборами**")
        if cs.empty:
            st.caption("Не найдено. По времени пересекаются только файлы 3 (cd477c43 / к2) и 6 (5f895b10 / к1) — "
                       "12.02.2026; совпадающих в пределах окна синхронности событий между ними нет. "
                       "Синхронизация часов разных приборов в любом случае не подтверждена.")
        else:
            st.dataframe(cs, hide_index=True, width="stretch")


def _scope_meta(m: dict, scope_split: str):
    runs = pd.DataFrame(m["detector_runs"])
    if scope_split == "all":
        runs["flagged_windows_eval"] = runs["flagged_windows_all"]
        si = source_info()
        hours = {r.source_id: (pd.Timestamp(r.time_end) - pd.Timestamp(r.time_start)).total_seconds() / 3600
                 for r in si.itertuples()}
    else:
        hours = {r["source_id"]: r["test_hours"] for r in data_report()["source_info"]}
    return runs, hours


def page_comparison():
    header("Сравнение алгоритмов", "Одинаковый период, одинаковые правила объединения срабатываний в события")
    if not require_results():
        return
    m = metrics()
    ev = events()
    cfg = (manifest() or {}).get("config", load_config())
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        si = source_info()
        src = st.selectbox("Источник", ["all"] + si["source_id"].tolist(), key="cmp_src",
                           format_func=lambda s: "Все источники" if s == "all" else
                           source_label(si.set_index("source_id").loc[s].to_dict() | {"source_id": s}))
    with c2:
        scope = st.selectbox("Период", ["test", "all"], key="cmp_scope",
                             format_func=lambda s: "Тестовая часть (последние 20 % каждого источника)"
                             if s == "test" else "Весь период")
    runs, hours = _scope_meta(m, scope)
    sources = None if src == "all" else [src]
    cmp = compare_detectors(ev, runs, hours, cfg, split="test" if scope == "test" else None, sources=sources)
    tab = pd.DataFrame(cmp["table"]).set_index("detector")
    with c3:
        st.markdown(f"<div class='fa-note' style='margin-top:1.8rem'>Наблюдаемое время в выборке: "
                    f"<b>{cmp['hours']:.1f} ч</b>. Совпадение событий: тот же источник, пересечение по времени "
                    f"±{cfg['evaluation']['match_time_tol_s']:.0f} с и по координате ±{cfg['evaluation']['match_x_tol']} ед.</div>",
                    unsafe_allow_html=True)

    labels = m.get("labels", {})
    present = [d for d in DET_ORDER if d in tab.index]
    cols = st.columns(len(present))
    for col, d in zip(cols, present):
        r = tab.loc[d]
        rr = runs[runs["detector"] == d] if sources is None else runs[(runs["detector"] == d) & runs["source_id"].isin(sources)]
        thr = rr["threshold"]
        mode = rr["threshold_mode"].iloc[0]
        thr_txt = f"{thr.iloc[0]:.2f}σ (фикс.)" if mode == "fixed" else f"{thr.min():.3f}–{thr.max():.3f} (квантиль)"
        lp = None
        if labels.get("available"):
            lp = next((x for x in labels["per_detector"] if x["detector"] == d), None)
        prf = (f"{fmt_num(lp['precision'])} / {fmt_num(lp['recall'])} / {fmt_num(lp['f1'])}" if lp
               else "<span class='fa-note'>нет разметки — не вычисляются</span>")
        col.markdown(
            f"<div class='fa-det'><div class='name'><span class='fa-dot' style='background:{DET_COLORS[d]}'></span>"
            f"{m['detector_labels'][d]}</div>" + kv_html([
                ("Найдено событий", f"<b>{int(r['events'])}</b>"),
                ("Событий в час", fmt_num(r["events_per_hour"], 2)),
                ("Окон-срабатываний", f"{int(r['flagged_windows']):,} из {int(r['windows']):,}".replace(",", " ")),
                ("Совпали с другими", str(int(r["matched_with_other"]))),
                ("Уникальные находки", str(int(r["unique"]))),
                ("Найдены всеми методами", str(int(r["consensus_all"])) if pd.notna(r["consensus_all"]) else "—"),
                ("Порог", thr_txt),
                ("Срабатываний на validation", f"{r['validation_alarm_rate']:.2%} окон"),
                ("Время (обучение + оценка)", f"{r['fit_time_s'] + r['score_time_s']:.2f} с"),
                ("Пик памяти", f"{r['peak_mem_mb']:.0f} МБ"),
                ("Precision / Recall / F1", prf),
            ]) + "</div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs(["По метрикам", "По типам событий", "Наложение событий", "Примеры находок",
                                      "Регрессия: MAE / RMSE", "Синтетический тест"])
    with t1:
        a, b = st.columns(2)
        long = []
        for d in DET_ORDER:
            if d in tab.index:
                for k, lbl in (("events", "найдено"), ("matched_with_other", "совпали с другими"),
                               ("unique", "уникальные")):
                    long.append(dict(metric=lbl, detector=DET_SHORT[d], n=int(tab.loc[d, k])))
        with a:
            st.markdown("**События: найдено, совпадения, уникальные**")
            st.plotly_chart(charts.grouped_bars(pd.DataFrame(long), "metric", "detector", "n",
                                                {DET_SHORT[k]: v for k, v in DET_COLORS.items()},
                                                order_series=[DET_SHORT[d] for d in present]),
                            width="stretch")
        with b:
            st.markdown("**Попарное совпадение событий**")
            ov = pd.DataFrame(cmp["overlap"])
            if not ov.empty:
                ov["пара"] = ov["detector_a"].map(DET_SHORT) + " ↔ " + ov["detector_b"].map(DET_SHORT)
                ov["A: совпало"] = ov["a_matched"].astype(str) + " из " + ov["a_total"].astype(str)
                ov["B: совпало"] = ov["b_matched"].astype(str) + " из " + ov["b_total"].astype(str)
                st.dataframe(ov[["пара", "A: совпало", "B: совпало"]], hide_index=True, width="stretch")
            wa = pd.DataFrame(m.get("window_agreement_test", []))
            if not wa.empty and scope == "test" and src == "all":
                wa["пара"] = wa["detector_a"].map(DET_SHORT) + " ↔ " + wa["detector_b"].map(DET_SHORT)
                st.markdown("**Совпадение на уровне окон (Жаккар, test)**")
                st.dataframe(wa[["пара", "intersection", "union", "jaccard"]].round(3), hide_index=True,
                             width="stretch")
            st.markdown("**Распределение оценок событий (оценка / порог)**")
            ev_s = ev if sources is None else ev[ev["source_id"].isin(sources)]
            if scope == "test":
                ev_s = ev_s[ev_s["split"] == "test"]
            st.plotly_chart(charts.score_hist(ev_s, height=220), width="stretch")
        if not labels.get("available"):
            st.info(f"Precision / Recall / F1 не вычисляются: {labels.get('reason', 'нет разметки')}. "
                    f"Разметку можно добавить в карточке события (Каталог) или в {LABELS_PATH.relative_to(LABELS_PATH.parents[2])}; "
                    "после этого перезапустите пайплайн.")
        else:
            st.markdown(f"**Метрики по экспертной разметке** (меток: {labels['n_labels']}, аномалий: "
                        f"{labels['n_anomaly']}, норм: {labels['n_normal']}, неопределённо: {labels['n_uncertain']})")
            st.dataframe(pd.DataFrame(labels["per_detector"]).round(3), hide_index=True, width="stretch")
    ev_scope = ev if sources is None else ev[ev["source_id"].isin(sources)]
    if scope == "test":
        ev_scope = ev_scope[ev_scope["split"] == "test"]
    with t2:
        st.plotly_chart(charts.type_bars(ev_scope, height=340), width="stretch")
    with t3:
        if src == "all":
            st.info("Выберите один источник, чтобы наложить события всех алгоритмов на общую ось времени и координаты.")
        else:
            si1 = source_info().set_index("source_id").loc[src].to_dict()
            off = si1["tz_offset_h"] if si1["tz_known"] else None
            st.markdown("**Временная шкала событий**")
            st.plotly_chart(charts.timeline(ev_scope, off), width="stretch")
            st.markdown("**События в координатах «время × расстояние»**")
            st.plotly_chart(charts.space_time_overlay(ev_scope, off, si1["x_unit"]), width="stretch")
    with t4:
        cols = st.columns(len(present))
        for col, d in zip(cols, present):
            with col:
                st.markdown(f"**{DET_SHORT[d]}: уникальные находки**")
                u = ev_scope[ev_scope["event_id"].isin(cmp["unique_ids"].get(d, []))]
                u = u.sort_values("anomaly_score_norm", ascending=False).head(10)
                if u.empty:
                    st.caption("Нет уникальных находок.")
                else:
                    st.dataframe(display_table(u)[["ID", "Участок x", "Тип события", "Оценка/порог"]],
                                 hide_index=True, width="stretch")
        st.caption("Пропуски можно показать только при наличии разметки (или в синтетическом тесте). "
                   "Уникальная находка — не обязательно ложная тревога; проверяйте её в карточке события.")
    with t5:
        reg = m.get("regression", {})
        rows = []
        for s, r in reg.items():
            for part, lbl in (("test_all", "все test-окна"), ("test_quasi_normal", "квази-нормальные (|z|<3)")):
                if r.get(part):
                    rows.append(dict(источник=s, выборка=lbl, n=r[part]["n"], MAE=r[part]["mae"], RMSE=r[part]["rmse"],
                                     **{"MAE «как в прошлом кадре»": r[part]["mae_persistence"],
                                        "RMSE «как в прошлом кадре»": r[part]["rmse_persistence"]}))
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        st.caption("Ошибка прогноза значения бина на следующем кадре, в единицах сигнала (по всей видимости °C). "
                   "«Квази-нормальные» окна выбраны по |z|<3 — это не экспертная разметка.")
        coef = pd.DataFrame({s: r.get("coef", {}) for s, r in reg.items()}).T
        if not coef.empty:
            st.markdown("**Коэффициенты Ridge по источникам**")
            st.dataframe(coef.round(3), width="stretch")
    with t6:
        syn = m.get("synthetic")
        if not syn or syn.get("error"):
            st.info(f"Синтетический тест не выполнялся. {syn.get('error', '') if syn else ''}")
        else:
            st.warning(syn["note"])
            st.markdown(f"Источник: файл {syn['source_id']}.csv · внесено {syn['n_injected']} отклонений в test-часть · "
                        f"шум бина (медиана σ) {syn['noise']:.3f}. Амплитуда — в σ шума; порог A и C = "
                        f"{cfg['detectors']['statistical']['min_threshold']}σ.")
            rows = []
            for r in syn["per_detector"]:
                row = dict(алгоритм=DET_SHORT[r["detector"]], найдено=f"{r['found']} из {r['injected']}",
                           recall=r["recall"], **{"событий в test": r["test_events"]})
                for bname, bv in r["recall_by_amplitude"].items():
                    row[f"recall {bname} (n={bv['n']})"] = bv["recall"]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.download_button("⬇ Экспорт таблицы сравнения (CSV)",
                       tab.reset_index().to_csv(index=False).encode("utf-8-sig"), "comparison.csv", "text/csv")


def page_spatial():
    header("Пространственный обзор", "Профиль сигнала и плотность событий вдоль волокна (без географической привязки)")
    if not require_results():
        return
    st.info("Географических координат кабелей в данных нет, соответствие «прибор/канал → кабель» не предоставлено, "
            "поэтому карта местности не строится. Ось — расстояние вдоль волокна.")
    c1, _ = st.columns([2.5, 3])
    with c1:
        si = source_selector("sp_src")
    prep = prepared(si["source_id"])
    xc = (prep["bin_x_start"] + prep["bin_x_end"]) / 2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(prep["vb"], [5, 50, 95], axis=0)
        mx = np.nanmax(prep["vb"], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xc, y=q[2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xc, y=q[0], fill="tonexty", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             name="5–95 перцентиль во времени"))
    fig.add_trace(go.Scatter(x=xc, y=q[1], line=dict(color="#2a78d6", width=2), name="медиана"))
    fig.add_trace(go.Scatter(x=xc, y=mx, line=dict(color=MUTED, width=1, dash="dot"), name="максимум"))
    plot_layout(fig, height=320)
    fig.update_xaxes(title=f"координата, {x_unit_short(si['x_unit'])}")
    fig.update_yaxes(title="значение")
    st.markdown("**Профиль сигнала вдоль волокна за весь период**")
    st.plotly_chart(fig, width="stretch")
    ev = events()
    ev = ev[ev["source_id"] == si["source_id"]]
    fig2 = go.Figure()
    bins = np.linspace(xc[0], xc[-1], 80)
    for d in DET_ORDER:
        e = ev[ev["detector"] == d]
        h, _ = np.histogram(e["x_peak"], bins=bins)
        fig2.add_trace(go.Bar(x=(bins[:-1] + bins[1:]) / 2, y=h, name=DET_SHORT[d],
                              marker=dict(color=DET_COLORS[d], line=dict(color="#fff", width=1))))
    plot_layout(fig2, height=280)
    fig2.update_layout(barmode="group", bargap=0.1)
    fig2.update_xaxes(title=f"координата пика, {x_unit_short(si['x_unit'])}")
    fig2.update_yaxes(title="событий")
    st.markdown("**Плотность событий вдоль волокна**")
    st.plotly_chart(fig2, width="stretch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(prep["vb"], axis=0)
    fig3 = go.Figure(go.Scatter(x=xc, y=sd, line=dict(color="#2a78d6", width=1.5), name="std"))
    fig3.add_trace(go.Scatter(x=xc, y=prep["sigma"], line=dict(color=MUTED, width=1, dash="dot"),
                              name="робастный шум σ (train)"))
    plot_layout(fig3, height=240)
    fig3.update_xaxes(title=f"координата, {x_unit_short(si['x_unit'])}")
    st.markdown("**Изменчивость во времени по координате**")
    st.plotly_chart(fig3, width="stretch")


def page_about():
    header("О проекте", "Методы, источники, ограничения, параметры и статус расчёта")
    man = manifest()
    dr = data_report()
    if man:
        a, b, c, d = st.columns(4)
        a.markdown(stat_tile("Статус последнего расчёта", "успешно" if man["status"] == "ok" else man["status"]),
                   unsafe_allow_html=True)
        b.markdown(stat_tile("Завершён (UTC)", man["finished_utc"][:19].replace("T", " ")), unsafe_allow_html=True)
        c.markdown(stat_tile("Длительность", f"{man['duration_s']:.0f} с", f"config {man['config_hash']}"),
                   unsafe_allow_html=True)
        d.markdown(stat_tile("Событий / источников", f"{man['n_events']} / {man['n_sources']}"), unsafe_allow_html=True)
    from src.config import ARTIFACTS_DIR
    failed = ARTIFACTS_DIR / "last_failed_run.json"
    if failed.exists() and (not man or failed.stat().st_mtime > (ARTIFACTS_DIR / "results" / "run_manifest.json").stat().st_mtime):
        st.error("Последний запуск завершился с ошибкой: " + json.loads(failed.read_text(encoding="utf-8"))["error"])

    t1, t2, t3, t4 = st.tabs(["Методы", "Источники и качество данных", "Параметры и пересчёт", "Ограничения"])
    with t1:
        st.markdown(METHODS_MD)
    with t2:
        if dr:
            rep = pd.DataFrame(dr["sources"])
            cols = {"file": "файл", "dts_id": "прибор", "channel_id": "канал", "rows_valid": "кадров",
                    "time_start": "начало (UTC)", "time_end": "конец (UTC)", "tz_offset_h": "пояс, ч",
                    "dt_median_s": "шаг, с", "dt_max_s": "макс. разрыв, с", "n_points_grid": "точек",
                    "x_step": "шаг x", "x_max": "x макс", "x_unit": "единицы x",
                    "rows_truncated_profile": "укороченных профилей", "pts_sentinel": "значений −300",
                    "tail_columns_cropped": "обрезано столбцов", "rows_rejected": "отбраковано строк",
                    "duplicate_timestamps": "дубликатов времени", "pts_size_mismatch": "несовпадений points_size",
                    "has_points_size": "есть points_size", "value_p01": "p1", "value_median": "медиана",
                    "value_p99": "p99"}
            st.dataframe(rep[list(cols)].rename(columns=cols), hide_index=True, width="stretch")
            si = source_info()
            st.dataframe(si[["source_id", "n_frames", "n_bins", "bin_points", "n_segments", "split_counts",
                             "test_hours"]].astype(str), hide_index=True, width="stretch")
            if dr.get("failures"):
                st.error(f"Файлы, которые не удалось загрузить: {dr['failures']}")
            st.markdown("Подробный отчёт: `docs/DATA_REPORT.md`, ноутбук `notebooks/eda.ipynb`.")
    with t3:
        cfg = (man or {}).get("config", load_config())
        st.json(cfg, expanded=False)
        st.markdown("**Пересчёт с другими параметрами** — изменения сохраняются в "
                    f"`{USER_CONFIG_PATH.relative_to(USER_CONFIG_PATH.parents[1])}` и применяются ко всем запускам.")
        with st.form("rerun"):
            c1, c2, c3 = st.columns(3)
            thr_a = c1.number_input("Порог A, σ", 2.0, 50.0, float(cfg["detectors"]["statistical"]["min_threshold"]), 0.5)
            thr_c = c2.number_input("Порог C, σ", 2.0, 50.0, float(cfg["detectors"]["regression"]["min_threshold"]), 0.5)
            rate = c3.number_input("Доля срабатываний IF на validation", 0.0001, 0.05,
                                   float(cfg["detectors"]["target_rate"]), 0.0005, format="%.4f")
            thr_d = st.number_input("Порог D (автоэнкодер), σ", 2.0, 50.0,
                                    float(cfg["detectors"].get("autoencoder", {}).get("min_threshold", 6.0)), 0.5)
            ae_on = st.checkbox("Автоэнкодер (P2) включён",
                                value=bool(cfg["detectors"].get("autoencoder", {}).get("enabled", True)))
            c4, c5, c6 = st.columns(3)
            win = c4.number_input("Окно локальной нормы, мин", 5.0, 360.0,
                                  float(cfg["preprocess"]["baseline_window_min"]), 5.0)
            gap = c5.number_input("Макс. разрыв внутри события, с", 30.0, 3600.0, float(cfg["events"]["max_gap_s"]), 30.0)
            syn = c6.checkbox("Синтетический тест", value=bool(cfg["evaluation"]["synthetic"]["enabled"]))
            go_ = st.form_submit_button("Пересчитать", type="primary")
        if go_:
            over = {"detectors": {"statistical": {"min_threshold": thr_a}, "regression": {"min_threshold": thr_c},
                                  "autoencoder": {"min_threshold": thr_d, "enabled": ae_on},
                                  "target_rate": rate},
                    "preprocess": {"baseline_window_min": win}, "events": {"max_gap_s": gap},
                    "evaluation": {"synthetic": {"enabled": syn}}}
            save_user_config(over)
            from src.pipeline import run_pipeline, setup_logging
            setup_logging()
            bar = st.progress(0.0, "Запуск…")
            try:
                mm = run_pipeline(load_config(), progress=lambda f, msg: bar.progress(min(f, 1.0), msg))
                st.cache_data.clear()
                st.success(f"Готово: {mm['n_events']} событий за {mm['duration_s']} с.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Ошибка пересчёта: {e}. Предыдущие результаты сохранены.")
        if USER_CONFIG_PATH.exists() and st.button("Сбросить параметры к значениям по умолчанию"):
            USER_CONFIG_PATH.unlink()
            st.success("Пользовательские параметры удалены. Нажмите «Пересчитать», чтобы применить.")
    with t4:
        st.markdown(LIMITS_MD)


METHODS_MD = """
**Пайплайн.** CSV → валидация (`src/data_loader.py`) → бины ~1 м, локальная норма, общий сдвиг кадра, робастный шум
(`src/preprocess.py`) → признаки окон «кадр × бин» (`src/features.py`) → три детектора → объединение срабатываний в
события (`src/events.py`) → повторы, синхронность, кластеры (`src/patterns.py`) → сравнение (`src/evaluation.py`).
Интерфейс только читает сохранённые результаты.

| Метод | Идея | Оценка (больше = необычнее) | Порог |
|---|---|---|---|
| **A. Статистический** | отклонение бина от медианы его прошлых кадров за 60 мин, минус общий сдвиг кадра | \\|dev\\| / (1.4826·MAD на train) | фиксированный, 6σ |
| **B. Isolation Forest** | нехарактерные сочетания 8 признаков окна (z, скорость, контраст с соседями, разброс, длительность, ширина) | −score_samples | квантиль 99.8 % на validation, ≥ 0.6 |
| **C. Регрессия + остатки** | Ridge прогнозирует изменение бина за кадр по прошлым кадрам, норме, соседям и общему сдвигу соседних бинов | \\|остаток\\| / (1.4826·MAD остатков на train) | фиксированный, 6σ |
| **D. Автоэнкодер (P2)** | сеть 35→24→6→24→35 восстанавливает патч z (5 прошлых кадров × 7 бинов); обучение на train без 1 % самых экстремальных патчей | ошибка восстановления центра / (1.4826·MAD на train) | фиксированный, 6σ |

**Разбиение** — по времени внутри каждого источника: 60 % train / 20 % validation / 20 % test. Норма, лаги и признаки
используют только прошлые кадры; через разрывы > 15 мин история не переносится.
**События**: срабатывания одного канала объединяются при разрыве ≤ 300 с и ≤ 2 бинов; одиночное окно — только при
оценке ≥ 1.5× порога. **Сравнение** — на test-части всех источников, одинаковые правила для всех методов.
"""

LIMITS_MD = """
- **Нет экспертной разметки** → Precision/Recall/F1 не вычисляются. Синтетический тест проверяет только техническую
  работоспособность.
- **Соответствие «канал → кабель» не предоставлено**: `cable_id` пуст, в интерфейсе — «кабель не сопоставлен».
- **Единицы.** Значения похожи на температуру DTS в °C (диапазон 6–56), шаг x 0.10/0.25 похож на метры — это не
  подтверждено. В файле 5 x идёт с шагом 1.0: вероятно, это индекс точки. Допустимые точки 0–1688 совпадают по длине
  с каналом 2 прибора cd477c43 (файл 3), но это гипотеза.
- **Время.** Файлы 1 и 6 записаны в UTC+03, файлы 2–4 в UTC+00, у файла 5 часовой пояс не указан, поэтому его время
  показано как есть. Синхронизация часов разных приборов неизвестна: совпадения между приборами — только кандидаты.
- **Файл 5**: значение −300 в столбцах 1689–2363 всех кадров трактуется как заполнитель за концом волокна и
  обрезается (с подсчётом в отчёте).
- **Файл 1**: режим опроса меняется (4 мин → 30 с), идут крупные одновременные изменения многих участков — отсюда
  большое число событий и «синхронных» групп.
- **Повторяемость и периодичность** оцениваются на рядах длиной 9–48 ч. Утверждать регулярность на таком объёме
  нельзя, пометка «возможно периодическое» требует ручной проверки.
- Категории событий (импульс, ступенька и т. п.) описывают форму сигнала, а **не физическую причину**.
- Автоэнкодер (P2) — дополнительный эксперимент: небольшая полносвязная сеть sklearn на 35-мерных патчах,
  без подбора архитектуры. Его результаты не заменяют три основных метода.
"""
