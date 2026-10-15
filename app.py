"""Fiber Analytics — Streamlit-интерфейс.

Запуск:  streamlit run app.py
Перед первым запуском рассчитайте результаты:  python -m src.pipeline
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Fiber Analytics", page_icon="〰️", layout="wide",
                   initial_sidebar_state="expanded")

from ui import views  # noqa: E402
from ui.common import inject_css, manifest  # noqa: E402

inject_css()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

pages = [
    st.Page(views.page_detection, title="Обнаружение событий", icon=":material/radar:", default=True),
    st.Page(views.page_catalog, title="Каталог событий", icon=":material/list_alt:", url_path="catalog"),
    st.Page(views.page_classification, title="Классификация", icon=":material/category:", url_path="classification"),
    st.Page(views.page_comparison, title="Сравнение алгоритмов", icon=":material/compare_arrows:",
            url_path="comparison"),
    st.Page(views.page_spatial, title="Пространственный обзор", icon=":material/timeline:", url_path="spatial"),
    st.Page(views.page_about, title="О проекте", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages, position="sidebar")
with st.sidebar:
    m = manifest()
    if m:
        st.caption(f"Расчёт: {m['finished_utc'][:16].replace('T', ' ')} UTC · {m['n_events']} событий · "
                   f"config {m['config_hash']}")
nav.run()
