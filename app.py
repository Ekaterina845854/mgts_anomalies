"""Fiber Analytics: streamlit run app.py (сначала python -m src.pipeline)."""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Fiber Analytics · Итоги", page_icon="〰️", layout="wide",
                   initial_sidebar_state="expanded")

from ui import accessible  # noqa: E402
from ui.common import inject_css, manifest  # noqa: E402

inject_css()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

pages = [
    st.Page(accessible.page_summary, title="Итоги", icon=":material/dashboard:", default=True),
    st.Page(accessible.page_events, title="События", icon=":material/radar:", url_path="events"),
    st.Page(accessible.page_patterns, title="Закономерности", icon=":material/timeline:", url_path="patterns"),
    st.Page(accessible.page_methods, title="Сравнение методов", icon=":material/compare_arrows:", url_path="comparison"),
    st.Page(accessible.page_data_limits, title="Данные и ограничения", icon=":material/info:", url_path="data"),
]
nav = st.navigation(pages, position="sidebar")
with st.sidebar:
    m = manifest()
    if m:
        st.caption(f"Расчёт: {m.get('finished_utc', '—')[:16].replace('T', ' ')} UTC · "
                   f"{m.get('n_events', 0)} событий · config {m.get('config_hash', '—')}")
nav.run()
