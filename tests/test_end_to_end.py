"""Сквозной запуск пайплайна на СИНТЕТИЧЕСКИХ файлах во временном каталоге."""
import pandas as pd

from src.pipeline import run_pipeline
from src.repository import ResultStore
from tests.conftest import make_synthetic_csv


def test_pipeline_end_to_end(tmp_path, cfg, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    make_synthetic_csv(data / "a.csv", seed=1)
    make_synthetic_csv(data / "b.csv", seed=2, with_size=False, tz=None, channel=2)
    cfg["data"]["files"] = ["a.csv", "b.csv", "missing.csv"]
    cfg["evaluation"]["synthetic"]["source"] = "a.csv"
    cfg["evaluation"]["synthetic"]["n_events"] = 5
    import src.pipeline as pl
    monkeypatch.setattr(pl, "LABELS_PATH", tmp_path / "labels" / "labels.csv")
    art = tmp_path / "artifacts"
    m = run_pipeline(cfg, data, art)
    assert m["status"] == "ok" and m["n_sources"] == 2
    assert m["failures"] == [{"file": "missing.csv", "error": "файл не найден"}]
    st = ResultStore(art)
    ev = st.load_events()
    assert set(ev["detector"]) == {"statistical", "isolation_forest", "regression", "autoencoder"}
    met = st.load_json("metrics.json")
    assert met["labels"]["available"] is False  # нет разметки -> нет P/R/F1
    assert {r["detector"] for r in met["comparison"]["table"]} == {"statistical", "isolation_forest", "regression", "autoencoder"}
    assert st.load_prepared("a")["vb"].shape[1] == 100
    assert (art / "results" / "events.csv").exists()
    # повторный запуск воспроизводит результат
    m2 = run_pipeline(cfg, data, art)
    ev2 = ResultStore(art).load_events()
    assert m2["config_hash"] == m["config_hash"]
    pd.testing.assert_frame_equal(ev.drop(columns=["features"]), ev2.drop(columns=["features"]))
