"""Интерфейсный слой без моделей и Streamlit."""
import pandas as pd

from ui.presentation import (HIGH_SHARE, MEDIUM_SHARE, add_presentation_fields, event_title,
                             method_counts, representative_events)


def sample():
    t = pd.Timestamp("2026-01-01T10:00:00Z")
    rows = []
    for idx, (detector, s, x, amp, shape, persistent, score, source) in enumerate([
        ("statistical", 0, 100, 7, True, True, 3.1, "4"),
        ("regression", 60, 102, 7, True, True, 2.8, "4"),
        ("autoencoder", 4000, 800, -3, False, False, 1.1, "4"),
        ("statistical", 0, 100, 2, None, False, 1.1, "5"),
        ("statistical", 100, 101, 7, True, True, 3.0, "4"),
    ]):
        rows.append(dict(event_id=f"e{idx}", source_id=source, detector=detector,
                         start_time=t + pd.Timedelta(seconds=s),
                         end_time=t + pd.Timedelta(seconds=s + 180),
                         x_start=x, x_end=x + 20, amplitude=amp,
                         abrupt=shape, persistent=persistent, duration_class="длительное" if persistent else "кратковременное",
                         x_unit="индекс", anomaly_score_norm=score, amplitude_z=5.2 if amp > 3 else 2.0))
    return pd.DataFrame(rows)


def test_title_and_units():
    df = sample()
    assert event_title(df.iloc[0]) == "Резкое устойчивое повышение сигнала на участке 100–120 ед. X"
    assert "снижение" in event_title(df.iloc[2])
    e = df.iloc[0].copy()
    e["x_unit"] = "м"
    assert event_title(e).endswith("100–120 м")


def test_counts_unique_algorithms_not_records():
    df = sample()
    counts = method_counts(df)
    assert counts.loc[0] == 2  # совпали statistical и regression
    assert counts.loc[1] == 2
    assert counts.loc[4] == 2  # второй statistical не даёт третьего голоса
    assert counts.loc[2] == 1
    assert counts.loc[3] == 1  # разные источники никогда не объединяются


def test_attention_without_claiming_failure():
    df = add_presentation_fields(sample())
    assert df.loc[0, "attention"] == "Высокий"
    assert df.loc[2, "attention"] == "Низкий"
    assert df.loc[0, "method_count"] == 2
    assert "авария" not in df.loc[0, "title"].lower()


def test_representatives_not_duplicate_same_event():
    df = add_presentation_fields(sample())
    picked = representative_events(df, 5)
    assert len(picked) == 3
    assert len(set(picked["event_id"])) == 3


def test_attention_is_a_pyramid_across_scales():
    rows = []
    for i in range(100):
        # у IF узкая шкала оценки, у статистики все |z| заведомо велики
        for det, score, z in (("isolation_forest", 1.0 + i / 1000, 3 + i / 50), ("statistical", 1 + i / 10, 6 + i / 5)):
            rows.append(dict(event_id=f"{det}{i}", source_id=str(i), detector=det,
                             start_time=pd.Timestamp("2026-01-01T00:00:00Z"),
                             end_time=pd.Timestamp("2026-01-01T00:03:00Z"),
                             x_start=0, x_end=1, amplitude=1, x_unit="м",
                             anomaly_score_norm=score, amplitude_z=z))
    df = add_presentation_fields(pd.DataFrame(rows))
    share = df["attention"].value_counts(normalize=True)
    assert abs(share["Высокий"] - HIGH_SHARE) < 0.02
    assert abs(share["Средний"] - MEDIUM_SHARE) < 0.02
    assert share["Высокий"] < share["Средний"] < share["Низкий"]
    high = df[df["attention"] == "Высокий"]
    assert set(high["detector"]) == {"isolation_forest", "statistical"}


def test_empty_catalog():
    e = add_presentation_fields(sample().iloc[:0])
    assert e.empty
    assert list(method_counts(e)) == []
    assert representative_events(e).empty
