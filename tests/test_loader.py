"""Чтение CSV, разбор points, points_size, повреждённые строки, время и дубликаты."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataFormatError, load_file, parse_points, parse_timestamp
from tests.conftest import make_synthetic_csv


def test_parse_points_json_and_fast_path_agree():
    s = "[[0, 1.5], [0.25, 2.0], [0.5, -3e-1]]"
    arr, err = parse_points(s)
    assert err is None and arr.shape == (3, 2)
    assert np.allclose(arr[:, 1], [1.5, 2.0, -0.3])


@pytest.mark.parametrize("raw,reason", [
    ("", "points_empty"),
    ("not json", "points_not_json"),
    ("[[0, 1], [1]]", "points_ragged_or_non_numeric"),
    ("[[0, 1, 2]]", "points_not_pairs"),
    ("[]", "points_not_list"),
    ("[[0, \"a\"]]", "points_ragged_or_non_numeric"),
])
def test_parse_points_rejects_bad(raw, reason):
    arr, err = parse_points(raw)
    assert arr is None and err == reason


def test_parse_points_does_not_execute_code():
    arr, err = parse_points("__import__('os').system('echo hacked')")
    assert arr is None and err == "points_not_json"


def test_parse_timestamp_variants():
    ts, off, err = parse_timestamp("2026-09-14 00:03:19+03")
    assert err is None and off == 3.0 and ts == pd.Timestamp("2026-09-13 21:03:19", tz="UTC")
    ts, off, err = parse_timestamp("2025-10-16 12:00:06")
    assert err is None and off is None and ts.hour == 12
    _, _, err = parse_timestamp("16/10/2025 12:00")
    assert err == "timestamp_unparseable"


def test_reads_both_schemas(tmp_path, cfg):
    a, b = tmp_path / "with_size.csv", tmp_path / "no_size.csv"
    make_synthetic_csv(a, n_frames=20, with_size=True)
    make_synthetic_csv(b, n_frames=20, with_size=False, tz=None)
    sa, sb = load_file(a, cfg)[0], load_file(b, cfg)[0]
    assert sa.report["has_points_size"] and not sb.report["has_points_size"]
    assert sa.values.shape == sb.values.shape == (20, 400)
    assert sa.tz_known and sa.tz_offset_h == 3.0
    assert not sb.tz_known  # пояс не указан


def test_points_size_mismatch_and_corrupt_rows_are_reported(tmp_path, cfg):
    p = tmp_path / "bad.csv"
    make_synthetic_csv(p, n_frames=10)
    df = pd.read_csv(p)
    df.loc[2, "points_size"] = 999
    df.loc[3, "points"] = "[[0, 1], [0.25"
    df.loc[4, "timestamp"] = "garbage"
    df.to_csv(p, index=False)
    src = load_file(p, cfg)[0]
    reasons = set(src.issues["reason"])
    assert {"points_size_mismatch", "points_not_json", "timestamp_unparseable"} <= reasons
    assert src.n_frames == 8  # строки 3 и 4 отбракованы, строка 2 сохранена
    assert src.report["pts_size_mismatch"] == 1


def test_duplicates_unsorted_sentinel_truncation(tmp_path, cfg):
    p = tmp_path / "dup.csv"
    make_synthetic_csv(p, n_frames=10, n_points=40)
    df = pd.read_csv(p)
    df = pd.concat([df, df.iloc[[5]]])                      # дубликат времени
    df = df.iloc[::-1]                                       # обратный порядок
    pts = df.iloc[0]["points"]
    arr = np.array(eval_free(pts))
    arr[-5:, 1] = -300.0                                     # заполнитель
    df.iloc[0, df.columns.get_loc("points")] = to_json(arr)
    df.iloc[1, df.columns.get_loc("points")] = to_json(np.array(eval_free(df.iloc[1]["points"]))[:30])  # укороченный
    df.iloc[1, df.columns.get_loc("points_size")] = 30
    df.to_csv(p, index=False)
    src = load_file(p, cfg)[0]
    assert src.report["duplicate_timestamps"] == 1
    assert not src.report["timestamps_monotonic_in_file"]
    assert src.times.is_monotonic_increasing and src.n_frames == 10
    assert src.report["pts_sentinel"] == 5
    assert src.report["rows_truncated_profile"] == 1
    assert np.isnan(src.values).sum() == 5 + 10  # -300 -> NaN, хвост укороченного профиля -> NaN


def test_no_valid_rows_raises(tmp_path, cfg):
    p = tmp_path / "empty.csv"
    pd.DataFrame(dict(dts_id=["x"], channel_id=[1], timestamp=["bad"], points=["[]"])).to_csv(p, index=False)
    with pytest.raises(DataFormatError):
        load_file(p, cfg)


def test_missing_column_raises(tmp_path, cfg):
    p = tmp_path / "schema.csv"
    pd.DataFrame(dict(dts_id=["x"], timestamp=["2026-01-01 00:00:00"])).to_csv(p, index=False)
    with pytest.raises(DataFormatError):
        load_file(p, cfg)


def eval_free(s):
    import json
    return json.loads(s)


def to_json(a):
    import json
    return json.dumps(a.tolist())
