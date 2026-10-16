"""Синтетические фикстуры; с реальными данными не смешиваются."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_CONFIG  # noqa: E402


@pytest.fixture
def cfg():
    c = copy.deepcopy(DEFAULT_CONFIG)
    c["detectors"]["isolation_forest"]["n_estimators"] = 50
    c["detectors"]["isolation_forest"]["train_sample"] = 20000
    return c


def make_synthetic_csv(path: Path, n_frames=240, n_points=400, dx=0.25, dt_s=60, with_size=True,
                       tz="+03", anomaly=True, seed=0, dts="synthetic-test-device", channel=1,
                       amp=6.0) -> dict:
    """Синтетический CSV в формате реальных файлов; аномалия amp в кадрах 200..205 на x≈50."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_points) * dx
    static = 20 + 3 * np.sin(x / 7.0)
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    rows = []
    truth = dict(t0=200, t1=205, x0=48.0, x1=52.0, amp=amp)
    for k in range(n_frames):
        v = static + 0.3 * np.sin(k / 50) + rng.normal(0, 0.2, n_points)
        if anomaly and truth["t0"] <= k <= truth["t1"]:
            v[(x >= truth["x0"]) & (x <= truth["x1"])] += amp
        pts = json.dumps([[float(a), float(b)] for a, b in zip(x, v)])
        ts = (t0 + pd.Timedelta(seconds=k * dt_s)).strftime("%Y-%m-%d %H:%M:%S") + (tz or "")
        row = dict(dts_id=dts, channel_id=channel, timestamp=ts, points=pts)
        if with_size:
            row["points_size"] = n_points
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return truth


@pytest.fixture
def synthetic_source(tmp_path, cfg):
    from src.data_loader import load_file
    p = tmp_path / "synthetic.csv"
    truth = make_synthetic_csv(p)
    src = load_file(p, cfg)[0]
    return src, truth
