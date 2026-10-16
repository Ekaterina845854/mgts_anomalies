"""Воспроизводимый EDA: python -m src.eda -> docs/eda/ (используется в notebooks/eda.ipynb)."""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT, load_config  # noqa: E402
from .data_loader import SourceData, epoch_s, load_all  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "eda"
SUMMARY_COLS = ["file", "dts_id", "channel_id", "rows_raw", "rows_valid", "rows_rejected", "time_start",
                "time_end", "tz_known", "tz_offset_h", "dt_median_s", "dt_min_s", "dt_max_s",
                "points_min", "points_max", "rows_truncated_profile", "x_step", "x_max", "x_unit",
                "has_points_size", "pts_size_mismatch", "pts_sentinel", "tail_columns_cropped",
                "duplicate_timestamps", "nan_fraction", "value_p01", "value_median", "value_p99"]


def load_sources(cfg=None) -> list[SourceData]:
    cfg = cfg or load_config()
    sources, failures = load_all(cfg, DATA_DIR, ARTIFACTS_DIR / "processed")
    if failures:
        print("Не загружены:", failures)
    return sources


def summary_table(sources: list[SourceData]) -> pd.DataFrame:
    return pd.DataFrame([s.report for s in sources]).set_index("source_id")[SUMMARY_COLS]


def noise_stats(sources: list[SourceData]) -> pd.DataFrame:
    """Шум между кадрами и вдоль волокна, временная изменчивость."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in sources:
            v = s.values
            rows.append(dict(source_id=s.source_id,
                             frame_diff_std_median=float(np.nanmedian(np.nanstd(np.diff(v, axis=0), axis=0))),
                             spatial_diff_std=float(np.nanstd(np.diff(v, axis=1))),
                             temporal_std_median=float(np.nanmedian(np.nanstd(v, axis=0))),
                             temporal_std_p95=float(np.nanpercentile(np.nanstd(v, axis=0), 95))))
    return pd.DataFrame(rows).set_index("source_id")


def plot_overview(sources: list[SourceData], out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    n = len(sources)
    # 1. распределения значений
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8), sharey=False)
    for ax, s in zip(np.atleast_1d(axes), sources):
        v = s.values[np.isfinite(s.values)]
        ax.hist(v[:: max(1, v.size // 200000)], bins=80, color="#2a78d6")
        ax.set_title(f"{s.source_id}.csv", fontsize=9)
        ax.set_xlabel("значение")
    fig.suptitle("Распределение значений (после замены −300 на NaN)", fontsize=10)
    fig.tight_layout()
    paths.append(out / "values_hist.png")
    fig.savefig(paths[-1], dpi=90)
    plt.close(fig)
    # 2. интервалы между кадрами
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8))
    for ax, s in zip(np.atleast_1d(axes), sources):
        dt = np.diff(epoch_s(s.times))
        ax.hist(np.clip(dt, 0, 5 * np.median(dt)), bins=50, color="#1baf7a")
        ax.set_title(f"{s.source_id}.csv · медиана {np.median(dt):.0f} с, макс {dt.max():.0f} с", fontsize=8)
        ax.set_xlabel("шаг, с (обрезано на 5×медиана)")
    fig.tight_layout()
    paths.append(out / "time_steps.png")
    fig.savefig(paths[-1], dpi=90)
    plt.close(fig)
    # 3. длина профилей и пропуски по координате
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8))
    for ax, s in zip(np.atleast_1d(axes), sources):
        ax.plot(s.x, np.isnan(s.values).mean(axis=0), color="#eb6834")
        ax.set_title(f"{s.source_id}.csv · доля NaN по x", fontsize=9)
        ax.set_xlabel("x")
    fig.tight_layout()
    paths.append(out / "nan_by_x.png")
    fig.savefig(paths[-1], dpi=90)
    plt.close(fig)
    # 4. тепловые карты сырых значений и отклонений от медианы
    for s in sources:
        v = s.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = v - np.nanmedian(v, axis=0)
        fig, ax = plt.subplots(1, 3, figsize=(16, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
        ext = [0, len(v), s.x[0], s.x[-1]]
        im = ax[0].imshow(v.T, aspect="auto", origin="lower", cmap="Blues", extent=ext,
                          vmin=np.nanpercentile(v, 1), vmax=np.nanpercentile(v, 99.5))
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title(f"{s.source_id}.csv: значение (кадр × x)", fontsize=9)
        im = ax[1].imshow(d.T, aspect="auto", origin="lower", cmap="RdBu_r", extent=ext, vmin=-3, vmax=3)
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title("минус медиана по времени", fontsize=9)
        for k in np.linspace(0, len(v) - 1, 4).astype(int):
            ax[2].plot(s.x, v[k], lw=0.7, label=f"кадр {k}")
        ax[2].legend(fontsize=7)
        ax[2].set_title("профили сигнала", fontsize=9)
        fig.tight_layout()
        paths.append(out / f"heatmap_{s.source_id}.png")
        fig.savefig(paths[-1], dpi=80)
        plt.close(fig)
    return paths


def main():
    sources = load_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    summ = summary_table(sources)
    noise = noise_stats(sources)
    summ.join(noise).to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    issues = {s.source_id: (s.issues["reason"].value_counts().to_dict() if len(s.issues) else {})
              for s in sources}
    pd.Series(issues, name="issues").to_csv(OUT / "issues.csv", encoding="utf-8-sig")
    paths = plot_overview(sources)
    pd.set_option("display.width", 200)
    print(summ.T)
    print(noise)
    print("Проблемы строк:", issues)
    print("Графики:", [p.name for p in paths])


if __name__ == "__main__":
    main()
