"""Слой доступа к результатам пайплайна (единый источник правды для интерфейса).

artifacts/
  processed/              кэш разобранных CSV (ключ - отпечаток файла + настройки загрузки)
  results/                результаты последнего УСПЕШНОГО запуска
    run_manifest.json     статус, время, конфигурация, хэш, отпечатки данных, версии
    data_report.json      профиль качества по источникам
    events.parquet        каталог событий всех детекторов (контракт - events.EVENT_COLUMNS)
    metrics.json          сравнение детекторов, регрессия, разметка, синтетический тест
    patterns/*.csv        группы повторений, совпадения между приборами, автокорреляция
    prepared/<src>.npz    бинированный сигнал, отклонения, разбиение
    scores/<src>__<det>.npz  оценки окон
Запуск пишет в results_staging/ и заменяет results/ только после успеха.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR
from .data_loader import to_epoch_ns
from .preprocess import Prepared


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _clean(o):
    """Рекурсивно заменяет NaN/inf на None (JSON не поддерживает NaN)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=1, default=_json_default),
                    encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe(sid: str) -> str:
    return sid.replace("|", "_").replace("/", "_")


class ResultStore:
    def __init__(self, root: Path | None = None, results_name: str = "results"):
        self.root = Path(root) if root else ARTIFACTS_DIR
        self.dir = self.root / results_name

    # ---- запись
    def save_prepared(self, prep: Prepared) -> None:
        p = self.dir / "prepared" / f"{_safe(prep.source_id)}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, times=to_epoch_ns(prep.times), t_sec=prep.t_sec, segment=prep.segment,
            split=prep.split.astype("U10"), bin_x_start=prep.bin_x_start, bin_x_end=prep.bin_x_end,
            vb=prep.vb, dev=prep.dev, z=prep.z, sigma=prep.sigma, cm=prep.cm)

    def save_scores(self, source_id: str, detector: str, scores: np.ndarray) -> None:
        p = self.dir / "scores" / f"{_safe(source_id)}__{detector}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, scores=scores.astype(np.float32))

    def save_events(self, events: pd.DataFrame) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        events.to_parquet(self.dir / "events.parquet", index=False)

    def save_table(self, name: str, df: pd.DataFrame) -> None:
        p = self.dir / "patterns" / f"{name}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, encoding="utf-8")

    def save_json(self, name: str, obj) -> None:
        write_json(self.dir / name, obj)

    # ---- чтение
    def exists(self) -> bool:
        return (self.dir / "run_manifest.json").exists()

    def manifest(self) -> dict | None:
        p = self.dir / "run_manifest.json"
        return read_json(p) if p.exists() else None

    def load_json(self, name: str):
        p = self.dir / name
        return read_json(p) if p.exists() else None

    def load_events(self) -> pd.DataFrame:
        p = self.dir / "events.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def load_table(self, name: str) -> pd.DataFrame:
        p = self.dir / "patterns" / f"{name}.csv"
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def load_prepared(self, source_id: str) -> dict:
        z = np.load(self.dir / "prepared" / f"{_safe(source_id)}.npz")
        d = {k: z[k] for k in z.files}
        d["times"] = pd.DatetimeIndex(pd.to_datetime(d["times"], unit="ns", utc=True))
        return d

    def load_scores(self, source_id: str, detector: str) -> np.ndarray | None:
        p = self.dir / "scores" / f"{_safe(source_id)}__{detector}.npz"
        return np.load(p)["scores"] if p.exists() else None

    def mtime(self) -> float:
        p = self.dir / "run_manifest.json"
        return p.stat().st_mtime if p.exists() else 0.0


def promote_staging(root: Path, staging: str = "results_staging", final: str = "results") -> None:
    """Атомарная (насколько позволяет ФС) замена результатов после успешного запуска."""
    root = Path(root)
    src, dst, old = root / staging, root / final, root / f"{final}_old"
    if old.exists():
        shutil.rmtree(old)
    if dst.exists():
        dst.rename(old)
    src.rename(dst)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
