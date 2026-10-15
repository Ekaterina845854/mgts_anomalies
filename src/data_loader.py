"""Загрузка и валидация исходных CSV.

Обнаруженный на реальных файлах формат (см. docs/DATA_REPORT.md):
  * столбцы dts_id, channel_id, timestamp, points и (только в 1.csv, 6.csv) points_size;
  * points - JSON-массив пар [x, value]; x начинается с 0 и растёт с постоянным шагом;
  * timestamp - "YYYY-MM-DD HH:MM:SS+HH" либо без часового пояса (5.csv);
  * 5.csv содержит заполнитель -300 за пределами волокна.

Разбор points выполняется только через json.loads (без eval/literal_eval: других
форматов в данных не обнаружено). Каждая отбракованная строка или точка
сохраняется с причиной в отчёте качества, ничего не удаляется молча.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["dts_id", "channel_id", "timestamp", "points"]
OPTIONAL_COLUMNS = ["points_size"]
_TS_RE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*(Z|[+-]\d{2}(?::?\d{2})?)?\s*$"
)


class DataFormatError(ValueError):
    """Файл не соответствует ни одной из поддерживаемых схем."""


@dataclass
class SourceData:
    """Один источник сопоставимых сигналов: один прибор + один канал + один файл."""

    source_id: str
    file_name: str
    dts_id: str
    channel_id: int
    times: pd.DatetimeIndex  # UTC (для источников без пояса - "как есть", см. tz_known)
    x: np.ndarray  # координата вдоль волокна, общая сетка
    values: np.ndarray  # (кадры, точки) float32, NaN = нет данных
    tz_known: bool
    tz_offset_h: float | None
    x_unit: str
    report: dict = field(default_factory=dict)
    issues: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_frames(self) -> int:
        return self.values.shape[0]

    @property
    def n_points(self) -> int:
        return self.values.shape[1]

    @property
    def dx(self) -> float:
        return float(np.median(np.diff(self.x))) if len(self.x) > 1 else 1.0

    @property
    def dt_median_s(self) -> float:
        if len(self.times) < 2:
            return float("nan")
        return float(np.median(np.diff(to_epoch_ns(self.times))) / 1e9)


# --------------------------------------------------------------------------- parsing

def parse_timestamp(raw: str) -> tuple[pd.Timestamp | None, float | None, str | None]:
    """Возвращает (время в UTC, смещение пояса в часах или None, ошибка)."""
    m = _TS_RE.match(str(raw))
    if not m:
        return None, None, "timestamp_unparseable"
    date, time, tz = m.groups()
    try:
        ts = pd.Timestamp(f"{date} {time}")
    except (ValueError, TypeError):
        return None, None, "timestamp_invalid"
    if tz is None:
        return ts.tz_localize("UTC"), None, None
    if tz == "Z":
        return ts.tz_localize("UTC"), 0.0, None
    sign = -1 if tz[0] == "-" else 1
    digits = tz[1:].replace(":", "")
    hours = int(digits[:2])
    minutes = int(digits[2:4]) if len(digits) >= 4 else 0
    offset = sign * (hours + minutes / 60)
    return (ts - pd.Timedelta(hours=offset)).tz_localize("UTC"), offset, None


def to_epoch_ns(times) -> np.ndarray:
    """Наносекунды от эпохи независимо от внутреннего разрешения pandas (в pandas 3 бывает 'us')."""
    return np.asarray(pd.DatetimeIndex(times).as_unit("ns").asi8, dtype=np.int64)


def epoch_s(s) -> np.ndarray:
    """Секунды от эпохи для Series/Index дат (корректно при любом разрешении: s, ms, us, ns)."""
    return to_epoch_ns(pd.DatetimeIndex(s)) / 1e9


def _parse_points_fast(s: str) -> np.ndarray | None:
    """Быстрый разбор строго вида [[x, v], [x, v], ...]; None, если структура иная."""
    if not (s.startswith("[[") and s.endswith("]]")):
        return None
    body = s[2:-2].replace(" ", "")
    n_pairs = body.count("],[") + 1
    if body.count("[") != n_pairs - 1 or body.count("]") != n_pairs - 1:
        return None
    flat = body.replace("],[", ",")
    if flat.count(",") != 2 * n_pairs - 1:
        return None
    try:
        arr = np.array(flat.split(","), dtype=np.float64)
    except ValueError:
        return None
    return arr.reshape(n_pairs, 2)


def parse_points(raw: str) -> tuple[np.ndarray | None, str | None]:
    """JSON-массив пар [x, value] -> массив (n, 2) float64."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "points_empty"
    fast = _parse_points_fast(raw.strip())
    if fast is not None:
        return fast, None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, RecursionError):
        return None, "points_not_json"
    if not isinstance(obj, list) or len(obj) == 0:
        return None, "points_not_list"
    try:
        arr = np.asarray(obj, dtype=np.float64)
    except (ValueError, TypeError):
        return None, "points_ragged_or_non_numeric"
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None, "points_not_pairs"
    return arr, None


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    h.update(f"{st.st_size}".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- loading

def read_raw_csv(path: Path) -> pd.DataFrame:
    """Чтение CSV как строк и проверка схемы."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataFormatError(f"{path.name}: отсутствуют обязательные столбцы {missing}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if extra:
        log.warning("%s: неизвестные столбцы %s игнорируются", path.name, extra)
    return df


def load_file(path: Path, cfg: dict) -> list[SourceData]:
    """Загружает файл и возвращает по одному SourceData на пару (прибор, канал)."""
    path = Path(path)
    df = read_raw_csv(path)
    dcfg = cfg["data"]
    sentinels = np.asarray(dcfg.get("sentinel_values", []), dtype=np.float64)
    vmin, vmax = dcfg.get("valid_value_range", [-np.inf, np.inf])
    has_size = "points_size" in df.columns

    issues: list[dict] = []
    rows: list[dict] = []
    point_stats = dict(nonfinite_x=0, nonfinite_v=0, sentinel=0, out_of_range=0,
                       negative_x=0, unsorted_rows=0, duplicate_x=0, size_mismatch=0,
                       size_missing=0)

    for i, r in enumerate(df.itertuples(index=False)):
        rec = r._asdict()
        dts = str(rec["dts_id"]).strip()
        if not dts:
            issues.append(dict(row=i, reason="dts_id_empty", detail=""))
            continue
        try:
            ch = int(float(rec["channel_id"]))
        except (TypeError, ValueError):
            issues.append(dict(row=i, reason="channel_id_invalid", detail=str(rec["channel_id"])[:40]))
            continue
        ts, offset, err = parse_timestamp(rec["timestamp"])
        if err:
            issues.append(dict(row=i, reason=err, detail=str(rec["timestamp"])[:40]))
            continue
        arr, err = parse_points(rec["points"])
        if err:
            issues.append(dict(row=i, reason=err, detail=str(rec["points"])[:60]))
            continue
        x, v = arr[:, 0], arr[:, 1]
        if not np.isfinite(x).all():
            point_stats["nonfinite_x"] += int((~np.isfinite(x)).sum())
            issues.append(dict(row=i, reason="x_nonfinite", detail=""))
            continue
        if (x < 0).any():
            point_stats["negative_x"] += int((x < 0).sum())
        if has_size:
            raw_size = str(rec["points_size"]).strip()
            if raw_size == "":
                point_stats["size_missing"] += 1
            else:
                try:
                    if int(float(raw_size)) != len(arr):
                        point_stats["size_mismatch"] += 1
                        issues.append(dict(row=i, reason="points_size_mismatch",
                                           detail=f"{raw_size} vs {len(arr)}"))
                except ValueError:
                    point_stats["size_mismatch"] += 1
                    issues.append(dict(row=i, reason="points_size_invalid", detail=raw_size[:20]))
        if (np.diff(x) < 0).any():
            point_stats["unsorted_rows"] += 1
            order = np.argsort(x, kind="stable")
            x, v = x[order], v[order]
        dup = np.r_[False, np.diff(x) == 0]
        if dup.any():
            # повторяющаяся координата: оставляем первое значение, считаем остальные
            point_stats["duplicate_x"] += int(dup.sum())
            x, v = x[~dup], v[~dup]
        v = v.copy()
        nf = ~np.isfinite(v)
        point_stats["nonfinite_v"] += int(nf.sum())
        if sentinels.size:
            sm = np.isin(v, sentinels)
            point_stats["sentinel"] += int(sm.sum())
            v[sm] = np.nan
        oor = np.isfinite(v) & ((v < vmin) | (v > vmax))
        point_stats["out_of_range"] += int(oor.sum())
        v[oor | nf] = np.nan
        rows.append(dict(row=i, dts_id=dts, channel_id=ch, ts=ts, offset=offset, x=x, v=v))

    if not rows:
        raise DataFormatError(f"{path.name}: нет ни одной корректной строки")

    issues_df = pd.DataFrame(issues, columns=["row", "reason", "detail"])
    pairs = sorted({(r["dts_id"], r["channel_id"]) for r in rows})
    out = []
    for dts, ch in pairs:
        sub = [r for r in rows if r["dts_id"] == dts and r["channel_id"] == ch]
        sid = path.stem if len(pairs) == 1 else f"{path.stem}|{dts[:8]}|{ch}"
        out.append(_assemble_source(sid, path.name, dts, ch, sub, issues_df, point_stats,
                                    len(df), has_size))
    return out


def _assemble_source(sid, fname, dts, ch, sub, issues_df, point_stats, n_raw, has_size) -> SourceData:
    issues = [issues_df] if len(issues_df) else []
    # --- порядок и дубликаты времени
    raw_order_ts = [r["ts"] for r in sub]  # sub уже в порядке строк файла
    monotonic = all(a <= b for a, b in zip(raw_order_ts[:-1], raw_order_ts[1:]))
    sub = sorted(sub, key=lambda r: (r["ts"], r["row"]))
    dedup, dups = [], []
    for r in sub:
        if dedup and r["ts"] == dedup[-1]["ts"]:
            dups.append(dict(row=r["row"], reason="duplicate_timestamp", detail=str(r["ts"])))
            continue
        dedup.append(r)
    if dups:
        issues.append(pd.DataFrame(dups))
    sub = dedup

    # --- часовой пояс
    offsets = {r["offset"] for r in sub}
    tz_known = None not in offsets
    tz_offset = next(iter(offsets)) if len(offsets) == 1 else None

    # --- общая сетка x: самый длинный профиль; остальные должны быть его префиксами
    ref = max(sub, key=lambda r: len(r["x"]))["x"]
    n = len(ref)
    vals = np.full((len(sub), n), np.nan, dtype=np.float32)
    regridded = truncated = 0
    lengths = np.empty(len(sub), dtype=int)
    for k, r in enumerate(sub):
        m = len(r["x"])
        lengths[k] = m
        if m <= n and np.allclose(r["x"], ref[:m], rtol=1e-9, atol=1e-9):
            vals[k, :m] = r["v"]
            truncated += m < n
        else:
            # сетка отличается - линейная интерполяция внутри исходного диапазона
            ok = np.isfinite(r["v"])
            inside = (ref >= r["x"][0]) & (ref <= r["x"][-1])
            if ok.sum() >= 2:
                vals[k, inside] = np.interp(ref[inside], r["x"][ok], r["v"][ok])
            regridded += 1

    # --- обрезка хвоста, где нет ни одного значения (например, -300 за концом волокна)
    col_valid = np.isfinite(vals).any(axis=0)
    last = int(np.max(np.nonzero(col_valid)[0])) + 1 if col_valid.any() else 0
    cropped = n - last
    vals, x = vals[:, :last], ref[:last]

    times = pd.DatetimeIndex([r["ts"] for r in sub]).as_unit("ns")
    is_integer_x = np.allclose(x, np.round(x)) and np.median(np.diff(x)) == 1.0
    x_unit = "индекс точки (единицы не подтверждены)" if is_integer_x else "м (предположительно)"
    dt = np.diff(to_epoch_ns(times)) / 1e9 if len(times) > 1 else np.array([np.nan])
    report = dict(
        source_id=sid, file=fname, dts_id=dts, channel_id=ch,
        rows_raw=n_raw, rows_valid=len(sub), rows_rejected=int(len(issues_df)),
        duplicate_timestamps=len(dups), timestamps_monotonic_in_file=bool(monotonic),
        tz_known=tz_known, tz_offset_h=tz_offset,
        time_start=str(times[0]), time_end=str(times[-1]),
        dt_median_s=float(np.median(dt)), dt_min_s=float(np.min(dt)), dt_max_s=float(np.max(dt)),
        n_points_grid=int(last), points_min=int(lengths.min()), points_max=int(lengths.max()),
        rows_truncated_profile=int(truncated), rows_regridded=int(regridded),
        tail_columns_cropped=int(cropped), x_step=float(np.median(np.diff(x))) if len(x) > 1 else None,
        x_max=float(x[-1]) if len(x) else None, x_unit=x_unit,
        has_points_size=has_size, nan_fraction=float(np.isnan(vals).mean()),
        value_p01=float(np.nanpercentile(vals, 1)), value_median=float(np.nanmedian(vals)),
        value_p99=float(np.nanpercentile(vals, 99)),
        **{f"pts_{k}": v for k, v in point_stats.items()},
    )
    issues_all = pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(
        columns=["row", "reason", "detail"])
    return SourceData(sid, fname, dts, ch, times, x.astype(np.float64), vals, tz_known,
                      tz_offset, x_unit, report, issues_all)


# --------------------------------------------------------------------------- cache

def load_source_cached(path: Path, cfg: dict, cache_dir: Path) -> list[SourceData]:
    """Загружает файл, переиспользуя кэш processed/*.npz при неизменном файле и настройках."""
    path = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((file_fingerprint(path) + json.dumps(cfg["data"], sort_keys=True))
                       .encode()).hexdigest()[:12]
    meta_path = cache_dir / f"{path.stem}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            try:
                return [_load_npz(cache_dir, m) for m in meta["sources"]]
            except (OSError, KeyError, ValueError) as e:
                log.warning("Кэш %s повреждён (%s), перечитываю CSV", meta_path.name, e)
    log.info("Разбор %s ...", path.name)
    sources = load_file(path, cfg)
    metas = []
    for s in sources:
        fn = f"{s.source_id.replace('|', '_')}.npz"
        np.savez_compressed(cache_dir / fn, values=s.values, x=s.x, times=to_epoch_ns(s.times))
        s.issues.to_csv(cache_dir / fn.replace(".npz", ".issues.csv"), index=False)
        metas.append(dict(npz=fn, source_id=s.source_id, file_name=s.file_name, dts_id=s.dts_id,
                          channel_id=s.channel_id, tz_known=s.tz_known, tz_offset_h=s.tz_offset_h,
                          x_unit=s.x_unit, report=s.report))
    meta_path.write_text(json.dumps(dict(key=key, sources=metas), ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return sources


def _load_npz(cache_dir: Path, m: dict) -> SourceData:
    z = np.load(cache_dir / m["npz"])
    issues_path = cache_dir / m["npz"].replace(".npz", ".issues.csv")
    issues = pd.read_csv(issues_path) if issues_path.exists() else pd.DataFrame()
    return SourceData(m["source_id"], m["file_name"], m["dts_id"], int(m["channel_id"]),
                      pd.DatetimeIndex(pd.to_datetime(z["times"], unit="ns", utc=True)), z["x"], z["values"],
                      m["tz_known"], m["tz_offset_h"], m["x_unit"], m["report"], issues)


def load_all(cfg: dict, data_dir: Path, cache_dir: Path) -> tuple[list[SourceData], list[dict]]:
    """Загружает все файлы; ошибки по отдельным файлам не останавливают остальные."""
    sources, failures = [], []
    for fname in cfg["data"]["files"]:
        p = Path(data_dir) / fname
        if not p.exists():
            failures.append(dict(file=fname, error="файл не найден"))
            log.error("%s: файл не найден", fname)
            continue
        try:
            sources.extend(load_source_cached(p, cfg, cache_dir))
        except (DataFormatError, OSError, ValueError) as e:
            failures.append(dict(file=fname, error=str(e)))
            log.exception("%s: ошибка загрузки", fname)
    return sources, failures
