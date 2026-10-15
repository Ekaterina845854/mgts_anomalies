"""Сквозной пайплайн: CSV -> валидация -> нормализация -> признаки -> детекторы ->
события -> закономерности -> оценка -> artifacts/results.

Запуск:  python -m src.pipeline            (конфигурация по умолчанию + config/user_config.json)
         python -m src.pipeline --no-synthetic --config path/to/override.json
Результат детерминирован при одинаковых данных и конфигурации (фиксированные random_state).
"""
from __future__ import annotations

import argparse
import logging
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, DATA_DIR, LABELS_PATH, PROJECT_ROOT, config_hash, load_config
from .data_loader import SourceData, file_fingerprint, load_all
from .autoencoder import detect_autoencoder
from .detectors import DETECTOR_LABELS, detect_isolation_forest, detect_statistical
from .evaluation import (compare_detectors, label_template_path, labeled_metrics, load_labels,
                         synthetic_benchmark, window_agreement)
from .events import aggregate_events, validate_event_frame
from .features import compute_features
from .patterns import apply_patterns
from .preprocess import prepare
from .regression import detect_regression
from .repository import ResultStore, promote_staging

log = logging.getLogger("pipeline")
DETECTORS = {
    "statistical": detect_statistical,
    "isolation_forest": detect_isolation_forest,
    "regression": detect_regression,
    "autoencoder": detect_autoencoder,
}


def setup_logging(level=logging.INFO) -> None:
    logdir = PROJECT_ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(logdir / "pipeline.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(fmt))
        root.addHandler(sh)
    root.setLevel(level)


def run_source(src: SourceData, cfg: dict) -> dict:
    """Все детекторы и агрегация для одного источника. Ошибка детектора не роняет остальные."""
    prep = prepare(src, cfg)
    feats = compute_features(prep)
    results, events, errors = {}, [], {}
    for name, fn in DETECTORS.items():
        if not cfg["detectors"].get(name, {}).get("enabled", True):
            continue
        try:
            r = fn(prep, feats, cfg)
            results[name] = r
            events.append(aggregate_events(prep, r, feats, cfg))
        except (ValueError, np.linalg.LinAlgError, MemoryError) as e:
            log.exception("%s / %s: ошибка детектора", src.source_id, name)
            errors[name] = str(e)
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    return dict(prep=prep, feats=feats, results=results, events=ev, errors=errors)


def run_pipeline(cfg: dict | None = None, data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR,
                 synthetic: bool | None = None, progress=None) -> dict:
    """Полный запуск; возвращает манифест. progress(fraction, message) - для интерфейса."""
    cfg = cfg or load_config()
    t_start = time.time()
    started = datetime.now(timezone.utc).isoformat()
    staging_name = "results_staging"
    staging = Path(artifacts_dir) / staging_name
    if staging.exists():
        shutil.rmtree(staging)
    store = ResultStore(artifacts_dir, staging_name)
    say = progress or (lambda f, m: None)
    stage_times = {}

    def stage(name, t0):
        stage_times[name] = round(time.time() - t0, 2)
        log.info("этап %s: %.1f c", name, stage_times[name])

    try:
        t0 = time.time()
        say(0.02, "Загрузка и валидация CSV")
        sources, failures = load_all(cfg, data_dir, Path(artifacts_dir) / "processed")
        stage("load", t0)
        if not sources:
            raise RuntimeError(f"Ни один файл не загружен: {failures}")

        all_events, meta_rows, agreement_acc, reg_metrics, src_info = [], [], {}, {}, []
        test_hours, dt_by_source, source_times = {}, {}, {}
        for i, src in enumerate(sources):
            t0 = time.time()
            say(0.05 + 0.6 * i / len(sources), f"Источник {src.source_id}: детекторы и события")
            out = run_source(src, cfg)
            prep = out["prep"]
            store.save_prepared(prep)
            test = prep.split_mask("test")
            tsec = prep.t_sec[test]
            test_hours[src.source_id] = float((tsec.max() - tsec.min()) / 3600) if test.sum() > 1 else 0.0
            dt_by_source[src.source_id] = prep.dt_median_s
            source_times[src.source_id] = (float(prep.times[0].timestamp()), float(prep.times[-1].timestamp()))
            flags_test = {}
            for name, r in out["results"].items():
                store.save_scores(src.source_id, name, r.scores)
                fl = r.flags
                flags_test[name] = fl[test]
                meta_rows.append(dict(
                    source_id=src.source_id, detector=name, threshold=r.threshold,
                    threshold_mode=r.config.get("threshold_mode"),
                    validation_alarm_rate=r.config.get("validation_alarm_rate"),
                    flagged_windows_eval=int(fl[test].sum()),
                    windows_eval=int(np.isfinite(r.scores[test]).sum()),
                    flagged_windows_all=int(fl.sum()), fit_time_s=r.fit_time_s,
                    score_time_s=r.score_time_s, peak_mem_mb=r.peak_mem_mb,
                    score_transform=r.score_transform, config=r.config))
                if name == "regression":
                    reg_metrics[src.source_id] = dict(r.extra.get("metrics", {}),
                                                      coef=r.extra.get("coef"),
                                                      intercept=r.extra.get("intercept"))
            for row in window_agreement(flags_test):
                key = (row["detector_a"], row["detector_b"])
                acc = agreement_acc.setdefault(key, [0, 0])
                acc[0] += row["intersection"]
                acc[1] += row["union"]
            all_events.append(out["events"])
            src_info.append(dict(
                source_id=src.source_id, file=src.file_name, dts_id=src.dts_id,
                channel_id=src.channel_id, cable_id=None, x_unit=src.x_unit,
                tz_known=src.tz_known, tz_offset_h=src.tz_offset_h,
                n_frames=prep.shape[0], n_bins=prep.shape[1], bin_points=prep.bin_points,
                dt_median_s=prep.dt_median_s, n_segments=prep.params["n_segments"],
                split_counts=prep.params["split_counts"], x_min=float(prep.bin_x_start[0]),
                x_max=float(prep.bin_x_end[-1]), time_start=prep.times[0].isoformat(),
                time_end=prep.times[-1].isoformat(), test_hours=test_hours[src.source_id],
                detector_errors=out["errors"]))
            stage(f"source_{src.source_id}", t0)

        t0 = time.time()
        say(0.7, "Закономерности: повторы, синхронность, кластеры")
        events = pd.concat(all_events, ignore_index=True)
        events, pat = apply_patterns(events, cfg, dt_by_source, source_times)
        errs = validate_event_frame(events)
        if errs:
            raise RuntimeError(f"Нарушен контракт событий: {errs}")
        store.save_events(events)
        events.to_csv(store.dir / "events.csv", index=False, encoding="utf-8-sig")
        for name, df in pat.items():
            store.save_table(name, df)
        stage("patterns", t0)

        t0 = time.time()
        say(0.8, "Оценка и сравнение детекторов")
        meta = pd.DataFrame(meta_rows)
        comparison = compare_detectors(events, meta, test_hours, cfg, split="test")
        comparison_by_source = {s: compare_detectors(events, meta, test_hours, cfg, split="test",
                                                     sources=[s]) for s in test_hours}
        label_template_path(LABELS_PATH)
        try:
            labels = load_labels(LABELS_PATH)
            label_eval = labeled_metrics(events, labels, cfg)
        except (ValueError, KeyError) as e:
            label_eval = dict(available=False, reason=f"ошибка чтения разметки: {e}", n_labels=0)
        stage("evaluation", t0)

        syn = None
        do_syn = cfg["evaluation"]["synthetic"]["enabled"] if synthetic is None else synthetic
        if do_syn:
            t0 = time.time()
            say(0.85, "Синтетический тест (инъекции в копию данных)")
            target = cfg["evaluation"]["synthetic"]["source"].replace(".csv", "")
            cand = [s for s in sources if s.source_id == target] or sources[:1]
            try:
                syn = synthetic_benchmark(cand[0], cfg)
            except (ValueError, RuntimeError) as e:
                syn = dict(error=str(e))
            stage("synthetic", t0)

        store.save_json("metrics.json", dict(
            comparison=comparison, comparison_by_source=comparison_by_source,
            window_agreement_test=[dict(detector_a=a, detector_b=b, intersection=v[0], union=v[1],
                                        jaccard=v[0] / v[1] if v[1] else None)
                                   for (a, b), v in agreement_acc.items()],
            detector_runs=meta.drop(columns=["config"]).to_dict("records"),
            detector_configs={f"{r['source_id']}|{r['detector']}": r["config"] for r in meta_rows},
            regression=reg_metrics, labels=label_eval, synthetic=syn,
            detector_labels=DETECTOR_LABELS))
        store.save_json("data_report.json", dict(
            sources=[s.report for s in sources], failures=failures,
            issues={s.source_id: s.issues["reason"].value_counts().to_dict() if len(s.issues) else {}
                    for s in sources},
            source_info=src_info))
        manifest = dict(
            status="ok", started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
            duration_s=round(time.time() - t_start, 1), config=cfg, config_hash=config_hash(cfg),
            data_fingerprints={f: file_fingerprint(Path(data_dir) / f) for f in cfg["data"]["files"]
                               if (Path(data_dir) / f).exists()},
            n_events=int(len(events)), n_sources=len(sources), failures=failures,
            stage_times_s=stage_times, python=platform.python_version(),
            versions=_versions())
        store.save_json("run_manifest.json", manifest)
        promote_staging(Path(artifacts_dir), staging_name, "results")
        say(1.0, "Готово")
        log.info("Пайплайн завершён: %d событий за %.1f c", len(events), time.time() - t_start)
        return manifest
    except Exception as e:  # noqa: BLE001 - фиксируем любую ошибку в статусе запуска
        log.exception("Пайплайн завершился с ошибкой")
        fail = dict(status="failed", started_utc=started, error=str(e),
                    traceback=traceback.format_exc(), config_hash=config_hash(cfg))
        from .repository import write_json
        write_json(Path(artifacts_dir) / "last_failed_run.json", fail)
        raise


def _versions() -> dict:
    import scipy
    import sklearn
    return dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fiber Analytics: запуск пайплайна")
    ap.add_argument("--config", type=Path, default=None, help="JSON с переопределениями параметров")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--no-synthetic", action="store_true", help="не запускать синтетический тест")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    m = run_pipeline(cfg, args.data_dir, synthetic=False if args.no_synthetic else None)
    print(f"OK: событий {m['n_events']}, источников {m['n_sources']}, {m['duration_s']} c, "
          f"config {m['config_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
