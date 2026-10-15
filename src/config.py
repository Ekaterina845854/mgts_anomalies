"""Конфигурация пайплайна.

Все параметры собраны в одном словаре, который сохраняется рядом с результатами
(artifacts/run_manifest.json). Значения по умолчанию подобраны по результатам EDA
(см. docs/DATA_REPORT.md): шаг по времени 10–240 с, шаг по координате ~0.1–0.25 м
(для файла 5 — 1.0 в неподтверждённых единицах), шум между кадрами ~0.35–1.3.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LABELS_PATH = DATA_DIR / "labels" / "labels.csv"
USER_CONFIG_PATH = PROJECT_ROOT / "config" / "user_config.json"

DEFAULT_CONFIG: dict = {
    "data": {
        "files": ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv"],
        # Значения, которые встречаются как заполнитель за пределами волокна.
        # Обнаружено в 5.csv: -300 ровно в столбцах 1689..2363 во всех кадрах.
        "sentinel_values": [-300.0],
        # Диапазон физически правдоподобных значений; всё вне диапазона -> NaN
        # с сохранением причины. Широкий намеренно: отбраковываем только мусор.
        "valid_value_range": [-60.0, 200.0],
    },
    "preprocess": {
        # Разрыв во времени больше gap_factor * медианный шаг начинает новый сегмент
        # (скользящая норма и лаги не переносятся через разрыв).
        "gap_factor": 5.0,
        "gap_min_s": 900.0,
        # Целевой размер пространственного бина в единицах x. Для файла 5
        # (x с шагом 1.0, единицы не подтверждены) используется bin_points_fallback.
        "bin_target_x": 1.0,
        "bin_points_fallback": 4,
        # Скользящая (только прошлые кадры) медиана - "локальная норма".
        "baseline_window_min": 60.0,
        "baseline_min_frames": 5,
        # Доли разбиения по времени внутри каждого источника.
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
        # Нижняя граница шума (доля медианного шума источника) - защита от MAD=0.
        "noise_floor_frac": 0.25,
        "noise_floor_abs": 1e-3,
    },
    "detectors": {
        # threshold_mode: "fixed" - порог min_threshold в шкале метода (σ шума для A и C);
        # "quantile" - квантиль (1 - target_rate) оценок на validation, не ниже min_threshold.
        "target_rate": 0.002,
        "statistical": {"threshold_mode": "fixed", "min_threshold": 6.0},
        "isolation_forest": {
            "n_estimators": 150,
            "max_samples": 512,
            "train_sample": 60000,
            "threshold_mode": "quantile",
            # 0.6: у исходного IF оценка > 0.5 уже "подозрительна"; нижняя граница
            # не даёт порогу опуститься в шум на спокойных источниках.
            "min_threshold": 0.6,
            "random_state": 42,
        },
        "regression": {
            "alpha": 1.0,
            "n_lags": 3,
            "train_sample": 200000,
            "threshold_mode": "fixed",
            "min_threshold": 6.0,
            "random_state": 42,
        },
        # P2: дополнительный эксперимент; включается/выключается флагом enabled.
        "autoencoder": {
            "enabled": True,
            "hidden": [24, 6, 24],
            "alpha": 1e-4,
            "max_iter": 80,
            "train_sample": 40000,
            "trim_quantile": 0.01,
            "threshold_mode": "fixed",
            "min_threshold": 6.0,
            "random_state": 42,
        },
    },
    "events": {
        # Срабатывания одного прибора/канала объединяются, если между ними не
        # больше max_gap_s по времени и max_gap_bins бинов по координате.
        "max_gap_s": 300.0,
        "max_gap_bins": 2,
        # Событие должно содержать хотя бы min_cells окон; одиночное окно
        # допускается только при оценке >= single_cell_score_mult * порог.
        "min_cells": 2,
        "single_cell_score_mult": 1.5,
        # Описательные категории (не физические причины).
        "short_max_s": 600.0,
        "local_max_x": 5.0,
        "abrupt_rise_frac": 0.6,
        "step_persist_z": 3.0,
        "context_window_min": 30.0,
    },
    "patterns": {
        "repeat_x_tol": 2.0,
        "min_repeats_periodic": 4,
        "periodic_cv_max": 0.3,
        "sync_window_s": 120.0,
        "dbscan_eps": 0.8,
        "dbscan_min_samples": 5,
    },
    "evaluation": {
        "match_time_tol_s": 300.0,
        "match_x_tol": 2.0,
        "synthetic": {
            "enabled": True,
            "source": "3.csv",
            "n_events": 40,
            "amplitude_noise_mult": [4.0, 20.0],
            "random_state": 7,
        },
    },
}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None) -> dict:
    """Конфигурация по умолчанию + пользовательские переопределения (если есть)."""
    path = Path(path) if path else USER_CONFIG_PATH
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        cfg = deep_update(cfg, json.loads(path.read_text(encoding="utf-8")))
    return cfg


def save_user_config(overrides: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else USER_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
