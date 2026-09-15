#!/usr/bin/env python3
"""Offline shadow-mode MVP for kiln time-series prediction.

The source parquet is read-only. No controller or DCS integration is present.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


CHEMISTRY_COLUMNS = [
    "出磨生料CaO",
    "出磨生料SiO2",
    "出磨生料Al2O3",
    "出磨生料Fe2O3",
    "出磨生料MgO",
    "出磨生料SO3",
    "出磨生料K2O",
    "出磨生料Na2O",
    "出磨生料Cl",
]

CHEMISTRY_EXPLANATION_COLUMNS = ["出磨生料KH", "出磨生料SM", "出磨生料IM"]

PROCESS_COLUMNS = [
    "二次风温",
    "三次风温",
    "分解炉出口温度A",
    "分解炉出口温度B",
    "窑主传电流",
    "预热器出口O2含量",
    "分解炉出口CO含量",
    "预热器出口压力",
    "篦冷机层压",
    "窑喂料量反馈值",
    "分解炉喂煤量反馈值",
    "窑头喂煤量反馈值",
    "窑速度反馈值",
    "高温风机反馈值",
    "篦冷机速度反馈值",
    "窑头排风机转速反馈值",
    "出磨生料KH",
    "出磨生料SM",
    "出磨生料IM",
]

LABEL_COLUMNS = ["窑况等级", "出窑熟料游离钙"]

# These fields exist in the source file but are not allowed in the base
# predictive contract.  Several are direct target/log/RTO derivatives.
FORBIDDEN_FEATURE_FIELDS = [
    "窑况等级",
    "窑况等级_1",
    "出窑熟料游离钙",
    "窑况日志",
    "窑况日志_1",
    "二次风温趋势",
    "二次风温趋势_1",
    "窑电流趋势",
    "窑电流趋势_1",
    "分解炉出口CO趋势",
    "RTO预热器出口压力目标值推荐_1",
    "RTO窑喂料量目标值推荐_1",
    "RTO分解炉出口目标值推荐_1",
    "RTO窑头秤压力目标值推荐_1",
    "RTO窑喂料量目标值推荐",
    "RTO分解炉出口目标值推荐",
    "RTO窑头秤压力目标值推荐",
    "RTO窑产量目标最终选择",
    "RTO窑头秤压力最终目标选择",
    "RTO分解炉出口温度最终目标选择",
    "RTO预热器出口压力最终目标选择",
    "RTO篦冷机层压最终目标选择",
    "RTO篦冷机层压目标值_1",
]

READ_COLUMNS = list(
    dict.fromkeys(
        ["time"] + CHEMISTRY_COLUMNS + CHEMISTRY_EXPLANATION_COLUMNS + PROCESS_COLUMNS + LABEL_COLUMNS
    )
)

PROCESS_RANGES = {
    "二次风温": (0.0, 1400.0),
    "三次风温": (0.0, 1400.0),
    "分解炉出口温度A": (0.0, 1200.0),
    "分解炉出口温度B": (0.0, 1200.0),
    "窑主传电流": (0.0, 1000.0),
    "预热器出口O2含量": (0.0, 25.0),
    "分解炉出口CO含量": (0.0, 2.0),
    "预热器出口压力": (-7000.0, 500.0),
    "篦冷机层压": (0.0, 200.0),
    "窑喂料量反馈值": (0.0, 500.0),
    "分解炉喂煤量反馈值": (0.0, 30.0),
    "窑头喂煤量反馈值": (0.0, 25.0),
    "窑速度反馈值": (0.0, 5.0),
    "高温风机反馈值": (0.0, 900.0),
    "篦冷机速度反馈值": (0.0, 70.0),
    "窑头排风机转速反馈值": (0.0, 50.0),
}


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["config_dir"] = str(path.parent.resolve())
    return cfg


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    output = Path(cfg["output_dir"])
    if not output.is_absolute():
        output = Path(cfg["config_dir"]) / output
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(parents=True, exist_ok=True)
    return output


def clean_batch(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for column, (lower, upper) in PROCESS_RANGES.items():
        if column in frame:
            bad = (frame[column] < lower) | (frame[column] > upper)
            frame.loc[bad, column] = np.nan
    return frame


def aggregate_complete_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["minute"] = frame["time"].dt.floor("min")
    last_columns = set(CHEMISTRY_COLUMNS + CHEMISTRY_EXPLANATION_COLUMNS + LABEL_COLUMNS)
    agg = {column: ("last" if column in last_columns else "mean") for column in READ_COLUMNS if column != "time"}
    result = frame.groupby("minute", sort=True).agg(agg)
    result.index.name = "time"
    return result


def build_minute_cache(data_path: Path, cache_path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(data_path)
    missing = sorted(set(READ_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Parquet 缺少必要字段: {missing}")

    minute_frames: list[pd.DataFrame] = []
    carry: pd.DataFrame | None = None
    for batch_number, batch in enumerate(
        parquet.iter_batches(columns=READ_COLUMNS, batch_size=65_536), start=1
    ):
        frame = batch.to_pandas()
        if "time" not in frame.columns and frame.index.name == "time":
            frame = frame.reset_index()
        frame = clean_batch(frame)
        frame["time"] = pd.to_datetime(frame["time"])
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        last_minute = frame["time"].iloc[-1].floor("min")
        complete = frame[frame["time"].dt.floor("min") < last_minute]
        carry = frame[frame["time"].dt.floor("min") == last_minute]
        if not complete.empty:
            minute_frames.append(aggregate_complete_minutes(complete))
        if batch_number % 10 == 0:
            print(f"  已处理 {batch_number} 个数据批次")

    if carry is not None and not carry.empty:
        minute_frames.append(aggregate_complete_minutes(carry))
    minute = pd.concat(minute_frames).sort_index()
    minute = minute[~minute.index.duplicated(keep="last")]
    minute.to_parquet(cache_path)
    return minute


def load_or_build_minutes(cfg: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    cache = output_dir / "minute_cache.parquet"
    if cfg.get("reuse_minute_cache", True) and cache.exists():
        print(f"复用一分钟缓存: {cache}")
        return pd.read_parquet(cache)
    print("读取原始 Parquet 并建立一分钟缓存……")
    return build_minute_cache(Path(cfg["data_path"]), cache)


def stable_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["窑喂料量反馈值"] > 300)
        & (frame["窑速度反馈值"] > 3)
        & (frame["窑主传电流"] > 400)
        & (frame["分解炉出口温度A"] > 800)
    )


def fit_material_clusters(
    minute: pd.DataFrame, cfg: dict[str, Any], output_dir: Path, fit_end: pd.Timestamp
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    # Forward filling is causal; the imputer/scaler/K-Means fit is restricted to
    # the training period so no test chemistry can affect cluster identities.
    causal_chemistry = minute[CHEMISTRY_COLUMNS].ffill()
    observed = causal_chemistry.notna().all(axis=1)
    changed = observed & causal_chemistry.ne(causal_chemistry.shift()).any(axis=1)
    fit_events = causal_chemistry.loc[changed & (causal_chemistry.index < fit_end)].copy()
    if len(fit_events) < 3:
        raise ValueError("训练期出磨生料成分变化事件不足，无法聚类")
    imputer = SimpleImputer(strategy="median").fit(fit_events)
    imputed_events = imputer.transform(fit_events)
    scaler = RobustScaler().fit(imputed_events)
    scaled = scaler.transform(imputed_events)

    rng = np.random.default_rng(cfg["random_seed"])
    sample_index = rng.choice(len(scaled), size=min(5_000, len(scaled)), replace=False)
    scores: dict[int, float] = {}
    fitted: dict[int, KMeans] = {}
    max_k = min(int(cfg["cluster_k_max"]), len(fit_events) - 1)
    for k in range(int(cfg["cluster_k_min"]), max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=cfg["random_seed"]).fit(scaled)
        fitted[k] = model
        scores[k] = float(silhouette_score(scaled[sample_index], model.labels_[sample_index]))
    if not scores:
        raise ValueError("训练期事件不足以覆盖配置的聚类数范围")
    best_k = max(scores, key=scores.get)
    model = fitted[best_k]

    transformed = scaler.transform(imputer.transform(causal_chemistry))
    cluster = pd.Series(model.predict(transformed), index=minute.index, name="material_cluster")
    distance = pd.Series(
        np.min(model.transform(transformed), axis=1), index=minute.index, name="material_cluster_distance"
    )
    profile_source = minute.loc[fit_events.index, CHEMISTRY_COLUMNS + CHEMISTRY_EXPLANATION_COLUMNS].copy()
    profile_source["material_cluster"] = model.labels_
    profile = profile_source.groupby("material_cluster").mean(numeric_only=True)
    profile.to_csv(output_dir / "material_cluster_profiles.csv", encoding="utf-8-sig")
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "columns": CHEMISTRY_COLUMNS,
            "fit_end": str(fit_end),
            "pca": None,
        },
        output_dir / "models" / "material_cluster.joblib",
    )
    meta = {
        "selected_k": int(best_k),
        "silhouette_by_k": {str(k): value for k, value in scores.items()},
        "chemistry_change_events": int(len(fit_events)),
        "fit_start": str(fit_events.index.min()),
        "fit_end": str(fit_end),
        "fit_rows": int(len(fit_events)),
        "missing_handling": "causal_ffill_then_training_period_median_imputer",
        "scaler_fit_period": "training_only",
        "pca": None,
        "cluster_centers_fit_period": "training_only",
        "future_fill_used": False,
    }
    return cluster, distance, meta


def build_features(
    minute: pd.DataFrame,
    cluster: pd.Series,
    distance: pd.Series,
    windows: list[int],
) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for column in PROCESS_COLUMNS:
        series = minute[column].astype("float32")
        features[column] = series
        for window in windows:
            rolling = series.rolling(window, min_periods=max(3, window // 2))
            features[f"{column}__mean_{window}m"] = rolling.mean().astype("float32")
            features[f"{column}__std_{window}m"] = rolling.std().astype("float32")
            features[f"{column}__delta_{window}m"] = (series - series.shift(window - 1)).astype(
                "float32"
            )
    frame = pd.DataFrame(features, index=minute.index)
    frame["material_cluster"] = cluster.astype("float32")
    frame["material_cluster_distance"] = distance.astype("float32")
    return frame


def chronological_split(index: pd.Index) -> tuple[pd.Timestamp, pd.Timestamp]:
    ordered = pd.Index(index).sort_values()
    return ordered[int(len(ordered) * 0.70)], ordered[int(len(ordered) * 0.85)]


def regression_pipeline(alpha: float = 10.0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


def classification_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=400, solver="lbfgs", random_state=20260915
                ),
            ),
        ]
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(true_array) & np.isfinite(pred_array)
    true_array = true_array[finite]
    pred_array = pred_array[finite]
    return {
        "mae": float(mean_absolute_error(true_array, pred_array)),
        "rmse": float(math.sqrt(mean_squared_error(true_array, pred_array))),
        "r2": float(r2_score(true_array, pred_array)),
    }


def classification_metrics(
    y_true: pd.Series, y_pred: np.ndarray, labels: list[int], risk_label: int | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    if risk_label is not None:
        result["risk_class_recall"] = float(
            recall_score(y_true, y_pred, labels=[risk_label], average="macro", zero_division=0)
        )
    return result


def assert_feature_contract(features: pd.DataFrame) -> None:
    """Reject target-derived or otherwise forbidden columns at the boundary."""
    allowed_derived = {"material_cluster", "material_cluster_distance"}
    for column in features.columns:
        if column in FORBIDDEN_FEATURE_FIELDS or column in LABEL_COLUMNS:
            raise AssertionError(f"禁止字段进入特征: {column}")
        if column in allowed_derived:
            continue
        base = column.split("__", 1)[0]
        if base not in PROCESS_COLUMNS:
            raise AssertionError(f"未登记的特征字段: {column}")
        if column != base and not any(column.startswith(f"{base}__") for base in PROCESS_COLUMNS):
            raise AssertionError(f"未登记的派生特征: {column}")


def assert_model_feature_contract(
    feature_columns: list[str], allowed_extra: set[str] | None = None
) -> None:
    allowed_extra = allowed_extra or set()
    base_columns = [column for column in feature_columns if column not in allowed_extra]
    assert_feature_contract(pd.DataFrame(columns=base_columns))
    if any(column in FORBIDDEN_FEATURE_FIELDS or column in LABEL_COLUMNS for column in feature_columns):
        forbidden = [column for column in feature_columns if column in FORBIDDEN_FEATURE_FIELDS or column in LABEL_COLUMNS]
        raise AssertionError(f"禁止字段进入模型: {forbidden}")


def future_window_aggregate(
    series: pd.Series, start_minutes: int, end_minutes: int, aggregation: str
) -> pd.Series:
    """Return an explicitly forward-looking target without changing input order.

    Reversing a shifted series makes the rolling operation cover t+start through
    t+end.  ``min_periods`` equal to the window width prevents partial targets.
    """
    if start_minutes < 0 or end_minutes < start_minutes:
        raise ValueError("future window must satisfy 0 <= start <= end")
    width = end_minutes - start_minutes + 1
    shifted = series.shift(-start_minutes)
    reverse = shifted.iloc[::-1]
    rolling = reverse.rolling(width, min_periods=width)
    if aggregation == "mean":
        result = rolling.mean()
    elif aggregation == "min":
        result = rolling.min()
    elif aggregation == "max":
        result = rolling.max()
    else:
        raise ValueError(f"unsupported future aggregation: {aggregation}")
    return result.iloc[::-1].rename(series.name)


def extract_change_events(series: pd.Series) -> pd.Series:
    """Keep the first observed value and subsequent actual value changes only."""
    previous = series.shift()
    event_mask = series.notna() & (previous.isna() | series.ne(previous))
    return series.loc[event_mask]


def assert_no_time_overlap(
    train_index: pd.Index,
    validation_index: pd.Index,
    test_index: pd.Index,
    purge_minutes: int = 0,
) -> None:
    """Check disjoint reference times and a target-horizon purge at boundaries."""
    train = pd.DatetimeIndex(train_index)
    validation = pd.DatetimeIndex(validation_index)
    test = pd.DatetimeIndex(test_index)
    if train.intersection(validation).size or train.intersection(test).size:
        raise AssertionError("时间切分存在样本交叉")
    if validation.intersection(test).size:
        raise AssertionError("验证集和测试集存在样本交叉")
    purge = pd.Timedelta(minutes=purge_minutes)
    if len(train) and len(test) and train.max() + purge >= test.min():
        raise AssertionError("训练标签窗口可能跨入测试期")
    if len(train) and len(validation) and train.max() + purge >= validation.min():
        raise AssertionError("训练标签窗口可能跨入验证期")
    if len(validation) and len(test) and validation.max() + purge >= test.min():
        raise AssertionError("验证标签窗口可能跨入测试期")


def time_split_masks(
    valid: pd.Series, horizon_minutes: int
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    index = pd.DatetimeIndex(valid.index)
    valid_index = index[valid.to_numpy()]
    train_boundary, validation_boundary = chronological_split(valid_index)
    purge = pd.Timedelta(minutes=horizon_minutes)
    train = valid & (index < train_boundary - purge)
    validation = valid & (index >= train_boundary) & (index < validation_boundary - purge)
    test = valid & (index >= validation_boundary)
    assert_no_time_overlap(index[train], index[validation], index[test], horizon_minutes)
    metadata = {
        "train_boundary": str(train_boundary),
        "validation_boundary": str(validation_boundary),
        "purge_minutes": int(horizon_minutes),
    }
    return train, validation, test, metadata


def sample_metadata(
    index: pd.Index,
    target: pd.Series | None = None,
    stride_rows: int | None = None,
    categorical: bool = False,
) -> dict[str, Any]:
    ordered = pd.DatetimeIndex(index).sort_values()
    result: dict[str, Any] = {"rows": int(len(ordered))}
    if len(ordered):
        result.update({"start": str(ordered[0]), "end": str(ordered[-1])})
    if stride_rows is not None:
        result["rows_after_stride"] = int(len(ordered[::stride_rows]))
    if target is not None and len(ordered):
        values = target.loc[ordered].dropna()
        if categorical:
            result["label_distribution"] = {
                str(key): int(value) for key, value in values.value_counts().sort_index().items()
            }
        elif len(values):
            result["label_summary"] = {
                "non_null": int(len(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
    return result


def weekly_periods(index: pd.Index) -> list[pd.Period]:
    periods = pd.PeriodIndex(pd.DatetimeIndex(index).to_period("W-SUN")).unique()
    return list(periods.sort_values())


def weekly_regression_validation(
    features: pd.DataFrame,
    target: pd.Series,
    baseline: pd.Series,
    valid: pd.Series,
    horizon_minutes: int,
    stride: int,
    alpha: float,
    min_train_weeks: int,
) -> dict[str, Any]:
    valid_index = pd.DatetimeIndex(target.index[valid & target.notna() & baseline.notna()]).sort_values()
    folds: list[dict[str, Any]] = []
    periods = weekly_periods(valid_index)
    for fold_number, test_week in enumerate(periods[min_train_weeks:], start=1):
        test_start = pd.Timestamp(test_week.start_time)
        test_end = pd.Timestamp(test_week.end_time)
        test_index = valid_index[(valid_index >= test_start) & (valid_index <= test_end)]
        train_index = valid_index[valid_index < test_start - pd.Timedelta(minutes=horizon_minutes)]
        if len(train_index) < 2 or len(test_index) == 0:
            continue
        assert_no_time_overlap(train_index[::stride], pd.DatetimeIndex([]), test_index, horizon_minutes)
        train_fit_index = train_index[::stride]
        model = regression_pipeline(alpha=alpha)
        model.fit(features.loc[train_fit_index], target.loc[train_fit_index])
        model_pred = model.predict(features.loc[test_index])
        model_metrics = regression_metrics(target.loc[test_index], model_pred)
        baseline_metrics = regression_metrics(target.loc[test_index], baseline.loc[test_index].to_numpy())
        folds.append(
            {
                "fold": int(fold_number),
                "test_week": str(test_week),
                "purge_minutes": int(horizon_minutes),
                "train": sample_metadata(train_index, target, stride),
                "test": sample_metadata(test_index, target),
                "baseline": baseline_metrics,
                "model": model_metrics,
                "model_beats_baseline": bool(model_metrics["mae"] < baseline_metrics["mae"]),
            }
        )
    model_mae = [fold["model"]["mae"] for fold in folds]
    baseline_mae = [fold["baseline"]["mae"] for fold in folds]
    return {
        "folds": folds,
        "summary": {
            "fold_count": int(len(folds)),
            "mean_model_mae": float(np.mean(model_mae)) if model_mae else None,
            "mean_baseline_mae": float(np.mean(baseline_mae)) if baseline_mae else None,
            "mean_improvement": (
                float(1.0 - np.mean(model_mae) / np.mean(baseline_mae))
                if model_mae and np.mean(baseline_mae) > 0
                else None
            ),
            "all_folds_model_beats_baseline": bool(
                folds and all(fold["model_beats_baseline"] for fold in folds)
            ),
        },
    }


def weekly_classification_validation(
    features: pd.DataFrame,
    target: pd.Series,
    valid: pd.Series,
    horizon_minutes: int,
    stride: int,
    min_train_weeks: int,
    labels: list[int],
    baseline: pd.Series | None = None,
    risk_label: int | None = None,
) -> dict[str, Any]:
    valid_index = pd.DatetimeIndex(target.index[valid & target.notna()]).sort_values()
    if baseline is not None:
        valid_index = valid_index[baseline.loc[valid_index].notna()]
    folds: list[dict[str, Any]] = []
    periods = weekly_periods(valid_index)
    for fold_number, test_week in enumerate(periods[min_train_weeks:], start=1):
        test_start = pd.Timestamp(test_week.start_time)
        test_end = pd.Timestamp(test_week.end_time)
        test_index = valid_index[(valid_index >= test_start) & (valid_index <= test_end)]
        train_index = valid_index[valid_index < test_start - pd.Timedelta(minutes=horizon_minutes)]
        if len(train_index) < 2 or len(test_index) == 0:
            continue
        train_fit_index = train_index[::stride]
        if target.loc[train_fit_index].nunique(dropna=True) < 2:
            folds.append(
                {
                    "fold": int(fold_number),
                    "test_week": str(test_week),
                    "status": "skipped_insufficient_train_classes",
                    "train": sample_metadata(train_index, target, stride, categorical=True),
                    "test": sample_metadata(test_index, target, categorical=True),
                }
            )
            continue
        assert_no_time_overlap(train_fit_index, pd.DatetimeIndex([]), test_index, horizon_minutes)
        model = classification_pipeline()
        model.fit(features.loc[train_fit_index], target.loc[train_fit_index].astype(int))
        model_pred = model.predict(features.loc[test_index])
        model_metrics = classification_metrics(target.loc[test_index].astype(int), model_pred, labels, risk_label)
        if baseline is None:
            majority = int(target.loc[train_fit_index].mode().iloc[0])
            baseline_pred = np.full(len(test_index), majority, dtype=int)
            baseline_name = "train_majority"
        else:
            baseline_pred = baseline.loc[test_index].astype(int).to_numpy()
            baseline_name = "persistence"
        baseline_metrics = classification_metrics(
            target.loc[test_index].astype(int), baseline_pred, labels, risk_label
        )
        folds.append(
            {
                "fold": int(fold_number),
                "test_week": str(test_week),
                "purge_minutes": int(horizon_minutes),
                "train": sample_metadata(train_index, target, stride, categorical=True),
                "test": sample_metadata(test_index, target, categorical=True),
                "baseline_name": baseline_name,
                "baseline": baseline_metrics,
                "model": model_metrics,
                "model_beats_baseline": bool(model_metrics["macro_f1"] > baseline_metrics["macro_f1"]),
            }
        )
    completed = [fold for fold in folds if fold.get("status") != "skipped_insufficient_train_classes"]
    macro = [fold["model"]["macro_f1"] for fold in completed]
    baseline_macro = [fold["baseline"]["macro_f1"] for fold in completed]
    summary: dict[str, Any] = {
        "fold_count": int(len(completed)),
        "skipped_fold_count": int(len(folds) - len(completed)),
        "mean_model_macro_f1": float(np.mean(macro)) if macro else None,
        "mean_baseline_macro_f1": float(np.mean(baseline_macro)) if baseline_macro else None,
        "all_folds_model_beats_baseline": bool(
            completed and all(fold["model_beats_baseline"] for fold in completed)
        ),
    }
    if risk_label is not None:
        recalls = [fold["model"].get("risk_class_recall") for fold in completed]
        summary["mean_model_risk_class_recall"] = float(np.mean(recalls)) if recalls else None
    return {"folds": folds, "summary": summary}


def fit_time_series_models(
    minute: pd.DataFrame,
    features: pd.DataFrame,
    stable: pd.Series,
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    start = int(cfg["temperature_future_start_minutes"])
    end = int(cfg["temperature_future_end_minutes"])
    future_temp = future_window_aggregate(minute["二次风温"], start, end, "mean")
    recent_temp = minute["二次风温"].rolling(10, min_periods=5).mean()
    delta_temp = future_temp - recent_temp
    deadband = float(cfg["temperature_trend_deadband_c"])
    temp_trend = pd.Series(np.nan, index=minute.index, dtype="float64")
    temp_trend.loc[delta_temp.notna()] = np.select(
        [delta_temp < -deadband, delta_temp > deadband], [0, 2], default=1
    )[delta_temp.notna()]
    future_stable = stable.shift(-end).fillna(False).astype(bool)
    valid_temp = stable & future_stable & future_temp.notna() & recent_temp.notna()
    train_mask, val_mask, test_mask, split_meta = time_split_masks(valid_temp, end)
    stride = int(cfg["train_stride_minutes"])
    train_rows = np.flatnonzero(train_mask.to_numpy())[::stride]
    val_rows = np.flatnonzero(val_mask.to_numpy())
    test_rows = np.flatnonzero(test_mask.to_numpy())
    if not len(train_rows) or not len(test_rows):
        raise ValueError("二次风温时间切分后训练集或测试集为空")

    temp_model = regression_pipeline(alpha=20.0)
    temp_model.fit(features.iloc[train_rows], future_temp.iloc[train_rows])
    temp_pred = temp_model.predict(features.iloc[test_rows])
    temp_metrics = regression_metrics(future_temp.iloc[test_rows], temp_pred)
    persistence = regression_metrics(
        future_temp.iloc[test_rows], recent_temp.iloc[test_rows].to_numpy()
    )
    temp_metrics["persistence_baseline_mae"] = persistence["mae"]
    temp_metrics["train_rows_after_stride"] = int(len(train_rows))
    temp_metrics["validation_rows"] = int(len(val_rows))
    temp_metrics["test_rows"] = int(len(test_rows))
    temp_metrics["test_start"] = str(minute.index[test_rows[0]])
    temp_metrics["test_end"] = str(minute.index[test_rows[-1]])
    temp_metrics["target_definition"] = {
        "name": "future_secondary_air_temperature_mean",
        "window_start_minutes": start,
        "window_end_minutes": end,
        "formula": "mean(y[t+25:t+35])",
        "eligibility": "stable_at_t_and_at_window_end",
    }
    temp_metrics["primary_split"] = {
        **split_meta,
        "train": sample_metadata(minute.index[train_mask], future_temp, stride),
        "validation": sample_metadata(minute.index[val_mask], future_temp),
        "test": sample_metadata(minute.index[test_mask], future_temp),
    }
    temp_metrics["weekly_rolling_validation"] = weekly_regression_validation(
        features,
        future_temp,
        recent_temp,
        valid_temp,
        end,
        stride,
        alpha=20.0,
        min_train_weeks=int(cfg.get("weekly_min_train_weeks", 4)),
    )

    trend_valid = valid_temp & temp_trend.notna()
    trend_train, trend_val, trend_test, trend_split_meta = time_split_masks(trend_valid, end)
    trend_train_rows = np.flatnonzero(trend_train.to_numpy())[::stride]
    trend_val_rows = np.flatnonzero(trend_val.to_numpy())
    trend_test_rows = np.flatnonzero(trend_test.to_numpy())
    trend_model = classification_pipeline()
    trend_model.fit(features.iloc[trend_train_rows], temp_trend.iloc[trend_train_rows].astype(int))
    trend_pred = trend_model.predict(features.iloc[trend_test_rows])
    trend_metrics = classification_metrics(
        temp_trend.iloc[trend_test_rows].astype(int), trend_pred, labels=[0, 1, 2]
    )
    trend_metrics.update(
        {
            "validation_rows": int(len(trend_val_rows)),
            "test_rows": int(len(trend_test_rows)),
            "target_definition": {
                "name": "future_temperature_trend_vs_recent_10m_mean",
                "window_start_minutes": start,
                "window_end_minutes": end,
                "deadband_c": deadband,
                "class_mapping": {"0": "下降", "1": "稳定", "2": "上升"},
            },
            "primary_split": {
                **trend_split_meta,
                "train": sample_metadata(minute.index[trend_train], temp_trend, stride, categorical=True),
                "validation": sample_metadata(minute.index[trend_val], temp_trend, categorical=True),
                "test": sample_metadata(minute.index[trend_test], temp_trend, categorical=True),
            },
            "weekly_rolling_validation": weekly_classification_validation(
                features,
                temp_trend,
                trend_valid,
                end,
                stride,
                int(cfg.get("weekly_min_train_weeks", 4)),
                labels=[0, 1, 2],
            ),
        }
    )

    grade = minute["窑况等级"].round().ffill()
    mapped_grade = grade.map({1.0: 0, 2.0: 0, 3.0: 1, 4.0: 2, 5.0: 2})
    horizon = int(cfg["kiln_state_horizon_minutes"])
    state_start = int(cfg.get("kiln_state_label_start_minutes", start))
    state_end = int(cfg.get("kiln_state_label_end_minutes", end))
    future_grade_worst = future_window_aggregate(mapped_grade, state_start, state_end, "min")
    future_grade_center = mapped_grade.shift(-horizon)

    def evaluate_kiln_variant(
        target: pd.Series, label_name: str, purge_minutes: int
    ) -> tuple[dict[str, Any], Pipeline]:
        valid_grade = stable & mapped_grade.notna() & target.notna()
        grade_train, grade_val, grade_test, grade_split_meta = time_split_masks(valid_grade, purge_minutes)
        grade_train_rows = np.flatnonzero(grade_train.to_numpy())[::stride]
        grade_val_rows = np.flatnonzero(grade_val.to_numpy())
        grade_test_rows = np.flatnonzero(grade_test.to_numpy())
        if not len(grade_train_rows) or not len(grade_test_rows):
            raise ValueError(f"窑况标签 {label_name} 时间切分后训练集或测试集为空")
        kiln_model = classification_pipeline()
        kiln_model.fit(features.iloc[grade_train_rows], target.iloc[grade_train_rows].astype(int))
        kiln_pred = kiln_model.predict(features.iloc[grade_test_rows])
        kiln_metrics = classification_metrics(
            target.iloc[grade_test_rows].astype(int), kiln_pred, labels=[0, 1, 2], risk_label=0
        )
        kiln_metrics.update(
            {
                "validation_rows": int(len(grade_val_rows)),
                "test_rows": int(len(grade_test_rows)),
                "test_start": str(minute.index[grade_test_rows[0]]),
                "test_end": str(minute.index[grade_test_rows[-1]]),
                "target_definition": {
                    "name": label_name,
                    "class_mapping": {"0": "差", "1": "中", "2": "好"},
                    "future_window_start_minutes": state_start if label_name == "window_worst" else None,
                    "future_window_end_minutes": state_end if label_name == "window_worst" else None,
                    "center_horizon_minutes": horizon if label_name == "center_30m" else None,
                    "aggregation": "minimum_over_future_window" if label_name == "window_worst" else "center_timestamp",
                    "label_fill": "causal_ffill_after_minute_last_observation",
                },
                "primary_split": {
                    **grade_split_meta,
                    "train": sample_metadata(minute.index[grade_train], target, stride, categorical=True),
                    "validation": sample_metadata(minute.index[grade_val], target, categorical=True),
                    "test": sample_metadata(minute.index[grade_test], target, categorical=True),
                },
                "persistence_baseline": classification_metrics(
                    target.iloc[grade_test_rows].astype(int),
                    mapped_grade.iloc[grade_test_rows].astype(int).to_numpy(),
                    labels=[0, 1, 2],
                    risk_label=0,
                ),
                "weekly_rolling_validation": weekly_classification_validation(
                    features,
                    target,
                    valid_grade,
                    purge_minutes,
                    stride,
                    int(cfg.get("weekly_min_train_weeks", 4)),
                    labels=[0, 1, 2],
                    baseline=mapped_grade,
                    risk_label=0,
                ),
            }
        )
        return kiln_metrics, kiln_model

    kiln_metrics, kiln_model = evaluate_kiln_variant(
        future_grade_worst, "window_worst", state_end
    )
    center_metrics, center_model = evaluate_kiln_variant(future_grade_center, "center_30m", horizon)

    joblib.dump(temp_model, output_dir / "models" / "secondary_air_temperature.joblib")
    joblib.dump(trend_model, output_dir / "models" / "secondary_air_trend.joblib")
    joblib.dump(kiln_model, output_dir / "models" / "kiln_state.joblib")
    joblib.dump(center_model, output_dir / "models" / "kiln_state_center_30m.joblib")

    tail = pd.DataFrame(
        {
            "actual_future_secondary_air_temp": future_temp.iloc[test_rows],
            "predicted_future_secondary_air_temp": temp_pred,
            "actual_temp_trend": temp_trend.iloc[test_rows],
            "predicted_temp_trend": trend_pred,
        },
        index=minute.index[test_rows],
    ).tail(2_000)
    return (
        {
            "secondary_air_temperature": temp_metrics,
            "secondary_air_trend": trend_metrics,
            "kiln_state": kiln_metrics,
            "kiln_state_center": center_metrics,
        },
        {
            "temp": temp_model,
            "trend": trend_model,
            "kiln": kiln_model,
            "kiln_center": center_model,
        },
        tail,
    )


def fit_fcao_models(
    minute: pd.DataFrame,
    features: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fcao = minute["出窑熟料游离钙"]
    event_series = extract_change_events(fcao)
    events = pd.DataFrame({"fcao": event_series})
    events["previous_fcao"] = events["fcao"].shift(1)
    events["delta_fcao"] = events["fcao"] - events["previous_fcao"]
    deadband = float(cfg["fcao_trend_deadband"])
    events["trend"] = np.select(
        [events["delta_fcao"] < -deadband, events["delta_fcao"] > deadband], [0, 2], default=1
    )
    delay = pd.Timedelta(minutes=int(cfg["fcao_process_delay_minutes"]))
    feature_times = events.index - delay
    # ``pad`` is deliberately used instead of nearest: nearest can select a
    # minute after the causal feature time and leak future process values.
    positions = features.index.get_indexer(
        feature_times, method="pad", tolerance=pd.Timedelta("1min")
    )
    keep = positions >= 0
    events = events.iloc[np.flatnonzero(keep)].copy()
    positions = positions[keep]
    x = features.iloc[positions].copy()
    x.index = events.index
    x["previous_fcao"] = events["previous_fcao"]
    valid = events["previous_fcao"].notna()
    events = events[valid]
    x = x.loc[events.index]
    assert_model_feature_contract(list(x.columns), allowed_extra={"previous_fcao"})
    valid_events = pd.Series(True, index=events.index)
    train_mask, validation_mask, test_mask, split_meta = time_split_masks(valid_events, 0)

    reg = regression_pipeline(alpha=10.0)
    reg.fit(x.loc[train_mask.index[train_mask]], events.loc[train_mask.index[train_mask], "fcao"])
    reg_pred = reg.predict(x.loc[test_mask.index[test_mask]])
    reg_metrics = regression_metrics(
        events.loc[test_mask.index[test_mask], "fcao"], reg_pred
    )
    reg_metrics["persistence_baseline_mae"] = float(
        mean_absolute_error(
            events.loc[test_mask.index[test_mask], "fcao"],
            events.loc[test_mask.index[test_mask], "previous_fcao"],
        )
    )
    reg_metrics["event_samples"] = int(len(events))
    reg_metrics["test_events"] = int(test_mask.sum())
    reg_metrics["feature_time_alignment"] = "pad_at_event_time_minus_process_delay"
    reg_metrics["process_delay_minutes"] = int(cfg["fcao_process_delay_minutes"])
    reg_metrics["event_definition"] = (
        "first_observed_value_and_subsequent_actual_value_changes_only"
    )
    reg_metrics["primary_split"] = {
        **split_meta,
        "train": sample_metadata(train_mask.index[train_mask], events["fcao"]),
        "validation": sample_metadata(validation_mask.index[validation_mask], events["fcao"]),
        "test": sample_metadata(test_mask.index[test_mask], events["fcao"]),
    }
    reg_metrics["weekly_rolling_validation"] = weekly_regression_validation(
        x,
        events["fcao"],
        events["previous_fcao"],
        valid_events,
        0,
        1,
        alpha=10.0,
        min_train_weeks=int(cfg.get("weekly_min_train_weeks", 4)),
    )

    cls = classification_pipeline()
    cls.fit(
        x.loc[train_mask.index[train_mask]],
        events.loc[train_mask.index[train_mask], "trend"].astype(int),
    )
    cls_pred = cls.predict(x.loc[test_mask.index[test_mask]])
    cls_metrics = classification_metrics(
        events.loc[test_mask.index[test_mask], "trend"].astype(int), cls_pred, labels=[0, 1, 2]
    )
    cls_metrics.update(
        {
            "process_delay_minutes": int(cfg["fcao_process_delay_minutes"]),
            "event_definition": "previous_event_delta_with_deadband",
            "deadband": deadband,
            "primary_split": {
                **split_meta,
                "train": sample_metadata(train_mask.index[train_mask], events["trend"], categorical=True),
                "validation": sample_metadata(validation_mask.index[validation_mask], events["trend"], categorical=True),
                "test": sample_metadata(test_mask.index[test_mask], events["trend"], categorical=True),
            },
            "weekly_rolling_validation": weekly_classification_validation(
                x,
                events["trend"],
                valid_events,
                0,
                1,
                int(cfg.get("weekly_min_train_weeks", 4)),
                labels=[0, 1, 2],
            ),
        }
    )

    joblib.dump(reg, output_dir / "models" / "fcao_regression.joblib")
    joblib.dump(cls, output_dir / "models" / "fcao_trend.joblib")
    joblib.dump(list(x.columns), output_dir / "models" / "fcao_feature_columns.joblib")
    return {"fcao_regression": reg_metrics, "fcao_trend": cls_metrics}, {"reg": reg, "trend": cls}


def model_promotion_status(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the stated acceptance gates without changing labels or test scope."""
    temp_weekly = metrics["secondary_air_temperature"]["weekly_rolling_validation"]["summary"]
    trend_weekly = metrics["secondary_air_trend"]["weekly_rolling_validation"]["summary"]
    kiln_weekly = metrics["kiln_state"]["weekly_rolling_validation"]["summary"]
    fcao_weekly = metrics["fcao_regression"]["weekly_rolling_validation"]["summary"]

    temperature_passed = bool(
        temp_weekly["fold_count"] > 0
        and temp_weekly["all_folds_model_beats_baseline"]
        and temp_weekly["mean_improvement"] is not None
        and temp_weekly["mean_improvement"] >= 0.10
    )
    trend_passed = bool(
        trend_weekly["fold_count"] > 0
        and trend_weekly["mean_model_macro_f1"] is not None
        and trend_weekly["mean_model_macro_f1"] >= 0.60
    )
    kiln_passed = bool(
        kiln_weekly["fold_count"] > 0
        and kiln_weekly.get("mean_model_risk_class_recall") is not None
        and kiln_weekly["mean_model_risk_class_recall"] >= 0.70
    )
    fcao_primary = metrics["fcao_regression"]
    fcao_passed = bool(
        fcao_weekly["fold_count"] > 0
        and fcao_weekly["all_folds_model_beats_baseline"]
        and fcao_primary["mae"] < fcao_primary["persistence_baseline_mae"]
    )
    statuses = {
        "secondary_air_temperature": {
            "passed": temperature_passed,
            "status": "promoted" if temperature_passed else "not_promoted",
            "reason": "all_weekly_folds_and_mean_improvement_ge_10pct"
            if temperature_passed
            else "weekly_or_10pct_gate_failed",
        },
        "secondary_air_trend": {
            "passed": trend_passed,
            "status": "promoted" if trend_passed else "not_promoted",
            "reason": "weekly_mean_macro_f1_ge_0.60"
            if trend_passed
            else "weekly_mean_macro_f1_gate_failed",
        },
        "kiln_state": {
            "passed": kiln_passed,
            "status": "promoted" if kiln_passed else "not_promoted",
            "reason": "weekly_mean_bad_recall_ge_0.70"
            if kiln_passed
            else "weekly_mean_bad_recall_gate_failed",
        },
        "fcao": {
            "passed": fcao_passed,
            "status": "promoted" if fcao_passed else "not_promoted",
            "reason": "all_weekly_folds_beat_persistence_and_primary_test_beats_it"
            if fcao_passed
            else "baseline_or_weekly_gate_failed",
        },
    }
    return {
        **statuses,
        "objective_direction_allowed": bool(all(item["passed"] for item in statuses.values())),
        "actuator_recommendation_allowed": False,
    }


def objective_direction_for_prediction(
    predicted_temperature: float, target: float, deadband: float, promoted: bool
) -> str | None:
    if not promoted:
        return None
    if predicted_temperature < target - deadband:
        return "需要升高二次风温"
    if predicted_temperature > target + deadband:
        return "需要降低二次风温"
    return "保持二次风温"


def build_direction_signal(
    predicted_temperature: float, target: float, deadband: float, promoted: bool
) -> dict[str, Any]:
    return {
        "objective_direction": objective_direction_for_prediction(
            predicted_temperature, target, deadband, promoted
        ),
        "actuator_recommendation": None,
    }


def latest_shadow_signal(
    minute: pd.DataFrame,
    features: pd.DataFrame,
    stable: pd.Series,
    models: dict[str, Any],
    fcao_models: dict[str, Any],
    cfg: dict[str, Any],
    validation_status: dict[str, Any],
) -> dict[str, Any]:
    latest_index = minute.index[stable & features.notna().any(axis=1)][-1]
    x = features.loc[[latest_index]].copy()
    predicted_temp = float(models["temp"].predict(x)[0])
    trend_class = int(models["trend"].predict(x)[0])
    trend_name = {0: "下降", 1: "稳定", 2: "上升"}[trend_class]
    kiln_name_map = {0: "差", 1: "中", 2: "好"}
    worst_model = models["kiln"]
    center_model = models.get("kiln_center", worst_model)
    worst_name = kiln_name_map[int(worst_model.predict(x)[0])]
    center_name = kiln_name_map[int(center_model.predict(x)[0])]

    def probability_map(model: Pipeline) -> dict[str, float]:
        probabilities = model.predict_proba(x)[0]
        class_order = model.named_steps["model"].classes_
        return {
            kiln_name_map[int(label)]: float(probability)
            for label, probability in zip(class_order, probabilities)
        }

    center_probability_map = probability_map(center_model)
    worst_probability_map = probability_map(worst_model)
    target = float(cfg["secondary_air_target_c"])
    deadband = float(cfg["temperature_trend_deadband_c"])
    objective_allowed = bool(validation_status.get("objective_direction_allowed", False))
    direction_signal = build_direction_signal(predicted_temp, target, deadband, objective_allowed)

    previous_fcao = float(minute.loc[:latest_index, "出窑熟料游离钙"].dropna().iloc[-1])
    fcao_x = x.copy()
    fcao_x["previous_fcao"] = previous_fcao
    predicted_fcao = float(fcao_models["reg"].predict(fcao_x)[0])
    fcao_trend = int(fcao_models["trend"].predict(fcao_x)[0])
    return {
        "timestamp": str(latest_index),
        "mode": "offline_shadow_only",
        "predicted_secondary_air_temperature_25_to_35_min_mean": predicted_temp,
        "secondary_air_temperature_target": target,
        "temperature_trend": trend_name,
        "predicted_kiln_state_30_min": center_name,
        "predicted_kiln_state_25_to_35_min_worst": worst_name,
        "kiln_state_probabilities": center_probability_map,
        "kiln_state_probabilities_25_to_35_min_worst": worst_probability_map,
        "predicted_fcao_after_configured_process_delay": predicted_fcao,
        "fcao_trend": {0: "下降", 1: "稳定", 2: "上升"}[fcao_trend],
        "objective_direction": direction_signal["objective_direction"],
        "actuator_recommendation": direction_signal["actuator_recommendation"],
        "objective_direction_status": "available" if objective_allowed else "blocked_model_not_promoted",
        "validation_status": validation_status,
        "safety_note": "尚未完成操纵量动态辨识；不得据此调整煤量、窑速或风机。",
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = sha256_file(path)
    return result


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def audit_source_provenance(data_path: Path) -> dict[str, Any]:
    """Audit source fields that could make the kiln label a rule/log replay."""
    parquet = pq.ParquetFile(data_path)
    names = set(parquet.schema_arrow.names)
    present_forbidden = sorted(names.intersection(FORBIDDEN_FEATURE_FIELDS))
    audit: dict[str, Any] = {
        "forbidden_source_fields_present": present_forbidden,
        "excluded_from_base_features": sorted(set(FORBIDDEN_FEATURE_FIELDS)),
        "kiln_label_provenance": "not_independent_if_log_grade_matches_label",
    }
    columns = [
        column
        for column in ["窑况等级", "窑况等级_1", "窑况日志"]
        if column in names
    ]
    counts = {"label_log_common_nonnull": 0, "label_log_grade_equal": 0}
    duplicate_counts = {"common_nonnull": 0, "equal": 0, "mismatch": 0}
    if columns:
        for batch in parquet.iter_batches(columns=columns, batch_size=200_000):
            frame = batch.to_pandas()
            if "窑况日志" in frame:
                log = frame["窑况日志"].astype("string")
                parsed = pd.to_numeric(
                    log.str.extract(r"等级\s*(\d+)", expand=False), errors="coerce"
                )
                if "窑况等级" in frame:
                    common = frame["窑况等级"].notna() & parsed.notna()
                    counts["label_log_common_nonnull"] += int(common.sum())
                    counts["label_log_grade_equal"] += int(
                        (frame.loc[common, "窑况等级"] == parsed.loc[common]).sum()
                    )
            if "窑况等级" in frame and "窑况等级_1" in frame:
                common = frame["窑况等级"].notna() & frame["窑况等级_1"].notna()
                duplicate_counts["common_nonnull"] += int(common.sum())
                duplicate_counts["equal"] += int(
                    (frame.loc[common, "窑况等级"] == frame.loc[common, "窑况等级_1"]).sum()
                )
                duplicate_counts["mismatch"] += int(
                    (frame.loc[common, "窑况等级"] != frame.loc[common, "窑况等级_1"]).sum()
                )
    audit["kiln_log_grade_comparison"] = counts
    audit["duplicate_kiln_grade_comparison"] = duplicate_counts
    audit["interpretation"] = (
        "窑况日志中的等级与窑况等级完全一致，故窑况任务至少包含日志/规则复现成分；"
        "RTO、日志和趋势字段未进入基础特征。"
    )
    return audit


def write_run_manifest(
    manifest_path: Path,
    config_path: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    audit: dict[str, Any],
    data_summary: dict[str, Any],
    feature_columns: list[str],
    cluster_meta: dict[str, Any],
    promotion_status: dict[str, Any],
    started_at: float,
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    code_paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        project_root / "README.md",
        project_root / "LUNA_HANDOFF.md",
        project_root / "requirements.txt",
    ]
    code_hashes = {
        str(path): file_metadata(path)
        for path in code_paths
        if path.exists() and path.is_file()
    }
    run_root = config_path.parent
    generated_files = [
        file_metadata(path)
        for path in sorted(run_root.rglob("*"))
        if path.is_file() and path.resolve() != manifest_path.resolve()
    ]
    manifest = {
        "stage": "A",
        "status": "REVIEW_REQUIRED",
        "run_id": output_dir.parent.name,
        "started_at_epoch": started_at,
        "completed_at_epoch": time.time(),
        "runtime_seconds": float(time.time() - started_at),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib"]
            },
        },
        "source_input": {
            "before": source_before,
            "after": source_after,
            "unchanged": source_before == source_after,
            "parquet_schema_columns": sorted(pq.ParquetFile(Path(cfg["data_path"])).schema_arrow.names),
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
            "values": {key: value for key, value in cfg.items() if key != "config_dir"},
        },
        "code_hashes": code_hashes,
        "feature_contract": {
            "columns_ordered": feature_columns,
            "base_process_columns": PROCESS_COLUMNS,
            "fcao_history_columns": ["previous_fcao"],
            "forbidden_fields": sorted(set(FORBIDDEN_FEATURE_FIELDS)),
            "feature_count": int(len(feature_columns)),
        },
        "cluster_fit": cluster_meta,
        "data_summary": data_summary,
        "source_provenance_audit": audit,
        "promotion_status": promotion_status,
        "generated_files": generated_files,
    }
    manifest_path.write_text(
        json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_html_report(
    output_dir: Path, metrics: dict[str, Any], signal: dict[str, Any], data_summary: dict[str, Any]
) -> None:
    def metric_cards(section: dict[str, Any]) -> str:
        cards = []
        for key, value in section.items():
            if isinstance(value, (int, float)):
                shown = f"{value:.4f}" if isinstance(value, float) else str(value)
                cards.append(f"<div class='card'><span>{html.escape(key)}</span><strong>{shown}</strong></div>")
        return "".join(cards)

    metric_sections = "".join(
        f"<h3>{html.escape(name)}</h3><div class='grid'>{metric_cards(section)}</div>"
        for name, section in metrics.items()
    )
    signal_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))}</td></tr>"
        for key, value in signal.items()
    )
    summary_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in data_summary.items()
    )
    page = f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>窑系统预测 MVP</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f6f8;color:#18212b;margin:0}}
main{{max-width:1100px;margin:32px auto;padding:0 20px 60px}}h1{{margin-bottom:6px}}.sub{{color:#5d6875}}
.notice{{background:#fff4d6;border-left:5px solid #e3a008;padding:14px 16px;margin:24px 0;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.card{{background:white;border-radius:12px;padding:16px;box-shadow:0 2px 12px #14213d12}}
.card span{{display:block;color:#66717e;font-size:13px;margin-bottom:8px}}.card strong{{font-size:24px}}
section{{background:white;border-radius:14px;padding:22px;margin-top:18px;box-shadow:0 2px 12px #14213d10}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #e9edf2;vertical-align:top}}th{{width:38%;color:#52606d}}
</style></head><body><main>
<h1>窑系统预测 MVP</h1><p class='sub'>离线影子模式 · 不连接 DCS · 不下发控制动作</p>
<div class='notice'>当前结果用于验证数据和建模链路是否可运行，不代表现场可执行的控制策略。</div>
<section><h2>最新影子信号</h2><table>{signal_rows}</table></section>
<section><h2>数据摘要</h2><table>{summary_rows}</table></section>
<section><h2>时间测试集指标</h2>{metric_sections}</section>
</main></body></html>"""
    (output_dir / "report.html").write_text(page, encoding="utf-8")


def assert_output_safe(data_path: Path, output_dir: Path) -> None:
    source = data_path.resolve()
    output = output_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError("输出目录不能是原始 Parquet 或其子目录")


def main() -> None:
    started_at = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    output_dir = resolve_output_dir(cfg)
    source_path = Path(cfg["data_path"]).resolve()
    assert_output_safe(source_path, output_dir)
    source_before = file_metadata(source_path)
    source_audit = audit_source_provenance(source_path)

    minute = load_or_build_minutes(cfg, output_dir)
    minute = minute.sort_index()
    stable = stable_mask(minute)
    print(f"一分钟记录: {len(minute):,}; 稳定工况: {stable.sum():,} ({stable.mean():.1%})")

    print("拟合原料成分聚类……")
    ordered_index = pd.DatetimeIndex(minute.index).sort_values()
    cluster_boundary = ordered_index[int(len(ordered_index) * 0.70)]
    max_target_horizon = max(
        int(cfg["temperature_future_end_minutes"]),
        int(cfg.get("kiln_state_label_end_minutes", cfg["temperature_future_end_minutes"])),
    )
    cluster_fit_end = cluster_boundary - pd.Timedelta(minutes=max_target_horizon)
    cluster, distance, cluster_meta = fit_material_clusters(
        minute, cfg, output_dir, cluster_fit_end
    )
    print(f"选定原料聚类数: {cluster_meta['selected_k']}")

    print("构造因果滑窗特征……")
    features = build_features(minute, cluster, distance, cfg["feature_windows_minutes"])
    assert_feature_contract(features)
    feature_contract = {
        "columns_ordered": list(features.columns),
        "base_process_columns": PROCESS_COLUMNS,
        "derived_features": [column for column in features.columns if "__" in column],
        "forbidden_fields": sorted(set(FORBIDDEN_FEATURE_FIELDS)),
        "missing_handling": "model_pipeline_training_median_imputer",
        "rolling_window_causality": "current_and_prior_rows_only",
    }
    (output_dir / "feature_contract.json").write_text(
        json.dumps(to_jsonable(feature_contract), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(list(features.columns), output_dir / "models" / "feature_columns.joblib")

    print("训练二次风温、趋势和窑况模型……")
    time_metrics, time_models, prediction_tail = fit_time_series_models(
        minute, features, stable, cfg, output_dir
    )

    print("训练游离钙事件模型……")
    fcao_metrics, fcao_models = fit_fcao_models(minute, features, cfg, output_dir)
    metrics = {**time_metrics, **fcao_metrics, "material_clustering": cluster_meta}
    promotion_status = model_promotion_status(metrics)

    signal = latest_shadow_signal(
        minute, features, stable, time_models, fcao_models, cfg, promotion_status
    )
    data_summary = {
        "source_file": str(cfg["data_path"]),
        "minute_rows": int(len(minute)),
        "start": str(minute.index.min()),
        "end": str(minute.index.max()),
        "stable_minutes": int(stable.sum()),
        "stable_fraction": float(stable.mean()),
        "feature_count": int(features.shape[1]),
        "material_clusters": int(cluster_meta["selected_k"]),
        "kiln_label_nonnull_minutes": int(minute["窑况等级"].notna().sum()),
        "fcao_change_events_after_initial": int(fcao_metrics["fcao_regression"]["event_samples"]),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "latest_shadow_signal.json").write_text(
        json.dumps(to_jsonable(signal), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_tail.to_csv(output_dir / "predictions_tail.csv", encoding="utf-8-sig")
    write_html_report(output_dir, metrics, signal, data_summary)
    print(f"完成。报告: {output_dir / 'report.html'}")
    sys.stdout.flush()
    source_after = file_metadata(source_path)
    if source_before != source_after:
        raise RuntimeError("原始 Parquet 在运行前后元数据或哈希发生变化")
    manifest_path = Path(
        cfg.get("run_manifest_path", str(args.config.resolve().parent / "run_manifest.json"))
    )
    write_run_manifest(
        manifest_path,
        args.config.resolve(),
        cfg,
        output_dir,
        source_before,
        source_after,
        source_audit,
        data_summary,
        list(features.columns),
        cluster_meta,
        promotion_status,
        started_at,
    )


if __name__ == "__main__":
    main()
