from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Any, Callable, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.core.groupby.generic import SeriesGroupBy
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from synthflow.birth import (
    IMPUTATION_MIDPOINT_VALUES,
    THRESHOLD_ALL_K_WAYS_MAX_ABS_DIFF,
)
from synthflow.utils import compute_complete_counts

LOGGER = logging.getLogger(__name__)

PRNG = np.random.default_rng()
LOGGER.info(
    f"_apply_resize_transformation:"
    f" Using PRNG with seed {PRNG.bit_generator._seed_seq=}"
)

SequenceData = Union[Sequence, ArrayLike, pd.Series, pd.DataFrame]
Metric_FN = Callable[[SequenceData, SequenceData], Any]

FREQ_MODES = ("diff", "ratio")
FREQ_FUNCS: dict[str, Callable] = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "rmse": lambda x: np.sqrt(np.mean(x**2)),
}


@dataclass
class Metric:
    name: str
    fun: Metric_FN


def _make_diff(metric_fn: Callable[[SequenceData], float]) -> Metric_FN:
    def func(x, y):
        return np.abs(metric_fn(x) - metric_fn(y))

    return func


def _make_ratio(metric_fn: Callable[[SequenceData], float]) -> Metric_FN:
    def func(x, y):
        try:
            return metric_fn(x) / metric_fn(y)
        except ZeroDivisionError:
            return np.inf

    return func


def _chi2homogeneity(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    assert x.dtype.name == "category" and y.dtype.name == "category"
    assert x.dtype == y.dtype
    obs = pd.DataFrame({"x": x.value_counts(), "y": y.value_counts()})
    obs = obs[(obs != 0).any(axis=1)]  # drop all zeros rows
    return stats.chi2_contingency(obs)[:3]


def _compute_pMSE(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, float]:
    """https://arxiv.org/pdf/1805.09392.pdf"""
    X = pd.concat([df1, df2])
    y = pd.Series([False] * len(df1) + [True] * len(df2))
    X, y = shuffle(X, y)
    cls = GradientBoostingClassifier()
    cls.fit(X, y)
    p = cls.predict_proba(X)[:, 1]
    # 0.25 means good prediction, 0 means random
    return {"score": ((p - 0.5) ** 2).mean(), "acc": cls.score(X, y)}


def _compute_wasserstein(x: pd.Series, y: pd.Series) -> float:
    x, y = x.astype(int), y.astype(int)
    x_bins, y_bins = np.arange(x.max() + 1), np.arange(y.max() + 1)
    x_counts, y_counts = np.bincount(x), np.bincount(y)
    return stats.wasserstein_distance(x_bins, y_bins, x_counts, y_counts) * len(x)


def _run_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], Any],
    y_bounds: None | tuple[float, float] = None,
) -> dict[str, float]:
    model = model_factory().fit(X, y)
    y_pred = model.predict(X)

    if y_bounds is not None:
        y_pred = np.clip(y_pred, *y_bounds)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return {
        "R2": r2,
        "coef": np.concatenate((model.coef_, [model.intercept_])),
        "MAE": mae,
    }, model


def _compute_cross_mae(
    model1: Any,
    model2: Any,
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    y1: pd.Series,
    y2: pd.Series,
    y_bounds: None | tuple[float, float] = None,
):
    y_pred_1_by_2 = model2.predict(X1)
    y_pred_2_by_1 = model1.predict(X2)

    y_pred_1_by_1 = model1.predict(X1)
    y_pred_2_by_2 = model2.predict(X2)

    if y_bounds is not None:
        y_pred_1_by_2 = np.clip(y_pred_1_by_2, *y_bounds)
        y_pred_2_by_1 = np.clip(y_pred_2_by_1, *y_bounds)
        y_pred_1_by_1 = np.clip(y_pred_1_by_1, *y_bounds)
        y_pred_2_by_2 = np.clip(y_pred_2_by_2, *y_bounds)

    mae_corss_1_by_2 = np.abs(
        mean_absolute_error(y1, y_pred_1_by_1) - mean_absolute_error(y1, y_pred_1_by_2)
    )
    mae_corss_2_by_1 = np.abs(
        mean_absolute_error(y2, y_pred_2_by_2) - mean_absolute_error(y2, y_pred_2_by_1)
    )

    return {"MAE/cross/1-by-2": mae_corss_1_by_2, "MAE/cross/2-by-1": mae_corss_2_by_1}


def _compute_linear_regression(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    model_factory_1: Callable[[], Any],
    model_factory_2: None | Callable[[], Any] = None,
    y_bounds: None | tuple[float, float] = None,
    scale_by_second: bool = True,
) -> dict[str, float]:
    assert (df1.columns == df2.columns).all(), (df1.columns, df2.columns)

    if model_factory_2 is None:
        model_factory_2 = model_factory_1

    X1, y1 = df1.iloc[:, :-1], df1.iloc[:, -1]
    X2, y2 = df2.iloc[:, :-1], df2.iloc[:, -1]

    if scale_by_second:
        scaler = StandardScaler().fit(X2)
        X1 = scaler.transform(X1)
        X2 = scaler.transform(X2)

    results1, model1 = _run_linear_regression(X1, y1, model_factory_1)
    results2, model2 = _run_linear_regression(X2, y2, model_factory_2)

    diff_matrics = {
        f"coef/error/diff/{column}": np.abs(coef1 - coef2)
        for column, coef1, coef2 in zip(df1.columns, results1["coef"], results2["coef"])
    }

    cross_mae_metrics = _compute_cross_mae(model1, model2, X1, X2, y1, y2, y_bounds)

    metrics = {**diff_matrics, **cross_mae_metrics}
    metrics["coef/error/diff/max"] = max(diff_matrics.values())
    metrics["coef/error/diff/total"] = sum(diff_matrics.values())
    metrics["R2/diff"] = np.abs(results1["R2"] - results2["R2"])
    metrics["MAE/diff"] = np.abs(results1["MAE"] - results2["MAE"])
    return metrics


SINGLE_METRICS = {
    "nan": _make_diff(lambda x: np.isnan(x).sum()),
    "mean": _make_diff(np.mean),
    "median": _make_diff(np.median),
    "std": _make_ratio(np.std),
    "max": _make_diff(np.max),
    "min": _make_diff(np.min),
    "q1": _make_diff(lambda a: np.quantile(a, 0.25)),
    "q3": _make_diff(lambda a: np.quantile(a, 0.75)),
    "cdf": lambda x, y: stats.ks_2samp(x, y)[1],  # show p-value only
    "chi2": lambda x, y: _chi2homogeneity(x, y)[1],  # show p-value only
    "wasserstein": _compute_wasserstein,
}


def _apply_resize_transformation(xs, size: int, imputation_value: Any):
    assert size > 0

    xs = np.array(xs)

    if len(xs) == size:  # as-is
        return xs.copy()
    elif len(xs) > size:  # truncation
        return PRNG.permutation(xs)[:size]
    elif len(xs) < size:  # imputation
        imputation_array = np.array([imputation_value] * (size - len(xs)))
        return PRNG.permutation(np.concatenate([xs, imputation_array]))


def _compute_mean_resized_by_second(x: SeriesGroupBy, y: SeriesGroupBy):
    num_records = y.size().sum()
    if x.size().sum() != num_records:
        LOGGER.warn(
            f"Got two group-by series with different total size:"
            f" {x.size().sum()=} and {y.size().sum()=}"
        )

    # Based on the first acceptance criteria, we can assume w.h.p. that
    # the 1way marginals of the synthetic dataset are at most within this gap
    lower_margin = int(np.ceil(num_records * THRESHOLD_ALL_K_WAYS_MAX_ABS_DIFF) + 1)

    sizes_by_second = np.maximum(y.size() - lower_margin, 1)

    mean_column_name = x.mean().name
    imputation_value = IMPUTATION_MIDPOINT_VALUES[mean_column_name]

    def calc_resized_mean(xs):
        size = sizes_by_second.loc[xs.name]
        resized_xs = _apply_resize_transformation(xs, size, imputation_value)
        return resized_xs.mean()

    return {
        "max/diff": (x.apply(calc_resized_mean) - y.apply(np.mean))
        .abs()
        .fillna(0)
        .max(),
        "min/size_by_second": sizes_by_second.min(),
    }


TWO_METRICS = {
    "median": lambda x, y: (x.agg(np.median) - y.agg(np.median)).abs().fillna(0).max(),
    "mean": lambda x, y: (x.agg(np.mean) - y.agg(np.mean)).abs().fillna(0).max(),
    "mean-resize-by-second": _compute_mean_resized_by_second,
}


def _compute_frequencies(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, float]:
    counts1 = df1.value_counts(normalize=False)
    counts2 = df2.value_counts(normalize=False)

    diff = ((counts1 / len(df1)) - (counts2 / len(df2))).abs().fillna(0)

    (counts1_complete, counts2_complete) = compute_complete_counts(
        counts1, counts2, fill_value=0
    )

    # Pascal correction
    counts1_complete = (counts1_complete + 1) / (len(df1) + 1)
    counts2_complete = (counts2_complete + 1) / (len(df1) + 1)

    ratio1 = counts1_complete / counts2_complete
    ratio2 = counts2_complete / counts1_complete
    ratio = np.maximum(ratio1, ratio2)

    return {
        f"{mode}/{name}": func(vals)
        for mode, vals in [("diff", diff), ("ratio", ratio)]
        for name, func in FREQ_FUNCS.items()
    }


def _compute_correlations(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, float]:
    results = {}
    for method in ("pearson", "kendall", "spearman"):
        corr1 = df1.corr(method)
        corr2 = df2.corr(method)
        diff = (corr1 - corr2).abs()
        for col1, col2 in combinations(df1.columns, r=2):
            results[f"{method}/diff/{col1}/{col2}"] = diff.loc[col1, col2]

        results[f"{method}/diff/max"] = max(results.values())

    return results


JOINT_METRICS = {
    "frequencies": _compute_frequencies,
    "corr": _compute_correlations,
    "pMSE": _compute_pMSE,
    "lr": partial(
        _compute_linear_regression, model_factory_1=lambda: LinearRegression()
    ),
}
