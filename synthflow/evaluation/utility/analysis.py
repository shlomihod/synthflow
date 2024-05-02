from __future__ import annotations

import itertools as it
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Hashable, Iterable, Sequence

import pandas as pd
from tqdm import tqdm

from synthflow.evaluation.utility.analysis_structures import (
    Analysis,
    ColumnAnalysis,
    ColumnAnalysis1Way,
    ColumnAnalysis2Way,
    JointAnalysis,
)
from synthflow.evaluation.utility.metrics import (
    FREQ_FUNCS,
    FREQ_MODES,
    JOINT_METRICS,
    SINGLE_METRICS,
    TWO_METRICS,
    Metric_FN,
)
from synthflow.utils import n_way_gen

N_WAYS_UP_TO_MAX = None


@dataclass
class Result:
    val: float
    analysis: Analysis
    name: str


def _bin_column(
    df: pd.DataFrame, column: Hashable, bins: dict[Hashable, Sequence[float]]
) -> pd.Series:
    return pd.cut(df[column], bins[column], right=False)


def _bin_table(df: pd.DataFrame, bins: dict[Hashable, Sequence[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: _bin_column(df, name, bins) for name, column in df.items()}
    )


def _get_groupby_2d(
    df: pd.DataFrame, target: str, by: str, bins: dict[Hashable, Sequence[float]]
):
    by_categories = _bin_column(df, by, bins)
    return df.groupby(by_categories)[target]


def _produce_results(
    metric_fn: Metric_FN,
    real_data: Sequence,
    sythn_data: Sequence,
    an: Analysis,
) -> Sequence[Result]:
    output = metric_fn(real_data, sythn_data)
    if isinstance(output, Real):
        return [Result(float(output), an, an.metric)]
    elif isinstance(output, dict):
        return [
            Result(float(val), an, f"{an.metric}/{name}")
            for name, val in output.items()
        ]
    else:
        raise TypeError("A metric should return either a number or a dict of numbers.")


def _perform_analysis(
    an: ColumnAnalysis | JointAnalysis,
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    bins: dict[Hashable, Sequence[float]],
) -> Sequence[Result]:
    if isinstance(an, ColumnAnalysis1Way):
        real_target, synth_target = (
            (
                _bin_column(real_df, an.target, bins),
                _bin_column(synth_df, an.target, bins),
            )
            if an.binning
            else (real_df[an.target], synth_df[an.target])
        )
        metric_fn = SINGLE_METRICS[an.metric]
        return _produce_results(metric_fn, real_target, synth_target, an)

    elif isinstance(an, ColumnAnalysis2Way):
        assert an.binning, "ColumnAnalysis2Way must have binning set to True."
        metric_fn = TWO_METRICS[an.metric]
        real_grp2d = _get_groupby_2d(real_df, an.target, an.by, bins)
        synth_grp2d = _get_groupby_2d(synth_df, an.target, an.by, bins)
        return _produce_results(metric_fn, real_grp2d, synth_grp2d, an)

    elif isinstance(an, JointAnalysis):
        metric_fn = JOINT_METRICS[an.metric]
        if isinstance(an.targets, Iterable):
            real_df = real_df[list(an.targets)]
            synth_df = synth_df[list(an.targets)]
        if an.binning:
            assert isinstance(an.targets, Iterable)
            real_df = _bin_table(real_df, bins)
            synth_df = _bin_table(synth_df, bins)

        return _produce_results(metric_fn, real_df, synth_df, an)

    else:
        raise TypeError()


def _evaluate_two(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    bins: dict[Hashable, Sequence[float]],
    is_minimal_evaluation: bool = False,
    user_analysis: Sequence[Any] = [],
):
    single_column_analyses = [
        ColumnAnalysis1Way(metric, column, binning=(metric == "chi2"))
        for column in df1.columns
        for metric in SINGLE_METRICS.keys()
    ]

    n_way_up_to = (
        len(df1.columns)
        if N_WAYS_UP_TO_MAX is None
        else min(N_WAYS_UP_TO_MAX, len(df1.columns))
    )
    n_ways = list(range(1, n_way_up_to + 1))

    joint_analyses = [
        JointAnalysis("frequencies", targets=tuple(columns_nway), binning=binning)
        for columns_nway in n_way_gen(df1.columns, n_ways)
        for binning in [False, True]
    ]

    joint_analyses += [
        JointAnalysis("lr", binning=False),
        JointAnalysis("corr", binning=False),
    ]

    if not is_minimal_evaluation:
        joint_analyses += [JointAnalysis("pMSE", binning=False)]

    analyses = single_column_analyses + user_analysis + joint_analyses  # type: ignore
    result_lists = (_perform_analysis(an, df1, df2, bins) for an in tqdm(analyses))
    results = list(it.chain.from_iterable(result_lists))

    results_df = pd.DataFrame(
        [
            {"val": result.val, "name": result.name, **asdict(result.analysis)}
            for result in results
        ]
    )

    frequencies_rows = results_df[results_df["metric"] == "frequencies"]
    assert set(frequencies_rows["targets"].apply(len)) == set(n_ways)

    for name, func in FREQ_FUNCS.items():
        for mode in FREQ_MODES:
            for binning in [False, True]:
                for n_way in [[n] for n in n_ways] + [n_ways]:
                    n_way_frequencies_rows = frequencies_rows[
                        (frequencies_rows["targets"].apply(len).isin(n_way))
                        & (frequencies_rows["binning"] == binning)
                        & (frequencies_rows["name"] == f"frequencies/{mode}/{name}")
                    ]

                    assert len(n_way_frequencies_rows)

                    metric = f"{''.join(map(str, n_way))}way/{name}/frequencies"

                    result_name = f"{metric}/{mode}/{'bins' if binning else 'raw'}"
                    result_val = func(n_way_frequencies_rows["val"])
                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame(
                                [
                                    {
                                        "metric": metric,
                                        "name": result_name,
                                        "val": result_val,
                                        "target": None,
                                        "targets": None,
                                        "by": None,
                                        "binning": binning,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

    return results_df
