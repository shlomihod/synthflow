from __future__ import annotations

from typing import Any, Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from synthflow.evaluation.utility.analysis import _evaluate_two


def evaluate_utility(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    bins: dict[str, Sequence[float]],
    subsampling_frac: float,
    subsampling_random_seed: None | int = None,
    processed_real_df: None | pd.DataFrame = None,
    is_minimal_evaluation: bool = False,
    user_analysis: list[Any] = [],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the utility of synthtic data compared to variants of real data.


    Args:
        real_df (DataFrame): The real (original) DataFrame.
        synth_df (DataFrame): The synthetic DataFrame.
        bins (dict): Binning per column to be used for metrics with conditioning.
        subsampling_frac (float): Which proportion of the data to use
                                  for the subsampling dataset.
        subsampling_random_seed (int, otptional): Random seed for subsampling.
        processed_real_df (DataFrame, optional): The processed real DataFrame.
        is_minimal_evaluation (bool): Whether to conduct a minimal analysis,
                                      and skip computational-heavy analyses.
                                      Default to False.
        user_analysis (list): Additional analyses to run given by the user.
    Returns:
        DataFrame: The results DataFrame of the utility evaluation.
        DataFrame: The subsampled DataFrame.
    """

    synth_results_df = _evaluate_two(
        real_df, synth_df, bins, is_minimal_evaluation, user_analysis
    )

    assert len(real_df) == len(synth_df), f"{len(real_df)=} ; {len(synth_df)=}"

    result_columns = []

    n_subsampling = int(len(real_df) * subsampling_frac)
    rng = np.random.default_rng(subsampling_random_seed)
    idx = rng.choice(np.arange(len(real_df)), n_subsampling, replace=False)
    subsampled_real_df = real_df.iloc[idx]

    subsampling_results_df = _evaluate_two(
        real_df, subsampled_real_df, bins, is_minimal_evaluation, user_analysis
    )

    ON_MERGE_COLUMNS = ["name", "metric", "target", "binning", "by", "targets"]

    results_df = pd.merge(
        synth_results_df,
        subsampling_results_df,
        on=ON_MERGE_COLUMNS,
        suffixes=("_r_s", "_r_sr"),
        sort=True,
    )

    result_columns = ["val_r_s", "val_r_sr"]

    if processed_real_df is not None:
        assert len(real_df) == len(processed_real_df)
        subsampled_processed_real_df = processed_real_df.iloc[idx]

        processed_real_results_df = _evaluate_two(
            processed_real_df, real_df, bins, is_minimal_evaluation, user_analysis
        )
        processed_synth_results_df = _evaluate_two(
            processed_real_df, synth_df, bins, is_minimal_evaluation, user_analysis
        )
        subsampled_processed_real_results_df = _evaluate_two(
            subsampled_processed_real_df,
            real_df,
            bins,
            is_minimal_evaluation,
            user_analysis,
        )

        processed_results_df = pd.merge(
            processed_real_results_df,
            processed_synth_results_df,
            on=ON_MERGE_COLUMNS,
            suffixes=("_pr_r", "_pr_s"),
            sort=True,
        )

        processed_results_df = pd.merge(
            processed_results_df,
            subsampled_processed_real_results_df.rename({"val": "val_spr_r"}, axis=1),
            on=ON_MERGE_COLUMNS,
            sort=True,
        )

        results_df = pd.merge(
            results_df,
            processed_results_df,
            on=ON_MERGE_COLUMNS,
            sort=True,
        )

        result_columns = ["val_pr_r", "val_pr_s", "val_spr_r", "val_r_sr", "val_r_s"]

    result_columns += ON_MERGE_COLUMNS
    results_df = results_df[result_columns]

    return results_df, subsampled_real_df
