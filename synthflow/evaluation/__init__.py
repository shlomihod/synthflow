from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pandas.api.types as ptypes  # type: ignore

from synthflow.evaluation.acceptance import AcceptanceCriteria, check_acceptance
from synthflow.evaluation.faithfulness import evaluate_faithfulness
from synthflow.evaluation.preperation import (  # noqa: F401
    _convert_dates,
    _unpack_categoricals,
    numerify,
)
from synthflow.evaluation.privacy import evaluate_privacy
from synthflow.evaluation.utility import evaluate_utility
from synthflow.evaluation.utility.analysis import ColumnAnalysis2Way  # noqa: F401
from synthflow.release import private_acceptance_criteria

LOGGER = logging.getLogger(__name__)


def evaluate(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    bins: dict[str, Sequence[float]],
    column_weights: dict[str, float],
    alphas: Sequence[float],
    face_privacy_up_to: int,
    acceptence_criteria: Sequence[AcceptanceCriteria],
    high_sensitivity_epsilon: float,
    low_sensitivity_epsilon: float,
    conditional_mean_epsilons: dict[float],
    complex_query_fit_epsilon: float,
    complex_query_eval_epsilon: float,
    ratio_max_clipping_factor: float,
    subsampling_frac: float,
    subsampling_random_seed: None | int = None,
    processed_real_df: None | pd.DataFrame = None,
    is_minimal_evaluation: bool = False,
    user_analysis: Sequence[Any] = (),
    quasi_snsitive_pairs: None | tuple[tuple[tuple[str], str]] = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    real_df = numerify(real_df)
    synth_df = numerify(synth_df)

    if processed_real_df is not None:
        processed_real_df = numerify(processed_real_df.copy())

    utility_df, subsampled_real_df = evaluate_utility(
        real_df,
        synth_df,
        bins,
        subsampling_frac,
        subsampling_random_seed,
        processed_real_df,
        is_minimal_evaluation,
        user_analysis,
    )
    if not is_minimal_evaluation:
        privacy_df = evaluate_privacy(
            real_df,
            synth_df,
            face_privacy_up_to,
            processed_real_df,
            quasi_snsitive_pairs,
        )
        faithfulness_df = evaluate_faithfulness(
            processed_real_df, synth_df, column_weights, alphas
        ).assign(comparison="val_pr_s")
    else:
        privacy_df, faithfulness_df = pd.DataFrame(), pd.DataFrame()

    acceptance_df = check_acceptance(
        acceptence_criteria, utility_df, privacy_df, faithfulness_df
    )

    dp_acceptance_df, dp_accountant = private_acceptance_criteria(
        acceptance_df,
        high_sensitivity_epsilon,
        low_sensitivity_epsilon,
        conditional_mean_epsilons,
        complex_query_fit_epsilon,
        complex_query_eval_epsilon,
        ratio_max_clipping_factor,
        synth_df,
        utility_df,
        processed_real_df,
    )

    return (
        utility_df,
        privacy_df,
        faithfulness_df,
        subsampled_real_df,
        acceptance_df,
        dp_acceptance_df,
        dp_accountant,
    )
