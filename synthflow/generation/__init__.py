from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any, Sequence

import numpy as np
import pandas as pd  # type: ignore
import pandera as pa

from synthflow.generation.models import _fit_sample
from synthflow.generation.post import _project_dataset, _reject_sampling
from synthflow.generation.processing import DATETIME_RESOLUTIONS, _process

LOGGER = logging.getLogger(__name__)


def generate(
    real_df: pd.DataFrame,
    config: dict[str, Any],
    schema: pa.DataFrameSchema,
    constraints: Sequence[Any] | None = None,
    ubersampling: float = 1,
) -> tuple[
    pd.DataFrame, Any, pa.DataFrameSchema, pd.DataFrame, pd.DataFrame, None | list
]:
    """
    Generate synthetic data.

    Args:
        real_df (DataFrame): The real/original DataFrame.
        config (dict): The configuration, including transformation,
                       model and hyperparameters.
        schema (DataFrameSchema): The schema of the input DataFrame.
        constraints (Sequence): Sequence of boolean functions
                                that all synthetic row should satisfy.
        ubersampling (float): Oversampling factor to allow filtering
                              due to the constraints.

    Returns:
        DataFrame: The synthetic DataFrame.
        Any: The fitted model used to generate the synthetic data.
        DataFrameSchema: The schema of the synthetic DataFrame.
        Sequence: The names of the categorical columns.
        DataFrame: The real processed DataFrame.
        DataFrame: Rows that were filtered due to the constrains.

    Raises:
        RuntimeError : Mismatch betweern number of rows (e.g., requsted vs. sampled).
    """

    processed_df, boundaries, categoricals, rev_transformations, _ = _process(
        real_df, config["transformations"], config["categoricals"], schema, is_rev=False
    )

    (
        real_processed_df,
        _,
        _,
        _,
        _,
    ) = _process(processed_df, rev_transformations, [], schema, is_rev=True)

    num_records = len(real_df)
    if constraints:
        num_records = int(math.ceil(num_records * ubersampling))

    samples_df, model = _fit_sample(
        processed_df,
        boundaries,
        config["model"],
        config["epsilon"],
        config["delta"],
        config["hparams"],
        categoricals,
        num_records,
    )

    try:
        transcript = deepcopy(model.transcript)
        assert isinstance(transcript, list)
        LOGGER.info("Transcript found in the model.")
    except AttributeError:
        transcript = []
        LOGGER.warning("No transcript found in the model.")

    transcript.append(("Unprocessed", samples_df))

    if len(samples_df) != num_records:
        raise RuntimeError(
            f"Number of sampled rows ({len(samples_df)})"
            f" is different than the number of requested rows ({num_records})."
        )

    (
        synth_df,
        _,
        _,
        _,
        synth_dataset_schema,
    ) = _process(samples_df, rev_transformations, [], schema, is_rev=True)

    transcript.append(("Sampled", synth_df))

    if constraints is not None and constraints:
        synth_df, constraints_df = _reject_sampling(synth_df, constraints)
        synth_df = synth_df.iloc[: len(real_df)].reset_index(drop=True)
        transcript.append(("Constrained", synth_df))
    else:
        constraints_df = pd.DataFrame()

    if config["dataset_projection"] is not None:
        synth_df = _project_dataset(synth_df, config["dataset_projection"])
        transcript.append(("Projected", synth_df))

    if len(real_df) != len(synth_df):
        raise RuntimeError(
            f"Number of synthetic rows ({len(synth_df)})"
            f" is different than the number of real rows ({len(real_df)})."
        )

    return (
        synth_df,
        model,
        synth_dataset_schema,
        real_processed_df,
        constraints_df,
        transcript,
    )
