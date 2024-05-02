from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import pandas.api.types as ptypes  # type: ignore

LOGGER = logging.getLogger(__name__)


def _adjust_categorical_column_weights(
    weights: list[int], binning: list[int]
) -> list[int]:
    """
    Adjust a faithfulness binning (column weights) according to a column's binning.

    Args:
        weights (list): The pre-set faithfulness binning.
        binning (list): The categorical binning of a column.

    Returns:
        list: The adjusted faithfulness binning.
    """

    assert len(weights) > 3 and len(binning) > 3

    binning_internal_stops = [-np.inf] + sorted(binning)[1:-1] + [np.inf]

    weights_internal_stops = [-np.inf] + sorted(weights)[1:-1] + [np.inf]

    adjusted_coloumn_weights = []

    weights_iter = iter(weights_internal_stops)
    binning_iter = iter(binning_internal_stops)

    weight_value = prev_weight_value = next(weights_iter)
    binning_value = next(binning_iter)
    while True:
        if weight_value == binning_value:
            adjusted_coloumn_weights.append(weight_value)

        elif binning_value < weight_value:
            while binning_value < weight_value:
                binning_value = next(binning_iter)
            adjusted_coloumn_weights.append(weight_value)

        elif weight_value < binning_value:
            adjusted_coloumn_weights.append(np.nextafter(prev_weight_value, np.inf))

            weight_value = next(weights_iter)

            while weight_value < binning_value:
                prev_weight_value = weight_value
                weight_value = next(weights_iter)

            adjusted_coloumn_weights.append(np.nextafter(binning_value, -np.inf))
            adjusted_coloumn_weights.append(binning_value)

        else:
            raise ValueError()

        try:
            prev_weight_value = weight_value
            weight_value = next(weights_iter)
        except StopIteration:
            break

        try:
            binning_value = next(binning_iter)
        except StopIteration:
            break

    return adjusted_coloumn_weights


def prepare_column_weights(
    column_weights: dict[str, float], df: pd.DataFrame
) -> dict[str, float]:
    """
    Adjust a faithfulness column weights according to a DataFrame column types.

    A change is made only for categorical columns.

    Args:
        column_weights (list): The adjusted faithfulness' column weights.
        df (DataFrame): The DataFrame.

    Returns:
    """

    column_weights = deepcopy(column_weights)

    for column, weights in column_weights.items():
        if isinstance(weights, list) and ptypes.is_categorical_dtype(df[column].dtype):
            binning = sorted(
                set.union(
                    *(
                        {interval.left, interval.right}
                        for interval in df[column].cat.categories
                    )
                )
            )

            weights = column_weights[column]

            adjusted_weights = _adjust_categorical_column_weights(weights, binning)

            if weights[1:-1] != adjusted_weights[1:-1]:
                LOGGER.info(
                    f"IMPORTANT: `{column}` cost weights for faithfulness"
                    " changed to binning because the latter"
                    " is a refierment of the former."
                )

            column_weights[column] = adjusted_weights

        return column_weights
