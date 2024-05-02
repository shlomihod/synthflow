from __future__ import annotations

import logging

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pandas.api.types as ptypes  # type: ignore

LOGGER = logging.getLogger(__name__)


def _unpack_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if ptypes.is_categorical_dtype(df[column]):  # type: ignore
            transform = {}

            for int_val, cat_val in enumerate(df[column].cat.categories):
                assert cat_val.closed_left and cat_val.open_right
                if cat_val.right == np.inf:
                    actual_val = cat_val.left
                elif cat_val.right - cat_val.left > 1:
                    actual_val = cat_val.mid
                else:
                    actual_val = cat_val.left
                transform[int_val] = actual_val

            # have to transform to float before replace,
            # because the replacement occurs in the original type
            # so we might get an overflow (e.g., if the type is int8)
            df[column] = df[column].cat.codes.astype(float).replace(transform)

    return df


def _convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if ptypes.is_datetime64_any_dtype(df[column]):  # type: ignore
            df[column] = df[column].dt.strftime("%j").apply(int)

    return df


def numerify(df):
    """
    Convert DataFrame into float values for evaluation.

    Logic:
    - `bool` and `int` are converted naturally to an float/
    - Datetime is converted to the day of the year
    - Intervals betweet integers (categorical) are convertes as follow:
      - If it contains only a single integer, then it is converted to this number.
      - If it contains more than a single integer, then it is converted
        to the average of the two edges, which could be non-integer.

    Args:
        df (DataFrame): The DataFrame to be converted.

    Returns:
        DataFrame: The DataFrame after conversion.
    """
    return _unpack_categoricals(_convert_dates(df)).astype(float)
