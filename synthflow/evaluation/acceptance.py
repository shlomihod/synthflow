from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


@dataclass
class AcceptanceCriteria:
    name: str
    selector: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], int | float]
    expected: int | float


def check_acceptance(
    criteria: Sequence[AcceptanceCriteria],
    utility_df: pd.DataFrame,
    privcay_df: pd.DataFrame,
    faithfulness_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether an evaluation is accepted according to given criteria.

    Args:
        df (Sequence): Sequence of criterion as an `AcceptanceCriteria` object.
        utility_df (DataFrame): The utility evaluation DataFrame.
        privcay_df (DataFrame): The privacy evaluation DataFrame.
        faithfulness_df (DataFrame): The faithfulness evaluation DataFrame.

    Returns:
        DataFrame: The acceptance results.
    """

    results = []
    for criterion in criteria:
        try:
            actual = criterion.selector(utility_df, privcay_df, faithfulness_df)
        # TODO: In the furture, when updating pandas version, change to
        # pd.errors.UndefinedVariableError
        except (IndexError, KeyError, pd.core.computation.ops.UndefinedVariableError):
            actual = np.nan

        results.append(
            {"name": criterion.name, "expected": criterion.expected, "actual": actual}
        )

    df = pd.DataFrame(results)
    df["check"] = df["actual"] < df["expected"]

    check_all = df["check"].all()
    df = df.append(
        {"name": "all", "actual": check_all, "expected": True, "check": check_all},
        ignore_index=True,
    )

    return df
