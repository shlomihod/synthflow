from __future__ import annotations

import itertools as it
from collections.abc import Sequence
from typing import Generator

import pandas as pd

ZERO_CLOSE_ATOL = 1e-10


def n_way_gen(seq: Sequence, n_ways: Sequence[int]) -> Generator[tuple, None, None]:
    """
    Generate all n-way tuples of a sequence.

    Args:
        seq (Sequence): Sequence of elements to create tuples from.
        n_ways (Sequence): Values of n-way tuples to generate.

    Returns:
        tuple: N-way tuple.
    """

    yield from it.chain(*(it.combinations(seq, n_way) for n_way in n_ways))


def compute_complete_counts(
    counts1: pd.Series, counts2: pd.Series, fill_value: int | float = 0
) -> tuple[pd.Series, pd.Series]:
    """
    Complete indices of two value counts to the union of their indices.

    Args:
        counts1 (panda.Series): First value counts.
        counts2 (panda.Series): Second value counts.
        fill_value (int, float): Count fill value for a missing index. Default to 0.

    Returns:
        pd.Series: Completed first value counts.
        pd.Series: Completed second value counts.
    """

    union_index = list(set(counts1.index) | set(counts2.index))

    counts1_complete = counts1.reindex(union_index, fill_value=fill_value)
    counts2_complete = counts2.reindex(union_index, fill_value=fill_value)

    assert isinstance(counts1_complete, pd.Series)
    assert isinstance(counts2_complete, pd.Series)

    return counts1_complete, counts2_complete


def interval_number(x: int | float) -> pd.Interval:
    """
    Create a pd.Interval containing a single number.

    Args:
        x (int, float): Number.

    Returns:
        pd.Interval: Interval containing a single number.
    """

    return pd.Interval(x, x, closed="both")


def get_single_row_by_query(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Retrieve a query-single-matching row from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame.
        query (str): The retrieval query.

    Returns:
        pandas.Series: The retrieved row.
    """

    row = df.query(expr)
    assert len(row) == 1
    return row.iloc[0]
