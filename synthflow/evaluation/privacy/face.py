from __future__ import annotations

from typing import Sequence

import pandas as pd  # type: ignore

FACE_PRIVACY_METRICS = ["unique", "k-anonymity", "l-diversity"]


def _count_occurrences(occurrences: pd.Series, up_to_k: int) -> pd.Series:
    return (
        occurrences.value_counts()
        .sort_index()
        .loc[1:up_to_k]
        .reindex(range(1, up_to_k + 1))
        .fillna(0)
        .astype(int)
    )


def _calc_uniques_and_more(df: pd.DataFrame, up_to: int) -> pd.Series:
    """
    Evaluate the uniquness of rows within a dataset.

    Args:
        df (DataFrame): The DataFrame to be evaluated.
        up_to (int): Evaluate up to that value of number of uniquness.
    Returns:
        Series: The result Series of the uniquness.
    """
    return _count_occurrences(df[list(df.columns)].value_counts(), up_to)


def _calc_k_anonymity(
    df: pd.DataFrame, quasi_columns: Sequence[str], up_to: int
) -> pd.Series:
    """
    Evaluate the k-anonymity of a dataset.

    Args:
        df (DataFrame): The DataFrame to be evaluated.
        up_to (int): Evaluate up to that value of the param k.
        quasi_snsitive_pairs (list): List of quasi columns.
    Returns:
        Series: The result Series of the k-anonymity.
    """
    return _count_occurrences(df[list(quasi_columns)].value_counts(), up_to)


def _calc_l_diversity(
    df: pd.DataFrame, quasi_columns: Sequence[str], sensitive_column: str, up_to: int
) -> pd.Series:
    """
    Evaluate the l-diversity of a dataset.

    Args:
        df (DataFrame): The DataFrame to be evaluated.
        up_to (int): Evaluate up to that value of the param l.
        quasi_snsitive_pairs (list): List of (quasi, sensitive) column pairs.
    Returns:
        Series: The result Series of the l-diversity.
    """
    return _count_occurrences(
        df.groupby(list(quasi_columns))[sensitive_column].nunique(), up_to
    )


def evaluate_face_privacy(
    df: pd.DataFrame,
    up_to: int,
    quasi_snsitive_pairs: None | tuple[tuple[tuple[str], str]] = None,
):
    """
    Evaluate the face privacy of a dataset.
    (1) uniques; (2) k-anonymity; (3) l-diversity

    Args:
        df (DataFrame): The DataFrame to be evaluated.
        up_to (int): Order of face privacy metrics.
        quasi_snsitive_pairs (list): List of (quasi, sensitive) column pairs.
    Returns:
        DataFrame: The results DataFrame of the face privacy evaluation.
    """

    quasi_seq, _ = zip(*quasi_snsitive_pairs)
    quasi_only = list(set(quasi_seq))

    unique_dfs = [
        _calc_uniques_and_more(df, up_to).to_frame(name="value").assign(metric="unique")
    ]

    k_anonymity_dfs = [
        pd.DataFrame({"value": _calc_k_anonymity(df, quasi, up_to)}).assign(
            quasi=";".join(quasi), metric="k-anonymity"
        )
        for quasi in quasi_only
    ]

    l_diversity_dfs = [
        pd.DataFrame({"value": _calc_l_diversity(df, quasi, sensitive, up_to)}).assign(
            quasi=";".join(quasi), sensitive=sensitive, metric="l-diversity"
        )
        for quasi, sensitive in quasi_snsitive_pairs
    ]

    face_privacy_df = (
        pd.concat(unique_dfs + k_anonymity_dfs + l_diversity_dfs)
        .rename_axis("param")
        .reset_index()
    )

    face_privacy_df["name"] = (
        face_privacy_df["metric"] + "/" + face_privacy_df["param"].astype(str)
    )

    for metric in FACE_PRIVACY_METRICS:
        for param in range(1, up_to + 1):
            max_value = face_privacy_df.query(f"name == '{metric}/{param}'")[
                "value"
            ].max()
            face_privacy_df = pd.concat(
                [
                    face_privacy_df,
                    pd.DataFrame(
                        [
                            {
                                "metric": metric,
                                "param": param,
                                "name": f"{metric}/{param}/max",
                                "value": max_value,
                            }
                        ]
                    ),
                ]
            )

    face_privacy_df = face_privacy_df.assign(type="face")

    return face_privacy_df
