from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore


def _reject_sampling(
    df: pd.DataFrame, constraints: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    passing_constraints_masks = [
        df.apply(constraint_fn, axis=1) for constraint_fn in constraints
    ]
    passing_mask = np.all(passing_constraints_masks, axis=0)
    constraints_df = df[~passing_mask].reset_index(drop=True)
    return df[passing_mask], constraints_df


def _project_single_min_count(df: pd.DataFrame, order: int) -> pd.DataFrame:
    counts = df.value_counts()

    # make sure that the minimal count is exactly `order`
    assert not counts.min() < order
    if counts.min() > order:
        return df

    order_mask = counts == order
    order_df = pd.DataFrame(list(counts[order_mask].index), columns=df.columns)

    # take n/(n+1) for row configurations repeating n times, and drop 1/(n+1) of them
    n_base_cases = len(order_df) // (order + 1)
    n_choosen = order * n_base_cases
    choosen_mask = np.array([True] * n_choosen + [False] * (len(order_df) - n_choosen))
    np.random.shuffle(choosen_mask)
    order_df["_choosen"] = choosen_mask

    merge_df = pd.merge(df, order_df, how="left", on=list(df.columns), indicator=True)
    larger_order_mask = merge_df["_merge"] == "left_only"
    choosen_exact_mask = merge_df["_choosen"] == True  # noqa: E712

    n_missing = len(df) - (larger_order_mask.sum() + (order + 1) * n_choosen)

    choosen_exact_indices = np.nonzero(choosen_exact_mask.values)[0]

    missing_indices = np.random.choice(choosen_exact_indices, size=n_missing)
    missing_df = df.iloc[missing_indices]

    final_df = pd.concat(
        [
            df[larger_order_mask],
            pd.concat([df[choosen_exact_mask].drop_duplicates()] * (order + 1)),
            missing_df,
        ]
    )

    final_df = final_df.reset_index(drop=True)
    final_counts = final_df.value_counts()

    assert (df.columns == final_df.columns).all()
    assert len(final_df) == len(df), len(final_df)
    assert final_counts.min() > order, (order, final_counts.min())
    assert (
        counts[counts > 2 * (order + 1)].sort_index()
        == final_counts[final_counts > 2 * (order + 1)].sort_index()
    ).all()

    return final_df


def _project_multiple_min_count(df: pd.DataFrame, up_to_order: int) -> pd.DataFrame:
    for order in range(1, up_to_order + 1):
        df = _project_single_min_count(df, order)
    return df


def _project_dataset(df: pd.DataFrame, projection: str) -> pd.DataFrame:
    if projection is None:
        return df

    elif projection.get("name") == "min-count":
        return _project_multiple_min_count(df, projection["order"])

    else:
        raise ValueError(f"Projection `{projection}` is not valid.")
