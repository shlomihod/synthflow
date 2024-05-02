import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from synthflow.evaluation import numerify
from synthflow.evaluation.utility import evaluate_utility
from synthflow.evaluation.utility.metrics import (
    _apply_resize_transformation,
    _compute_correlations,
    _compute_linear_regression,
    _compute_mean_resized_by_second,
)
from tests.utils import basic_all_categorical_real_df  # noqa: F401


def test_compute_correlations():
    "Test correlation calculation."

    df = pd.DataFrame({"A": [0] * 5 + [1] * 5, "B": [0] * 4 + [1] + [1] * 5})
    df1 = df.copy().assign(C=df["A"] - df["B"] + 1)
    df2 = df.copy().assign(C=df["A"])

    results = _compute_correlations(df1, df2)

    assert results["kendall/diff/A/B"] == 0
    assert results["spearman/diff/A/B"] == 0

    np.testing.assert_allclose(results["pearson/diff/A/C"], 1 - 1 / 3)
    np.testing.assert_allclose(
        results["pearson/diff/B/C"], 0.8164965809277258 - (-0.2721655269759086)
    )

    for method in ("pearson", "kendall", "spearman"):
        assert results[f"{method}/diff/A/B"] == 0
        np.testing.assert_allclose(
            results[f"{method}/diff/max"],
            max(
                val
                for name, val in results.items()
                if method in name and "max" not in name
            ),
        )


def test_compute_linear_regression():
    "Test linear regression calculation."

    rng = np.random.default_rng(1987)

    n_records = 100000

    df = pd.DataFrame(
        {"A": rng.normal(0, 1, n_records), "B": rng.normal(0, 1, n_records)}
    )
    df1 = df.copy().assign(C=3 * df["A"] + df["B"] + 1)
    df2 = df.copy().assign(C=df["A"] + 2 * df["B"] + 1)

    results = _compute_linear_regression(
        df1, df2, lambda: LinearRegression(), scale_by_second=False
    )

    np.testing.assert_allclose(results["coef/error/diff/A"], 2)
    np.testing.assert_allclose(results["coef/error/diff/B"], 1)
    np.testing.assert_allclose(results["coef/error/diff/max"], 2)
    np.testing.assert_allclose(results["coef/error/diff/total"], 3)

    np.testing.assert_allclose(results["MAE/diff"], 0, atol=1e-10)
    np.testing.assert_allclose(results["R2/diff"], 0, atol=1e-10)

    np.testing.assert_allclose(results["MAE/cross/1-by-2"], 1.7883245100546452)
    np.testing.assert_allclose(results["MAE/cross/2-by-1"], 1.7883245100546448)

    results = _compute_linear_regression(
        df1, df2, lambda: LinearRegression(), scale_by_second=True
    )

    np.testing.assert_allclose(results["coef/error/diff/A"], 2, atol=1e-2)
    np.testing.assert_allclose(results["coef/error/diff/B"], 1, atol=1e-2)
    np.testing.assert_allclose(results["coef/error/diff/max"], 2, atol=1e-2)
    np.testing.assert_allclose(results["coef/error/diff/total"], 3, atol=1e-1)

    np.testing.assert_allclose(results["MAE/diff"], 0, atol=1e-10)
    np.testing.assert_allclose(results["R2/diff"], 0, atol=1e-10)

    np.testing.assert_allclose(
        results["MAE/cross/1-by-2"], 1.7883245100546452, atol=1e-2
    )
    np.testing.assert_allclose(
        results["MAE/cross/2-by-1"], 1.7883245100546448, atol=1e-2
    )


def test_apply_resize_transformation_as_is():
    xs = [1, 2, 3]
    size = 3
    imputation_value = 0
    result = _apply_resize_transformation(xs, size, imputation_value)
    assert np.array_equal(result, xs)


def test_apply_resize_transformation_truncation():
    xs = [1, 2, 3]
    size = 2
    imputation_value = 0
    result = _apply_resize_transformation(xs, size, imputation_value)
    assert len(result) == size
    assert set(result).issubset(set(xs))


def test_apply_resize_transformation_imputation():
    xs = [1, 2]
    size = 3
    imputation_value = 0
    result = _apply_resize_transformation(xs, size, imputation_value)
    assert len(result) == size
    assert set(result) == {1, 2, 0}


def test_compute_mean_resized_by_second_different_group_sizes():
    df_x = pd.DataFrame(
        {
            "is_female": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "parity": [1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
        }
    )

    df_y = pd.DataFrame(
        {
            "is_female": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            "parity": [5, 6, 7, 7, 7, 8, 8, 8, 9, 9],
        }
    )

    x = df_x.groupby("is_female")["parity"]
    y = df_y.groupby("is_female")["parity"]

    result = _compute_mean_resized_by_second(x, y)

    np.testing.assert_allclose(result["max/diff"], 14 / 3)
    assert result["min/size_by_second"] == 2


def test_evaluate_utility(basic_all_categorical_real_df):  # noqa: F811
    "Test utility evaluation on same real and synth dataframes."

    (
        df,
        schema,
        categoricals,
        params,
        processed_df,
        rev_params,
        rev_processed_df,
    ) = basic_all_categorical_real_df

    numerified_df = numerify(df)
    numerified_rev_processed_df = numerify(rev_processed_df)

    evaluation_bins = {
        "binary": [0, 1, 2],
        "nominal": [0, 5, np.inf],
        "date": np.arange(0, 366 + 31, 31),
    }

    results_df, subsampled_real_df = evaluate_utility(
        numerified_df,
        numerified_df,
        evaluation_bins,
        0.9,
        1234,
        numerified_rev_processed_df,
    )

    # real = synth, so same results
    np.testing.assert_array_almost_equal(results_df["val_pr_r"], results_df["val_pr_s"])

    mask_one = results_df["name"].str.contains("(ratio|cdf|chi2|std)", regex=True) & (
        ~results_df["name"].str.contains("lr")
    )
    mask_half = results_df["name"] == "pMSE/acc"
    mask_zero = ~(mask_one | mask_half)

    assert (results_df.loc[mask_one, "val_r_s"] == 1).all()
    assert (results_df.loc[mask_half, "val_r_s"] == 0.5).all()
    assert (results_df.loc[mask_zero, "val_r_s"] == 0).all()

    # no replacement
    assert not subsampled_real_df.duplicated().any()
