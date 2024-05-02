import logging

import numpy as np
import pandas as pd
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms.laplace import Laplace
from diffprivlib.models import LinearRegression as LinearRegressionDP
from scipy import stats
from sklearn.linear_model import LinearRegression as LinearRegressionNotDP
from sklearn.preprocessing import StandardScaler

from synthflow.birth import BOUNDARIES
from synthflow.evaluation.preperation import numerify
from synthflow.evaluation.utility.metrics import _compute_linear_regression
from synthflow.utils import get_single_row_by_query

LOGGER = logging.getLogger(__name__)


def calc_ratio_possible_sensitivity(min_synth_count, clipping_factor):
    return (
        1 / min_synth_count,
        clipping_factor - 1 / (1 / clipping_factor + 1 / min_synth_count),
    )


def calc_ratio_max_sensitivity(min_synth_count, clipping_factor):
    return max(calc_ratio_possible_sensitivity(min_synth_count, clipping_factor))


def _apply_mech_spent_budget(accountant, mech, value):
    private_value = mech.randomise(value)
    accountant.spend(epsilon=mech.epsilon, delta=mech.delta)
    return private_value


def _private_high_sensitivity_criteria(
    accountant,
    dp_acceptance_df,
    epsilon,
    utility_df,
    synth_df,
    ratio_max_clipping_factor,
):
    """
    Let c be a column in the data, and v be a value in a column.
    Let r(c, v) and s(c, v) be the counts of value v in clolumn c
    in the real and synthetic data, respectively.
    The ratio_max is the max_c max_v max { r/s, s/r }$.
    To release it, we need to calculate the global sensitivity of the column ration_max.

    Claim: Let q_1, ..., q_k be queries with sensitivty d_1, ..., d_k.
    Then the sensitivity of max_i q_i is max_i d_i.

    Therefore, we need to find the maximial global sensitivity of r/s and s/r.

    Let min_synth_count be the minimal count of any value
    in any column in the synthetic data.

    The global sensitivity of r/s is 1/min_synth_count.

    The global sensitivity of s/r is much more complicated.
    It is max_synth_cout/2.

    But, if we clip the ratio of s/r to
    [1/ratio_max_clipping_factor, ratio_max_clipping_factor],
    we bound r to [s / clipping_factr, s * ratio_max_clipping_factor],

    Let f(r, s) = s/r - s/(r+1), so max f(r) is the global sensitivy of s/r.
    f(r, *) is monotonically decreasing for r > 0, so it gets its maximum
    for the minimal r, which is paired_min_real_count = s/ratio_max_clipping_factor.

    So we get max_s max_r f(r, s) = max_s f(paired_min_real_count, s)
    = ratio_max_clipping_factor - 1 / (1/ratio_max_clipping_factor + 1/min_synth_count).
    And this is the global sensitivity of the clipped s/r.

    So the global sensitivity of the max ratio query is
    max { 1/min_synth_count,
          ratio_max_clipping_factor - 1
          / (1/ratio_max_clipping_factor + 1/min_synth_count) }.

    Note: Although we do not neex to clip r/s, for simplicity we do.
    """

    raw_one_way_mask = (
        utility_df["targets"]
        .replace({np.nan: None})
        .apply(lambda x: False if x is None else len(x) == 1)
        & ~utility_df["binning"]
    )
    ratio_max = utility_df.loc[
        (utility_df["name"] == "frequencies/ratio/max") & raw_one_way_mask, "val_pr_s"
    ].max()

    clipped_ratio_max = np.clip(
        ratio_max, 1 / ratio_max_clipping_factor, ratio_max_clipping_factor
    )

    min_synth_count = min(
        synth_df[column].value_counts().min() for column in synth_df.columns
    )

    ratio_max_sensitivity = calc_ratio_max_sensitivity(
        min_synth_count, ratio_max_clipping_factor
    )

    ratio_max_mech = Laplace(
        epsilon=epsilon, delta=0, sensitivity=ratio_max_sensitivity
    )

    dp_ratio_max = _apply_mech_spent_budget(
        accountant, ratio_max_mech, clipped_ratio_max
    )

    expected = get_single_row_by_query(
        dp_acceptance_df, "name =='utility/1way/max/frequencies/ratio/raw/val_pr_s'"
    )["expected"]

    dp_ratio_max_row = pd.DataFrame(
        {
            "name": "utility/1way/max/frequencies/ratio/raw/val_pr_s",
            "expected": expected,
            "actual": dp_ratio_max,
            "lower": 1,
            "upper": ratio_max_clipping_factor,
            "epsilon": ratio_max_mech.epsilon,
            "var": ratio_max_mech.variance(None),
            "sensitivity": ratio_max_mech.sensitivity,
            "mech": "laplace",
        },
        index=[0],
    )

    dp_acceptance_df = dp_acceptance_df[
        dp_acceptance_df["name"] != "utility/1way/max/frequencies/ratio/raw/val_pr_s"
    ]

    return pd.concat([dp_ratio_max_row, dp_acceptance_df]).reset_index(drop=True)


def _private_low_sensitivity_criteria(
    accountant, dp_acceptance_df, epsilon, num_records, boundaries
):
    unit_interval_criteria_mask = dp_acceptance_df["name"].str.contains(
        "|".join(["frequencies/diff", "faithfulness/β"])
    )
    dp_acceptance_df.loc[unit_interval_criteria_mask, ["lower", "upper"]] = 0, 1

    lap_mask = unit_interval_criteria_mask
    num_values = lap_mask.sum()

    def mean_lap(row):
        mech = Laplace(
            epsilon=epsilon / num_values,
            delta=0,
            sensitivity=(row["upper"] - row["lower"]) / num_records,
        )

        dp_value = _apply_mech_spent_budget(accountant, mech, row["actual"])

        return pd.Series(
            {
                "actual": dp_value,
                "var": mech.variance(None),
                "epsilon": mech.epsilon,
                "sensitivity": mech.sensitivity,
                "mech": "laplace",
            }
        )

    dp_acceptance_df.loc[
        lap_mask, ["actual", "var", "epsilon", "sensitivity", "mech"]
    ] = dp_acceptance_df.loc[lap_mask].apply(mean_lap, axis=1)
    return dp_acceptance_df


def _private_conditional_mean_criteria(
    accountant, dp_acceptance_df, epsilons, utility_df, boundaries
):
    mean_criteria_mask = dp_acceptance_df["name"].str.contains("mean")
    mean_lower, mean_upper = zip(
        *dp_acceptance_df.loc[mean_criteria_mask, "name"]
        .str.split("/")
        .str[2]
        .map(boundaries)
        .values
    )

    dp_acceptance_df.loc[mean_criteria_mask, "lower"] = mean_lower
    dp_acceptance_df.loc[mean_criteria_mask, "upper"] = mean_upper

    lap_mask = mean_criteria_mask

    def mean_lap(row):
        target_column = row["name"].split("/")[2]
        sizes = utility_df.query(
            f"name == 'mean-resize-by-second/min/size_by_second'"
            f" & target == '{target_column}'"
        )["val_pr_s"]
        mech = Laplace(
            epsilon=epsilons[target_column],
            delta=0,
            sensitivity=(row["upper"] - row["lower"]) / sizes.min(),
        )

        dp_value = _apply_mech_spent_budget(accountant, mech, row["actual"])

        return pd.Series(
            {
                "actual": dp_value,
                "var": mech.variance(None),
                "epsilon": mech.epsilon,
                "sensitivity": mech.sensitivity,
                "mech": "laplace",
                "extra": list(sizes),
            }
        )

    dp_acceptance_df.loc[
        lap_mask, ["actual", "var", "epsilon", "sensitivity", "mech", "extra"]
    ] = dp_acceptance_df.loc[lap_mask].apply(mean_lap, axis=1)
    return dp_acceptance_df


def _private_complex_query_criteria(
    accountant,
    dp_acceptance_df,
    complex_query_fit_epsilon,
    complex_query_eval_epsilon,
    num_records,
    processed_real_df,
    synth_df,
    boundaries,
):
    bounds = [boundaries[column] for column in processed_real_df.columns]
    lower, upper = zip(*bounds)
    _, bounds_y = (lower[:-1], upper[:-1]), (lower[-1], upper[-1])

    def not_dp_lr_factory():
        return

    X = StandardScaler().fit_transform(synth_df.iloc[:, :-1])
    bounds_X = X.min(axis=0), X.max(axis=0)
    dp_model = LinearRegressionDP(
        epsilon=complex_query_fit_epsilon,
        bounds_X=bounds_X,
        bounds_y=bounds_y,
        accountant=accountant,
    )

    processed_real_df = numerify(processed_real_df)
    synth_df = numerify(synth_df)

    results = _compute_linear_regression(
        processed_real_df,
        synth_df,
        lambda: dp_model,
        lambda: LinearRegressionNotDP(),
        bounds_y,
    )

    mech_cross_mae = Laplace(
        epsilon=complex_query_eval_epsilon,
        delta=0,
        sensitivity=2 * (bounds_y[1] - bounds_y[0]) / num_records,
        # Difference between two n*MAE is bounded by
        # 2|U-L|/n
    )

    dp_value_cross_mae = _apply_mech_spent_budget(
        accountant, mech_cross_mae, results["MAE/cross/1-by-2"]
    )

    expected_coef = get_single_row_by_query(
        dp_acceptance_df, "name == 'utility/lr/coef/error/diff/total/val_pr_s'"
    )["expected"]

    expected_cross_mae = get_single_row_by_query(
        dp_acceptance_df, "name == 'utility/lr/MAE/cross/1-by-2/val_pr_s'"
    )["expected"]

    dp_lr = pd.DataFrame(
        [
            {
                "name": "utility/lr/coef/error/diff/total/val_pr_s",
                "expected": expected_coef,
                "actual": results["coef/error/diff/total"],
                "epsilon": complex_query_fit_epsilon,
                "mech": "lr",
                "extra": list(dp_model.coef_) + [dp_model.intercept_],
            },
            {
                "name": "utility/lr/MAE/cross/1-by-2/val_pr_s",
                "expected": expected_cross_mae,
                "actual": dp_value_cross_mae,
                "lower": bounds_y[0],
                "upper": bounds_y[1],
                "var": mech_cross_mae.variance(None),
                "epsilon": mech_cross_mae.epsilon,
                "sensitivity": mech_cross_mae.sensitivity,
                "mech": "laplace",
            },
        ],
        index=[5, 6],
    )

    dp_acceptance_df = dp_acceptance_df[
        ~dp_acceptance_df["name"].str.contains("utility/lr")
    ]

    return pd.concat([dp_acceptance_df, dp_lr]).sort_index().reset_index(drop=True)


def optimize_max_ratio_by_pseudo_threshold(
    prob_pass, min_synth_count, threshold, epsilon
):
    """
    We want to reduce the sd of the nosie. The SD (by the GS) is monitically decreasing
    as function of the clipping factor (given a fixed epsilon).
    Therefore, we could set the clipping factor to the actual threshold,
    and then set "pesudo-threshold" which is lower,
    and that the probability of the noise + pesudo threshold
    to pass the actual threshold is bounded by a probability prob_pass.

    So we want to calculate:
    param = GS(k) / eps
    CDF(threshold - pseudo_threshold, param) = prob_pass
    pesudo_threshold = threshold - PPF(prob_pass, param)
    """

    clipping_factor = threshold
    global_sensitivity = calc_ratio_max_sensitivity(min_synth_count, clipping_factor)
    scale = global_sensitivity / epsilon
    pseudo_threshold = threshold - stats.laplace.ppf(1 - prob_pass, scale=scale)
    return pseudo_threshold, clipping_factor, global_sensitivity, scale


def private_acceptance_criteria(
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
):
    accountant = BudgetAccountant()

    num_records = len(processed_real_df)

    dp_acceptance_df = acceptance_df.copy()
    dp_acceptance_df = dp_acceptance_df.assign(
        lower=None, upper=None, epsilon=None, var=None, mech=None
    )

    dp_acceptance_df = _private_high_sensitivity_criteria(
        accountant,
        dp_acceptance_df,
        high_sensitivity_epsilon,
        utility_df,
        synth_df,
        ratio_max_clipping_factor,
    )

    dp_acceptance_df = _private_low_sensitivity_criteria(
        accountant, dp_acceptance_df, low_sensitivity_epsilon, num_records, BOUNDARIES
    )

    dp_acceptance_df = _private_conditional_mean_criteria(
        accountant, dp_acceptance_df, conditional_mean_epsilons, utility_df, BOUNDARIES
    )

    dp_acceptance_df = _private_complex_query_criteria(
        accountant,
        dp_acceptance_df,
        complex_query_fit_epsilon,
        complex_query_eval_epsilon,
        num_records,
        processed_real_df,
        synth_df,
        BOUNDARIES,
    )

    all_mask = dp_acceptance_df["name"] == "all"
    assert all_mask.sum() == 1

    dp_acceptance_df.loc[~all_mask, "check"] = (
        dp_acceptance_df.loc[~all_mask, "actual"]
        < dp_acceptance_df.loc[~all_mask, "expected"]
    )
    dp_acceptance_df.loc[all_mask, "check"] = dp_acceptance_df.loc[
        ~all_mask, "check"
    ].all()
    dp_acceptance_df.loc[all_mask, "actual"] = dp_acceptance_df.loc[
        all_mask, "check"
    ].astype(float)
    dp_acceptance_df.loc[all_mask, "epsilon"] = dp_acceptance_df.loc[
        ~all_mask, "epsilon"
    ].sum()

    spent_epsilon, spent_delta = accountant.total()

    np.testing.assert_allclose(
        spent_epsilon,
        high_sensitivity_epsilon
        + low_sensitivity_epsilon
        + sum(conditional_mean_epsilons.values())
        + complex_query_fit_epsilon
        + complex_query_eval_epsilon,
    )

    assert spent_delta == 0

    return dp_acceptance_df, accountant
