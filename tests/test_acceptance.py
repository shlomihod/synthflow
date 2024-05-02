from copy import deepcopy

import pandas as pd

from synthflow.evaluation.acceptance import AcceptanceCriteria, check_acceptance


def test_check_acceptance():
    """Test whether evaluation is accepted."""

    utility_df = pd.DataFrame(
        {
            "metrics": ["median", "mean1", "mean2", "mean3", "var"],
            "values": [1, 1.2, 1.1, 1.5, 3.5],
        }
    )

    privacy_df = pd.DataFrame()

    faithfulness_df = pd.DataFrame([{"α": 1, "β": 0.5}])

    criteria_base = [
        AcceptanceCriteria(
            "median", lambda u, p, f: u.query("metrics == 'median'")["values"].item(), 2
        ),
        AcceptanceCriteria(
            "max_mean",
            lambda u, p, f: u[u["metrics"].str.startswith("mean")]["values"].max(),
            1.7,
        ),
        AcceptanceCriteria(
            "faithfulness", lambda u, p, f: f.query("α == 1")["β"].item(), 0.75
        ),
    ]

    criteria_accept_all = deepcopy(criteria_base)
    df_accept_all = check_acceptance(
        criteria_accept_all, utility_df, privacy_df, faithfulness_df
    )
    assert df_accept_all.equals(
        pd.DataFrame(
            {
                "name": {0: "median", 1: "max_mean", 2: "faithfulness", 3: "all"},
                "expected": {0: 2.0, 1: 1.7, 2: 0.75, 3: 1.0},
                "actual": {0: 1.0, 1: 1.5, 2: 0.5, 3: 1.0},
                "check": {0: True, 1: True, 2: True, 3: True},
            }
        )
    )

    criteria_fail_on_threshold = deepcopy(criteria_base)
    criteria_fail_on_threshold[0] = AcceptanceCriteria(
        "median", lambda u, p, f: u.query("metrics == 'median'")["values"].item(), 1
    )
    df_fail_on_threshold = check_acceptance(
        criteria_fail_on_threshold, utility_df, privacy_df, faithfulness_df
    )
    assert df_fail_on_threshold.equals(
        pd.DataFrame(
            {
                "name": {0: "median", 1: "max_mean", 2: "faithfulness", 3: "all"},
                "expected": {0: 1.0, 1: 1.7, 2: 0.75, 3: 1.0},
                "actual": {0: 1.0, 1: 1.5, 2: 0.5, 3: 0.0},
                "check": {0: False, 1: True, 2: True, 3: False},
            }
        )
    )

    criteria_fail_above_threshold = deepcopy(criteria_base)
    criteria_fail_above_threshold[0] = AcceptanceCriteria(
        "median", lambda u, p, f: u.query("metrics == 'median'")["values"].item(), 0.5
    )
    df_fail_above_threshold = check_acceptance(
        criteria_fail_above_threshold, utility_df, privacy_df, faithfulness_df
    )
    assert df_fail_above_threshold.equals(
        pd.DataFrame(
            {
                "name": {0: "median", 1: "max_mean", 2: "faithfulness", 3: "all"},
                "expected": {0: 0.5, 1: 1.7, 2: 0.75, 3: 1.0},
                "actual": {0: 1.0, 1: 1.5, 2: 0.5, 3: 0.0},
                "check": {0: False, 1: True, 2: True, 3: False},
            }
        )
    )
