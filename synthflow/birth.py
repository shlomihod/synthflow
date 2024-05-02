"""
Gravidity = the number of times a woman is or has been pregnant
Parity = the number of times a woman carried the pregnancies to a viable gestational age

https://en.wikipedia.org/wiki/Gravidity_and_parity
https://www.perinatology.com/Q&A/qanda46.htm
"""
import itertools as it
from datetime import datetime

import numpy as np
import pandera as pa

from synthflow.evaluation.acceptance import AcceptanceCriteria
from synthflow.evaluation.utility.analysis_structures import ColumnAnalysis2Way
from synthflow.utils import get_single_row_by_query, interval_number

ALPHAS = [1]

REAL_DATASET_SCHEMA = pa.DataFrameSchema(
    {
        "mother_age": pa.Column("int64", pa.Check(lambda x: x.between(12, 60))),
        "parity": pa.Column("int64", pa.Check(lambda x: x.between(1, 24))),
        "gestation_week": pa.Column(int, pa.Check(lambda x: x.between(17, 47))),
        "is_female": pa.Column(bool),
        "date_of_birth": pa.Column(datetime, coerce=True),
        "birth_weight": pa.Column("int64", pa.Check(lambda x: x.between(200, 8200))),
    },
    strict=False,
    ordered=False,
)

BOUNDARIES = {
    "mother_age": (17, 46),
    "parity": (1, 11),
    "gestation_week": (28, 43),
    "is_female": (0, 1),
    "date_of_birth": (0, 365),
    "birth_weight": (1400, 4600),
}

IMPUTATION_MIDPOINT_VALUES = {
    column: (lower + upper) / 2 for column, (lower, upper) in BOUNDARIES.items()
}

# Here we must have both left and right ultimate edges
EVALUATION_COLUMN_BINS = {
    "mother_age": [0, 25, 30, 35, np.inf],
    "birth_weight": [0, 2500, 4000, np.inf],
    "parity": [1, 2, 4, np.inf],
    "gestation_week": [0, 37, np.inf],
    "date_of_birth": np.arange(0, 366 + 31, 31),
    "is_female": [0, 1, 2],
}

# Here we do not have to give both boundaries
DATA_TRANSFORMATION_SPECS = {
    "is_female": {"type": "binary"},
    "mother_age": {
        "type": "continuous",
        "lower": 17,
        "upper": 45,
        "bins": [
            [17, 18, 20, 25, 30, 35, 40, 43, 45, 46],
            [17, 18, 20, 25, 30, 35, 37, 40, 43, 45, 46],
        ],
        "also_unit_binning": True,
    },
    "birth_weight": {
        "type": "continuous",
        "lower": 1400,
        "upper": 4500,
        "bins": [
            list(range(1400, 4700, 100)),
        ],
        "also_unit_binning": False,
    },
    "parity": {
        "type": "continuous",
        "lower": 1,
        "upper": 11,
        "bins": [
            [1, 2, 4, 7, 11, 12],
        ],
        "also_unit_binning": True,
    },
    "gestation_week": {
        "type": "continuous",
        "lower": 28,
        "upper": 42,
        "bins": [
            [28, 29, 32, 34, 37, 42, 43],
        ],
        "also_unit_binning": True,
    },
    "date_of_birth": {
        "type": "datetime",
        "format": "%d/%m/%Y",
    },
}


DATE_TRANSFORMATION_ONLY_AS_MONTH = True

COLUMN_WEIGHTS = {
    "mother_age": [0, 18, 20, 25, 30, 35, 37, 40, 43, np.inf],
    "birth_weight": 1 / 100,
    "date_of_birth": 1 / 14,
    "gestation_week": [0, 29, 32, 34, 37, 42, np.inf],
    "is_female": 100,
    "parity": 100,
}


MOH_GEN_CONFIG = {
    "id": "moh",
    "epsilon": 1e500,
    "delta": 0,
    "model": "ID",
    "hparams": {},
    "transformations": {
        "is_female": [{"name": "binary"}],
        "date_of_birth": [
            {"name": "datetime", "format": "%d/%m/%Y", "resolution": "month"}
        ],
        "mother_age": [{"name": "clipping", "lower": 18, "upper": 50}],
        "gestation_week": [
            {"name": "clipping", "lower": 25, "upper": 50},
            {"name": "binning", "bins": [25, 40, 51]},
        ],
        "parity": [
            {"name": "clipping", "lower": 1, "upper": 20},
            {"name": "binning", "bins": [1, 2, 3, 4, 5, 6, 7, 8, 21]},
        ],
        "birth_weight": [
            {"name": "clipping", "lower": 1000, "upper": 5000},
            {"name": "resolution", "scale": 100},
        ],
    },
    "categoricals": [
        "mother_age",
        "gestation_week",
        "parity",
    ],
    "num_categorical": 4,
    "num_cat_cells": 491520,
    "space_limited": False,
    "num_one_hot": 0,
    "gpu": False,
    "categorical_mode": "none",
    "tier": 0,
}


USER_ANALYSIS_BY_METRIC = [
    [
        ColumnAnalysis2Way(metric, "birth_weight", column)
        for column in (
            "is_female",
            "parity",
            "gestation_week",
            "mother_age",
        )
    ]
    + [
        ColumnAnalysis2Way(
            metric,
            "gestation_week",
            column,
        )
        for column in (
            "parity",
            "mother_age",
        )
    ]
    + [ColumnAnalysis2Way(metric, "parity", "mother_age")]
    for metric in ["mean", "median", "mean-resize-by-second"]
]


USER_ANALYSIS = list(it.chain(*USER_ANALYSIS_BY_METRIC))


THRESHOLD_ALL_K_WAYS_MAX_ABS_DIFF = 0.01


ACCEPTANCE_CRITERIA = [
    AcceptanceCriteria(
        "utility/1way/max/frequencies/ratio/raw/val_pr_s",
        lambda utility_df, _, __: get_single_row_by_query(
            utility_df, "name == '1way/max/frequencies/ratio/raw'"
        )["val_pr_s"],
        1.4,  # see comment below
    ),
    AcceptanceCriteria(
        "utility/123456way/max/frequencies/diff/raw/val_pr_s",
        lambda utility_df, _, __: get_single_row_by_query(
            utility_df, "name == '123456way/max/frequencies/diff/raw'"
        )["val_pr_s"],
        THRESHOLD_ALL_K_WAYS_MAX_ABS_DIFF,
    ),
    AcceptanceCriteria(
        "utility/mean/parity/l_inf/val_pr_s",
        lambda utility_df, _, __: utility_df.query(
            "(name == 'mean-resize-by-second/max/diff' & target == 'parity')"
            " | (name == 'mean' & target == 'parity' & by.isna())"
        )["val_pr_s"].max(),
        0.3,
    ),
    AcceptanceCriteria(
        "utility/mean/birth_weight/l_inf/val_pr_s",
        lambda utility_df, _, __: utility_df.query(
            "(name == 'mean-resize-by-second/max/diff' & target == 'birth_weight')"
            " | (name == 'mean' & target == 'birth_weight' & by.isna())"
        )["val_pr_s"].max(),
        100,
    ),
    AcceptanceCriteria(
        "utility/mean/gestation_week/l_inf/val_pr_s",
        lambda utility_df, _, __: utility_df.query(
            "(name == 'mean-resize-by-second/max/diff' & target == 'gestation_week')"
            " | (name == 'mean' & target == 'gestation_week' & by.isna())"
        )["val_pr_s"].max(),
        1,
    ),
    AcceptanceCriteria(
        "utility/lr/coef/error/diff/total/val_pr_s",
        lambda utility_df, _, __: get_single_row_by_query(
            utility_df, "name == 'lr/coef/error/diff/total'"
        )["val_pr_s"],
        30,
    ),
    AcceptanceCriteria(
        "utility/lr/MAE/cross/1-by-2/val_pr_s",
        lambda utility_df, _, __: get_single_row_by_query(
            utility_df, "name == 'lr/MAE/cross/1-by-2'"
        )["val_pr_s"],
        5,  # should be easy to get according to experiments on the NVSS data
    ),
    AcceptanceCriteria(
        "faithfulness/β/ɑ=1/val_pr_s",
        lambda _, __, faithfulness_df: get_single_row_by_query(
            faithfulness_df, "ɑ == 1 & comparison == 'val_pr_s'"
        )["β"],
        0.05,
    ),
    AcceptanceCriteria(
        "privacy/face/unique/1/max",
        lambda _, privacy_df, __: get_single_row_by_query(
            privacy_df, "name == 'unique/1/max'"
        )["value"],
        1,
    ),
    AcceptanceCriteria(
        "privacy/face/k-anonymity/1/max",
        lambda _, privacy_df, __: get_single_row_by_query(
            privacy_df, "name == 'k-anonymity/1/max'"
        )["value"],
        1,
    ),
]

# optimize_max_ratio_by_pseudo_threshold(prob_pass=.05, min_synth_count=50,
#                                        threshold=2, epsilon=.3)
# pseudo_threshold = 1.4095935658989616 ≈ 1.4
# prob_pass ≈ 4.82%
RATIO_MAX_CLIPPING_FACTOR = 2
HIGH_SENSITIVITY_EPSILON = 0.3
LOW_SENSITIVITY_EPSILON = 0.02
COMPLEX_QUERY_FIT_EPSILON = 0.43

# Justification: used for BW, and with eps=.02 the SD is ~3.16, which is quite high
# for anticipated actual value ~0, so let's double epsilon to get SD ~1.5
COMPLEX_QUERY_EVAL_EPSILON = 0.04

# Justification: taking into account the sensitivity of the conditioining process
# of EVALUATION_COLUMN_BINS
CONDITIONAL_MEAN_EPSILONS = {
    "parity": 0.01,
    "birth_weight": 0.17,
    "gestation_week": 0.02,
}


def _cannot_be_less_one_and_more_second(col1, val1, col2, val2):
    return lambda r: ~((r[col1] < val1) & (r[col2] > val2))


CONSTRAINTS = (
    [
        _cannot_be_less_one_and_more_second(
            "mother_age", mother_age_limit, "parity", parity_limit
        )
        for mother_age_limit, parity_limit in [
            (interval_number(23), interval_number(6)),
            (interval_number(20), interval_number(3)),
        ]
    ]
    # https://www.canada.ca/en/public-health/services/injury-prevention/health-surveillance-epidemiology-division/maternal-infant-health/birth-weight-gestational.html
    + [
        _cannot_be_less_one_and_more_second(
            "gestation_week", gestation_week_limit, "birth_weight", birth_weight_limit
        )
        for gestation_week_limit, birth_weight_limit in [
            (interval_number(29), interval_number(3000 - 1)),
            (interval_number(34), interval_number(4000 - 1)),
        ]
    ]
)

CONSTRAINTS_UBERSAMPLING = 1.1


QUASI_SENSITIVE_PAIRS = [
    (("mother_age", "parity", "is_female", "date_of_birth"), "birth_weight"),
    (("mother_age", "parity", "is_female", "date_of_birth"), "gestation_week"),
    (
        ("mother_age", "parity", "is_female", "date_of_birth", "gestation_week"),
        "birth_weight",
    ),
    (
        ("mother_age", "parity", "is_female", "date_of_birth", "birth_weight"),
        "gestation_week",
    ),
]


FACE_PRIVACY_UP_TO = 6
