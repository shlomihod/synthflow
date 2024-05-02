import json
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from synthflow.birth import REAL_DATASET_SCHEMA

REAL_DATA_10K_PATH = (
    Path(__file__).parent / "data" / "birth_2019_nber_us_SINGLETON_SAMPLE10K.csv"
)
REAL_DATA_200K_PATH = (
    Path(__file__).parent / "data" / "birth_2019_nber_us_SINGLETON_SAMPLE200K.csv.gz"
)
GEN_CONFIG_PATH = (
    Path(__file__).parent / "data" / "PBNTheta-cpu-6-7-41580-0-2921-3d1a-3b41.json"
)
GEN_CONFIG_WITH_PROJECTION_PATH = (
    Path(__file__).parent / "data" / "PBNTheta-cpu-6-7-41580-0-2921-3d1a-f51f.json"
)
SYNTH_DATA_200K_PATH = Path(__file__).parent / "data" / "synth.pkl.gz"
PROCESSED_REAL_DATA_200K_PATH = Path(__file__).parent / "data" / "processed.pkl.gz"

WANDB_ONLINE_RUN_DIR_PATH = "synthflow/100hhmek"


@pytest.fixture
def basic_all_categorical_real_df():
    df = pd.DataFrame(
        {
            "binary": [True, False] * 5,
            "nominal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "date": [f"25/{month}/2020" for month in range(3, 13)],
        }
    )

    schema = pa.DataFrameSchema(
        {
            "binary": pa.Column(pa.dtypes.Bool),
            "nominal": pa.Column(pa.dtypes.Int, pa.Check(lambda x: x.between(1, 10))),
            "date": pa.Column(pa.dtypes.DateTime, coerce=True),
        },
        strict=False,
    )

    df = schema.validate(df)

    categoricals = list(df.columns)

    params = {
        "binary": [{"name": "binary"}],
        "nominal": [{"name": "binning", "bins": [1, 2, 4, 7, 11]}],
        "date": [{"name": "datetime", "resolution": "month", "format": "%Y-%m-%d"}],
    }

    processed_df = pd.DataFrame(
        {
            "binary": [1, 0] * 5,
            "nominal": [0] + [1] * 2 + [2] * 3 + [3] * 4,
            "date": range(2, 12),
        }
    )

    rev_params = {
        "nominal": [
            {
                "name": "rev_binning",
                "categories": [
                    pd.Interval(x, y, closed="left")
                    for x, y in [(1, 2), (2, 4), (4, 7), (7, 11)]
                ],
            }
        ],
        "date": [{"name": "rev_datetime", "resolution": "month", "year": 2020}],
    }

    rev_processed_df = pd.DataFrame(
        {
            "binary": df["binary"],
            "nominal": pd.Categorical(
                [
                    pd.Interval(x, y, closed="left")
                    for x, y in [(1, 2)] + [(2, 4)] * 2 + [(4, 7)] * 3 + [(7, 11)] * 4
                ],
                ordered=True,
            ),
            "date": [
                pd.Timestamp(year=date.year, month=date.month, day=15)
                for date in df["date"]
            ],
        }
    )

    return df, schema, categoricals, params, processed_df, rev_params, rev_processed_df


@pytest.fixture
def nvss_10k_data():
    real_df = pd.read_csv(REAL_DATA_10K_PATH)
    real_df = REAL_DATASET_SCHEMA.validate(real_df)

    with open(GEN_CONFIG_PATH) as f:
        gen_config = json.load(f)

    with open(GEN_CONFIG_WITH_PROJECTION_PATH) as f:
        gen_config_with_projection = json.load(f)

    return real_df, gen_config, gen_config_with_projection
