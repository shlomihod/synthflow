import numpy as np
import pandas as pd
import pytest

from synthflow.birth import REAL_DATASET_SCHEMA
from synthflow.generation import generate
from synthflow.generation.post import _project_single_min_count
from synthflow.utils import interval_number
from tests.utils import (  # noqa: F401
    GEN_CONFIG_PATH,
    PROCESSED_REAL_DATA_200K_PATH,
    REAL_DATA_10K_PATH,
    basic_all_categorical_real_df,
    nvss_10k_data,
)


def test_generate_id(basic_all_categorical_real_df):  # noqa: F811
    """Test generating synthetic data with ID model."""

    (
        df,
        schema,
        categoricals,
        params,
        procesed_df,
        rev_params,
        rev_processed_df,
    ) = basic_all_categorical_real_df

    config = {
        "categoricals": categoricals,
        "delta": 0,
        "dp": "pure",
        "epsilon": 4.0,
        "dataset_projection": None,
        "hparams": {},
        "model": "ID",
        "transformations": {
            "binary": [{"name": "binary"}],
            "nominal": [{"name": "binning", "bins": [1, 2, 4, 7, 11]}],
            "date": [{"name": "datetime", "resolution": "month", "format": "%Y-%m-%d"}],
        },
    }

    synth_df_, _, _, real_processed_df_, _, _ = generate(df, config, schema)
    assert (rev_processed_df == real_processed_df_).all().all()
    assert synth_df_.equals(real_processed_df_)


def test_generate_constraints(nvss_10k_data):  # noqa: F811
    """Test generating synthetic data with PBN model, NVSS data and constraints."""

    real_df, gen_config, _ = nvss_10k_data

    constraints = [
        lambda r: r["mother_age"] < interval_number(37),
        lambda r: r["is_female"] | (r["parity"] < interval_number(4)),
    ]

    synth_df_, _, _, _, constraints_df, _ = generate(
        real_df, gen_config, REAL_DATASET_SCHEMA, constraints, ubersampling=2
    )

    assert len(synth_df_) == len(real_df)

    assert synth_df_["mother_age"].max() < interval_number(37)
    assert synth_df_.groupby("is_female")["parity"].max()[False] < interval_number(4)


def test_generate_wrong_hparam(nvss_10k_data):  # noqa: F811
    """Test generation failure due to passing a non-exiting hparam."""

    real_df, gen_config, _ = nvss_10k_data
    gen_config["hparams"]["fake"] = 1

    with pytest.raises(Exception) as e:
        generate(real_df, gen_config, REAL_DATASET_SCHEMA)
    assert e.type == AssertionError


def test_generate_not_enogth_rows(nvss_10k_data):  # noqa: F811
    """Test generation failure due to using constrains without enough rows left."""

    real_df, gen_config, _ = nvss_10k_data

    constraints = [
        lambda r: r["mother_age"] < interval_number(37),
        lambda r: r["is_female"] | (r["parity"] < interval_number(4)),
    ]

    with pytest.raises(Exception) as e:
        generate(
            real_df, gen_config, REAL_DATASET_SCHEMA, constraints, ubersampling=1.01
        )
    assert e.type == RuntimeError


def _projection_tester(x_df, y_df, order):
    x_counts = x_df.value_counts()
    y_counts = y_df.value_counts()

    assert (x_counts <= order).any()
    assert not (y_counts <= order).any()

    x_higher_order_index = list(x_counts[x_counts > order + 1].index)
    y_higher_order_index = list(y_counts[y_counts > order + 1].index)
    joint_higher_order_index = list(
        set(x_higher_order_index) & set(y_higher_order_index)
    )
    assert (
        x_counts.loc[joint_higher_order_index] == y_counts[joint_higher_order_index]
    ).all()

    x_lower_order_index = list(x_counts[x_counts < order + 1].index)
    joint_at_order_index = list(set(x_lower_order_index) & set(y_counts.index))
    np.testing.assert_allclose(
        len(joint_at_order_index) / len(x_lower_order_index),
        1 / (order + 1),
        rtol=0.2,
    )


def test_project_single_min_count():
    """Test generating min count projection."""

    order = 1
    x_df = pd.read_pickle(PROCESSED_REAL_DATA_200K_PATH)

    y_df = _project_single_min_count(x_df, order)

    _projection_tester(x_df, y_df, order)


def test_projection(nvss_10k_data):  # noqa: F811
    """Test generating synthetic data with doubling projection."""

    real_df, _, config = nvss_10k_data

    config["model"] = "ID"
    config["hparams"] = {}

    synth_df_, _, _, real_processed_df_, _, _ = generate(
        real_df, config, REAL_DATASET_SCHEMA
    )

    _projection_tester(
        real_processed_df_, synth_df_, order=config["dataset_projection"]["order"]
    )
