import numpy as np
import pytest

from synthflow.generation.processing import _prepare_dtypes, _process
from tests.utils import basic_all_categorical_real_df  # noqa: F401


def test_process_missing_param(basic_all_categorical_real_df):  # noqa: F811
    """Test that params with a missing column should fail."""

    df, schema, categoricals, *_ = basic_all_categorical_real_df

    params = {"binary": None, "nominal": None}

    with pytest.raises(Exception) as e:
        _process(df, params, categoricals, schema, is_rev=False)
    assert e.type == AssertionError
    assert "Generation config file should contain all columns in the dataset." in str(
        e.value
    )


def test_process_bin_min_max(basic_all_categorical_real_df):  # noqa: F811
    """Test that binning that do not include the boundary values should fail."""

    df, schema, categoricals, *_ = basic_all_categorical_real_df

    base_params = {
        "binary": [{"name": "binary"}],
        "nominal": [{"name": "binning", "bins": []}],
        "date": [{"name": "datetime", "resolution": "month", "format": "%Y-%m-%d"}],
    }

    edge_error_msg = (
        "The lower-most bin edge of column `nominal`"
        " is higher than the minimal value in the data"
        " OR the higher-most bin edge is lower"
        " than the maximal valuein the data,"
        " you should have clip this column"
        " or change the bin edges independently of the real data."
    )

    lower_wrong_params = base_params.copy()
    lower_wrong_params["nominal"][0]["bins"] = [2, 4, 7, 11]

    with pytest.raises(Exception) as e:
        _process(df, lower_wrong_params, categoricals, schema, is_rev=False)
    assert e.type == AssertionError
    assert edge_error_msg in str(e.value)

    upper_wrong_params = base_params.copy()
    upper_wrong_params["nominal"][0]["bins"] = [1, 2, 4, 7, 10]

    with pytest.raises(Exception) as e:
        _process(df, upper_wrong_params, categoricals, schema, is_rev=False)
    assert e.type == AssertionError
    assert edge_error_msg in str(e.value)


def test_process_forward_backward(basic_all_categorical_real_df):  # noqa: F811
    """Test applying preprocessing and then postprocessing (rev) and its invariance"""

    (
        df,
        schema,
        categoricals,
        params,
        procesed_df,
        rev_params,
        rev_processed_df,
    ) = basic_all_categorical_real_df

    # forward

    processed_df, boundaries, _, rev_params_, _ = _process(
        df, params, categoricals, schema, is_rev=False
    )

    assert (processed_df == processed_df).all().all()

    assert boundaries == {"binary": (0, 1), "nominal": (0, 3), "date": (0, 11)}

    # backward

    rev_processed_df_, _, _, _, rev_processed_schema = _process(
        processed_df, rev_params_, [], schema, is_rev=True
    )

    # binary variables are identified from the original scheme
    assert rev_params == rev_params_

    assert (rev_processed_df == rev_processed_df_).all().all()


def test_prepare_dtypes_forward_backward(basic_all_categorical_real_df):  # noqa: F811
    """Test applying dtype preperation and then rev preperation and its invariance"""

    df, schema, *_ = basic_all_categorical_real_df

    # forward

    prepared_df, prepared_schema = _prepare_dtypes(df, schema, is_rev=False)

    prepared_df_dtypes = prepared_df.dtypes.to_dict()
    assert (
        prepared_df_dtypes["binary"] == np.dtype("int64")
        and prepared_df_dtypes["nominal"] == np.dtype("float64")
        and prepared_df_dtypes["date"] == np.dtype("<M8[ns]")
    )

    prepared_schema.validate(prepared_df)

    # backward

    rev_prepred_df, _ = _prepare_dtypes(prepared_df, schema, is_rev=True)

    rev_prepared_df_dtypes = rev_prepred_df.dtypes.to_dict()
    assert (
        rev_prepared_df_dtypes["binary"] == np.dtype("bool")
        and rev_prepared_df_dtypes["nominal"] == np.dtype("int64")
        and rev_prepared_df_dtypes["date"] == np.dtype("<M8[ns]")
    )

    schema.validate(rev_prepred_df)
