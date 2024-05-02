from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pandas.api.types as ptypes  # type: ignore
import pandera as pa
from rdt import HyperTransformer  # type: ignore
from rdt import transformers  # type: ignore

DATETIME_RESOLUTIONS = {"dayofyear", "week", "month"}


# TODO: assume no float, tailored made for birth dataset
def _prepare_dtypes(
    df: pd.DataFrame, schema: pa.DataFrameSchema, is_rev: bool
) -> tuple[pd.DataFrame, pa.DataFrameSchema]:
    """
    Convert types of DataFrame's columns according to a schema and rev/no-rev.

    If not reverse (is_rev=True), then the types should be all float or ints.
    The function does not handle pandas' categorical and datatime types.

    Force int64 anf float64 for consistency between Windows and Unix.
    See here: https://github.com/unionai-oss/pandera/issues/726

    Logic:
    - `bool`
      - Forward (no-rev): to int
      - Backward (rev): clip(0, 1) and then to bool
    - numeric (e.g., int, float)
      - Forward (no-rev): to float
      - Backward (rev): round() and then to int

    Args:
        df (DataFrame): The DataFrame to be converted.
        schema (DataFrameSchema): The Schema of the DataFrame.
        is_rev (bool): Whether this is a reverse transformation application.

    Returns:
        DataFrame: The DataFrame after conversion.
        DataFrameSchema: The schema of the converted DataFrame.
    """

    df = df.copy()
    schema = deepcopy(schema)

    for column in df.columns:
        dtype = schema.columns[column].dtype.type

        if ptypes.is_categorical_dtype(df[column].dtype):
            assert is_rev, f"{column} type cannot be categorical when preprocessing."
            schema = schema.remove_columns([column])
            # TODO: should be checked but raises an exception
            # TypeError: Cannot compare a Categorical for op __ge__
            # with a scalar, which is not a category.
            # schema = schema.update_column(
            #      column,
            #      dtype=pa.Category(ordered=True),
            #  )

        elif ptypes.is_bool_dtype(dtype):
            if is_rev:
                df[column] = df[column].clip(0, 1).round().astype(bool)
                schema = schema.update_column(column, dtype=bool)
            else:
                df[column] = df[column].astype("int64")
                schema = schema.update_column(column, dtype="int64")

        elif ptypes.is_numeric_dtype(dtype):
            if is_rev:
                df[column] = df[column].round().astype("int64")
                schema = schema.update_column(column, dtype="int64")
            else:
                df[column] = df[column].astype("float64")
                schema = schema.update_column(column, dtype="float64")

        elif ptypes.is_datetime64_dtype(dtype):
            # assert not is_rev, (
            #   f"{column} type cannot be datetime when postprocessing."
            # )
            pass

        else:
            raise TypeError(
                f"{column} type cannot be {dtype},"
                " it should be either bool or numeric or categorical or datetime."
            )

    return df, schema


def _process(
    df: pd.DataFrame,
    params: dict[str, Any],
    categoricals: Sequence[str],
    schema: pa.DataFrameSchema,
    is_rev: bool = False,
) -> tuple[
    pd.DataFrame,
    dict[str, tuple[float, float]],
    Sequence[str],
    defaultdict,
    pa.DataFrameSchema,
]:
    """
    Process DataFrame according to a given sequence of transformations per column.
    The function doecuments its steps to allow applying the reverse transformations,
    if possible.

    Args:
        df (DataFrame): The DataFrame to be processed.
        params (dict): The transforamations per column.
        categoricals (Sequence): The names of the categorical columns.
        schema (DataFrameSchema): The schema of the input DataFrame.
        is_rev (bool): Whether this is a reverse transformation application.

    Returns:
        DataFrame: The processed DataFrame
        dict: Boundary values (min and max) of each column.
        Sequence: The names of the categorical columns.
        defaultdict: The reverse transformations per column.
        DataFrameSchema: The schema of the processed DataFrame.


    Raises:
        ValueError: If transformation parameters are not valid.
    """

    assert is_rev or (
        set(df.columns) == set(params.keys())
    ), "Generation config file should contain all columns in the dataset."

    processed_df = df.copy()
    schema = deepcopy(schema)
    categoricals = categoricals[:]
    boundaries = {}

    if not is_rev:
        processed_df, processed_schema = _prepare_dtypes(processed_df, schema, is_rev)

    rev_params: defaultdict = defaultdict(list)

    for column in params:
        was_bounded = False

        for args in params[column]:
            assert "name" in args, "params items should have the key `name`."

            rev_args = None

            if args["name"] == "binary":
                assert (
                    processed_df[column].nunique() == 2
                ), f"Binary {column} column must have exactly two values."
                was_bounded = True
                boundaries[column] = (0, 1)

            elif args["name"] == "clipping":
                processed_df[column] = processed_df[column].clip(
                    args["lower"], args["upper"]
                )
                rev_args = args.copy()
                boundaries[column] = (args["lower"], args["upper"])
                was_bounded = True

            elif args["name"] == "resolution":
                assert (
                    was_bounded
                ), f"Column {column} musb be bounded before changing resulotion."
                assert ptypes.is_numeric_dtype(processed_df[column])
                processed_df[column] = (
                    (processed_df[column] / args["scale"]).round().astype(int)
                )
                rev_args = {"name": "rev_resolution", "scale": args["scale"]}
                boundaries[column] = (
                    np.floor(boundaries[column][0] / args["scale"]).astype(int),
                    np.ceil(boundaries[column][1] / args["scale"]).astype(int),
                )

            elif args["name"] == "rev_resolution":
                assert ptypes.is_numeric_dtype(processed_df[column])
                processed_df[column] *= args["scale"]

            elif args["name"] == "binning":
                processed_df[column] = pd.cut(
                    processed_df[column], args["bins"], right=False  #
                )
                categories = list(processed_df[column].cat.categories)
                assert (-1 != processed_df[column].cat.codes).all(), (
                    f"The lower-most bin edge of column `{column}`"
                    " is higher than the minimal value in the data"
                    " OR the higher-most bin edge is lower than the maximal"
                    " valuein the data,"
                    " you should have clip this column or change the bin edges"
                    " independently of the real data."
                )
                processed_df[column] = processed_df[column].cat.codes
                rev_args = {"name": "rev_binning", "categories": categories}
                boundaries[column] = (0, len(categories) - 1)
                was_bounded = True

                # ugly hack
                rev_params[column] = [
                    ra for ra in rev_params[column] if ra["name"] != "clipping"
                ]

            elif args["name"] == "rev_binning":
                processed_df[column] = (
                    processed_df[column]
                    .clip(lower=0, upper=len(args["categories"]) - 1)
                    .round()
                    .astype(int)
                    .astype(pd.CategoricalDtype(ordered=True))
                )
                processed_df[column] = processed_df[column].cat.rename_categories(
                    dict(enumerate(args["categories"]))
                )

            elif args["name"] == "one_hot":
                assert ptypes.is_integer_dtype(processed_df[column])
                ohot = transformers.categorical.OneHotEncodingTransformer()
                hyper = HyperTransformer({column: ohot})
                processed_df = hyper.fit_transform(processed_df)

                categoricals.remove(column)
                for i in range(ohot._num_dummies):
                    dummy_column = f"{column}#{i}"
                    boundaries[dummy_column] = (0, 1)
                    categoricals.append(dummy_column)
                    was_bounded = True

                rev_args = {"name": "rev_one_hot", "transformer": hyper}

            elif args["name"] == "rev_one_hot":
                transformer = args["transformer"]
                processed_df = transformer.reverse_transform(processed_df)

            elif args["name"] == "datetime":
                processed_df[column] = pd.to_datetime(
                    processed_df[column], format=args["format"]
                )
                # TODO: tailored made for brith dataset
                assert processed_df[column].dt.year.nunique() == 1
                year = processed_df[column].dt.year.iloc[0]

                if args["resolution"] == "dayofyear":
                    format = "%j"
                    boundaries[column] = (0, 365)
                    dec_to_zero = True
                elif args["resolution"] == "week":
                    # Week number of the year (Sunday as the first day of the week)
                    format = "%U"
                    boundaries[column] = (0, 53)
                    dec_to_zero = False
                elif args["resolution"] == "month":
                    format = "%m"
                    boundaries[column] = (0, 11)
                    dec_to_zero = True

                else:
                    raise ValueError(
                        f'Resolution `{args["resolution"]}`` is not valid'
                        f" for column `{column}` with the `datetime` transformation."
                    )
                processed_df[column] = (
                    processed_df[column]
                    .dt.strftime(format)
                    .apply(int)
                    .astype(int)
                    .apply(lambda x: x - 1 if dec_to_zero else x)
                )
                rev_args = {
                    "name": "rev_datetime",
                    "resolution": args["resolution"],
                    "year": year,
                }
                was_bounded = True

            elif args["name"] == "rev_datetime":
                if args["resolution"] == "dayofyear":
                    n_days_in_year = (
                        datetime(day=31, month=12, year=args["year"])
                        - datetime(day=1, month=1, year=args["year"])
                    ).days + 1
                    day_of_year = (
                        (processed_df[column].round() + 1)
                        .clip(1, n_days_in_year)
                        .astype(int)
                    )
                    day_with_year = [f"{day:03d}/{args['year']}" for day in day_of_year]
                    processed_df[column] = pd.to_datetime(day_with_year, format="%j/%Y")

                elif args["resolution"] == "week":
                    weeks = processed_df[column].round().clip(0, 53).astype(int)
                    middle_week_with_year = [
                        f"Wednesday/{week:02d}/{args['year']}" for week in weeks
                    ]
                    processed_df[column] = pd.to_datetime(
                        middle_week_with_year, format="%A/%U/%Y"
                    )
                elif args["resolution"] == "month":
                    months = (processed_df[column].round() + 1).clip(1, 12).astype(int)
                    middle_month_with_year = [
                        f"15/{month:02d}/{args['year']}" for month in months
                    ]
                    processed_df[column] = pd.to_datetime(
                        middle_month_with_year, format="%d/%m/%Y"
                    )
                else:
                    raise ValueError(
                        f'Resolution `{args["resolution"]}`` is not valid'
                        f" for column `{column}`` with the"
                        " `rev_datetime` transformation."
                    )

            else:
                raise ValueError(f"{args['name']} is not a valid transformation.")

            if rev_args is not None:
                rev_params[column].insert(0, rev_args)

            assert (
                is_rev
                or args["name"] == "one_hot"
                or (
                    processed_df[column].between(
                        boundaries[column][0], boundaries[column][1]
                    )
                ).all()
            ), (column, processed_df[column].min(), processed_df[column].max())

            assert is_rev or was_bounded, (
                f"Column {column} must be bounded in preprocessing"
                " (use either clippin or binning)."
            )

    if is_rev:
        processed_df, processed_schema = _prepare_dtypes(processed_df, schema, is_rev)

    assert all(
        lower == 0
        for column, (lower, _) in boundaries.items()
        if column in categoricals
    )

    return processed_df, boundaries, categoricals, rev_params, processed_schema
