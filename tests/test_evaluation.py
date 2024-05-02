import pandas as pd

from synthflow.evaluation import numerify
from tests.utils import basic_all_categorical_real_df  # noqa: F401


def test_numerify(basic_all_categorical_real_df):  # noqa: F811
    """Test the convertion a DataFrame to floats for evaluation."""
    df, _, _, _, _, _, rev_processed_df = basic_all_categorical_real_df

    # simulate the type of category codes
    df["nominal"] = df["nominal"].astype("int8")

    assert numerify(df).equals(
        pd.DataFrame(
            {
                "binary": {
                    0: 1.0,
                    1: 0.0,
                    2: 1.0,
                    3: 0.0,
                    4: 1.0,
                    5: 0.0,
                    6: 1.0,
                    7: 0.0,
                    8: 1.0,
                    9: 0.0,
                },
                "nominal": {
                    0: 1.0,
                    1: 2.0,
                    2: 3.0,
                    3: 4.0,
                    4: 5.0,
                    5: 6.0,
                    6: 7.0,
                    7: 8.0,
                    8: 9.0,
                    9: 10.0,
                },
                "date": {
                    0: 85.0,
                    1: 116.0,
                    2: 146.0,
                    3: 177.0,
                    4: 207.0,
                    5: 238.0,
                    6: 269.0,
                    7: 299.0,
                    8: 330.0,
                    9: 360.0,
                },
            }
        )
    )

    assert numerify(rev_processed_df).equals(
        pd.DataFrame(
            {
                "binary": {
                    0: 1.0,
                    1: 0.0,
                    2: 1.0,
                    3: 0.0,
                    4: 1.0,
                    5: 0.0,
                    6: 1.0,
                    7: 0.0,
                    8: 1.0,
                    9: 0.0,
                },
                "nominal": {
                    0: 1.0,
                    1: 3.0,
                    2: 3.0,
                    3: 5.5,
                    4: 5.5,
                    5: 5.5,
                    6: 9.0,
                    7: 9.0,
                    8: 9.0,
                    9: 9.0,
                },
                "date": {
                    0: 75.0,
                    1: 106.0,
                    2: 136.0,
                    3: 167.0,
                    4: 197.0,
                    5: 228.0,
                    6: 259.0,
                    7: 289.0,
                    8: 320.0,
                    9: 350.0,
                },
            }
        )
    )
