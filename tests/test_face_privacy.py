import numpy as np
import pandas as pd  # type: ignore
import pytest

from synthflow.evaluation.privacy.face import (
    _calc_k_anonymity,
    _calc_l_diversity,
    _calc_uniques_and_more,
    _count_occurrences,
    evaluate_face_privacy,
)


@pytest.fixture
def face_privacy_data_df():
    return pd.DataFrame(
        {
            "a": [False, True] * 5,
            "b": list("ijk") * 3 + ["n"],
            "c": [1, 2, 1, 2, 3, 1, 1, 3, 4, 5],
        }
    )


def test_count_occurrences():
    assert _count_occurrences(
        pd.Series([0] + [1] * 3 + [2] * 3 + [6] * 10).value_counts(), 3
    ).to_dict() == {1: 1, 2: 0, 3: 2}


def test_calc_uniques_and_more(face_privacy_data_df):
    """Test unique count values."""
    _calc_uniques_and_more(face_privacy_data_df, 3).to_dict() == {1: 8, 2: 1, 3: 0}


def test_calc_k_anonymity(face_privacy_data_df):
    """Test k-anonymity values."""
    assert _calc_k_anonymity(face_privacy_data_df, ["a", "b"], 3).to_dict() == {
        1: 4,
        2: 3,
        3: 0,
    }
    assert _calc_k_anonymity(face_privacy_data_df, ["a", "c"], 3).to_dict() == {
        1: 5,
        2: 1,
        3: 1,
    }


def test_calc_l_diversity(face_privacy_data_df):
    """Test l-diversity values."""
    _calc_l_diversity(face_privacy_data_df, ["a", "b"], "c", 3).to_dict() == {
        1: 5,
        2: 2,
        3: 0,
    }
    _calc_l_diversity(face_privacy_data_df, ["a", "c"], "b", 3).to_dict() == {
        1: 5,
        2: 2,
        3: 0,
    }


def test_evaluate_face_privacy(face_privacy_data_df):
    """Test face privacy evaluation."""
    face_privacy_evaluation_df = evaluate_face_privacy(
        face_privacy_data_df, 3, [(("a", "b"), "c"), (("a", "c"), "b")]
    )
    (
        face_privacy_evaluation_df.sort_values(
            by=list(face_privacy_evaluation_df.columns)
        )
        .reset_index(drop=True)
        .to_dict()
    ) == {
        "param": {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 2,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 3,
            17: 3,
            18: 3,
            19: 3,
            20: 3,
            21: 3,
            22: 3,
            23: 3,
        },
        "value": {
            0: 4,
            1: 5,
            2: 5,
            3: 5,
            4: 5,
            5: 5,
            6: 8,
            7: 8,
            8: 1,
            9: 1,
            10: 1,
            11: 2,
            12: 2,
            13: 2,
            14: 3,
            15: 3,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 1,
            23: 1,
        },
        "metric": {
            0: "k-anonymity",
            1: "k-anonymity",
            2: "k-anonymity",
            3: "l-diversity",
            4: "l-diversity",
            5: "l-diversity",
            6: "unique",
            7: "unique",
            8: "k-anonymity",
            9: "unique",
            10: "unique",
            11: "l-diversity",
            12: "l-diversity",
            13: "l-diversity",
            14: "k-anonymity",
            15: "k-anonymity",
            16: "k-anonymity",
            17: "l-diversity",
            18: "l-diversity",
            19: "l-diversity",
            20: "unique",
            21: "unique",
            22: "k-anonymity",
            23: "k-anonymity",
        },
        "quasi": {
            0: "('a', 'b')",
            1: "('a', 'c')",
            2: np.nan,
            3: "('a', 'b')",
            4: "('a', 'c')",
            5: np.nan,
            6: np.nan,
            7: np.nan,
            8: "('a', 'c')",
            9: np.nan,
            10: np.nan,
            11: "('a', 'b')",
            12: "('a', 'c')",
            13: np.nan,
            14: "('a', 'b')",
            15: np.nan,
            16: "('a', 'b')",
            17: "('a', 'b')",
            18: "('a', 'c')",
            19: np.nan,
            20: np.nan,
            21: np.nan,
            22: "('a', 'c')",
            23: np.nan,
        },
        "sensitive": {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: "c",
            4: "b",
            5: np.nan,
            6: np.nan,
            7: np.nan,
            8: np.nan,
            9: np.nan,
            10: np.nan,
            11: "c",
            12: "b",
            13: np.nan,
            14: np.nan,
            15: np.nan,
            16: np.nan,
            17: "c",
            18: "b",
            19: np.nan,
            20: np.nan,
            21: np.nan,
            22: np.nan,
            23: np.nan,
        },
        "name": {
            0: "k-anonymity/1",
            1: "k-anonymity/1",
            2: "k-anonymity/1/max",
            3: "l-diversity/1",
            4: "l-diversity/1",
            5: "l-diversity/1/max",
            6: "unique/1",
            7: "unique/1/max",
            8: "k-anonymity/2",
            9: "unique/2",
            10: "unique/2/max",
            11: "l-diversity/2",
            12: "l-diversity/2",
            13: "l-diversity/2/max",
            14: "k-anonymity/2",
            15: "k-anonymity/2/max",
            16: "k-anonymity/3",
            17: "l-diversity/3",
            18: "l-diversity/3",
            19: "l-diversity/3/max",
            20: "unique/3",
            21: "unique/3/max",
            22: "k-anonymity/3",
            23: "k-anonymity/3/max",
        },
        "type": {
            0: "face",
            1: "face",
            2: "face",
            3: "face",
            4: "face",
            5: "face",
            6: "face",
            7: "face",
            8: "face",
            9: "face",
            10: "face",
            11: "face",
            12: "face",
            13: "face",
            14: "face",
            15: "face",
            16: "face",
            17: "face",
            18: "face",
            19: "face",
            20: "face",
            21: "face",
            22: "face",
            23: "face",
        },
    }
