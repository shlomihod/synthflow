import logging

import numpy as np
import pandas as pd
import pytest

from synthflow.birth import COLUMN_WEIGHTS, DATA_TRANSFORMATION_SPECS
from synthflow.evaluation.faithfulness.core import (
    ALGORITHMS,
    _build_graph_mat,
    _compute_beta,
    _compute_beta_by_algorithm,
    _compute_beta_push_relabel_igraph,
    _compute_radius_neighbors,
    evaluate_faithfulness,
    scale_columns_by_weights,
)
from synthflow.evaluation.faithfulness.weights import _adjust_categorical_column_weights

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def matching_params():
    inds = np.array(
        [np.array([0, 1]), np.array([1]), np.array([], dtype=np.int64), np.array([2])],
        dtype=object,
    )

    column_weights = {"a": 1 / 10, "b": 10, "c": [1, 2, 3, 4, 10]}

    x = pd.DataFrame({"a": [1, 3, 1, 20], "b": [1, 1, 0, 1], "c": [2, 2, 1, 6]})
    y = pd.DataFrame({"a": [1, 3, 4, 1], "b": [1, 1, 1, 0], "c": [2, 3, 6, 1]})

    density = 2 / 3

    return inds, x, y, column_weights, density


def test_scale_columns_by_weights():
    """Test scaling column values according to weights."""

    x = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [0, 1, 1, 1], "c": [1, 2, 3, 6], "d": [0, 1.5, 3.5, 0]}
    )
    column_weights = {"a": 1 / 10, "b": 10, "c": [1, 2, 3, 4, 10]}

    actual = scale_columns_by_weights(x, column_weights)
    desired = pd.DataFrame(
        {
            "a": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
            "b": {0: 0, 1: 10, 2: 10, 3: 10},
            "c": {0: 1, 1: 2, 2: 3, 3: 4},
            "d": {0: 0.0, 1: 1.5, 2: 3.5, 3: 0.0},
        }
    )

    pd.testing.assert_frame_equal(actual, desired)


def test_compute_radius_neighbors(matching_params):
    """Test finding nearest neighbors."""

    column_weights = {"a": 1 / 10, "b": 10, "c": [1, 2, 3, 4, 10]}

    # exact match
    x = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 1, 1], "c": [1, 2, 3, 6]})
    y = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 1, 1], "c": [1, 2, 3, 6]})

    inds, density = _compute_radius_neighbors(x, y, 1, 1, column_weights)

    np.testing.assert_equal(
        inds, np.array([np.array(0), np.array(1), np.array(2), np.array(3)])
    )
    np.testing.assert_allclose(density, 2 / 3)

    # approx match
    inds, density = _compute_radius_neighbors(
        matching_params[1], matching_params[2], 1, 1, matching_params[3]
    )

    assert all(
        np.array_equal(arr1, arr2) for arr1, arr2 in zip(inds, matching_params[0])
    )
    np.testing.assert_allclose(density, matching_params[4])


def test_build_graph_mat():
    """Test building adjency matrix."""

    np.testing.assert_equal(
        _build_graph_mat([[1, 2], [0], [3], []]).toarray(),
        np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_compute_beta_push_relabel_igraph():
    """Test calculation of beta using igrpah's push-relabel algorithm."""

    # C  R
    # 0--0
    #   /
    #  /
    # 1--1

    # 2  2
    #  \
    #   \
    # 3  3
    beta, matching, unmatched_indices = _compute_beta_push_relabel_igraph(
        [[0, 1], [1], [], [2]]
    )  # R[C]

    assert beta == 1 / 4

    np.testing.assert_array_equal(matching[0], np.array([0, 1, 2]))  # C
    np.testing.assert_array_equal(matching[1], np.array([0, 1, 3]))  # R

    #                             C    R
    assert unmatched_indices == ({3}, {2})


def test_compute_beta_by_algorithm(matching_params):
    """
    Test that calculation of beta using different algorithm
    is in alignment with igrpah's push-relabel algorithm.
    """

    # assume that this is correct:
    beta, matching, unmatched_indices = _compute_beta_push_relabel_igraph(
        matching_params[0]
    )

    for algorithm in ALGORITHMS:
        try:
            beta_, matching_, unmatched_indices_ = _compute_beta_by_algorithm(
                matching_params[0], algorithm
            )
        except AssertionError:
            LOGGER.warn(f"Skipping testing algorithm `{algorithm}`")
        assert beta == beta_
        np.testing.assert_array_equal(matching[0], matching_[0])
        np.testing.assert_array_equal(matching[1], matching_[1])
        assert unmatched_indices == unmatched_indices_


def test_compute_beta(matching_params):
    """
    Test that calculation of beta
    is in alignment with igrpah's push-relabel algorithm.
    """

    # assume that this is correct:
    beta, matching, unmatched_indices = _compute_beta_push_relabel_igraph(
        matching_params[0]
    )

    beta_, density_, candidate_, _, matching_, unmatched_indices_ = _compute_beta(
        matching_params[1], matching_params[2], 1, "auto", matching_params[3]
    )

    assert beta == beta_
    np.testing.assert_allclose(density_, 2 / 3)
    assert candidate_ == "push_relabel_igraph"
    assert (tuple(matching[0]), tuple(matching[1])) == matching_
    assert (
        tuple(unmatched_indices[0]),
        tuple(unmatched_indices[1]),
    ) == unmatched_indices_


def test_evaluate_faithfulness(matching_params):
    """
    Test that calculation of faithfulness results
    is in alignment with igrpah's push-relabel algorithm.
    """

    # assume that this is correct:
    beta, matching, unmatched_indices = _compute_beta_push_relabel_igraph(
        matching_params[0]
    )

    alpha = 1

    faithfulness_df = evaluate_faithfulness(
        matching_params[1], matching_params[2], matching_params[3], [alpha]
    )

    assert len(faithfulness_df) == 1
    faithfulness_result = faithfulness_df.loc[0]

    alpha_ = faithfulness_result["ɑ"].item()
    beta_ = faithfulness_result["β"].item()
    candidate_ = faithfulness_result["algorithm"]
    matching_ = faithfulness_result["matching"]
    unmatched_indices_ = faithfulness_result["unmatched_indices"]

    assert alpha == alpha_
    assert beta == beta_
    assert candidate_ == "push_relabel_igraph"
    assert (tuple(matching[0]), tuple(matching[1])) == matching_
    assert (
        tuple(unmatched_indices[0]),
        tuple(unmatched_indices[1]),
    ) == unmatched_indices_


def test_adjust_categorical_column_weights():
    """Test adjusting categorical column weights."""

    # no gap
    assert _adjust_categorical_column_weights([1, 2, 3, 4], [1, 2, 3, 4]) == [
        -np.inf,
        2,
        3,
        np.inf,
    ]

    # single gap
    assert _adjust_categorical_column_weights([1, 2, 3, 4, 5, 6], [1, 2, 4, 5, 6]) == [
        -np.inf,
        2,
        np.nextafter(2, np.inf),
        np.nextafter(4, -np.inf),
        4,
        5,
        np.inf,
    ]

    # double gap
    assert _adjust_categorical_column_weights([1, 2, 3, 4, 5, 6, 7], [1, 2, 6, 7]) == [
        -np.inf,
        2,
        np.nextafter(2, np.inf),
        np.nextafter(6, -np.inf),
        6,
        np.inf,
    ]

    # right edge is looser for weights
    assert _adjust_categorical_column_weights([1, 3, 5, 9], [1, 3, 5, 7, 9]) == [
        -np.inf,
        3,
        5,
        np.inf,
    ]

    # too short
    with pytest.raises(Exception) as e:
        _adjust_categorical_column_weights([1, 2, 3], [1, 2, 3])
    assert e.type == AssertionError

    assert len(DATA_TRANSFORMATION_SPECS["mother_age"]["bins"]) == 2
    assert _adjust_categorical_column_weights(
        COLUMN_WEIGHTS["mother_age"], DATA_TRANSFORMATION_SPECS["mother_age"]["bins"][0]
    ) == [
        -np.inf,
        18,
        20,
        25,
        30,
        35,
        np.nextafter(35, np.inf),
        np.nextafter(40, -np.inf),
        40,
        43,
        np.inf,
    ]
    assert _adjust_categorical_column_weights(
        COLUMN_WEIGHTS["mother_age"], DATA_TRANSFORMATION_SPECS["mother_age"]["bins"][1]
    ) == [-np.inf, 18, 20, 25, 30, 35, 37, 40, 43, np.inf]
    assert (
        _adjust_categorical_column_weights(
            COLUMN_WEIGHTS["mother_age"],
            list(
                range(
                    DATA_TRANSFORMATION_SPECS["mother_age"]["lower"],
                    DATA_TRANSFORMATION_SPECS["mother_age"]["upper"] + 2,
                )
            ),
        )
        == [-np.inf] + COLUMN_WEIGHTS["mother_age"][1:]
    )

    assert len(DATA_TRANSFORMATION_SPECS["gestation_week"]["bins"]) == 1
    assert _adjust_categorical_column_weights(
        COLUMN_WEIGHTS["gestation_week"], [28, 29, 32, 34, 37, 42, 43]
    ) == [-np.inf, 29, 32, 34, 37, 42, np.inf]
    assert (
        _adjust_categorical_column_weights(
            COLUMN_WEIGHTS["gestation_week"],
            list(
                range(
                    DATA_TRANSFORMATION_SPECS["gestation_week"]["lower"],
                    DATA_TRANSFORMATION_SPECS["gestation_week"]["upper"] + 2,
                )
            ),
        )
        == [-np.inf] + COLUMN_WEIGHTS["gestation_week"][1:]
    )
