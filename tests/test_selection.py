from functools import reduce
from operator import mul
from pathlib import Path

import numpy as np
import pytest

from synthflow.birth import DATA_TRANSFORMATION_SPECS
from synthflow.generation import DATETIME_RESOLUTIONS
from synthflow.selection import (
    MODEL_SPECS,
    PROJECTION_UP_TO_MIN_COUNT_ORDER,
    build_binary_column_transformations,
    build_categorical_column_transformations,
    build_continuous_column_transformations,
    build_datatime_transforms,
    iterate_data_generation_configs,
    iterate_generation_configs,
    span_configs,
)


@pytest.fixture
def basic_data_transformation_spec():
    return {
        "binary": {"type": "binary"},
        "continuous_1": {
            "type": "continuous",
            "lower": 1,
            "upper": 10,
            "bins": [[1, 2, 11], [1, 2, 3, 11]],
            "also_unit_binning": False,
        },
        "continuous_2": {
            "type": "continuous",
            "lower": 1,
            "upper": 10,
            "bins": [[1, 2, 11], [1, 2, 3, 11]],
            "also_unit_binning": True,
        },
        "date": {"type": "datetime", "format": "%d/%m/%Y"},
    }


@pytest.fixture
def basic_data_gen_configs(basic_data_transformation_spec):
    data_gen_configs = list(
        iterate_data_generation_configs(
            basic_data_transformation_spec, "only", False, False
        )
    )
    data_gen_configs_size = (
        len(basic_data_transformation_spec["continuous_1"]["bins"])
        + basic_data_transformation_spec["continuous_1"]["also_unit_binning"]
    ) * (
        len(basic_data_transformation_spec["continuous_2"]["bins"])
        + basic_data_transformation_spec["continuous_2"]["also_unit_binning"]
    )
    all_data_configs, *_ = zip(*data_gen_configs)

    return data_gen_configs, all_data_configs, data_gen_configs_size


def test_build_continuous_column_transformations():
    """Test building transformations for a continuous column."""

    assert build_continuous_column_transformations(0, 1) == [
        {"name": "clipping", "lower": 0, "upper": 1}
    ]
    assert build_continuous_column_transformations(0, 1, 0.5) == [
        {"name": "clipping", "lower": 0, "upper": 1},
        {"name": "resolution", "scale": 0.5},
    ]

    with pytest.raises(Exception) as e:
        build_continuous_column_transformations(0, 0)
    assert e.type == AssertionError

    with pytest.raises(Exception) as e:
        build_continuous_column_transformations(1, 0)
    assert e.type == AssertionError

    with pytest.raises(Exception) as e:
        build_continuous_column_transformations(0, 1, 0)
    assert e.type == AssertionError

    with pytest.raises(Exception) as e:
        build_continuous_column_transformations(0, 1, -0.5)
    assert e.type == AssertionError


def test_build_binary_column_transformations():
    """Test building transformations for a binary column."""

    assert build_binary_column_transformations() == [{"name": "binary"}]


def test_build_categorical_column_transformations():
    """Test building transformations for a categorical column."""

    assert build_categorical_column_transformations(0, 1, [0, 1], False) == [
        {"name": "clipping", "upper": 1, "lower": 0},
        {"name": "binning", "bins": [0, 1]},
    ]

    assert build_categorical_column_transformations(0, 1, [0, 1], True) == [
        {"name": "clipping", "upper": 1, "lower": 0},
        {"name": "binning", "bins": [0, 1]},
        {"name": "one_hot"},
    ]

    with pytest.raises(Exception) as e:
        build_categorical_column_transformations(0, 1, [1, 0], False)
    assert e.type == AssertionError


def test_build_datetime_column_transformations():
    """Test building transformations for a datetime column."""

    for resolution in DATETIME_RESOLUTIONS:
        assert build_datatime_transforms("abc", resolution, False) == [
            {"name": "datetime", "format": "abc", "resolution": resolution}
        ]

    assert build_datatime_transforms("abc", resolution, True) == [
        {"name": "datetime", "format": "abc", "resolution": resolution},
        {"name": "one_hot"},
    ]

    with pytest.raises(Exception) as e:
        build_datatime_transforms("abc", "hour", False)
    assert e.type == AssertionError


def test_model_spec_testing_assumptions():
    # we test only part of the arguemnts of the function
    # so let's make sure that the model used are in this region
    for name, spec in MODEL_SPECS.items():
        assert spec["dp"] == "pure"
        assert spec["categorical_mode"] == "only"
        assert not spec["universe_limited"]
        assert not spec["try_one_hot"]


def test_iterate_data_generation_configs(
    basic_data_transformation_spec, basic_data_gen_configs
):
    """Test building data generation configs."""

    data_gen_configs, all_data_configs, data_gen_configs_size = basic_data_gen_configs

    assert len(data_gen_configs) == data_gen_configs_size
    all_data_configs, *_ = zip(*data_gen_configs)
    assert len(data_gen_configs) == len(np.unique([repr(x) for x in all_data_configs]))

    for (
        config,
        represented_as_categorical,
        num_cat_cells,
        num_one_hot,
    ) in data_gen_configs:
        assert config["binary"] == [{"name": "binary"}]
        assert len(config["date"]) == 1

        assert config["date"][0]["name"] == "datetime"
        assert config["date"][0]["format"] == "%d/%m/%Y"
        assert config["date"][0]["resolution"] in DATETIME_RESOLUTIONS

        assert len(config["continuous_1"]) == 2
        assert len(config["continuous_2"]) == 2

        assert config["continuous_1"][0] == {
            "name": "clipping",
            "upper": 10,
            "lower": 1,
        }
        assert config["continuous_2"][0] == {
            "name": "clipping",
            "upper": 10,
            "lower": 1,
        }

        assert (
            config["continuous_1"][1]["bins"]
            in basic_data_transformation_spec["continuous_1"]["bins"]
        )
        assert config["continuous_2"][1]["bins"] in basic_data_transformation_spec[
            "continuous_2"
        ]["bins"] + [list(range(1, 12))]

        assert num_one_hot == 0


def test_iterate_generation_configs(
    basic_data_transformation_spec, basic_data_gen_configs
):
    """Test building data + model generation configs."""

    data_gen_configs, all_data_configs, data_gen_configs_size = basic_data_gen_configs

    gen_cofings = list(
        iterate_generation_configs(MODEL_SPECS, basic_data_transformation_spec, 4, 0)
    )

    hparam_size = reduce(mul, map(len, MODEL_SPECS["PBNTheta"]["hparams"].values()), 1)
    projection_size = (
        PROJECTION_UP_TO_MIN_COUNT_ORDER  # = | 1, PROJECTION_UP_TO_MIN_COUNT_ORDER] |
    )

    assert len(gen_cofings) == data_gen_configs_size * hparam_size * projection_size
    assert len(gen_cofings) == len(np.unique([repr(x) for x in gen_cofings]))
    assert len(gen_cofings) == len({config["id"] for config in gen_cofings})

    for config in gen_cofings:
        assert config["model"] == "PBNTheta"
        assert config["epsilon"] == 4
        assert config["delta"] == 0
        assert not config["gpu"]
        assert config["transformations"] in all_data_configs
        assert set(config["categoricals"]) == set(basic_data_transformation_spec.keys())
        for hparam_name, hparam_value in config["hparams"].items():
            assert hparam_value in MODEL_SPECS[config["model"]]["hparams"][hparam_name]


def test_span_configs(tmpdir):
    """Test saved gen config json files against generated one."""

    gen_config_dir = tmpdir.join("gen-config")

    counter = span_configs(gen_config_dir, 4, 0)
    paths = list(Path(gen_config_dir).glob("*.json"))

    gen_cofings = list(
        iterate_generation_configs(MODEL_SPECS, DATA_TRANSFORMATION_SPECS, 4, 0)
    )

    assert len(gen_cofings) == counter
    assert len(paths) == counter

    path_ids = {Path(p).stem.split("-", 6)[-1] for p in paths}
    config_ids = {config["id"] for config in gen_cofings}
    assert path_ids == config_ids
