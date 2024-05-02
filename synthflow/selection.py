from __future__ import annotations

import hashlib
import itertools as it
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from synthflow.birth import (  # noqa: F401
    DATA_TRANSFORMATION_SPECS,
    DATE_TRANSFORMATION_ONLY_AS_MONTH,
    MOH_GEN_CONFIG,
)
from synthflow.generation import DATETIME_RESOLUTIONS

PROJECTION_UP_TO_MIN_COUNT_ORDER = 2

HPARAMS_BASE = {
    "QUERY_BASED": {
        "num_query": [128, 1024, 4096],  # 512,
        "num_iterations": [100, 1000],  # 500,
        "num_inner_updates": [25, 100],
    },
    "GAN": {
        "epochs": [300],
        "generator_lr": [2e-4, 2e-5],
        "discriminator_lr": [2e-4, 2e-5],
        "generator_decay": [1e-6],
        "discriminator_decay": [1e-6],
        # "discriminator_steps": [1, 5], # NOT IN USE IN smartnoise
        "batch_size": [500],
        "noise_multiplier": [1e-3, 0.1, 1, 5],
        "max_per_sample_grad_norm": [0.1, 1, 5],
        "loss": ["cross_entropy", "wasserstein"],
    },
    "PBN_BASED": {
        "EPSILON_SPLIT": [0.1, 0.25, 0.5, 0.75],
        "DEGREE": [1, 2, 3, 4],
        "THETA": [2, 4, 8, 16, 20, 25, 30, 35, 40, 50, 60, 100],
    },
}


MODEL_SPECS = {
    #     "GC": {
    #         "dp": "none",
    #         "categorical_mode": "indifferent",
    #         "universe_limited": False,
    #         "try_one_hot": True,
    #         "gpu": False,
    #         "hparams": {'field_distributions': [None]},
    #     },
    # Insperation for the hparams values:
    # https://github.com/terranceliu/iterative-dp
    # https://arxiv.org/pdf/2106.07153.pdf
    # "MWEM": {
    #     "dp": "pure",
    #     "categorical_mode": "only",
    #     "try_one_hot": False,
    #     "universe_limited": True,
    #     "gpu": False,
    #     "hparams": {
    #         "q_count": HPARAMS_BASE["QUERY_BASED"]["num_query"],
    #         "iterations": HPARAMS_BASE["QUERY_BASED"]["num_iterations"],
    #         "mult_weights_iterations":
    #             HPARAMS_BASE["QUERY_BASED"]["num_inner_updates"],
    #     },
    #     "filter_out": lambda hp: hp["q_count"] < hp["iterations"],
    # },
    # "PEP": {
    #     "dp": "approx",
    #     "categorical_mode": "only",
    #     "try_one_hot": False,
    #     "universe_limited": True,
    #     "gpu": False,
    #     "hparams": {
    #         "marginal": [2, 3, 4],
    #         "workload": HPARAMS_BASE["QUERY_BASED"]["num_query"],
    #         "T": HPARAMS_BASE["QUERY_BASED"]["num_iterations"],
    #         "iters": HPARAMS_BASE["QUERY_BASED"]["num_inner_updates"],
    #     },
    #     "filter_out": lambda hp: hp["workload"] < hp["T"],
    #
    # https://dl.acm.org/doi/pdf/10.1145/3134428
    # Original version
    "PBNTheta": {
        "dp": "pure",
        "categorical_mode": "only",
        "universe_limited": False,
        "try_one_hot": False,
        "gpu": False,
        "hparams": {
            "theta": HPARAMS_BASE["PBN_BASED"]["THETA"],
            "epsilon_split": HPARAMS_BASE["PBN_BASED"]["EPSILON_SPLIT"],
        },
    },
    # https://dl.acm.org/doi/pdf/10.1145/3134428
    # Modified version
    # "PBNDegree": {
    #     "dp": "pure",
    #     "categorical_mode": "only",
    #     "universe_limited": False,
    #     "try_one_hot": False,
    #     "gpu": False,
    #     "hparams": {
    #         "degree": HPARAMS_BASE["PBN_BASED"]["DEGREE"],
    #         "epsilon_split": HPARAMS_BASE["PBN_BASED"]["EPSILON_SPLIT"],
    #     },
    # },
    #     "CTGAN": {
    #         "dp": "none",
    #         "categorical_mode": "only",
    #         "universe_limited": False,
    #         "try_one_hot": False,
    #         "gpu": True,
    #         "hparams": {
    #             "epochs": [100, 200, 300],
    #             "generator_lr": [2e-4],
    #             "discriminator_lr": [2e-4],
    #             "generator_decay": [1e-6],
    #             "discriminator_decay": [1e-6],
    #             "discriminator_steps": [1, 5],
    #             "batch_size": [500],
    #         },
    #     },
    #    "DPGAN": {
    #        "dp": "approx",
    #        "categorical_mode": "only",
    #        "universe_limited": False,
    #        "try_one_hot": False,
    #        "gpu": True,
    #        "hparams": {
    #            "epochs": HPARAMS_BASE["GAN"]["epochs"],
    #            "generator_lr": HPARAMS_BASE["GAN"]["generator_lr"],
    #            "discriminator_lr": HPARAMS_BASE["GAN"]["discriminator_lr"],
    #            "generator_decay": HPARAMS_BASE["GAN"]["generator_decay"],
    #            "discriminator_decay": HPARAMS_BASE["GAN"]["discriminator_decay"],
    #            # "discriminator_steps": HPARAMS_BASE["GAN"]["discriminator_steps"],
    #            "batch_size": HPARAMS_BASE["GAN"]["batch_size"],
    #            "sigma": HPARAMS_BASE["GAN"]["noise_multiplier"],
    #            "max_per_sample_grad_norm":
    #               HPARAMS_BASE["GAN"]["max_per_sample_grad_norm"],
    #            },
    #    },
    #    "PATEGAN": {
    #        "dp": "approx",
    #        "categorical_mode": "only",
    #        "universe_limited": False,
    #        "try_one_hot": False,
    #        "gpu": True,
    #        "hparams": {
    #            "generator_lr": HPARAMS_BASE["GAN"]["generator_lr"],
    #            "discriminator_lr": HPARAMS_BASE["GAN"]["discriminator_lr"],
    #            "generator_decay": HPARAMS_BASE["GAN"]["generator_decay"],
    #            "discriminator_decay": HPARAMS_BASE["GAN"]["discriminator_decay"],
    #            "batch_size": HPARAMS_BASE["GAN"]["batch_size"],
    #            },
    #    },
    # "DPCTGAN": {
    #     "dp": "approx",
    #     "categorical_mode": "only",
    #     "universe_limited": False,
    #     "try_one_hot": False,
    #     "gpu": True,
    #     "hparams": {
    #         "epochs": HPARAMS_BASE["GAN"]["epochs"],
    #         "generator_lr": HPARAMS_BASE["GAN"]["generator_lr"],
    #         "discriminator_lr": HPARAMS_BASE["GAN"]["discriminator_lr"],
    #         "generator_decay": HPARAMS_BASE["GAN"]["generator_decay"],
    #         "discriminator_decay": HPARAMS_BASE["GAN"]["discriminator_decay"],
    #         # "discriminator_steps": HPARAMS_BASE["GAN"]["discriminator_steps"],
    #         "batch_size": HPARAMS_BASE["GAN"]["batch_size"],
    #         "sigma": HPARAMS_BASE["GAN"]["noise_multiplier"],
    #         "max_per_sample_grad_norm":
    #           HPARAMS_BASE["GAN"]["max_per_sample_grad_norm"],
    #         "loss": HPARAMS_BASE["GAN"]["loss"],
    #     },
    # },
    # "PATECTGAN": {
    #     "dp": "approx",
    #     "categorical_mode": "only",
    #     "universe_limited": False,
    #     "try_one_hot": False,
    #     "gpu": True,
    #     "hparams": {
    #         "epochs": HPARAMS_BASE["GAN"]["epochs"],
    #         "generator_lr": HPARAMS_BASE["GAN"]["generator_lr"],
    #         "discriminator_lr": HPARAMS_BASE["GAN"]["discriminator_lr"],
    #         "generator_decay": HPARAMS_BASE["GAN"]["generator_decay"],
    #         "discriminator_decay": HPARAMS_BASE["GAN"]["discriminator_decay"],
    #         # "discriminator_steps": HPARAMS_BASE["GAN"]["discriminator_steps"],
    #         "batch_size": HPARAMS_BASE["GAN"]["batch_size"],
    #         "noise_multiplier": HPARAMS_BASE["GAN"]["noise_multiplier"],
    #         "loss": ["cross_entropy"],
    #         "regularization": [None, "dragan"],
    #     },
    # },
}

# only:
#   - only categorical variables
#   - and try unit binning
# matters:
#   - categorical and continuous
#   - and try unit binning
# indifferent:
#   - categorical and continuous
#   - but do NOT try unit binning
CATEGORICAL_MODES = [
    "only",
    "matters",
    "indifferent",
]


@dataclass(eq=True, frozen=True)
class CategoricalSetting:
    as_category: bool
    as_one_hot: None | bool


@dataclass(eq=True, frozen=True)
class DateTimeSetting:
    resolution: str
    as_category: bool
    as_one_hot: None | bool


class SkipConfig(Exception):
    pass


def _dict_hash(d):
    msg = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha224(msg).hexdigest()[:4]


def _is_sorted(seq: list) -> bool:
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def build_continuous_column_transformations(
    lower: float, upper: float, scale: None | float = None
) -> Sequence[dict[str, Any]]:
    """
    Build a sequence of transformation to process a continuous column.

    Args:
        lower (float): Lowe value for clipping.
        upper (float): Upper value for clipping.
        scale (float, optional): Dcaling factor after clipping. Defaults to None.

    Returns:
        list[dict]: The squence of transformations.
    """

    transformations = []
    assert lower < upper
    transformations.append({"name": "clipping", "lower": lower, "upper": upper})
    if scale is not None:
        assert scale > 0
        transformations.append({"name": "resolution", "scale": scale})
    return transformations


def build_binary_column_transformations() -> Sequence[dict[str, str]]:
    """
    Build a sequence of transformation to process a binary column.

    Returns:
        list[dict]: A sequence of a single binary transformation.
    """

    return [{"name": "binary"}]


def build_categorical_column_transformations(
    lower: float,
    upper: float,
    bins: list,
    #     as_unit_binning: bool,
    as_one_hot: bool,
) -> Sequence[dict[str, Any]]:
    """
    Build a sequence of transformation to process a categorical column.

    Args:
        lower (float): Lowe value for clipping.
        upper (float): Upper value for clipping.
        bins (list): A sequence of bins for `pandas.cut` with `right=False`.
        as_one_hot (bool): Whether to apply one hot encoding
                           to the categories after binning.

    Returns:
        list[dict]: The squence of transformations.
    """

    assert _is_sorted(bins)

    transformations = [
        {"name": "clipping", "upper": upper, "lower": lower},
        {"name": "binning", "bins": bins},
    ]

    if as_one_hot:
        transformations.append({"name": "one_hot"})

    return transformations


def build_datatime_transforms(
    format: str, resolution: str, as_one_hot: None | bool
) -> Sequence[dict[str, str]]:
    """
    Build a sequence of transformation to process a datatime column.

    Args:
        format (str): Datetime format.
        resolution (str): Resolution for categories,
                          must be either `dayofyear`, `week` or `month`.
        bins (list): A sequence of bins for `pandas.cut` with `right=False`.
        as_one_hot (bool): Whether to apply one hot encoding
                           to the categories after binning.

    Returns:
        list[dict]: The squence of transformations.
    """

    assert resolution in DATETIME_RESOLUTIONS
    transformations = [{"name": "datetime", "format": format, "resolution": resolution}]

    if as_one_hot:
        transformations.append({"name": "one_hot"})
    return transformations


def iterate_data_generation_configs(
    data_transformation_spec: dict[str, dict[str, Any]],
    categorical_mode: str,
    universe_limited: bool,
    try_one_hot: bool,
) -> tuple[dict[str, Any], Sequence[str], int, int]:
    """
    Create data cofing iterator.

    Args:
        data_transformation_spec (dict): Data transformation specification format.
        categorical_mode (str): Mode for categories,
                                must be either `only`, `matters` or `indifferent`.
        universe_limited (bool): Whether to limit the number of total cells.
        try_one_hot (bool): Whether to apply one hot encoding
                            to the categories after binning.

    Returns:
        dict: Data config.
        list: Column names that are represented as categorical.
        int: Number of total cells.
        int: number of one hot encoded columns.
    """

    base_config = {}
    base_represented_as_categorical = []

    continuous_columns_by_specs = []
    datetime_columns_by_specs = []

    for column, spec in data_transformation_spec.items():
        if spec["type"] == "binary":
            base_config[column] = build_binary_column_transformations()
            base_represented_as_categorical.append(column)
        elif spec["type"] == "continuous":
            continuous_columns_by_specs.append(column)
        elif spec["type"] == "datetime":
            datetime_columns_by_specs.append(column)
        else:
            raise ValueError(
                "`type` in `data_transformation_spec`"
                "should be either `binary` or `continuos`"
            )

    assert categorical_mode in CATEGORICAL_MODES
    as_categorical_space = (
        [False, True] if categorical_mode in {"matters", "indifferent"} else [True]
    )
    as_one_hot_space = [False, True] if try_one_hot else [False]

    unit_binning_is_meaningful_for_model = categorical_mode in {
        "only",
        "matters",
    }

    datetime_resolution_as_categorical = (
        ["month", "week"]
        if not (universe_limited or DATE_TRANSFORMATION_ONLY_AS_MONTH)
        else ["month"]
    )

    categorical_setting_space = set()
    datetime_resulotion_space = set()

    for as_categorical in as_categorical_space:
        if as_categorical:
            for as_one_hot in as_one_hot_space:
                datetime_resulotion_space.update(
                    [
                        DateTimeSetting(resolution, as_categorical, as_one_hot)
                        for resolution in datetime_resolution_as_categorical
                    ]
                )

                categorical_setting_space.add(
                    CategoricalSetting(True, as_one_hot)  # , as_unit_binning)
                )

        else:
            categorical_setting_space.add(CategoricalSetting(False, None))  # , None))
            datetime_resulotion_space.update(
                [
                    DateTimeSetting(resolution, False, None)
                    for resolution in DATETIME_RESOLUTIONS
                ]
            )

    def product_with_bins(as_categorical_product):
        for as_categorical_columns in as_categorical_product:
            first_cat_cols = second_cat_cols = list(as_categorical_columns)
            all_colum_bins = []

            for column, setting in zip(continuous_columns_by_specs, first_cat_cols):
                spec = data_transformation_spec[column]
                if setting.as_category:
                    colum_bins = deepcopy(spec["bins"])
                    if (
                        unit_binning_is_meaningful_for_model
                        and spec["also_unit_binning"]
                    ):
                        unit_bins = list(range(spec["lower"], spec["upper"] + 2))
                        assert unit_bins not in colum_bins, column
                        colum_bins.append(unit_bins)

                else:
                    colum_bins = [None]

                all_colum_bins.append(colum_bins)

            all_column_bin_product = (
                dict(zip(continuous_columns_by_specs, column_bins))
                for column_bins in it.product(*all_colum_bins)
            )

            for column_bins in all_column_bin_product:
                spec_setting = {}

                for column, setting in zip(
                    continuous_columns_by_specs, second_cat_cols
                ):
                    spec = deepcopy(data_transformation_spec[column])
                    spec["bins"] = column_bins[column]
                    spec_setting[column] = {**asdict(setting), **spec}

                yield spec_setting

    as_categorical_product = it.product(
        categorical_setting_space, repeat=len(continuous_columns_by_specs)
    )

    as_categorical_bins_product = product_with_bins(as_categorical_product)

    which_datatime_resolutions_product = it.product(
        datetime_resulotion_space, repeat=len(datetime_columns_by_specs)
    )

    representations_products = it.product(
        as_categorical_bins_product, which_datatime_resolutions_product
    )

    for (
        as_categorical_bins_columns,
        which_datatime_resolutions_columns,
    ) in representations_products:
        try:
            config = deepcopy(base_config)
            represented_as_categorical = deepcopy(base_represented_as_categorical)

            num_cat_cells = 1
            num_one_hot = 0

            for column, spec_setting in as_categorical_bins_columns.items():
                if spec_setting["as_category"]:
                    num_one_hot += bool(spec_setting["as_one_hot"])

                    config[column] = build_categorical_column_transformations(
                        spec_setting["lower"],
                        spec_setting["upper"],
                        spec_setting["bins"],
                        spec_setting["as_one_hot"],
                    )
                    represented_as_categorical.append(column)
                    num_cat_cells *= len(spec_setting["bins"])
                else:
                    config[column] = build_continuous_column_transformations(
                        spec_setting["lower"],
                        spec_setting["upper"],
                        spec_setting.get("resolution"),
                    )

            for column, setting in zip(
                datetime_columns_by_specs, which_datatime_resolutions_columns
            ):
                num_one_hot += bool(setting.as_one_hot)

                config[column] = build_datatime_transforms(
                    data_transformation_spec[column]["format"],
                    setting.resolution,
                    setting.as_one_hot,
                )
                if setting.as_category:
                    represented_as_categorical.append(column)

            yield config, represented_as_categorical, num_cat_cells, num_one_hot

        except SkipConfig:
            continue


def iterate_generation_configs(
    model_specs: dict[str, dict[str, Any]],
    data_transformation_spec: dict[str, dict[str, Any]],
    epsilon: float,
    delta: float,
) -> dict[str, Any]:
    """
    Create model × data cofing iterator.

    Args:
        model_specs (dict): Model specification.
        data_transformation_spec (dict): Data transformation specification format.
        epsilon (float): Differential privacy epsilon parameter.
        delta (float): Differential privacy delta parameter.

    Returns:
        dict: Model × data config.
    """

    for model_name, spec in model_specs.items():
        base_config = {
            field: spec[field] for field in ("categorical_mode", "dp", "gpu")
        }
        base_config["model"] = model_name

        if spec["dp"] == "none":
            base_config["epsilon"], base_config["delta"] = float("inf"), 0
        elif spec["dp"] == "pure":
            base_config["epsilon"], base_config["delta"] = epsilon, 0
        elif spec["dp"] == "approx":
            base_config["epsilon"], base_config["delta"] = epsilon, delta
        else:
            raise ValueError("`dp` should be either `none` or `pure` or `apprix`.")

        data_transformation_iter = iterate_data_generation_configs(
            data_transformation_spec,
            spec["categorical_mode"],
            spec["universe_limited"],
            spec["try_one_hot"],
        )
        data_transformation_iter = list(data_transformation_iter)

        hparams_enumerated = map(enumerate, spec["hparams"].values())

        hparams_product = it.product(*hparams_enumerated)

        hparams_configurations = (zip(*prod) for prod in hparams_product)

        hparams_iter = (
            (sum(enums), dict(zip(spec["hparams"].keys(), prod)))
            for enums, prod in hparams_configurations
        )
        hparams_iter = list(hparams_iter)

        projections_iter = [
            {"name": "min-count", "order": order}
            for order in range(1, PROJECTION_UP_TO_MIN_COUNT_ORDER + 1)
        ]  # + [None]

        print(
            model_name,
            spec["categorical_mode"],
            spec["universe_limited"],
            spec["try_one_hot"],
            "#trans =",
            len(data_transformation_iter),
            "#hparams =",
            len(hparams_iter),
            "#projections =",
            len(projections_iter),
        )

        for (
            (tier, hparams),
            (
                transformations,
                represented_as_categorical,
                num_cat_cells,
                num_one_hot,
            ),
            (dataset_projection),
        ) in it.product(hparams_iter, data_transformation_iter, projections_iter):
            if "filter_out" in MODEL_SPECS[model_name]:
                if MODEL_SPECS[model_name]["filter_out"](hparams):
                    continue

            config = deepcopy(base_config)

            config["transformations"] = transformations
            config["categoricals"] = represented_as_categorical
            config["hparams"] = hparams

            config["tier"] = tier
            config["num_categorical"] = len(represented_as_categorical)
            config["num_cat_cells"] = num_cat_cells
            config["num_one_hot"] = num_one_hot

            config["dataset_projection"] = dataset_projection

            config["trans_id"] = _dict_hash(transformations)
            config["hparams_id"] = _dict_hash(hparams)
            config["projections_id"] = _dict_hash([dataset_projection])

            config["id"] = (
                config["trans_id"]
                + "-"
                + config["hparams_id"]
                + "-"
                + config["projections_id"]
            )

            yield config


def span_configs(dest_path: str, epsilon: float, delta: float) -> int:
    """
    Span model × data cofings and save them as JSON file.

    Args:
        dest_path (str): Path to the destination directory of gen config JSON files.
        epsilon (float): Differential privacy epsilon parameter.
        delta (float): Differential privacy delta parameter.

    Returns:
        int: Number of generate configs.
    """

    config_iter = iterate_generation_configs(
        MODEL_SPECS, DATA_TRANSFORMATION_SPECS, epsilon, delta
    )
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=False)

    for counter, config in enumerate(config_iter):
        filename = (
            "-".join(
                [
                    config["model"],
                    "gpu" if config["gpu"] else "cpu",
                    str(config["num_categorical"]),
                    str(config["tier"]),
                    str(config["num_cat_cells"]),
                    str(config["num_one_hot"]),
                    config["id"],
                ]
            )
            + ".json"
        )

        config_path = dest_path / filename

        assert not config_path.exists(), config_path
        with open(config_path, "w") as f:
            json.dump(config, f)

    return counter + 1
