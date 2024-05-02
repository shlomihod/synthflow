"""Synthflow.
Usage:
    synthflow execute <real-data-path> <gen-config-path> [--synth-data-name-path=PATH] [--processed-data-name-path=PATH] [--minimal] [--subsampling=FRAC] [--random-seed=SEED] [--verbose] [--wandb] [--wandb-api-key=KEY] [--wandb-group=NAME] [--non-private-evaluation]
    synthflow evaluate <real-data-path> <synth-data-path> <processed-data-path> [--minimal] [--subsampling=FRAC] [--random-seed=SEED] [--verbose]
    synthflow span <gen-config-directory-path> <epsilon> <delta>
    synthflow parallel <real-data-path> <gen-config-directory-path> (--grid | --random) --wandb-api-key=KEY [--wandb-group=NAME] [--num-workers=NUM] [--ray-connect-auto] [--dry-run] [--verbose]
    synthflow report <output-notebook-path> <run_path> [--wandb-mode=MODE] [--minimal]

Options:
    --minimal                         Conduct only minimal evaluation
    --subsampling=FRAC                Subsamplig data fraction for evaluation [default: .4]
    --random-seed=SEED                Random seed for subsampled dataset
    --synth-data-name-path=PATH       Path for saving the synthethic data
    --processed-data-name-path=PATH   Path for savint the processed real data
    --grid                            Conduct (sorted) grid search
    --random                          Conduct random search
    --wandb                           Use Weight and Biases
    --wandb-mode=MODE                 Set mode for Weight and Biases [default: online]
    --wandb-api-key=KEY               Get Weight and Biases API key here: https://wandb.ai/authorize
    --wandb-group=NAME                Set a group name for the run at Weight and Biases
    --num-workers=NUM                 Number of Ray workers, set to number of CPUs as default
    --ray-connect-auto                Automatic connect to existing Ray cluster
    --dry-run                         Do not run, print config path only
    --verbose                         Print more text
    --non-private-evaluation          Show non-private evaluation results
"""  # noqa: E501

from __future__ import annotations

import itertools as it
import json
import logging
import os
import pickle
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
import pandas.api.types as ptypes  # type: ignore
import wandb
from docopt import docopt  # type: ignore
from pandera import DataFrameSchema
from pandera.errors import SchemaError
from rich.console import Console
from schema import And, Or, Schema, Use  # type: ignore

from synthflow import create_report, evaluate, generate, span_configs
from synthflow.birth import (
    ACCEPTANCE_CRITERIA,
    ALPHAS,
    COLUMN_WEIGHTS,
    COMPLEX_QUERY_EVAL_EPSILON,
    COMPLEX_QUERY_FIT_EPSILON,
    CONDITIONAL_MEAN_EPSILONS,
    CONSTRAINTS,
    CONSTRAINTS_UBERSAMPLING,
    EVALUATION_COLUMN_BINS,
    FACE_PRIVACY_UP_TO,
    HIGH_SENSITIVITY_EPSILON,
    LOW_SENSITIVITY_EPSILON,
    QUASI_SENSITIVE_PAIRS,
    RATIO_MAX_CLIPPING_FACTOR,
    REAL_DATASET_SCHEMA,
    USER_ANALYSIS,
)
from synthflow.evaluation.faithfulness import prepare_column_weights

LOGGER = logging.getLogger(__name__)


WANDB_MODES = ["online", "offline"]


pd.set_option(
    "display.max_columns",
    None,
    "display.expand_frame_repr",
    False,
    "display.float_format",
    lambda x: f"{x:.5f}",
    "display.max_colwidth",
    None,
)


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def _cat2str(df):
    df = df.copy()
    for column in df.columns:
        if ptypes.is_categorical_dtype(df[column]):
            df[column] = df[column].astype(str)
    return df


def _wanb_log_artifact(obj: Any, name: str, type_: str, as_file: bool = False):
    artifcat = wandb.Artifact(name, type_)
    if as_file:
        artifcat.add_file(obj, name)
    else:
        artifcat.add(obj, name)
    wandb.log_artifact(artifcat)


def _print_validate_coerce(
    df: pd.DataFrame, name: str, schema: DataFrameSchema, console: Any | None = None
) -> pd.DataFrame:
    # Validation for catergorical columns does not work
    # We have to make sure that there is no None or NaN
    assert not df.isna().any().any()

    # Order df according to the schema
    # Required for the Logistic Regression utility evaluation
    # Because we assume that we try to predict the last column from the rest
    schema_columns = list(REAL_DATASET_SCHEMA.columns.keys())
    assert sorted(df.columns) == sorted(schema_columns)
    df = df[schema_columns]

    try:
        validated_df = schema.validate(df)
    except SchemaError as err:
        for idx in err.failure_cases["index"]:
            if console is not None:
                console.print(df.loc[idx])
        raise err
    finally:
        if console is not None:
            console.print(f"{name} Dataset", style="red")
            df.info(null_counts=True)
            console.print()

    return validated_df


def report_main(arguments: dict[str, str]):
    console = Console()

    wandb_mode = Schema(lambda x: x in WANDB_MODES).validate(arguments["--wandb-mode"])

    notebook_dir_path = Path(arguments["<output-notebook-path>"])
    notebook_dir_path.mkdir(parents=True, exist_ok=True)

    console.print(f"Create report at `{notebook_dir_path}`")
    console.print(f"W&B run path: {arguments['<run_path>']}")
    console.print(f"W&B mode: {wandb_mode}")

    create_report(
        str(notebook_dir_path),
        arguments["<run_path>"],
        wandb_mode,
        arguments["--minimal"],
    )


def span_main(arguments: dict[str, str]):
    console = Console()

    epsilon = Schema(And(Use(float), lambda x: float(x) > 0)).validate(
        arguments["<epsilon>"]
    )
    delta = Schema(And(Use(float), lambda x: float(x) >= 0)).validate(
        arguments["<delta>"]
    )
    console.print(f"Spanning configs at with ε={epsilon} and δ={delta}...")
    cofnig_count = span_configs(
        arguments["<gen-config-directory-path>"], epsilon, delta
    )
    console.print(f"{cofnig_count} configs were spanned!")


def parallel_main(arguments: dict[str, str]):
    import ray

    console = Console()

    Schema(Use(open)).validate(arguments["<real-data-path>"])

    num_workes = Schema(Or(And(Use(int), lambda x: x >= 1), None)).validate(
        arguments["--num-workers"]
    )

    console.rule("Init Ray")

    address = "auto" if arguments["--ray-connect-auto"] else None

    import torch

    num_gpus = torch.cuda.device_count()
    has_gpus = num_gpus > 0
    console.print(f"#gpu = {num_gpus}")
    if address is not None:
        console.print("BUT IT WAS SET IN THE `ray start` COMMAND")

    ray.init(
        address=address,
        num_cpus=num_workes,
        # num_gpus=num_gpus,  # detect automatically
        log_to_driver=bool(arguments["--verbose"]),
        runtime_env={"env_vars": {"WANDB_API_KEY": arguments["--wandb-api-key"]}},
    )

    @ray.remote(num_gpus=0, num_cpus=1, max_calls=1)
    def execute_main_with_cpu(arguments):
        execute_main(arguments)
        return arguments["<gen-config-path>"]

    @ray.remote(num_gpus=has_gpus / 2, num_cpus=1, max_calls=1)
    def execute_main_with_gpu(arguments):
        execute_main(arguments)
        return arguments["<gen-config-path>"]

    console.rule("Processing generation config files...")

    config_dir_path = Path(arguments["<gen-config-directory-path>"])
    config_paths = config_dir_path.glob("*.json")

    by_model_metadatas = defaultdict(list)
    for path in config_paths:
        model, compute, num_categoricals, tier, num_cat_cells, num_one_hot, *_ = str(
            path.name
        ).split("-")

        by_model_metadatas[model].append(
            {
                "path": path,
                "model": model,
                "compute": compute,
                "num_categoricals": num_categoricals,
                "tier": tier,
                "num_cat_cells": num_cat_cells,
                "num_one_hot": num_one_hot,
            }
        )

    if arguments["--grid"]:

        def key(x):
            return (
                x["tier"],
                x["num_one_hot"],
                x["num_cat_cells"],
                x["num_categoricals"],
            )

        ordered_by_model_metadatas = {
            model: sorted(metadatas, key=key)
            for model, metadatas in by_model_metadatas.items()
        }
    else:
        ordered_by_model_metadatas = {}
        num_ordered, num_more_complex = 0, 0

        for model, metadatas in by_model_metadatas.items():

            def key(x):
                return x["num_one_hot"] == "0"

            ordered_by_model_metadatas[model] = [
                metadata for metadata in metadatas if key(metadata)
            ]

            more_complex_metadatas = [
                metadata for metadata in metadatas if not key(metadata)
            ]

            num_ordered += len(ordered_by_model_metadatas[model])
            num_more_complex += len(more_complex_metadatas)

            random.shuffle(more_complex_metadatas)
            ordered_by_model_metadatas[model].extend(more_complex_metadatas)

        console.print(
            f"# grid search = {num_ordered}        # random search = {num_more_complex}"
        )

    model_interleaved_metatdatas = it.zip_longest(
        *ordered_by_model_metadatas.values(), fillvalue=None
    )
    config_metadatas = [
        metadata
        for metadata in it.chain.from_iterable(model_interleaved_metatdatas)
        if metadata is not None
    ]

    refs = []

    console.print(
        Counter(
            f"{metadata['model']}@{metadata['compute']}"
            for metadata in config_metadatas
        )
    )

    console.rule("Creating tasks...")

    for metadata in config_metadatas:
        main_args = {
            "execute": True,
            "<real-data-path>": arguments["<real-data-path>"],
            "<gen-config-path>": metadata["path"],
            "--minimal": True,
            "--wandb": True,
            "--subsampling": 0.4,
            "--verbose": True,
            "--random-seed": None,
            "--wandb-group": arguments["--wandb-group"],
        }

        if not arguments["--dry-run"]:
            if metadata["compute"] == "gpu":
                refs.append(execute_main_with_gpu.remote(main_args))
            else:
                refs.append(execute_main_with_cpu.remote(main_args))

        else:
            print(main_args)

    if not arguments["--dry-run"]:
        console.rule("Waiting to tasks...")

        finished_config_dir_path = config_dir_path / "finished"
        os.makedirs(finished_config_dir_path, exist_ok=True)

        unfinished = refs
        while unfinished:
            finished, unfinished = ray.wait(unfinished)  # , num_returns=1)

            for path in ray.get(finished):
                console.print(f"FINISHED: {path.name}")
                shutil.move(path, finished_config_dir_path / path.name)

    console.rule("Done!")


def _process_subampling_arguments(arguments: dict[str, str]):
    random_seed = Schema(Or(None, Use(int))).validate(arguments["--random-seed"])

    Schema(Use(open)).validate(arguments["<real-data-path>"])
    subsampling_frac = Schema(And(Use(float), lambda x: 0 < float(x) <= 1)).validate(
        arguments["--subsampling"]
    )

    return random_seed, subsampling_frac


def _run_evaluation(
    real_df: pd.DataFrame,
    synth_df,
    subsampling_frac,
    random_seed,
    processed_real_df,
    is_non_private_evaluation,
    is_minimal,
):
    console = Console()

    column_weights = prepare_column_weights(COLUMN_WEIGHTS, synth_df)

    (
        utility_df,
        privacy_df,
        faithfulness_df,
        subsampled_real_df,
        acceptance_df,
        dp_acceptance_df,
        dp_accountant,
    ) = evaluate(
        real_df,
        synth_df,
        EVALUATION_COLUMN_BINS,
        column_weights,
        ALPHAS,
        FACE_PRIVACY_UP_TO,
        ACCEPTANCE_CRITERIA,
        HIGH_SENSITIVITY_EPSILON,
        LOW_SENSITIVITY_EPSILON,
        CONDITIONAL_MEAN_EPSILONS,
        COMPLEX_QUERY_FIT_EPSILON,
        COMPLEX_QUERY_EVAL_EPSILON,
        RATIO_MAX_CLIPPING_FACTOR,
        subsampling_frac,
        random_seed,
        processed_real_df,
        is_minimal,
        USER_ANALYSIS,
        QUASI_SENSITIVE_PAIRS,
    )

    rows_to_show = (
        (
            utility_df["name"].isin(
                [
                    "mean",
                    "mean-resize-by-second",
                    "median",
                    "pMSE/score",
                    "pMSE/acc",
                    "std",
                ]
            )
        )
        | utility_df["name"].str.startswith("lr")
        | utility_df["name"].str.contains("1way/max/frequencies/ratio/raw")
        | utility_df["name"].str.contains(
            r"\d+way/max/frequencies/diff/raw", regex=True
        )
        | (
            utility_df["name"].str.startswith("corr")
            & utility_df["name"].str.endswith("diff/max")
        )
    )
    val_columns = [column for column in utility_df.columns if "val" in column]
    columns_to_show = val_columns + ["name", "target", "by", "targets", "binning"]
    succinct_utility_df = (
        utility_df.loc[rows_to_show, columns_to_show]
        .fillna("")
        .sort_values(by=["name", "target", "by"])
    )  # ignore

    if is_non_private_evaluation:
        console.print("Utility", style="blue")
        console.print(f"{EVALUATION_COLUMN_BINS=}")
        console.print(f"{subsampling_frac=}")
        console.print(f"{random_seed=}")

        with pd.option_context("display.max_rows", None):
            print(succinct_utility_df)
        console.print()

    if is_non_private_evaluation:
        console.print("Privacy", style="blue")
        print(privacy_df)
        console.print()

    if not is_minimal and is_non_private_evaluation:
        console.print("Faithfulness", style="blue")
        console.print(f"{column_weights=}")
        print(faithfulness_df.drop(["matching", "unmatched_indices"], axis=1))
        console.print()

    if is_non_private_evaluation:
        console.print("Acceptance", style="blue")
        print(acceptance_df)
        console.print()

    if not dp_acceptance_df.empty:
        console.print("DP Acceptance", style="blue")
        console.print(f"{HIGH_SENSITIVITY_EPSILON=}")
        console.print(f"{LOW_SENSITIVITY_EPSILON=}")
        console.print(f"{CONDITIONAL_MEAN_EPSILONS=}")
        console.print(f"{COMPLEX_QUERY_FIT_EPSILON=}")
        console.print(f"{COMPLEX_QUERY_EVAL_EPSILON=}")
        console.print(f"{RATIO_MAX_CLIPPING_FACTOR=}")
        print(dp_acceptance_df)
        console.print()

    console.print()

    return (
        utility_df,
        succinct_utility_df,
        privacy_df,
        faithfulness_df,
        subsampled_real_df,
        acceptance_df,
        dp_acceptance_df,
        column_weights,
        val_columns,
        dp_accountant,
    )


def execute_main(arguments: dict[str, str]):
    console = Console()

    random_seed, subsampling_frac = _process_subampling_arguments(arguments)

    if arguments["<gen-config-path>"] is not None:
        Schema(Use(open)).validate(arguments["<gen-config-path>"])
        with open(arguments["<gen-config-path>"]) as f:
            generate_config = json.load(f)

    if arguments["--wandb"]:
        wandb.login()
        wandb.init(
            config={**generate_config, **arguments},
            project="synthflow",
            group=arguments["--wandb-group"],
        )

    console.rule("Load Data")
    real_df = pd.read_csv(arguments["<real-data-path>"])
    real_df = _print_validate_coerce(real_df, "Real", REAL_DATASET_SCHEMA, console)

    console.rule("Generation")

    console.print()
    console.print(f"ε={generate_config['epsilon']} and δ={generate_config['delta']}")
    console.print()
    console.print(f"gen-config-path={arguments['<gen-config-path>']}")
    console.print()
    console.print(f"model={generate_config['model']}")
    console.print()
    console.print(f"categorical_mode={generate_config['categorical_mode']}")
    console.print()
    console.print(f"transformations={generate_config['transformations']}")
    console.print()
    console.print(f"tier={generate_config['tier']}")
    console.print()
    console.print(f"hparams={generate_config['hparams']}")
    console.print()
    console.print(f"{CONSTRAINTS_UBERSAMPLING=}")
    if CONSTRAINTS:
        console.print(CONSTRAINTS)
    console.print()
    console.print(f"dataset_projection={generate_config['dataset_projection']}")
    console.print()

    console.rule("Fitting & Sampling")
    (
        synth_df,
        model,
        synth_dataset_schema,
        processed_real_df,
        constraints_df,
        transcript,
    ) = generate(
        real_df,
        generate_config,
        REAL_DATASET_SCHEMA,
        # arguments["--wandb"],
        CONSTRAINTS,
        CONSTRAINTS_UBERSAMPLING,
    )

    console.rule("Synthetic & Processed Data")

    synth_df = _print_validate_coerce(
        synth_df, "Synthetic", synth_dataset_schema, console
    )

    processed_real_df = _print_validate_coerce(
        processed_real_df, "Processed", synth_dataset_schema, console
    )

    if CONSTRAINTS:
        constraints_df = _print_validate_coerce(
            constraints_df,
            "Constraints",
            synth_dataset_schema,
            console if arguments["--non-private-evaluation"] else None,
        )

    synth_data_name_path = arguments.get("--synth-data-name-path")
    if synth_data_name_path:
        (Path(synth_data_name_path).parent).mkdir(parents=True, exist_ok=True)
        synth_df.to_csv(synth_data_name_path + ".csv", index=False)
        synth_df.to_pickle(synth_data_name_path + ".pckl")

    processed_data_name_path = arguments.get("--processed-data-name-path")

    if processed_data_name_path is not None:
        (Path(processed_data_name_path).parent).mkdir(parents=True, exist_ok=True)
        processed_real_df.to_csv(processed_data_name_path + ".csv", index=False)
        processed_real_df.to_pickle(processed_data_name_path + ".pckl")

    console.rule("Evaluation")

    (
        utility_df,
        succinct_utility_df,
        privacy_df,
        faithfulness_df,
        subsampled_real_df,
        acceptance_df,
        dp_acceptance_df,
        column_weights,
        val_columns,
        dp_accountant,
    ) = _run_evaluation(
        real_df,
        synth_df,
        subsampling_frac,
        random_seed,
        processed_real_df,
        arguments["--non-private-evaluation"],
        arguments["--minimal"],
    )

    epsilon_generation = generate_config["epsilon"]
    delta_generation = generate_config["delta"]
    if not dp_acceptance_df.empty:
        dp_acceptance_all_record = dp_acceptance_df.query("name == 'all'")
        assert len(dp_acceptance_all_record) == 1
        epsilon_acceptance = dp_acceptance_all_record["epsilon"].iloc[0]
    else:
        epsilon_acceptance = 0

    delta_acceptance = 0
    total_epsilon = epsilon_generation + epsilon_acceptance
    total_delta = delta_generation + delta_acceptance
    dp_info = {
        "total": {"ε": total_epsilon, "δ": total_delta},
        "generation": {"ε": epsilon_generation, "δ": delta_generation},
        "acceptance": {"ε": epsilon_acceptance, "δ": delta_acceptance},
    }
    console.rule(f"DP Info {dp_info}")

    console.print()

    if arguments["--wandb"]:
        console.rule("W&B Logging")

        wandb.log(
            {
                f"dp_info/{source}/{param}": value
                for source, budget_params in dp_info.items()
                for param, value in budget_params.items()
            }
        )

        wandb.log(
            {
                "specific_config/alphas": ALPHAS,
                "specific_config/column_weights": column_weights,
                "specific_config/constraints_ubersampling": CONSTRAINTS_UBERSAMPLING,
                "specific_config/evaluation_column_bins": EVALUATION_COLUMN_BINS,
                "specific_config/high_sensitivity_epsilon": HIGH_SENSITIVITY_EPSILON,
                "specific_config/low_sensitivity_epsilon": LOW_SENSITIVITY_EPSILON,
                "specific_config/complex_query_fit_epsilon": COMPLEX_QUERY_FIT_EPSILON,
                "specific_config/complex_query_eval_epsilon": COMPLEX_QUERY_EVAL_EPSILON,  # noqa: E501
                "specific_config/ratio_max_clipping_factor": RATIO_MAX_CLIPPING_FACTOR,
            }
        )

        for _, r in succinct_utility_df.iterrows():
            key = f"utility/{r['name']}"
            if r["target"]:
                key += f"/{r['target']}"
            if r["by"]:
                key += f"/{r['by']}"
            wandb.log({f"{key}/{val_col}": r[val_col] for val_col in val_columns})

        if privacy_df is not None:
            for _, r in privacy_df.iterrows():
                key = f"privacy/face/{r['name']}"
                if r["quasi"]:
                    key += f"/{r['quasi']}"
                if r["sensitive"]:
                    key += f"/{r['sensitive']}"
                wandb.log({key: r["value"]})

        if faithfulness_df is not None:
            for _, r in faithfulness_df.iterrows():
                wandb.log(
                    {
                        f"faithfulness/β/ɑ={r['ɑ']}/{r['comparison']}": r["β"],
                        f"faithfulness/density/ɑ={r['ɑ']}/{r['comparison']}": r[
                            "E/½V²"
                        ],
                        f"faithfulness/algorithm/ɑ={r['ɑ']}/{r['comparison']}": r[
                            "algorithm"
                        ],
                    }
                )

        for _, r in acceptance_df.iterrows():
            wandb.log(
                {
                    f"acceptance/{r['name']}/{column}": r[column]
                    for column in ["expected", "actual", "check"]
                }
            )

        if not dp_acceptance_df.empty:
            for _, r in dp_acceptance_df.iterrows():
                wandb.log(
                    {
                        f"dp_acceptance/{r['name']}/{column}": r[column]
                        for column in [
                            "expected",
                            "actual",
                            "check",
                            "lower",
                            "upper",
                            "epsilon",
                            "var",
                            "mech",
                            "check",
                        ]
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdirname:
            tempdirpath = Path(tmpdirname)

            for type_, name, obj in [
                ("model", "model", model),
                ("dataset", "real", real_df),
                ("dataset", "synth", synth_df),
                ("dataset", "processed_real", processed_real_df),
                ("dataset", "subsampled_real", subsampled_real_df),
                ("dataset", "constraints", constraints_df),
                ("aux", "column_weights", column_weights),
                ("aux", "dp_info", dp_info),
                ("aux", "transcript", transcript),
            ]:
                if obj is not None:
                    assert type_ in ("model", "aux", "dataset")
                    filepath = tempdirpath / name
                    with open(filepath, "wb") as f:
                        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    _wanb_log_artifact(f.name, name, type_, as_file=True)

        wandb.log(
            {
                "utility": wandb.Table(dataframe=utility_df),
                "privacy": wandb.Table(dataframe=privacy_df),
                "faithfulness": wandb.Table(dataframe=faithfulness_df),
                "acceptance": wandb.Table(dataframe=acceptance_df),
                "dp_acceptance": wandb.Table(dataframe=dp_acceptance_df),
            }
        )

    wandb.finish()

    console.rule("Done!")

    return dp_acceptance_df


def evaluate_main(arguments: dict[str, str]):
    console = Console()

    random_seed, subsampling_frac = _process_subampling_arguments(arguments)

    console.rule("Load Data")
    real_df = pd.read_csv(arguments["<real-data-path>"])
    real_df = _print_validate_coerce(real_df, "Real", REAL_DATASET_SCHEMA, console)

    synth_df = pd.read_pickle(arguments["<synth-data-path>"])
    processed_real_df = pd.read_pickle(arguments["<processed-data-path>"])

    (
        utility_df,
        succinct_utility_df,
        privacy_df,
        faithfulness_df,
        subsampled_real_df,
        acceptance_df,
        _,
        column_weights,
        val_columns,
        _,
    ) = _run_evaluation(
        real_df,
        synth_df,
        subsampling_frac,
        random_seed,
        processed_real_df,
        False,
        arguments["--minimal"],
    )

    return utility_df, privacy_df, faithfulness_df, acceptance_df


def main(arguments: dict[str, str]):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if arguments["--verbose"] else logging.INFO,
    )

    if arguments["span"]:
        return span_main(arguments)
    elif arguments["parallel"]:
        return parallel_main(arguments)
    elif arguments["report"]:
        return report_main(arguments)
    elif arguments["execute"]:
        return execute_main(arguments)
    elif arguments["evaluate"]:
        return evaluate_main(arguments)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
