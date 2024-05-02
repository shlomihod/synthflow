from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

DELTA_EXPONENT_DATA_SIZE = 1.5


def assert_keys(d: dict, optional, must=None):
    must = set() if must is None else set(must)
    allowed = set(optional) | must
    keys = set(d.keys())
    assert must.issubset(keys), f"Given keys {keys} must have all {must}"
    assert keys.issubset(allowed), f"Given keys {keys} should be only from {allowed}"


def _fit_sample(
    processed_df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]],
    model_name: str,
    epsilon: float,
    delta: float,
    hparams: dict[str, Any],
    categoricals: Sequence[str],
    num_records: None | int = None,
) -> tuple[pd.DataFrame, Any]:
    hparams = deepcopy(hparams)

    if num_records is None:
        num_records = len(processed_df)

    if delta != 0:
        maximial_acceptable_delta = (
            1 / (num_records**DELTA_EXPONENT_DATA_SIZE) + np.finfo(float).eps
        )

        assert (
            delta <= maximial_acceptable_delta
        ), f"δ is too big. It should be not more then 1/N**{DELTA_EXPONENT_DATA_SIZE}"

    if model_name == "ID":
        # currenty we use this model for evaluation,
        # so no check should be done
        # assert epsilon == float("inf") and delta == 0
        # assert_keys(hparams, {})
        model = None
        samples_df = processed_df.copy()

    elif model_name == "GC":
        from sdv.tabular import GaussianCopula

        assert epsilon == float("inf") and delta == 0
        assert_keys(hparams, {"field_distributions"})

        if hparams.get("field_distributions") == "none":
            hparams["field_distributions"]

        field_transformers = {column: "float" for column in processed_df.columns}

        model = GaussianCopula(field_transformers=field_transformers, **hparams)
        model.fit(processed_df)
        samples_df = model.sample(num_records)

    elif model_name == "DPGC":
        raise NotImplementedError()

    elif model_name.startswith("PBN"):
        from synthflow.generation.models.privbayes import PrivBayes

        assert delta == 0
        assert_keys(hparams, {"degree", "theta", "epsilon_split"})
        model = PrivBayes(epsilon=epsilon, **hparams)
        samples_df = model.fit_sample(processed_df, ranges, categoricals, num_records)

    elif model_name == "MWEM":
        from snsynth.mwem import MWEMSynthesizer  # type: ignore

        assert delta == 0
        assert_keys(
            hparams,
            {
                "q_count",
                "iterations",
                "mult_weights_iterations",
                "splits",
                "split_factor",
            },
        )
        assert set(categoricals) == set(processed_df.columns)
        # synthflow is responsible to binning, so `max_bin_count`` is set to inf
        model = MWEMSynthesizer(epsilon=epsilon, max_bin_count=float("inf"), **hparams)
        samples_df = model.fit_sample(processed_df)

    elif model_name == "PEP":
        from iterative_dp import PEPSynth

        assert delta > 0
        assert_keys(
            hparams,
            {
                "marginal",
                "workload",
                "workload_seed",
                "T",
                "iters",
            },
        )
        hparams["workload_seed"] = hparams.get("workload_seed")
        assert set(categoricals) == set(processed_df.columns)
        model = PEPSynth(epsilon, delta, **hparams)
        samples_df = model.fit_sample(processed_df)

    elif model_name == "CTGAN":
        from ctgan.synthesizers import CTGANSynthesizer  # type: ignore

        assert epsilon == float("inf") and delta == 0

        # epochs (300): Number of training epochs
        # generator_lr (2e-4): Learning rate for the generator
        # discriminator_lr (2e-4): Learning rate for the discriminator
        # generator_decay (1e-6): Weight decay for the generator
        # discriminator_decay (1e-6): Weight decay for the discriminator
        # discriminator_steps (1): Number of discriminator updates
        #                          to do for each generator update
        # embedding_dim (128): Dimension of input z to the generator
        # generator_dim ([256, 256]): Dimension of each generator layer
        # discriminator_dim ([256, 256]): Dimension of each discriminator layer
        # batch_size (500): Batch size. Must be an even number
        # log_frequency (True): Whether to use log frequency of categorical levels
        #                       in conditional sampling
        # pac (10): Number of samples to group together when applying the discriminator

        assert_keys(
            hparams,
            {
                "epochs",
                "generator_lr",
                "discriminator_lr",
                "generator_decay",
                "discriminator_decay",
                "discriminator_steps",
                "embedding_dim",
                "generator_dim",
                "discriminator_dim",
                "batch_size",
                "log_frequency",
                "pac",
            },
        )
        model = CTGANSynthesizer(verbose=True, **hparams)
        model.fit(processed_df, discrete_columns=categoricals)
        samples_df = model.sample(num_records)

    elif model_name == "DPGAN":
        from snsynth.pytorch import PytorchDPSynthesizer  # type: ignore
        from snsynth.pytorch.nn import DPGAN  # type: ignore

        assert_keys(
            hparams,
            {
                "discriminator_lr",
                "generator_lr",
                "discriminator_decay",
                "generator_decay",
                "sigma",
                "max_per_sample_grad_norm",
                "latent_dim",
                "batch_size",
                "epochs",
            },
        )

        # DP currently works only for categorigal columns
        assert set(categoricals) == set(processed_df.columns)

        model = PytorchDPSynthesizer(
            epsilon, DPGAN(epsilon=epsilon, delta=delta, **hparams), None
        )

        model.fit(processed_df, categorical_columns=categoricals)
        samples_df = model.sample(num_records)

    elif model_name == "PATEGAN":
        from snsynth.pytorch import PytorchDPSynthesizer  # type: ignore
        from snsynth.pytorch.nn import PATEGAN  # type: ignore

        assert_keys(
            hparams,
            {
                "latent_dim",
                "batch_size",
                "teacher_iters",
                "student_iters",
                "generator_lr",
                "generator_decay",
                "discriminator_dim",
                "discriminator_lr",
                "discriminator_decay",
                "noise_multiplier",
                "regularization",
                "sample_per_teacher",
                "moments_order",
                "noise_multiplier",
            },
        )

        # DP currently works only for categorigal columns
        assert set(categoricals) == set(processed_df.columns)

        model = PytorchDPSynthesizer(
            epsilon,
            PATEGAN(epsilon=epsilon, delta=delta, verbose=True, **hparams),
            None,
        )
        model.fit(processed_df, categorical_columns=categoricals)
        samples_df = model.sample(num_records)

    elif model_name == "DPCTGAN":
        from snsynth.pytorch import PytorchDPSynthesizer  # type: ignore
        from snsynth.pytorch.nn import DPCTGAN

        assert_keys(
            hparams,
            {
                "epochs",
                "generator_lr",
                "discriminator_lr",
                "generator_decay",
                "discriminator_decay",
                "discriminator_steps",
                "embedding_dim",
                "generator_dim",
                "discriminator_dim",
                "batch_size",
                "pac",
                "sigma",
                "max_per_sample_grad_norm",
                "category_epsilon_pct",
                "loss",
            },
        )

        # DP currently works only for categorigal columns
        assert set(categoricals) == set(processed_df.columns)

        model = PytorchDPSynthesizer(
            epsilon, DPCTGAN(delta=delta, verbose=True, **hparams), None
        )

        model.fit(processed_df, categorical_columns=categoricals)
        samples_df = model.sample(num_records)

    elif model_name == "PATECTGAN":
        from snsynth.pytorch import PytorchDPSynthesizer  # type: ignore
        from snsynth.pytorch.nn import PATECTGAN  # type: ignore

        assert_keys(
            hparams,
            {
                "epochs",
                "generator_lr",
                "discriminator_lr",
                "generator_decay",
                "discriminator_decay",
                "discriminator_steps",
                "embedding_dim",
                "generator_dim",
                "discriminator_dim",
                "batch_size",
                "pac",
                "teacher_iters",
                "student_iters",
                "regularization",
                "sample_per_teacher",
                "moments_order",
                "noise_multiplier",
                "loss",
            },
        )

        # DP currently works only for categorigal columns
        assert set(categoricals) == set(processed_df.columns)

        model = PytorchDPSynthesizer(
            epsilon, PATECTGAN(delta=delta, verbose=True, **hparams), None
        )
        model.fit(processed_df, categorical_columns=categoricals)
        samples_df = model.sample(num_records)

    else:
        raise ValueError(f"Model `{model_name}` does not exist.")

    return samples_df, model
