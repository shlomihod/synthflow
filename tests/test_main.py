import os
from unittest import mock

import numpy as np
import pytest

from synthflow.__main__ import evaluate_main, execute_main, report_main, span_main
from tests.utils import (
    GEN_CONFIG_PATH,
    PROCESSED_REAL_DATA_200K_PATH,
    REAL_DATA_200K_PATH,
    SYNTH_DATA_200K_PATH,
    WANDB_ONLINE_RUN_DIR_PATH,
)


@pytest.fixture
def wandb_offline():
    os.envriorment["WANDB_MODE"] = "offline"


def test_span_main(tmpdir):
    gen_config_dir = tmpdir.join("gen-config")
    arguments = {
        "<gen-config-directory-path>": str(gen_config_dir),
        "<epsilon>": 4,
        "<delta>": 0,
    }
    span_main(arguments)


@mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
def test_execute_main():
    arguments = {
        "<real-data-path>": REAL_DATA_200K_PATH,
        "<gen-config-path>": GEN_CONFIG_PATH,
        "--verbose": True,
        "--random-seed": "42",
        "--subsampling": "0.4",
        "--wandb": True,
        "--wandb-group": "testing",
        "--minimal": False,
        "--constraints-ubersampling": "1.01",
        "--acceptance-eps-low-sens": "0.05",
        "--acceptance-eps-high-sens": "0.2",
        "--non-private-evaluation": True,
    }
    print(arguments)
    dp_acceptance_df = execute_main(arguments)

    assert len(dp_acceptance_df) == 11
    laplace_only_dp_acceptance_df = (
        dp_acceptance_df[dp_acceptance_df["mech"] == "laplace"]
        .drop(columns=["name", "extra", "actual", "expected", "check", "mech"])
        .astype(float)
    )

    assert not laplace_only_dp_acceptance_df.isna().any().any()
    print(laplace_only_dp_acceptance_df.dtypes)
    print(laplace_only_dp_acceptance_df)

    np.testing.assert_allclose(
        laplace_only_dp_acceptance_df["lower"], [1, 0, 1, 1400, 28, 1400, 0]
    )
    np.testing.assert_allclose(
        laplace_only_dp_acceptance_df["upper"], [2, 1, 11, 4600, 43, 4600, 1]
    )
    np.testing.assert_allclose(
        laplace_only_dp_acceptance_df["epsilon"],
        [0.3, 0.01, 0.01, 0.17, 0.02, 0.04, 0.01],
    )

    # skip `1way/max/frequencies/ratio/raw`
    # and mean-resize-by-second
    # because sensitivity is depands on synthetic data counts
    np.testing.assert_allclose(
        laplace_only_dp_acceptance_df["var"].iloc[[1, 5, 6]],
        [5e-07, 1.28, 5e-07],
    )
    np.testing.assert_allclose(
        laplace_only_dp_acceptance_df["sensitivity"].iloc[[1, 5, 6]],
        [5e-06, 0.032, 5e-06],
    )


def test_evaluate_main():
    arguments = {
        "<real-data-path>": REAL_DATA_200K_PATH,
        "<synth-data-path>": SYNTH_DATA_200K_PATH,
        "<processed-data-path>": PROCESSED_REAL_DATA_200K_PATH,
        "--verbose": True,
        "--random-seed": "42",
        "--subsampling": "0.4",
        "--minimal": False,
    }
    print(arguments)
    *_, acceptance_df = evaluate_main(arguments)

    assert not acceptance_df.query("name == 'all'")["check"].item()


def test_online_report_main(tmpdir):
    import wandb

    wandb.login()

    reports_dir = str(tmpdir.join("reports"))

    arguments = {
        "<output-notebook-path>": reports_dir,
        "<run_path>": WANDB_ONLINE_RUN_DIR_PATH,
        "--wandb-mode": "online",
        "--minimal": True,
    }
    report_main(arguments)
