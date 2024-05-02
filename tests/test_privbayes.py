import numpy as np
import pandas as pd
from scipy import stats
import pytest

from synthflow.birth import REAL_DATASET_SCHEMA
from synthflow.generation import generate
from synthflow.utils import interval_number
from tests.utils import (  # noqa: F401
    nvss_10k_data,
)

PVALUE_THRESHOLD = 1/20

def test_privbayes_transcript(nvss_10k_data):  # noqa: F811
    """Test privbayes transcript, both for structure and noise distribution."""

    real_df, _, gen_config = nvss_10k_data

    constraints = [
            lambda r: r["mother_age"] < interval_number(37),
            lambda r: r["is_female"] | (r["parity"] < interval_number(4)),
    ]
    _, _, _, _, _, transcript = generate(
        real_df, gen_config, REAL_DATASET_SCHEMA, constraints, ubersampling=2
    )

    assert [x for x, y in transcript[1:]] == ["Unprocessed", "Sampled", "Constrained", "Projected"]

    logs = transcript[0].splitlines()
    hist_logs = (line[1:].split(" ") for line in logs if "0x" in line and "<-" not in line)
    counts, random_noise, noisy_counts = zip(*hist_logs)

    counts = np.array([int(x) for x in counts])
    random_noise = np.array([float.fromhex(x) for x in random_noise])
    noisy_counts = np.array([float.fromhex(x) for x in noisy_counts])

    # Test that the noisy counts are the sume of random noise and counts
    np.testing.assert_allclose(noisy_counts, counts + random_noise)

    lap_scale_set = {line.split(" ")[-1] for line in logs if "with Laplace noise scale" in line}
    assert len(lap_scale_set) == 1
    lap_scale = float(next(iter(lap_scale_set)))

    # Test that the noise is Laplace distributed
    ks_result = stats.kstest(1/lap_scale * random_noise, stats.laplace.cdf)

    assert ks_result.pvalue > PVALUE_THRESHOLD


