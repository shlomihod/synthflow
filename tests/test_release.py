import secrets

import numpy as np
from diffprivlib.mechanisms import Laplace, LaplaceFolded, LaplaceTruncated
from diffprivlib.utils import check_random_state
from scipy import stats

from synthflow.release import optimize_max_ratio_by_pseudo_threshold


def test_diffprivlib_use_secure_random():
    """Test whether diffprivlib uses cryptographic secure random.

    secrets.SystemRandom -> random.SystemRandom -> os.urandom

    Windows, Python 3.8: CryptGenRandom
    """

    laplace_mech = Laplace(epsilon=1, sensitivity=1)
    assert isinstance(laplace_mech._rng, secrets.SystemRandom)

    laplace_truncated_mech = LaplaceTruncated(
        epsilon=1, sensitivity=1, lower=0, upper=float("inf")
    )
    assert isinstance(laplace_truncated_mech._rng, secrets.SystemRandom)

    laplace_folded_mech = LaplaceFolded(
        epsilon=1, sensitivity=1, lower=0, upper=float("inf")
    )
    assert isinstance(laplace_folded_mech._rng, secrets.SystemRandom)

    rng = check_random_state(None, secure=True)
    assert isinstance(rng, secrets.SystemRandom)


def test_optimize_max_ratio_by_pseudo_threshold_args():
    (
        pseudo_threshold,
        clipping_factor,
        _,
        scale,
    ) = optimize_max_ratio_by_pseudo_threshold(
        prob_pass=0.05, min_synth_count=100, threshold=2, epsilon=0.3
    )

    assert clipping_factor == 2

    computed_prob_pass = 1 - stats.laplace.cdf(2 - pseudo_threshold, scale=scale)
    np.testing.assert_allclose(0.05, computed_prob_pass)


def test_optimize_max_ratio_by_pseudo_prob_pass():
    (
        pseudo_threshold,
        clipping_factor,
        global_sensitivity,
        scale,
    ) = optimize_max_ratio_by_pseudo_threshold(
        prob_pass=0.05, min_synth_count=100, threshold=2, epsilon=0.3
    )

    mech = Laplace(epsilon=0.3, sensitivity=global_sensitivity)
    np.testing.assert_allclose(
        0.05,
        np.mean([mech.randomise(pseudo_threshold) > 2 for _ in range(10**4)]),
        atol=0.01,
    )
