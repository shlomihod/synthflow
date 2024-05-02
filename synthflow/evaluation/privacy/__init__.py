from __future__ import annotations

import pandas as pd  # type: ignore

from synthflow.evaluation.privacy.face import evaluate_face_privacy


def evaluate_privacy(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    face_up_to: int,
    processed_real_df: None | pd.DataFrame = None,
    quasi_snsitive_pairs: None | tuple[tuple[tuple[str], str]] = None,
):
    return evaluate_face_privacy(synth_df, face_up_to, quasi_snsitive_pairs)
