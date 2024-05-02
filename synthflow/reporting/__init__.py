from __future__ import annotations

from pathlib import Path

import papermill as pm  # type: ignore

INPUT_NOTEBOOK_NAME = "template.ipynb"
INPUT_NOTEBOOK_PATH = Path(__file__).parent / INPUT_NOTEBOOK_NAME


def create_report(
    notebook_dir_path: str, run_path: str, mode: bool = False, is_minimal: bool = False
):
    name = run_path.replace("/", "_")
    output_notebook_path = str(Path(notebook_dir_path) / name) + ".ipynb"
    pm.execute_notebook(
        INPUT_NOTEBOOK_PATH,
        output_notebook_path,
        parameters=dict(
            run_path=run_path,
            root_path=notebook_dir_path,
            mode=mode,
            is_minimal=is_minimal,
        ),
    )
