"""Smoke tests for runnable examples."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIRECTORY = _REPOSITORY_ROOT / "examples"
_RUNNABLE_EXAMPLES = (
    "bulk_publication_lookup_with_cdist.py",
    "catalog_lookup_with_fuzzy_dict.py",
    "database_ids_with_keyed_index.py",
    "mutable_reference_data.py",
    "ordered_list_cleanup_with_fuzzy_list.py",
    "record_lookup_with_sequence_index.py",
    "reference_data_normalization_with_fuzzy_dict.py",
    "vocabulary_lookup_with_fuzzy_set.py",
)


@pytest.mark.parametrize("script_name", _RUNNABLE_EXAMPLES)
def test_example_script_runs(script_name: str) -> None:
    """Verify that a documented example script runs successfully."""
    python_path = str(_REPOSITORY_ROOT / "src")
    if inherited_python_path := os.environ.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, inherited_python_path))
    subprocess.run(
        [sys.executable, str(_EXAMPLES_DIRECTORY / script_name)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        env=os.environ | {"PYTHONPATH": python_path},
        text=True,
        timeout=30,
    )
