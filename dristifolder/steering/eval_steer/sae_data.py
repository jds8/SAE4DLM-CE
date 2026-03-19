"""
File discovery utilities for SAE steering JSON files.

This module provides a helper to locate steering result JSON files under
a given root directory. The JSON files are assumed to follow a filename
pattern like:

    steer_*_layer*_l0_*.json

Examples:
    steer_dlm_layer1_l0_80.json
    steer_qwen2.5_layer1_l0_320.json
    steer_llama3_layer10_l0_160.json
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union


def find_steering_json_files(root_dir: Union[str, Path]) -> List[Path]:
    """
    Recursively find all steering JSON files under the given root directory.

    Parameters
    ----------
    root_dir:
        Directory under which to search for steering JSON files.

    Returns
    -------
    List[Path]
        Sorted list of Paths to JSON files matching the pattern
        "steer_*_layer*_l0_*.json".
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    # Use rglob to be robust to possible nested subdirectories in the future.
    paths = sorted(root.rglob("steer_*_layer*_l0_*.json"))
    return paths
