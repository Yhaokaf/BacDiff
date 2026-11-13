from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_run_dir(config: dict[str, Any]) -> Path:
    """Resolve the output directory for one configured run."""
    if "output_dir" in config and "run_name" in config:
        return Path(config["output_dir"]) / str(config["run_name"])
    return Path(config["output"]["root_dir"])
