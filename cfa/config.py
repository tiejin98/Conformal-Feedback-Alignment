"""Configuration loading and management."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Package root (where pyproject.toml lives)
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def get_default_config_path() -> Path:
    """Return path to the default config, checking multiple locations.

    When installed via pip, configs/ may not be in the package.
    Falls back to looking relative to the current working directory.
    """
    # 1. Relative to project root (dev / git clone)
    candidate = PROJECT_ROOT / "configs" / "default.yaml"
    if candidate.exists():
        return candidate

    # 2. Relative to CWD (user's working directory)
    candidate = Path.cwd() / "configs" / "default.yaml"
    if candidate.exists():
        return candidate

    # 3. Bundled inside the package
    candidate = PACKAGE_ROOT / "configs" / "default.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find configs/default.yaml. "
        "Please specify --config path/to/config.yaml explicitly."
    )


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file with environment variable support.

    Args:
        config_path: Path to YAML config file. Defaults to configs/default.yaml.

    Returns:
        Configuration dictionary.
    """
    load_dotenv()

    if config_path is None:
        config_path = get_default_config_path()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Inject environment variables for sensitive values
    config["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    config["hf_token"] = os.environ.get("HF_TOKEN", "")

    return config


def get_output_dir(config: dict, stage: str) -> Path:
    """Get the output directory for a pipeline stage, creating it if needed.

    Args:
        config: Configuration dictionary.
        stage: Stage name (generation, calibration, feedback, inference, evaluation).

    Returns:
        Path to the stage output directory.
    """
    output_dir = Path(config["data"]["output_dir"]) / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
