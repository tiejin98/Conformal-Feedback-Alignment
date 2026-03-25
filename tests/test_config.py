"""Tests for configuration loading."""

import os
import pytest
from pathlib import Path
from cfa.config import load_config, get_output_dir


PROJECT_ROOT = Path(__file__).parent.parent


class TestLoadConfig:
    def test_loads_default_config(self):
        config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert "model" in config
        assert "data" in config
        assert "sft" in config
        assert "generation" in config
        assert "conformal" in config
        assert "dpo" in config

    def test_loads_mwe_config(self):
        config = load_config(str(PROJECT_ROOT / "configs" / "mwe.yaml"))
        assert config["generation"]["calibration_size"] == 5
        assert config["generation"]["sampling_num"] == 5

    def test_env_variables_injected(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("HF_TOKEN", "test-hf-456")
        config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert config["openai_api_key"] == "test-key-123"
        assert config["hf_token"] == "test-hf-456"

    def test_missing_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert config["openai_api_key"] == ""
        assert config["hf_token"] == ""

    def test_default_values(self):
        config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert config["generation"]["seed"] == 42
        assert config["conformal"]["accuracy_threshold"] == 0.7
        assert len(config["conformal"]["quantile_bars"]) == 2


class TestGetOutputDir:
    def test_creates_directory(self, tmp_path):
        config = {"data": {"output_dir": str(tmp_path / "test_outputs")}}
        result = get_output_dir(config, "generation")
        assert result.exists()
        assert result == tmp_path / "test_outputs" / "generation"

    def test_nested_stage(self, tmp_path):
        config = {"data": {"output_dir": str(tmp_path / "out")}}
        result = get_output_dir(config, "calibration")
        assert result.name == "calibration"
        assert result.parent.name == "out"
