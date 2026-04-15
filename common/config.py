from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from common.env import get_env, load_env_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "data_pipeline" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _merge_env(config: dict[str, Any]) -> dict[str, Any]:
    load_env_file()
    merged = dict(config)
    freesound = dict(merged.get("freesound", {}))
    hf = dict(merged.get("huggingface", {}))

    freesound_client_id = get_env("FREESOUND_CLIENT_ID")
    freesound_client_secret = get_env("FREESOUND_CLIENT_SECRET")
    freesound_access_token = get_env("FREESOUND_ACCESS_TOKEN")
    freesound_refresh_token = get_env("FREESOUND_REFRESH_TOKEN")
    hf_token = get_env("HF_TOKEN")

    if freesound_client_id:
        freesound["client_id"] = freesound_client_id
    if freesound_client_secret:
        freesound["client_secret"] = freesound_client_secret
    if freesound_access_token:
        freesound["access_token"] = freesound_access_token
    if freesound_refresh_token:
        freesound["refresh_token"] = freesound_refresh_token
    if hf_token:
        hf["token"] = hf_token

    if freesound:
        merged["freesound"] = freesound
    if hf:
        merged["huggingface"] = hf
    return merged


def load_project_config() -> dict[str, Any]:
    return _merge_env(_read_yaml(CONFIG_PATH))


def load_pipeline_config() -> dict[str, Any]:
    return _merge_env(_read_yaml(PIPELINE_CONFIG_PATH))
