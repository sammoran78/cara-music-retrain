from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(env_path: Path | None = None) -> dict[str, str]:
    path = env_path or ENV_PATH
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        if key:
            loaded[key] = os.environ.get(key, value)
    return loaded


def get_env(key: str, default: str | None = None) -> str | None:
    if key not in os.environ:
        load_env_file()
    return os.environ.get(key, default)
