from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_custom_metadata(info: dict[str, Any], audio: Any) -> dict[str, Any]:
    audio_path = Path(info["path"])
    sidecar_path = audio_path.with_suffix(audio_path.suffix + ".json")
    if not sidecar_path.exists():
        return {"prompt": "high quality audio"}
    with sidecar_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "prompt": data.get("prompt", "high quality audio"),
        "cara_codeword": data.get("cara_codeword", ""),
        "cara_pool_name": data.get("cara_pool_name", ""),
        "cara_family_codeword": data.get("cara_family_codeword", ""),
        "cara_soft_targets": data.get("cara_soft_targets", []),
        "source": data.get("source", ""),
        "source_id": data.get("source_id", ""),
    }
