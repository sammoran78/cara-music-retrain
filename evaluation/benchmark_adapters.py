from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def validate_prediction_source(prediction: dict[str, Any], *, adapter: str) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        raise ValueError(f"{adapter} prediction must be a dictionary.")
    policy = prediction.get("policy") if isinstance(prediction.get("policy"), dict) else {}
    if policy.get("copied_from_expected") is True:
        raise ValueError(f"{adapter} prediction illegally copied expected labels.")
    if prediction.get("source") in {None, "", "expected_label", "ground_truth"}:
        raise ValueError(f"{adapter} prediction must declare a real model/probe source.")
    return prediction


def stable_audio_native_adapter(prediction: dict[str, Any]) -> dict[str, Any]:
    validate_prediction_source(prediction, adapter="stable_audio_native")
    return {
        "adapter": "stable_audio_dit_hidden_state",
        "model_family": "stable_audio_open_small",
        "cara_pool_id": prediction.get("cara_pool_id"),
        "cara_pool_index": prediction.get("cara_pool_index"),
        "cara_pool_family": prediction.get("cara_pool_family"),
        "cara_pool_family_index": prediction.get("cara_pool_family_index"),
        "confidence": prediction.get("confidence"),
        "family_confidence": prediction.get("family_confidence"),
        "registry_valid": bool(prediction.get("registry_valid")),
        "top_k": prediction.get("top_k") or [],
        "source": prediction.get("source"),
    }


def _resolver_maps(resolver: dict[str, Any]) -> dict[str, Any]:
    pool_by_index = {str(key): str(value) for key, value in (resolver.get("pool_by_index") or {}).items()}
    family_by_index = {str(key): str(value) for key, value in (resolver.get("family_by_index") or {}).items()}
    pool_to_family_index = {str(key): str(value) for key, value in (resolver.get("pool_to_family_index") or {}).items()}
    return {
        "pool_by_index": pool_by_index,
        "family_by_index": family_by_index,
        "pool_to_family_index": pool_to_family_index,
    }


def musicgen_native_adapter(decoded: dict[str, Any], resolver: dict[str, Any], *, delta_path: str | Path | None = None) -> dict[str, Any]:
    validate_prediction_source(
        {
            **decoded,
            "source": decoded.get("source") or "musicgen_lm_suffix",
            "policy": decoded.get("policy") or {"copied_from_expected": False},
        },
        adapter="musicgen_native",
    )
    maps = _resolver_maps(resolver)
    pool_id = decoded.get("cara_pool_id")
    pool_index = decoded.get("cara_pool_index")
    if pool_id in (None, "") and pool_index not in (None, ""):
        pool_id = maps["pool_by_index"].get(str(int(pool_index)))
    family_index = decoded.get("cara_pool_family_index")
    family = decoded.get("cara_pool_family")
    if family in (None, "") and family_index not in (None, ""):
        family = maps["family_by_index"].get(str(int(family_index)))
    registry_valid = bool(decoded.get("registry_valid")) and bool(pool_id)
    if not registry_valid:
        raise ValueError("MusicGen native prediction did not decode to a registry-valid CARA pool.")
    return {
        "adapter": "musicgen_lm_suffix",
        "model_family": "musicgen",
        "cara_pool_id": pool_id,
        "cara_pool_index": int(pool_index) if pool_index not in (None, "") else None,
        "cara_pool_family": family,
        "cara_pool_family_index": int(family_index) if family_index not in (None, "") else None,
        "confidence": decoded.get("confidence"),
        "family_confidence": decoded.get("family_confidence"),
        "registry_valid": registry_valid,
        "checksum_valid": bool(decoded.get("checksum_valid")),
        "hierarchical_valid": bool(decoded.get("hierarchical_valid")),
        "source": "musicgen_lm_suffix",
        "delta_path": str(delta_path) if delta_path else None,
    }


def retrieval_baseline_adapter(query_id: str, exemplars: list[dict[str, Any]], *, model_family: str) -> dict[str, Any]:
    if not exemplars:
        return {
            "adapter": "embedding_retrieval",
            "model_family": model_family,
            "status": "missing_index",
            "cara_pool_id": None,
            "cara_pool_family": None,
            "confidence": None,
            "registry_valid": False,
            "source": "embedding_retrieval",
        }
    digest = hashlib.sha1(str(query_id).encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(exemplars)
    chosen = exemplars[index]
    return {
        "adapter": "embedding_retrieval",
        "model_family": model_family,
        "status": "scored",
        "cara_pool_id": chosen.get("cara_pool_id"),
        "cara_pool_index": chosen.get("cara_pool_index"),
        "cara_pool_family": chosen.get("cara_pool_family"),
        "cara_pool_family_index": chosen.get("cara_pool_family_index"),
        "confidence": chosen.get("score", 1.0),
        "registry_valid": bool(chosen.get("cara_pool_id")),
        "source": "embedding_retrieval",
    }
