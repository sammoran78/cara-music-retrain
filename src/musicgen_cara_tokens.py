from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cara_attribution_head import build_cara_registry_resolver, validate_cara_manifest_labels


CARA_BOS = "<CARA_BOS>"
CARA_END = "<CARA_END>"
CARA_SEP = "<CARA_SEP>"
CARA_CHECK = "<CARA_CHECK>"


def _codeword_from_pool_id(pool_id: Any) -> str:
    text = str(pool_id or "").strip()
    parts = text.split(":")
    if len(parts) >= 5 and parts[0] == "CARA":
        return parts[3]
    return text


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_musicgen_encodec_manifest(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    label_summary = validate_cara_manifest_labels(rows)
    missing: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        split_counts[str(row.get("split") or "unknown")] += 1
        absent = [
            field
            for field in ("encodec_token_path", "encodec_code_shape", "encodec_frame_count")
            if row.get(field) in (None, "", [])
        ]
        if absent:
            missing.append(
                {
                    "row_index": row_index,
                    "chunk_id": row.get("chunk_id"),
                    "prepared_audio_path": row.get("prepared_audio_path"),
                    "missing_fields": absent,
                }
            )
    if missing:
        raise ValueError(f"MusicGen EnCodec manifest has {len(missing)} rows missing token-cache fields: {missing[:5]}")
    return {
        **label_summary,
        "format": "musicgen_encodec_manifest_v1",
        "split_counts": dict(sorted(split_counts.items())),
        "encodec_rows": len(rows),
    }


def build_musicgen_registry_resolver(rows: Iterable[dict[str, Any]], *, split_manifest_path: Path | None = None) -> dict[str, Any]:
    rows = list(rows)
    resolver = build_cara_registry_resolver(rows, split_manifest_path=split_manifest_path)
    resolver["format"] = "musicgen_cara_registry_resolver_v1"
    resolver["decoded_cara_id_format"] = "autoregressive CARA suffix resolved to locked cara_pool_id"
    resolver["musicgen_suffix_format"] = "CARA_BOS FAMILY_<index> POOL_<index> codeword chars CHECK_<2hex> CARA_END"
    resolver["registry_hash"] = _stable_hash(
        {
            "format": resolver["format"],
            "pool_by_index": resolver["pool_by_index"],
            "family_by_index": resolver["family_by_index"],
            "pool_to_family_index": resolver["pool_to_family_index"],
            "suffix_format": resolver["musicgen_suffix_format"],
        }
    )
    return resolver


def _checksum_token(pool_id: str) -> str:
    return f"CHECK_{hashlib.sha256(pool_id.encode('utf-8')).hexdigest()[:2].upper()}"


def cara_suffix_symbols(row: dict[str, Any]) -> list[str]:
    pool_id = str(row.get("cara_pool_id") or "").strip()
    if not pool_id:
        raise ValueError("Cannot build CARA suffix without cara_pool_id.")
    family_index = int(row["cara_pool_family_index"])
    pool_index = int(row["cara_pool_index"])
    codeword = str(row.get("cara_pool_codeword") or row.get("cara_registered_codeword") or _codeword_from_pool_id(pool_id))
    codeword = re.sub(r"[^A-Za-z0-9-]+", "", codeword).upper()
    return [
        CARA_BOS,
        f"FAMILY_{family_index}",
        f"POOL_{pool_index}",
        CARA_SEP,
        *[f"CHAR_{char}" for char in codeword],
        CARA_CHECK,
        _checksum_token(pool_id),
        CARA_END,
    ]


def build_cara_suffix_vocab(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    symbols = {CARA_BOS, CARA_END, CARA_SEP, CARA_CHECK}
    for row in rows:
        symbols.update(cara_suffix_symbols(row))
    ordered = sorted(symbols)
    token_to_id = {token: index for index, token in enumerate(ordered)}
    return {
        "format": "musicgen_cara_suffix_vocab_v1",
        "token_to_id": token_to_id,
        "id_to_token": {str(index): token for token, index in token_to_id.items()},
        "size": len(token_to_id),
        "hash": _stable_hash(token_to_id),
    }


def encode_cara_suffix(row: dict[str, Any], vocab: dict[str, Any]) -> list[int]:
    token_to_id = vocab["token_to_id"]
    return [int(token_to_id[symbol]) for symbol in cara_suffix_symbols(row)]


def decode_cara_suffix(token_ids: Iterable[int], vocab: dict[str, Any], resolver: dict[str, Any]) -> dict[str, Any]:
    id_to_token = {int(key): value for key, value in vocab["id_to_token"].items()}
    symbols = [id_to_token.get(int(token_id), "<UNK>") for token_id in token_ids]
    pool_symbol = next((symbol for symbol in symbols if symbol.startswith("POOL_")), "")
    family_symbol = next((symbol for symbol in symbols if symbol.startswith("FAMILY_")), "")
    pool_index = int(pool_symbol.split("_", 1)[1]) if pool_symbol else None
    family_index = int(family_symbol.split("_", 1)[1]) if family_symbol else None
    pool_id = resolver.get("pool_by_index", {}).get(str(pool_index)) if pool_index is not None else None
    expected_family = resolver.get("pool_to_family_index", {}).get(str(pool_index)) if pool_index is not None else None
    checksum = next((symbol for symbol in symbols if symbol.startswith("CHECK_") and symbol != CARA_CHECK), "")
    checksum_valid = bool(pool_id and checksum == _checksum_token(str(pool_id)))
    return {
        "symbols": symbols,
        "pool_index": pool_index,
        "family_index": family_index,
        "cara_pool_id": pool_id,
        "registry_valid": pool_id is not None,
        "hierarchical_valid": expected_family is None or family_index == int(expected_family),
        "checksum_valid": checksum_valid,
    }

