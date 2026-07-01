from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicgen_cara_tokens import (
    build_cara_suffix_vocab,
    build_musicgen_registry_resolver,
    cara_suffix_symbols,
    decode_cara_suffix,
    encode_cara_suffix,
    validate_musicgen_encodec_manifest,
)
from musicgen_lm_cara_trainer import _filter_rows_with_enough_encodec_frames


def _rows() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-a",
            "split": "train",
            "prepared_audio_path": "musicgen/audio/train/a.wav",
            "cara_pool_id": "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q",
            "cara_pool_index": 0,
            "cara_pool_family": "Ambient",
            "cara_pool_family_index": 1,
            "encodec_token_path": "tokens/train/chunk-a.pt",
            "encodec_code_shape": [1, 4, 1500],
            "encodec_frame_count": 1500,
        },
        {
            "chunk_id": "chunk-b",
            "split": "validation",
            "prepared_audio_path": "musicgen/audio/validation/b.wav",
            "cara_pool_id": "CARA:AUD:1:7BCD-EFGH-IJKL:2A",
            "cara_pool_index": 1,
            "cara_pool_family": "Percussion",
            "cara_pool_family_index": 2,
            "encodec_token_path": "tokens/validation/chunk-b.pt",
            "encodec_code_shape": [1, 4, 1200],
            "encodec_frame_count": 1200,
        },
    ]


def test_musicgen_encodec_manifest_requires_token_and_cara_fields() -> None:
    rows = _rows()
    rows[0].pop("encodec_token_path")

    with pytest.raises(ValueError, match="missing token-cache fields"):
        validate_musicgen_encodec_manifest(rows)


def test_cara_suffix_round_trips_through_registry() -> None:
    rows = _rows()
    resolver = build_musicgen_registry_resolver(rows)
    vocab = build_cara_suffix_vocab(rows)
    encoded = encode_cara_suffix(rows[0], vocab)
    decoded = decode_cara_suffix(encoded, vocab, resolver)

    assert cara_suffix_symbols(rows[0])[0] == "<CARA_BOS>"
    assert decoded["cara_pool_id"] == "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q"
    assert decoded["registry_valid"] is True
    assert decoded["hierarchical_valid"] is True
    assert decoded["checksum_valid"] is True


def test_musicgen_lm_trainer_filters_too_short_encodec_rows_before_loading() -> None:
    rows = _rows()
    rows.append(
        {
            "chunk_id": "chunk-too-short",
            "split": "train",
            "prepared_audio_path": "musicgen/audio/train/too-short.wav",
            "cara_pool_id": "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q",
            "cara_pool_index": 0,
            "cara_pool_family": "Ambient",
            "cara_pool_family_index": 1,
            "encodec_token_path": "tokens/train/too-short.pt",
            "encodec_code_shape": [1, 4, 1],
            "encodec_frame_count": 1,
        }
    )

    kept, rejected = _filter_rows_with_enough_encodec_frames(rows, min_frames=2)

    assert [row["chunk_id"] for row in kept] == ["chunk-a", "chunk-b"]
    assert rejected == [
        {
            "chunk_id": "chunk-too-short",
            "split": "train",
            "prepared_audio_path": "musicgen/audio/train/too-short.wav",
            "encodec_token_path": "tokens/train/too-short.pt",
            "encodec_frame_count": 1,
            "reject_reason": "encodec_frame_count_lt_2",
        }
    ]


def test_musicgen_azure_jobs_use_real_lm_trainer() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "azureml/jobs/07b_musicgen_trainer_preflight.yml",
        "azureml/jobs/08_smoke_musicgen_ar_trainer.yml",
        "azureml/jobs/12_full_musicgen_cara_strong_trainer.yml",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "python musicgen_lm_cara_trainer.py" in text
        assert "python smoke_musicgen_ar_trainer.py" not in text
        assert "--max_token_frames" in text
        assert "--min_encodec_frames" in text
        assert "--model_dtype" in text

    full_job = (root / "azureml/jobs/12_full_musicgen_cara_strong_trainer.yml").read_text(encoding="utf-8")
    assert "placeholder over cached EnCodec tokens" not in full_job
    assert "real MusicGen LM CARA-Strong full fine-tune" in full_job

    backend = (root / "gui/backend/main.py").read_text(encoding="utf-8")
    assert "_TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION" in backend
    assert "implementation_matches" in backend
    assert "effective_batch_size = max(1, min(requested_batch_size, 2))" in backend
