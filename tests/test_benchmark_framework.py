from __future__ import annotations

import pytest

from evaluation.benchmark_adapters import (
    musicgen_native_adapter,
    retrieval_baseline_adapter,
    stable_audio_native_adapter,
)
from evaluation.benchmark_spec import (
    CONDITION_HELDOUT_AUDIO,
    CONDITION_NO_TAG,
    CONDITION_OPEN_QUALITY,
    CONDITION_SHUFFLED_LABEL,
    CONDITION_TAG_PRESENT,
    CONDITION_TAG_WITHHELD,
    PROMPT_CONDITIONS,
    build_prompt_manifest_v2,
    comparison_cards,
    default_metric_rows,
    model_lanes,
)
from evaluation.metrics_v2 import balanced_accuracy, bootstrap_ci, ece, macro_f1, summarize_prediction_rows, topk_accuracy


def _rows() -> list[dict[str, object]]:
    return [
        {
            "example_id": "ex-a",
            "source_id": "a",
            "split": "validation",
            "split_group_key": "artist:a",
            "prompt": "Warm ambient texture CARA:AUD:1:AAAA-BBBB-CCCC:Q1 AAAA-BBBB-CCCC",
            "cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1",
            "cara_pool_codeword": "AAAA-BBBB-CCCC",
            "cara_pool_index": 0,
            "cara_pool_family": "Ambient",
            "cara_pool_family_index": 0,
        },
        {
            "example_id": "ex-b",
            "source_id": "b",
            "split": "test",
            "split_group_key": "artist:b",
            "prompt": "Percussion loop CARA:AUD:1:DDDD-EEEE-FFFF:X9 DDDD-EEEE-FFFF",
            "cara_pool_id": "CARA:AUD:1:DDDD-EEEE-FFFF:X9",
            "cara_pool_codeword": "DDDD-EEEE-FFFF",
            "cara_pool_index": 1,
            "cara_pool_family": "Percussion",
            "cara_pool_family_index": 1,
        },
    ]


def test_prompt_manifest_v2_has_required_conditions_and_deterministic_ids() -> None:
    first = build_prompt_manifest_v2(_rows(), seeds=1, condition_limit=2)
    second = build_prompt_manifest_v2(_rows(), seeds=1, condition_limit=2)

    assert [row["prompt_id"] for row in first] == [row["prompt_id"] for row in second]
    assert {row["condition"] for row in first} == set(PROMPT_CONDITIONS)
    assert {row["condition"] for row in first} == {
        CONDITION_TAG_PRESENT,
        CONDITION_TAG_WITHHELD,
        CONDITION_NO_TAG,
        CONDITION_SHUFFLED_LABEL,
        CONDITION_HELDOUT_AUDIO,
        CONDITION_OPEN_QUALITY,
    }
    assert all(row["source"]["source_group_key"] for row in first if row["condition"] != CONDITION_OPEN_QUALITY)


def test_prompt_manifest_v2_withheld_and_no_tag_do_not_leak_cara_tokens() -> None:
    rows = build_prompt_manifest_v2(_rows(), seeds=1, condition_limit=2)
    protected = [row for row in rows if row["condition"] in {CONDITION_TAG_WITHHELD, CONDITION_NO_TAG}]

    for row in protected:
        prompt = row["prompt"]
        expected = row["expected"]
        assert expected["cara_pool_id"] not in prompt
        assert expected["cara_pool_codeword"] not in prompt


def test_prompt_manifest_v2_shuffled_labels_are_actually_shuffled() -> None:
    rows = build_prompt_manifest_v2(_rows(), seeds=1, condition_limit=2)
    shuffled = [row for row in rows if row["condition"] == CONDITION_SHUFFLED_LABEL]
    original_by_example = {row["example_id"]: row["cara_pool_id"] for row in _rows()}

    assert {row["expected"]["cara_pool_id"] for row in shuffled} == {row["cara_pool_id"] for row in _rows()}
    assert all(
        row["expected"]["cara_pool_id"] != original_by_example[row["source"]["source_example_id"]]
        for row in shuffled
    )


def test_model_lanes_include_core_lanes_and_future_models_without_schema_changes() -> None:
    lanes = model_lanes(
        [
            {
                "model_id": "future_model_candidate",
                "label": "Future Model",
                "family": "future",
                "architecture": "diffusion",
                "variant": "cara_strong",
                "checkpoint_uri": "future/checkpoint",
                "output_uri": None,
                "native_prediction_adapter": "future_adapter",
                "generation_adapter": "future_generation",
                "native_cara_output": True,
                "baseline_role": "candidate",
            }
        ]
    )

    ids = {lane["model_id"] for lane in lanes}
    assert "diffusion_cara_strong_full_modest_arch" in ids
    assert "musicgen_cara_strong_full" in ids
    assert "retrieval_baseline" in ids
    assert "future_model_candidate" in ids
    for lane in lanes:
        assert {"model_id", "family", "architecture", "variant", "checkpoint_uri", "output_uri", "native_prediction_adapter", "generation_adapter", "status"} <= set(lane)


def test_metrics_v2_cover_peer_review_rows() -> None:
    rows = [
        {
            "expected_pool_id": "pool-a",
            "predicted_pool_id": "pool-a",
            "top_k": ["pool-a", "pool-b"],
            "expected_family": "fam-a",
            "predicted_family": "fam-a",
            "confidence": 0.9,
            "exact": True,
            "registry_valid": True,
            "tier": "exact_pool",
        },
        {
            "expected_pool_id": "pool-b",
            "predicted_pool_id": "pool-a",
            "top_k": ["pool-a", "pool-b"],
            "expected_family": "fam-b",
            "predicted_family": "fam-a",
            "confidence": 0.6,
            "exact": False,
            "registry_valid": True,
            "tier": "exception",
        },
    ]

    assert topk_accuracy(rows, k=1) == 0.5
    assert topk_accuracy(rows, k=3) == 1.0
    assert balanced_accuracy(rows) == 0.5
    assert macro_f1(rows) == pytest.approx(1 / 3)
    assert ece(rows) == pytest.approx(0.35)
    summary = summarize_prediction_rows(rows)
    assert summary["registry_valid_rate"] == 1.0
    assert summary["exception_rate"] == 0.5
    ci = bootstrap_ci([1, 0, 1], samples=50)
    assert set(ci) == {"mean", "ci_low", "ci_high"}


def test_long_form_metric_rows_and_comparison_cards_are_schema_stable() -> None:
    rows = default_metric_rows(
        [
            {
                "model_id": "candidate",
                "variant": "cara_strong",
                "native_cara_output": True,
                "status": "Ready",
            }
        ]
    )
    assert rows
    assert {"model_id", "variant", "evidence_lane", "suite_id", "condition", "metric_id", "value", "status"} <= set(rows[0])
    assert comparison_cards(rows)


def test_adapters_reject_expected_label_copying_and_emit_registry_predictions() -> None:
    with pytest.raises(ValueError, match="copied expected"):
        stable_audio_native_adapter({"source": "expected_label", "policy": {"copied_from_expected": True}})


def test_native_extractor_exceptions_are_not_reported_as_zero_accuracy() -> None:
    pytest.importorskip("torch")
    from benchmark_testing_stable_audio_score import _lane_metrics

    rows = [
        {
            "expected": {
                "cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1",
                "cara_pool_index": 0,
                "cara_pool_family": "Percussion",
                "cara_pool_family_index": 0,
            },
            "native_cara_prediction": {
                "status": "exception",
                "error": "RuntimeError('tap failed')",
                "cara_pool_id": None,
                "cara_pool_index": None,
                "cara_pool_family": None,
                "cara_pool_family_index": None,
            },
        }
    ]
    resolver = {
        "pool_by_index": {"0": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
        "family_by_index": {"0": "Percussion"},
        "pool_to_family_index": {"0": 0},
    }

    metrics = _lane_metrics(rows, "native", resolver)

    assert metrics["status"] == "extractor_failed"
    assert metrics["exception_rate"] == 1.0
    assert "pool_exact_accuracy" not in metrics

    stable = stable_audio_native_adapter(
        {
            "source": "stable_audio_dit_generation_hidden_state_tap",
            "cara_pool_id": "pool-a",
            "cara_pool_family": "fam-a",
            "confidence": 0.8,
            "registry_valid": True,
        }
    )
    assert stable["registry_valid"] is True

    resolver = {
        "pool_by_index": {"0": "pool-a"},
        "family_by_index": {"0": "fam-a"},
        "pool_to_family_index": {"0": "0"},
    }
    musicgen = musicgen_native_adapter(
        {
            "cara_pool_index": 0,
            "cara_pool_family_index": 0,
            "registry_valid": True,
            "checksum_valid": True,
            "hierarchical_valid": True,
        },
        resolver,
        delta_path="checkpoints/musicgen_lm_cara_delta.pt",
    )
    assert musicgen["cara_pool_id"] == "pool-a"
    assert musicgen["source"] == "musicgen_lm_suffix"

    retrieval = retrieval_baseline_adapter(
        "query-a",
        [{"cara_pool_id": "pool-a", "cara_pool_family": "fam-a", "score": 0.7}],
        model_family="musicgen",
    )
    assert retrieval["model_family"] == "musicgen"
    assert retrieval["registry_valid"] is True


def test_stable_audio_native_extractor_aligns_generation_branch_features() -> None:
    torch = pytest.importorskip("torch")
    from benchmark_testing_stable_audio_score import NativeCaraHiddenStateExtractor

    extractor = NativeCaraHiddenStateExtractor.__new__(NativeCaraHiddenStateExtractor)
    extractor.last_feature_alignment = {}
    extractor.tapper = type(
        "FakeTapper",
        (),
        {
            "features": [
                torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
                torch.tensor([[10.0, 20.0]]),
            ]
        },
    )()

    features = extractor._pooled_generation_features(batch_size=1)

    assert features.tolist() == [[2.0, 4.0, 10.0, 20.0]]
    assert extractor.last_feature_alignment["usable_feature_count"] == 2
    assert extractor.last_feature_alignment["adjustments"][0]["mode"] == "mean_over_generation_branches"
    assert extractor.last_feature_alignment["feature_dim_mode"] == "exact"

    extractor.expected_feature_dim = 6
    padded = extractor._pooled_generation_features(batch_size=1)

    assert padded.tolist() == [[2.0, 4.0, 10.0, 20.0, 0.0, 0.0]]
    assert extractor.last_feature_alignment["actual_feature_dim"] == 4
    assert extractor.last_feature_alignment["expected_feature_dim"] == 6
    assert extractor.last_feature_alignment["feature_dim_mode"] == "right_zero_pad_to_checkpoint_width"
