from evaluation.cara_repairability import (
    EXACT_POOL,
    FAMILY_OR_GENRE,
    REPAIRABLE_POOL,
    aggregate_repairability,
    repairability_schema,
    resolve_prediction,
)


def _resolver():
    return {
        "pool_by_index": {
            "0": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1",
            "1": "CARA:AUD:1:AAAA-BBBB-CCCD:Q2",
            "2": "CARA:AUD:1:DDDD-EEEE-FFFF:X9",
        },
        "family_by_index": {"0": "Percussion/Beats", "1": "Tonal/Orchestral"},
        "pool_to_family_index": {"0": 0, "1": 0, "2": 1},
    }


def test_exact_pool_match():
    decision = resolve_prediction(
        {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
        {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
        _resolver(),
    )
    assert decision.tier == EXACT_POOL
    assert decision.pool_exact_correct
    assert decision.family_correct


def test_checksum_repair_to_known_payload():
    decision = resolve_prediction(
        {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:ZZ"},
        {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
        _resolver(),
    )
    assert decision.tier == REPAIRABLE_POOL
    assert decision.resolved_pool_id == "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"
    assert decision.pool_repaired_correct


def test_registry_valid_wrong_pool_is_mechanical_exact_not_checksum_repair():
    decision = resolve_prediction(
        {"cara_pool_id": "CARA:AUD:1:DDDD-EEEE-FFFF:X9"},
        {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
        _resolver(),
    )

    assert decision.tier == EXACT_POOL
    assert decision.registry_valid
    assert decision.repair_method == "exact_registry_match"
    assert not decision.pool_exact_correct


def test_family_fallback_from_family_index():
    decision = resolve_prediction(
        {"cara_pool_family_index": 1},
        {"cara_pool_id": "CARA:AUD:1:DDDD-EEEE-FFFF:X9"},
        _resolver(),
    )
    assert decision.tier == FAMILY_OR_GENRE
    assert decision.family_correct


def test_aggregate_rates_are_mutually_exclusive():
    decisions = [
        resolve_prediction(
            {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
            {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
            _resolver(),
        ),
        resolve_prediction(
            {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:ZZ"},
            {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1"},
            _resolver(),
        ),
    ]
    summary = aggregate_repairability(decisions)
    assert summary["tier_counts"][EXACT_POOL] == 1
    assert summary["tier_counts"][REPAIRABLE_POOL] == 1
    assert summary["correct_tier_counts"][EXACT_POOL] == 1
    assert summary["correct_tier_counts"][REPAIRABLE_POOL] == 1
    assert summary["pool_exact_accuracy"] == 0.5
    assert summary["pool_repaired_accuracy"] == 0.5
    assert summary["pool_recovered_accuracy"] == 1.0


def test_wrong_unique_repair_is_not_counted_as_correct_repairability():
    decisions = [
        resolve_prediction(
            {"cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:ZZ"},
            {"cara_pool_id": "CARA:AUD:1:DDDD-EEEE-FFFF:X9"},
            _resolver(),
        )
    ]

    summary = aggregate_repairability(decisions)

    assert summary["tier_counts"][REPAIRABLE_POOL] == 1
    assert summary["correct_tier_counts"][REPAIRABLE_POOL] == 0
    assert summary["correct_tier_counts"]["unattributable"] == 1
    assert summary["pool_repaired_accuracy"] == 0.0


def test_aggregate_accepts_benchmark_scoring_rows_with_extra_audit_fields():
    rows = [
        {
            "tier": EXACT_POOL,
            "predicted_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1",
            "resolved_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1",
            "expected_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:Q1",
            "predicted_family": "Percussion/Beats",
            "resolved_family": "Percussion/Beats",
            "expected_family": "Percussion/Beats",
            "registry_valid": True,
            "repair_method": "exact_registry_match",
            "repair_distance": 0,
            "prompt_id": "benchmark-row-001",
            "audio_path": "audio/model/suite/example.wav",
            "confidence": 0.91,
        }
    ]

    summary = aggregate_repairability(rows)

    assert summary["tier_counts"][EXACT_POOL] == 1
    assert summary["pool_exact_accuracy"] == 1.0


def test_schema_includes_unattributable_bucket():
    schema = repairability_schema()
    assert [tier["id"] for tier in schema["tiers"]][-1] == "unattributable"
