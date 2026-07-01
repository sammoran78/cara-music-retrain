from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CARA_POOL_RE = re.compile(r"^CARA:AUD:(?P<version>\d+):(?P<payload>[A-Z0-9-]+):(?P<check>[A-Z0-9]+)$")

EXACT_POOL = "exact_pool"
REPAIRABLE_POOL = "repairable_pool"
FAMILY_OR_GENRE = "family_or_genre"
UNATTRIBUTABLE = "unattributable"


@dataclass(frozen=True)
class RepairDecision:
    tier: str
    predicted_pool_id: str | None
    resolved_pool_id: str | None
    expected_pool_id: str | None
    predicted_family: str | None
    resolved_family: str | None
    expected_family: str | None
    registry_valid: bool
    repair_method: str | None = None
    repair_distance: int | None = None

    @property
    def pool_exact_correct(self) -> bool:
        return bool(self.expected_pool_id and self.resolved_pool_id == self.expected_pool_id and self.tier == EXACT_POOL)

    @property
    def pool_repaired_correct(self) -> bool:
        return bool(
            self.expected_pool_id
            and self.resolved_pool_id == self.expected_pool_id
            and self.tier in {EXACT_POOL, REPAIRABLE_POOL}
        )

    @property
    def family_correct(self) -> bool:
        return bool(self.expected_family and self.resolved_family == self.expected_family)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "predicted_pool_id": self.predicted_pool_id,
            "resolved_pool_id": self.resolved_pool_id,
            "expected_pool_id": self.expected_pool_id,
            "predicted_family": self.predicted_family,
            "resolved_family": self.resolved_family,
            "expected_family": self.expected_family,
            "registry_valid": self.registry_valid,
            "repair_method": self.repair_method,
            "repair_distance": self.repair_distance,
            "pool_exact_correct": self.pool_exact_correct,
            "pool_repaired_correct": self.pool_repaired_correct,
            "family_correct": self.family_correct,
        }


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _parse_pool_id(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    match = CARA_POOL_RE.match(value.strip().upper())
    return match.groupdict() if match else None


def _payload(value: str | None) -> str | None:
    parsed = _parse_pool_id(value)
    return parsed["payload"] if parsed else None


def _normalise_index_map(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return {str(key): item for key, item in value.items()}


def normalise_resolver(resolver: dict[str, Any]) -> dict[str, Any]:
    if "pool_by_index" in resolver:
        pool_by_index = _normalise_index_map(resolver.get("pool_by_index"))
        family_by_index = _normalise_index_map(resolver.get("family_by_index"))
        pool_to_family_index = _normalise_index_map(resolver.get("pool_to_family_index"))
    else:
        pool_index = resolver.get("pool_index") or {}
        family_index = resolver.get("family_index") or {}
        pool_by_index = {str(index): pool_id for pool_id, index in pool_index.items()}
        family_by_index = {str(index): family for family, index in family_index.items()}
        pool_to_family_index = {}

    pool_to_family_name: dict[str, str] = {}
    for pool_index, pool_id in pool_by_index.items():
        family_index = pool_to_family_index.get(str(pool_index))
        family = family_by_index.get(str(family_index)) if family_index is not None else None
        if family:
            pool_to_family_name[str(pool_id)] = str(family)

    return {
        **resolver,
        "pool_by_index": pool_by_index,
        "family_by_index": family_by_index,
        "pool_to_family_index": pool_to_family_index,
        "pool_to_family_name": {**pool_to_family_name, **(resolver.get("pool_to_family_name") or {})},
    }


def load_resolver(path: str | Path) -> dict[str, Any]:
    return normalise_resolver(json.loads(Path(path).read_text(encoding="utf-8")))


def _family_for_pool(pool_id: str | None, resolver: dict[str, Any]) -> str | None:
    if not pool_id:
        return None
    pool_to_family = resolver.get("pool_to_family_name") or {}
    if pool_id in pool_to_family:
        return str(pool_to_family[pool_id])
    pool_by_index = resolver.get("pool_by_index") or {}
    pool_to_family_index = resolver.get("pool_to_family_index") or {}
    family_by_index = resolver.get("family_by_index") or {}
    for pool_index, candidate in pool_by_index.items():
        if candidate == pool_id:
            family_index = pool_to_family_index.get(str(pool_index))
            if family_index is not None and str(family_index) in family_by_index:
                return str(family_by_index[str(family_index)])
    return None


def _family_from_index(index: int | str | None, resolver: dict[str, Any]) -> str | None:
    if index in (None, ""):
        return None
    return (resolver.get("family_by_index") or {}).get(str(index))


def _pool_from_index(index: int | str | None, resolver: dict[str, Any]) -> str | None:
    if index in (None, ""):
        return None
    return (resolver.get("pool_by_index") or {}).get(str(index))


def _nearest_unique_pool(predicted_pool_id: str, resolver: dict[str, Any], *, max_distance: int) -> tuple[str, int] | None:
    predicted_payload = _payload(predicted_pool_id)
    pool_ids = [str(pool_id) for pool_id in (resolver.get("pool_by_index") or {}).values()]
    if not pool_ids:
        return None

    distances: list[tuple[int, str]] = []
    for pool_id in pool_ids:
        left = predicted_payload or predicted_pool_id
        right = _payload(pool_id) or pool_id
        distances.append((levenshtein_distance(left, right), pool_id))
    distances.sort(key=lambda item: (item[0], item[1]))
    if not distances or distances[0][0] > max_distance:
        return None
    if len(distances) > 1 and distances[1][0] == distances[0][0]:
        return None
    return distances[0][1], distances[0][0]


def resolve_prediction(
    prediction: dict[str, Any],
    expected: dict[str, Any] | None,
    resolver: dict[str, Any],
    *,
    pool_repair_distance: int = 2,
    family_fallback_distance: int = 4,
) -> RepairDecision:
    resolver = normalise_resolver(resolver)
    expected = expected or {}
    predicted_pool_id = prediction.get("cara_pool_id") or _pool_from_index(prediction.get("cara_pool_index"), resolver)
    predicted_pool_id = str(predicted_pool_id) if predicted_pool_id not in (None, "") else None
    expected_pool_id = expected.get("cara_pool_id") or _pool_from_index(expected.get("cara_pool_index"), resolver)
    expected_pool_id = str(expected_pool_id) if expected_pool_id not in (None, "") else None

    predicted_family = (
        prediction.get("cara_pool_family")
        or _family_from_index(prediction.get("cara_pool_family_index"), resolver)
        or _family_for_pool(predicted_pool_id, resolver)
    )
    expected_family = (
        expected.get("cara_pool_family")
        or _family_from_index(expected.get("cara_pool_family_index"), resolver)
        or _family_for_pool(expected_pool_id, resolver)
    )

    known_pool_ids = set(str(pool_id) for pool_id in (resolver.get("pool_by_index") or {}).values())
    registry_valid = bool(predicted_pool_id and predicted_pool_id in known_pool_ids)
    if predicted_pool_id and registry_valid:
        resolved_family = _family_for_pool(predicted_pool_id, resolver) or predicted_family
        return RepairDecision(
            tier=EXACT_POOL,
            predicted_pool_id=predicted_pool_id,
            resolved_pool_id=predicted_pool_id,
            expected_pool_id=expected_pool_id,
            predicted_family=str(predicted_family) if predicted_family else None,
            resolved_family=str(resolved_family) if resolved_family else None,
            expected_family=str(expected_family) if expected_family else None,
            registry_valid=True,
            repair_method="exact_registry_match",
            repair_distance=0,
        )

    if predicted_pool_id:
        parsed = _parse_pool_id(predicted_pool_id)
        if parsed:
            same_payload = [
                pool_id for pool_id in known_pool_ids if _payload(pool_id) == parsed["payload"]
            ]
            if len(same_payload) == 1:
                repaired_pool = same_payload[0]
                return RepairDecision(
                    tier=REPAIRABLE_POOL,
                    predicted_pool_id=predicted_pool_id,
                    resolved_pool_id=repaired_pool,
                    expected_pool_id=expected_pool_id,
                    predicted_family=str(predicted_family) if predicted_family else None,
                    resolved_family=_family_for_pool(repaired_pool, resolver),
                    expected_family=str(expected_family) if expected_family else None,
                    registry_valid=False,
                    repair_method="checksum_repair",
                    repair_distance=0,
                )
        nearest = _nearest_unique_pool(predicted_pool_id, resolver, max_distance=pool_repair_distance)
        if nearest:
            repaired_pool, distance = nearest
            return RepairDecision(
                tier=REPAIRABLE_POOL,
                predicted_pool_id=predicted_pool_id,
                resolved_pool_id=repaired_pool,
                expected_pool_id=expected_pool_id,
                predicted_family=str(predicted_family) if predicted_family else None,
                resolved_family=_family_for_pool(repaired_pool, resolver),
                expected_family=str(expected_family) if expected_family else None,
                registry_valid=False,
                repair_method="unique_payload_edit_distance",
                repair_distance=distance,
            )

    resolved_family = str(predicted_family) if predicted_family else None
    if not resolved_family and predicted_pool_id:
        nearest_family = _nearest_unique_pool(predicted_pool_id, resolver, max_distance=family_fallback_distance)
        if nearest_family:
            repaired_pool, distance = nearest_family
            resolved_family = _family_for_pool(repaired_pool, resolver)
            if resolved_family:
                return RepairDecision(
                    tier=FAMILY_OR_GENRE,
                    predicted_pool_id=predicted_pool_id,
                    resolved_pool_id=None,
                    expected_pool_id=expected_pool_id,
                    predicted_family=None,
                    resolved_family=resolved_family,
                    expected_family=str(expected_family) if expected_family else None,
                    registry_valid=False,
                    repair_method="family_from_nearest_payload",
                    repair_distance=distance,
                )

    if resolved_family:
        return RepairDecision(
            tier=FAMILY_OR_GENRE,
            predicted_pool_id=predicted_pool_id,
            resolved_pool_id=predicted_pool_id if registry_valid else None,
            expected_pool_id=expected_pool_id,
            predicted_family=str(predicted_family) if predicted_family else resolved_family,
            resolved_family=resolved_family,
            expected_family=str(expected_family) if expected_family else None,
            registry_valid=registry_valid,
            repair_method="family_output",
            repair_distance=None,
        )

    return RepairDecision(
        tier=UNATTRIBUTABLE,
        predicted_pool_id=predicted_pool_id,
        resolved_pool_id=None,
        expected_pool_id=expected_pool_id,
        predicted_family=None,
        resolved_family=None,
        expected_family=str(expected_family) if expected_family else None,
        registry_valid=False,
    )


def aggregate_repairability(decisions: list[RepairDecision | dict[str, Any]]) -> dict[str, Any]:
    decision_fields = set(RepairDecision.__dataclass_fields__)
    default_decision = {
        "tier": UNATTRIBUTABLE,
        "predicted_pool_id": None,
        "resolved_pool_id": None,
        "expected_pool_id": None,
        "predicted_family": None,
        "resolved_family": None,
        "expected_family": None,
        "registry_valid": False,
    }
    rows = [
        item
        if isinstance(item, RepairDecision)
        else RepairDecision(
            **{
                **default_decision,
                **{key: value for key, value in item.items() if key in decision_fields},
            }
        )
        for item in decisions
    ]
    total = len(rows)
    tiers = Counter(row.tier for row in rows)
    repair_methods = Counter(row.repair_method or "none" for row in rows)
    labelled = [row for row in rows if row.expected_pool_id or row.expected_family]
    labelled_count = len(labelled)
    exact_correct = sum(1 for row in labelled if row.pool_exact_correct)
    repaired_correct = sum(1 for row in labelled if row.tier == REPAIRABLE_POOL and row.pool_repaired_correct)
    recovered_correct = exact_correct + repaired_correct
    family_correct = sum(
        1
        for row in labelled
        if not row.pool_exact_correct and not (row.tier == REPAIRABLE_POOL and row.pool_repaired_correct) and row.family_correct
    )
    correct_tiers = {
        EXACT_POOL: exact_correct,
        REPAIRABLE_POOL: repaired_correct,
        FAMILY_OR_GENRE: family_correct,
        UNATTRIBUTABLE: max(labelled_count - exact_correct - repaired_correct - family_correct, 0),
    }

    def rate(value: int, denominator: int = total) -> float | None:
        if denominator <= 0:
            return None
        return value / denominator

    return {
        "total": total,
        "correctness_semantics": "expected_label_correctness",
        "labelled_total": labelled_count,
        "tier_counts": {tier: tiers.get(tier, 0) for tier in [EXACT_POOL, REPAIRABLE_POOL, FAMILY_OR_GENRE, UNATTRIBUTABLE]},
        "tier_rates": {tier: rate(tiers.get(tier, 0)) for tier in [EXACT_POOL, REPAIRABLE_POOL, FAMILY_OR_GENRE, UNATTRIBUTABLE]},
        "correct_tier_counts": correct_tiers,
        "correct_tier_rates": {tier: rate(correct_tiers.get(tier, 0), labelled_count) for tier in [EXACT_POOL, REPAIRABLE_POOL, FAMILY_OR_GENRE, UNATTRIBUTABLE]},
        "repair_method_counts": dict(sorted(repair_methods.items())),
        "pool_exact_accuracy": rate(exact_correct, labelled_count),
        "pool_repaired_accuracy": rate(repaired_correct, labelled_count),
        "pool_recovered_accuracy": rate(recovered_correct, labelled_count),
        "family_or_genre_accuracy": rate(family_correct, labelled_count),
        "registry_valid_rate": rate(sum(1 for row in rows if row.registry_valid)),
        "unattributable_rate": rate(correct_tiers.get(UNATTRIBUTABLE, 0), labelled_count),
    }


def repairability_schema() -> dict[str, Any]:
    return {
        "format": "cara_repairability_v1",
        "tiers": [
            {
                "id": EXACT_POOL,
                "label": "Exact pool",
                "description": "Predicted CARA pool-id is registry-valid and exactly matches the expected pool.",
                "counts_as_pool_success": True,
            },
            {
                "id": REPAIRABLE_POOL,
                "label": "Repairable pool",
                "description": "Predicted CARA pool-id is invalid or malformed but repairs uniquely to the expected registry pool.",
                "counts_as_pool_success": True,
            },
            {
                "id": FAMILY_OR_GENRE,
                "label": "Family / genre fallback",
                "description": "Pool could not be resolved, but the prediction is attributable to the correct CARA family or genre.",
                "counts_as_pool_success": False,
            },
            {
                "id": UNATTRIBUTABLE,
                "label": "Unattributable",
                "description": "No valid pool, unique repair, or family-level attribution could be recovered.",
                "counts_as_pool_success": False,
            },
        ],
    }
