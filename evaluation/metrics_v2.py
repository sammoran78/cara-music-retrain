from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable


def _label(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value not in (None, ""):
        return str(value)
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    value = expected.get(key)
    return str(value) if value not in (None, "") else None


def _prediction_topk(row: dict[str, Any], *, k: int = 3) -> list[str]:
    top_k = row.get("top_k")
    if isinstance(top_k, list):
        values = []
        for item in top_k[:k]:
            if isinstance(item, dict) and item.get("cara_pool_id") not in (None, ""):
                values.append(str(item["cara_pool_id"]))
            elif item not in (None, ""):
                values.append(str(item))
        return values
    value = row.get("predicted_pool_id") or row.get("cara_pool_id")
    return [str(value)] if value not in (None, "") else []


def topk_accuracy(rows: list[dict[str, Any]], *, k: int = 1) -> float | None:
    labelled = [row for row in rows if _label(row, "expected_pool_id") or _label(row, "cara_pool_id")]
    if not labelled:
        return None
    correct = 0
    for row in labelled:
        expected = _label(row, "expected_pool_id") or _label(row, "cara_pool_id")
        if expected in _prediction_topk(row, k=k):
            correct += 1
    return correct / len(labelled)


def balanced_accuracy(rows: list[dict[str, Any]]) -> float | None:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        expected = _label(row, "expected_pool_id") or _label(row, "cara_pool_id")
        if expected:
            by_label[expected].append(row)
    if not by_label:
        return None
    recalls = []
    for expected, items in by_label.items():
        correct = sum(1 for row in items if _prediction_topk(row, k=1)[:1] == [expected])
        recalls.append(correct / len(items))
    return sum(recalls) / len(recalls)


def macro_f1(rows: list[dict[str, Any]]) -> float | None:
    labels = sorted(
        {
            expected
            for row in rows
            for expected in [_label(row, "expected_pool_id") or _label(row, "cara_pool_id")]
            if expected
        }
    )
    if not labels:
        return None
    predicted = [(_label(row, "expected_pool_id") or _label(row, "cara_pool_id"), (_prediction_topk(row, k=1) or [None])[0]) for row in rows]
    scores = []
    for label in labels:
        tp = sum(1 for expected, pred in predicted if expected == label and pred == label)
        fp = sum(1 for expected, pred in predicted if expected != label and pred == label)
        fn = sum(1 for expected, pred in predicted if expected == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def family_accuracy(rows: list[dict[str, Any]]) -> float | None:
    labelled = [row for row in rows if row.get("expected_family") not in (None, "")]
    if not labelled:
        return None
    correct = sum(1 for row in labelled if str(row.get("expected_family")) == str(row.get("predicted_family")))
    return correct / len(labelled)


def ece(rows: list[dict[str, Any]], *, bins: int = 10) -> float | None:
    with_conf = [row for row in rows if isinstance(row.get("confidence"), (int, float))]
    if not with_conf:
        return None
    total = len(with_conf)
    value = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [row for row in with_conf if lower < float(row["confidence"]) <= upper or (index == 0 and float(row["confidence"]) == 0.0)]
        if not bucket:
            continue
        avg_conf = sum(float(row["confidence"]) for row in bucket) / len(bucket)
        avg_acc = sum(1.0 for row in bucket if bool(row.get("exact"))) / len(bucket)
        value += len(bucket) / total * abs(avg_conf - avg_acc)
    return value


def brier_score(rows: list[dict[str, Any]]) -> float | None:
    with_conf = [row for row in rows if isinstance(row.get("confidence"), (int, float))]
    if not with_conf:
        return None
    return sum((float(row["confidence"]) - (1.0 if row.get("exact") else 0.0)) ** 2 for row in with_conf) / len(with_conf)


def tier_rates(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {"repaired_rate": None, "degraded_rate": None, "exception_rate": None}
    tiers = Counter(str(row.get("tier") or row.get("attribution_state") or "unknown") for row in rows)
    total = len(rows)
    return {
        "repaired_rate": tiers.get("repairable_pool", 0) / total + tiers.get("repaired", 0) / total,
        "degraded_rate": tiers.get("family_or_genre", 0) / total + tiers.get("degraded", 0) / total,
        "exception_rate": tiers.get("unattributable", 0) / total + tiers.get("exception", 0) / total,
    }


def bootstrap_ci(values: Iterable[float], *, samples: int = 500, seed: int = 20260609) -> dict[str, float | None]:
    data = [float(value) for value in values]
    if not data:
        return {"mean": None, "ci_low": None, "ci_high": None}
    if len(data) == 1:
        return {"mean": data[0], "ci_low": data[0], "ci_high": data[0]}
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, samples)):
        draw = [rng.choice(data) for _ in data]
        means.append(sum(draw) / len(draw))
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return {"mean": sum(data) / len(data), "ci_low": means[low_index], "ci_high": means[high_index]}


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = tier_rates(rows)
    return {
        "exact_pool_top1": topk_accuracy(rows, k=1),
        "exact_pool_top3": topk_accuracy(rows, k=3),
        "balanced_accuracy": balanced_accuracy(rows),
        "macro_f1": macro_f1(rows),
        "family_accuracy": family_accuracy(rows),
        "ece": ece(rows),
        "brier": brier_score(rows),
        "registry_valid_rate": sum(1 for row in rows if row.get("registry_valid")) / len(rows) if rows else None,
        **rates,
    }
