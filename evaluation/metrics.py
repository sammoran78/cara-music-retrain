from __future__ import annotations

from collections import Counter


def parse_attr_string(attr_string: str) -> list[tuple[str, int]]:
    if not attr_string.startswith("ATTR|") or not attr_string.endswith("|END"):
        return []
    body = attr_string[5:-4]
    slots = []
    for part in body.split("|"):
        if "@" not in part:
            continue
        codeword, probability = part.rsplit("@", 1)
        slots.append((codeword, int(probability)))
    return slots


def top1_accuracy(results: list[dict], key: str) -> float:
    if not results:
        return 0.0
    correct = 0
    for row in results:
        slots = parse_attr_string(row.get(key, ""))
        if slots and slots[0][0] == row.get("expected_pool"):
            correct += 1
    return correct / len(results)


def top3_accuracy(results: list[dict], key: str) -> float:
    if not results:
        return 0.0
    correct = 0
    for row in results:
        slots = parse_attr_string(row.get(key, ""))
        if row.get("expected_pool") in [codeword for codeword, _ in slots]:
            correct += 1
    return correct / len(results)


def state_distribution(results: list[dict]) -> dict[str, int]:
    return dict(Counter(row.get("head_state", "unknown") for row in results))


def compute_all_metrics(results: list[dict]) -> dict[str, object]:
    return {
        "head_top1": top1_accuracy(results, "head_attribution"),
        "head_top3": top3_accuracy(results, "head_attribution"),
        "nn_top1": top1_accuracy(results, "nn_attribution"),
        "nn_top3": top3_accuracy(results, "nn_attribution"),
        "keyword_top1": top1_accuracy(results, "keyword_attribution"),
        "prior_top1": top1_accuracy(results, "prior_attribution"),
        "random_top1": top1_accuracy(results, "random_attribution"),
        "head_state_distribution": state_distribution(results),
    }
