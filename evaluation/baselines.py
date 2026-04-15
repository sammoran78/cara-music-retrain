from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np


def _normalise_slots(pairs: list[tuple[str, float]], limit: int = 3) -> str:
    top = pairs[:limit]
    if not top:
        return "ATTR|UNKNOWN@100|UNKNOWN@00|UNKNOWN@00|END"
    total = sum(value for _, value in top) or 1.0
    probs = [round(value / total * 100) for _, value in top]
    probs[0] += 100 - sum(probs)
    return "ATTR|" + "|".join(f"{cw}@{prob:02d}" for (cw, _), prob in zip(top, probs)) + "|END"


def nn_attribution(generated_latent, index_vectors, faiss_metadata, k: int = 20):
    if len(index_vectors) == 0:
        return "ATTR|UNKNOWN@100|UNKNOWN@00|UNKNOWN@00|END"
    query = generated_latent.flatten().astype("float32")
    query /= np.linalg.norm(query) + 1e-8
    scores = index_vectors @ query
    top_indices = np.argsort(scores)[::-1][:k]
    pool_votes: dict[str, float] = {}
    for idx in top_indices:
        meta = faiss_metadata[int(idx)]
        codeword = meta.get("codeword") or meta.get("primary_pool") or "UNKNOWN"
        pool_votes[codeword] = pool_votes.get(codeword, 0.0) + float(scores[int(idx)])
    return _normalise_slots(sorted(pool_votes.items(), key=lambda item: -item[1]))


def keyword_attribution(prompt: str, keyword_pool_map: dict[str, dict[str, float]], prior_distribution: dict[str, float] | None = None):
    pool_scores: dict[str, float] = {}
    prompt_lower = prompt.lower()
    for keyword, pool_weights in keyword_pool_map.items():
        if keyword in prompt_lower:
            for codeword, weight in pool_weights.items():
                pool_scores[codeword] = pool_scores.get(codeword, 0.0) + float(weight)
    if not pool_scores:
        return prior_attribution(prior_distribution or {})
    return _normalise_slots(sorted(pool_scores.items(), key=lambda item: -item[1]))


def prior_attribution(pool_size_distribution: dict[str, float]):
    return _normalise_slots(sorted(pool_size_distribution.items(), key=lambda item: -item[1]))


def random_attribution(codewords: list[str]):
    if not codewords:
        return "ATTR|UNKNOWN@100|UNKNOWN@00|UNKNOWN@00|END"
    picks = random.sample(codewords, k=min(3, len(codewords)))
    while len(picks) < 3:
        picks.append(picks[-1])
    return _normalise_slots([(codeword, weight) for codeword, weight in zip(picks, [0.5, 0.3, 0.2])])


def build_keyword_pool_map(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    mapping: dict[str, Counter[str]] = {}
    for row in rows:
        codeword = row.get("codeword") or row.get("primary_pool") or "UNKNOWN"
        tags = json.loads(row.get("original_tags", "[]")) if row.get("original_tags", "").startswith("[") else []
        for tag in tags:
            mapping.setdefault(str(tag).lower(), Counter())[codeword] += 1
    return {keyword: dict(counter) for keyword, counter in mapping.items()}
