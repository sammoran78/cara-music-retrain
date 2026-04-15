from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _parse_json_list(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text_hash_features(parts: Iterable[str], dim: int = 64) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for part in parts:
        for token in str(part).lower().replace("_", " ").replace("-", " ").split():
            vector[hash(token) % dim] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def build_file_embedding(row: dict[str, str], dim: int = 64) -> np.ndarray:
    tags = _parse_json_list(row.get("tags", ""))
    moods = _parse_json_list(row.get("mood_tags", ""))
    pools = _parse_json_list(row.get("all_pools", ""))
    numeric_features = np.array(
        [
            _safe_float(row.get("duration_s", "")),
            _safe_float(row.get("bpm", "")),
        ],
        dtype=np.float32,
    )
    text_vector = _text_hash_features(
        [
            *tags,
            *moods,
            *pools,
            row.get("genre_tier1", ""),
            row.get("genre_tier2", ""),
            row.get("key", ""),
            row.get("license", ""),
        ],
        dim=dim - len(numeric_features),
    )
    combined = np.concatenate([text_vector, numeric_features])
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    return combined.astype(np.float32)


def compute_pool_centroids(file_embeddings: dict[str, np.ndarray], pool_assignments: dict[str, str]) -> dict[str, np.ndarray]:
    pool_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for source_id, pool_id in pool_assignments.items():
        embedding = file_embeddings.get(source_id)
        if embedding is not None:
            pool_vectors[pool_id].append(embedding)
    return {pool_id: np.mean(vectors, axis=0) for pool_id, vectors in pool_vectors.items() if vectors}


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    denom = (np.linalg.norm(query) * np.linalg.norm(matrix, axis=1)) + 1e-8
    return np.dot(matrix, query) / denom


def compute_soft_targets(
    file_embedding: np.ndarray,
    pool_centroids: dict[str, np.ndarray],
    primary_pool_id: str,
    num_slots: int = 3,
    primary_boost: float = 2.0,
    temperature: float = 0.1,
) -> list[tuple[str, int]]:
    pool_ids = list(pool_centroids.keys())
    centroid_matrix = np.array([pool_centroids[pid] for pid in pool_ids], dtype=np.float32)
    sims = _cosine_similarity(file_embedding, centroid_matrix)
    if primary_pool_id in pool_ids:
        sims[pool_ids.index(primary_pool_id)] += primary_boost
    exp_sims = np.exp(sims / max(temperature, 1e-6))
    probs = exp_sims / exp_sims.sum()
    top_indices = np.argsort(probs)[-num_slots:][::-1]
    top_probs = np.round(probs[top_indices] / probs[top_indices].sum() * 100).astype(int)
    top_probs[0] += 100 - int(top_probs.sum())
    return [(pool_ids[idx], int(prob)) for idx, prob in zip(top_indices, top_probs)]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_soft_targets(rows: list[dict[str, str]], embedding_dim: int = 64) -> list[dict[str, str]]:
    file_embeddings = {row["source_id"]: build_file_embedding(row, dim=embedding_dim) for row in rows}
    pool_assignments = {row["source_id"]: row["primary_pool"] for row in rows}
    centroids = compute_pool_centroids(file_embeddings, pool_assignments)
    output_rows: list[dict[str, str]] = []
    for row in rows:
        soft_targets = compute_soft_targets(file_embeddings[row["source_id"]], centroids, row["primary_pool"])
        output_rows.append(
            {
                "source": row["source"],
                "source_id": row["source_id"],
                "primary_pool": row["primary_pool"],
                "soft_targets": json.dumps([
                    {"pool_id": pool_id, "probability": probability} for pool_id, probability in soft_targets
                ]),
            }
        )
    return output_rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "source_id", "primary_pool", "soft_targets"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/genre_mapped.csv")
    parser.add_argument("--pool-assignments", default="data/pool_assignments.csv")
    parser.add_argument("--output", default="data/soft_targets.csv")
    parser.add_argument("--embedding-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = {row["source_id"]: row for row in load_rows(Path(args.input))}
    assignment_rows = load_rows(Path(args.pool_assignments))
    merged_rows = [{**source_rows.get(row["source_id"], {}), **row} for row in assignment_rows]
    output_rows = build_soft_targets(merged_rows, embedding_dim=args.embedding_dim)
    write_rows(Path(args.output), output_rows)
    print(json.dumps({"total_rows": len(output_rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
