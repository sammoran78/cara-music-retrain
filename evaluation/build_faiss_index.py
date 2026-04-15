from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


class SimpleIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query: np.ndarray, k: int):
        scores = np.dot(self.vectors, query.T).squeeze(-1)
        indices = np.argsort(scores)[::-1][:k]
        return scores[indices][None, :], indices[None, :]


def build_index(latents_dir: Path, pool_assignments_path: Path, output_path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(pool_assignments_path.open("r", encoding="utf-8", newline="")))
    vectors = []
    metadata = []
    for row in rows:
        latent_path = latents_dir / f"{row['source_id']}.npy"
        if not latent_path.exists():
            continue
        latent = np.load(latent_path).flatten().astype("float32")
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent = latent / norm
        vectors.append(latent)
        metadata.append({"source_id": row["source_id"], "primary_pool": row["primary_pool"], "codeword": row.get("codeword", ""), "soft_targets": row.get("all_pools", "[]")})
    matrix = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "faiss_index.npy", matrix)
    (output_path / "faiss_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"vectors": len(vectors), "dim": int(matrix.shape[1]) if matrix.size else 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-dir", default="data/latents")
    parser.add_argument("--pool-assignments", default="data/pool_assignments.csv")
    parser.add_argument("--output-dir", default="evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_index(Path(args.latents_dir), Path(args.pool_assignments), Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
