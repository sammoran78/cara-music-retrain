from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from registry.common import DEFAULT_MODALITY, DEFAULT_VERSION, build_codeword, compute_checksum, generate_distant_payloads


def generate_codebook(
    pool_definitions: list[dict[str, Any]],
    modality: str = DEFAULT_MODALITY,
    version: str = DEFAULT_VERSION,
) -> list[dict[str, Any]]:
    payloads = generate_distant_payloads(len(pool_definitions))
    codebook: list[dict[str, Any]] = []
    for pool_def, payload in zip(pool_definitions, payloads):
        entry = {
            **pool_def,
            "modality": modality,
            "payload": payload,
            "version": version,
            "checksum": compute_checksum(modality, payload, version),
            "codeword": build_codeword(modality, payload, version),
            "level": pool_def.get("level", "pool"),
        }
        codebook.append(entry)
    return codebook


def load_pool_definitions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "pools" in data:
        data = data["pools"]
    if not isinstance(data, list):
        raise ValueError("Pool definitions must be a list or a dict with a 'pools' key")
    return data


def write_codewords_csv(path: Path, codebook: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for entry in codebook for key in entry.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(codebook)


def write_pools_json(path: Path, codebook: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(codebook, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--codewords-out", default="registry/codewords.csv")
    parser.add_argument("--pools-out", default="registry/pools.json")
    parser.add_argument("--modality", default=DEFAULT_MODALITY)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_definitions = load_pool_definitions(Path(args.input))
    codebook = generate_codebook(pool_definitions, modality=args.modality, version=args.version)
    write_codewords_csv(Path(args.codewords_out), codebook)
    write_pools_json(Path(args.pools_out), codebook)


if __name__ == "__main__":
    main()
