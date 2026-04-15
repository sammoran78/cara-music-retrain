from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from registry.common import build_codeword


ROOT_PAYLOAD = "ROOTMS"


def _family_payload(name: str) -> str:
    alnum = "".join(ch for ch in name.upper() if ch.isalnum())
    return f"FAM{alnum[:3]:<3}".replace(" ", "X")[:6]


def build_hierarchy(
    codebook: list[dict[str, Any]],
    modality: str,
    version: str,
    root_name: str = "All licensed music pools",
) -> dict[str, dict[str, Any]]:
    family_entries: dict[str, dict[str, Any]] = {}
    family_children: dict[str, list[str]] = defaultdict(list)
    root_codeword = build_codeword(modality, ROOT_PAYLOAD, version)

    for entry in codebook:
        if entry.get("level") != "pool":
            continue
        family_name = entry.get("genre_tier1") or entry.get("family_name") or "Unclassified"
        family_payload = _family_payload(str(family_name))
        family_codeword = build_codeword(modality, family_payload, version)
        family_entries[family_codeword] = {
            "level": "family",
            "name": str(family_name),
            "parent": root_codeword,
            "children": family_children[family_codeword],
        }
        family_children[family_codeword].append(entry["codeword"])

    hierarchy: dict[str, dict[str, Any]] = {
        root_codeword: {
            "level": "root",
            "name": root_name,
            "children": sorted(family_entries.keys()),
        }
    }

    for family_codeword, family_entry in family_entries.items():
        hierarchy[family_codeword] = family_entry

    for entry in codebook:
        if entry.get("level") != "pool":
            continue
        family_name = entry.get("genre_tier1") or entry.get("family_name") or "Unclassified"
        family_codeword = build_codeword(modality, _family_payload(str(family_name)), version)
        hierarchy[entry["codeword"]] = {
            "level": "pool",
            "name": entry.get("pool_name") or entry.get("name") or entry["codeword"],
            "parent": family_codeword,
            "children": [],
        }

    return hierarchy


def load_codebook(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Codebook input must be a JSON list")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="registry/pools.json")
    parser.add_argument("--output", default="registry/hierarchy.json")
    parser.add_argument("--modality", default="M")
    parser.add_argument("--version", default="A1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    codebook = load_codebook(Path(args.input))
    hierarchy = build_hierarchy(codebook, modality=args.modality, version=args.version)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(hierarchy, handle, indent=2)


if __name__ == "__main__":
    main()
