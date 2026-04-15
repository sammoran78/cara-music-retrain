from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from registry.common import codeword_parts, compute_checksum, extract_payload, hamming_distance


class CARACodebook:
    def __init__(self, codebook_path: str | Path, hierarchy_path: str | Path):
        self.codewords: dict[str, dict[str, Any]] = {}
        self.hierarchy: dict[str, dict[str, Any]] = {}
        self.idx_to_codeword: dict[int, str] = {}
        self.codeword_to_idx: dict[str, int] = {}
        self._load_codebook(Path(codebook_path))
        self._load_hierarchy(Path(hierarchy_path))

    def _load_codebook(self, path: Path) -> None:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            with path.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError("Codebook data must be a list")
        for idx, entry in enumerate(rows):
            codeword = entry["codeword"]
            payload = extract_payload(codeword)
            self.codewords[payload] = entry
            self.idx_to_codeword[idx] = codeword
            self.codeword_to_idx[codeword] = idx

    def _load_hierarchy(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            self.hierarchy = json.load(handle)

    def is_registered(self, codeword: str) -> bool:
        try:
            payload = extract_payload(codeword)
        except ValueError:
            return False
        return payload in self.codewords

    def is_registered_payload(self, codeword: str) -> bool:
        try:
            payload = extract_payload(codeword)
        except ValueError:
            return False
        return payload in self.codewords

    def checksum_valid(self, codeword: str) -> bool:
        try:
            modality, payload, version, claimed_checksum = codeword_parts(codeword)
        except ValueError:
            return False
        expected = compute_checksum(modality, payload, version)
        return claimed_checksum == expected

    def nearest_codeword(self, malformed: str) -> tuple[str | None, int]:
        try:
            mal_payload = extract_payload(malformed)
        except ValueError:
            return None, 999999
        best_dist = 999999
        best_cw: str | None = None
        for payload, entry in self.codewords.items():
            if len(mal_payload) != len(payload):
                continue
            dist = hamming_distance(mal_payload, payload)
            if dist < best_dist:
                best_dist = dist
                best_cw = entry["codeword"]
        return best_cw, best_dist

    def all_within_distance(self, malformed: str, max_dist: int) -> list[str]:
        try:
            mal_payload = extract_payload(malformed)
        except ValueError:
            return []
        results: list[str] = []
        for payload, entry in self.codewords.items():
            if len(mal_payload) != len(payload):
                continue
            if hamming_distance(mal_payload, payload) <= max_dist:
                results.append(entry["codeword"])
        return results

    def get_parent_codeword(self, codeword: str) -> str | None:
        if codeword in self.hierarchy:
            return self.hierarchy[codeword].get("parent")
        nearest, _dist = self.nearest_codeword(codeword)
        if nearest and nearest in self.hierarchy:
            return self.hierarchy[nearest].get("parent")
        return None

    def get_root_codeword(self, modality: str) -> str | None:
        for codeword, entry in self.hierarchy.items():
            if entry.get("level") == "root" and codeword.startswith(f"{modality}-"):
                return codeword
        return None

    def recompute_checksum(self, codeword: str) -> str:
        modality, payload, version, _checksum = codeword_parts(codeword)
        new_checksum = compute_checksum(modality, payload, version)
        return f"{modality}-{payload}-{version}-{new_checksum}"
