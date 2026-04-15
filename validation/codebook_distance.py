from __future__ import annotations

from registry.common import hamming_distance


def nearest_codewords(codebook_payloads: list[str], malformed_payload: str, max_distance: int) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    for payload in codebook_payloads:
        if len(payload) != len(malformed_payload):
            continue
        distance = hamming_distance(payload, malformed_payload)
        if distance <= max_distance:
            results.append((payload, distance))
    return sorted(results, key=lambda item: item[1])
