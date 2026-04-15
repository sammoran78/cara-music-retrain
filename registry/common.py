from __future__ import annotations

import secrets
from typing import Iterable

import crcmod

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PAYLOAD_LENGTH = 6
MIN_HAMMING_DISTANCE = 3
DEFAULT_MODALITY = "M"
DEFAULT_VERSION = "A1"

_crc8_func = crcmod.predefined.mkCrcFun("crc-8")


def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        raise ValueError("Hamming distance requires strings of equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def compute_checksum(modality: str, payload: str, version: str) -> str:
    prefix = f"{modality}-{payload}-{version}"
    crc = _crc8_func(prefix.encode("ascii"))
    return f"{crc:02X}"


def build_codeword(modality: str, payload: str, version: str = DEFAULT_VERSION) -> str:
    checksum = compute_checksum(modality, payload, version)
    return f"{modality}-{payload}-{version}-{checksum}"


def generate_distant_payloads(
    count: int,
    min_hamming: int = MIN_HAMMING_DISTANCE,
    charset: str = CHARSET,
    payload_length: int = PAYLOAD_LENGTH,
    max_attempt_multiplier: int = 1000,
) -> list[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if min_hamming < 0:
        raise ValueError("min_hamming must be non-negative")
    payloads: list[str] = []
    attempts = 0
    max_attempts = max(1, count * max_attempt_multiplier)
    while len(payloads) < count and attempts < max_attempts:
        candidate = "".join(secrets.choice(charset) for _ in range(payload_length))
        if all(hamming_distance(candidate, existing) >= min_hamming for existing in payloads):
            payloads.append(candidate)
        attempts += 1
    if len(payloads) < count:
        raise RuntimeError(
            f"Could not generate {count} payloads with Hamming >= {min_hamming}"
        )
    return payloads


def extract_payload(codeword: str) -> str:
    parts = codeword.split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid codeword format: {codeword}")
    return parts[1]


def codeword_parts(codeword: str) -> tuple[str, str, str, str]:
    parts = codeword.split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid codeword format: {codeword}")
    modality, payload, version, checksum = parts
    return modality, payload, version, checksum


def normalise_probability_bins(values: Iterable[float], total: int = 100) -> list[int]:
    rounded = [int(round(value)) for value in values]
    if not rounded:
        return []
    rounded[0] += total - sum(rounded)
    return rounded
