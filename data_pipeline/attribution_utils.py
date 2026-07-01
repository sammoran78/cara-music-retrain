from __future__ import annotations

import csv
import io
import re
import urllib.request
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
    ".m4a",
    ".aac",
    ".wma",
}

SOUND_URL_RE = re.compile(r"freesound\.org/(?:people/[^/]+/sounds/|sounds/)(\d+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\+[a-z0-9]+)?")

LICENSE_DISPLAY_MAP = {
    "cc0": "CC0",
    "cc-by": "CC-BY",
    "cc-by-sa": "CC-BY-SA",
    "cc-by-nc": "CC-BY-NC",
    "cc-by-nc-sa": "CC-BY-NC-SA",
    "cc-by-nd": "CC-BY-ND",
    "sampling+": "CC-Sampling+",
    "unknown": "Unknown",
}


def load_csv_rows(source: str) -> list[dict[str, str]]:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=120) as response:
            text = response.read().decode("utf-8", errors="replace")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def extract_sound_id_from_url(url: str) -> int | None:
    match = SOUND_URL_RE.search(url or "")
    if not match:
        return None
    return int(match.group(1))


def normalize_license(raw_license: str | None) -> str:
    if not raw_license:
        return "unknown"
    cleaned = raw_license.strip().lower().replace("_", "-").replace(" ", "")
    if "creativecommons.org" in cleaned:
        if "/publicdomain/zero/" in cleaned or "/zero/" in cleaned:
            return "cc0"
        if "/licenses/by/" in cleaned and "/by-sa/" not in cleaned and "/by-nc/" not in cleaned and "/by-nd/" not in cleaned:
            return "cc-by"
        if "/licenses/by-sa/" in cleaned:
            return "cc-by-sa"
        if "/licenses/by-nc-sa/" in cleaned:
            return "cc-by-nc-sa"
        if "/licenses/by-nc/" in cleaned:
            return "cc-by-nc"
        if "/licenses/by-nd/" in cleaned:
            return "cc-by-nd"
        if "/licenses/sampling+/" in cleaned or "sampling+/" in cleaned:
            return "sampling+"
    if cleaned in {"cc0", "publicdomain", "public-domain", "creativecommonszero", "zero"}:
        return "cc0"
    if cleaned in {"ccby", "cc-by", "creativecommonsby"}:
        return "cc-by"
    if cleaned in {"ccbysa", "cc-by-sa", "creativecommonsbysa"}:
        return "cc-by-sa"
    if cleaned in {"ccbync", "cc-by-nc", "creativecommonsbync"}:
        return "cc-by-nc"
    if cleaned in {"ccbyncsa", "cc-by-nc-sa", "creativecommonsbyncsa"}:
        return "cc-by-nc-sa"
    if cleaned in {"ccbynd", "cc-by-nd", "creativecommonsbynd"}:
        return "cc-by-nd"
    if cleaned in {"sampling+", "samplingplus", "ccsampling+", "cc-sampling+"}:
        return "sampling+"
    return cleaned or "unknown"


def display_license(normalized_license: str) -> str:
    return LICENSE_DISPLAY_MAP.get(normalized_license, normalized_license.upper() if normalized_license else "Unknown")


def canonical_source_id(raw_id: str | None, url: str | None) -> str:
    raw_value = (raw_id or "").strip()
    if raw_value.isdigit():
        return raw_value
    url_id = extract_sound_id_from_url(url or "")
    return str(url_id) if url_id is not None else ""


def strip_audio_extension(title: str) -> str:
    text = (title or "").strip()
    suffix = Path(text).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return Path(text).stem.strip()
    return text


def infer_audio_extension(title: str, url: str = "") -> str:
    for candidate in [title, Path(url).name]:
        suffix = Path(candidate or "").suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            return suffix
    return ""


def normalise_text(value: str) -> str:
    return " ".join(re.sub(r"[_\-/]+", " ", (value or "").lower()).split())


def tokenize_text(*parts: str) -> list[str]:
    tokens: list[str] = []
    for part in parts:
        tokens.extend(TOKEN_RE.findall(normalise_text(part)))
    return tokens
