from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from common.env import ENV_PATH

API_BASE = "https://freesound.org/apiv2"
TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"


class FreesoundRateLimitError(RuntimeError):
    pass


class FreesoundAuthError(RuntimeError):
    pass


class FreesoundClient:
    def __init__(self, config: dict[str, Any]):
        freesound_cfg = config.get("freesound", {})
        self.client_id = freesound_cfg.get("client_id", "")
        self.client_secret = freesound_cfg.get("client_secret", "")
        self.access_token = freesound_cfg.get("access_token", "")
        self.refresh_token = freesound_cfg.get("refresh_token", "")
        self.rate_limit_delay = float(freesound_cfg.get("rate_limit_delay_seconds", 0.5))
        self.session = requests.Session()

    def _persist_tokens(self) -> None:
        existing: dict[str, str] = {}
        if ENV_PATH.exists():
            for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                existing[key.strip()] = value.strip()
        existing["FREESOUND_ACCESS_TOKEN"] = self.access_token
        if self.refresh_token:
            existing["FREESOUND_REFRESH_TOKEN"] = self.refresh_token
        ENV_PATH.write_text("\n".join(f"{key}={value}" for key, value in existing.items()) + "\n", encoding="utf-8")

    def _refresh_access_token(self) -> bool:
        if not (self.client_id and self.client_secret and self.refresh_token):
            return False
        response = self.session.post(
            TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
            timeout=120,
        )
        if response.status_code >= 400:
            return False
        payload = response.json()
        self.access_token = str(payload.get("access_token") or "")
        self.refresh_token = str(payload.get("refresh_token") or self.refresh_token or "")
        if not self.access_token:
            return False
        self._persist_tokens()
        return True

    def _auth_headers(self) -> dict[str, str]:
        if not self.access_token:
            return {}
        return {"Authorization": f"Bearer {self.access_token}"}

    def _auth_params(self) -> dict[str, str]:
        if self.access_token:
            return {}
        if self.client_id:
            return {"token": self.client_id}
        return {}

    def _request(self, method: str, url: str, authenticated: bool = False, **kwargs):
        retried_auth = False
        while True:
            headers = dict(kwargs.get("headers", {}))
            params = dict(kwargs.get("params", {}))
            if authenticated:
                headers.update(self._auth_headers())
                params.update(self._auth_params())
            response = None
            for attempt in range(5):
                response = self.session.request(method, url, headers=headers, params=params, timeout=120, **{k: v for k, v in kwargs.items() if k not in {"headers", "params"}})
                if response.status_code == 401 and authenticated:
                    if retried_auth:
                        raise FreesoundAuthError(
                            "Freesound authentication failed. Re-run the OAuth helper to refresh or re-authorize: "
                            "python3 data_pipeline/00_freesound_oauth_helper.py refresh-token --write-env"
                        )
                    if not self._refresh_access_token():
                        raise FreesoundAuthError(
                            "Freesound access token expired and could not be refreshed automatically. Re-run one of: "
                            "python3 data_pipeline/00_freesound_oauth_helper.py refresh-token --write-env "
                            "or re-authorize with authorize-url/exchange-code."
                        )
                    retried_auth = True
                    break
                if response.status_code != 429:
                    break
                retry_after = float(response.headers.get("Retry-After", self.rate_limit_delay * max(4, attempt + 2)))
                time.sleep(retry_after)
            if response is not None and response.status_code == 401 and authenticated and retried_auth:
                continue
            if response is not None and response.status_code == 429:
                raise FreesoundRateLimitError(f"Rate limited after retries for url: {url}")
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response

    def fetch_sound(self, sound_id: int) -> dict[str, Any]:
        return self._request("GET", f"{API_BASE}/sounds/{sound_id}/", authenticated=True).json()

    def fetch_analysis(self, sound_id: int) -> dict[str, Any]:
        return self._request("GET", f"{API_BASE}/sounds/{sound_id}/analysis/", authenticated=True).json()

    def download_original(self, sound_id: int) -> requests.Response:
        if not self.access_token:
            raise RuntimeError("Freesound access token required for original-quality downloads")
        return self._request("GET", f"{API_BASE}/sounds/{sound_id}/download/", authenticated=True, stream=True, allow_redirects=True)


def safe_suffix_from_metadata(metadata: dict[str, Any]) -> str:
    original_filename = str(metadata.get("original_filename") or metadata.get("filename") or metadata.get("name") or "")
    suffix = Path(original_filename).suffix.lower()
    return suffix or ".wav"
