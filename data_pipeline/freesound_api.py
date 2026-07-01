from __future__ import annotations

import json
import os
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
        self.requests_per_minute = int(freesound_cfg.get("requests_per_minute", 50))
        self.requests_per_day = int(freesound_cfg.get("requests_per_day", 1800))
        self.api_usage_path = Path(freesound_cfg.get("api_usage_path", "data/freesound_api_usage.json"))
        self.session = requests.Session()

    def _today_key(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime())

    def _load_usage(self) -> dict[str, Any]:
        if not self.api_usage_path.exists():
            return {"day": self._today_key(), "daily_count": 0, "minute_window_started_at": 0.0, "minute_count": 0}
        try:
            payload = json.loads(self.api_usage_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        return {
            "day": str(payload.get("day") or self._today_key()),
            "daily_count": int(payload.get("daily_count", 0) or 0),
            "minute_window_started_at": float(payload.get("minute_window_started_at", 0.0) or 0.0),
            "minute_count": int(payload.get("minute_count", 0) or 0),
        }

    def _write_usage(self, usage: dict[str, Any]) -> None:
        self.api_usage_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_usage_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")

    def _throttle(self) -> None:
        if self.requests_per_minute <= 0 and self.requests_per_day <= 0:
            return
        while True:
            usage = self._load_usage()
            now = time.time()
            today = self._today_key()
            if usage["day"] != today:
                usage = {"day": today, "daily_count": 0, "minute_window_started_at": now, "minute_count": 0}
            if now - usage["minute_window_started_at"] >= 60:
                usage["minute_window_started_at"] = now
                usage["minute_count"] = 0

            minute_blocked = self.requests_per_minute > 0 and usage["minute_count"] >= self.requests_per_minute
            day_blocked = self.requests_per_day > 0 and usage["daily_count"] >= self.requests_per_day
            if not minute_blocked and not day_blocked:
                return

            if day_blocked:
                tomorrow = time.mktime(time.strptime(f"{today} 23:59:59", "%Y-%m-%d %H:%M:%S")) + 1
                sleep_for = max(tomorrow - now, self.rate_limit_delay)
            else:
                sleep_for = max((usage["minute_window_started_at"] + 60) - now, self.rate_limit_delay)
            time.sleep(sleep_for)

    def _record_request(self) -> None:
        usage = self._load_usage()
        now = time.time()
        today = self._today_key()
        if usage["day"] != today:
            usage = {"day": today, "daily_count": 0, "minute_window_started_at": now, "minute_count": 0}
        if now - usage["minute_window_started_at"] >= 60:
            usage["minute_window_started_at"] = now
            usage["minute_count"] = 0
        usage["daily_count"] += 1
        usage["minute_count"] += 1
        self._write_usage(usage)

    def get_usage_snapshot(self) -> dict[str, Any]:
        usage = self._load_usage()
        return {
            "day": usage["day"],
            "daily_count": usage["daily_count"],
            "minute_count": usage["minute_count"],
            "requests_per_day": self.requests_per_day,
            "requests_per_minute": self.requests_per_minute,
        }

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
        os.environ["FREESOUND_ACCESS_TOKEN"] = self.access_token
        if self.refresh_token:
            os.environ["FREESOUND_REFRESH_TOKEN"] = self.refresh_token

    def _reload_tokens_from_env_file(self) -> None:
        if not ENV_PATH.exists():
            return
        values: dict[str, str] = {}
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        self.client_id = values.get("FREESOUND_CLIENT_ID") or self.client_id
        self.client_secret = values.get("FREESOUND_CLIENT_SECRET") or self.client_secret
        self.access_token = values.get("FREESOUND_ACCESS_TOKEN") or self.access_token
        self.refresh_token = values.get("FREESOUND_REFRESH_TOKEN") or self.refresh_token

    def _refresh_access_token(self) -> tuple[bool, str]:
        self._reload_tokens_from_env_file()
        if not (self.client_id and self.client_secret and self.refresh_token):
            missing = []
            if not self.client_id:
                missing.append("FREESOUND_CLIENT_ID")
            if not self.client_secret:
                missing.append("FREESOUND_CLIENT_SECRET")
            if not self.refresh_token:
                missing.append("FREESOUND_REFRESH_TOKEN")
            return False, f"missing {', '.join(missing)}"
        try:
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
        except requests.RequestException as exc:
            return False, f"refresh request failed: {exc}"
        if response.status_code >= 400:
            body = response.text.strip().replace("\n", " ")[:500]
            return False, f"refresh HTTP {response.status_code}: {body}"
        try:
            payload = response.json()
        except ValueError as exc:
            return False, f"refresh response was not JSON: {exc}"
        self.access_token = str(payload.get("access_token") or "")
        self.refresh_token = str(payload.get("refresh_token") or self.refresh_token or "")
        if not self.access_token:
            return False, "refresh response did not include access_token"
        self._persist_tokens()
        return True, "refreshed"

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
                self._throttle()
                response = self.session.request(method, url, headers=headers, params=params, timeout=120, **{k: v for k, v in kwargs.items() if k not in {"headers", "params"}})
                self._record_request()
                if response.status_code == 401 and authenticated:
                    if retried_auth:
                        raise FreesoundAuthError(
                            "Freesound authentication failed. Re-run the OAuth helper to refresh or re-authorize: "
                            "python3 data_pipeline/00_freesound_oauth_helper.py refresh-token --write-env"
                        )
                    refreshed, refresh_reason = self._refresh_access_token()
                    if not refreshed:
                        raise FreesoundAuthError(
                            "Freesound access token expired and could not be refreshed automatically "
                            f"({refresh_reason}). Re-run one of: "
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

    def fetch_sound(self, sound_id: int, fields: list[str] | None = None) -> dict[str, Any]:
        params = {}
        if fields:
            params["fields"] = ",".join(fields)
        return self._request("GET", f"{API_BASE}/sounds/{sound_id}/", authenticated=True, params=params).json()

    def fetch_sounds_bulk(self, sound_ids: list[int], fields: list[str] | None = None) -> dict[int, dict[str, Any]]:
        if not sound_ids:
            return {}
        indexed: dict[int, dict[str, Any]] = {}
        field_list = list(fields) if fields else None
        if field_list and "id" not in field_list:
            field_list = ["id", *field_list]
        page_size = max(len(sound_ids), 1)
        params = {
            "filter": "id:(" + " OR ".join(str(sid) for sid in sound_ids) + ")",
            "page_size": str(min(page_size, 150)),
        }
        if field_list:
            params["fields"] = ",".join(field_list)
        payload = self._request("GET", f"{API_BASE}/search/text/", authenticated=True, params=params).json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        for item in results:
            try:
                indexed[int(item.get("id"))] = item
            except Exception:
                continue
        return indexed

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
