from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.env import ENV_PATH, get_env, load_env_file

AUTHORIZE_URL = "https://freesound.org/apiv2/oauth2/authorize/"
TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"
DEFAULT_CALLBACK_URL = "http://freesound.org/home/app_permissions/permission_granted/"


def build_authorize_url(client_id: str, callback_url: str = DEFAULT_CALLBACK_URL) -> str:
    query = urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": callback_url,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    code: str,
    callback_url: str = DEFAULT_CALLBACK_URL,
) -> dict[str, object]:
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": callback_url,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def refresh_access_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> dict[str, object]:
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def update_env_tokens(env_path: Path, access_token: str, refresh_token: str | None) -> None:
    existing: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            existing[key.strip()] = value.strip()
    existing["FREESOUND_ACCESS_TOKEN"] = access_token
    if refresh_token:
        existing["FREESOUND_REFRESH_TOKEN"] = refresh_token
    lines = [f"{key}={value}" for key, value in existing.items()]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["authorize-url", "exchange-code", "refresh-token"])
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--client-secret", default=None)
    parser.add_argument("--callback-url", default=DEFAULT_CALLBACK_URL)
    parser.add_argument("--code", default=None)
    parser.add_argument("--refresh-token", default=None)
    parser.add_argument("--write-env", action="store_true")
    parser.add_argument("--env-path", default=str(ENV_PATH))
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    client_id = args.client_id or get_env("FREESOUND_CLIENT_ID") or ""
    client_secret = args.client_secret or get_env("FREESOUND_CLIENT_SECRET") or ""
    env_path = Path(args.env_path)

    if not client_id:
        raise RuntimeError("Missing Freesound client ID. Set FREESOUND_CLIENT_ID in .env or pass --client-id.")

    if args.command == "authorize-url":
        print(build_authorize_url(client_id=client_id, callback_url=args.callback_url))
        return

    if not client_secret:
        raise RuntimeError("Missing Freesound client secret. Set FREESOUND_CLIENT_SECRET in .env or pass --client-secret.")

    if args.command == "exchange-code":
        if not args.code:
            raise RuntimeError("Missing authorization code. Pass it with --code.")
        payload = exchange_code_for_tokens(
            client_id=client_id,
            client_secret=client_secret,
            code=args.code,
            callback_url=args.callback_url,
        )
    else:
        refresh_token = args.refresh_token or get_env("FREESOUND_REFRESH_TOKEN") or ""
        if not refresh_token:
            raise RuntimeError("Missing refresh token. Pass --refresh-token or set FREESOUND_REFRESH_TOKEN in .env.")
        payload = refresh_access_token(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
        )

    if args.write_env:
        update_env_tokens(
            env_path=env_path,
            access_token=str(payload.get("access_token") or ""),
            refresh_token=str(payload.get("refresh_token") or "") or None,
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
