"""Persist and manage per-client requirements profiles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.services.self_correction.config import DEFAULT_CLIENT_ID

STORE = Path(__file__).resolve().parents[2] / "store"
PROFILES_PATH = STORE / "requirements_profiles.json"

DEFAULT_PROFILE_FIELDS = {
    "tone_style": "",
    "detail_level": "",
    "must_include": "",
    "must_avoid": "",
    "format_expectations": "",
    "domain_constraints": "",
    "extra": {},
}


def ensure_profiles_file() -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    if not PROFILES_PATH.exists():
        PROFILES_PATH.write_text("{}")


def _read_profiles() -> dict:
    ensure_profiles_file()
    try:
        data = json.loads(PROFILES_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_profiles(data: dict) -> None:
    ensure_profiles_file()
    PROFILES_PATH.write_text(json.dumps(data, indent=2))


def empty_profile(client_id: str = DEFAULT_CLIENT_ID) -> dict:
    return {
        "client_id": client_id,
        **DEFAULT_PROFILE_FIELDS,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def get_requirements_profile(client_id: str) -> dict | None:
    profiles = _read_profiles()
    return profiles.get(client_id)


def save_requirements_profile(client_id: str, profile: dict) -> dict:
    profiles = _read_profiles()
    saved = {
        **DEFAULT_PROFILE_FIELDS,
        **profile,
        "client_id": client_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    profiles[client_id] = saved
    _write_profiles(profiles)
    return saved


def list_requirements_profiles() -> dict[str, dict]:
    return _read_profiles()


def resolve_requirements_profile(client_id: str, allow_empty: bool = True) -> dict:
    profile = get_requirements_profile(client_id)
    if profile:
        return profile
    if allow_empty:
        return empty_profile(client_id)
    raise ValueError(f"No requirements profile for client_id={client_id}")


def profile_to_prompt_context(profile: dict) -> str:
    parts = []
    mapping = [
        ("Tone / style", profile.get("tone_style")),
        ("Detail level", profile.get("detail_level")),
        ("Must include", profile.get("must_include")),
        ("Must avoid", profile.get("must_avoid")),
        ("Format expectations", profile.get("format_expectations")),
        ("Domain constraints", profile.get("domain_constraints")),
    ]
    for label, value in mapping:
        if value and str(value).strip():
            parts.append(f"- {label}: {value}")
    extra = profile.get("extra") or {}
    for key, value in extra.items():
        if value and str(value).strip():
            parts.append(f"- {key}: {value}")
    return "\n".join(parts) if parts else "(No specific requirements configured)"
