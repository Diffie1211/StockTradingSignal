import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv

try:
    import streamlit as st
except Exception:
    st = None


load_dotenv()


SUBSCRIPTION_UNIVERSES = [
    "S&P 500",
    "Nasdaq-100",
    "Dow 30",
    "S&P 400 MidCap",
]

UNIVERSE_TO_FLAG = {
    "S&P 500": "scan_sp500",
    "Nasdaq-100": "scan_nasdaq100",
    "Dow 30": "scan_dow30",
    "S&P 400 MidCap": "scan_sp400",
}


def get_secret(name: str, default=None):
    env_value = os.getenv(name)
    if env_value:
        return env_value

    if st is not None:
        try:
            return st.secrets.get(name, default)
        except Exception:
            return default

    return default


def get_supabase_url() -> str | None:
    value = get_secret("SUPABASE_URL")
    if value:
        return value.rstrip("/")
    return None


def get_supabase_service_role_key() -> str | None:
    return get_secret("SUPABASE_SERVICE_ROLE_KEY")


def get_app_public_url() -> str:
    return get_secret("APP_PUBLIC_URL", "http://localhost:8501").rstrip("/")


def subscription_config_ready() -> bool:
    return bool(get_supabase_url() and get_supabase_service_role_key())


def require_supabase_config() -> tuple[str, str]:
    supabase_url = get_supabase_url()
    service_key = get_supabase_service_role_key()

    if not supabase_url or not service_key:
        raise RuntimeError(
            "Missing Supabase config. Add SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY to .env or Streamlit Secrets."
        )

    return supabase_url, service_key


def supabase_headers(prefer: str | None = None) -> dict[str, str]:
    _, service_key = require_supabase_config()

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


def table_url() -> str:
    supabase_url, _ = require_supabase_config()
    return f"{supabase_url}/rest/v1/subscriptions"


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()

    if not email:
        raise ValueError("Email address is required.")

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise ValueError("Please enter a valid email address.")

    return email


def flags_from_selected_universes(selected_universes: list[str]) -> dict[str, bool]:
    selected_set = set(selected_universes)

    flags = {}

    for universe, flag_name in UNIVERSE_TO_FLAG.items():
        flags[flag_name] = universe in selected_set

    return flags


def flags_to_selected_universes(subscription: dict[str, Any]) -> list[str]:
    selected = []

    for universe, flag_name in UNIVERSE_TO_FLAG.items():
        if bool(subscription.get(flag_name)):
            selected.append(universe)

    return selected


def selected_universes_from_subscription(subscription: dict[str, Any]) -> list[str]:
    return flags_to_selected_universes(subscription)


def _get_one_by_filter(filter_name: str, filter_value: str) -> dict[str, Any] | None:
    response = requests.get(
        table_url(),
        headers=supabase_headers(),
        params={
            "select": "*",
            filter_name: f"eq.{filter_value}",
            "limit": "1",
        },
        timeout=30,
    )

    response.raise_for_status()
    rows = response.json()

    if not rows:
        return None

    return rows[0]


def get_subscription_by_email(email: str) -> dict[str, Any] | None:
    email = normalize_email(email)
    return _get_one_by_filter("email", email)


def get_subscription_by_token(token: str) -> dict[str, Any] | None:
    token = (token or "").strip()

    if not token:
        return None

    return _get_one_by_filter("token", token)


def upsert_subscription(email: str, selected_universes: list[str]) -> dict[str, Any]:
    email = normalize_email(email)

    if not selected_universes:
        raise ValueError("Please choose at least one scan list.")

    existing = get_subscription_by_email(email)
    now = datetime.now(timezone.utc).isoformat()
    flags = flags_from_selected_universes(selected_universes)

    if existing:
        token = existing["token"]

        payload = {
            "is_active": True,
            "updated_at": now,
            **flags,
        }

        response = requests.patch(
            table_url(),
            headers=supabase_headers(prefer="return=representation"),
            params={"email": f"eq.{email}"},
            json=payload,
            timeout=30,
        )
    else:
        token = secrets.token_urlsafe(32)

        payload = {
            "email": email,
            "token": token,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            **flags,
        }

        response = requests.post(
            table_url(),
            headers=supabase_headers(prefer="return=representation"),
            json=payload,
            timeout=30,
        )

    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise RuntimeError("Supabase did not return the saved subscription.")

    return rows[0]


def update_subscription_by_token(token: str, selected_universes: list[str]) -> dict[str, Any]:
    token = (token or "").strip()

    if not token:
        raise ValueError("Missing subscription token.")

    if not selected_universes:
        raise ValueError("Please choose at least one scan list.")

    existing = get_subscription_by_token(token)

    if not existing:
        raise ValueError("Subscription link is invalid.")

    now = datetime.now(timezone.utc).isoformat()
    flags = flags_from_selected_universes(selected_universes)

    payload = {
        "is_active": True,
        "updated_at": now,
        **flags,
    }

    response = requests.patch(
        table_url(),
        headers=supabase_headers(prefer="return=representation"),
        params={"token": f"eq.{token}"},
        json=payload,
        timeout=30,
    )

    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise RuntimeError("Supabase did not return the updated subscription.")

    return rows[0]


def unsubscribe_by_token(token: str) -> None:
    token = (token or "").strip()

    if not token:
        raise ValueError("Missing subscription token.")

    payload = {
        "is_active": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    response = requests.patch(
        table_url(),
        headers=supabase_headers(),
        params={"token": f"eq.{token}"},
        json=payload,
        timeout=30,
    )

    response.raise_for_status()


def unsubscribe_by_email(email: str) -> None:
    email = normalize_email(email)

    payload = {
        "is_active": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    response = requests.patch(
        table_url(),
        headers=supabase_headers(),
        params={"email": f"eq.{email}"},
        json=payload,
        timeout=30,
    )

    response.raise_for_status()


def get_active_subscriptions() -> list[dict[str, Any]]:
    response = requests.get(
        table_url(),
        headers=supabase_headers(),
        params={
            "select": "*",
            "is_active": "eq.true",
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def format_manage_link(token: str) -> str:
    return f"{get_app_public_url()}/?token={token}"
