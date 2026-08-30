"""FYERS login/token handling.

FYERS uses OAuth-style auth: a human has to log in via browser once to get
an `auth_code`, which is exchanged for an `access_token`. That token is
valid for a single trading day (FYERS invalidates it every day around
market open) - there is no way around this, it's how FYERS' API works.

Run `python3 fyers_login.py` from backend/ whenever the token expires.
"""

import json

from fyers_apiv3 import fyersModel

import config


def get_login_url() -> str:
    session = fyersModel.SessionModel(
        client_id=config.FYERS_CLIENT_ID,
        secret_key=config.FYERS_SECRET_KEY,
        redirect_uri=config.FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )
    return session.generate_authcode()


def exchange_auth_code(auth_code: str) -> str:
    """Exchanges a one-time auth_code (from the redirect URL after login)
    for an access_token, and persists it to disk."""
    session = fyersModel.SessionModel(
        client_id=config.FYERS_CLIENT_ID,
        secret_key=config.FYERS_SECRET_KEY,
        redirect_uri=config.FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(auth_code)
    response = session.generate_token()

    if "access_token" not in response:
        raise RuntimeError(f"FYERS token exchange failed: {response}")

    access_token = response["access_token"]
    config.FYERS_TOKEN_PATH.write_text(json.dumps({"access_token": access_token}))
    return access_token


def load_saved_token() -> str | None:
    if not config.FYERS_TOKEN_PATH.exists():
        return None
    try:
        return json.loads(config.FYERS_TOKEN_PATH.read_text()).get("access_token")
    except (json.JSONDecodeError, OSError):
        return None
