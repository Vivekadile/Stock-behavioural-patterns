"""One-time (well, once-a-day) manual FYERS login.

FYERS access tokens expire daily, so this has to be re-run whenever live
prices stop updating. Usage:

    cd backend
    python3 fyers_login.py

It prints a login URL - open it in a browser, log in with your FYERS
credentials, and you'll be redirected to your configured redirect_uri with
an `auth_code=...` query parameter in the URL. Paste that full redirected
URL (or just the auth_code value) back into this script when prompted.
"""

import sys
from urllib.parse import urlparse, parse_qs

import config
from services import fyers_auth


def extract_auth_code(pasted: str) -> str:
    pasted = pasted.strip()
    if "auth_code=" not in pasted:
        return pasted  # assume the user pasted just the code itself
    query = urlparse(pasted).query
    code = parse_qs(query).get("auth_code")
    if not code:
        raise ValueError("Could not find auth_code in the pasted URL.")
    return code[0]


def main():
    if not config.FYERS_CONFIGURED:
        print("FYERS_CLIENT_ID / FYERS_SECRET_KEY / FYERS_REDIRECT_URI are not set.")
        print("Copy backend/.env.example to backend/.env and fill them in first.")
        sys.exit(1)

    login_url = fyers_auth.get_login_url()
    print("\n1. Open this URL in your browser and log in to FYERS:\n")
    print(f"   {login_url}\n")
    print("2. After login you'll be redirected to your redirect_uri - the page")
    print("   itself may not load (that's fine), just copy the full URL from")
    print("   the address bar.\n")

    pasted = input("3. Paste the redirected URL (or just the auth_code) here: ")
    auth_code = extract_auth_code(pasted)

    token = fyers_auth.exchange_auth_code(auth_code)
    print(f"\nSuccess. Access token saved to {config.FYERS_TOKEN_PATH}")
    print(f"Token (first 12 chars): {token[:12]}...")
    print("\nRestart the backend (uvicorn) so it picks up the new token.")


if __name__ == "__main__":
    main()
