"""Loads FYERS credentials from backend/.env (gitignored, never committed).

Live price integration is entirely optional: if these are unset, the app
falls back to the stored historical dataset everywhere - see
services/fyers_service.py.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "")
FYERS_SECRET_KEY = os.getenv("FYERS_SECRET_KEY", "")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "")

FYERS_TOKEN_PATH = BACKEND_DIR / ".fyers_token.json"

FYERS_CONFIGURED = bool(FYERS_CLIENT_ID and FYERS_SECRET_KEY and FYERS_REDIRECT_URI)
