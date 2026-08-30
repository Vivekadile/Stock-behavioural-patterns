"""Live LTP (last traded price) from FYERS, layered on top of the stored
dataset - never replacing it. If FYERS isn't configured, isn't logged in,
or a quote call fails for any reason, every function here returns None and
callers (market_service.py, prediction_service.py) fall back to the
stored historical close silently. Live prices only ever change what
`last_close` / `nifty50_close` display - they do not feed the prediction
model, which is trained on the stored dataset.
"""

import time

import config
from services import fyers_auth

_client = None
_client_token = None

_CACHE_TTL_SECONDS = 5
_cache: dict[str, tuple[float, float]] = {}  # fyers_symbol -> (fetched_at, ltp)

# Our dataset's Symbol -> FYERS trading symbol, where they differ.
# Kept in sync with the overrides used elsewhere in the pipeline
# (src/fetch_data.py, src/fetch_fundamentals.py, src/fetch_filing_dates.py).
FYERS_SYMBOL_OVERRIDES = {
    "MM": "M&M",  # NSE ticker is "M&M"; sanitized to "MM" throughout this project
    "TATAMOTORS": "TMPV",  # renamed post-2025 demerger
}

NIFTY50_FYERS_SYMBOL = "NSE:NIFTY50-INDEX"


def to_fyers_symbol(symbol: str) -> str:
    nse_symbol = FYERS_SYMBOL_OVERRIDES.get(symbol, symbol)
    return f"NSE:{nse_symbol}-EQ"


def is_available() -> bool:
    return config.FYERS_CONFIGURED and fyers_auth.load_saved_token() is not None


def _get_client():
    """Recreates the FYERS client if the on-disk token changed (e.g. after
    re-running fyers_login.py) without needing a server restart."""
    global _client, _client_token

    if not config.FYERS_CONFIGURED:
        return None

    token = fyers_auth.load_saved_token()
    if token is None:
        return None

    if _client is None or token != _client_token:
        from fyers_apiv3 import fyersModel

        _client = fyersModel.FyersModel(
            client_id=config.FYERS_CLIENT_ID,
            token=token,
            is_async=False,
            log_path="",
        )
        _client_token = token

    return _client


def _fetch_ltp_uncached(fyers_symbols: list[str]) -> dict[str, float]:
    client = _get_client()
    if client is None:
        return {}

    try:
        response = client.quotes({"symbols": ",".join(fyers_symbols)})
    except Exception:
        return {}

    if not isinstance(response, dict) or response.get("s") != "ok":
        return {}

    prices = {}
    for row in response.get("d", []):
        try:
            prices[row["n"]] = float(row["v"]["lp"])
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def get_ltp_batch(symbols: list[str]) -> dict[str, float]:
    """Returns {our_symbol: live_ltp} for whichever symbols FYERS actually
    returned a price for. Missing/failed symbols are simply absent from
    the result - callers should fall back to stored data for those."""
    if not symbols:
        return {}

    fyers_map = {to_fyers_symbol(s): s for s in symbols}
    now = time.time()

    to_fetch = [
        fs for fs in fyers_map
        if fs not in _cache or now - _cache[fs][0] > _CACHE_TTL_SECONDS
    ]
    if to_fetch:
        fresh = _fetch_ltp_uncached(to_fetch)
        for fs, price in fresh.items():
            _cache[fs] = (now, price)

    return {
        fyers_map[fs]: _cache[fs][1]
        for fs in fyers_map
        if fs in _cache
    }


def get_ltp(symbol: str) -> float | None:
    return get_ltp_batch([symbol]).get(symbol)


def get_nifty_ltp() -> float | None:
    now = time.time()
    if NIFTY50_FYERS_SYMBOL in _cache and now - _cache[NIFTY50_FYERS_SYMBOL][0] <= _CACHE_TTL_SECONDS:
        return _cache[NIFTY50_FYERS_SYMBOL][1]

    fresh = _fetch_ltp_uncached([NIFTY50_FYERS_SYMBOL])
    if NIFTY50_FYERS_SYMBOL not in fresh:
        return None

    _cache[NIFTY50_FYERS_SYMBOL] = (now, fresh[NIFTY50_FYERS_SYMBOL])
    return fresh[NIFTY50_FYERS_SYMBOL]
