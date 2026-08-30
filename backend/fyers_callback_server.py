"""One-shot local HTTP server that catches the FYERS OAuth redirect and
exchanges the auth_code automatically - avoids the manual copy-paste of the
address bar (which browsers truncate / replace with an internal error URL
once the connection fails).

Requires backend/.env's FYERS_REDIRECT_URI to be exactly
"http://127.0.0.1:3000/callback" (also registered on myapi.fyers.in for
this app). Usage:

    cd backend
    python3 fyers_callback_server.py

Then open the printed login URL, log in, and this script does the rest.
"""

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import config
from services import fyers_auth

PORT = 3000
result = {}
done = threading.Event()


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        auth_code = query.get("auth_code", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if auth_code:
            result["auth_code"] = auth_code
            self.wfile.write(b"<h2>Login captured. You can close this tab and go back to the terminal.</h2>")
        else:
            self.wfile.write(b"<h2>No auth_code found in redirect - check the terminal for details.</h2>")

        done.set()

    def log_message(self, format, *args):
        pass  # silence default request logging


def main():
    if not config.FYERS_CONFIGURED:
        print("FYERS_CLIENT_ID / FYERS_SECRET_KEY / FYERS_REDIRECT_URI not set in backend/.env")
        sys.exit(1)

    if config.FYERS_REDIRECT_URI != f"http://127.0.0.1:{PORT}/callback":
        print(f"FYERS_REDIRECT_URI in .env is '{config.FYERS_REDIRECT_URI}', "
              f"but this catcher listens on http://127.0.0.1:{PORT}/callback - they must match exactly "
              f"(and match what's registered on myapi.fyers.in for this app).")
        sys.exit(1)

    login_url = fyers_auth.get_login_url()
    print("\nOpen this URL in your browser and log in to FYERS:\n")
    print(f"  {login_url}\n")
    print(f"Waiting for the redirect on http://127.0.0.1:{PORT}/callback ...\n")

    server = HTTPServer(("127.0.0.1", PORT), CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    done.wait(timeout=300)
    server.shutdown()

    auth_code = result.get("auth_code")
    if not auth_code:
        print("Timed out or no auth_code received. Try again.")
        sys.exit(1)

    token = fyers_auth.exchange_auth_code(auth_code)
    print(f"\nSuccess. Access token saved to {config.FYERS_TOKEN_PATH}")
    print(f"Token (first 12 chars): {token[:12]}...")
    print("\nRestart the backend (uvicorn) so it picks up the new token.")


if __name__ == "__main__":
    main()
