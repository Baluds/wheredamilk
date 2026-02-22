"""
serve.py — HTTP server for WhereDaMilk frontend + launcher for main.py

Run this to view the frontend:
    python serve.py

Then open: http://localhost:8000
Click "Initialize Project" to launch the vision app.
"""

import http.server
import socketserver
import os
import sys
import json
import signal
import subprocess
from pathlib import Path

PORT = 8000
SCRIPT_DIR = Path(__file__).parent
MAIN_PROCESS = None


def _kill_main():
    """Kill main.py process and its entire process group. Returns True if killed."""
    global MAIN_PROCESS
    if MAIN_PROCESS and MAIN_PROCESS.poll() is None:
        try:
            os.killpg(os.getpgid(MAIN_PROCESS.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            MAIN_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(MAIN_PROCESS.pid), signal.SIGKILL)
        print(f"[serve] Stopped main.py (PID {MAIN_PROCESS.pid})")
        MAIN_PROCESS = None
        return True
    MAIN_PROCESS = None
    return False


class WhereDaMilkHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        global MAIN_PROCESS

        if self.path == "/api/start":
            # Kill any leftover process first
            _kill_main()

            try:
                python = sys.executable
                main_py = str(SCRIPT_DIR / "main.py")
                MAIN_PROCESS = subprocess.Popen(
                    [python, main_py],
                    cwd=str(SCRIPT_DIR),
                    preexec_fn=os.setsid,
                )
                print(f"[serve] Started main.py (PID {MAIN_PROCESS.pid})")
                self._json_response(200, {"status": "started", "pid": MAIN_PROCESS.pid})
            except Exception as e:
                print(f"[serve] Failed to start main.py: {e}")
                self._json_response(500, {"status": "error", "message": str(e)})

        elif self.path == "/api/stop":
            if _kill_main():
                self._json_response(200, {"status": "stopped"})
            else:
                self._json_response(200, {"status": "not_running"})

        elif self.path == "/api/status":
            running = MAIN_PROCESS is not None and MAIN_PROCESS.poll() is None
            self._json_response(200, {"running": running})

        else:
            self.send_error(404)

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server():
    os.chdir(SCRIPT_DIR)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), WhereDaMilkHandler) as httpd:
        print()
        print("=" * 60)
        print("  WhereDaMilk Frontend Server")
        print("=" * 60)
        print(f"  Server running at: http://localhost:{PORT}")
        print(f"  Serving from: {SCRIPT_DIR}")
        print("  Press Ctrl+C to stop the server")
        print("=" * 60)
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            _kill_main()
            print("\n  Server stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    run_server()
