"""
serve.py — Simple HTTP server to serve the WhereDaMilk frontend

Run this to view the frontend:
    python serve.py

Then open: http://localhost:8000
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8000
HANDLER = http.server.SimpleHTTPRequestHandler


def run_server():
    """Start the development server"""
    # Change to the directory where this script is located
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    with socketserver.TCPServer(("", PORT), HANDLER) as httpd:
        print()
        print("=" * 60)
        print("🥛 WhereDaMilk Frontend Server")
        print("=" * 60)
        print(f"✓ Server running at: http://localhost:{PORT}")
        print(f"✓ Serving from: {script_dir}")
        print("✓ Press Ctrl+C to stop the server")
        print("=" * 60)
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✓ Server stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    run_server()
