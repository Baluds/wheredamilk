"""
app.py — Optional Flask API for wheredamilk.

Exposes two endpoints that trigger the vision pipeline remotely
(e.g. from a mobile app or voice assistant integration).

Endpoints:
    POST /find   body: {"query": "milk"}  → start "find" mode
    POST /read                            → trigger "read" mode once
    GET  /status                          → return current mode + direction

The webcam loop runs in a background thread.  The Flask server shares
state with it via a ThreadSafeState object.
"""

import threading
import time
import cv2
from flask import Flask, request, jsonify

from vision.yolo import YOLODetector
from vision.ocr import OCRReader
from logic.match import find_best_match
from logic.direction import compute_direction
from logic.tracker import IoUTracker
from utils.tts import TTSEngine

# ── Shared state ──────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.mode = "idle"
        self.query = ""
        self.direction = ""
        self.target_locked = False
        self.target_box = None
        self.lock = threading.Lock()

    def set_find(self, query: str):
        with self.lock:
            self.mode = "find"
            self.query = query.lower().strip()
            self.direction = ""
            self.target_locked = False
            self.target_box = None

    def set_read(self):
        with self.lock:
            self.mode = "read"
            self.direction = ""
            self.target_locked = False
            self.target_box = None

    def as_dict(self):
        with self.lock:
            return {
                "mode": self.mode,
                "query": self.query,
                "direction": self.direction,
                "target_locked": self.target_locked,
            }


state = AppState()

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/find", methods=["POST"])
def find():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    state.set_find(query)
    return jsonify({"status": "started", "query": query})


@app.route("/read", methods=["POST"])
def read():
    state.set_read()
    return jsonify({"status": "reading"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(state.as_dict())


# ── Vision loop (background thread) ──────────────────────────────────────────

FRAME_W = 640
FRAME_H = 480
SKIP_N  = 2
TOP_K   = 2


def vision_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[vision] Cannot open webcam — vision thread exiting.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    detector = YOLODetector()
    ocr = OCRReader()
    tracker = IoUTracker()
    tts = TTSEngine()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % SKIP_N != 0:
            continue

        boxes = detector.detect(frame)

        with state.lock:
            mode = state.mode
            query = state.query
            target_box = state.target_box
            target_locked = state.target_locked

        if mode == "find" and query:
            candidates = boxes[:TOP_K]
            if not target_locked:
                texts = [ocr.read_text(frame, b) for b in candidates]
                idx = find_best_match(texts, query)
                if idx != -1:
                    target_box = candidates[idx]
                    target_locked = True
                    tts.speak(f"Found {query}!")

            if target_locked and target_box:
                target_box = tracker.update(boxes, target_box)
                direction = compute_direction(target_box, FRAME_W, FRAME_H)
                tts.speak(f"{query} — {direction}")
                with state.lock:
                    state.direction = direction
                    state.target_box = target_box
                    state.target_locked = True

        elif mode == "read":
            if boxes:
                b = max(boxes, key=lambda x: (x["x2"] - x["x1"]) * (x["y2"] - x["y1"]))
                text = ocr.read_text(frame, b)
                print(f"[vision] Read mode detected text: {text}")
                tts.speak_once(text or "No text found.")
            else:
                tts.speak_once("Nothing detected.")
            with state.lock:
                state.mode = "idle"

    cap.release()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=vision_loop, daemon=True)
    t.start()
    print("wheredamilk Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
