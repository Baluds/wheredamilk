"""
main.py — wheredamilk webcam main loop (voice-controlled).

SPEAK to control the app:
    "find <item>"  → Find mode: scan for item, lock on, give spoken directions
    "what is this" → Wait 2-3 seconds, identify detected object, speak once
    "read"         → OCR largest box, speak the text once
    "stop"         → Cancel current mode, return to idle
    "quit"         → Exit

Press  q  in the OpenCV window to also quit.

Frame pipeline:
    1. Capture 640×480 frame
    2. Skip odd frames (process every 2nd)
    3. YOLOv8 detection → top-2 boxes by confidence
    4. "find" mode → OCR candidates until target locked, then track + guide
    5. "what" mode → wait 2-3 seconds, then OCR largest box and speak once
    6. "read" mode → OCR largest box, speak once, reset to idle
"""

import sys
import cv2

# Load .env file (ELEVEN_API_KEY etc.) before any module imports that need them
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on shell env vars

from vision.yolo import YOLODetector
from vision.ocr import OCRReader
from vision.depth import DepthEstimator
from logic.match import find_best_match
from logic.direction import compute_direction
from logic.tracker import IoUTracker
from utils.tts import TTSEngine
from utils.speech import SpeechListener

# ── Constants ─────────────────────────────────────────────────────────────────
FRAME_W = 640
FRAME_H = 480
TOP_K   = 2     # max candidates for OCR
SKIP_N  = 2     # process every Nth frame

# Colours (BGR)
COL_DEFAULT = (0, 200, 0)
COL_TARGET  = (0, 0, 255)
COL_LOCKED  = (255, 64, 0)
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ───────────────────────────────────────────────────────────────────

def largest_box(boxes: list[dict]) -> dict | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))


def draw_box(frame, box: dict, colour, label: str = ""):
    cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), colour, 2)
    if label:
        cv2.putText(
            frame, label,
            (box["x1"], max(box["y1"] - 6, 14)),
            FONT, 0.55, colour, 2,
        )


def overlay_status(frame, mode: str, direction: str, locked: bool, what_waiting: bool = False):
    status = f"Mode: {mode}"
    if direction:
        status += f"  |  {direction}"
    if locked:
        status += "  [LOCKED]"
    if what_waiting:
        status += "  [ANALYZING...]"
    cv2.putText(frame, status, (10, 24), FONT, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, 'Say: "find <item>" | "what is this" | "read" | "stop" | "quit"',
                (10, FRAME_H - 12), FONT, 0.45, (200, 200, 200), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("▶  wheredamilk starting …")

    # ---- Hardware init ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # ---- Module init ----
    detector  = YOLODetector()
    ocr       = OCRReader()
    depth_est = DepthEstimator()      # MiDaS — graceful no-op if transformers not installed
    tracker   = IoUTracker()
    tts       = TTSEngine()
    listener  = SpeechListener().start()

    # ---- State ----
    mode          = "idle"
    query         = ""
    target_box    = None
    target_locked = False
    direction     = ""
    frame_count   = 0
    what_wait_frames = 0  # Counter for "what" mode delay (2-3 seconds)

    tts.speak("wheredamilk is ready.")

    # ---- Main loop ----
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ── Voice command ─────────────────────────────────────────────────────
        cmd = listener.get_command()
        if cmd:
            action, arg = cmd

            if action == "quit":
                break

            elif action == "stop":
                mode          = "idle"
                target_box    = None
                target_locked = False
                direction     = ""
                tts.reset_throttle()  # Allow new announcements on next mode
                tts.speak_once("Stopped.")

            elif action == "find":
                query         = arg
                mode          = "find"
                target_box    = None
                target_locked = False
                direction     = ""
                tts.reset_throttle()  # Allow new announcements on next mode
                tts.speak_once(f"Looking for {query}.")
                print(f"[main] Find mode: query='{query}'")

            elif action == "what":
                mode             = "what"
                target_box       = None
                target_locked    = False
                what_wait_frames = 0
                tts.reset_throttle()  # Allow new announcements on next mode
                tts.speak_once("Analyzing object. Please hold still.")
                print("[main] What mode: waiting 2-3 seconds before identification.")

            elif action == "read":
                mode          = "read"
                target_box    = None
                target_locked = False
                tts.reset_throttle()  # Allow new announcements on next mode
                tts.speak_once("Reading.")
                print("[main] Read mode.")

        # ── Quit via OpenCV window key ──────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # ── Frame skip ────────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % SKIP_N != 0:
            cv2.imshow("wheredamilk", frame)
            continue

        # ── Detection ─────────────────────────────────────────────────────────
        boxes = detector.detect(frame)

        # Draw all detections faintly
        for b in boxes:
            draw_box(frame, b, COL_DEFAULT, b["cls_name"])

        if mode == "what":
            # Wait 2-3 seconds (approximately 75 raw frames at ~30fps)
            what_wait_frames += 1
            b = largest_box(boxes)
            if b is not None:
                draw_box(frame, b, COL_TARGET, "waiting…")
            
            if what_wait_frames >= 75:  # ~2.5 seconds
                if b is not None:
                    # Perform OCR and identify the object
                    text = ocr.read_text(frame, b)
                    obj_class = b["cls_name"]
                    result = f"{obj_class}"
                    if text:
                        result += f": {text}"
                    print(f"[main] What: {result}")
                    tts.speak_once(result)
                else:
                    tts.speak_once("Nothing detected.")
                mode = "idle"
                what_wait_frames = 0
                tts.reset_throttle()  # Reset throttle for next mode

        elif mode == "find" and query:
            candidates = boxes[:TOP_K]

            if not target_locked:
                # OCR the top candidates to find the target
                texts = [ocr.read_text(frame, b) for b in candidates]
                idx   = find_best_match(texts, query)
                if idx != -1:
                    target_box    = candidates[idx]
                    target_locked = True
                    tts.speak(f"Found {query}!")
                    print(f"[main] Target locked: {target_box}")
                else:
                    # Visual hint on scanning candidates
                    for b in candidates:
                        draw_box(frame, b, COL_TARGET, "scanning…")

            if target_locked and target_box is not None:
                target_box = tracker.update(boxes, target_box)
                depth_val  = depth_est.box_depth(frame, target_box)   # None if model unavailable
                direction  = compute_direction(target_box, FRAME_W, FRAME_H, depth_val)
                src = "MiDaS" if depth_val is not None else "bbox-area"
                print(f"[main] Direction ({src}): {direction}")
                tts.speak(f"{query} — {direction}")
                draw_box(frame, target_box, COL_LOCKED, f"TARGET: {query}")

        elif mode == "read":
            b = largest_box(boxes)
            if b is not None:
                draw_box(frame, b, COL_TARGET, "reading…")
                text = ocr.read_text(frame, b)
                read_result = text if text else "No text found."
                print(f"[main] Read: {read_result}")
                tts.speak_once(read_result)
            else:
                tts.speak_once("Nothing detected.")
            tts.reset_throttle()
            mode = "idle"

        # ── HUD overlay ───────────────────────────────────────────────────────
        overlay_status(frame, mode, direction, target_locked, mode == "what")
        cv2.imshow("wheredamilk", frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    listener.stop()
    tts.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("wheredamilk stopped.")


if __name__ == "__main__":
    main()
