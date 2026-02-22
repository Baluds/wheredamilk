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
from vision.gemini import GeminiAnalyzer
from logic.match import find_best_match
from logic.tracker import IoUTracker
from logic.modes import FindModeHandler, WhatModeHandler, ReadModeHandler, DetailsModeHandler
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


def largest_box_excluding(boxes: list[dict], exclude_classes: list[str] = None) -> dict | None:
    """Pick largest box, excluding certain classes (e.g., person).
    Falls back to largest box if all boxes are in exclude list."""
    if not boxes:
        return None
    if exclude_classes is None:
        exclude_classes = ["person"]
    
    filtered = [b for b in boxes if b["cls_name"] not in exclude_classes]
    if filtered:
        return max(filtered, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    # Fall back to largest box if all excluded
    return max(boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))


def get_box_position(box: dict, frame_w: int, frame_h: int, mirror_horizontal: bool = True) -> str:
    """Determine if bounding box is on left/center/right and top/middle/bottom of screen.
    
    Args:
        box: Bounding box dict with x1, x2, y1, y2
        frame_w: Frame width
        frame_h: Frame height
        mirror_horizontal: If True, flip left/right (for mirrored webcam). Default True.
    
    Returns a phrase like "top-left", "center", "bottom-right" describing the screen position.
    Useful for "what mode" to tell user where the object is visible on-screen.
    """
    # Calculate box centre
    cx = (box["x1"] + box["x2"]) / 2
    cy = (box["y1"] + box["y2"]) / 2
    
    # Horizontal position (1/3 divisions)
    if cx < frame_w / 3:
        horizontal = "left"
    elif cx < 2 * frame_w / 3:
        horizontal = "center"
    else:
        horizontal = "right"
    
    # Mirror horizontal if webcam is mirrored
    if mirror_horizontal:
        if horizontal == "left":
            horizontal = "right"
        elif horizontal == "right":
            horizontal = "left"
        # "center" stays "center"
    
    # Vertical position (1/3 divisions)
    if cy < frame_h / 3:
        vertical = "top"
    elif cy < 2 * frame_h / 3:
        vertical = "middle"
    else:
        vertical = "bottom"
    
    # Combine: if both are centre, just say "center", otherwise say both
    if horizontal == "center" and vertical == "middle":
        return "center"
    elif vertical == "middle":
        return horizontal
    elif horizontal == "center":
        return vertical
    else:
        return f"{vertical}-{horizontal}"



def draw_box(frame, box: dict, colour, label: str = ""):
    cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), colour, 2)
    if label:
        cv2.putText(
            frame, label,
            (box["x1"], max(box["y1"] - 6, 14)),
            FONT, 0.55, colour, 2,
        )


def overlay_status(frame, mode: str, direction: str, locked: bool, what_waiting: bool = False, loading: bool = False):
    status = f"Mode: {mode}"
    if direction:
        status += f"  |  {direction}"
    if locked:
        status += "  [LOCKED]"
    if what_waiting:
        status += "  [ANALYZING...]"
    if loading:
        status += "  [⏳ LOADING...]"
    cv2.putText(frame, status, (10, 24), FONT, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, 'Say: "find <item>" | "what is this" | "read" | "tell me more" | "stop" | "quit"',
                (10, FRAME_H - 12), FONT, 0.45, (200, 200, 200), 1)


def draw_loading_spinner(frame, position=(FRAME_W // 2, FRAME_H // 2), frame_num: int = 0):
    """Draw a simple loading spinner animation."""
    x, y = position
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    # Use frame number to cycle through spinner animation
    spinner_idx = (frame_num // 3) % len(spinner_chars)
    
    # Draw a circle background
    cv2.circle(frame, (x, y), 40, (100, 100, 100), -1)
    cv2.circle(frame, (x, y), 40, (255, 255, 255), 2)
    
    # Draw loading text
    cv2.putText(frame, "🔄 Analyzing with Gemini...", (x - 120, y + 60), FONT, 0.5, (0, 255, 0), 2)


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
    tracker   = IoUTracker()
    tts       = TTSEngine()
    gemini    = GeminiAnalyzer()      # Gemini Vision API for detailed product analysis
    listener  = SpeechListener().start()

    # ---- Mode handlers ----
    find_handler = FindModeHandler(detector, ocr, tracker, tts, gemini)
    what_handler = WhatModeHandler(detector, ocr, tracker, tts, gemini)
    read_handler = ReadModeHandler(detector, ocr, tracker, tts, gemini)
    details_handler = DetailsModeHandler(detector, ocr, tracker, tts, gemini)

    # ---- State ----
    mode          = "idle"
    frame_count   = 0

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
                if mode != "idle":
                    mode_handler = locals().get(f"{mode}_handler")
                    if mode_handler:
                        mode_handler.reset_state()
                mode = "idle"
                tts.reset_throttle()
                tts.speak_once("Stopped.")

            elif action == "find":
                find_handler.start(arg)
                mode = "find"

            elif action == "what":
                what_handler.start()
                mode = "what"

            elif action == "read":
                read_handler.start()
                mode = "read"

            elif action == "details":
                if details_handler.start(frame):
                    mode = "details"
                else:
                    mode = "idle"

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
            is_complete, announcement = what_handler.process(boxes, frame, get_box_position)
            if is_complete:
                tts.speak_once(announcement)
                what_handler.reset_state()
                mode = "idle"

        elif mode == "find":
            target_locked, target_box = find_handler.process(boxes, frame, get_box_position)
            if target_locked and target_box is not None:
                draw_box(frame, target_box, COL_LOCKED, f"TARGET: {find_handler.query}")

        elif mode == "read":
            is_complete, announcement = read_handler.process(boxes, frame, get_box_position)
            if is_complete:
                tts.speak_once(announcement)
                read_handler.reset_state()
                mode = "idle"

        elif mode == "details":
            is_complete, result_text = details_handler.process()
            if is_complete:
                if result_text:
                    tts.speak_once(result_text)
                details_handler.reset_state()
                mode = "idle"
            else:
                # Show loading spinner while processing
                draw_loading_spinner(frame, frame_num=frame_count)

        # ── HUD overlay ───────────────────────────────────────────────────────
        locked = mode == "find" and find_handler.target_locked
        loading = mode == "details" and details_handler.loading
        waiting = mode == "what"
        overlay_status(frame, mode, "", locked, waiting, loading)
        cv2.imshow("wheredamilk", frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    listener.stop()
    tts.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("wheredamilk stopped.")


if __name__ == "__main__":
    main()
