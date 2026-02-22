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
from vision.gemini import GeminiAnalyzer
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
    depth_est = DepthEstimator()      # MiDaS — graceful no-op if transformers not installed
    tracker   = IoUTracker()
    tts       = TTSEngine()
    gemini    = GeminiAnalyzer()      # Gemini Vision API for detailed product analysis
    listener  = SpeechListener().start()

    # ---- State ----
    mode          = "idle"
    query         = ""
    target_box    = None
    target_locked = False
    direction     = ""
    frame_count   = 0
    what_wait_frames = 0  # Counter for "what" mode delay (2-3 seconds)
    details_loading = False  # For Gemini analysis (details mode)
    details_frame_capture = None  # Store frame for Gemini analysis

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

            elif action == "details":
                if not gemini.available():
                    tts.speak_once("Gemini API is not configured. Please set your API key.")
                    print("[main] Gemini not available")
                else:
                    mode = "details"
                    details_loading = True
                    details_frame_capture = frame.copy()
                    tts.reset_throttle()
                    tts.speak_once("Analyzing product details. Please wait.")
                    print("[main] Details mode: capturing frame for Gemini analysis")

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
            # Wait 1-2 seconds (approximately 40 raw frames at ~30fps)
            what_wait_frames += 1
            b = largest_box_excluding(boxes)  # Exclude "person" class
            if b is not None:
                draw_box(frame, b, COL_TARGET, "waiting…")
            
            if what_wait_frames >= 40:  
                # Run OCR on ALL detected boxes
                enriched_boxes = ocr.enrich_detections(frame, boxes)
                
                # Print all detected text to terminal
                print(f"\n[main] ════════ WHAT MODE: ALL DETECTIONS ════════")
                all_text_items = []
                for enriched_box in enriched_boxes:
                    cls_name = enriched_box["cls_name"]
                    text = enriched_box["text"]
                    confidence = enriched_box["text_conf"]
                    
                    if text:
                        print(f"  [{cls_name}] Text: '{text}' (OCR conf: {confidence:.2f})")
                        all_text_items.append(f"{cls_name}: {text}")
                    else:
                        print(f"  [{cls_name}] No text detected")
                
                # Speak the main object (largest non-person box)
                if b is not None:
                    text = b.get("text", ocr.read_text(frame, b))
                    obj_class = b["cls_name"]
                    position = get_box_position(b, FRAME_W, FRAME_H)
                    
                    # Build announcement: "cup on your left" or "bottle center: WATER"
                    result = f"{obj_class} (on your {position})"
                    if text:
                        result += f": {text}"
                    
                    print(f"[main] Speaking: {result}")
                    tts.speak_once(result)
                else:
                    tts.speak_once("Nothing detected.")
                
                print(f"[main] ════════════════════════════════════════════\n")
                
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
            b = largest_box_excluding(boxes)  # Exclude "person" class
            
            # Run OCR on ALL detected boxes
            enriched_boxes = ocr.enrich_detections(frame, boxes)
            
            # Print all detected text to terminal
            print(f"\n[main] ════════ READ MODE: ALL DETECTIONS ════════")
            all_text = []
            for enriched_box in enriched_boxes:
                cls_name = enriched_box["cls_name"]
                text = enriched_box["text"]
                confidence = enriched_box["text_conf"]
                
                if text:
                    print(f"  [{cls_name}] '{text}' (OCR conf: {confidence:.2f})")
                    all_text.append(text)
            
            # Speak text from the largest non-person box
            if b is not None:
                draw_box(frame, b, COL_TARGET, "reading…")
                text = b.get("text", ocr.read_text(frame, b))
                obj_class = b["cls_name"]
                position = get_box_position(b, FRAME_W, FRAME_H)
                
                # Build announcement with position
                read_result = text if text else "No text found."
                read_result = f"{obj_class} (on your {position}): {read_result}"
                
                print(f"[main] Speaking (primary object): {read_result}")
                tts.speak_once(read_result)
            else:
                tts.speak_once("Nothing detected.")
            
            print(f"[main] ════════════════════════════════════════════\n")
            
            tts.reset_throttle()
            mode = "idle"

        elif mode == "details":
            # Show loading indicator
            if details_loading and details_frame_capture is not None:
                draw_loading_spinner(frame, frame_num=frame_count)
                
                # Send frame to Gemini (only once)
                if details_frame_capture is not None:
                    print("[main] 🔄 Sending frame to Gemini for analysis...")
                    result = gemini.identify_product(
                        details_frame_capture,
                        query="What is the main product in this image? List visible text. Give a brief description about the product, brand, ingredients if food, and any other useful information in 2-3 lines. "
                    )
                    
                    if result.get("success"):
                        response_text = result.get("answer", "No information available.")
                        print(f"\n[main] ════════ GEMINI ANALYSIS ════════")
                        print(f"{response_text}")
                        print(f"[main] ════════════════════════════════════\n")
                        
                        # Speak the response
                        tts.speak_once(response_text)
                    else:
                        error_msg = f"Analysis failed: {result.get('error', 'Unknown error')}"
                        print(f"[main] ❌ {error_msg}")
                        tts.speak_once("Analysis failed. Please try again.")
                    
                    details_loading = False
                    details_frame_capture = None
                    tts.reset_throttle()
                    mode = "idle"

        # ── HUD overlay ───────────────────────────────────────────────────────
        overlay_status(frame, mode, direction, target_locked, mode == "what", loading=details_loading)
        cv2.imshow("wheredamilk", frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    listener.stop()
    tts.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("wheredamilk stopped.")


if __name__ == "__main__":
    main()
