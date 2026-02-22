# wheredamilk Architecture Guide

## Overview

wheredamilk is a voice-controlled assistive vision app with 4 distinct modes. The codebase is organized modularly to make each mode easy to understand, test, and extend.

---

## Mode Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Voice Command (Mic)                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                  utils/speech.py
              (parse_command returns action)
                         ↓
        ┌────────────────┴────────────────┐
        ↓                                 ↓
   known command?                    unknown → ignore
        ↓
   ┌────┴─────┬──────┬────────┐
   ↓          ↓      ↓        ↓
 FIND       WHAT   READ    DETAILS
   ↓          ↓      ↓        ↓
 [handler initialization]
   ↓
 Main loop: frame → boxes → handler.process()
   ↓
 Result → TTS → HUD overlay → OpenCV display
   ↓
 Mode complete? → reset_state() → idle
```

---

## Mode Handlers (logic/modes.py)

Each mode is a class with a consistent interface:

### FindModeHandler
**Purpose:** Search for an object by name, lock onto it, track silently.

**Interface:**
```python
handler = FindModeHandler(detector, ocr, tracker, tts, gemini)
handler.start(query="milk")                    # Initialize
target_locked, box = handler.process(boxes, frame, get_box_position)  # Per frame
handler.reset_state()                          # Cleanup
```

**Internal Logic:**
1. STEP 1: Try to match query against YOLO class names
2. STEP 2: If no match, fall back to OCR text matching
3. Once locked: silently track (no repeated announcements)

**State:**
- `target_box`: The box we're tracking
- `target_locked`: Are we locked on?
- `location_announced`: Did we announce the location?
- `query`: What are we looking for?

---

### WhatModeHandler
**Purpose:** Identify object and position (wait 2-3s for user to stabilize).

**Interface:**
```python
handler = WhatModeHandler(detector, ocr, tracker, tts, gemini)
handler.start()                                # Initialize
is_complete, announcement = handler.process(boxes, frame, get_box_position, largest_box_excluding)
handler.reset_state()                          # Cleanup
```

**Returns:** `(is_complete, announcement_text)`
- `is_complete=True` means mode is done, return to idle
- `announcement_text` is what to speak to user

**Internal Logic:**
1. Wait `WAIT_THRESHOLD` frames (~40 frames ≈ 1.3 sec at 30fps)
2. Find largest non-person object
3. Build announcement: "I see a bottle on your left."
4. Return to idle

---

### ReadModeHandler
**Purpose:** Extract text from largest non-person object immediately.

**Interface:**
```python
handler = ReadModeHandler(detector, ocr, tracker, tts, gemini)
handler.start()
is_complete, announcement = handler.process(boxes, frame, get_box_position, largest_box_excluding)
handler.reset_state()
```

**Returns:** `(is_complete, announcement_text)`

**Internal Logic:**
1. Find largest non-person object
2. OCR the text: "COCA-COLA CLASSIC"
3. Announce: "The text reads: COCA-COLA CLASSIC"
4. Return to idle

---

### DetailsModeHandler
**Purpose:** Send frame to Google Gemini Vision, get detailed analysis.

**Interface:**
```python
handler = DetailsModeHandler(detector, ocr, tracker, tts, gemini)
success = handler.start(frame)                 # Capture frame, check API availability
is_complete, result = handler.process()        # Call Gemini (async-friendly)
handler.reset_state()                          # Cleanup
```

**Returns from process():** `(is_complete, result_text)`

**Internal Logic:**
1. Capture current frame
2. Send to Gemini with analysis prompt
3. Wait for response (2-5 seconds)
4. Return detailed product info
5. Announce and return to idle

---

## Main Loop Flow

```python
# 1. Initialize handlers
find_handler = FindModeHandler(...)
what_handler = WhatModeHandler(...)
# etc.

while True:
    frame = capture_frame()
    cmd = listener.get_command()
    
    # 2. Handle voice command
    if cmd:
        action, arg = cmd
        if action == "find":
            find_handler.start(arg)
            mode = "find"
        # etc.
    
    # 3. Process current mode
    boxes = detector.detect(frame)
    
    if mode == "find":
        target_locked, box = find_handler.process(boxes, frame, get_box_position)
        if target_locked:
            draw_box(frame, box, COLOR_LOCKED)
    
    elif mode == "what":
        is_complete, announcement = what_handler.process(boxes, ...)
        if is_complete:
            tts.speak_once(announcement)
            what_handler.reset_state()
            mode = "idle"
    
    # etc.
    
    # 4. Display overlay
    draw_hud(frame, mode, ...)
    cv2.imshow("wheredamilk", frame)
```

---

## Adding a New Mode

To add a new mode (e.g., "compare" mode):

1. **Create handler class** in `logic/modes.py`:
```python
class CompareModeHandler(ModeHandler):
    def start(self, query: str):
        self.query = query
        self.tts.speak_once(f"Comparing {query}...")
    
    def process(self, boxes, frame, ...) -> Tuple[bool, str]:
        # Your logic here
        return is_complete, result_text
    
    def reset_state(self):
        self.query = ""
```

2. **Initialize handler** in `main()`:
```python
compare_handler = CompareModeHandler(detector, ocr, tracker, tts, gemini)
```

3. **Add command parsing** in `utils/speech.py`:
```python
if text.startswith("compare "):
    return ("compare", text[len("compare "):])
```

4. **Add mode dispatch** in main loop:
```python
elif action == "compare":
    compare_handler.start(arg)
    mode = "compare"

elif mode == "compare":
    is_complete, result = compare_handler.process(boxes, ...)
    if is_complete:
        tts.speak_once(result)
        compare_handler.reset_state()
        mode = "idle"
```

That's it! The handler pattern makes adding modes straightforward.

---

## Testing

Since handlers are decoupled, you can test them in isolation:

```python
# test_modes.py
import unittest
from unittest.mock import Mock
from logic.modes import FindModeHandler

class TestFindMode(unittest.TestCase):
    def setUp(self):
        # Mock all dependencies
        self.mock_detector = Mock()
        self.mock_ocr = Mock()
        self.mock_tracker = Mock()
        self.mock_tts = Mock()
        self.mock_gemini = Mock()
        
        self.handler = FindModeHandler(
            self.mock_detector,
            self.mock_ocr,
            self.mock_tracker,
            self.mock_tts,
            self.mock_gemini
        )
    
    def test_find_by_yolo_class(self):
        # Create fake boxes with YOLO class "bottle"
        boxes = [
            {"cls_name": "bottle", "x1": 0, "y1": 0, "x2": 100, "y2": 100}
        ]
        
        # Start find mode
        self.handler.start("bottle")
        
        # Process
        is_locked, box = self.handler.process(boxes, None, lambda b: "center")
        
        # Assert
        self.assertTrue(is_locked)
        self.assertIsNotNone(box)
        self.mock_tts.speak_once.assert_called()
```

---

## Performance Notes

- **YOLOv8n:** ~30ms per frame on CPU
- **EasyOCR:** ~300-500ms per box (heavy)
- **Gemini API:** ~2-5s per request (async-friendly)
- **Frame rate:** Processing every 2nd frame = ~15fps overhead minimal

---

## Dependencies

- `ultralytics` — YOLOv8 detection
- `easyocr` — Text recognition
- `google-generativeai` — Gemini Vision API
- `elevenlabs` — Premium TTS (optional)
- `edge-tts` — Fallback TTS (free)
- `opencv-python` — Vision pipeline
- `SpeechRecognition` — Microphone input

---

## Key Functions

### get_box_position(box, frame_w, frame_h)
Maps bounding box to screen position (9-zone grid).

Returns: "top-left", "center", "bottom-right", etc.

### largest_box_excluding(boxes, exclude_classes=["person"])
Gets largest box, optionally excluding certain classes.

### draw_box(frame, box, colour, label)
Draws rectangle and label on frame.

### overlay_status(frame, mode, ...)
Draws HUD status bar.

---

## Future Enhancements

- [ ] Multi-target comparison ("compare milk and juice")
- [ ] Confidence-gated OCR (skip if YOLO conf < threshold)
- [ ] Re-lock after occlusion
- [ ] Vertical guidance ("look higher")
- [ ] Caching Gemini responses per frame hash
- [ ] iOS/Android companion app
