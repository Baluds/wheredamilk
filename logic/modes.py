"""
logic/modes.py — Modular mode handlers for wheredamilk.

Each mode (find, what, read, details) is encapsulated in its own handler class
for cleaner, more maintainable code.

Modes:
  - find: Search for object by YOLO class or OCR text, lock & track silently
  - what: Identify object class and position (2-3 second delay)
  - read: Extract text from largest non-person box
  - details: Use Gemini Vision for detailed product analysis
"""

from typing import Dict, List, Tuple, Optional


class ModeHandler:
    """Base class for mode handlers."""
    
    def __init__(self, detector, ocr, tracker, tts, gemini):
        self.detector = detector
        self.ocr = ocr
        self.tracker = tracker
        self.tts = tts
        self.gemini = gemini
    
    def reset_state(self):
        """Reset state when exiting this mode. Override in subclasses."""
        pass
    
    def largest_non_person_box(self, boxes: List[Dict]) -> Optional[Dict]:
        """Get largest bounding box, explicitly excluding person class.
        
        If all boxes are persons, falls back to largest box.
        This ensures we never lock onto or interact with people.
        """
        if not boxes:
            return None
        
        # Filter out persons
        non_person_boxes = [b for b in boxes if b["cls_name"] != "person"]
        
        if non_person_boxes:
            # Return largest non-person box by area
            return max(non_person_boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
        
        # Fallback: if all boxes are persons, return largest box anyway
        return max(boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))


class FindModeHandler(ModeHandler):
    """
    Find mode: Search for an object by name.
    
    Strategy:
      1. Try to match query against YOLO class names (fast, reliable)
      2. Fall back to OCR text matching (slower, more flexible)
      3. Once locked, track target silently (no repeated announcements)
    """
    
    def __init__(self, detector, ocr, tracker, tts, gemini):
        super().__init__(detector, ocr, tracker, tts, gemini)
        self.target_box = None
        self.target_locked = False
        self.location_announced = False
        self.query = ""
    
    def start(self, query: str) -> None:
        """Initialize find mode with a query."""
        self.query = query
        self.target_box = None
        self.target_locked = False
        self.location_announced = False
        self.tts.reset_throttle()
        self.tts.speak_once(f"Looking for {query}.")
    
    def process(self, boxes: List[Dict], frame, get_box_position) -> Tuple[bool, Optional[Dict]]:
        """
        Process a frame in find mode.
        
        Returns:
            (target_locked, target_box)
        """
        # Filter out person class to avoid locking onto people
        candidates = [b for b in boxes if b["cls_name"] != "person"][:2]  # TOP_K=2
        if not candidates:
            candidates = boxes[:2]
        
        if not self.target_locked:
            # STEP 1: Try YOLO class match
            from logic.match import find_best_match
            class_names = [b["cls_name"] for b in candidates]
            idx = find_best_match(class_names, self.query)
            
            if idx != -1:
                self.target_box = candidates[idx]
                self.target_locked = True
                frame_h, frame_w = frame.shape[:2]
                position = get_box_position(self.target_box, frame_w, frame_h)
                self.tts.speak_once(f"Found {self.query} on your {position}!")
                self.location_announced = True
                return True, self.target_box
            
            # STEP 2: Fall back to OCR text match
            texts = [self.ocr.read_text(frame, b) for b in candidates]
            idx = find_best_match(texts, self.query)
            
            if idx != -1:
                self.target_box = candidates[idx]
                self.target_locked = True
                frame_h, frame_w = frame.shape[:2]
                position = get_box_position(self.target_box, frame_w, frame_h)
                self.tts.speak_once(f"Found {self.query} on your {position}!")
                self.location_announced = True
                return True, self.target_box
        
        # Keep tracking if locked
        if self.target_locked and self.target_box is not None:
            self.target_box = self.tracker.update(boxes, self.target_box)
            return True, self.target_box
        
        return False, self.target_box
    
    def reset_state(self):
        self.target_box = None
        self.target_locked = False
        self.location_announced = False
        self.query = ""


class WhatModeHandler(ModeHandler):
    """
    What mode: Identify object and its position.
    
    Steps:
      1. Wait 2-3 seconds (lets user stabilize the frame)
      2. OCR all boxes to get full context
      3. Announce object class and position once
      4. Return to idle
    """
    
    def __init__(self, detector, ocr, tracker, tts, gemini):
        super().__init__(detector, ocr, tracker, tts, gemini)
        self.wait_frames = 0
        self.WAIT_THRESHOLD = 40  # ~40 frames at ~30fps = 1.3 seconds
    
    def start(self) -> None:
        """Initialize what mode."""
        self.wait_frames = 0
        self.tts.reset_throttle()
        self.tts.speak_once("Analyzing object. Please hold still.")
    
    def process(self, boxes: List[Dict], frame, get_box_position) -> Tuple[bool, str]:
        """
        Process a frame in what mode.
        
        Returns:
            (is_complete, announcement_text) — is_complete=True means return to idle
        """
        self.wait_frames += 1
        
        if self.wait_frames < self.WAIT_THRESHOLD:
            return False, ""
        
        # Run full OCR
        enriched_boxes = self.ocr.enrich_detections(frame, boxes)
        b = self.largest_non_person_box(boxes)
        
        if b is not None:
            text = b.get("text", self.ocr.read_text(frame, b))
            obj_class = b["cls_name"]
            frame_h, frame_w = frame.shape[:2]
            position = get_box_position(b, frame_w, frame_h)
            
            # Announce only the object (what it is and where it is)
            announcement = f"I see a {obj_class} on your {position}."
            if text:
                announcement += f" It says: {text}"
            
            return True, announcement
        
        return True, "Nothing detected."
    
    def reset_state(self):
        self.wait_frames = 0


class ReadModeHandler(ModeHandler):
    """
    Read mode: Extract and read text from largest non-person box.
    
    Steps:
      1. Immediately find largest object (exclude person)
      2. OCR the text
      3. Announce the text once
      4. Return to idle
    """
    
    def start(self) -> None:
        """Initialize read mode."""
        self.tts.reset_throttle()
        self.tts.speak_once("Reading.")
    
    def process(self, boxes: List[Dict], frame, get_box_position) -> Tuple[bool, str]:
        """
        Process a frame in read mode.
        
        Returns:
            (is_complete, announcement_text)
        """
        b = self.largest_non_person_box(boxes)
        
        if b is not None:
            text = b.get("text", self.ocr.read_text(frame, b))
            obj_class = b["cls_name"]
            frame_h, frame_w = frame.shape[:2]
            position = get_box_position(b, frame_w, frame_h)
            
            # Announce only the text (what it says)
            if text:
                announcement = f"The text reads: {text}"
            else:
                announcement = f"No text found on the {obj_class}."
            
            return True, announcement
        
        return True, "Nothing detected."


class DetailsModeHandler(ModeHandler):
    """
    Details mode: Use Gemini Vision API for detailed product analysis.
    
    Steps:
      1. Capture current frame
      2. Send to Gemini with analysis prompt
      3. Get detailed product information
      4. Read the response aloud
      5. Return to idle
    """
    
    def __init__(self, detector, ocr, tracker, tts, gemini):
        super().__init__(detector, ocr, tracker, tts, gemini)
        self.loading = False
        self.captured_frame = None
    
    def start(self, frame) -> bool:
        """Initialize details mode with frame capture."""
        if not self.gemini.available():
            self.tts.speak_once("Gemini API is not configured. Please set your API key.")
            return False
        
        self.loading = True
        self.captured_frame = frame.copy()
        self.tts.reset_throttle()
        self.tts.speak_once("Analyzing product details. Please wait.")
        return True
    
    def process(self) -> Tuple[bool, Optional[str]]:
        """
        Process Gemini analysis.
        
        Returns:
            (is_complete, result_text) — is_complete=True means return to idle
        """
        if self.loading and self.captured_frame is not None:
            result = self.gemini.identify_product(
                self.captured_frame,
                query="What is the main product in this image? List visible text. "
                      "Give a brief description about the product, brand, ingredients if food, "
                      "and any other useful information in 2-3 lines."
            )
            
            if result.get("success"):
                response_text = result.get("answer", "No information available.")
                self.loading = False
                self.captured_frame = None
                return True, response_text
            else:
                error_msg = f"Analysis failed: {result.get('error', 'Unknown error')}"
                self.loading = False
                self.captured_frame = None
                return True, f"Analysis failed. Please try again."
        
        return False, None
    
    def reset_state(self):
        self.loading = False
        self.captured_frame = None
