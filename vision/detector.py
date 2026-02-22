"""
vision/detector.py — Unified YOLO + OCR detection pipeline.

Simplifies using YOLO detection and PaddleOCR together in a single step.
"""

from vision.yolo import YOLODetector
from vision.ocr import OCRReader


class UnifiedDetector:
    """Combines YOLOv8 detection + OCR text recognition."""
    
    def __init__(self, yolo_model: str = "yolov8n.pt"):
        self.yolo = YOLODetector(yolo_model)
        self.ocr = OCRReader()
    
    def detect_and_read(self, frame, enriched: bool = True) -> list[dict]:
        """
        Run YOLO detection + OCR reading on a frame.
        
        Args:
            frame: Input frame (numpy array)
            enriched: If True, includes 'text' and 'text_conf' fields
        
        Returns:
            List of detection dicts with structure:
                {
                    'x1', 'y1', 'x2', 'y2',    # bounding box
                    'conf',                     # YOLO confidence
                    'cls_name',                 # class name
                    'text',                     # OCR-detected text (if enriched=True)
                    'text_conf'                 # OCR confidence (if enriched=True)
                }
        """
        boxes = self.yolo.detect(frame)
        if enriched:
            return self.ocr.enrich_detections(frame, boxes)
        return boxes
    
    def detect_and_read_top_k(self, frame, k: int = 2) -> list[dict]:
        """
        Detect objects and read text from top K detections by confidence.
        Useful for performance when you only care about top matches.
        
        Args:
            frame: Input frame
            k: Number of top detections to process with OCR
        
        Returns:
            List of up to K enriched detection dicts
        """
        boxes = self.yolo.detect(frame)
        top_boxes = boxes[:k]
        return self.ocr.enrich_detections(frame, top_boxes)
    
    def get_detections_by_class(self, frame, class_name: str) -> list[dict]:
        """
        Detect objects of a specific class and read text.
        
        Args:
            frame: Input frame
            class_name: Class to filter for (e.g., 'bottle', 'text', 'label')
        
        Returns:
            List of enriched detections for that class only
        """
        boxes = self.yolo.detect(frame)
        filtered = [b for b in boxes if b['cls_name'].lower() == class_name.lower()]
        return self.ocr.enrich_detections(frame, filtered)
