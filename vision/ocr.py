"""
vision/ocr.py — PaddleOCR wrapper with YOLO integration.

Reads text from cropped regions and enriches detection boxes with OCR text.
"""

import numpy as np
from paddleocr import PaddleOCR


class OCRReader:
    def __init__(self):
        # use_angle_cls helps with rotated text; lang='en' for English
        self._ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def read_text(self, frame, box: dict) -> str:
        """
        Crop `frame` to the region defined by `box` and run OCR.
        Returns joined string of all detected text lines, or ''.
        """
        x1 = max(0, box["x1"])
        y1 = max(0, box["y1"])
        x2 = min(frame.shape[1], box["x2"])
        y2 = min(frame.shape[0], box["y2"])

        if x2 <= x1 or y2 <= y1:
            return ""

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        result = self._ocr.predict(crop)
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    try:
                        ocr_data = line[1]
                        if isinstance(ocr_data, (list, tuple)) and len(ocr_data) >= 2:
                            text, conf = ocr_data[0], ocr_data[1]
                            if conf > 0.5:
                                texts.append(text)
                    except (TypeError, IndexError, ValueError):
                        pass
        return " ".join(texts).strip()

    def read_text_with_confidence(self, frame, box: dict) -> tuple[str, float]:
        """
        Crop `frame` to the region defined by `box` and run OCR.
        Returns tuple of (text_string, average_confidence) or ('', 0.0).
        Useful for filtering low-confidence OCR results.
        """
        x1 = max(0, box["x1"])
        y1 = max(0, box["y1"])
        x2 = min(frame.shape[1], box["x2"])
        y2 = min(frame.shape[0], box["y2"])

        if x2 <= x1 or y2 <= y1:
            return "", 0.0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0

        result = self._ocr.predict(crop)
        texts = []
        confidences = []
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    try:
                        ocr_data = line[1]
                        if isinstance(ocr_data, (list, tuple)) and len(ocr_data) >= 2:
                            text, conf = ocr_data[0], ocr_data[1]
                            if conf > 0.5:
                                texts.append(text)
                                confidences.append(conf)
                    except (TypeError, IndexError, ValueError):
                        pass
        
        avg_conf = np.mean(confidences) if confidences else 0.0
        return " ".join(texts).strip(), float(avg_conf)

    def enrich_detections(self, frame, boxes: list[dict]) -> list[dict]:
        """
        Process multiple YOLO detection boxes with OCR simultaneously.
        Returns enriched boxes with 'text' and 'text_conf' fields.
        
        Example:
            boxes = detector.detect(frame)
            enriched = ocr.enrich_detections(frame, boxes)
            for box in enriched:
                print(box['cls_name'], "→", box['text'])
        """
        enriched = []
        for box in boxes:
            text, conf = self.read_text_with_confidence(frame, box)
            box_copy = box.copy()
            box_copy["text"] = text
            box_copy["text_conf"] = conf
            enriched.append(box_copy)
        return enriched
