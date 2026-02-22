"""
vision/ocr.py — PaddleOCR wrapper.

Reads text from a cropped region of a frame.
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

        result = self._ocr.ocr(crop, cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text, conf = line[1]
                    if conf > 0.5:
                        texts.append(text)
        return " ".join(texts).strip()
