"""
vision/yolo.py — YOLOv8 detection wrapper.

Returns detections sorted by confidence (highest first).
Each detection is a dict:
    {x1, y1, x2, y2, conf, cls_name}
"""

from ultralytics import YOLO
import cv2

FRAME_W = 640
FRAME_H = 480


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        # Warm-up run to avoid first-frame latency
        import numpy as np
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype="uint8")
        self.model(dummy, verbose=False)

    def detect(self, frame) -> list[dict]:
        """
        Run YOLOv8 on a frame (any size – will be resized internally by YOLO).
        Returns list of box dicts sorted by conf descending.
        """
        results = self.model(frame, imgsz=640, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                boxes.append(
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "conf": conf,
                        "cls_name": cls_name,
                    }
                )
        boxes.sort(key=lambda b: b["conf"], reverse=True)
        return boxes
