"""
logic/tracker.py — Simple IoU-based single-target tracker.

When YOLO detections are available each frame, the tracker picks
whichever new detection overlaps most with the previously known box.
If no detection has IoU above the threshold, the previous box is
returned unchanged (handles momentary occlusion).
"""

IOU_MIN = 0.15  # minimum IoU to consider a detection as the same target


def _iou(a: dict, b: dict) -> float:
    """Compute Intersection-over-Union between two boxes."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class IoUTracker:
    """
    Tracks a single target across frames using IoU matching.

    Usage:
        tracker = IoUTracker()
        target = tracker.update(new_detections, previous_target_box)
    """

    def update(self, detections: list[dict], prev_box: dict) -> dict:
        """
        Find the detection closest (by IoU) to `prev_box`.

        Args:
            detections: list of current box dicts from YOLO.
            prev_box: last known target box dict.

        Returns:
            Best-matching box dict, or `prev_box` if no good match.
        """
        if not detections:
            return prev_box

        best_box = prev_box
        best_iou = IOU_MIN  # must beat this to count as a match

        for det in detections:
            score = _iou(prev_box, det)
            if score > best_iou:
                best_iou = score
                best_box = det

        return best_box
