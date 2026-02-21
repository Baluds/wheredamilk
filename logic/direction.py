"""
logic/direction.py — Spatial direction computation.

Combines horizontal position (left/right/ahead from bbox centre) with
depth estimation:

  PRIMARY  — MiDaS depth value via vision.depth.DepthEstimator (0=close, 1=far)
  FALLBACK — bbox area fraction (used when depth model is unavailable)

Returns a human-friendly phrase like:
    "on your left — keep going"
    "ahead — almost there"
    "stop, it's right in front of you"
"""

# ── Fallback area thresholds (fraction of total frame area) ──────────────────
AREA_SMALL = 0.03    # < 3%   → move forward
AREA_MED   = 0.10    # < 10%  → keep going
AREA_CLOSE = 0.25    # < 25%  → almost there
                     # ≥ 25%  → stop


def compute_direction(
    box: dict,
    frame_w: int,
    frame_h: int,
    depth_val: float | None = None,
) -> str:
    """
    Compute a spoken direction phrase for the given box.

    Args:
        box:       dict with x1, y1, x2, y2 keys (pixel coords).
        frame_w:   frame width in pixels.
        frame_h:   frame height in pixels.
        depth_val: normalised MiDaS depth [0=close, 1=far], or None to
                   fall back to bbox-area heuristic.

    Returns:
        A human-readable direction string.
    """
    cx = (box["x1"] + box["x2"]) / 2.0

    # ── Horizontal ────────────────────────────────────────────────────────────
    if cx < frame_w * 0.33:
        horiz = "on your left"
    elif cx > frame_w * 0.66:
        horiz = "on your right"
    else:
        horiz = "ahead"

    # ── Distance ──────────────────────────────────────────────────────────────
    if depth_val is not None:
        # MiDaS: 0 = very close, 1 = very far
        if depth_val > 0.75:
            dist = "move forward"
        elif depth_val > 0.50:
            dist = "keep going"
        elif depth_val > 0.25:
            dist = "almost there"
        else:
            return "stop, it's right in front of you"
    else:
        # Fallback: bbox area as proxy for distance
        w = box["x2"] - box["x1"]
        h = box["y2"] - box["y1"]
        area_frac = (w * h) / (frame_w * frame_h)

        if area_frac < AREA_SMALL:
            dist = "move forward"
        elif area_frac < AREA_MED:
            dist = "keep going"
        elif area_frac < AREA_CLOSE:
            dist = "almost there"
        else:
            return "stop, it's right in front of you"

    return f"{horiz} — {dist}"
