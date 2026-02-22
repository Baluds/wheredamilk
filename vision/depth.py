"""
vision/depth.py — MiDaS monocular depth estimator (single RGB webcam).

Uses Intel MiDaS via HuggingFace transformers to predict a relative
depth map from a standard RGB frame.  No depth camera required.

Returns depth at the centre of a given bounding box, as a value in
[0.0, 1.0] where 1.0 = furthest, 0.0 = closest.

Install:
    pip install transformers timm torch

Usage:
    estimator = DepthEstimator()          # loads model once
    depth = estimator.box_depth(frame, box)   # 0.0 (close) → 1.0 (far)
"""

import os
import numpy as np
import cv2

# Ensure HuggingFace downloads are allowed (overrides TRANSFORMERS_OFFLINE if set in shell)
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# MiDaS small is fast enough for near-real-time on CPU (~50–120 ms/frame)
MODEL_ID = "Intel/dpt-hybrid-midas"


class DepthEstimator:
    def __init__(self):
        if not _HF_AVAILABLE:
            print("[depth] transformers not installed — depth estimation disabled.")
            print("[depth] Install: pip install transformers timm torch")
            self._pipe = None
            return

        try:
            print("[depth] Loading MiDaS depth model (first run downloads ~400 MB)…")
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=MODEL_ID,
            )
            print("[depth] MiDaS ready.")
        except Exception as exc:
            print(f"[depth] Could not load MiDaS: {exc}")
            print("[depth] Depth estimation disabled — will use bbox-area fallback.")
            self._pipe = None

    # ── Public API ────────────────────────────────────────────────────────────

    def available(self) -> bool:
        return self._pipe is not None

    def depth_map(self, frame) -> np.ndarray | None:
        """
        Estimate a depth map for the full frame.

        Returns a float32 array (H×W) where higher = further away,
        normalised to [0, 1].  Returns None if model unavailable.
        """
        if not self.available():
            return None

        # HuggingFace pipeline expects a PIL Image or numpy RGB array
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb)

        result = self._pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)

        # Resize to match the frame in case model output differs
        if depth.shape[:2] != frame.shape[:2]:
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return depth

    def box_depth(self, frame, box: dict) -> float | None:
        """
        Return the median normalised depth at the centre region of `box`.

        Returns:
            float in [0, 1]  where 0 = very close, 1 = very far.
            None if model unavailable.
        """
        dm = self.depth_map(frame)
        if dm is None:
            return None

        # Sample the inner 50% of the box to avoid edge noise
        cx = (box["x1"] + box["x2"]) // 2
        cy = (box["y1"] + box["y2"]) // 2
        hw = max(1, (box["x2"] - box["x1"]) // 4)
        hh = max(1, (box["y2"] - box["y1"]) // 4)

        x1 = max(0, cx - hw)
        x2 = min(dm.shape[1] - 1, cx + hw)
        y1 = max(0, cy - hh)
        y2 = min(dm.shape[0] - 1, cy + hh)

        patch = dm[y1:y2, x1:x2]
        return float(np.median(patch)) if patch.size > 0 else None

    def depth_to_phrase(self, depth_val: float) -> str:
        """
        Convert a normalised depth value → human-readable distance phrase.

        Thresholds tuned for typical indoor (~3 m max) scenarios.
        """
        if depth_val > 0.75:
            return "move forward"
        elif depth_val > 0.50:
            return "keep going"
        elif depth_val > 0.25:
            return "almost there"
        else:
            return "stop, it's right in front of you"
