"""Scene cut and lighting transition detector."""

from typing import Optional, Tuple
import cv2
import numpy as np


class SceneCutDetector:
    """Detects abrupt scene cuts, camera transitions, and lighting shifts."""

    def __init__(self, corr_threshold: float = 0.65, diff_threshold: float = 40.0):
        self.corr_threshold = corr_threshold
        self.diff_threshold = diff_threshold
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_hist: Optional[np.ndarray] = None

    def compute_frame_brightness(self, frame: np.ndarray) -> float:
        """Compute mean normalized luminance of the frame in [0, 1]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray) / 255.0)

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Check if current frame is a scene cut relative to previous frame.
        
        Returns:
            is_cut: bool, True if a hard transition was detected.
            brightness: float, normalized brightness of current frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray) / 255.0)

        # Compute HSV color histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if self.prev_gray is None or self.prev_hist is None:
            self.prev_gray = gray
            self.prev_hist = hist
            return False, brightness

        # 1. Histogram correlation (1.0 = identical, < 0.65 = major change)
        hist_corr = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)

        # 2. Mean pixel difference
        mean_diff = float(np.mean(cv2.absdiff(self.prev_gray, gray)))

        # Update cache
        self.prev_gray = gray
        self.prev_hist = hist

        # Detect cut
        is_cut = (hist_corr < self.corr_threshold) or (mean_diff > self.diff_threshold)
        return is_cut, brightness

    def reset(self):
        """Reset internal state."""
        self.prev_gray = None
        self.prev_hist = None
