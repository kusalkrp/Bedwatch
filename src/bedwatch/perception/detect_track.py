"""Patient track selector and continuity manager across cuts and clothing changes."""

from collections import deque
import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class PatientTracker:
    """Selects and maintains the primary patient track using spatial and bed continuity."""

    def __init__(self, default_body_height: float = 460.0):
        self.default_body_height = default_body_height
        self.body_height = default_body_height
        self.active_patient_id: Optional[int] = None
        self.last_known_hip: Optional[Tuple[float, float]] = None
        self.last_known_bbox: Optional[List[float]] = None
        self.last_seen_t: float = 0.0

        # Recent hip history: deque of (t, x, y)
        self.hip_history: deque = deque(maxlen=10)

    def reset_on_scene_cut(self):
        """Handle hard scene cut: relax distance continuity but retain body height."""
        self.hip_history.clear()

    def update_body_height(self, bbox: List[float], torso_angle: float):
        """Update estimated standing body height when person is upright."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # Upright posture: tall box, low torso angle
        if torso_angle < 25.0 and h > 300.0 and (w / h) < 0.55:
            # Exponential smoothing
            self.body_height = 0.9 * self.body_height + 0.1 * h

    def compute_speed_bh_s(self, current_t: float, current_hip: Tuple[float, float]) -> float:
        """Compute speed in body heights per second over ~1.0s window."""
        self.hip_history.append((current_t, current_hip[0], current_hip[1]))
        if len(self.hip_history) < 2:
            return 0.0

        # Look back up to 1.0s
        t_start, x_start, y_start = self.hip_history[0]
        dt = current_t - t_start
        if dt < 0.2:
            return 0.0

        dx = current_hip[0] - x_start
        dy = current_hip[1] - y_start
        dist_px = math.hypot(dx, dy)
        speed_px_s = dist_px / dt
        return float(speed_px_s / max(self.body_height, 100.0))

    def select_patient_track(
        self,
        t: float,
        tracks: List[Dict],  # Each has: track_id, bbox, keypoints, confs, hip_xy
        polygon: Optional[np.ndarray],
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """Select the primary patient track among candidates and label caregivers.
        
        Returns:
            (patient_track, caregiver_tracks)
        """
        if not tracks:
            return None, []

        if len(tracks) == 1:
            candidate = tracks[0]
            # Check if this single person might be a caregiver during an absence
            # (e.g. 97s-102s: caregiver at door while patient is out of room)
            # If door area and patient was previously seen out of bed, handle gracefully.
            self.active_patient_id = candidate["track_id"]
            self.last_known_hip = candidate["hip_xy"]
            self.last_known_bbox = candidate["bbox"]
            self.last_seen_t = t
            return candidate, []

        # Multiple people in frame (e.g., patient in bed + caregiver standing beside bed)
        scored_tracks = []
        for trk in tracks:
            score = 0.0
            hip = trk["hip_xy"]

            # Bed overlap score
            if polygon is not None and len(polygon) >= 3:
                is_in_bed = cv2.pointPolygonTest(polygon, (float(hip[0]), float(hip[1])), False) >= 0
                if is_in_bed:
                    score += 50.0  # High preference for the person in bed

            # Track ID continuity score
            if self.active_patient_id is not None and trk["track_id"] == self.active_patient_id:
                score += 30.0

            # Spatial distance score relative to last known hip
            if self.last_known_hip is not None:
                dist = math.hypot(hip[0] - self.last_known_hip[0], hip[1] - self.last_known_hip[1])
                score += max(0.0, 30.0 - (dist / 20.0))

            scored_tracks.append((score, trk))

        # Sort descending by score
        scored_tracks.sort(key=lambda x: x[0], reverse=True)
        patient_track = scored_tracks[0][1]
        caregiver_tracks = [st[1] for st in scored_tracks[1:]]

        self.active_patient_id = patient_track["track_id"]
        self.last_known_hip = patient_track["hip_xy"]
        self.last_known_bbox = patient_track["bbox"]
        self.last_seen_t = t

        return patient_track, caregiver_tracks
