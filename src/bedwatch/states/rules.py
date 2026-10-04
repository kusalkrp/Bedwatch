"""State score estimator: turns FeatureRecord into soft scores for the 8 core states."""

from dataclasses import dataclass
from typing import Dict
import numpy as np

from bedwatch.perception.features import FeatureRecord


CORE_STATES = [
    "LYING_IN_BED",
    "SITTING_ON_BED",
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
    "OUT_OF_BED",
    "LYING_ON_FLOOR",
    "UNKNOWN",
]


@dataclass
class StateScores:
    t: float
    scores: Dict[str, float]
    quality: float

    def get_top_state(self) -> str:
        return max(self.scores.items(), key=lambda x: x[1])[0]


class StateEstimator:
    """Computes soft probability scores [0, 1] for all 8 states from a FeatureRecord."""

    def __init__(self, min_conf: float = 0.3):
        self.min_conf = min_conf

    def estimate_scores(self, rec: FeatureRecord) -> StateScores:
        """Compute score dictionary for one feature record."""
        # Check for OUT_OF_BED (person not visible in frame)
        if rec.primary_track == -1 or rec.n_persons == 0:
            scores = {s: 0.01 for s in CORE_STATES}
            scores["OUT_OF_BED"] = 0.95
            return StateScores(t=rec.t, scores=scores, quality=1.0)

        # Check for UNKNOWN (severe occlusion or low confidence)
        if not rec.core_visible or rec.visible_frac < 0.25 or rec.mean_kpt_conf < 0.25:
            scores = {s: 0.05 for s in CORE_STATES}
            scores["UNKNOWN"] = 0.85
            return StateScores(t=rec.t, scores=scores, quality=rec.mean_kpt_conf)

        scores = {s: 0.001 for s in CORE_STATES}

        torso_angle = rec.torso_angle_deg
        aspect = rec.body_aspect
        hip_bed = rec.hip_in_bed
        speed = rec.speed_bh_s
        hip_ank_ratio = rec.hip_to_ankle_ratio
        dist_bh = rec.dist_to_bed_bh

        # Extract keypoint geometry for scale-invariant knee/thigh flexion
        kpts = np.array(rec.keypoints)
        hip_pts = [kpts[i] for i in [11, 12] if len(kpts) > i and kpts[i, 2] > 0.25]
        knee_pts = [kpts[i] for i in [13, 14] if len(kpts) > i and kpts[i, 2] > 0.25]

        thigh_angle = None
        if hip_pts and knee_pts:
            hip_y = np.mean([p[1] for p in hip_pts])
            hip_x = np.mean([p[0] for p in hip_pts])
            knee_y = np.mean([p[1] for p in knee_pts])
            knee_x = np.mean([p[0] for p in knee_pts])
            thigh_dy = abs(knee_y - hip_y)
            thigh_dx = abs(knee_x - hip_x)
            thigh_angle = float(np.degrees(np.arctan2(thigh_dy, max(1.0, thigh_dx))))

        # 1. Posture indicators
        # Horizontal / Reclined (lying) - requires significant torso recline
        is_horizontal = (torso_angle > 48.0) or (torso_angle > 38.0 and aspect > 1.15)

        # 2. Bed overlap indicators
        on_bed = hip_bed > 0.4 or dist_bh < -0.05
        off_bed = hip_bed < 0.2 and dist_bh > 0.10

        # 3. Motion indicators
        is_moving = speed >= 0.18

        # --- A. Lying states ---
        if is_horizontal:
            if on_bed:
                scores["LYING_IN_BED"] = 0.85 + 0.15 * hip_bed
            elif off_bed:
                scores["LYING_ON_FLOOR"] = 0.90
            else:
                scores["LYING_IN_BED"] = 0.55
                scores["LYING_ON_FLOOR"] = 0.35
                scores["UNKNOWN"] = 0.10

        # --- B. Upright / Seated / Standing states ---
        else:
            if is_moving and off_bed:
                scores["WALKING"] = 0.85 + 0.15 * min(1.0, speed / 0.5)
                scores["STANDING"] = 0.10
            elif is_moving and speed >= 0.26:
                scores["WALKING"] = 0.80
                scores["STANDING"] = 0.15
            else:
                # Stationary / slow upright posture
                is_seated_thigh = (thigh_angle is not None and thigh_angle < 72.0)
                is_standing_thigh = (thigh_angle is not None and thigh_angle >= 78.0)

                if on_bed:
                    # In elderly care, upright person on bed is SITTING_ON_BED
                    # unless standing on the floor right beside the edge
                    is_edge_standing = (
                        dist_bh >= -0.12 and
                        aspect < 0.40 and
                        (is_standing_thigh or thigh_angle is None) and
                        hip_ank_ratio >= 0.41
                    )

                    if is_edge_standing:
                        scores["STANDING"] = 0.82
                        scores["SITTING_ON_BED"] = 0.15
                    else:
                        scores["SITTING_ON_BED"] = 0.85 + 0.12 * hip_bed
                        scores["STANDING"] = 0.10

                elif off_bed:
                    # Away from bed: sitting in chair vs standing
                    if is_standing_thigh:
                        is_chair_seated = False
                    elif is_seated_thigh:
                        is_chair_seated = True
                    else:
                        is_chair_seated = (aspect > 0.40 and hip_ank_ratio < 0.40)

                    if is_chair_seated:
                        scores["SITTING_OUTSIDE_BED"] = 0.85
                        scores["STANDING"] = 0.10
                    else:
                        scores["STANDING"] = 0.82
                        scores["SITTING_OUTSIDE_BED"] = 0.12
                else:
                    # Borderline bed edge
                    if is_seated_thigh or aspect > 0.45:
                        scores["SITTING_ON_BED"] = 0.70
                        scores["STANDING"] = 0.20
                    else:
                        scores["STANDING"] = 0.70
                        scores["SITTING_ON_BED"] = 0.20

        # Baseline small unknown
        scores["UNKNOWN"] = max(scores["UNKNOWN"], 0.02)

        # Normalize to sum to 1.0
        total = sum(scores.values())
        norm_scores = {s: v / total for s, v in scores.items()}
        return StateScores(t=rec.t, scores=norm_scores, quality=rec.mean_kpt_conf)

