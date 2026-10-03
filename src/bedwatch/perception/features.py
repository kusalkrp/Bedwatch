"""Feature definitions and geometric calculations for perception records."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FeatureRecord:
    t: float
    view: str
    scene_cut: bool
    primary_track: int
    n_persons: int
    bbox: List[float]
    keypoints: List[List[float]]
    torso_angle_deg: float
    body_aspect: float
    hip_in_bed: float
    dist_to_bed_bh: float
    speed_bh_s: float
    hip_to_ankle_ratio: float
    core_visible: bool
    visible_frac: float
    mean_kpt_conf: float
    brightness: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureRecord":
        return cls(**data)


def compute_torso_angle_deg(
    keypoints: np.ndarray,
    keypoint_confs: np.ndarray,
    min_conf: float = 0.25,
) -> float:
    """Compute torso angle from vertical in degrees [0, 90].
    
    0 degrees = perfectly upright (standing or sitting straight).
    90 degrees = perfectly horizontal (lying flat).
    """
    # Shoulders: 5 (l_sh), 6 (r_sh); Hips: 11 (l_hip), 12 (r_hip)
    sh_pts = [keypoints[i] for i in [5, 6] if keypoint_confs[i] >= min_conf]
    hip_pts = [keypoints[i] for i in [11, 12] if keypoint_confs[i] >= min_conf]

    if not sh_pts or not hip_pts:
        return 45.0  # Ambiguous fallback

    sh_mid = np.mean(sh_pts, axis=0)
    hip_mid = np.mean(hip_pts, axis=0)

    dx = abs(float(sh_mid[0] - hip_mid[0]))
    dy = abs(float(sh_mid[1] - hip_mid[1]))

    if dy < 1e-4:
        return 90.0

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    return float(np.clip(angle_deg, 0.0, 90.0))


def compute_hip_to_ankle_ratio(
    keypoints: np.ndarray,
    keypoint_confs: np.ndarray,
    body_height: float,
    min_conf: float = 0.25,
) -> float:
    """Compute vertical distance from hip center to ankle center divided by standing body height."""
    hip_pts = [keypoints[i] for i in [11, 12] if keypoint_confs[i] >= min_conf]
    ank_pts = [keypoints[i] for i in [15, 16] if keypoint_confs[i] >= min_conf]

    if not hip_pts or not ank_pts or body_height <= 0:
        return 0.5  # Standard upright default

    hip_y = np.mean(hip_pts, axis=0)[1]
    ank_y = np.mean(ank_pts, axis=0)[1]

    vert_dist = abs(float(ank_y - hip_y))
    ratio = vert_dist / max(body_height, 100.0)
    return float(np.clip(ratio, 0.0, 1.0))


def compute_core_visibility(
    keypoint_confs: np.ndarray,
    min_conf: float = 0.3,
) -> Tuple[bool, float, float]:
    """Evaluate visibility of core joints (shoulders, hips) and overall pose quality."""
    core_indices = [5, 6, 11, 12]
    core_visible = all(keypoint_confs[i] >= min_conf for i in core_indices)

    visible_mask = keypoint_confs >= min_conf
    visible_frac = float(np.mean(visible_mask))

    if np.any(visible_mask):
        mean_kpt_conf = float(np.mean(keypoint_confs[visible_mask]))
    else:
        mean_kpt_conf = 0.0

    return core_visible, visible_frac, mean_kpt_conf
