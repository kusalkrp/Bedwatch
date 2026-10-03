"""Bed geometry calculations: polygon containment, signed distance, and overlap."""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def compute_signed_distance_to_polygon(polygon: np.ndarray, point: Tuple[float, float]) -> float:
    """Compute signed distance in pixels from point (x, y) to polygon.
    
    Positive if outside the polygon, negative if inside, zero on the boundary.
    (Note: cv2.pointPolygonTest returns positive inside, so we negate it to match our convention).
    """
    if polygon is None or len(polygon) < 3:
        return 999.0
    pt = (float(point[0]), float(point[1]))
    # cv2: > 0 inside, < 0 outside
    cv_dist = cv2.pointPolygonTest(polygon, pt, True)
    return -float(cv_dist)


def compute_bed_features(
    polygon: Optional[np.ndarray],
    keypoints: np.ndarray,
    keypoint_confs: np.ndarray,
    body_height: float,
    min_conf: float = 0.25,
) -> Tuple[float, float]:
    """Compute (hip_in_bed, dist_to_bed_bh) from person keypoints and mattress polygon.
    
    Args:
        polygon: ndarray of shape (N, 2) or None if moving view.
        keypoints: ndarray of shape (17, 2) COCO points:
                   [0: nose, 5: l_sh, 6: r_sh, 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ank, 16: r_ank]
        keypoint_confs: ndarray of shape (17,) confidences.
        body_height: person's estimated standing height in pixels.
        min_conf: threshold for keypoint inclusion.
        
    Returns:
        hip_in_bed: float in [0.0, 1.0], soft fraction of hip and torso points inside polygon.
        dist_to_bed_bh: signed distance in body heights (negative inside, positive outside).
    """
    if polygon is None or body_height <= 0:
        return 0.0, 999.0

    # Torso/hip keypoints: 5 (l_sh), 6 (r_sh), 11 (l_hip), 12 (r_hip)
    torso_indices = [5, 6, 11, 12]
    hip_indices = [11, 12]

    # Calculate hip center
    valid_hip_pts = [keypoints[i] for i in hip_indices if keypoint_confs[i] >= min_conf]
    if valid_hip_pts:
        hip_center = np.mean(valid_hip_pts, axis=0)
    else:
        # Fallback to any valid torso point or bounding box center
        valid_torso = [keypoints[i] for i in torso_indices if keypoint_confs[i] >= min_conf]
        if valid_torso:
            hip_center = np.mean(valid_torso, axis=0)
        else:
            return 0.0, 999.0

    # Signed distance to polygon boundary in body heights
    pixel_dist = compute_signed_distance_to_polygon(polygon, (hip_center[0], hip_center[1]))
    dist_to_bed_bh = pixel_dist / max(body_height, 50.0)

    # Compute soft hip_in_bed fraction based on hips and torso points
    in_count = 0.0
    weight_sum = 0.0

    # Hips get higher weight
    for idx in hip_indices:
        if keypoint_confs[idx] >= min_conf:
            pt = (float(keypoints[idx][0]), float(keypoints[idx][1]))
            is_inside = cv2.pointPolygonTest(polygon, pt, False) >= 0
            in_count += 2.0 if is_inside else 0.0
            weight_sum += 2.0

    # Shoulders get standard weight
    for idx in [5, 6]:
        if keypoint_confs[idx] >= min_conf:
            pt = (float(keypoints[idx][0]), float(keypoints[idx][1]))
            is_inside = cv2.pointPolygonTest(polygon, pt, False) >= 0
            in_count += 1.0 if is_inside else 0.0
            weight_sum += 1.0

    hip_in_bed = in_count / weight_sum if weight_sum > 0 else 0.0

    # If hip center is deeply inside the polygon, ensure hip_in_bed is close to 1.0
    if dist_to_bed_bh < -0.1 and hip_in_bed > 0.4:
        hip_in_bed = max(hip_in_bed, 0.9)
    elif dist_to_bed_bh > 0.15:
        hip_in_bed = min(hip_in_bed, 0.2)

    return float(np.clip(hip_in_bed, 0.0, 1.0)), float(dist_to_bed_bh)
