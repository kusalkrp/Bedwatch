"""YOLO11-pose inference and tracking wrapper."""

from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PoseEstimator:
    """Runs YOLO11-pose estimation with ByteTrack tracking."""

    def __init__(
        self,
        model_name: str = "yolo11s-pose.pt",
        tracker_type: str = "bytetrack.yaml",
        min_keypoint_conf: float = 0.25,
        brightness_threshold: float = 0.15,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.tracker_type = tracker_type
        self.min_keypoint_conf = min_keypoint_conf
        self.brightness_threshold = brightness_threshold

        # CLAHE for low-light enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_if_dim(self, frame: np.ndarray, brightness: float) -> np.ndarray:
        """Apply adaptive CLAHE enhancement if frame brightness is low."""
        if brightness >= self.brightness_threshold:
            return frame

        # Convert to LAB and apply CLAHE on L-channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def process_frame(
        self,
        frame: np.ndarray,
        brightness: float,
    ) -> List[Dict]:
        """Run pose tracking on the frame.
        
        Returns a list of dicts for each detected track:
            track_id: int
            bbox: [x1, y1, x2, y2]
            keypoints: ndarray of shape (17, 2)
            confs: ndarray of shape (17,)
            hip_xy: Tuple[float, float]
        """
        img_to_run = self.preprocess_if_dim(frame, brightness)

        # Run YOLO with ByteTrack
        results = self.model.track(
            img_to_run,
            persist=True,
            tracker=self.tracker_type,
            verbose=False,
            device=self.device,
        )

        res = results[0]
        detected_tracks: List[Dict] = []

        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return detected_tracks

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        # Track IDs might be None if ByteTrack is still initializing in the first frame
        track_ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(boxes_xyxy))
        kpts_xy = res.keypoints.xy.cpu().numpy()
        kpts_conf = res.keypoints.conf.cpu().numpy()

        for i, tid in enumerate(track_ids):
            bbox = [float(v) for v in boxes_xyxy[i]]
            kpts = kpts_xy[i]
            confs = kpts_conf[i]

            # Hip midpoint (11: l_hip, 12: r_hip)
            hip_x = float((kpts[11, 0] + kpts[12, 0]) / 2.0)
            hip_y = float((kpts[11, 1] + kpts[12, 1]) / 2.0)

            # Fallback if hips are not detected (e.g. keypoint confs == 0)
            if hip_x == 0.0 and hip_y == 0.0:
                hip_x = (bbox[0] + bbox[2]) / 2.0
                hip_y = bbox[1] + (bbox[3] - bbox[1]) * 0.6

            detected_tracks.append({
                "track_id": int(tid),
                "bbox": bbox,
                "keypoints": kpts,
                "confs": confs,
                "hip_xy": (hip_x, hip_y),
            })

        return detected_tracks
