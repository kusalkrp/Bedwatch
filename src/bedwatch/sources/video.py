"""Video decoding, perception pipeline orchestration, and feature extraction."""

from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union
import cv2
import numpy as np

from bedwatch.config import BedwatchConfig
from bedwatch.perception.bed import compute_bed_features
from bedwatch.perception.detect_track import PatientTracker
from bedwatch.perception.features import (
    FeatureRecord,
    compute_core_visibility,
    compute_hip_to_ankle_ratio,
    compute_torso_angle_deg,
)
from bedwatch.perception.pose import PoseEstimator
from bedwatch.perception.scenecut import SceneCutDetector
from bedwatch.sources.features import write_feature_cache


class VideoSource:
    """Decodes video, runs perception models, and yields FeatureRecord instances."""

    def __init__(self, video_path: Union[str, Path], config: BedwatchConfig):
        self.video_path = str(video_path)
        self.config = config
        self.sampling_fps = config.sampling_fps

        # Initialize perception modules
        self.scenecut_detector = SceneCutDetector(
            corr_threshold=0.65,
            diff_threshold=40.0,
        )
        self.pose_estimator = PoseEstimator(
            model_name=config.perception.pose_model,
            tracker_type=config.perception.tracker_type,
            min_keypoint_conf=config.perception.min_keypoint_conf,
            brightness_threshold=config.perception.brightness_threshold,
        )
        self.patient_tracker = PatientTracker(default_body_height=460.0)

    def iter_records(
        self,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> Iterator[FeatureRecord]:
        """Yield FeatureRecords sampled at sampling_fps."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_duration = total_video_frames / video_fps if total_video_frames > 0 else 0.0
        total_samples = int(total_duration * self.sampling_fps) if total_duration > 0 else 0

        sample_idx = 0
        while True:
            # Timestamp for this sample
            t = sample_idx / self.sampling_fps
            target_frame_idx = int(round(t * video_fps))

            if total_video_frames > 0 and target_frame_idx >= total_video_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # 1. Resolve camera view and bed polygon
            view_name, polygon = self.config.views.get_view_for_time(t)

            # 2. Scene cut & brightness
            scene_cut, brightness = self.scenecut_detector.process_frame(frame)
            if scene_cut:
                self.patient_tracker.reset_on_scene_cut()

            # 3. Pose estimation with ByteTrack
            detected_tracks = self.pose_estimator.process_frame(frame, brightness)

            # 4. Patient track selection
            patient_track, caregiver_tracks = self.patient_tracker.select_patient_track(
                t=t,
                tracks=detected_tracks,
                polygon=polygon,
            )

            # 5. Feature computation
            if patient_track is not None:
                bbox = patient_track["bbox"]
                kpts = patient_track["keypoints"]  # (17, 2)
                confs = patient_track["confs"]      # (17,)
                primary_id = int(patient_track["track_id"])

                # Torso angle
                torso_angle = compute_torso_angle_deg(
                    keypoints=kpts,
                    keypoint_confs=confs,
                    min_conf=self.config.perception.min_keypoint_conf,
                )

                # Body aspect ratio: width / height
                bw = max(1.0, bbox[2] - bbox[0])
                bh = max(1.0, bbox[3] - bbox[1])
                body_aspect = float(bw / bh)

                # Update body height
                self.patient_tracker.update_body_height(bbox, torso_angle)

                # Bed features: hip_in_bed and distance
                hip_in_bed, dist_to_bed_bh = compute_bed_features(
                    polygon=polygon,
                    keypoints=kpts,
                    keypoint_confs=confs,
                    body_height=self.patient_tracker.body_height,
                    min_conf=self.config.perception.min_keypoint_conf,
                )

                # Speed
                speed_bh_s = self.patient_tracker.compute_speed_bh_s(
                    current_t=t,
                    current_hip=patient_track["hip_xy"],
                )

                # Hip to ankle ratio
                hip_to_ankle_ratio = compute_hip_to_ankle_ratio(
                    keypoints=kpts,
                    keypoint_confs=confs,
                    body_height=self.patient_tracker.body_height,
                    min_conf=self.config.perception.min_keypoint_conf,
                )

                # Core visibility
                core_vis, vis_frac, mean_conf = compute_core_visibility(
                    keypoint_confs=confs,
                    min_conf=self.config.perception.min_keypoint_conf,
                )

                # Format keypoints as list of [x, y, conf]
                formatted_kpts = [
                    [float(kpts[i, 0]), float(kpts[i, 1]), float(confs[i])]
                    for i in range(17)
                ]
            else:
                # No patient detected
                primary_id = -1
                bbox = [0.0, 0.0, 0.0, 0.0]
                formatted_kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
                torso_angle = 45.0
                body_aspect = 1.0
                hip_in_bed = 0.0
                dist_to_bed_bh = 999.0
                speed_bh_s = 0.0
                hip_to_ankle_ratio = 0.5
                core_vis = False
                vis_frac = 0.0
                mean_conf = 0.0

            rec = FeatureRecord(
                t=round(float(t), 3),
                view=view_name,
                scene_cut=bool(scene_cut),
                primary_track=primary_id,
                n_persons=len(detected_tracks),
                bbox=[round(v, 1) for v in bbox],
                keypoints=[[round(v[0], 1), round(v[1], 1), round(v[2], 2)] for v in formatted_kpts],
                torso_angle_deg=round(float(torso_angle), 2),
                body_aspect=round(float(body_aspect), 3),
                hip_in_bed=round(float(hip_in_bed), 3),
                dist_to_bed_bh=round(float(dist_to_bed_bh), 3),
                speed_bh_s=round(float(speed_bh_s), 3),
                hip_to_ankle_ratio=round(float(hip_to_ankle_ratio), 3),
                core_visible=bool(core_vis),
                visible_frac=round(float(vis_frac), 3),
                mean_kpt_conf=round(float(mean_conf), 3),
                brightness=round(float(brightness), 3),
            )

            if progress_callback:
                progress_callback(t, sample_idx + 1, total_samples)

            yield rec
            sample_idx += 1

        cap.release()


def extract_features(
    video_path: Union[str, Path],
    cache_path: Union[str, Path],
    config: BedwatchConfig,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> List[FeatureRecord]:
    """Extract features from video and save to JSONL cache."""
    source = VideoSource(video_path=video_path, config=config)
    records = []
    for rec in source.iter_records(progress_callback=progress_callback):
        records.append(rec)
    write_feature_cache(records, cache_path)
    return records
