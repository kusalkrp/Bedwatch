"""Video annotator: renders polygon, skeleton, state labels, and alert banners onto video."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from bedwatch.config import BedwatchConfig
from bedwatch.events.confidence import Event
from bedwatch.perception.features import FeatureRecord
from bedwatch.states.segments import Segment, format_hms


# COCO skeleton bones (joint pairs)
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Facial
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]


def render_annotated_video(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    records: List[FeatureRecord],
    segments: List[Segment],
    events: List[Event],
    config: BedwatchConfig,
    max_frames: Optional[int] = None,
):
    """Render annotated MP4 video with telemetry overlays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total_frames = min(total_frames, max_frames)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Fast segment state lookup by time
    def get_state_at(t_val: float) -> Tuple[str, float]:
        for seg in segments:
            if seg.start <= t_val < seg.end:
                return seg.state, seg.confidence
        return "UNKNOWN", 0.5

    # Fast event lookup by time
    def get_active_event_at(t_val: float) -> Optional[Event]:
        for ev in events:
            if ev.start_s - 1.0 <= t_val <= ev.confirmed_s + 2.0:
                return ev
        return None

    # Map records by closest timestamp
    rec_times = np.array([r.t for r in records])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if max_frames and frame_idx >= max_frames:
            break

        t = frame_idx / fps

        # 1. Draw mattress polygon
        view_name, polygon = config.views.get_view_for_time(t)
        if polygon is not None and len(polygon) >= 3:
            poly_overlay = frame.copy()
            cv2.fillPoly(poly_overlay, [polygon], (0, 180, 0))
            cv2.addWeighted(poly_overlay, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

        # 2. Get closest perception record
        if len(rec_times) > 0:
            nearest_idx = int(np.argmin(np.abs(rec_times - t)))
            rec = records[nearest_idx]

            # Draw bbox and skeleton if patient visible
            if rec.primary_track != -1:
                x1, y1, x2, y2 = [int(v) for v in rec.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                kpts = rec.keypoints
                for bone in COCO_BONES:
                    pt1 = kpts[bone[0]]
                    pt2 = kpts[bone[1]]
                    if pt1[2] > 0.25 and pt2[2] > 0.25:
                        p1 = (int(pt1[0]), int(pt1[1]))
                        p2 = (int(pt2[0]), int(pt2[1]))
                        cv2.line(frame, p1, p2, (0, 255, 255), 2)

                for kpt in kpts:
                    if kpt[2] > 0.25:
                        cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 4, (0, 0, 255), -1)

        # 3. State & telemetry HUD banner (Top-Left)
        state_str, state_conf = get_state_at(t)
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (20, 20), (450, 130), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.6, frame, 0.4, 0, frame)

        time_str = f"TIME: {format_hms(t)} ({t:.1f}s)"
        state_display = f"STATE: {state_str} ({state_conf:.2f})"
        view_display = f"VIEW: {view_name}"

        cv2.putText(frame, "BEDWATCH TELEMETRY", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, time_str, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, state_display, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
        cv2.putText(frame, view_display, (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 4. Contextual Decision & Event Banner (Top-Right)
        active_ev = get_active_event_at(t)
        if active_ev:
            decision = active_ev.decision
            badge_color = (0, 0, 255) if decision == "ALERT" else ((0, 165, 255) if decision == "MONITOR" else (0, 200, 0))

            ev_bg = frame.copy()
            cv2.rectangle(ev_bg, (w - 420, 20), (w - 20, 110), (0, 0, 0), -1)
            cv2.addWeighted(ev_bg, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"DECISION: {decision}", (w - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2)
            cv2.putText(frame, f"EVENT: {active_ev.event.upper()}", (w - 400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"CONF: {active_ev.confidence:.2f}", (w - 400, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 500 == 0:
            total_str = f"/{total_frames}" if total_frames > 0 else ""
            pct_str = f" ({frame_idx/total_frames*100:.1f}%)" if total_frames > 0 else ""
            print(f"  Rendering annotated video: frame {frame_idx}{total_str}{pct_str}...", flush=True)

    cap.release()
    writer.release()
    print(f"  Finished rendering {frame_idx} frames to {out_path}!", flush=True)
