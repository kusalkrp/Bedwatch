"""Configuration loader and schema validation for Bedwatch."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import numpy as np


@dataclass
class PerceptionConfig:
    pose_model: str = "yolo11s-pose.pt"
    tracker_type: str = "bytetrack.yaml"
    min_keypoint_conf: float = 0.3
    brightness_threshold: float = 0.15
    scene_cut_threshold: float = 0.35


@dataclass
class DecoderConfig:
    min_dwell_s: Dict[str, float] = field(default_factory=lambda: {
        "STANDING": 0.6,
        "WALKING": 1.0,
        "SITTING_ON_BED": 1.0,
        "LYING_IN_BED": 2.0,
        "SITTING_OUTSIDE_BED": 1.5,
        "OUT_OF_BED": 2.0,
        "LYING_ON_FLOOR": 1.5,
        "UNKNOWN": 0.6,
    })
    penalty: Dict[str, float] = field(default_factory=lambda: {
        "self": 0.0,
        "allowed": 1.0,
        "disallowed": 20.0,
        "unknown": 2.0,
    })


@dataclass
class EventsConfig:
    move_away_min_s: float = 2.5
    move_away_min_bh: float = 0.5
    approach_max_bh: float = 0.8
    lying_confirm_s: float = 2.0
    floor_confirm_s: float = 2.0
    dwell_ref_s: float = 10.0


@dataclass
class AgentConfig:
    enabled: bool = True
    max_steps: int = 3
    vlm_calls_per_video: int = 20
    keyframes_per_call: int = 3


@dataclass
class AlertsConfig:
    profile: str = "demo"
    sit_edge_monitor_s: float = 10.0
    unknown_monitor_s: float = 2.0
    absence_monitor_s: float = 20.0
    absence_alert_s: float = 40.0
    lost_alert_s: float = 60.0
    floor_alert_s: float = 3.0


@dataclass
class ViewsConfig:
    views: Dict[str, Any] = field(default_factory=dict)
    schedule: List[Dict[str, Any]] = field(default_factory=list)

    def get_view_for_time(self, t: float) -> Tuple[str, Optional[np.ndarray]]:
        """Return (view_name, bed_polygon_ndarray) for the given timestamp in seconds."""
        for item in self.schedule:
            if item["start_s"] <= t < item["end_s"]:
                view_name = item["view"]
                view_info = self.views.get(view_name, {})
                poly_pts = view_info.get("bed_polygon")
                if poly_pts:
                    return view_name, np.array(poly_pts, dtype=np.int32)
                return view_name, None

        # Fallback to view_main
        poly_pts = self.views.get("view_main", {}).get("bed_polygon")
        if poly_pts:
            return "view_main", np.array(poly_pts, dtype=np.int32)
        # Fallback to any view with a bed_polygon
        for name, info in self.views.items():
            pts = info.get("bed_polygon")
            if pts:
                return name, np.array(pts, dtype=np.int32)
        return "view_main", None


@dataclass
class BedwatchConfig:
    sampling_fps: float = 5.0
    views_config_path: str = "configs/views.yaml"
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)


def load_config(
    config_path: Optional[str] = None,
    profile: Optional[str] = None,
    video_path: Optional[str] = None,
    views_config: Optional[str] = None,
) -> BedwatchConfig:
    """Load and merge configuration from YAML files with automatic video-specific views discovery."""
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Pick config path
    if config_path is None:
        if profile == "production":
            config_path = str(base_dir / "configs" / "default.yaml")
        else:
            config_path = str(base_dir / "configs" / "demo.yaml")

    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = base_dir / cfg_file

    data: Dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            data = yaml.safe_load(f) or {}

    # Resolve views file:
    # 1. Explicit override if provided
    # 2. Video-specific views_{stem}.yaml if present
    # 3. views_config from YAML or default configs/views.yaml
    views_file = None
    if views_config is not None:
        views_file = Path(views_config)
    elif video_path is not None:
        v_stem = Path(video_path).stem
        candidate = base_dir / "configs" / f"views_{v_stem}.yaml"
        if candidate.exists():
            views_file = candidate

    if views_file is None:
        views_rel = data.get("views_config", "configs/views.yaml")
        views_file = Path(views_rel)

    if not views_file.is_absolute():
        views_file = base_dir / views_file

    views_data: Dict[str, Any] = {}
    if views_file.exists():
        with open(views_file, "r") as f:
            views_data = yaml.safe_load(f) or {}

    views_cfg = ViewsConfig(
        views=views_data.get("views", {}),
        schedule=views_data.get("schedule", []),
    )

    perception_data = data.get("perception", {})
    decoder_data = data.get("decoder", {})
    events_data = data.get("events", {})
    agent_data = data.get("agent", {})
    alerts_data = data.get("alerts", {})

    cfg = BedwatchConfig(
        sampling_fps=float(data.get("sampling_fps", 5.0)),
        views_config_path=str(views_file),
        perception=PerceptionConfig(**perception_data),
        decoder=DecoderConfig(**decoder_data),
        events=EventsConfig(**events_data),
        agent=AgentConfig(**agent_data),
        alerts=AlertsConfig(**alerts_data),
        views=views_cfg,
    )

    return cfg

