"""Segment data structure and duration summary aggregators."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


IN_BED_STATES = {"LYING_IN_BED", "SITTING_ON_BED"}
OUT_OF_BED_STATES = {
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
    "OUT_OF_BED",
    "LYING_ON_FLOOR",
    "UNKNOWN",
}


@dataclass
class Segment:
    start: float
    end: float
    state: str
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_duration_str(seconds: float) -> str:
    """Format seconds into 'Xm Ys' or 'Ys' string."""
    sec = int(round(seconds))
    mins = sec // 60
    rem_sec = sec % 60
    if mins > 0:
        return f"{mins}m {rem_sec:02d}s"
    return f"{rem_sec}s"


def format_hms(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    sec = int(round(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def merge_contiguous_segments(segments: List[Segment]) -> List[Segment]:
    """Merge adjacent segments having the exact same state."""
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.state == prev.state:
            # Weighted average confidence
            d1 = prev.duration_s
            d2 = seg.duration_s
            new_conf = (prev.confidence * d1 + seg.confidence * d2) / max(1e-4, d1 + d2)
            prev.end = seg.end
            prev.confidence = round(new_conf, 3)
        else:
            merged.append(seg)
    return merged


def compute_durations_summary(
    segments: List[Segment],
    total_observation_sec: float,
    bed_exit_count: int = 0,
    bed_return_count: int = 0,
    floor_event_count: int = 0,
) -> Dict[str, Any]:
    """Calculate duration summaries matching assignment specifications."""
    activity_durations: Dict[str, float] = {
        "lying_in_bed": 0.0,
        "sitting_on_bed": 0.0,
        "sitting_outside_bed": 0.0,
        "standing": 0.0,
        "walking": 0.0,
        "out_of_bed": 0.0,
        "lying_on_floor": 0.0,
        "unknown": 0.0,
    }

    # Sum durations
    for seg in segments:
        key = seg.state.lower()
        if key in activity_durations:
            activity_durations[key] += seg.duration_s

    # In bed vs out of bed
    time_in_bed = activity_durations["lying_in_bed"] + activity_durations["sitting_on_bed"]
    time_out_of_bed = sum(
        activity_durations[k]
        for k in activity_durations
        if k not in ("lying_in_bed", "sitting_on_bed")
    )

    # Longest continuous out-of-bed period
    longest_out_of_bed = 0.0
    current_out_streak = 0.0
    for seg in segments:
        if seg.state in OUT_OF_BED_STATES:
            current_out_streak += seg.duration_s
            if current_out_streak > longest_out_of_bed:
                longest_out_of_bed = current_out_streak
        else:
            current_out_streak = 0.0

    final_state = segments[-1].state.lower() if segments else "unknown"

    summary = {
        "observation_duration_sec": round(total_observation_sec),
        "activity_duration_sec": {k: round(v) for k, v in activity_durations.items()},
        "activity_summary_formatted": {k: format_duration_str(v) for k, v in activity_durations.items()},
        "bed_exit_count": bed_exit_count,
        "bed_return_count": bed_return_count,
        "floor_event_count": floor_event_count,
        "total_in_bed_sec": round(time_in_bed),
        "total_out_of_bed_sec": round(time_out_of_bed),
        "bed_summary_formatted": {
            "time_in_bed": format_duration_str(time_in_bed),
            "time_out_of_bed": format_duration_str(time_out_of_bed),
            "bed_exit_count": bed_exit_count,
        },
        "longest_out_of_bed_period_sec": round(longest_out_of_bed),
        "final_state": final_state,
    }
    return summary
