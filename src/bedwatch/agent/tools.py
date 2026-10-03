"""Agent tools for inspecting temporal context and rechecking geometry."""

from typing import Any, Dict, List, Optional
import numpy as np

from bedwatch.config import BedwatchConfig
from bedwatch.perception.bed import compute_bed_features
from bedwatch.perception.features import FeatureRecord
from bedwatch.states.segments import Segment


class AgentContextTools:
    """Bounded tools available to the verification agent."""

    def __init__(
        self,
        segments: List[Segment],
        records: List[FeatureRecord],
        config: BedwatchConfig,
    ):
        self.segments = segments
        self.records = records
        self.config = config
        self.rec_by_time = {r.t: r for r in records}

    def get_previous_segment(self, t: float, window_s: float = 15.0) -> Optional[Dict[str, Any]]:
        """Get the segment occurring immediately prior to timestamp t."""
        candidates = [s for s in self.segments if s.end <= t + 0.1 and (t - s.end) <= window_s]
        if not candidates:
            return None
        prev_seg = candidates[-1]
        return {
            "state": prev_seg.state,
            "start": prev_seg.start,
            "end": prev_seg.end,
            "duration_s": prev_seg.duration_s,
            "confidence": prev_seg.confidence,
        }

    def get_next_segment(self, t: float, window_s: float = 15.0) -> Optional[Dict[str, Any]]:
        """Get the segment occurring immediately after timestamp t."""
        candidates = [s for s in self.segments if s.start >= t - 0.1 and (s.start - t) <= window_s]
        if not candidates:
            return None
        next_seg = candidates[0]
        return {
            "state": next_seg.state,
            "start": next_seg.start,
            "end": next_seg.end,
            "duration_s": next_seg.duration_s,
            "confidence": next_seg.confidence,
        }

    def recheck_bed_overlap(self, t: float, window_s: float = 1.0) -> Dict[str, Any]:
        """Recompute mattress polygon overlap over a small temporal window."""
        window_recs = [r for r in self.records if abs(r.t - t) <= window_s and r.primary_track != -1]
        if not window_recs:
            return {"mean_hip_in_bed": 0.0, "mean_dist_bh": 999.0, "sample_count": 0}

        mean_hip_in_bed = float(np.mean([r.hip_in_bed for r in window_recs]))
        mean_dist_bh = float(np.mean([r.dist_to_bed_bh for r in window_recs if r.dist_to_bed_bh < 900.0]))
        return {
            "mean_hip_in_bed": round(mean_hip_in_bed, 3),
            "mean_dist_bh": round(mean_dist_bh, 3),
            "sample_count": len(window_recs),
        }
