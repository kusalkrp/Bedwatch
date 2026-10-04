"""Bed event data models and confidence calculation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class Event:
    event: str  # "bed_exit", "bed_return", "floor_lying"
    start_time: str  # "HH:MM:SS"
    confirmed_time: str  # "HH:MM:SS"
    start_s: float
    confirmed_s: float
    previous_state: str
    current_state: str
    confidence: float
    decision: str  # "NORMAL", "MONITOR", "ALERT"
    trace_id: str
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "start_time": self.start_time,
            "confirmed_time": self.confirmed_time,
            "previous_state": self.previous_state.lower(),
            "current_state": self.current_state.lower(),
            "confidence": round(self.confidence, 2),
            "decision": self.decision,
            "trace_id": self.trace_id,
        }


def compute_event_confidence(
    chain_completeness: float,
    mean_seg_conf: float,
    prior_bed_dwell_s: float,
    dwell_ref_s: float = 10.0,
) -> float:
    """Compute event confidence based on chain completeness and prior dwell."""
    occupancy_factor = 0.5 + 0.5 * min(1.0, max(0.0, prior_bed_dwell_s) / max(1.0, dwell_ref_s))
    conf = chain_completeness * mean_seg_conf * occupancy_factor
    return float(min(1.0, max(0.1, conf)))
