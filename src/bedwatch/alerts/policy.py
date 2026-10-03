"""Contextual alert policy engine (NORMAL, MONITOR, ALERT)."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from bedwatch.config import AlertsConfig
from bedwatch.events.confidence import Event
from bedwatch.states.segments import Segment


ALERT_LEVELS = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}


@dataclass
class AlertDecision:
    level: str  # "NORMAL", "MONITOR", "ALERT"
    rule_id: str
    rationale: str
    timestamp_s: float


class AlertPolicy:
    """Evaluates safety rules against segment timelines and detected events."""

    def __init__(self, config: AlertsConfig):
        self.config = config

    def evaluate_timeline(
        self,
        segments: List[Segment],
        events: List[Event],
    ) -> List[AlertDecision]:
        """Evaluate contextual alert rules across the timeline."""
        decisions: List[AlertDecision] = []

        # 1. Floor lying rule -> ALERT
        for seg in segments:
            if seg.state == "LYING_ON_FLOOR" and seg.duration_s >= self.config.floor_alert_s:
                decisions.append(
                    AlertDecision(
                        level="ALERT",
                        rule_id="RULE-FLOOR-01",
                        rationale=f"Patient detected on floor for {seg.duration_s:.1f}s (exceeds threshold {self.config.floor_alert_s}s)",
                        timestamp_s=seg.start + self.config.floor_alert_s,
                    )
                )

        # 2. Prolonged bed edge sitting -> MONITOR
        for seg in segments:
            if seg.state == "SITTING_ON_BED" and seg.duration_s >= self.config.sit_edge_monitor_s:
                decisions.append(
                    AlertDecision(
                        level="MONITOR",
                        rule_id="RULE-BED-EDGE-01",
                        rationale=f"Sitting on bed edge for {seg.duration_s:.1f}s (may indicate difficulty standing)",
                        timestamp_s=seg.start + self.config.sit_edge_monitor_s,
                    )
                )

        # 3. Prolonged UNKNOWN -> MONITOR
        for seg in segments:
            if seg.state == "UNKNOWN" and seg.duration_s >= self.config.unknown_monitor_s:
                decisions.append(
                    AlertDecision(
                        level="MONITOR",
                        rule_id="RULE-UNKNOWN-01",
                        rationale=f"Patient state unconfirmed for {seg.duration_s:.1f}s",
                        timestamp_s=seg.start + self.config.unknown_monitor_s,
                    )
                )

        # 4. Out-of-bed absence monitoring / alert
        current_absence = 0.0
        absence_start = 0.0
        for seg in segments:
            if seg.state not in ("LYING_IN_BED", "SITTING_ON_BED"):
                if current_absence == 0.0:
                    absence_start = seg.start
                current_absence += seg.duration_s

                if current_absence >= self.config.absence_alert_s:
                    decisions.append(
                        AlertDecision(
                            level="ALERT",
                            rule_id="RULE-ABSENCE-ALERT-01",
                            rationale=f"Prolonged absence from bed for {current_absence:.1f}s",
                            timestamp_s=absence_start + self.config.absence_alert_s,
                        )
                    )
                elif current_absence >= self.config.absence_monitor_s:
                    decisions.append(
                        AlertDecision(
                            level="MONITOR",
                            rule_id="RULE-ABSENCE-MONITOR-01",
                            rationale=f"Out of bed absence for {current_absence:.1f}s",
                            timestamp_s=absence_start + self.config.absence_monitor_s,
                        )
                    )
            else:
                current_absence = 0.0

        # Sort chronologically
        decisions.sort(key=lambda d: d.timestamp_s)
        return decisions
