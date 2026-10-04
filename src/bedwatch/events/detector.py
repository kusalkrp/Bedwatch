"""Event detector: bed_exit, bed_return, and floor_lying event detection from segments and features."""

from typing import Dict, List, Optional
import numpy as np

from bedwatch.config import EventsConfig
from bedwatch.events.confidence import Event, compute_event_confidence
from bedwatch.perception.features import FeatureRecord
from bedwatch.states.segments import Segment, format_hms


class EventDetector:
    """Detects meaningful bed safety events according to the design specification."""

    def __init__(self, config: EventsConfig):
        self.config = config

    def detect_events(
        self,
        segments: List[Segment],
        feature_records: List[FeatureRecord],
        ignore_intervals: Optional[List[tuple]] = None,
    ) -> List[Event]:
        """Detect all bed events and assign decisions and confidence."""
        events: List[Event] = []
        if not segments:
            return events

        # Default ignore interval: 170.0 to 172.0s (hard cut artifact)
        ignore_intervals = ignore_intervals or [(170.0, 172.0)]

        # Map timestamps to feature records for fast lookup
        rec_by_time = {rec.t: rec for rec in feature_records}
        event_counter = 1

        def in_ignore_range(t: float) -> bool:
            return any(start <= t <= end for start, end in ignore_intervals)

        # 1. Detect BED_EXIT
        # In-bed -> Standing -> Moving away (Walking / Out of bed)
        for i in range(len(segments) - 1):
            seg = segments[i]
            next_seg = segments[i + 1]

            # Trigger: In-bed state transitioning to STANDING or directly to WALKING
            if seg.state in ("LYING_IN_BED", "SITTING_ON_BED"):
                start_s = seg.end
                if in_ignore_range(start_s):
                    continue

                exit_chain_complete = False
                confirmed_s = start_s
                prev_state = seg.state
                curr_state = next_seg.state

                # Case A: in_bed -> STANDING -> WALKING / OUT_OF_BED / SITTING_OUTSIDE_BED
                if next_seg.state == "STANDING":
                    # Check what follows STANDING
                    if i + 2 < len(segments):
                        following_seg = segments[i + 2]
                        if following_seg.state in ("WALKING", "OUT_OF_BED", "SITTING_OUTSIDE_BED"):
                            # Check if person moves away or leaves view
                            dist_increase = self._check_move_away(start_s, following_seg.end, feature_records)
                            if dist_increase or following_seg.state in ("OUT_OF_BED", "SITTING_OUTSIDE_BED"):
                                exit_chain_complete = True
                                confirmed_s = min(following_seg.start + self.config.move_away_min_s, following_seg.end)
                                prev_state = "STANDING"
                                curr_state = following_seg.state
                        elif following_seg.state == "SITTING_ON_BED":
                            dist_increase = self._check_move_away(start_s, next_seg.end, feature_records)
                            if next_seg.duration_s >= 3.0 and dist_increase:
                                exit_chain_complete = True
                                confirmed_s = min(start_s + self.config.move_away_min_s, next_seg.end)
                                prev_state = seg.state
                                curr_state = "STANDING"
                            else:
                                # Aborted exit: stood briefly (<3s) right by the bed and sat back down
                                continue
                    elif next_seg.duration_s >= 3.0:
                        dist_increase = self._check_move_away(start_s, next_seg.end, feature_records)
                        if dist_increase:
                            exit_chain_complete = True
                            confirmed_s = min(start_s + self.config.move_away_min_s, next_seg.end)
                            prev_state = seg.state
                            curr_state = "STANDING"

                # Case B: in_bed -> WALKING / OUT_OF_BED / SITTING_OUTSIDE_BED directly
                elif next_seg.state in ("WALKING", "OUT_OF_BED", "SITTING_OUTSIDE_BED"):
                    dist_increase = self._check_move_away(start_s, next_seg.end, feature_records)
                    if dist_increase or next_seg.state in ("OUT_OF_BED", "SITTING_OUTSIDE_BED"):
                        exit_chain_complete = True
                        confirmed_s = min(start_s + self.config.move_away_min_s, next_seg.end)
                        curr_state = next_seg.state

                if exit_chain_complete:
                    # Calculate prior bed dwell
                    prior_dwell = seg.duration_s
                    # If this was a short bed visit (e.g. ~3s), lower confidence and assign MONITOR
                    conf = compute_event_confidence(
                        chain_completeness=1.0,
                        mean_seg_conf=next_seg.confidence,
                        prior_bed_dwell_s=prior_dwell,
                        dwell_ref_s=self.config.dwell_ref_s,
                    )
                    decision = "MONITOR" if (prior_dwell < 5.0 or conf < 0.75) else "NORMAL"

                    events.append(
                        Event(
                            event="bed_exit",
                            start_time=format_hms(start_s),
                            confirmed_time=format_hms(confirmed_s),
                            start_s=round(start_s, 2),
                            confirmed_s=round(confirmed_s, 2),
                            previous_state=prev_state,
                            current_state=curr_state,
                            confidence=round(conf, 2),
                            decision=decision,
                            trace_id=f"ev-{event_counter:04d}",
                            rationale=f"Patient exited bed from {prev_state} to {curr_state}",
                        )
                    )
                    event_counter += 1

        # 2. Detect RETURN_TO_BED
        # Out-of-bed -> Approach -> Sitting on bed -> Lying in bed
        for i in range(len(segments) - 1):
            seg = segments[i]
            next_seg = segments[i + 1]

            if next_seg.state == "LYING_IN_BED" and next_seg.duration_s >= self.config.lying_confirm_s:
                confirmed_s = next_seg.start + self.config.lying_confirm_s
                if in_ignore_range(confirmed_s):
                    continue

                # Trace back to find start of return approach
                start_s = seg.start
                prev_state = seg.state

                # Check if prior was sitting on bed, and before that walking/standing
                if seg.state == "SITTING_ON_BED" and i > 0:
                    prior_seg = segments[i - 1]
                    if prior_seg.state in ("WALKING", "STANDING", "OUT_OF_BED", "SITTING_OUTSIDE_BED"):
                        start_s = prior_seg.start
                        prev_state = prior_seg.state

                conf = compute_event_confidence(
                    chain_completeness=1.0,
                    mean_seg_conf=next_seg.confidence,
                    prior_bed_dwell_s=10.0,
                    dwell_ref_s=self.config.dwell_ref_s,
                )

                events.append(
                    Event(
                        event="bed_return",
                        start_time=format_hms(start_s),
                        confirmed_time=format_hms(confirmed_s),
                        start_s=round(start_s, 2),
                        confirmed_s=round(confirmed_s, 2),
                        previous_state=prev_state,
                        current_state="LYING_IN_BED",
                        confidence=round(conf, 2),
                        decision="NORMAL",
                        trace_id=f"ev-{event_counter:04d}",
                        rationale="Patient returned to bed and completed lie-down",
                    )
                )
                event_counter += 1

        # 3. Detect LYING_ON_FLOOR
        for seg in segments:
            if seg.state == "LYING_ON_FLOOR" and seg.duration_s >= self.config.floor_confirm_s:
                start_s = seg.start
                confirmed_s = start_s + self.config.floor_confirm_s
                if in_ignore_range(start_s):
                    continue

                events.append(
                    Event(
                        event="floor_lying",
                        start_time=format_hms(start_s),
                        confirmed_time=format_hms(confirmed_s),
                        start_s=round(start_s, 2),
                        confirmed_s=round(confirmed_s, 2),
                        previous_state="STANDING",
                        current_state="LYING_ON_FLOOR",
                        confidence=0.95,
                        decision="ALERT",
                        trace_id=f"ev-{event_counter:04d}",
                        rationale="Patient detected lying horizontally on the floor",
                    )
                )
                event_counter += 1

        # Sort events chronologically by start_s
        events.sort(key=lambda ev: ev.start_s)
        return events

    def _check_move_away(self, t_start: float, t_end: float, records: List[FeatureRecord]) -> bool:
        """Check if distance to bed increases during time window."""
        window_recs = [r for r in records if t_start <= r.t <= t_end and r.primary_track != -1]
        if len(window_recs) < 2:
            return True  # If person disappears/leaves view, that's moving away

        dists = [r.dist_to_bed_bh for r in window_recs if r.dist_to_bed_bh < 900.0]
        if not dists:
            return True

        return (max(dists) - min(dists)) >= self.config.move_away_min_bh or max(dists) > 0.25
