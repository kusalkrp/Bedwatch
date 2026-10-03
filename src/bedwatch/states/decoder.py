"""Constrained Viterbi temporal decoder with transition graph and minimum dwell smoothing."""

import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from bedwatch.config import DecoderConfig
from bedwatch.perception.features import FeatureRecord
from bedwatch.states.rules import CORE_STATES, StateEstimator, StateScores
from bedwatch.states.segments import Segment, merge_contiguous_segments


# Allowed bidirectional transition pairs
ALLOWED_EDGES: Set[Tuple[str, str]] = {
    ("LYING_IN_BED", "SITTING_ON_BED"),
    ("SITTING_ON_BED", "LYING_IN_BED"),
    ("SITTING_ON_BED", "STANDING"),
    ("STANDING", "SITTING_ON_BED"),
    ("STANDING", "WALKING"),
    ("WALKING", "STANDING"),
    ("STANDING", "SITTING_OUTSIDE_BED"),
    ("SITTING_OUTSIDE_BED", "STANDING"),
    ("WALKING", "OUT_OF_BED"),
    ("OUT_OF_BED", "WALKING"),
    ("OUT_OF_BED", "STANDING"),
    ("STANDING", "OUT_OF_BED"),
    # Transitions to LYING_ON_FLOOR
    ("STANDING", "LYING_ON_FLOOR"),
    ("WALKING", "LYING_ON_FLOOR"),
    ("SITTING_ON_BED", "LYING_ON_FLOOR"),
    ("SITTING_OUTSIDE_BED", "LYING_ON_FLOOR"),
    ("LYING_IN_BED", "LYING_ON_FLOOR"),
    ("LYING_ON_FLOOR", "SITTING_OUTSIDE_BED"),
    ("LYING_ON_FLOOR", "STANDING"),
}


class TemporalDecoder:
    """Decodes frame scores into state segments using Viterbi with minimum dwell."""

    def __init__(self, config: DecoderConfig, states: Optional[List[str]] = None):
        self.config = config
        self.states = states or CORE_STATES
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

        # Build transition cost matrix
        self.transition_matrix = np.full((self.n_states, self.n_states), config.penalty.get("disallowed", 20.0))
        for i, s1 in enumerate(self.states):
            for j, s2 in enumerate(self.states):
                if s1 == s2:
                    self.transition_matrix[i, j] = config.penalty.get("self", 0.0)
                elif (s1, s2) in ALLOWED_EDGES:
                    self.transition_matrix[i, j] = config.penalty.get("allowed", 1.0)
                elif s1 == "UNKNOWN" or s2 == "UNKNOWN":
                    self.transition_matrix[i, j] = config.penalty.get("unknown", 2.0)

    def decode(
        self,
        score_records: List[StateScores],
        scene_cuts: Optional[List[bool]] = None,
    ) -> List[Segment]:
        """Run constrained Viterbi pass over score sequence."""
        if not score_records:
            return []

        T = len(score_records)
        K = self.n_states

        # Emission negative log probabilities
        emission = np.zeros((T, K))
        for t_idx, rec in enumerate(score_records):
            for s_idx, state_name in enumerate(self.states):
                prob = max(1e-6, rec.scores.get(state_name, 1e-6))
                emission[t_idx, s_idx] = -math.log(prob)

        # DP table: cost[t, k], backpointer[t, k]
        cost = np.full((T, K), np.inf)
        backpointer = np.zeros((T, K), dtype=int)

        # Initialize t = 0
        cost[0] = emission[0]

        # Forward pass
        for t in range(1, T):
            is_cut = scene_cuts[t] if scene_cuts is not None else False

            if is_cut:
                # Scene cut reset: uniform transition prior
                cost[t] = emission[t] + np.min(cost[t - 1])
                backpointer[t] = np.argmin(cost[t - 1])
            else:
                for k in range(K):
                    total_costs = cost[t - 1] + self.transition_matrix[:, k] + emission[t, k]
                    best_prev = int(np.argmin(total_costs))
                    cost[t, k] = total_costs[best_prev]
                    backpointer[t, k] = best_prev

        # Backward pass
        best_path_indices = [0] * T
        best_path_indices[-1] = int(np.argmin(cost[-1]))
        for t in range(T - 2, -1, -1):
            best_path_indices[t] = backpointer[t + 1, best_path_indices[t + 1]]

        # Convert state indices to frame segments
        decoded_states = [self.states[idx] for idx in best_path_indices]
        timestamps = [rec.t for rec in score_records]

        # Calculate time step dt
        dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.2

        raw_segments = []
        seg_start = timestamps[0]
        cur_state = decoded_states[0]
        conf_acc = [score_records[0].scores[cur_state]]

        for i in range(1, T):
            st = decoded_states[i]
            if st == cur_state:
                conf_acc.append(score_records[i].scores[cur_state])
            else:
                seg_end = timestamps[i]
                raw_segments.append(
                    Segment(
                        start=round(seg_start, 2),
                        end=round(seg_end, 2),
                        state=cur_state,
                        confidence=round(float(np.mean(conf_acc)), 3),
                    )
                )
                seg_start = timestamps[i]
                cur_state = st
                conf_acc = [score_records[i].scores[cur_state]]

        # Append final segment
        final_end = timestamps[-1] + dt
        raw_segments.append(
            Segment(
                start=round(seg_start, 2),
                end=round(final_end, 2),
                state=cur_state,
                confidence=round(float(np.mean(conf_acc)), 3),
            )
        )

        # Minimum dwell smoothing
        smoothed = self._apply_minimum_dwell(raw_segments)
        return merge_contiguous_segments(smoothed)

    def _apply_minimum_dwell(self, segments: List[Segment]) -> List[Segment]:
        """Absorb segments shorter than minimum dwell into valid neighbours."""
        if len(segments) <= 1:
            return segments

        min_dwells = self.config.min_dwell_s
        result = list(segments)
        changed = True

        while changed:
            changed = False
            for i, seg in enumerate(result):
                min_req = min_dwells.get(seg.state, 1.0)
                if seg.duration_s < min_req:
                    # Candidate for absorption
                    if i > 0 and i < len(result) - 1:
                        # Choose neighbor with higher compatibility
                        prev_seg = result[i - 1]
                        next_seg = result[i + 1]
                        if prev_seg.duration_s >= next_seg.duration_s:
                            prev_seg.end = seg.end
                        else:
                            next_seg.start = seg.start
                        result.pop(i)
                        changed = True
                        break
                    elif i > 0:
                        result[i - 1].end = seg.end
                        result.pop(i)
                        changed = True
                        break
                    elif i < len(result) - 1:
                        result[i + 1].start = seg.start
                        result.pop(i)
                        changed = True
                        break

            if changed:
                result = merge_contiguous_segments(result)

        return result
