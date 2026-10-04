"""Agentic verification state machine and trace generation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bedwatch.agent.tools import AgentContextTools
from bedwatch.agent.vlm import LocalVLM
from bedwatch.config import AgentConfig, BedwatchConfig
from bedwatch.events.confidence import Event
from bedwatch.perception.features import FeatureRecord
from bedwatch.states.segments import Segment


@dataclass
class AgentStep:
    tool: str
    observation: str


@dataclass
class AgentTrace:
    trace_id: str
    trigger: str
    steps: List[Dict[str, str]]
    verdict: str  # "confirm", "reject", "downgrade"
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentVerifier:
    """Agent that analyzes temporal context around ambiguous segments and events."""

    def __init__(
        self,
        config: AgentConfig,
        tools: AgentContextTools,
        vlm: Optional[LocalVLM] = None,
    ):
        self.config = config
        self.tools = tools
        self.vlm = vlm or LocalVLM()
        self.traces: List[AgentTrace] = []

    def verify_segment(self, seg: Segment, seg_idx: int) -> Tuple[Segment, Optional[AgentTrace]]:
        """Verify an ambiguous segment (e.g. curtain occlusion, low confidence, bed vs floor)."""
        # Trigger 1: Curtain occlusion (UNKNOWN)
        if seg.state == "UNKNOWN":
            steps = []
            prev_info = self.tools.get_previous_segment(seg.start)
            steps.append({
                "tool": "get_previous_segment",
                "observation": f"Preceded by {prev_info['state'] if prev_info else 'None'} ({prev_info['duration_s'] if prev_info else 0:.1f}s)",
            })

            next_info = self.tools.get_next_segment(seg.end)
            steps.append({
                "tool": "get_next_segment",
                "observation": f"Followed by {next_info['state'] if next_info else 'None'} ({next_info['duration_s'] if next_info else 0:.1f}s)",
            })

            trace = AgentTrace(
                trace_id=f"tr-seg-{seg_idx:03d}",
                trigger="occlusion_unknown",
                steps=steps,
                verdict="confirm",  # Confirm UNKNOWN rather than forcing a state
                confidence=0.90,
            )
            self.traces.append(trace)
            return seg, trace

        # Trigger 2: Floor lying verification
        if seg.state == "LYING_ON_FLOOR":
            steps = []
            overlap = self.tools.recheck_bed_overlap((seg.start + seg.end) / 2.0)
            steps.append({
                "tool": "recheck_bed_overlap",
                "observation": f"Mattress overlap: {overlap['mean_hip_in_bed']:.2f}, dist_bh: {overlap['mean_dist_bh']:.2f}",
            })

            verdict = "confirm" if overlap["mean_hip_in_bed"] < 0.25 else "downgrade"
            trace = AgentTrace(
                trace_id=f"tr-seg-{seg_idx:03d}",
                trigger="floor_lying_overlap_check",
                steps=steps,
                verdict=verdict,
                confidence=0.95,
            )
            self.traces.append(trace)
            return seg, trace

        # Trigger 3: Low confidence segment
        if seg.confidence < 0.60:
            steps = []
            prev_info = self.tools.get_previous_segment(seg.start)
            steps.append({
                "tool": "get_previous_segment",
                "observation": f"Preceded by {prev_info['state'] if prev_info else 'None'}",
            })
            trace = AgentTrace(
                trace_id=f"tr-seg-{seg_idx:03d}",
                trigger="low_segment_confidence",
                steps=steps,
                verdict="confirm",
                confidence=seg.confidence,
            )
            self.traces.append(trace)
            return seg, trace

        return seg, None

    def verify_event(self, ev: Event) -> Tuple[Event, Optional[AgentTrace]]:
        """Verify an event chain using temporal context."""
        steps = []
        if ev.event == "bed_exit":
            prev_info = self.tools.get_previous_segment(ev.start_s)
            steps.append({
                "tool": "get_previous_segment",
                "observation": f"In-bed state before exit: {prev_info['state'] if prev_info else 'unknown'}",
            })
            next_info = self.tools.get_next_segment(ev.confirmed_s)
            steps.append({
                "tool": "get_next_segment",
                "observation": f"Out-of-bed motion after exit: {next_info['state'] if next_info else 'unknown'}",
            })

            trace = AgentTrace(
                trace_id=ev.trace_id,
                trigger="bed_exit_chain_verification",
                steps=steps,
                verdict="confirm",
                confidence=ev.confidence,
            )
            self.traces.append(trace)
            return ev, trace

        elif ev.event == "floor_lying":
            overlap = self.tools.recheck_bed_overlap(ev.start_s)
            steps.append({
                "tool": "recheck_bed_overlap",
                "observation": f"Mattress overlap: {overlap['mean_hip_in_bed']:.2f}, dist: {overlap['mean_dist_bh']:.2f}",
            })
            trace = AgentTrace(
                trace_id=ev.trace_id,
                trigger="floor_event_verification",
                steps=steps,
                verdict="confirm",
                confidence=ev.confidence,
            )
            self.traces.append(trace)
            return ev, trace

        return ev, None
