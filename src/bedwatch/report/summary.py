"""JSON report writer for summary, events, and agent traces."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from bedwatch.agent.graph import AgentTrace
from bedwatch.events.confidence import Event


def write_json_report(data: Any, output_path: Union[str, Path]):
    """Write dictionary or list to JSON file with indent=2."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_all_reports(
    summary_dict: Dict[str, Any],
    events: List[Event],
    traces: List[AgentTrace],
    output_dir: Union[str, Path],
):
    """Write summary.json, events.json, traces.json into output_dir."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. summary.json
    write_json_report(summary_dict, out_dir / "summary.json")

    # 2. events.json
    events_data = [ev.to_dict() for ev in events]
    write_json_report(events_data, out_dir / "events.json")

    # 3. traces.json
    traces_data = [tr.to_dict() for tr in traces]
    write_json_report(traces_data, out_dir / "traces.json")
