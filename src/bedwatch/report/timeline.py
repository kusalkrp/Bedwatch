"""Timeline report generation matching assignment output format."""

from pathlib import Path
from typing import List, Union
from bedwatch.states.segments import Segment, format_hms


def generate_timeline_text(segments: List[Segment]) -> str:
    """Format segments into contiguous text lines: 'HH:MM:SS - HH:MM:SS STATE'."""
    lines = []
    for seg in segments:
        start_str = format_hms(seg.start)
        end_str = format_hms(seg.end)
        lines.append(f"{start_str} - {end_str} {seg.state}")
    return "\n".join(lines) + "\n"


def write_timeline_file(segments: List[Segment], output_path: Union[str, Path]):
    """Write timeline text to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = generate_timeline_text(segments)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
