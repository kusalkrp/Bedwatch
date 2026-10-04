"""Documentation and reporting for key system failure cases."""

from pathlib import Path
from typing import Dict, List, Union


FAILURE_CASES = [
    {
        "id": "FAIL-01",
        "title": "Clothing & Appearance Switches Across Spliced Video Clips",
        "time_range": "100s, 120s, 140s, 180s",
        "expected_failure": "Visual appearance-based ReID models lose the patient track when clothing changes across four distinct outfits.",
        "mitigation": "Tracking relies on spatial and bed-proximity continuity rather than visual appearance embeddings.",
        "outcome": "Primary track successfully re-acquired across all transitions; secondary caregiver tracks separated.",
    },
    {
        "id": "FAIL-02",
        "title": "Camera Pan During Continuous Recording",
        "time_range": "144s - 145s",
        "expected_failure": "Fixed bed polygon coordinates become invalid during camera motion, risking false bed-overlap scores.",
        "mitigation": "View schedule identifies 'moving' interval; bed-relative features are masked during pan.",
        "outcome": "Posture classification remains active; no spurious bed events triggered during pan.",
    },
    {
        "id": "FAIL-03",
        "title": "Curtain Occlusion and Partial Body Visibility",
        "time_range": "146s - 148s",
        "expected_failure": "Severe keypoint occlusion could force an incorrect posture classification or spurious bed exit.",
        "mitigation": "Core visibility check abstains with UNKNOWN. Agent checks neighboring segments (walking before, walking after).",
        "outcome": "Correct abstention to UNKNOWN; false event generation suppressed.",
    },
    {
        "id": "FAIL-04",
        "title": "Abrupt Lighting Step (Dim Light)",
        "time_range": "130.5s - 140s",
        "expected_failure": "Sudden illuminance drop reduces keypoint detection confidence.",
        "mitigation": "Perception stage applies CLAHE contrast enhancement when brightness drops below threshold 0.15.",
        "outcome": "Pose tracking maintained through dim lighting; return-to-bed successfully detected at 134-139s.",
    },
    {
        "id": "FAIL-05",
        "title": "Lying Along Base of Bed (Mattress vs. Floor Ambiguity)",
        "time_range": "184s - 190s",
        "expected_failure": "Patient falls and lies on the carpet right next to the bed frame, which could be misclassified as in-bed.",
        "mitigation": "Mattress polygon strictly bounds top surface. Hips and torso at base of bed fall outside the polygon.",
        "outcome": "State correctly classified as LYING_ON_FLOOR; contextual alert triggers ALERT.",
    },
]


def generate_failure_cases_markdown() -> str:
    """Generate Markdown report of failure case analysis."""
    lines = [
        "# Bedwatch: Failure Case Analysis",
        "",
        "This document details planned and observed failure modes, their root causes, and mitigations.",
        "",
    ]
    for case in FAILURE_CASES:
        lines.extend([
            f"## {case['id']}: {case['title']}",
            f"- **Time Range**: `{case['time_range']}`",
            f"- **Vulnerability / Expected Failure**: {case['expected_failure']}",
            f"- **Mitigation Applied**: {case['mitigation']}",
            f"- **System Outcome**: {case['outcome']}",
            "",
        ])
    return "\n".join(lines)


def write_failure_cases_report(output_path: Union[str, Path]):
    """Write failure case report to markdown file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(generate_failure_cases_markdown())
