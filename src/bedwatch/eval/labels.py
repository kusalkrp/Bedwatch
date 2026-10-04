"""Ground truth CSV parser and label loader."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import pandas as pd


@dataclass
class GroundTruthInterval:
    start_s: float
    end_s: float
    state: str
    note: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class GroundTruthEvent:
    event: str
    start_s: float
    confirmed_s: float
    counts_as_event: str  # "yes", "no", "ignore"
    note: str


def load_ground_truth_labels(csv_path: Union[str, Path]) -> List[GroundTruthInterval]:
    """Load ground truth state intervals from CSV."""
    df = pd.read_csv(csv_path)
    intervals = []
    for _, row in df.iterrows():
        intervals.append(
            GroundTruthInterval(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                state=str(row["state"]).strip(),
                note=str(row.get("note", "")),
            )
        )
    return intervals


def load_ground_truth_events(csv_path: Union[str, Path]) -> List[GroundTruthEvent]:
    """Load ground truth events from CSV."""
    df = pd.read_csv(csv_path)
    events = []
    for _, row in df.iterrows():
        events.append(
            GroundTruthEvent(
                event=str(row["event"]).strip(),
                start_s=float(row["start_s"]),
                confirmed_s=float(row["confirmed_s"]),
                counts_as_event=str(row["counts_as_event"]).strip().lower(),
                note=str(row.get("note", "")),
            )
        )
    return events
