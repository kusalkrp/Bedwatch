"""Feature record storage and streaming from JSONL cache."""

import json
from pathlib import Path
from typing import Iterator, List, Union
from bedwatch.perception.features import FeatureRecord


def write_feature_cache(records: List[FeatureRecord], cache_path: Union[str, Path]):
    """Write list of FeatureRecord objects to a JSONL file."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")


def read_feature_cache(cache_path: Union[str, Path]) -> List[FeatureRecord]:
    """Read list of FeatureRecord objects from a JSONL file."""
    records = []
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(FeatureRecord.from_dict(json.loads(line)))
    return records


class FeatureFileSource:
    """Streams FeatureRecord objects from a pre-computed JSONL cache file."""

    def __init__(self, cache_path: Union[str, Path]):
        self.cache_path = Path(cache_path)

    def iter_records(self) -> Iterator[FeatureRecord]:
        """Yield FeatureRecord instances line-by-line."""
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {self.cache_path}")
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield FeatureRecord.from_dict(json.loads(line))
