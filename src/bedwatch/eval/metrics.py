"""Evaluation metrics for state classification, bed events, and activity durations."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from bedwatch.eval.labels import GroundTruthEvent, GroundTruthInterval
from bedwatch.events.confidence import Event
from bedwatch.states.segments import Segment


def evaluate_state_classification(
    pred_segments: List[Segment],
    gt_intervals: List[GroundTruthInterval],
    duration_s: float = 190.0,
    sample_hz: float = 1.0,
) -> Dict[str, Any]:
    """Sample predictions and ground truth on a 1 Hz grid and compute metrics."""
    time_points = np.arange(0.5, duration_s, 1.0 / sample_hz)

    def get_gt_at(t: float) -> str:
        for itv in gt_intervals:
            if itv.start_s <= t < itv.end_s:
                return itv.state
        return "UNKNOWN"

    def get_pred_at(t: float) -> str:
        for seg in pred_segments:
            if seg.start <= t < seg.end:
                return seg.state
        return "UNKNOWN"

    gt_labels = [get_gt_at(t) for t in time_points]
    pred_labels = [get_pred_at(t) for t in time_points]

    # Strict accuracy
    strict_correct = sum(1 for p, g in zip(pred_labels, gt_labels) if p == g)
    strict_acc = strict_correct / len(time_points) if time_points.size > 0 else 0.0

    # Tolerant accuracy (boundary matching +/- 1s)
    tolerant_correct = 0
    for i, t in enumerate(time_points):
        p = pred_labels[i]
        g = gt_labels[i]
        if p == g:
            tolerant_correct += 1
        else:
            # Check adjacent 1s window in GT
            near_gt = {get_gt_at(t - 1.0), get_gt_at(t + 1.0)}
            if p in near_gt:
                tolerant_correct += 1
    tolerant_acc = tolerant_correct / len(time_points) if time_points.size > 0 else 0.0

    # Unique labels present
    unique_labels = sorted(list(set(gt_labels) | set(pred_labels)))

    macro_f1 = float(f1_score(gt_labels, pred_labels, labels=unique_labels, average="macro", zero_division=0))
    conf_mat = confusion_matrix(gt_labels, pred_labels, labels=unique_labels)

    # Find top confused pairs
    confused_pairs = []
    for i, true_lbl in enumerate(unique_labels):
        for j, pred_lbl in enumerate(unique_labels):
            if i != j and conf_mat[i, j] > 0:
                confused_pairs.append({
                    "true_state": true_lbl,
                    "pred_state": pred_lbl,
                    "count_sec": int(conf_mat[i, j]),
                })
    confused_pairs.sort(key=lambda x: x["count_sec"], reverse=True)

    return {
        "strict_accuracy": round(float(strict_acc), 4),
        "tolerant_accuracy": round(float(tolerant_acc), 4),
        "macro_f1": round(macro_f1, 4),
        "unique_states": unique_labels,
        "confusion_matrix": conf_mat.tolist(),
        "top_confused_pairs": confused_pairs[:5],
    }


def evaluate_bed_events(
    pred_events: List[Event],
    gt_events: List[GroundTruthEvent],
    tolerance_s: float = 3.0,
) -> Dict[str, Any]:
    """Evaluate event precision, recall, and false detections with 3s matching."""
    true_gt = [g for g in gt_events if g.counts_as_event == "yes"]
    neg_gt = [g for g in gt_events if g.counts_as_event == "no"]

    event_types = ["bed_exit", "bed_return", "floor_lying"]
    results = {}

    for etype in event_types:
        gt_subset = [g for g in true_gt if g.event == etype]
        pred_subset = [p for p in pred_events if p.event == etype]

        matched_gt = set()
        matched_pred = set()

        for p_idx, p in enumerate(pred_subset):
            for g_idx, g in enumerate(gt_subset):
                if g_idx not in matched_gt and abs(p.start_s - g.start_s) <= tolerance_s:
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)
                    break

        tp = len(matched_gt)
        fp = len(pred_subset) - len(matched_pred)
        fn = len(gt_subset) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[etype] = {
            "ground_truth_count": len(gt_subset),
            "predicted_count": len(pred_subset),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1": round(float(f1), 3),
        }

    return results


def evaluate_durations(
    pred_summary: Dict[str, Any],
    gt_intervals: List[GroundTruthInterval],
) -> Dict[str, Any]:
    """Compare predicted duration vs ground truth duration per activity."""
    gt_durations: Dict[str, float] = {}
    for itv in gt_intervals:
        k = itv.state.lower()
        gt_durations[k] = gt_durations.get(k, 0.0) + itv.duration_s

    pred_durations = pred_summary.get("activity_duration_sec", {})
    comparison = {}

    for state, gt_sec in gt_durations.items():
        pred_sec = pred_durations.get(state, 0)
        abs_err = abs(pred_sec - round(gt_sec))
        comparison[state] = {
            "ground_truth_sec": round(gt_sec),
            "predicted_sec": pred_sec,
            "abs_error_sec": abs_err,
        }

    return comparison
