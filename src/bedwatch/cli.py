"""Bedwatch Command Line Interface (CLI)."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bedwatch.agent.graph import AgentVerifier
from bedwatch.agent.tools import AgentContextTools
from bedwatch.agent.vlm import LocalVLM
from bedwatch.alerts.policy import AlertPolicy
from bedwatch.config import load_config
from bedwatch.eval.failure_cases import write_failure_cases_report
from bedwatch.eval.labels import load_ground_truth_events, load_ground_truth_labels
from bedwatch.eval.metrics import evaluate_bed_events, evaluate_durations, evaluate_state_classification
from bedwatch.events.detector import EventDetector
from bedwatch.report.annotate import render_annotated_video
from bedwatch.report.summary import write_all_reports
from bedwatch.report.timeline import write_timeline_file
from bedwatch.sources.features import read_feature_cache, write_feature_cache
from bedwatch.sources.video import extract_features
from bedwatch.states.decoder import TemporalDecoder
from bedwatch.states.rules import StateEstimator
from bedwatch.states.segments import compute_durations_summary


console = Console()


def cmd_define_bed(args):
    """Interactively define bed polygon on a reference frame and save to views.yaml."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        console.print(f"[bold red]Error:[/bold red] Cannot open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(args.timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        console.print(f"[bold red]Error:[/bold red] Failed to read frame at {args.timestamp}s")
        sys.exit(1)

    points = []

    if args.points:
        import ast
        points = ast.literal_eval(args.points)
        console.print(f"[green]Using provided points:[/green] {points}")
    else:
        # Interactive point selection
        window_name = f"Define Bed Polygon for {args.view} (Click mattress corners, press ENTER to save)"
        display_frame = frame.copy()

        def mouse_cb(event, x, y, flags, param):
            nonlocal display_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([int(x), int(y)])
                cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                if len(points) > 1:
                    cv2.line(display_frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
                cv2.imshow(window_name, display_frame)
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.clear()
                display_frame = frame.copy()
                cv2.imshow(window_name, display_frame)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_cb)
        cv2.imshow(window_name, display_frame)
        console.print(f"[cyan]Click the mattress corners in order. Right click to reset. Press ENTER or 'q' when done.[/cyan]")
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32, ord('q'), ord('Q'), 27):
                break
        cv2.destroyAllWindows()

    if len(points) < 3:
        console.print("[yellow]Warning: At least 3 points required for a polygon. Aborting.[/yellow]")
        return

    # Save to views config YAML
    import yaml
    out_yaml = Path(args.out) if args.out else Path("configs") / f"views_{Path(args.video).stem}.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if out_yaml.exists():
        with open(out_yaml, "r") as f:
            data = yaml.safe_load(f) or {}

    views = data.get("views", {})
    views[args.view] = {
        "description": f"Mattress polygon for {args.view}",
        "bed_polygon": points,
    }
    data["views"] = views

    # Default schedule if not present or single-view
    if "schedule" not in data or not data["schedule"] or len(data.get("schedule", [])) <= 1:
        data["schedule"] = [
            {"start_s": 0.0, "end_s": 999999.0, "view": args.view}
        ]

    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    console.print(Panel(f"[bold green]Successfully saved bed polygon for '{args.view}' to {out_yaml}![/bold green]\nPoints: {points}"))


def cmd_extract_features(args):
    """Run perception stage and cache features to JSONL."""
    cfg = load_config(
        args.config,
        args.profile,
        video_path=args.video,
        views_config=getattr(args, "views_config", None),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{Path(args.video).stem}_features.jsonl"

    console.print(Panel(f"[bold cyan]Extracting Features[/bold cyan]\nVideo: {args.video}\nViews Config: {cfg.views_config_path}\nOutput: {cache_path}"))
    t0 = time.time()

    def progress(t, cur, total):
        if cur % 100 == 0 or cur == total:
            console.print(f"  Frame {cur}/{total} ({t:.1f}s) - {time.time()-t0:.1f}s elapsed")

    records = extract_features(
        video_path=args.video,
        cache_path=cache_path,
        config=cfg,
        progress_callback=progress,
    )
    elapsed = time.time() - t0
    console.print(f"[bold green]Complete![/bold green] Extracted {len(records)} records in {elapsed:.1f}s ({len(records)/elapsed:.1f} fps)")


def cmd_analyze(args):
    """Run reasoning pipeline (states, decoder, agent, events, alerts, reports)."""
    cfg = load_config(
        args.config,
        args.profile,
        video_path=args.video,
        views_config=getattr(args, "views_config", None),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(f"[bold cyan]Bedwatch Analysis[/bold cyan]\nVideo: {args.video}\nViews Config: {cfg.views_config_path}\nProfile: [yellow]{cfg.alerts.profile}[/yellow]\nOutput: {out_dir}"))

    # Load or extract features
    if args.cache and Path(args.cache).exists():
        console.print(f"Loading features from cache: {args.cache}")
        records = read_feature_cache(args.cache)
    else:
        cache_path = out_dir / f"{Path(args.video).stem}_features.jsonl"
        if cache_path.exists():
            console.print(f"Loading features from existing output cache: {cache_path}")
            records = read_feature_cache(cache_path)
        else:
            console.print(f"Extracting features from video: {args.video} ...")
            records = extract_features(args.video, cache_path, cfg)

    # 1. State estimation
    estimator = StateEstimator()
    score_records = [estimator.estimate_scores(r) for r in records]
    scene_cuts = [r.scene_cut for r in records]

    # 2. Temporal decoding
    decoder = TemporalDecoder(cfg.decoder)
    segments = decoder.decode(score_records, scene_cuts)

    # 3. Agentic verification loop
    tools = AgentContextTools(segments, records, cfg)
    vlm = LocalVLM(enabled=args.use_vlm)
    verifier = AgentVerifier(cfg.agent, tools, vlm)

    verified_segments = []
    for idx, seg in enumerate(segments):
        v_seg, _ = verifier.verify_segment(seg, idx)
        verified_segments.append(v_seg)

    # 4. Event detection
    event_detector = EventDetector(cfg.events)
    events = event_detector.detect_events(verified_segments, records)

    # Verify events with agent
    verified_events = []
    for ev in events:
        v_ev, _ = verifier.verify_event(ev)
        verified_events.append(v_ev)

    # 5. Alert policy evaluation
    alert_policy = AlertPolicy(cfg.alerts)
    alert_decisions = alert_policy.evaluate_timeline(verified_segments, verified_events)

    # 6. Generate reports
    total_sec = records[-1].t if records else 190.0
    n_exits = len([e for e in verified_events if e.event == "bed_exit"])
    n_returns = len([e for e in verified_events if e.event == "bed_return"])
    n_floors = len([e for e in verified_events if e.event == "floor_lying"])

    summary = compute_durations_summary(
        segments=verified_segments,
        total_observation_sec=total_sec,
        bed_exit_count=n_exits,
        bed_return_count=n_returns,
        floor_event_count=n_floors,
    )

    write_timeline_file(verified_segments, out_dir / "timeline.txt")
    write_all_reports(summary, verified_events, verifier.traces, out_dir)
    write_failure_cases_report(out_dir / "failure_cases.md")

    # Render annotated video if requested
    if args.render_video:
        console.print("Rendering annotated video (this may take ~1-2 min)...")
        render_annotated_video(
            video_path=args.video,
            output_path=out_dir / "annotated.mp4",
            records=records,
            segments=verified_segments,
            events=verified_events,
            config=cfg,
            max_frames=args.max_render_frames,
        )
        console.print(f"[green]Saved annotated video to {out_dir / 'annotated.mp4'}[/green]")

    # Print summary tables
    t_summary = Table(title="Activity Duration Summary")
    t_summary.add_column("Activity State", style="cyan")
    t_summary.add_column("Duration (sec)", justify="right")
    t_summary.add_column("Formatted", justify="right")
    for k, v in summary["activity_duration_sec"].items():
        t_summary.add_row(k.upper(), str(v), summary["activity_summary_formatted"][k])
    console.print(t_summary)

    t_events = Table(title="Detected Bed & Safety Events")
    t_events.add_column("Event", style="bold")
    t_events.add_column("Start Time")
    t_events.add_column("Confirmed")
    t_events.add_column("Confidence", justify="right")
    t_events.add_column("Decision", style="magenta")
    for ev in verified_events:
        t_events.add_row(ev.event, ev.start_time, ev.confirmed_time, f"{ev.confidence:.2f}", ev.decision)
    console.print(t_events)


def cmd_evaluate(args):
    """Evaluate predicted outputs against hand-labeled ground truth."""
    pred_dir = Path(args.pred)
    summary_file = pred_dir / "summary.json"
    events_file = pred_dir / "events.json"
    timeline_file = pred_dir / "timeline.txt"

    if not summary_file.exists() or not events_file.exists():
        console.print(f"[bold red]Error:[/bold red] Predictions not found in {pred_dir}")
        sys.exit(1)

    with open(summary_file) as f:
        pred_summary = json.load(f)
    with open(events_file) as f:
        pred_events_raw = json.load(f)

    # Reconstruct events and segments
    gt_intervals = load_ground_truth_labels(args.gt)
    gt_events = load_ground_truth_events(args.events)

    from bedwatch.events.confidence import Event
    from bedwatch.states.segments import Segment

    pred_events = [
        Event(
            event=e["event"],
            start_time=e["start_time"],
            confirmed_time=e["confirmed_time"],
            start_s=float(e["start_time"].split(":")[-1]) + float(e["start_time"].split(":")[-2]) * 60,
            confirmed_s=float(e["confirmed_time"].split(":")[-1]) + float(e["confirmed_time"].split(":")[-2]) * 60,
            previous_state=e["previous_state"],
            current_state=e["current_state"],
            confidence=float(e["confidence"]),
            decision=e["decision"],
            trace_id=e["trace_id"],
        )
        for e in pred_events_raw
    ]

    # Reconstruct segments from timeline.txt
    pred_segments = []
    with open(timeline_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                # 00:00:00 - 00:00:25 LYING_IN_BED
                t1_parts = parts[0].split(":")
                t2_parts = parts[2].split(":")
                s1 = int(t1_parts[0]) * 3600 + int(t1_parts[1]) * 60 + float(t1_parts[2])
                s2 = int(t2_parts[0]) * 3600 + int(t2_parts[1]) * 60 + float(t2_parts[2])
                pred_segments.append(Segment(start=s1, end=s2, state=parts[3]))

    # Compute evaluation
    state_res = evaluate_state_classification(pred_segments, gt_intervals, duration_s=190.0)
    event_res = evaluate_bed_events(pred_events, gt_events, tolerance_s=3.0)
    dur_res = evaluate_durations(pred_summary, gt_intervals)

    console.print(Panel("[bold green]Bedwatch Evaluation Results[/bold green]"))

    console.print(f"[bold]State Strict Accuracy:[/bold] {state_res['strict_accuracy']*100:.1f}%")
    console.print(f"[bold]State Tolerant Accuracy (+/-1s):[/bold] {state_res['tolerant_accuracy']*100:.1f}%")
    console.print(f"[bold]State Macro F1 Score:[/bold] {state_res['macro_f1']:.3f}\n")

    t_ev = Table(title="Event Detection Performance (3s window)")
    t_ev.add_column("Event Type")
    t_ev.add_column("GT Count", justify="right")
    t_ev.add_column("Pred Count", justify="right")
    t_ev.add_column("True Positives", justify="right")
    t_ev.add_column("Precision", justify="right")
    t_ev.add_column("Recall", justify="right")
    t_ev.add_column("F1", justify="right")

    for etype, stats in event_res.items():
        t_ev.add_row(
            etype,
            str(stats["ground_truth_count"]),
            str(stats["predicted_count"]),
            str(stats["true_positives"]),
            f"{stats['precision']*100:.1f}%",
            f"{stats['recall']*100:.1f}%",
            f"{stats['f1']:.3f}",
        )
    console.print(t_ev)

    t_dur = Table(title="Activity Duration Accuracy (sec)")
    t_dur.add_column("Activity")
    t_dur.add_column("Ground Truth", justify="right")
    t_dur.add_column("Predicted", justify="right")
    t_dur.add_column("Absolute Error", justify="right")

    for act, stats in dur_res.items():
        t_dur.add_row(
            act.upper(),
            f"{stats['ground_truth_sec']}s",
            f"{stats['predicted_sec']}s",
            f"{stats['abs_error_sec']}s",
        )
    console.print(t_dur)

    eval_out = pred_dir / "eval_results.json"
    with open(eval_out, "w") as f:
        json.dump({"state_metrics": state_res, "event_metrics": event_res, "duration_metrics": dur_res}, f, indent=2)
    console.print(f"\n[green]Saved evaluation results to {eval_out}[/green]")


def cmd_process(args):
    """End-to-end processing pipeline for a new video."""
    import copy
    video_path = Path(args.video)
    video_stem = video_path.stem
    
    # 1. Check if view config exists, otherwise run define-bed
    views_config = Path("configs") / f"views_{video_stem}.yaml"
    if not views_config.exists():
        console.print(Panel(f"[yellow]No bed configuration found for '{video_stem}'. Launching interactive setup.[/yellow]"))
        define_args = copy.copy(args)
        define_args.view = "view_main"
        define_args.timestamp = 5.0
        define_args.points = None
        define_args.out = str(views_config)
        cmd_define_bed(define_args)
        
    out_dir = Path("outputs") / video_stem
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_stem}_features.jsonl"
    
    # 2. Extract features if cache doesn't exist
    if not cache_path.exists():
        console.print(Panel(f"[bold cyan]Running Perception Stage on {video_path.name}[/bold cyan]"))
        extract_args = copy.copy(args)
        extract_args.out = str(cache_dir)
        extract_args.config = None
        extract_args.views_config = str(views_config)
        cmd_extract_features(extract_args)
        
    # 3. Run analysis
    console.print(Panel(f"[bold cyan]Running Analysis Stage on {video_path.name}[/bold cyan]"))
    analyze_args = copy.copy(args)
    analyze_args.cache = str(cache_path)
    analyze_args.out = str(out_dir)
    analyze_args.config = None
    analyze_args.views_config = str(views_config)
    analyze_args.use_vlm = False
    analyze_args.render_video = not args.no_render
    analyze_args.max_render_frames = None
    cmd_analyze(analyze_args)


def main():
    parser = argparse.ArgumentParser(prog="bedwatch", description="Bedwatch Agentic Vision System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # define-bed
    p_bed = subparsers.add_parser("define-bed", help="Define bed mattress polygon on reference frame")
    p_bed.add_argument("--video", required=True, help="Input video path")
    p_bed.add_argument("--view", default="view_main", help="View name (e.g. view_main)")
    p_bed.add_argument("--timestamp", type=float, default=5.0, help="Timestamp in seconds for reference frame")
    p_bed.add_argument("--points", default=None, help="Optional direct list of points '[[x1,y1],...]'")
    p_bed.add_argument("--out", default=None, help="Output views YAML path (defaults to configs/views_<video>.yaml)")

    # extract-features
    p_feat = subparsers.add_parser("extract-features", help="Extract and cache perception features")
    p_feat.add_argument("--video", required=True, help="Input video path")
    p_feat.add_argument("--out", default="cache", help="Output directory for feature cache")
    p_feat.add_argument("--config", default=None, help="Config YAML path")
    p_feat.add_argument("--views-config", default=None, help="Views YAML path")
    p_feat.add_argument("--profile", default="demo", choices=["demo", "production"])

    # analyze
    p_ana = subparsers.add_parser("analyze", help="Run full analysis pipeline")
    p_ana.add_argument("--video", required=True, help="Input video path")
    p_ana.add_argument("--cache", default=None, help="Path to pre-computed JSONL feature cache")
    p_ana.add_argument("--profile", default="demo", choices=["demo", "production"])
    p_ana.add_argument("--out", default="outputs", help="Output directory")
    p_ana.add_argument("--config", default=None, help="Config YAML path")
    p_ana.add_argument("--views-config", default=None, help="Views YAML path")
    p_ana.add_argument("--use-vlm", action="store_true", help="Enable local VLM verification")
    p_ana.add_argument("--render-video", action="store_true", help="Render annotated MP4 video")
    p_ana.add_argument("--max-render-frames", type=int, default=None, help="Limit frames to render")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate predictions against ground truth")
    p_eval.add_argument("--pred", default="outputs", help="Directory with predictions")
    p_eval.add_argument("--gt", default="csv/combined_ground_truth_draft.csv", help="Ground truth states CSV")
    p_eval.add_argument("--events", default="csv/combined_events_draft.csv", help="Ground truth events CSV")

    # process
    p_proc = subparsers.add_parser("process", help="End-to-end processing pipeline for a new video")
    p_proc.add_argument("--video", required=True, help="Input video path")
    p_proc.add_argument("--no-render", action="store_true", help="Skip rendering the annotated video")
    p_proc.add_argument("--profile", default="demo", choices=["demo", "production"], help="Alert profile to use")

    args = parser.parse_args()
    if args.command == "define-bed":
        cmd_define_bed(args)
    elif args.command == "extract-features":
        cmd_extract_features(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "process":
        cmd_process(args)


if __name__ == "__main__":
    main()
