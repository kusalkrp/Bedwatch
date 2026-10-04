# Bedwatch: Agentic Vision System for Elderly Care & Bed Safety

Bedwatch is an agentic AI + Computer Vision system designed to analyze continuous indoor video of an elderly resident. It monitors resident safety by answering four fundamental operational questions:
1. **Activity Timeline**: What is the person doing over time?
2. **Bed Safety Events**: When do they exit or return to bed?
3. **Temporal Accounting**: How much time is spent in each activity state (and total time in bed vs. out of bed)?
4. **Contextual Alerts**: Does an event or situation warrant normal logging, active monitoring, or an immediate alert?

The system is built with open-source models (YOLO11-pose with ByteTrack, OpenCV, SciPy, Scikit-Learn) and runs entirely locally on a single consumer GPU (**NVIDIA RTX 3060, 6 GB VRAM**), with zero external API dependencies.

---

## 1. System Architecture

Bedwatch strictly decouples **perception** (pixels $\rightarrow$ per-frame geometric features) from **reasoning** (features $\rightarrow$ state timeline, event chains, and alert decisions).

```mermaid
flowchart TD
    SRC["Frame Source<br/>(og.mp4 / HEVC)"] --> CUT["Scene Cut & Lighting Detector<br/>(Histogram + CLAHE)"]
    CUT --> POSE["Perception Stage<br/>(YOLO11-Pose + ByteTrack)"]
    POSE --> POLY["Bed Geometry Engine<br/>(Per-view Mattress Polygons)"]
    POLY --> CACHE[("Feature Cache<br/>(og_features.jsonl)")]
    
    CACHE --> EST["State Estimator<br/>(Soft Scores for 8 States)"]
    EST --> VITERBI["Temporal Decoder<br/>(Constrained Viterbi + Min Dwell)"]
    VITERBI --> EVT["Event Detector<br/>(Bed Exit, Return, Floor Lying)"]
    
    VITERBI <--> AGENT["Agent Verifier<br/>(Temporal Context & Tools)"]
    EVT <--> AGENT
    
    EVT --> ALERT["Alert Policy Engine<br/>(NORMAL / MONITOR / ALERT)"]
    ALERT --> OUT["Structured Deliverables<br/>(timeline.txt, summary.json, events.json, annotated.mp4)"]
    OUT --> EVAL["Ground Truth Evaluator<br/>(Strict/Tolerant Accuracy, Confusion Matrix)"]
```

### Architectural Principles
- **Decoupled Perception & Reasoning**: The perception stage processes raw video frames into standardized JSONL `FeatureRecord`s. The downstream reasoning pipeline is 100% CPU-executable and runs in seconds, enabling instant iteration and simulation without reprocessing heavy video frames.
- **Geometry First, VLM Last**: Pose keypoints, bounding box aspect ratios, and polygon distances handle the vast majority of frames explainably and efficiently without heavy foundation models. A local VLM is loaded only as a bounded fallback for ambiguous keyframes.
- **Explicit Temporal Modeling**: Avoids frame-by-frame flickering by decoding state scores with a **constrained Viterbi pass** over an allowed transition graph, supplemented by per-state **minimum dwell time smoothing**.
- **First-Class Abstention**: When evidence is obscured (e.g. behind curtains or severe occlusion), the system outputs `UNKNOWN` rather than forcing an inaccurate guess.

---

## 2. Activity State Taxonomy

Bedwatch tracks **8 discrete activity states**:

| State | Definition & Evidence | Bed Status |
|---|---|---|
| `LYING_IN_BED` | Horizontal posture ($\text{torso angle} > 48^\circ$, or $>38^\circ$ with aspect $> 1.15$) with hips and torso inside the mattress surface polygon. | In Bed |
| `SITTING_ON_BED` | Upright or reclined seated posture on mattress top surface ($\text{thigh angle} < 72^\circ$ or seated aspect); stationary speed. | In Bed |
| `SITTING_OUTSIDE_BED`| Seated posture off the bed (e.g. armchair near window, $\text{thigh angle} < 72^\circ$, compressed vertical ratio, or seated aspect). | Out of Bed |
| `STANDING` | Upright posture ($\text{torso angle} < 25^\circ$) with extended vertical legs ($\text{thigh angle} \ge 78^\circ$, narrow aspect $\le 0.38$, $\text{hip-to-ankle ratio} \ge 0.41$). | Out of Bed |
| `WALKING` | Upright traveling posture with sustained locomotion ($\text{speed} \ge 0.18\text{ bh/s}$). | Out of Bed |
| `OUT_OF_BED` | Resident not visible in camera view after having exited the bed area. | Out of Bed |
| `LYING_ON_FLOOR` | Horizontal posture with hips and torso completely outside mattress surface; low vertical position. | Out of Bed |
| `UNKNOWN` | Person present but keypoints occluded or unreadable (e.g. curtain occlusion). | Out of Bed |

> **Why add `LYING_ON_FLOOR`?**  
> Clinical safety requires differentiating normal rest (`LYING_IN_BED`) from a potential collapse (`LYING_ON_FLOOR`). Adding this state enables an immediate, uncompromised `ALERT` decision without needing ad-hoc heuristics.

---

## 3. Allowed Transition Graph & Viterbi Decoding

To enforce physically possible movements, the temporal decoder applies Viterbi pathfinding over an allowed transition graph:

```mermaid
stateDiagram-v2
    [*] --> LYING_IN_BED
    LYING_IN_BED --> SITTING_ON_BED
    SITTING_ON_BED --> LYING_IN_BED
    SITTING_ON_BED --> STANDING
    STANDING --> SITTING_ON_BED
    STANDING --> WALKING
    WALKING --> STANDING
    STANDING --> SITTING_OUTSIDE_BED
    SITTING_OUTSIDE_BED --> STANDING
    WALKING --> OUT_OF_BED
    OUT_OF_BED --> WALKING
    OUT_OF_BED --> STANDING
    STANDING --> LYING_ON_FLOOR
    WALKING --> LYING_ON_FLOOR
    SITTING_ON_BED --> LYING_ON_FLOOR
    LYING_ON_FLOOR --> STANDING
    LYING_ON_FLOOR --> SITTING_OUTSIDE_BED
```
*Transitions directly from `LYING_ON_FLOOR` to `LYING_IN_BED` are strictly disallowed (a person getting up from the floor must pass through standing or sitting).*

---

## 4. Bed Exit & Return Event Detection

Events are detected from contiguous state chains and geometric displacement:
- **`BED_EXIT`**:
  $$\text{In Bed (Lying / Sitting)} \longrightarrow \text{Standing} \longrightarrow \text{Moving away from bed / Out of bed}$$
  - Confirmed once distance to bed increases by $\ge 0.5\text{ body heights}$ or person leaves view.
  - Negative situations suppressed: turning in bed, sitting up, brief stand-and-sit-back, and aborted exits (one step then sitting back down).
- **`RETURN_TO_BED`**:
  $$\text{Out of bed} \longrightarrow \text{Approach bed} \longrightarrow \text{Sit on bed} \longrightarrow \text{Lie down in bed}$$
  - Confirmed after lying down on mattress for $\ge 2.0\text{s}$.
  - Negative situations suppressed: visiting bed edge without lying down (false return attempt).
- **`FLOOR_LYING`**:
  $$\text{Any non-floor state} \longrightarrow \text{LYING\_ON\_FLOOR}$$
  - Confirmed after $\ge 2.0\text{s}$ on floor $\implies$ Triggers immediate **`ALERT`**.

---

## 5. Agentic Verification Loop

When temporal ambiguity arises, an agentic verification loop gathers context before confirming states or events:

```mermaid
flowchart LR
    SCAN["Scan Segment / Event"] --> CHECK{"Ambiguity?<br/>(Low conf, UNKNOWN,<br/>floor vs bed)"}
    CHECK -- No --> PASS["Confirm State / Event"]
    CHECK -- Yes --> PLAN["Plan Cheapest Tool"]
    PLAN --> ACT["Execute Tool"]
    ACT --> UPDATE["Update Evidence"]
    UPDATE --> DECIDE{"Decided or<br/>Budget Exhausted?"}
    DECIDE -- No --> PLAN
    DECIDE -- Yes --> TRACE["Emit AgentTrace & Verdict"]
```

### Bounded Agent Tools
1. `get_previous_segment(t, window_s)`: Inspects what the resident was doing immediately prior (e.g. verifies whether the resident was in bed before standing).
2. `get_next_segment(t, window_s)`: Inspects succeeding motion (e.g. verifies if walking followed a brief curtain occlusion at 146–148s).
3. `recheck_bed_overlap(t)`: Re-samples mattress polygon containment across adjacent frames to disambiguate bed vs. floor lying.
4. `vlm_describe(keyframes)`: Bounded local vision-language query on keyframes if geometric evidence is borderline.

---

## 6. Contextual Alert Policy

The system produces three discrete safety decisions:

| Level | Condition | Clinical / Operational Rationale |
|---|---|---|
| **`NORMAL`** | Routine daily activities: lying, sitting, standing, walking, seated in chair, or brief bed departures. | Expected daily living patterns. |
| **`MONITOR`** | - Sitting on bed edge $> \text{sit\_edge\_monitor\_s}$<br/>- Out of bed absence $> \text{absence\_monitor\_s}$<br/>- Unconfirmed state $> \text{unknown\_monitor\_s}$<br/>- Low-confidence bed exit (brief bed stay) | Early warning: edge sitting may signal dizziness, weakness, or hesitation before standing. |
| **`ALERT`** | - `LYING_ON_FLOOR` confirmed $> \text{floor\_alert\_s}$<br/>- Out-of-bed absence $> \text{absence\_alert\_s}$<br/>- Resident lost from view $> \text{lost\_alert\_s}$ | Immediate safety intervention required for suspected falls or wander risk. |

### Alert Policy Profiles: Unvalidated Production Placeholders vs. Scaled Demo Profile

> **Note on Thresholds**: The production parameters listed below (e.g. 15 s floor lying, 300 s edge-sitting, 600 s room absence) are **unvalidated heuristic placeholders** reflecting operational assumptions rather than clinically validated criteria. They must be calibrated against institutional elder-care protocols and clinical risk guidelines before real-world deployment.
>
> For evaluating the 190-second benchmark video (`og.mp4`), Bedwatch uses a scaled `demo` profile (`configs/demo.yaml`) that scales temporal thresholds so rapid events (such as the 6-second fall at 184–190 s) trigger appropriate decisions within the short recording window:

| Parameter | Production (Unvalidated Placeholder) | Demo (Scaled for Video) | Operational Rationale |
|---|---|---|---|
| `floor_alert_s` | 15s | 3s | Enables the 6s floor event (184–190s) to trigger `ALERT`. |
| `sit_edge_monitor_s` | 300s | 10s | Scaled by 30x for rapid event demonstration. |
| `absence_monitor_s` | 600s | 20s | Early warning for prolonged room absence. |
| `absence_alert_s` | 1200s | 40s | Critical alarm for unreturned room departure. |

---

## 7. Installation & Quick Start

### Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.12)
- NVIDIA GPU with CUDA support (tested on RTX 3060 Laptop GPU, 6 GB VRAM) or CPU
- Docker (optional, for containerized execution)
- Windows / Linux / macOS

### Option A: Local Installation

```bash
# Clone the repository
git clone https://github.com/kusalkrp/Bedwatch.git
cd Bedwatch

# (Optional) Create and activate virtual environment
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

# Install package in editable mode
pip install -e .
```

### Option B: Docker Container

A production `Dockerfile` is included for fully reproducible containerized runs.

> ⚠️ **Important GUI Note for New Videos:** The `process` command interactively pops up a window for you to draw the bed polygon on the first frame of a *new* video. Because Docker containers cannot natively open UI windows on your host monitor without complex X11 display mapping, you must either:
> 1. Run the `define-bed` command natively on your host machine *first* to generate the `configs/views_<video>.yaml` configuration, OR
> 2. Process videos that already have an existing configuration (like `og.mp4` or `test_3.mp4`).

```bash
# Build the Docker image
docker build -t bedwatch:latest .

# 1. Process a video end-to-end (GPU accelerated, assuming views config exists)
docker run --rm --gpus all \
  -v "${PWD}/cache:/app/cache" \
  -v "${PWD}/outputs:/app/outputs" \
  -v "${PWD}/my_video.mp4:/app/my_video.mp4" \
  bedwatch:latest process --video my_video.mp4 --out outputs/

# 2. Run ground-truth evaluation in container
docker run --rm -v "${PWD}/outputs:/app/outputs" bedwatch:latest evaluate

# 3. Run fast CPU reasoning pipeline on cached features
docker run --rm \
  -v "${PWD}/outputs:/app/outputs" \
  -v "${PWD}/cache:/app/cache" \
  bedwatch:latest analyze --video og.mp4 --cache cache/og_features.jsonl --profile demo --out outputs/
```

---

### Running the CLI (Native)

#### 🚀 Primary Command: Process Any New Video End-to-End
To analyze **any new video**, use the unified `process` command. It runs perception, temporal decoding, agent verification, event detection, and video rendering in a single command:

```bash
# Process any video (creates outputs/<video_stem>/ with timeline, summary, events, and annotated video)
python -m bedwatch.cli process --video path/to/patient_video.mp4

# Skip video re-rendering if only telemetry reports are needed:
python -m bedwatch.cli process --video path/to/patient_video.mp4 --no-render
```

**How Bedwatch handles new videos:**
1. **Bed Geometry Auto-Discovery:** Bedwatch automatically looks for a camera-specific config (`configs/views_<video_stem>.yaml`). If one doesn't exist, it uses `configs/views.yaml` or interactively launches a 4-point bed polygon calibration on the first frame.
2. **Feature Caching:** Extracted pose and bounding box features are cached in `cache/<video_stem>_features.jsonl` so re-running analysis or tuning thresholds takes < 2 seconds.
3. **Structured Outputs:** Results are saved directly into `outputs/<video_stem>/` (`timeline.txt`, `summary.json`, `events.json`, `annotated.mp4`).

---

#### Modular Pipeline Subcommands

If you prefer to run pipeline stages individually or inspect intermediate steps:

#### 1. Calibrate Bed Polygon for New Views (One-time Setup)
Calibrate mattress polygon coordinates for a new camera angle or room layout:
```bash
python -m bedwatch.cli define-bed --video new_video.mp4 --view view_main --timestamp 5.0
```

#### 2. Extract Features (GPU Perception Stage)
Decodes raw video, runs YOLO11-pose with ByteTrack, and streams geometry features to JSONL:
```bash
python -m bedwatch.cli extract-features --video new_video.mp4 --out cache/ --profile production
```

#### 3. Run Analysis Pipeline (Fast CPU Reasoning)
Runs state estimation, Viterbi temporal decoding, agent verification, event detection, and alert policy in ~2 seconds:
```bash
python -m bedwatch.cli analyze --video og.mp4 --cache cache/og_features.jsonl --profile demo --out outputs/
```

#### 4. Render Annotated Video with Telemetry Overlay
Generates `outputs/annotated.mp4` with pose skeletons, mattress polygon, real-time activity state, and event alerts:
```bash
python -m bedwatch.cli analyze --video og.mp4 --cache cache/og_features.jsonl --profile demo --out outputs/ --render-video
```

#### 5. Evaluate Ground Truth (Benchmark Verification)
Evaluates generated outputs against hand-annotated state labels and event records:
```bash
python -m bedwatch.cli evaluate --pred outputs/ --gt csv/combined_ground_truth.csv --events csv/combined_events.csv
```


### Sample Deliverables (Deliverables Preview)

The pipeline produces four structured deliverables in `outputs/`:

#### 1. Activity Timeline (`outputs/timeline.txt`)
Contiguous time-stamped sequence of patient activity states formatted as `HH:MM:SS - HH:MM:SS STATE`:
```text
00:00:00 - 00:00:22 LYING_IN_BED
00:00:22 - 00:00:45 SITTING_ON_BED
00:00:45 - 00:00:57 STANDING
00:00:57 - 00:01:03 SITTING_OUTSIDE_BED
00:01:03 - 00:01:06 STANDING
00:01:06 - 00:01:08 WALKING
...
00:03:01 - 00:03:05 SITTING_ON_BED
00:03:05 - 00:03:10 LYING_ON_FLOOR
```

#### 2. Detected Bed & Safety Events (`outputs/events.json`)
Structured event entries containing timing, confidence, and contextual alert decision:
```json
[
  {
    "event": "bed_exit",
    "start_time": "00:00:45",
    "confirmed_time": "00:01:00",
    "previous_state": "standing",
    "current_state": "sitting_outside_bed",
    "confidence": 0.84,
    "decision": "NORMAL",
    "trace_id": "ev-0001"
  },
  {
    "event": "bed_return",
    "start_time": "00:01:08",
    "confirmed_time": "00:01:19",
    "previous_state": "standing",
    "current_state": "lying_in_bed",
    "confidence": 0.89,
    "decision": "NORMAL",
    "trace_id": "ev-0005"
  },
  {
    "event": "floor_lying",
    "start_time": "00:03:05",
    "confirmed_time": "00:03:07",
    "previous_state": "standing",
    "current_state": "lying_on_floor",
    "confidence": 0.95,
    "decision": "ALERT",
    "trace_id": "ev-0008"
  }
]
```

#### 3. Temporal Accounting Summary (`outputs/summary.json`)
Aggregated durations per activity state, total in-bed vs. out-of-bed accounting, and event counts:
```json
{
  "observation_duration_sec": 190,
  "activity_duration_sec": {
    "lying_in_bed": 50,
    "sitting_on_bed": 65,
    "sitting_outside_bed": 9,
    "standing": 38,
    "walking": 18,
    "out_of_bed": 2,
    "lying_on_floor": 5,
    "unknown": 3
  },
  "total_in_bed_sec": 115,
  "total_out_of_bed_sec": 75,
  "bed_exit_count": 4,
  "bed_return_count": 3,
  "floor_event_count": 1,
  "longest_out_of_bed_period_sec": 27,
  "final_state": "lying_on_floor"
}
```

---

## 8. Evaluation Results

**Data and caveats.** One 190.07 s AI-generated video (`og.mp4`, one subject, four outfits, two camera views), labelled by hand at one-second resolution and reviewed against the video. All numbers below are **in-sample**: the heuristics, bed ROIs, and thresholds were developed and calibrated directly while inspecting this video. They show what the pipeline does on this footage and are not an estimate of performance on new footage. Events are scored on only 5 exits, 3 returns and 1 floor event, so each missed event moves a percentage by 20 to 33 points. Raw counts are shown next to percentages.

All figures come from one run (`outputs/eval_results.json`, `outputs/summary.json`, `outputs/timeline.txt`, `outputs/events.json`): commit `81845d8`, configuration `configs/demo.yaml` (demo profile).

### 8.1 State classification

| Metric | Value |
|---|---|
| Strict accuracy (1 Hz) | 70.5% (134 of 190 s) |
| Tolerant accuracy (±1 s at boundaries) | 83.2% |
| Macro F1 | 0.707 |

Per-state results (rows are ground truth, from the confusion matrix):

| State | GT (s) | Correct (s) | Recall | Where the rest went |
|---|---|---|---|---|
| LYING_IN_BED | 52 | 50 | 96% | 2 s sitting on bed |
| SITTING_ON_BED | 43 | 40 | 93% | 3 s walking |
| SITTING_OUTSIDE_BED | 9 | 9 | 100% | none |
| STANDING | 28 | 14 | 50% | 11 s sitting on bed, 1 s sitting outside, 1 s unknown, 1 s walking |
| WALKING | 40 | 11 | 28% | 23 s standing, 6 s sitting on bed |
| OUT_OF_BED | 10 | 3 | 30% | 4 s walking, 2 s sitting on bed, 1 s standing |
| UNKNOWN | 2 | 2 | 100% | one extra second predicted unknown (from STANDING) |
| LYING_ON_FLOOR | 6 | 5 | 83% | 1 s sitting on bed |

The two weak states are WALKING and STANDING. Predicted STANDING is only 37% precise (14 of 38 s), because most walking is labelled standing (Failure case F1).

### 8.2 Duration estimation

From `summary.json`. Errors are absolute, in seconds.

| State | Ground truth | Predicted | Error |
|---|---|---|---|
| LYING_IN_BED | 52 s | 50 s | 2 s |
| SITTING_ON_BED | 43 s | 65 s | 22 s |
| STANDING | 28 s | 38 s | 10 s |
| WALKING | 40 s | 18 s | 22 s |
| SITTING_OUTSIDE_BED | 9 s | 9 s | 0 s |
| OUT_OF_BED | 10 s | 2 s | 8 s |
| UNKNOWN | 2 s | 3 s | 1 s |
| LYING_ON_FLOOR | 6 s | 5 s | 1 s |
| Total | 190 s | 190 s | 0 s |

A zero error does not mean every second was right: errors can cancel (see the confusion matrix above). Time in bed was predicted at 115 s (lying 50 s + sitting on bed 65 s) and out of bed at 75 s. `timeline.txt` formats segment boundary timestamps to whole seconds (`HH:MM:SS`), causing cumulative integer-boundary quantization differences of up to 3 s per state compared to the exact floating-point duration aggregates in `summary.json` (for example, summing individual integer segment durations in `timeline.txt` gives 62 s for sitting on bed versus 65 s in `summary.json`).

### 8.3 Bed events

Predicted events are matched one-to-one with ground-truth events. Two matching rules are reported, because the system's "start time" and the ground-truth "start time" are defined slightly differently:

- **Start-time rule:** predicted start within ±3 s of the ground-truth start (the rule in `eval_results.json`).
- **Interval-overlap rule:** the predicted [start, confirmed] interval overlaps the ground-truth interval (touching counts).

| Event | GT | Predicted | Rule | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| bed_exit | 5 | 4 | Start ±3 s | 2 | 2 | 3 | 0.50 | 0.40 | 0.444 |
| bed_exit | 5 | 4 | Interval overlap | 2 | 2 | 3 | 0.50 | 0.40 | 0.444 |
| bed_return | 3 | 3 | Start ±3 s | 0 | 3 | 3 | 0.00 | 0.00 | 0.000 |
| bed_return | 3 | 3 | Interval overlap | 2 | 1 | 1 | 0.67 | 0.67 | 0.667 |
| floor_lying | 1 | 1 | either | 1 | 0 | 0 | 1.00 | 1.00 | 1.000 |

Event by event:

| Ground truth | Predicted | Outcome |
|---|---|---|
| Exit, 41-44 s | Event start 45 s, confirmed 60 s | Late by 1 s at the start, but confirmed 16 s late. Not matched by either rule |
| Exit, 93-95 s | Event start 97 s, confirmed 100 s | Not matched. The patient is already out of frame at 97 s, and a caregiver is at the door (F4) |
| Exit, 123-125 s | none | Missed (F1) |
| Exit, 142-144 s | Event 144-145 s | Matched |
| Exit, 157-159 s | Event 157-163 s, confidence 0.58, MONITOR | Matched |
| Return, 72-77 s | Event 68-79 s | Matched by interval overlap, 4 s early by start |
| Return, 105-117 s | Event 109-119 s | Matched by interval overlap, 4 s late by start |
| Return, 132-139 s | none | Missed (F3) |
| none (hard cut at 170 s) | Return 171-174 s | False positive (F5) |
| Floor lying, 184 s | Event 185-187 s, ALERT | Matched, one second late |

Two of the three "false positive" exits and two of the three "false positive" returns under the start-time rule are late or early detections of real events. The interval-overlap rule was adopted after seeing these results, so the stricter start-time numbers are shown as well.

### 8.4 Alerts and the agent

- `floor_lying` was the only ALERT (1 of 1). Two exits were MONITOR (confidence 0.67 and 0.58), and the rest were NORMAL.
- The verifier produced 7 traces (2 segment checks, 4 exit checks, 1 floor check). All 7 verdicts were "confirm". It never rejected or downgraded a result in this run, and the three returns were not verified. It supplied explanations but did not change any outcome here.

### 8.5 Manual Calibration versus Automatic Detection Comparison

To evaluate how automated bed detection performs relative to human calibration, the full pipeline was run on `og.mp4` under both regimes:
1. **Manual Calibration**: Human-annotated mattress polygon for `view_main` and `view_panned` with explicit camera switch schedule (`configs/views.yaml`).
2. **Automatic Detection**: Single bed polygon automatically extracted by YOLO segmentation without camera-pan scheduling (`outputs/auto_eval/`).

| Metric | Manual Calibration (`views.yaml`) | Automatic Detection (`auto_eval`) | Delta |
|---|---|---|---|
| **Strict Accuracy (1 Hz)** | **70.5%** (134 of 190 s) | **64.2%** (122 of 190 s) | -6.3% |
| **Tolerant Accuracy (±1 s)** | **83.2%** | **76.8%** | -6.4% |
| **Macro F1** | **0.707** | **0.558** | -0.149 |
| **Bed Exits (3 s window)** | 2 of 5 TP (40.0% recall, 50% prec) | 0 of 5 TP (0.0% recall, 0.0% prec) | -40.0% |
| **Bed Returns (3 s window)** | 0 of 3 TP (interval: 2 of 3) | 0 of 3 TP | 0 |
| **Floor Event Detection** | **1 of 1 (100% recall, ALERT)** | **0 of 1 (0.0% recall — MISSED)** | -100% |
| **Floor Event Classification** | **ALERT** (`floor_lying`) | **False NORMAL (`bed_return`)** | Critical Failure |
| **Floor State Duration Error** | 1 s (5 s pred vs 6 s GT) | **6 s (0 s pred vs 6 s GT)** | +5 s |
| **Polygon IoU with `view_main`** | 1.000 | 0.366 (36.6%) | -63.4% |
| **Polygon IoU with `view_panned`**| 1.000 | 0.054 (5.4%) | -94.6% |

#### Key Takeaways from the Comparison:
1. **Side Panel / Skirt Overflow**: A generic YOLO bed mask segments the entire furniture item down to the floor legs/skirting. In `og.mp4`, the auto-polygon extends down to the floor where the patient lands during the 184–190 s fall. Because the patient's hips fall inside this extended polygon (4 of 31 fall frames), the decoder classifies the fall as lying in bed, falsely triggering a `bed_return` (NORMAL) instead of an emergency `floor_lying` (ALERT).
2. **Fixed Polygon on Panned Cameras**: A single fixed polygon cannot adapt to pan/tilt camera motion. At 144–170 s, `og.mp4` pans across the room where the initial polygon retains only 5.4% IoU with the panned bed.
3. **Mitigations Implemented**:
   - **Multi-Frame Consensus Voting**: Samples multiple frames across the video and takes the consensus occupancy mask, preventing patient occlusion notches.
   - **Mattress Elevation Trimming**: Automatically trims the lower 12% of the mask to exclude the floor legs and side rails.
   - **Camera Motion Guard**: Compares background keypoints across checkpoints to detect camera movement and flag a warning when single-view polygons are unreliable.
   - **Bed Unknown Fallback**: If zero beds or multiple conflicting beds are detected, falls back to `bed_polygon: null`, suppresses bed-relative events, and tracks posture states only.

---

## 9. Failure Cases

Observed in this run. Each entry gives the time, predicted versus true state, the evidence, the likely cause, and what would fix it. "Likely" marks hypotheses not yet tested.

### F1. Walking is read as standing (23 of 40 s)

- **Where:** the walk away from the bed at 43-48 s (the first exit), and the walk at 124-127 s (the exit at 123-125 s).
- **Predicted vs true:** `timeline.txt` shows STANDING from 45 s to 57 s, where the ground truth is WALKING, STANDING, WALKING.
- **Impact:** an exit needs movement away from the bed. The first exit is confirmed at 60 s instead of 44 s, and the exit at 123-125 s is never confirmed.
- **Likely cause:** the walking threshold (0.18 body heights per second) is too high for the apparent motion in this wide-angle, high-mounted view, sampled at 5 fps.
- **Fix to try:** log `speed_bh_s` on ground-truth walking frames and set the threshold from that distribution, then check on held-out footage.

### F2. Standing beside the bed is read as sitting on the bed (11 of 28 s)

- **Predicted vs true:** STANDING predicted as SITTING_ON_BED for 11 s.
- **Likely cause:** the mattress polygon is drawn in image space. A person standing on the floor between the bed and the camera has hips and torso that project into the polygon.
- **Fix to try:** keep the mattress-top polygon for lying and sitting, add the bed's floor footprint, and use the ankle positions to decide "on the bed or beside it".

### F3. The dim-light return at 132-139 s is missed

- **Predicted vs true:** `timeline.txt` shows SITTING_ON_BED from 136 s to 144 s. The ground truth is sitting at 136-139 s, lying at 139-141 s, then sitting again.
- **Likely cause:** the generated clip has only about 2 s of lying, the same length as the 2.0 s lying confirmation, and the frames are dim.
- **Fix to try:** test a shorter lying confirmation (1.0 s) and check pose confidence in the dim frames.

### F4. A caregiver at the door is treated as the patient (about 97-103 s)

- **Evidence:** 4 s of ground-truth OUT_OF_BED are predicted as WALKING, and a BED_EXIT event starts at 97 s. The patient left the frame at about 95 s, and the caregiver enters at the door at 97 s. Confirmed on the final `annotated.mp4` at 99 s and 101 s (`outputs/frames/fail_99.jpg` and `outputs/frames/fail_101.jpg`): the tracker attaches the skeleton and bounding box directly to the caregiver.
- **Likely cause:** the tracker adopts any person who appears near the door right after the patient leaves.
- **Fix to try:** treat frames with a second person near the door as ambiguous, and send them to the agent instead of adopting the new track.

### F5. A hard cut creates a false return (171-174 s)

- **Predicted vs true:** a RETURN_TO_BED event (confidence 0.97) right after the scene cut at about 170 s, where the patient stands, sits and lies within 2 s with no approach.
- **Likely cause:** event chains are not closed at scene cuts, and the ground-truth ignore window (170-172 s) is not applied by the evaluator.
- **Fix to try:** reset chain state at a detected cut and apply the ignore window.

### F6. Generalization to New Camera Angles (Limitation)

- **Evidence:** When evaluating entirely new videos using the `process` CLI command, predictions for standing vs. walking and sitting edge vs. sitting outside bed can degrade if the new video has a drastically different camera mount (e.g., top-down ceiling mount vs. wide-angle wall mount).
- **Likely cause:** The temporal and geometric heuristics (like `walking_speed > 0.18 bh/s` or `torso_angle < 25°`) are static configurations currently optimized for the `og.mp4` viewing angle.
- **Fix to try:** Rather than using global fixed thresholds, compute dynamic thresholds relative to the camera's intrinsic perspective or learn these parameters via the HSMM proposed in Future Work.

### What worked

- Curtain occlusion: UNKNOWN at 145-148 s against a ground truth of 146-148 s.
- Floor lying: LYING_ON_FLOOR for 5 of 6 s, event at 185 s, decision ALERT.
- Lying in bed (96%), sitting on the bed (93%) and sitting outside the bed (100%).

### Not verified in this run

- Tracker identity across the four outfit changes: track IDs were not logged, so no claim is made in either direction.
- Camera pan (144-145 s): no spurious event appears in `events.json`, but the exit at 144-145 s falls inside the pan.
- CLAHE in dim light: not compared with and without.

### Evidence frames to attach

```bash
# Extract telemetry verification frames from annotated.mp4
for t in 45 99 101 140 172 185; do
  ffmpeg -y -ss $t -i outputs/annotated.mp4 -frames:v 1 outputs/frames/fail_$t.jpg
done
```

## 10. Conclusions, Limitations & Future Work

Bedwatch successfully demonstrates an agentic, lightweight approach to continuous indoor video monitoring for elderly care. By intentionally decoupling geometric perception from temporal reasoning, the system efficiently handles a 190-second video on a consumer GPU in under 5 seconds of reasoning time. It accurately distinguishes critical safety events (like lying on the floor) from normal resting behaviors (lying in bed) while generating comprehensive, auditable event traces.

### Current Limitations & Difficulties
- **Static Geometric Heuristics**: As noted in Failure Case F1 and F6, relying on fixed angular and speed thresholds (`bh/s`) makes the system brittle to drastically different camera mounting angles.
- **Tracker Identity Swaps**: In environments with caregivers or multiple residents (F4), simple IoU/ByteTrack tracking can erroneously attach to a new person entering the frame if the patient is briefly occluded.
- **Fixed Bed Polygons during Camera Motion**: A static polygon cannot adapt well to PTZ (Pan-Tilt-Zoom) camera movements without background homography tracking.

### Future Work (What We Would Do Differently With More Time)
1. **Learned Semi-Markov Model / Small Transformer**: Replace heuristic Viterbi transition costs with an explicit duration-aware Hidden Semi-Markov Model (HSMM) trained on real-world elder care datasets, removing the need for hand-tuned thresholds.
2. **Dynamic Background Homography**: Implement automatic camera view tracking using background feature matching (e.g., ORB or SIFT) to keep the bed polygon anchored properly even if the camera pans or zooms.
3. **Multi-Camera Association**: Extend patient tracking across multiple rooms (bedroom, hallway, bathroom) using re-identification with spatio-temporal transit graphs.
4. **Edge Deployment**: Quantize YOLO11-pose to TensorRT / ONNX INT8 to run at 30 fps on low-power edge gateways (e.g. NVIDIA Jetson Orin Nano).

---

## License
Apache-2.0
