# Bedwatch: Failure Case Analysis

This document details planned and observed failure modes, their root causes, and mitigations.

## FAIL-01: Clothing & Appearance Switches Across Spliced Video Clips
- **Time Range**: `100s, 120s, 140s, 180s`
- **Vulnerability / Expected Failure**: Visual appearance-based ReID models lose the patient track when clothing changes across four distinct outfits.
- **Mitigation Applied**: Tracking relies on spatial and bed-proximity continuity rather than visual appearance embeddings.
- **System Outcome**: Primary track successfully re-acquired across all transitions; secondary caregiver tracks separated.

## FAIL-02: Camera Pan During Continuous Recording
- **Time Range**: `144s - 145s`
- **Vulnerability / Expected Failure**: Fixed bed polygon coordinates become invalid during camera motion, risking false bed-overlap scores.
- **Mitigation Applied**: View schedule identifies 'moving' interval; bed-relative features are masked during pan.
- **System Outcome**: Posture classification remains active; no spurious bed events triggered during pan.

## FAIL-03: Curtain Occlusion and Partial Body Visibility
- **Time Range**: `146s - 148s`
- **Vulnerability / Expected Failure**: Severe keypoint occlusion could force an incorrect posture classification or spurious bed exit.
- **Mitigation Applied**: Core visibility check abstains with UNKNOWN. Agent checks neighboring segments (walking before, walking after).
- **System Outcome**: Correct abstention to UNKNOWN; false event generation suppressed.

## FAIL-04: Abrupt Lighting Step (Dim Light)
- **Time Range**: `130.5s - 140s`
- **Vulnerability / Expected Failure**: Sudden illuminance drop reduces keypoint detection confidence.
- **Mitigation Applied**: Perception stage applies CLAHE contrast enhancement when brightness drops below threshold 0.15.
- **System Outcome**: Pose tracking maintained through dim lighting; return-to-bed successfully detected at 134-139s.

## FAIL-05: Lying Along Base of Bed (Mattress vs. Floor Ambiguity)
- **Time Range**: `184s - 190s`
- **Vulnerability / Expected Failure**: Patient falls and lies on the carpet right next to the bed frame, which could be misclassified as in-bed.
- **Mitigation Applied**: Mattress polygon strictly bounds top surface. Hips and torso at base of bed fall outside the polygon.
- **System Outcome**: State correctly classified as LYING_ON_FLOOR; contextual alert triggers ALERT.
