# Badminton Post-Match Coaching Assistant (BMCA)

## Product Requirements Document (PRD) v2.0

**Status:** Engineering PRD
**Target Users:** Competitive Players, Coaches, Academies, Performance Analysts, Sports Scientists, Parents of Competitive Juniors

---

# 1. Product Vision

Create an AI-powered badminton analyst that converts recorded match videos into:

* Stroke-by-stroke analysis
* Player movement analysis
* Tactical understanding
* Fitness and workload insights
* Coach-grade recommendations

The system should emulate the workflow of an elite badminton performance analyst reviewing match footage.

---

# 2. Design Principles

## Principle 1 – Research Reuse First

Mandatory integration of proven research assets:

* TrackNetV3
* BST (Badminton Stroke Transformer)
* ShuttleSet
* MMAction2 ecosystem

Avoid retraining foundational models unless necessary.

---

## Principle 2 – Shuttle First

Shuttle trajectory is the most important signal.

Signal priority:

1. Shuttle trajectory
2. Hit frame localization
3. Court position
4. Pose information
5. Rally context

---

## Principle 3 – Explainability

Every coaching recommendation must reference measurable evidence.

---

## Principle 4 – Hardware Flexibility

Must support:

* CPU-only execution
* Consumer GPUs
* Cloud deployment

---

# 3. Supported Use Cases

## MVP

### Single Match Analysis

Upload a single match and generate a complete coaching report.

### Player Development Tracking

Compare multiple matches over time.

### Coach Review Workflow

Interactive timeline with synchronized analytics.

---

## Future

### Tournament Analytics

Multi-match tournament analysis.

### Opponent Scouting

Opponent tendency detection and forecasting.

---

# 4. System Architecture

```text
VIDEO

|
V

COURT UNDERSTANDING

|
V

PLAYER UNDERSTANDING

|
V

SHUTTLE UNDERSTANDING

|
V

HIT FRAME LOCALIZATION

|
V

STROKE CLIP GENERATION

|
V

MULTI-MODAL STROKE ENGINE

|
V

RALLY UNDERSTANDING

|
V

TACTICAL STATE ENGINE

|
V

FITNESS ENGINE

|
V

COACHING INTELLIGENCE ENGINE

|
V

REPORT GENERATOR
```

---

# 5. Core Modules

---

# Module 1 – Court Understanding Engine

## Purpose

Generate a stable court coordinate system.

## Inputs

* Match video

## Outputs

```json
{
  "court_lines": [],
  "net_line": [],
  "homography_matrix": []
}
```

## Functional Requirements

Detect:

* Singles sidelines
* Baselines
* Service lines
* Net

Generate:

* Pixel-to-court coordinate transform

## Research Sources

* ShuttleSet
* MonoTrack

---

# Module 2 – Player Understanding Engine

## Purpose

Maintain persistent player identity throughout the match.

## Player Selection

Option A:

* User clicks target player

Option B:

* Select near-side player

Option C:

* Select far-side player

## Tracking

Detect:

* Player A
* Player B

Track across entire match.

## Identity Persistence

### Level 1

Court-side constraint

### Level 2

Color histogram matching

### Level 3

Re-identification embeddings

## Output

```json
{
  "player_id": 1,
  "is_target": true
}
```

---

# Module 3 – Pose Engine

## Purpose

Extract biomechanical information.

## Preferred Models

1. RTMPose
2. ViTPose
3. YOLOv8-Pose

## Keypoints

Required joints:

* Shoulder
* Elbow
* Wrist
* Hip
* Knee
* Ankle

## Temporal Smoothing

Mandatory:

* One-Euro Filter
* Kalman Filter

## Output

Pose sequences for both players.

---

# Module 4 – Shuttle Understanding Engine

## Purpose

Track shuttle throughout the match.

## Engine

TrackNetV3

## Output

```json
{
  "frame": 100,
  "x": 120,
  "y": 340,
  "confidence": 0.99
}
```

---

# Module 5 – 3D Shuttle Reconstruction

## Phase

Phase 2

## Research Basis

MonoTrack

## Purpose

Recover 3D shuttle flight paths.

## Outputs

```json
{
  "x": 1.2,
  "y": 3.4,
  "z": 5.6
}
```

## Derived Metrics

* Apex height
* Flight duration
* Landing zone
* Shot steepness
* Trajectory depth

---

# Module 6 – Hit Frame Localization Engine

## Priority

Highest priority module.

## Purpose

Determine exact shuttle impact frame.

## Signals

* Shuttle trajectory reversal
* Shuttle speed peak
* Minimum shuttle-racket distance
* Wrist velocity peak

## Output

```json
{
  "frame": 23421,
  "confidence": 0.97
}
```

---

# Module 7 – Stroke Clip Builder

## Purpose

Create temporal windows around each hit.

## Default Window

30 frames

## Extended Window

100 frames

## Output

```json
{
  "start_frame": 23000,
  "hit_frame": 23421,
  "end_frame": 23600
}
```

---

# Module 8 – Normalization Contract

## Purpose

Ensure compatibility with BST preprocessing.

## Requirements

Mandatory normalization of:

* Pose coordinates
* Player coordinates
* Shuttle coordinates
* Court coordinates

## Validation

```json
{
  "normalization_passed": true
}
```

Inference must not proceed if validation fails.

---

# Module 9 – Multi-Modal Stroke Engine

## MVP Model

BST-CG-AP

## Inputs

* Pose
* Shuttle trajectory
* Court coordinates
* Rally context

## Future Benchmark Models

* TemPose
* TemPose-TF
* ST-GCN
* 2s-AGCN
* PoseC3D
* Multi-modal Transformer Fusion

## Classification Strategy

### Stage 1

Overhead vs Underarm

### Stage 2

Stroke subtype

## Output

```json
{
  "stroke": "smash",
  "confidence": 0.94
}
```

---

# Module 10 – Rally Understanding Engine

## Purpose

Understand rally structure.

## Outputs

* Rally start
* Rally end
* Winner
* Error type
* Stroke sequence

## Example

```json
{
  "rally_id": 102,
  "length": 14,
  "winner": "player_1"
}
```

---

# Module 11 – Tactical State Engine

## Purpose

Convert strokes into tactical understanding.

## Tactical States

* Attack
* Defense
* Neutral
* Pressure
* Recovery
* Transition

## Output

```json
{
  "state": "attack",
  "confidence": 0.92
}
```

## Derived Metrics

* Initiative %
* Attack retention
* Pressure creation
* Pressure escape
* Counterattack frequency

---

# Module 12 – Court Control Engine

## Purpose

Estimate rally control.

## Metrics

* Court control %
* Space created
* Opponent displacement
* Recovery risk

## Output

```json
{
  "court_control": 0.74
}
```

---

# Module 13 – Movement Engine

## Purpose

Analyze footwork and court movement.

## Metrics

* Distance covered
* Explosive movements
* Direction changes
* Recovery efficiency
* Base return time
* Adjustment steps
* Split-step frequency
* Balance score

## Output

```json
{
  "distance_m": 423
}
```

---

# Module 14 – Fitness Engine

## Purpose

Infer fatigue and workload.

## Metrics

* Movement speed decline
* Recovery decline
* Court coverage decline
* Reaction delay
* Work rate

## Output

```json
{
  "fatigue_score": 72
}
```

---

# Module 15 – Stroke Influence Engine

## Purpose

Estimate contribution of each shot to rally outcome.

## Example

```json
{
  "stroke": "drop",
  "influence": 0.23
}
```

---

# Module 16 – Forecasting Engine

## Phase

Phase 3

## Predict

* Next stroke
* Next movement
* Opponent response

## Purpose

Support scouting and advanced analytics.

---

# Module 17 – Coaching Intelligence Engine

## Purpose

Generate coach-grade recommendations.

## Inputs

* Technical analytics
* Tactical analytics
* Movement analytics
* Fitness analytics

## Outputs

* Strengths
* Weaknesses
* Tactical recommendations
* Technical recommendations
* Drill suggestions

## Example

```json
{
  "finding": "Late recovery after rear-court clears"
}
```

---

# 6. Analytics Requirements

## Technical Analytics

* Preparation timing
* Contact height
* Contact location
* Body alignment
* Footwork efficiency
* Recovery mechanics
* Balance

---

## Tactical Analytics

* Shot distribution
* Pattern chains
* Cross-court vs straight
* Front-back variation
* Attack conversion
* Defensive resilience

---

## Fitness Analytics

* Distance covered
* Recovery time
* Intensity score
* Fatigue score

---

## Behavioral Analytics

* Pressure tendencies
* End-game tendencies
* Shot selection under stress

---

# 7. Data Storage

```text
video.mp4

court.parquet

players.parquet

poses.parquet

shuttle.parquet

hits.parquet

strokes.parquet

rallies.parquet

tactics.parquet

fitness.parquet

report.json
```

---

# 8. Performance Targets

## GPU

### RTX 4060

1-hour match ≤ 15 minutes

### RTX 5070

1-hour match ≤ 8 minutes

---

## CPU

16-core CPU

1-hour match ≤ 60 minutes

---

## Memory

Maximum memory usage ≤ 16 GB

---

# 9. Evaluation Framework

## Stroke Classification

Target:

* > 90% accuracy for major stroke classes

---

## Player Attribution

Target:

* > 98% accuracy

---

## Rally Segmentation

Target:

* > 95% accuracy

---

## Tactical State Classification

Track attack/defense/neutral accuracy.

---

## Hit Frame Localization

Measure frame error relative to labeled ground truth.

---

## Confusion Matrix

Generated for every training run and release candidate.

---

# 10. Product Roadmap

## Phase 1 – Foundation

* Court calibration
* Player tracking
* Pose extraction
* TrackNet integration
* Hit frame localization
* BST integration

---

## Phase 2 – Advanced Analytics

* 3D shuttle reconstruction
* Court control engine
* Tactical state engine
* Movement analytics
* Fitness analytics

---

## Phase 3 – Intelligence Layer

* Forecasting
* Opponent scouting
* Match comparison
* Personalized coaching

---

# 11. Future Research Backlog

Maintain architecture extensibility for:

* Multi-camera fusion
* 3D pose estimation
* Wearable integration
* Heart-rate synchronization
* Real-time coaching
* Doubles analysis
* VideoMAE
* InternVideo
* VideoMamba
* Vision-Language Coaching Agents
* RAG-based coaching knowledge bases
* Personalized athlete digital twins
* Foundation sports video models
* Automated drill generation
* Longitudinal athlete development tracking

---

# Research Assets and External Dependencies

## Datasets

* ShuttleSet
* BadmintonDB
* FineGYM (benchmarking)
* NTU60
* NTU120
* Kinetics

## Models

* TrackNetV3
* BST-CG-AP
* BST-CG
* BST-AP
* BST-0
* RTMPose
* ViTPose
* ST-GCN
* 2s-AGCN
* PoseC3D
* TemPose
* TemPose-TF

## Frameworks

* MMAction2
* PyTorch
* OpenCV
* ONNX Runtime

---

# Product Differentiator

The product's primary differentiator is not stroke classification.

The differentiator is transforming:

**Video → Match Understanding → Tactical Understanding → Coaching Intelligence**

while maintaining explainability, player-specific attribution, and coach-actionable recommendations.
