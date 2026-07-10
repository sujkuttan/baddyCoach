# Early Pipeline TrackNet and InpaintNet Design

## Goal

Make shuttle tracking use the installed TrackNet and InpaintNet checkpoints according to their trained contracts, preserve genuine TrackNet detections, and prevent invalid shuttle coordinates from contaminating downstream hit and stroke logic.

## Scope

The change covers the backend and its parallel Colab implementation:

- TrackNet input and decoding use the checkpoint contract: a static background plus eight RGB frames, overlapping weighted heatmap ensembling, thresholding, and largest-connected-component centre extraction.
- InpaintNet uses the checkpoint-compatible temporal U-Net. It receives normalized, gap-filled input with a missing-frame mask; only frames originally missing from TrackNet are replaced by its predictions.
- Shuttle cleanup rejects out-of-court and physically impossible court-space detections rather than clamping them to court edges. Rejected points remain missing and are marked for downstream consumers.
- Pose bbox fallback is true same-player linear interpolation when detections bracket a frame; nearest-neighbour fallback remains for one-sided gaps.
- Player-tracking logs use the keyword-only `PipelineLogger` API.
- Colab keeps the same TrackNet and InpaintNet semantics as the backend.

## Deliberate Non-Goals

- Do not add a per-frame optical-flow homography stream. The current static court homography remains the coordinate reference; a moving-camera feature requires an artifact contract shared by every geometry consumer.
- Do not replace RTMPose or BaddyCoach's two-player stitching with the older YOLO-pose approach from `baddyAnalysis`.
- Do not change hit, ownership, BST, or coaching thresholds as part of this work.

## Data Flow

`frames → TrackNet raw detections → confidence/missing mask → masked InpaintNet repair → shuttle cleanup → court projection → out-of-bounds/speed rejection → downstream artifacts`

Raw TrackNet coordinates and confidences remain available. The repaired track carries explicit provenance so hit detection and analytics can distinguish observed points from repaired points. Court-space rejection happens before computing court-space velocity and direction; invalid points are not projected onto court boundaries.

## Error Handling and Model Health

InpaintNet loading validates every checkpoint tensor by key and shape. A mismatch disables InpaintNet and records a model-health failure instead of silently running randomly initialized layers. TrackNet continues to fail visibly if its checkpoint is incompatible.

## Testing

Regression tests will prove:

- the official InpaintNet checkpoint is fully compatible;
- observed coordinates remain unchanged after repair while only missing coordinates are reconstructed;
- TrackNet decoding selects a valid connected component rather than a single-pixel argmax and accepts the background-plus-eight-frame input contract;
- out-of-court and impossible-speed points become missing instead of edge-clamped;
- bracketing pose detections interpolate a bbox, and one-sided gaps retain nearest-bbox fallback;
- player tracking with live YOLO-result logging does not raise `TypeError`.

Backend and Colab behavior will be kept in parity for the TrackNet/InpaintNet path.
