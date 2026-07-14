# BST Input Quality Gate Design

## Goal

Improve stroke-classification accuracy by sending BST only clips supported by reliable upstream evidence, and make every abstention or fallback auditable.

## Scope

This design changes the quality assessment and admission of BST inputs. It does not retrain BST, add a new model, or change the default clip-boundary convention. Those remain separate, label-driven experiments after this gate establishes a trustworthy baseline.

## Problem

The stroke stage currently receives a cleaned shuttle track, pose rows, and player boxes, but it does not carry enough provenance into the pre-inference decision:

- short shuttle gaps are interpolated and receive a confidence that passes the normal shuttle gate;
- court-rejected points can still enter the pixel-coordinate shuttle tensor;
- bbox interpolation has no maximum span when a clip is assembled;
- missing pose frames are zero-filled but only counted after the clip is built.

These conditions can produce a structurally valid tensor that does not describe a real stroke. An accuracy-first pipeline must abstain rather than make a confident model prediction from such a tensor.

## Design

### Clip provenance

`_build_clip()` will collect per-frame provenance for the unpadded portion of a clip (`0:video_len`):

- `shuttle_observed`: a confidence-qualified raw TrackNet observation;
- `shuttle_repaired`: an InpaintNet-repaired point that was not observed;
- `shuttle_interpolated`: a cleaner-filled point (`was_interpolated=True`);
- `shuttle_court_rejected`: a point carrying `court_rejected=True`;
- `pose_present_far` and `pose_present_near`: a pose row with meaningful keypoints for each BST player slot;
- `pose_keypoint_confidence_far` and `pose_keypoint_confidence_near`: median COCO keypoint confidence across meaningful pose frames;
- `bbox_gap_far` and `bbox_gap_near`: frame distance to the source detection used for bbox normalization.

In resolution (pixel) mode, court-rejected shuttle points still enter the BST shuttle tensor using their image-space coordinates (homography OOB is a court-projection signal, not missing pixels). In court-normalized mode, court-rejected points are encoded as missing (`[0, 0]`). Existing cleaned/repaired points retain current tensor semantics; their provenance affects admission, not the tensor shape. This preserves BST’s fixed input contract.

### Quality evaluation

Add a pure helper, `evaluate_bst_clip_quality(clip, provenance)`, returning a serializable record:

```python
{
    "eligible": bool,
    "score": float,
    "reasons": list[str],
    "observed_shuttle_frames": int,
    "repaired_shuttle_frames": int,
    "interpolated_shuttle_frames": int,
    "court_rejected_shuttle_frames": int,
    "max_shuttle_gap_frames": int,
    "far_pose_coverage": float,
    "near_pose_coverage": float,
    "far_pose_median_confidence": float,
    "near_pose_median_confidence": float,
    "max_bbox_gap_frames": int,
}
```

`score` starts at `1.0` and applies the following deterministic penalties, clamped to `[0.0, 1.0]`:

- subtract `0.35` when the observed-shuttle fraction is below `0.35`;
- subtract `0.25` when a raw-observation gap exceeds `7` frames;
- subtract `0.20` for one or more court-rejected shuttle points;
- subtract `0.20` when either player has pose coverage below `0.70`;
- subtract `0.15` when either player’s median keypoint confidence is below `0.35`;
- subtract `0.15` when a bbox requires a source detection more than `10` frames away.

A clip is BST-eligible only when all hard checks hold: `video_len >= 15`, observed-shuttle fraction `>= 0.35`, maximum raw-observation gap `<= 7`, no court-rejected point, both pose coverages `>= 0.70`, both median confidences `>= 0.35`, and maximum bbox gap `<= 10`. Its quality score must also be at least `0.70`.

All threshold values live in `Settings` so the later manual-label evaluation can tune them without changing stage code.

### Bbox and keypoint rules

`_interpolate_bboxes()` receives `max_gap_frames=settings.bst_max_bbox_interp_gap` and returns both its bbox and its source-gap distance. A gap larger than 10 does not get extrapolated. The existing confidence-aware keypoint bbox path becomes the normalization fallback when a detection bbox is unavailable.

Before joint and bone construction, keypoints with confidence below `settings.bst_min_keypoint_confidence` (`0.35`) are zeroed. This makes missing-joint semantics consistent for detection-bbox and keypoint-bbox normalization paths.

### Inference routing

The stroke stage builds every clip and quality record before model batching.

- Eligible clips are batched through `BSTClassifier.predict_from_clips()` unchanged. In court-normalized mode, court-rejected shuttle points are zeroed (`[0, 0]`); in resolution mode they retain their image-space coordinates.
- Ineligible clips do not reach BST. They produce `stroke_type="unknown"`, `stroke_confidence=0.0`, `shuttleset_class_id=0`, and `is_bst_fallback=True`.
- The existing context/physics stages may still turn an `unknown` into a class only when their own existing evidence permits it. The shot retains its original quality reasons regardless of downstream override.

The stage must not manufacture a rule-based BST fallback for an ineligible clip: that would undermine the accuracy-first decision and make it difficult to distinguish abstention from classification.

### Persisted output and observability

Every shot records these fields in `shots.parquet` and the report payload:

- `bst_input_eligible`, `bst_input_quality_score`, `bst_input_quality_reasons`;
- all numeric fields returned by `evaluate_bst_clip_quality`;
- `bst_input_route` with one of `"bst"`, `"quality_abstain"`, or `"downstream_override"`.

At debug level 1 or greater, persist `debug_bst_input_quality.parquet`, one row per candidate clip. The stroke-stage summary logs the number and percentage of clips sent to BST, abstained by each reason, and later overridden by context/physics.

## Data flow

```text
raw TrackNet + repair/interpolation provenance + court rejection
                         \
pose rows + keypoint confidence + bbox source distances --> clip provenance
                                                           |
                                             quality evaluator
                                               /          \
                                          eligible       ineligible
                                             |              |
                                           BST batch      unknown + reasons
                                             \              /
                                           existing context/physics
                                                     |
                                           persisted shot audit fields
```

## Evaluation

Use a manually labeled held-out set with whole-video or whole-rally splitting. Report all of:

- accepted-shot accuracy (BST-eligible clips only);
- overall accuracy, treating abstentions explicitly rather than as correct predictions;
- BST coverage (`eligible / all candidates`);
- unknown/abstention rate;
- per-class precision and recall; and
- reason distribution and quality-score calibration.

The gate is accepted only if accepted-shot accuracy increases without a disproportionate coverage loss. The baseline and candidate runs must use identical hit frames and label matching.

## Tests

Unit tests cover each hard rejection condition, score clamping, provenance counting, court-rejected zeroing, bbox-gap capping, low-confidence keypoint zeroing, and the guarantee that ineligible clips never enter the BST batch. Stage tests verify persisted audit fields and debug-artifact gating. Colab parity tests ensure the same quality evaluator and routing behavior are used in `colab/pipeline.py`.
