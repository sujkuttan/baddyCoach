# Owner Attribution Quality Design

## Goal

Prevent weak or unvalidated signals from assigning the wrong player to a shot.
Player-specific coaching and analytics must consume only ownership assignments
with defensible evidence. Shots without sufficient evidence remain in rally and
stroke totals with an explicit `unknown` owner.

## Findings that constrain the design

`aimplayer_alpha` is an internal BST feature-gating value, not a calibrated
owner probability. In preserved runs it was tightly centred around 0.5 and
never passed the current confidence gate. Its sign was near chance against the
available manual side labels. It therefore cannot assign an owner or act as a
Viterbi emission. The BST Top/Bottom class prefix is also diagnostic-only.

The current Viterbi decoder has a 0.95 alternate transition probability. With
neutral emissions it selects an arbitrary initial owner and then alternates.
The scorer's turn-prior feature is currently neutral because all per-shot
emissions are calculated before any owner has been assigned.

## Production attribution behaviour

1. The ownership scorer produces only geometry and pose-based emissions:
   trajectory, court-side feasibility, wrist proximity, racket motion, and
   pose-contact feasibility. It records BST alpha/class evidence for audit but
   does not include either in the score.
2. Each shot receives a local evidence confidence and margin. An emission is
   an **anchor** only when both meet configurable thresholds and the required
   evidence is present.
3. Viterbi decodes only spans bounded by anchors. It may fill a short span
   between two compatible anchors but may not choose the first owner of a
   rally, bridge an unbounded prefix/suffix, or fill a span above the configured
   maximum length.
4. Non-anchor shots not safely bridged are assigned `player_id=None`,
   `side="unknown"`, `owner_confident=False`, and an `owner_reason` explaining
   why. No default near-player fill is permitted.
5. A shot assigned locally or by a bounded Viterbi bridge has
   `owner_confident=True`, its source (`local_anchor` or `viterbi_bridge`), and
   the evidence score/margin persisted in `shots.parquet`.
6. Player-specific coaching and analytics filter on `owner_confident=True`.
   Rally/stroke totals retain all shots, including unknown owners.

## Offline calibration and evaluation

Add a script that joins manual side labels to shots and evaluates attribution
using leave-one-rally-out splits. It must report coverage, strict accuracy on
assigned shots, overall accuracy with abstentions counted as incorrect,
abstention rate, and per-source metrics.

The script may fit a regularized logistic calibration model from the existing
geometry/pose sub-scores and margins. It must never write a runtime model or
change production settings automatically. It produces a versioned JSON report
and only recommends deployment when held-out assigned-shot accuracy and
coverage both exceed the evidence-only baseline by configured minimums.

`aimplayer_alpha`, its raw cosine similarities, and Top/Bottom class prefix
may appear as diagnostic columns in the report but are excluded from fitted
features until a separately labelled validation set demonstrates predictive
value.

## Configuration

All tunables live in `Settings`:

- owner anchor minimum confidence and margin;
- minimum count of non-neutral independent signals;
- maximum Viterbi bridge span;
- enabled/disabled Viterbi bridge flag;
- calibration evaluation matching tolerance and deployment improvement gates.

Defaults prioritize precision over ownership coverage.

## Testing

Tests cover: neutral BST alpha cannot affect an emission; an unanchored rally
does not alternate by default; local anchors assign their supported side;
Viterbi only fills a bounded short span; long/unbounded spans remain unknown;
unknown owners are excluded from player-specific analytics; and the offline
report correctly calculates coverage, assigned accuracy, overall accuracy, and
per-source metrics.

