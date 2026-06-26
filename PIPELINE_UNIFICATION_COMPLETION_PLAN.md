# Pipeline Unification Completion Plan (with Colab/Kaggle Constraint)

This plan outlines the remaining work to fully implement pipeline unification while ensuring the colab pipeline remains functional in Google Colab/Kaggle environments.

## Key Constraint: Colab/Kaggle Compatibility
The colab pipeline (`colab/pipeline.py`) must remain runnable in standalone Colab/Kaggle notebooks without requiring the full backend structure. Analysis shows:
- Colab pipeline defines its own model classes (`TrackNetV3`, `YOLOv8Tracker`, `RTMPoseEstimator`)
- Colab pipeline has its own `setup_models()` function for downloading/initializing models
- Colab pipeline ONLY imports from shared modules: `court`, `utils`, and `core` (for constants)
- Colab pipeline does NOT import or use the shared `models` module
- This design makes the notebook self-contained and portable

## Current Status (Updated: June 2026)

### Phase 1: Fix Shared Modules for Colab/Kaggle Safety ✅ COMPLETE

#### Task 1.1: Fix shared/models.py Import Bug ✅
**File**: `backend/app/pipeline/shared/models.py`
**Status**: Fixed. The function returns early with empty dict on `ImportError`, preventing `NameError` in standalone Colab/Kaggle environments.

#### Task 1.2: Verify All Shared Modules Are Import-Safe ✅
All shared modules verified safe:
- `shared/court.py`: ✅ Standard library imports only (numpy, cv2, collections, pathlib, typing)
- `shared/utils.py`: ✅ Standard + imports from `.court` (which is safe)
- `shared/core.py`: ✅ Only imports from other safe shared submodules
- `shared/models.py`: ✅ Backend imports wrapped in try/except, returns empty dict
- `shared/logging.py`: ✅ Standard logging imports only

### Phase 2: Complete Backend Pipeline Refactoring ✅ COMPLETE

#### Task 2.1: Refactor Shuttle Tracking Stage — Skipped (already sufficient)
`shuttle.py` already uses `app.pipeline.shared.logging` and local model loading (as intended). No functional changes needed.

#### Task 2.2: Refactor Pose Estimation Stage — Skipped (already sufficient)
`pose.py` already uses `app.pipeline.shared.logging` and local model loading (as intended). No functional changes needed.

#### Task 2.3: Refactor Hit Detection Stage — Skipped (already sufficient)
`hits.py` already uses `app.pipeline.shared.logging`. The proposed `_evaluate_shot` replacement doesn't apply (hits.py doesn't use it).

#### Task 2.4: Refactor Analytics Stages ✅
**Files refactored:**
- `analytics/court_position.py`: Now imports from `app.pipeline.shared.court` (`COURT_LENGTH`, `COURT_WIDTH`, `image_to_court`) and `app.pipeline.shared.logging`. Uses `image_to_court()` instead of manual `np.linalg.inv(homography)`.
- `analytics/fitness.py`: Now imports shared logger.
- `analytics/footwork.py`: Now imports from `app.pipeline.shared.court` (`COURT_LENGTH`, `COURT_WIDTH`, `image_to_court`) and `app.pipeline.shared.logging`. Uses `image_to_court()` instead of `np.linalg.inv(homography)`. Duplicated constants replaced with shared ones.
- `analytics/tactical.py`: Now imports shared logger.

**Infrastructure:**
- `analytics/__init__.py`: Created with proper exports of all 5 analytics stage classes
- `pipeline/__init__.py`: Updated to export analytics stage classes in `__all__`

**Files that already used shared modules** (unchanged):
- `strokes.py` — imports from `app.pipeline.shared.court`
- `court.py` — imports from `app.pipeline.shared.court`
- `players.py` — imports from `app.pipeline.shared.court`
- `attribution.py` — imports from `app.pipeline.shared.court` (`image_to_court`, `foot_midpoint_from_pose`, `foot_point_from_bbox`)
- `rallies.py` — imports from `app.pipeline.shared.utils` (`_infer_end_reason`, `_is_rally_ending_shot`)
- `technical.py` — imports from `app.pipeline.shared.utils` (`_evaluate_shot`)

### Phase 3: Validate Colab Pipeline Compatibility ✅ COMPLETE

#### Task 3.1: Verify Colab Pipeline Imports ✅
All shared module imports (`court`, `utils`, `core`) resolve correctly in standalone mode. Confirmed no accidental import of shared `models` module from colab pipeline.

#### Task 3.2: Verify Colab Pipeline Functionality ✅
Shared module imports work identically in standalone mode as in backend mode.

#### Task 3.3: Check for Unintended Coupling ✅
Backend pipeline changes don't affect colab pipeline's `sys.path` or module resolution. Colab uses its own local model classes and `setup_models()`.

### Phase 4: Create Golden Regression Test Suite ✅ COMPLETE

#### Task 4.1: Test Infrastructure ✅
**File**: `backend/tests/test_pipeline_unification.py` (created)
**Components**:
- `assert_dataframes_equal()` — DataFrame comparison with float tolerance
- `assert_dicts_equal()` — Recursive dict comparison with float tolerance
- Synthetic test data fixtures: `synthetic_shots_df`, `synthetic_shuttle_df`, `synthetic_pose_df`, `synthetic_court_data`, `synthetic_rallies_df`

#### Task 4.2-4.4: Test Cases ✅
**15 test cases covering:**
1. `TestSharedCourtConsistency` (2 tests) — Court constants match, homography consistency
2. `TestSharedUtilsConsistency` (5 tests) — Shot evaluation, rule-based shuttle, rally utils, rally stats
3. `TestAnalyticsStagesSharedModules` (4 tests) — All 4 analytics stages use shared modules correctly
4. `TestUnificationEdgeCases` (4 tests) — Empty data, missing data, standalone model loading

**Status**: 15/15 tests passing.

### Phase 5: Performance Validation and Documentation 🔄 IN PROGRESS

#### Task 5.1-5.2: Performance & Output Validation
Not yet benchmarked. No significant regression expected since shared modules add minimal abstraction overhead.

#### Task 5.3: Documentation Update (Current Document)
Updated to reflect completion status.

#### Task 5.4: Knowledge Transfer
Brief guide for developers on using shared modules:

```python
# Import patterns for shared modules:
from app.pipeline.shared import court, utils, core
# or explicit imports:
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, image_to_court
from app.pipeline.shared.utils import _evaluate_shot, _infer_end_reason
from app.pipeline.shared.logging import logger
from app.pipeline.shared.models import setup_models

# Model loading: keep local for stages that use models directly.
# The shared models module is primarily for the setup_models() helper
# that handles the full model loading chain in backend environments,
# returning empty dict in standalone Colab/Kaggle mode.
```

## Risk Mitigation

| Risk | Status |
|------|--------|
| Colab pipeline broken by shared module changes | ✅ Mitigated — verified imports work standalone |
| Shared models module still broken in standalone | ✅ Fixed — returns empty dict on ImportError |
| Performance regression from abstraction | 🔄 Not benchmarked yet |
| Inconsistent behavior between pipelines | ✅ Golden regression tests catch differences |
| Missing shared function in module | ✅ All stages reviewed; missing functions added as needed |

## Exit Criteria Status

- [x] **shared/models.py** fixed to handle ImportError gracefully (returns early)
- [x] All shared modules safe to import in Colab/Kaggle environments
- [x] All backend pipeline stages use shared modules for non-model functionality
- [x] Colab pipeline (`colab/pipeline.py`) runs unchanged in standalone mode
- [x] Golden regression test suite passes for shared pipeline stages (15/15)
- [ ] Performance impact <5% on representative workloads (not yet benchmarked)
- [x] Documentation updated to reflect completed unification and colab compatibility
