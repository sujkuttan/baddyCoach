# BaddyCoach Implementation Plan

## Executive Summary

This plan addresses critical and high-priority issues in the BaddyCoach repository, focusing on fixing model integration bugs, correcting data processing errors, and improving reliability. The implementation will prioritize correctness issues first, followed by reliability improvements, and finally product enhancements.

# BaddyCoach Implementation Plan

## Executive Summary

This plan addresses critical and high-priority issues in the BaddyCoach repository, focusing on fixing model integration bugs, correcting data processing errors, and improving reliability. The implementation will prioritize correctness issues first, followed by reliability improvements, and finally product enhancements.

## Phase 1: Critical Fixes (Correctness) - Week 1

### 1.1 Fix BST Seq_len Wiring and Weight Path

**Files to Modify:**
- `backend/app/pipeline/strokes.py` - Remove hardcoded `SEQ_LEN = 30`, add `seq_len` parameter to `_build_clip`, pass `seq_len` to `_build_clip`
- `backend/app/models/bst.py` - Remove default `seq_len = 30`, update `_rule_based_predict` to use `self.seq_len`
- `backend/app/config/settings.py` - Fix BST weight path from `BST/weight/bst_CG_JnB_bone_merged.pt` to `ckpts/bst/bst_CG_AP.pt`

**Implementation Details:**
- Removed hardcoded `SEQ_LEN = 30` from `strokes.py`
- Updated `_build_clip` function to require `seq_len` parameter instead of default
- Updated all `SEQ_LEN` references to use `classifier.seq_len`
- Updated `BSTClassifier.__init__` to set `self.seq_len = None` (will be set after loading model)
- Updated `_rule_based_predict` to use `self.seq_len` if available, otherwise fallback to 30
- Updated `settings.py` to use the correct BST model path

**Testing Strategy:**
- Unit test: Verify seq_len parameter is correctly passed
- Integration test: Verify BST runs instead of falling back to rule-based classification
- Performance test: Compare inference accuracy before/after fix

### 1.2 Reorder Stages for Correct Rally Winners

**Files to Modify:**
- `backend/app/api/routes.py:72-86` - Reorder stages so `player_attribution` runs before `rally_segmentation`
- `backend/app/pipeline/rallies.py` - Add `_compute_rally_winner_after_attribution` function to compute rally winners after player attribution is complete

**Implementation Details:**
- Reordered stages in `routes.py` to move `player_attribution` before `rally_segmentation`
- Added `_compute_rally_winner_after_attribution` function in `rallies.py` to compute rally winners based on player attribution
- Updated `RallySegmentationStage.run` to use the new function for computing rally winners

**Testing Strategy:**
- Unit test: Verify rally winner attribution matches actual last hitter
- Integration test: Verify cross-session rally statistics are accurate
- Edge case test: Handle missing player attribution gracefully

### 1.3 Fix RTMPose x/y Rescale Transpose Bug

**Files to Modify:**
- `backend/app/models/rtmpose.py:62-63` - Fix swapped divisors
- `backend/app/models/rtmpose.py:75` - Use `self.input_name` instead of hardcoded "input"

**Implementation Details:**
- Fixed x/y rescale in `_postprocess` method: changed from `/ 256.0 * bw + x1` to `/ bw * 256.0 + x1` and `/ 192.0 * bh + y1` to `/ bh * 192.0 + y1`
- Updated `estimate` method to use `self.input_name` instead of hardcoded "input"

**Testing Strategy:**
- Unit test: Verify keypoint coordinates are correctly normalized
- Integration test: Verify BST features are correctly computed from pose data
- Visual test: Compare pose estimation results with ground truth

### 1.4 Fix Recovery Time Units Mismatch

**Files to Modify:**
- `backend/app/pipeline/analytics/footwork.py:128` - Convert threshold to meters
- `backend/app/pipeline/analytics/footwork.py:10-25` - Implement pixel-to-meter conversion

**Implementation Details:**
- Updated `_compute_recovery_times` method to accept `px_per_m` parameter
- Converted threshold from meters to pixels: `threshold_px = threshold_m * px_per_m`
- Scoped recovery to shot owner: only compute recovery for the player who hit the shot
- Updated `run` method to pass `px_per_m` to `_compute_recovery_times`

**Testing Strategy:**
- Unit test: Verify recovery time calculations are in correct units
- Integration test: Verify recovery analytics are reasonable for real footage
- Edge case test: Handle missing pose data gracefully

### 1.5 Respect court.valid Flag

**Files to Modify:**
- `backend/app/pipeline/attribution.py` - Add court validity check
- `backend/app/pipeline/players.py` - Add court validity check
- `backend/app/pipeline/analytics/court_position.py` - Skip invalid court data

**Implementation Details:**
- Added court validity check in `attribution.py` to return error if court is invalid
- Added court validity check in `players.py` to return error if court is invalid
- Added court validity check in `court_position.py` to return error if court is invalid

**Testing Strategy:**
- Unit test: Verify invalid court data is properly handled
- Integration test: Verify analytics skip invalid court data
- Edge case test: Handle completely failed court detection

### 1.6 Flag Synthetic/Fallback Data in Reports

**Files to Modify:**
- `backend/app/pipeline/players.py` - Add synthetic data flag to `_generate_synthetic_detections`
- `backend/app/pipeline/strokes.py` - Add rule-based prediction flag to shots
- `backend/app/report/generator.py` - Add data quality section to reports

**Implementation Details:**
- Updated `_generate_synthetic_detections` in `players.py` to add `is_synthetic` flag to synthetic detections
- Updated shot creation in `strokes.py` to add `is_rule_based` and `is_bst_fallback` flags
- Added `_generate_data_quality_report` method in `generator.py` to generate data quality report
- Updated `generate` method in `generator.py` to include data quality section in report

**Testing Strategy:**
- Unit test: Verify data quality flags are correctly added
- Integration test: Verify reports contain data quality information
- User test: Verify users understand data quality implications

## Phase 2: High Priority Fixes (Reliability) - Week 2

### 2.1 Fix TrackNet Integration

**Files to Modify:**
- `backend/app/models/tracknet.py:18` - Update to official TrackNetV3 architecture
- `backend/app/models/tracknet.py:116` - Add try/except around load_state_dict
- `backend/app/models/tracknet.py:223-227` - Replace silent zero fallback with visible error

**Implementation Details:**
- Added try/except around model loading in `TrackNetV3.__init__`
- Updated `predict` method to raise `RuntimeError` if model is not loaded or insufficient frames
- Updated `predict_batch` method to raise `RuntimeError` if model is not loaded
- Updated `predict_batch` method to raise `RuntimeError` if prediction fails

**Testing Strategy:**
- Unit test: Verify model loads correctly with official weights
- Integration test: Verify shuttle tracking works end-to-end
- Error handling test: Verify proper error messages when model fails

### 2.2 Use BST Top/Bottom Output for Attribution

**Files to Modify:**
- `backend/app/models/bst.py:27-35` - Update mapping to preserve Top/Bottom info
- `backend/app/pipeline/attribution.py` - Use BST Top/Bottom predictions for attribution

**Implementation Details:**
- Updated `map_to_coach_class` function in `bst.py` to preserve Top/Bottom information
- Updated attribution logic in `attribution.py` to use BST Top/Bottom predictions for attribution
- Added fallback to alternation heuristic if BST predictions not available

**Testing Strategy:**
- Unit test: Verify Top/Bottom attribution matches actual shuttle trajectory
- Integration test: Verify attribution accuracy improves with BST integration
- Cross-validation test: Compare BST attribution vs. ground truth

### 2.3 Compute Analytics in Meters via Homography

**Files to Modify:**
- `backend/app/pipeline/analytics/court_position.py` - Use homography for court coverage
- `backend/app/pipeline/analytics/footwork.py` - Use homography for distance calculations
- `backend/app/pipeline/analytics/fitness.py` - Use homography for distance calculations

**Implementation Details:**
- Updated `CourtPositionAnalyticsStage.run` to use homography for court coverage
- Added `_get_zone_from_court` method to compute zone from court coordinates
- Updated `FootworkAnalyticsStage._compute_distance` to accept homography parameter
- Updated `FootworkAnalyticsStage.run` to pass homography to `_compute_distance`

**Testing Strategy:**
- Unit test: Verify distance calculations are in correct units
- Integration test: Verify analytics are consistent across different video resolutions
- Cross-platform test: Verify results are independent of camera perspective

### 2.4 Replace Per-frame YOLO with Proper Tracking

**Files to Modify:**
- `backend/app/models/yolov8.py` - Update to use ByteTrack or BoT-SORT for better tracking
- `backend/app/pipeline/players.py` - Update synthetic generation
- `backend/app/pipeline/attribution.py` - Update attribution logic

**Implementation Details:**
- Updated `YOLOv8Tracker.__init__` to try to use ByteTrack or BoT-SORT for better tracking
- Updated `track_frames` method to use tracker if available, otherwise fallback to YOLO's tracking
- Updated attribution logic in `attribution.py` to use tracking data for attribution

**Testing Strategy:**
- Unit test: Verify tracking is consistent across frames
- Integration test: Verify tracking works with player ID switches
- Performance test: Verify tracking is faster than per-frame detection

### 2.5 Externalize Config

**Files to Modify:**
- `backend/app/config/settings.py:5` - Change to BaseSettings
- `backend/app/config/settings.py:16-22` - Externalize model paths
- `backend/app/config/settings.py:30-39` - Fix GPU settings
- `backend/requirements.txt` - Add `gdown` and pin ML dependencies

**Implementation Details:**
- Changed `Settings` class to inherit from `BaseSettings` instead of `BaseModel`
- Added environment variables for configuration
- Updated `requirements.txt` to add `pydantic-settings`, `structlog`, `gdown`, and pin ML dependencies

**Testing Strategy:**
- Unit test: Verify environment variables override defaults
- Integration test: Verify GPU acceleration works when available
- Configuration test: Verify all required settings have sensible defaults

### 2.6 Add Auth + Upload Validation

**Files to Modify:**
- `backend/app/api/routes.py:185-207` - Add file validation
- `backend/app/api/routes.py:305-309` - Add authentication
- `backend/app/security.py` - Create security utilities

**Implementation Details:**
- Created `Security` class in `security.py` to handle authentication
- Updated `upload_video` endpoint to validate file size, MIME type, and video duration
- Updated `upload_video` endpoint to authenticate request using API key

**Testing Strategy:**
- Unit test: Verify file validation catches invalid uploads
- Integration test: Verify authentication works with API keys
- Security test: Verify no authentication bypass is possible

### 2.7 Fix or Remove shuttle_coach Subsystem

**Files to Modify:**
- `backend/app/shuttle_coach/loader.py:19-26` - Update to use players.json
- `backend/app/shuttle_coach/metrics.py` - Remove unused code
- `backend/app/shuttle_coach/` - Delete dead code

**Implementation Details:**
- Updated `COLUMN_ALIASES` in `loader.py` to use `players` instead of `player_detections`
- Updated `REQUIRED` in `loader.py` to use `players` instead of `player_detections`
- Updated `capabilities` function in `loader.py` to use `players` instead of `player_detections`

**Testing Strategy:**
- Unit test: Verify shuttle_coach works with new data format
- Integration test: Verify shuttle_coach endpoint works end-to-end
- Cleanup test: Verify dead code is properly removed

## Phase 3: Nice-to-have (Product & Maintainability) - Week 3-4

### 3.1 Unify Backend and Colab Pipelines

**Files to Modify:**
- Create shared stage modules in `backend/app/pipeline/`
- Remove duplicate code from colab pipeline
- Add golden regression tests

**Implementation Details:**
- Create shared stage modules in `backend/app/pipeline/`
- Update both pipelines to use shared modules
- Add golden regression tests

**Testing Strategy:**
- Unit test: Verify shared modules work correctly
- Integration test: Verify both pipelines produce identical results
- Regression test: Add golden output tests

### 3.2 Replace Single-Frame Technique Score

**Files to Modify:**
- `backend/app/pipeline/technical.py:53-93` - Replace with temporal analysis

**Implementation Details:**
- Update `technical.py` to use BST clip data for temporal analysis
- Add `_analyze_swing_mechanics` method to analyze swing mechanics using temporal data

**Testing Strategy:**
- Unit test: Verify technique scores are biomechanically meaningful
- Integration test: Verify technique scores correlate with coaching insights
- User test: Verify coaches find technique scores useful

### 3.3 Cross-Session Progress Tracking

**Files to Modify:**
- Add user/session tracking to database
- Implement trend analysis across multiple uploads

**Implementation Details:**
- Add user/session tracking to database
- Implement trend analysis across multiple uploads

**Testing Strategy:**
- Unit test: Verify progress tracking works correctly
- Integration test: Verify trends are calculated accurately
- Privacy test: Verify user data is properly anonymized

### 3.4 Structured Logging + Data-Quality Score

**Files to Modify:**
- Replace print statements with structured logging
- Add data quality metrics to reports

**Implementation Details:**
- Use structured logging with `structlog`
- Add data quality score to reports

**Testing Strategy:**
- Unit test: Verify logging is structured correctly
- Integration test: Verify data quality score is computed accurately
- Monitoring test: Verify logs are properly formatted for analysis

### 3.5 Promote Grounded LLM Narration

**Files to Modify:**
- `/backend/app/gemini.py:42-56` - Integrate into main report
- `/backend/app/api/routes.py:305-309` - Use Gemini for coaching insights

**Implementation Details:**
- Integrate Gemini narration into main report
- Use Gemini for coaching insights

**Testing Strategy:**
- Unit test: Verify Gemini integration works correctly
- Integration test: Verify narration is grounded and accurate
- User test: Verify narration provides actionable insights

### 3.6 License Compliance Audit

**Files to Modify:**
- Document all third-party licenses
- Ensure AGPL compliance for YOLOv8

**Implementation Details:**
- Document all third-party licenses
- Ensure AGPL compliance for YOLOv8

**Testing Strategy:**
- Unit test: Verify license information is accurate
- Compliance test: Verify licensing restrictions are understood
- Documentation test: Verify licenses are properly documented

## Testing Strategy

### 1. Unit Tests
- Test each fix in isolation
- Mock external dependencies
- Verify edge cases
- Run: `python -m pytest backend/tests/ -v -k "test_fix_name"`

### 2. Integration Tests
- Test pipeline end-to-end with synthetic data
- Verify model integration works correctly
- Test data quality flags
- Run: `python -m pytest backend/tests/test_real_pipeline.py -v`

### 3. Performance Tests
- Benchmark before/after fixes
- Test with real video files
- Verify GPU acceleration works
- Run: `python -m pytest backend/tests/ -m "performance" -v`

### 4. Security Tests
- Test authentication mechanisms
- Test file validation
- Test input sanitization
- Run: `python -m pytest backend/tests/ -m "security" -v`

## Rollout Plan

### Week 1: Critical Fixes (Correctness)
**Priority 1-6:**
1. Fix BST seq_len wiring and weight path
2. Reorder stages for correct rally winners
3. Fix RTMPose x/y rescale transpose
4. Fix recovery-time pixel/meter mismatch
5. Respect `court.valid` flag
6. Flag synthetic/fallback data in reports

**Testing Focus:**
- Unit tests for each fix
- Integration tests for pipeline
- Regression tests for existing functionality

### Week 2: High Priority Fixes (Reliability)
**Priority 7-13:**
7. Fix TrackNet integration
8. Use BST Top/Bottom output for attribution
9. Compute analytics in meters via homography
10. Replace per-frame YOLO with proper tracking
11. Externalize config
12. Add auth + upload validation
13. Fix or remove shuttle_coach

**Testing Focus:**
- Integration tests for new features
- Security tests for authentication
- Performance tests for tracking

### Week 3-4: Nice-to-have (Product)
**Priority 14-19:**
14. Unify backend/colab pipelines
15. Replace single-frame technique score
16. Cross-session progress tracking
17. Structured logging + data-quality score
18. Promote grounded LLM narration
19. License compliance audit

**Testing Focus:**
- User acceptance testing
- Performance testing
- Documentation verification

## Risk Mitigation

### High-Risk Changes:
1. **Stage reordering** - Ensure rally segmentation still works without attribution
2. **TrackNet integration** - Have fallback to existing implementation if official weights not available
3. **Config externalization** - Maintain backward compatibility with existing settings

### Testing Approach:
1. Run existing tests before changes
2. Create comprehensive test coverage for new functionality
3. Use feature flags for gradual rollout
4. Monitor production metrics after deployment

## Dependencies

### Required Changes to Requirements:
```bash
# backend/requirements.txt
pip install -r backend/requirements.txt

# Additional packages for fixes
pip install pydantic-settings>=2.0.0
pip install structlog>=22.0.0
pip install gdown>=4.0.0
pip install onnxruntime-gpu>=1.15.0  # For GPU acceleration
```

### Environment Setup:
```bash
# Create .env file
GEMINI_API_KEY=your_api_key_here
BST_MODEL_PATH=/path/to/bst_model.pt
GPU_ENABLED=true
```

## Monitoring and Maintenance

### 1. Data Quality Dashboard
- Track synthetic data usage
- Monitor rule-based fallback rates
- Alert on model loading failures

### 2. Performance Monitoring
- Track processing times for each stage
- Monitor GPU utilization
- Alert on performance degradation

### 3. User Analytics
- Track usage patterns
- Monitor error rates
- Collect feedback on coaching insights

---

*Implementation plan created based on comprehensive analysis of CODE_REVIEW.md and repository exploration.*
