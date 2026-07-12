# Court-Space Rejection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Reject implausible shuttle detections after image-to-court projection without changing original image-space detections.

**Architecture:** Keep image_to_court() as a pure unclamped projection. Apply bounds and speed policy only in _add_court_space_columns(); a rejected row keeps raw x/y but all derived court kinematics are NaN.

**Tech Stack:** Python 3, NumPy, pandas, OpenCV, pydantic-settings, pytest.

---

## File structure

- backend/app/config/settings.py — rejection thresholds, overridable via environment.
- backend/app/pipeline/shuttle.py — court-space enrichment and rejection.
- backend/tests/test_shuttle.py — focused regression tests.
- colab/pipeline.py / backend/tests/test_colab_pipeline.py — parity verification where Colab has a separate shuttle implementation.

### Task 1: Add configurable rejection thresholds

**Files:**

- Modify: backend/app/config/settings.py:36-45
- Test: backend/tests/test_shuttle.py

- [ ] **Step 1: Write the failing behavior test**

~~~python
def test_court_enrichment_rejects_out_of_bounds_and_impossible_speed(monkeypatch):
    from app.pipeline import shuttle

    monkeypatch.setattr(shuttle.settings, "shuttle_oob_margin_meters", 0.25)
    monkeypatch.setattr(shuttle.settings, "shuttle_max_speed_mps", 20.0)
    detections = pd.DataFrame([
        {"frame": 0, "x": 13.5, "y": 1.0, "confidence": 0.9},
        {"frame": 1, "x": 14.0, "y": 1.0, "confidence": 0.9},
        {"frame": 2, "x": 1.2, "y": 1.0, "confidence": 0.9},
        {"frame": 3, "x": 5.2, "y": 1.0, "confidence": 0.9},
    ])

    enriched = _add_court_space_columns(detections, np.eye(3), fps=30.0)

    assert enriched["court_rejected"].tolist() == [False, True, False, True]
    assert enriched.loc[0, "x_court"] == 13.5
    assert enriched.loc[1, ["x", "y"]].tolist() == [14.0, 1.0]
    assert np.isnan(enriched.loc[1, ["x_court", "y_court", "speed_court", "direction_x", "direction_y"]]).all()
    assert np.isnan(enriched.loc[3, ["x_court", "y_court", "speed_court", "direction_x", "direction_y"]]).all()
~~~

- [ ] **Step 2: Run it before implementation**

Run: cd backend && python3 -m pytest tests/test_shuttle.py::test_court_enrichment_rejects_out_of_bounds_and_impossible_speed -v

Expected: FAIL because the settings or court_rejected output does not exist.

- [ ] **Step 3: Define unit-explicit settings**

Add immediately after existing shuttle-cleaning settings:

~~~python
shuttle_oob_margin_meters: float = 1.0  # allowed court-space overshoot before rejection
shuttle_max_speed_mps: float = 80.0  # reject consecutive court-space detections above this speed
~~~

- [ ] **Step 4: Re-run the focused test**

Run: cd backend && python3 -m pytest tests/test_shuttle.py::test_court_enrichment_rejects_out_of_bounds_and_impossible_speed -v

Expected: still FAIL until Task 2 is implemented.

### Task 2: Reject invalid projected points without clamping

**Files:**

- Modify: backend/app/pipeline/shuttle.py:12-75
- Test: backend/tests/test_shuttle.py

- [ ] **Step 1: Confirm the behavior test still fails**

Run: cd backend && python3 -m pytest tests/test_shuttle.py::test_court_enrichment_rejects_out_of_bounds_and_impossible_speed -v

Expected: FAIL with missing or incorrect rejection behavior.

- [ ] **Step 2: Allocate derived outputs separately from raw detections**

At the start of _add_court_space_columns(), create derived arrays:

~~~python
court_xs = np.full(len(df), np.nan, dtype=np.float64)
court_ys = np.full(len(df), np.nan, dtype=np.float64)
speeds = np.full(len(df), np.nan, dtype=np.float64)
dir_xs = np.full(len(df), np.nan, dtype=np.float64)
dir_ys = np.full(len(df), np.nan, dtype=np.float64)
court_rejected = np.zeros(len(df), dtype=bool)
~~~

Never assign to df["x"] or df["y"].

- [ ] **Step 3: Check unclamped bounds after projection**

Use image_to_court(H, (float(x), float(y))) directly — do not call clamp_to_court:

~~~python
out_of_bounds = (
    not np.isfinite(cx) or not np.isfinite(cy)
    or cx < -settings.shuttle_oob_margin_meters
    or cx > COURT_LENGTH + settings.shuttle_oob_margin_meters
    or cy < -settings.shuttle_oob_margin_meters
    or cy > COURT_WIDTH + settings.shuttle_oob_margin_meters
)
if out_of_bounds:
    court_rejected[i] = True
    prev_cx, prev_cy = None, None
    continue
~~~

A valid point within the margin retains its actual projected position. An out-of-bounds or non-finite point retains NaN derived values.

- [ ] **Step 4: Reject only the later side of an impossible consecutive speed pair**

For accepted consecutive rows:

~~~python
dx, dy = cx - prev_cx, cy - prev_cy
speed = np.hypot(dx, dy) * fps
if speed > settings.shuttle_max_speed_mps:
    court_rejected[i] = True
    court_xs[i] = court_ys[i] = np.nan
    prev_cx, prev_cy = None, None
    continue
speeds[i] = speed
norm = np.hypot(dx, dy) + 1e-8
dir_xs[i], dir_ys[i] = dx / norm, dy / norm
~~~

Reset prev_cx/prev_cy after a missing, out-of-bounds, or speed-rejected row so gaps are not treated as one-frame teleports.

- [ ] **Step 5: Publish all derived columns**

~~~python
df = df.copy()
df["x_court"] = court_xs
df["y_court"] = court_ys
df["speed_court"] = speeds
df["direction_x"] = dir_xs
df["direction_y"] = dir_ys
df["court_rejected"] = court_rejected
return df
~~~

- [ ] **Step 6: Run the regression file**

Run: cd backend && python3 -m pytest tests/test_shuttle.py -v

Expected: PASS, including artifact persistence and empty-input coverage.

- [ ] **Step 7: Commit the backend change**

~~~bash
git add backend/app/config/settings.py backend/app/pipeline/shuttle.py backend/tests/test_shuttle.py
git commit -m "fix: reject invalid court-space shuttle points"
~~~

### Task 3: Verify Colab parity and final behavior

**Files:**

- Modify if needed: colab/pipeline.py
- Test: backend/tests/test_colab_pipeline.py
- Test: backend/tests/test_shuttle.py

- [ ] **Step 1: Locate Colab shuttle ownership**

Run: rg -n -C 3 "ShuttleTrackingStage|_add_court_space_columns|x_court|speed_court" colab/pipeline.py

Expected: identify whether Colab delegates to app.pipeline.shuttle.ShuttleTrackingStage or has a duplicate enrichment path.

- [ ] **Step 2: Keep policies identical when Colab has local enrichment**

If Colab delegates to the backend stage, make no duplicate implementation. If it maintains local court enrichment, mirror these exact rules: unclamped projection, margin bounds, speed limit, court_rejected, NaN derived values for rejected points, and preservation of raw coordinates.

- [ ] **Step 3: Add a parity test for the delegation case**

~~~python
def test_colab_uses_backend_shuttle_stage():
    source = Path("../colab/pipeline.py").read_text()
    assert "ShuttleTrackingStage" in source
    assert "from app.pipeline.shuttle import" in source
~~~

If Colab has a local helper, replace this source test with a functional comparison against the backend helper using the same identity-homography dataframe and equal_nan=True.

- [ ] **Step 4: Run final focused verification**

Run: cd backend && python3 -m pytest tests/test_shuttle.py tests/test_colab_pipeline.py -v

Expected: PASS; hardware/model-gated tests may be SKIPPED.

- [ ] **Step 5: Commit a Colab-only change if one was required**

~~~bash
git add colab/pipeline.py backend/tests/test_colab_pipeline.py
git commit -m "fix: keep Colab court-space rejection in parity"
~~~

## Plan self-review

- **Spec coverage:** The plan covers configurable thresholds, unclamped bounds checks, non-finite projections, speed teleports, reset-after-gap behavior, raw-coordinate preservation, regression tests, and Colab parity.
- **Placeholder scan:** No TBD/TODO or unspecified implementation/test steps remain.
- **Type consistency:** The plan consistently uses court_rejected, shuttle_oob_margin_meters, shuttle_max_speed_mps, and float court-space outputs.

