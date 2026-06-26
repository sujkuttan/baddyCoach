# Badminton Court Position Extraction — PRD + Technical Design

**Component name (working):** `shuttle-court-positions`
**One-liner:** Take an uploaded handheld badminton video, detect the court automatically, and output per-frame 2D player positions in real-world court coordinates, using a per-frame homography that is robust to camera motion.

---

## Part 1 — Product Requirements (PRD)

### 1.1 Problem

Handheld phone footage of a badminton match contains players moving on a court, but every measurement we care about (distance covered, court coverage, position heatmaps, time in each zone) requires positions in **court metres**, not image pixels. Pixel positions are meaningless across a moving camera: the same court location lands on different pixels every frame as the phone pans and shakes. We need a pipeline that converts each player's image position into a stable court coordinate, frame by frame.

### 1.2 Goal

Given one uploaded video, produce a per-frame record of each player's `(x, y)` position on a canonical badminton court (origin at a fixed court corner, units in metres), plus the per-frame homography used to compute it and a confidence/validity flag.

### 1.3 Non-goals

- 3D reconstruction or shuttle height (this is strictly on-court-plane 2D).
- Doubles (SoloShuttlePose targets singles; doubles is a later extension).
- Real-time/live processing (this is offline batch over an uploaded file).
- Shot classification, tactics, or scoring — downstream consumers can build on the position output.

### 1.4 Users / consumers

- A downstream analytics model or notebook that ingests court positions.
- A coach-facing visualization (heatmaps, movement trails) built on top of the position stream.

### 1.5 User stories

1. As an analyst, I upload an MP4 and get back a CSV/JSON of per-frame player court positions so I can compute distance covered.
2. As a developer, I get the per-frame homography matrix so I can project any other image point (e.g. shuttle) onto the court later.
3. As a user with shaky handheld footage, I get a validity flag per frame so I can discard frames where the court could not be reliably located.

### 1.6 Functional requirements

- **FR1** — Accept a single video file (mp4/mov) as input.
- **FR2** — Detect the court automatically (no manual corner clicking).
- **FR3** — Detect players and extract a ground-contact point (foot midpoint) per player.
- **FR4** — Compute a homography per frame (or per stable segment) mapping image → court metres.
- **FR5** — Output per-frame: frame index, timestamp, per-player court `(x, y)`, homography (9 floats), validity flag, reprojection error.
- **FR6** — Emit a visual overlay video (optional flag) for QA: court template reprojected onto the frame + player dots.

### 1.7 Non-functional requirements

- **NFR1 — Robustness to handheld motion:** position output should remain stable (sub-decimetre jitter on a stationary player) despite camera shake.
- **NFR2 — Graceful degradation:** frames where the court is occluded or off-screen are flagged invalid, not silently wrong.
- **NFR3 — Reproducibility:** same input + config → same output.
- **NFR4 — Throughput:** offline; target faster-than-realtime on a single GPU but no hard latency bound.

### 1.8 Success metrics

- **Court-plane accuracy:** reprojection error of the court template < 3 px median on valid frames; positional error < ~15 cm near camera on a calibration clip with known marks.
- **Coverage:** ≥ 90% of rally frames flagged valid on representative handheld clips.
- **Stability:** stationary-player position jitter < 10 cm std-dev across valid frames.

### 1.9 Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Court detector assumes broadcast angle; handheld angles vary | Run detection per-frame; add homography smoothing/tracking; allow swapping in a learned court-keypoint detector |
| Phone lens distortion bends court lines | Undistort frames before detection (calibrate phone once, or approximate) |
| Court partially out of frame on some shots | Validity flag + temporal interpolation from neighbouring valid frames |
| SoloShuttlePose is singles-only | Scope to singles now; doubles is explicit non-goal |
| Per-frame detection is noisy/flickery | Temporal smoothing of the homography (see Tech Design §2.6) |

---

## Part 2 — Technical Design

### 2.0 Source frameworks (real repos)

- **SoloShuttlePose** — base for automatic court + player + net detection (singles, MIT license): https://github.com/sunwuzhou03/SoloShuttlePose
  - Entry point `main.py`; source under `src/`; CLI flags include `--folder_path`, `--result_path`, `--force`, `--court`, `--net`, `--players`, `--ball`, `--trajectory`, `--traj_len`. Results (court keypoints, player detections) are written under `--result_path`. See `docs/run.md` and `docs/Quick-Start.md`.
- **Automated-Hit-frame-Detection** — learned **Court Keypoint-RCNN** (`court_kpRCNN.pth`), more robust to varied angles; good drop-in if SoloShuttlePose's detector struggles on handheld: https://github.com/arthur900530/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis
- **MonoTrack** — reference for a robust deep court-detection model (fixed-camera broadcast assumption, so reference only): https://github.com/jhwang7628/monotrack
- **OpenCV** — homography (`cv2.findHomography`), perspective transform, undistortion, optical flow for homography tracking.

> Note: SoloShuttlePose is ~79% Jupyter Notebook, ~21% Python, MIT-licensed. Treat its `src/` modules as a library to import, and wrap them rather than fork heavily.

### 2.1 High-level architecture

```
Uploaded video
      │
      ▼
[0] Ingest & undistort frames ───────────────┐
      │                                       │ (camera intrinsics, optional)
      ▼                                       │
[1] Court detection per frame  ◄──────────────┘
      │   (SoloShuttlePose court module → image-space court keypoints)
      ▼
[2] Court-keypoint → template correspondence
      │   (match detected keypoints to canonical court model in metres)
      ▼
[3] Per-frame homography  H_t = findHomography(image_kpts, court_model_m)
      │
      ▼
[4] Homography validation + temporal smoothing
      │   (reprojection error gate; smooth across frames; fill gaps)
      ▼
[5] Player detection → foot midpoint (image space)
      │   (SoloShuttlePose players module)
      ▼
[6] Project foot point through H_t → court (x, y) in metres
      │
      ▼
[7] Emit per-frame records (CSV/JSON) + optional overlay video
```

### 2.2 Canonical court model

Define the court once in metres. Full doubles outer boundary is 13.40 m (length) × 6.10 m (width). Place origin at one corner, x along length, y along width.

```python
# court_model.py
import numpy as np

# Canonical badminton court key reference points, in metres.
# Origin (0,0) at one outer corner; x = along length (13.40 m), y = along width (6.10 m).
COURT_LENGTH = 13.40
COURT_WIDTH  = 6.10
SINGLES_INSET = 0.46          # singles sidelines are 0.46 m inside the doubles sidelines
SHORT_SERVICE_FROM_NET = 1.98 # short service line distance from the net
NET_X = COURT_LENGTH / 2.0    # net runs across the middle

# A dictionary of named court landmarks the detector is expected to localize.
# Keep names aligned to whatever the court detector outputs so matching is 1:1.
COURT_POINTS_M = {
    "outer_tl": (0.0,            0.0),
    "outer_tr": (0.0,            COURT_WIDTH),
    "outer_bl": (COURT_LENGTH,   0.0),
    "outer_br": (COURT_LENGTH,   COURT_WIDTH),
    "net_l":    (NET_X,          0.0),
    "net_r":    (NET_X,          COURT_WIDTH),
    "short_service_top_l": (NET_X - SHORT_SERVICE_FROM_NET, 0.0),
    "short_service_top_r": (NET_X - SHORT_SERVICE_FROM_NET, COURT_WIDTH),
    "short_service_bot_l": (NET_X + SHORT_SERVICE_FROM_NET, 0.0),
    "short_service_bot_r": (NET_X + SHORT_SERVICE_FROM_NET, COURT_WIDTH),
}
```

> Important for your downstream model: whatever ordering/þnaming convention the original training data used for court coordinates, mirror it here exactly. The homography is only as meaningful as the consistency of this model across training and inference.

### 2.3 Ingest & undistortion

```python
# ingest.py
import cv2
import numpy as np

def frame_iter(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, idx / fps, frame
        idx += 1
    cap.release()

def make_undistorter(K, dist, size):
    """K: 3x3 intrinsics, dist: distortion coeffs, size: (w, h).
    If you have not calibrated the phone, skip undistortion (return identity)."""
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha=0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_16SC2)
    def undistort(frame):
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    return undistort
```

If you have not calibrated the phone, run without undistortion first; add it once you can capture a checkerboard clip with the same phone/lens.

### 2.4 Court detection wrapper (SoloShuttlePose)

SoloShuttlePose runs as a CLI over a folder and writes results to `--result_path`. The integration-friendly approach is to import its court-detection module directly from `src/` and call it per frame. Because the repo is notebook-heavy, the exact symbol names should be confirmed against `src/` — the wrapper below isolates that dependency so only one function changes if the API differs.

```python
# court_detect.py
# Thin wrapper around SoloShuttlePose's court detection.
# Confirm the import path/symbol against the repo's src/ (e.g. src.tools.court or similar).
import numpy as np

class CourtDetector:
    def __init__(self, weights_path=None, device="cuda"):
        # Lazy import so the rest of the pipeline is testable without the model.
        from src.models.court import CourtDetectNet   # <-- verify actual path in repo
        self.net = CourtDetectNet(weights_path, device=device)

    def detect(self, frame_bgr):
        """Return dict {landmark_name: (u, v)} in image pixels, or None if not found.
        Map SoloShuttlePose's raw court keypoint output to the names in COURT_POINTS_M."""
        raw = self.net.infer(frame_bgr)      # <-- shape/format per repo
        if raw is None:
            return None
        # EXAMPLE mapping — adapt indices to the repo's keypoint ordering:
        return {
            "outer_tl": tuple(raw[0]),
            "outer_tr": tuple(raw[1]),
            "outer_br": tuple(raw[2]),
            "outer_bl": tuple(raw[3]),
            # ... add net / service-line points if the detector provides them
        }
```

If SoloShuttlePose's detector proves flaky on handheld angles, swap this class's internals for arthur900530's Court Keypoint-RCNN (`court_kpRCNN.pth`) — same interface, different model.

### 2.5 Per-frame homography

```python
# homography.py
import cv2
import numpy as np
from court_model import COURT_POINTS_M

def compute_homography(image_points: dict, min_points: int = 4):
    """image_points: {name: (u, v)}. Returns (H, reproj_err, n_used) or (None, inf, 0).
    H maps image pixels -> court metres."""
    names = [n for n in image_points if n in COURT_POINTS_M]
    if len(names) < min_points:
        return None, float("inf"), 0

    src = np.array([image_points[n] for n in names], dtype=np.float64)      # pixels
    dst = np.array([COURT_POINTS_M[n] for n in names], dtype=np.float64)    # metres

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return None, float("inf"), 0

    # Reprojection error in METRES, then we also report a pixel-space check.
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err_m = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
    return H, err_m, int(mask.sum())

def image_to_court(H, uv):
    """Project a single image point (u, v) to court metres (x, y)."""
    pt = np.array([[uv]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])
```

### 2.6 Validation + temporal smoothing (the handheld-critical part)

Per-frame detection flickers. Two defenses: gate on reprojection error, and smooth the homography across time. Smoothing a homography directly is unprincipled (its 8 DOF aren't independent), so smooth in a stable parameterization: decompose into the 4 court-corner image positions (which vary smoothly with camera motion), low-pass those, and recompute H.

```python
# smoothing.py
import numpy as np
import cv2
from collections import deque
from court_model import COURT_POINTS_M

ERR_GATE_M = 0.20   # reject frames whose court reprojection error exceeds 20 cm

class HomographySmoother:
    """Smooths the FOUR outer court corners (in image space) over time,
    then recomputes H from the smoothed corners. Robust to per-frame flicker."""
    def __init__(self, alpha=0.6, win=5):
        self.alpha = alpha
        self.win = win
        self.buf = deque(maxlen=win)   # recent corner sets, each 4x2 image pts
        self.corner_names = ["outer_tl", "outer_tr", "outer_br", "outer_bl"]

    def _corners_from_image_points(self, image_points):
        if not all(n in image_points for n in self.corner_names):
            return None
        return np.array([image_points[n] for n in self.corner_names], dtype=np.float64)

    def update(self, image_points):
        corners = self._corners_from_image_points(image_points)
        if corners is None:
            # fall back to prediction from buffer if available
            if not self.buf:
                return None
            corners = self.buf[-1]
        self.buf.append(corners)
        # median over window (robust) then EMA toward it
        med = np.median(np.stack(self.buf), axis=0)
        if len(self.buf) == 1:
            smoothed = corners
        else:
            smoothed = self.alpha * med + (1 - self.alpha) * self.buf[-1]
        # recompute H from smoothed corners
        dst = np.array([COURT_POINTS_M[n] for n in self.corner_names], dtype=np.float64)
        H, _ = cv2.findHomography(smoothed, dst, cv2.RANSAC, 5.0)
        return H
```

An alternative / complement for fast pans: track the homography between consecutive frames with optical flow on static background points and chain it (`H_t = H_det` when detection is good, else `H_t = H_{t-1} · H_interframe`), re-anchoring whenever a clean detection arrives. Start with corner-smoothing; add flow chaining only if gaps are long.

### 2.7 Player foot point (SoloShuttlePose)

The homography is valid only on the court plane (z = 0), so the player's **foot midpoint** is the correct ground-contact point — not torso or head.

```python
# players.py
import numpy as np

def foot_midpoint_from_pose(keypoints_xy, conf=None, conf_thr=0.3):
    """COCO-17 ankles are indices 15 (left) and 16 (right).
    Returns (u, v) midpoint of ankles, or None if both low-confidence."""
    L_ANKLE, R_ANKLE = 15, 16
    pts = []
    for i in (L_ANKLE, R_ANKLE):
        if conf is None or conf[i] >= conf_thr:
            pts.append(keypoints_xy[i])
    if not pts:
        return None
    pts = np.array(pts, dtype=np.float64)
    return tuple(pts.mean(axis=0))

def foot_point_from_bbox(bbox_xyxy):
    """Fallback when pose is unavailable: bottom-center of the player box."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, float(y2))
```

### 2.8 Orchestrator & output schema

```python
# run_pipeline.py
import csv, json
import numpy as np
from ingest import frame_iter
from court_detect import CourtDetector
from homography import compute_homography, image_to_court
from smoothing import HomographySmoother, ERR_GATE_M
from players import foot_midpoint_from_pose, foot_point_from_bbox

def run(video_path, out_csv, court_weights=None, player_detector=None,
        smooth=True, emit_overlay=False):
    court = CourtDetector(court_weights)
    smoother = HomographySmoother() if smooth else None

    rows = []
    for idx, ts, frame in frame_iter(video_path):
        image_points = court.detect(frame)            # {name:(u,v)} or None
        H, err_m, n_used = (None, float("inf"), 0)
        if image_points:
            H, err_m, n_used = compute_homography(image_points)
            if smoother is not None:
                Hs = smoother.update(image_points)
                if Hs is not None:
                    H = Hs

        valid = (H is not None) and (err_m <= ERR_GATE_M)

        # players: replace with SoloShuttlePose player module output
        players = player_detector(frame) if player_detector else []
        for p in players:
            foot = (foot_midpoint_from_pose(p["keypoints"], p.get("conf"))
                    or foot_point_from_bbox(p["bbox"]))
            if valid and foot is not None:
                cx, cy = image_to_court(H, foot)
            else:
                cx, cy = (None, None)
            rows.append({
                "frame": idx, "timestamp": round(ts, 4),
                "player_id": p["id"],
                "foot_u": None if foot is None else round(foot[0], 2),
                "foot_v": None if foot is None else round(foot[1], 2),
                "court_x": None if cx is None else round(cx, 3),
                "court_y": None if cy is None else round(cy, 3),
                "valid": valid,
                "reproj_err_m": None if err_m == float("inf") else round(err_m, 4),
                "H": None if H is None else H.flatten().tolist(),
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r["H"] = json.dumps(r["H"])
            w.writerow(r)
    return out_csv
```

**Output schema (one row per player per frame):**

| field | type | meaning |
|-------|------|---------|
| `frame` | int | 0-based frame index |
| `timestamp` | float | seconds from start |
| `player_id` | int | stable id (top/bottom player) |
| `foot_u`, `foot_v` | float | foot point in image pixels |
| `court_x`, `court_y` | float\|null | position on court in metres (null if frame invalid) |
| `valid` | bool | court located & within error gate |
| `reproj_err_m` | float | court template reprojection error (metres) |
| `H` | json | 3×3 homography, row-major, image→court |

### 2.9 Colab quick-start

```python
# CELL 1 — clone + deps
!git clone https://github.com/sunwuzhou03/SoloShuttlePose.git
%cd SoloShuttlePose
!pip install -q opencv-python-headless numpy torch torchvision
# follow docs/Quick-Start.md to fetch the model weights

# CELL 2 — upload video
from google.colab import files
up = files.upload()
import os; os.makedirs("videos", exist_ok=True)
video = next(iter(up)); os.replace(video, f"videos/{video}")

# CELL 3 — sanity-run SoloShuttlePose's own pipeline first (verify weights work)
!python main.py --folder_path videos --result_path results \
                --court True --players True --ball False

# CELL 4 — then run the custom court-position pipeline (files from this design)
# Put court_model.py, ingest.py, court_detect.py, homography.py, smoothing.py,
# players.py, run_pipeline.py in the repo root, then:
from run_pipeline import run
run(f"videos/{video}", "results/court_positions.csv",
    court_weights="<path to court weights>", smooth=True)
```

### 2.10 Validation & QA

- **Overlay video:** reproject the full court model through `H_t` back onto each frame; lines should hug the painted court. Drift = bad homography.
- **Stationary check:** ask someone to stand still through several pans; their `court_x, court_y` should barely move on valid frames (NFR1).
- **Known-distance check:** measure a real on-court distance (e.g. service line to net) and confirm the projected metres match.
- **Coverage report:** % valid frames per rally; investigate rallies below threshold.

### 2.11 Build order (milestones)

1. **M1 — Static-camera spike:** one homography for a tripod clip; validate overlay. Proves the court-model + findHomography path.
2. **M2 — Per-frame detection:** wire SoloShuttlePose court module; per-frame H with error gate; overlay video.
3. **M3 — Handheld robustness:** add corner-smoothing; (optional) optical-flow chaining; coverage + jitter metrics.
4. **M4 — Players → positions:** foot midpoint + projection; emit CSV/JSON schema.
5. **M5 — Hardening:** undistortion, doubles investigation, swap-in learned court detector if needed.

### 2.12 Open questions to confirm before building

- Which exact court-keypoint ordering/naming did the **downstream model's training data** use? The court model in §2.2 must match it for positions to be comparable.
- Does the downstream model expect positions in metres, or normalized court units (0–1)? Trivial to change in `COURT_POINTS_M`, but must match.
- Singles only, confirmed? (SoloShuttlePose constraint.)
- Is phone-camera calibration available for undistortion, or do we start without it?
