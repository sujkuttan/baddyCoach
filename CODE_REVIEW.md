# BaddyCoach — Critical Code Review

## Executive Summary

BaddyCoach is an ambitious, well-scaffolded badminton post-match analysis system: a 13-stage FastAPI pipeline (court → players → shuttle → pose → hits → strokes → rallies → attribution → 5 analytics → coach), a clean React dashboard, and a faithful port of the BST-CG transformer. The *architecture* is genuinely good — staged, artifact-based, testable. But the **core ML value chain is not reliably wired end-to-end**: the BST classifier very likely falls back to a crude rule-based heuristic in the backend (sequence-length mismatch + 20–40% "unknown" rate the team already documents), TrackNetV3 is a non-standard custom reimplementation that wastes most of its own output and has no trajectory rectification, the SoloShuttlePose court detector is the only one of the three integrated reasonably well, and several analytics ("technique score", "recovery time", "court coverage") are pixel-space heuristics that are either perspective-skewed or outright broken. There is also a stage-ordering bug that systematically biases rally winners. Coaching output is a YAML rule engine over these shaky metrics — analytics, not coaching.

**Maturity rating: Advanced prototype / pre-MVP.** It runs and produces a plausible-looking report, but a meaningful fraction of the numbers shown to a user are unreliable or computed from fallback paths, and it would not survive contact with real broadcast/club footage outside the developer's environment without fixes.

The single most emphasized concern in the brief — *are TrackNetV3 and BST actually integrated and leveraged?* — the honest answer is: **partially for the court model, poorly for TrackNet, and probably not at all in the live backend for BST.** Details below.

---

## Model Integration Assessment (the headline issue)

### TrackNetV3 — minimally/incorrectly leveraged
- `backend/app/models/tracknet.py:17-18` defines a **custom UNet with `in_channels=27, num_classes=8`** — 9 RGB frames in, 8 heatmap channels out. This is *not* the published TrackNetV3 architecture (which is a 3-frame / 3-output VGG-style net with an accompanying **InpaintNet** for trajectory rectification). So this requires bespoke custom-trained weights; the official `TrackNet_best.pt` will not `load_state_dict` into it, and `__init__` has **no try/except around `load_state_dict`** (`tracknet.py:116`) so a key mismatch crashes the whole shuttle stage.
- Even on its own terms it **throws away 7 of its 8 output channels** — both `predict` (`tracknet.py:163`) and `predict_batch` (`tracknet.py:207`) take only `[:, 0]`. The multi-frame heatmap design is half-built.
- If the model fails to load, `predict_batch` silently returns **all-zero positions for every frame** (`tracknet.py:223-227`) — no error surfaced. Zero shuttle data then poisons hit detection, strokes, and every analytic downstream.
- **InpaintNet is referenced** (`settings.py:17`) and required by the downloader (`model_downloader.py:200-201`) but **never used** — the original repo's trajectory denoising/gap-filling is simply not implemented. Missing shuttle detections are only crudely linearly interpolated per-clip later (`strokes.py:130-137`).

### BST — likely not running in the backend; falls back to a skewed heuristic
- The transformer port itself (`bst_model.py`) is faithful (TCN + temporal/cross/interactional transformers + Clean Gate). Good.
- **Critical: sequence-length mismatch.** `bst.py` detects the checkpoint's true `seq_len` from `embedding_tem` and stores it (`bst.py:79`), but the backend clip builder uses a hardcoded `SEQ_LEN = 30` (`strokes.py:14`) and **never reads `classifier.seq_len`**. The colab pipeline *does* adapt to the detected length (`colab/pipeline.py:1348-1350`) — which is strong evidence the real weights are *not* seq_len 30 (the downloader comment even says CG-AP is **seq_len=100**, `model_downloader.py:10`). A 30-length clip added to a 101-length positional embedding is a shape error → caught at `bst.py:165` → **every clip falls back to `_rule_based_predict`.** The team's own note ("BST weights... Falls back to rule-based classification", `AGENTS.md:129`) and "~20-40% unknown predictions" (`AGENTS.md:95`) corroborate that BST inference is unreliable in practice.
- **Weight path inconsistency:** settings points to `BST/weight/bst_CG_JnB_bone_merged.pt` (`settings.py:22`) while the downloader writes `ckpts/bst/bst_CG_AP.pt` (`model_downloader.py:27`) — a different variant *and* a different path. After running the downloader the configured path still doesn't exist → silent rule-based fallback.
- **The model's own player-side prediction is discarded.** BST outputs 25 classes with `Top_/Bottom_` distinction, but `map_to_coach_class` (`bst.py:27-35`) collapses both halves to 12 classes, throwing away who hit the shuttle — then the pipeline re-derives attribution with a separate heuristic. A free, model-grade signal is wasted.
- **Two divergent, dead preprocessing implementations** exist alongside the real one: `bst_preprocessing.py` (its `prepare_stroke_clips_from_pipeline` feeds **all-zero player positions** at `L325` and **un-normalized raw-pixel joints**) and the 144-dim `bst_features.py` (a completely different feature scheme incompatible with BST-CG's input contract). Neither is used by the live pipeline (confirmed: only tests/docs import them) — they're confusing dead code that a maintainer could wire in by mistake.

### SoloShuttlePose — the best-integrated of the three
- The court KeypointRCNN port (`court.py:70-168`) is reasonable: correct BGR handling, the 6→4 corner selection, geometric validation, a sensible fallback chain (model → color/Hough → proportional), and temporal homography smoothing. This is the one external model used competently.
- **But:** SoloShuttlePose's player-region filtering and pose pipeline are *not* reused — the app rolls its own YOLOv8 top-2 + RTMPose path instead, which is weaker (see below). And when court detection falls back, `valid=False` is set (`court.py:208`) but downstream code (attribution, footwork) **uses the homography regardless of the `valid` flag** → garbage court coordinates pass silently.

**Other models worth integrating** ("other implementations" question): TrackNetV3's own InpaintNet (already half-referenced); ShuttleSet/CoachAI's player-region court filtering; MMPose RTMO/RTMPose-x for stronger pose; ByteTrack/BoT-SORT (already in ultralytics) for stable 2-player tracking instead of per-frame top-2; and using BST's native Top/Bottom output for attribution.

---

## 1. Completeness

**Implemented:** video upload + H.264 transcode, court homography, player tracking, shuttle tracking, pose, hit localization, stroke classification (model or fallback), rally segmentation, player attribution, court/footwork/fitness/tactical/technical analytics, a YAML rule coach, a separate "shuttle_coach" metrics+RAG system, WebSocket progress, and a full React dashboard. That's broad coverage of "what a badminton coach should have" on paper.

**Incomplete / dead-ends / stubs:**
- **`shuttle_coach` endpoint is a dead-end for backend jobs.** Its loader *requires* a `player_detections.parquet` table (`loader.py:19-26`), but the backend stores players as `players.json` (`artifacts.py:12-15`, `players.py:128`) — so `analyze()` raises `Missing required tables: ['player_detections']` and the `/api/shuttle-coach/analyze/{job}` endpoint returns 400 on every real job. Its `movement`/`tactical` capabilities also need `court_x/court_y` columns the backend never writes. This whole subsystem (metrics, feedback, Gemini RAG) was built for the colab data layout and is effectively non-functional in the deployed API.
- **Two parallel coaching engines** (`coach/engine.py` used in the pipeline, vs `shuttle_coach/` used by the broken endpoint) — overlapping purpose, no shared logic.
- `serving_player_id` is always `None` (`rallies.py:144`); `match_id` always `None`.
- No progress tracking *across sessions* (per-player history/trends over multiple uploads) — "progress tracking" as a coaching feature doesn't exist; each job is independent.
- No drill *recommendation logic* beyond static strings attached to YAML rules.

**Missing error handling / validation:**
- Upload accepts any file with an allowed extension — **no size limit, no MIME/content sniffing, no duration check** (`routes.py:185-207`) despite `max_video_length_seconds` existing in settings.
- Pipeline assumes exactly 2 singles players throughout; doubles, warm-up footage, or replays will silently produce wrong attribution.

---

## 2. Effectiveness & Correctness

- **Stage-ordering bug → broken rally winners.** Stages run rally_segmentation (index 7) *before* player_attribution (index 8) (`routes.py:79-80`). But `rallies.py` computes `winner_player_id` from `last_shot["player_id"]` (`rallies.py:129-142`) — which **isn't assigned yet**. So `last_pid` is NaN, and the "opponent wins on error" branch always resolves to `player_1` (`rallies.py:140`). Rally win attribution is systematically skewed. (AGENTS.md claims this ordering is intentional because attribution needs rally structure — true for *attribution*, but rally *winner* computation must move after attribution.)
- **Hit detection has no absolute notion of a hit.** It flags the top 10% of frames by a weighted score via a fixed 90th-percentile threshold (`hits.py:37`). Number of "hits" scales with video length, not actual shots — false positives on sparse footage, missed hits on dense rallies.
- **RTMPose keypoint rescale looks transposed.** Crops are resized to `(192, 256)` (W×H) (`rtmpose.py:36`) but decoded as `x/256` and `y/192` (`rtmpose.py:62-63`) — width and height divisors swapped. For non-square boxes this mislocates every joint, corrupting BST features, technique scores, and footwork. Also `estimate` hardcodes input name `"input"` (`rtmpose.py:75`) while `estimate_batch` uses `self.input_name` (`rtmpose.py:87`).
- **Pose cross-contamination.** When a player's bbox is missing for a frame, pose estimation borrows *another player's* bbox (`pose.py:138-157`) and records the resulting keypoints under the original player_id — injecting the wrong player's body into the data.
- **"Technique score" is not technique analysis.** `technical.py:53-93` uses a single frame, only the **left** shoulder/wrist (index 5/9) regardless of handedness, arbitrary pixel divisors (`/100`, `/80`…) plus constant offsets so the floor is ~0.2–0.4. It's perspective-dependent and biomechanically meaningless.
- **Coach output is rule-based, not AI.** `coach/engine.py` is a YAML threshold engine over the above metrics. The only LLM (Gemini, grounded with citation enforcement — actually a nice anti-hallucination design, `gemini.py:42-56`) sits behind the broken shuttle_coach endpoint, so users never see it.

Net: outputs *look* like coaching but are a chain of heuristics, several of which are broken or fed by fallback data.

---

## 3. Hardcoding & Configuration

- **`Settings` is a `BaseModel`, not `BaseSettings`** (`settings.py:5`) — there is **no environment/`.env` override**. All model paths are hardcoded relative paths (`ckpts/...`, `BST/weight/...`, `settings.py:16-22`); nothing works outside a CWD that contains those exact folders.
- **GPU is effectively never enabled via the API:** `gpu_enabled` defaults `False` and routes always pass `StageConfig(gpu_enabled=False)` (`routes.py:37`), so `settings.device` returns `"cpu"` (`settings.py:30-39`). The entire `gpu_batch.py` tier system is dead in the deployed path, and `requirements.txt` pins **CPU-only `onnxruntime`** (`requirements.txt:13`), so RTMPose can't use CUDA even if enabled.
- **Magic numbers everywhere:** `fps = 30.0` hardcoded in strokes (`strokes.py:286`), rallies (`rallies.py:145`), fitness (`fitness.py:33`) — actual video FPS is read but ignored. Default frame size `1280×720` (`strokes.py:195`, `court_position.py:29`). `court_mid_y = 360/300/600` fallbacks (`attribution.py:31`, `players.py:29`). Court dims `13.4/5.18` repeated in ~6 files. Hit-weight constants (`hits.py:13-16`), `LOOKBACK=5`, recovery `threshold=0.3`, jump filter `500px`, dedup gap `8`, rally gap `60/45/25/15`, proportional-corner ratios `0.08/0.72/0.28` (`court.py:164-168`). None are externalized.
- **Fake data presented as real:** `_generate_synthetic_detections` fabricates static player boxes when YOLO returns nothing (`players.py:65-77`); those then drive attribution, court coverage, and fitness with no flag in the report telling the user the data is synthetic.
- **Hardcoded `gemini-2.0-flash`** model id (`gemini.py:32`).
- **Dependency drift:** `gdown` is used (`model_downloader.py:44`) but absent from `requirements.txt`; ML libs are unpinned (`torch>=`, `ultralytics>=`) — reproducibility risk.

---

## 4. Gaps & Misses

**Architecture/coupling:** No model abstraction layer — each stage hardcodes its model import and re-derives `vid_w/vid_h` and court dims independently. The colab pipeline is a **3,400-line re-implementation** of the whole backend with models inlined and the coach rules duplicated (backend reads YAML, colab hardcodes them) — high drift risk and no test comparing the two outputs. Three separate BST-preprocessing implementations. Two coach engines.

**Security:**
- **No authentication/authorization on any endpoint** (`routes.py`) — anyone can upload, list all jobs, and fetch any job's video/report by id.
- **Unvalidated upload** (size/content) → trivial disk-fill DoS.
- `ffmpeg` invoked via subprocess on the uploaded file (`routes.py:166-177`) — args are list-form (no shell injection), acceptable, but unbounded 600s transcode per upload is a resource risk.
- `torch.load(..., weights_only=False)` (`bst.py:60`, `court.py:93`) deserializes arbitrary pickles — fine for trusted local weights, dangerous if weight paths ever become user-controlled.
- Gemini API key read from env and passed around (`routes.py:305-309`) — OK, but the user `question` is forwarded to the LLM (low-risk prompt injection).

**Performance:** CPU-only path means per-frame UNet (TrackNet) + per-crop ONNX (RTMPose) + YOLO over potentially thousands of frames — minutes-to-hours per video. No result caching, no early exit, no streaming. `rally_lookup` is O(rallies × shots) (`rallies.py:115-120`). Frames are all decoded into a Python list in memory (`routes.py:148-163`) — a long 1080p clip can OOM.

**Tests/logging/monitoring:** ~70 tests exist but lean on synthetic inputs and mocked models — they validate plumbing, not real inference (e.g. they wouldn't catch the seq_len fallback or the RTMPose transpose). Logging is `print()` statements (`bst.py:103`) — no structured logging, no metrics, no per-stage timing surfaced.

**Data/privacy:** User videos are stored on disk under `data/jobs/{id}` indefinitely with no retention/cleanup, no encryption, no consent flow, and are served unauthenticated. For footage of identifiable people this is a real privacy gap.

**Frontend (from sub-review):** No baked-in mock data (good), clean API/WS layer, but missing null-guards on nested report fields (`report.tactical[player]`, `report.shots.map`) will crash on partial/malformed reports; no auth.

---

## 5. Usefulness & Value

**Value proposition:** "Upload a match, get shot distribution, rally stats, court coverage, fatigue trend, and coaching tips" is genuinely attractive to club players and coaches. The dashboard is the strongest part — it presents this credibly.

**Where it falls short:** The insights are only as good as the underlying numbers, and today many are unreliable: stroke types come from a clear-biased fallback heuristic, "technique scores" are meaningless, "recovery time" is essentially always 0 (see §6), court coverage measures shuttle pixels not player movement, and rally winners are mis-attributed. A coach who knows the sport will spot these quickly and lose trust. There's no cross-session progress tracking, no comparison to reference technique, and no actionable drill logic beyond canned strings — so it's **analytics, not coaching**. Combining shuttle tracking + stroke classification *could* yield genuinely actionable insight (e.g. "you lift cross-court under pressure and get punished"), but the current pipeline doesn't connect shot type → outcome → recommendation in a trustworthy way.

**What would make a player keep using it:** reliable stroke labels, real per-player movement/distance, multi-match progress trends, and one or two *specific, correct* observations per match. What will make them abandon it: numbers that are obviously wrong for footage they understand.

---

## 6. Data Skewness (specifically requested)

This is a real and multi-pronged problem:

1. **Stroke-type over-classification via fallback.** When BST doesn't run (likely, per §"BST") or returns `unknown`, `_rule_based_predict` is used — and it returns **`clear` in 4 of its branches including the default** (`bst.py:200-213`), with `smash`/`net_shot` as the other frequent outputs. Result: stroke distribution collapses toward 2–3 classes regardless of reality.
2. **"Unknown → second-best" override** (`bst.py:152-160`) and **majority-vote relabeling** of low-confidence shots to the neighbor majority (`strokes.py:271-284`) both **amplify whichever class is already dominant** — a classic skew accelerator.
3. **Rally-winner skew to player_1** from the stage-ordering bug (§2, `rallies.py:140`).
4. **Per-player shot counts are artificially forced ~50/50** by the strict-alternation assumption (`attribution.py:74-75`) — this *hides* real shot-count asymmetry rather than measuring it, and if the rally's first hitter is mis-detected, an entire rally's shots flip to the wrong player.
5. **Court-coverage skew by perspective.** `court_position` uses the *shuttle's* pixel position in a 3×3 pixel grid, not the player's court position, and ignores the homography entirely (`court_position.py:44-46, 73-74`) — the far player occupies fewer/higher pixels, systematically distorting zone distribution.
6. **Distance-covered skew between players.** Footwork sums frame-to-frame **pixel** displacement with a single uniform pixels-per-meter scale (`footwork.py:51-64`) — the far player's real movement maps to fewer pixels, so their distance is consistently *under*-reported relative to the near player. Noise integration also inflates absolute distances.
7. **Recovery time is effectively broken (not just skewed):** `base_position` is in pixels but the return `threshold = 0.3` (`footwork.py:128`) is a court-meter-scale value, so `distances < 0.3` is almost never true → `recovery_times` is usually empty → `avg_recovery` ≈ 0 for everyone. The recovery loop also scans *all* players' poses, not the shot owner's (`footwork.py:119`).
8. **Rally segmentation accuracy** is hostage to the percentile-based hit detector (§2) — over- or under-segmenting based on video length.

---

## 7. Upgrades & Recommendations (prioritized)

### Critical (correctness — do first)
1. **Fix BST seq_len wiring**: read `classifier.seq_len` in `strokes.py` instead of the constant `30`, and reconcile the weight path (`settings.py:22` vs `model_downloader.py:27`). Then verify BST actually runs (log model-vs-fallback rate per job). *Quick win, highest impact.*
2. **Move rally-winner computation after player_attribution** (reorder stages in `routes.py:79-80` or recompute winners in attribution). *Quick win.*
3. **Fix the RTMPose x/y rescale transpose** (`rtmpose.py:62-63`) and unify the input-name handling. Validate against a known image. *Quick win, fixes pose-derived everything.*
4. **Fix recovery-time units** (`footwork.py:128`) and scope it to the shot's player. *Quick win.*
5. **Stop using the homography when `court["valid"]` is False** (or down-weight/flag those court coordinates) — propagate the validity flag into attribution/analytics.
6. **Surface fallback/synthetic data in the report** — never present synthetic detections (`players.py:65-77`) or rule-based strokes as if they were model output; add a confidence/data-quality banner.

### High (reliability & integration)
7. **Make TrackNet integration honest**: either adopt the real published TrackNetV3 (3-frame) with its weights and add **InpaintNet** trajectory rectification, or document this as a custom net and ship matching weights — and wrap loading in try/except that degrades visibly, not silently to zeros.
8. **Leverage BST's Top/Bottom output for attribution** instead of (or to cross-check) the alternation heuristic; reconcile the two `COACH_STROKE_CLASSES` lists and mapping functions (`bst.py:12` vs `bst_model.py:383`).
9. **Compute court coverage and distance in court meters via the homography**, not pixel grids — this removes the per-player perspective skew in one move.
10. **Replace per-frame top-2 YOLO** (`yolov8.py:104`) with proper tracking + court-region filtering (reuse SoloShuttlePose's approach) so umpire/audience don't get tracked; this also fixes the AGENTS.md/code contradiction about top-2.
11. **Externalize config**: switch to `pydantic-settings` with env/.env, lift court dims/FPS/thresholds/paths into config, read real video FPS, and add `onnxruntime-gpu` + actually honor `gpu_enabled` in the API.
12. **Add auth + upload validation** (size, duration, MIME) + a retention/cleanup policy for uploaded videos.
13. **Either fix or remove the `shuttle_coach` subsystem** for backend jobs (write `player_detections.parquet` with `court_x/court_y`, or repoint it at the existing artifacts) — and delete the dead `bst_features.py` / `bst_preprocessing.py`.

### Nice-to-have (product & maintainability)
14. **Unify the backend and colab pipelines** (import shared stage code instead of a 3,400-line fork) and add a golden-output regression test across both. The colab pipeline needs to be able to run standalone on google colab or kaggle notebook infra.
15. **Replace the single-frame "technique score"** with temporal swing kinematics (use the BST clip you already build) and handedness detection; or drop it until it's meaningful.
16. **Cross-session progress tracking** (per-player history) — the actual coaching differentiator.
17. **Real, structured logging + per-stage timing + a data-quality score** in each report.
18. **Promote the grounded Gemini narration** (a genuinely good design) into the main report once its inputs are trustworthy.
19. **License audit**: ultralytics YOLOv8 is **AGPL-3.0** (commercial implications), and TrackNetV3 / BST / SoloShuttlePose each carry their own terms — document and comply before any distribution.

---

## Prioritized Action List

1. ☐ Wire `classifier.seq_len` into clip building + fix BST weight path; log model-vs-fallback rate. *(Critical, quick)*
2. ☐ Reorder so rally winners compute after attribution. *(Critical, quick)*
3. ☐ Fix RTMPose x/y divisor transpose + input-name. *(Critical, quick)*
4. ☐ Fix recovery-time pixel/meter unit mismatch + scope to shot owner. *(Critical, quick)*
5. ☐ Respect `court.valid`; don't use bad homography. *(Critical)*
6. ☐ Flag synthetic/fallback data in the report. *(Critical)*
7. ☐ TrackNet: adopt official arch+InpaintNet OR ship matching weights + fail loudly. *(High)*
8. ☐ Use BST Top/Bottom for attribution; merge duplicate class maps. *(High)*
9. ☐ Court coverage & distance in meters via homography. *(High)*
10. ☐ Proper player tracking + court-region filter; kill per-frame top-2. *(High)*
11. ☐ `pydantic-settings` config; real FPS; GPU honored; `onnxruntime-gpu`; add `gdown`, pin ML deps. *(High)*
12. ☐ Auth, upload validation, video retention/privacy policy. *(High)*
13. ☐ Fix or remove `shuttle_coach`; delete dead BST preprocessing modules. *(High)*
14. ☐ Unify backend/colab; golden regression test. *(Nice-to-have)*
15. ☐ Temporal technique analysis + handedness. *(Nice-to-have)*
16. ☐ Cross-session progress tracking. *(Nice-to-have)*
17. ☐ Structured logging + per-stage timing + data-quality score. *(Nice-to-have)*
18. ☐ Promote grounded LLM narration into main report. *(Nice-to-have)*
19. ☐ License compliance audit (AGPL YOLOv8 etc.). *(Nice-to-have)*
