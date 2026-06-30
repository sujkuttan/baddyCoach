import numpy as np
import pandas as pd
from collections import Counter

from app.config.settings import settings
from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, image_to_court, clamp_to_unit
from app.pipeline.shared.bst_preproc import normalize_joints, normalize_joints_court, create_bones, BONE_PAIRS
from app.pipeline.shared.physics import apply_physics_ensemble
from app.models.bst import COACH_STROKE_CLASSES


def _temporal_resample(arr: np.ndarray, target_len: int,
                       zero_is_missing: bool = False) -> np.ndarray:
    """Resample a temporal array to target_len via linear interpolation.

    Supports any trailing dimensions (T, ...) → (target_len, ...).
    When zero_is_missing=True, rows that are all-zero in the source are
    treated as missing data — interpolation only bridges gaps up to
    the nearest valid neighbour, and regions beyond valid data stay zero.

    Args:
        arr: Input array with shape (T, ...).
        target_len: Desired length of the first dimension.
        zero_is_missing: If True, treat all-zero rows as missing values.
    """
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 0 or target_len == 0:
        return np.zeros((target_len, *arr.shape[1:]), dtype=arr.dtype)

    orig = np.arange(arr.shape[0], dtype=np.float64)
    target = np.linspace(0, arr.shape[0] - 1, target_len, dtype=np.float64)

    out = np.zeros((target_len, *arr.shape[1:]), dtype=arr.dtype)

    if not zero_is_missing:
        for idx in np.ndindex(arr.shape[1:]):
            col = arr[(slice(None),) + idx]
            out[(slice(None),) + idx] = np.interp(target, orig, col)
        return out

    # Zero-is-missing mode: interpolate only between non-zero rows.
    # Flatten trailing dims to find per-row zero mask.
    flat = arr.reshape(arr.shape[0], -1)
    row_valid = np.any(flat != 0, axis=1)
    valid_idx = np.where(row_valid)[0]

    if len(valid_idx) < 2:
        # Zero or one valid rows → nothing to interpolate
        return out

    for idx in np.ndindex(arr.shape[1:]):
        col = arr[(slice(None),) + idx]
        # Interpolate only between valid source points
        interp_col = np.interp(target, valid_idx, col[valid_idx])
        out[(slice(None),) + idx] = interp_col

    # Mask out regions in the output that map to source regions OUTSIDE
    # the valid-data hull [valid_idx.min(), valid_idx.max()] — regions
    # before the first valid row and after the last valid row.
    vmin, vmax = valid_idx.min(), valid_idx.max()
    out_flat = out.reshape(target_len, -1)
    for t in range(target_len):
        # Map target position back to source index
        src_pos = t * (arr.shape[0] - 1) / (target_len - 1)
        if src_pos < vmin or src_pos > vmax:
            out_flat[t] = 0.0
    return out.reshape(target_len, *arr.shape[1:])


def _get_keypoints_for_frame(pose_df: pd.DataFrame, frame: int, player_id: str) -> np.ndarray | None:
    """Get (17, 3) keypoints for a frame/player from pose dataframe."""
    if pose_df is None or len(pose_df) == 0:
        return None
    row = pose_df[(pose_df['frame'] == frame) & (pose_df['player_id'] == player_id)]
    if len(row) == 0:
        return None
    raw = row.iloc[0]['keypoints']
    kps = np.array(raw.tolist()) if hasattr(raw, 'tolist') else np.array(raw)
    if kps.ndim == 2 and kps.shape[1] >= 2 and kps.shape[0] >= 17:
        return kps
    return None


def _build_clip(
    frames: list[int],
    shuttle_df: pd.DataFrame | None,
    pose_df: pd.DataFrame | None,
    vid_w: float,
    vid_h: float,
    court_length: float,
    court_width: float,
    seq_len: int,
    player_sides: dict | None = None,
    player_detections: dict | None = None,
    homography: np.ndarray | None = None,
    original_len: int | None = None,
    player_ids: list | None = None,
) -> dict:
    """Build a BST clip from a sequence of frame indices.

    This follows the official BST preprocessing:
    1. Joints normalized by bbox diagonal + center_align (range [-0.X, 0.X])
    2. Bones computed as endpoint differences
    3. Shuttle normalized by court dimensions (range [0, 1])
    4. Position = feet midpoint in court-normalized coords via homography

    Player ordering: p_idx=0 is ALWAYS the "far" player, p_idx=1 is "near".

    When homography is provided, positions are projected to court coordinates
    then normalized by court dimensions (matching BST official preprocessing).
    Falls back to pixel-normalized positions when homography is unavailable.
    """
    n_frames_orig = original_len if original_len is not None else len(frames)
    joints = np.zeros((seq_len, 2, 17, 2), dtype=np.float32)
    shuttle = np.zeros((seq_len, 2), dtype=np.float32)
    pos = np.zeros((seq_len, 2, 2), dtype=np.float32)

    # Build per-frame player lookup: {frame: {side: pid}}
    # Handles YOLO tracking ID switches (player_1 might become player_3 mid-clip)
    frame_player_map = {}
    if player_detections:
        for p in player_detections:
            pid = p.get("id", "")
            side = p.get("side", "near")
            for d in p.get("detections", []):
                f = d["frame"]
                if f not in frame_player_map:
                    frame_player_map[f] = {}
                frame_player_map[f][side] = pid

    # Build detection bbox lookup: {pid: {frame: bbox}}
    det_bbox_lookup = {}
    if player_detections:
        for p in player_detections:
            pid = p.get("id", "")
            det_bbox_lookup[pid] = {}
            for d in p.get("detections", []):
                det_bbox_lookup[pid][d["frame"]] = d["bbox"]

    def _interpolate_bboxes(lookup, target_frames):
        """Linearly interpolate bbox for missing frames."""
        existing = sorted(lookup.keys())
        if not existing:
            return {}
        result = {}
        for f in target_frames:
            if f in lookup:
                result[f] = lookup[f]
            else:
                before = [ef for ef in existing if ef <= f]
                after = [ef for ef in existing if ef >= f]
                if before and after:
                    bf, af = before[-1], after[0]
                    if bf == af:
                        result[f] = lookup[bf]
                    else:
                        ratio = (f - bf) / (af - bf)
                        result[f] = tuple(
                            lookup[bf][i] + ratio * (lookup[af][i] - lookup[bf][i])
                            for i in range(4)
                        )
                elif before:
                    result[f] = lookup[before[-1]]
                elif after:
                    result[f] = lookup[after[0]]
        return result

    # Build interpolated bbox lookups for each player
    clip_frames_set = set(frames[:seq_len])
    interpolated_bboxes = {}
    for pid in list(det_bbox_lookup.keys()):
        interpolated_bboxes[pid] = _interpolate_bboxes(det_bbox_lookup[pid], clip_frames_set)

    # Debug counters for missing data
    bbox_diags = {}
    for pid, iboxes in interpolated_bboxes.items():
        diags = [np.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2) for _, b in iboxes.items()]
        if diags:
            bbox_diags[f"bbox_diag_{pid}_mean"] = float(np.mean(diags))
            bbox_diags[f"bbox_diag_{pid}_std"] = float(np.std(diags))

    debug_clip_stats = {
        "n_frames": min(len(frames), seq_len),
        "n_missing_bbox": 0,
        "n_missing_pose": 0,
        "frame_start": int(frames[0]) if frames else 0,
        "frame_end": int(frames[-1]) if frames else 0,
        **bbox_diags,
    }

    for t, frame in enumerate(frames[:seq_len]):
        if shuttle_df is not None:
            s_row = shuttle_df[shuttle_df['frame'] == frame]
            if len(s_row) > 0:
                s_conf = float(s_row.iloc[0].get('confidence', 1.0))
                if s_conf < settings.shuttle_min_conf:
                    continue
                sx = float(s_row.iloc[0]['x'])
                sy = float(s_row.iloc[0]['y'])
                if settings.bst_shuttle_norm == "court" and homography is not None:
                    sx, sy = image_to_court(homography, (sx, sy))
                    shuttle[t, 0] = max(0.0, min(1.0, sx / court_length if court_length > 0 else 0))
                    shuttle[t, 1] = max(0.0, min(1.0, sy / court_width if court_width > 0 else 0))
                else:
                    shuttle[t, 0] = max(0.0, min(1.0, sx / vid_w if vid_w > 0 else 0))
                    shuttle[t, 1] = max(0.0, min(1.0, sy / vid_h if vid_h > 0 else 0))

    # DO NOT interpolate missing shuttle coordinates. The shuttle array
    # keeps zeros for frames without detections, preserving the sparsity
    # pattern so the model can distinguish real vs missing positions.
    # Interpolation was creating fake continuous trajectories (100/100
    # frames "valid" for every clip), destroying the discriminative signal.

    for t, frame in enumerate(frames[:seq_len]):
        # Resolve active players for THIS frame.
        # When one side is missing (e.g. far player not detected in this frame),
        # infer the missing player as whoever is NOT assigned to near — avoids
        # silently putting player_1 in both slots.
        frame_players = frame_player_map.get(frame, {})
        near_pid = frame_players.get('near')
        far_pid = frame_players.get('far')
        if far_pid is None:
            if near_pid in player_ids and len(player_ids) > 1:
                far_pid = [p for p in player_ids if p != near_pid][0]
            else:
                far_pid = 'player_2'
        if near_pid is None:
            if far_pid in player_ids and len(player_ids) > 1:
                near_pid = [p for p in player_ids if p != far_pid][0]
            else:
                near_pid = 'player_1'
        player_order = [far_pid, near_pid]

        for p_idx, pid in enumerate(player_order):
            kps = _get_keypoints_for_frame(pose_df, frame, pid)
            if kps is not None:
                coords = kps[:, :2]
                # Use interpolated detection bbox for stable normalization
                det_bbox = interpolated_bboxes.get(pid, {}).get(frame)
                if det_bbox is None:
                    # Fall back to keypoint bbox if no detection at all
                    debug_clip_stats["n_missing_bbox"] += 1
                if settings.bst_joint_norm == "court" and homography is not None:
                    joints[t, p_idx] = normalize_joints_court(coords, homography)
                else:
                    joints[t, p_idx] = normalize_joints(coords, det_bbox=det_bbox)

                feet_x = (coords[15, 0] + coords[16, 0]) / 2
                feet_y = max(coords[15, 1], coords[16, 1])
                if homography is not None:
                    court_x, court_y = image_to_court(homography, (feet_x, feet_y))
                    court_x, court_y = clamp_to_unit(court_x / court_length if court_length > 0 else 0,
                                                     court_y / court_width  if court_width  > 0 else 0)
                    pos[t, p_idx, 0] = court_x
                    pos[t, p_idx, 1] = court_y
                else:
                    pos[t, p_idx, 0] = max(0.0, min(1.0, feet_x / vid_w))
                    pos[t, p_idx, 1] = max(0.0, min(1.0, feet_y / vid_h))
            else:
                debug_clip_stats["n_missing_pose"] += 1

    bones = np.zeros((seq_len, 2, len(BONE_PAIRS), 2), dtype=np.float32)
    amp = settings.joint_velocity_amplification
    if amp > 0:
        vel = np.diff(joints, axis=0)  # (seq_len-1, 2, 17, 2)
        vel_mag = np.linalg.norm(vel, axis=-1)  # (seq_len-1, 2, 17)
        vel_mag = np.concatenate([np.zeros((1, 2, 17)), vel_mag], axis=0)
    for i in range(2):
        for t in range(seq_len):
            vm = vel_mag[t, i] if amp > 0 else None
            bones[t, i] = create_bones(joints[t, i], velocity_mag=vm, amp_factor=amp)
    JnB = np.concatenate([joints, bones], axis=-2)
    return {
        'JnB': JnB.reshape(seq_len, 2, -1),
        'shuttle': shuttle,
        'pos': pos,
        'video_len': min(n_frames_orig, seq_len),
        'vid_w': vid_w,
        'vid_h': vid_h,
        'court_length': court_length,
        'court_width': court_width,
        '_debug_clip': debug_clip_stats,
    }


class StrokeClassificationStage:
    name = "stroke_classification"
    input_keys = ["hits", "shuttle", "pose", "court"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        hits_df = artifacts.get_parquet("hits")
        if hits_df is None or len(hits_df) == 0:
            return StageResult.success(metadata={"shot_count": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        pose_df = artifacts.get_parquet("pose")
        court = artifacts.get("court") or {}

        from app.pipeline.shared.models import get_bst

        classifier = get_bst()
        if classifier is None:
            return StageResult.from_error("BST model not available")

        court_length = court.get("court_length", COURT_LENGTH)
        court_width = court.get("court_width", COURT_WIDTH)
        homography = np.array(court["homography"]) if court.get("homography") is not None else None

        vid_w, vid_h = settings.default_frame_width, settings.default_frame_height
        video_res = artifacts.get("video_resolution")
        if video_res:
            vid_w = float(video_res.get("width", vid_w))
            vid_h = float(video_res.get("height", vid_h))
        elif shuttle_df is not None and len(shuttle_df) > 0:
            vid_w = max(float(shuttle_df["x"].max()), 640)
            vid_h = max(float(shuttle_df["y"].max()), 480)

        # Get player side info for consistent ordering (p0=Far, p1=Near)
        players_data = artifacts.get("players") or {}
        player_sides = {}
        for p in players_data.get("players", []):
            player_sides[p["id"]] = p.get("side", "near")

        # Get player detection data for bbox normalization
        player_list = players_data.get("players", [])
        player_ids = [p["id"] for p in player_list if p.get("id")]

        # Deduplicate hits by frame: keep the highest-confidence hit per frame.
        # Duplicate hits can arise from TrackNet producing multiple shuttle
        # detections at the same frame, or from hits.py gap-based dedup missing
        # same-frame duplicates. Without dedup, each duplicate produces a
        # separate shot at the same timestamp with independently classified
        # (potentially different) stroke types.
        hits_df = hits_df.loc[hits_df.groupby("frame")["confidence"].idxmax()].reset_index(drop=True)

        shots = []
        bst_clips_registry = {}
        hit_frames_sorted = sorted(int(h["frame"]) for h in hits_df.to_dict('records'))

        # Phase 1: build all clips (fast, no model inference)
        all_clips = []
        clip_hit_pairs = []  # (frame, hit_dict, frames_list)

        # Log suspiciously close consecutive hits (double-detect diagnostic)
        for i in range(1, len(hit_frames_sorted)):
            gap = hit_frames_sorted[i] - hit_frames_sorted[i - 1]
            if gap < 5:
                logger.warning("Tight hit gap: %d frames between hits at frames %d and %d",
                               gap, hit_frames_sorted[i - 1], hit_frames_sorted[i])

        for _, hit in hits_df.iterrows():
            frame = int(hit["frame"])
            hit_pos = hit_frames_sorted.index(frame)
            use_midpoint = settings.bst_clip_boundary == "midpoint"

            if use_midpoint:
                # Midpoint-to-midpoint convention:
                #   - start = midpoint(prev_hit, curr_hit) → preparation phase
                #   - end   = midpoint(curr_hit, next_hit) → next approach
                #   - contact frame is in the middle of the clip
                #   - temporal resample maps variable-length clips to seq_len
                prev_hit = hit_frames_sorted[hit_pos - 1] if hit_pos > 0 else max(0, frame - 20)
                next_hit = hit_frames_sorted[hit_pos + 1] if hit_pos + 1 < len(hit_frames_sorted) else frame + 20
                start_frame = (prev_hit + frame) // 2
                end_frame = (frame + next_hit) // 2

                # Enforce minimum clip length so short exchanges don't collapse
                if end_frame - start_frame < settings.bst_min_clip_frames:
                    half_floor = settings.bst_min_clip_frames // 2
                    start_frame = max(0, frame - half_floor)
                    end_frame = frame + (settings.bst_min_clip_frames - half_floor)

                clip_frames = list(range(start_frame, end_frame))
                original_n_frames = len(clip_frames)

                # Build clip with actual (non-padded) frame range
                clip = _build_clip(
                    clip_frames, shuttle_df, pose_df,
                    vid_w, vid_h, court_length, court_width,
                    seq_len=original_n_frames,
                    player_sides=player_sides, player_detections=player_list,
                    homography=homography,
                    original_len=original_n_frames,
                    player_ids=player_ids,
                )

                # Temporal resample to match model's expected seq_len
                if original_n_frames != classifier.seq_len:
                    clip['JnB'] = _temporal_resample(clip['JnB'], classifier.seq_len)
                    clip['shuttle'] = _temporal_resample(clip['shuttle'], classifier.seq_len,
                                                        zero_is_missing=True)
                    clip['pos'] = _temporal_resample(clip['pos'], classifier.seq_len)
                clip['video_len'] = min(original_n_frames, classifier.seq_len)

            else:
                # Hit-start convention (default):
                #   - start at the current hit (position 0 = the stroke launch)
                #   - end at the next hit (one inter-hit segment, not two)
                #   - positional encoding expects the stroke at a fixed position
                #   - frames beyond video_len are masked by the model
                #   - bst_min_clip_frames: floor on real frames so short exchanges
                #     don't fall back to unknown (14% of clips <20 frames)
                start_frame = frame
                if hit_pos < len(hit_frames_sorted) - 1:
                    end_frame = min(
                        frame + classifier.seq_len,
                        max(frame + settings.bst_min_clip_frames,
                            hit_frames_sorted[hit_pos + 1])
                    )
                else:
                    end_frame = frame + classifier.seq_len

                # Truncate clip at shuttle landing: if the shuttle stops moving
                # (speed near zero for several frames) before end_frame, end the
                # clip there to avoid dead air padding.
                if shuttle_df is not None:
                    seg = shuttle_df[
                        (shuttle_df["frame"] >= start_frame) &
                        (shuttle_df["frame"] <= end_frame)
                    ].copy().sort_values("frame")
                    if len(seg) > 10:
                        sx = seg["x"].values.astype(np.float64)
                        sy = seg["y"].values.astype(np.float64)
                        frame_gaps = np.diff(seg["frame"].values, prepend=seg["frame"].values[0])
                        spd = np.sqrt(np.diff(sx, prepend=sx[0])**2 + np.diff(sy, prepend=sy[0])**2) / np.maximum(frame_gaps, 1)
                        land_frames = settings.rally_dead_frames
                        streak = 0
                        for i, s in enumerate(spd):
                            if s < settings.rally_dead_speed_px:
                                streak += 1
                                if streak >= land_frames:
                                    land_frame = int(seg.iloc[i - land_frames + 1]["frame"])
                                    if end_frame - land_frame > land_frames * 2:
                                        end_frame = land_frame + 5
                                    break
                            else:
                                streak = 0

                clip_frames = list(range(start_frame, end_frame))
                original_n_frames = len(clip_frames)
                while len(clip_frames) < classifier.seq_len:
                    clip_frames.append(clip_frames[-1] if clip_frames else frame)
                clip_frames = clip_frames[:classifier.seq_len]

                clip = _build_clip(
                    clip_frames, shuttle_df, pose_df,
                    vid_w, vid_h, court_length, court_width,
                    seq_len=classifier.seq_len,
                    player_sides=player_sides, player_detections=player_list,
                    homography=homography,
                    original_len=original_n_frames,
                    player_ids=player_ids,
                )

            bst_clips_registry[int(frame)] = {"frames": clip_frames}
            all_clips.append(clip)
            clip_hit_pairs.append((frame, hit, clip_frames))

        # Phase 2: batch inference (GPU-efficient, all clips in one call)
        batch_size = config.extra.get("bst_batch", 32)
        debug_level = config.debug_level

        # Collect debug info for logits (needed for calibration) when requested
        collect_debug = debug_level >= 1 or settings.report_include_logits
        bst_debug_collector = [] if collect_debug else None
        all_results, probs_matrix = classifier.predict_from_clips(
            all_clips, batch_size=batch_size,
            debug_collector=bst_debug_collector,
            return_probs=True,
        )

        if bst_debug_collector is not None and len(bst_debug_collector) > 0:
            artifacts.set_parquet("debug_bst_outputs", pd.DataFrame(bst_debug_collector))

        # Phase 3: build shot records from results
        for i, ((frame, hit, clip_frames), (stroke_type, confidence, raw_class_id, alpha, aim_attention_p0, aim_attention_p1)) in enumerate(zip(clip_hit_pairs, all_results)):
            # Track if this specific shot fell back to rule-based prediction
            # raw_class_id == 0 catches three paths:
            #   1. Model never loaded (all clips fallback)
            #   2. Model predicted "unknown" with uniformly low logits
            #   3. Runtime exception during inference
            is_rule_based = raw_class_id == 0

            shot = {
                "frame": frame,
                "hit_confidence": float(hit["confidence"]),
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
                "shuttleset_class_id": raw_class_id,
                "aimplayer_alpha": alpha,
                "aim_attention_p0": aim_attention_p0,
                "aim_attention_p1": aim_attention_p1,
                "is_rule_based": is_rule_based,
                "is_bst_fallback": is_rule_based,
            }

            if is_rule_based and bst_debug_collector is not None and i < len(bst_debug_collector):
                ev = bst_debug_collector[i].get("rule_evidence", {})
                top3 = bst_debug_collector[i].get("rule_top3", [])
                if ev:
                    shot["rule_evidence"] = ev
                if top3:
                    shot["rule_top3"] = top3

            if bst_debug_collector is not None and i < len(bst_debug_collector):
                raw_conf = bst_debug_collector[i].get("bst_raw_confidence")
                if raw_conf is not None and not is_rule_based:
                    shot["bst_raw_confidence"] = raw_conf

            if settings.report_include_logits and bst_debug_collector is not None and i < len(bst_debug_collector):
                logits_str = bst_debug_collector[i].get("logits_all")
                if logits_str:
                    shot["logits"] = logits_str

            # Add clip debug info (level 1+)
            if debug_level >= 1 and i < len(all_clips):
                clip = all_clips[i]
                debug_clip = clip.get('_debug_clip', {})
                shot["clip_n_frames"] = debug_clip.get("n_frames", 0)
                shot["clip_n_missing_bbox"] = debug_clip.get("n_missing_bbox", 0)
                shot["clip_n_missing_pose"] = debug_clip.get("n_missing_pose", 0)
                shot["clip_frame_start"] = debug_clip.get("frame_start", 0)
                shot["clip_frame_end"] = debug_clip.get("frame_end", 0)

                # Shuttle coverage: how many frames have valid detections
                sv = (clip['shuttle'][:, 0] != 0) | (clip['shuttle'][:, 1] != 0)
                shot["clip_shuttle_valid"] = int(sv.sum())

                # JnB variance within the clip
                shot["clip_jnb_std"] = float(clip['JnB'].std())

                # Per-player foot position mean (pos is the only absolute-position feature)
                for p_idx, side in enumerate(["far", "near"]):
                    px = clip['pos'][:, p_idx, 0]
                    py = clip['pos'][:, p_idx, 1]
                    valid = (px != 0) | (py != 0)
                    if valid.any():
                        shot[f"clip_pos_{side}_x_mean"] = float(px[valid].mean())
                        shot[f"clip_pos_{side}_y_mean"] = float(py[valid].mean())
                    else:
                        shot[f"clip_pos_{side}_x_mean"] = 0.0
                        shot[f"clip_pos_{side}_y_mean"] = 0.0

                # Bbox diagonal per player from debug stats
                for pid_key in ['player_1', 'player_2']:
                    d_mean = debug_clip.get(f"bbox_diag_{pid_key}_mean")
                    d_std = debug_clip.get(f"bbox_diag_{pid_key}_std")
                    if d_mean is not None:
                        shot[f"bbox_diag_{pid_key}"] = d_mean
                        shot[f"bbox_diag_{pid_key}_std"] = d_std

            shots.append(shot)

        # Phase 3a: context fusion layer — nudge BST logits by physics/context
        fps = float(config.processing_fps or settings.fps)
        shuttle_raw = artifacts.get_parquet("shuttle_raw")
        if settings.fusion_enabled and probs_matrix is not None and len(shots) > 0:
            from app.pipeline.shared.context_fusion import ContextFusion
            fusion = ContextFusion.from_settings()
            probs_matrix = fusion.fuse(
                shots, probs_matrix,
                shuttle_raw, pose_df, court, fps, vid_w, vid_h,
            )

        # Phase 3b: hierarchical family classifier — reduce cross-family noise
        if settings.hierarchical_enabled and probs_matrix is not None and probs_matrix.shape[0] > 0:
            from app.pipeline.shared.hierarchical_classifier import hierarchical_refine
            probs_matrix = hierarchical_refine(probs_matrix, penalty=settings.hierarchical_penalty)

        # Phase 3b.5: confusion-pair correction — resolve within-family ambiguities
        if settings.confusion_pair_enabled and probs_matrix is not None and probs_matrix.shape[0] > 0:
            from app.pipeline.shared.confusion_pairs import resolve_confusion_pairs
            probs_matrix = resolve_confusion_pairs(
                probs_matrix, shots, shuttle_raw,
                pose_df, court, fps, vid_w, vid_h,
                boost=settings.confusion_pair_boost,
            )

        # Phase 3c: physics-consistency gate + BST × physics ensemble

        # Build 25-class name list matching predict_from_clips output ordering
        physics_classes = ["unknown"] + COACH_STROKE_CLASSES + COACH_STROKE_CLASSES

        shots = apply_physics_ensemble(
            shots, probs_matrix, physics_classes,
            shuttle_raw, pose_df, court, fps, vid_w, vid_h,
        )

        # Post-classification temporal smoothing: overwrite unknown strokes
        # with the majority type from nearby shots. Determinate predictions
        # (even low-confidence) are preserved to avoid rule-based bias from
        # dominating the neighborhood vote.
        if len(shots) > 2:
            for i in range(len(shots)):
                stype = shots[i]["stroke_type"]
                if stype != "unknown":
                    continue
                neighbors = []
                win = settings.stroke_smoothing_window
                for j in range(max(0, i - win), min(len(shots), i + win + 1)):
                    if j != i and shots[j]["stroke_type"] != "unknown":
                        neighbors.append(shots[j]["stroke_type"])
                if neighbors:
                    majority, count = Counter(neighbors).most_common(1)[0]
                    if majority != stype and count >= settings.stroke_smoothing_majority_count:
                        shots[i]["stroke_type"] = majority
                        shots[i]["stroke_confidence"] = 0.3

        # Temporal dedup: merge consecutive shots within 0.2s that share the
        # same stroke type. When the same hit is detected at multiple nearby
        # frames, BST produces similar classifications — keep only the one
        # with the highest confidence.
        if len(shots) > 1:
            max_gap = max(1, int(fps * settings.stroke_dedup_gap_seconds))
            shots = sorted(shots, key=lambda s: s["frame"])
            deduped = [shots[0]]
            for s in shots[1:]:
                prev = deduped[-1]
                gap = s["frame"] - prev["frame"]
                same_type = (
                    s["stroke_type"] == prev["stroke_type"]
                    or s["stroke_type"] == "unknown"
                    or prev["stroke_type"] == "unknown"
                )
                if gap <= max_gap and same_type:
                    # Merge: keep the higher-confidence shot
                    if s["stroke_confidence"] > prev["stroke_confidence"]:
                        deduped[-1] = s
                else:
                    deduped.append(s)
            shots = deduped

        for i, s in enumerate(shots):
            s["shot_id"] = i + 1
            s["start_ts"] = round(s["frame"] / fps, 3)

        # Compute ts_end for each shot: next shot's start_ts, or last shot gets +1s window
        for i, s in enumerate(shots):
            if i < len(shots) - 1:
                s["ts_end"] = shots[i + 1]["start_ts"]
            else:
                clip_end = s.get("clip_frame_end", s["frame"] + fps)
                s["ts_end"] = round(clip_end / fps, 3)

        shots_df = pd.DataFrame(shots)
        artifacts.set_parquet("shots", shots_df)
        artifacts.set("bst_clips", bst_clips_registry)

        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={
                "shot_count": len(shots),
                "stroke_distribution": self._compute_distribution(shots),
            }
        )

    @staticmethod
    def _compute_distribution(shots):
        if not shots:
            return {}
        dist = Counter(s["stroke_type"] for s in shots)
        total = len(shots)
        return {k: v / total for k, v in dist.items()}
