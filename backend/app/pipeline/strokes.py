import numpy as np
import pandas as pd
from collections import Counter

from app.config.settings import settings
from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, image_to_court, clamp_to_unit
from app.pipeline.shared.bst_preproc import (
    normalize_joints, normalize_joints_court, normalize_joints_hip_centered,
    create_bones, BONE_PAIRS,
)
from app.pipeline.shared.bst_input_quality import evaluate_aim_alpha_quality, evaluate_bst_clip_quality
from app.pipeline.shared.physics import apply_physics_ensemble, summarize_physics_sources
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


def _can_temporally_deduplicate(previous: dict, current: dict, gap: int, max_gap: int) -> bool:
    """Keep quality-abstained shots as auditable records, even when nearby."""
    if gap > max_gap:
        return False
    if not previous.get("bst_input_eligible", True) or not current.get("bst_input_eligible", True):
        return False
    return (
        current["stroke_type"] == previous["stroke_type"]
        or current["stroke_type"] == "unknown"
        or previous["stroke_type"] == "unknown"
    )


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
    shuttle_raw: pd.DataFrame | None = None,
    hit_frame: int | None = None,
    alpha_probe_offset: int = 0,
    racket_detections: list | None = None,
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
        """Linearly interpolate only short bbox gaps and report source distance."""
        existing = sorted(lookup.keys())
        if not existing:
            return {}, {}
        result = {}
        gaps = {}
        for f in target_frames:
            if f in lookup:
                result[f] = lookup[f]
                gaps[f] = 0
            else:
                before = [ef for ef in existing if ef <= f]
                after = [ef for ef in existing if ef >= f]
                if before and after:
                    bf, af = before[-1], after[0]
                    if bf == af:
                        result[f] = lookup[bf]
                        gaps[f] = 0
                    elif af - bf <= settings.bst_max_bbox_interp_gap:
                        ratio = (f - bf) / (af - bf)
                        result[f] = tuple(
                            lookup[bf][i] + ratio * (lookup[af][i] - lookup[bf][i])
                            for i in range(4)
                        )
                        gaps[f] = max(f - bf, af - f)
                elif before and f - before[-1] <= settings.bst_max_bbox_interp_gap:
                    result[f] = lookup[before[-1]]
                    gaps[f] = f - before[-1]
                elif after and after[0] - f <= settings.bst_max_bbox_interp_gap:
                    result[f] = lookup[after[0]]
                    gaps[f] = after[0] - f
                else:
                    gaps[f] = settings.bst_max_bbox_interp_gap + 1
                if f not in gaps:
                    gaps[f] = settings.bst_max_bbox_interp_gap + 1
        return result, gaps

    # Build interpolated bbox lookups for each player
    clip_frames_set = set(frames[:seq_len])
    interpolated_bboxes = {}
    bbox_gaps = {}
    for pid in list(det_bbox_lookup.keys()):
        interpolated_bboxes[pid], bbox_gaps[pid] = _interpolate_bboxes(
            det_bbox_lookup[pid], clip_frames_set
        )

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

    provenance = {
        "video_len": min(n_frames_orig, seq_len),
        "contact_frame_index": 0,
        "shuttle_observed": [],
        "shuttle_repaired": [],
        "shuttle_interpolated": [],
        "shuttle_court_rejected": [],
        "pose_present_far": [],
        "pose_present_near": [],
        "pose_keypoint_confidence_far": [],
        "pose_keypoint_confidence_near": [],
        "bbox_gap_far": [],
        "bbox_gap_near": [],
        "resolved_far_pid": [],
        "resolved_near_pid": [],
        "wrist_shuttle_distance_far": [],
        "wrist_shuttle_distance_near": [],
    }
    if hit_frame is not None and frames:
        provenance["contact_frame_index"] = min(range(len(frames)), key=lambda i: abs(frames[i] - hit_frame))
    shuttle_lookup = {}
    if shuttle_df is not None:
        shuttle_lookup = {int(row["frame"]): row for _, row in shuttle_df.iterrows()}
    raw_shuttle_lookup = {}
    if shuttle_raw is not None:
        raw_shuttle_lookup = {int(row["frame"]): row for _, row in shuttle_raw.iterrows()}

    for t, frame in enumerate(frames[:seq_len]):
        clean_row = shuttle_lookup.get(frame)
        raw_row = raw_shuttle_lookup.get(frame)
        raw_repaired = bool(raw_row.get("was_repaired", False)) if raw_row is not None else False
        raw_observed = bool(
            raw_row is not None
            and not raw_repaired
            and float(raw_row.get("confidence", 0.0)) >= settings.shuttle_min_conf
            and np.isfinite(float(raw_row.get("x", np.nan)))
            and np.isfinite(float(raw_row.get("y", np.nan)))
        )
        interpolated = bool(clean_row.get("was_interpolated", False)) if clean_row is not None else False
        court_rejected = bool(clean_row.get("court_rejected", False)) if clean_row is not None else False
        provenance["shuttle_observed"].append(raw_observed)
        provenance["shuttle_repaired"].append(raw_repaired)
        provenance["shuttle_interpolated"].append(interpolated)
        provenance["shuttle_court_rejected"].append(court_rejected)

        # Provenance always records court_rejected. Resolution/pixel BST
        # inputs keep image-space coords even when court projection is OOB
        # (homography mismatch must not blank the only channel BST was
        # trained to see in resolution mode). Court-normalized mode still
        # blanks rejected points.
        use_for_tensor = clean_row is not None and (
            not court_rejected or settings.bst_shuttle_norm != "court"
        )
        if use_for_tensor:
            # Feed a coord to BST when it is raw-observed, OR InpaintNet-repaired
            # (real model estimate, enabled by default), OR linear-interpolated
            # (fabric, disabled by default). When require_raw_observation is off,
            # any clean coord is fed regardless of provenance.
            feed = (
                (not settings.bst_shuttle_require_raw_observation)
                or raw_observed
                or (raw_repaired and settings.bst_shuttle_use_repaired)
                or (interpolated and settings.bst_shuttle_use_interpolated)
            )
            if not feed:
                continue  # leave zeros; provenance already recorded
            s_conf = float(clean_row.get('confidence', 1.0))
            if s_conf >= settings.shuttle_min_conf:
                sx = float(clean_row['x'])
                sy = float(clean_row['y'])
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

    # ── Racket head points + presence (Scope A racket channel) ─────
    racket_head = np.zeros((seq_len, 2, 2), dtype=np.float32)
    racket_present = np.zeros((seq_len, 2), dtype=bool)
    racket_lookup = {}
    if racket_detections is not None:
        for rd in racket_detections:
            racket_lookup.setdefault(int(rd["frame"]), {})[rd["player_side"]] = rd
    for t, frame in enumerate(frames[:seq_len]):
        rframe = racket_lookup.get(int(frame))
        if not rframe:
            continue
        for p_idx, side in ((0, "far"), (1, "near")):
            rd = rframe.get(side)
            if rd is None:
                continue
            hp = rd.get("head_point")
            if hp is not None:
                racket_head[t, p_idx] = [float(hp[0]), float(hp[1])]
                racket_present[t, p_idx] = True

    for t, frame in enumerate(frames[:seq_len]):
        # Resolve active players for THIS frame.
        # When one side is missing (e.g. far player not detected in this frame),
        # infer the missing player as whoever is NOT assigned to near — avoids
        # silently putting player_1 in both slots.
        frame_players = frame_player_map.get(frame, {})
        near_pid = frame_players.get('near')
        far_pid = frame_players.get('far')
        if far_pid is None:
            if player_ids and near_pid in player_ids and len(player_ids) > 1:
                far_pid = [p for p in player_ids if p != near_pid][0]
            else:
                far_pid = 'player_2'
        if near_pid is None:
            if player_ids and far_pid in player_ids and len(player_ids) > 1:
                near_pid = [p for p in player_ids if p != far_pid][0]
            else:
                near_pid = 'player_1'
        player_order = [far_pid, near_pid]
        provenance["resolved_far_pid"].append(far_pid)
        provenance["resolved_near_pid"].append(near_pid)

        for p_idx, pid in enumerate(player_order):
            kps = _get_keypoints_for_frame(pose_df, frame, pid)
            side = "far" if p_idx == 0 else "near"
            bbox_gap = bbox_gaps.get(pid, {}).get(
                frame, settings.bst_max_bbox_interp_gap + 1
            )
            provenance[f"bbox_gap_{side}"].append(bbox_gap)
            wrist_distance = np.nan
            if kps is not None:
                coords = kps[:, :2]
                conf = kps[:, 2] if kps.shape[1] >= 3 else np.ones(len(kps))
                valid_keypoints = (
                    np.isfinite(coords).all(axis=1)
                    & (conf >= settings.bst_min_keypoint_confidence)
                    & ~np.all(coords == 0.0, axis=1)
                )
                n_valid = int(valid_keypoints.sum())
                dense_enough = n_valid >= int(17 * settings.bst_min_valid_keypoints_fraction)
                if not dense_enough:
                    provenance[f"pose_present_{side}"].append(False)
                    provenance[f"pose_keypoint_confidence_{side}"].append(0.0)
                    provenance[f"wrist_shuttle_distance_{side}"].append(np.nan)
                    joints[t, p_idx] = 0.0
                    continue
                masked_coords = coords.copy()
                masked_coords[~valid_keypoints] = 0.0
                provenance[f"pose_present_{side}"].append(bool(valid_keypoints.any()))
                provenance[f"pose_keypoint_confidence_{side}"].append(
                    float(np.median(conf[valid_keypoints])) if valid_keypoints.any() else 0.0
                )
                # Coords are (17, 2); pass confidence for keypoint-bbox masking.
                # Keypoint-derived bbox structurally eliminates pose/bbox mismatch.
                if interpolated_bboxes.get(pid, {}).get(frame) is None:
                    debug_clip_stats["n_missing_bbox"] += 1
                if settings.bst_joint_norm == "court" and homography is not None:
                    joints[t, p_idx] = normalize_joints_court(masked_coords, homography)
                elif settings.bst_joint_norm == "hip_centered":
                    joints[t, p_idx] = normalize_joints_hip_centered(
                        masked_coords, vid_w=vid_w, vid_h=vid_h, conf=conf,
                    )
                else:
                    # Use keypoint-derived bbox (det_bbox=None) so the bounding
                    # box is always the extent of the keypoints themselves.
                    # This guarantees normalized joints stay in ~[-0.5, 0.5].
                    # The YOLO det_bbox path (interpolated_bboxes) produced
                    # out-of-range joints when the detection bbox did not
                    # contain the keypoints (player/bbox association drift).
                    joints[t, p_idx] = normalize_joints(
                        masked_coords, det_bbox=None, bbox_margin=settings.bst_bbox_margin,
                        conf=conf, min_confidence=settings.bst_min_keypoint_confidence,
                    )

                feet_x = (masked_coords[15, 0] + masked_coords[16, 0]) / 2
                feet_y = max(masked_coords[15, 1], masked_coords[16, 1])
                if homography is not None:
                    court_x, court_y = image_to_court(homography, (feet_x, feet_y))
                    court_x, court_y = clamp_to_unit(court_x / court_length if court_length > 0 else 0,
                                                     court_y / court_width  if court_width  > 0 else 0)
                    pos[t, p_idx, 0] = court_x
                    pos[t, p_idx, 1] = court_y
                else:
                    pos[t, p_idx, 0] = max(0.0, min(1.0, feet_x / vid_w))
                    pos[t, p_idx, 1] = max(0.0, min(1.0, feet_y / vid_h))

                clean_row = shuttle_lookup.get(frame)
                bbox = interpolated_bboxes.get(pid, {}).get(frame)
                cr = bool(clean_row.get("court_rejected", False)) if clean_row is not None else False
                if (
                    clean_row is not None
                    and (not cr or settings.bst_shuttle_norm != "court")
                    and float(clean_row.get("confidence", 0.0)) >= settings.shuttle_min_conf
                    and kps.shape[1] >= 3
                    and float(kps[10, 2]) >= settings.bst_min_keypoint_confidence
                    and bbox is not None
                ):
                    bbox_h = max(float(bbox[3] - bbox[1]), 1.0)
                    wrist_xy = kps[10, :2]
                    wrist_distance = float(np.linalg.norm(wrist_xy - np.array([float(clean_row["x"]), float(clean_row["y"])])) / bbox_h)
            else:
                debug_clip_stats["n_missing_pose"] += 1
                provenance[f"pose_present_{side}"].append(False)
                provenance[f"pose_keypoint_confidence_{side}"].append(0.0)
            provenance[f"wrist_shuttle_distance_{side}"].append(wrist_distance)

    video_len = min(len(frames), seq_len)
    finite = joints[:video_len][np.isfinite(joints[:video_len])]
    provenance["joint_abs_mean"] = float(np.mean(np.abs(finite))) if finite.size else 0.0
    # Fraction of frames where every joint coordinate is all-zero OR non-finite.
    # This catches collapsed/NaN skeletons that still pass pose_present (a
    # collapsed skeleton has valid keypoints at one point -> pose_present=True,
    # high confidence, joint_abs_mean~0) and would otherwise be fed to BST.
    joints_view = joints[:video_len]
    if video_len:
        real_joints = np.isfinite(joints_view) & (joints_view != 0.0)
        frame_has_real = real_joints.any(axis=(1, 2, 3))
        degenerate_fraction = float(1.0 - frame_has_real.mean())
    else:
        degenerate_fraction = 0.0
    provenance["joint_degenerate_fraction"] = degenerate_fraction

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
        'racket_head': racket_head,
        'racket_present': racket_present,
        'video_len': min(n_frames_orig, seq_len),
        'vid_w': vid_w,
        'vid_h': vid_h,
        'court_length': court_length,
        'court_width': court_width,
        '_debug_clip': debug_clip_stats,
        '_bst_provenance': provenance,
        '_alpha_probe_offset': alpha_probe_offset,
    }


def _prepare_bst_clip_for_hit(
    frame: int,
    hit_pos: int,
    hit_frames_sorted: list[int],
    classifier,
    shuttle_df: pd.DataFrame | None,
    shuttle_raw: pd.DataFrame | None,
    pose_df: pd.DataFrame | None,
    vid_w: float,
    vid_h: float,
    court_length: float,
    court_width: float,
    player_sides: dict | None,
    player_list: list,
    homography: np.ndarray | None,
    player_ids: list[str],
    alpha_probe_offset: int = 0,
    racket_detections: list | None = None,
) -> tuple[dict, list[int]]:
    anchor_frame = max(0, int(frame + alpha_probe_offset))
    use_midpoint = settings.bst_clip_boundary == "midpoint"

    if use_midpoint:
        prev_hit = hit_frames_sorted[hit_pos - 1] if hit_pos > 0 else max(0, anchor_frame - 20)
        next_hit = hit_frames_sorted[hit_pos + 1] if hit_pos + 1 < len(hit_frames_sorted) else anchor_frame + 20
        start_frame = (prev_hit + anchor_frame) // 2
        end_frame = (anchor_frame + next_hit) // 2

        if end_frame - start_frame < settings.bst_min_clip_frames:
            half_floor = settings.bst_min_clip_frames // 2
            start_frame = max(0, anchor_frame - half_floor)
            end_frame = anchor_frame + (settings.bst_min_clip_frames - half_floor)

        clip_frames = list(range(start_frame, end_frame))
        original_n_frames = len(clip_frames)
        clip = _build_clip(
            clip_frames, shuttle_df, pose_df,
            vid_w, vid_h, court_length, court_width,
            seq_len=original_n_frames,
            player_sides=player_sides, player_detections=player_list,
            homography=homography,
            original_len=original_n_frames,
            player_ids=player_ids,
            shuttle_raw=shuttle_raw,
            hit_frame=anchor_frame,
            alpha_probe_offset=alpha_probe_offset,
            racket_detections=racket_detections,
        )
        if original_n_frames != classifier.seq_len:
            clip['JnB'] = _temporal_resample(clip['JnB'], classifier.seq_len)
            clip['shuttle'] = _temporal_resample(clip['shuttle'], classifier.seq_len, zero_is_missing=True)
            clip['pos'] = _temporal_resample(clip['pos'], classifier.seq_len)
            # After resampling, the real motion is spread across ALL seq_len
            # frames — so the whole sequence is valid. Setting video_len to the
            # pre-resample length would mask out (1 - orig/seq_len) of the clip,
            # dropping the (now-centered) hit outside the attended window.
            clip['video_len'] = classifier.seq_len
        else:
            clip['video_len'] = original_n_frames
        return clip, clip_frames

    start_frame = anchor_frame
    if hit_pos < len(hit_frames_sorted) - 1:
        end_frame = min(
            anchor_frame + classifier.seq_len,
            max(anchor_frame + settings.bst_min_clip_frames, hit_frames_sorted[hit_pos + 1]),
        )
    else:
        end_frame = anchor_frame + classifier.seq_len

    if shuttle_df is not None:
        seg = shuttle_df[
            (shuttle_df["frame"] >= start_frame) &
            (shuttle_df["frame"] <= end_frame)
        ].copy().sort_values("frame")
        if len(seg) > 10:
            sx = seg["x"].values.astype(np.float64)
            sy = seg["y"].values.astype(np.float64)
            frame_gaps = np.diff(seg["frame"].values, prepend=seg["frame"].values[0])
            spd = np.sqrt(np.diff(sx, prepend=sx[0]) ** 2 + np.diff(sy, prepend=sy[0]) ** 2) / np.maximum(frame_gaps, 1)
            land_frames = settings.rally_dead_frames
            streak = 0
            for i, speed in enumerate(spd):
                if speed < settings.rally_dead_speed_px:
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
        clip_frames.append(clip_frames[-1] if clip_frames else anchor_frame)
    clip_frames = clip_frames[:classifier.seq_len]

    clip = _build_clip(
        clip_frames, shuttle_df, pose_df,
        vid_w, vid_h, court_length, court_width,
        seq_len=classifier.seq_len,
        player_sides=player_sides, player_detections=player_list,
        homography=homography,
        original_len=original_n_frames,
        player_ids=player_ids,
        shuttle_raw=shuttle_raw,
        hit_frame=anchor_frame,
        alpha_probe_offset=alpha_probe_offset,
        racket_detections=racket_detections,
    )
    return clip, clip_frames


class StrokeClassificationStage:
    name = "stroke_classification"
    input_keys = ["hits", "shuttle", "pose", "court"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        hits_df = artifacts.get_parquet("hits")
        if hits_df is None or len(hits_df) == 0:
            return StageResult.success(metadata={"shot_count": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        shuttle_raw = artifacts.get_parquet("shuttle_raw")
        pose_df = artifacts.get_parquet("pose")
        racket_detections = artifacts.get("racket_detections")
        court = artifacts.get("court") or {}

        from app.pipeline.shared.models import get_bst

        classifier = get_bst()
        if classifier is None:
            return StageResult.from_error("BST model not available")

        court_length = court.get("court_length", COURT_LENGTH)
        court_width = court.get("court_width", COURT_WIDTH)
        # Guard against invalid court geometry: a degenerate (e.g. rectangular
        # proportional-fallback) homography produces silently-wrong court-space
        # features. When court is invalid, null the homography so all downstream
        # `if homography is not None` paths fall back to pixel-space normalization.
        court_valid = bool(court.get("valid", False)) if isinstance(court, dict) else False
        if not court_valid:
            logger.warning(
                "Court geometry invalid; BST clip court-space features fall back to pixel space"
            )
        homography = (
            np.array(court["homography"])
            if (court_valid and court.get("homography") is not None)
            else None
        )

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
            clip, clip_frames = _prepare_bst_clip_for_hit(
                frame=frame,
                hit_pos=hit_pos,
                hit_frames_sorted=hit_frames_sorted,
                classifier=classifier,
                shuttle_df=shuttle_df,
                shuttle_raw=shuttle_raw,
                pose_df=pose_df,
                vid_w=vid_w,
                vid_h=vid_h,
                court_length=court_length,
                court_width=court_width,
                player_sides=player_sides,
                player_list=player_list,
                homography=homography,
                player_ids=player_ids,
                alpha_probe_offset=0,
                racket_detections=racket_detections,
            )

            bst_clips_registry[int(frame)] = {"frames": clip_frames}
            all_clips.append(clip)
            clip_hit_pairs.append((frame, hit, clip_frames))

        # Phase 2: evaluate quality, then batch only evidence-supported clips.
        batch_size = config.extra.get("bst_batch", 32)
        debug_level = config.debug_level
        collect_debug = debug_level >= 1 or settings.report_include_logits
        quality_records = [
            evaluate_bst_clip_quality(clip["_bst_provenance"])
            for clip in all_clips
        ]
        alpha_quality_records = [
            evaluate_aim_alpha_quality(clip["_bst_provenance"])
            for clip in all_clips
        ]
        eligible_indices = [
            i for i, quality in enumerate(quality_records)
            if not settings.bst_input_quality_enabled or quality["eligible"]
        ]
        eligible_clips = [all_clips[i] for i in eligible_indices]
        abstained_result = ("unknown", 0.0, 0, 0.5, 0.0, 0.0)
        all_results = [abstained_result for _ in all_clips]
        probs_matrix = np.zeros((len(all_clips), 25), dtype=np.float32)
        bst_debug_by_clip = {}

        if eligible_clips:
            eligible_debug = [] if collect_debug else None
            eligible_results, eligible_probs = classifier.predict_from_clips(
                eligible_clips, batch_size=batch_size,
                debug_collector=eligible_debug,
                return_probs=True,
            )
            for result_idx, clip_idx in enumerate(eligible_indices):
                all_results[clip_idx] = eligible_results[result_idx]
                if eligible_probs is not None:
                    probs_matrix[clip_idx] = eligible_probs[result_idx]
                if eligible_debug is not None and result_idx < len(eligible_debug):
                    entry = dict(eligible_debug[result_idx])
                    entry["clip_index"] = clip_idx
                    bst_debug_by_clip[clip_idx] = entry

        if bst_debug_by_clip:
            artifacts.set_parquet("debug_bst_outputs", pd.DataFrame(bst_debug_by_clip.values()))
        if debug_level >= 1:
            artifacts.set_parquet("debug_bst_input_quality", pd.DataFrame([
                {**quality_records[i], **{
                    "aim_alpha_reliable": alpha_quality_records[i]["reliable"],
                    "aim_alpha_quality_score": alpha_quality_records[i]["score"],
                    "aim_alpha_quality_reasons": alpha_quality_records[i]["reasons"],
                    "aim_alpha_contact_window_valid": alpha_quality_records[i]["contact_window_valid"],
                    "aim_alpha_pose_balance_score": alpha_quality_records[i]["pose_balance_score"],
                    "aim_alpha_identity_stable": alpha_quality_records[i]["identity_stable"],
                    "aim_alpha_contact_separation": alpha_quality_records[i]["contact_separation"],
                }}
                for i in range(len(quality_records))
            ]))

        logger.info(
            "BST input quality routing",
            total_clips=len(all_clips),
            bst_eligible=len(eligible_indices),
            quality_abstained=len(all_clips) - len(eligible_indices),
            abstention_reasons=str(Counter(
                reason for quality in quality_records for reason in quality["reasons"]
            )),
        )

        probe_results_by_clip = {}
        if settings.aim_alpha_enabled and eligible_indices:
            probe_requests = []
            for clip_idx in eligible_indices:
                if not alpha_quality_records[clip_idx]["reliable"]:
                    continue
                frame, _, _ = clip_hit_pairs[clip_idx]
                hit_pos = hit_frames_sorted.index(int(frame))
                for offset in settings.aim_alpha_probe_offsets:
                    if offset == 0 or abs(offset) > settings.aim_alpha_max_anchor_shift:
                        continue
                    probe_clip, _ = _prepare_bst_clip_for_hit(
                        frame=int(frame),
                        hit_pos=hit_pos,
                        hit_frames_sorted=hit_frames_sorted,
                        classifier=classifier,
                        shuttle_df=shuttle_df,
                        shuttle_raw=shuttle_raw,
                        pose_df=pose_df,
                        vid_w=vid_w,
                        vid_h=vid_h,
                        court_length=court_length,
                        court_width=court_width,
                        player_sides=player_sides,
                        player_list=player_list,
                        homography=homography,
                        player_ids=player_ids,
                        alpha_probe_offset=offset,
                        racket_detections=racket_detections,
                    )
                    probe_requests.append((clip_idx, offset, probe_clip))
            if probe_requests:
                probe_outputs = classifier.predict_from_clips([item[2] for item in probe_requests], batch_size=batch_size)
                if isinstance(probe_outputs, tuple):
                    probe_outputs = probe_outputs[0]
                for (clip_idx, offset, _), result in zip(probe_requests, probe_outputs):
                    probe_results_by_clip.setdefault(clip_idx, {})[offset] = float(result[3])

        # ── Phase 2b: MMAction2 ensemble (optional) ──────────────────────
        if settings.mmaction2_enabled and eligible_indices:
            from app.pipeline.shared.models import get_mmaction2
            mma_clf = get_mmaction2()
            if mma_clf is not None:
                logger.info("Running MMAction2 ensemble", mode=settings.mmaction2_mode, weight=settings.mmaction2_ensemble_weight)
                mma_results, mma_probs = mma_clf.predict_from_clips(
                    eligible_clips, batch_size=batch_size, return_probs=True,
                )
                if mma_probs is not None and mma_probs.shape == (len(eligible_indices), 25):
                    w = settings.mmaction2_ensemble_weight
                    eligible_probs = (1.0 - w) * probs_matrix[eligible_indices] + w * mma_probs
                    probs_matrix[eligible_indices] = eligible_probs
                    # Re-derive results from ensembled probs
                    for result_idx, probs in enumerate(eligible_probs):
                        pred_idx = int(np.argmax(probs))
                        confidence = float(probs[pred_idx])
                        stroke_type = "unknown"
                        if pred_idx > 0:
                            from app.models.bst import map_to_coach_class
                            stroke_type = map_to_coach_class(pred_idx)
                        # Preserve alpha/sims from BST result if available
                        clip_idx = eligible_indices[result_idx]
                        orig = all_results[clip_idx]
                        alpha = orig[3] if orig and len(orig) > 3 else 0.5
                        p0 = orig[4] if orig and len(orig) > 4 else 0.0
                        p1 = orig[5] if orig and len(orig) > 5 else 0.0
                        all_results[clip_idx] = (stroke_type, confidence, pred_idx, alpha, p0, p1)
                    logger.info("MMAction2 ensemble applied", shots=len(eligible_indices))
                else:
                    logger.warning(
                        "MMAction2 ensemble skipped: shape mismatch",
                        bst_shape=str((len(eligible_indices), 25)),
                        mma_shape=str(mma_probs.shape if mma_probs is not None else None),
                    )

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
            quality = quality_records[i]
            alpha_quality = dict(alpha_quality_records[i])
            probe_offsets = probe_results_by_clip.get(i, {})
            probe_alphas = [alpha] + [probe_offsets[offset] for offset in sorted(probe_offsets)]
            stability_span = float(max(probe_alphas) - min(probe_alphas)) if probe_alphas else 0.0
            probe_dirs = {
                "far" if probe_alpha > 0.5 else "near" if probe_alpha < 0.5 else "uncertain"
                for probe_alpha in probe_alphas
            }
            if "far" in probe_dirs and "near" in probe_dirs:
                alpha_quality["reliable"] = False
                alpha_quality["reasons"] = list(alpha_quality["reasons"]) + ["probe_direction_flip"]
            if stability_span > settings.aim_alpha_max_stability_span:
                alpha_quality["reliable"] = False
                alpha_quality["reasons"] = list(alpha_quality["reasons"]) + ["probe_span_too_wide"]
            alpha_route = "alpha_ok" if alpha_quality["reliable"] else (
                "alpha_abstain_instability" if probe_offsets else "alpha_abstain_quality"
            )
            shot.update({
                "bst_input_eligible": bool(quality["eligible"]),
                "bst_input_quality_score": float(quality["score"]),
                "bst_input_quality_reasons": quality["reasons"],
                "bst_input_route": "bst" if (i in eligible_indices and not quality["soft"])
                                 else ("low_quality_bst" if (i in eligible_indices and quality["soft"])
                                       else "quality_abstain"),
                "bst_input_observed_shuttle_frames": quality["observed_shuttle_frames"],
                "bst_input_repaired_shuttle_frames": quality["repaired_shuttle_frames"],
                "bst_input_interpolated_shuttle_frames": quality["interpolated_shuttle_frames"],
                "bst_input_court_rejected_shuttle_frames": quality["court_rejected_shuttle_frames"],
                "bst_input_max_shuttle_gap_frames": quality["max_shuttle_gap_frames"],
                "bst_input_far_pose_coverage": quality["far_pose_coverage"],
                "bst_input_near_pose_coverage": quality["near_pose_coverage"],
                "bst_input_far_pose_median_confidence": quality["far_pose_median_confidence"],
                "bst_input_near_pose_median_confidence": quality["near_pose_median_confidence"],
                "bst_input_max_bbox_gap_frames": quality["max_bbox_gap_frames"],
                "aim_alpha_reliable": bool(alpha_quality["reliable"]),
                "aim_alpha_quality_score": float(alpha_quality["score"]),
                "aim_alpha_quality_reasons": list(dict.fromkeys(alpha_quality["reasons"])),
                "aim_alpha_route": alpha_route,
                "aim_alpha_contact_window_valid": bool(alpha_quality["contact_window_valid"]),
                "aim_alpha_pose_balance_score": float(alpha_quality["pose_balance_score"]),
                "aim_alpha_identity_stable": bool(alpha_quality["identity_stable"]),
                "aim_alpha_contact_separation": float(alpha_quality["contact_separation"]),
                "aim_alpha_stability_span": stability_span,
            })
            debug_entry = bst_debug_by_clip.get(i, {})

            if is_rule_based and debug_entry:
                ev = debug_entry.get("rule_evidence", {})
                top3 = debug_entry.get("rule_top3", [])
                if ev:
                    shot["rule_evidence"] = ev
                if top3:
                    shot["rule_top3"] = top3

            if debug_entry:
                raw_conf = debug_entry.get("bst_raw_confidence")
                if raw_conf is not None and not is_rule_based:
                    shot["bst_raw_confidence"] = raw_conf

            if settings.report_include_logits and debug_entry:
                logits_str = debug_entry.get("logits_all")
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
        if settings.fusion_enabled and eligible_indices:
            from app.pipeline.shared.context_fusion import ContextFusion
            fusion = ContextFusion.from_settings()
            probs_matrix[eligible_indices] = fusion.fuse(
                [shots[i] for i in eligible_indices], probs_matrix[eligible_indices],
                shuttle_df, shuttle_raw, pose_df, court, fps, vid_w, vid_h,
            )

        # Phase 3b: hierarchical family classifier — reduce cross-family noise
        if settings.hierarchical_enabled and eligible_indices:
            from app.pipeline.shared.hierarchical_classifier import hierarchical_refine
            probs_matrix[eligible_indices] = hierarchical_refine(
                probs_matrix[eligible_indices], penalty=settings.hierarchical_penalty
            )

        # Phase 3b.5: confusion-pair correction — resolve within-family ambiguities
        if settings.confusion_pair_enabled and eligible_indices:
            from app.pipeline.shared.confusion_pairs import resolve_confusion_pairs
            probs_matrix[eligible_indices] = resolve_confusion_pairs(
                probs_matrix[eligible_indices], [shots[i] for i in eligible_indices], shuttle_df, shuttle_raw,
                pose_df, court, fps, vid_w, vid_h,
                boost=settings.confusion_pair_boost,
            )

        # Phase 3c: physics-consistency gate + BST × physics ensemble

        # Build 25-class name list matching predict_from_clips output ordering
        physics_classes = ["unknown"] + COACH_STROKE_CLASSES + COACH_STROKE_CLASSES

        shots = apply_physics_ensemble(
            shots, probs_matrix, physics_classes,
            shuttle_df, shuttle_raw, pose_df, court, fps, vid_w, vid_h,
        )

        for shot in shots:
            if not shot["bst_input_eligible"]:
                if shot.get("stroke_source") in {"bst", "bst_no_physics"}:
                    shot["stroke_source"] = "quality_abstain"
                if shot["stroke_type"] != "unknown":
                    shot["bst_input_route"] = "downstream_override"
                    shot["bst_input_override_source"] = "physics"

        # Spec 3: clips admitted on the soft quality tier (between
        # bst_quality_score_min and bst_quality_score_soft) keep their BST
        # prediction but are down-weighted so they don't inflate confidence.
        if settings.bst_low_quality_discount < 1.0:
            for shot in shots:
                if shot.get("bst_input_route") == "low_quality_bst":
                    shot["stroke_source"] = "low_quality_bst"
                    disc = float(settings.bst_low_quality_discount)
                    if "stroke_confidence" in shot:
                        shot["stroke_confidence"] = float(shot["stroke_confidence"]) * disc
                    if "calibrated_confidence" in shot:
                        shot["calibrated_confidence"] = float(shot["calibrated_confidence"]) * disc

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

        for shot in shots:
            if shot["bst_input_route"] == "quality_abstain" and shot["stroke_type"] != "unknown":
                shot["bst_input_route"] = "downstream_override"
                shot["bst_input_override_source"] = "temporal_smoothing"
                shot["stroke_source"] = "temporal_smoothing"

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
                if _can_temporally_deduplicate(prev, s, gap, max_gap):
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
        physics_summary = summarize_physics_sources(shots)
        artifacts.set("physics_summary", physics_summary)

        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={
                "shot_count": len(shots),
                "stroke_distribution": self._compute_distribution(shots),
                "physics_summary": physics_summary,
            }
        )

    @staticmethod
    def _compute_distribution(shots):
        if not shots:
            return {}
        dist = Counter(s["stroke_type"] for s in shots)
        total = len(shots)
        return {k: v / total for k, v in dist.items()}
