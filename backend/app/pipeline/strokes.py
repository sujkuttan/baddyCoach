import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, image_to_court
from app.pipeline.shared.bst_preproc import normalize_joints, create_bones, BONE_PAIRS


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
    debug_clip_stats = {
        "n_frames": min(len(frames), seq_len),
        "n_missing_bbox": 0,
        "n_missing_pose": 0,
    }

    for t, frame in enumerate(frames[:seq_len]):
        if shuttle_df is not None:
            s_row = shuttle_df[shuttle_df['frame'] == frame]
            if len(s_row) > 0:
                shuttle[t, 0] = float(s_row.iloc[0]['x']) / court_length
                shuttle[t, 1] = float(s_row.iloc[0]['y']) / court_width

    # Interpolate missing shuttle coordinates (0.0 = missing)
    for dim in range(2):
        shuttle_series = pd.Series(shuttle[:, dim])
        mask = shuttle_series == 0.0
        if mask.any() and (~mask).any():
            shuttle_series = shuttle_series.replace(0, np.nan)
            shuttle_series = shuttle_series.interpolate(method='linear').bfill().ffill()
            shuttle[:, dim] = shuttle_series.values

    for t, frame in enumerate(frames[:seq_len]):
        # Resolve active players for THIS frame
        frame_players = frame_player_map.get(frame, {})
        far_pid = frame_players.get('far', 'player_1')
        near_pid = frame_players.get('near', 'player_2')
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
                joints[t, p_idx] = normalize_joints(coords, det_bbox=det_bbox)

                feet_x = (coords[15, 0] + coords[16, 0]) / 2
                feet_y = max(coords[15, 1], coords[16, 1])
                if homography is not None:
                    court_x, court_y = image_to_court(homography, (feet_x, feet_y))
                    pos[t, p_idx, 0] = court_x / court_length if court_length > 0 else 0
                    pos[t, p_idx, 1] = court_y / court_width if court_width > 0 else 0
                else:
                    pos[t, p_idx, 0] = feet_x / vid_w
                    pos[t, p_idx, 1] = feet_y / vid_h
            else:
                debug_clip_stats["n_missing_pose"] += 1

    bones = np.zeros((seq_len, 2, len(BONE_PAIRS), 2), dtype=np.float32)
    for i in range(2):
        for t in range(seq_len):
            bones[t, i] = create_bones(joints[t, i])
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
        from app.config.settings import settings

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

        hit_frames = sorted(int(h["frame"]) for h in hits_df.to_dict('records'))

        # Get player side info for consistent ordering (p0=Far, p1=Near)
        players_data = artifacts.get("players") or {}
        player_sides = {}
        for p in players_data.get("players", []):
            player_sides[p["id"]] = p.get("side", "near")

        # Get player detection data for bbox normalization
        player_list = players_data.get("players", [])

        # Deduplicate hits by frame: keep the highest-confidence hit per frame.
        # Duplicate hits can arise from TrackNet producing multiple shuttle
        # detections at the same frame, or from hits.py gap-based dedup missing
        # same-frame duplicates. Without dedup, each duplicate produces a
        # separate shot at the same timestamp with independently classified
        # (potentially different) stroke types.
        hits_df = hits_df.loc[hits_df.groupby("frame")["confidence"].idxmax()].reset_index(drop=True)

        shots = []
        bst_clips_registry = {}
        previous_shots = []
        hit_frames_sorted = sorted(int(h["frame"]) for h in hits_df.to_dict('records'))

        # Phase 1: build all clips (fast, no model inference)
        all_clips = []
        clip_hit_pairs = []  # (frame, hit_dict, frames_list)
        for _, hit in hits_df.iterrows():
            frame = int(hit["frame"])

            hit_pos = hit_frames_sorted.index(frame)
            if hit_pos > 0:
                start_frame = hit_frames_sorted[hit_pos - 1]
            else:
                start_frame = max(0, frame - classifier.seq_len // 2)

            if hit_pos < len(hit_frames_sorted) - 1:
                end_frame = hit_frames_sorted[hit_pos + 1] + 2
            else:
                end_frame = frame + classifier.seq_len // 2 + 1

            clip_frames = list(range(start_frame, end_frame))

            if len(clip_frames) > classifier.seq_len:
                hit_offset = frame - start_frame
                half = classifier.seq_len // 2
                clip_start = max(0, hit_offset - half)
                clip_end = clip_start + classifier.seq_len
                if clip_end > len(clip_frames):
                    clip_end = len(clip_frames)
                    clip_start = max(0, clip_end - classifier.seq_len)
                clip_frames = clip_frames[clip_start:clip_end]

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
            )

            bst_clips_registry[int(frame)] = {"frames": clip_frames}
            all_clips.append(clip)
            clip_hit_pairs.append((frame, hit, clip_frames))

        # Phase 2: batch inference (GPU-efficient, all clips in one call)
        batch_size = config.extra.get("bst_batch", 32)
        debug_level = config.debug_level

        # Collect debug info if debug_level >= 1
        bst_debug_collector = [] if debug_level >= 1 else None
        all_results = classifier.predict_from_clips(
            all_clips, batch_size=batch_size,
            debug_collector=bst_debug_collector,
        )

        if bst_debug_collector is not None and len(bst_debug_collector) > 0:
            artifacts.set_parquet("debug_bst_outputs", pd.DataFrame(bst_debug_collector))

        # Phase 3: build shot records from results
        for i, ((frame, hit, clip_frames), (stroke_type, confidence, raw_class_id)) in enumerate(zip(clip_hit_pairs, all_results)):
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
                "is_rule_based": is_rule_based,
                "is_bst_fallback": is_rule_based,
            }

            # Add clip debug info (level 1+)
            if debug_level >= 1 and i < len(all_clips):
                debug_clip = all_clips[i].get('_debug_clip', {})
                shot["clip_n_frames"] = debug_clip.get("n_frames", 0)
                shot["clip_n_missing_bbox"] = debug_clip.get("n_missing_bbox", 0)
                shot["clip_n_missing_pose"] = debug_clip.get("n_missing_pose", 0)

            shots.append(shot)

            previous_shots.append({
                "stroke_type": stroke_type,
                "frame": frame,
                "stroke_confidence": confidence,
            })

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
                    from collections import Counter
                    majority, count = Counter(neighbors).most_common(1)[0]
                    if majority != stype and count >= settings.stroke_smoothing_majority_count:
                        shots[i]["stroke_type"] = majority
                        shots[i]["stroke_confidence"] = 0.3

        fps = float(config.processing_fps or settings.fps)

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
        from collections import Counter
        dist = Counter(s["stroke_type"] for s in shots)
        total = len(shots)
        return {k: v / total for k, v in dist.items()}
