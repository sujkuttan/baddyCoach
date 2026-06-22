import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH

BONE_PAIRS = [
    (0,1),(0,2),(1,2),(1,3),(2,4),
    (3,5),(4,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


def _normalize_joints_bstdiag(coords: np.ndarray, det_bbox: tuple | None = None) -> np.ndarray:
    """Normalize joints using bbox diagonal with center_align.

    Matches the official BST ``normalize_joints`` preprocessing:
    - Origin = top-left of the player bounding box
    - Scale = diagonal distance of the bounding box
    - center_align=True shifts origin to bbox center

    Args:
        coords: (17, 2) keypoints in pixel coords
        det_bbox: optional (x1, y1, x2, y2) detection bbox for stable normalization.
                  If None, falls back to keypoint bbox (less stable).

    Returns:
        (17, 2) normalized joints, range roughly [-0.X, 0.X]
    """
    if det_bbox is not None:
        bbox_min = np.array([det_bbox[0], det_bbox[1]], dtype=np.float64)
        bbox_max = np.array([det_bbox[2], det_bbox[3]], dtype=np.float64)
    else:
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)

    diag = np.linalg.norm(bbox_max - bbox_min)
    if diag < 1e-6:
        diag = 1.0

    normalized = (coords - bbox_min) / diag
    center = (bbox_min + bbox_max) / 2.0
    normalized -= (center - bbox_min) / diag
    return normalized.astype(np.float32)


def _create_bones(joints: np.ndarray) -> np.ndarray:
    """Create bone vectors from joint positions. joints: (17, 2) -> bones: (19, 2)"""
    bones = []
    for s, e in BONE_PAIRS:
        sj, ej = joints[s], joints[e]
        bones.append(ej - sj if np.any(sj != 0) and np.any(ej != 0) else np.zeros(2, dtype=np.float32))
    return np.array(bones, dtype=np.float32)



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
) -> dict:
    """Build a BST clip from a sequence of frame indices.

    This follows the official BST preprocessing:
    1. Joints normalized by bbox diagonal + center_align (range [-0.X, 0.X])
    2. Bones computed as endpoint differences
    3. Shuttle normalized by video resolution (range [0, 1])
    4. Position = feet midpoint in court-normalized coords (range [0, 1])

    Player ordering: p_idx=0 is ALWAYS the "far" player, p_idx=1 is "near".
    """
    n_frames = len(frames)
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

    for t, frame in enumerate(frames[:seq_len]):
        if shuttle_df is not None:
            s_row = shuttle_df[shuttle_df['frame'] == frame]
            if len(s_row) > 0:
                shuttle[t, 0] = float(s_row.iloc[0]['x']) / vid_w
                shuttle[t, 1] = float(s_row.iloc[0]['y']) / vid_h

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
                # Use detection bbox for stable normalization (issue 3 fix)
                det_bbox = det_bbox_lookup.get(pid, {}).get(frame)
                joints[t, p_idx] = _normalize_joints_bstdiag(coords, det_bbox=det_bbox)

                feet_x = (coords[15, 0] + coords[16, 0]) / 2
                feet_y = max(coords[15, 1], coords[16, 1])
                pos[t, p_idx, 0] = feet_x / vid_w
                pos[t, p_idx, 1] = feet_y / vid_h

    bones = np.zeros((seq_len, 2, len(BONE_PAIRS), 2), dtype=np.float32)
    for i in range(2):
        for t in range(seq_len):
            bones[t, i] = _create_bones(joints[t, i])
    JnB = np.concatenate([joints, bones], axis=-2)
    return {
        'JnB': JnB.reshape(seq_len, 2, -1),
        'shuttle': shuttle,
        'pos': pos,
        'video_len': min(n_frames, seq_len),
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

        from app.models.bst import BSTClassifier
        from app.config.settings import settings

        model_path = str(settings.bst_model_path) if settings.bst_model_path else None
        classifier = BSTClassifier(model_path, device=settings.device)

        court_length = court.get("court_length", COURT_LENGTH)
        court_width = court.get("court_width", COURT_WIDTH)

        vid_w, vid_h = 1280, 720
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

        shots = []
        previous_shots = []
        hit_frames_sorted = sorted(int(h["frame"]) for h in hits_df.to_dict('records'))

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

            while len(clip_frames) < classifier.seq_len:
                clip_frames.append(clip_frames[-1] if clip_frames else frame)
            clip_frames = clip_frames[:classifier.seq_len]

            clip = _build_clip(
                clip_frames, shuttle_df, pose_df,
                vid_w, vid_h, court_length, court_width,
                seq_len=classifier.seq_len,
                player_sides=player_sides, player_detections=player_list,
            )

            stroke_type, confidence = classifier.predict_single(clip)

            # Add flag if prediction is rule-based (fallback)
            is_rule_based = classifier.model is None  # BST failed to load

            shot = {
                "frame": frame,
                "hit_confidence": float(hit["confidence"]),
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
                "is_rule_based": is_rule_based,
                "is_bst_fallback": is_rule_based,
            }
            shots.append(shot)

            previous_shots.append({
                "stroke_type": stroke_type,
                "frame": frame,
                "stroke_confidence": confidence,
            })

        if len(shots) > 2:
            for i in range(len(shots)):
                if shots[i]["stroke_confidence"] >= 0.25 or shots[i]["stroke_type"] == "unknown":
                    continue
                neighbors = []
                for j in range(max(0, i - 2), min(len(shots), i + 3)):
                    if j != i and shots[j]["stroke_type"] != "unknown":
                        neighbors.append(shots[j]["stroke_type"])
                if neighbors:
                    from collections import Counter
                    majority = Counter(neighbors).most_common(1)[0]
                    if majority[0] != shots[i]["stroke_type"] and majority[1] >= 3:
                        shots[i]["stroke_type"] = majority[0]
                        shots[i]["stroke_confidence"] = 0.3

        fps = 30.0
        for i, s in enumerate(shots):
            s["shot_id"] = i + 1
            s["start_ts"] = round(s["frame"] / fps, 3)

        shots_df = pd.DataFrame(shots)
        artifacts.set_parquet("shots", shots_df)

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
