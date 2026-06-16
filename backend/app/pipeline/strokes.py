import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult

BONE_PAIRS = [
    (0,1),(0,2),(1,2),(1,3),(2,4),
    (3,5),(4,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

SEQ_LEN = 30


def _normalize_joints_bstdiag(coords: np.ndarray) -> np.ndarray:
    """Normalize joints using bbox diagonal with center_align.

    Matches the official BST ``normalize_joints`` preprocessing:
    - Origin = top-left of the player bounding box
    - Scale = diagonal distance of the bounding box
    - center_align=True shifts origin to bbox center

    Args:
        coords: (17, 2) keypoints in pixel coords (or [0,1] RTMPose coords)

    Returns:
        (17, 2) normalized joints, range roughly [-0.X, 0.X]
    """
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    if diag < 1e-6:
        diag = 1.0

    normalized = (coords - bbox_min) / diag
    center = (bbox_min + bbox_max) / (2 * diag)
    normalized -= center
    return normalized.astype(np.float32)


def _create_bones(joints: np.ndarray) -> np.ndarray:
    """Create bone vectors from joint positions. joints: (17, 2) -> bones: (19, 2)"""
    bones = []
    for s, e in BONE_PAIRS:
        sj, ej = joints[s], joints[e]
        bones.append(ej - sj if np.any(sj != 0) and np.any(ej != 0) else np.zeros(2, dtype=np.float32))
    return np.array(bones, dtype=np.float32)


def _normalize_position(feet_coords: np.ndarray, court_length: float, court_width: float) -> np.ndarray:
    """Convert pixel feet position to court coordinates [0, 1].

    Uses homography-style mapping: pixel y maps to court length, pixel x maps to court width.
    For a standard badminton broadcast view, feet_y maps to court纵向 and feet_x to court横向.
    """
    return feet_coords.astype(np.float32)


def _get_keypoints_for_frame(pose_df: pd.DataFrame, frame: int, player_id: str) -> np.ndarray | None:
    """Get (17, 3) keypoints for a frame/player from pose dataframe."""
    if pose_df is None or len(pose_df) == 0:
        return None
    row = pose_df[(pose_df['frame'] == frame) & (pose_df['player_id'] == player_id)]
    if len(row) == 0:
        return None
    kps = np.array(row.iloc[0]['keypoints'])
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
    seq_len: int = SEQ_LEN,
) -> dict:
    """Build a BST clip from a sequence of frame indices.

    This follows the official BST preprocessing:
    1. Joints normalized by bbox diagonal + center_align (range [-0.X, 0.X])
    2. Bones computed as endpoint differences
    3. Shuttle normalized by video resolution (range [0, 1])
    4. Position = feet midpoint in court-normalized coords (range [0, 1])
    """
    n_frames = len(frames)
    joints = np.zeros((seq_len, 2, 17, 2), dtype=np.float32)
    shuttle = np.zeros((seq_len, 2), dtype=np.float32)
    pos = np.zeros((seq_len, 2, 2), dtype=np.float32)

    for t, frame in enumerate(frames[:seq_len]):
        if shuttle_df is not None:
            s_row = shuttle_df[shuttle_df['frame'] == frame]
            if len(s_row) > 0:
                shuttle[t, 0] = float(s_row.iloc[0]['x']) / vid_w
                shuttle[t, 1] = float(s_row.iloc[0]['y']) / vid_h

        for p_idx, pid in enumerate(['player_1', 'player_2']):
            kps = _get_keypoints_for_frame(pose_df, frame, pid)
            if kps is not None:
                coords = kps[:, :2]
                joints[t, p_idx] = _normalize_joints_bstdiag(coords)

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

        court_length = court.get("court_length", 13.4)
        court_width = court.get("court_width", 5.18)

        vid_w, vid_h = 1280, 720
        if shuttle_df is not None and len(shuttle_df) > 0:
            vid_w = max(float(shuttle_df["x"].max()), 640)
            vid_h = max(float(shuttle_df["y"].max()), 480)

        hit_frames = sorted(int(h["frame"]) for h in hits_df.to_dict('records'))

        shots = []
        previous_shots = []

        for _, hit in hits_df.iterrows():
            frame = int(hit["frame"])
            hit_pos = hit_frames.index(frame) if frame in hit_frames else 0

            start_frame = hit_frames[hit_pos - 1] if hit_pos > 0 else max(0, frame - SEQ_LEN // 2)
            end_frame = hit_frames[hit_pos + 1] + 2 if hit_pos < len(hit_frames) - 1 else frame + SEQ_LEN // 2 + 1

            clip_frames = list(range(start_frame, end_frame))
            while len(clip_frames) < SEQ_LEN:
                clip_frames.append(clip_frames[-1] if clip_frames else frame)
            clip_frames = clip_frames[:SEQ_LEN]

            clip = _build_clip(
                clip_frames, shuttle_df, pose_df,
                vid_w, vid_h, court_length, court_width,
            )

            stroke_type, confidence = classifier.predict_single(clip)

            shot = {
                "frame": frame,
                "hit_confidence": float(hit["confidence"]),
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
            }
            shots.append(shot)

            previous_shots.append({
                "stroke_type": stroke_type,
                "frame": frame,
                "stroke_confidence": confidence,
            })

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
