"""BST preprocessing utilities.

Handles normalization, bone creation, and sequence extraction for BST model input.
Based on the BST paper's data preparation pipeline.
"""

import numpy as np
from typing import Optional


# COCO skeleton format: 17 joints
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Bone pairs for COCO format
BONE_PAIRS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),    # head
    (3, 5), (4, 6),                              # ears to shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),            # arms
    (5, 6), (5, 11), (6, 12), (11, 12),         # torso
    (11, 13), (13, 15), (12, 14), (14, 16)      # legs
]


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution.
    
    Args:
        arr: (T, 2) array of (x, y) positions
        v_width: Video width
        v_height: Video height
    
    Returns:
        Normalized array (T, 2) with values in [0, 1]
    """
    return arr / np.array([v_width, v_height])


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    center_align: bool = True
) -> np.ndarray:
    """Normalize joints by bounding box diagonal distance.
    
    Args:
        arr: (T, M, J, 2) array of joint positions (T=frames, M=people, J=joints)
        bbox: (T, M, 4) array of bounding boxes (x1, y1, x2, y2)
        center_align: If True, center of bbox is origin
    
    Returns:
        Normalized array (T, M, J, 2)
    """
    diag = np.linalg.norm(bbox[:, :, 2:] - bbox[:, :, :2], axis=-1, keepdims=True)
    diag = np.where(diag == 0, 1, diag)
    
    arr_x = arr[:, :, :, 0]
    arr_y = arr[:, :, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, :, None, 0]) / diag, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, :, None, 1]) / diag, 0.0)
    
    if center_align:
        center = (bbox[:, :, :2] + bbox[:, :, 2:]) / 2
        c_normalized = (center - bbox[:, :, :2]) / diag
        x_normalized -= c_normalized[:, :, None, 0]
        y_normalized -= c_normalized[:, :, None, 1]
    
    return np.stack((x_normalized, y_normalized), axis=-1)


def normalize_position(
    arr: np.ndarray,
    court_corners: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize player position by court boundary.
    
    Args:
        arr: (T, M, 2) array of player positions (feet midpoint)
        court_corners: (4, 2) array of court corners (optional)
    
    Returns:
        Normalized array (T, M, 2) with values in [0, 1]
    """
    if court_corners is not None:
        # Use actual court corners
        x_min, y_min = court_corners.min(axis=0)
        x_max, y_max = court_corners.max(axis=0)
    else:
        # Estimate from data
        x_min, y_min = arr.min(axis=(0, 1))
        x_max, y_max = arr.max(axis=(0, 1))
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    
    normalized = (arr - np.array([x_min, y_min])) / np.array([width, height])
    return np.clip(normalized, 0, 1)


def create_bones(joints: np.ndarray, pairs: list = None) -> np.ndarray:
    """Create bone vectors from joint positions.
    
    Args:
        joints: (T, M, J, 2) array of joint positions
        pairs: List of (start, end) joint index pairs
    
    Returns:
        bones: (T, M, B, 2) array of bone vectors
    """
    if pairs is None:
        pairs = BONE_PAIRS
    
    bones = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
        bones.append(bone)
    
    return np.stack(bones, axis=-2)


def extract_stroke_clip(
    frames: list,
    hit_frame_idx: int,
    seq_len: int = 30,
    all_hit_frames: Optional[list] = None,
) -> tuple:
    """Extract a stroke clip around the hit frame.
    
    Uses the BST paper's clipping strategy:
    - Start from previous opponent's hit frame
    - End at next opponent's hit frame (with small epsilon)
    
    Args:
        frames: List of frame data (pose, shuttle, position)
        hit_frame_idx: Index of the hit frame in frames list
        seq_len: Target sequence length
        all_hit_frames: List of all hit frame indices (for smart clipping)
    
    Returns:
        clip_frames: List of frame data for the clip
        actual_len: Actual sequence length (before padding)
    """
    if all_hit_frames is None:
        # Fixed-width clipping
        start = max(0, hit_frame_idx - seq_len // 2)
        end = min(len(frames), start + seq_len)
        start = max(0, end - seq_len)
    else:
        # Smart clipping based on hit frames
        hit_frames_sorted = sorted(all_hit_frames)
        pos = hit_frames_sorted.index(hit_frame_idx)
        
        # Previous opponent's hit (or start with margin)
        if pos > 0:
            start = hit_frames_sorted[pos - 1]
        else:
            start = max(0, hit_frame_idx - seq_len // 2)
        
        # Next opponent's hit (or end with margin)
        if pos < len(hit_frames_sorted) - 1:
            end = hit_frames_sorted[pos + 1] + 2  # small epsilon
        else:
            end = min(len(frames), hit_frame_idx + seq_len // 2 + 1)
    
    clip_frames = frames[start:end]
    actual_len = len(clip_frames)
    
    return clip_frames, actual_len


def prepare_bst_input(
    clip_frames: list,
    seq_len: int = 30,
    player1_id: str = 'player_1',
    player2_id: str = 'player_2',
) -> dict:
    """Prepare BST model input from a clip of frames.
    
    Args:
        clip_frames: List of frame dicts with keys:
            - pose: dict mapping player_id -> (17, 3) keypoints
            - shuttle: (x, y) position
            - position: dict mapping player_id -> (x, y) court position
        seq_len: Target sequence length
        player1_id: ID for player 1 (Top)
        player2_id: ID for player 2 (Bottom)
    
    Returns:
        dict with keys:
            - JnB: (seq_len, 2, 72) pose + bones
            - shuttle: (seq_len, 2) shuttle positions
            - pos: (seq_len, 2, 2) player positions
            - video_len: int (actual length)
    """
    n_frames = len(clip_frames)
    
    # Initialize arrays
    joints = np.zeros((seq_len, 2, 17, 2), dtype=np.float32)
    shuttle = np.zeros((seq_len, 2), dtype=np.float32)
    positions = np.zeros((seq_len, 2, 2), dtype=np.float32)
    
    for t, frame in enumerate(clip_frames[:seq_len]):
        # Extract shuttle position
        if 'shuttle' in frame:
            shuttle[t] = frame['shuttle'][:2]
        
        # Extract pose for both players
        for p_idx, pid in enumerate([player1_id, player2_id]):
            if 'pose' in frame and pid in frame['pose']:
                kps = frame['pose'][pid]
                if kps is not None and len(kps) == 17:
                    joints[t, p_idx] = kps[:, :2]
            
            # Extract position
            if 'position' in frame and pid in frame['position']:
                positions[t, p_idx] = frame['position'][pid][:2]
    
    # Create bones
    bones = create_bones(joints)
    
    # Concatenate joints and bones: (seq_len, 2, 17+19, 2) -> (seq_len, 2, 72)
    JnB = np.concatenate([joints, bones], axis=-2)  # (seq_len, 2, 36, 2)
    JnB = JnB.reshape(seq_len, 2, -1)  # (seq_len, 2, 72)
    
    return {
        'JnB': JnB,
        'shuttle': shuttle,
        'pos': positions,
        'video_len': min(n_frames, seq_len),
    }


def prepare_stroke_clips_from_pipeline(
    shuttle_data: list,
    pose_data: list,
    shots_data: list,
    frame_width: int = 1280,
    frame_height: int = 720,
    seq_len: int = 30,
) -> list:
    """Prepare BST input clips from pipeline data.
    
    Args:
        shuttle_data: List of dicts with frame, x, y, confidence
        pose_data: List of dicts with frame, player_id, keypoints
        shots_data: List of dicts with frame, hit_confidence
        frame_width: Video width
        frame_height: Video height
        seq_len: Target sequence length
    
    Returns:
        List of dicts with BST inputs for each shot
    """
    import pandas as pd
    
    shuttle_df = pd.DataFrame(shuttle_data) if shuttle_data else pd.DataFrame()
    pose_df = pd.DataFrame(pose_data) if pose_data else pd.DataFrame()
    
    # Get all hit frames
    hit_frames = [s['frame'] for s in shots_data]
    
    # Group pose by frame
    pose_by_frame = {}
    if len(pose_df) > 0:
        for frame_idx in pose_df['frame'].unique():
            frame_poses = pose_df[pose_df['frame'] == frame_idx]
            pose_dict = {}
            for _, row in frame_poses.iterrows():
                pid = row['player_id']
                kps = np.array(row['keypoints'])
                if kps.shape == (17, 3):
                    kps = kps[:, :2]  # Take x, y only
                pose_dict[pid] = kps
            pose_by_frame[frame_idx] = pose_dict
    
    clips = []
    
    for shot in shots_data:
        hit_frame = shot['frame']
        
        # Find frame range for this clip
        hit_pos = hit_frames.index(hit_frame)
        if hit_pos > 0:
            start_frame = hit_frames[hit_pos - 1]
        else:
            start_frame = max(0, hit_frame - seq_len // 2)
        
        if hit_pos < len(hit_frames) - 1:
            end_frame = hit_frames[hit_pos + 1] + 2
        else:
            end_frame = hit_frame + seq_len // 2 + 1
        
        # Extract frames in range
        clip_frames = []
        for f in range(start_frame, end_frame):
            frame_data = {}
            
            # Shuttle
            if len(shuttle_df) > 0:
                shuttle_row = shuttle_df[shuttle_df['frame'] == f]
                if len(shuttle_row) > 0:
                    x = float(shuttle_row.iloc[0]['x']) / frame_width
                    y = float(shuttle_row.iloc[0]['y']) / frame_height
                    frame_data['shuttle'] = np.array([x, y], dtype=np.float32)
                else:
                    frame_data['shuttle'] = np.zeros(2, dtype=np.float32)
            else:
                frame_data['shuttle'] = np.zeros(2, dtype=np.float32)
            
            # Pose
            frame_data['pose'] = pose_by_frame.get(f, {})
            
            # Position (use shuttle position as proxy for player position)
            frame_data['position'] = {}
            
            clip_frames.append(frame_data)
        
        # Pad/truncate to seq_len
        while len(clip_frames) < seq_len:
            clip_frames.append({
                'shuttle': np.zeros(2, dtype=np.float32),
                'pose': {},
                'position': {},
            })
        clip_frames = clip_frames[:seq_len]
        
        # Prepare BST input
        bst_input = prepare_bst_input(clip_frames, seq_len)
        bst_input['shot_frame'] = hit_frame
        bst_input['hit_confidence'] = shot.get('hit_confidence', 1.0)
        
        clips.append(bst_input)
    
    return clips
