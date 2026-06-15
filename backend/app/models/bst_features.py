"""Feature extraction pipeline for BST stroke classification.

Extracts 144-dimensional feature vectors from shuttle trajectory,
pose keypoints, court position, and rally context.
"""

import numpy as np
import pandas as pd


STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]


class BSTFeatureExtractor:
    """Extracts 144-dim feature vectors for BST stroke classification.
    
    Feature layout (144 dims total):
    - Shuttle trajectory (24): velocity/accel over 8-frame window
    - Shuttle position (6): current x, y, speed, direction
    - Pose joints (48): 17 keypoints x (x, y) normalized
    - Pose dynamics (12): joint velocities
    - Body orientation (6): torso angle, lean, arm extension
    - Court position (6): normalized court coords
    - Rally context (42): previous 3 shots encoded
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        court_length: float = 13.4,
        court_width: float = 5.18,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.court_length = court_length
        self.court_width = court_width
    
    def extract(
        self,
        shuttle_df: pd.DataFrame | None,
        pose_df: pd.DataFrame | None,
        target_frame: int,
        player_id: str,
        previous_shots: list[dict],
    ) -> np.ndarray:
        """Extract 144-dim feature vector for a single hit frame."""
        features = []
        features.append(self._extract_shuttle_trajectory(shuttle_df, target_frame))
        features.append(self._extract_shuttle_position(shuttle_df, target_frame))
        features.append(self._extract_pose_joints(pose_df, target_frame, player_id))
        features.append(self._extract_pose_dynamics(pose_df, target_frame, player_id))
        features.append(self._extract_body_orientation(pose_df, target_frame, player_id))
        features.append(self._extract_court_position(shuttle_df, target_frame))
        features.append(self._encode_previous_shots(previous_shots, target_frame))
        
        combined = np.concatenate(features)
        assert combined.shape == (144,), f"Expected 144 dims, got {combined.shape}"
        return combined
    
    def _extract_shuttle_trajectory(self, shuttle_df, target_frame):
        """24 dims: velocity/acceleration over 8-frame window."""
        if shuttle_df is None or len(shuttle_df) == 0:
            return np.zeros(24)
        
        window = shuttle_df[
            (shuttle_df["frame"] >= target_frame - 8) &
            (shuttle_df["frame"] <= target_frame)
        ].sort_values("frame")
        
        if len(window) < 2:
            return np.zeros(24)
        
        x = window["x"].values.astype(np.float64)
        y = window["y"].values.astype(np.float64)
        
        x = x / self.frame_width
        y = y / self.frame_height
        
        dx = np.diff(x)
        dy = np.diff(y)
        ddx = np.diff(dx) if len(dx) > 1 else np.array([0.0])
        ddy = np.diff(dy) if len(dy) > 1 else np.array([0.0])
        
        speed = np.sqrt(dx**2 + dy**2)
        
        features = np.array([
            np.mean(dx), np.std(dx), np.min(dx), np.max(dx),
            np.mean(dy), np.std(dy), np.min(dy), np.max(dy),
            np.mean(ddx), np.std(ddx), np.min(ddx), np.max(ddx),
            np.mean(ddy), np.std(ddy), np.min(ddy), np.max(ddy),
            np.mean(speed),
            np.max(speed),
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            np.mean(np.abs(ddx + ddy)),
            dx[-1] if len(dx) > 0 else 0,
            dy[-1] if len(dy) > 0 else 0,
            x[-1] - x[0] if len(x) > 1 else 0,
        ])
        
        return features[:24]
    
    def _extract_shuttle_position(self, shuttle_df, target_frame):
        """6 dims: current x, y, speed, direction, height, distance_from_net."""
        if shuttle_df is None or len(shuttle_df) == 0:
            return np.zeros(6)
        
        row = shuttle_df[shuttle_df["frame"] == target_frame]
        if len(row) == 0:
            nearby = shuttle_df[
                (shuttle_df["frame"] >= target_frame - 2) &
                (shuttle_df["frame"] <= target_frame + 2)
            ]
            if len(nearby) == 0:
                return np.zeros(6)
            row = nearby.iloc[[-1]]
        
        x = float(row.iloc[0]["x"]) / self.frame_width
        y = float(row.iloc[0]["y"]) / self.frame_height
        
        prev_rows = shuttle_df[shuttle_df["frame"] == target_frame - 1]
        if len(prev_rows) > 0:
            prev_x = float(prev_rows.iloc[0]["x"]) / self.frame_width
            prev_y = float(prev_rows.iloc[0]["y"]) / self.frame_height
            dx = x - prev_x
            dy = y - prev_y
            speed = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)
        else:
            speed = 0
            direction = 0
        
        height = y
        dist_from_net = abs(y - 0.5)
        
        return np.array([x, y, speed, direction, height, dist_from_net])
    
    def _extract_pose_joints(self, pose_df, target_frame, player_id):
        """48 dims: 17 keypoints x (x, y) normalized by bounding box."""
        if pose_df is None or len(pose_df) == 0:
            return np.zeros(48)
        
        row = pose_df[
            (pose_df["frame"] == target_frame) &
            (pose_df["player_id"] == player_id)
        ]
        
        if len(row) == 0:
            nearby = pose_df[
                (pose_df["frame"] >= target_frame - 2) &
                (pose_df["frame"] <= target_frame + 2) &
                (pose_df["player_id"] == player_id)
            ]
            if len(nearby) == 0:
                return np.zeros(48)
            row = nearby.iloc[[-1]]
        
        kps = np.array(row.iloc[0]["keypoints"])
        if kps.shape != (17, 3):
            kps = np.array(kps.tolist())
        if kps.shape != (17, 3):
            return np.zeros(48)
        
        coords = kps[:, :2]
        
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        diag = np.linalg.norm(bbox_max - bbox_min)
        if diag == 0:
            diag = 1
        
        coords_norm = (coords - bbox_min) / diag
        
        center = (bbox_min + bbox_max) / 2
        coords_centered = coords_norm - center / diag
        
        flat = coords_centered.flatten()
        
        return np.pad(flat, (0, 48 - len(flat)))[:48]
    
    def _extract_pose_dynamics(self, pose_df, target_frame, player_id):
        """12 dims: wrist, elbow, shoulder velocities."""
        if pose_df is None or len(pose_df) == 0:
            return np.zeros(12)
        
        curr_row = pose_df[
            (pose_df["frame"] == target_frame) &
            (pose_df["player_id"] == player_id)
        ]
        prev_row = pose_df[
            (pose_df["frame"] == target_frame - 1) &
            (pose_df["player_id"] == player_id)
        ]
        
        if len(curr_row) == 0 or len(prev_row) == 0:
            return np.zeros(12)
        
        curr_kps = np.array(curr_row.iloc[0]["keypoints"])
        prev_kps = np.array(prev_row.iloc[0]["keypoints"])
        
        if curr_kps.shape != (17, 3) or prev_kps.shape != (17, 3):
            return np.zeros(12)
        
        key_joints = [5, 7, 9, 10]  # shoulder, elbow, wrist L, wrist R
        velocities = []
        
        for joint in key_joints:
            dx = (curr_kps[joint, 0] - prev_kps[joint, 0]) / self.frame_width
            dy = (curr_kps[joint, 1] - prev_kps[joint, 1]) / self.frame_height
            velocities.extend([dx, dy, np.sqrt(dx**2 + dy**2)])
        
        return np.array(velocities)[:12]
    
    def _extract_body_orientation(self, pose_df, target_frame, player_id):
        """6 dims: torso angle, lean, arm extension metrics."""
        if pose_df is None or len(pose_df) == 0:
            return np.zeros(6)
        
        row = pose_df[
            (pose_df["frame"] == target_frame) &
            (pose_df["player_id"] == player_id)
        ]
        
        if len(row) == 0:
            return np.zeros(6)
        
        kps = np.array(row.iloc[0]["keypoints"])
        if kps.shape != (17, 3):
            kps = np.array(kps.tolist())
        if kps.shape != (17, 3):
            return np.zeros(6)
        
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
        LEFT_ELBOW, RIGHT_ELBOW = 7, 8
        LEFT_WRIST, RIGHT_WRIST = 9, 10
        LEFT_HIP, RIGHT_HIP = 11, 12
        
        torso_vec = kps[LEFT_HIP, :2] - kps[LEFT_SHOULDER, :2]
        torso_angle = np.arctan2(torso_vec[1], torso_vec[0])
        
        shoulder_center = (kps[LEFT_SHOULDER, :2] + kps[RIGHT_SHOULDER, :2]) / 2
        hip_center = (kps[LEFT_HIP, :2] + kps[RIGHT_HIP, :2]) / 2
        lean = (shoulder_center[0] - hip_center[0]) / self.frame_width
        
        left_arm = np.linalg.norm(kps[LEFT_WRIST, :2] - kps[LEFT_SHOULDER, :2])
        right_arm = np.linalg.norm(kps[RIGHT_WRIST, :2] - kps[RIGHT_SHOULDER, :2])
        torso_len = np.linalg.norm(kps[LEFT_SHOULDER, :2] - kps[LEFT_HIP, :2])
        
        if torso_len == 0:
            torso_len = 1
        
        left_ext = left_arm / torso_len
        right_ext = right_arm / torso_len
        racket_arm_ext = max(left_ext, right_ext)
        
        return np.array([
            torso_angle / np.pi,
            lean,
            left_ext / 3,
            right_ext / 3,
            racket_arm_ext / 3,
            (left_ext - right_ext) / 3,
        ])
    
    def _extract_court_position(self, shuttle_df, target_frame):
        """6 dims: normalized court x, y, distance from corners."""
        if shuttle_df is None or len(shuttle_df) == 0:
            return np.zeros(6)
        
        row = shuttle_df[shuttle_df["frame"] == target_frame]
        if len(row) == 0:
            return np.zeros(6)
        
        x = float(row.iloc[0]["x"]) / self.frame_width
        y = float(row.iloc[0]["y"]) / self.frame_height
        
        court_x = x
        court_y = y
        
        dist_tl = np.sqrt(court_x**2 + court_y**2)
        dist_tr = np.sqrt((1 - court_x)**2 + court_y**2)
        dist_bl = np.sqrt(court_x**2 + (1 - court_y)**2)
        dist_br = np.sqrt((1 - court_x)**2 + (1 - court_y)**2)
        
        return np.array([court_x, court_y, dist_tl, dist_tr, dist_bl, dist_br])
    
    def _encode_previous_shots(self, previous_shots, current_frame):
        """42 dims: encode last 3 shots (type one-hot + frame gap)."""
        features = []
        
        recent = [s for s in previous_shots if s["frame"] < current_frame][-3:]
        
        for shot in recent:
            stroke = shot.get("stroke_type", "clear")
            one_hot = np.zeros(12)
            if stroke in STROKE_CLASSES:
                one_hot[STROKE_CLASSES.index(stroke)] = 1
            
            gap = (current_frame - shot["frame"]) / 100
            conf = shot.get("stroke_confidence", 0.8)
            
            features.extend(one_hot)
            features.extend([gap, conf])
        
        while len(features) < 42:
            features.append(0)
        
        return np.array(features[:42])
