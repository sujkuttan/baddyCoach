"""Context fusion layer: nudge BST logits by physics/context likelihoods.

Runs before the physics gate (which catches hard contradictions).
Fusion provides soft guidance where BST is uncertain between
physically-plausible classes.
"""

import numpy as np
from app.config.settings import settings
from app.pipeline.shared.physics import extract_physics_features
from app.models.bst import COACH_STROKE_CLASSES


# ── Helper: sigmoid ─────────────────────────────────────────────

def _sigmoid(x, center, width):
    if width <= 0:
        return 0.5
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


# ── Shuttle trajectory likelihood sigmoid params ────────────────
# (speed_center, speed_width, dir_center, dir_width)
# speed in m/s (from homography), v_down in px/frame (>0 descending)
# Direction mode: 'descend' → sigmoid(v_down), 'ascend' → sigmoid(-v_down),
#                 'flat' → gaussian decay from 0
_STROKE_SIGMOIDS = {
    'smash':       (9.0, 0.8,  1.5, 0.5),
    'drop':        (4.0, 1.5,  1.0, 0.8),
    'clear':       (6.0, 1.5,  0.0, 2.0),
    'drive':       (7.0, 1.0,  0.0, 0.6),
    'lift':        (5.0, 1.5, -1.5, 0.8),
    'net_shot':    (2.0, 0.8,  0.0, 1.5),
    'block':       (4.0, 1.5,  1.0, 0.8),
    'push':        (4.5, 1.5,  0.0, 1.0),
    'rush':        (7.0, 1.0,  0.5, 0.8),
    'cross_court': (6.0, 1.5,  0.0, 1.5),
    'short_serve': (2.0, 0.8, -0.5, 0.8),
    'long_serve':  (5.0, 1.5, -1.5, 0.8),
}

_INVERSE_SPEED = {'drop', 'net_shot', 'block', 'short_serve'}
_DIRECTION_MODE = {
    'smash': 'descend', 'drop': 'descend', 'block': 'descend', 'rush': 'descend',
    'lift': 'ascend', 'short_serve': 'ascend', 'long_serve': 'ascend',
    'clear': 'flat', 'drive': 'flat', 'net_shot': 'flat', 'push': 'flat',
    'cross_court': 'flat',
}


# ── Court zone likelihood ───────────────────────────────────────
# Per-class score for each zone
_ZONE_LIKELIHOOD = {
    'smash':       {'front': 0.20, 'mid': 0.40, 'back': 0.60},
    'drop':        {'front': 0.60, 'mid': 0.30, 'back': 0.15},
    'clear':       {'front': 0.10, 'mid': 0.20, 'back': 0.85},
    'drive':       {'front': 0.20, 'mid': 0.70, 'back': 0.30},
    'lift':        {'front': 0.10, 'mid': 0.30, 'back': 0.80},
    'net_shot':    {'front': 0.90, 'mid': 0.25, 'back': 0.05},
    'block':       {'front': 0.40, 'mid': 0.55, 'back': 0.20},
    'push':        {'front': 0.70, 'mid': 0.40, 'back': 0.10},
    'rush':        {'front': 0.20, 'mid': 0.60, 'back': 0.20},
    'cross_court': {'front': 0.20, 'mid': 0.50, 'back': 0.50},
    'short_serve': {'front': 0.85, 'mid': 0.10, 'back': 0.05},
    'long_serve':  {'front': 0.10, 'mid': 0.30, 'back': 0.80},
}


# ── Contact height likelihood ───────────────────────────────────
_CONTACT_LIKELIHOOD = {
    'smash':       {'overhead': 0.90, 'side': 0.40, 'underarm': 0.10, 'low': 0.05},
    'drop':        {'overhead': 0.85, 'side': 0.50, 'underarm': 0.15, 'low': 0.10},
    'clear':       {'overhead': 0.85, 'side': 0.50, 'underarm': 0.10, 'low': 0.05},
    'drive':       {'overhead': 0.25, 'side': 0.80, 'underarm': 0.35, 'low': 0.20},
    'lift':        {'overhead': 0.05, 'side': 0.20, 'underarm': 0.85, 'low': 0.60},
    'net_shot':    {'overhead': 0.05, 'side': 0.15, 'underarm': 0.80, 'low': 0.70},
    'block':       {'overhead': 0.15, 'side': 0.55, 'underarm': 0.50, 'low': 0.30},
    'push':        {'overhead': 0.05, 'side': 0.30, 'underarm': 0.75, 'low': 0.60},
    'rush':        {'overhead': 0.75, 'side': 0.50, 'underarm': 0.15, 'low': 0.10},
    'cross_court': {'overhead': 0.40, 'side': 0.55, 'underarm': 0.40, 'low': 0.25},
    'short_serve': {'overhead': 0.05, 'side': 0.10, 'underarm': 0.80, 'low': 0.85},
    'long_serve':  {'overhead': 0.05, 'side': 0.10, 'underarm': 0.80, 'low': 0.75},
}


# ── Rally context transition table ──────────────────────────────
_CONTEXT_TABLE = {
    'smash':      {'block': 0.35, 'lift': 0.25, 'clear': 0.20, 'net_shot': 0.15, 'smash': 0.05},
    'block':      {'lift': 0.30, 'clear': 0.20, 'net_shot': 0.20, 'drive': 0.15, 'block': 0.15},
    'lift':       {'smash': 0.30, 'clear': 0.25, 'drop': 0.20, 'lift': 0.15, 'net_shot': 0.10},
    'clear':      {'smash': 0.25, 'drop': 0.25, 'clear': 0.20, 'lift': 0.15, 'net_shot': 0.15},
    'drop':       {'lift': 0.30, 'net_shot': 0.30, 'clear': 0.15, 'smash': 0.15, 'drop': 0.10},
    'drive':      {'drive': 0.30, 'smash': 0.20, 'block': 0.20, 'lift': 0.15, 'clear': 0.15},
    'net_shot':   {'lift': 0.35, 'net_shot': 0.20, 'push': 0.20, 'clear': 0.10, 'drop': 0.15},
    'push':       {'lift': 0.30, 'clear': 0.20, 'net_shot': 0.25, 'push': 0.15, 'drive': 0.10},
    'rush':       {'block': 0.30, 'lift': 0.30, 'clear': 0.20, 'net_shot': 0.10, 'rush': 0.10},
    'cross_court': {'clear': 0.25, 'smash': 0.20, 'lift': 0.20, 'drive': 0.20, 'drop': 0.15},
    'short_serve': {'lift': 0.40, 'net_shot': 0.30, 'clear': 0.15, 'smash': 0.10, 'drop': 0.05},
    'long_serve':  {'smash': 0.35, 'clear': 0.30, 'lift': 0.20, 'drop': 0.10, 'net_shot': 0.05},
}

_DEFAULT_TRANSITION = 0.15  # fallback when stroke not in table


# ── Likelihood functions ────────────────────────────────────────

def _shuttle_likelihood(stroke: str, feats) -> float:
    if not feats.usable:
        return 0.5
    sig = _STROKE_SIGMOIDS.get(stroke)
    if sig is None:
        return 0.5
    speed_center, speed_width, dir_center, dir_width = sig
    speed = feats.speed_mps if feats.speed_mps is not None else (
        feats.speed_norm * 20.0 if feats.speed_norm is not None else None
    )
    v_down = feats.v_down

    if speed is None and v_down is None:
        return 0.5
    score = 1.0
    if speed is not None:
        s = _sigmoid(speed, speed_center, speed_width)
        if stroke in _INVERSE_SPEED:
            s = 1.0 - s
        score *= s
    if v_down is not None:
        mode = _DIRECTION_MODE.get(stroke, 'flat')
        if mode == 'descend':
            score *= _sigmoid(v_down, dir_center, dir_width)
        elif mode == 'ascend':
            score *= _sigmoid(-v_down, dir_center, dir_width)
        else:  # 'flat' — gaussian decay from 0
            score *= np.exp(-0.5 * (v_down / max(dir_width, 1e-6)) ** 2)
    return float(np.clip(score, 0.05, 0.95))


def _zone_likelihood(stroke: str, feats) -> float:
    zone = feats.zone
    if zone is None:
        return 0.5
    lookup = _ZONE_LIKELIHOOD.get(stroke, {})
    return lookup.get(zone, 0.5)


def _height_likelihood(stroke: str, contact: str) -> float:
    if contact is None:
        return 0.5
    lookup = _CONTACT_LIKELIHOOD.get(stroke, {})
    return lookup.get(contact, 0.5)


def _context_likelihood(next_stroke: str, prev_stroke: str) -> float:
    if prev_stroke is None:
        return 0.5
    trans = _CONTEXT_TABLE.get(prev_stroke, {})
    return trans.get(next_stroke, _DEFAULT_TRANSITION)


# ── Logit helpers ───────────────────────────────────────────────

def _logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """Recover logits from softmax probabilities (up to additive constant)."""
    logits = np.log(np.clip(probs, 1e-10, 1.0 - 1e-10))
    logits -= logits.max(axis=1, keepdims=True)
    return logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ── ContextFusion class ─────────────────────────────────────────

class ContextFusion:
    """Nudge BST logits by physics/context likelihoods before the physics gate.

    Recoverable logits from softmax → add per-class physics biases →
    re-softmax → update shots and probs_matrix in-place.
    """

    def __init__(self, w_shuttle: float, w_zone: float, w_height: float,
                 w_context: float, logit_clip: float = 2.0):
        self.w_shuttle = w_shuttle
        self.w_zone = w_zone
        self.w_height = w_height
        self.w_context = w_context
        self.logit_clip = logit_clip

    @classmethod
    def from_settings(cls) -> "ContextFusion":
        return cls(
            w_shuttle=settings.fusion_shuttle_weight,
            w_zone=settings.fusion_zone_weight,
            w_height=settings.fusion_height_weight,
            w_context=settings.fusion_context_weight,
            logit_clip=settings.fusion_logit_clip,
        )

    def fuse(self, shots: list, probs_matrix: np.ndarray,
             shuttle_cleaned, shuttle_raw, pose_df, court, fps, vid_w, vid_h) -> np.ndarray:
        """Fuse BST probabilities with physics/context features.

        Args:
            shots: List of shot dicts (must have 'frame', 'stroke_type').
            probs_matrix: (N, 25) softmax probabilities from BST.
            shuttle_cleaned, shuttle_raw, pose_df, court, fps, vid_w, vid_h: physics features.

        Returns:
            Adjusted (N, 25) softmax probabilities.
        """
        n_shots = len(shots)
        if n_shots == 0:
            return probs_matrix
        n_classes = probs_matrix.shape[1]

        logits = _logits_from_probs(probs_matrix)
        cls_names = ["unknown"] + COACH_STROKE_CLASSES + COACH_STROKE_CLASSES

        court_corners = court.get("corners_pixel", []) if court else []
        net_y = ((court_corners[0][1] + court_corners[2][1]) / 2
                 if len(court_corners) >= 3 else None)

        prev_stroke = None

        for i, shot in enumerate(shots):
            frame = shot["frame"]
            feats = None
            if shuttle_cleaned is not None and len(shuttle_cleaned) > 0:
                feats = extract_physics_features(
                    frame, shuttle_cleaned, pose_df, "player_1",
                    court, fps, vid_w, vid_h, shuttle_raw,
                )

            # Contact height: try both players if pose available, take max per class
            contact_p0 = None
            contact_p1 = None
            if pose_df is not None and len(pose_df) > 0:
                from app.pipeline.shared.physics import contact_height
                contact_p0 = contact_height(pose_df, frame, "player_1", net_y)
                contact_p1 = contact_height(pose_df, frame, "player_2", net_y)

            for c in range(1, n_classes):  # skip unknown (index 0)
                cls = cls_names[c]
                bias = 0.0

                if feats is not None and feats.usable:
                    bias += self.w_shuttle * _shuttle_likelihood(cls, feats)
                    bias += self.w_zone * _zone_likelihood(cls, feats)

                    # Height: take max of both players (identity unknown at this point)
                    h_p0 = _height_likelihood(cls, contact_p0)
                    h_p1 = _height_likelihood(cls, contact_p1)
                    bias += self.w_height * max(h_p0, h_p1)

                if prev_stroke is not None:
                    bias += self.w_context * _context_likelihood(cls, prev_stroke)

                bias = float(np.clip(bias, -self.logit_clip, self.logit_clip))
                logits[i, c] += bias

            # Determine top class for context tracking
            top_idx = int(np.argmax(logits[i, 1:])) + 1
            top_cls = cls_names[top_idx]
            if top_cls and top_cls != "unknown":
                prev_stroke = top_cls

        adjusted_probs = _softmax(logits)

        # Update shots with adjusted stroke types and confidences
        for i, shot in enumerate(shots):
            top_idx = int(np.argmax(adjusted_probs[i]))
            if top_idx > 0:
                shot["stroke_type"] = cls_names[top_idx]
                shot["stroke_confidence"] = float(adjusted_probs[i, top_idx])

        return adjusted_probs
