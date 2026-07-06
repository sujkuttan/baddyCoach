"""Stroke feature extraction for rule-based classification.

Transforms a BST clip dict into the rich feature vector described in
badminton_stroke_type_algorithm.docx (Spec), supporting hierarchical
family→specific classification with far-side mirroring.

Extracted features are in court-normalized [0,1] space (by court_length/
court_width) or in derived physical units (m/s through fps).
"""

import numpy as np

# ── Thresholds in court-normalized [0,1] space ─────────────────────
# These assume shuttle is normalized by (court_length, court_width).
# When clip uses resolution-normalization, a rough conversion is applied.

# Speeds in normalized displacement per frame (at 30fps equivalent)
# Converted from spec's m/s: v_norm = v_mps / (court_length * fps)
# e.g. SMASH_SPEED = 12 m/s / (13.4 * 30) ≈ 0.030
_SMASH_SPEED = 0.030
_DRIVE_SPEED = 0.020
_PUSH_SPEED = 0.012
_NET_SHOT_SPEED = 0.008
_BLOCK_SPEED = 0.010
_HIGH_INCOMING = 0.025
_DROP_SPEED = 0.015

# Directions — dy of post-hit trajectory (shuttle.y diff per frame)
_DESCEND_THRESH = 0.003      # positive dy = descending (y increasing)
_ASCEND_THRESH = -0.003      # negative dy = ascending (y decreasing)
_FLAT_THRESH = 0.002         # |dy| below this = flat

# Landing zones — normalized shuttle.x (0..1, ~0.5 = net)
_FRONT_X = 0.78              # front court (near side: x > 0.78)
_MID_X = 0.62                # mid court boundary
_BACK_X = 0.50               # rear court boundary

# Contact zones — from player pos.x
_ZONE_FRONT = 0.78
_ZONE_BACK = 0.62

# Arc / curvature measure
_HIGH_ARC = 0.020            # std of post-hit shuttle.x around linear fit


def extract_clip_features(clip: dict) -> dict:
    """Extract stroke features from a BST clip dict.

    Returns a flat dict with keys matching the spec's feature vector
    (adapted to available clip data). All spatial features are in
    court-normalized [0,1] space; velocities are in normalized-displacement
    per frame (convert to m/s via court_length × fps where needed).
    """
    seq_len = len(clip.get('shuttle', []))
    shuttle = clip.get('shuttle', np.zeros((seq_len, 2)))
    pos = clip.get('pos', np.zeros((seq_len, 2, 2)))
    jnb = clip.get('JnB', np.zeros((seq_len, 2, 72)))
    video_len = clip.get('video_len', seq_len)
    court_length = clip.get('court_length', 13.4)
    court_width = clip.get('court_width', 6.10)

    feats: dict = {}

    # ── Contact point (shuttle at hit frame, index 0) ──────────────
    contact_x = float(shuttle[0, 0]) if len(shuttle) > 0 else 0.5
    contact_y = float(shuttle[0, 1]) if len(shuttle) > 0 else 0.5
    feats['contact_x'] = contact_x
    feats['contact_y'] = contact_y

    # ── Post-hit shuttle trajectory (frames 0..video_len-1) ────────
    post = shuttle[:video_len]
    valid = (post[:, 0] != 0) | (post[:, 1] != 0)
    n_valid = int(valid.sum())
    feats['n_valid_frames'] = n_valid

    if n_valid < 3:
        feats['usable'] = False
        feats.update({
            'outgoing_speed': 0.0, 'outgoing_dy': 0.0, 'outgoing_angle': 0.0,
            'trajectory_curvature': 0.0, 'landing_x': contact_x,
            'landing_y': contact_y, 'max_speed': 0.0,
        })
        return feats
    feats['usable'] = True
    traj = post[valid]

    # Speeds
    dx = np.diff(traj[:, 0])
    dy = np.diff(traj[:, 1])
    speeds = np.sqrt(dx ** 2 + dy ** 2)
    feats['max_speed'] = float(np.max(speeds)) if len(speeds) > 0 else 0.0
    feats['outgoing_speed'] = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    feats['outgoing_speed_top5'] = float(np.mean(np.sort(speeds)[-5:])) if len(speeds) >= 5 else feats['outgoing_speed']

    # Direction
    feats['outgoing_dy'] = float(np.mean(dy)) if len(dy) > 0 else 0.0
    mean_dx = float(np.mean(dx)) if len(dx) > 0 else 0.0
    feats['outgoing_angle'] = float(np.arctan2(np.mean(dy), mean_dx + 1e-10)) if len(dy) > 0 else 0.0

    # Landing zone
    feats['landing_x'] = float(traj[-1, 0])
    feats['landing_y'] = float(traj[-1, 1])

    # Trajectory curvature: std of trajectory around linear fit
    if len(traj) >= 4:
        t_vals = np.linspace(0, 1, len(traj))
        coeffs = np.polyfit(t_vals, traj[:, 0], 1)
        linear_fit = np.polyval(coeffs, t_vals)
        feats['trajectory_curvature'] = float(np.std(traj[:, 0] - linear_fit))
    else:
        feats['trajectory_curvature'] = 0.0

    # ── Pre-hit (incoming) shuttle from frames before clip? ─────────
    # Not available in between-2-hits clips. Estimate from first few
    # post-hit frames to derive incoming direction.
    feats['incoming_speed'] = feats['outgoing_speed']
    feats['reversal_angle'] = 0.0

    # ── Player position at contact ──────────────────────────────────
    # pos[t, 0] = far player, pos[t, 1] = near player
    # Infer hitter: the player closer to the shuttle at contact
    if len(pos) > 0:
        p_far = pos[0, 0] if len(pos[0]) > 0 else np.array([0.5, 0.5])
        p_near = pos[0, 1] if len(pos[0]) > 1 else np.array([0.5, 0.5])
        dist_far = np.sqrt((p_far[0] - contact_x) ** 2 + (p_far[1] - contact_y) ** 2)
        dist_near = np.sqrt((p_near[0] - contact_x) ** 2 + (p_near[1] - contact_y) ** 2)
        is_near_hitter = dist_near <= dist_far
        hitter_pos = p_near if is_near_hitter else p_far
        feats['player_x'] = float(hitter_pos[0])
        feats['player_y'] = float(hitter_pos[1])
        feats['hitter_side'] = 'near' if is_near_hitter else 'far'

        # Far-side mirroring: mirror far-player's shuttle trajectory
        if not is_near_hitter:
            feats['contact_x'] = 1.0 - contact_x
            feats['landing_x'] = 1.0 - traj[-1, 0]
            feats['player_x'] = 1.0 - float(hitter_pos[0])
            feats['outgoing_dy'] = -feats['outgoing_dy']
            feats['outgoing_angle'] = -feats['outgoing_angle']
    else:
        feats['player_x'] = 0.5
        feats['player_y'] = 0.5
        feats['hitter_side'] = 'unknown'

    # Distance from net
    feats['distance_to_net'] = abs(feats['player_x'] - 0.5)

    # ── Joint angles at contact from JnB ────────────────────────────
    # JnB[t, p, :34] = 17 joints × 2 coords (COCO order)
    # Determine which player index (0=far, 1=near) is the hitter
    hitter_p_idx = 1 if feats['hitter_side'] == 'near' else 0
    if len(jnb) > 0:
        j = jnb[0, hitter_p_idx, :34]
        if j.shape[0] == 34:
            joints = j.reshape(17, 2)
            # Elbow angle — COCO-17: 5=L_shoulder, 6=R_shoulder, 7=L_elbow
            # TODO: this mixes left/right arm chains — needs arm+handedness detection
            v1 = joints[5] - joints[6]
            v2 = joints[7] - joints[6]
            n1 = np.linalg.norm(v1) + 1e-10
            n2 = np.linalg.norm(v2) + 1e-10
            cos_elbow = float(np.dot(v1, v2) / (n1 * n2))
            feats['elbow_angle'] = float(np.arccos(np.clip(cos_elbow, -1, 1)))

            # Shoulder angle (COCO: 5=shoulder, 6=elbow, 11=hip)
            v_sh = joints[5] - joints[11]
            n_sh = np.linalg.norm(v_sh) + 1e-10
            cos_shoulder = float(np.dot(v1, v_sh / n_sh))
            feats['shoulder_angle'] = float(np.arccos(np.clip(cos_shoulder, -1, 1)))

            # Torso rotation: angle between shoulder-midline and vertical
            # Use shoulders (5,6) and hips (11,12)
            shoulder_mid = (joints[5] + joints[6]) / 2
            hip_mid = (joints[11] + joints[12]) / 2
            torso_vec = shoulder_mid - hip_mid
            feats['torso_rotation'] = float(np.arctan2(torso_vec[0], abs(torso_vec[1]) + 1e-10))

            # Knee flexion (COCO: 11=hip, 13=knee, 15=ankle)
            v_k1 = joints[13] - joints[11]
            v_k2 = joints[15] - joints[13]
            nk1 = np.linalg.norm(v_k1) + 1e-10
            nk2 = np.linalg.norm(v_k2) + 1e-10
            cos_knee = float(np.dot(v_k1, v_k2) / (nk1 * nk2))
            feats['knee_flexion'] = float(np.arccos(np.clip(cos_knee, -1, 1)))
        else:
            feats['elbow_angle'] = 0.0
            feats['shoulder_angle'] = 0.0
            feats['torso_rotation'] = 0.0
            feats['knee_flexion'] = 0.0
    else:
        feats['elbow_angle'] = 0.0
        feats['shoulder_angle'] = 0.0
        feats['torso_rotation'] = 0.0
        feats['knee_flexion'] = 0.0

    # ── Contact height from shuttle.y (court-normalized) ────────────
    # y ranges [0,1] across court width. Low y = near sideline.
    # "Above head" classification: use elbow angle as proxy
    feats['contact_height'] = contact_y

    return feats


# ── Hierarchical classification ────────────────────────────────────


def classify_family(feats: dict) -> str:
    """Level-1 family classification per Spec §6."""
    if not feats.get('usable', False):
        return 'mid_height'

    contact_x = feats['contact_x']
    player_x = feats['player_x']
    outgoing_dy = feats['outgoing_dy']
    max_speed = feats['max_speed']
    outgoing_speed = feats['outgoing_speed']
    distance_to_net = feats['distance_to_net']

    # Serve: first stroke check is handled by caller; use contact + speed
    if contact_x > _FRONT_X and max_speed < _NET_SHOT_SPEED and distance_to_net < 0.15:
        return 'serve'

    # Overhead: fast descending trajectory OR high contact + speed
    if outgoing_dy > _DESCEND_THRESH and max_speed > _SMASH_SPEED * 0.6:
        return 'overhead'
    if outgoing_dy > _DESCEND_THRESH * 0.5 and max_speed > _DRIVE_SPEED:
        return 'overhead'

    # Defensive block: incoming fast, outgoing slow (estimated from trajectory)
    if max_speed > _HIGH_INCOMING and outgoing_speed < _BLOCK_SPEED:
        return 'defensive_block'

    # Net: near net, low speed
    if distance_to_net < 0.08 and max_speed < _DRIVE_SPEED:
        return 'net'

    # Underhand: ascending trajectory
    if outgoing_dy < _ASCEND_THRESH:
        return 'underhand'

    return 'mid_height'


def classify_overhead(feats: dict) -> str:
    """Spec §7.1: smash, drop, clear."""
    max_speed = feats['max_speed']
    landing_x = feats['landing_x']
    outgoing_dy = feats['outgoing_dy']
    curvature = feats['trajectory_curvature']

    is_descend = outgoing_dy > _DESCEND_THRESH * 0.5
    is_deep = landing_x < _BACK_X
    is_front = landing_x > _FRONT_X
    is_high_arc = curvature > _HIGH_ARC

    if is_descend and max_speed > _SMASH_SPEED:
        return 'smash'
    if is_descend and is_front and max_speed <= _DROP_SPEED:
        return 'drop'
    if is_high_arc and is_deep:
        return 'clear'
    if is_descend and max_speed > _SMASH_SPEED * 0.7:
        return 'smash'

    return 'overhead_unknown'


def classify_underhand(feats: dict) -> str:
    """Spec §7.2: lift, defensive_lift."""
    landing_x = feats['landing_x']
    outgoing_dy = feats['outgoing_dy']
    max_speed = feats['max_speed']
    outgoing_speed = feats['outgoing_speed']

    is_ascend = outgoing_dy < _ASCEND_THRESH
    is_deep = landing_x < _BACK_X

    if is_ascend and is_deep:
        if max_speed > _HIGH_INCOMING:
            return 'defensive_lift'
        return 'lift'
    if is_ascend and not is_deep:
        return 'soft_lift_or_push'

    return 'underhand_unknown'


def classify_net(feats: dict) -> str:
    """Spec §7.3: net_shot, net_lift, push."""
    max_speed = feats['max_speed']
    outgoing_speed = feats['outgoing_speed']
    landing_x = feats['landing_x']
    outgoing_dy = feats['outgoing_dy']
    contact_x = feats['contact_x']

    is_front = landing_x > _FRONT_X
    is_ascend = outgoing_dy < _ASCEND_THRESH

    if max_speed < _NET_SHOT_SPEED and is_front:
        return 'net_shot'
    if is_ascend and not is_front:
        return 'net_lift'
    if max_speed > _PUSH_SPEED and abs(outgoing_dy) < _FLAT_THRESH:
        return 'push'

    if max_speed > _NET_SHOT_SPEED and outgoing_dy > _DESCEND_THRESH:
        return 'net_kill'

    return 'net_unknown'


def classify_mid_height(feats: dict) -> str:
    """Spec §7.4: drive, block, push."""
    max_speed = feats['max_speed']
    outgoing_speed = feats['outgoing_speed']
    outgoing_dy = feats['outgoing_dy']

    is_flat = abs(outgoing_dy) < _FLAT_THRESH

    if is_flat and max_speed > _DRIVE_SPEED:
        return 'drive'
    if max_speed > _HIGH_INCOMING and outgoing_speed < _BLOCK_SPEED:
        return 'block'
    if is_flat and max_speed > _PUSH_SPEED:
        return 'push'

    return 'mid_height_unknown'


def classify_serve(feats: dict) -> str:
    """Spec §7.5: short_serve, long_serve."""
    max_speed = feats['max_speed']
    landing_x = feats['landing_x']
    outgoing_dy = feats['outgoing_dy']

    if max_speed < _NET_SHOT_SPEED and landing_x > _FRONT_X:
        return 'short_serve'
    if landing_x < _BACK_X and outgoing_dy < _ASCEND_THRESH:
        return 'long_serve'

    return 'serve_unknown'


def classify_by_family(family: str, feats: dict) -> str:
    """Route to the correct family-specific classifier."""
    if family == 'serve':
        stroke = classify_serve(feats)
    elif family == 'overhead':
        stroke = classify_overhead(feats)
    elif family == 'underhand':
        stroke = classify_underhand(feats)
    elif family == 'net':
        stroke = classify_net(feats)
    elif family == 'defensive_block':
        stroke = classify_mid_height(feats)
    else:
        stroke = classify_mid_height(feats)

    # Map internal family fallbacks to user-facing types
    if stroke == 'mid_height_unknown':
        stroke = 'drive'
    elif stroke == 'overhead_unknown':
        stroke = 'clear'
    elif stroke == 'underhand_unknown':
        stroke = 'lift'
    elif stroke == 'net_unknown':
        stroke = 'net_shot'

    return stroke


# ── Confidence, evidence, top-3 ────────────────────────────────────

# Family → plausible strokes for top-3 generation
_FAMILY_STROKES = {
    'overhead': ['smash', 'drop', 'clear', 'overhead_unknown'],
    'underhand': ['lift', 'defensive_lift', 'soft_lift_or_push', 'underhand_unknown'],
    'net': ['net_shot', 'net_lift', 'net_kill', 'push', 'net_unknown'],
    'mid_height': ['drive', 'block', 'push', 'mid_height_unknown'],
    'defensive_block': ['block', 'drive', 'mid_height_unknown'],
    'serve': ['short_serve', 'long_serve', 'serve_unknown'],
}


def _evidence_consistent(stroke: str, feats: dict, evidence: dict) -> bool:
    """Check if evidence supports the predicted stroke type."""
    landing = evidence.get('landing_zone', '')
    traj = evidence.get('outgoing_trajectory', '')
    contact = evidence.get('contact_height', '')
    zone = evidence.get('player_zone', '')

    signatures = {
        'smash': lambda: traj == 'descending' and landing in ('mid court', 'deep (rear court)'),
        'clear': lambda: traj == 'ascending' and landing == 'deep (rear court)',
        'drop': lambda: traj == 'descending' and landing == 'short (front court)',
        'drive': lambda: traj in ('flat', 'descending') and landing in ('mid court', 'short (front court)'),
        'lift': lambda: traj == 'ascending' and contact == 'below waist',
        'defensive_lift': lambda: traj == 'ascending' and landing == 'deep (rear court)',
        'net_shot': lambda: contact == 'below waist' and zone in ('front court', 'mid court'),
        'block': lambda: traj in ('flat', 'descending') and landing in ('short (front court)', 'mid court'),
        'push': lambda: traj in ('flat', 'ascending') and landing == 'mid court',
        'soft_lift_or_push': lambda: traj == 'ascending' and landing == 'mid court',
        'short_serve': lambda: landing == 'short (front court)',
        'cross_court': lambda: 'rear' in landing or 'mid' in landing,
    }
    checker = signatures.get(stroke)
    return checker() if checker else True


def estimate_confidence(stroke: str, feats: dict) -> float:
    """Estimate confidence 0-1 based on feature margins and evidence consistency."""
    if not feats.get('usable', False):
        return 0.10

    max_speed = feats['max_speed']
    n_valid = feats.get('n_valid_frames', 0)

    # Base confidence from data quality
    if n_valid < 5:
        base = 0.30
    elif n_valid < 10:
        base = 0.50
    else:
        base = 0.70

    # Boost for high-speed strokes (cleaner signal)
    if stroke in ('smash', 'drive') and max_speed > _SMASH_SPEED:
        base += 0.15
    # Penalty for low-speed strokes (more ambiguous)
    if stroke in ('net_shot', 'block', 'push') and max_speed < _NET_SHOT_SPEED * 2:
        base -= 0.05

    # Evidence consistency: if evidence contradicts the stroke, penalize
    evidence = _build_evidence(stroke, feats)
    if not _evidence_consistent(stroke, feats, evidence):
        base -= 0.20

    return min(0.85, max(0.10, base))


def _build_evidence(stroke: str, feats: dict) -> dict:
    """Build structured evidence dict per Spec §11."""
    contact_x = feats['contact_x']
    player_x = feats['player_x']
    outgoing_dy = feats['outgoing_dy']
    landing_x = feats['landing_x']

    # Contact height description
    elbow = feats.get('elbow_angle', 0)
    if elbow > 2.0:
        contact_h = 'above head (extended arm)'
    elif elbow > 1.0:
        contact_h = 'waist to shoulder'
    else:
        contact_h = 'below waist'

    # Player zone
    if player_x > _ZONE_FRONT:
        zone = 'front court'
    elif player_x < _ZONE_BACK:
        zone = 'back court'
    else:
        zone = 'mid court'

    # Outgoing trajectory
    if outgoing_dy > _DESCEND_THRESH:
        traj = 'descending'
    elif outgoing_dy < _ASCEND_THRESH:
        traj = 'ascending'
    else:
        traj = 'flat'

    # Landing
    if landing_x > _FRONT_X:
        landing = 'short (front court)'
    elif landing_x < _BACK_X:
        landing = 'deep (rear court)'
    else:
        landing = 'mid court'

    return {
        'contact_height': contact_h,
        'player_zone': zone,
        'outgoing_trajectory': traj,
        'landing_zone': landing,
    }


def top3_alternatives(feats: dict, chosen: str) -> list[dict]:
    """Return top-3 alternative stroke types with confidence."""
    # Determine family from features
    family = classify_family(feats)
    candidates = _FAMILY_STROKES.get(family, ['unknown'])

    scored = []
    for stroke in candidates:
        if stroke == chosen:
            continue
        c = estimate_confidence(stroke, feats) * 0.5
        scored.append({'stroke': stroke, 'confidence': round(c, 3)})

    scored.sort(key=lambda x: x['confidence'], reverse=True)
    return scored[:3]
