import numpy as np
import pytest

from app.pipeline.shared.bst_validator import (
    BSTInputValidator, ValidationResult, ValidationError,
)


def _make_good_clip(seq_len=100):
    """Create a clip that should pass all checks.

    Builds anatomically plausible COCO-17 joints in bbox-relative
    center-aligned space: joint y increases from head (≈ -0.3) down
    to ankles (≈ 0.3), and shuttle/pos in [0,1] range.
    """
    T = seq_len
    # Joints shape: (T, 2, 72) = (T, 2 players, 17 joints×2 + 19 bones×2)
    joints = np.zeros((T, 2, 72), dtype=np.float32)
    for p in range(2):
        for t in range(T):
            # Build COCO-17 y coords increasing from head to feet.
            # Anatomical ordering: eyes (idx 1,2) above nose (idx 0), so
            # swap the first three linspace values to put L_eye/R_eye highest.
            kp_y = np.linspace(-0.3, 0.3, 17)
            # Swap: nose(0) gets largest y (lowest), L_eye(1) gets smallest y (highest),
            # R_eye(2) gets middle → both eyes above nose anatomically.
            kp_y[0], kp_y[1], kp_y[2] = kp_y[2], kp_y[0], kp_y[1]
            kp_x = np.random.uniform(-0.1, 0.1, 17)
            kp = np.stack([kp_x, kp_y], axis=1)  # (17, 2)
            joints[t, p, :34] = kp.ravel()
            # Bones (non-zero, plausible magnitude)
            bones = np.random.uniform(-0.05, 0.05, (19, 2)).astype(np.float32)
            joints[t, p, 34:] = bones.ravel()

    shuttle = np.zeros((T, 2), dtype=np.float32)
    mid = T // 2
    # Shuttle starts at contact (frame 0) at mid-trajectory y.
    # Use linear trajectory from 0.3→0.7 but offset frame 0 to 0.5 so it's interior.
    y_vals = np.linspace(0.3, 0.7, T)
    y_vals[0] = 0.5
    for t in range(T):
        frac = t / max(T - 1, 1)
        shuttle[t, 0] = 0.5
        shuttle[t, 1] = y_vals[t]

    # Player positions: p0 (far) has smaller court-x (depth) than p1 (near).
    # Court-x = pos[..., 0], where x=0 is far end, x=1 is near end.
    pos = np.zeros((T, 2, 2), dtype=np.float32)
    pos[:, 0, 0] = 0.2  # far player depth (close to x=0 / far end)
    pos[:, 1, 0] = 0.8  # near player depth (close to x=1 / near end)

    return {
        'JnB': joints,
        'shuttle': shuttle,
        'pos': pos,
        'video_len': T,
        'vid_w': 1920,
        'vid_h': 1080,
        'court_length': 13.4,
        'court_width': 6.1,
        '_debug_clip': {
            'frame_start': 0,
            'frame_end': T - 1,
        },
    }


def test_validator_default_construction():
    v = BSTInputValidator()
    assert v.seq_len == 100
    assert v.level == "warn"
    assert v.center_align is True


def test_validator_good_clip_passes():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    result = v.validate_clip(clip)
    assert result.passed, f"expected all checks to pass, got {result}"
    assert result.n_errors == 0
    assert result.n_warnings == 0


def test_validator_player_order_reversed():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Swap player order on the depth axis: p0 becomes near (larger x), p1 becomes far (smaller x)
    clip['pos'][:, 0, 0] = 0.8  # p0 has larger x → near player (should be far)
    clip['pos'][:, 1, 0] = 0.2  # p1 has smaller x → far player (should be near)
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("REVERSED" in w for w in result.warnings)


def test_validator_seq_len_mismatch():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip(seq_len=50)
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("sequence length mismatch" in e.lower() for e in result.errors)


def test_validator_shuttle_out_of_range():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    clip['shuttle'][5, 0] = 2.5
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("outside" in w and "x range" in w for w in result.warnings)


def test_validator_joints_out_of_range():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Set a joint value well outside [-0.6, 0.6]
    clip['JnB'][0, 0, 0] = 5.0
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("exceed" in w for w in result.warnings)


def test_validator_no_center_align():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Shift all joints to create mean far from 0
    clip['JnB'][:, :, :34] += 0.5
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("center" in w.lower() for w in result.warnings)


def test_validator_player_pos_out_of_range():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    clip['pos'][3, 0, 0] = -0.5
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("outside" in w for w in result.warnings)


def test_validator_missing_joints():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Zero out player 0 joints entirely
    clip['JnB'][:, 0, :34] = 0.0
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("all-zero" in w for w in result.warnings)


def test_validator_hit_frame_at_extreme():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Put shuttle at extreme y at frame 0
    clip['shuttle'][0, 1] = 0.98
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("extreme" in w for w in result.warnings)


def test_validator_bone_edges_all_zero():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Zero out the bone portion
    clip['JnB'][:, :, 34:] = 0.0
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("Bone" in w or "zero" in w.lower() for w in result.warnings)


def test_validator_wrong_bone_dim():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    clip['JnB'] = np.zeros((100, 2, 50), dtype=np.float32)
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("dimension mismatch" in e for e in result.errors)


def test_validator_level_off_skips_all():
    v = BSTInputValidator(seq_len=100, level="off")
    clip = _make_good_clip()
    clip['JnB'] = np.zeros((50, 2, 72), dtype=np.float32)  # wrong seq_len
    result = v.validate_clip(clip)
    assert result.passed
    assert result.n_checks == 0


def test_validator_level_error_raises():
    v = BSTInputValidator(seq_len=100, level="error")
    clip = _make_good_clip(seq_len=50)
    with pytest.raises(ValidationError):
        v.validate_clip(clip)


def test_validator_batch_passes():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    JnB = np.stack([clip['JnB']])
    shuttle = np.stack([clip['shuttle']])
    pos = np.stack([clip['pos']])
    result = v.validate_batch(JnB, shuttle, pos)
    assert result.passed


def test_validator_batch_nan_detected():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    JnB = np.stack([clip['JnB']])
    shuttle = np.stack([clip['shuttle']])
    pos = np.stack([clip['pos']])
    JnB[0, 0, 0] = np.nan
    result = v.validate_batch(JnB, shuttle, pos)
    assert not result.passed
    assert any("NaN" in w for w in result.warnings)


def test_validator_batch_wrong_seq_len():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip(seq_len=30)
    JnB = np.stack([clip['JnB']])
    result = v.validate_batch(
        JnB,
        np.stack([clip['shuttle']]),
        np.stack([clip['pos']]),
    )
    assert not result.passed
    assert any("sequence length" in e.lower() for e in result.errors)


def test_validator_batch_wrong_dtype():
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    JnB = np.stack([clip['JnB']]).astype(np.float64)
    shuttle = np.stack([clip['shuttle']])
    pos = np.stack([clip['pos']])
    result = v.validate_batch(JnB, shuttle, pos)
    assert not result.passed
    assert any("dtype" in w for w in result.warnings)


def test_validator_anatomy_violation_detected():
    """Joints with inverted y ordering should trigger joint order warning."""
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Invert y ordering for first few frames: make L_ankle above L_knee
    # Joint index 13 = L_knee, 15 = L_ankle; in the flattened array:
    # indices 13*2+1=27 (L_knee y), 15*2+1=31 (L_ankle y)
    # Swap them: make ankle_y < knee_y (wrong — should be higher)
    for p in range(2):
        for t in range(5):
            clip['JnB'][t, p, 31] = clip['JnB'][t, p, 27] - 0.1
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("anatomical" in w or "violated" in w for w in result.warnings)


def test_validator_merge_combines_results():
    r1 = ValidationResult(n_checks=2, n_passed=2)
    r2 = ValidationResult(
        n_checks=1, n_errors=1, n_warnings=1,
        errors=["bad data"],
        warnings=["bad warning"],
    )
    merged = r1.merge(r2)
    assert merged.n_checks == 3
    assert merged.n_passed == 2
    assert merged.n_errors == 1
    assert merged.n_warnings == 1
    assert not merged.passed


def test_validator_serialized_has_all_src_locations():
    """Verify all 10 checks produce warnings with [...] source loc."""
    v = BSTInputValidator(seq_len=100)
    clip = _make_good_clip()
    # Break everything simultaneously
    clip['pos'][:, 0, 0] = 0.9  # player order reversed on depth axis
    clip['JnB'] = np.zeros((50, 2, 72), dtype=np.float32)  # wrong seq_len
    clip['shuttle'][:] = 5.0  # way out of range
    clip['pos'][:] = -1.0  # out of range
    result = v.validate_clip(clip)
    # Every warning/error should contain a [file:line] reference
    for msg in result.warnings + result.errors:
        assert "[" in msg and "]" in msg, f"Missing source location: {msg}"


# ── Clip boundary mode tests ─────────────────────────────────────────

def test_validator_midpoint_mode_in_flight_passes():
    """Midpoint mode: shuttle at frame 0 should be in-flight (not at extreme or center)."""
    v = BSTInputValidator(seq_len=100, clip_boundary="midpoint")
    clip = _make_good_clip()
    # Frame 0 shuttle at y=0.3 in a [0.1, 0.9] trajectory → y_frac=0.25 (in-flight)
    T = len(clip['shuttle'])
    clip['shuttle'][:, 1] = np.linspace(0.1, 0.9, T)
    clip['shuttle'][0, 1] = 0.3
    result = v.validate_clip(clip)
    assert result.passed


def test_validator_midpoint_detects_hit_start_leak():
    """Midpoint mode: frame-0 shuttle centered in broad trajectory → "
    "suspicious (looks like hit_start convention leaking through)."""
    v = BSTInputValidator(seq_len=100, clip_boundary="midpoint")
    clip = _make_good_clip()
    # Build a broad trajectory [0.1, 0.9] with frame 0 at center (0.5)
    T = len(clip['shuttle'])
    clip['shuttle'][:, 1] = np.linspace(0.1, 0.9, T)
    clip['shuttle'][0, 1] = 0.5
    result = v.validate_clip(clip)
    # y_frac = (0.5 - 0.1) / (0.9 - 0.1) = 0.5 → at_contact range
    assert not result.passed
    assert any("centered" in w or "contact" in w for w in result.warnings)


def test_validator_midpoint_detects_grounded_shuttle():
    """Midpoint mode: shuttle at frame 0 at y extreme = warning."""
    v = BSTInputValidator(seq_len=100, clip_boundary="midpoint")
    clip = _make_good_clip()
    # Frame 0 shuttle very close to the ground (y=0.0, extreme low)
    clip['shuttle'][0, 1] = 0.0
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("extreme" in w for w in result.warnings)


def test_validator_midpoint_midframe_contact_check():
    """Midpoint mode: mid-frame shuttle should be near trajectory middle."""
    v = BSTInputValidator(seq_len=100, clip_boundary="midpoint")
    clip = _make_good_clip()
    # Make the mid-frame shuttle at trajectory extreme
    mid = len(clip['shuttle']) // 2
    clip['shuttle'][mid, 1] = clip['shuttle'][:, 1].max()  # at max extreme
    result = v.validate_clip(clip)
    assert not result.passed
    assert any("mid-frame" in w for w in result.warnings)


def test_validator_clip_boundary_constructor():
    """clip_boundary defaults to hit_start."""
    v = BSTInputValidator()
    assert v.clip_boundary == "hit_start"
    v2 = BSTInputValidator(clip_boundary="midpoint")
    assert v2.clip_boundary == "midpoint"
