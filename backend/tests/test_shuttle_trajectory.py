import numpy as np
import pandas as pd

from app.config.settings import settings
from app.pipeline.shared.shuttle_utils import clean_trajectory


def _make_df(x, y, conf=0.95):
    n = len(x)
    conf_arr = np.full(n, conf, dtype=np.float32) if np.isscalar(conf) else np.array(conf, dtype=np.float32)
    return pd.DataFrame({
        "frame": np.arange(n),
        "x": np.array(x, dtype=np.float32),
        "y": np.array(y, dtype=np.float32),
        "confidence": conf_arr,
    })


def test_confidence_gate_marks_low_conf_points():
    df = _make_df([0, 10, 20, 30], [0, 10, 20, 30],
                  conf=[0.95, 0.10, 0.95, 0.95])
    cleaned = clean_trajectory(df, settings)
    # Low-conf point at idx 1 gets gated then interpolated (1-frame gap)
    assert cleaned["was_interpolated"].iloc[1] == True
    assert cleaned["was_interpolated"].iloc[0] == False
    assert cleaned["was_interpolated"].iloc[2] == False


def test_spike_removal_removes_there_and_back():
    x = [0, 300, 5, 10, 15]
    y = [0, 0, 0, 0, 0]
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    # Spike at idx 1 is removed (set to NaN) then interpolated (1-frame gap)
    assert cleaned["was_interpolated"].iloc[1] == True, "Spike at idx 1 should be interpolated"
    assert cleaned["was_interpolated"].iloc[0] == False, "First point preserved"
    assert cleaned["was_interpolated"].iloc[2] == False, "Returned-to point preserved"


def test_sustained_fast_move_preserved():
    """A 250px/frame sustained smash is NOT a there-and-back spike."""
    x = [0, 250, 500, 750, 1000]
    y = [0, 0, 0, 0, 0]
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    for i in range(len(x)):
        assert not np.isnan(cleaned["x"].iloc[i]), (
            f"Sustained fast move at idx {i} should survive"
        )
        assert not np.isnan(cleaned["y"].iloc[i])


def test_consecutive_spikes_removed():
    x = [0, 300, 280, 5, 10, 15]
    y = [0, 0, 0, 0, 0, 0]
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    # Both spikes removed then interpolated (2-frame gap, ≤7 frames)
    assert cleaned["was_interpolated"].iloc[1] == True
    assert cleaned["was_interpolated"].iloc[2] == True
    assert cleaned["was_interpolated"].iloc[3] == False


def test_gap_interpolation_fills_short_gap():
    x = [0, np.nan, np.nan, np.nan, 40]
    y = [0, np.nan, np.nan, np.nan, 40]
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    assert not np.isnan(cleaned["x"].iloc[2]), "Gap of 3 should be interpolated"
    assert cleaned["was_interpolated"].iloc[1] == True
    assert cleaned["was_interpolated"].iloc[2] == True
    # Endpoints were valid, not interpolated
    assert cleaned["was_interpolated"].iloc[0] == False
    assert cleaned["was_interpolated"].iloc[4] == False


def test_long_gap_not_interpolated():
    """Gaps > shuttle_max_interp_gap (7) stay NaN."""
    n = settings.shuttle_max_interp_gap + 2  # 9 NaN between two valid points
    x = [0.0] + [np.nan] * n + [100.0]
    y = [0.0] + [np.nan] * n + [100.0]
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    # First 7 NaN frames are filled, last 2 stay NaN
    assert not np.isnan(cleaned["x"].iloc[settings.shuttle_max_interp_gap])
    assert np.isnan(cleaned["x"].iloc[settings.shuttle_max_interp_gap + 1])


def test_was_interpolated_provenance():
    x = [0, np.nan, np.nan, 30, np.nan, np.nan, 60]
    y = [0, np.nan, np.nan, 30, np.nan, np.nan, 60]
    df = _make_df(x, y, conf=[0.95] * 7)
    cleaned = clean_trajectory(df, settings)
    # Indices 1,2: short gap filled
    assert cleaned["was_interpolated"].iloc[1] == True
    assert cleaned["was_interpolated"].iloc[2] == True
    # Indices 4,5: gap > 7 may or may not be filled depending on gap length
    # (gap from 30→60 over indices 4,5 = 3 frames, <=7 → filled)
    assert cleaned["was_interpolated"].iloc[4] == True
    assert cleaned["was_interpolated"].iloc[5] == True
    # Original valid points not interpolated
    assert cleaned["was_interpolated"].iloc[0] == False
    assert cleaned["was_interpolated"].iloc[3] == False
    assert cleaned["was_interpolated"].iloc[6] == False


def test_smoothing_preserves_nan_gaps():
    """Gap > max_interp_gap (7) stays NaN after interpolation + smoothing."""
    n_gap = settings.shuttle_max_interp_gap + 2  # 9 NaN between valid points
    x = [0.0, 5.0] + [np.nan] * n_gap + [25.0, 30.0]
    y = [0.0, 5.0] + [np.nan] * n_gap + [25.0, 30.0]
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    # Indices 2..2+max_interp_gap-1 = 7 filled indices, rest stay NaN
    last_filled = 2 + settings.shuttle_max_interp_gap - 1
    first_nan = 2 + settings.shuttle_max_interp_gap
    assert not np.isnan(cleaned["x"].iloc[last_filled]), f"Idx {last_filled} should be filled"
    assert np.isnan(cleaned["x"].iloc[first_nan]), f"Idx {first_nan} should stay NaN"


def test_clean_trajectory_preserves_normal_data():
    x = list(range(20))
    y = list(range(20))
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    for i in range(len(df)):
        assert not np.isnan(cleaned["x"].iloc[i])
        assert not np.isnan(cleaned["y"].iloc[i])
        assert cleaned["was_interpolated"].iloc[i] == False


def test_empty_dataframe():
    df = pd.DataFrame(columns=["frame", "x", "y", "confidence"])
    cleaned = clean_trajectory(df, settings)
    assert len(cleaned) == 0


def test_all_valid_no_cleaning():
    x = list(range(20))
    y = list(range(20))
    df = _make_df(x, y)
    cleaned = clean_trajectory(df, settings)
    for i in range(len(df)):
        assert not np.isnan(cleaned["x"].iloc[i])
        assert not np.isnan(cleaned["y"].iloc[i])
        assert cleaned["was_interpolated"].iloc[i] == False
