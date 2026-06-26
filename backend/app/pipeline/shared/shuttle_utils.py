import numpy as np
import pandas as pd


def clean_trajectory(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Clean shuttle trajectory in-place: gate → spike reject → gap fill → smooth.

    Modifies x, y columns in-place and adds a was_interpolated column.
    """
    x = df["x"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)
    conf = df["confidence"].values.astype(np.float64)

    max_jump = settings.shuttle_max_jump_px
    max_interp = settings.shuttle_max_interp_gap
    smooth_win = settings.shuttle_smooth_window
    min_conf = settings.shuttle_clean_min_conf

    # 1. Confidence gate — mark low-confidence detections as missing
    low_conf = conf < min_conf
    x[low_conf] = np.nan
    y[low_conf] = np.nan

    # 2. Physical-velocity outlier reject (there-and-back spike check)
    # Walk clean trajectory points; when a point jumps from the last good
    # trajectory point, search ahead for a return point within max_jump.
    # If found (and direction reverses), all intermediate points are spikes.
    valid = ~np.isnan(x) & ~np.isnan(y)
    valid_idx = np.where(valid)[0]
    spike = np.zeros(len(x), dtype=bool)

    i = 1
    while i < len(valid_idx):
        last_good = valid_idx[i - 1]
        # Skip already-spiked intermediate trajectory points
        for k in range(i - 1, -1, -1):
            if not spike[valid_idx[k]]:
                last_good = valid_idx[k]
                break

        curr_i = valid_idx[i]
        dx1 = x[curr_i] - x[last_good]
        dy1 = y[curr_i] - y[last_good]
        d1 = np.sqrt(dx1 * dx1 + dy1 * dy1)

        if d1 <= max_jump:
            i += 1
            continue

        # Search ahead for a return to the original trajectory
        found_return = False
        for j in range(i + 1, len(valid_idx)):
            next_j = valid_idx[j]
            d_return = np.sqrt(
                (x[next_j] - x[last_good]) ** 2 + (y[next_j] - y[last_good]) ** 2
            )
            d2 = np.sqrt(
                (x[next_j] - x[curr_i]) ** 2 + (y[next_j] - y[curr_i]) ** 2
            )

            if d_return < max_jump and d2 > max_jump * 0.3:
                # next_j is back near the trajectory — check direction reversal
                dot = dx1 * (x[next_j] - x[curr_i]) + dy1 * (y[next_j] - y[curr_i])
                if dot < 0:
                    for k in range(i, j):
                        spike[valid_idx[k]] = True
                    i = j + 1
                    found_return = True
                break
            elif d_return > d1 * 1.5:
                # Sustained move away from trajectory — not a spike
                break

        if not found_return:
            i += 1

    x[spike] = np.nan
    y[spike] = np.nan

    # Save pre-interp state for provenance
    x_before_interp = x.copy()

    # 3. Gap interpolation (linear, up to max_interp frames)
    if max_interp > 0:
        x_filled = pd.Series(x).interpolate(method="linear", limit=max_interp).values
        y_filled = pd.Series(y).interpolate(method="linear", limit=max_interp).values
    else:
        x_filled = x.copy()
        y_filled = y.copy()

    was_interpolated = np.isnan(x_before_interp) & ~np.isnan(x_filled)

    # Bump confidence for interpolated frames so they pass the BST
    # confidence gate (strokes.py:136) and reach the model's shuttle channel.
    if was_interpolated.any():
        sentinel_conf = max(min_conf, settings.shuttle_min_conf + 0.05)
        conf_series = df["confidence"].values.astype(np.float64)
        conf_series[was_interpolated] = np.maximum(
            conf_series[was_interpolated], sentinel_conf
        )
        df["confidence"] = conf_series

    x[:] = x_filled
    y[:] = y_filled

    # 4. Light smoothing (moving median, preserve NaN gaps)
    if smooth_win >= 3:
        nan_x = np.isnan(x)
        nan_y = np.isnan(y)
        if nan_x.any():
            x_smooth = (
                pd.Series(np.where(nan_x, np.nan, x))
                .rolling(window=smooth_win, center=True, min_periods=1)
                .median()
                .values
            )
            x[:] = np.where(nan_x, np.nan, x_smooth)
        if nan_y.any():
            y_smooth = (
                pd.Series(np.where(nan_y, np.nan, y))
                .rolling(window=smooth_win, center=True, min_periods=1)
                .median()
                .values
            )
            y[:] = np.where(nan_y, np.nan, y_smooth)

    df["x"] = x
    df["y"] = y
    df["was_interpolated"] = was_interpolated
    return df
