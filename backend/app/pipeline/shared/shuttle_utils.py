import numpy as np
import pandas as pd


class ShuttleSmoother:
    """Clean noisy TrackNetV3 shuttle tracks, fill short gaps, and
    compute derived features (velocity, acceleration, curvature).

    Preserves abrupt changes that could indicate hit contact and marks
    interpolated frames so they can be down-weighted downstream.
    """

    def __init__(self, settings):
        self.max_jump = settings.shuttle_max_jump_px
        self.max_interp = settings.shuttle_max_interp_gap
        self.smooth_win = settings.shuttle_smooth_window
        self.min_conf = settings.shuttle_clean_min_conf
        self.smooth_method = getattr(settings, 'shuttle_smooth_method', 'median')

    # ── Core cleaning ──────────────────────────────────────────────

    def _confidence_gate(self, x, y, conf):
        """Mark low-confidence frames as missing (NaN)."""
        low = conf < self.min_conf
        x[low] = np.nan
        y[low] = np.nan
        return x, y

    def _reject_spikes(self, x, y):
        """There-and-back spike removal.

        When a point jumps beyond max_jump from the last good trajectory
        point, search ahead for a return.  If direction reverses and the
        return point is back near the original trajectory, the intermediate
        points are considered spikes and set to NaN.
        """
        valid = ~np.isnan(x) & ~np.isnan(y)
        valid_idx = np.where(valid)[0]
        spike = np.zeros(len(x), dtype=bool)

        i = 1
        while i < len(valid_idx):
            last_good = valid_idx[i - 1]
            for k in range(i - 1, -1, -1):
                if not spike[valid_idx[k]]:
                    last_good = valid_idx[k]
                    break

            curr_i = valid_idx[i]
            dx1 = x[curr_i] - x[last_good]
            dy1 = y[curr_i] - y[last_good]
            d1 = np.sqrt(dx1 * dx1 + dy1 * dy1)

            if d1 <= self.max_jump:
                i += 1
                continue

            found_return = False
            for j in range(i + 1, len(valid_idx)):
                next_j = valid_idx[j]
                d_return = np.sqrt(
                    (x[next_j] - x[last_good]) ** 2 + (y[next_j] - y[last_good]) ** 2
                )
                d2 = np.sqrt(
                    (x[next_j] - x[curr_i]) ** 2 + (y[next_j] - y[curr_i]) ** 2
                )

                if d_return < self.max_jump and d2 > self.max_jump * 0.3:
                    dot = dx1 * (x[next_j] - x[curr_i]) + dy1 * (y[next_j] - y[curr_i])
                    if dot < 0:
                        for k in range(i, j):
                            spike[valid_idx[k]] = True
                        i = j + 1
                        found_return = True
                    break
                elif d_return > d1 * 1.5:
                    break

            if not found_return:
                i += 1

        x[spike] = np.nan
        y[spike] = np.nan
        return x, y

    def smooth(self, shuttle_track: pd.DataFrame) -> pd.DataFrame:
        """Clean shuttle trajectory: confidence gate → spike reject → smooth.

        Returns a copy with cleaned ``x``, ``y`` and ``was_interpolated``.
        Interpolated frames get a bumped confidence sentinel so they pass
        downstream confidence gates.
        """
        df = shuttle_track.copy()
        x = df["x"].values.astype(np.float64)
        y = df["y"].values.astype(np.float64)
        conf = df["confidence"].values.astype(np.float64)

        self._confidence_gate(x, y, conf)
        self._reject_spikes(x, y)
        x_before_interp = x.copy()
        df = self.interpolate_missing(df, x, y, x_before_interp, conf)
        x = df["x"].values.astype(np.float64)
        y = df["y"].values.astype(np.float64)

        # Light smoothing
        if self.smooth_win >= 3:
            if self.smooth_method == 'savgol':
                try:
                    from scipy.signal import savgol_filter
                    nan_x = np.isnan(x)
                    nan_y = np.isnan(y)
                    if not nan_x.all():
                        x_filled = pd.Series(np.where(nan_x, np.nan, x)).interpolate(limit=2).values
                        x[:] = np.where(nan_x, np.nan,
                            savgol_filter(np.where(nan_x, x_filled, x),
                                          window_length=min(self.smooth_win, len(x) - 1 if len(x) % 2 == 0 else len(x)),
                                          polyorder=2, mode='nearest'))
                    if not nan_y.all():
                        y_filled = pd.Series(np.where(nan_y, np.nan, y)).interpolate(limit=2).values
                        y[:] = np.where(nan_y, np.nan,
                            savgol_filter(np.where(nan_y, y_filled, y),
                                          window_length=min(self.smooth_win, len(y) - 1 if len(y) % 2 == 0 else len(y)),
                                          polyorder=2, mode='nearest'))
                except ImportError:
                    pass  # fall through to rolling median

            # Default: moving median (preserves edges better than mean)
            if self.smooth_method != 'savgol' or 'savgol' not in dir():
                nan_x = np.isnan(x)
                nan_y = np.isnan(y)
                if nan_x.any():
                    x_smooth = (
                        pd.Series(np.where(nan_x, np.nan, x))
                        .rolling(window=self.smooth_win, center=True, min_periods=1)
                        .median().values
                    )
                    x[:] = np.where(nan_x, np.nan, x_smooth)
                if nan_y.any():
                    y_smooth = (
                        pd.Series(np.where(nan_y, np.nan, y))
                        .rolling(window=self.smooth_win, center=True, min_periods=1)
                        .median().values
                    )
                    y[:] = np.where(nan_y, np.nan, y_smooth)

        df["x"] = x
        df["y"] = y
        return df

    def interpolate_missing(self, shuttle_track: pd.DataFrame,
                            x: np.ndarray | None = None,
                            y: np.ndarray | None = None,
                            x_before_interp: np.ndarray | None = None,
                            conf: np.ndarray | None = None) -> pd.DataFrame:
        """Fill short gaps via linear interpolation (up to ``max_interp`` frames).

        Sets ``was_interpolated`` and bumps confidence on filled frames.
        """
        df = shuttle_track.copy() if x is None else shuttle_track
        if x is None:
            x = df["x"].values.astype(np.float64)
            y = df["y"].values.astype(np.float64)
        if x_before_interp is None:
            x_before_interp = x.copy()

        min_conf = self.min_conf
        if self.max_interp > 0:
            x_filled = pd.Series(x).interpolate(method="linear", limit=self.max_interp).values
            y_filled = pd.Series(y).interpolate(method="linear", limit=self.max_interp).values
        else:
            x_filled = x.copy()
            y_filled = y.copy()

        was_interpolated = np.isnan(x_before_interp) & ~np.isnan(x_filled)

        if was_interpolated.any() and conf is not None:
            from app.config.settings import settings
            sentinel_conf = max(min_conf, settings.shuttle_min_conf + 0.05)
            conf[was_interpolated] = np.maximum(conf[was_interpolated], sentinel_conf)

        if x is None or y is None or conf is None:
            df["x"] = x_filled
            df["y"] = y_filled
        df["was_interpolated"] = was_interpolated
        return df

    # ── Derived features ───────────────────────────────────────────

    def compute_velocity(self, shuttle_track: pd.DataFrame) -> np.ndarray:
        """Per-frame speed (units/frame) from consecutive valid positions.

        NaN at the first frame and any frame where either endpoint is missing
        or interpolated (when ``was_interpolated`` column exists).
        """
        x = shuttle_track["x"].values.astype(np.float64)
        y = shuttle_track["y"].values.astype(np.float64)
        vel = np.full(len(x), np.nan, dtype=np.float64)

        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.sqrt(dx * dx + dy * dy)
        valid = ~(np.isnan(x[:-1]) | np.isnan(x[1:]))
        vel[1:] = np.where(valid, speed, np.nan)

        if "was_interpolated" in shuttle_track.columns:
            interp = shuttle_track["was_interpolated"].values.astype(bool)
            for t in range(1, len(vel)):
                if interp[t - 1] or interp[t]:
                    vel[t] = np.nan

        return vel

    def compute_acceleration(self, shuttle_track: pd.DataFrame) -> np.ndarray:
        """Per-frame acceleration (units/frame²) as first difference of velocity."""
        vel = self.compute_velocity(shuttle_track)
        acc = np.full(len(vel), np.nan, dtype=np.float64)
        acc[1:] = np.diff(vel)
        return acc

    def compute_curvature(self, shuttle_track: pd.DataFrame) -> np.ndarray:
        """Per-frame Menger curvature from 3 consecutive points.

        Curvature = 4 * triangle_area / (|AB| * |BC| * |CA|).
        High values = sharp turns (potential hit contact).
        NaN when any of the 3 points is unavailable.
        """
        x = shuttle_track["x"].values.astype(np.float64)
        y = shuttle_track["y"].values.astype(np.float64)
        curv = np.full(len(x), np.nan, dtype=np.float64)

        for t in range(1, len(x) - 1):
            if np.isnan(x[t - 1]) or np.isnan(x[t]) or np.isnan(x[t + 1]):
                continue

            a = np.array([x[t - 1], y[t - 1]])
            b = np.array([x[t], y[t]])
            c = np.array([x[t + 1], y[t + 1]])

            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)

            if ab < 1e-6 or bc < 1e-6 or ca < 1e-6:
                continue

            # Triangle area via cross product
            area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
            curv[t] = 4.0 * area / (ab * bc * ca)

        return curv


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
