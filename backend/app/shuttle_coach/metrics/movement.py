import numpy as np
import pandas as pd

from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class RecoveryTime(Metric):
    metric_id = "movement.recovery_time"
    requires = {"movement"}

    def compute(self, m) -> list[MetricResult]:
        results = []
        fps = 30.0

        for pid in m.player_ids:
            pos = m.positions_of(pid)
            if pos.empty or "court_x" not in pos.columns:
                continue

            base_x = pos["court_x"].median()
            base_y = pos["court_y"].median()

            player_shots = m.shots_of(pid)
            recovery_times = []

            for _, shot in player_shots.iterrows():
                hit_frame = shot.get("hit_frame")
                if hit_frame is None:
                    continue
                hit_frame = int(hit_frame)

                after = pos[pos["frame"] > hit_frame]
                if after.empty:
                    continue

                for _, row in after.iterrows():
                    dx = row["court_x"] - base_x
                    dy = row["court_y"] - base_y
                    dist = float(np.sqrt(dx * dx + dy * dy))
                    if dist <= 1.0:
                        frames = row["frame"] - hit_frame
                        recovery_times.append(frames / fps)
                        break

            if recovery_times:
                results.append(MetricResult(
                    metric_id=self.metric_id,
                    player_id=pid,
                    value=float(np.mean(recovery_times)),
                    unit="seconds",
                    sample_size=len(recovery_times),
                    confidence=1.0,
                    context={"base_x": float(base_x), "base_y": float(base_y)},
                ))
        return results


@register
class CourtCoverage(Metric):
    metric_id = "movement.court_coverage"
    requires = {"movement"}

    ZONE_EDGES_X = [0.0, 4.0, 8.0, 13.4]
    ZONE_EDGES_Y = [0.0, 3.05, 6.10]
    ZONE_NAMES = ["rear_left", "rear_right", "mid_left", "mid_right", "front_left", "front_right"]

    def compute(self, m) -> list[MetricResult]:
        results = []

        for pid in m.player_ids:
            pos = m.positions_of(pid)
            if pos.empty or "court_x" not in pos.columns:
                continue

            cx = pos["court_x"].values
            cy = pos["court_y"].values

            x_bins = np.searchsorted(self.ZONE_EDGES_X[1:], cx, side="right")
            y_bins = np.searchsorted(self.ZONE_EDGES_Y[1:], cy, side="right")

            zone_indices = y_bins * 2 + x_bins
            counts = np.zeros(6, dtype=float)
            for zi in zone_indices:
                if 0 <= zi < 6:
                    counts[zi] += 1

            total = counts.sum()
            if total == 0:
                continue

            pcts = (counts / total * 100.0).round(2)
            zones = {name: float(pcts[i]) for i, name in enumerate(self.ZONE_NAMES)}

            results.append(MetricResult(
                metric_id=self.metric_id,
                player_id=pid,
                value=zones,
                unit="percent",
                sample_size=int(total),
                confidence=1.0,
                context={},
            ))
        return results


@register
class DistancePerRally(Metric):
    metric_id = "movement.distance_per_rally"
    requires = {"movement"}

    def compute(self, m) -> list[MetricResult]:
        results = []

        for pid in m.player_ids:
            pos = m.positions_of(pid)
            if pos.empty or "court_x" not in pos.columns:
                continue

            sorted_pos = pos.sort_values("frame")
            dx = sorted_pos["court_x"].diff().fillna(0.0)
            dy = sorted_pos["court_y"].diff().fillna(0.0)
            total_dist = float(np.sqrt(dx * dx + dy * dy).sum())

            results.append(MetricResult(
                metric_id=self.metric_id,
                player_id=pid,
                value=round(total_dist, 2),
                unit="meters",
                sample_size=len(sorted_pos),
                confidence=1.0,
                context={},
            ))
        return results
