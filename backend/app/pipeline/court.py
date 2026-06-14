import numpy as np
import cv2
from pathlib import Path

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class CourtDetectionStage:
    name = "court_detection"
    input_keys = []
    output_keys = ["court"]

    # Standard badminton court dimensions in meters (singles)
    COURT_LENGTH = 13.4
    COURT_WIDTH = 5.18
    NET_HEIGHT = 1.55

    def run(self, artifacts: ArtifactStore, config: StageConfig, corners: list[tuple[int, int]] | None = None) -> StageResult:
        if corners is None or len(corners) != 4:
            return StageResult.from_error("Court corners are required (4 points). Provide via manual calibration.")

        src_points = np.array(corners, dtype=np.float32)

        dst_points = np.array([
            [0, 0],
            [self.COURT_WIDTH, 0],
            [0, self.COURT_LENGTH],
            [self.COURT_WIDTH, self.COURT_LENGTH],
        ], dtype=np.float32)

        homography, _ = cv2.findHomography(src_points, dst_points)

        if homography is None:
            return StageResult.from_error("Failed to compute homography matrix")

        court_data = {
            "homography": homography.tolist(),
            "corners_pixel": [list(c) for c in corners],
            "court_length": self.COURT_LENGTH,
            "court_width": self.COURT_WIDTH,
            "net_height": self.NET_HEIGHT,
        }

        artifacts.set("court", court_data)

        return StageResult.success(
            artifacts={"court": artifacts.path("court")},
            metadata={"homography_computed": True}
        )
