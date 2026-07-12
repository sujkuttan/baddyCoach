import inspect
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn


def _synthetic_court_frame(corners):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for a, b in [(corners[0], corners[1]), (corners[2], corners[3]), (corners[0], corners[2]), (corners[1], corners[3])]:
        cv2.line(frame, tuple(a), tuple(b), (255, 255, 255), 5, cv2.LINE_AA)
    return frame


def test_colab_pipeline_accepts_manual_court_corners():
    import colab.pipeline as pipeline

    assert "court_corners" in inspect.signature(pipeline.run_pipeline).parameters
    corners = pipeline._parse_court_corners_arg("100,500,1180,500,250,150,1030,150")
    assert corners == [(100, 500), (1180, 500), (250, 150), (1030, 150)]


def test_colab_court_detection_uses_shared_hough_trapezoid_detector():
    import colab.pipeline as pipeline
    from app.pipeline.shared.court import compute_homography

    frame = _synthetic_court_frame([[160, 650], [1120, 650], [390, 170], [890, 170]])

    corners = pipeline.detect_court_from_frame(frame)

    assert corners is not None
    H, valid = compute_homography(corners)
    assert H is not None
    assert valid is True


def test_colab_tracknet_uses_masked_inpaintnet_repair_api():
    """Colab TrackNet must use the same coords+mask InpaintNet repair contract."""
    import colab.pipeline as pipeline

    class ConstantRepairNet(nn.Module):
        def forward(self, coords, mask):
            assert coords.shape == (1, 3, 2)
            assert torch.equal(mask[0, :, 0], torch.tensor([0.0, 1.0, 0.0]))
            return torch.tensor([[[0.1, 0.2], [0.5, 0.25], [0.9, 0.8]]])

    tracker = pipeline.TrackNetV3(model_path="missing.pt", device="cpu")
    tracker.inpaintnet = ConstantRepairNet()
    repaired = tracker._rectify_trajectory(
        [(10.0, 20.0, 0.9), None, (90.0, 80.0, 0.8)], 100, 100
    )

    assert repaired == [(10.0, 20.0, 0.9), (50.0, 25.0, 0.0), (90.0, 80.0, 0.8)]


def test_colab_delegates_court_space_enrichment_to_backend_helper():
    source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

    assert "from app.pipeline.shuttle import _add_court_space_columns" in source
    assert "_add_court_space_columns(shuttle_df, np.array(court[\"homography\"]), float(video_fps))" in source


def test_colab_exports_bst_input_quality_debug_artifact():
    source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

    assert '"debug_bst_input_quality"' in source


def test_colab_pose_fallback_interpolates_same_side_bboxes():
    import colab.pipeline as pipeline

    bbox = pipeline._interpolate_pose_bbox(
        1,
        "near",
        [0, 1, 2],
        {
            0: [{"side": "near", "bbox": [0.0, 0.0, 10.0, 10.0]}],
            2: [{"side": "near", "bbox": [10.0, 10.0, 20.0, 20.0]}],
        },
    )

    assert bbox == [5.0, 5.0, 15.0, 15.0]


def test_colab_pose_fallback_uses_same_side_detection_from_previous_batch():
    import colab.pipeline as pipeline

    bbox = pipeline._interpolate_pose_bbox(
        100,
        "near",
        [100, 101],
        {99: [{"side": "near", "bbox": [10.0, 20.0, 30.0, 40.0]}]},
    )

    assert bbox == [10.0, 20.0, 30.0, 40.0]


def test_colab_pose_fallback_does_not_cross_player_sides():
    import colab.pipeline as pipeline

    bbox = pipeline._interpolate_pose_bbox(
        100,
        "near",
        [100, 101],
        {99: [{"side": "far", "bbox": [10.0, 20.0, 30.0, 40.0]}]},
    )

    assert bbox is None


def test_colab_preserves_aim_alpha_quality_fields_in_outputs():
    source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

    assert '"aim_alpha_reliable"' in source
    assert '"aim_alpha_route"' in source
