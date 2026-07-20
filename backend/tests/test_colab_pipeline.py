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


def test_colab_resolves_manual_corners_from_output_dir_then_repo_root(tmp_path, monkeypatch):
    import json
    import colab.pipeline as pipeline

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output_path = out_dir / "report.json"

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "manual_corners.json").write_text(
        json.dumps({"corners": [[1, 2], [3, 4], [5, 6], [7, 8]]})
    )
    monkeypatch.setattr(pipeline, "_REPO_ROOT", repo_root, raising=False)

    corners = pipeline._resolve_manual_corners(str(output_path))
    assert corners == [(1, 2), (3, 4), (5, 6), (7, 8)]

    (out_dir / "manual_corners.json").write_text(
        json.dumps({"corners": [[10, 20], [30, 40], [50, 60], [70, 80]]})
    )
    corners = pipeline._resolve_manual_corners(str(output_path))
    assert corners == [(10, 20), (30, 40), (50, 60), (70, 80)]


def test_colab_falls_back_to_manual_corners_when_auto_invalid():
    """When auto-detection yields invalid geometry, colab uses the manual
    corners fallback instead of accepting the degenerate auto result."""
    import colab.pipeline as pipeline

    bad_auto = [(80, 500), (1800, 500), (120, 100), (1760, 100)]  # near-rectangular
    good_manual = [(148, 637), (1184, 641), (466, 77), (831, 76)]  # trapezoid

    corners, method, valid = pipeline._select_court_corners(
        auto_corners=bad_auto,
        manual_fallback=good_manual,
        vid_w=1920,
        vid_h=1080,
    )
    assert valid is True
    assert method == "manual_fallback"
    assert corners == good_manual

    corners, method, valid = pipeline._select_court_corners(
        auto_corners=[(100, 500), (1180, 500), (250, 150), (1030, 150)],
        manual_fallback=good_manual,
        vid_w=1920,
        vid_h=1080,
    )
    assert valid is True
    assert method != "manual_fallback"


def test_colab_hrnet_decode_applies_msra_subpixel_refinement():
    """Colab _decode_hrnet must apply the MMPose MSRAHeatmap sub-pixel
    refinement (move 0.25 index toward the higher neighbor) before rescaling,
    matching mmpose.codecs.utils.refinement.refine_keypoints."""
    import numpy as np
    import colab.pipeline as pipeline

    est = pipeline.RTMPoseEstimator.__new__(pipeline.RTMPoseEstimator)
    est.model_type = "hrnet"

    K, H, W = 17, 64, 48
    heatmap = np.zeros((K, H, W), dtype=np.float32)
    # Joint 0: peak at (x=10, y=20); higher neighbor on +x and +y side.
    heatmap[0, 20, 10] = 1.0
    heatmap[0, 20, 11] = 0.8
    heatmap[0, 20, 9] = 0.2
    heatmap[0, 21, 10] = 0.8
    heatmap[0, 19, 10] = 0.2

    crop_info = (0, 0, W, H)  # 1:1 crop -> index space == pixel space
    kps = est._decode_hrnet([heatmap[None]], crop_info)

    # Reference: index refined by +0.25 in x and +0.25 in y, then * (crop/W).
    exp_x = (10 + 0.25) / W * W
    exp_y = (20 + 0.25) / H * H
    assert abs(kps[0, 0] - exp_x) < 1e-4
    assert abs(kps[0, 1] - exp_y) < 1e-4
    assert abs(kps[0, 2] - 1.0) < 1e-6


def test_colab_hrnet_decode_no_refinement_at_border():
    """Border keypoints get no sub-pixel shift (matches reference guards)."""
    import numpy as np
    import colab.pipeline as pipeline

    est = pipeline.RTMPoseEstimator.__new__(pipeline.RTMPoseEstimator)
    est.model_type = "hrnet"

    K, H, W = 17, 64, 48
    heatmap = np.zeros((K, H, W), dtype=np.float32)
    heatmap[0, 0, 0] = 1.0  # top-left corner -> guards prevent refinement

    kps = est._decode_hrnet([heatmap[None]], (0, 0, W, H))
    assert abs(kps[0, 0] - 0.0) < 1e-4
    assert abs(kps[0, 1] - 0.0) < 1e-4


def test_colab_uses_continuity_aware_tracknet_candidate_selection():
    source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

    assert "_extract_component_candidates" in source
    assert "_select_detection_candidate" in source
    assert "tracknet_component_motion_weight" in source


def test_colab_reuses_backend_tracknet_crop_and_merge_helpers():
    source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

    assert "from app.models.tracknet import (" in source
    assert "_court_crop_rect" in source
    assert "_gate_tracknet_spikes" in source
    assert "_merge_far_tile_tracks" in source


def test_colab_exposes_racket_settings():
    """Colab must expose the 7 racket CLI flags and wire them to settings."""
    source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

    flags = [
        "--racket-enabled",
        "--racket-min-conf",
        "--racket-proximity-blend",
        "--racket-head-margin",
        "--racket-motion-weight",
        "--racket-dist-weight",
        "--racket-model-path",
    ]
    assert all(flag in source for flag in flags)
    # And the settings mapping must wire at least the enable/disable toggle.
    assert "settings.racket_enabled" in source
