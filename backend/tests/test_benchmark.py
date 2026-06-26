"""Tests for the benchmark harness.

Marked ``benchmark`` — auto-skips when no manifest/data present.
Fails when results fall below the release gate.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from benchmarks.runner import (
    BenchmarkRunner,
    _hit_f1,
    _stroke_accuracy,
    _attribution_accuracy,
    _homography_error,
    _shuttle_tracking,
)


pytestmark = pytest.mark.benchmark

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(__file__).parent.parent.joinpath("benchmarks", "manifest").exists(),
    reason="No benchmark manifests found — run with --runbenchmark to force",
)


# ═══════════════════════════════════════════════════════════════════
# Unit tests for individual scorers
# ═══════════════════════════════════════════════════════════════════


class TestHitF1:
    def test_perfect(self):
        gt = [{"frame": 10}, {"frame": 20}, {"frame": 30}]
        pred = [{"frame": 10}, {"frame": 20}, {"frame": 30}]
        r = _hit_f1(pred, gt)
        assert r["f1"] == 1.0
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0

    def test_within_tolerance(self):
        gt = [{"frame": 10}, {"frame": 20}]
        pred = [{"frame": 12}, {"frame": 22}]
        r = _hit_f1(pred, gt, tolerance=3)
        assert r["f1"] == 1.0

    def test_miss(self):
        gt = [{"frame": 10}, {"frame": 20}]
        pred = [{"frame": 50}]
        r = _hit_f1(pred, gt)
        assert r["tp"] == 0
        assert r["f1"] == 0.0


class TestStrokeAccuracy:
    def test_exact(self):
        gt = [{"frame": 10, "stroke_type": "smash"},
              {"frame": 20, "stroke_type": "clear"}]
        pred = [{"frame": 10, "stroke_type": "smash"},
                {"frame": 20, "stroke_type": "clear"}]
        r = _stroke_accuracy(pred, gt)
        assert r["accuracy"] == 1.0
        assert r["macro_f1"] == 1.0

    def test_wrong(self):
        gt = [{"frame": 10, "stroke_type": "smash"}]
        pred = [{"frame": 10, "stroke_type": "clear"}]
        r = _stroke_accuracy(pred, gt)
        assert r["accuracy"] == 0.0


class TestAttribution:
    def test_correct(self):
        gt = [{"frame": 10, "player_side": "near"},
              {"frame": 20, "player_side": "far"}]
        pred = [{"frame": 10, "player_side": "near"},
                {"frame": 20, "player_side": "far"}]
        r = _attribution_accuracy(pred, gt)
        assert r["accuracy"] == 1.0

    def test_wrong(self):
        gt = [{"frame": 10, "player_side": "near"}]
        pred = [{"frame": 10, "player_side": "far"}]
        r = _attribution_accuracy(pred, gt)
        assert r["accuracy"] == 0.0


class TestHomography:
    def test_identical(self):
        court = {"corners_pixel": [[0, 0], [100, 0], [0, 100], [100, 100]]}
        r = _homography_error(court, court)
        assert r["mean_error_px"] == 0.0


class TestShuttleTracking:
    def test_perfect(self):
        gt = [{"frame": 10, "x": 50, "y": 50}]
        pred = [{"frame": 10, "x": 50, "y": 50, "confidence": 0.9}]
        r = _shuttle_tracking(pred, gt)
        assert r["detection_rate"] == 1.0

    def test_low_conf_ignored(self):
        gt = [{"frame": 10, "x": 50, "y": 50}]
        pred = [{"frame": 10, "x": 50, "y": 50, "confidence": 0.1}]
        r = _shuttle_tracking(pred, gt)
        assert r["detection_rate"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Integration test for BenchmarkRunner
# ═══════════════════════════════════════════════════════════════════


class TestBenchmarkRunner:
    @pytest.fixture
    def temp_manifest_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                "clip_id": "test_clip_001",
                "video": "clips/test_clip_001.mp4",
                "fps": 30,
                "court_corners_px": [[0, 0], [100, 0], [0, 100], [100, 100]],
                "hits": [
                    {"frame": 10, "stroke_type": "smash", "player_side": "near"},
                    {"frame": 25, "stroke_type": "clear", "player_side": "far"},
                ],
            }
            manifest_path = Path(tmp) / "test_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            yield Path(tmp)

    def test_runner_loads_manifests(self, temp_manifest_dir):
        runner = BenchmarkRunner(manifest_dir=temp_manifest_dir)
        manifests = runner.load_manifests()
        assert len(manifests) == 1
        assert manifests[0]["clip_id"] == "test_clip_001"

    def test_runner_evaluate_clip(self, temp_manifest_dir):
        runner = BenchmarkRunner(manifest_dir=temp_manifest_dir)
        manifest = runner.load_manifests()[0]

        predictions = {
            "hits": [{"frame": 10}, {"frame": 25}],
            "strokes": [{"frame": 10, "stroke_type": "smash"},
                        {"frame": 25, "stroke_type": "clear"}],
            "attribution": [{"frame": 10, "player_side": "near"},
                            {"frame": 25, "player_side": "far"}],
            "court": {"corners_pixel": [[0, 0], [100, 0], [0, 100], [100, 100]]},
            "shuttle": [{"frame": 10, "x": 50, "y": 50, "confidence": 0.9}],
        }

        result = runner.evaluate_clip(manifest, predictions)
        assert result["hit_detection"]["f1"] == 1.0
        assert result["stroke_classification"]["accuracy"] == 1.0
        assert result["attribution"]["accuracy"] == 1.0

    def test_runner_all_with_custom_predictor(self, temp_manifest_dir):
        runner = BenchmarkRunner(manifest_dir=temp_manifest_dir)

        def predictor_fn(manifest):
            return {
                "hits": [{"frame": 10}, {"frame": 25}],
                "strokes": [{"frame": 10, "stroke_type": "smash"},
                            {"frame": 25, "stroke_type": "clear"}],
                "attribution": [{"frame": 10, "player_side": "near"},
                                {"frame": 25, "player_side": "far"}],
                "court": {"corners_pixel": [[0, 0], [100, 0], [0, 100], [100, 100]]},
            "shuttle": [{"frame": 10, "x": 50, "y": 50, "confidence": 0.9},
                        {"frame": 25, "x": 60, "y": 60, "confidence": 0.85}],
        }

        results = runner.run_all(predictor_fn)
        assert results["n_clips"] == 1
        assert results["gate"]["overall"] is True

    def test_runner_save_results(self, temp_manifest_dir):
        runner = BenchmarkRunner(manifest_dir=temp_manifest_dir)
        with tempfile.TemporaryDirectory() as tmp:
            runner.results_dir = Path(tmp)
            results = {"clips": [], "aggregates": {"hit_detection": {"f1": 0.95}},
                       "gate": {"overall": True}, "n_clips": 0,
                       "timestamp": "2025-01-01"}
            path = runner.save_results(results)
            assert path.exists()
            assert path.suffix == ".json"
            md_files = list(Path(tmp).glob("*.md"))
            assert len(md_files) == 1


# ═══════════════════════════════════════════════════════════════════
# Gate integration test
# ═══════════════════════════════════════════════════════════════════


class TestReleaseGate:
    def test_gate_check_passes(self):
        runner = BenchmarkRunner()
        aggregates = {
            "hit_detection": {"f1": 0.95, "precision": 1.0, "recall": 0.9},
            "stroke_classification": {"accuracy": 0.85, "macro_f1": 0.75},
            "attribution": {"accuracy": 0.95},
            "homography": {"mean_error_px": 0.0, "n_points": 4},
            "shuttle_tracking": {"detection_rate": 0.85, "mean_px_error": 5.0, "n_matched": 10},
        }
        gate = runner._check_gate(aggregates)
        assert gate["overall"] is True

    def test_gate_check_fails(self):
        runner = BenchmarkRunner()
        aggregates = {
            "hit_detection": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
            "stroke_classification": {"accuracy": 0.85, "macro_f1": 0.75},
            "attribution": {"accuracy": 0.95},
            "homography": {"mean_error_px": 0.0, "n_points": 4},
            "shuttle_tracking": {"detection_rate": 0.85, "mean_px_error": 5.0, "n_matched": 10},
        }
        gate = runner._check_gate(aggregates)
        assert gate["overall"] is False
        assert gate["hit_detection"]["passed"] is False
