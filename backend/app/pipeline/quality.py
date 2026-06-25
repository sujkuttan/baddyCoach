"""Runtime data-quality gate — DataQualityStage.

Runs last (before report assembly) and writes ``quality.json`` with
trust tiers and per-capability flags.  Downstream consumers (engine.py,
report generator, frontend) use these to suppress or down-rank insights
when data is unreliable.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.pipeline.shared.models import get_model_health


def _bst_fallback_rate(shots_df) -> float:
    if shots_df is None or len(shots_df) == 0:
        return 1.0
    if "is_bst_fallback" not in shots_df.columns:
        return 0.0
    fallback = shots_df["is_bst_fallback"].sum()
    return float(fallback / max(len(shots_df), 1))


def _mean_stroke_confidence(shots_df) -> float:
    if shots_df is None or len(shots_df) == 0:
        return 0.0
    if "stroke_confidence" not in shots_df.columns:
        return 0.0
    return float(shots_df["stroke_confidence"].mean())


def _court_xy_coverage(shots_df) -> float:
    if shots_df is None or len(shots_df) == 0:
        return 0.0
    if "court_x" not in shots_df.columns:
        return 0.0
    valid = shots_df["court_x"].notna()
    return float(valid.sum() / max(len(shots_df), 1))


def _pose_coverage(pose_df, n_frames_expected: int) -> float:
    if pose_df is None or len(pose_df) == 0:
        return 0.0
    if n_frames_expected == 0:
        return 0.0
    n_unique = pose_df["frame"].nunique() if "frame" in pose_df.columns else len(pose_df)
    return min(1.0, n_unique / n_frames_expected)


def _shuttle_detection_rate(shuttle_df, min_conf: float = 0.5) -> float:
    if shuttle_df is None or len(shuttle_df) == 0:
        return 0.0
    if "confidence" in shuttle_df.columns:
        detected = (shuttle_df["confidence"] > min_conf).sum()
        return float(detected / max(len(shuttle_df), 1))
    return 1.0


def compute_quality(artifacts: ArtifactStore) -> dict:
    """Compute quality.json contents from pipeline artifacts.

    Returns a dict with tier, quality_score, capability_trust, caveats, etc.
    """
    court = artifacts.get("court") or {}
    court_valid = bool(court.get("valid", False))

    shots_df = artifacts.get_parquet("shots")
    pose_df = artifacts.get_parquet("pose")
    shuttle_df = artifacts.get_parquet("shuttle")

    n_shots = len(shots_df) if shots_df is not None else 0
    n_rallies = 0
    rallies_df = artifacts.get_parquet("rallies")
    if rallies_df is not None:
        n_rallies = len(rallies_df)

    # Estimate expected frames from video resolution metadata
    res = artifacts.get("video_resolution") or {}
    fps = settings.fps or 30.0
    vid_info = artifacts.get("video_info") or {}
    duration_f = vid_info.get("duration_frames", 0)
    n_frames_expected = int(duration_f) if duration_f > 0 else 0

    shuttle_rate = _shuttle_detection_rate(shuttle_df, settings.quality_shuttle_conf_thr)
    bst_fb = _bst_fallback_rate(shots_df)
    mean_conf = _mean_stroke_confidence(shots_df)
    xy_cov = _court_xy_coverage(shots_df)
    pose_cov = _pose_coverage(pose_df, n_frames_expected)

    model_health = get_model_health()
    if not model_health:
        model_health = {}

    # ── Quality score (0-1) weighted blend ──────────────────────
    score = 1.0
    penalties = []
    if not court_valid:
        score -= 0.20
        penalties.append("court_invalid")
    if bst_fb > settings.quality_max_fallback_patterns:
        score -= 0.15
        penalties.append("high_bst_fallback")
    if shuttle_rate < 0.5:
        score -= 0.10
        penalties.append("low_shuttle_detection")
    if n_shots < settings.quality_min_shots_tactical:
        score -= 0.10
        penalties.append("too_few_shots")
    if pose_cov < 0.3:
        score -= 0.10
        penalties.append("low_pose_coverage")
    if mean_conf < settings.quality_min_stroke_conf:
        score -= 0.10
        penalties.append("low_stroke_confidence")

    q_score = max(0.0, min(1.0, score))
    if q_score >= 0.75:
        tier = "high"
    elif q_score >= 0.45:
        tier = "medium"
    else:
        tier = "low"

    # ── Capability trust flags ──────────────────────────────────
    rtmpose_health = model_health.get("rtmpose", {})
    rtmpose_loaded = rtmpose_health.get("loaded", True) if rtmpose_health else True

    capability_trust = {
        "tactical": court_valid and bst_fb < 0.4 and n_shots >= settings.quality_min_shots_tactical and mean_conf >= settings.quality_min_stroke_conf,
        "patterns": court_valid and bst_fb < settings.quality_max_fallback_patterns
                    and xy_cov > 0.6 and n_shots >= 20 and mean_conf >= settings.quality_min_stroke_conf,
        "technique": pose_cov > 0.5 and rtmpose_loaded,
        "movement": court_valid and xy_cov > 0.3 and n_shots >= 5 and mean_conf >= settings.quality_min_stroke_conf,
        "progress": q_score >= 0.45,
    }

    # ── Caveats ─────────────────────────────────────────────────
    caveats = []
    if not court_valid:
        caveats.append("Court not detected — spatial analytics unavailable")
    if bst_fb > 0.2:
        caveats.append(f"BST fell back on {bst_fb:.0%} of shots")
    if shuttle_rate < 0.5:
        caveats.append(f"Low shuttle detection rate ({shuttle_rate:.0%})")
    if pose_cov < 0.3:
        caveats.append(f"Low pose coverage ({pose_cov:.0%} of frames)")
    if n_shots < settings.quality_min_shots_tactical:
        caveats.append(f"Only {n_shots} shots — insufficient for tactical analysis")
    if mean_conf < settings.quality_min_stroke_conf:
        caveats.append(f"Low mean stroke confidence ({mean_conf:.2f})")
    untrusted = [k for k, v in capability_trust.items() if not v]
    if untrusted:
        caveats.append(f"Low-confidence run — {', '.join(untrusted)} insights hidden")

    return {
        "court_valid": court_valid,
        "shuttle_detection_rate": round(shuttle_rate, 4),
        "pose_coverage": round(pose_cov, 4),
        "bst_fallback_rate": round(bst_fb, 4),
        "mean_stroke_confidence": round(mean_conf, 4),
        "court_xy_coverage": round(xy_cov, 4),
        "n_shots": n_shots,
        "n_rallies": n_rallies,
        "model_health": model_health,
        "quality_score": round(q_score, 4),
        "tier": tier,
        "capability_trust": capability_trust,
        "caveats": caveats,
    }


class DataQualityStage:
    """Pipeline stage that computes and persists data-quality metadata.

    Intended to run LAST (after all analytics stages) so it can read
    every artifact.  Writes quality.json for use by report generator +
    frontend.
    """

    name = "data_quality"
    input_keys: list[str] = []
    output_keys = ["data_quality"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        quality = compute_quality(artifacts)

        artifacts.set("data_quality", quality)

        # Persist to quality.json for downstream consumers
        quality_path = Path(artifacts.job_dir) / "quality.json"
        quality_path.write_text(json.dumps(quality, indent=2, default=str))

        # Standalone model_health.json (for monitoring/debugging)
        model_health = get_model_health()
        if model_health:
            mh_path = Path(artifacts.job_dir) / "model_health.json"
            mh_path.write_text(json.dumps(model_health, indent=2, default=str))

        logger.info(f"Data quality: tier={quality['tier']}, score={quality['quality_score']}, "
                    f"caveats={len(quality['caveats'])}")

        return StageResult.success(
            artifacts={"quality_json": str(quality_path)},
            metadata={"tier": quality["tier"], "quality_score": quality["quality_score"]},
        )
