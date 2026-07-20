import json
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import (
    image_to_court, foot_midpoint_from_pose,
    COURT_LENGTH, COURT_WIDTH,
)
from app.pipeline.shared.ownership_quality import assign_rally_owners
from app.pipeline.shared.ownership_scorer import OwnershipScorer
from app.pipeline.shared.logging import logger
from app.config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL NEAR/FAR CONVENTION  (verified against gold labels — Task 2.3)
# ─────────────────────────────────────────────────────────────────────────────
# The `side` column of `shots_df` MUST share the exact meaning of the human
# annotation `side` column in `labels_enriched_new.csv` ("near"/"far" = which
# player hit the shot). An inverted convention would flip EVERY attribution
# metric for no good reason, so the mapping below is intentionally direct.
#
#   • "near" = the camera-NEAR / lower-court player. In image coordinates the
#     court y-axis increases downward, so the player with the larger median
#     court-y (nearer the bottom of the frame) is "near". Player tracking
#     (`players.py::_resolve_sides`) assigns `player_1` → "near" and
#     `player_2` → "far" by this rule, independent of track index.
#
#   • BST AimPlayer `alpha > 0.5` ⇒ the FAR player hit the shot, which maps to
#     `side="far"`. `alpha < 0.5` ⇒ "near". (This is used only for diagnostics
#     in `attention_alpha_owner`; it is NOT the emission that sets `side`.)
#
#   • `OwnerDecision.side` ("near"/"far", decided by the Viterbi/anchor logic in
#     `ownership_quality.assign_rally_owners`) is written DIRECTLY to
#     `shots_df["side"]` below with NO inversion. Do NOT add any
#     `side = "far" if decision.side == "near" else "near"` flip — that would
#     invert the convention vs the gold labels.
#
# Regression guard: `tests/test_attribution.py::test_near_far_convention_vs_labels`
# asserts the committed-only attribution match rate (non-null side AND
# `owner_uncertain == False`) against the gold labels stays > 50%. The current
# real run reports ~62.5% committed match, confirming the convention is NOT
# inverted.
# ─────────────────────────────────────────────────────────────────────────────


class PlayerAttributionStage:
    name = "player_attribution"
    input_keys = ["shots", "shuttle", "players", "court", "pose", "rallies"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"attributed": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        players_data = artifacts.get("players")
        court = artifacts.get("court") or {}
        pose_df = artifacts.get_parquet("pose")
        rallies_df = artifacts.get_parquet("rallies")

        if players_data is None:
            return StageResult.from_error("Player data required for attribution")

        if not court.get("valid", False):
            logger.warning("Court geometry invalid; continuing attribution with pixel/fallback cues")

        court_corners = court.get("corners_pixel", [])
        if court_corners and len(court_corners) >= 3:
            court_mid_y = (court_corners[0][1] + court_corners[2][1]) / 2
        else:
            court_mid_y = settings.default_frame_height / 2

        H = None
        if "homography" in court and court["homography"] is not None:
            H = np.array(court["homography"])

        shuttle_y_map = {}
        if shuttle_df is not None and len(shuttle_df) > 0:
            shuttle_sorted = shuttle_df.sort_values("frame").reset_index(drop=True)
            shuttle_y_map = dict(zip(shuttle_sorted["frame"].astype(int), shuttle_sorted["y"].astype(float)))

        if "player_id" not in shots_df.columns:
            shots_df["player_id"] = None
        shots_df["owner_uncertain"] = False
        shots_df["owner_confident"] = False
        shots_df["owner_source"] = "unknown"
        shots_df["owner_reason"] = "unassigned"
        shots_df["side"] = "unknown"
        if config.debug_level >= 1:
            shots_df["attribution_tier"] = "none"

        scorer = OwnershipScorer.from_settings()

        rally_scores: dict[int, list[dict[str, float]]] = {}
        rally_candidates: dict[int, list[int]] = {}
        players_by_side = {
            p["side"]: p["id"]
            for p in players_data.get("players", [])
            if p.get("side") in {"near", "far"} and p.get("id")
        }

        for idx, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            rid = shot.get("rally_id")

            score_result = scorer.score(
                shuttle_df=shuttle_df, pose_df=pose_df,
                players_data=players_data, court_data=court,
                frame=frame, prev_owner=None,
                shot=shot,
                racket_detections=artifacts.get("racket_detections"),
            )

            rally_id = int(rid) if pd.notna(rid) else -1
            if rally_id not in rally_scores:
                rally_scores[rally_id] = []
                rally_candidates[rally_id] = []
            rally_scores[rally_id].append(score_result)
            rally_candidates[rally_id].append(idx)

            for key in ("near_score", "far_score", "trajectory_near", "trajectory_far",
                        "court_side_near", "court_side_far", "proximity_near", "proximity_far",
                        "motion_near", "motion_far", "pose_near", "pose_far",
                        "turn_near", "turn_far", "bst_diag_near", "bst_diag_far"):
                if key in score_result:
                    shots_df.at[idx, f"ownership_{key}"] = score_result[key]

        for rally_id, score_list in rally_scores.items():
            indices = rally_candidates[rally_id]
            if not indices:
                continue
            decisions = assign_rally_owners(indices, score_list, players_by_side, settings)
            for idx in indices:
                decision = decisions[idx]
                shots_df.at[idx, "player_id"] = decision.player_id
                # OwnerDecision.side is "near"/"far" and is written verbatim to
                # shots_df["side"] — see the CANONICAL NEAR/FAR CONVENTION block
                # above. Intentionally NO inversion.
                shots_df.at[idx, "side"] = decision.side
                shots_df.at[idx, "owner_confident"] = decision.confident
                shots_df.at[idx, "owner_source"] = decision.source
                shots_df.at[idx, "owner_reason"] = decision.reason
                shots_df.at[idx, "owner_uncertain"] = not decision.confident
                if config.debug_level >= 1:
                    shots_df.at[idx, "attribution_tier"] = decision.source

        # ── Post-attribution consistency: BST AimPlayer vs external owner ──
        # After Viterbi assigns final owners, check if BST's internal AimPlayer
        # attention agrees. Flag conflicts for debugging.
        for idx, shot in shots_df.iterrows():
            alpha = shot.get("aimplayer_alpha")
            side = shot.get("side")
            alpha_reliable = bool(shot.get("aim_alpha_reliable", False))
            if alpha is None or side not in {"near", "far"} or not alpha_reliable:
                shots_df.at[idx, "attention_owner_match"] = None
                shots_df.at[idx, "attention_alpha_owner"] = None
                continue
            alpha_owner = "far" if alpha > 0.5 else ("near" if alpha < 0.5 else None)
            if alpha_owner is None:
                shots_df.at[idx, "attention_owner_match"] = None
                shots_df.at[idx, "attention_alpha_owner"] = None
                continue
            shots_df.at[idx, "attention_alpha_owner"] = alpha_owner
            shots_df.at[idx, "attention_owner_match"] = (side == alpha_owner)

        if config.debug_level >= 1:
            matches = shots_df["attention_owner_match"].value_counts().to_dict()
            logger.info("Attention-owner consistency", matches=str(matches))

        if config.debug_level >= 1:
            tier_counts = shots_df["attribution_tier"].value_counts().to_dict()
            logger.info("Attribution tiers", tiers=str(tier_counts))

        # ── Court-space foot position ──
        if H is not None and pose_df is not None:
            for idx, shot in shots_df.iterrows():
                frame = int(shot["frame"])
                pid = shot.get("player_id")
                if not pid:
                    continue
                row_matches = pose_df[(pose_df["frame"] == frame) & (pose_df["player_id"] == pid)]
                if len(row_matches) > 0:
                    row = row_matches.iloc[0]
                    kps = np.array(row["keypoints"].tolist()) if hasattr(row["keypoints"], 'tolist') else np.array(row["keypoints"])
                    if kps.shape == (17, 3) and np.any(kps != 0):
                        foot = foot_midpoint_from_pose(kps[:, :2], kps[:, 2])
                        if foot is not None:
                            try:
                                cx, cy = image_to_court(H, foot)
                                cx = max(0.0, min(COURT_LENGTH, cx))
                                cy = max(0.0, min(COURT_WIDTH, cy))
                                shots_df.at[idx, "court_x"] = round(cx, 3)
                                shots_df.at[idx, "court_y"] = round(cy, 3)
                            except Exception:
                                pass

        # ── Post-attribution confidence calibration ──
        if settings.report_include_logits:
            from app.models.bst import BSTClassifier
            T_far, T_near = BSTClassifier.load_calibration_cache()
            n_calibrated = 0
            for idx, shot in shots_df.iterrows():
                logits_raw = shot.get("logits")
                if not logits_raw or shot.get("is_rule_based", False):
                    continue
                try:
                    logits = np.array(json.loads(logits_raw) if isinstance(logits_raw, str) else logits_raw)
                except Exception:
                    continue
                side = shot.get("side", "near")
                T = T_far if side == "far" else T_near
                _, calibrated_conf, top3 = BSTClassifier.calibrate_probs(logits, T)
                shots_df.at[idx, "calibrated_confidence"] = calibrated_conf
                shots_df.at[idx, "calibrated_top3"] = json.dumps(top3)
                n_calibrated += 1
            if n_calibrated > 0:
                logger.info("Confidence calibration",
                            T_far=f"{T_far:.3f}", T_near=f"{T_near:.3f}",
                            n_calibrated=n_calibrated)

        artifacts.set_parquet("shots", shots_df)

        counts = shots_df["player_id"].value_counts().to_dict()
        return StageResult.success(
            artifacts={"shots": artifacts.path("shots"), "rallies": artifacts.path("rallies")},
            metadata={"attributed": len(shots_df), "distribution": counts, "court_mid_y": court_mid_y}
        )
