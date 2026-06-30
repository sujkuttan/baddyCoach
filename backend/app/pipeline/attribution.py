import json
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import (
    image_to_court, foot_midpoint_from_pose,
    COURT_LENGTH, COURT_WIDTH,
)
from app.pipeline.shared.ownership_scorer import OwnershipScorer, ViterbiConfig, assign_hit_owners_viterbi
from app.pipeline.shared.logging import logger
from app.config.settings import settings


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
            return StageResult.from_error("Court detection is invalid, cannot perform attribution")

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

        # ── Tier 1: BST model-based attribution (AimPlayer alpha + class_id) ──
        if "player_id" not in shots_df.columns:
            shots_df["player_id"] = None
        shots_df["owner_uncertain"] = False
        if config.debug_level >= 1:
            shots_df["attribution_tier"] = "none"

        from app.models.bst import get_shuttleset_class_info, SHUTTLESET_CLASSES
        bst_side_to_playerside = {"top": "far", "bottom": "near"}
        alpha_conf_thresh = 0.15
        if "shuttleset_class_id" in shots_df.columns:
            max_known_id = len(SHUTTLESET_CLASSES) - 1
            for idx, shot in shots_df.iterrows():
                if pd.notna(shot.get("player_id")):
                    continue

                class_id = shot.get("shuttleset_class_id", 0)
                alpha = shot.get("aimplayer_alpha", 0.5)
                conf = shot.get("stroke_confidence", 0)
                used_signal = None

                # Signal A: AimPlayer alpha
                if abs(alpha - 0.5) > alpha_conf_thresh:
                    player_side = "far" if alpha > 0.5 else "near"
                    for p in players_data.get("players", []):
                        if p.get("side") == player_side:
                            shots_df.at[idx, "player_id"] = p["id"]
                            shots_df.at[idx, "side"] = player_side
                            used_signal = "bst_alpha"
                            break

                # Signal B: shuttleset_class_id prefix
                if used_signal is None and class_id > 0 and class_id <= max_known_id:
                    if conf >= settings.attribution_bst_min_conf:
                        _, side = get_shuttleset_class_info(class_id)
                        if side is not None:
                            player_side = bst_side_to_playerside[side]
                            for p in players_data.get("players", []):
                                if p.get("side") == player_side:
                                    shots_df.at[idx, "player_id"] = p["id"]
                                    shots_df.at[idx, "side"] = player_side
                                    used_signal = "bst_class_id"
                                    break

                if used_signal and config.debug_level >= 1:
                    shots_df.at[idx, "attribution_tier"] = used_signal

        # ── Tier 2: OwnershipScorer emissions + Viterbi rally-level assignment ──
        # Replaces old threshold-based + greedy-alternation + fallback cascade.
        scorer = OwnershipScorer.from_settings()
        viterbi_config = ViterbiConfig.from_settings()

        # Build frame-order list of shots per rally to track prev_owner for turn prior
        rally_shots_order: dict[int, list[int]] = {}
        if rallies_df is not None and len(rallies_df) > 0:
            for _, rally in rallies_df.iterrows():
                r_id = int(rally["rally_id"])
                r_mask = (shots_df["frame"] >= int(rally["start_frame"])) & \
                         (shots_df["frame"] <= int(rally["end_frame"]))
                r_order = shots_df[r_mask].sort_values("frame").index.tolist()
                if r_order:
                    rally_shots_order[r_id] = r_order

        # Collect emissions per rally for unassigned shots
        rally_emissions: dict[int, list[dict[str, float]]] = {}
        rally_candidates: dict[int, list[tuple]] = {}

        for idx, shot in shots_df.iterrows():
            if pd.notna(shot.get("player_id")):
                continue

            frame = int(shot["frame"])

            # Determine prev_owner for turn prior
            prev_owner = None
            rid = shot.get("rally_id")
            if pd.notna(rid) and int(rid) in rally_shots_order:
                order = rally_shots_order[int(rid)]
                pos = order.index(idx) if idx in order else -1
                if pos > 0:
                    prev_idx = order[pos - 1]
                    prev_owner = shots_df.at[prev_idx, "player_id"]

            # Run the scorer
            score_result = scorer.score(
                shuttle_df=shuttle_df, pose_df=pose_df,
                players_data=players_data, court_data=court,
                frame=frame, prev_owner=prev_owner,
            )

            ns = score_result["near_score"]
            fs = score_result["far_score"]

            rally_id = int(rid) if pd.notna(rid) else -1
            if rally_id not in rally_emissions:
                rally_emissions[rally_id] = []
                rally_candidates[rally_id] = []
            rally_emissions[rally_id].append({"near": ns, "far": fs})
            rally_candidates[rally_id].append(idx)

            # Store ownership scores unconditionally (needed for uncertainty check)
            for key in ("near_score", "far_score", "trajectory_near", "trajectory_far",
                        "court_side_near", "court_side_far", "proximity_near", "proximity_far",
                        "motion_near", "motion_far", "pose_near", "pose_far",
                        "turn_near", "turn_far"):
                if key in score_result:
                    shots_df.at[idx, f"ownership_{key}"] = score_result[key]

        # Run Viterbi per rally to assign owners globally
        min_conf = settings.confidence_min_owner_confidence
        uncert_margin = settings.confidence_uncertain_margin
        for rally_id, emissions_list in rally_emissions.items():
            indices = rally_candidates[rally_id]
            if len(indices) == 0:
                continue
            owners = assign_hit_owners_viterbi(
                candidates=indices,
                emissions=emissions_list,
                config=viterbi_config,
            )
            for idx, owner, emission in zip(indices, owners, emissions_list):
                for p in players_data.get("players", []):
                    if p.get("side") == owner:
                        shots_df.at[idx, "player_id"] = p["id"]
                        shots_df.at[idx, "side"] = owner
                        if config.debug_level >= 1:
                            shots_df.at[idx, "attribution_tier"] = "viterbi"
                        break
                # Uncertainty flag: low max-score or near-far gap too small
                owner_score = emission[owner]
                other_score = emission["near" if owner == "far" else "far"]
                uncertain = (owner_score < min_conf) or \
                            (abs(owner_score - other_score) < uncert_margin)
                shots_df.at[idx, "owner_uncertain"] = uncertain

        # ── Post-attribution consistency: BST AimPlayer vs external owner ──
        # After Viterbi assigns final owners, check if BST's internal AimPlayer
        # attention agrees. Flag conflicts for debugging.
        for idx, shot in shots_df.iterrows():
            alpha = shot.get("aimplayer_alpha")
            side = shot.get("side")
            if alpha is None or side is None:
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

        # ── Derive side from player_id ──
        _side_lookup = {}
        for _p in players_data.get("players", []):
            _side_lookup[_p["id"]] = _p.get("side", "near")
        if "side" not in shots_df.columns:
            shots_df["side"] = shots_df["player_id"].map(_side_lookup).fillna("near")
        else:
            shots_df["side"] = shots_df["side"].fillna(shots_df["player_id"].map(_side_lookup).fillna("near"))

        if config.debug_level >= 1:
            tier_counts = shots_df["attribution_tier"].value_counts().to_dict()
            logger.info("Attribution tiers", tiers=str(tier_counts))

        # ── Court-space foot position ──
        if H is not None and pose_df is not None:
            for idx, shot in shots_df.iterrows():
                frame = int(shot["frame"])
                pid = shot.get("player_id", "player_1")
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
