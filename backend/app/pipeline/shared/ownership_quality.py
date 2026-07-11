from dataclasses import dataclass

import pandas as pd


@dataclass
class OwnerDecision:
    side: str
    player_id: str | None
    confident: bool
    source: str
    reason: str


def count_independent_signals(score: dict, neutral_epsilon: float) -> int:
    signal_pairs = [
        ("trajectory_near", "trajectory_far"),
        ("court_side_near", "court_side_far"),
        ("proximity_near", "proximity_far"),
        ("motion_near", "motion_far"),
        ("pose_near", "pose_far"),
    ]
    return sum(abs(float(score.get(near_key, 0.5)) - float(score.get(far_key, 0.5))) >= neutral_epsilon for near_key, far_key in signal_pairs)


def is_anchor(
    score: dict,
    min_confidence: float,
    min_margin: float,
    min_signals: int,
    neutral_epsilon: float,
) -> bool:
    confidence = max(float(score.get("near_score", 0.5)), float(score.get("far_score", 0.5)))
    margin = abs(float(score.get("near_score", 0.5)) - float(score.get("far_score", 0.5)))
    return (
        confidence >= min_confidence
        and margin >= min_margin
        and count_independent_signals(score, neutral_epsilon) >= min_signals
    )


def assign_rally_owners(
    indices: list[int],
    scores: list[dict],
    players_by_side: dict[str, str],
    settings,
) -> dict[int, OwnerDecision]:
    decisions = {
        idx: OwnerDecision(side="unknown", player_id=None, confident=False, source="unknown", reason="no_anchor")
        for idx in indices
    }
    anchors: list[tuple[int, int, str]] = []

    for pos, (idx, score) in enumerate(zip(indices, scores)):
        if not is_anchor(
            score,
            settings.ownership_min_anchor_confidence,
            settings.ownership_min_anchor_margin,
            settings.ownership_min_anchor_signals,
            settings.ownership_signal_neutral_epsilon,
        ):
            continue
        side = "near" if float(score["near_score"]) >= float(score["far_score"]) else "far"
        anchors.append((pos, idx, side))
        decisions[idx] = OwnerDecision(
            side=side,
            player_id=players_by_side.get(side),
            confident=True,
            source="local_anchor",
            reason="local_evidence",
        )

    for (left_pos, left_idx, left_side), (right_pos, right_idx, right_side) in zip(anchors, anchors[1:]):
        gap = right_pos - left_pos - 1
        if not settings.ownership_viterbi_bridge_enabled or gap <= 0 or gap > settings.ownership_viterbi_max_bridge_shots:
            continue

        expected_right = left_side if (right_pos - left_pos) % 2 == 0 else ("far" if left_side == "near" else "near")
        if right_side != expected_right:
            continue

        cur_side = "far" if left_side == "near" else "near"
        for pos in range(left_pos + 1, right_pos):
            idx = indices[pos]
            decisions[idx] = OwnerDecision(
                side=cur_side,
                player_id=players_by_side.get(cur_side),
                confident=True,
                source="viterbi_bridge",
                reason=f"bounded_bridge:{left_idx}->{right_idx}",
            )
            cur_side = "far" if cur_side == "near" else "near"

    return decisions


def confident_owner_shots(shots_df) -> pd.DataFrame:
    if not isinstance(shots_df, pd.DataFrame) or shots_df.empty:
        return pd.DataFrame()
    if "player_id" not in shots_df.columns:
        return shots_df.iloc[0:0].copy()
    if "owner_confident" in shots_df.columns:
        mask = shots_df["owner_confident"].fillna(False) & shots_df["player_id"].notna()
    else:
        mask = shots_df["player_id"].notna()
    return shots_df[mask].copy()
