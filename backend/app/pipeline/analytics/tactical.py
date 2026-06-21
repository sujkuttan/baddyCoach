from collections import Counter, defaultdict

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class TacticalAnalyticsStage:
    name = "tactical_analytics"
    input_keys = ["shots", "court", "shuttle"]
    output_keys = ["tactical_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.from_error("Shot data required")

        tactical = {}
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]

            shot_dist = Counter(player_shots["stroke_type"].tolist())
            total = sum(shot_dist.values())
            shot_distribution = {k: v / total for k, v in shot_dist.items()}

            stroke_sequence = player_shots["stroke_type"].tolist()
            ngrams = self._extract_ngrams(stroke_sequence, n=3)

            rally_openers = Counter()
            rally_enders = Counter()
            rally_shots = defaultdict(list)
            for _, shot in player_shots.iterrows():
                rid = shot.get("rally_id")
                if rid is not None:
                    rally_shots[rid].append(shot["stroke_type"])
            for rid, strokes in rally_shots.items():
                if strokes:
                    rally_openers[strokes[0]] += 1
                    rally_enders[strokes[-1]] += 1

            tactical[player_id] = {
                "shot_distribution": shot_distribution,
                "total_shots": total,
                "common_patterns": ngrams,
                "unique_strokes": list(shot_dist.keys()),
                "rally_openers": dict(rally_openers),
                "rally_enders": dict(rally_enders),
            }

        artifacts.set("tactical_analytics", tactical)

        return StageResult.success(
            artifacts={"tactical_analytics": artifacts.path("tactical_analytics")},
            metadata={"shot_distribution": {k: v["shot_distribution"] for k, v in tactical.items()}}
        )

    @staticmethod
    def _extract_ngrams(sequence: list[str], n: int = 3) -> list[dict]:
        if len(sequence) < n:
            return []

        ngram_counts = Counter()
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngram_counts[ngram] += 1

        return [
            {"pattern": " → ".join(ng), "count": c}
            for ng, c in ngram_counts.most_common(5)
        ]
