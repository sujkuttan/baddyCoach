import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.rallies import RallySegmentationStage


def test_rally_segmentation_groups_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": [0, 5, 10, 15, 50, 55, 60, 100, 105, 110],
        "stroke_type": ["serve", "clear", "drop", "net_shot", "serve", "smash", "clear", "serve", "drop", "clear"],
        "player_id": ["player_1", "player_2", "player_1", "player_2",
                      "player_2", "player_1", "player_2", "player_1", "player_2", "player_1"],
        "stroke_confidence": [0.9] * 10,
    })
    store.set_parquet("shots", shots_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=20)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    assert len(rallies_df) == 3
    assert "rally_id" in rallies_df.columns
    assert "start_frame" in rallies_df.columns
    assert "end_frame" in rallies_df.columns
    assert "shot_count" in rallies_df.columns
