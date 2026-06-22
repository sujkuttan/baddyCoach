import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import RallySegmentationStage, _is_rally_ending_shot, _infer_end_reason


def test_rally_segmentation_groups_shots(tmp_job_dir):
    """Test basic rally segmentation with time gaps."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    # Rallies separated by large gaps (>60 frames)
    shots_df = pd.DataFrame({
        "frame": [0, 5, 10, 15, 20, 100, 105, 110, 115, 200, 205, 210],
        "stroke_type": ["serve", "clear", "drop", "smash", "clear",
                       "serve", "drop", "lift", "clear",
                       "serve", "smash", "drop"],
        "player_id": ["player_1", "player_2", "player_1", "player_2", "player_1",
                      "player_2", "player_1", "player_2", "player_1",
                      "player_1", "player_2", "player_1"],
        "stroke_confidence": [0.9, 0.7, 0.6, 0.8, 0.7,
                             0.9, 0.6, 0.5, 0.7,
                             0.9, 0.8, 0.7],
    })
    store.set_parquet("shots", shots_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=60)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    assert len(rallies_df) == 3
    assert "rally_id" in rallies_df.columns
    assert "start_frame" in rallies_df.columns
    assert "end_frame" in rallies_df.columns
    assert "shot_count" in rallies_df.columns


def test_is_rally_ending_shot_high_conf_smash():
    # High-confidence smash with moderate gap ends rally
    assert _is_rally_ending_shot("smash", 0.7, 30) is True
    assert _is_rally_ending_shot("smash", 0.6, 26) is True
    # High-confidence smash with small gap does NOT end rally
    assert _is_rally_ending_shot("smash", 0.5, 20) is False
    assert _is_rally_ending_shot("smash", 0.7, 10) is False


def test_is_rally_ending_shot_net_shot():
    # Net shot with gap > 15 ends rally
    assert _is_rally_ending_shot("net_shot", 0.3, 20) is True
    assert _is_rally_ending_shot("net_shot", 0.9, 16) is True
    # Net shot with small gap does NOT end rally
    assert _is_rally_ending_shot("net_shot", 0.5, 5) is False


def test_is_rally_ending_shot_large_gap():
    # Large gap always ends rally regardless of stroke type
    assert _is_rally_ending_shot("clear", 0.6, 50) is True
    assert _is_rally_ending_shot("lift", 0.5, 46) is True


def test_is_rally_ending_shot_normal_shot():
    # Normal shots with small gaps do NOT end rallies
    assert _is_rally_ending_shot("clear", 0.6, 15) is False
    assert _is_rally_ending_shot("drop", 0.4, 10) is False
    assert _is_rally_ending_shot("lift", 0.5, 12) is False


def test_rally_segmentation_with_net_shot_ending(tmp_job_dir):
    """Net shot should end a rally with moderate gap."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    # Rally 1: serve -> clear -> drop -> smash -> net_shot (ends rally at frame 18, gap=7 to next)
    # Rally 2: serve -> clear -> drop
    shots_df = pd.DataFrame({
        "frame": [0, 5, 10, 15, 18, 35, 40, 45],
        "stroke_type": ["serve", "clear", "drop", "smash", "net_shot", "serve", "clear", "drop"],
        "player_id": ["player_1", "player_2", "player_1", "player_2",
                      "player_1", "player_2", "player_1", "player_2"],
        "stroke_confidence": [0.9, 0.7, 0.6, 0.8, 0.5, 0.9, 0.7, 0.6],
    })
    store.set_parquet("shots", shots_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=60)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    # Rally 1 ends at frame 18 (net_shot + gap=17), Rally 2 starts at frame 35
    assert len(rallies_df) == 2
    assert rallies_df.iloc[0]["end_frame"] == 18
    assert rallies_df.iloc[1]["start_frame"] == 35
