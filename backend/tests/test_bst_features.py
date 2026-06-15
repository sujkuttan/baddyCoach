import numpy as np
import pandas as pd
import pytest
from app.models.bst_features import BSTFeatureExtractor


@pytest.fixture
def sample_shuttle_df():
    """10 frames of shuttle trajectory."""
    return pd.DataFrame({
        "frame": range(10),
        "x": np.linspace(100, 400, 10),
        "y": np.linspace(200, 350, 10),
        "confidence": np.full(10, 0.9),
    })


@pytest.fixture
def sample_pose_df():
    """2 players, 10 frames of pose keypoints."""
    rows = []
    for frame in range(10):
        for pid in ["player_1", "player_2"]:
            kps = np.random.rand(17, 3).astype(np.float32)
            kps[:, :2] *= np.array([640, 480])
            kps[:, 2] = 0.9
            rows.append({"frame": frame, "player_id": pid, "keypoints": kps.tolist()})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_court():
    return {"court_length": 13.4, "court_width": 5.18}


def test_feature_extractor_returns_144_dims(sample_shuttle_df, sample_pose_df, sample_court):
    extractor = BSTFeatureExtractor(
        frame_width=640, frame_height=480,
        court_length=13.4, court_width=5.18
    )
    features = extractor.extract(
        shuttle_df=sample_shuttle_df,
        pose_df=sample_pose_df,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    assert features.shape == (144,)


def test_feature_extractor_handles_missing_shuttle(sample_pose_df):
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    features = extractor.extract(
        shuttle_df=None,
        pose_df=sample_pose_df,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    assert features.shape == (144,)
    assert not np.all(features == 0)


def test_feature_extractor_handles_missing_pose(sample_shuttle_df):
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    features = extractor.extract(
        shuttle_df=sample_shuttle_df,
        pose_df=None,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    assert features.shape == (144,)
    assert not np.all(features == 0)


def test_feature_extractor_normalizes_values(sample_shuttle_df, sample_pose_df):
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    features = extractor.extract(
        shuttle_df=sample_shuttle_df,
        pose_df=sample_pose_df,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    assert np.all(np.abs(features) < 10)


def test_previous_shots_encoding():
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    prev_shots = [
        {"stroke_type": "clear", "frame": 0},
        {"stroke_type": "smash", "frame": 3},
        {"stroke_type": "drop", "frame": 6},
    ]
    encoding = extractor._encode_previous_shots(prev_shots, current_frame=10)
    assert encoding.shape == (42,)


def test_bst_classifier_fallback_when_no_model():
    """Test BST classifier falls back to rule-based when no model."""
    from app.models.bst import BSTClassifier, STROKE_CLASSES
    classifier = BSTClassifier(None, device="cpu")
    
    features = np.random.rand(144).astype(np.float32)
    stroke_type, confidence = classifier.predict(features)
    
    assert stroke_type in STROKE_CLASSES
    assert confidence > 0


def test_bst_classifier_handles_corrupt_checkpoint(tmp_path):
    """Test BST classifier handles corrupt checkpoint gracefully."""
    from app.models.bst import BSTClassifier, STROKE_CLASSES
    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_text("not a real checkpoint")
    
    classifier = BSTClassifier(str(corrupt_path), device="cpu")
    
    features = np.random.rand(144).astype(np.float32)
    stroke_type, confidence = classifier.predict(features)
    
    assert stroke_type in STROKE_CLASSES


def test_bst_rule_based_smash_detection():
    """Test rule-based fallback detects smash-like features."""
    from app.models.bst import BSTClassifier
    classifier = BSTClassifier(None, device="cpu")
    
    # Create features with high speed and downward trajectory
    features = np.zeros(144)
    features[16] = 0.4  # shuttle_speed > 0.3
    features[22] = 0.15  # shuttle_dy > 0.1
    
    stroke_type, confidence = classifier._rule_based_predict(features)
    assert stroke_type == "smash"
