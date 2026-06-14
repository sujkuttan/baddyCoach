import numpy as np
from pathlib import Path


def test_tracknet_predict_returns_position():
    from app.models.tracknet import TrackNetV3
    model_path = Path("ckpts/TrackNet_best.pt")
    if not model_path.exists():
        return
    model = TrackNetV3(str(model_path), device="cpu")
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    result = model.predict(frames)
    assert len(result) == 1
    assert 'x' in result[0]
    assert 'y' in result[0]
    assert 'confidence' in result[0]
    assert 0 <= result[0]['confidence'] <= 1


def test_tracknet_predict_batch():
    from app.models.tracknet import TrackNetV3
    model_path = Path("ckpts/TrackNet_best.pt")
    if not model_path.exists():
        return
    model = TrackNetV3(str(model_path), device="cpu")
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(6)]
    results = model.predict_batch(frames, batch_size=3)
    assert len(results) == 4
    for r in results:
        assert 'x' in r
        assert 'y' in r