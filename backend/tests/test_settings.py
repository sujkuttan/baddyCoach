def test_racket_settings_defaults():
    from app.config.settings import settings
    assert settings.racket_enabled is True
    assert settings.racket_min_conf == 0.4
    assert settings.racket_proximity_blend == 0.5
    assert settings.racket_head_margin == 0.1
    assert settings.racket_model_path is not None
