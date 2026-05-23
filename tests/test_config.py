from gridpulse.config import load_settings


def test_default_settings_are_hackathon_first():
    settings = load_settings()

    assert settings.app.name == "GridPulse"
    assert settings.grid.mode == "networkx"
    assert settings.grid.zone_count == 8
    assert settings.validation.enable_pandapower is False
    assert settings.references.baseline == "OpenEMS"
