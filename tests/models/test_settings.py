import pytest

from src.models.settings import ModelSettings
from src.util._exceptions import UsageError


def test_model_settings_initialization():
    settings = ModelSettings(
        temperature=0.7, top_p=0.9, max_tokens=100, sword_choice="auto", parallel_sword_calls=True
    )

    assert settings.temperature == 0.7
    assert settings.top_p == 0.9
    assert settings.max_tokens == 100
    assert settings.sword_choice == "auto"
    assert settings.parallel_sword_calls is True


def test_model_settings_validation_valid():
    settings = ModelSettings(temperature=1.0, top_p=0.5, max_tokens=1000)
    settings.validate()  # Should not raise any exceptions


def test_model_settings_validation_invalid_temperature():
    settings = ModelSettings(temperature=2.1)
    with pytest.raises(UsageError, match="temperature must be between 0 and 2"):
        settings.validate()


def test_model_settings_validation_invalid_top_p():
    settings = ModelSettings(top_p=1.1)
    with pytest.raises(UsageError, match="top_p must be between 0 and 1"):
        settings.validate()


def test_model_settings_validation_invalid_max_tokens():
    settings = ModelSettings(max_tokens=0)
    with pytest.raises(UsageError, match="max_tokens must be at least 1"):
        settings.validate()


def test_model_settings_resolve():
    base_settings = ModelSettings(temperature=0.7, top_p=0.9, max_tokens=100)

    override_settings = ModelSettings(temperature=0.8, max_tokens=200)

    resolved = base_settings.resolve(override_settings)

    assert resolved.temperature == 0.8  # Overridden
    assert resolved.top_p == 0.9  # Kept from base
    assert resolved.max_tokens == 200  # Overridden


def test_model_settings_resolve_with_none():
    base_settings = ModelSettings(temperature=0.7, top_p=0.9, max_tokens=100)

    resolved = base_settings.resolve(None)

    assert resolved.temperature == 0.7
    assert resolved.top_p == 0.9
    assert resolved.max_tokens == 100
