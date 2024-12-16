import pytest
from utilities.StyleRegistry import StyleRegistry
from utilities.ConfigManager import ConfigManager
from utilities.Logger import Logger
import os

def test_style_registry_random_selection():
    registry = StyleRegistry()
    random_style = registry.get_random_style_image("impressionism")
    assert random_style is not None, "Random style image should be selected."
    assert "impressionism" in random_style, "Selected image should belong to the requested category."

def test_config_manager(tmp_path):
    config_path = tmp_path / "config.json"
    config_data = {"output_dir": "images/output", "default_style": "impressionism"}
    ConfigManager.save_config(config_data, config_path)
    loaded_config = ConfigManager.load_config(config_path)
    assert loaded_config == config_data, "Config should be saved and loaded correctly."

def test_logger():
    logger = Logger.setup_logger()
    assert logger is not None, "Logger should be set up successfully."
