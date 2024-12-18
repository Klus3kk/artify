import json
import os


class ConfigManager:
    @staticmethod
    def load_config(config_path, default_config=None):
        """
        Load configuration from a file. If the file does not exist and default_config is provided,
        create the file with the default configuration.

        :param config_path: Path to the configuration file.
        :param default_config: A dictionary with default configuration values.
        :return: Loaded configuration as a dictionary.
        """
        if not os.path.exists(config_path):
            if default_config is not None:
                ConfigManager.save_config(default_config, config_path)
                return default_config
            raise FileNotFoundError(f"Config file '{config_path}' not found.")

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file '{config_path}': {e}")

    @staticmethod
    def save_config(config, config_path):
        """
        Save configuration to a file.

        :param config: Dictionary containing configuration values.
        :param config_path: Path to save the configuration file.
        """
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            raise IOError(f"Failed to save config to '{config_path}': {e}")
