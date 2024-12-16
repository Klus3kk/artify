import json

class ConfigManager:
    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
        
    @staticmethod
    def save_config(config, config_path):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)