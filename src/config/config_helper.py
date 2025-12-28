import configparser
from pathlib import Path

class ConfigHelper():
    def __init__(self):
        config_path = Path(__file__).resolve().parent / "config.ini"
        config = configparser.ConfigParser()
        config.read(config_path)
        self.config = config

    def get_config(self):
        return self.config