import configparser
from datasets import load_dataset
from pathlib import Path


class HuggingFaceDatasetHelper():
    def __init__(self):
        self.config = self.load_config()
        self.load_dataset()

    def load_dataset(self):
        dataset_name = self.config['dataset']['dataset_name']
        raw_datasets = load_dataset(dataset_name)
        self.raw_datasets = raw_datasets

    def load_config(self):
        config = configparser.ConfigParser()
        config_path = Path(__file__).resolve().parent.parent / "config.ini"
        config.read(config_path)
        return config