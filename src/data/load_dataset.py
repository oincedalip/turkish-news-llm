from datasets import load_dataset
from src.config.config_helper import ConfigHelper

import logging

LOG_FORMAT = '%(asctime)-15s| %(levelname)-7s| %(name)s | %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class HuggingFaceDataset():
    def __init__(self):
        config_helper = ConfigHelper()
        self.config = config_helper.get_config()
        self.load_dataset()

    def load_dataset(self):
        dataset_name = self.config['dataset']['dataset_name']
        raw_datasets = load_dataset(dataset_name)
        self.raw_datasets = raw_datasets
        logging.info('dataset loaded successfully')