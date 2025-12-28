from transformers import AutoModelForSequenceClassification
from src.config.config_helper import ConfigHelper

import logging

LOG_FORMAT = '%(asctime)-15s| %(levelname)-7s| %(name)s | %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class Modeling():
    def __init__(self):
        config_helper = ConfigHelper()
        self.config = config_helper.get_config()
        self.initialize_model()


    def initialize_model(self):
        checkpoint = self.config['modeling']['base_model_name']
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        logging.info('base model is initialized successfully')
