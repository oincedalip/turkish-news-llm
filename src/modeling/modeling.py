from transformers import AutoModelForSequenceClassification
from src.config.config_helper import ConfigHelper

class Modeling():
    def __init__(self):
        config_helper = ConfigHelper()
        self.config = config_helper.get_config()
        self.initialize_model()


    def initialize_model(self):
        checkpoint = self.config['modeling']['base_model_name']
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
