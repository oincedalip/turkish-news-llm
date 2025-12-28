from transformers import AutoTokenizer
from src.config.config_helper import ConfigHelper
from src.data.load_dataset import HuggingFaceDataset

class DataTokenizer():
    def __init__(self, dataset: HuggingFaceDataset):
        config_helper = ConfigHelper()
        self.config = config_helper.get_config()
        self.raw_datasets = dataset.raw_datasets
        self.tokenize()


    def tokenize(self):
        checkpoint = self.config['modeling']['base_model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenized_datasets = self.raw_datasets.map(self._tokenize_function, batched=True)


    def _tokenize_function(self, example):
        return self.tokenizer(example["sentence1"], example["sentence2"], truncation=True)

