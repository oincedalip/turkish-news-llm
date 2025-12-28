from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

from src.config.config_helper import ConfigHelper
from src.data.load_dataset import HuggingFaceDataset
from src.modeling.tokenizer import DataTokenizer
from src.modeling.collator import Collator
from src.modeling.modeling import Modeling

import logging

LOG_FORMAT = '%(asctime)-15s| %(levelname)-7s| %(name)s | %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class Training():
    def __init__(self):
        config_helper = ConfigHelper()
        self.config = config_helper.get_config()

    def train(self):
        dataset = HuggingFaceDataset()
        tokenizer = DataTokenizer(dataset)
        collator = Collator(tokenizer=tokenizer)
        model = Modeling()
        training_args = self._get_training_args()

        trainer = Trainer(
            model.model,
            training_args,
            train_dataset=tokenizer.tokenized_datasets["train"],
            eval_dataset=tokenizer.tokenized_datasets["validation"],
            data_collator=collator.collator,
            processing_class=tokenizer.tokenizer,
            compute_metrics=Training.compute_metrics,
        )
        logging.info('Starting training')
        trainer.train()


    def _get_training_args(self):
        output_model_name = self.config['training']['output_model_name']
        training_args = TrainingArguments(output_model_name,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            push_to_hub=True)
        return training_args
    
    @staticmethod
    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
