from transformers import DataCollatorWithPadding
from src.modeling.tokenizer import DataTokenizer

class Collator():
    def __init__(self, tokenizer: DataTokenizer):
        self.collator = DataCollatorWithPadding(tokenizer=tokenizer.tokenizer)