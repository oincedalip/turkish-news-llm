from transformers import DataCollatorWithPadding

class Collator():
    def __init__(self, tokenizer):
        self.collator = DataCollatorWithPadding(tokenizer=tokenizer)