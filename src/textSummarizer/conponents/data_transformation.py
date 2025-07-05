import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer, PegasusTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.tokenizer = PegasusTokenizer.from_pretrained(str(config.tokenizer_name), use_fast=False)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'],
                max_length=128,
                truncation=True
            )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        processed_dataset = dataset.map(self.convert_examples_to_features, batched=True)
        processed_dataset.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))

