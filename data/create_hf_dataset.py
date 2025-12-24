import configparser
import opendatasets as od
import pandas as pd
import re
import shutil
import random
import logging

from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

from tqdm.auto import tqdm

LOG_FORMAT = '%(asctime)-15s| %(levelname)-7s| %(name)s | %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class DatasetCreator:
    def __init__(self):
        self.config = self.load_config()
        self.load_data()
        self.preprocess_data()
        self.create_dataset()
        # self.save_dataset()
        self.cleanup_files()

    def load_config(self):
        config = configparser.ConfigParser()
        config_path = Path(__file__).resolve().parent / "config.ini"
        config.read(config_path)
        return config

    def load_data(self):
        # Logic to load data from the source
        kaggle_url = self.config['kaggle']['kaggle_dataset_url']
        local_data_path = self.config['kaggle']['local_data_path']
        od.download(kaggle_url)
        logging.info('Successfully downloaded kaggle data')
        data = pd.read_csv(local_data_path)
        self.data = data

    def preprocess_data(self):
        # Logic to preprocess the loaded data
        all_data = []
        number_of_paragraphs = 0
        number_of_sentences = 0
        for i in range(len(self.data)):
            paragraphs = self._split_paragraphs(self.data.iloc[i].text)
            for paragraph in paragraphs:
                paragraph_sentences = []
                if paragraph.strip():
                    sentences = self._split_on_punctuation(paragraph)
                    for sentence in sentences:
                        processed_sentence = self._process_sentence(sentence)
                        if processed_sentence:
                            paragraph_sentences.append(processed_sentence)
                            number_of_sentences += 1
                    all_data.append(paragraph_sentences)
                    number_of_paragraphs += 1
        self.all_data = all_data
        logging.info(f'Preprocessing completed: {number_of_paragraphs} paragraphs and {number_of_sentences} sentences')

    def create_positive_samples(self, min_sentence_length=8):
        first_sentences = []
        second_sentences = []
        progress_bar = tqdm(range(len(self.all_data)))
        logging.info(f'Creating positive samples from {len(self.all_data)} paragraphs')
        for paragraph in self.all_data:
            if len(paragraph) > 1:
                for i in range(len(paragraph) - 1):
                    sentence1 = paragraph[i]
                    sentence2 = paragraph[i + 1]
                    if sentence1 and sentence2 \
                        and len(sentence1.split(' ')) >= min_sentence_length \
                        and len(sentence2.split(' ')) >= min_sentence_length:
                        first_sentences.append(sentence1)
                        second_sentences.append(sentence2)
            progress_bar.update(1)

        positive_examples = pd.DataFrame({'sentence1': first_sentences, 'sentence2': second_sentences})
        positive_examples['label'] = 1
        logging.info(f'Created {positive_examples.shape[0]} positive examples')
        return positive_examples

    def create_negative_samples(self, min_sentence_length=8, sample_size=10000):
        first_sentences = []
        second_sentences = []
        progress_bar = tqdm(range(sample_size))
        logging.info(f'Creating {sample_size} negative samples')

        def get_random_sentence():
            try:
                random_index = random.randint(0, len(self.all_data) - 1)
                paragraph = self.all_data[random_index]
                random_sentence_index = random.randint(0, len(paragraph) - 1)
                return paragraph[random_sentence_index], random_index, random_sentence_index
            except:
                return get_random_sentence()

        for _ in range(sample_size):
            sentence1, paragraph_index1, _ = get_random_sentence()
            while len(sentence1.split()) < min_sentence_length:
                sentence1, paragraph_index1, _ = get_random_sentence()
            sentence2, paragraph_index2, _ = get_random_sentence()
            while len(sentence2.split()) < min_sentence_length or paragraph_index1 == paragraph_index2:
                sentence2, paragraph_index2, _ = get_random_sentence()

            first_sentences.append(sentence1)
            second_sentences.append(sentence2)
            progress_bar.update(1)

        negative_examples = pd.DataFrame({'sentence1': first_sentences, 'sentence2': second_sentences})
        negative_examples['label'] = 0
        logging.info(f'Created {negative_examples.shape[0]} negative examples')
        return negative_examples

    def create_dataset(self):
        # Logic to create a Hugging Face dataset
        positive_examples = self.create_positive_samples()
        negative_sample_size = len(positive_examples)
        negative_examples = self.create_negative_samples(sample_size=negative_sample_size)
        df = pd.concat([positive_examples, negative_examples], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)

        df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

        self.create_huggingface_dataset(df_train, df_val, df_test)

    def create_huggingface_dataset(self, df_train, df_val, df_test):
        features = Features({
            "sentence1": Value("string"),
            "sentence2": Value("string"),
            "label": ClassLabel(
                names=["not_continuation", "continuation"]
            )
        })

        # Ensure label dtype is int
        for df in [df_train, df_val, df_test]:
            df["label"] = df["label"].astype(int)

        train_ds = Dataset.from_pandas(
            df_train,
            features=features,
            preserve_index=False
        )

        val_ds = Dataset.from_pandas(
            df_val,
            features=features,
            preserve_index=False
        )

        test_ds = Dataset.from_pandas(
            df_test,
            features=features,
            preserve_index=False
        )

        self.dataset = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        })
        logging.info('Created Huggingface Dataset object')

    def save_dataset(self):
        # Logic to save the dataset to the specified path
        path = self.config['huggingface']['dataset_path']
        self.dataset.push_to_hub(path)
        logging.info('Pushed the dataset to Huggingface hub')

    def cleanup_files(self):
        try:
            local_data_path = self.config['kaggle']['local_data_path']
            folder_to_remove = Path(local_data_path).resolve().parent
            shutil.rmtree(folder_to_remove)
            logging.info('Removed intermediary data files downloaded from kaggle')
        except:
            logging.error('Downloaded data files could not be removed')

    def _split_on_punctuation(self, string):
        return re.split(r'[.?!\n]+', string)

    def _split_paragraphs(self, string):
        return re.split(r'[\n]{2,}', string)

    def _process_sentence(self, string):
        chars = '\\`*_{}[]()>#+-"'
        for c in chars:
            string = string.replace(c, "")
        return string.strip()