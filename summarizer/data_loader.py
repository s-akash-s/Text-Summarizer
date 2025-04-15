# summarizer/data_loader.py

from datasets import load_dataset
from typing import Tuple

class DataLoader:
    def __init__(self, dataset_name: str = "samsum", split_ratio: Tuple[float, float] = (0.9, 0.1)):
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio

    def load_data(self):
        print(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, trust_remote_code=True)

        train_test_split = dataset['train'].train_test_split(test_size=self.split_ratio[1])
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        return train_dataset, test_dataset
