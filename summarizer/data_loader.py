# summarizer/data_loader.py

from datasets import load_dataset
from typing import Tuple
from logger import get_logger
from config import DATASET_NAME, SPLIT_RATIO

logger = get_logger(__name__)

class DataLoader:
    def __init__(self, dataset_name: str = DATASET_NAME, split_ratio: Tuple[float, float] = SPLIT_RATIO):
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio

    def load_data(self):
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, trust_remote_code=True)

        train_test_split = dataset['train'].train_test_split(test_size=self.split_ratio[1])
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        return train_dataset, test_dataset
