# summarizer/pipeline.py

from data_loader import DataLoader
from model import SummarizationModel
from logger import get_logger
from config import MODEL_NAME

logger = get_logger(__name__)

class SummarizationPipeline:
    def __init__(self, model_name=MODEL_NAME):
        self.loader = DataLoader()
        self.model = SummarizationModel(model_name=model_name)
        self.train_data, self.test_data = self.loader.load_data()

    def run_inference(self, num_samples=5):
        logger.info(f"Running inference on {num_samples} test samples...")
        results = []
        for i in range(num_samples):
            item = self.test_data[i]
            dialogue = item["dialogue"]
            reference = item["summary"]
            generated = self.model.summarize(dialogue)

            results.append({
                "dialogue": dialogue,
                "reference": reference,
                "generated": generated
            })
        return results
