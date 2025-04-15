# summarizer/pipeline.py

from data_loader import DataLoader
from model import SummarizationModel

class SummarizationPipeline:
    def __init__(self, model_name="t5-small"):
        self.loader = DataLoader()
        self.model = SummarizationModel(model_name=model_name)
        self.train_data, self.test_data = self.loader.load_data()

    def run_inference(self, num_samples=5):
        print(f"\nRunning inference on {num_samples} test samples...\n")
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
