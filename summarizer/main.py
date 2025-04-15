# summarizer/main.py

from pipeline import SummarizationPipeline
from evaluator import Evaluator
from config import NUM_SAMPLES
from logger import get_logger


logger = get_logger(__name__)

if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    results = pipeline.run_inference(num_samples=NUM_SAMPLES)


    for i, res in enumerate(results):
        print(f"\nExample {i+1}:")
        print("-" * 60)
        print("DIALOGUE:\n", res['dialogue'])
        print("REFERENCE SUMMARY:\n", res['reference'])
        print("GENERATED SUMMARY:\n", res['generated'])

    evaluator = Evaluator()
    logger.info("Evaluating generated summaries using ROUGE scores...")
    metrics = evaluator.evaluate(results)

    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
