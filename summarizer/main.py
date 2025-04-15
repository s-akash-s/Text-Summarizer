# summarizer/main.py

from pipeline import SummarizationPipeline
from evaluator import Evaluator

if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    results = pipeline.run_inference(num_samples=10)

    for i, res in enumerate(results):
        print(f"\nExample {i+1}:")
        print("-" * 60)
        print("DIALOGUE:\n", res['dialogue'])
        print("REFERENCE SUMMARY:\n", res['reference'])
        print("GENERATED SUMMARY:\n", res['generated'])

    evaluator = Evaluator()
    metrics = evaluator.evaluate(results)

    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
