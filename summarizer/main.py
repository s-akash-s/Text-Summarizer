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






# import argparse
# from data_loader import DataLoader
# from model import SummarizationModel
# from pipeline import SummarizationPipeline
# from evaluator import Evaluator

# def main():
 
#     parser = argparse.ArgumentParser(description="Run the summarization pipeline.")
#     parser.add_argument("--model_name", type=str, default="google/pegasus-large", help="Hugging Face model name")
#     parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to summarize")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to run model on: 'cuda' or 'cpu'")
#     parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")

#     args = parser.parse_args()

   
#     data_loader = DataLoader()
#     dataset = data_loader.load_dataset(split=args.split)
#     print(f"Loaded {len(dataset)} samples from {args.split} set.")

    
#     model = SummarizationModel(model_name=args.model_name, device=args.device)

 
#     pipeline = SummarizationPipeline(model)
#     outputs = pipeline.run(dataset, num_samples=args.num_samples)

   
#     evaluator = Evaluator()
#     rouge_scores = evaluator.evaluate(outputs)
#     print("\n🔍 ROUGE Evaluation Scores:")
#     for metric, score in rouge_scores.items():
#         print(f"{metric.upper()}: {score:.4f}")

# if __name__ == "__main__":
#     main()
