# summarizer/evaluator.py

from rouge_score import rouge_scorer
from typing import List, Dict

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate(self, results: List[Dict[str, str]]) -> Dict[str, float]:
        rouge1_scores = []
        rouge2_scores = []
        rougel_scores = []

        for result in results:
            reference = result["reference"]
            generated = result["generated"]

            scores = self.scorer.score(reference, generated)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougel_scores.append(scores["rougeL"].fmeasure)

        return {
            "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores),
            "ROUGE-2": sum(rouge2_scores) / len(rouge2_scores),
            "ROUGE-L": sum(rougel_scores) / len(rougel_scores),
        }
