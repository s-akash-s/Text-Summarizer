# summarizer/model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class SummarizationModel:
    def __init__(self, model_name: str = "google/pegasus-large", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def summarize(self, text: str, max_input_length=1024, max_output_length=128) -> str:
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input_length
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_output_length,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
