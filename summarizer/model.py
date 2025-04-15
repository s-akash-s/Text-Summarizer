# summarizer/model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from logger import get_logger
from config import *

logger = get_logger(__name__)

class SummarizationModel:
    def __init__(self, model_name: str = "google/pegasus-large", device: str = DEVICE):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def summarize(self, text: str) -> str:
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_INPUT_LENGTH
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=NUM_BEAMS,
            max_length=MAX_OUTPUT_LENGTH,
            early_stopping=EARLY_STOPPING
        )


        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
