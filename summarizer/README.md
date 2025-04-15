# 📝 Modular Text Summarization Pipeline (Self-hosted + Local Inference)

This project implements a **modular, class-based summarization pipeline** using a **self-hosted Transformer model** to summarize dialogue-based data (SAMSum dataset). The solution leverages models like T5, BART, and Pegasus — all running locally with no reliance on external APIs.

---

## 📌 Project Highlights

- 💬 Works on **dialogue summarization** using the [SAMSum dataset](https://huggingface.co/datasets/samsum)
- 🧠 Utilizes **state-of-the-art Transformer models** (`google/pegasus-large`)
- ⚙️ Built with **modular, class-based architecture**
- 🧪 Includes ROUGE-based **evaluation**
- 🧵 Captures model performance insights with comparisons
- 🧱 Easily extensible and GPU-compatible

---

## 🧩 Folder Structure

```
summarizer/
├── data_loader.py     # Loads and splits dataset
├── model.py           # Summarization model wrapper
├── pipeline.py        # Inference orchestrator
├── evaluator.py       # ROUGE evaluation
├── main.py            # Entry point to run the pipeline
├── requirements.txt   # Python dependencies
├── INSIGHTS.md        # Findings & observations
└── README.md          # This file
```

---

## ⚙️ Installation

1. Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd summarizer
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate on macOS/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If you have an NVIDIA GPU:

```bash
# Make sure PyTorch is installed with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Running the Project

```bash
python main.py
```

This will:
- Load and preprocess the SAMSum dataset
- Initialize the selected model (default: `google/pegasus-large`)
- Run inference on a small sample
- Evaluate with ROUGE scores

---

## 🧠 Models Compared

| Model Name               | Inference Quality | Notes |
|--------------------------|-------------------|-------|
| `t5-small`               | ❌ Basic, shallow summaries |
| `facebook/bart-large-cnn`| 🟡 Better fluency, weak abstraction |
| `google/pegasus-xsum`    | 🟡 More abstract, some improvement |
| **`google/pegasus-large`** | ✅ Best quality + structure + ROUGE scores |

---

## 📊 Evaluation

Evaluation is done using **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** metrics:

| Metric    | Score   |
|-----------|---------|
| ROUGE-1   | 0.3165  |
| ROUGE-2   | 0.1477  |
| ROUGE-L   | 0.2610  |

Evaluation was based on 10 randomly selected test samples from the SAMSum dataset.

---

## 🔍 Sample Output

Example:
```
DIALOGUE:
Helen: Not this time. I need to finish the project. My boss will kill me if I put off the deadline once more.
Tricia: Can't you just pop in for one drink?

REFERENCE:
Helen is not going to the party because she needs to finish a project.

GENERATED SUMMARY:
I need to finish the project. My boss will kill me if I put off the deadline once more.
```

---

## 🧩 Design Choices

- Modular class-based structure for **clarity and maintainability**
- Evaluator module isolated to allow **plug-and-play metrics**
- `pipeline.py` used as orchestration layer for **data → model → evaluation**
- Fully local model execution, using **Hugging Face Transformers**

---

## 🔮 Future Improvements

- 🧪 Fine-tuning Pegasus or BART on SAMSum for dialogue-specific learning
- 🧵 Add speaker-tag processing for improved context understanding
- 🎛️ Integrate hyperparameter tuning (beam size, length penalty)
- 📈 Add visualizations (ROUGE score trends, token lengths, etc.)


