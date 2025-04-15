# summarizer/config.py

# Model and device settings
MODEL_NAME = "google/pegasus-large" # Options: 't5-small', 'facebook/bart-base', 'google/pegasus-xsum'
DEVICE = None  # Auto-detects CUDA/CPU if None

# Inference settings
NUM_SAMPLES = 10
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 128
NUM_BEAMS = 4
EARLY_STOPPING = True

# Dataset settings
DATASET_NAME = "samsum"
SPLIT_RATIO = (0.9, 0.1)

# Generation Hyperparameters
GENERATION_ARGS = {
    "num_beams": 4,
    "length_penalty": 1.2,
    "max_length": 60,
    "min_length": 20,
}