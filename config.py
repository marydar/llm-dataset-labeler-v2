import os
from dotenv import load_dotenv


load_dotenv()


# OpenRouter
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

BASE_URL = "https://openrouter.ai/api/v1"


MODEL = "google/gemini-2.5-flash"


# Generation
BATCH_SIZE = 10

MAX_TOKENS = 1500

TEMPERATURE = 0.2


# Safety
REQUEST_DELAY_MIN = 4
REQUEST_DELAY_MAX = 8
KEEP_NOT_RELATED = False
NOT_RELATED_LABEL = "Not Related"

# Output
CHECKPOINT_PATH = "output/checkpoint.json"

FINAL_DATASET_PATH = "output/labeled_dataset"