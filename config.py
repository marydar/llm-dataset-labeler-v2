import os
from dotenv import load_dotenv


load_dotenv()


# OpenRouter
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

BASE_URL = "https://openrouter.ai/api/v1"


# MODEL = "google/gemini-2.5-flash"
# MODEL ="google/gemma-4-31b-it:free"
# MODEL = "nvidia/nemotron-3.5-content-safety:free"
# MODEL = "openai/gpt-oss-20b:free"   
# MODEL= "poolside/laguna-xs-2.1:free",
# MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


# Nvidia
OPENROUTER_API_KEY = os.getenv(
    "NVIDIA_API_KEY"
)

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
REASONING = False

REQUESTS_BEFORE_SLEEP = 5
SLEEP_SECONDS = 5

#

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