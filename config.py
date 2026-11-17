import os
from dotenv import load_dotenv


load_dotenv()


# OpenRouter
# OPENROUTER_API_KEY = os.getenv(
#     "OPENROUTER_API_KEY"
# )

# BASE_URL = "https://openrouter.ai/api/v1"


# MODEL = "google/gemini-2.5-flash"
# MODEL ="google/gemma-4-31b-it:free"
# MODEL = "nvidia/nemotron-3.5-content-safety:free"
# MODEL = "openai/gpt-oss-20b:free"   
# MODEL= "poolside/laguna-xs-2.1:free",
# MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


# Nvidia
# NVIDIA_API_KEY = os.getenv(
#     "NVIDIA_API_KEY"
# )

# BASE_URL = "https://integrate.api.nvidia.com/v1"
# MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
# MODEL = "deepseek-ai/deepseek-v4-pro-0813"
# MODEL = "moonshotai/kimi-k3"



# ============================================================
# API
# ============================================================
# Hooshyar
HOOSHYAR_API_KEY = os.getenv(
    "HOOSHYAR_API_KEY"
)
BASE_URL = "https://hooshyar.payampardaz.com/api/v1"
MODEL = "gpt-6-luna"
# MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL"


REASONING = False
REQUESTS_BEFORE_SLEEP = 30
SLEEP_SECONDS = 30

# Generation
BATCH_SIZE = 10
MAX_TOKENS = 1500
TEMPERATURE = 0.2


# Safety
REQUEST_DELAY_MIN = 4
REQUEST_DELAY_MAX = 8

# Output
CHECKPOINT_PATH = "output/checkpoint.json"
FINAL_DATASET_PATH = "output/labeled_dataset"
HF_REPO = "maryamdar/topic-classification-persian-dataset-real-labeled"

#Load source dataset
SOURCE_DATASET  = "amiralifirouzi/Persian-SFT-102K"
CONFIG_NAME = None
# the range of the source dataset to be used for labeling
TEXT_COLUMNS=["messages"]
SPLIT="train"
START_IDX=0
END_IDX=5000