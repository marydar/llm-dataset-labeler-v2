from tqdm import tqdm

from src.loader import load_text_dataset
from src.classifier import classify_batch

from src.utils import (
    load_json,
    save_json,
    random_delay
)

from src.exporter import save_dataset

from config import *


def main():

    labels = load_json(
        "labels.json"
    )

    print("start")
    # texts = load_text_dataset(
    #     "lmsys/lmsys-chat-1m",
    #     conversation_column="conversation"
    # )
    texts = load_text_dataset(
        "lmsys/lmsys-chat-1m",
        conversation_column="conversation",
        language_column="language"
    )
    print("end")
    print(len(texts))
    # return

#     texts = [
#     "How can I protect my website from SQL injection attacks?",
#     "How do I train a neural network using PyTorch?",
#     "What are the symptoms of diabetes?",
#     "How do I open a bank account?",
#     "Best exercises for building muscle?"
# ]


    # ==========================
    # TEST MODE
    # Only one batch
    # ==========================

    texts = texts[:BATCH_SIZE]


    print(
        f"Testing with {len(texts)} texts"
    )


    results = []


    labeled = classify_batch(
        texts,
        labels
    )


    results.extend(
        labeled
    )

    MIN_CONFIDENCE = 0.8


    results = [
        x for x in results
        if x["confidence"] >= MIN_CONFIDENCE
    ]
    
    save_json(
        results,
        CHECKPOINT_PATH
    )


    save_dataset(
        results,
        FINAL_DATASET_PATH
    )


    print(
        "Finished:",
        len(results)
    )


if __name__ == "__main__":
    main()