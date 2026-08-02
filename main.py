from tqdm import tqdm

from src.loader import load_text_dataset
from src.classifier import classify_batch

from src.utils import (
    load_json,
    save_json,
    random_delay
)

from src.exporter import save_dataset
from src.exporter import push_to_hub

from config import *



def main():

    labels = load_json(
        "labels.json"
    )


    print("Loading dataset...")

    texts = load_text_dataset(
        "lmsys/lmsys-chat-1m",
        conversation_column="conversation",
        language_column="language",
        max_samples=40
    )


    print(
        f"Loaded {len(texts)} texts"
    )


    results = []


    # Process batches
    for i in tqdm(
        range(
            0,
            len(texts),
            BATCH_SIZE
        ),
        desc="Classifying batches"
    ):


        batch = texts[
            i:i+BATCH_SIZE
        ]


        labeled = classify_batch(
            batch,
            labels
        )


        results.extend(
            labeled
        )


        # Save checkpoint after every batch
        save_json(
            results,
            CHECKPOINT_PATH
        )


        # Avoid hitting OpenRouter limits
        random_delay(
            REQUEST_DELAY_MIN,
            REQUEST_DELAY_MAX
        )


    print(
        f"Before filtering: {len(results)}"
    )


    # Remove low confidence
    MIN_CONFIDENCE = 0.8


    results = [
        x for x in results
        if x.get(
            "confidence",
            0
        ) >= MIN_CONFIDENCE
    ]


    # Remove Not Related
    results = [
        x for x in results
        if x["label"] != "Not Related"
    ]


    print(
        f"After filtering: {len(results)}"
    )


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
    
    push_to_hub(
        results,
        "maryamdar/llm-chats-labeled"
    )



if __name__ == "__main__":
    main()