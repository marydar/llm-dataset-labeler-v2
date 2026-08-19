import time
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

    # texts = load_text_dataset(
    #     "lmsys/lmsys-chat-1m",
    #     conversation_column="conversation",
    #     language_column="language",
    #     max_samples=40
    # )
    texts = load_text_dataset(
        "lmsys/lmsys-chat-1m",
        start_idx=3400,
        end_idx=3500
    )


    print(
        f"Loaded {len(texts)} texts"
    )


    results = []


    # Process batches
    request_count = 0

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

        request_count += 1

        # Save checkpoint after every batch
        save_json(
            results,
            CHECKPOINT_PATH
        )

        if request_count % REQUESTS_BEFORE_SLEEP == 0:
            print(
                f"\n{REQUESTS_BEFORE_SLEEP} requests completed. Sleeping for {SLEEP_SECONDS} seconds..."
            )
            time.sleep(SLEEP_SECONDS)  # Sleep for seconds                      


    print(
        f"Before filtering: {len(results)}"
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
        "maryamdar/llm-chats-labeled-v2"  
    )



if __name__ == "__main__":
    main()