import time
from tqdm import tqdm

from src.remove_checkpoint_duplicates import remove_checkpoint_duplicates
from src.shuffle_loader import load_text_dataset
from src.classifier_three_try import classify_batch

from src.utils import (
    load_json,
    save_json,
    random_delay
)

from src.exporter import save_dataset
from src.exporter import push_to_hub
import os

from config import *



def main():

    labels = load_json(
        "labels.json"
    )


    print("Loading dataset...")

    # for the dataset that had many languages
    # texts = load_text_dataset(
    #     dataset_name=SOURCE_DATASET,
    #     conversation_column=TEXT_COLUMNS,
    #     language_column="language",
    #     start_idx=START_IDX,
    #     end_idx=END_IDX,
    # )
    
    texts = load_text_dataset(
        dataset_name=SOURCE_DATASET,
        config_name=CONFIG_NAME,
        text_columns=TEXT_COLUMNS,
        split=SPLIT,
        start_idx=START_IDX,
        end_idx=END_IDX,
    )
    # print(texts[2])
    # return 
    texts = remove_checkpoint_duplicates(
        texts,
        checkpoint_path=CHECKPOINT_PATH
    )


    print(
        f"Loaded {len(texts)} texts"
    )


    results = []
    
    if os.path.exists(CHECKPOINT_PATH):
        results = load_json(CHECKPOINT_PATH)
        print(f"Loaded {len(results)} results from checkpoint")
    else:
        results = []
        print("No checkpoint found. Starting from empty results.")

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
        HF_REPO
    )



if __name__ == "__main__":
    main()