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


    texts = load_text_dataset(
        "YOUR_DATASET_NAME",
        text_column="text"
    )


    results=[]



    for i in tqdm(
        range(
            0,
            len(texts),
            BATCH_SIZE
        )
    ):


        batch=texts[
            i:i+BATCH_SIZE
        ]


        labeled = classify_batch(
            batch,
            labels
        )


        results.extend(
            labeled
        )


        save_json(
            results,
            CHECKPOINT_PATH
        )


        random_delay(
            REQUEST_DELAY_MIN,
            REQUEST_DELAY_MAX
        )



    save_dataset(
        results,
        FINAL_DATASET_PATH
    )


    print(
        "Finished:",
        len(results)
    )



if __name__=="__main__":
    main()