from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
import hashlib


def load_text_dataset(
    dataset_name,
    text_columns,
    config_name=None,
    split="train",
    start_idx=0,
    end_idx=None,
    separator="\n\n",
    shuffle_seed=42,
    cache_dir="./dataset_cache",
):
    """
    Load a Hugging Face text dataset.

    The dataset is:
    1. Loaded
    2. Shuffled ONCE using shuffle_seed
    3. Saved locally
    4. On future runs, the already-shuffled dataset is loaded from disk
    5. start_idx:end_idx is then selected

    This means:

        Day 1:  start_idx=0,   end_idx=100
        Day 2:  start_idx=100, end_idx=200

    will operate on the SAME shuffled dataset.
    """

    # ---------------------------------------------------------
    # Allow a single column name
    # ---------------------------------------------------------

    if isinstance(text_columns, str):
        text_columns = [text_columns]

    # ---------------------------------------------------------
    # Create a unique cache path
    # ---------------------------------------------------------

    cache_key = (
        f"{dataset_name}_"
        f"{config_name}_"
        f"{split}_"
        f"seed_{shuffle_seed}"
    )

    cache_key = hashlib.md5(
        cache_key.encode("utf-8")
    ).hexdigest()

    shuffled_cache_path = os.path.join(
        cache_dir,
        cache_key
    )

    os.makedirs(cache_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load already shuffled dataset if it exists
    # ---------------------------------------------------------

    if os.path.exists(shuffled_cache_path):

        print("Loading previously shuffled dataset...")

        dataset = load_from_disk(
            shuffled_cache_path
        )

    # ---------------------------------------------------------
    # Otherwise load + shuffle + save
    # ---------------------------------------------------------

    else:

        print("Loading dataset from Hugging Face...")

        if config_name is None:

            dataset = load_dataset(
                dataset_name,
                split=split
            )

        else:

            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split
            )

        print(
            f"Shuffling dataset with fixed seed: {shuffle_seed}"
        )

        dataset = dataset.shuffle(
            seed=shuffle_seed
        )

        print(
            f"Saving shuffled dataset to: "
            f"{shuffled_cache_path}"
        )

        dataset.save_to_disk(
            shuffled_cache_path
        )

    # ---------------------------------------------------------
    # Select requested range AFTER shuffling
    # ---------------------------------------------------------

    if end_idx is None:
        end_idx = len(dataset)

    end_idx = min(end_idx, len(dataset))

    print(
        f"Processing shuffled rows "
        f"{start_idx} -> {end_idx}"
    )

    dataset = dataset.select(
        range(start_idx, end_idx)
    )

    # ---------------------------------------------------------
    # Extract texts
    # ---------------------------------------------------------

    texts = []

    for item in tqdm(dataset):

        parts = []

        for column in text_columns:

            value = item.get(column)

            if value is None:
                continue

            value = str(value).strip()

            if value:
                parts.append(value)

        # Skip rows where all requested columns are empty

        if not parts:
            continue

        text = separator.join(parts)

        # Remove very short texts

        # if len(text.split()) < 5:
        #     continue

        texts.append(text)

    return texts