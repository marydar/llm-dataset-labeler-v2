
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
import hashlib


def load_text_dataset(
    dataset_name,
    text_columns=None,
    subset=None,
    split="train",
    start_idx=0,
    end_idx=None,
    separator="\n\n",
    shuffle_seed=42,
    cache_dir="./dataset_cache",
    messages_column="messages",
    user_only=True,
):
    """
    Load a Hugging Face text dataset.

    Supports:
    1. Regular text columns.
    2. Conversation columns containing messages with role/content.
    3. Extracting only user messages (role == "user").

    The dataset is:
    - Loaded from Hugging Face.
    - Shuffled ONCE using shuffle_seed.
    - Saved locally.
    - Reused on future runs.
    - Sliced using start_idx:end_idx after shuffling.
    """

    # ---------------------------------------------------------
    # Create unique cache path
    # ---------------------------------------------------------

    cache_key = (
        f"{dataset_name}_"
        f"{subset}_"
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
    # Load previously shuffled dataset
    # ---------------------------------------------------------

    if os.path.exists(shuffled_cache_path):

        print("Loading previously shuffled dataset...")

        dataset = load_from_disk(shuffled_cache_path)

    # ---------------------------------------------------------
    # Otherwise load + shuffle + save
    # ---------------------------------------------------------

    else:

        print("Loading dataset from Hugging Face...")

        if subset is None:
            dataset = load_dataset(
                dataset_name,
                split=split
            )
        else:
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split
            )

        print(f"Shuffling dataset with fixed seed: {shuffle_seed}")

        dataset = dataset.shuffle(seed=shuffle_seed)

        print(f"Saving shuffled dataset to: {shuffled_cache_path}")

        dataset.save_to_disk(shuffled_cache_path)

    # ---------------------------------------------------------
    # Select requested range AFTER shuffling
    # ---------------------------------------------------------

    if end_idx is None:
        end_idx = len(dataset)

    end_idx = min(end_idx, len(dataset))

    if start_idx < 0 or start_idx >= end_idx:
        raise ValueError(
            f"Invalid range: start_idx={start_idx}, end_idx={end_idx}"
        )

    print(f"Processing shuffled rows {start_idx} -> {end_idx}")

    dataset = dataset.select(range(start_idx, end_idx))

    # ---------------------------------------------------------
    # Extract texts
    # ---------------------------------------------------------

    texts = []

    for item in tqdm(dataset):

        # -----------------------------------------------------
        # Conversation format: extract user messages only
        # -----------------------------------------------------

        if user_only:

            messages = item.get(messages_column)

            if not isinstance(messages, list):
                continue

            for message in messages:

                if not isinstance(message, dict):
                    continue

                if message.get("role") != "user":
                    continue

                content = message.get("content")

                if content is None:
                    continue

                content = str(content).strip()

                if content:
                    texts.append(content)

        # -----------------------------------------------------
        # Regular text columns
        # -----------------------------------------------------

        else:

            if isinstance(text_columns, str):
                columns = [text_columns]
            else:
                columns = text_columns or []

            parts = []

            for column in columns:

                value = item.get(column)

                if value is None:
                    continue

                value = str(value).strip()

                if value:
                    parts.append(value)

            if parts:
                texts.append(separator.join(parts))

    print(f"\nExtracted {len(texts):,} texts")

    return texts
