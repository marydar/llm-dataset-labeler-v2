from datasets import load_dataset
from tqdm import tqdm


def load_text_dataset(
    dataset_name,
    text_columns,
    config_name=None,
    split="train",
    start_idx=0,
    end_idx=None,
    separator="\n\n",
):
    """
    Flexible Hugging Face text dataset loader.

    text_columns:
        A single column name or a list of column names.
        Multiple columns are concatenated in the given order.

    config_name:
        Hugging Face dataset configuration name.
        Required for datasets with multiple configs.
    """

    # Allow a single column name
    if isinstance(text_columns, str):
        text_columns = [text_columns]

    # Load dataset
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

    # Select requested range
    if end_idx is None:
        end_idx = len(dataset)

    dataset = dataset.select(range(start_idx, end_idx))

    print(f"Processing rows {start_idx} -> {end_idx}")

    texts = []

    for item in tqdm(dataset):
        
        parts = []

        # Extract columns in requested order
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

        # Combine columns
        text = separator.join(parts)

        # Remove very short texts
        if len(text.split()) < 5:
            continue

        texts.append(text)

    return texts