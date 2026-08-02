from datasets import load_dataset
from tqdm import tqdm


def load_text_dataset(
        dataset_name,
        split="train",
        conversation_column="conversation",
        language_column="language",
        start_idx=0,
        end_idx=None
):

    dataset = load_dataset(
        dataset_name,
        split=split
    )

    if end_idx is None:
        end_idx = len(dataset)

    dataset = dataset.select(range(start_idx, end_idx))

    print(f"Processing rows {start_idx} -> {end_idx}")

    texts = []

    for item in tqdm(dataset):

        if item.get(language_column) != "English":
            continue

        conversation = item.get(conversation_column)

        if not conversation:
            continue

        for msg in conversation:

            if msg["role"] == "user":

                text = msg["content"].strip()

                if len(text.split()) >= 5:
                    texts.append(text)

                break

    return texts