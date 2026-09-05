from datasets import load_dataset
from tqdm import tqdm
import json


def is_attack(
    item,
    moderation_column="openai moderation",
    redacted_column="redacted"
):
    """
    Returns True if the item should be considered an attack
    and therefore filtered out.
    """
    
    return False

    # Filter redacted samples
    if item.get(redacted_column) is True:
        return True

    moderation = item.get(moderation_column)

    if not moderation:
        return False

    try:
        # The column may be stored as a JSON string
        if isinstance(moderation, str):
            moderation = json.loads(moderation)

        # Your data contains a list with one moderation result
        if isinstance(moderation, list):
            if not moderation:
                return False

            moderation = moderation[0]

        # Filter explicitly flagged samples
        if moderation.get("flagged", False):
            return True

        # Filter samples where any moderation category is True
        categories = moderation.get("categories", {})

        if any(categories.values()):
            return True

    except Exception:
        # If the moderation field cannot be parsed,
        # don't filter the sample.
        return False

    return False


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

        # Remove non-English
        if item.get(language_column) != "English":
            continue

        # Remove attacks
        if is_attack(item):
            continue

        conversation = item.get(conversation_column)

        if not conversation:
            continue

        for msg in conversation:

            if msg["role"] == "user":

                text = msg["content"].strip()

                # Remove very short texts
                if len(text.split()) >= 5:
                    texts.append(text)

                break

    return texts