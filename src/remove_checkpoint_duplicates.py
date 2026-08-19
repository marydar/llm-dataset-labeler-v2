import json
def remove_checkpoint_duplicates(
    texts,
    checkpoint_path="checkpoint.json"
):
    """
    Remove texts that already exist in the checkpoint file.
    """

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

    except FileNotFoundError:
        print("Checkpoint file not found. Nothing to remove.")
        return texts

    # All texts that have already been processed
    existing_texts = {
        item["text"]
        for item in checkpoint
        if item.get("text")
    }

    original_count = len(texts)

    # Keep only texts that are NOT already in checkpoint
    texts = [
        text
        for text in texts
        if text not in existing_texts
    ]

    removed_count = original_count - len(texts)

    print(f"Texts before checkpoint filtering: {original_count}")
    print(f"Already in checkpoint: {removed_count}")
    print(f"Texts remaining for LLM: {len(texts)}")

    return texts