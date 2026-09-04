from collections import defaultdict
from datasets import load_dataset


DATASET_NAME = "maryamdar/topic_classification_dataset_gen"
dataset = load_dataset(DATASET_NAME)
for split in dataset:
    ds = dataset[split]

    text_to_indices = defaultdict(list)

    for i, text in enumerate(ds["text"]):
        text_to_indices[text].append(i)

    duplicates = {
        text: indices
        for text, indices in text_to_indices.items()
        if len(indices) > 1
    }

    print(f"\n{'=' * 60}")
    print(f"Split: {split}")
    print(f"{'=' * 60}")

    print(f"Total rows: {len(ds)}")
    print(f"Unique texts: {len(text_to_indices)}")
    print(f"Texts appearing multiple times: {len(duplicates)}")

    if duplicates:
        print("\nDuplicate texts:\n")

        for text, indices in duplicates.items():
            print("-" * 60)
            print(f"Text: {text}")
            print(f"Indices: {indices}")

            print("Rows:")
            for idx in indices:
                print(
                    f"  index={idx}, "
                    f"old_label={ds['old_label'][idx]}, "
                    f"label={ds['label'][idx]}"
                    f"gen={ds['generation_group'][idx]}"
                    f"model={ds['generator_model'][idx]}"
                )