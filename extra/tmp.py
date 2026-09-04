# from datasets import load_dataset

# # =========================
# # CONFIG
# # =========================

# SOURCE_DATASET = "maryamdar/topic-classification-dataset-v2"
# NEW_DATASET = "maryamdar/topic-classification-dataset-new-Medical-Law"

# PARENT_LABELS_TO_REMOVE = {
#     "Medical",
#     "Law",
# }

# # =========================
# # LOAD
# # =========================

# dataset = load_dataset(
#     SOURCE_DATASET,
#     split="train"
# )

# print(f"Original rows: {len(dataset)}")

# # =========================
# # FILTER
# # =========================

# filtered_dataset = dataset.filter(
#     lambda row: row["parent_label"] not in PARENT_LABELS_TO_REMOVE
# )

# print(f"Filtered rows: {len(filtered_dataset)}")
# print(f"Removed rows: {len(dataset) - len(filtered_dataset)}")

# # =========================
# # PUSH TO NEW DATASET
# # =========================

# filtered_dataset.push_to_hub(
#     NEW_DATASET
# )

import json
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASET_ID = "maryamdar/topic-classification-dataset-new-Medical-Law"
JSON_PATH = "classification_results_all_labels.json"

MISCLASSIFIED_COLUMN = "was_misclassified"


# ============================================================
# 1. LOAD CLASSIFICATION JSON
# ============================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Total samples in JSON: {len(results)}")


# ============================================================
# 2. KEEP ONLY MISCLASSIFIED SAMPLES
# ============================================================

misclassified = [
    item
    for item in results
    if item["llm_prediction"] != item["label"]
]

print(f"Misclassified samples: {len(misclassified)}")


# ============================================================
# 3. GET THE TEXTS OF MISCLASSIFIED SAMPLES
# ============================================================

misclassified_texts = {
    item["text"]
    for item in misclassified
}

print(
    f"Unique misclassified texts: "
    f"{len(misclassified_texts)}"
)


# ============================================================
# 4. LOAD HUGGING FACE DATASET
# ============================================================

dataset = load_dataset(
    DATASET_ID,
    split="train"
)

print(f"Dataset size: {len(dataset)}")


# ============================================================
# 5. ADD MISCLASSIFICATION COLUMN
# ============================================================

dataset = dataset.add_column(
    MISCLASSIFIED_COLUMN,
    [
        text in misclassified_texts
        for text in dataset["text"]
    ]
)


# ============================================================
# 6. CHECK RESULT
# ============================================================

num_marked = sum(dataset[MISCLASSIFIED_COLUMN])

print(
    f"Rows marked as misclassified: {num_marked}"
)

print(dataset)


# ============================================================
# 7. PUSH UPDATED DATASET
# ============================================================

dataset.push_to_hub(DATASET_ID)

print("Dataset updated successfully!")