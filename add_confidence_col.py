from datasets import load_dataset, Value

# dataset = load_dataset("maryamdar/llm-chats-labeled-v2")

from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================

# Dataset you are borrowing the texts FROM
SOURCE_DATASET_NAME = "maryamdar/topic-classification-dataset-real"

# Split you are borrowing from
SOURCE_SPLIT = "train"

# Dataset you want to save/push the labeled texts TO
OUTPUT_DATASET_NAME = "maryamdar/topic-classification-dataset-real"

# Name to store in the new column
# Change this to whatever name you want
ORIGINAL_DATASET_NAME = "lmsys/lmsys-chat-1m"


# ============================================================
# LOAD SOURCE DATASET
# ============================================================

dataset = load_dataset(
    SOURCE_DATASET_NAME,
    split=SOURCE_SPLIT
)

print("Loaded dataset:")
print(dataset)
print("Columns:", dataset.column_names)


# ============================================================
# ADD ORIGINAL DATASET COLUMN
# ============================================================

dataset = dataset.add_column(
    "source_dataset",
    [ORIGINAL_DATASET_NAME] * len(dataset)
)


# # Optional: keep track of which split the text came from
# dataset = dataset.add_column(
#     "original_dataset_split",
#     [SOURCE_SPLIT] * len(dataset)
# )


# ============================================================
# CHECK
# ============================================================

print("\nUpdated columns:")
print(dataset.column_names)

print("\nFirst example:")
print(dataset[0])


# ============================================================
# PUSH TO HUGGING FACE
# ============================================================

dataset.push_to_hub(
    OUTPUT_DATASET_NAME,
    split="train"
)

print(
    f"\nSuccessfully pushed dataset to: {OUTPUT_DATASET_NAME}"
)
