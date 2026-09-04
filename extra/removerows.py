from datasets import load_dataset

# Load dataset
dataset = load_dataset("maryamdar/prompt-batch-test", split="train")

# Remove rows where:
# generation_group == "A"
# label == "B"
# dataset index < 160000
# labels_to_remove = [
#     "Corporate",
#     "Personal",
#     "Corporate Law",
# ]
labels_to_remove = [
    "Medication & Pharmacology",
]
dataset = dataset.filter(
    lambda x, idx: not (
        x["generation_group"] ==  "second_label_desc_Medication_single"
        and x["label"] in labels_to_remove
    ),
    with_indices=True
)

# Push filtered dataset to Hugging Face
dataset.push_to_hub("maryamdar/prompt-batch-test")