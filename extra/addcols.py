from datasets import load_dataset

# Load dataset
dataset = load_dataset("maryamdar/topic_classification_dataset_gen")



# Add train-selection column
dataset["train"] = dataset["train"].add_column(
    "use_for_train", [True] * len(dataset["train"])
)

dataset.push_to_hub("maryamdar/topic_classification_dataset_gen")