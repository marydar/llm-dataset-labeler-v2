from datasets import load_dataset, Value

dataset = load_dataset("maryamdar/llm-chats-labeled-v2")

for split in dataset:
    dataset[split] = dataset[split].add_column(
        "confidence_score",
        [None] * len(dataset[split])
    ).cast_column(
        "confidence_score",
        Value("float32")
    )

# Check
print(dataset["train"].column_names)

# Save changes back to HF
dataset.push_to_hub("maryamdar/llm-chats-labeled-v2")