from datasets import load_dataset

old_dataset = load_dataset(
    "maryamdar/llm-chats-labeled-v2",
    revision="211ca0d848e72382c5cfd4f1a553acc1ac8a28ab"
)

old_dataset.push_to_hub("maryamdar/llm-chats-labeled-v2")