from datasets import load_dataset, Features, Value

old_dataset = load_dataset(
    "maryamdar/topic-classification-dataset-real",
    revision="c41e69a7be62eb2b4cdebefdfc15215a0dd59e07",
    split="train"
)

# features = Features({
#     "text": Value("string"),
#     "label": Value("string"),
#     "parent_label": Value("string"),
#     "generator_model": Value("string"),
#     "source": Value("string"),
#     "confidence_score": Value("float32"),
#     "source_dataset": Value("string"),
# })

# old_dataset = old_dataset.cast(features)

old_dataset.push_to_hub(
    "maryamdar/topic-classification-dataset-real",
    # split="train"
)